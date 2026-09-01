"""L2 host pool + page-hash tests.

Runs where torch is available (i.e. inside the FreeToken image); the pinned
extension is stubbed so no GPU is needed:

    python -m pytest tests/kvcache/test_hicache_host_pool.py -q

The indexing convention is the thing under test. `host_indices` are TOKEN slots
and `get_data_page` takes a page's first token slot, not a page number — every
storage backend is written against that, and getting it wrong reads the wrong
page's bytes without erroring.
"""

import hashlib
import importlib
import random

import pytest
import torch

from freetoken.kvcache.hicache import hashing


@pytest.fixture
def pool(monkeypatch):
    """HostKVCache with the CUDA-only pinned allocator stubbed out."""
    import freetoken.kernel.pinned as pinned

    monkeypatch.setattr(
        pinned, "alloc_pinned_tensor", lambda *shape, dtype: torch.empty(*shape, dtype=dtype)
    )
    monkeypatch.setattr(pinned, "device_ptr", lambda t: t.data_ptr())

    host_pool = importlib.reload(
        importlib.import_module("freetoken.kvcache.hicache.host_pool")
    )
    return host_pool.HostKVCache(
        device_size=100,
        page_size=8,
        bytes_per_token=16,
        dtype=torch.uint8,
        host_to_device_ratio=4.0,
    )


class TestAllocator:
    def test_alloc_is_contiguous_and_accounted(self, pool):
        a = pool.alloc(16)
        assert a.tolist() == list(range(16))
        b = pool.alloc(8)
        assert b.tolist() == list(range(16, 24))
        assert pool.available_size() == pool.size - 24

    def test_free_returns_slots(self, pool):
        a = pool.alloc(16)
        pool.free(a)
        assert pool.available_size() == pool.size

    def test_alloc_must_be_page_multiple(self, pool):
        with pytest.raises(ValueError):
            pool.alloc(5)

    def test_exhaustion_returns_none_not_raise(self, pool):
        # The controller evicts host pages and retries on None; raising here
        # would turn a normal full-pool condition into a crash.
        assert pool.alloc(pool.size * 2) is None

    def test_host_must_exceed_device(self):
        import freetoken.kvcache.hicache.host_pool as hp

        with pytest.raises(ValueError):
            # A load-back holds L2 slots while moving pages to L1, so an L2 no
            # larger than L1 can deadlock with every page pinned.
            hp.HostKVCache(
                device_size=1000,
                page_size=8,
                bytes_per_token=16,
                dtype=torch.uint8,
                host_size_bytes=8 * 16 * 4,
            )


class TestPageViews:
    def test_index_is_a_token_slot_not_a_page_number(self, pool):
        # Token slot 8 is page 1, NOT page 8.
        assert pool.get_data_page(8).data_ptr() == pool.kv_buffer[1].data_ptr()
        assert pool.get_data_page(0).data_ptr() == pool.kv_buffer[0].data_ptr()

    def test_page_is_bytes_per_token_times_page_size(self, pool):
        assert pool.get_data_page(0).numel() == 16 * 8

    def test_unaligned_index_rejected(self, pool):
        with pytest.raises(ValueError):
            pool.get_data_page(3)

    def test_round_trip(self, pool):
        src = torch.arange(128, dtype=torch.uint8)
        pool.set_from_flat_data_page(8, src)
        assert torch.equal(pool.get_data_page(8), src)
        # neighbouring pages untouched
        assert not torch.equal(pool.get_data_page(0), src)

    def test_wrong_size_write_rejected(self, pool):
        with pytest.raises(ValueError):
            pool.set_from_flat_data_page(0, torch.zeros(7, dtype=torch.uint8))

    def test_buffer_meta_sizes(self, pool):
        ptrs, sizes = pool.get_page_buffer_meta([0, 8, 16])
        assert sizes == [128, 128, 128]
        assert len(set(ptrs)) == 3


class TestPageHashing:
    """The hash IS the L3 key; drift makes every cross-process lookup a miss."""

    @staticmethod
    def _upstream(token_ids, prior=None):
        # transcribed from sglang mem_cache/utils.py:365-380
        h = hashlib.sha256()
        if prior:
            h.update(bytes.fromhex(prior))
        for t in token_ids:
            h.update(t.to_bytes(4, byteorder="little", signed=False))
        return h.hexdigest()

    def test_matches_sglang_bit_for_bit(self):
        random.seed(7)
        for trial in range(200):
            toks = [random.randint(0, 2**31) for _ in range(random.randint(1, 64))]
            prior = None if trial % 3 == 0 else self._upstream([trial])
            assert hashing.get_hash_str(toks, prior) == self._upstream(toks, prior)

    def test_chain_excludes_partial_tail(self):
        # A short final page has no stable identity: hashing it would make the
        # same prefix hash differently once the page fills.
        toks = list(range(64 * 5 + 13))
        assert len(hashing.chain_page_hashes(toks, 64)) == 5

    def test_prefix_property(self):
        # A prefix's page hashes must be a prefix of the longer sequence's, or
        # batch_exists' "consecutive existing keys from the start" is meaningless.
        toks = list(range(64 * 5))
        full = hashing.chain_page_hashes(toks, 64)
        assert hashing.chain_page_hashes(toks[: 64 * 3], 64) == full[:3]

    def test_differing_prefix_diverges(self):
        a = hashing.chain_page_hashes([1] * 64 + [2] * 64, 64)
        b = hashing.chain_page_hashes([9] * 64 + [2] * 64, 64)
        assert a[0] != b[0]
        # chaining means the SECOND page differs too, though its tokens match
        assert a[1] != b[1]
