"""DSV4 pages to an L3 tier: what gets written, what gets skipped, what is said
about it when it does not all land.

The backend here is a dict. That is deliberate — everything worth pinning is on
this side of the interface: which pages a pool can supply, which keys they get,
that a partial failure is reported rather than swallowed, and that the staging
buffer is released whether or not the write worked.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.dsv4_cost_model import dsv4_pool_sizes
from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
from freetoken.kvcache.hicache.dsv4_codec import POOL_FULL, POOL_WINDOW
from freetoken.kvcache.hicache.dsv4_l3 import DSV4L3Tier
from freetoken.kvcache.hicache.storage import HiCacheStorage, PoolTransferResult
from freetoken.models.deepseek_v4.args import DeepseekV4Args

DEVICE = torch.device("cpu")
P = 128
RATIOS = (0, 0, 4, 128, 4, 128, 4, 0)


class FakeStorage(HiCacheStorage):
    """A dict with the v2 surface. `fail_after` makes writes stop landing."""

    def __init__(self, fail_after: int | None = None):
        self.blobs: dict[str, bytes] = {}
        self.fail_after = fail_after
        self.writes = 0

    def batch_set_v2(self, transfers, extra_info=None):
        out = {}
        for t in transfers:
            host = self.registered_pools[t.name]
            oks = []
            for i, key in enumerate(t.keys):
                if self.fail_after is not None and self.writes >= self.fail_after:
                    oks.append(False)
                    continue
                page = host.get_data_page(int(t.host_indices[i].item()), flat=True)
                self.blobs[key] = bytes(page.view(torch.uint8).tolist())
                self.writes += 1
                oks.append(True)
            out[t.name] = oks
        return out

    # The v1 surface is abstract on HiCacheStorage. DSV4 does not use it — its
    # pages need the multi-pool v2 path — but the ABC still demands the methods.
    def get(self, key, target_location=None, target_sizes=None): raise NotImplementedError
    def set(self, key, value=None, target_location=None, target_sizes=None): raise NotImplementedError
    def batch_get(self, keys, target_locations=None, target_sizes=None): raise NotImplementedError
    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None): raise NotImplementedError
    def exists(self, key): return key in self.blobs

    def batch_exists_v2(self, transfers, extra_info=None):
        res = PoolTransferResult.empty()
        for t in transfers:
            n = 0
            for key in t.keys:
                if key in self.blobs:
                    n += 1
                else:
                    break
            res.update_kv_hit_pages(n)
        return res


def _pool(num_pages=8):
    args = DeepseekV4Args(
        n_layers=8, compress_ratios=RATIOS, max_seq_len=1024,
        head_dim=512, index_head_dim=128, window_size=P,
    )
    sizes = dsv4_pool_sizes(num_pages=num_pages, args=args, swa_ratio=0.5, P=P)
    return DSV4PagedKVCache(sizes=sizes, args=args, device=DEVICE,
                            dtype=torch.bfloat16, P=P, n_scratch=1)


def _tier(pool, storage, staging_pages=2):
    return DSV4L3Tier(pool, storage, staging_pages=staging_pages)


def _bind(pool, n):
    for i in range(n):
        pool.bind_window_pages(i * P, i * P)


def test_a_window_resident_page_is_written_to_both_pools():
    pool = _pool()
    _bind(pool, 1)
    st = FakeStorage()
    tier = _tier(pool, st)
    rep = tier.write_pages([(0, "h0")])
    assert rep.complete, rep
    assert tier.key(POOL_FULL, "h0") in st.blobs
    assert tier.key(POOL_WINDOW, "h0") in st.blobs


def test_a_page_that_slid_out_of_the_window_still_stores_its_history():
    """The two pools have different lifetimes, and that is the point. An old
    page keeps its compressed history and has no window rows; writing only the
    full tier for it is correct, not a partial failure."""
    pool = _pool()
    _bind(pool, 2)
    pool.unbind_window_pages(torch.arange(0, P, dtype=torch.int64))  # page 0 slid out
    st = FakeStorage()
    tier = _tier(pool, st)
    rep = tier.write_pages([(0, "h0"), (P, "h1")])

    assert tier.key(POOL_FULL, "h0") in st.blobs
    assert tier.key(POOL_WINDOW, "h0") not in st.blobs
    assert tier.key(POOL_WINDOW, "h1") in st.blobs
    assert rep.skipped_not_resident == 1
    assert rep.complete, "dropping an out-of-window page is normal, not a failure"


def test_pages_already_in_the_backend_are_not_rewritten():
    pool = _pool()
    _bind(pool, 3)
    st = FakeStorage()
    tier = _tier(pool, st)
    pages = [(0, "h0"), (P, "h1"), (2 * P, "h2")]
    tier.write_pages(pages)
    before = st.writes
    rep = tier.write_pages(pages)
    assert st.writes == before, "a second write of the same prefix should be free"
    assert rep.skipped_present[POOL_FULL] == 3


def test_only_a_leading_run_of_existing_pages_is_skipped():
    """Pages are chained, so a gap in the middle cannot be patched by writing
    the tail — a reader stops at the gap either way. Skipping must therefore be
    a prefix, never a set-difference."""
    pool = _pool()
    _bind(pool, 3)
    st = FakeStorage()
    tier = _tier(pool, st)
    pages = [(0, "h0"), (P, "h1"), (2 * P, "h2")]
    tier.write_pages(pages)
    del st.blobs[tier.key(POOL_FULL, "h1")]      # punch a hole in the middle
    before = st.writes
    tier.write_pages(pages)
    # h0 is skipped; h1 and h2 are both rewritten even though h2 was present.
    assert st.writes - before == 2


def test_a_write_that_does_not_land_is_reported_rather_than_swallowed():
    """A page marked stored but absent is worse than one never written: the
    caller stops trying, and every later reader misses."""
    pool = _pool()
    _bind(pool, 3)
    st = FakeStorage(fail_after=2)
    tier = _tier(pool, st)
    rep = tier.write_pages([(0, "h0"), (P, "h1"), (2 * P, "h2")])
    assert not rep.complete
    assert rep.stored[POOL_FULL] == 2
    assert "3 attempted" in str(rep)


def test_the_staging_pool_is_released_even_when_the_write_fails():
    """It is sized for one chunk and nothing holds it between calls; a leak here
    turns the next write into a hard failure that looks unrelated."""
    pool = _pool()
    _bind(pool, 4)
    class Boom(FakeStorage):
        def batch_set_v2(self, transfers, extra_info=None):
            raise RuntimeError("backend down")
    st = Boom()
    tier = _tier(pool, st, staging_pages=2)
    with pytest.raises(RuntimeError, match="backend down"):
        tier.write_pages([(0, "h0"), (P, "h1")])
    assert tier.staging[POOL_FULL].available_size() == 2 * P


def test_more_pages_than_staging_slots_are_written_in_chunks():
    pool = _pool(num_pages=8)
    _bind(pool, 4)
    st = FakeStorage()
    tier = _tier(pool, st, staging_pages=2)
    rep = tier.write_pages([(i * P, f"h{i}") for i in range(4)])
    assert rep.complete, rep
    assert rep.stored[POOL_FULL] == 4


def test_the_key_leads_with_the_layout_signature():
    """Geometry is a compatibility boundary, not a detail. Two engines with
    different compress ratios write blobs of the same length for the same
    tokens; reading one as the other is corruption, not a miss."""
    pool = _pool()
    tier = _tier(pool, FakeStorage())
    k = tier.key(POOL_FULL, "abc")
    assert k.startswith(tier.codec.layout_signature + "/")
    assert k.endswith("/abc")
    assert POOL_FULL in k


def test_a_staging_page_that_does_not_match_the_codec_is_refused():
    """The host pool addresses by token slot, so a page's bytes must divide by
    the page size. A geometry where they do not would silently stage short."""
    pool = _pool()
    tier = _tier(pool, FakeStorage())
    assert tier.staging[POOL_FULL].bytes_per_page == tier.codec.full_page_bytes
    assert tier.staging[POOL_WINDOW].bytes_per_page == tier.codec.window_page_bytes


def test_writing_nothing_is_not_an_error():
    pool = _pool()
    tier = _tier(pool, FakeStorage())
    rep = tier.write_pages([])
    assert rep.complete and not rep.attempted
