"""DSV4 page codec: gather a page out of the five tiers and put it back.

The round trip is the whole contract, so most of these tests are variations on
"fill the pool with known bytes, take a page out, destroy the pool, put the page
back, and check nothing else moved". CPU-only, no model.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.dsv4_cost_model import dsv4_pool_sizes
from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
from freetoken.kvcache.hicache.dsv4_codec import POOL_FULL, POOL_WINDOW, DSV4PageCodec
from freetoken.models.deepseek_v4.args import DeepseekV4Args

DEVICE = torch.device("cpu")
P = 128
RATIOS = (0, 0, 4, 128, 4, 128, 4, 0)


def _pool(num_pages=8, swa_ratio=0.5, n_scratch=1):
    args = DeepseekV4Args(
        n_layers=8, compress_ratios=RATIOS, max_seq_len=1024,
        head_dim=512, index_head_dim=128, window_size=P,
    )
    sizes = dsv4_pool_sizes(num_pages=num_pages, args=args, swa_ratio=swa_ratio, P=P)
    pool = DSV4PagedKVCache(
        sizes=sizes, args=args, device=DEVICE, dtype=torch.bfloat16, P=P, n_scratch=n_scratch
    )
    return pool


def _fill(pool, seed=0):
    """Distinct, reproducible content in every tier."""
    g = torch.Generator().manual_seed(seed)
    for L, ratio in enumerate(RATIOS):
        pool.window_pool[L].normal_(generator=g)
        if ratio == 0:
            continue
        pool.cmp_pool[L].normal_(generator=g)
        pool.state_ring[L].buffer.normal_(generator=g)
        if ratio == 4:
            pool.idx_pool[L].normal_(generator=g)
            pool.indexer_state_ring[L].buffer.normal_(generator=g)


def _snapshot(pool):
    out = []
    for L, ratio in enumerate(RATIOS):
        out.append(pool.window_pool[L].clone())
        if ratio == 0:
            continue
        out.append(pool.cmp_pool[L].clone())
        out.append(pool.state_ring[L].buffer.clone())
        if ratio == 4:
            out.append(pool.idx_pool[L].clone())
            out.append(pool.indexer_state_ring[L].buffer.clone())
    return out


def test_a_page_round_trips_through_bytes():
    """The contract. Both pools, byte-exact."""
    pool = _pool()
    pool.bind_window_pages(0, 0)
    _fill(pool, seed=1)
    codec = DSV4PageCodec(pool)

    full_blob = codec.gather(0, POOL_FULL)
    win_blob = codec.gather(0, POOL_WINDOW)
    assert full_blob is not None and win_blob is not None
    before = _snapshot(pool)

    # Destroy everything, then restore only page 0 and check it came back.
    for L, ratio in enumerate(RATIOS):
        pool.window_pool[L].zero_()
        if ratio:
            pool.cmp_pool[L].zero_()
            pool.state_ring[L].buffer.zero_()
            if ratio == 4:
                pool.idx_pool[L].zero_()
                pool.indexer_state_ring[L].buffer.zero_()
    codec.scatter(0, POOL_FULL, full_blob)
    codec.scatter(0, POOL_WINDOW, win_blob)

    for L, ratio in enumerate(RATIOS):
        assert torch.equal(pool.window_pool[L][:P], _snapshot_window(before, L))
        if ratio == 0:
            continue
        n = P // ratio
        assert torch.equal(pool.cmp_pool[L][:n], _snapshot_cmp(before, L))
        rs = pool.ring_size(L)
        assert torch.equal(pool.state_ring[L].buffer[:rs], _snapshot_state(before, L, rs))


def _idx_of(L, kind):
    """Index into the flat snapshot list for layer L."""
    i = 0
    for LL, ratio in enumerate(RATIOS):
        order = ["window"] + (["cmp", "state"] if ratio else []) + (["idx", "istate"] if ratio == 4 else [])
        if LL == L:
            return i + order.index(kind)
        i += len(order)
    raise KeyError(L)


def _snapshot_window(snap, L):
    return snap[_idx_of(L, "window")][:P]


def _snapshot_cmp(snap, L):
    n = P // RATIOS[L]
    return snap[_idx_of(L, "cmp")][:n]


def _snapshot_state(snap, L, rs):
    return snap[_idx_of(L, "state")][:rs]


def test_restoring_one_page_does_not_disturb_another():
    """Pages must be isolated — that is what makes them independently cacheable.

    The ring is the one that could go wrong: its rows are addressed by a block
    base derived from the window slot, and an off-by-one in that arithmetic
    would silently overwrite the neighbouring page's carry state.
    """
    pool = _pool()
    pool.bind_window_pages(0, 0)
    pool.bind_window_pages(P, P)
    _fill(pool, seed=2)
    codec = DSV4PageCodec(pool)

    page1_before = [
        pool.state_ring[L].buffer[
            (P // P) * pool.ring_size(L) : (P // P) * pool.ring_size(L) + pool.ring_size(L)
        ].clone()
        for L, r in enumerate(RATIOS) if r
    ]
    blob = codec.gather(0, POOL_WINDOW)
    codec.scatter(0, POOL_WINDOW, blob)

    k = 0
    for L, r in enumerate(RATIOS):
        if not r:
            continue
        rs = pool.ring_size(L)
        base = (P // P) * rs
        assert torch.equal(pool.state_ring[L].buffer[base : base + rs], page1_before[k]), (
            f"layer {L}: restoring page 0 disturbed page 1's ring block"
        )
        k += 1


def test_a_page_outside_the_window_has_no_window_blob():
    """Sliding-window semantics, and why the two pools have different hit policies.

    A page that has slid out keeps its compressed history and loses its window
    rows. A caller must be able to tell that apart from a failure — `gather`
    returns None rather than raising, and the FULL tier is still there.
    """
    pool = _pool()
    pool.bind_window_pages(0, 0)
    codec = DSV4PageCodec(pool)
    assert codec.window_resident(0)
    assert codec.gather(0, POOL_WINDOW) is not None

    pool.unbind_window_pages(torch.arange(0, P, dtype=torch.int64))
    assert not codec.window_resident(0)
    assert codec.gather(0, POOL_WINDOW) is None
    assert codec.gather(0, POOL_FULL) is not None, "the full tier outlives the window"


def test_restoring_a_window_page_that_is_not_bound_is_refused():
    """Loudly, not silently. Without the guard the write lands on whatever slot
    the stale mapping happens to point at — another sequence's page."""
    pool = _pool()
    pool.bind_window_pages(0, 0)
    codec = DSV4PageCodec(pool)
    blob = codec.gather(0, POOL_WINDOW)
    pool.unbind_window_pages(torch.arange(0, P, dtype=torch.int64))
    with pytest.raises(RuntimeError, match="no window slots bound"):
        codec.scatter(0, POOL_WINDOW, blob)


def test_a_blob_of_the_wrong_size_is_refused():
    pool = _pool()
    pool.bind_window_pages(0, 0)
    codec = DSV4PageCodec(pool)
    with pytest.raises(ValueError, match="layout wants"):
        codec.scatter(0, POOL_FULL, torch.zeros(7, dtype=torch.uint8))


def test_layout_signature_separates_incompatible_geometries():
    """The signature goes in the storage key. Two engines with different ratios
    produce blobs that are the same shape for the same tokens and mean different
    things; without this they would read each other's bytes."""
    a = DSV4PageCodec(_pool())
    args = DeepseekV4Args(
        n_layers=8, compress_ratios=(0, 0, 4, 4, 4, 128, 4, 0), max_seq_len=1024,
        head_dim=512, index_head_dim=128, window_size=P,
    )
    sizes = dsv4_pool_sizes(num_pages=8, args=args, swa_ratio=0.5, P=P)
    b = DSV4PageCodec(
        DSV4PagedKVCache(sizes=sizes, args=args, device=DEVICE,
                         dtype=torch.bfloat16, P=P, n_scratch=1)
    )
    assert a.layout_signature != b.layout_signature


def test_the_scratch_row_is_recleared_after_a_restore():
    """`set`/`set_blocks` re-clear it on every write; a restore that skipped it
    would hand the compressor the previous sequence's leftovers as 'empty'."""
    pool = _pool()
    pool.bind_window_pages(0, 0)
    codec = DSV4PageCodec(pool)
    blob = codec.gather(0, POOL_WINDOW)
    for L, r in enumerate(RATIOS):
        if r:
            pool.state_ring[L].buffer[-1].fill_(1234.0)
    codec.scatter(0, POOL_WINDOW, blob)
    for L, r in enumerate(RATIOS):
        if not r:
            continue
        ring = pool.state_ring[L]
        assert torch.all(ring.buffer[-1, : ring.item_size] == 0)
        assert torch.all(torch.isinf(ring.buffer[-1, ring.item_size :]))


def test_a_ratio_that_does_not_divide_the_page_is_refused_at_construction():
    """A page would straddle a compressed row, and no contiguous slice could
    describe it."""
    args = DeepseekV4Args(
        n_layers=2, compress_ratios=(4, 128), max_seq_len=1024,
        head_dim=512, index_head_dim=128, window_size=P,
    )
    sizes = dsv4_pool_sizes(num_pages=8, args=args, swa_ratio=0.5, P=P)
    pool = DSV4PagedKVCache(sizes=sizes, args=args, device=DEVICE,
                            dtype=torch.bfloat16, P=P, n_scratch=1)
    pool.compress_ratios = (4, 48)  # 48 does not divide 128
    with pytest.raises(ValueError, match="does not divide page size"):
        DSV4PageCodec(pool)
