"""Assembly: what `--hicache-*` builds, and what happens when it cannot.

The contract worth pinning is the failure one. Every layer of this tier is
written so that its absence costs nothing and its failure costs nothing beyond
a reprefill — which is only true if the assembly refuses to raise. An engine
that will not start because a storage endpoint is unreachable would be a worse
engine than one with no tier at all.
"""

from __future__ import annotations

import types

import torch

from freetoken.kvcache.dsv4_cost_model import dsv4_pool_sizes
from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
from freetoken.kvcache.hicache.attach import attach_l3, detach_l3
from freetoken.kvcache.swa_radix_cache import SWARadixCache
from freetoken.models.deepseek_v4.args import DeepseekV4Args

from test_dsv4_l3 import P, RATIOS  # noqa: E402

DEVICE = torch.device("cpu")


def _pool():
    args = DeepseekV4Args(
        n_layers=8, compress_ratios=RATIOS, max_seq_len=1024,
        head_dim=512, index_head_dim=128, window_size=P,
    )
    sizes = dsv4_pool_sizes(num_pages=8, args=args, swa_ratio=0.5, P=P)
    return DSV4PagedKVCache(sizes=sizes, args=args, device=DEVICE,
                            dtype=torch.bfloat16, P=P, n_scratch=1)


class _Manager:
    """Only the surface `attach_l3` touches."""

    def __init__(self):
        self.prefix_cache = SWARadixCache(
            device=DEVICE, page_size=P, sliding_window_size=P
        )
        self.l3_writer = None
        self.l3_prefetcher = None


def _cfg(**over):
    base = dict(
        hicache_storage_backend=None,
        hicache_storage_backend_extra_config=None,
        hicache_staging_pages=2,
        hicache_prefetch_deadline_s=0.25,
        hicache_max_inflight=2,
        hicache_write_queue_bytes=1 << 20,
        served_model_name="test",
    )
    base.update(over)
    return types.SimpleNamespace(**base)


_GOOD = (
    '{"module_path": "test_l3_attach", "class_name": "OKBackend"}'
)


class OKBackend:
    def __init__(self, storage_config, extra):
        self.storage_config = storage_config
        self.extra = extra
        self.registered_pools = {}

    def register_mem_host_pool_v2(self, host_pool, name):
        self.registered_pools[name] = host_pool


class BrokenBackend:
    def __init__(self, storage_config, extra):
        raise RuntimeError("endpoint unreachable")


def test_no_backend_means_no_tier_and_no_cost():
    m = _Manager()
    assert attach_l3(m, _pool(), _cfg()) is False
    assert m.l3_writer is None and m.l3_prefetcher is None
    assert m.prefix_cache.enable_page_hash is False, (
        "hashing costs a digest per page on the scheduler thread and buys "
        "nothing without a tier to name pages for"
    )


def test_a_configured_backend_builds_both_halves():
    m = _Manager()
    assert attach_l3(m, _pool(), _cfg(
        hicache_storage_backend="dynamic",
        hicache_storage_backend_extra_config=_GOOD,
    )) is True
    try:
        assert m.l3_writer is not None and m.l3_prefetcher is not None
        assert m.prefix_cache.enable_page_hash is True
        assert m.l3_prefetcher.deadline_s == 0.25
        assert m.l3_prefetcher.max_inflight == 2
    finally:
        detach_l3(m)


def test_a_backend_that_will_not_build_leaves_the_engine_alone():
    """The whole tier's promise is that it can be switched on without a fallback
    plan. An assembly that raised would make an unreachable endpoint a startup
    failure instead of a missing cache."""
    m = _Manager()
    ok = attach_l3(m, _pool(), _cfg(
        hicache_storage_backend="dynamic",
        hicache_storage_backend_extra_config=
        '{"module_path": "test_l3_attach", "class_name": "BrokenBackend"}',
    ))
    assert ok is False
    assert m.l3_writer is None and m.l3_prefetcher is None
    assert m.prefix_cache.enable_page_hash is False


def test_a_malformed_extra_config_is_a_missing_tier_not_a_crash():
    m = _Manager()
    for extra in ('{"module_path": "x"}', "not json at all", "{}"):
        assert attach_l3(m, _pool(), _cfg(
            hicache_storage_backend="dynamic",
            hicache_storage_backend_extra_config=extra,
        )) is False
        assert m.l3_writer is None


def test_the_backend_gets_the_geometry_it_needs():
    """FreeToken has no tensor parallelism at all, DSV4 is MLA, and page_first
    is the only layout the zero-copy interfaces accept. A backend that read
    these wrong would address the wrong bytes rather than fail."""
    m = _Manager()
    attach_l3(m, _pool(), _cfg(
        hicache_storage_backend="dynamic",
        hicache_storage_backend_extra_config=_GOOD,
    ))
    try:
        sc = m.l3_writer.tier.storage.storage_config
        assert (sc.tp_rank, sc.tp_size, sc.pp_rank, sc.pp_size) == (0, 1, 0, 1)
        assert sc.is_mla_model is True
        assert sc.is_page_first_layout is True
    finally:
        detach_l3(m)


def test_detach_stops_the_threads_and_the_hashing():
    """A rebuild reallocates every tier buffer. A writer still holding gathered
    bytes would persist a layout that no longer exists."""
    m = _Manager()
    attach_l3(m, _pool(), _cfg(
        hicache_storage_backend="dynamic",
        hicache_storage_backend_extra_config=_GOOD,
    ))
    w = m.l3_writer
    detach_l3(m)
    assert m.l3_writer is None and m.l3_prefetcher is None
    assert m.prefix_cache.enable_page_hash is False
    assert w._thread is None


def test_detach_is_safe_when_nothing_was_attached():
    m = _Manager()
    detach_l3(m)
    detach_l3(m)


# --------------------------------------------------------------------------- #
# hit-rate accounting
# --------------------------------------------------------------------------- #
def test_the_ways_a_lookup_fails_are_counted_apart():
    """miss / expired / declined are not interchangeable, and reporting them as
    one number would hide the only two a knob can fix. A miss says the data is
    not there; expired says it is there and the deadline or backend is too slow;
    declined says the in-flight bound refused it."""
    from freetoken.kvcache.hicache.dsv4_l3 import L3Stats

    st = L3Stats()
    st.bump(lookups=4, hits=1, misses=1, expired=1, declined=1)
    d = st.snapshot()
    assert (d["misses"], d["expired"], d["declined"]) == (1, 1, 1)
    assert d["hit_rate"] == 0.25


def test_adopted_is_counted_apart_from_hit():
    """A fetch can succeed and still buy nothing — admission may find L1 already
    had those pages. hits/lookups says the data is there; adopted/lookups says
    it was worth fetching, and only the second justifies the tier."""
    from freetoken.kvcache.hicache.dsv4_l3 import L3Stats

    st = L3Stats()
    st.bump(lookups=10, hits=8, adopted=3)
    d = st.snapshot()
    assert d["hit_rate"] == 0.8 and d["adopt_rate"] == 0.3


def test_the_rate_is_defined_with_no_lookups():
    """The summary runs on a timer and will be asked before anything happens."""
    from freetoken.kvcache.hicache.dsv4_l3 import L3Stats

    d = L3Stats().snapshot()
    assert d["hit_rate"] == 0.0 and d["adopt_rate"] == 0.0
    assert "L3 summary" in L3Stats().line()


def test_counting_is_safe_from_two_threads():
    """The fetch thread and the scheduler both bump these."""
    import threading
    from freetoken.kvcache.hicache.dsv4_l3 import L3Stats

    st = L3Stats()
    N, T = 20000, 8

    def worker():
        for _ in range(N):
            # Several fields per call: read-modify-write is not atomic under the
            # GIL, and more bytecodes per bump means more chances to be switched
            # out mid-increment. With one field and a few thousand iterations
            # the race is real but rarely observed, which makes for a test that
            # passes with the lock removed.
            st.bump(lookups=1, hits=1, pages_offered=2, bytes_read=64)

    ts = [threading.Thread(target=worker) for _ in range(T)]
    [t.start() for t in ts]; [t.join() for t in ts]
    d = st.snapshot()
    assert d["lookups"] == N * T, f"lost {N * T - d['lookups']} increments"
    assert d["pages_offered"] == 2 * N * T
    assert d["bytes_read"] == 64 * N * T


def test_a_prefetch_records_every_outcome_it_reaches():
    """End to end through the real prefetcher, not the counters in isolation."""
    import time as _t
    from freetoken.kvcache.hicache.dsv4_l3 import DSV4L3Tier
    from freetoken.kvcache.hicache.l3_prefetch import L3Prefetcher, Status
    from test_dsv4_l3 import FakeStorage, P as _P

    pool = _pool()
    pool.bind_window_pages(0, 0)
    tier = DSV4L3Tier(pool, FakeStorage(), staging_pages=2)
    tier.write_pages([(0, "h0")])
    pf = L3Prefetcher(tier, deadline_s=5)
    try:
        pf.start("a", ["h0"])
        for _ in range(200):
            if pf.poll("a")[0] is Status.READY:
                break
            _t.sleep(0.01)
        pf.release("a")
        pf.start("b", ["absent"])
        for _ in range(200):
            if pf.poll("b")[0] is Status.MISS:
                break
            _t.sleep(0.01)
        pf.release("b")
        d = tier.stats.snapshot()
        assert d["lookups"] == 2, d
        assert d["hits"] == 1 and d["misses"] == 1, d
        assert d["hit_rate"] == 0.5
        assert d["pages_offered"] == 1 and d["bytes_read"] > 0
        assert d["writes"] >= 1 and d["pages_stored"] >= 1
    finally:
        pf.stop()


def test_a_slow_backend_is_counted_expired_not_missed():
    """Through the real prefetcher, because the distinction only exists there.

    A miss says the pages are not in storage; expired says they are and the
    fetch ran out of time. Conflating them points at the wrong fix — one is a
    cold cache, the other is a deadline or a slow backend, and only the second
    is a knob.
    """
    import threading as _th
    import time as _t
    from freetoken.kvcache.hicache.dsv4_l3 import DSV4L3Tier
    from freetoken.kvcache.hicache.l3_prefetch import L3Prefetcher, Status
    from test_dsv4_l3 import FakeStorage

    pool = _pool()
    pool.bind_window_pages(0, 0)
    tier = DSV4L3Tier(pool, FakeStorage(), staging_pages=2)
    tier.write_pages([(0, "h0")])          # the pages ARE there
    gate = _th.Event()

    class Slow(FakeStorage):
        def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
            gate.wait(timeout=10)
            return super().batch_exists_v2(keys, pool_transfers, extra_info)

    slow = Slow()
    slow.registered_pools = tier.storage.registered_pools
    slow.blobs = tier.storage.blobs
    tier.storage = slow

    before = tier.stats.snapshot()
    pf = L3Prefetcher(tier, deadline_s=0.05)
    try:
        pf.start("slow", ["h0"])
        for _ in range(300):
            if pf.poll("slow")[0] is Status.EXPIRED:
                break
            _t.sleep(0.01)
        d = tier.stats.snapshot()
        assert d["expired"] - before["expired"] == 1, d
        assert d["misses"] - before["misses"] == 0, (
            "a fetch that ran out of time was booked as a cold cache"
        )
    finally:
        gate.set()
        pf.release("slow")
        pf.stop()
