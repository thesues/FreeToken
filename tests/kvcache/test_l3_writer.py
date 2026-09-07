"""The write queue: what the scheduler pays, and what it is told afterwards.

The split under test is the whole design — the gather happens on the calling
thread so nothing reads device memory concurrently, and only the network write
is deferred. These tests pin that split, the backpressure behaviour, and that a
storage failure is reported rather than raised into the scheduler.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch

from freetoken.kvcache.dsv4_cost_model import dsv4_pool_sizes
from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
from freetoken.kvcache.hicache.dsv4_codec import POOL_FULL, POOL_WINDOW
from freetoken.kvcache.hicache.dsv4_l3 import DSV4L3Tier
from freetoken.kvcache.hicache.l3_writer import L3Writer
from freetoken.models.deepseek_v4.args import DeepseekV4Args

from test_dsv4_l3 import FakeStorage, P, RATIOS  # noqa: E402

DEVICE = torch.device("cpu")


def _pool(num_pages=8):
    args = DeepseekV4Args(
        n_layers=8, compress_ratios=RATIOS, max_seq_len=1024,
        head_dim=512, index_head_dim=128, window_size=P,
    )
    sizes = dsv4_pool_sizes(num_pages=num_pages, args=args, swa_ratio=0.5, P=P)
    return DSV4PagedKVCache(sizes=sizes, args=args, device=DEVICE,
                            dtype=torch.bfloat16, P=P, n_scratch=1)


def _bind(pool, n):
    for i in range(n):
        pool.bind_window_pages(i * P, i * P)


def _settle(w, want=1, timeout=10.0):
    out, t0 = [], time.time()
    while len(out) < want and time.time() - t0 < timeout:
        out += w.collect()
        time.sleep(0.01)
    return out


def test_a_submitted_prefix_reaches_storage():
    pool = _pool(); _bind(pool, 2)
    st = FakeStorage()
    w = L3Writer(DSV4L3Tier(pool, st, staging_pages=2))
    try:
        assert w.submit([(0, "h0"), (P, "h1")], tag="req-1")
        [res] = _settle(w)
        assert res.ok and res.tag == "req-1", res
        assert len(st.blobs) == 4      # two pages x two tiers
    finally:
        w.stop()


def test_the_gather_happens_before_submit_returns():
    """The device pool is only safe to read on this thread.

    Determinism matters more than realism here: gating the storage backend is
    not enough, because a deferred gather would run before the backend is
    reached and would usually still win the race against the test. So the writer
    thread is held back entirely, the pool is overwritten while the submission
    sits in the queue, and only then is the thread allowed to run. If the gather
    were deferred, the stored bytes would be the overwritten ones.

    In the engine, that overwrite is the next request reusing the pages.
    """
    pool = _pool(); _bind(pool, 1)
    # The pool is born zeroed, so it has to hold something first — otherwise the
    # "overwrite" below changes nothing and the test cannot fail either way.
    g = torch.Generator().manual_seed(5)
    for L, r in enumerate(RATIOS):
        if r:
            pool.cmp_pool[L].normal_(generator=g)
    st = FakeStorage()
    tier = DSV4L3Tier(pool, st, staging_pages=2)

    class Held(L3Writer):
        def _ensure_thread(self):
            pass          # nothing consumes the queue until we say so

    w = Held(tier)
    try:
        want = bytes(tier.codec.gather(0, POOL_FULL).tolist())
        w.submit([(0, "h0")])
        for L, r in enumerate(RATIOS):     # the next request reuses these pages
            if r:
                pool.cmp_pool[L].fill_(0)
        assert bytes(tier.codec.gather(0, POOL_FULL).tolist()) != want, (
            "the overwrite did not change the page; the test proves nothing"
        )

        L3Writer._ensure_thread(w)         # now let it drain
        _settle(w)
        assert st.blobs[(POOL_FULL, tier.key("h0"))] == want, (
            "the stored bytes are the pool's CURRENT contents, so the gather "
            "ran on the writer thread rather than at submit time"
        )
    finally:
        w.stop()


def test_a_page_outside_the_window_contributes_only_its_history():
    pool = _pool(); _bind(pool, 2)
    pool.unbind_window_pages(torch.arange(0, P, dtype=torch.int64))
    st = FakeStorage()
    tier = DSV4L3Tier(pool, st, staging_pages=2)
    w = L3Writer(tier)
    try:
        w.submit([(0, "h0"), (P, "h1")])
        _settle(w)
        assert (POOL_FULL, tier.key("h0")) in st.blobs
        assert (POOL_WINDOW, tier.key("h0")) not in st.blobs
    finally:
        w.stop()


def test_a_storage_failure_is_reported_not_raised():
    """The scheduler must survive a backend outage. A prefix that did not land
    costs a reprefill next time; an exception crossing into the loop costs
    every request in flight."""
    pool = _pool(); _bind(pool, 1)

    class Down(FakeStorage):
        def batch_set_v2(self, transfers, extra_info=None):
            raise RuntimeError("backend down")

    w = L3Writer(DSV4L3Tier(pool, Down(), staging_pages=2))
    try:
        w.submit([(0, "h0")], tag="doomed")
        [res] = _settle(w)
        assert not res.ok and "backend down" in (res.error or "")
        assert res.tag == "doomed"
    finally:
        w.stop()


def test_the_staging_pool_survives_a_failed_write():
    """Only the writer thread allocates from it, so a leak would turn every
    later write into an exhaustion error that names the wrong cause."""
    pool = _pool(); _bind(pool, 1)

    class Down(FakeStorage):
        def batch_set_v2(self, transfers, extra_info=None):
            raise RuntimeError("backend down")

    tier = DSV4L3Tier(pool, Down(), staging_pages=2)
    w = L3Writer(tier)
    try:
        w.submit([(0, "h0")])
        _settle(w)
        assert tier.staging[POOL_FULL].available_size() == 2 * P
    finally:
        w.stop()


def test_backpressure_drops_rather_than_blocking():
    """Blocking here would stall decoding for every request to save one prefix.
    The drop is counted so the trade is visible rather than mysterious."""
    pool = _pool(); _bind(pool, 1)
    release = threading.Event()

    class Slow(FakeStorage):
        def batch_set_v2(self, transfers, extra_info=None):
            release.wait(timeout=10)
            return super().batch_set_v2(transfers)

    w = L3Writer(DSV4L3Tier(pool, Slow(), staging_pages=2), max_queued_bytes=1)
    try:
        assert w.submit([(0, "h0")]) is True      # first one is admitted
        t0 = time.time()
        assert w.submit([(0, "h0")]) is False     # queue already over budget
        assert time.time() - t0 < 1.0, "submit must not wait on the writer"
        assert w.dropped_submissions == 1
    finally:
        release.set()
        w.stop()


def test_the_byte_budget_is_returned_after_a_write():
    pool = _pool(); _bind(pool, 1)
    w = L3Writer(DSV4L3Tier(pool, FakeStorage(), staging_pages=2))
    try:
        w.submit([(0, "h0")])
        _settle(w)
        assert w.queued_bytes() == 0
    finally:
        w.stop()


def test_submitting_nothing_starts_no_thread():
    """A cache that never writes should not carry a thread around."""
    pool = _pool()
    w = L3Writer(DSV4L3Tier(pool, FakeStorage(), staging_pages=2))
    assert w.submit([]) is True
    assert w._thread is None
    w.stop()


def test_stop_is_safe_without_a_thread_and_twice():
    pool = _pool()
    w = L3Writer(DSV4L3Tier(pool, FakeStorage(), staging_pages=2))
    w.stop()
    w.stop()
