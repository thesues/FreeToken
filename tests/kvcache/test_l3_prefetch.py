"""Prefetch: the deadline, the bound, and the staging accounting.

The property everything here defends is that a prefetch can save work and must
never cost latency. So the tests are mostly about what happens when storage is
slow, absent, or lying — not about the happy path, which is one test.
"""

from __future__ import annotations

import threading
import time

import torch

from freetoken.kvcache.dsv4_cost_model import dsv4_pool_sizes
from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
from freetoken.kvcache.hicache.dsv4_codec import POOL_FULL, POOL_WINDOW
from freetoken.kvcache.hicache.dsv4_l3 import DSV4L3Tier
from freetoken.kvcache.hicache.l3_prefetch import L3Prefetcher, Status
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


def _stocked(staging_pages=4):
    """A tier whose backend already holds one page under 'h0'."""
    pool = _pool()
    pool.bind_window_pages(0, 0)
    st = FakeStorage()
    tier = DSV4L3Tier(pool, st, staging_pages=staging_pages)
    tier.write_pages([(0, "h0")])
    return tier, st


def _swap_storage(tier, storage):
    """Replace a tier's backend, carrying the pool registration across.

    `DSV4L3Tier.__init__` calls `register_mem_host_pool_v2` on the storage it
    was built with, so a bare `tier.storage = other` leaves the new object
    without `registered_pools` and every v2 call raises AttributeError. That
    turns a test of the success path into a test of the exception path — which
    is how the "abandoned fetch frees its slots" case went untested here.
    """
    storage.registered_pools = tier.storage.registered_pools
    storage.blobs = tier.storage.blobs
    tier.storage = storage
    return storage


def _await(pf, key, want=Status.READY, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s, r = pf.poll(key)
        if s is want:
            return s, r
        time.sleep(0.005)
    return pf.poll(key)


def test_a_stored_prefix_comes_back_ready():
    tier, _ = _stocked()
    pf = L3Prefetcher(tier)
    try:
        assert pf.start("r1", ["h0"])
        status, ready = _await(pf, "r1")
        assert status is Status.READY and ready.n_pages == 1
        assert POOL_FULL in ready.slots and POOL_WINDOW in ready.slots
    finally:
        pf.release("r1")
        pf.stop()


def test_nothing_stored_is_a_miss_not_a_wait():
    """A miss has to be distinguishable from "not yet", or the scheduler holds
    the request until the deadline for a prefix that was never there."""
    tier, _ = _stocked()
    pf = L3Prefetcher(tier)
    try:
        pf.start("r1", ["nope"])
        status, _ = _await(pf, "r1", want=Status.MISS)
        assert status is Status.MISS
    finally:
        pf.release("r1")
        pf.stop()


def test_a_slow_backend_expires_instead_of_holding_the_request():
    """The rule: a prefetch may save work, never add latency. A backend that
    takes longer than the deadline must hand the request back to L1."""
    tier, _ = _stocked()
    release = threading.Event()

    class Slow(FakeStorage):
        def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
            release.wait(timeout=10)
            return super().batch_exists_v2(keys, pool_transfers, extra_info)

    _swap_storage(tier, Slow())
    pf = L3Prefetcher(tier, deadline_s=0.05)
    try:
        pf.start("r1", ["h0"])
        t0 = time.time()
        status, _ = _await(pf, "r1", want=Status.EXPIRED, timeout=3)
        assert status is Status.EXPIRED
        assert time.time() - t0 < 1.0, "the deadline did not bound the wait"
        assert pf.expired == 1
    finally:
        release.set()
        pf.release("r1")
        pf.stop()


def test_an_expired_fetch_still_returns_its_staging_slots():
    """The pages may arrive after the scheduler gave up. Keeping them would
    drain the staging pool one abandoned fetch at a time until nothing can be
    prefetched at all."""
    tier, _ = _stocked(staging_pages=2)
    gate = threading.Event()

    class Slow(FakeStorage):
        def batch_get_v2(self, transfers, extra_info=None):
            gate.wait(timeout=10)
            return super().batch_get_v2(transfers)

    _swap_storage(tier, Slow())
    free_before = tier.staging[POOL_FULL].available_size()
    pf = L3Prefetcher(tier, deadline_s=0.05)
    try:
        pf.start("r1", ["h0"])
        # The fetch must actually be holding staging before the deadline, or
        # there is nothing to leak and this test passes without testing.
        t0 = time.time()
        while (tier.staging[POOL_FULL].available_size() == free_before
               and time.time() - t0 < 3):
            time.sleep(0.005)
        assert tier.staging[POOL_FULL].available_size() < free_before, (
            "the fetch never reached the staging allocation"
        )
        _await(pf, "r1", want=Status.EXPIRED, timeout=3)
        pf.release("r1")            # the scheduler moves on
        gate.set()                  # the fetch lands afterwards
        t0 = time.time()
        while (tier.staging[POOL_FULL].available_size() != free_before
               and time.time() - t0 < 5):
            time.sleep(0.01)
        assert tier.staging[POOL_FULL].available_size() == free_before, (
            "an abandoned fetch kept its staging slots"
        )
    finally:
        gate.set()
        pf.stop()


def test_too_many_in_flight_are_declined_not_queued():
    """Queueing would turn a busy moment into added latency for everything
    behind it. Declining just means the request proceeds on L1."""
    tier, _ = _stocked()
    gate = threading.Event()

    class Slow(FakeStorage):
        def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
            gate.wait(timeout=10)
            return super().batch_exists_v2(keys, pool_transfers, extra_info)

    _swap_storage(tier, Slow())
    pf = L3Prefetcher(tier, deadline_s=10, max_inflight=1)
    try:
        assert pf.start("r1", ["h0"]) is True
        t0 = time.time()
        assert pf.start("r2", ["h0"]) is False
        assert time.time() - t0 < 0.5, "start must not wait"
        assert pf.declined == 1
    finally:
        gate.set()
        pf.release("r1")
        pf.stop()


def test_release_is_required_and_idempotent():
    tier, _ = _stocked()
    free_before = tier.staging[POOL_FULL].available_size()
    pf = L3Prefetcher(tier)
    try:
        pf.start("r1", ["h0"])
        _await(pf, "r1")
        assert tier.staging[POOL_FULL].available_size() < free_before
        pf.release("r1")
        pf.release("r1")
        assert tier.staging[POOL_FULL].available_size() == free_before
    finally:
        pf.stop()


def test_a_backend_that_raises_becomes_a_miss():
    """A storage outage must look like an empty cache, not an exception on the
    scheduler thread."""
    tier, _ = _stocked()

    class Down(FakeStorage):
        def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
            raise RuntimeError("backend down")

    _swap_storage(tier, Down())
    pf = L3Prefetcher(tier)
    try:
        pf.start("r1", ["h0"])
        status, _ = _await(pf, "r1", want=Status.MISS)
        assert status is Status.MISS
    finally:
        pf.release("r1")
        pf.stop()


def test_a_failed_fetch_leaks_no_staging():
    tier, _ = _stocked()
    free_before = tier.staging[POOL_FULL].available_size()

    class Down(FakeStorage):
        def batch_get_v2(self, transfers, extra_info=None):
            raise RuntimeError("backend down")

    _swap_storage(tier, Down())
    pf = L3Prefetcher(tier)
    try:
        pf.start("r1", ["h0"])
        _await(pf, "r1", want=Status.MISS)
        assert tier.staging[POOL_FULL].available_size() == free_before
    finally:
        pf.release("r1")
        pf.stop()


def test_starting_the_same_key_twice_does_not_double_fetch():
    tier, _ = _stocked()
    pf = L3Prefetcher(tier)
    try:
        assert pf.start("r1", ["h0"]) is True
        assert pf.start("r1", ["h0"]) is True
        _await(pf, "r1")
        assert len(pf._entries) == 1
    finally:
        pf.release("r1")
        pf.stop()


def test_an_empty_hash_list_starts_nothing():
    tier, _ = _stocked()
    pf = L3Prefetcher(tier)
    assert pf.start("r1", []) is False
    assert pf._thread is None
    pf.stop()
