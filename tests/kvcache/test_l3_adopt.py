"""`CacheManager.adopt_l3_prefix` — the consumer of a ready L3 fetch.

This path had no test, and it had never run: every fetch it was handed was
rejected by a length check that compared a page count against a token count, so
the hit rate stayed at zero and nothing downstream was ever exercised. Both
tests here fail on the pre-fix code, one for the rejection and one for the leak
that followed it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from test_dsv4_l3 import FakeStorage, P, RATIOS, _bind, _scramble  # noqa: E402

from freetoken.core import Req, SamplingParams  # noqa: E402
from freetoken.kvcache.dsv4_cost_model import dsv4_pool_sizes  # noqa: E402
from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache  # noqa: E402
from freetoken.kvcache.hicache.dsv4_l3 import DSV4L3Tier  # noqa: E402
from freetoken.kvcache.hicache.l3_prefetch import L3Prefetcher, Status  # noqa: E402
from freetoken.models.deepseek_v4.args import DeepseekV4Args  # noqa: E402
from freetoken.scheduler.cache import CacheManager  # noqa: E402

DEVICE = torch.device("cpu")
MRR = 4
HASHES = ["a0", "a1", "a2"]


def _stack(num_pages=32):
    args = DeepseekV4Args(
        n_layers=8, compress_ratios=RATIOS, max_seq_len=8192,
        head_dim=512, index_head_dim=128, window_size=P,
    )
    sizes = dsv4_pool_sizes(num_pages=num_pages + 1, args=args, swa_ratio=1.0, P=P)
    pool = DSV4PagedKVCache(sizes=sizes, args=args, device=DEVICE, P=P, n_scratch=MRR + 1)
    pool._init_paged_state(MRR, True)
    pt = torch.zeros(MRR + 1, args.max_seq_len, dtype=torch.int32)
    pt[MRR].fill_(num_pages * P)
    pool.full_loc_map = pt
    cm = CacheManager(num_pages=num_pages, page_size=P, page_table=pt, type="swa_radix",
                      swa_pool=pool, sliding_window_size=P)
    return cm, pool, pt


def _stocked(cm, pool):
    """A tier whose backend holds HASHES, written the way the scheduler writes.

    One growing prefix per chunk, so every page is some commit's trailing page
    and carries window rows — which is what makes the whole run restorable.
    """
    _bind(pool, len(HASHES))
    tier = DSV4L3Tier(pool, FakeStorage(), staging_pages=8)
    pages = [(i * P, h) for i, h in enumerate(HASHES)]
    for i in range(1, len(pages) + 1):
        assert tier.write_pages(pages[:i]).complete
    # The pool now has to look like a fresh process: the bytes are someone
    # else's and nothing is bound, exactly as after a restart.
    _scramble(pool, 7)
    pool.reset_paged_state() if hasattr(pool, "reset_paged_state") else None
    pool.full_to_window.fill_(-1)
    cm.l3_prefetcher = L3Prefetcher(tier, deadline_s=30.0)
    return tier


def _ready(cm, uid):
    pf = cm.l3_prefetcher
    assert pf.start(uid, HASHES)
    for _ in range(2000):
        status, ready = pf.poll(uid)
        if status is not Status.WAITING:
            return status, ready
        time_sleep()
    raise AssertionError("the fetch never settled")


def time_sleep():
    import time
    time.sleep(0.005)


def _req(uid, n_tokens):
    ids = torch.arange(n_tokens, dtype=torch.int32)
    r = Req(input_ids=ids, table_idx=uid, cached_len=0, output_len=1,
            uid=uid, sampling_params=SamplingParams(), cache_handle=None)
    r.input_len = n_tokens
    return r


def test_a_ready_fetch_is_actually_adopted():
    """The whole point of the tier, and the thing that never happened.

    `_allocate` returns PAGE bases; the rest of the path works in token slots.
    Comparing the two rejected every fetch, so this asserts the prefix really
    grows — not merely that the call returned.
    """
    cm, pool, _ = _stack()
    _stocked(cm, pool)
    req = _req(1, len(HASHES) * P)
    status, ready = _ready(cm, req.uid)
    assert status is Status.READY, status
    assert ready.n_pages == len(HASHES), ready

    handle = cm.match_req(req).cuda_handle
    assert handle.cached_len == 0, "nothing is in L1 yet"
    _, cached = cm.adopt_l3_prefix(req, handle)
    # One page short of the fetch, and that is right: a radix match never covers
    # the whole sequence — something has to be left to compute.
    assert cached == (len(HASHES) - 1) * P, f"adopted {cached}, wanted a prefix"


def test_a_failed_adoption_returns_every_slot_it_took():
    """A rejected adoption must leave the pool exactly as it found it.

    `_free` strides by `page_size`, so handing it page bases returned one slot
    in `page_size` of what was taken and leaked the rest — a leak that only
    showed up as an eviction assert much later, under a different request.
    """
    cm, pool, _ = _stack()
    tier = _stocked(cm, pool)
    req = _req(2, len(HASHES) * P)
    status, _ = _ready(cm, req.uid)
    assert status is Status.READY

    def boom(*a, **k):
        raise RuntimeError("scatter fails after the slots are taken")

    tier.codec.scatter = boom
    handle = cm.match_req(req).cuda_handle
    before = len(cm.free_slots)
    _, cached = cm.adopt_l3_prefix(req, handle)
    assert cached == 0, "a failed adoption must not claim a prefix"
    assert len(cm.free_slots) == before, (
        f"leaked {before - len(cm.free_slots)} of {len(HASHES)} pages"
    )


def test_a_fetch_that_was_not_ready_yet_gives_its_staging_back():
    """The leak that made the tier look like it hit and adopted nothing.

    `add_one_req` starts the fetch and the SAME scheduler pass polls it, so on
    an idle engine WAITING is the normal answer. Releasing only on a terminal
    status left the entry — and its staging slots — held for the life of the
    process, because continuation chunks never reach `_try_allocate_one` and
    nothing polls that uid again. Ablation: gate the release on
    `status is not Status.WAITING` and this goes red.
    """
    cm, pool, _ = _stack()
    _stocked(cm, pool)
    req = _req(7, len(HASHES) * P)
    pf = cm.l3_prefetcher
    assert pf.start(req.uid, HASHES)          # in flight, not yet ready

    handle = cm.match_req(req).cuda_handle
    cm.adopt_l3_prefix(req, handle)           # polls WAITING and must let go
    assert req.uid not in pf._entries, "the fetch was left holding its slots"

    # And the pool is whole again: a second fetch of the same size can be made.
    for name, host in pf.tier.staging.items():
        got = host.alloc(pf.max_pages * pool.P)
        assert got is not None, f"{name} staging never came back"
        host.free(got)


def test_one_trailing_page_is_not_enough_to_match():
    """Why the window tail is two pages and not one.

    `match_req` matches `input_ids[:input_len - 1]`, so a page-aligned prompt
    offers 127 tokens of its own last page and `align_down(127, P)` is 0. With a
    one-page tail the live run at the end is under a page, the matcher commits
    nothing, and the restore moves every byte for no reuse. This is the ablation
    for `window_tail_pages` returning 2.
    """
    cm, pool, _ = _stack()
    tier = _stocked(cm, pool)
    tier.window_pages = 1                     # the value this used to compute
    req = _req(8, len(HASHES) * P)
    status, _ = _ready(cm, req.uid)
    assert status is Status.READY
    handle = cm.match_req(req).cuda_handle
    _, cached = cm.adopt_l3_prefix(req, handle)
    assert cached == 0, f"a one-page tail should match nothing, got {cached}"


def test_adoption_extends_a_prefix_l1_already_partly_holds():
    """The only shape that exercises the commit's page-table splice.

    `have == 0` short-circuits it, and that is the one case that ever ran — so
    three bugs sat in this path unseen: it read `req.table_idx` from a
    `PendingReq`, which has no such field; it unlocked a handle `match_req`
    never locked, which asserts AFTER `insert` has taken the pages; and it
    locked the handle that `_try_allocate_one` locks again.

    Ablation: splice from `self.page_table[req.table_idx]` and this goes red
    (AttributeError -> the request falls back to its L1 prefix).
    """
    cm, pool, _ = _stack()
    _stocked(cm, pool)
    req = _req(9, len(HASHES) * P)

    # Seed L1 with the first page, so the adoption has to extend rather than
    # start from nothing.
    seed = cm._page_to_token(cm._allocate(1))
    cm.ensure_swa_slots(len(seed))
    cm.swa_pool.alloc_swa(seed)
    cm.prefix_cache.insert(req.input_ids[:P], seed,
                           swa_evicted_seqlen=0, update_kv_after_len=0)
    handle = cm.match_req(req).cuda_handle
    assert handle.cached_len == P, f"L1 seed did not take: {handle.cached_len}"

    status, ready = _ready(cm, req.uid)
    assert status is Status.READY and ready.n_pages == len(HASHES)
    new_handle, cached = cm.adopt_l3_prefix(req, handle)
    assert cached == (len(HASHES) - 1) * P, f"extended to {cached}, not 256"
    # Returned unlocked: the caller (`_try_allocate_one`) is what locks it, and
    # locking here too leaked a lock and a window page on every adoption.
    assert cm.prefix_cache.full_protected == 0, "adopt left a lock behind"


class _FakeDecode:
    def __init__(self, running=()):
        self.running_reqs = set(running)


def _pending(uid, n_tokens):
    """What the admission loop actually iterates — not a `Req`."""
    from freetoken.scheduler.utils import PendingReq
    return PendingReq(uid, torch.arange(n_tokens, dtype=torch.int32),
                      SamplingParams())


def _prefill_mgr(cm, pending, running=()):
    """Just the surface `_l3_fetch_worth_waiting_for` reads."""
    from freetoken.scheduler.prefill import PrefillManager
    m = PrefillManager.__new__(PrefillManager)
    m.cache_manager = cm
    m.decode_manager = _FakeDecode(running)
    m.pending_list = list(pending)
    return m


def test_a_lone_request_waits_a_pass_for_its_own_fetch():
    """Otherwise the fetch is ALWAYS still in flight when it is consulted.

    `add_one_req` starts it and the same scheduler pass admits, so on an idle
    engine the poll is WAITING every time: the fetch lands, is scored a hit, and
    is never used. Holding the request back costs nothing here because nothing
    is behind it — which is the only condition under which this is allowed.
    """
    cm, pool, _ = _stack()
    _stocked(cm, pool)
    req = _pending(11, len(HASHES) * P)
    pf = cm.l3_prefetcher
    assert pf.start(req.uid, HASHES)                    # in flight
    mgr = _prefill_mgr(cm, [req])
    assert mgr._l3_fetch_worth_waiting_for(req), "a lone request refused to wait"

    _ready(cm, req.uid)                                 # once it lands
    assert not mgr._l3_fetch_worth_waiting_for(req), "kept waiting after READY"
    pf.release(req.uid)


def test_a_request_with_company_never_waits():
    """`schedule_next_batch` breaks rather than skips, so waiting with anything
    queued behind would hold up the whole queue — the 'never add latency' rule
    broken at the queue level instead of the request level."""
    cm, pool, _ = _stack()
    _stocked(cm, pool)
    req = _pending(12, len(HASHES) * P)
    other = _pending(13, len(HASHES) * P)
    pf = cm.l3_prefetcher
    assert pf.start(req.uid, HASHES)
    assert not _prefill_mgr(cm, [req, other])._l3_fetch_worth_waiting_for(req), \
        "waited with a request queued behind it"
    assert not _prefill_mgr(cm, [req], running=[object()])._l3_fetch_worth_waiting_for(req), \
        "waited while something was decoding"
    pf.release(req.uid)
