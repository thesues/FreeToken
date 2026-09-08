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
