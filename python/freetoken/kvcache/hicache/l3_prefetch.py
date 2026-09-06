"""Fetch a prefix from L3 while the scheduler gets on with something else.

The rule this is built around: **a prefetch may save work but must never make a
request slower than it would have been without an L3 tier.** Everything here
follows from that. There is a deadline; missing it admits the request on
whatever L1 already had. There is a bounded number of fetches in flight;
exceeding it declines rather than queues. Nothing blocks the scheduler thread.

The split mirrors the writer. The background thread does the two network round
trips — the existence check and the read — and lands the bytes in the staging
pool. The scheduler thread does the device writes, because that is the only
thread allowed to: `CacheManager` and the radix caches take no locks and the
page table is its alone.

Staging slots are held from the read until the scheduler consumes them, rather
than copied out into private buffers. That is the whole purpose of a pinned
staging area, and copying instead would cost a second pass over ~161 MiB for a
long prefix — the exact overhead this is supposed to avoid. It does mean the
pool has to be sized for the fetches in flight, which `max_inflight` bounds.

Cancellation is cooperative for the same reason. A deadline that fired while
the read is still running cannot take the slots back: the writer is inside
them. It marks the entry abandoned, and the thread frees on its way out.
"""

from __future__ import annotations

import logging
from freetoken.utils.logger import init_logger
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

import torch

from .dsv4_codec import POOL_FULL, POOL_WINDOW
from .dsv4_l3 import DSV4L3Tier
from .storage import PoolTransfer

# `init_logger`, not a bare `getLogger`: this process configures no root
# handler, so a bare logger falls back to Python's lastResort handler,
# which drops everything below WARNING. Every INFO line in this package
# — "L3 tier attached", per-lookup results, write progress — was being
# discarded, which is why a cross-restart miss could only be diagnosed
# by counting keys in the cluster by hand.
logger = init_logger(__name__)


class Status(str, Enum):
    WAITING = "waiting"      # in flight; ask again next iteration
    READY = "ready"          # pages are in staging, take them
    MISS = "miss"            # storage had nothing useful
    EXPIRED = "expired"      # deadline passed; admit on L1 alone


@dataclass
class Ready:
    """Pages sitting in staging, addressed by token slot.

    `pages` is per pool: `(page_index, staging_slot)`. The scheduler allocates
    device pages, scatters from `staging_slot`, and then calls `release`.
    """

    n_pages: int
    slots: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class _Entry:
    hashes: list[str]
    started: float
    status: Status = Status.WAITING
    ready: Ready | None = None
    abandoned: bool = False


class L3Prefetcher:
    """Non-blocking prefix fetch, with a deadline the scheduler can rely on."""

    def __init__(
        self,
        tier: DSV4L3Tier,
        *,
        deadline_s: float = 0.25,
        max_inflight: int = 2,
    ) -> None:
        self.tier = tier
        self.deadline_s = deadline_s
        self.max_inflight = max_inflight
        self._entries: dict[object, _Entry] = {}
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stopping = False
        self.expired = 0
        self.declined = 0

    # ----- scheduler thread ------------------------------------------------ #

    def start(self, key: object, page_hashes: list[str]) -> bool:
        """Begin a fetch. Idempotent per key; False when declined.

        Declining is not a failure — it means the request proceeds on L1 alone,
        which is what it would have done anyway. Queueing instead would turn a
        busy moment into added latency for every request behind it.
        """
        if not page_hashes:
            return False
        with self._lock:
            if key in self._entries:
                return True
            inflight = sum(1 for e in self._entries.values()
                           if e.status is Status.WAITING)
            if inflight >= self.max_inflight:
                self.declined += 1
                self.tier.stats.bump(declined=1)
                return False
            self._entries[key] = _Entry(list(page_hashes), time.monotonic())
        self.tier.stats.bump(lookups=1)
        self._ensure_thread()
        self._q.put(key)
        return True

    def poll(self, key: object) -> tuple[Status, Ready | None]:
        """Where this fetch stands. Never blocks, never waits on the network."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return Status.MISS, None
            if entry.status is Status.WAITING:
                if time.monotonic() - entry.started > self.deadline_s:
                    # Give up on the fetch, not on the request. The pages may
                    # still arrive; `release` will drop them.
                    entry.status = Status.EXPIRED
                    entry.abandoned = True
                    self.expired += 1
                    self.tier.stats.bump(expired=1)
                    return Status.EXPIRED, None
                return Status.WAITING, None
            return entry.status, entry.ready

    def release(self, key: object) -> None:
        """Done with this fetch — hand the staging slots back.

        Must be called for every `start`, including expired ones, or the
        staging pool drains one fetch at a time until nothing can be prefetched
        at all.
        """
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                return
            entry.abandoned = True
            ready = entry.ready
            entry.ready = None
        if ready:
            self._free(ready)

    def stop(self, *, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._stopping = True
        self._q.put(None)
        self._thread.join(timeout=timeout)
        self._thread = None
        for key in list(self._entries):
            self.release(key)

    # ----- fetch thread ---------------------------------------------------- #

    def _ensure_thread(self) -> None:
        if self._thread is None and not self._stopping:
            self._thread = threading.Thread(
                target=self._run, name="l3-prefetch", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            key = self._q.get()
            if key is None:
                return
            with self._lock:
                entry = self._entries.get(key)
            if entry is None or entry.abandoned:
                continue
            try:
                ready = self._fetch(entry)
            except Exception as e:  # noqa: BLE001
                logger.warning("L3 prefetch failed: %s", e)
                self.tier.stats.bump(errors=1)
                ready = None
            with self._lock:
                entry = self._entries.get(key)
                if entry is None or entry.abandoned:
                    # The scheduler stopped waiting. Free rather than keep: an
                    # abandoned fetch that held its slots would starve the next.
                    stale, ready = ready, None
                else:
                    entry.ready = ready
                    entry.status = Status.READY if ready else Status.MISS
                    if ready:
                        # Counted here rather than at the fetch, so a result the
                        # scheduler had already abandoned is not scored a hit.
                        self.tier.stats.bump(
                            hits=1, pages_offered=ready.n_pages,
                            bytes_read=ready.n_pages * (
                                self.tier.codec.full_page_bytes
                                + self.tier.codec.window_page_bytes),
                        )
                    else:
                        self.tier.stats.bump(misses=1)
                    stale = None
            if stale:
                self._free(stale)

    def _fetch(self, entry: _Entry) -> Ready | None:
        pages = [(0, h) for h in entry.hashes]     # bases are the caller's problem
        n = self.tier.restorable_prefix(pages)
        if n == 0:
            return None
        want = entry.hashes[:n]

        slots: dict[str, torch.Tensor] = {}
        try:
            for pool in (POOL_FULL, POOL_WINDOW):
                keys = [self.tier.key(h) for h in want]
                host = self.tier.staging[pool]
                got = host.alloc(len(keys) * self.tier.pool.P)
                if got is None:
                    raise RuntimeError(
                        f"{pool}: staging pool could not supply {len(keys)} pages. "
                        "max_inflight is what bounds this; either it is too high "
                        "for the pool or a previous fetch was never released."
                    )
                slots[pool] = got
                idx = got[:: self.tier.pool.P].contiguous()
                results = self.tier.storage.batch_get_v2(
                    [PoolTransfer(name=pool, host_indices=idx, keys=keys)]
                ).get(pool, [])
                # Stop at the first gap for the same reason the direct read
                # does: the pages are a chain, and a prefix with a hole in it is
                # not a prefix.
                ok = 0
                for r in results:
                    if not r:
                        break
                    ok += 1
                if pool == POOL_FULL and ok < len(keys):
                    n = ok
                    if n == 0:
                        raise _Miss()
                    want = want[:n]
        except _Miss:
            self._free(Ready(0, slots))
            return None
        except Exception:
            self._free(Ready(0, slots))
            raise
        return Ready(n, slots)

    def _free(self, ready: Ready) -> None:
        for pool, got in ready.slots.items():
            self.tier.staging[pool].free(got)


class _Miss(Exception):
    """Storage had the keys a moment ago and not now. Not an error."""
