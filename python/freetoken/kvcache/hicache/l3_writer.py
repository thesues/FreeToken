"""Push finished prefixes to L3 without stalling the scheduler.

The scheduler is a single synchronous loop with no async handle machinery, and
neither `CacheManager` nor any radix cache takes a lock — they are strictly
single-threaded by design. So a background thread here may do exactly one of
two things: talk over a `queue.Queue` that the scheduler drains at the top of
its loop, or touch device memory under a reference the scheduler granted.

This takes the first option only, and the split is measured rather than
guessed. On the deployed geometry (43 layers, 21 at ratio 4, 20 at 128,
head_dim 512, page 128) a page is 17.86 MiB — but 17.02 MiB of that is the
window tier, and `sliding_window` equals the page size, so exactly ONE page per
sequence is window-resident at a time. A 22k-token prefix is therefore
171 x 0.84 MiB of history plus one 17 MiB window page: about 161 MiB, roughly
7 ms of device-to-host copy at the 23 GiB/s this node measures.

7 ms on the scheduler thread, once at the end of a turn, is a decode step. The
network write is the part that is not: ~40 ms at the KV tier's measured peak,
and far more when it is not. So the gather runs inline and the write runs
behind a queue. That also removes the need for the "backup in flight" reference
class the design notes called for — nothing here reads a device tensor off the
scheduler thread, so eviction cannot race a copy.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

import torch

from .dsv4_codec import POOL_FULL, POOL_WINDOW
from .dsv4_l3 import DSV4L3Tier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingPage:
    pool: str
    key: str
    blob: torch.Tensor  # host bytes, already detached from the device pool


@dataclass
class WriteOutcome:
    """One submission's result, collected by the scheduler at the loop top.

    Carries the sequence's identity so a caller can tell which prefix is now
    durable — a report with no way to attribute it is a log line, not a signal.
    """

    tag: object
    stored: int
    attempted: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.stored == self.attempted


class L3Writer:
    """Gathers on the calling thread, writes on its own.

    Started lazily and stopped explicitly. Not a daemon that outlives its owner:
    a half-written prefix left behind by an abrupt exit is exactly the state
    `batch_exists_v2`'s prefix semantics are designed to tolerate, but leaking
    the thread would keep the storage handle alive past engine shutdown.
    """

    def __init__(self, tier: DSV4L3Tier, *, max_queued_bytes: int = 512 << 20) -> None:
        self.tier = tier
        self.max_queued_bytes = max_queued_bytes
        self._q: queue.Queue = queue.Queue()
        self._out: queue.Queue = queue.Queue()
        self._queued_bytes = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stopping = False
        self.dropped_submissions = 0

    # ----- scheduler side -------------------------------------------------- #

    def submit(self, pages: list[tuple[int, str]], *, tag: object = None) -> bool:
        """Gather `(full_page_base, page_hash)` now; write later.

        Returns False when the queue is over its byte budget. Dropping is the
        correct response to backpressure here: a prefix that misses L3 costs a
        reprefill next time, while blocking the scheduler costs every request
        in flight. The drop is counted, not silent.
        """
        if not pages:
            return True
        with self._lock:
            if self._queued_bytes >= self.max_queued_bytes:
                self.dropped_submissions += 1
                logger.warning(
                    "L3 write queue over budget (%.1f MiB); dropping a %d-page prefix",
                    self._queued_bytes / 2**20, len(pages),
                )
                self.tier.stats.bump(write_drops=1)
                return False

        batch: list[PendingPage] = []
        for pool in (POOL_FULL, POOL_WINDOW):
            for base, page_hash in pages:
                blob = self.tier.codec.gather(base, pool)
                if blob is None:
                    # Window tier only: the page has slid out and keeps its
                    # history instead. Normal, and the reason the two tiers are
                    # separate objects.
                    continue
                batch.append(PendingPage(pool, self.tier.key(page_hash), blob))
        if not batch:
            return True

        nbytes = sum(p.blob.numel() for p in batch)
        with self._lock:
            self._queued_bytes += nbytes
        self._ensure_thread()
        self._q.put((tag, batch, nbytes))
        return True

    def collect(self) -> list[WriteOutcome]:
        """Everything finished since the last call. Never blocks."""
        out: list[WriteOutcome] = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    def queued_bytes(self) -> int:
        with self._lock:
            return self._queued_bytes

    def stop(self, *, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._stopping = True
        self._q.put(None)
        self._thread.join(timeout=timeout)
        self._thread = None

    # ----- writer side ----------------------------------------------------- #

    def _ensure_thread(self) -> None:
        if self._thread is None and not self._stopping:
            self._thread = threading.Thread(
                target=self._run, name="l3-writer", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            tag, batch, nbytes = item
            try:
                stored = self._write(batch)
                self.tier.stats.bump(writes=1, pages_stored=stored, bytes_written=nbytes)
                if stored != len(batch):
                    self.tier.stats.bump(write_failures=1)
                self._out.put(WriteOutcome(tag, stored, len(batch)))
            except Exception as e:  # noqa: BLE001
                # A storage failure must not take the engine with it. The prefix
                # simply is not durable, which the next reader discovers as a
                # miss — the outcome a caller would have reached anyway.
                logger.warning("L3 write failed: %s", e)
                self.tier.stats.bump(write_failures=1)
                self._out.put(WriteOutcome(tag, 0, len(batch), error=str(e)))
            finally:
                with self._lock:
                    self._queued_bytes -= nbytes

    def _write(self, batch: list[PendingPage]) -> int:
        from .storage import PoolTransfer

        stored = 0
        for pool in (POOL_FULL, POOL_WINDOW):
            pages = [p for p in batch if p.pool == pool]
            if not pages:
                continue
            host = self.tier.staging[pool]
            for chunk in _chunks(pages, self.tier.staging_pages):
                slots = host.alloc(len(chunk) * self.tier.pool.P)
                if slots is None:
                    raise RuntimeError(
                        f"{pool}: staging pool exhausted. Only this thread allocates "
                        "from it, so this means a previous write left an allocation."
                    )
                try:
                    idx = slots[:: self.tier.pool.P].contiguous()
                    for i, p in enumerate(chunk):
                        host.set_from_flat_data_page(int(idx[i].item()), p.blob)
                    results = self.tier.storage.batch_set_v2(
                        [PoolTransfer(name=pool, host_indices=idx,
                                      keys=[p.key for p in chunk])]
                    )
                    stored += sum(1 for ok in results.get(pool, []) if ok)
                finally:
                    host.free(slots)
        return stored


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]
