"""DSV4 prefix pages to and from an L3 storage tier.

What this buys, stated narrowly so it is not oversold: a prefix survives the
process. In-VRAM reuse is already near-total on a warm engine — a follow-up turn
matches its history in the radix tree and reprefills almost nothing — so L3 is
not a speed-up over an L1 hit and never will be. It is what a restart, an
eviction, or a second replica has to fall back on, and today those fall back on
recomputing the whole prompt.

Shape of one page's worth of storage:

    <layout_signature>/dsv4_full/<page_hash>       compressed KV + indexer KV
    <layout_signature>/dsv4_window/<page_hash>     window KV + both state rings

Two objects, not five, because the five tiers split cleanly by lifetime — see
`dsv4_codec`. The layout signature is a prefix rather than a suffix so that a
geometry change leaves a whole namespace behind rather than interleaving with
the pages still in use.

The staging pool is a pinned host buffer, not a cache. Under the L3-only scope
there is no L2 tier: nothing is kept there between operations, nothing is
evicted from it, and its only job is to give the zero-copy storage interface a
DMA-able address to read from and write into. Sizing it is therefore a
throughput knob, not a hit-rate one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from ..dsv4_paged_pool import DSV4PagedKVCache
from .dsv4_codec import POOL_FULL, POOL_WINDOW, DSV4PageCodec
from .host_pool import HostKVCache
from .storage import HiCacheStorage, PoolHitPolicy, PoolTransfer

logger = logging.getLogger(__name__)


@dataclass
class WriteReport:
    """What actually landed. Never a bare success flag.

    A page that failed to write must not be treated as cached: the next process
    would ask for it, miss, and reprefill — which is correct but slow — while a
    caller that assumed success could instead mark the prefix stored and skip
    the write forever.
    """

    attempted: dict[str, int] = field(default_factory=dict)
    stored: dict[str, int] = field(default_factory=dict)
    skipped_present: dict[str, int] = field(default_factory=dict)
    skipped_not_resident: int = 0

    @property
    def complete(self) -> bool:
        return all(self.stored.get(k, 0) + self.skipped_present.get(k, 0) == n
                   for k, n in self.attempted.items())

    def __str__(self) -> str:
        parts = [
            f"{p}: {self.stored.get(p, 0)} stored"
            f" + {self.skipped_present.get(p, 0)} already there"
            f" / {n} attempted"
            for p, n in self.attempted.items()
        ]
        if self.skipped_not_resident:
            parts.append(f"{self.skipped_not_resident} pages no longer in the window")
        return "; ".join(parts)


class DSV4L3Tier:
    """Moves whole DSV4 pages between the device pool and an L3 backend."""

    def __init__(
        self,
        pool: DSV4PagedKVCache,
        storage: HiCacheStorage,
        *,
        staging_pages: int = 8,
    ) -> None:
        if staging_pages < 1:
            raise ValueError("need at least one staging page")
        self.pool = pool
        self.storage = storage
        self.codec = DSV4PageCodec(pool)
        self.staging_pages = staging_pages

        # One staging pool per storage pool: the two carry different bytes per
        # page, and `get_data_page` derives its stride from that.
        self.staging: dict[str, HostKVCache] = {}
        for name, page_bytes in (
            (POOL_FULL, self.codec.full_page_bytes),
            (POOL_WINDOW, self.codec.window_page_bytes),
        ):
            # Sized by bytes, not by a ratio against a device pool. There is no
            # L2 tier under the L3-only scope: this buffer holds one chunk in
            # flight and nothing between calls, so `device_size` — which exists
            # for the load-back deadlock the constructor guards against — is
            # zero here rather than a number invented to satisfy the guard.
            host = HostKVCache(
                device_size=0,
                page_size=pool.P,
                bytes_per_token=page_bytes // pool.P,
                host_size_bytes=staging_pages * page_bytes,
            )
            self.staging[name] = host
            self.storage.register_mem_host_pool_v2(host, name)

        for name, page_bytes in (
            (POOL_FULL, self.codec.full_page_bytes),
            (POOL_WINDOW, self.codec.window_page_bytes),
        ):
            got = self.staging[name].bytes_per_page
            if got != page_bytes:
                raise ValueError(
                    f"{name}: staging page is {got} bytes, a codec page is {page_bytes}. "
                    f"A DSV4 page is {page_bytes} bytes and must divide evenly by the "
                    f"page size {pool.P} for the pool's token-slot addressing to hold."
                )

    # ----- keys ------------------------------------------------------------ #

    def key(self, pool_name: str, page_hash: str) -> str:
        """The storage name of one page.

        The layout signature leads because it is the compatibility boundary: two
        engines with different head dims or compress ratios produce blobs of the
        same length for the same tokens, and reading one as the other is silent
        corruption rather than a miss.
        """
        return f"{self.codec.layout_signature}/{pool_name}/{page_hash}"

    # ----- write ----------------------------------------------------------- #

    def write_pages(
        self,
        pages: list[tuple[int, str]],
        *,
        skip_existing: bool = True,
    ) -> WriteReport:
        """Store `(full_page_base, page_hash)` pairs, oldest prefix first.

        Order matters and is the caller's responsibility: `batch_exists_v2`
        answers "how many consecutive pages from the start exist", so a caller
        that shuffles the list gets a meaningless prefix length back.
        """
        report = WriteReport()
        if not pages:
            return report

        for pool_name in (POOL_FULL, POOL_WINDOW):
            todo = self._resident_subset(pages, pool_name, report)
            if not todo:
                continue
            report.attempted[pool_name] = len(todo)
            if skip_existing:
                have = self._existing_prefix(pool_name, todo)
                if have:
                    report.skipped_present[pool_name] = have
                    todo = todo[have:]
            stored = 0
            for chunk in _chunks(todo, self.staging_pages):
                stored += self._write_chunk(pool_name, chunk)
            report.stored[pool_name] = stored

        if not report.complete:
            logger.warning("DSV4 L3 write incomplete — %s", report)
        return report

    def _resident_subset(
        self, pages: list[tuple[int, str]], pool_name: str, report: WriteReport
    ) -> list[tuple[int, str]]:
        """The pages this pool can actually supply.

        FULL always can. WINDOW cannot for pages that have slid out, and those
        are the OLDEST — so the window's contribution is a suffix, which is
        exactly what `PoolHitPolicy.TRAILING_PAGES` describes on the way back in.
        Dropping them is normal operation, not a failure.
        """
        if pool_name == POOL_FULL:
            return list(pages)
        keep = [p for p in pages if self.codec.window_resident(p[0])]
        report.skipped_not_resident = len(pages) - len(keep)
        return keep

    def _existing_prefix(self, pool_name: str, todo: list[tuple[int, str]]) -> int:
        """How many leading pages the backend already holds.

        Only a prefix is skippable: pages are chained, so a gap in the middle
        cannot be filled by writing the tail — a later reader stops at the gap
        regardless.
        """
        keys = [self.key(pool_name, h) for _, h in todo]
        transfer = PoolTransfer(
            name=pool_name,
            keys=keys,
            hit_policy=(
                PoolHitPolicy.ALL_PAGES if pool_name == POOL_FULL
                else PoolHitPolicy.TRAILING_PAGES
            ),
        )
        try:
            result = self.storage.batch_exists_v2([transfer])
        except NotImplementedError:
            return 0
        return int(getattr(result, "kv_hit_pages", 0) or 0)

    def _write_chunk(self, pool_name: str, chunk: list[tuple[int, str]]) -> int:
        host = self.staging[pool_name]
        slots = host.alloc(len(chunk) * self.pool.P)
        if slots is None:
            raise RuntimeError(
                f"{pool_name}: staging pool could not supply {len(chunk)} pages. "
                "It is sized for one chunk at a time and nothing should hold it "
                "between calls; a failure here means a previous write leaked its "
                "allocation."
            )
        try:
            host_indices = slots[:: self.pool.P].contiguous()
            for i, (base, _) in enumerate(chunk):
                blob = self.codec.gather(base, pool_name)
                assert blob is not None, "residency was checked before staging"
                host.set_from_flat_data_page(int(host_indices[i].item()), blob)
            transfer = PoolTransfer(
                name=pool_name,
                host_indices=host_indices,
                keys=[self.key(pool_name, h) for _, h in chunk],
            )
            results = self.storage.batch_set_v2([transfer])
            return sum(1 for ok in results.get(pool_name, []) if ok)
        finally:
            host.free(slots)


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]
