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
from freetoken.utils.logger import init_logger
from dataclasses import dataclass, field

import torch

from ..dsv4_paged_pool import DSV4PagedKVCache
from .dsv4_codec import POOL_FULL, POOL_WINDOW, DSV4PageCodec
from .host_pool import HostKVCache
from .storage import HiCacheStorage, PoolHitPolicy, PoolTransfer

# `init_logger`, not a bare `getLogger`: this process configures no root
# handler, so a bare logger falls back to Python's lastResort handler,
# which drops everything below WARNING. Every INFO line in this package
# — "L3 tier attached", per-lookup results, write progress — was being
# discarded, which is why a cross-restart miss could only be diagnosed
# by counting keys in the cluster by hand.
logger = init_logger(__name__)


class L3Stats:
    """What the tier actually did, counted where it happens.

    The number this exists for is the hit rate, and the reason it is a set of
    counters rather than one is that the ways a lookup can fail to help are not
    interchangeable:

      miss     the pages are not in storage. Nothing to tune; the prefix was
               never written, or was written by an engine with another layout.
      expired  the pages are there and the fetch did not finish in time. The
               deadline or the backend is the problem, not the cache.
      declined the in-flight bound refused it. The bound is the problem.

    Reporting those as one "miss" would hide the only two that a knob can fix.

    `adopted` is separate from `hits` on purpose. A fetch can succeed and still
    not help — admission may find L1 already had those pages, or no room to put
    them. hits/lookups says whether the data is there; adopted/lookups says
    whether it was worth fetching, and only the second justifies the tier.
    """

    __slots__ = ("_lock", "lookups", "hits", "adopted", "misses", "expired",
                 "declined", "errors", "pages_offered", "pages_adopted",
                 "bytes_read", "writes", "pages_stored", "pages_present",
                 "pages_no_window", "bytes_written", "write_failures",
                 "write_drops",
                 # Why an offered hit was not adopted. `adopted=0` next to
                 # `hits>0` used to conflate five different outcomes, one of
                 # which (a commit that returned a shorter prefix than it
                 # claimed) was being reported as a success.
                 "late", "redundant", "no_room", "adopt_errors", "adopt_short")

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        for f in self.__slots__[1:]:
            setattr(self, f, 0)

    def bump(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def snapshot(self) -> dict:
        with self._lock:
            d = {f: getattr(self, f) for f in self.__slots__[1:]}
        look = d["lookups"] or 1
        d["hit_rate"] = round(d["hits"] / look, 4)
        d["adopt_rate"] = round(d["adopted"] / look, 4)
        return d

    def line(self) -> str:
        """One log line. Rates first, because that is what anyone reads."""
        d = self.snapshot()
        return (
            f"L3 summary: lookups={d['lookups']} hit={d['hit_rate']:.1%} "
            f"adopted={d['adopt_rate']:.1%} | hits={d['hits']} adopted={d['adopted']} "
            f"miss={d['misses']} expired={d['expired']} declined={d['declined']} "
            f"errors={d['errors']} | pages offered={d['pages_offered']} "
            f"adopted={d['pages_adopted']} | read={d['bytes_read'] / 2**20:.1f} MiB | "
            f"not-adopted: late={d['late']} redundant={d['redundant']} "
            f"no-room={d['no_room']} err={d['adopt_errors']} "
            f"short={d['adopt_short']} | "
            f"writes={d['writes']} pages stored={d['pages_stored']} "
            f"present={d['pages_present']} no-window={d['pages_no_window']} "
            f"wrote={d['bytes_written'] / 2**20:.1f} MiB "
            f"failed={d['write_failures']} dropped={d['write_drops']}"
        )


@dataclass
class ReadReport:
    """How much of the candidate prefix came back, and from where.

    `pages` is the answer the scheduler acts on: the number of LEADING pages now
    restored in the device pool. Anything past it must be prefilled. It is never
    optimistic — a page counted here has both tiers it needs already scattered
    in.
    """

    pages: int = 0
    loaded: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        detail = ", ".join(f"{p}: {n}" for p, n in self.loaded.items())
        return f"{self.pages} pages restored ({detail})" if detail else "nothing restored"


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


def window_tail_pages(pool) -> int:
    """Trailing pages of a prefix that must carry window rows.

    Two, and both halves of that matter.

    One is the window itself: `sliding_window_size` is P, so exactly one page
    of a sequence is bound at any time and everything older is tombstoned out
    of it — which is fine, because a tombstoned prefix DOES match. What the
    matcher requires is a live run of at least one full page at the END
    (`SWARadixCache.match_prefix`), not a tombstone-free path.

    The second page pays for the token the matcher drops. `match_req` matches
    `input_ids[:input_len - 1]`, so a page-aligned prompt offers 127 tokens of
    its own last page, `align_down(127, P)` is 0, and a one-page tail matches
    nothing. The engine's own finish-time retention keeps the same two pages.

    This was the pool's capacity (98) until it was measured. That made the
    sidecar's TRAILING_PAGES window wider than any prefix, so `lo` was always 0
    and the policy degenerated into ALL_PAGES anchored at the head: one page
    missing its window rows anywhere near the front zeroed the whole prefix.
    """
    return max(1, -(-pool.sliding_window_size // pool.P) + 1)


class DSV4L3Tier:
    """Moves whole DSV4 pages between the device pool and an L3 backend."""

    def __init__(
        self,
        pool: DSV4PagedKVCache,
        storage: HiCacheStorage,
        *,
        staging_pages: int = 8,
        window_staging_pages: int | None = None,
        writer_pages: int | None = None,
        window_writer_pages: int | None = None,
    ) -> None:
        if staging_pages < 1:
            raise ValueError("need at least one staging page")
        self.pool = pool
        self.storage = storage
        self.codec = DSV4PageCodec(pool)
        self.staging_pages = staging_pages
        # How many trailing pages of a prefix carry window rows. DSV4's window
        # is one page wide (`sliding_window_size` returns P), so this is 1 — and
        # that is the difference between a restore moving 17 MiB and moving 17
        # MiB per page. Everything older keeps its full tier and is tombstoned
        # out of the window on insert, so its window rows are never read back
        # even when they were stored.
        self.window_pages = window_tail_pages(pool)
        # Per pool, because the two are wanted in wildly different quantities: a
        # fetch takes a long prefix from the full tier and a fixed tail from the
        # window tier. One number for both would size the expensive pool by the
        # cheap pool's needs.
        if window_staging_pages is None:
            window_staging_pages = staging_pages
        if window_staging_pages < 1:
            raise ValueError("need at least one window staging page")
        self.staging_capacity = {
            POOL_FULL: staging_pages,
            POOL_WINDOW: window_staging_pages,
        }
        # The writer and the prefetcher allocate from the SAME pools, and a
        # prefetcher holds its slots from the read until the scheduler consumes
        # them — so without a reserved share, fetches in flight can occupy the
        # pool and every write fails while they wait, or one writer chunk takes
        # the pool and every fetch fails. Each side is bounded to its own share;
        # the pool is sized to hold both. Defaults to the whole pool, which is
        # right when there is only one client.
        self.writer_capacity = {
            POOL_FULL: writer_pages or staging_pages,
            POOL_WINDOW: window_writer_pages or window_staging_pages,
        }
        # One per tier, shared by the writer and the prefetcher: they are two
        # halves of the same cache and a hit rate split across two objects is a
        # hit rate nobody computes.
        self.stats = L3Stats()

        # One staging pool per storage pool: the two carry different bytes per
        # page, and `get_data_page` derives its stride from that.
        self.staging: dict[str, HostKVCache] = {}
        for name, page_bytes in (
            (POOL_FULL, self.codec.full_page_bytes),
            (POOL_WINDOW, self.codec.window_page_bytes),
        ):
            capacity = self.staging_capacity[name]
            # Sized by bytes, not by a ratio against a device pool. There is no
            # L2 tier under the L3-only scope: this buffer holds one chunk in
            # flight and nothing between calls, so `device_size` — which exists
            # for the load-back deadlock the constructor guards against — is
            # zero here rather than a number invented to satisfy the guard.
            # OUTSIDE inference mode, deliberately. `launch.py` builds the
            # scheduler under `with torch.inference_mode()`, so a tensor
            # allocated here inherits that and becomes an INFERENCE tensor —
            # and PyTorch forbids an in-place update to one from outside the
            # mode. The L3 writer runs on its own thread, which is outside it,
            # so `set_from_flat_data_page`'s `dst.copy_(...)` raised
            # "Inplace update to inference tensor outside InferenceMode is not
            # allowed" on every write. Reads were unaffected, so the tier
            # looked alive: it logged lookups and misses while storing nothing.
            #
            # These buffers are staging for a host<->storage transfer. They hold
            # nothing between calls and are not part of any autograd or
            # inference graph, so inference-tensor semantics were never wanted
            # here — the mode was inherited by accident of where the tier is
            # constructed, not chosen.
            with torch.inference_mode(False):
                host = HostKVCache(
                    device_size=0,
                    page_size=pool.P,
                    bytes_per_token=page_bytes // pool.P,
                    host_size_bytes=capacity * page_bytes,
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

    def key(self, page_hash: str) -> str:
        """The storage name of one page, WITHOUT a pool segment.

        The layout signature leads because it is the compatibility boundary: two
        engines with different head dims or compress ratios produce blobs of the
        same length for the same tokens, and reading one as the other is silent
        corruption rather than a miss.

        The pool is deliberately absent. The backend adds it — that is the whole
        point of `PoolTransfer.name`, and `batch_exists_v2` probes a sidecar by
        re-scoping the PRIMARY key list into that pool's segment. Embedding the
        pool here made the two disagree: writes went to
        `.../dsv4_window/<sig>/dsv4_window/<hash>` while probes asked for
        `.../dsv4_window/<sig>/kv/<hash>`, so every window page read as absent
        and the prefix collapsed to zero however much was really stored.
        """
        return f"{self.codec.layout_signature}/{page_hash}"

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
                    self.stats.bump(pages_present=have)
                    todo = todo[have:]
            stored = 0
            for chunk in _chunks(todo, self.writer_capacity[pool_name]):
                stored += self._write_chunk(pool_name, chunk)
            report.stored[pool_name] = stored
            # Counted here as well as in `L3Writer`: they are two write paths
            # into the same tier, and a counter that only sees one of them means
            # something different depending on which was used.
            page_bytes = (self.codec.full_page_bytes if pool_name == POOL_FULL
                          else self.codec.window_page_bytes)
            self.stats.bump(writes=1, pages_stored=stored,
                            bytes_written=stored * page_bytes)
            if stored != len(todo):
                self.stats.bump(write_failures=1)

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
        self.stats.bump(pages_no_window=report.skipped_not_resident)
        return keep

    def _existing_prefix(self, pool_name: str, todo: list[tuple[int, str]]) -> int:
        """How many leading pages the backend already holds in THIS pool.

        Only a prefix is skippable: pages are chained, so a gap in the middle
        cannot be filled by writing the tail — a later reader stops at the gap
        regardless.

        A sidecar pool has to be asked as a sidecar. `batch_exists` scopes its
        keys into the DEFAULT segment, so handing it window keys as the primary
        list answers about the full tier; and since a page hash is the same
        string in both pools, that answer is "present" for every window page
        whose full page was already stored — so the window tier stopped being
        written at all, silently, and a restore found its history and no window
        rows.
        """
        keys = [self.key(h) for _, h in todo]
        try:
            if pool_name == POOL_FULL:
                result = self.storage.batch_exists_v2(keys)
                return int(getattr(result, "kv_hit_pages", 0) or 0)
            probe = PoolTransfer(name=pool_name, keys=keys,
                                 hit_policy=PoolHitPolicy.ALL_PAGES)
            result = self.storage.batch_exists_v2(keys, [probe])
            extra = getattr(result, "extra_pool_hit_pages", None) or {}
            return int(extra.get(pool_name, 0))
        except NotImplementedError:
            return 0

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
                keys=[self.key(h) for _, h in chunk],
            )
            results = self.storage.batch_set_v2([transfer])
            return sum(1 for ok in results.get(pool_name, []) if ok)
        finally:
            host.free(slots)


    # ----- read ------------------------------------------------------------ #

    def restorable_prefix(self, pages: list[tuple[int, str]]) -> int:
        """How many LEADING pages of this candidate the backend can supply.

        The two tiers are asked as one question, because the answer is one
        number and the interface already computes it: the full tier is the
        primary key list — every page of a usable prefix needs its history — and
        the window tier rides along as a secondary pool with TRAILING_PAGES,
        since only the sliding tail needs window rows. `kv_hit_pages` comes back
        as the minimum across pools, so a missing window page at the tail
        shortens the prefix rather than being silently ignored.

        Asking the two separately and taking a minimum here would be the same
        arithmetic done worse: it would not know how many trailing pages the
        window actually has to cover, which depends on the window size the
        backend was told about, not on anything visible from this side.
        """
        if not pages:
            return 0
        full_keys = [self.key(h) for _, h in pages]
        # `keys` is read for its LENGTH — it sizes the trailing window the
        # sidecar has to cover; the pages actually probed are `full_keys`
        # re-scoped into this pool's segment.
        #
        # The length is the window width, not a residency probe of the local
        # pool. Residency was the wrong source twice over: a prefetcher has no
        # page bases to probe (its pages are not allocated yet, so it passed
        # zero for all of them and asked one question N times), and the write
        # side now stores only the trailing pages, so a set derived from local
        # residency describes something the backend never held.
        window = PoolTransfer(
            name=POOL_WINDOW,
            keys=[self.key(h) for _, h in pages[-self.window_pages:]],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        try:
            result = self.storage.batch_exists_v2(full_keys, [window])
        except NotImplementedError:
            return 0
        kv_hit = int(getattr(result, "kv_hit_pages", 0) or 0)
        # Logged here, unthrottled, because the periodic summary is not enough
        # to answer "did this lookup hit". That summary is emitted from the
        # scheduler loop, rate-limited, and only while the engine is busy — so
        # after a single request the engine goes idle and the counters it
        # updated are never printed. Diagnosing a cross-restart miss took a
        # full day partly because the one number that settled it was invisible.
        logger.info(
            "L3 restorable_prefix: asked=%d window_trailing=%d -> kv_hit=%d pools=%s",
            len(pages), self.window_pages, kv_hit,
            getattr(result, "extra_pool_hit_pages", None),
        )
        return min(kv_hit, len(pages))

    def read_pages(self, pages: list[tuple[int, str]]) -> ReadReport:
        """Restore a prefix into the device pool.

        `pages` are `(full_page_base, page_hash)` for pages the caller has
        ALREADY allocated — window pages included, and bound, because a window
        blob has nowhere to land otherwise. Returns how many leading pages are
        now genuinely in the pool.
        """
        report = ReadReport()
        n = self.restorable_prefix(pages)
        if n == 0:
            return report
        want = pages[:n]

        for pool_name in (POOL_FULL, POOL_WINDOW):
            subset = (
                want if pool_name == POOL_FULL
                else [p for p in want if self.codec.window_resident(p[0])]
            )
            if not subset:
                continue
            got = 0
            for chunk in _chunks(subset, self.staging_capacity[pool_name]):
                got += self._read_chunk(pool_name, chunk)
            report.loaded[pool_name] = got
            if got != len(subset):
                # A short read means the prefix this report describes is not
                # actually in the pool. Reporting the requested length would
                # hand the scheduler a prefix whose tail is uninitialised
                # memory — worse than reprefilling, because it is wrong rather
                # than slow.
                logger.warning(
                    "DSV4 L3 read short on %s: %d of %d pages; not claiming the prefix",
                    pool_name, got, len(subset),
                )
                return ReadReport(pages=0, loaded=report.loaded)
        report.pages = n
        return report

    def _read_chunk(self, pool_name: str, chunk: list[tuple[int, str]]) -> int:
        host = self.staging[pool_name]
        slots = host.alloc(len(chunk) * self.pool.P)
        if slots is None:
            raise RuntimeError(
                f"{pool_name}: staging pool could not supply {len(chunk)} pages; "
                "a previous transfer leaked its allocation."
            )
        try:
            host_indices = slots[:: self.pool.P].contiguous()
            transfer = PoolTransfer(
                name=pool_name,
                host_indices=host_indices,
                keys=[self.key(h) for _, h in chunk],
            )
            results = self.storage.batch_get_v2([transfer]).get(pool_name, [])
            got = 0
            for i, (base, _) in enumerate(chunk):
                if i >= len(results) or not results[i]:
                    # Stop at the first gap rather than scattering past it: the
                    # pages are a chain, and a later page restored over a missing
                    # earlier one would leave the pool holding a prefix that
                    # never existed.
                    break
                blob = host.get_data_page(int(host_indices[i].item()), flat=True)
                self.codec.scatter(base, pool_name, blob.view(torch.uint8))
                got += 1
            return got
        finally:
            host.free(slots)


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]
