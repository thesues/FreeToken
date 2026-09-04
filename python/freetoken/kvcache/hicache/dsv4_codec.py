"""Serialise one page of DeepSeek-V4 KV out of the paged pool, and put it back.

Every other family in this engine keeps a page of KV in one buffer, so an L3
tier can address it with a single index. DSV4 does not: a page lives in up to
five per-layer structures whose presence depends on that layer's compress
ratio, and two of them are rings.

    window_pool[L]         every layer            the sliding window's KV
    cmp_pool[L]            ratio != 0             compressed KV, whole history
    state_ring[L]          ratio != 0             the attention compressor's carry
    idx_pool[L]            ratio == 4             the indexer's own KV
    indexer_state_ring[L]  ratio == 4             the indexer compressor's carry

That is why the v2 pool interface exists. The five split cleanly in two by
lifetime, and the split is what makes them addressable:

    FULL tier    cmp + idx        indexed off the full-history anchor, so a page
                                  keeps its rows for as long as the sequence
                                  lives -> PoolHitPolicy.ALL_PAGES
    WINDOW tier  window + rings   indexed off the sliding window, so only the
                                  tail of a sequence has them at all
                                  -> PoolHitPolicy.TRAILING_PAGES

The rings look like the hard part and are not. `CompressStateRing.get_blocks`
takes a block base of `(window_slot // P) * ring_size`, which is a pure function
of the slot -- no dependence on write history -- and the pool guarantees window
pages are 1:1 page-bound to full pages, so distinct pages own disjoint blocks.
A page's carry state is therefore self-contained and movable. `set_blocks` puts
it back. Those two primitives already existed for the copy-on-write path; this
module is mostly addressing arithmetic on top of them.

Blobs come out as flat `uint8`. A page mixes bf16 tier data with fp32 ring
state, and an L3 blob is bytes regardless -- keeping the byte view here means
the storage layer never has to know the tier dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..dsv4_paged_pool import DSV4PagedKVCache

# Pool names for `PoolTransfer` / `register_mem_host_pool_v2`. Two, not five:
# the tiers that share a lifetime also share a hit policy, and splitting them
# further would put five keys and five round trips behind every page.
POOL_FULL = "dsv4_full"
POOL_WINDOW = "dsv4_window"


@dataclass(frozen=True)
class _Span:
    """One contiguous run of rows in one tier, and where it sits in the blob."""

    layer: int
    tier: str  # "cmp" | "idx" | "window" | "state" | "istate"
    rows: int
    width: int
    itemsize: int
    offset: int  # byte offset into the pool's page blob

    @property
    def nbytes(self) -> int:
        return self.rows * self.width * self.itemsize


class DSV4PageCodec:
    """Page <-> flat bytes, for the two DSV4 L3 pools.

    The layout is computed once from the pool's geometry and then fixed: a blob
    written by one process must be readable by another, so nothing here may
    depend on runtime state. `layout_signature` is the guard — it goes in the
    storage key, and a pool built with different geometry produces a different
    signature rather than silently mis-parsing someone else's bytes.
    """

    def __init__(self, pool: DSV4PagedKVCache) -> None:
        self.pool = pool
        self.P = pool.P
        self._full: list[_Span] = []
        self._window: list[_Span] = []

        # Validate before building anything off these ratios: every span below is
        # a contiguous slice, and that only holds if a page maps onto a whole
        # number of compressed rows.
        for L, ratio in enumerate(pool.compress_ratios[: pool._n_layers]):
            if ratio and self.P % ratio:
                raise ValueError(
                    f"layer {L}: compress ratio {ratio} does not divide page size {self.P}; "
                    "a page would straddle a compressed row and could not be serialised"
                )

        kv_itemsize = pool.window_pool[0].element_size()
        head_dim = pool.head_dim
        index_head_dim = pool.index_head_dim

        off = 0
        for L, ratio in enumerate(pool.compress_ratios[: pool._n_layers]):
            if ratio == 0:
                continue
            # Compressed KV: one row per `ratio` full tokens, so P // ratio rows
            # for a page. The divisibility that makes this exact is checked above.
            self._full.append(_Span(L, "cmp", self.P // ratio, head_dim, kv_itemsize, off))
            off += self._full[-1].nbytes
            if ratio == 4:
                self._full.append(
                    _Span(L, "idx", self.P // 4, index_head_dim, kv_itemsize, off)
                )
                off += self._full[-1].nbytes
        self.full_page_bytes = off

        off = 0
        for L, ratio in enumerate(pool.compress_ratios[: pool._n_layers]):
            self._window.append(_Span(L, "window", self.P, head_dim, kv_itemsize, off))
            off += self._window[-1].nbytes
            if ratio == 0:
                continue
            ring = pool.state_ring[L]
            assert ring is not None
            self._window.append(
                _Span(L, "state", pool.ring_size(L), ring.buffer.shape[1],
                      ring.buffer.element_size(), off)
            )
            off += self._window[-1].nbytes
            if ratio == 4:
                iring = pool.indexer_state_ring[L]
                assert iring is not None
                self._window.append(
                    _Span(L, "istate", 8, iring.buffer.shape[1],
                          iring.buffer.element_size(), off)
                )
                off += self._window[-1].nbytes
        self.window_page_bytes = off


    @property
    def layout_signature(self) -> str:
        """Identifies the byte layout. Belongs in the storage key.

        Two engines that disagree on layers, ratios, head dims or page size
        produce different blobs of the same length for the same tokens. Without
        this in the key, one would read the other's bytes as its own.
        """
        ratios = ",".join(str(r) for r in self.pool.compress_ratios[: self.pool._n_layers])
        return (
            f"dsv4-P{self.P}-h{self.pool.head_dim}-i{self.pool.index_head_dim}"
            f"-r{ratios}-f{self.full_page_bytes}-w{self.window_page_bytes}"
        )

    # ----- addressing ------------------------------------------------------ #

    def _tier_rows(self, span: _Span, full_page_base: int) -> tuple[torch.Tensor, int]:
        """(buffer, first row) for this span's slice of the page.

        Every tier is a contiguous run, which is why a page can be moved with
        slices rather than a gather: `cmp_rows` is floor division and the page
        base is page-aligned, so P consecutive full locs map onto P//ratio
        consecutive compressed rows.
        """
        pool, L = self.pool, span.layer
        if span.tier == "cmp":
            ratio = pool.compress_ratios[L]
            base = full_page_base // ratio
            buf = pool.cmp_pool[L]
            assert buf is not None
            if base + span.rows > pool.cmp_scratch_base[L]:
                raise IndexError(
                    f"layer {L}: page at {full_page_base} reaches the cmp scratch region "
                    f"(rows {base}..{base + span.rows} vs scratch at {pool.cmp_scratch_base[L]})"
                )
            return buf, base
        if span.tier == "idx":
            base = full_page_base // 4
            buf = pool.idx_pool[L]
            assert buf is not None
            if base + span.rows > pool.idx_scratch_base[L]:
                raise IndexError(
                    f"layer {L}: page at {full_page_base} reaches the idx scratch region"
                )
            return buf, base
        # WINDOW tier: all three are keyed off the window slot the page is bound to.
        win = self._window_page_base(full_page_base)
        if span.tier == "window":
            return pool.window_pool[L], win
        ring = pool.state_ring[L] if span.tier == "state" else pool.indexer_state_ring[L]
        assert ring is not None
        rs = pool.ring_size(L) if span.tier == "state" else 8
        return ring.buffer, (win // self.P) * rs

    def _window_page_base(self, full_page_base: int) -> int:
        """The window slot this full page is bound to, or -1 if it is not resident.

        Page-atomic by construction (`bind_window_pages` binds a whole page), so
        reading the first slot is enough — but a partially bound page would be a
        correctness hazard rather than a performance one, so it is checked.
        """
        slots = self.pool.full_to_window[full_page_base : full_page_base + self.P]
        first = int(slots[0].item())
        if first < 0:
            return -1
        expected = torch.arange(
            first, first + self.P, dtype=slots.dtype, device=slots.device
        )
        if not torch.equal(slots, expected):
            raise RuntimeError(
                f"full page {full_page_base} is not contiguously window-bound "
                f"(first slot {first}); the page-atomic invariant this codec relies on is broken"
            )
        return first

    def window_resident(self, full_page_base: int) -> bool:
        """Is this page still inside the sliding window?

        Pages that have slid out keep their FULL tier and lose their WINDOW
        tier. That is the whole reason the two are separate pools with different
        hit policies, and a caller must not treat a missing window blob as an
        error.
        """
        return self._window_page_base(full_page_base) >= 0

    # ----- gather / scatter ------------------------------------------------ #

    def gather(self, full_page_base: int, pool_name: str) -> torch.Tensor | None:
        """One page of one tier, as flat bytes. `None` if it is not resident.

        Bytes rather than tensors because a page mixes bf16 tier data with fp32
        ring state; the storage layer stays dtype-agnostic.
        """
        spans, nbytes = self._spans_for(pool_name)
        if pool_name == POOL_WINDOW and not self.window_resident(full_page_base):
            return None
        out = torch.empty(nbytes, dtype=torch.uint8)
        for sp in spans:
            buf, base = self._tier_rows(sp, full_page_base)
            rows = buf[base : base + sp.rows]
            out[sp.offset : sp.offset + sp.nbytes] = rows.reshape(-1).view(torch.uint8)
        return out

    def scatter(self, full_page_base: int, pool_name: str, blob: torch.Tensor) -> None:
        """Write a page back. The inverse of `gather`, same layout."""
        spans, nbytes = self._spans_for(pool_name)
        if blob.numel() != nbytes:
            raise ValueError(
                f"{pool_name}: blob is {blob.numel()} bytes, layout wants {nbytes} — "
                "the writer's geometry differs from this pool's (see layout_signature)"
            )
        if pool_name == POOL_WINDOW and not self.window_resident(full_page_base):
            raise RuntimeError(
                f"page {full_page_base} has no window slots bound; allocate the window page "
                "before restoring its blob"
            )
        for sp in spans:
            buf, base = self._tier_rows(sp, full_page_base)
            dst = buf[base : base + sp.rows]
            src = blob[sp.offset : sp.offset + sp.nbytes].view(dst.dtype).reshape(dst.shape)
            dst.copy_(src)
        if pool_name == POOL_WINDOW:
            # `set_blocks`/`set` re-clear the scratch row after every write, and
            # a restore that skipped it would leave whatever the previous
            # sequence left in the row the compressor treats as empty.
            for L, ratio in enumerate(self.pool.compress_ratios[: self.pool._n_layers]):
                if ratio == 0:
                    continue
                self.pool.state_ring[L]._clear_scratch()
                if ratio == 4:
                    self.pool.indexer_state_ring[L]._clear_scratch()

    def _spans_for(self, pool_name: str) -> tuple[list[_Span], int]:
        if pool_name == POOL_FULL:
            return self._full, self.full_page_bytes
        if pool_name == POOL_WINDOW:
            return self._window, self.window_page_bytes
        raise KeyError(f"unknown DSV4 pool {pool_name!r}")
