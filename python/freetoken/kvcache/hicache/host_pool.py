"""L2: a pinned, device-mapped host page pool sitting under the GPU KV cache.

Shape of the port
-----------------
sglang subclasses its `HostKVCache` per attention family (MHA / MLA / NSA), because
its L1<->L2 transfer kernels are written against each family's tensor layout.
FreeToken has seven pool families whose layouts do not agree — and, more to the
point, it has **no KV read path at all** (every pool implements `store_kv`, none
implements a gather-out). Since that read path has to be written from scratch
anyway, this pool stays family-agnostic: a flat `[num_pages, bytes_per_page]`
byte slab. Packing a page's K/V for a given family is the caller's job.

That also matches what the L3 side actually needs. `AutumnKVCacheStorage`, the
backend this port exists to reach, touches exactly one method on the host pool:
`get_data_page(idx, flat=True)`, from which it takes a byte view. The other
zero-copy backends are the same shape (flat page + raw pointer).

Layout is `page_first` and only `page_first`: one page's bytes are contiguous, so
`get_data_page` is a single slice and an L3 backend can DMA straight into the
slab. `layer_first` — sglang's default — interleaves layers across a page and
makes zero-copy impossible; there is no reason to carry it here.

Indexing convention (inherited from sglang, easy to get wrong)
-------------------------------------------------------------
`host_indices` are **token** slots, not page numbers, and `get_data_page` takes
the page's *first token* slot. Allocation is in whole pages and contiguous within
a single `alloc()` call. Kept identical to sglang deliberately: the controller
and every storage backend are written against it.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import torch

from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr

logger = logging.getLogger(__name__)


def _synchronized(func):
    """Serialize pool mutation: the controller's prefetch/backup threads call
    alloc/free while the scheduler thread is doing the same."""

    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class HostKVCache:
    """Pinned host page pool addressed by token slot.

    `bytes_per_token` is how many bytes one token's KV occupies once packed by
    the caller's per-family writer — for an MLA/DSA model that is the latent slab
    plus the index-key slab, summed over layers.
    """

    def __init__(
        self,
        *,
        device_size: int,
        page_size: int,
        bytes_per_token: int,
        dtype: torch.dtype = torch.uint8,
        host_size_bytes: Optional[int] = None,
        host_to_device_ratio: float = 3.0,
    ):
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")
        if bytes_per_token < 1:
            raise ValueError(f"bytes_per_token must be >= 1, got {bytes_per_token}")

        self.page_size = int(page_size)
        self.bytes_per_token = int(bytes_per_token)
        self.bytes_per_page = self.bytes_per_token * self.page_size
        self.dtype = dtype
        self.layout = "page_first"
        self.lock = threading.RLock()

        if host_size_bytes is not None:
            token_capacity = int(host_size_bytes) // self.bytes_per_token
        else:
            token_capacity = int(device_size * host_to_device_ratio)

        self.page_num = max(1, token_capacity // self.page_size)
        self.size = self.page_num * self.page_size

        # sglang asserts host > device and the protocol genuinely depends on it:
        # a load-back moves pages L2 -> L1 while their L2 slots stay allocated, so
        # an L2 no larger than L1 can deadlock with every page pinned by an
        # in-flight load. Fail here rather than at 3am.
        if self.size <= device_size:
            raise ValueError(
                f"host pool ({self.size} tokens) must exceed the device pool "
                f"({device_size} tokens); raise host_size_bytes or the ratio"
            )

        itemsize = torch.empty((), dtype=dtype).element_size()
        if self.bytes_per_page % itemsize:
            raise ValueError(
                f"bytes_per_page={self.bytes_per_page} is not a multiple of "
                f"{dtype} itemsize {itemsize}"
            )
        elems_per_page = self.bytes_per_page // itemsize

        # Exact-size pinned + device-mapped. NOT torch.empty(pin_memory=True):
        # its caching allocator rounds up to the next power of two, so a 70 GiB
        # pool would reserve 128 GiB (see freetoken/kernel/pinned.py).
        self.kv_buffer = alloc_pinned_tensor(
            self.page_num, elems_per_page, dtype=dtype
        )
        self.device_ptr = device_ptr(self.kv_buffer)

        self.mem_state: Optional[torch.Tensor] = None
        self.free_slots: Optional[torch.Tensor] = None
        self.clear()

        logger.info(
            "hicache L2: %d pages x %d B = %.2f GiB pinned (page_size=%d, "
            "bytes_per_token=%d)",
            self.page_num,
            self.bytes_per_page,
            self.page_num * self.bytes_per_page / 2**30,
            self.page_size,
            self.bytes_per_token,
        )

    # ── allocation ────────────────────────────────────────────────────────

    @_synchronized
    def clear(self) -> None:
        self.mem_state = torch.zeros((self.size,), dtype=torch.uint8)
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    @_synchronized
    def available_size(self) -> int:
        return int(self.free_slots.numel())

    @_synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Reserve `need_size` token slots (a whole number of pages).

        Returns contiguous slots, or None when the pool is full — the caller is
        expected to evict host pages and retry, not to crash.
        """
        if need_size % self.page_size:
            raise ValueError(
                f"need_size={need_size} must be a multiple of page_size="
                f"{self.page_size}"
            )
        if need_size > self.free_slots.numel():
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @_synchronized
    def free(self, indices: torch.Tensor) -> int:
        if indices.numel() == 0:
            return 0
        self.free_slots = torch.cat([self.free_slots, indices.cpu()])
        return int(indices.numel())

    # ── page views (the whole L3-facing surface) ──────────────────────────

    def get_data_page(self, index: int, flat: bool = True) -> torch.Tensor:
        """View of the page whose FIRST TOKEN is slot `index`.

        `index` is a token slot, not a page number — sglang's convention, which
        every storage backend is written against. Must be page-aligned.
        """
        if index % self.page_size:
            raise ValueError(
                f"index={index} is not page-aligned (page_size={self.page_size})"
            )
        page = self.kv_buffer[index // self.page_size]
        return page.flatten() if flat else page

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """Write a flat page back into the pool at token slot `index`."""
        dst = self.get_data_page(index, flat=True)
        if data_page.numel() != dst.numel():
            raise ValueError(
                f"page size mismatch: got {data_page.numel()} elements, "
                f"expected {dst.numel()}"
            )
        dst.copy_(data_page.flatten())

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """A zeroed, pinned page-sized scratch buffer.

        The controller's non-zero-copy path reads an L3 page into one of these
        and then scatters it in. Pinned so that path stays DMA-able.
        """
        return alloc_pinned_tensor(
            self.bytes_per_page // torch.empty((), dtype=self.dtype).element_size(),
            dtype=self.dtype,
        ).zero_()

    def get_page_buffer_meta(self, indices) -> tuple[list[int], list[int]]:
        """Raw (address, length) pairs for zero-copy RDMA-style backends."""
        itemsize = self.kv_buffer.element_size()
        ptrs, sizes = [], []
        for idx in indices:
            page = self.get_data_page(int(idx), flat=True)
            ptrs.append(page.data_ptr())
            sizes.append(page.numel() * itemsize)
        return ptrs, sizes

    def get_size_per_token(self) -> int:
        return self.bytes_per_token
