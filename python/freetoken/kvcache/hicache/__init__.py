"""HiCache: hierarchical KV cache (L1 GPU -> L2 pinned host -> L3 external store).

Ported from sglang so that storage backends written against sglang's
`HiCacheStorage` load into FreeToken unmodified — see storage.py for why the ABC
is kept byte-compatible.

Status: L3 interface, backend factory, operation primitives, page hashing and
the L2 host pool are in place. The controller, the per-family KV read path and
the scheduler hooks are not yet written.
"""

from freetoken.kvcache.hicache.hashing import chain_page_hashes, get_hash_str
from freetoken.kvcache.hicache.storage import HiCacheStorage, HiCacheStorageConfig

__all__ = [
    "HiCacheStorage",
    "HiCacheStorageConfig",
    "get_hash_str",
    "chain_page_hashes",
]
