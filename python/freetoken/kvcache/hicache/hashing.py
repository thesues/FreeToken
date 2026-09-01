"""Page hashing for the L3 key namespace — ported from sglang.

Upstream: sglang/python/sglang/srt/mem_cache/utils.py (`get_hash_str`,
`hash_str_to_int64`) @ a757c1e3f.

**This must stay bit-identical to sglang's.** The hash IS the L3 key: a backend
holds pages written by whoever ran before, and a drifted hash silently turns
every lookup into a miss (best case) or collides distinct prefixes (worst).
The EAGLE bigram branch is dropped — FreeToken has no speculative-decoding path
that feeds tuples in here — but the scalar encoding is untouched: 4-byte
little-endian unsigned per token id.

The chaining is the load-bearing part. Each page's hash folds in its parent's,
so a hash identifies a whole prefix rather than one page's contents. That is
what makes `batch_exists`'s "number of consecutive existing keys from the start"
answer a prefix-length question, and it is why a page can never be reused under
a different prefix.

FreeToken has no equivalent of this today: its radix tree keys children on a
Python tuple of the page's token ids (`_get_key_fn` in radix_cache.py), which is
an in-process dict key — not stable across processes, not a content digest, and
therefore unusable as a shared-storage identity.
"""

import hashlib
from typing import List, Optional


def get_hash_str(token_ids: List[int], prior_hash: Optional[str] = None) -> str:
    """Hash one page of token ids, chained onto the parent page's hash.

    `prior_hash` is the previous page's digest (hex) or None at the sequence
    start. Pass exactly `page_size` token ids: a short final page must not be
    hashed, or the same prefix hashes differently depending on where it was cut.
    """
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


def hash_str_to_int64(hash_str: str) -> int:
    """Convert a SHA256 hex digest to a signed 64-bit integer for events.

    Takes the first 16 hex characters (64 bits) into the signed int64 range.
    """
    uint64_val = int(hash_str[:16], 16)
    if uint64_val >= 2**63:
        return uint64_val - 2**64
    return uint64_val


def chain_page_hashes(
    token_ids: List[int], page_size: int, prior_hash: Optional[str] = None
) -> List[str]:
    """Hash a token run into one digest per WHOLE page, chained left to right.

    A trailing partial page is not hashed — it has no stable identity yet, and
    emitting one would make the same prefix hash differently once the page fills.
    Returns an empty list when the run is shorter than one page.
    """
    hashes: List[str] = []
    h = prior_hash
    for start in range(0, len(token_ids) - page_size + 1, page_size):
        h = get_hash_str(token_ids[start : start + page_size], h)
        hashes.append(h)
    return hashes
