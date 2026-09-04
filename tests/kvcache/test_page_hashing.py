"""Page hashes on the radix tree: the L3 key namespace.

The tree's own child key is a Python tuple hash — an in-process dict key, not
stable across processes. A shared storage tier needs a name for a page that two
engines agree on. These tests pin the properties that make the chained digest
usable as that name: same prefix same key, different prefix different key, and
survives the tree restructuring itself.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.hicache.hashing import chain_page_hashes, get_hash_str
from freetoken.kvcache.radix_cache import RadixPrefixCache
from freetoken.kvcache.swa_radix_cache import SWARadixCache

DEVICE = torch.device("cpu")
PAGE = 4


def _ids(*vals):
    return torch.tensor(list(vals), dtype=torch.int32, device=DEVICE)


def _idx(n, start=0):
    return torch.arange(start, start + n, dtype=torch.int32, device=DEVICE)


def _cache(page_size=PAGE, hashing=True):
    c = RadixPrefixCache(device=DEVICE, page_size=page_size)
    c.enable_page_hash = hashing
    return c


# --------------------------------------------------------------------------- #
# the digest itself
# --------------------------------------------------------------------------- #
def test_the_same_page_under_a_different_prefix_gets_a_different_digest():
    """The chaining is the point. Without it a page would be reusable under any
    prefix, and a hit would hand back KV computed for different context."""
    a = chain_page_hashes([1, 2, 3, 4, 9, 9, 9, 9], PAGE)
    b = chain_page_hashes([5, 6, 7, 8, 9, 9, 9, 9], PAGE)
    assert a[0] != b[0]
    assert a[1] != b[1], "the second page differs only by what came before it"


def test_a_partial_trailing_page_is_not_hashed():
    """A short page has no stable identity: the same prefix would hash
    differently depending on where the request happened to be cut."""
    assert chain_page_hashes([1, 2, 3], PAGE) == []
    assert len(chain_page_hashes([1, 2, 3, 4, 5], PAGE)) == 1


def test_chaining_matches_hashing_page_by_page():
    """`chain_page_hashes` is a loop over `get_hash_str`; pin that it stays one."""
    toks = [11, 12, 13, 14, 21, 22, 23, 24]
    h1 = get_hash_str(toks[:4], None)
    h2 = get_hash_str(toks[4:], h1)
    assert chain_page_hashes(toks, PAGE) == [h1, h2]


# --------------------------------------------------------------------------- #
# on the tree
# --------------------------------------------------------------------------- #
def test_insert_gives_every_whole_page_a_digest():
    c = _cache()
    ids = _ids(1, 2, 3, 4, 5, 6, 7, 8)
    res = c.insert_prefix(ids, _idx(8))
    node = res.handle.node
    assert node.hash_value == chain_page_hashes(ids.tolist(), PAGE)


def test_hashing_off_leaves_the_field_alone():
    """It costs a SHA-256 per page on the scheduler thread and buys nothing
    without a storage tier, so it must be genuinely off, not merely unused."""
    c = _cache(hashing=False)
    res = c.insert_prefix(_ids(1, 2, 3, 4), _idx(4))
    assert res.handle.node.hash_value is None


def test_a_split_divides_the_digests_where_it_divides_the_tokens():
    """`split_at` restructures the tree under a diverging prefix. The digests
    must follow the tokens exactly — an off-by-one page here turns every key
    after the split into a silent miss.
    """
    c = _cache()
    first = _ids(1, 2, 3, 4, 5, 6, 7, 8)
    c.insert_prefix(first, _idx(8))
    # Diverge after the first page: forces the 2-page node to split at 4.
    second = _ids(1, 2, 3, 4, 9, 9, 9, 9)
    c.insert_prefix(second, _idx(8, start=100))

    want = chain_page_hashes(first.tolist(), PAGE)
    node, _ = c._tree_walk(first)
    got: list[str] = []
    while not node.is_root():
        # Per NODE, not just per path: an off-by-one split leaves the
        # concatenation intact while moving a hash onto a node whose tokens it
        # does not describe, and the storage key for that node would then name
        # someone else's page. Checking only the concatenation misses it.
        assert len(node.hash_value or []) == node.length // PAGE, (
            f"node of {node.length} tokens carries {len(node.hash_value or [])} "
            f"page digests; it should carry {node.length // PAGE}"
        )
        got = (node.hash_value or []) + got
        node = node.parent
    assert got == want, "the digests along the path no longer describe the prefix"


def test_a_shared_prefix_keeps_one_identity_across_both_branches():
    """Two sequences sharing a prefix must agree on that prefix's page keys —
    that agreement is what lets one of them reuse what the other stored."""
    c = _cache()
    c.insert_prefix(_ids(1, 2, 3, 4, 5, 6, 7, 8), _idx(8))
    c.insert_prefix(_ids(1, 2, 3, 4, 9, 9, 9, 9), _idx(8, start=100))

    def path(ids):
        node, _ = c._tree_walk(ids)
        out: list[str] = []
        while not node.is_root():
            out = (node.hash_value or []) + out
            node = node.parent
        return out

    a = path(_ids(1, 2, 3, 4, 5, 6, 7, 8))
    b = path(_ids(1, 2, 3, 4, 9, 9, 9, 9))
    assert a[0] == b[0], "the shared first page must have one key"
    assert a[1] != b[1]


def test_prior_hash_walks_past_nodes_that_carry_none():
    """A node can hold fewer than one page (or none) after a split, so the chain
    has to look further up rather than stop at the parent."""
    c = _cache()
    c.insert_prefix(_ids(1, 2, 3, 4, 5, 6, 7, 8), _idx(8))
    node, _ = c._tree_walk(_ids(1, 2, 3, 4, 5, 6, 7, 8))
    want = chain_page_hashes([1, 2, 3, 4, 5, 6, 7, 8], PAGE)[-1]
    assert node.prior_hash() == want


def test_an_unaligned_split_is_refused_rather_than_shifting_every_key():
    """Both caches align their split positions down to a page. If a future
    caller stops doing that, the digests would quietly stop describing the
    tokens under them — worth an assert, not a wrong answer."""
    c = _cache()
    c.insert_prefix(_ids(1, 2, 3, 4, 5, 6, 7, 8), _idx(8))
    node, _ = c._tree_walk(_ids(1, 2, 3, 4, 5, 6, 7, 8))
    with pytest.raises(AssertionError, match="not page-aligned"):
        node.split_at(3)


# --------------------------------------------------------------------------- #
# the cache DSV4 actually uses
# --------------------------------------------------------------------------- #
def test_the_swa_cache_hashes_the_same_way():
    """DSV4 is forced onto `swa_radix`, so the plain cache being right is not
    enough. Both share `RadixTreeNode`, and this pins that the sharing holds."""
    c = SWARadixCache(device=DEVICE, page_size=PAGE, sliding_window_size=64)
    c.enable_page_hash = True
    ids = _ids(1, 2, 3, 4, 5, 6, 7, 8)
    c.insert(ids, _idx(8), swa_evicted_seqlen=0)
    node = c.root
    got: list[str] = []
    while node.children:
        node = next(iter(node.children.values()))
        got += node.hash_value or []
    assert got == chain_page_hashes(ids.tolist(), PAGE)
