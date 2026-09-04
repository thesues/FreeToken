"""Where the L3 writer attaches, and which pages it is given.

The load-bearing rule is the source of the pages. `insert` dedups: a page the
tree already had gets the request's copy freed, and that page can be handed to
the next allocation immediately. Persisting from the request's own indices
would therefore store whatever the next request wrote there. The tree's
`committed_pages` is the canonical answer, and these tests are mostly about
that distinction.
"""

from __future__ import annotations

import torch

from freetoken.kvcache.hicache.hashing import chain_page_hashes
from freetoken.kvcache.swa_radix_cache import SWARadixCache

DEVICE = torch.device("cpu")
PAGE = 4
WINDOW = 8


def _ids(*v):
    return torch.tensor(list(v), dtype=torch.int32, device=DEVICE)


def _idx(n, start=0):
    return torch.arange(start, start + n, dtype=torch.int32, device=DEVICE)


def _cache(hashing=True):
    c = SWARadixCache(device=DEVICE, page_size=PAGE, sliding_window_size=WINDOW)
    c.enable_page_hash = hashing
    return c


def test_committed_pages_names_every_whole_page_of_the_prefix():
    c = _cache()
    ids = _ids(1, 2, 3, 4, 5, 6, 7, 8)
    c.insert(ids, _idx(8, start=100), swa_evicted_seqlen=0)
    pages = c.committed_pages(ids)
    assert [h for _, h in pages] == chain_page_hashes(ids.tolist(), PAGE)
    assert [p for p, _ in pages] == [100, 104], "page base = the first slot of each page"


def test_the_pages_come_from_the_tree_not_from_the_caller():
    """The rule this whole hook depends on.

    A second request with the same prefix has its own copies deduped away by
    `insert`; those slots go back on the free list and the next allocation may
    own them. `committed_pages` must name the tree's slots, which stay valid.
    """
    c = _cache()
    ids = _ids(1, 2, 3, 4, 5, 6, 7, 8)
    c.insert(ids, _idx(8, start=100), swa_evicted_seqlen=0)
    # A second request brings its own pages for the same tokens.
    c.insert(ids, _idx(8, start=900), swa_evicted_seqlen=0)
    pages = c.committed_pages(ids)
    assert [p for p, _ in pages] == [100, 104], (
        "the second request's slots were returned to the free list; naming them "
        "would persist whatever the next allocation writes there"
    )


def test_a_partial_match_yields_only_the_pages_it_covers():
    """Reading a prefix of a node needs no split — the tokens line up from the
    node's start, so the hashes do too."""
    c = _cache()
    long = _ids(1, 2, 3, 4, 5, 6, 7, 8)
    c.insert(long, _idx(8, start=100), swa_evicted_seqlen=0)
    short = _ids(1, 2, 3, 4)
    pages = c.committed_pages(short)
    assert len(pages) == 1
    assert pages[0] == (100, chain_page_hashes(long.tolist(), PAGE)[0])


def test_a_diverging_prefix_stops_where_it_diverges():
    c = _cache()
    c.insert(_ids(1, 2, 3, 4, 5, 6, 7, 8), _idx(8, start=100), swa_evicted_seqlen=0)
    pages = c.committed_pages(_ids(1, 2, 3, 4, 9, 9, 9, 9))
    assert len(pages) == 1, "only the shared page is committed under this prefix"


def test_no_hashes_means_nothing_to_persist():
    """A node inserted while hashing was off has no identity to store under, and
    every page after it is chained onto that missing identity. Stopping is the
    only honest answer; guessing a hash would name someone else's page."""
    c = _cache(hashing=False)
    ids = _ids(1, 2, 3, 4, 5, 6, 7, 8)
    c.insert(ids, _idx(8, start=100), swa_evicted_seqlen=0)
    assert c.committed_pages(ids) == []


def test_the_walk_does_not_restructure_the_tree():
    """`match_prefix` splits nodes and stamps timestamps, which is right for
    serving a request and wrong for deciding what to persist — this runs on the
    scheduler thread inside `cache_req` and must not move anything."""
    c = _cache()
    c.insert(_ids(1, 2, 3, 4, 5, 6, 7, 8), _idx(8, start=100), swa_evicted_seqlen=0)

    def shape(node):
        return (node.length, sorted(shape(ch) for ch in node.children.values()))

    before = shape(c.root)
    c.committed_pages(_ids(1, 2, 3, 4))     # a partial match: the split-inducing case
    assert shape(c.root) == before, "the read-only walk restructured the tree"


def test_an_unknown_prefix_yields_nothing():
    c = _cache()
    assert c.committed_pages(_ids(7, 7, 7, 7)) == []


def test_a_gap_in_the_hashes_stops_the_walk_rather_than_skipping_it():
    """The case that matters, which "hashing was never on" does not cover.

    If an early node has no digest and a later one does, the later hash was
    chained by `prior_hash`, which walks PAST unhashed ancestors — so it folds
    in a shorter prefix than the one it actually sits under. It names a
    different sequence's page. Carrying on would store real KV under that wrong
    name, which is worse than storing nothing: a future reader would hit it.
    """
    c = _cache(hashing=False)
    head = _ids(1, 2, 3, 4)
    c.insert(head, _idx(4, start=100), swa_evicted_seqlen=0)
    assert c.committed_pages(head) == []

    c.enable_page_hash = True                       # turned on mid-run
    longer = _ids(1, 2, 3, 4, 5, 6, 7, 8)
    c.insert(longer, _idx(8, start=200), swa_evicted_seqlen=0)

    tail_node = next(iter(
        n for n in c.root.children.values() for n in n.children.values()
    ), None)
    assert tail_node is not None and tail_node.hash_value, (
        "the later node should have digests; the test needs the mixed tree"
    )
    assert c.committed_pages(longer) == [], (
        "the walk carried past the unhashed head and offered the tail's digest, "
        "which was chained without those tokens"
    )
