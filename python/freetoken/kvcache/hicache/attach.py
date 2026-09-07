"""Build the L3 tier from configuration, and hang it off the cache manager.

One function, and it is deliberately the only place that knows how the pieces
fit together. Everything below it — the codec, the tier, the writer, the
prefetcher — is constructible without any of this and is tested that way; this
is the assembly, and assembly is where an engine usually grows a knot.

Nothing here is on by default. An engine with no `--hicache-*` arguments gets a
cache manager whose `l3_writer` and `l3_prefetcher` stay None, and every call
site is written so that None costs one attribute lookup.

The backend is loaded by module path rather than named in a table, which is how
sglang's `dynamic` backend works and why autumn needs no code here at all: the
storage class is whatever `--hicache-storage-backend-extra-config` names.
"""

from __future__ import annotations

import importlib
import json
import logging
from freetoken.utils.logger import init_logger

from .dsv4_l3 import DSV4L3Tier, window_pages_for
from .l3_prefetch import L3Prefetcher
from .l3_writer import L3Writer

# `init_logger`, not a bare `getLogger`: this process configures no root
# handler, so a bare logger falls back to Python's lastResort handler,
# which drops everything below WARNING. Every INFO line in this package
# — "L3 tier attached", per-lookup results, write progress — was being
# discarded, which is why a cross-restart miss could only be diagnosed
# by counting keys in the cluster by hand.
logger = init_logger(__name__)


def attach_l3(cache_manager, kv_pool, config) -> bool:
    """Wire an L3 tier onto `cache_manager`. False when none is configured.

    Failing to attach is never fatal. A misconfigured or unreachable backend
    leaves the engine exactly as it would have been without the flag — which is
    the same contract every other part of this tier keeps, and the reason it can
    be turned on in production without a fallback plan.
    """
    spec = getattr(config, "hicache_storage_backend", None)
    if not spec:
        return False

    try:
        storage = _load_backend(config)
        # The longest prefix worth fetching is bounded by the device pool it
        # has to land in: `full_to_window` carries one entry per full token
        # slot, so its length in pages is that bound.
        pool_pages = max(1, kv_pool.full_to_window.numel() // kv_pool.P)
        window_pages = window_pages_for(kv_pool)
        # Bounded by the window pool, not the full pool. A prefix longer than
        # the window tier can cover comes back with its head unbound, goes into
        # the tree tombstoned, and matches nothing — all of the transfer cost
        # and none of the benefit.
        max_pages = int(getattr(config, "hicache_prefetch_pages", 0)
                        or min(pool_pages, window_pages))
        max_inflight = int(getattr(config, "hicache_max_inflight", 2))
        # The writer allocates from the same pools as the prefetcher, so it gets
        # a reserved share rather than competing for one: fetches in flight
        # would otherwise hold the pool while every write failed. 32 full pages
        # is a ~27 MiB write chunk.
        writer_full = int(getattr(config, "hicache_writer_pages", 0) or 32)
        budget = int(getattr(config, "hicache_staging_pages", 0) or 0)
        if budget:
            # An explicit budget caps memory; the prefetch shrinks to fit it
            # rather than the tier refusing to attach. A pool that is merely
            # smaller than ideal should still restore short prefixes.
            writer_full = max(1, min(writer_full, budget // 4))
            fits = max(1, (budget - writer_full) // max_inflight)
            if fits < max_pages:
                logger.warning(
                    "L3 prefetch capped at %d pages, not %d: staging budget is "
                    "%d pages across %d in flight plus a %d-page writer share",
                    fits, max_pages, budget, max_inflight, writer_full,
                )
                max_pages = fits
        staging_pages = max_pages * max_inflight + writer_full

        tier = DSV4L3Tier(
            kv_pool, storage,
            staging_pages=staging_pages,
            window_staging_pages=staging_pages,
            writer_pages=writer_full,
            window_writer_pages=writer_full,
        )
    except Exception:  # noqa: BLE001
        logger.exception("could not build the L3 tier; continuing without one")
        return False

    cache_manager.l3_writer = L3Writer(
        tier,
        max_queued_bytes=int(getattr(config, "hicache_write_queue_bytes", 512 << 20)),
    )
    # The deadline has to outlast a full-size fetch or none ever lands. A fetch
    # moves both tiers of the prefix, ~17.9 MiB per page, so at the ~470 MiB/s
    # measured against this backend a full-size one is a few seconds. Waiting
    # costs nothing but a staging slot: `poll` never blocks, so
    # a fetch still running means this request proceeds without it and the next
    # one with the same prefix finds it ready.
    cache_manager.l3_prefetcher = L3Prefetcher(
        tier,
        deadline_s=float(getattr(config, "hicache_prefetch_deadline_s", 6.0)),
        max_inflight=max_inflight,
        max_pages=max_pages,
    )
    # The digests are what L3 names pages by, and computing them costs a hash
    # per page on the scheduler thread — so they are enabled here, with the
    # tier, and nowhere else.
    cache_manager.prefix_cache.enable_page_hash = True

    logger.info(
        "L3 tier attached: %s, page %d bytes full / %d bytes window, layout %s"
        "; prefetch <=%d pages x%d inflight, staging %d pages per tier "
        "(%.0f MiB)",
        type(storage).__name__, tier.codec.full_page_bytes,
        tier.codec.window_page_bytes, tier.codec.layout_signature,
        max_pages, max_inflight, staging_pages,
        staging_pages * (tier.codec.full_page_bytes
                         + tier.codec.window_page_bytes) / (1 << 20),
    )
    return True


def detach_l3(cache_manager) -> None:
    """Stop the threads. Called on shutdown and before a cache rebuild.

    A rebuild reallocates every tier buffer, which invalidates the codec's view
    of them; a writer still holding gathered bytes would be writing a page
    layout that no longer exists.
    """
    for name in ("l3_writer", "l3_prefetcher"):
        obj = getattr(cache_manager, name, None)
        if obj is not None:
            obj.stop()
            setattr(cache_manager, name, None)
    if getattr(cache_manager, "prefix_cache", None) is not None:
        cache_manager.prefix_cache.enable_page_hash = False


def _load_backend(config):
    """Instantiate the storage class named in the extra config.

    Kept identical in shape to sglang's `dynamic` backend so a backend written
    against that — autumn's is — needs no changes to run here.
    """
    extra = getattr(config, "hicache_storage_backend_extra_config", None) or "{}"
    if isinstance(extra, str):
        extra = json.loads(extra)
    module_path = extra.get("module_path")
    class_name = extra.get("class_name")
    if not module_path or not class_name:
        raise ValueError(
            "hicache-storage-backend-extra-config needs module_path and class_name; "
            f"got {sorted(extra)}"
        )
    from .storage import HiCacheStorageConfig

    # Every field is required, and every value here is fixed by what FreeToken
    # is: one rank (no tensor parallelism at all — `server/args.py` rejects it),
    # MLA on the DSV4 path, and page-first because that is the only layout the
    # zero-copy backends accept.
    storage_config = HiCacheStorageConfig(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=getattr(config, "served_model_name", None)
        or getattr(config, "model_path", None),
        extra_config=extra,
    )
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(storage_config, extra)
