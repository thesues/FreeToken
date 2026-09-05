from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"

    # ── L3 KV tier ───────────────────────────────────────────────────────────
    # These mirror the `--hicache-*` options in `server/args.py` and MUST keep
    # its defaults; a caller building a config directly should get the same
    # engine the CLI builds.
    #
    # They were missing, and the way they were missing is worth remembering.
    # The flags existed and parsed; `attach.py` read them with
    # `getattr(config, "hicache_storage_backend", None)`, which returns None
    # for an absent attribute rather than raising. So WITHOUT the flags the
    # tier silently did not attach and everything looked fine, and WITH them
    # `ServerArgs(**kwargs)` raised `unexpected keyword argument` before the
    # server ever started. The only configuration that could reveal the gap
    # was the only one nobody had run.
    hicache_storage_backend: str | None = None
    hicache_storage_backend_extra_config: str | None = None
    hicache_staging_pages: int = 8
    hicache_prefetch_deadline_s: float = 0.25
    hicache_max_inflight: int = 2
    hicache_write_queue_bytes: int = 512 << 20
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/freetoken_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/freetoken_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/freetoken_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
