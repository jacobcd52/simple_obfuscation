from __future__ import annotations

import os
import sys
import time
import threading
from typing import Optional
from contextlib import contextmanager

_started = False
_last_beat: float = time.time()
_timeout_s: float = 10.0
_lock = threading.Lock()


def _get_rank_world() -> str:
    try:
        import torch.distributed as dist  # local import to avoid hard dep

        if dist.is_available() and dist.is_initialized():
            return f"rank{dist.get_rank()}/{dist.get_world_size()}"
    except Exception:
        pass
    return "rank?-?"


def _watchdog_loop():
    global _last_beat
    tag = _get_rank_world()
    while True:
        time.sleep(1.0)
        with _lock:
            idle = time.time() - _last_beat
            timeout = _timeout_s
        if idle > timeout:
            msg = f"[Watchdog][{tag}] No heartbeat for {idle:.1f}s (> {timeout}s). Exiting."
            try:
                print(msg, flush=True)
            except Exception:
                pass
            # Hard-exit to avoid hanging collectives
            os._exit(90)


def init_watchdog(timeout_s: float = 10.0):
    global _started, _timeout_s
    if _started:
        return
    _timeout_s = max(1.0, float(timeout_s))
    _started = True
    t = threading.Thread(target=_watchdog_loop, daemon=True)
    t.start()
    print(f"[Watchdog][{_get_rank_world()}] Started with timeout={_timeout_s}s", flush=True)


def heartbeat(tag: Optional[str] = None):
    global _last_beat
    with _lock:
        _last_beat = time.time()
    if tag:
        try:
            print(f"[HB][{_get_rank_world()}] {tag}", flush=True)
        except Exception:
            pass


@contextmanager
def heartbeat_guard(tag: str, interval_s: float = 1.0):
    """Context manager that sends periodic heartbeats while a long op runs."""
    stop_flag = False

    def _pumper():
        while not stop_flag:
            heartbeat(tag)
            time.sleep(interval_s)

    t = threading.Thread(target=_pumper, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop_flag = True
        # best-effort final beat
        heartbeat(f"{tag}_done")


