from __future__ import annotations

from typing import Iterable

import torch


def robust_clip_grad_norm(accelerator, params: Iterable[torch.nn.Parameter], max_norm: float, *, module_name: str = "") -> None:
    try:
        torch.nn.utils.clip_grad_norm_(
            [p for p in params if getattr(p, "grad", None) is not None],
            max_norm,
            foreach=False,
        )
        return
    except Exception as e2:
        print(f"[GradClip][{module_name}] Non-foreach grad clip failed: {e2}", flush=True)


