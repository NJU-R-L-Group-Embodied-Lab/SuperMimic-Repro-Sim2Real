"""Cosine learning-rate schedule (paper Table A.5): decay to `final_ratio` over `decay_fraction` of training, then constant."""

from __future__ import annotations

import math


def cosine_lr(step: int, total_steps: int, base_lr: float, final_ratio: float = 0.03, decay_fraction: float = 250_000 / 320_000) -> float:
    decay_steps = max(1, int(total_steps * decay_fraction))
    if step >= decay_steps:
        return base_lr * final_ratio
    cos = 0.5 * (1.0 + math.cos(math.pi * step / decay_steps))
    return base_lr * (final_ratio + (1.0 - final_ratio) * cos)
