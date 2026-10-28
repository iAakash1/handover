"""Learning-rate schedule preview (pure math, for the plan/report).

mlx-vlm owns the optimizer at run time; this module reproduces the LR curve so
the plan can show warmup + decay without launching anything. Linear warmup to
``learning_rate`` over ``warmup_steps``, then cosine / linear / constant decay to
``min_learning_rate`` across the remaining steps.
"""

from __future__ import annotations

import math

from plantdx.training.config import OptimConfig


def lr_at(step: int, total_steps: int, optim: OptimConfig) -> float:
    """Return the scheduled LR at ``step`` (0-indexed) of ``total_steps``."""
    warmup = max(0, optim.warmup_steps)
    peak = optim.learning_rate
    floor = optim.min_learning_rate
    if step < warmup and warmup > 0:
        return peak * (step + 1) / warmup
    decay_steps = max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / decay_steps))
    if optim.scheduler == "constant":
        return peak
    if optim.scheduler == "linear":
        return peak + (floor - peak) * progress
    # cosine
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def preview_curve(
    total_steps: int, optim: OptimConfig, *, points: int = 6
) -> list[tuple[int, float]]:
    """Return ``points`` evenly spaced ``(step, lr)`` samples for the report."""
    if total_steps <= 0:
        return []
    if total_steps == 1:
        return [(0, lr_at(0, 1, optim))]
    steps = sorted({round(i * (total_steps - 1) / (points - 1)) for i in range(points)})
    return [(s, lr_at(s, total_steps, optim)) for s in steps]
