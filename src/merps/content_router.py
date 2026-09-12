"""Nested content-backbone routing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class BackboneDecision:
    selected: str
    baseline: str
    baseline_mae: float
    best_mae: float
    inner_gain: float
    used_alternative: bool


def choose_backbone(
    inner_mae_by_model: Mapping[str, float],
    *,
    baseline: str,
    minimum_inner_gain: float,
) -> BackboneDecision:
    """Choose an alternative only from inner-fold scores with exact fallback."""
    if baseline not in inner_mae_by_model:
        raise KeyError(f"Missing baseline model: {baseline}")
    if float(minimum_inner_gain) < 0.0:
        raise ValueError("minimum_inner_gain must be non-negative")
    scores = {name: float(value) for name, value in inner_mae_by_model.items()}
    if not scores or not all(np.isfinite(value) for value in scores.values()):
        raise ValueError("All inner MAEs must be finite")
    baseline_mae = scores[baseline]
    best = min(
        scores,
        key=lambda name: (scores[name], 0 if name == baseline else 1, name),
    )
    best_mae = scores[best]
    gain = baseline_mae - best_mae
    use_alternative = (
        best != baseline and gain >= float(minimum_inner_gain)
    )
    selected = best if use_alternative else baseline
    return BackboneDecision(
        selected=selected,
        baseline=baseline,
        baseline_mae=baseline_mae,
        best_mae=best_mae,
        inner_gain=float(gain),
        used_alternative=bool(use_alternative),
    )
