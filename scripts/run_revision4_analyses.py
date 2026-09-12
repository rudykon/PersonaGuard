#!/usr/bin/env python3
"""Revision-4 audits for estimator, reference, sensing, and actionability.

This script keeps the released observations fixed while addressing the fourth
pre-submission review. It audits signal time axes and array-level quality,
compares temporal-deviation estimands and DTW objectives, propagates reference
construction through a smaller full-pipeline bootstrap, models the balanced
emotion-category facet, clarifies the strongest no-sensor baseline, evaluates
reservation-aware fNIRS pooling, and tests participant-calibrated signed-bias
trace correction on non-calibration videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import struct
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from merps.innovation.data import (  # noqa: E402
    load_innovation_index,
    load_reservation_masks,
    outer_subject_folds,
)
from merps.innovation.normative_dynamics import (  # noqa: E402
    WarpConfig,
    _alignment_signal,
    decompose_trial,
    smooth_curve,
)
from merps.innovation.personalization import additive_personalization  # noqa: E402
from run_phase_personalization import (  # noqa: E402
    ADDITIVE_SHRINKAGE,
    calibration_sets,
    select_hyperparameters,
)
from run_revision2_analyses import (  # noqa: E402
    emotion_categories,
    interval,
    safe_correlation,
    trial_rows,
)
from run_revision3_analyses import (  # noqa: E402
    METHODS,
    balanced_variance_components,
    crossfit_geometry,
    reliability_for_k,
    run_nested_sensing,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "MER_PS_trainval"
DEFAULT_FEATURE_CACHE = PROJECT_ROOT / "data" / "feature_cache"
DEFAULT_INNOVATION_CACHE = PROJECT_ROOT / "data" / "innovation_cache"
DEFAULT_REVISION2_DIR = PROJECT_ROOT / "artifacts" / "revision2"
DEFAULT_REVISION3_DIR = PROJECT_ROOT / "artifacts" / "revision3"
DEFAULT_TRAIT_DIR = PROJECT_ROOT / "artifacts" / "phase_trait"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "revision4"

VARIANT_NAMES = (
    "consensus",
    "axis_mean",
    "multivariate",
    "valence_only",
    "arousal_only",
)
OBJECTIVE_NAMES = (
    "direct_step",
    "mixed_step_mean",
    "path_mean",
    "step_zero",
)
CALIBRATION_BUDGETS = (1, 2, 4, 8)
MANAGED_FILES = {
    "baseline_comparison.csv",
    "category_gstudy.csv",
    "dtw_objective_audit.csv",
    "estimand_calibration.csv",
    "estimand_robustness.csv",
    "estimand_targets.npz",
    "full_pipeline_bootstrap.csv",
    "reservation_sensing_results.csv",
    "run_manifest.json",
    "sampling_rate_audit.csv",
    "signal_quality_summary.csv",
    "signed_correction.csv",
    "summary.json",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument(
        "--innovation-cache", type=Path, default=DEFAULT_INNOVATION_CACHE
    )
    parser.add_argument("--revision2-dir", type=Path, default=DEFAULT_REVISION2_DIR)
    parser.add_argument("--revision3-dir", type=Path, default=DEFAULT_REVISION3_DIR)
    parser.add_argument("--trait-dir", type=Path, default=DEFAULT_TRAIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--geometry-bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--calibration-repeats", type=int, default=100)
    parser.add_argument("--calibration-inner-repeats", type=int, default=30)
    parser.add_argument("--full-pipeline-repeats", type=int, default=200)
    parser.add_argument(
        "--full-pipeline-workers",
        type=int,
        default=min(12, max(1, (os.cpu_count() or 2) // 2)),
    )
    parser.add_argument("--skip-reservation-sensing", action="store_true")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def manifest_path(path: Path) -> str:
    """Return an anonymous, portable path for the run manifest."""
    resolved = resolve(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def manifest_configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: manifest_path(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_output(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = list(path.iterdir())
    if not existing:
        return
    if not overwrite:
        raise FileExistsError(f"{path} is not empty; pass --overwrite")
    unknown = [item for item in existing if item.name not in MANAGED_FILES]
    if unknown:
        raise FileExistsError(
            "Refusing to overwrite unmanaged files: "
            + ", ".join(sorted(item.name for item in unknown))
        )
    for item in existing:
        if item.is_file() or item.is_symlink():
            item.unlink()


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def interval_gate_status(
    point: float, low: float, high: float, threshold: float = 0.70
) -> str:
    if low > threshold:
        return "PASS"
    if high < threshold:
        return "FAILED"
    return "PARTIAL"


@dataclass(frozen=True)
class PathAudit:
    mapping: np.ndarray
    path: tuple[tuple[int, int], ...]
    local_cost: float
    warp_cost: float
    step_cost: float
    total_cost: float
    path_length: int
    non_diagonal_steps: int


def _signal_matrix(values: np.ndarray, config: WarpConfig) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return _alignment_signal(values, config)[:, None]
    if values.ndim != 2:
        raise ValueError(f"Expected one- or two-dimensional curve, got {values.shape}")
    return np.column_stack(
        [_alignment_signal(values[:, axis], config) for axis in range(values.shape[1])]
    )


def _solve_additive_path(
    participant: np.ndarray,
    reference: np.ndarray,
    config: WarpConfig,
    *,
    step_penalty: float,
    per_node_offset: float = 0.0,
) -> PathAudit:
    participant = np.asarray(participant, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    n, m = len(participant), len(reference)
    if n < 2 or m < 2:
        mapping = np.zeros(n, dtype=np.int16)
        return PathAudit(mapping, ((0, 0),), 0.0, 0.0, 0.0, 0.0, 1, 0)
    band = max(abs(n - m), int(config.band_seconds))
    cost = np.full((n, m), np.inf, dtype=np.float64)
    back = np.full((n, m), -1, dtype=np.int8)
    for i in range(n):
        start = max(0, i - band)
        end = min(m, i + band + 1)
        for j in range(start, end):
            local = float(np.mean(np.abs(participant[i] - reference[j])))
            local += config.warp_penalty * abs(i - j) / max(band, 1)
            local -= per_node_offset
            if i == 0 and j == 0:
                cost[i, j] = local
                back[i, j] = 0
                continue
            best = np.inf
            direction = -1
            if i > 0 and j > 0 and cost[i - 1, j - 1] < best:
                best = cost[i - 1, j - 1]
                direction = 0
            if i > 0 and cost[i - 1, j] + step_penalty < best:
                best = cost[i - 1, j] + step_penalty
                direction = 1
            if j > 0 and cost[i, j - 1] + step_penalty < best:
                best = cost[i, j - 1] + step_penalty
                direction = 2
            if np.isfinite(best):
                cost[i, j] = local + best
                back[i, j] = direction
    if not np.isfinite(cost[-1, -1]):
        raise RuntimeError("Constrained DTW failed to connect both endpoints")

    path: list[tuple[int, int]] = []
    i, j = n - 1, m - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        direction = int(back[i, j])
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        elif direction == 2:
            j -= 1
        else:
            raise RuntimeError("Invalid DTW backtracking direction")
    path.reverse()

    matches: list[list[int]] = [[] for _ in range(n)]
    for participant_time, reference_time in path:
        matches[participant_time].append(reference_time)
    mapping = np.empty(n, dtype=np.int16)
    previous = 0
    for position, values in enumerate(matches):
        if values:
            previous = int(math.floor(float(np.median(values)) + 0.5))
        mapping[position] = previous
    mapping = np.maximum.accumulate(np.clip(mapping, 0, m - 1)).astype(np.int16)

    local_cost = 0.0
    warp_cost = 0.0
    non_diagonal = 0
    for position, (pi, rj) in enumerate(path):
        local_cost += float(np.mean(np.abs(participant[pi] - reference[rj])))
        warp_cost += config.warp_penalty * abs(pi - rj) / max(band, 1)
        if position and (
            pi - path[position - 1][0], rj - path[position - 1][1]
        ) != (1, 1):
            non_diagonal += 1
    step_cost = float(step_penalty * non_diagonal)
    total = float(local_cost + warp_cost + step_cost)
    return PathAudit(
        mapping=mapping,
        path=tuple(path),
        local_cost=float(local_cost),
        warp_cost=float(warp_cost),
        step_cost=step_cost,
        total_cost=total,
        path_length=len(path),
        non_diagonal_steps=non_diagonal,
    )


def _solve_mixed_step_mean_path(
    participant: np.ndarray,
    reference: np.ndarray,
    config: WarpConfig,
) -> PathAudit:
    """Exactly minimize local-plus-warp sum plus lambda_s K / L.

    For a path joining fixed endpoints, the number of non-diagonal transitions
    K is determined by its node count L: K = 2L - n - m. Dynamic programming
    therefore keeps the best local-plus-warp cost for every feasible L, then
    applies the mixed-normalization term at the endpoint.
    """

    participant = np.asarray(participant, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    n, m = len(participant), len(reference)
    if n < 2 or m < 2:
        mapping = np.zeros(n, dtype=np.int16)
        return PathAudit(mapping, ((0, 0),), 0.0, 0.0, 0.0, 0.0, 1, 0)

    band = max(abs(n - m), int(config.band_seconds))
    maximum_length = n + m - 1
    costs: list[list[np.ndarray | None]] = [
        [None for _ in range(m)] for _ in range(n)
    ]
    backs: list[list[np.ndarray | None]] = [
        [None for _ in range(m)] for _ in range(n)
    ]
    for i in range(n):
        start = max(0, i - band)
        end = min(m, i + band + 1)
        for j in range(start, end):
            local = float(np.mean(np.abs(participant[i] - reference[j])))
            local += config.warp_penalty * abs(i - j) / max(band, 1)
            current = np.full(maximum_length + 1, np.inf, dtype=np.float64)
            directions = np.full(maximum_length + 1, -1, dtype=np.int8)
            if i == 0 and j == 0:
                current[1] = local
                directions[1] = 0
            else:
                for direction, previous_i, previous_j in (
                    (0, i - 1, j - 1),
                    (1, i - 1, j),
                    (2, i, j - 1),
                ):
                    if previous_i < 0 or previous_j < 0:
                        continue
                    previous = costs[previous_i][previous_j]
                    if previous is None:
                        continue
                    candidate = previous[:-1] + local
                    better = candidate < current[1:]
                    indices = np.flatnonzero(better) + 1
                    current[indices] = candidate[better]
                    directions[indices] = direction
            costs[i][j] = current
            backs[i][j] = directions

    endpoint = costs[-1][-1]
    if endpoint is None:
        raise RuntimeError("Mixed-normalization DTW failed to reach the endpoint")
    lengths = np.arange(maximum_length + 1, dtype=np.int32)
    non_diagonal_by_length = 2 * lengths - n - m
    valid = (
        np.isfinite(endpoint)
        & (lengths > 0)
        & (non_diagonal_by_length >= 0)
    )
    objective = np.full(maximum_length + 1, np.inf, dtype=np.float64)
    objective[valid] = endpoint[valid] + (
        float(config.step_penalty)
        * non_diagonal_by_length[valid]
        / lengths[valid]
    )
    selected_length = int(np.argmin(objective))
    if not np.isfinite(objective[selected_length]):
        raise RuntimeError("Mixed-normalization DTW found no feasible path length")

    path: list[tuple[int, int]] = []
    i, j, length = n - 1, m - 1, selected_length
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        directions = backs[i][j]
        if directions is None:
            raise RuntimeError("Missing mixed-normalization backtracking state")
        direction = int(directions[length])
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        elif direction == 2:
            j -= 1
        else:
            raise RuntimeError("Invalid mixed-normalization backtracking direction")
        length -= 1
    path.reverse()

    matches: list[list[int]] = [[] for _ in range(n)]
    for participant_time, reference_time in path:
        matches[participant_time].append(reference_time)
    mapping = np.empty(n, dtype=np.int16)
    previous_reference = 0
    for position, values in enumerate(matches):
        if values:
            previous_reference = int(math.floor(float(np.median(values)) + 0.5))
        mapping[position] = previous_reference
    mapping = np.maximum.accumulate(np.clip(mapping, 0, m - 1)).astype(np.int16)

    local_cost = 0.0
    warp_cost = 0.0
    non_diagonal = 0
    for position, (participant_time, reference_time) in enumerate(path):
        local_cost += float(
            np.mean(
                np.abs(
                    participant[participant_time] - reference[reference_time]
                )
            )
        )
        warp_cost += (
            config.warp_penalty
            * abs(participant_time - reference_time)
            / max(band, 1)
        )
        if position and (
            participant_time - path[position - 1][0],
            reference_time - path[position - 1][1],
        ) != (1, 1):
            non_diagonal += 1
    if non_diagonal != 2 * len(path) - n - m:
        raise RuntimeError("Mixed-normalization path-count identity failed")
    step_cost = float(config.step_penalty * non_diagonal / len(path))
    return PathAudit(
        mapping=mapping,
        path=tuple(path),
        local_cost=float(local_cost),
        warp_cost=float(warp_cost),
        step_cost=step_cost,
        total_cost=float(local_cost + warp_cost + step_cost),
        path_length=len(path),
        non_diagonal_steps=non_diagonal,
    )


def audited_dtw(
    participant_curve: np.ndarray,
    reference_curve: np.ndarray,
    config: WarpConfig,
    *,
    objective: str = "direct_step",
) -> PathAudit:
    participant = _signal_matrix(participant_curve, config)
    reference = _signal_matrix(reference_curve, config)
    if objective == "direct_step":
        return _solve_additive_path(
            participant,
            reference,
            config,
            step_penalty=float(config.step_penalty),
        )
    if objective == "mixed_step_mean":
        return _solve_mixed_step_mean_path(
            participant, reference, config
        )
    if objective == "step_zero":
        return _solve_additive_path(
            participant, reference, config, step_penalty=0.0
        )
    if objective != "path_mean":
        raise ValueError(f"Unknown DTW objective: {objective}")

    ratio = 0.0
    result = _solve_additive_path(
        participant,
        reference,
        config,
        step_penalty=float(config.step_penalty),
    )
    ratio = result.total_cost / max(result.path_length, 1)
    for _ in range(30):
        candidate = _solve_additive_path(
            participant,
            reference,
            config,
            step_penalty=float(config.step_penalty),
            per_node_offset=ratio,
        )
        updated = candidate.total_cost / max(candidate.path_length, 1)
        result = candidate
        if abs(updated - ratio) < 1e-10:
            break
        ratio = updated
    return result


def consensus_mapping(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, int]:
    raw = np.floor(
        (np.asarray(first, dtype=np.float64) + np.asarray(second, dtype=np.float64))
        / 2.0
        + 0.5
    ).astype(np.int16)
    raw = np.clip(raw, 0, max(len(raw) - 1, 0))
    projected = np.maximum.accumulate(raw)
    return projected.astype(np.int16), int(np.sum(projected != raw))


def trial_estimands(
    target: np.ndarray,
    template: np.ndarray,
    config: WarpConfig,
) -> tuple[dict[str, float], dict[str, object]]:
    axis = decompose_trial(target, template, config)
    multivariate = audited_dtw(target, template, config, objective="direct_step")
    times = np.arange(len(target), dtype=np.float64)
    q_values = {
        "consensus": float(np.abs(axis.consensus_lag).mean()),
        "axis_mean": float(
            0.5
            * (
                np.abs(axis.lag_from_valence).mean()
                + np.abs(axis.lag_from_arousal).mean()
            )
        ),
        "multivariate": float(np.abs(times - multivariate.mapping).mean()),
        "valence_only": float(np.abs(axis.lag_from_valence).mean()),
        "arousal_only": float(np.abs(axis.lag_from_arousal).mean()),
    }
    raw_consensus, projection_changes = consensus_mapping(
        axis.mapping_from_valence, axis.mapping_from_arousal
    )
    diagnostics = {
        "signed_bias": float(axis.consensus_lag.mean()),
        "projection_changes": projection_changes,
        "consensus_matches_implementation": bool(
            np.array_equal(raw_consensus, axis.consensus_mapping)
        ),
        "maximum_consensus_jump": int(np.diff(axis.consensus_mapping).max(initial=0)),
        "maximum_consensus_band_deviation": int(
            np.abs(np.arange(len(target)) - axis.consensus_mapping).max()
        ),
    }
    return q_values, diagnostics


def target_cube(index) -> list[np.ndarray]:
    cubes: list[np.ndarray] = []
    for video in range(1, 16):
        traces = []
        for subject in range(1, 25):
            rows = trial_rows(index, subject, video)
            traces.append(index.targets[rows].astype(np.float64))
        cubes.append(np.stack(traces, axis=0))
    return cubes


def reference_from_positions(cube: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return smooth_curve(np.median(cube[positions], axis=0), 3)


def build_estimand_matrices(index) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    cubes = target_cube(index)
    config = WarpConfig()
    matrices = {
        name: np.empty((24, 15), dtype=np.float64) for name in VARIANT_NAMES
    }
    signed = np.empty((24, 15), dtype=np.float64)
    projection_changes = 0
    consensus_mismatches = 0
    maximum_jump = 0
    maximum_band = 0
    all_positions = np.arange(24, dtype=np.int16)
    for participant in range(24):
        reference_positions = all_positions[all_positions != participant]
        for video, cube in enumerate(cubes):
            template = reference_from_positions(cube, reference_positions)
            values, diagnostics = trial_estimands(cube[participant], template, config)
            for name, value in values.items():
                matrices[name][participant, video] = value
            signed[participant, video] = float(diagnostics["signed_bias"])
            projection_changes += int(diagnostics["projection_changes"])
            consensus_mismatches += int(
                not bool(diagnostics["consensus_matches_implementation"])
            )
            maximum_jump = max(maximum_jump, int(diagnostics["maximum_consensus_jump"]))
            maximum_band = max(
                maximum_band, int(diagnostics["maximum_consensus_band_deviation"])
            )
        print(f"estimand LOPO participant={participant + 1}/24", flush=True)
    return matrices, signed, {
        "trials": 360,
        "projection_changed_positions": projection_changes,
        "consensus_reconstruction_mismatches": consensus_mismatches,
        "maximum_consensus_mapping_jump": maximum_jump,
        "maximum_consensus_band_deviation": maximum_band,
        "mathematical_note": (
            "The rounded average of two nondecreasing mappings is itself "
            "nondecreasing, so cumulative-max projection is a verified no-op "
            "for all released trials. The consensus is a band-respecting "
            "summary mapping, not a claim that it is the unique original path."
        ),
    }


def geometry_tensor(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = (
        "trajectory_range",
        "trajectory_variance",
        "joystick_path_length_per_second",
        "mean_joystick_speed",
        "axis_correlation",
        "group_reference_gradient",
        "magnitude_residual",
        "distance_from_center",
        "floor_ceiling_occupancy",
    )
    rows.sort(key=lambda row: (int(row["subject"]), int(row["video"])))
    return np.asarray(
        [[float(row[name]) for name in names] for row in rows], dtype=np.float64
    ).reshape(24, 15, len(names))


def estimator_robustness(
    matrices: dict[str, np.ndarray],
    geometry: np.ndarray,
    *,
    bootstrap_repeats: int,
    geometry_bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    consensus_profile = matrices["consensus"].mean(axis=1)
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    rng = np.random.default_rng(seed)
    for name in VARIANT_NAMES:
        matrix = matrices[name]
        components = balanced_variance_components(matrix)
        reliability = reliability_for_k(components, 15)
        residual = crossfit_geometry(matrix, geometry)
        adjusted_components = balanced_variance_components(residual)
        adjusted_reliability = reliability_for_k(adjusted_components, 15)
        g_boot = np.empty(bootstrap_repeats, dtype=np.float64)
        phi_boot = np.empty_like(g_boot)
        negative_participant = 0
        negative_video = 0
        for repeat in range(bootstrap_repeats):
            participants = rng.integers(0, 24, 24)
            videos = rng.integers(0, 15, 15)
            draw = matrix[np.ix_(participants, videos)]
            draw_components = balanced_variance_components(draw)
            draw_rel = reliability_for_k(draw_components, 15)
            g_boot[repeat] = draw_rel["relative_g"]
            phi_boot[repeat] = draw_rel["absolute_phi"]
            negative_participant += int(draw_components["raw_participant"] < 0)
            negative_video += int(draw_components["raw_video"] < 0)

        relative_low, relative_high = np.quantile(g_boot, (0.025, 0.975))
        phi_low, phi_high = np.quantile(phi_boot, (0.025, 0.975))
        retained_boot = np.empty(
            geometry_bootstrap_repeats, dtype=np.float64
        )
        adjusted_g_boot = np.empty_like(retained_boot)
        for repeat in range(geometry_bootstrap_repeats):
            participants = rng.integers(0, 24, 24)
            videos = rng.integers(0, 15, 15)
            matrix_draw = matrix[np.ix_(participants, videos)]
            geometry_draw = geometry[participants][:, videos]
            residual_draw = crossfit_geometry(matrix_draw, geometry_draw)
            raw_draw = balanced_variance_components(matrix_draw)
            adjusted_draw = balanced_variance_components(residual_draw)
            retained_boot[repeat] = (
                adjusted_draw["participant"]
                / max(raw_draw["participant"], 1e-12)
            )
            adjusted_g_boot[repeat] = reliability_for_k(
                adjusted_draw, 15
            )["relative_g"]
        retained_low, retained_high = np.quantile(
            retained_boot, (0.025, 0.975)
        )
        adjusted_g_low, adjusted_g_high = np.quantile(
            adjusted_g_boot, (0.025, 0.975)
        )
        row = {
            "estimand": name,
            "profile_spearman_vs_consensus": safe_correlation(
                matrix.mean(axis=1), consensus_profile, spearman=True
            ),
            "trial_mae_vs_consensus_seconds": float(
                np.abs(matrix - matrices["consensus"]).mean()
            ),
            "participant_variance_seconds_squared": components["participant"],
            "raw_participant_variance_seconds_squared": components[
                "raw_participant"
            ],
            "video_variance_seconds_squared": components["video"],
            "raw_video_variance_seconds_squared": components["raw_video"],
            "residual_variance_seconds_squared": components["residual"],
            "bootstrap_negative_participant_draws": negative_participant,
            "bootstrap_negative_participant_fraction": float(
                negative_participant / bootstrap_repeats
            ),
            "bootstrap_negative_video_draws": negative_video,
            "bootstrap_negative_video_fraction": float(
                negative_video / bootstrap_repeats
            ),
            "relative_g_15": reliability["relative_g"],
            "relative_g_15_ci95_low": float(relative_low),
            "relative_g_15_ci95_high": float(relative_high),
            "relative_g_15_gate_status": interval_gate_status(
                reliability["relative_g"],
                float(relative_low),
                float(relative_high),
            ),
            "absolute_phi_15": reliability["absolute_phi"],
            "absolute_phi_15_ci95_low": float(phi_low),
            "absolute_phi_15_ci95_high": float(phi_high),
            "absolute_phi_15_gate_status": interval_gate_status(
                reliability["absolute_phi"],
                float(phi_low),
                float(phi_high),
            ),
            "geometry_adjusted_participant_variance": adjusted_components[
                "participant"
            ],
            "geometry_variance_retained_fraction": float(
                adjusted_components["participant"]
                / max(components["participant"], 1e-12)
            ),
            "geometry_variance_retained_ci95_low": float(retained_low),
            "geometry_variance_retained_ci95_high": float(retained_high),
            "geometry_bootstrap_draws": geometry_bootstrap_repeats,
            "geometry_adjusted_relative_g_15": adjusted_reliability[
                "relative_g"
            ],
            "geometry_adjusted_relative_g_15_ci95_low": float(
                adjusted_g_low
            ),
            "geometry_adjusted_relative_g_15_ci95_high": float(
                adjusted_g_high
            ),
            "geometry_profile_spearman_raw_vs_adjusted": safe_correlation(
                matrix.mean(axis=1), residual.mean(axis=1), spearman=True
            ),
        }
        rows.append(row)
        summary[name] = row
    return rows, summary


def fold_estimand_matrices(
    index,
) -> list[dict[str, object]]:
    cubes = target_cube(index)
    config = WarpConfig()
    output: list[dict[str, object]] = []
    for fold, (training_subjects, validation_subjects) in enumerate(
        outer_subject_folds()
    ):
        training_positions = np.asarray(training_subjects, dtype=int) - 1
        validation_positions = np.asarray(validation_subjects, dtype=int) - 1
        training = {
            name: np.empty((len(training_positions), 15), dtype=np.float64)
            for name in VARIANT_NAMES
        }
        validation = {
            name: np.empty((len(validation_positions), 15), dtype=np.float64)
            for name in VARIANT_NAMES
        }
        training_b = np.empty((len(training_positions), 15), dtype=np.float64)
        validation_b = np.empty((len(validation_positions), 15), dtype=np.float64)
        validation_templates: list[list[np.ndarray]] = [
            [np.empty((0, 2)) for _ in range(15)]
            for _ in range(len(validation_positions))
        ]
        for local, participant in enumerate(training_positions):
            reference_positions = training_positions[training_positions != participant]
            for video, cube in enumerate(cubes):
                template = reference_from_positions(cube, reference_positions)
                values, diagnostics = trial_estimands(cube[participant], template, config)
                for name, value in values.items():
                    training[name][local, video] = value
                training_b[local, video] = float(diagnostics["signed_bias"])
        templates = [
            reference_from_positions(cube, training_positions) for cube in cubes
        ]
        for local, participant in enumerate(validation_positions):
            for video, cube in enumerate(cubes):
                template = templates[video]
                values, diagnostics = trial_estimands(cube[participant], template, config)
                for name, value in values.items():
                    validation[name][local, video] = value
                validation_b[local, video] = float(diagnostics["signed_bias"])
                validation_templates[local][video] = template
        output.append(
            {
                "fold": fold,
                "training_subjects": np.asarray(training_subjects, dtype=np.int16),
                "validation_subjects": np.asarray(validation_subjects, dtype=np.int16),
                "training": training,
                "validation": validation,
                "training_b": training_b,
                "validation_b": validation_b,
                "validation_templates": validation_templates,
                "cubes": cubes,
            }
        )
        print(f"estimand outer fold={fold + 1}/5", flush=True)
    return output


def estimator_calibration(
    folds: list[dict[str, object]],
    *,
    repeats: int,
    inner_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold_info in folds:
        fold = int(fold_info["fold"])
        validation_subjects = np.asarray(fold_info["validation_subjects"])
        for variant_position, name in enumerate(VARIANT_NAMES):
            training = np.asarray(fold_info["training"][name], dtype=np.float64)
            validation = np.asarray(fold_info["validation"][name], dtype=np.float64)
            video_mean = training.mean(axis=0)
            for budget in CALIBRATION_BUDGETS:
                selection = select_hyperparameters(
                    training,
                    budget=budget,
                    repeats=inner_repeats,
                    seed=seed + fold * 10000 + variant_position * 1000 + budget * 10,
                )
                subsets = calibration_sets(
                    participants=len(validation_subjects),
                    budget=budget,
                    repeats=repeats,
                    seed=(
                        seed
                        + 500000
                        + fold * 10000
                        + variant_position * 1000
                        + budget * 10
                    ),
                )
                for local, subject in enumerate(validation_subjects):
                    target = validation[local]
                    for repeat in range(repeats):
                        calibration = subsets[(local, repeat)]
                        evaluation = np.setdiff1d(np.arange(15), calibration)
                        prediction = additive_personalization(
                            video_mean,
                            calibration,
                            target[calibration],
                            float(selection["additive_shrinkage"]),
                        )
                        baseline_mae = float(
                            np.abs(video_mean[evaluation] - target[evaluation]).mean()
                        )
                        calibrated_mae = float(
                            np.abs(prediction[evaluation] - target[evaluation]).mean()
                        )
                        rows.append(
                            {
                                "fold": fold,
                                "estimand": name,
                                "subject": int(subject),
                                "budget": budget,
                                "repeat": repeat,
                                "selected_shrinkage": float(
                                    selection["additive_shrinkage"]
                                ),
                                "baseline_mae_seconds": baseline_mae,
                                "calibrated_mae_seconds": calibrated_mae,
                                "gain_seconds": baseline_mae - calibrated_mae,
                            }
                        )
    summary: dict[str, object] = {}
    for name in VARIANT_NAMES:
        summary[name] = {}
        for budget in CALIBRATION_BUDGETS:
            selected = [
                row
                for row in rows
                if row["estimand"] == name and int(row["budget"]) == budget
            ]
            participant_means = np.asarray(
                [
                    np.mean(
                        [
                            float(row["gain_seconds"])
                            for row in selected
                            if int(row["subject"]) == subject
                        ]
                    )
                    for subject in range(1, 25)
                ]
            )
            summary[name][str(budget)] = {
                "mean_gain_seconds": float(participant_means.mean()),
                "participant_median_gain_seconds": float(
                    np.median(participant_means)
                ),
                "participants_improved": int(np.sum(participant_means > 0)),
            }
    return rows, summary


def objective_audit(
    index,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cubes = target_cube(index)
    config = WarpConfig()
    rows: list[dict[str, object]] = []
    q = {name: np.empty((24, 15), dtype=np.float64) for name in OBJECTIVE_NAMES}
    for training_subjects, validation_subjects in outer_subject_folds():
        training_positions = np.asarray(training_subjects, dtype=int) - 1
        templates = [
            reference_from_positions(cube, training_positions) for cube in cubes
        ]
        for subject in validation_subjects:
            participant = int(subject) - 1
            for video, cube in enumerate(cubes):
                mappings: dict[str, list[np.ndarray]] = {
                    name: [] for name in OBJECTIVE_NAMES
                }
                for axis in range(2):
                    for objective in OBJECTIVE_NAMES:
                        result = audited_dtw(
                            cube[participant, :, axis],
                            templates[video][:, axis],
                            config,
                            objective=objective,
                        )
                        mappings[objective].append(result.mapping)
                        rows.append(
                            {
                                "subject": int(subject),
                                "video": video + 1,
                                "axis": "valence" if axis == 0 else "arousal",
                                "objective": objective,
                                "local_cost": result.local_cost,
                                "warp_cost": result.warp_cost,
                                "step_cost": result.step_cost,
                                "total_cost": result.total_cost,
                                "step_cost_fraction": float(
                                    result.step_cost / max(result.total_cost, 1e-12)
                                ),
                                "trial_nodes": len(cube[participant]),
                                "path_length": result.path_length,
                                "non_diagonal_steps": result.non_diagonal_steps,
                            }
                        )
                for objective in OBJECTIVE_NAMES:
                    mapping, _ = consensus_mapping(*mappings[objective])
                    q[objective][participant, video] = float(
                        np.abs(np.arange(len(mapping)) - mapping).mean()
                    )
        print(
            f"DTW objective fold with held-out={validation_subjects.tolist()} complete",
            flush=True,
        )

    direct_rows = [row for row in rows if row["objective"] == "direct_step"]
    step_fractions = np.asarray(
        [float(row["step_cost_fraction"]) for row in direct_rows]
    )
    rng = np.random.default_rng(seed)
    summary: dict[str, object] = {
        "implementation_audit": (
            "The executed and manuscript objectives directly add lambda_s for "
            "every non-diagonal transition; no division by path length occurs."
        ),
        "production_objective": "direct_step",
        "sensitivity_objectives": {
            "mixed_step_mean": "sum(local+warp) + lambda_s K / path_length",
            "path_mean": "sum(local+warp+direct_step) / path_length",
            "step_zero": "sum(local+warp) with lambda_s set to zero",
        },
        "direct_step_cost_fraction": quantile_summary(step_fractions),
        "objectives": {},
    }
    for objective in OBJECTIVE_NAMES:
        matrix = q[objective]
        components = balanced_variance_components(matrix)
        reliability = reliability_for_k(components, 15)
        profile = matrix.mean(axis=1)
        difference = matrix - q["direct_step"]
        boot = np.empty(bootstrap_repeats, dtype=np.float64)
        for repeat in range(bootstrap_repeats):
            participants = rng.integers(0, 24, 24)
            videos = rng.integers(0, 15, 15)
            boot[repeat] = difference[np.ix_(participants, videos)].mean()
        objective_rows = [
            row for row in rows if row["objective"] == objective
        ]
        path_lengths = np.asarray(
            [float(row["path_length"]) for row in objective_rows]
        )
        path_ratios = np.asarray(
            [
                float(row["path_length"]) / float(row["trial_nodes"])
                for row in objective_rows
            ]
        )
        non_diagonal_fractions = np.asarray(
            [
                float(row["non_diagonal_steps"])
                / max(float(row["path_length"]) - 1.0, 1.0)
                for row in objective_rows
            ]
        )
        objective_values = np.asarray(
            [
                float(row["total_cost"])
                / (
                    float(row["path_length"])
                    if objective == "path_mean"
                    else 1.0
                )
                for row in objective_rows
            ]
        )
        summary["objectives"][objective] = {
            "path_length": quantile_summary(path_lengths),
            "path_length_to_trial_nodes_ratio": quantile_summary(path_ratios),
            "non_diagonal_transition_fraction": quantile_summary(
                non_diagonal_fractions
            ),
            "objective_value": quantile_summary(objective_values),
            "mean_q_seconds": float(matrix.mean()),
            "mean_q_difference_vs_direct_seconds": float(difference.mean()),
            "mean_q_difference_ci95": interval(boot),
            "profile_spearman_vs_direct": safe_correlation(
                profile, q["direct_step"].mean(axis=1), spearman=True
            ),
            "relative_g_15": reliability["relative_g"],
            "absolute_phi_15": reliability["absolute_phi"],
        }
    return rows, summary


def category_variance_components(matrix: np.ndarray) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (24, 5, 3):
        raise ValueError(f"Expected [24, 5, 3], got {values.shape}")
    participants, categories, videos = values.shape
    grand = float(values.mean())
    participant_mean = values.mean(axis=(1, 2))
    category_mean = values.mean(axis=(0, 2))
    video_mean = values.mean(axis=0)
    participant_category_mean = values.mean(axis=2)

    ss_participant = categories * videos * float(
        np.square(participant_mean - grand).sum()
    )
    ss_category = participants * videos * float(
        np.square(category_mean - grand).sum()
    )
    ss_video_category = participants * float(
        np.square(video_mean - category_mean[:, None]).sum()
    )
    pc_residual = (
        participant_category_mean
        - participant_mean[:, None]
        - category_mean[None, :]
        + grand
    )
    ss_pc = videos * float(np.square(pc_residual).sum())
    fitted = (
        participant_category_mean[:, :, None]
        + video_mean[None, :, :]
        - category_mean[None, :, None]
    )
    ss_residual = float(np.square(values - fitted).sum())

    ms_participant = ss_participant / (participants - 1)
    ms_video_category = ss_video_category / (categories * (videos - 1))
    ms_pc = ss_pc / ((participants - 1) * (categories - 1))
    ms_residual = ss_residual / (
        (participants - 1) * categories * (videos - 1)
    )
    raw_participant = (ms_participant - ms_pc) / (categories * videos)
    raw_video_category = (ms_video_category - ms_residual) / participants
    raw_pc = (ms_pc - ms_residual) / videos
    return {
        "participant": float(max(raw_participant, 0.0)),
        "video_within_category": float(max(raw_video_category, 0.0)),
        "participant_by_category": float(max(raw_pc, 0.0)),
        "residual_participant_by_video": float(max(ms_residual, 0.0)),
        "raw_participant": float(raw_participant),
        "raw_video_within_category": float(raw_video_category),
        "raw_participant_by_category": float(raw_pc),
        "raw_residual": float(ms_residual),
        "category_fixed_effect_ss": float(ss_category),
    }


def category_reliability(
    components: dict[str, float], videos_per_category: int
) -> dict[str, float]:
    categories = 5
    participant = components["participant"]
    video = components["video_within_category"]
    pc = components["participant_by_category"]
    residual = components["residual_participant_by_video"]
    relative_error = pc / categories + residual / (
        categories * videos_per_category
    )
    absolute_error = relative_error + video / (
        categories * videos_per_category
    )
    return {
        "relative_g": float(
            participant / (participant + relative_error)
            if participant + relative_error > 0
            else 0.0
        ),
        "absolute_phi": float(
            participant / (participant + absolute_error)
            if participant + absolute_error > 0
            else 0.0
        ),
    }


def category_order(categories: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(categories)
    unique = sorted(set(int(value) for value in values))
    groups = [np.flatnonzero(values == value) for value in unique]
    if len(groups) != 5 or any(len(group) != 3 for group in groups):
        raise ValueError("Expected five categories with three videos each")
    return groups


def category_gstudy(
    matrix: np.ndarray,
    categories: np.ndarray,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    groups = category_order(categories)
    cube = np.stack([matrix[:, group] for group in groups], axis=1)
    components = category_variance_components(cube)
    rows = []
    for count in (1, 2, 3):
        reliability = category_reliability(components, count)
        rows.append(
            {
                "videos_per_category": count,
                "total_videos": count * 5,
                "relative_g": reliability["relative_g"],
                "absolute_phi": reliability["absolute_phi"],
            }
        )

    rng = np.random.default_rng(seed)
    simple_g = np.empty(bootstrap_repeats, dtype=np.float64)
    simple_phi = np.empty_like(simple_g)
    category_g = np.empty_like(simple_g)
    category_phi = np.empty_like(simple_g)
    negative = {
        "participant": 0,
        "video_within_category": 0,
        "participant_by_category": 0,
    }
    for repeat in range(bootstrap_repeats):
        participants = rng.integers(0, 24, 24)
        sampled_groups = [rng.choice(group, size=3, replace=True) for group in groups]
        videos = np.concatenate(sampled_groups)
        draw = matrix[np.ix_(participants, videos)]
        simple = balanced_variance_components(draw)
        simple_rel = reliability_for_k(simple, 15)
        simple_g[repeat] = simple_rel["relative_g"]
        simple_phi[repeat] = simple_rel["absolute_phi"]
        draw_cube = np.stack(
            [matrix[np.ix_(participants, group)] for group in sampled_groups],
            axis=1,
        )
        category_components = category_variance_components(draw_cube)
        negative["participant"] += int(category_components["raw_participant"] < 0)
        negative["video_within_category"] += int(
            category_components["raw_video_within_category"] < 0
        )
        negative["participant_by_category"] += int(
            category_components["raw_participant_by_category"] < 0
        )
        category_rel = category_reliability(category_components, 3)
        category_g[repeat] = category_rel["relative_g"]
        category_phi[repeat] = category_rel["absolute_phi"]
    return rows, {
        "categories_treated_as_fixed": True,
        "variance_components": components,
        "balanced_dstudy": {str(row["total_videos"]): row for row in rows},
        "category_stratified_fast_bootstrap": {
            "simple_relative_g_15_ci95": interval(simple_g),
            "simple_absolute_phi_15_ci95": interval(simple_phi),
            "category_relative_g_15_ci95": interval(category_g),
            "category_absolute_phi_15_ci95": interval(category_phi),
            "negative_raw_component_counts": negative,
            "draws": bootstrap_repeats,
        },
    }


_FULL_BOOT_CUBES: list[np.ndarray] | None = None
_FULL_BOOT_GROUPS: list[np.ndarray] | None = None


def _init_full_pipeline_worker(
    cubes: list[np.ndarray], groups: list[np.ndarray]
) -> None:
    global _FULL_BOOT_CUBES, _FULL_BOOT_GROUPS
    _FULL_BOOT_CUBES = cubes
    _FULL_BOOT_GROUPS = groups


def _full_pipeline_draw(seed: int) -> dict[str, float | int]:
    if _FULL_BOOT_CUBES is None or _FULL_BOOT_GROUPS is None:
        raise RuntimeError("Full-pipeline worker was not initialized")
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, 24, 24)
    config = WarpConfig()
    q = np.empty((24, 15), dtype=np.float64)
    pseudo_positions = np.arange(24)
    for pseudo in range(24):
        reference_positions = pseudo_positions[pseudo_positions != pseudo]
        source_positions = selected[reference_positions]
        for video, cube in enumerate(_FULL_BOOT_CUBES):
            template = reference_from_positions(cube, source_positions)
            result = decompose_trial(cube[selected[pseudo]], template, config)
            q[pseudo, video] = float(np.abs(result.consensus_lag).mean())
    sampled_groups = [
        rng.choice(group, size=3, replace=True) for group in _FULL_BOOT_GROUPS
    ]
    videos = np.concatenate(sampled_groups)
    simple_components = balanced_variance_components(q[:, videos])
    simple = reliability_for_k(simple_components, 15)
    category_cube = np.stack([q[:, group] for group in sampled_groups], axis=1)
    category_components = category_variance_components(category_cube)
    category = category_reliability(category_components, 3)
    return {
        "relative_g_15": simple["relative_g"],
        "absolute_phi_15": simple["absolute_phi"],
        "category_relative_g_15": category["relative_g"],
        "category_absolute_phi_15": category["absolute_phi"],
        "raw_participant_variance": simple_components["raw_participant"],
        "raw_video_variance": simple_components["raw_video"],
        "raw_category_participant_variance": category_components[
            "raw_participant"
        ],
        "raw_video_within_category_variance": category_components[
            "raw_video_within_category"
        ],
        "raw_participant_by_category_variance": category_components[
            "raw_participant_by_category"
        ],
    }


def full_pipeline_bootstrap(
    index,
    categories: np.ndarray,
    *,
    repeats: int,
    workers: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cubes = target_cube(index)
    groups = category_order(categories)
    seeds = [seed + repeat * 104729 for repeat in range(repeats)]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=max(1, workers),
        initializer=_init_full_pipeline_worker,
        initargs=(cubes, groups),
    ) as executor:
        for repeat, result in enumerate(
            executor.map(_full_pipeline_draw, seeds, chunksize=1), start=1
        ):
            rows.append({"draw": repeat, **result})
            if repeat % max(1, repeats // 10) == 0:
                print(
                    f"full-pipeline bootstrap={repeat}/{repeats}", flush=True
                )

    def values(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=np.float64)

    negative_counts = {
        "participant": int(np.sum(values("raw_participant_variance") < 0)),
        "video": int(np.sum(values("raw_video_variance") < 0)),
        "category_participant": int(
            np.sum(values("raw_category_participant_variance") < 0)
        ),
        "video_within_category": int(
            np.sum(values("raw_video_within_category_variance") < 0)
        ),
        "participant_by_category": int(
            np.sum(values("raw_participant_by_category_variance") < 0)
        ),
    }
    return rows, {
        "draws": repeats,
        "workers": workers,
        "protocol": (
            "Participants are resampled as newly indexed pseudo-participants; "
            "each pseudo-participant's leave-one-pseudo-index reference and "
            "DTW targets are rebuilt before category-stratified video resampling."
        ),
        "relative_g_15_ci95": interval(values("relative_g_15")),
        "absolute_phi_15_ci95": interval(values("absolute_phi_15")),
        "category_relative_g_15_ci95": interval(
            values("category_relative_g_15")
        ),
        "category_absolute_phi_15_ci95": interval(
            values("category_absolute_phi_15")
        ),
        "relative_g_15_above_070_fraction": float(
            np.mean(values("relative_g_15") > 0.70)
        ),
        "absolute_phi_15_above_070_fraction": float(
            np.mean(values("absolute_phi_15") > 0.70)
        ),
        "negative_raw_component_counts": negative_counts,
    }


def shifted_trace_mae(
    target: np.ndarray,
    template: np.ndarray,
    shift_seconds: float,
    *,
    band_seconds: int = 10,
) -> float:
    target = np.asarray(target, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    shift = float(np.clip(shift_seconds, -band_seconds, band_seconds))
    reference_times = np.arange(band_seconds, len(target) - band_seconds)
    if not len(reference_times):
        raise ValueError("Trial is too short for common-interior trace correction")
    participant_times = reference_times.astype(np.float64) + shift
    time = np.arange(len(target), dtype=np.float64)
    corrected = np.column_stack(
        [
            np.interp(participant_times, time, target[:, axis])
            for axis in range(target.shape[1])
        ]
    )
    return float(np.abs(corrected - template[reference_times]).mean())


def select_signed_shrinkage(
    matrix: np.ndarray,
    *,
    budget: int,
    repeats: int,
    seed: int,
) -> float:
    subsets = calibration_sets(
        participants=len(matrix), budget=budget, repeats=repeats, seed=seed
    )
    scores = {value: [] for value in ADDITIVE_SHRINKAGE}
    for participant in range(len(matrix)):
        reference = np.delete(matrix, participant, axis=0)
        video_mean = reference.mean(axis=0)
        target = matrix[participant]
        for repeat in range(repeats):
            calibration = subsets[(participant, repeat)]
            evaluation = np.setdiff1d(np.arange(15), calibration)
            for shrinkage in ADDITIVE_SHRINKAGE:
                prediction = additive_personalization(
                    video_mean,
                    calibration,
                    target[calibration],
                    float(shrinkage),
                )
                scores[shrinkage].append(
                    float(np.abs(prediction[evaluation] - target[evaluation]).mean())
                )
    return float(
        min(ADDITIVE_SHRINKAGE, key=lambda value: (np.mean(scores[value]), value))
    )


def signed_correction(
    folds: list[dict[str, object]],
    *,
    repeats: int,
    inner_repeats: int,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold_info in folds:
        fold = int(fold_info["fold"])
        training_b = np.asarray(fold_info["training_b"], dtype=np.float64)
        validation_b = np.asarray(fold_info["validation_b"], dtype=np.float64)
        validation_subjects = np.asarray(fold_info["validation_subjects"])
        validation_positions = validation_subjects.astype(int) - 1
        video_mean_b = training_b.mean(axis=0)
        cubes = fold_info["cubes"]
        templates = fold_info["validation_templates"]
        for budget in CALIBRATION_BUDGETS:
            shrinkage = select_signed_shrinkage(
                training_b,
                budget=budget,
                repeats=inner_repeats,
                seed=seed + fold * 1000 + budget * 10,
            )
            subsets = calibration_sets(
                participants=len(validation_subjects),
                budget=budget,
                repeats=repeats,
                seed=seed + 50000 + fold * 1000 + budget * 10,
            )
            for local, subject in enumerate(validation_subjects):
                target_b = validation_b[local]
                for repeat in range(repeats):
                    calibration = subsets[(local, repeat)]
                    evaluation = np.setdiff1d(np.arange(15), calibration)
                    predicted_b = additive_personalization(
                        video_mean_b,
                        calibration,
                        target_b[calibration],
                        shrinkage,
                    )
                    for video in evaluation:
                        target = cubes[video][validation_positions[local]]
                        template = templates[local][video]
                        no_correction = shifted_trace_mae(target, template, 0.0)
                        video_correction = shifted_trace_mae(
                            target, template, video_mean_b[video]
                        )
                        calibrated = shifted_trace_mae(
                            target, template, predicted_b[video]
                        )
                        oracle = shifted_trace_mae(target, template, target_b[video])
                        rows.append(
                            {
                                "fold": fold,
                                "subject": int(subject),
                                "budget": budget,
                                "repeat": repeat,
                                "video": int(video) + 1,
                                "selected_shrinkage": shrinkage,
                                "true_b_seconds": float(target_b[video]),
                                "video_mean_b_seconds": float(video_mean_b[video]),
                                "calibrated_b_seconds": float(predicted_b[video]),
                                "no_correction_trace_mae_units": no_correction,
                                "video_only_trace_mae_units": video_correction,
                                "calibrated_trace_mae_units": calibrated,
                                "oracle_b_trace_mae_units": oracle,
                                "video_gain_vs_none_units": no_correction
                                - video_correction,
                                "calibrated_gain_vs_none_units": no_correction
                                - calibrated,
                                "calibrated_gain_vs_video_units": video_correction
                                - calibrated,
                                "oracle_gain_vs_none_units": no_correction - oracle,
                                "signed_bias_absolute_error_video_seconds": abs(
                                    video_mean_b[video] - target_b[video]
                                ),
                                "signed_bias_absolute_error_calibrated_seconds": abs(
                                    predicted_b[video] - target_b[video]
                                ),
                            }
                        )

    rng = np.random.default_rng(seed + 90000)
    summary: dict[str, object] = {}
    for budget in CALIBRATION_BUDGETS:
        selected = [row for row in rows if int(row["budget"]) == budget]
        by_subject = {}
        for subject in range(1, 25):
            subject_rows = [row for row in selected if int(row["subject"]) == subject]
            by_subject[subject] = {
                name: float(np.mean([float(row[name]) for row in subject_rows]))
                for name in (
                    "video_gain_vs_none_units",
                    "calibrated_gain_vs_none_units",
                    "calibrated_gain_vs_video_units",
                    "oracle_gain_vs_none_units",
                    "signed_bias_absolute_error_video_seconds",
                    "signed_bias_absolute_error_calibrated_seconds",
                )
            }
        bootstrap = {
            name: np.empty(bootstrap_repeats, dtype=np.float64)
            for name in (
                "calibrated_gain_vs_none_units",
                "calibrated_gain_vs_video_units",
                "oracle_gain_vs_none_units",
            )
        }
        subjects = np.arange(1, 25)
        for repeat in range(bootstrap_repeats):
            draw = rng.choice(subjects, size=24, replace=True)
            for name in bootstrap:
                bootstrap[name][repeat] = float(
                    np.mean([by_subject[int(subject)][name] for subject in draw])
                )
        summary[str(budget)] = {
            "mean_video_gain_vs_none_units": float(
                np.mean([value["video_gain_vs_none_units"] for value in by_subject.values()])
            ),
            "mean_calibrated_gain_vs_none_units": float(
                np.mean(
                    [value["calibrated_gain_vs_none_units"] for value in by_subject.values()]
                )
            ),
            "calibrated_gain_vs_none_ci95": interval(
                bootstrap["calibrated_gain_vs_none_units"]
            ),
            "mean_calibrated_gain_vs_video_units": float(
                np.mean(
                    [value["calibrated_gain_vs_video_units"] for value in by_subject.values()]
                )
            ),
            "calibrated_gain_vs_video_ci95": interval(
                bootstrap["calibrated_gain_vs_video_units"]
            ),
            "mean_oracle_gain_vs_none_units": float(
                np.mean([value["oracle_gain_vs_none_units"] for value in by_subject.values()])
            ),
            "oracle_gain_vs_none_ci95": interval(
                bootstrap["oracle_gain_vs_none_units"]
            ),
            "mean_signed_bias_mae_video_seconds": float(
                np.mean(
                    [
                        value["signed_bias_absolute_error_video_seconds"]
                        for value in by_subject.values()
                    ]
                )
            ),
            "mean_signed_bias_mae_calibrated_seconds": float(
                np.mean(
                    [
                        value["signed_bias_absolute_error_calibrated_seconds"]
                        for value in by_subject.values()
                    ]
                )
            ),
            "participants_trace_improved_vs_none": int(
                np.sum(
                    [
                        value["calibrated_gain_vs_none_units"] > 0
                        for value in by_subject.values()
                    ]
                )
            ),
        }
    return rows, {
        "endpoint": (
            "Mean absolute valence/arousal trace error against the fold-safe "
            "reference over the common 10-s interior after a constant signed "
            "timestamp correction. This is proximal trace correction, not user benefit."
        ),
        "budgets": summary,
    }


def paired_crossed_interval(
    difference: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> list[float]:
    values = np.asarray(difference, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        participants = rng.integers(0, values.shape[0], values.shape[0])
        videos = rng.integers(0, values.shape[1], values.shape[1])
        draws[repeat] = values[np.ix_(participants, videos)].mean()
    return interval(draws)


def baseline_comparison_from_predictions(
    methods: Sequence[str],
    predictions: np.ndarray,
    target: np.ndarray,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    methods = [str(value) for value in methods]
    predictions = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    selected = (
        "video_mean",
        "context_only",
        "context_plus_blockwise_residual",
    )
    errors = {
        name: np.abs(predictions[methods.index(name)] - target) for name in selected
    }
    rows = []
    for name in selected:
        rows.append(
            {
                "method": name,
                "participant_macro_mae_seconds": float(
                    errors[name].mean(axis=1).mean()
                ),
            }
        )
    context_minus_video = errors["context_only"] - errors["video_mean"]
    video_minus_sensor = (
        errors["video_mean"] - errors["context_plus_blockwise_residual"]
    )
    context_minus_sensor = (
        errors["context_only"] - errors["context_plus_blockwise_residual"]
    )
    return rows, {
        "video_mean_advantage_over_context_seconds": float(
            context_minus_video.mean()
        ),
        "video_mean_advantage_over_context_ci95": paired_crossed_interval(
            context_minus_video, repeats=bootstrap_repeats, seed=seed
        ),
        "sensor_gain_over_video_mean_seconds": float(video_minus_sensor.mean()),
        "sensor_gain_over_video_mean_ci95": paired_crossed_interval(
            video_minus_sensor, repeats=bootstrap_repeats, seed=seed + 1
        ),
        "sensor_gain_over_context_seconds": float(context_minus_sensor.mean()),
        "sensor_gain_over_context_ci95": paired_crossed_interval(
            context_minus_sensor, repeats=bootstrap_repeats, seed=seed + 2
        ),
        "decision": (
            "Video mean is the point-best no-sensor baseline; its small advantage "
            "over context is reported with paired uncertainty. Deployment should "
            "retain the best no-sensor baseline rather than assume context wins."
        ),
    }


def baseline_comparison(
    revision3_dir: Path,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cache = np.load(
        revision3_dir / "matched_target_predictions.npz", allow_pickle=False
    )
    return baseline_comparison_from_predictions(
        [str(value) for value in cache["methods"].tolist()],
        cache["predictions"],
        cache["target"],
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )


MI_INT8 = 1
MI_UINT8 = 2
MI_INT32 = 5
MI_UINT32 = 6
MI_MATRIX = 14
MI_COMPRESSED = 15


def _parse_small_elements(buffer: bytes) -> tuple[str, tuple[int, ...]]:
    offset = 0

    def element() -> tuple[int, bytes]:
        nonlocal offset
        raw = struct.unpack_from("<I", buffer, offset)[0]
        small_nbytes = raw >> 16
        if small_nbytes:
            data_type = raw & 0xFFFF
            data = buffer[offset + 4 : offset + 4 + small_nbytes]
            offset += 8
            return data_type, data
        data_type, nbytes = struct.unpack_from("<II", buffer, offset)
        start = offset + 8
        end = start + nbytes
        offset = end + ((8 - nbytes % 8) % 8)
        return data_type, buffer[start:end]

    element()  # flags
    dim_type, dim_data = element()
    if dim_type not in (MI_INT32, MI_UINT32):
        raise ValueError("Unsupported MATLAB dimension type")
    dtype = "<i4" if dim_type == MI_INT32 else "<u4"
    dims = tuple(int(value) for value in np.frombuffer(dim_data, dtype=dtype))
    name_type, name_data = element()
    if name_type not in (MI_INT8, MI_UINT8):
        raise ValueError("Unsupported MATLAB variable-name type")
    return name_data.decode("utf-8").rstrip("\x00"), dims


def mat_v5_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    output: dict[str, tuple[int, ...]] = {}
    with path.open("rb") as handle:
        header = handle.read(128)
        if b"MATLAB 5.0 MAT-file" not in header[:116]:
            raise ValueError(f"Unsupported MAT file: {path}")
        while True:
            tag = handle.read(8)
            if not tag:
                break
            if len(tag) != 8:
                raise ValueError(f"Truncated MAT tag: {path}")
            data_type, nbytes = struct.unpack("<II", tag)
            if data_type == MI_MATRIX:
                preview = handle.read(min(nbytes, 512))
                name, dims = _parse_small_elements(preview)
                output[name] = dims
                handle.seek(nbytes - len(preview), 1)
            elif data_type == MI_COMPRESSED:
                raise ValueError(
                    f"Compressed MAT shape audit is unsupported for large file {path}"
                )
            else:
                handle.seek(nbytes, 1)
            padding = (8 - nbytes % 8) % 8
            if padding:
                handle.seek(padding, 1)
    return output


def sampling_rate_audit(
    data_root: Path, index
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    eeg_rates = []
    fnirs_rates = []
    eeg_baseline_counts = []
    fnirs_baseline_counts = []
    for subject in range(1, 25):
        subject_name = f"test_{subject}"
        subject_dir = data_root / "data" / subject_name
        eeg_video_shapes = mat_v5_shapes(subject_dir / "EEG_videos.mat")
        eeg_baseline_shapes = mat_v5_shapes(subject_dir / "EEG_baselines.mat")
        fnirs_video_shapes = mat_v5_shapes(subject_dir / "fNIRS_videos.mat")
        fnirs_baseline_shapes = mat_v5_shapes(subject_dir / "fNIRS_baselines.mat")
        for video in range(1, 16):
            key = f"video_{video}"
            label_count = len(trial_rows(index, subject, video))
            eeg_samples = int(eeg_video_shapes[key][-1])
            fnirs_samples = int(fnirs_video_shapes[key][-1])
            eeg_rate = eeg_samples / label_count
            fnirs_rate = fnirs_samples / label_count
            eeg_baseline = int(eeg_baseline_shapes[key][-1])
            fnirs_baseline = int(fnirs_baseline_shapes[key][-1])
            eeg_rates.append(eeg_rate)
            fnirs_rates.append(fnirs_rate)
            eeg_baseline_counts.append(eeg_baseline)
            fnirs_baseline_counts.append(fnirs_baseline)
            rows.append(
                {
                    "subject": subject,
                    "video": video,
                    "label_seconds": label_count,
                    "eeg_video_samples": eeg_samples,
                    "eeg_samples_per_label_second": eeg_rate,
                    "eeg_baseline_samples": eeg_baseline,
                    "fnirs_video_samples": fnirs_samples,
                    "fnirs_samples_per_label_second": fnirs_rate,
                    "fnirs_baseline_samples": fnirs_baseline,
                }
            )
        print(f"sampling audit subject={subject}/24", flush=True)
    eeg_rates_array = np.asarray(eeg_rates)
    fnirs_rates_array = np.asarray(fnirs_rates)
    return rows, {
        "dataset_card": {"eeg_hz": 200.0, "fnirs_hz": 47.62},
        "observed_video_samples_per_label_second": {
            "eeg": quantile_summary(eeg_rates_array),
            "fnirs": quantile_summary(fnirs_rates_array),
        },
        "observed_five_second_baseline_samples": {
            "eeg_unique": sorted(set(eeg_baseline_counts)),
            "fnirs_unique": sorted(set(fnirs_baseline_counts)),
        },
        "adopted_analysis_interpretation": (
            "Every EEG trial has exactly 1,000 array samples per 1-Hz label "
            "second and every stated five-second baseline has 5,000 samples. "
            "The analysis therefore treats 1,000 Hz as the released array-index "
            "density and reduces it to 200 Hz for CBraMod. This resolves internal "
            "array timing but does not assert the undocumented acquisition hardware rate."
        ),
    }


def signal_quality_audit(
    innovation_cache: Path,
    data_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    quality_files = sorted((innovation_cache / "trials").glob("*_quality.npy"))
    quality = np.concatenate(
        [np.load(path, mmap_mode="r", allow_pickle=False) for path in quality_files]
    ).astype(np.float64)
    masks = load_reservation_masks(data_root / "fNIRS_reservations.csv")
    counts = np.asarray([mask.sum() for mask in masks.values()], dtype=np.float64)
    names = (
        "eeg_log1p_median_peak_to_peak",
        "eeg_flat_channel_ratio",
        "eeg_log1p_median_derivative_rms",
        "eeg_log1p_50hz_to_broadband_ratio_x1000",
        "eeg_flat_or_extreme_channel_ratio",
        "fnirs_reserved_channel_ratio",
        "fnirs_motion_spike_ratio",
        "fnirs_extreme_robust_z_ratio",
        "fnirs_flat_reserved_channel_ratio",
    )
    rows = []
    for position, name in enumerate(names):
        values = quality[:, position]
        rows.append(
            {
                "metric": name,
                **quantile_summary(values),
                "nonzero_second_fraction": float(np.mean(values > 0)),
            }
        )
    return rows, {
        "seconds_screened": int(len(quality)),
        "trials_screened": len(quality_files),
        "all_trials_retained": True,
        "eeg": {
            "flat_channel_ratio_mean": float(quality[:, 1].mean()),
            "flat_or_extreme_channel_ratio_mean": float(quality[:, 4].mean()),
            "seconds_with_any_flat_or_extreme_channel_fraction": float(
                np.mean(quality[:, 4] > 0)
            ),
            "thresholds": "flat s.d.<0.1 microvolt; extreme peak-to-peak>500 microvolts",
        },
        "fnirs": {
            "reservation_file_used_for_qc": True,
            "reserved_channels_mean": float(counts.mean()),
            "reserved_channels_range": [int(counts.min()), int(counts.max())],
            "motion_spike_ratio_mean": float(quality[:, 6].mean()),
            "seconds_with_any_motion_spike_fraction": float(
                np.mean(quality[:, 6] > 0)
            ),
            "extreme_robust_z_ratio_mean": float(quality[:, 7].mean()),
            "flat_reserved_channel_ratio_mean": float(quality[:, 8].mean()),
        },
        "boundary": (
            "These are secondary array-level screens, not source acquisition QC; "
            "they do not reconstruct rereferencing, notch filtering, EOG/EMG, "
            "motion annotations, or provider-side bad-channel decisions."
        ),
    }


def build_reservation_aware_fnirs(
    index,
    feature_cache: Path,
    innovation_cache: Path,
    output_path: Path,
) -> None:
    source = np.load(
        feature_cache / "fnirs.npy", mmap_mode="r", allow_pickle=False
    )
    masks = np.load(
        innovation_cache / "reservation_masks.npy", allow_pickle=False
    ).astype(np.float32)
    if source.shape != (len(index.targets), 51, 90):
        raise ValueError(f"Unexpected fNIRS feature shape: {source.shape}")
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(index.targets), 180),
    )
    subject_positions = index.subject_numbers.astype(int) - 1
    for start in range(0, len(index.targets), 512):
        end = min(start + 512, len(index.targets))
        block = np.asarray(source[start:end], dtype=np.float32)
        mask = masks[subject_positions[start:end]][:, :, None]
        count = np.maximum(mask.sum(axis=1), 1.0)
        mean = (block * mask).sum(axis=1) / count
        variance = (
            np.square(block - mean[:, None, :]) * mask
        ).sum(axis=1) / count
        output[start:end, :90] = mean
        output[start:end, 90:] = np.sqrt(np.maximum(variance, 0.0))
    output.flush()


def reservation_sensing(
    index,
    primary_matrix: np.ndarray,
    feature_cache: Path,
    innovation_cache: Path,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
]:
    with tempfile.TemporaryDirectory(prefix="revision4_fnirs_", dir="/tmp") as raw:
        cache = Path(raw)
        os.symlink(innovation_cache / "cbramod", cache / "cbramod")
        os.symlink(
            innovation_cache / "handcrafted_eeg_pooled.npy",
            cache / "handcrafted_eeg_pooled.npy",
        )
        os.symlink(
            innovation_cache / "phase_sensing_targets",
            cache / "phase_sensing_targets",
        )
        build_reservation_aware_fnirs(
            index,
            feature_cache,
            innovation_cache,
            cache / "handcrafted_fnirs_pooled.npy",
        )
        predictions, rows, summary, selections, _ = run_nested_sensing(
            index,
            primary_matrix,
            cache,
            bootstrap_repeats=bootstrap_repeats,
            seed=seed,
        )
    baseline_rows, baseline_summary = baseline_comparison_from_predictions(
        METHODS,
        predictions,
        primary_matrix,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed + 1000,
    )
    reservation_summary = {
        "pooling": (
            "Per-participant fNIRS channel means and standard deviations use "
            "only channels marked 1 in fNIRS_reservations.csv."
        ),
        "nested_matched_target_sensing": summary,
        "selections": selections,
    }
    return rows, reservation_summary, baseline_rows, baseline_summary


def main() -> None:
    args = parse_args()
    data_root = resolve(args.data_root)
    feature_cache = resolve(args.feature_cache)
    innovation_cache = resolve(args.innovation_cache)
    revision2_dir = resolve(args.revision2_dir)
    revision3_dir = resolve(args.revision3_dir)
    trait_dir = resolve(args.trait_dir)
    output_dir = resolve(args.output_dir)
    prepare_output(output_dir, args.overwrite)
    started = time.perf_counter()
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": "running",
            "started_at_utc": utc_now(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "configuration": manifest_configuration(args),
        },
    )

    index = load_innovation_index(data_root)
    categories = emotion_categories(data_root / "Targeted_emotions.txt")

    sampling_rows, sampling_summary = sampling_rate_audit(data_root, index)
    write_rows(output_dir / "sampling_rate_audit.csv", sampling_rows)
    quality_rows, quality_summary = signal_quality_audit(
        innovation_cache, data_root
    )
    write_rows(output_dir / "signal_quality_summary.csv", quality_rows)

    matrices, signed_lopo, consensus_audit = build_estimand_matrices(index)
    np.savez_compressed(
        output_dir / "estimand_targets.npz",
        **{name: matrix.astype(np.float32) for name, matrix in matrices.items()},
        signed_bias=signed_lopo.astype(np.float32),
    )
    geometry = geometry_tensor(revision2_dir / "geometry_covariates.csv")
    estimand_rows, estimand_summary = estimator_robustness(
        matrices,
        geometry,
        bootstrap_repeats=args.bootstrap_repeats,
        geometry_bootstrap_repeats=args.geometry_bootstrap_repeats,
        seed=args.seed + 1000,
    )
    write_rows(output_dir / "estimand_robustness.csv", estimand_rows)

    folds = fold_estimand_matrices(index)
    calibration_rows, calibration_summary = estimator_calibration(
        folds,
        repeats=args.calibration_repeats,
        inner_repeats=args.calibration_inner_repeats,
        seed=args.seed + 2000,
    )
    write_rows(output_dir / "estimand_calibration.csv", calibration_rows)

    dtw_rows, dtw_summary = objective_audit(
        index,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed + 3000,
    )
    write_rows(output_dir / "dtw_objective_audit.csv", dtw_rows)

    category_rows, category_summary = category_gstudy(
        matrices["consensus"],
        categories,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed + 4000,
    )
    write_rows(output_dir / "category_gstudy.csv", category_rows)

    full_rows, full_summary = full_pipeline_bootstrap(
        index,
        categories,
        repeats=args.full_pipeline_repeats,
        workers=args.full_pipeline_workers,
        seed=args.seed + 5000,
    )
    write_rows(output_dir / "full_pipeline_bootstrap.csv", full_rows)

    signed_rows, signed_summary = signed_correction(
        folds,
        repeats=args.calibration_repeats,
        inner_repeats=args.calibration_inner_repeats,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed + 6000,
    )
    write_rows(output_dir / "signed_correction.csv", signed_rows)

    reservation_summary: dict[str, object]
    if args.skip_reservation_sensing:
        baseline_rows, baseline_summary = baseline_comparison(
            revision3_dir,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed + 7000,
        )
        reservation_summary = {"status": "skipped_by_cli"}
    else:
        primary_matrix = np.load(
            trait_dir / "phase_trait_matrix.npz", allow_pickle=False
        )["mean_absolute_lag"].astype(np.float64)
        (
            reservation_rows,
            reservation_summary,
            baseline_rows,
            baseline_summary,
        ) = reservation_sensing(
            index,
            primary_matrix,
            feature_cache,
            innovation_cache,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed + 8000,
        )
        write_rows(
            output_dir / "reservation_sensing_results.csv", reservation_rows
        )
    write_rows(output_dir / "baseline_comparison.csv", baseline_rows)

    summary = {
        "sampling_rate_audit": sampling_summary,
        "signal_quality_audit": quality_summary,
        "consensus_mapping_audit": consensus_audit,
        "estimand_robustness": estimand_summary,
        "estimand_calibration": calibration_summary,
        "dtw_objective_audit": dtw_summary,
        "category_gstudy": category_summary,
        "full_pipeline_reference_bootstrap": full_summary,
        "signed_trace_correction": signed_summary,
        "no_sensor_baseline": baseline_summary,
        "reservation_aware_sensing": reservation_summary,
        "claim_boundary": (
            "The new analyses test estimator dependence and proximal trace "
            "correction. They do not add a second session, unseen interfaces, "
            "real-time inference, or observed user benefit/harm."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": "complete",
            "started_at_utc": json.loads(
                (output_dir / "run_manifest.json").read_text(encoding="utf-8")
            )["started_at_utc"],
            "completed_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "configuration": manifest_configuration(args),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
