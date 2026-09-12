#!/usr/bin/env python3
"""Evaluate phase--amplitude decomposition of naturalistic affect trajectories.

Primary evidence is cross-dimensional: a warp estimated from valence must
improve arousal alignment, and vice versa. This prevents the guaranteed
in-sample error reduction of unconstrained time warping from being mistaken
for a scientific contribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.innovation.data import load_innovation_index, outer_subject_folds
from merps.innovation.normative_dynamics import (
    WarpConfig,
    circular_shift_null,
    decompose_trial,
    normative_trajectory,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "MER_PS_trainval"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "normative_dynamics"
MANAGED_FILES = {
    "participant_profiles.csv",
    "phase_targets.npz",
    "run_manifest.json",
    "summary.json",
    "trial_metrics.csv",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--band-seconds", type=int, default=10)
    parser.add_argument("--warp-penalty", type=float, default=0.15)
    parser.add_argument("--step-penalty", type=float, default=0.05)
    parser.add_argument("--smoothing-radius", type=int, default=2)
    parser.add_argument("--use-levels", action="store_true")
    parser.add_argument("--null-repeats", type=int, default=50)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--signflip-repeats", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_output(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = list(path.iterdir())
    if not existing:
        return
    if not overwrite:
        names = ", ".join(sorted(item.name for item in existing[:8]))
        raise FileExistsError(f"{path} is not empty ({names}); pass --overwrite")
    unknown = [item for item in existing if item.name not in MANAGED_FILES]
    if unknown:
        raise FileExistsError(
            "Refusing to overwrite unmanaged files: "
            + ", ".join(sorted(item.name for item in unknown))
        )
    for item in existing:
        if item.is_file() or item.is_symlink():
            item.unlink()


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def correlation(first: np.ndarray, second: np.ndarray, *, spearman: bool = False) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if spearman:
        first, second = _rankdata(first), _rankdata(second)
    if len(first) < 3 or first.std() < 1e-12 or second.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def percentile_interval(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def hierarchical_inference(
    records: list[dict[str, float | int]],
    *,
    value_key: str,
    bootstrap_repeats: int,
    signflip_repeats: int,
    seed: int,
) -> dict[str, object]:
    subjects = np.arange(1, 25, dtype=np.int16)
    videos = np.arange(1, 16, dtype=np.int16)
    matrix = np.full((len(subjects), len(videos)), np.nan, dtype=np.float64)
    for record in records:
        matrix[int(record["subject"]) - 1, int(record["video"]) - 1] = float(
            record[value_key]
        )
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"Incomplete participant-video matrix for {value_key}")
    observed = float(matrix.mean())
    rng = np.random.default_rng(seed)
    participant_boot = np.empty(bootstrap_repeats, dtype=np.float64)
    crossed_boot = np.empty(bootstrap_repeats, dtype=np.float64)
    for repeat in range(bootstrap_repeats):
        participant_draw = rng.integers(0, len(subjects), len(subjects))
        participant_boot[repeat] = matrix[participant_draw].mean()
        video_draw = rng.integers(0, len(videos), len(videos))
        crossed_boot[repeat] = matrix[np.ix_(participant_draw, video_draw)].mean()
    participant_values = matrix.mean(axis=1)
    signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(signflip_repeats, len(participant_values)),
    )
    null = (signs * participant_values[None, :]).mean(axis=1)
    signflip_p = float(
        (1 + np.sum(np.abs(null) >= abs(observed))) / (signflip_repeats + 1)
    )
    return {
        "observed_mean": observed,
        "participant_cluster_bootstrap": percentile_interval(participant_boot),
        "participant_video_crossed_bootstrap": percentile_interval(crossed_boot),
        "participant_signflip_p_two_sided": signflip_p,
        "participants": 24,
        "videos": 15,
        "trials": 360,
    }


def participant_profiles(
    records: list[dict[str, float | int]]
) -> tuple[list[dict[str, float | int]], dict[str, float]]:
    profiles = []
    for subject in range(1, 25):
        rows = [record for record in records if int(record["subject"]) == subject]
        odd = [record for record in rows if int(record["video"]) % 2 == 1]
        even = [record for record in rows if int(record["video"]) % 2 == 0]
        profiles.append(
            {
                "subject": subject,
                "mean_signed_lag": float(
                    np.mean([float(record["median_lag_seconds"]) for record in rows])
                ),
                "mean_absolute_lag": float(
                    np.mean(
                        [float(record["mean_absolute_lag_seconds"]) for record in rows]
                    )
                ),
                "mean_cross_gain": float(
                    np.mean([float(record["cross_gain_mean"]) for record in rows])
                ),
                "odd_video_signed_lag": float(
                    np.mean([float(record["median_lag_seconds"]) for record in odd])
                ),
                "even_video_signed_lag": float(
                    np.mean([float(record["median_lag_seconds"]) for record in even])
                ),
                "odd_video_absolute_lag": float(
                    np.mean(
                        [float(record["mean_absolute_lag_seconds"]) for record in odd]
                    )
                ),
                "even_video_absolute_lag": float(
                    np.mean(
                        [float(record["mean_absolute_lag_seconds"]) for record in even]
                    )
                ),
            }
        )
    odd_signed = np.asarray([row["odd_video_signed_lag"] for row in profiles])
    even_signed = np.asarray([row["even_video_signed_lag"] for row in profiles])
    odd_absolute = np.asarray([row["odd_video_absolute_lag"] for row in profiles])
    even_absolute = np.asarray([row["even_video_absolute_lag"] for row in profiles])
    reliability = {
        "signed_lag_split_half_pearson": correlation(odd_signed, even_signed),
        "signed_lag_split_half_spearman": correlation(
            odd_signed, even_signed, spearman=True
        ),
        "absolute_lag_split_half_pearson": correlation(odd_absolute, even_absolute),
        "absolute_lag_split_half_spearman": correlation(
            odd_absolute, even_absolute, spearman=True
        ),
    }
    return profiles, reliability


def run_null(
    trial_inputs: list[tuple[np.ndarray, np.ndarray]],
    config: WarpConfig,
    repeats: int,
    seed: int,
) -> np.ndarray:
    if repeats <= 0:
        return np.empty(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    output = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        gains = [
            circular_shift_null(target, template, config, rng)
            for target, template in trial_inputs
        ]
        output[repeat] = float(np.mean(gains))
        print(
            f"null repeat={repeat + 1:03d}/{repeats:03d} gain={output[repeat]:.5f}",
            flush=True,
        )
    return output


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    data_root = resolve(args.data_root)
    output_dir = resolve(args.output_dir)
    prepare_output(output_dir, args.overwrite)
    config = WarpConfig(
        band_seconds=args.band_seconds,
        warp_penalty=args.warp_penalty,
        step_penalty=args.step_penalty,
        smoothing_radius=args.smoothing_radius,
        use_derivative=not args.use_levels,
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": "running",
            "updated_at_utc": utc_now(),
            "configuration": {
                "data_root": str(data_root),
                "template_radius": args.radius,
                "warp": asdict(config),
                "null_repeats": args.null_repeats,
                "bootstrap_repeats": args.bootstrap_repeats,
                "signflip_repeats": args.signflip_repeats,
                "seed": args.seed,
            },
        },
    )
    started = time.perf_counter()
    index = load_innovation_index(data_root)
    n_samples = len(index.targets)
    fold_by_row = np.full(n_samples, -1, dtype=np.int8)
    mapping_v_by_row = np.full(n_samples, -1, dtype=np.int16)
    mapping_a_by_row = np.full(n_samples, -1, dtype=np.int16)
    lag_by_row = np.full(n_samples, np.nan, dtype=np.float32)
    amplitude_by_row = np.full((n_samples, 2), np.nan, dtype=np.float32)
    boundary_by_row = np.zeros(n_samples, dtype=bool)
    records: list[dict[str, float | int]] = []
    trial_inputs: list[tuple[np.ndarray, np.ndarray]] = []

    for fold, (training_subjects, validation_subjects) in enumerate(
        outer_subject_folds()
    ):
        templates = {
            video: normative_trajectory(
                index, training_subjects, video, radius=args.radius
            )[0]
            for video in range(1, 16)
        }
        for subject in validation_subjects:
            for video in range(1, 16):
                rows = np.flatnonzero(
                    (index.subject_numbers == int(subject))
                    & (index.videos == video)
                )
                rows = rows[np.argsort(index.timestamps[rows])]
                target = index.targets[rows].astype(np.float64)
                template = templates[video]
                if target.shape != template.shape:
                    raise RuntimeError(
                        f"Length mismatch subject={subject} video={video}: "
                        f"{target.shape} vs {template.shape}"
                    )
                decomposition = decompose_trial(target, template, config)
                fold_by_row[rows] = fold
                mapping_v_by_row[rows] = decomposition.mapping_from_valence
                mapping_a_by_row[rows] = decomposition.mapping_from_arousal
                lag_by_row[rows] = decomposition.consensus_lag
                amplitude_by_row[rows] = decomposition.amplitude_residual
                boundary_by_row[rows] = decomposition.boundary_mask
                record: dict[str, float | int] = {
                    "fold": fold,
                    "subject": int(subject),
                    "video": video,
                    "seconds": len(rows),
                }
                record.update(decomposition.metrics)
                records.append(record)
                trial_inputs.append((target, template))
        print(
            f"fold={fold} complete validation_subjects={validation_subjects.tolist()}",
            flush=True,
        )
    if (
        np.any(fold_by_row < 0)
        or np.any(mapping_v_by_row < 0)
        or np.any(mapping_a_by_row < 0)
        or not np.isfinite(lag_by_row).all()
        or not np.isfinite(amplitude_by_row).all()
    ):
        raise RuntimeError("OOF phase targets do not cover every sample exactly once")

    gain_inference = hierarchical_inference(
        records,
        value_key="cross_gain_mean",
        bootstrap_repeats=args.bootstrap_repeats,
        signflip_repeats=args.signflip_repeats,
        seed=args.seed + 1,
    )
    boundary_lag_inference = hierarchical_inference(
        [
            {
                **record,
                "boundary_minus_stable_lag": float(
                    record["boundary_absolute_lag_seconds"]
                )
                - float(record["stable_absolute_lag_seconds"]),
            }
            for record in records
        ],
        value_key="boundary_minus_stable_lag",
        bootstrap_repeats=args.bootstrap_repeats,
        signflip_repeats=args.signflip_repeats,
        seed=args.seed + 2,
    )
    boundary_amplitude_inference = hierarchical_inference(
        [
            {
                **record,
                "boundary_minus_stable_amplitude": float(
                    record["boundary_amplitude_deviation"]
                )
                - float(record["stable_amplitude_deviation"]),
            }
            for record in records
        ],
        value_key="boundary_minus_stable_amplitude",
        bootstrap_repeats=args.bootstrap_repeats,
        signflip_repeats=args.signflip_repeats,
        seed=args.seed + 3,
    )
    profiles, reliability = participant_profiles(records)
    null_distribution = run_null(
        trial_inputs, config, args.null_repeats, args.seed + 4
    )
    observed_gain = float(gain_inference["observed_mean"])
    permutation_p = (
        float(
            (1 + np.sum(null_distribution >= observed_gain))
            / (len(null_distribution) + 1)
        )
        if len(null_distribution)
        else None
    )
    participant_ci_low = float(
        gain_inference["participant_cluster_bootstrap"]["ci95_low"]
    )
    phase_supported = bool(
        observed_gain > 0
        and participant_ci_low > 0
        and permutation_p is not None
        and permutation_p < 0.05
    )
    baseline = np.mean(
        [
            (
                float(record["baseline_valence_mae"])
                + float(record["baseline_arousal_mae"])
            )
            / 2.0
            for record in records
        ]
    )
    summary = {
        "research_question": (
            "Do participant-specific temporal warps transfer across valence and "
            "arousal, indicating phase variation rather than mechanical DTW fit?"
        ),
        "primary_cross_dimension_gain": gain_inference,
        "baseline_trial_macro_mae": float(baseline),
        "phase_explained_fraction_of_baseline_mae": float(
            observed_gain / max(baseline, 1e-12)
        ),
        "trials_with_positive_cross_gain_fraction": float(
            np.mean([float(record["cross_gain_mean"]) > 0 for record in records])
        ),
        "circular_shift_null": {
            "repeats": int(len(null_distribution)),
            "mean": float(null_distribution.mean()) if len(null_distribution) else None,
            "ci95_low": (
                float(np.quantile(null_distribution, 0.025))
                if len(null_distribution)
                else None
            ),
            "ci95_high": (
                float(np.quantile(null_distribution, 0.975))
                if len(null_distribution)
                else None
            ),
            "one_sided_p_null_gain_at_least_observed": permutation_p,
        },
        "participant_phase_profile_reliability": reliability,
        "event_boundary_absolute_lag_effect": boundary_lag_inference,
        "event_boundary_amplitude_effect": boundary_amplitude_inference,
        "mean_lag_dimension_disagreement_seconds": float(
            np.mean(
                [
                    float(record["lag_dimension_disagreement_seconds"])
                    for record in records
                ]
            )
        ),
        "phase_gate": {
            "status": "pass" if phase_supported else "stop",
            "rule": (
                "cross-dimensional gain > 0, participant-bootstrap CI lower > 0, "
                "and circular-shift permutation p < 0.05"
            ),
            "next_step": (
                "predict phase/amplitude from physiology and evaluate selective sensing"
                if phase_supported
                else "do not build physiology models for this target; revise the construct"
            ),
        },
        "configuration": {
            "template_radius": args.radius,
            "warp": asdict(config),
            "outer_folds": 5,
            "participant_disjoint": True,
        },
        "literature_basis": [
            {
                "idea": "functional phase-amplitude separation",
                "doi": "10.1214/15-STS524",
            },
            {
                "idea": "continuous emotion annotation reaction-lag correction",
                "doi": "10.1109/TAFFC.2014.2334294",
            },
            {
                "idea": "time-varying annotator disagreement",
                "doi": "10.21437/Interspeech.2018-1933",
            },
            {
                "idea": "naturalistic intersubject affect consistency",
                "doi": "10.1016/j.bandc.2025.106295",
            },
        ],
        "elapsed_seconds": float(time.perf_counter() - started),
    }

    with (output_dir / "trial_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "participant_profiles.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profiles[0]))
        writer.writeheader()
        writer.writerows(profiles)
    np.savez_compressed(
        output_dir / "phase_targets.npz",
        sample_ids=index.sample_ids,
        subjects=index.subjects,
        subject_numbers=index.subject_numbers,
        videos=index.videos,
        timestamps=index.timestamps,
        targets=index.targets,
        outer_fold=fold_by_row,
        mapping_from_valence=mapping_v_by_row,
        mapping_from_arousal=mapping_a_by_row,
        consensus_lag_seconds=lag_by_row,
        amplitude_residual=amplitude_by_row,
        event_boundary=boundary_by_row,
        circular_shift_null_gain=null_distribution,
    )
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": "complete",
            "updated_at_utc": utc_now(),
            "configuration": summary["configuration"],
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "generated_files": sorted(MANAGED_FILES),
            "samples": n_samples,
            "trials": len(records),
            "phase_gate": summary["phase_gate"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
