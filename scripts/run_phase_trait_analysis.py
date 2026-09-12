#!/usr/bin/env python3
"""Treat temporal affect deviation as a participant-by-video measurement."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.innovation.data import load_innovation_index, outer_subject_folds
from merps.innovation.phase_trait import (
    corrected_item_total_correlations,
    crossed_bootstrap,
    estimate_variance_components,
    participant_label_permutation_test,
    reliability_curve,
    reliability_for_videos,
    videos_required,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "MER_PS_trainval"
DEFAULT_INNOVATION_CACHE = PROJECT_ROOT / "data" / "innovation_cache"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "phase_trait"
MANAGED_FILES = {
    "item_diagnostics.csv",
    "phase_trait_matrix.npz",
    "run_manifest.json",
    "summary.json",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--innovation-cache", type=Path, default=DEFAULT_INNOVATION_CACHE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--permutation-repeats", type=int, default=10000)
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


def held_out_trial_matrix(index, innovation_cache: Path) -> np.ndarray:
    """Assemble one fold-safe absolute-lag score for each participant/video."""

    matrix = np.full((24, 15), np.nan, dtype=np.float64)
    assigned = np.zeros(24, dtype=bool)
    for fold, (_, validation_subjects) in enumerate(outer_subject_folds()):
        target_cache = np.load(
            innovation_cache / "phase_sensing_targets" / f"fold_{fold}.npz",
            allow_pickle=False,
        )
        if (
            "target_version" not in target_cache.files
            or int(target_cache["target_version"]) != 2
        ):
            raise RuntimeError("Phase target cache is stale; rerun phase sensing")
        lag = target_cache["lag"]
        for subject in validation_subjects:
            subject_index = int(subject) - 1
            if assigned[subject_index]:
                raise RuntimeError(f"Participant {subject} appears in two folds")
            for video in range(1, 16):
                rows = np.flatnonzero(
                    (index.subject_numbers == int(subject))
                    & (index.videos == video)
                )
                matrix[subject_index, video - 1] = float(
                    np.abs(lag[rows]).mean()
                )
            assigned[subject_index] = True
    if not assigned.all() or not np.isfinite(matrix).all():
        raise RuntimeError("Held-out phase trait matrix is incomplete")
    return matrix


def main() -> None:
    args = parse_args()
    data_root = resolve(args.data_root)
    innovation_cache = resolve(args.innovation_cache)
    output_dir = resolve(args.output_dir)
    prepare_output(output_dir, args.overwrite)
    started = time.perf_counter()
    index = load_innovation_index(data_root)
    matrix = held_out_trial_matrix(index, innovation_cache)
    components = estimate_variance_components(matrix)
    curve = reliability_curve(components, matrix.shape[1])
    bootstrap = crossed_bootstrap(
        matrix,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
        maximum_videos=matrix.shape[1],
    )
    identity_test = participant_label_permutation_test(
        matrix,
        repeats=args.permutation_repeats,
        seed=args.seed + 1,
    )
    item_total = corrected_item_total_correlations(matrix)
    item_rows = [
        {
            "video": video + 1,
            "mean_absolute_lag_seconds": float(matrix[:, video].mean()),
            "participant_sd_seconds": float(matrix[:, video].std(ddof=1)),
            "corrected_item_total_correlation": float(item_total[video]),
        }
        for video in range(matrix.shape[1])
    ]
    bootstrap_curve = {
        int(value["videos"]): value
        for value in bootstrap["reliability_ci95"]
    }
    key_budgets = {}
    for count in (1, 2, 4, 8, 15):
        estimate = reliability_for_videos(components, count)
        interval = bootstrap_curve[count]
        key_budgets[str(count)] = {
            **estimate,
            "relative_g_ci95": interval["relative_g"],
            "absolute_phi_ci95": interval["absolute_phi"],
        }
    required = {
        str(threshold): {
            "relative_g": videos_required(
                components, threshold, coefficient="relative_g"
            ),
            "absolute_phi": videos_required(
                components, threshold, coefficient="absolute_phi"
            ),
        }
        for threshold in (0.60, 0.70, 0.80)
    }
    stable_trait_pass = (
        float(identity_test["p_one_sided"]) < 0.05
        and float(key_budgets["15"]["relative_g_ci95"][0]) >= 0.70
    )
    four_video_reliability = float(key_budgets["4"]["relative_g"])
    four_video_ci_low = float(key_budgets["4"]["relative_g_ci95"][0])
    if four_video_ci_low >= 0.70:
        brief_assessment = "high_reliability"
    elif four_video_reliability >= 0.50:
        brief_assessment = "moderate_not_high_stakes"
    else:
        brief_assessment = "low_reliability"
    summary = {
        "research_question": (
            "Is a participant's magnitude of temporal affect deviation a "
            "generalizable individual-difference measurement across videos?"
        ),
        "measurement": {
            "cell": "held-out participant-video mean absolute DTW lag",
            "participants": int(matrix.shape[0]),
            "videos": int(matrix.shape[1]),
            "target_leakage_control": (
                "each participant score uses the outer-fold normative template "
                "built only from other participants"
            ),
            "residual_component": (
                "participant-by-video interaction plus within-trial measurement error"
            ),
        },
        "g_study_variance_components": components.to_dict(),
        "d_study_key_budgets": key_budgets,
        "d_study_full_curve": curve,
        "videos_required_for_reliability": required,
        "crossed_participant_video_bootstrap": bootstrap,
        "participant_identity_permutation": identity_test,
        "item_diagnostics": item_rows,
        "trait_gate": {
            "status": "pass" if stable_trait_pass else "stop",
            "rule": (
                "participant identity permutation p < 0.05 and crossed-bootstrap "
                "lower CI for 15-video relative G >= 0.70"
            ),
            "interpretation": (
                "temporal-deviation magnitude is a stable cross-video trait"
                if stable_trait_pass
                else "do not claim a stable cross-video temporal-deviation trait"
            ),
            "brief_four_video_assessment": brief_assessment,
            "brief_calibration_caveat": (
                "four videos can improve prediction but do not establish a "
                "high-reliability individual diagnostic score"
            ),
        },
        "cross_domain_basis": [
            {
                "domain": "generalizability theory",
                "use": "decompose participant, video, and interaction/error variance",
                "doi": "10.1002/(SICI)1098-240X(199802)21:1<83::AID-NUR9>3.0.CO;2-P",
            },
            {
                "domain": "ERP reliability",
                "use": "precedent for stimulus/task G-studies in psychophysiology",
                "doi": "10.1016/j.ijpsycho.2021.02.015",
            },
            {
                "domain": "computerized adaptive testing",
                "use": "future training-only selection of informative calibration videos",
                "doi": "10.1177/014662168200600408",
            },
        ],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    np.savez_compressed(
        output_dir / "phase_trait_matrix.npz",
        mean_absolute_lag=matrix.astype(np.float32),
        subject_numbers=np.arange(1, 25, dtype=np.int16),
        videos=np.arange(1, 16, dtype=np.int16),
        target_version=np.asarray(2),
    )
    with (output_dir / "item_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(item_rows[0]))
        writer.writeheader()
        writer.writerows(item_rows)
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": "complete",
            "updated_at_utc": utc_now(),
            "configuration": {
                "bootstrap_repeats": args.bootstrap_repeats,
                "permutation_repeats": args.permutation_repeats,
                "seed": args.seed,
            },
            "trait_gate": summary["trait_gate"],
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "generated_files": sorted(MANAGED_FILES),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
