#!/usr/bin/env python3
"""Evaluate robust population priors plus bounded post-trial SAM anchors.

This is an algorithm experiment, not a manuscript generator.  It writes full
out-of-fold predictions, fold selections, hierarchy-aware uncertainty, and
fold-allocation sensitivity to ``artifacts/sparse_anchor_optimization``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.innovation.data import load_innovation_index  # noqa: E402
from merps.sparse_anchor import (  # noqa: E402
    DEFAULT_ANCHOR_CANDIDATES,
    DEFAULT_CONDITIONAL_CANDIDATES,
    DEFAULT_ENSEMBLE_WEIGHTS,
    DEFAULT_FUNCTIONAL_CANDIDATES,
    DEFAULT_PRIOR_CANDIDATES,
    build_trial_table,
    nested_sparse_anchor_oof,
    repeated_outer_folds,
    trial_gain_matrix,
    trial_macro_mae,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "MER_PS_trainval"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "sparse_anchor_optimization"
MANAGED_FILES = {
    "metrics.csv",
    "oof_predictions.npz",
    "repeat_metrics.csv",
    "results.json",
    "run_manifest.json",
    "trial_metrics.csv",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sam-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent CV repeats to evaluate concurrently.",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--refresh-summaries",
        action="store_true",
        help="Refresh metrics/result metadata from saved OOF outputs without refitting.",
    )
    return parser.parse_args(argv)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metrics(index, table, prediction: np.ndarray) -> dict[str, float]:
    error = np.abs(np.asarray(prediction, dtype=np.float64) - index.targets)
    participant_scores = [
        float(error[index.subject_numbers == subject].mean())
        for subject in sorted(set(int(value) for value in index.subject_numbers))
    ]
    return {
        "sample_mae": float(error.mean()),
        "valence_mae": float(error[:, 0].mean()),
        "arousal_mae": float(error[:, 1].mean()),
        "participant_macro_mae": float(np.mean(participant_scores)),
        "trial_macro_mae": trial_macro_mae(index, table, prediction),
    }


def evaluate_repeat(payload):
    """Worker entry point; it performs no writes and returns deterministic data."""

    repeat, index, table, plan, inner_folds = payload
    result = nested_sparse_anchor_oof(
        index,
        table,
        outer_folds=plan,
        inner_folds=inner_folds,
    )
    repeat_metrics = {
        name: metrics(index, table, prediction)
        for name, prediction in result.predictions.items()
    }
    return int(repeat), result, repeat_metrics


def gain_uncertainty(
    gain: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    """Participant and crossed participant-by-video uncertainty for MAE gain."""

    gain = np.asarray(gain, dtype=np.float64)
    if gain.ndim != 2:
        raise ValueError("gain must be a subject-by-video matrix")
    rng = np.random.default_rng(seed)
    observed = float(gain.mean())
    participant_mean = gain.mean(axis=1)
    participant_bootstrap = np.empty(repeats, dtype=np.float64)
    crossed_bootstrap = np.empty(repeats, dtype=np.float64)
    sign_flip = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled_subjects = rng.integers(0, gain.shape[0], size=gain.shape[0])
        sampled_videos = rng.integers(0, gain.shape[1], size=gain.shape[1])
        participant_bootstrap[repeat] = participant_mean[sampled_subjects].mean()
        crossed_bootstrap[repeat] = gain[
            sampled_subjects[:, None], sampled_videos[None, :]
        ].mean()
        signs = rng.choice((-1.0, 1.0), size=gain.shape[0])
        sign_flip[repeat] = float(np.mean(participant_mean * signs))
    interval = lambda values: [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]
    return {
        "observed_trial_macro_mae_gain": observed,
        "positive_means_comparator_is_better": True,
        "participants": int(gain.shape[0]),
        "videos": int(gain.shape[1]),
        "participant_cluster_bootstrap_ci95": interval(participant_bootstrap),
        "participant_video_crossed_bootstrap_ci95": interval(crossed_bootstrap),
        "participant_sign_flip_p_two_sided": float(
            (1 + np.sum(np.abs(sign_flip) >= abs(observed))) / (repeats + 1)
        ),
        "participants_improved": int(np.sum(participant_mean > 0.0)),
        "participant_gain_median": float(np.median(participant_mean)),
        "participant_gain_iqr": [
            float(np.quantile(participant_mean, 0.25)),
            float(np.quantile(participant_mean, 0.75)),
        ],
        "worst_participant_gain": float(participant_mean.min()),
        "best_participant_gain": float(participant_mean.max()),
        "resamples": int(repeats),
    }


def trial_rows(index, table, predictions: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for trial, rows in enumerate(table.rows):
        record: dict[str, object] = {
            "subject": int(table.subjects[trial]),
            "video": int(table.videos[trial]),
            "seconds": int(len(rows)),
            "sam_rating_valence": float(
                1.0 + (table.anchors[trial, 0] - 1.0) * 8.0 / 254.0
            ),
            "sam_rating_arousal": float(
                1.0 + (table.anchors[trial, 1] - 1.0) * 8.0 / 254.0
            ),
            "sam_anchor_continuous_valence": float(table.anchors[trial, 0]),
            "sam_anchor_continuous_arousal": float(table.anchors[trial, 1]),
        }
        for name, prediction in predictions.items():
            record[f"{name}_mae"] = float(
                np.abs(prediction[rows] - index.targets[rows]).mean()
            )
        record["canonical_gain_of_sparse_anchor"] = (
            record["canonical_prior_mae"]
            - record["bounded_sparse_anchor_mae"]
        )
        record["matched_prior_gain_of_sparse_anchor"] = (
            record["anchor_reference_prior_mae"]
            - record["bounded_sparse_anchor_mae"]
        )
        record["canonical_gain_of_conditional_sparse_anchor"] = (
            record["canonical_prior_mae"]
            - record["conditional_sparse_anchor_mae"]
        )
        record["bounded_gain_of_conditional_sparse_anchor"] = (
            record["bounded_sparse_anchor_mae"]
            - record["conditional_sparse_anchor_mae"]
        )
        record["conditional_prior_gain_of_sparse_anchor"] = (
            record["conditional_reference_prior_mae"]
            - record["conditional_sparse_anchor_mae"]
        )
        record["conditional_gain_of_functional_ensemble"] = (
            record["conditional_sparse_anchor_mae"]
            - record["ensemble_sparse_anchor_mae"]
        )
        record["canonical_gain_of_functional_ensemble"] = (
            record["canonical_prior_mae"]
            - record["ensemble_sparse_anchor_mae"]
        )
        rows_out.append(record)
    return rows_out


def build_results(
    *,
    index,
    table,
    predictions: dict[str, np.ndarray],
    selections: Sequence[dict[str, object]],
    repeat_rows: list[dict[str, object]],
    inner_folds: int,
    bootstrap_repeats: int,
    seed: int,
    elapsed_seconds: float,
) -> tuple[dict[str, object], dict[str, dict[str, float]]]:
    """Build deterministic summaries from saved predictions and selections."""

    primary_metrics = {
        name: metrics(index, table, prediction)
        for name, prediction in predictions.items()
    }
    effects = {
        "ensemble_sparse_anchor_vs_conditional_sparse_anchor": gain_uncertainty(
            trial_gain_matrix(
                index,
                table,
                predictions["conditional_sparse_anchor"],
                predictions["ensemble_sparse_anchor"],
            ),
            repeats=bootstrap_repeats,
            seed=seed + 300,
        ),
        "ensemble_sparse_anchor_vs_canonical_prior": gain_uncertainty(
            trial_gain_matrix(
                index,
                table,
                predictions["canonical_prior"],
                predictions["ensemble_sparse_anchor"],
            ),
            repeats=bootstrap_repeats,
            seed=seed + 400,
        ),
        "conditional_sparse_anchor_vs_canonical_prior": gain_uncertainty(
            trial_gain_matrix(
                index,
                table,
                predictions["canonical_prior"],
                predictions["conditional_sparse_anchor"],
            ),
            repeats=bootstrap_repeats,
            seed=seed + 500,
        ),
        "conditional_sparse_anchor_vs_bounded_sparse_anchor": gain_uncertainty(
            trial_gain_matrix(
                index,
                table,
                predictions["bounded_sparse_anchor"],
                predictions["conditional_sparse_anchor"],
            ),
            repeats=bootstrap_repeats,
            seed=seed + 600,
        ),
        "conditional_sparse_anchor_vs_conditional_reference_prior": (
            gain_uncertainty(
                trial_gain_matrix(
                    index,
                    table,
                    predictions["conditional_reference_prior"],
                    predictions["conditional_sparse_anchor"],
                ),
                repeats=bootstrap_repeats,
                seed=seed + 700,
            )
        ),
        "bounded_sparse_anchor_vs_canonical_prior": gain_uncertainty(
            trial_gain_matrix(
                index,
                table,
                predictions["canonical_prior"],
                predictions["bounded_sparse_anchor"],
            ),
            repeats=bootstrap_repeats,
            seed=seed + 1000,
        ),
        "bounded_sparse_anchor_vs_matched_selected_prior": gain_uncertainty(
            trial_gain_matrix(
                index,
                table,
                predictions["anchor_reference_prior"],
                predictions["bounded_sparse_anchor"],
            ),
            repeats=bootstrap_repeats,
            seed=seed + 2000,
        ),
        "independent_robust_prior_vs_canonical_prior": gain_uncertainty(
            trial_gain_matrix(
                index,
                table,
                predictions["canonical_prior"],
                predictions["independent_robust_prior"],
            ),
            repeats=bootstrap_repeats,
            seed=seed + 3000,
        ),
    }
    repeat_gains = np.asarray(
        [float(row["gain_vs_canonical_prior"]) for row in repeat_rows]
    )
    results = {
        "schema_version": "sparse-anchor-optimization-v7",
        "research_mode": {
            "name": "post_trial_sparse_anchor_reconstruction",
            "zero_interaction": False,
            "continuous_target_trace_used_at_inference": False,
            "post_trial_inputs": "one 1--9 SAM valence rating and one 1--9 SAM arousal rating",
            "real_time_prediction": False,
            "boundary": (
                "This mode reconstructs a completed trial after sparse self-report. "
                "It is not a zero-interaction or online EEG/fNIRS decoder."
            ),
        },
        "novelty_scope": {
            "project_delta": (
                "The project previously used only a zero-interaction video-time "
                "prior for absolute trajectories and did not exploit released "
                "post-trial SAM ratings."
            ),
            "algorithmic_combination": (
                "nested robust population-trajectory selection + fixed-scale "
                "uniform or Gaussian-weighted axis/joint same-video retrieval conditioned on "
                "sparse SAM ratings + shrinkage toward an unconditional prior + "
                "separately selected center/shape correction with a hard cap + "
                "a ridge-shrunk smooth residual-function expert + an inner-selected "
                "convex ensemble with an exact retrieval-only fallback"
            ),
            "scientific_novelty_status": (
                "incremental algorithmic combination, not the first use of "
                "sparse self-reports to recover dense emotion trajectories"
            ),
            "closest_prior_art": {
                "citation": (
                    "Jolly et al., Recovering Individual Emotional States from "
                    "Sparse Ratings Using Collaborative Filtering, Affective "
                    "Science 3, 799--817 (2022)"
                ),
                "doi": "10.1007/s42761-022-00161-2",
                "overlap": (
                    "recovers dense individual emotion ratings from sparse "
                    "self-reports using collaborative filtering, including KNN "
                    "and NNMF on naturalistic video time series"
                ),
                "remaining_delta": (
                    "one global post-trial two-axis SAM report rather than "
                    "observed time-point labels; known-stimulus trajectory "
                    "retrieval; participant-disjoint nested selection; explicit "
                    "population fallback and bounded correction"
                ),
            },
            "construct_risk": (
                "A single retrospective SAM response has no time localization "
                "and can differ from continuous ratings because of recall and "
                "peak-end effects; the method reconstructs a trajectory but "
                "does not make the global rating temporally resolved evidence."
            ),
        },
        "evidence_status": {
            "estimate_type": "iterative development-set nested out-of-fold estimate",
            "independent_confirmatory_test": False,
            "candidate_development_note": (
                "The conditional and joint retrieval candidate families were "
                "expanded after inspecting earlier OOF results on this same "
                "dataset. Participant-disjoint nesting prevents within-run "
                "target leakage, but it does not turn iterative reuse of the "
                "dataset into independent confirmation."
            ),
            "required_next_test": (
                "freeze code and candidates, then evaluate once on an untouched "
                "compatible dataset or prospectively collected cohort"
            ),
        },
        "protocol": {
            "outer_evaluation": "five participant-disjoint folds",
            "inner_selection": (
                f"{inner_folds} participant-disjoint folds within each outer training set"
            ),
            "anchor_mapping": "fixed affine map from SAM [1,9] to continuous labels [1,255]",
            "correction": (
                "m=median_t(prior); clip(center_weight*(SAM_anchor-m) "
                "- shape_weight*(prior_t-m), -cap, cap)"
            ),
            "weight_constraint": (
                "center_weight <= 0.30, shape_weight <= 0.20, cap <= 20; at "
                "least 80% of the population shape is retained before clipping"
            ),
            "selection_metric": "participant-by-video trial-macro MAE",
            "fallback": (
                "center_weight=shape_weight=0 is an explicit inner-fold candidate"
            ),
            "same_prior_increment": (
                "anchor_reference_prior is the exact prior jointly selected with the anchor"
            ),
            "conditional_retrieval": (
                "within each video, compare axis-specific absolute-SAM distance "
                "with a shared two-axis Euclidean neighborhood; inner selection "
                "chooses uniform KNN or a numerically stable Gaussian bandwidth, "
                "then shrinks the retrieved trajectory toward the selected prior"
            ),
            "conditional_selection": (
                "a second inner-only stage selects mode, neighbor count, bandwidth, "
                "retrieval mix and bounded anchor parameters; disabled retrieval is explicit"
            ),
            "functional_residual": (
                "within each video and axis, regress the dense residual function on "
                "the global SAM displacement, ridge-shrink and smooth its slope, "
                "then cap the resulting correction"
            ),
            "functional_ensemble": (
                "a final inner-only stage selects a convex weight between local "
                "trajectory retrieval and global residual-function regression; "
                "conditional weight 1.0 is an exact fallback"
            ),
        },
        "samples": int(len(index.targets)),
        "participants": int(len(set(int(value) for value in index.subject_numbers))),
        "trials": int(len(table.rows)),
        "primary_metrics": primary_metrics,
        "primary_effects": effects,
        "primary_selections": list(selections),
        "fold_allocation_sensitivity": {
            "repeats": int(len(repeat_rows)),
            "gain_vs_canonical_prior_min": float(repeat_gains.min()),
            "gain_vs_canonical_prior_max": float(repeat_gains.max()),
            "gain_vs_canonical_prior_mean": float(repeat_gains.mean()),
            "positive_partitions": int(np.sum(repeat_gains > 0.0)),
            "total_partitions": int(len(repeat_gains)),
        },
        "elapsed_seconds": float(elapsed_seconds),
    }
    return results, primary_metrics


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_root = resolve(args.data_root)
    sam_path = resolve(args.sam_path) if args.sam_path else data_root / "SAM_score.csv"
    output_dir = resolve(args.output_dir)
    if args.refresh_summaries:
        output_dir.mkdir(parents=True, exist_ok=True)
        required = {
            "oof_predictions.npz",
            "repeat_metrics.csv",
            "results.json",
        }
        missing = sorted(
            name for name in required if not (output_dir / name).is_file()
        )
        if missing:
            raise FileNotFoundError(
                "Cannot refresh summaries; missing: " + ", ".join(missing)
            )
    else:
        prepare_output(output_dir, args.overwrite)
    started = time.perf_counter()

    manifest = {
        "status": "running",
        "updated_at_utc": utc_now(),
        "algorithm": "sam_kernel_retrieval_plus_functional_residual_ensemble",
        "configuration": {
            "data_root": project_path(data_root),
            "sam_path": project_path(sam_path),
            "sam_sha256": file_sha256(sam_path),
            "inner_folds": int(args.inner_folds),
            "cv_repeats": int(args.cv_repeats),
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "workers": int(args.workers),
            "seed": int(args.seed),
            "prior_candidates": [spec.name for spec in DEFAULT_PRIOR_CANDIDATES],
            "anchor_candidates": [
                spec.name for spec in DEFAULT_ANCHOR_CANDIDATES
            ],
            "conditional_candidates": [
                spec.name for spec in DEFAULT_CONDITIONAL_CANDIDATES
            ],
            "functional_candidates": [
                spec.name for spec in DEFAULT_FUNCTIONAL_CANDIDATES
            ],
            "ensemble_weights": list(DEFAULT_ENSEMBLE_WEIGHTS),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "generated_files": [],
    }
    if not args.refresh_summaries:
        write_json(output_dir / "run_manifest.json", manifest)

    index = load_innovation_index(data_root)
    table = build_trial_table(index, sam_path)
    subjects = sorted(set(int(value) for value in index.subject_numbers))
    if args.refresh_summaries:
        saved = np.load(output_dir / "oof_predictions.npz", allow_pickle=False)
        predictions = {
            name: np.asarray(saved[name], dtype=np.float32)
            for name in (
                "canonical_prior",
                "independent_robust_prior",
                "anchor_reference_prior",
                "bounded_sparse_anchor",
                "conditional_reference_prior",
                "conditional_sparse_anchor",
                "functional_sparse_anchor",
                "ensemble_sparse_anchor",
            )
        }
        if not np.array_equal(saved["sample_ids"], index.sample_ids):
            raise RuntimeError("Saved OOF sample order does not match current data")
        previous = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        with (output_dir / "repeat_metrics.csv").open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            repeat_rows = list(csv.DictReader(handle))
        results, primary_metrics = build_results(
            index=index,
            table=table,
            predictions=predictions,
            selections=previous["primary_selections"],
            repeat_rows=repeat_rows,
            inner_folds=args.inner_folds,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed,
            elapsed_seconds=float(previous.get("elapsed_seconds", 0.0)),
        )
        write_rows(
            output_dir / "metrics.csv",
            [{"model": name, **value} for name, value in primary_metrics.items()],
        )
        write_rows(output_dir / "trial_metrics.csv", trial_rows(index, table, predictions))
        write_json(output_dir / "results.json", results)
        manifest.update(
            {
                "status": "complete",
                "updated_at_utc": utc_now(),
                "elapsed_seconds": results["elapsed_seconds"],
                "generated_files": sorted(
                    item.name
                    for item in output_dir.iterdir()
                    if item.name != "run_manifest.json"
                ),
                "summary_refresh_without_refit": True,
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        print("complete: summaries refreshed without refitting", flush=True)
        return

    plans = repeated_outer_folds(
        subjects,
        repeats=args.cv_repeats,
        folds=5,
        seed=args.seed,
    )

    tasks = [
        (repeat, index, table, plan, int(args.inner_folds))
        for repeat, plan in enumerate(plans)
    ]
    workers = max(1, min(int(args.workers), len(tasks)))
    if workers == 1:
        completed = [evaluate_repeat(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            completed = list(executor.map(evaluate_repeat, tasks))

    repeat_rows: list[dict[str, object]] = []
    primary = None
    for repeat, result, repeat_metrics in sorted(completed, key=lambda item: item[0]):
        canonical = repeat_metrics["canonical_prior"]["trial_macro_mae"]
        matched = repeat_metrics["anchor_reference_prior"]["trial_macro_mae"]
        bounded = repeat_metrics["bounded_sparse_anchor"]["trial_macro_mae"]
        conditional_reference = repeat_metrics[
            "conditional_reference_prior"
        ]["trial_macro_mae"]
        conditional = repeat_metrics["conditional_sparse_anchor"][
            "trial_macro_mae"
        ]
        functional = repeat_metrics["functional_sparse_anchor"][
            "trial_macro_mae"
        ]
        ensemble = repeat_metrics["ensemble_sparse_anchor"][
            "trial_macro_mae"
        ]
        repeat_rows.append(
            {
                "repeat": int(repeat),
                "canonical_prior_trial_macro_mae": canonical,
                "anchor_reference_prior_trial_macro_mae": matched,
                "bounded_sparse_anchor_trial_macro_mae": bounded,
                "conditional_reference_prior_trial_macro_mae": (
                    conditional_reference
                ),
                "conditional_sparse_anchor_trial_macro_mae": conditional,
                "functional_sparse_anchor_trial_macro_mae": functional,
                "ensemble_sparse_anchor_trial_macro_mae": ensemble,
                "gain_vs_canonical_prior": canonical - ensemble,
                "gain_vs_bounded_sparse_anchor": bounded - ensemble,
                "gain_vs_conditional_reference_prior": (
                    conditional_reference - ensemble
                ),
                "gain_vs_conditional_sparse_anchor": conditional - ensemble,
                "bounded_gain_vs_canonical_prior": canonical - bounded,
            }
        )
        print(
            f"repeat={repeat} canonical={canonical:.6f} "
            f"conditional={conditional:.6f} ensemble={ensemble:.6f} "
            f"gain={canonical - ensemble:.6f}",
            flush=True,
        )
        if repeat == 0:
            primary = result

    assert primary is not None
    results, primary_metrics = build_results(
        index=index,
        table=table,
        predictions=primary.predictions,
        selections=primary.selections,
        repeat_rows=repeat_rows,
        inner_folds=args.inner_folds,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
        elapsed_seconds=float(time.perf_counter() - started),
    )

    write_rows(
        output_dir / "metrics.csv",
        [{"model": name, **value} for name, value in primary_metrics.items()],
    )
    write_rows(output_dir / "repeat_metrics.csv", repeat_rows)
    write_rows(
        output_dir / "trial_metrics.csv",
        trial_rows(index, table, primary.predictions),
    )
    np.savez_compressed(
        output_dir / "oof_predictions.npz",
        sample_ids=index.sample_ids,
        target=index.targets,
        **primary.predictions,
    )
    write_json(output_dir / "results.json", results)

    generated = sorted(
        item.name for item in output_dir.iterdir() if item.name != "run_manifest.json"
    )
    manifest.update(
        {
            "status": "complete",
            "updated_at_utc": utc_now(),
            "elapsed_seconds": results["elapsed_seconds"],
            "generated_files": generated,
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)
    print(
        "complete: "
        f"trial-MAE={primary_metrics['ensemble_sparse_anchor']['trial_macro_mae']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
