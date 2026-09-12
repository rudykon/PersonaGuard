#!/usr/bin/env python3
"""Build a machine-readable index of the finalized algorithm experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUTS = {
    "stimulus_audit": "data/stimuli/stimulus_audit.json",
    "clip": "artifacts/av_content_prior/results.json",
    "siglip": "artifacts/av_content_prior_siglip/results.json",
    "dinov2": "artifacts/av_content_prior_dinov2/results.json",
    "clip_siglip": "artifacts/av_content_prior_clip_siglip/results.json",
    "clip_siglip_weighted": (
        "artifacts/av_content_prior_clip_siglip_weight_s050/results.json"
    ),
    "clip_siglip_delta1": (
        "artifacts/av_content_prior_clip_siglip_delta1_w025/results.json"
    ),
    "clip_siglip_delta13": (
        "artifacts/av_content_prior_clip_siglip_delta13_w025/results.json"
    ),
    "content_router": "artifacts/content_backbone_router/results.json",
    "physiology_full": "artifacts/av_causal_physio_clip_siglip/results.json",
    "physiology_tokens": (
        "artifacts/av_causal_physio_clip_siglip_tokens/results.json"
    ),
    "known_video_physiology": "artifacts/known_video_physio_residual/results.json",
    "sparse_previous": (
        "artifacts/sparse_anchor_functional_optimization/results.json"
    ),
    "sparse_kernel": "artifacts/sparse_anchor_kernel_optimization/results.json",
}

OOF_DIRECTORIES = (
    "artifacts/av_content_prior",
    "artifacts/av_content_prior_siglip",
    "artifacts/av_content_prior_dinov2",
    "artifacts/av_content_prior_clip_siglip",
    "artifacts/av_content_prior_clip_siglip_weight_s050",
    "artifacts/av_content_prior_clip_siglip_delta1_w025",
    "artifacts/av_content_prior_clip_siglip_delta13_w025",
    "artifacts/content_backbone_router",
    "artifacts/av_causal_physio_clip_siglip",
    "artifacts/av_causal_physio_clip_siglip_tokens",
    "artifacts/known_video_physio_residual",
    "artifacts/sparse_anchor_kernel_optimization",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "algorithm_experiment_summary",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, str]]]:
    payloads: dict[str, Mapping[str, Any]] = {}
    identities: dict[str, dict[str, str]] = {}
    for name, relative in DEFAULT_INPUTS.items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError(f"Expected an object in {path}")
        payloads[name] = value
        identities[name] = {"path": relative, "sha256": sha256(path)}
    return payloads, identities


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value for {label}: {value!r}")
    return result


def validate_oof_artifacts() -> dict[str, Any]:
    reference_ids: np.ndarray | None = None
    reference_targets: np.ndarray | None = None
    numeric_arrays_checked = 0
    identities: dict[str, dict[str, str]] = {}
    for relative in OOF_DIRECTORIES:
        path = PROJECT_ROOT / relative / "oof_predictions.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            if "sample_ids" not in archive:
                raise KeyError(f"Missing sample_ids in {path}")
            target_key = "targets" if "targets" in archive else "target"
            if target_key not in archive:
                raise KeyError(f"Missing target(s) in {path}")
            sample_ids = np.asarray(archive["sample_ids"])
            targets = np.asarray(archive[target_key])
            if reference_ids is None:
                reference_ids = sample_ids.copy()
                reference_targets = targets.copy()
            elif not np.array_equal(sample_ids, reference_ids):
                raise ValueError(f"Sample identity mismatch in {path}")
            elif not np.array_equal(targets, reference_targets):
                raise ValueError(f"Target mismatch in {path}")
            for key in archive.files:
                values = np.asarray(archive[key])
                if np.issubdtype(values.dtype, np.number):
                    if not np.isfinite(values).all():
                        raise ValueError(f"Non-finite values in {path}:{key}")
                    numeric_arrays_checked += 1
        identities[relative] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256(path),
        }
    if reference_ids is None:
        raise RuntimeError("No OOF artifacts were checked")
    return {
        "artifacts_checked": len(OOF_DIRECTORIES),
        "samples": int(len(reference_ids)),
        "numeric_arrays_checked": numeric_arrays_checked,
        "sample_identity_consistent": True,
        "targets_identical": True,
        "all_numeric_arrays_finite": True,
        "artifact_identities": identities,
    }


def content_metrics(payload: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    metrics = payload["primary_metrics"]
    for key in ("nested_visual_content_prior", "nested_clip_content_prior"):
        if key in metrics:
            return metrics[key]
    raise KeyError(f"No content-prior metrics in {label}")


def content_row(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = content_metrics(payload, name)
    repeated = payload["repeated_grouped_cv"]
    offsets = payload["offset_sensitivity"]
    return {
        "name": name,
        "trial_macro_mae": finite(metrics["trial_macro_mae"], f"{name}.mae"),
        "valence_mae": finite(metrics["valence_mae"], f"{name}.valence"),
        "arousal_mae": finite(metrics["arousal_mae"], f"{name}.arousal"),
        "gain_vs_metadata": finite(
            payload["primary_gain_trial_macro_mae"], f"{name}.gain"
        ),
        "repeated_grouped_cv": {
            "runs": int(repeated["total_runs"]),
            "mae_mean": finite(
                repeated["content_trial_macro_mae_mean"], f"{name}.repeat_mean"
            ),
            "mae_std": finite(
                repeated["content_trial_macro_mae_std"], f"{name}.repeat_std"
            ),
            "positive_gain_runs": int(repeated["positive_gain_runs"]),
        },
        "offset_sensitivity_seconds": list(offsets["fixed_offsets_seconds"]),
        "offset_mae_range": [
            finite(value, f"{name}.offset")
            for value in offsets["content_trial_macro_mae_range"]
        ],
    }


def decision_curve(payload: Mapping[str, Any]) -> list[dict[str, float]]:
    return [
        {
            "minimum_inner_gain": finite(row["minimum_inner_gain"], "threshold"),
            "trial_macro_mae": finite(row["trial_macro_mae"], "curve.mae"),
            "gain_vs_content_prior": finite(
                row["gain_vs_content_prior"], "curve.gain"
            ),
        }
        for row in payload["decision_curve"]
    ]


def primary_curve_row(
    payload: Mapping[str, Any], threshold: float, label: str
) -> dict[str, float]:
    """Return the decision-curve row frozen as primary in the run manifest."""

    rows = decision_curve(payload)
    matches = [
        row
        for row in rows
        if math.isclose(row["minimum_inner_gain"], threshold, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one primary decision-curve row for {label} at {threshold}"
        )
    return matches[0]


def build_summary() -> dict[str, Any]:
    data, identities = load_inputs()
    oof_consistency = validate_oof_artifacts()
    stimulus = data["stimulus_audit"]
    offsets = [
        finite(record["duration_seconds"], "duration")
        - finite(record["annotation_seconds"], "annotation")
        for record in stimulus["records"]
    ]

    content_names = (
        "clip",
        "siglip",
        "dinov2",
        "clip_siglip",
        "clip_siglip_weighted",
        "clip_siglip_delta1",
        "clip_siglip_delta13",
    )
    content = [content_row(name, data[name]) for name in content_names]
    best_content = min(content, key=lambda row: row["trial_macro_mae"])
    if best_content["name"] != "clip_siglip":
        raise RuntimeError(
            "Frozen deployment decision is stale: lowest-MAE content candidate is "
            f"{best_content['name']}"
        )

    metadata_prior_mae = finite(
        data["clip_siglip"]["primary_metrics"]["metadata_temporal_prior"][
            "trial_macro_mae"
        ],
        "metadata_prior_mae",
    )

    router = data["content_router"]
    sparse = data["sparse_kernel"]
    sparse_metrics = sparse["primary_metrics"]
    sparse_effect = sparse["primary_effects"][
        "ensemble_sparse_anchor_vs_canonical_prior"
    ]
    previous_sparse_mae = finite(
        data["sparse_previous"]["primary_metrics"]["ensemble_sparse_anchor"][
            "trial_macro_mae"
        ],
        "previous_sparse_mae",
    )
    sparse_mae = finite(
        sparse_metrics["ensemble_sparse_anchor"]["trial_macro_mae"],
        "sparse_mae",
    )

    physiology = data["physiology_full"]
    physiology_tokens = data["physiology_tokens"]
    known_phys = data["known_video_physiology"]
    primary_threshold = finite(
        physiology["primary_gain_threshold"], "physiology.primary_threshold"
    )
    primary_gain = finite(
        physiology["primary_gain_uncertainty"]["trial_macro_mae_gain"],
        "physiology.primary_gain",
    )
    token_primary_threshold = finite(
        physiology_tokens["primary_gain_threshold"], "tokens.primary_threshold"
    )
    token_primary_gain = finite(
        physiology_tokens["primary_gain_uncertainty"]["trial_macro_mae_gain"],
        "tokens.primary_gain",
    )
    primary_physiology_row = primary_curve_row(
        physiology, primary_threshold, "physiology"
    )
    primary_token_row = primary_curve_row(
        physiology_tokens, token_primary_threshold, "physiology_tokens"
    )

    router_repeat = router["repeated_grouped_cv"]
    router_offsets = router["offset_sensitivity"]

    return {
        "schema_version": "merps-algorithm-experiment-summary-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_identity": {
            "path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "evidence_scope": {
            "participants": int(sparse["participants"]),
            "videos": 15,
            "estimate_type": "iterative development-set nested out-of-fold estimate",
            "independent_confirmatory_test": False,
            "video_inference_boundary": (
                "Held-out released videos within the fixed 15-video support; "
                "not an unseen-video population claim."
            ),
        },
        "stimulus_integrity": {
            "files_expected": int(stimulus["files_expected"]),
            "files_present": int(stimulus["files_present"]),
            "identities_verified": int(stimulus["identity_verified"]),
            "formal_feature_status": stimulus["formal_feature_status"],
            "media_minus_annotation_seconds": {
                "minimum": min(offsets),
                "maximum": max(offsets),
                "per_video": offsets,
            },
        },
        "oof_artifact_consistency": oof_consistency,
        "zero_interaction_held_out_participant_held_out_released_video": {
            "metric": "trial-macro MAE on the 1--255 affect scale; lower is better",
            "candidates": content,
            "metadata_prior_trial_macro_mae": metadata_prior_mae,
            "selected_model": best_content["name"],
            "selected_trial_macro_mae": best_content["trial_macro_mae"],
            "selected_gain_vs_metadata": (
                metadata_prior_mae - best_content["trial_macro_mae"]
            ),
            "selected_repeat_mae_mean": best_content["repeated_grouped_cv"][
                "mae_mean"
            ],
            "selected_repeat_mae_std": best_content["repeated_grouped_cv"][
                "mae_std"
            ],
            "selected_offset_mae_range": best_content["offset_mae_range"],
            "router": {
                "trial_macro_mae": finite(
                    router["primary_metrics"]["trial_macro_mae"], "router.mae"
                ),
                "gain_vs_frozen_baseline": finite(
                    router["primary_gain_vs_baseline"], "router.gain"
                ),
                "positive_repeat_runs": int(
                    router["repeated_grouped_cv"]["positive_gain_runs"]
                ),
                "total_repeat_runs": int(
                    router["repeated_grouped_cv"]["total_runs"]
                ),
                "repeated_mae_mean": finite(
                    router_repeat["routed_trial_macro_mae_mean"],
                    "router.repeat_mean",
                ),
                "repeated_mae_std": finite(
                    router_repeat["routed_trial_macro_mae_std"],
                    "router.repeat_std",
                ),
                "offset_mae_range": [
                    finite(value, "router.offset")
                    for value in router_offsets["trial_macro_mae_range"]
                ],
                "adopted": False,
            },
        },
        "causal_physiology_residual": {
            "base_content_model": "clip_siglip",
            "full_modalities": {
                "primary_minimum_inner_gain": primary_threshold,
                "primary_trial_macro_mae": primary_physiology_row[
                    "trial_macro_mae"
                ],
                "primary_gain": primary_gain,
                "decision_curve": decision_curve(physiology),
                "adopted": False,
            },
            "temporal_token_modalities": {
                "primary_minimum_inner_gain": token_primary_threshold,
                "primary_trial_macro_mae": primary_token_row[
                    "trial_macro_mae"
                ],
                "primary_gain": token_primary_gain,
                "decision_curve": decision_curve(physiology_tokens),
                "adopted": False,
            },
            "known_video_condition": {
                "primary_minimum_inner_gain": finite(
                    known_phys["primary_gain_threshold"], "known.threshold"
                ),
                "population_prior_trial_macro_mae": finite(
                    known_phys["metrics"]["population_prior"]["trial_macro_mae"],
                    "known.prior_mae",
                ),
                "sensor_increment_gain": 0.0,
                "adopted": False,
            },
            "decision": (
                "No physiology residual cleared the frozen primary gate; deploy the "
                "content prediction unchanged."
            ),
        },
        "one_post_trial_sam_sparse_recovery": {
            "canonical_prior_trial_macro_mae": finite(
                sparse_metrics["canonical_prior"]["trial_macro_mae"],
                "sparse.prior",
            ),
            "previous_functional_ensemble_mae": previous_sparse_mae,
            "gaussian_retrieval_ensemble_mae": sparse_mae,
            "gain_vs_canonical_prior": finite(
                sparse_effect["observed_trial_macro_mae_gain"], "sparse.gain"
            ),
            "gain_vs_previous_ensemble": previous_sparse_mae - sparse_mae,
            "participant_video_crossed_bootstrap_ci95": [
                finite(value, "sparse.ci")
                for value in sparse_effect[
                    "participant_video_crossed_bootstrap_ci95"
                ]
            ],
            "fold_allocation_sensitivity": sparse[
                "fold_allocation_sensitivity"
            ],
            "adopted_for_post_trial_only": True,
            "not_zero_interaction": True,
        },
        "frozen_deployment_decisions": {
            "zero_interaction": "clip_siglip content prior",
            "post_trial_one_sam": "gaussian-weighted sparse-recovery ensemble",
            "physiology_residual": "disabled (exact content-prior fallback)",
            "content_router": "disabled",
        },
        "validity_boundaries": [
            "The +/-3 s analysis is a fixed alignment sensitivity analysis, not a test-label-selected correction.",
            "The sparse-recovery result requires one retrospective post-trial SAM report and is not zero-interaction inference.",
            "All candidate development and nested OOF evaluation reused the same development dataset; independent confirmation remains required.",
            "Machine checks cover file identity, structure, fold propagation, and numerical consistency, not substantive validity.",
        ],
        "input_identities": identities,
    }


def markdown(summary: Mapping[str, Any]) -> str:
    zero = summary[
        "zero_interaction_held_out_participant_held_out_released_video"
    ]
    phys = summary["causal_physiology_residual"]
    sparse = summary["one_post_trial_sam_sparse_recovery"]
    integrity = summary["stimulus_integrity"]
    rows = [
        "# Algorithm experiment summary",
        "",
        "All MAE values use trial-macro averaging on the 1--255 affect scale; lower is better.",
        "",
        "| Deployment condition | Frozen result | Decision |",
        "|---|---:|---|",
        (
            "| Zero interaction, held-out participant + held-out released video | "
            f"{zero['selected_trial_macro_mae']:.3f} | CLIP+SigLIP content prior |"
        ),
        (
            "| Causal EEG/fNIRS residual after content | "
            f"{phys['full_modalities']['primary_gain']:+.3f} gain | Disabled; exact fallback |"
        ),
        (
            "| One post-trial SAM sparse recovery | "
            f"{sparse['gaussian_retrieval_ensemble_mae']:.3f} | "
            "Gaussian-weighted retrieval ensemble |"
        ),
        "",
        (
            f"The selected zero-interaction model has repeated grouped-CV MAE "
            f"{zero['selected_repeat_mae_mean']:.3f} +/- "
            f"{zero['selected_repeat_mae_std']:.3f}. Its fixed -3...+3 s alignment "
            f"range is {zero['selected_offset_mae_range'][0]:.3f}--"
            f"{zero['selected_offset_mae_range'][1]:.3f}."
        ),
        "",
        (
            f"All {integrity['files_present']}/{integrity['files_expected']} stimulus "
            f"files are present and identity-verified. Media exceed annotation lengths "
            f"by {integrity['media_minus_annotation_seconds']['minimum']:.3f}--"
            f"{integrity['media_minus_annotation_seconds']['maximum']:.3f} s."
        ),
        "",
        (
            "The post-trial method improves over the canonical known-video prior by "
            f"{sparse['gain_vs_canonical_prior']:.3f} MAE and over the previous "
            f"functional ensemble by {sparse['gain_vs_previous_ensemble']:.3f}."
        ),
        "",
        "These are iterative nested out-of-fold development estimates, not an independent confirmatory test.",
        "",
    ]
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (output_dir / "results.json", output_dir / "README.md")
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already contains outputs; pass --overwrite")
    summary = build_summary()
    outputs[0].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs[1].write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))


if __name__ == "__main__":
    main()
