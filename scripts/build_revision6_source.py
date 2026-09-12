#!/usr/bin/env python3
"""Build the Revision-6 source from formal artifacts and authorized figure examples.

All manuscript tables, generated README blocks, and manuscript figures must read
paper_support/revision6_source.json rather than earlier revision directories. Upstream
artifacts remain the reproducible analysis record; this file is the sole
publication-facing numerical interface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper_support" / "revision6_source.json"

JSON_INPUTS = {
    "dense_trajectory_examples": ROOT / "paper_support" / "dense_trajectory_examples.json",
    "revision2": ROOT / "artifacts" / "revision2" / "summary.json",
    "revision3": ROOT / "artifacts" / "revision3" / "summary.json",
    "revision4": ROOT / "artifacts" / "revision4" / "summary.json",
    "revision5": ROOT / "artifacts" / "revision5" / "summary.json",
    "revision6": ROOT / "artifacts" / "revision6" / "summary.json",
    "measurement_robustness": ROOT
    / "artifacts"
    / "measurement_robustness"
    / "summary.json",
    "normative_dynamics": ROOT
    / "artifacts"
    / "normative_dynamics"
    / "summary.json",
    "evidence_traceability": ROOT / "paper_support" / "evidence_traceability.json",
    "protocol_replay_cases": ROOT / "paper_support" / "protocol_replay_cases.json",
    "protocol_replay": ROOT / "artifacts" / "revision6" / "protocol_replay.json",
    "protocol_rules": ROOT / "paper_support" / "protocol_rules.json",
    "external_reuse_cases": ROOT / "paper_support" / "external_reuse_cases.json",
    "adjacent_frameworks": ROOT / "paper_support" / "adjacent_frameworks.json",
    "algorithm_experiments": ROOT
    / "artifacts"
    / "algorithm_experiment_summary"
    / "results.json",
    "revision2_run_manifest": ROOT / "artifacts" / "revision2" / "run_manifest.json",
    "revision4_run_manifest": ROOT / "artifacts" / "revision4" / "run_manifest.json",
    "revision5_run_manifest": ROOT / "artifacts" / "revision5" / "run_manifest.json",
    "revision6_run_manifest": ROOT / "artifacts" / "revision6" / "run_manifest.json",
    "av_content_run_manifest": ROOT
    / "artifacts"
    / "av_content_prior_clip_siglip"
    / "run_manifest.json",
    "av_physio_run_manifest": ROOT
    / "artifacts"
    / "av_causal_physio_clip_siglip"
    / "run_manifest.json",
    "av_physio_tokens_run_manifest": ROOT
    / "artifacts"
    / "av_causal_physio_clip_siglip_tokens"
    / "run_manifest.json",
    "sparse_anchor_run_manifest": ROOT
    / "artifacts"
    / "sparse_anchor_kernel_optimization"
    / "run_manifest.json",
    "content_router_run_manifest": ROOT
    / "artifacts"
    / "content_backbone_router"
    / "run_manifest.json",
    "clip_feature_manifest": ROOT
    / "data"
    / "stimuli"
    / "features"
    / "foundation_av_manifest.json",
    "siglip_feature_manifest": ROOT
    / "data"
    / "stimuli"
    / "features"
    / "foundation_siglip_ast_manifest.json",
    "dinov2_feature_manifest": ROOT
    / "data"
    / "stimuli"
    / "features"
    / "foundation_dinov2_ast_manifest.json",
    "fusion_feature_manifest": ROOT
    / "data"
    / "stimuli"
    / "features"
    / "foundation_clip_siglip_ast_manifest.json",
}

CSV_INPUTS = {
    "estimand_robustness": ROOT
    / "artifacts"
    / "revision4"
    / "estimand_robustness.csv",
    "simulation_recovery": ROOT
    / "artifacts"
    / "measurement_robustness"
    / "simulation_recovery.csv",
    "sensing_models": ROOT / "artifacts" / "revision6" / "sensing_model_table.csv",
    "sensing_results": ROOT
    / "artifacts"
    / "revision6"
    / "primary_antialias_sensing_results.csv",
    "matched_identity_transfer": ROOT
    / "artifacts"
    / "revision6"
    / "matched_identity_transfer.csv",
    "category_heterogeneity": ROOT
    / "artifacts"
    / "revision6"
    / "category_heterogeneity.csv",
    "decision_threshold_curves": ROOT
    / "artifacts"
    / "revision6"
    / "decision_threshold_curves.csv",
    "repeated_grouped_cv": ROOT
    / "artifacts"
    / "revision6"
    / "repeated_grouped_cv.csv",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
PRIVATE_PATH_MARKERS = ("/home/", "/Users/", "file" + "://")


def publication_portable(value: object) -> object:
    """Return a publication-facing copy with workspace paths made relative.

    Upstream manifests are immutable analysis records and may preserve the
    machine path used for a run. The sole publication source instead exposes
    paths relative to the project root so that it is portable and anonymous.
    Dictionary keys are normalized as well because hash registries commonly
    use paths as keys.
    """

    if isinstance(value, dict):
        return {
            str(publication_portable(key)): publication_portable(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [publication_portable(item) for item in value]
    if isinstance(value, tuple):
        return [publication_portable(item) for item in value]
    if not isinstance(value, str):
        return value

    root = ROOT.resolve().as_posix()
    if value == root:
        return "."
    return value.replace(root + "/", "")


def iter_strings(value: object, location: str = "root"):
    if isinstance(value, dict):
        for key, item in value.items():
            yield f"{location}.<key>", str(key)
            yield from iter_strings(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from iter_strings(item, f"{location}[{index}]")
    elif isinstance(value, str):
        yield location, value


def assert_no_private_paths(value: object) -> None:
    leaks: list[str] = []
    for location, value in iter_strings(value):
        if any(marker in value for marker in PRIVATE_PATH_MARKERS):
            leaks.append(f"{location}: {value}")
        elif WINDOWS_ABSOLUTE_PATH.match(value):
            leaks.append(f"{location}: {value}")
    if leaks:
        preview = "; ".join(leaks[:5])
        raise RuntimeError(f"Publication source contains private paths: {preview}")


def build_statistical_registry(summaries: dict[str, object]) -> dict[str, object]:
    """Return the complete publication-facing analysis-family registry.

    Counts and seeds are read from committed manifests. Textual fields define
    reporting contracts and are intentionally explicit about exploratory or
    descriptive families; they are not retroactive preregistration.
    """

    manifests = {
        revision: summaries[f"{revision}_run_manifest"]
        for revision in ("revision2", "revision4", "revision5", "revision6")
    }
    for revision, manifest in manifests.items():
        if not isinstance(manifest, dict) or manifest.get("status") != "complete":
            raise RuntimeError(f"Incomplete run manifest for {revision}")

    algorithm_manifests = {
        name: summaries[name]
        for name in (
            "av_content_run_manifest",
            "av_physio_run_manifest",
            "av_physio_tokens_run_manifest",
            "sparse_anchor_run_manifest",
            "content_router_run_manifest",
        )
    }
    for name, manifest in algorithm_manifests.items():
        if not isinstance(manifest, dict) or manifest.get("status") != "complete":
            raise RuntimeError(f"Incomplete algorithm run manifest for {name}")

    def config(revision: str, key: str) -> object:
        return manifests[revision]["configuration"][key]

    def algorithm_config(manifest: str, key: str) -> object:
        return algorithm_manifests[manifest]["configuration"][key]

    software = {
        "python": manifests["revision6"].get("python", "not recorded"),
        "numpy": manifests["revision6"].get("numpy", "not recorded"),
        "note": "Revision 2 did not record package versions; later recomputations used the listed environment.",
    }
    rows = [
        {
            "id": "formula_target_audit",
            "endpoint": "Saved versus separately implemented b and q targets; structural q-at-least-absolute-b invariant",
            "population_unit": "360 participant-video trials (structural identity check)",
            "estimator_or_test": "Exact recomputation and equality/invariant checks",
            "uncertainty": "None; this is a deterministic artifact check",
            "multiplicity": "Not applicable",
            "resampling_or_folds": "None",
            "seed": str(config("revision5", "seed")),
            "status": "STRUCTURAL CHECK",
        },
        {
            "id": "same_trial_cross_axis",
            "endpoint": "Other-axis alignment gain and own-versus-donor warp advantage",
            "population_unit": "Participant; videos crossed as released support points",
            "estimator_or_test": "Mean gain, participant x video crossed bootstrap, participant sign-flip diagnostic, donor derangement",
            "uncertainty": f"Percentile crossed bootstrap; {config('revision2', 'permutation_repeats'):,} sign-flip/derangement draws",
            "multiplicity": "Axis-specific and combined contrasts are exploratory; no familywise claim",
            "resampling_or_folds": f"{config('revision2', 'bootstrap_repeats'):,} bootstrap draws; outer participant folds retained",
            "seed": str(config("revision2", "seed")),
            "status": "EXPLORATORY INTERFACE-DEPENDENCE DIAGNOSTIC",
        },
        {
            "id": "cross_video_transfer",
            "endpoint": "Held-out signed residual or absolute-magnitude prediction gain",
            "population_unit": "Participant, with held-out videos inside participant-safe folds",
            "estimator_or_test": "Nested shrinkage calibration over 1/2/4/8-video budgets",
            "uncertainty": f"Participant percentile bootstrap with {config('revision5', 'bootstrap_repeats'):,} draws",
            "multiplicity": "Four budgets are reported as an exploratory response curve; no confirmatory threshold claim",
            "resampling_or_folds": f"{config('revision2', 'calibration_repeats')} calibration subsets per budget",
            "seed": str(config("revision5", "seed")),
            "status": "EXPLORATORY TRANSFER ANALYSIS",
        },
        {
            "id": "category_heterogeneity",
            "endpoint": "Sum of squared deviations among five category mean gains",
            "population_unit": "Participant blocks; released video categories",
            "estimator_or_test": "Within-participant category-label permutation",
            "uncertainty": f"Monte Carlo p value from {config('revision6', 'permutation_repeats'):,} permutations",
            "multiplicity": "Budget-specific p values Holm-adjusted across four budgets; combined all-budget diagnostic reported separately",
            "resampling_or_folds": "Participant-blocked label permutations",
            "seed": str(config("revision6", "seed")),
            "status": "EXPLORATORY HETEROGENEITY TEST",
        },
        {
            "id": "estimator_reference_sensitivity",
            "endpoint": "Profile rank, q shift, path geometry, and shared-reference descriptive variance ratios",
            "population_unit": "Participants and released videos under the stated empirical support",
            "estimator_or_test": "Five estimands, four path objectives, parameter and reference-panel sensitivities",
            "uncertainty": f"Original-identity-excluding full-pipeline bootstrap ({config('revision5', 'full_pipeline_repeats'):,} draws); geometry bootstrap ({config('revision4', 'geometry_bootstrap_repeats'):,})",
            "multiplicity": "Sensitivity inventory; no reliability cutoff or confirmatory family",
            "resampling_or_folds": "Participant-cluster resampling with within-category video sampling and rebuilt references",
            "seed": str(config("revision5", "seed")),
            "status": "DESCRIPTIVE SENSITIVITY",
        },
        {
            "id": "optional_sensing_increment",
            "endpoint": "Participant-macro MAE gain versus video-mean and context-only comparators",
            "population_unit": "Outer-held-out participant",
            "estimator_or_test": "Nested grouped CV with fold-safe targets, features, scaling, selection, and disabled-residual option",
            "uncertainty": f"Participant percentile bootstrap ({config('revision6', 'bootstrap_repeats'):,} draws) plus complete split sensitivity",
            "multiplicity": "Model inventory is descriptive; the blockwise residual stack is the declared primary sensor route; no equivalence margin",
            "resampling_or_folds": f"5 outer x 4 inner folds; {config('revision6', 'grouped_cv_repeats')} complete participant partitions",
            "seed": str(config("revision6", "seed")),
            "status": "PRIMARY ROUTE COMPARISON, NOT PREREGISTERED",
        },
        {
            "id": "signed_calibration",
            "endpoint": "Reference-trace MAE gain versus video-only correction",
            "population_unit": "Participant",
            "estimator_or_test": "Outer-training shrinkage with 1/2/4/8 calibration-video budgets",
            "uncertainty": f"Participant percentile bootstrap ({config('revision5', 'bootstrap_repeats'):,} draws)",
            "multiplicity": "Four budgets form an exploratory burden-response curve; no smallest meaningful gain or utility test",
            "resampling_or_folds": f"{config('revision2', 'calibration_repeats')} calibration subsets per budget; participant-safe folds",
            "seed": str(config("revision5", "seed")),
            "status": "PROXIMAL, NOT USER-UTILITY EVIDENCE",
        },
        {
            "id": "decision_threshold_curves",
            "endpoint": "Counts of participants exceeding hypothetical benefit or degradation thresholds",
            "population_unit": "24 participant-level estimated gains",
            "estimator_or_test": "Exact integer count over a fixed 41-point grid",
            "uncertainty": "No interval; conditional Wilson bands deliberately removed",
            "multiplicity": "Post hoc descriptive sensitivity only; thresholds do not define a decision rule",
            "resampling_or_folds": "Uses participant estimates from the registered sensing/calibration pipelines",
            "seed": str(config("revision6", "seed")),
            "status": "DESCRIPTIVE DECISION CURVE",
        },
        {
            "id": "content_conditioned_trajectory",
            "endpoint": "Trial-macro valence/arousal MAE on the 1--255 scale and gain versus the metadata temporal prior",
            "population_unit": "Participant, with videos treated as fixed released support points and held out jointly in outer cells",
            "estimator_or_test": "Frozen visual representation, same-category trajectory retrieval, monotone content alignment, and inner-selected exact fallback",
            "uncertainty": "Five complete grouped-CV participant allocations plus a fixed -3 to +3 s alignment sensitivity range",
            "multiplicity": "Seven reported backbone/fusion candidates are a development inventory; no confirmatory winner claim",
            "resampling_or_folds": (
                f"{algorithm_config('av_content_run_manifest', 'subject_folds')} participant x "
                f"{algorithm_config('av_content_run_manifest', 'video_folds')} video outer cells; "
                f"{algorithm_config('av_content_run_manifest', 'inner_content_subject_folds')} participant x "
                f"{algorithm_config('av_content_run_manifest', 'inner_content_video_folds')} video inner folds; "
                f"{algorithm_config('av_content_run_manifest', 'repeat_count')} participant-fold allocations"
            ),
            "seed": str(algorithm_config("av_content_run_manifest", "seed")),
            "status": "ITERATIVE DEVELOPMENT-SET NESTED OOF ESTIMATE",
        },
        {
            "id": "causal_physiology_residual",
            "endpoint": "Trial-macro MAE gain over the frozen CLIP+SigLIP content prediction",
            "population_unit": "Participant, with released videos held out jointly inside the fixed 15-video support",
            "estimator_or_test": "Causal current/past EEG-fNIRS ridge residual with inner gain gate and exact zero-residual fallback",
            "uncertainty": "Frozen five-point inner-gain decision curve; no equivalence interval or user-utility claim",
            "multiplicity": "Full-feature and temporal-token routes are reported as gated development comparisons",
            "resampling_or_folds": (
                "Same 5 participant x 5 video outer cells as the content prior; "
                f"{algorithm_config('av_physio_run_manifest', 'inner_subject_folds')} participant x "
                f"{algorithm_config('av_physio_run_manifest', 'inner_video_folds')} video inner folds"
            ),
            "seed": str(algorithm_config("av_physio_run_manifest", "seed")),
            "status": "PRIMARY GATE FAILED; EXACT CONTENT FALLBACK",
        },
        {
            "id": "post_trial_sam_sparse_recovery",
            "endpoint": "Trial-macro MAE and gain over the canonical known-video population prior after one post-trial SAM pair",
            "population_unit": "Participant; videos crossed as the fixed released support in uncertainty propagation",
            "estimator_or_test": "Gaussian-weighted same-video trajectory retrieval plus bounded center/shape correction and functional residual ensemble",
            "uncertainty": (
                f"Participant-video crossed percentile bootstrap "
                f"({algorithm_config('sparse_anchor_run_manifest', 'bootstrap_repeats'):,} draws) "
                "and complete participant-fold allocation sensitivity"
            ),
            "multiplicity": "Candidate inventory and ensemble were developed on this dataset; independent confirmation is required",
            "resampling_or_folds": (
                f"5 outer participant folds x "
                f"{algorithm_config('sparse_anchor_run_manifest', 'inner_folds')} inner folds; "
                f"{algorithm_config('sparse_anchor_run_manifest', 'cv_repeats')} complete allocations"
            ),
            "seed": str(algorithm_config("sparse_anchor_run_manifest", "seed")),
            "status": "POST-TRIAL SPARSE-FEEDBACK ROUTE; NOT ZERO INTERACTION",
        },
    ]
    return {
        "schema_version": "statistical-analysis-registry-v1",
        "scope": "Complete registry for quantitative claims retained in the main manuscript and Supplementary Information.",
        "independent_unit_policy": "Participants are the population unit. Videos are fixed released support points or category-stratified empirical support as stated; no unseen-video population claim is made.",
        "software": software,
        "rows": rows,
    }


def build_source() -> dict[str, object]:
    paths = [*JSON_INPUTS.values(), *CSV_INPUTS.values()]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing formal Revision-6 source artifacts: "
            + ", ".join(portable(path) for path in missing)
        )

    summaries = {name: read_json(path) for name, path in JSON_INPUTS.items()}
    # The only trial-level release exception is the three authorized anonymous
    # Figure 5 examples. Full OOF archives and original identifiers stay private.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dense_example_release", ROOT / "scripts/prepare_dense_trajectory_examples.py"
    )
    release_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release_module)
    release_module.validate_release(summaries["dense_trajectory_examples"])
    revision6 = summaries["revision6"]
    if not isinstance(revision6, dict):
        raise TypeError("Revision-6 summary must be a JSON object")
    if revision6.get("schema_version") != "revision6-analysis-v2":
        raise RuntimeError("Revision-6 analysis summary is stale")
    if revision6.get("revision") != 6:
        raise RuntimeError("Expected Revision 6")

    algorithms = summaries["algorithm_experiments"]
    if not isinstance(algorithms, dict):
        raise TypeError("Algorithm experiment summary must be a JSON object")
    if algorithms.get("schema_version") != "merps-algorithm-experiment-summary-v2":
        raise RuntimeError("Algorithm experiment summary is stale")
    integrity = algorithms.get("stimulus_integrity", {})
    if (
        integrity.get("files_expected"),
        integrity.get("files_present"),
        integrity.get("identities_verified"),
    ) != (15, 15, 15):
        raise RuntimeError("Formal stimulus integrity audit is incomplete")
    oof = algorithms.get("oof_artifact_consistency", {})
    if not all(
        bool(oof.get(key))
        for key in (
            "sample_identity_consistent",
            "targets_identical",
            "all_numeric_arrays_finite",
        )
    ):
        raise RuntimeError("Algorithm OOF artifact consistency audit failed")

    for name in (
        "av_content_run_manifest",
        "av_physio_run_manifest",
        "av_physio_tokens_run_manifest",
        "sparse_anchor_run_manifest",
        "content_router_run_manifest",
        "clip_feature_manifest",
        "siglip_feature_manifest",
        "dinov2_feature_manifest",
        "fusion_feature_manifest",
    ):
        manifest = summaries[name]
        if not isinstance(manifest, dict) or manifest.get("status") != "complete":
            raise RuntimeError(f"Incomplete publication algorithm manifest: {name}")

    tables = {name: read_csv(path) for name, path in CSV_INPUTS.items()}
    tables["decision_threshold_curves"] = [
        {
            key: value
            for key, value in row.items()
            if key not in {"wilson_ci95_low", "wilson_ci95_high"}
        }
        for row in tables["decision_threshold_curves"]
    ]
    if not tables["decision_threshold_curves"]:
        raise RuntimeError("Decision-threshold curves are empty")
    if not tables["repeated_grouped_cv"]:
        raise RuntimeError("Repeated grouped-CV sensitivity is empty")

    registry = [
        {
            "id": name,
            "uri": portable(path),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(
            {**JSON_INPUTS, **CSV_INPUTS}.items(), key=lambda item: item[0]
        )
    ]
    statistical_registry = build_statistical_registry(summaries)
    source = {
        "schema_version": "revision6-publication-source-v1",
        "revision": 6,
        "policy": {
            "single_numeric_source": "paper_support/revision6_source.json",
            "anonymous_curve_exception": (
                "Three author-authorized Figure 5 examples and their plotted "
                "coordinates only; no original identities or complete OOF archives."
            ),
            "downstream_consumers": [
                "paper/body.tex generated tables and macros",
                "paper/body_zh.tex generated tables and macros",
                "paper/supplementary_information.tex generated table and macros",
                "README.md generated Revision-6 block",
                "paper/README.md generated Revision-6 block",
                "main-manuscript and supplementary figures with their source-data CSVs",
                "external-transfer and rule-ablation tables",
                "adjacent-framework comparison and statistical-analysis registry",
                "content, causal-residual, and sparse-feedback algorithm tables",
            ],
            "rule": (
                "Downstream publication files may not read artifacts/revision2 "
                "through artifacts/revision6 directly."
            ),
            "path_policy": (
                "Embedded workspace paths are project-relative; upstream run "
                "records remain unchanged and are referenced by content hash."
            ),
        },
        "source_artifacts": registry,
        "summaries": summaries,
        "tables": tables,
        "statistical_analysis_registry": statistical_registry,
    }
    source = publication_portable(source)
    assert_no_private_paths(source)
    return source


def serialized(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    expected = serialized(build_source())
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != expected:
            raise SystemExit(
                "Revision-6 publication source is stale; run "
                ".venv/bin/python scripts/build_revision6_source.py"
            )
        print(f"Revision-6 publication source is current: {portable(output)}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8")
    print(f"Wrote {portable(output)}")


if __name__ == "__main__":
    main()
