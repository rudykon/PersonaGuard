#!/usr/bin/env python3
"""Release three author-authorized anonymous examples from frozen OOF archives.

This is figure preparation, not model fitting. Full archives and the mapping
back to participant/video identities remain restricted and are never packaged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper_support/dense_trajectory_examples.json"
CURVES = ("target", "metadata", "content", "residual", "sam")
ARCHIVES = {
    "content": "artifacts/av_content_prior_clip_siglip/oof_predictions.npz",
    "residual": "artifacts/av_causal_physio_clip_siglip/oof_predictions.npz",
    "sam": "artifacts/sparse_anchor_kernel_optimization/oof_predictions.npz",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialized(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def validate_release(payload):
    """Closed schema: reject identifiers and unapproved extra curves/fields."""
    if set(payload) != {"schema_version", "authorization", "selection", "verification",
                        "archive_sha256", "conditions", "examples"}:
        raise ValueError("Unexpected release fields")
    if payload["schema_version"] != "anonymous-dense-examples-v1":
        raise ValueError("Unknown anonymous-example schema")
    if payload["authorization"] != {
        "basis": "author confirmation", "date": "2026-09-11",
        "scope": "three anonymous trial-curve illustrations and their plotted coordinates",
        "excludes": "complete archives, original identifiers, and participant-indexed mappings",
    }:
        raise ValueError("Anonymous-curve authorization is missing or changed")
    if set(payload["archive_sha256"]) != set(ARCHIVES):
        raise ValueError("Archive provenance incomplete")
    if set(payload["conditions"]) != set(CURVES):
        raise ValueError("Exactly five authorized curves are required")
    if set(payload["selection"]) != {
        "method", "rank_formula", "tie_break", "candidate_trials", "selected_trials",
        "excluded_trials", "selected_for_display_only", "criterion", "not_used_for_selection"
    }:
        raise ValueError("Unexpected selection metadata")
    if set(payload["verification"]) != {
        "sample_count", "sample_order_identical", "targets_identical",
        "residual_content_comparator_exact", "residual_primary_gate_exact",
        "full_trial_grid", "all_trial_macro_mae"
    }:
        raise ValueError("Unexpected verification metadata")
    if set(payload["verification"]["all_trial_macro_mae"]) != set(CURVES) - {"target"}:
        raise ValueError("Unexpected metric metadata")
    for sha in payload["archive_sha256"].values():
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Invalid archive hash")
    examples = payload["examples"]
    if len(examples) != 3 or [e["label"] for e in examples] != ["A", "B", "C"]:
        raise ValueError("Exactly three anonymous examples are authorized")
    for example in examples:
        if set(example) != {"label", "percentile", "rank", "time_seconds", "curves"}:
            raise ValueError("Identifier or unexpected field in example")
        times = example["time_seconds"]
        if not times or times != list(range(len(times))):
            raise ValueError("Expected complete one-second trial grid starting at zero")
        if set(example["curves"]) != set(CURVES):
            raise ValueError("Missing or extra curve")
        for values in example["curves"].values():
            if len(values) != len(times):
                raise ValueError("Incomplete trial curve")
            for pair in values:
                if len(pair) != 2 or not all(math.isfinite(v) and 1 <= v <= 255 for v in pair):
                    raise ValueError("Curves must be finite valence/arousal on the 1–255 scale")
    return payload


def prepare():
    import numpy as np

    source = json.loads((ROOT / "paper_support/revision6_source.json").read_text())
    registered = source["summaries"]["algorithm_experiments"]["oof_artifact_consistency"][
        "artifact_identities"]
    archive_hashes = {}
    arrays = {}
    for name, relative in ARCHIVES.items():
        path = ROOT / relative
        sha = digest(path)
        if sha != registered[str(Path(relative).parent)]["sha256"]:
            raise ValueError(f"Frozen archive hash mismatch: {name}")
        archive_hashes[name] = sha
        with np.load(path, allow_pickle=False) as data:
            arrays[name] = {key: data[key] for key in data.files}
    content, residual, sam = (arrays[name] for name in ARCHIVES)
    for other, target_key in ((residual, "targets"), (sam, "target")):
        if not np.array_equal(content["sample_ids"], other["sample_ids"]):
            raise ValueError("OOF sample identities/order disagree")
        if not np.array_equal(content["targets"], other[target_key]):
            raise ValueError("OOF targets disagree")
    if not np.array_equal(content["primary_content"], residual["content_prior"]):
        raise ValueError("Residual comparator differs from the frozen content prediction")
    if not np.array_equal(residual["primary_prediction"], residual["prediction_gain_0.1"]):
        raise ValueError("Residual candidate does not match the primary frozen gate")
    curves = {
        "target": content["targets"], "metadata": content["primary_metadata"],
        "content": content["primary_content"], "residual": residual["primary_prediction"],
        "sam": sam["ensemble_sparse_anchor"],
    }
    for values in curves.values():
        if values.shape != content["targets"].shape or not np.isfinite(values).all():
            raise ValueError("Incomplete or nonfinite predictions")
        if values.min() < 1 or values.max() > 255:
            raise ValueError("Unexpected scaling; do not clip figure data")

    groups = np.unique(np.column_stack((content["subject_numbers"], content["videos"])), axis=0)
    trial_rows = []
    for participant, video in groups:
        indices = np.flatnonzero((content["subject_numbers"] == participant) &
                                 (content["videos"] == video))
        indices = indices[np.argsort(content["timestamps"][indices], kind="stable")]
        if not np.array_equal(content["timestamps"][indices], np.arange(len(indices))):
            raise ValueError("Trial timestamps are not a complete one-second grid")
        if len(np.unique(content["primary_subject_fold_ids"][indices])) != 1:
            raise ValueError("A trial crosses participant folds")
        if len(np.unique(content["primary_video_fold_ids"][indices])) != 1:
            raise ValueError("A trial crosses video folds")
        errors = {name: float(np.abs(values[indices].astype(float) -
                                    curves["target"][indices].astype(float)).mean())
                  for name, values in curves.items() if name != "target"}
        trial_rows.append({"participant": int(participant), "video": int(video),
                           "indices": indices, "errors": errors})
    # Prespecified for this illustration before looking at individual curves.
    # Ties use the archive's participant/video ordering, never a desired gain.
    ordered = sorted(trial_rows, key=lambda r: (r["errors"]["content"], r["participant"], r["video"]))
    examples, private = [], []
    for label, percentile in zip("ABC", (25, 50, 75)):
        index = int(math.floor(percentile / 100 * (len(ordered) - 1) + .5))
        row = ordered[index]
        selected = row["indices"]
        examples.append({
            "label": label, "percentile": percentile, "rank": index + 1,
            "time_seconds": content["timestamps"][selected].astype(int).tolist(),
            "curves": {name: values[selected].astype(float).tolist()
                       for name, values in curves.items()},
        })
        private.append({"label": label, "participant": row["participant"], "video": row["video"],
                        "rank": index + 1, "rows": selected.tolist(), "trial_mae": row["errors"]})
    payload = {
        "schema_version": "anonymous-dense-examples-v1",
        "authorization": {
            "basis": "author confirmation", "date": "2026-09-11",
            "scope": "three anonymous trial-curve illustrations and their plotted coordinates",
            "excludes": "complete archives, original identifiers, and participant-indexed mappings",
        },
        "selection": {
            "method": "nearest ranked trial to content-prior MAE percentiles 25, 50, 75",
            "rank_formula": "floor(percentile/100 * (N-1) + 0.5) + 1",
            "tie_break": "stable original participant/video ordering (mapping retained privately)",
            "candidate_trials": len(ordered), "selected_trials": 3,
            "excluded_trials": 0, "selected_for_display_only": True,
            "criterion": "mean absolute error across all trial seconds and both axes",
            "not_used_for_selection": ["metadata gain", "residual gain", "SAM gain"],
        },
        "verification": {
            "sample_count": len(content["sample_ids"]), "sample_order_identical": True,
            "targets_identical": True, "residual_content_comparator_exact": True,
            "residual_primary_gate_exact": True, "full_trial_grid": True,
            "all_trial_macro_mae": {
                name: float(np.mean([r["errors"][name] for r in ordered]))
                for name in curves if name != "target"},
        },
        "archive_sha256": archive_hashes,
        "conditions": {
            "target": "Observed joystick report; valence then arousal, 1–255",
            "metadata": "Metadata only; participant × released-video holdout",
            "content": "Full video content; participant × released-video holdout",
            "residual": "Content + current/past EEG/fNIRS; primary full-modality candidate, not disabled fallback",
            "sam": "Post-trial SAM + same-video training library; participant holdout, known video",
        },
        "examples": examples,
    }
    return validate_release(payload), private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify against frozen restricted OOF files")
    args = parser.parse_args()
    payload, private = prepare()
    expected = serialized(payload)
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != expected:
            raise SystemExit("Anonymous example release is stale")
    else:
        OUTPUT.write_text(expected)
        private_path = ROOT / "artifacts/dense_trajectory_examples/private_selection.json"
        private_path.parent.mkdir(parents=True, exist_ok=True)
        private_path.write_text(serialized(private))
    print(json.dumps({"trials_checked": payload["selection"]["candidate_trials"],
                      "trial_macro_mae": payload["verification"]["all_trial_macro_mae"],
                      "selected_ranks": [e["rank"] for e in payload["examples"]],
                      "samples_per_example": [len(e["time_seconds"]) for e in payload["examples"]],
                      "status": "verified" if args.check else "prepared"}))


if __name__ == "__main__":
    main()
