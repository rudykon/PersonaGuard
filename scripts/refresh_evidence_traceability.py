#!/usr/bin/env python3
"""Refresh traceability hashes and register aggregate Revision-6 additions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "paper_support" / "evidence_traceability.json"

ADDITIONAL_ARTIFACTS = {
    "revision6_decision_threshold_curves": "artifacts/revision6/decision_threshold_curves.csv",
    "revision6_repeated_grouped_cv": "artifacts/revision6/repeated_grouped_cv.csv",
}

NODE_LINKS = {
    "eeg_fnirs_increment": {
        "revision6_decision_threshold_curves",
        "revision6_repeated_grouped_cv",
    },
    "signed_proximal_correction": {"revision6_decision_threshold_curves"},
    "downstream_user_utility": {"revision6_decision_threshold_curves"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refreshed() -> dict[str, object]:
    graph = json.loads(SCHEMA.read_text(encoding="utf-8"))
    registry = graph["artifact_registry"]
    by_id = {str(item["id"]): item for item in registry}
    for artifact_id, uri in ADDITIONAL_ARTIFACTS.items():
        if artifact_id not in by_id:
            item = {"id": artifact_id, "uri": uri, "sha256": "0" * 64}
            registry.append(item)
            by_id[artifact_id] = item
    for item in registry:
        path = ROOT / str(item["uri"])
        if not path.is_file():
            raise FileNotFoundError(f"Registered artifact is missing: {item['uri']}")
        item["sha256"] = sha256(path)

    node_by_id = {str(node["id"]): node for node in graph["nodes"]}
    for node_id, additions in NODE_LINKS.items():
        node = node_by_id[node_id]
        current = [str(value) for value in node["evidence_artifact_ids"]]
        node["evidence_artifact_ids"] = current + sorted(additions - set(current))
    return graph


def main() -> None:
    graph = refreshed()
    SCHEMA.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Refreshed {len(graph['artifact_registry'])} evidence artifacts in "
        "paper_support/evidence_traceability.json"
    )


if __name__ == "__main__":
    main()
