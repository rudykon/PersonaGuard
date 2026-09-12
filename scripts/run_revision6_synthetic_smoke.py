#!/usr/bin/env python3
"""Exercise Revision-6 publication structure using only synthetic aggregates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_node(node_id: str, status: str) -> dict[str, object]:
    table = {
        "node": node_id.replace("_", " ").title(),
        "evidence": "Synthetic aggregate evidence",
        "boundary": "Synthetic structural smoke only",
        "permitted_claim": "Pipeline structure executes without gated data",
        "decision": "Do not make an empirical manuscript claim",
    }
    return {
        "id": node_id,
        "status": status,
        "status_rationale": "Synthetic status used only to exercise schema validation.",
        "evidence_artifact_ids": ["synthetic_summary"],
        "decision_owner_role": "synthetic-test owner",
        "decision_date": "2026-08-09",
        "review_due_date": None,
        "review_trigger": "Replace whenever the publication schema changes.",
        "estimand": "synthetic structural estimand",
        "permits": [],
        "include_in_table": True,
        "table_en": table,
        "table_zh": {
            **table,
            "node": "合成结构节点",
            "evidence": "合成聚合证据",
            "boundary": "仅用于结构烟测",
            "permitted_claim": "无 gated 数据时结构可运行",
            "decision": "不得生成经验主张",
        },
    }


def run_smoke() -> dict[str, object]:
    analysis = load_module(
        "revision6_smoke_analysis", ROOT / "scripts" / "run_revision6_analyses.py"
    )
    graph_module = load_module(
        "revision6_smoke_graph",
        ROOT / "scripts" / "generate_evidence_traceability.py",
    )

    gains = np.linspace(-0.12, 0.12, 24, dtype=np.float64)
    threshold_rows = analysis.threshold_decision_rows(
        gains,
        analysis="synthetic_increment",
        route="synthetic_route",
        unit="synthetic_units",
        thresholds=(0.0, 0.05, 0.10),
    )
    if len(threshold_rows) != 6:
        raise RuntimeError("Synthetic threshold table has the wrong shape")
    forbidden = {"subject", "participant_id", "participant"}
    if forbidden & {key for row in threshold_rows for key in row}:
        raise RuntimeError("Synthetic aggregate export leaked an identity field")

    primary = analysis.participant_fold_plan(0, seed=2027)
    repeated = analysis.participant_fold_plan(1, seed=2027)
    for plan in (primary, repeated):
        held_out = [
            int(value)
            for _, validation in plan
            for value in np.asarray(validation)
        ]
        if sorted(held_out) != list(range(1, 25)):
            raise RuntimeError("Synthetic grouped fold plan is incomplete")
    primary_digest = analysis.fold_plan_digest(primary)
    repeated_digest = analysis.fold_plan_digest(repeated)
    if primary_digest == repeated_digest:
        raise RuntimeError("Synthetic repeated fold allocation did not change")

    original_root = graph_module.PROJECT_ROOT
    try:
        with tempfile.TemporaryDirectory(prefix="revision6-synthetic-smoke-") as tmp:
            temp_root = Path(tmp)
            artifact = temp_root / "aggregate" / "synthetic_summary.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text(
                json.dumps({"synthetic": True, "participants": 24}) + "\n",
                encoding="utf-8",
            )
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            graph = {
                "schema_version": graph_module.EXPECTED_SCHEMA_VERSION,
                "graph_type": "evidence_traceability_graph",
                "graph_instance_version": "synthetic-smoke-v1",
                "decision_snapshot_date": "2026-08-09",
                "machine_check_scope": {
                    "checked": [
                        "synthetic schema and required fields",
                        "synthetic artifact hash and typed edge",
                        "synthetic propagation-rule presence",
                    ],
                    "not_checked": [
                        "substantive validity or adequacy of synthetic evidence",
                        "analyst agreement, user benefit, fairness, or domain transfer",
                    ],
                },
                "artifact_registry": [
                    {
                        "id": "synthetic_summary",
                        "uri": "aggregate/synthetic_summary.json",
                        "sha256": digest,
                    }
                ],
                "status_propagation_rules": {
                    rule: "Synthetic rule exercised by the smoke test."
                    for rule in graph_module.REQUIRED_RULES
                },
                "nodes": [
                    synthetic_node("synthetic_evidence", "PARTIAL"),
                    synthetic_node("synthetic_deployment", "NOT_JUSTIFIED"),
                ],
                "edges": [
                    {
                        "source": "synthetic_evidence",
                        "target": "synthetic_deployment",
                        "type": "blocks_deployment",
                    }
                ],
            }
            graph_module.PROJECT_ROOT = temp_root
            graph_module.validate_graph(graph)
            rendered_rows = graph_module.table_rows(graph, "en")
            if "Synthetic Evidence" not in rendered_rows:
                raise RuntimeError("Synthetic evidence rows were not rendered")
    finally:
        graph_module.PROJECT_ROOT = original_root

    return {
        "status": "passed",
        "uses_gated_data": False,
        "participants": 24,
        "threshold_rows": len(threshold_rows),
        "fold_partitions": 2,
        "fold_digests_distinct": True,
        "machine_scope_structural_only": True,
        "synthetic_graph_nodes": 2,
        "synthetic_graph_edges": 1,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
