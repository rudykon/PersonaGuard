#!/usr/bin/env python3
"""Validate the evidence traceability graph and generate bilingual table rows."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = PROJECT_ROOT / "paper_support" / "evidence_traceability.json"
DEFAULT_EN = PROJECT_ROOT / "paper" / "generated" / "evidence_traceability_rows.tex"
DEFAULT_ZH = PROJECT_ROOT / "paper" / "generated" / "evidence_traceability_rows_zh.tex"

ALLOWED_STATUSES = {
    "AVAILABLE",
    "PARTIAL",
    "NOT_TESTED",
    "NO_DEMONSTRATED_INCREMENT",
    "NOT_JUSTIFIED",
}
ALLOWED_EDGE_TYPES = {
    "requires",
    "supports",
    "does_not_license",
    "blocks_deployment",
}
REQUIRED_RULES = {
    "no_automatic_status_propagation",
    "partial_does_not_license_retention_or_utility",
    "route_failure_is_local",
    "untested_utility_blocks_consequential_deployment",
    "unassessed_reference_equity_blocks_retention",
}
EXPECTED_SCHEMA_VERSION = "1.2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_NODE_LIFECYCLE_FIELDS = {
    "status_rationale",
    "evidence_artifact_ids",
    "decision_owner_role",
    "decision_date",
    "review_due_date",
    "review_trigger",
}
REQUIRED_MACHINE_SCOPE = {"checked", "not_checked"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-en", type=Path, default=DEFAULT_EN)
    parser.add_argument("--output-zh", type=Path, default=DEFAULT_ZH)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def validate_iso_date(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO date")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must use YYYY-MM-DD") from exc


def validate_artifact_registry(graph: dict[str, object]) -> set[str]:
    registry = graph.get("artifact_registry")
    if not isinstance(registry, list) or not registry:
        raise ValueError("artifact_registry must be a non-empty list")
    artifact_ids: set[str] = set()
    for artifact in registry:
        if not isinstance(artifact, dict):
            raise ValueError("Each artifact registry entry must be an object")
        artifact_id = str(artifact.get("id", ""))
        uri = str(artifact.get("uri", ""))
        expected_sha256 = str(artifact.get("sha256", ""))
        if not artifact_id or artifact_id in artifact_ids:
            raise ValueError(f"Artifact identifier is missing or duplicated: {artifact_id!r}")
        artifact_ids.add(artifact_id)
        uri_path = Path(uri)
        if not uri or uri_path.is_absolute() or ".." in uri_path.parts:
            raise ValueError(f"Artifact URI must be project-relative: {uri!r}")
        if not SHA256_RE.fullmatch(expected_sha256):
            raise ValueError(f"Artifact {artifact_id} lacks a lowercase SHA-256 digest")
        artifact_path = PROJECT_ROOT / uri_path
        if not artifact_path.is_file():
            raise ValueError(f"Registered artifact is missing: {uri}")
        actual_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {artifact_id}: "
                f"expected {expected_sha256}, observed {actual_sha256}"
            )
    return artifact_ids


def validate_graph(graph: dict[str, object]) -> None:
    if graph.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {EXPECTED_SCHEMA_VERSION}")
    if graph.get("graph_type") != "evidence_traceability_graph":
        raise ValueError("graph_type must be evidence_traceability_graph")
    if not isinstance(graph.get("graph_instance_version"), str):
        raise ValueError("graph_instance_version must be present")
    validate_iso_date(graph.get("decision_snapshot_date"), "decision_snapshot_date")
    machine_scope = graph.get("machine_check_scope")
    if not isinstance(machine_scope, dict) or set(machine_scope) != REQUIRED_MACHINE_SCOPE:
        raise ValueError("machine_check_scope must declare checked and not_checked")
    for field in sorted(REQUIRED_MACHINE_SCOPE):
        values = machine_scope[field]
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise ValueError(f"machine_check_scope.{field} must be a non-empty string list")
    if not any("substantive" in value.lower() for value in machine_scope["not_checked"]):
        raise ValueError("machine_check_scope must disclaim substantive validity")
    artifact_ids = validate_artifact_registry(graph)
    rules = graph.get("status_propagation_rules")
    if not isinstance(rules, dict) or not REQUIRED_RULES.issubset(rules):
        raise ValueError("One or more required status-propagation rules are missing")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("nodes and edges must be lists")
    if not all(isinstance(node, dict) for node in nodes):
        raise ValueError("Every node must be an object")
    if not all(isinstance(edge, dict) for edge in edges):
        raise ValueError("Every edge must be an object")
    ids = [str(node.get("id")) for node in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Node identifiers must be unique")
    node_by_id = {str(node["id"]): node for node in nodes}
    for node in nodes:
        node_id = str(node["id"])
        status = str(node.get("status"))
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Unknown status {status!r}")
        missing_lifecycle = REQUIRED_NODE_LIFECYCLE_FIELDS - set(node)
        if missing_lifecycle:
            raise ValueError(
                f"Node {node_id} lacks lifecycle fields: {sorted(missing_lifecycle)}"
            )
        if not str(node["status_rationale"]).strip():
            raise ValueError(f"Node {node_id} has an empty status_rationale")
        evidence_ids = node["evidence_artifact_ids"]
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError(f"Node {node_id} must cite aggregate evidence artifacts")
        unknown_artifacts = {str(value) for value in evidence_ids} - artifact_ids
        if unknown_artifacts:
            raise ValueError(
                f"Node {node_id} cites unknown artifacts: {sorted(unknown_artifacts)}"
            )
        if not str(node["decision_owner_role"]).strip():
            raise ValueError(f"Node {node_id} has an empty decision_owner_role")
        validate_iso_date(node["decision_date"], f"{node_id}.decision_date")
        review_due = node["review_due_date"]
        if review_due is not None:
            validate_iso_date(review_due, f"{node_id}.review_due_date")
        if not str(node["review_trigger"]).strip():
            raise ValueError(f"Node {node_id} has an empty review_trigger")
        for language in ("en", "zh"):
            table = node.get(f"table_{language}")
            if not isinstance(table, dict):
                raise ValueError(f"Node {node_id} lacks table_{language}")
            missing = {
                "node",
                "evidence",
                "boundary",
                "permitted_claim",
                "decision",
            } - set(table)
            if missing:
                raise ValueError(f"Node {node_id} lacks fields: {sorted(missing)}")
        if status == "PARTIAL":
            permits = {str(value) for value in node.get("permits", [])}
            forbidden = {"durable_retention", "user_benefit", "consequential_deployment"}
            if permits & forbidden:
                raise ValueError(
                    f"PARTIAL node {node_id} cannot permit {sorted(permits & forbidden)}"
                )
    for edge in edges:
        edge_type = str(edge.get("type"))
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if edge_type not in ALLOWED_EDGE_TYPES:
            raise ValueError(f"Unknown edge type {edge_type!r}")
        if source not in node_by_id or target not in node_by_id:
            raise ValueError(f"Edge endpoint is missing: {source} -> {target}")
        if edge_type == "blocks_deployment":
            source_status = str(node_by_id[source]["status"])
            target_status = str(node_by_id[target]["status"])
            if source_status != "AVAILABLE" and target_status == "AVAILABLE":
                raise ValueError(
                    f"Blocked target {target} cannot be AVAILABLE while "
                    f"{source} is {source_status}"
                )


def chinese_table_text(value: object) -> str:
    """Translate residual prose without changing the evidence-graph source."""

    text = str(value)
    for old, new in (
        ("video-only correction", "仅视频校正"),
        ("participant-held-out", "参与者留出"),
        ("发布 support 上", "发布支持集上的"),
        ("held-out", "留出"),
        ("joystick", "摇杆"),
        ("donor", "供体"),
        ("Session", "会话"),
        ("session", "会话"),
        ("trait", "特质"),
        (" min", " 分钟"),
    ):
        text = text.replace(old, new)
    return text


def table_rows(graph: dict[str, object], language: str) -> str:
    output = [
        (
            "% 由 paper_support/evidence_traceability.json 自动生成；请勿手工编辑。"
            if language == "zh"
            else "% Generated from paper_support/evidence_traceability.json; do not edit by hand."
        )
    ]
    status_zh = {
        "AVAILABLE": "可用",
        "PARTIAL": "部分",
        "NOT_TESTED": "未检验",
        "NO_DEMONSTRATED_INCREMENT": "未证明增量",
        "NOT_JUSTIFIED": "缺乏正当性",
    }
    for node in graph["nodes"]:
        if not bool(node.get("include_in_table", True)):
            continue
        table = node[f"table_{language}"]
        status = str(node.get("table_status", node["status"]))
        fields = [
            table["node"],
            status_zh[status.replace(" ", "_")] if language == "zh" else status.replace("_", " "),
            table["evidence"],
            table["boundary"],
            table["permitted_claim"],
            table["decision"],
        ]
        if language == "zh":
            fields = [chinese_table_text(field) for field in fields]
        output.append(" & ".join(fields) + r" \\")
    output.append(r"\bottomrule")
    return "\n".join(output) + "\n"


def main() -> None:
    args = parse_args()
    graph = json.loads(args.schema.read_text(encoding="utf-8"))
    validate_graph(graph)
    expected_en = table_rows(graph, "en")
    expected_zh = table_rows(graph, "zh")
    if args.check:
        actual_en = args.output_en.read_text(encoding="utf-8")
        actual_zh = args.output_zh.read_text(encoding="utf-8")
        if actual_en != expected_en or actual_zh != expected_zh:
            raise RuntimeError("Generated evidence-table rows are stale")
    else:
        args.output_en.parent.mkdir(parents=True, exist_ok=True)
        args.output_zh.parent.mkdir(parents=True, exist_ok=True)
        args.output_en.write_text(expected_en, encoding="utf-8")
        args.output_zh.write_text(expected_zh, encoding="utf-8")
    print(
        f"Validated {len(graph['nodes'])} nodes and {len(graph['edges'])} typed edges."
    )


if __name__ == "__main__":
    main()
