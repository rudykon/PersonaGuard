#!/usr/bin/env python3
"""Replay the route-audit protocol on worked and external HCI cases.

The evaluator is intentionally declarative. Route records contain evidence
states but no expected action or rule identifier; one frozen rule file resolves
both the MER-PS worked cases and literature-derived external cases. Machine
checks establish structural conformance and deterministic propagation only.
They do not validate the truth of the coding, the rules' substantive adequacy,
human agreement, usability, fairness, or benefit outside the coded boundary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "paper_support" / "evidence_traceability.json"
RULES_PATH = ROOT / "paper_support" / "protocol_rules.json"
CASES_PATH = ROOT / "paper_support" / "protocol_replay_cases.json"
EXTERNAL_CASES_PATH = ROOT / "paper_support" / "external_reuse_cases.json"
OUTPUT_PATH = ROOT / "artifacts" / "revision6" / "protocol_replay.json"
VALIDATOR_PATH = ROOT / "scripts" / "generate_evidence_traceability.py"
ORDER_REPLAYS = 100
SEED = 20270810

ALLOWED_STAGES = {
    "OFFLINE_ESTIMATION",
    "OPTIONAL_ACQUISITION",
    "REVERSIBLE_PERSONALIZATION",
    "CONSEQUENTIAL_PERSONALIZATION",
    "RETENTION_TRANSFER",
}
ALLOWED_CAPABILITIES = {"AVAILABLE", "PARTIAL", "NOT_AVAILABLE"}
FORBIDDEN_CASE_FIELDS = {"expected_action", "audited_action", "rule_id"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resolve-case-set",
        type=Path,
        help=(
            "Validate and resolve a supplied route-audit-cases-v2 file with the "
            "same declarative resolver, without running benchmark mutations."
        ),
    )
    parser.add_argument(
        "--external-case-set",
        action="store_true",
        help="Treat --resolve-case-set as an external set with DOI-anchored sources.",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=RULES_PATH,
        help="Rule file for --resolve-case-set (default: paper_support/protocol_rules.json).",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_value(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def load_graph_validator():
    spec = importlib.util.spec_from_file_location(
        "protocol_replay_graph_validator", VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load evidence-graph validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_field(record: dict[str, object], dotted: str) -> object:
    value: object = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Predicate references missing field: {dotted}")
        value = value[part]
    return value


def set_field(record: dict[str, object], dotted: str, replacement: object) -> None:
    container: dict[str, object] = record
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = container.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"Mutation references missing object: {dotted}")
        container = child
    container[parts[-1]] = replacement


def predicate_fields(predicate: dict[str, object]) -> set[str]:
    if "field" in predicate:
        return {str(predicate["field"])}
    fields: set[str] = set()
    for operator in ("all", "any"):
        children = predicate.get(operator)
        if children is not None:
            if not isinstance(children, list) or not children:
                raise ValueError(f"Predicate {operator} must be a non-empty list")
            for child in children:
                if not isinstance(child, dict):
                    raise ValueError("Predicate children must be objects")
                fields.update(predicate_fields(child))
    if not fields:
        raise ValueError("Predicate must contain field, all, or any")
    return fields


def matches(predicate: dict[str, object], record: dict[str, object]) -> bool:
    if "all" in predicate:
        return all(matches(child, record) for child in predicate["all"])
    if "any" in predicate:
        return any(matches(child, record) for child in predicate["any"])
    field = str(predicate["field"])
    observed = get_field(record, field)
    operators = [operator for operator in ("eq", "in", "not_in") if operator in predicate]
    if len(operators) != 1:
        raise ValueError(f"Predicate for {field} must declare exactly one operator")
    operator = operators[0]
    expected = predicate[operator]
    if operator == "eq":
        return observed == expected
    if not isinstance(expected, list) or not expected:
        raise ValueError(f"Predicate {operator} for {field} must be a non-empty list")
    return (observed in expected) if operator == "in" else (observed not in expected)


def validate_protocol(protocol: dict[str, object]) -> None:
    if protocol.get("schema_version") != "route-audit-protocol-v2":
        raise ValueError("Unexpected route-audit protocol schema")
    required = protocol.get("required_route_fields")
    vocabularies = protocol.get("evidence_fields")
    rules = protocol.get("rules")
    labels = protocol.get("action_labels")
    resolver = protocol.get("resolver")
    if not isinstance(required, list) or not required:
        raise ValueError("Protocol must declare required route fields")
    if not isinstance(vocabularies, dict) or not vocabularies:
        raise ValueError("Protocol must declare evidence vocabularies")
    if not isinstance(rules, list) or not rules:
        raise ValueError("Protocol must declare rules")
    if not isinstance(labels, dict) or not labels:
        raise ValueError("Protocol must declare action labels")
    if not isinstance(resolver, dict) or resolver.get("fallback_action") not in labels:
        raise ValueError("Protocol fallback action must have a label")

    rule_ids: list[str] = []
    priorities: list[int] = []
    referenced_fields: set[str] = set()
    legal_fields = set(required) | {
        f"evidence.{field}" for field in vocabularies
    }
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("Every protocol rule must be an object")
        for field in ("id", "priority", "kind", "predicate", "action", "derivation_sources"):
            if field not in rule:
                raise ValueError(f"Protocol rule lacks {field}")
        rule_ids.append(str(rule["id"]))
        priorities.append(int(rule["priority"]))
        if rule["kind"] not in {"constraint", "permission"}:
            raise ValueError("Rule kind must be constraint or permission")
        if rule["action"] not in labels:
            raise ValueError(f"Rule action lacks a label: {rule['action']}")
        if not isinstance(rule["derivation_sources"], list) or not rule["derivation_sources"]:
            raise ValueError("Every rule requires at least one derivation source")
        fields = predicate_fields(rule["predicate"])
        referenced_fields.update(fields)
        unknown = fields - legal_fields
        if unknown:
            raise ValueError(f"Rule predicate uses unknown fields: {sorted(unknown)}")
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("Protocol rule identifiers must be unique")
    if len(priorities) != len(set(priorities)):
        raise ValueError("Protocol rule priorities must be unique")
    unreferenced_evidence = {
        f"evidence.{field}" for field in vocabularies
    } - referenced_fields
    if unreferenced_evidence:
        raise ValueError(
            "Protocol declares evidence fields unused by every rule: "
            f"{sorted(unreferenced_evidence)}"
        )


def validate_source_locator(case: dict[str, object], external: bool) -> None:
    locator = str(case["source_locator"])
    if external:
        if not locator.startswith("doi:10.") or "#" not in locator:
            raise ValueError(f"External case requires DOI plus source anchor: {locator}")
        return
    relative = locator.split("#", 1)[0]
    path = ROOT / relative
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Case source escapes project root: {locator}") from error
    if not path.is_file():
        raise ValueError(f"Case source does not exist: {locator}")


def validate_case_set(
    specification: dict[str, object],
    protocol: dict[str, object],
    *,
    external: bool,
) -> None:
    if specification.get("schema_version") != "route-audit-cases-v2":
        raise ValueError("Unexpected route-audit case schema")
    if specification.get("protocol_version") != protocol.get("protocol_version"):
        raise ValueError("Case-set and protocol versions differ")
    cases = specification.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Route-audit case set must not be empty")
    required = set(protocol["required_route_fields"])
    vocabularies = protocol["evidence_fields"]
    ids: list[str] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every route case must be an object")
        missing = required - set(case)
        if missing:
            raise ValueError(f"Route case lacks fields: {sorted(missing)}")
        forbidden = FORBIDDEN_CASE_FIELDS & set(case)
        if forbidden:
            raise ValueError(f"Route case encodes its own answer: {sorted(forbidden)}")
        ids.append(str(case["id"]))
        if case["deployment_stage"] not in ALLOWED_STAGES:
            raise ValueError(f"Unknown deployment stage: {case['deployment_stage']}")
        if case["route_capability"] not in ALLOWED_CAPABILITIES:
            raise ValueError(f"Unknown route capability: {case['route_capability']}")
        evidence = case["evidence"]
        if not isinstance(evidence, dict) or set(evidence) != set(vocabularies):
            raise ValueError("Route evidence fields must exactly match the protocol schema")
        for field, allowed in vocabularies.items():
            if evidence[field] not in allowed:
                raise ValueError(
                    f"Illegal evidence value {field}={evidence[field]} for {case['id']}"
                )
        validate_source_locator(case, external)
    if len(ids) != len(set(ids)):
        raise ValueError("Route case identifiers must be unique within a case set")

    if external:
        sources = specification.get("sources")
        selection = specification.get("selection")
        if not isinstance(sources, list) or not isinstance(selection, dict):
            raise ValueError("External benchmark requires sources and selection")
        source_ids = [str(source["id"]) for source in sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("External source identifiers must be unique")
        if int(selection.get("source_count", -1)) != len(sources):
            raise ValueError("External source_count is stale")
        if int(selection.get("case_count", -1)) != len(cases):
            raise ValueError("External case_count is stale")
        for source in sources:
            if not str(source.get("doi", "")).startswith("10."):
                raise ValueError("Every external source requires a DOI")
            if not str(source.get("primary_source_url", "")).startswith("https://"):
                raise ValueError("Every external source requires an HTTPS primary source")
        unknown_sources = {str(case["source_id"]) for case in cases} - set(source_ids)
        if unknown_sources:
            raise ValueError(f"External cases cite unknown sources: {sorted(unknown_sources)}")


def evaluate_case(
    case: dict[str, object],
    protocol: dict[str, object],
    *,
    disabled_rules: set[str] | None = None,
) -> dict[str, object]:
    disabled = disabled_rules or set()
    ordered_rules = sorted(protocol["rules"], key=lambda rule: int(rule["priority"]))
    matched_rule: dict[str, object] | None = None
    for rule in ordered_rules:
        if str(rule["id"]) in disabled:
            continue
        if matches(rule["predicate"], case):
            matched_rule = rule
            break
    if matched_rule is None:
        action = str(protocol["resolver"]["fallback_action"])
        rule_id = "FALLBACK"
        rule_kind = "evidence_request"
        decisive_fields: list[str] = []
    else:
        action = str(matched_rule["action"])
        rule_id = str(matched_rule["id"])
        rule_kind = str(matched_rule["kind"])
        decisive_fields = sorted(predicate_fields(matched_rule["predicate"]))
    labels = protocol["action_labels"][action]
    return {
        "id": str(case["id"]),
        "source_id": str(case["source_id"]),
        "label_en": str(case["label_en"]),
        "label_zh": str(case["label_zh"]),
        "proposed_use_en": str(case["proposed_use_en"]),
        "proposed_use_zh": str(case["proposed_use_zh"]),
        "evidence_basis_en": str(case["evidence_basis_en"]),
        "evidence_basis_zh": str(case["evidence_basis_zh"]),
        "source_locator": str(case["source_locator"]),
        "naive_action": str(case["naive_action"]),
        "audited_action": action,
        "audited_decision_en": str(labels["en"]),
        "audited_decision_zh": str(labels["zh"]),
        "matched_rule_id": rule_id,
        "matched_rule_kind": rule_kind,
        "decisive_fields": decisive_fields,
        "decisive_values": {
            field: get_field(case, field) for field in decisive_fields
        },
        "decision_changed": str(case["naive_action"]) != action,
    }


def evaluate_cases(
    specification: dict[str, object],
    protocol: dict[str, object],
    *,
    disabled_rules: set[str] | None = None,
) -> list[dict[str, object]]:
    return sorted(
        [
            evaluate_case(case, protocol, disabled_rules=disabled_rules)
            for case in specification["cases"]
        ],
        key=lambda result: str(result["id"]),
    )


def order_invariance_audit(
    graph: dict[str, object],
    protocol: dict[str, object],
    worked: dict[str, object],
    external: dict[str, object],
    baseline: dict[str, object],
) -> int:
    rng = random.Random(SEED)
    expected = canonical(baseline)
    for _ in range(ORDER_REPLAYS):
        shuffled_graph = copy.deepcopy(graph)
        shuffled_protocol = copy.deepcopy(protocol)
        shuffled_worked = copy.deepcopy(worked)
        shuffled_external = copy.deepcopy(external)
        for field in ("nodes", "edges", "artifact_registry"):
            rng.shuffle(shuffled_graph[field])
        rng.shuffle(shuffled_protocol["rules"])
        rng.shuffle(shuffled_worked["cases"])
        rng.shuffle(shuffled_external["cases"])
        rng.shuffle(shuffled_external["sources"])
        observed = {
            "worked": evaluate_cases(shuffled_worked, shuffled_protocol),
            "external": evaluate_cases(shuffled_external, shuffled_protocol),
        }
        if canonical(observed) != expected:
            raise RuntimeError("Protocol replay changed after order permutation")
    return ORDER_REPLAYS


def invalid_mutation_audit(
    graph: dict[str, object],
    protocol: dict[str, object],
    worked: dict[str, object],
    external: dict[str, object],
    validator,
) -> tuple[int, int]:
    mutations: list[tuple[str, dict[str, object]]] = []

    duplicate = copy.deepcopy(graph)
    duplicate["nodes"][1]["id"] = duplicate["nodes"][0]["id"]
    mutations.append(("graph", duplicate))

    unknown_status = copy.deepcopy(graph)
    unknown_status["nodes"][0]["status"] = "ASSERTED_VALID"
    mutations.append(("graph", unknown_status))

    missing_rule = copy.deepcopy(graph)
    missing_rule["status_propagation_rules"].pop("route_failure_is_local")
    mutations.append(("graph", missing_rule))

    unknown_edge = copy.deepcopy(graph)
    unknown_edge["edges"][0]["type"] = "causes"
    mutations.append(("graph", unknown_edge))

    bad_hash = copy.deepcopy(graph)
    bad_hash["artifact_registry"][0]["sha256"] = "0" * 64
    mutations.append(("graph", bad_hash))

    forbidden_permit = copy.deepcopy(graph)
    partial = next(node for node in forbidden_permit["nodes"] if node["status"] == "PARTIAL")
    partial["permits"].append("user_benefit")
    mutations.append(("graph", forbidden_permit))

    answered_case = copy.deepcopy(worked)
    answered_case["cases"][0]["expected_action"] = "BOUND_TO_EVIDENCE_SCOPE"
    mutations.append(("worked", answered_case))

    illegal_value = copy.deepcopy(worked)
    illegal_value["cases"][0]["evidence"]["interpretation_scope"] = "VALID"
    mutations.append(("worked", illegal_value))

    missing_field = copy.deepcopy(external)
    missing_field["cases"][0]["evidence"].pop("user_outcome")
    mutations.append(("external", missing_field))

    unknown_source = copy.deepcopy(external)
    unknown_source["cases"][0]["source_id"] = "invented_source"
    mutations.append(("external", unknown_source))

    duplicate_rule = copy.deepcopy(protocol)
    duplicate_rule["rules"][1]["id"] = duplicate_rule["rules"][0]["id"]
    mutations.append(("protocol", duplicate_rule))

    unknown_predicate = copy.deepcopy(protocol)
    unknown_predicate["rules"][0]["predicate"] = {
        "field": "evidence.magic_validity", "eq": "SUPPORTED"
    }
    mutations.append(("protocol", unknown_predicate))

    rejected = 0
    for kind, mutation in mutations:
        try:
            if kind == "graph":
                validator.validate_graph(mutation)
            elif kind == "protocol":
                validate_protocol(mutation)
            elif kind == "worked":
                validate_case_set(mutation, protocol, external=False)
            else:
                validate_case_set(mutation, protocol, external=True)
        except (KeyError, TypeError, ValueError, RuntimeError):
            rejected += 1
    if rejected != len(mutations):
        raise RuntimeError("One or more malformed protocol records were accepted")
    return len(mutations), rejected


def route_locality_audit(
    worked: dict[str, object],
    external: dict[str, object],
    protocol: dict[str, object],
) -> tuple[int, int]:
    case_sets = {
        "worked": {str(case["id"]): case for case in worked["cases"]},
        "external": {str(case["id"]): case for case in external["cases"]},
    }
    probes = [
        ("worked", "trace_interpretation", "evidence.comparator_increment", "SUPPORTED"),
        ("worked", "optional_sensing", "evidence.user_control", "SUPPORTED"),
        ("worked", "signed_calibration", "evidence.persistence", "SUPPORTED"),
        ("worked", "retention_transfer", "evidence.comparator_increment", "SUPPORTED"),
        ("external", "supple_motor_access", "evidence.persistence", "SUPPORTED"),
    ]
    passed = 0
    for group, case_id, dotted, replacement in probes:
        case = case_sets[group][case_id]
        baseline = evaluate_case(case, protocol)["audited_action"]
        mutation = copy.deepcopy(case)
        set_field(mutation, dotted, replacement)
        observed = evaluate_case(mutation, protocol)["audited_action"]
        if observed == baseline:
            passed += 1
    if passed != len(probes):
        raise RuntimeError("A route-locality probe changed an unrelated decision")
    return len(probes), passed


def permission_boundary_audit(
    worked: dict[str, object],
    external: dict[str, object],
    protocol: dict[str, object],
) -> dict[str, object]:
    cases = [
        ("worked", case) for case in worked["cases"]
    ] + [("external", case) for case in external["cases"]]
    positive = [
        (case_set, case)
        for case_set, case in cases
        if evaluate_case(case, protocol)["audited_action"]
        == "PROCEED_WITHIN_EVALUATED_BOUNDARY"
    ]
    if not positive:
        raise RuntimeError("Permission-boundary audit requires a positive route")
    probes: list[tuple[str, list[tuple[str, object]]]] = [
        ("capability", [("route_capability", "NOT_AVAILABLE")]),
        ("scope", [("evidence.interpretation_scope", "PARTIAL")]),
        (
            "comparator",
            [("evidence.comparator_increment", "NO_DEMONSTRATED_INCREMENT")],
        ),
        ("evaluation", [("evidence.evaluation_mode", "PROXY_ONLY")]),
        ("outcome", [("evidence.user_outcome", "PROXIMAL_ONLY")]),
        (
            "control_or_reversibility",
            [
                ("evidence.user_control", "NOT_TESTED"),
                ("evidence.safety_reversibility", "NOT_TESTED"),
            ],
        ),
    ]
    passed = 0
    total = 0
    for case_set, case in positive:
        for dimension, changes in probes:
            total += 1
            mutation = copy.deepcopy(case)
            for dotted, replacement in changes:
                set_field(mutation, dotted, replacement)
            observed = evaluate_case(mutation, protocol)
            if observed["audited_action"] == "PROCEED_WITHIN_EVALUATED_BOUNDARY":
                raise RuntimeError(
                    "Positive permission leaked across a failed boundary: "
                    f"{case_set}/{case['id']}/{dimension}"
                )
            passed += 1
    return {
        "positive_case_count": len(positive),
        "positive_case_ids": [
            f"{case_set}:{case['id']}" for case_set, case in positive
        ],
        "dimensions": [dimension for dimension, _ in probes],
        "probes": total,
        "probes_passed": passed,
    }


def precedence_audit(
    worked: dict[str, object],
    protocol: dict[str, object],
) -> list[dict[str, object]]:
    by_id = {str(case["id"]): case for case in worked["cases"]}
    specifications = [
        (
            "R2_over_R1",
            "trace_interpretation",
            [("evidence.comparator_increment", "NO_DEMONSTRATED_INCREMENT")],
            "R2_COMPARATOR_INCREMENT",
        ),
        (
            "R2_over_R3",
            "signed_calibration",
            [("evidence.comparator_increment", "NO_DEMONSTRATED_INCREMENT")],
            "R2_COMPARATOR_INCREMENT",
        ),
        (
            "R3_over_R1",
            "signed_calibration",
            [],
            "R3_CONSEQUENCE_MATCH",
        ),
        (
            "R4_over_R2_and_R1",
            "retention_transfer",
            [("evidence.comparator_increment", "NO_DEMONSTRATED_INCREMENT")],
            "R4_RETENTION_TRANSFER",
        ),
    ]
    rows: list[dict[str, object]] = []
    for probe_id, case_id, changes, expected_rule in specifications:
        mutation = copy.deepcopy(by_id[case_id])
        for dotted, replacement in changes:
            set_field(mutation, dotted, replacement)
        observed = evaluate_case(mutation, protocol)
        if observed["matched_rule_id"] != expected_rule:
            raise RuntimeError(
                f"Declared precedence failed for {probe_id}: "
                f"{observed['matched_rule_id']} != {expected_rule}"
            )
        rows.append(
            {
                "id": probe_id,
                "case_id": case_id,
                "expected_rule_id": expected_rule,
                "observed_rule_id": observed["matched_rule_id"],
                "passed": True,
            }
        )
    return rows


def rule_ablation_audit(
    protocol: dict[str, object],
    worked: dict[str, object],
    external: dict[str, object],
) -> list[dict[str, object]]:
    all_cases = [
        ("worked", case) for case in worked["cases"]
    ] + [("external", case) for case in external["cases"]]
    rows: list[dict[str, object]] = []
    for rule in sorted(protocol["rules"], key=lambda item: str(item["id"])):
        rule_id = str(rule["id"])
        changed: list[dict[str, object]] = []
        for case_set, case in all_cases:
            baseline = evaluate_case(case, protocol)
            ablated = evaluate_case(case, protocol, disabled_rules={rule_id})
            if baseline["audited_action"] != ablated["audited_action"]:
                changed.append(
                    {
                        "case_set": case_set,
                        "case_id": str(case["id"]),
                        "baseline_action": baseline["audited_action"],
                        "ablated_action": ablated["audited_action"],
                        "replacement_rule_id": ablated["matched_rule_id"],
                    }
                )
        if not changed:
            raise RuntimeError(f"Rule ablation had no observable effect: {rule_id}")
        rows.append(
            {
                "rule_id": rule_id,
                "kind": str(rule["kind"]),
                "title_en": str(rule["title_en"]),
                "cases_changed": len(changed),
                "changes": changed,
                "interpretation": (
                    "permission lost" if rule["kind"] == "permission" else "constraint or route-specific rationale lost"
                ),
            }
        )
    return rows


def build_result() -> dict[str, object]:
    graph = read_json(GRAPH_PATH)
    protocol = read_json(RULES_PATH)
    worked = read_json(CASES_PATH)
    external = read_json(EXTERNAL_CASES_PATH)
    validator = load_graph_validator()
    validator.validate_graph(graph)
    validate_protocol(protocol)
    validate_case_set(worked, protocol, external=False)
    validate_case_set(external, protocol, external=True)

    worked_results = evaluate_cases(worked, protocol)
    external_results = evaluate_cases(external, protocol)
    combined = {"worked": worked_results, "external": external_results}
    order_replays = order_invariance_audit(
        graph, protocol, worked, external, combined
    )
    mutation_total, mutation_rejected = invalid_mutation_audit(
        graph, protocol, worked, external, validator
    )
    locality_total, locality_passed = route_locality_audit(
        worked, external, protocol
    )
    permission_boundary = permission_boundary_audit(worked, external, protocol)
    precedence_probes = precedence_audit(worked, protocol)
    ablations = rule_ablation_audit(protocol, worked, external)

    worked_actions = {str(row["audited_action"]) for row in worked_results}
    external_actions = {str(row["audited_action"]) for row in external_results}
    positive_external = sum(
        row["audited_action"] == "PROCEED_WITHIN_EVALUATED_BOUNDARY"
        for row in external_results
    )
    constrained_external = len(external_results) - positive_external
    source_records = [
        {
            **source,
            "coding_record_sha256": sha256_value(source),
        }
        for source in sorted(external["sources"], key=lambda item: str(item["id"]))
    ]
    return {
        "schema_version": "protocol-replay-result-v2",
        "protocol_version": protocol["protocol_version"],
        "source_records": {
            "evidence_graph": {"uri": "paper_support/evidence_traceability.json", "sha256": sha256(GRAPH_PATH)},
            "protocol_rules": {"uri": "paper_support/protocol_rules.json", "sha256": sha256(RULES_PATH)},
            "worked_cases": {"uri": "paper_support/protocol_replay_cases.json", "sha256": sha256(CASES_PATH)},
            "external_cases": {"uri": "paper_support/external_reuse_cases.json", "sha256": sha256(EXTERNAL_CASES_PATH)},
        },
        "machine_checked_scope": protocol["machine_check_scope"]["checked"],
        "not_machine_checked": protocol["machine_check_scope"]["not_checked"],
        "rule_inventory": [
            {
                "id": rule["id"],
                "priority": rule["priority"],
                "kind": rule["kind"],
                "title_en": rule["title_en"],
                "title_zh": rule["title_zh"],
                "risk_blocked_en": rule["risk_blocked_en"],
                "action": rule["action"],
                "derivation_sources": rule["derivation_sources"],
            }
            for rule in sorted(protocol["rules"], key=lambda item: int(item["priority"]))
        ],
        "case_results": worked_results,
        "worked_case_results": worked_results,
        "external_case_results": external_results,
        "external_source_records": source_records,
        "permission_boundary": permission_boundary,
        "precedence_probes": precedence_probes,
        "rule_ablation": ablations,
        "validation": {
            "case_count": len(worked_results),
            "distinct_audited_actions": len(worked_actions),
            "naive_decisions_changed": sum(bool(row["decision_changed"]) for row in worked_results),
            "order_permutations": order_replays,
            "order_invariant": True,
            "invalid_mutations": mutation_total,
            "invalid_mutations_rejected": mutation_rejected,
            "route_locality_probes": locality_total,
            "route_locality_probes_passed": locality_passed,
            "permission_boundary_probes": permission_boundary["probes"],
            "permission_boundary_probes_passed": permission_boundary["probes_passed"],
            "precedence_probes": len(precedence_probes),
            "precedence_probes_passed": sum(
                bool(row["passed"]) for row in precedence_probes
            ),
            "combined_distinct_actions": len(worked_actions | external_actions),
            "rule_ablations": len(ablations),
            "rule_ablations_with_action_change": sum(row["cases_changed"] > 0 for row in ablations),
        },
        "external_transfer": {
            "selection_strategy": external["selection"]["strategy"],
            "source_count": len(source_records),
            "case_count": len(external_results),
            "distinct_audited_actions": len(external_actions),
            "positive_bounded_actions": positive_external,
            "constrained_or_study_actions": constrained_external,
            "represented_without_schema_extension": True,
            "fallback_actions": sum(row["matched_rule_id"] == "FALLBACK" for row in external_results),
        },
        "interpretation_boundary": (
            "The same frozen schema and resolver represented worked and external published HCI routes, including positive and constraining actions, without a schema extension. The external coding was performed by the present authors after protocol development; it improves cross-source representational evidence but is not an independent analyst study, an inter-rater test, or proof that the substantive actions are correct."
        ),
    }


def serialized(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def resolve_case_set(
    case_set_path: Path,
    rules_path: Path,
    *,
    external: bool,
) -> dict[str, object]:
    protocol = read_json(rules_path)
    specification = read_json(case_set_path)
    validate_protocol(protocol)
    validate_case_set(specification, protocol, external=external)
    return {
        "schema_version": "protocol-case-set-resolution-v1",
        "protocol_version": protocol["protocol_version"],
        "case_set": specification.get("case_set", case_set_path.stem),
        "external_case_set": external,
        "protocol_sha256": sha256(rules_path),
        "case_set_sha256": sha256(case_set_path),
        "machine_checked_scope": (
            "Schema, declared vocabularies, source-locator form, and frozen "
            "rule resolution for the supplied records."
        ),
        "not_machine_checked": [
            "substantive validity of evidence coding",
            "human analyst agreement",
            "fairness, usability, or user benefit",
        ],
        "case_results": evaluate_cases(specification, protocol),
    }


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    if args.resolve_case_set is not None:
        if args.check:
            raise SystemExit("--check cannot be combined with --resolve-case-set")
        case_set_path = project_path(args.resolve_case_set)
        rules_path = project_path(args.rules)
        payload = serialized(
            resolve_case_set(
                case_set_path,
                rules_path,
                external=bool(args.external_case_set),
            )
        )
        if args.output is None:
            print(payload, end="")
            return
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        try:
            label = output.relative_to(ROOT).as_posix()
        except ValueError:
            label = output.as_posix()
        print(f"Wrote {label}")
        return

    if args.external_case_set:
        raise SystemExit("--external-case-set requires --resolve-case-set")
    if project_path(args.rules) != RULES_PATH:
        raise SystemExit("--rules is only used with --resolve-case-set")

    output = project_path(args.output or OUTPUT_PATH)
    expected = serialized(build_result())
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != expected:
            raise SystemExit(
                "Protocol replay output is stale; run "
                ".venv/bin/python scripts/run_protocol_replay.py"
            )
        print("Protocol replay result is current")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
