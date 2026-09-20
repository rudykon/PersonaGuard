#!/usr/bin/env python3
"""Generate three protocol-facing main-paper figures from Revision 6.

Protocol definition, evaluation design, and evaluation results are separated
so implementation evidence is not promoted into claims about analyst
agreement, usability, user benefit, safety, privacy, or deployment.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from vector_export import save_pdfua_embed
from manuscript_fonts import with_manuscript_fonts


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "revision6_source.json"
FIGURE_DIR = ROOT / "artifacts" / "figures"
DATA_DIR = Path(__file__).resolve().parent
FIGURE_WIDTH_MM = 182.1
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4
OUTPUT_SUFFIXES = (".svg", ".pdf", ".png", ".tiff")
RASTER_DPI = 600

COLORS = {
    "ink": "#172B4D",
    "muted": "#5B6573",
    "grid": "#D9E1E8",
    "panel": "#F7F9FB",
    "record": "#DCEAF7",
    "proceed": "#168C7E",
    "proceed_light": "#E3F2EE",
    "bound": "#756BB1",
    "bound_light": "#ECE9F5",
    "comparator": "#D9822B",
    "comparator_light": "#F8EBDC",
    "study": "#3D6FB6",
    "study_light": "#E4ECF7",
    "withhold": "#B55A52",
    "withhold_light": "#F5E5E3",
    "gray": "#6B778C",
    "gray_light": "#EEF1F4",
}

ACTION_STYLE = {
    "PROCEED_WITHIN_EVALUATED_BOUNDARY": (
        "Bounded permission", COLORS["proceed_light"], COLORS["proceed"],
    ),
    "BOUND_TO_EVIDENCE_SCOPE": (
        "Scope restriction", COLORS["bound_light"], COLORS["bound"],
    ),
    "RETAIN_EVALUATED_COMPARATOR": (
        "Retain comparator", COLORS["comparator_light"], COLORS["comparator"],
    ),
    "RUN_PREREGISTERED_USER_STUDY": (
        "In-context study", COLORS["study_light"], COLORS["study"],
    ),
    "WITHHOLD_RETENTION_AND_TRANSFER": (
        "Withhold retention / transfer", COLORS["withhold_light"], COLORS["withhold"],
    ),
    "REQUIRE_ROUTE_SPECIFIC_EVIDENCE": (
        "Evidence request", COLORS["gray_light"], COLORS["gray"],
    ),
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    }
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    common = {}  # Preserve the final 182.1-mm manuscript width.
    for suffix in OUTPUT_SUFFIXES:
        kwargs = dict(common)
        if suffix in {".png", ".tiff"}:
            kwargs["dpi"] = RASTER_DPI
        fig.savefig(FIGURE_DIR / f"{stem}{suffix}", **kwargs)
    save_pdfua_embed(fig, FIGURE_DIR / f"{stem}_embed.pdf", common)
    plt.close(fig)
    missing = [
        suffix
        for suffix in OUTPUT_SUFFIXES
        if not (FIGURE_DIR / f"{stem}{suffix}").is_file()
        or (FIGURE_DIR / f"{stem}{suffix}").stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Figure export contract failed for {stem}: {missing}")
    embed = FIGURE_DIR / f"{stem}_embed.pdf"
    if not embed.is_file() or embed.stat().st_size == 0:
        raise RuntimeError(f"Figure embed contract failed for {stem}")


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    facecolor: str = "#FFFFFF",
    edgecolor: str = COLORS["grid"],
    title_size: float = 6.2,
    body_size: float = 5.5,
    title_color: str = COLORS["ink"],
    align: str = "left",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=ax.transAxes,
        linewidth=0.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    text_x = x + 0.016 if align == "left" else x + width / 2
    horizontal = "left" if align == "left" else "center"
    ax.text(
        text_x,
        y + height * 0.70,
        title,
        transform=ax.transAxes,
        ha=horizontal,
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        text_x,
        y + height * 0.34,
        body,
        transform=ax.transAxes,
        ha=horizontal,
        va="center",
        fontsize=body_size,
        color=COLORS["muted"],
        linespacing=1.22,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["muted"],
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.9},
    )


def panel_heading(ax: plt.Axes, x: float, label: str, title: str) -> None:
    ax.text(
        x, 0.975, label, transform=ax.transAxes, ha="left", va="top",
        fontsize=8.2, fontweight="bold", color=COLORS["ink"],
    )
    ax.text(
        x + 0.043, 0.975, title, transform=ax.transAxes, ha="left", va="top",
        fontsize=7.5, fontweight="bold", color=COLORS["ink"],
    )


def figure_protocol_overview(protocol: dict[str, object]) -> None:
    """One route, ordered resolution, and a human-owned next action."""
    rules = protocol["rule_inventory"]
    expected_order = ["R4_RETENTION_TRANSFER", "R2_COMPARATOR_INCREMENT",
                      "R3_CONSEQUENCE_MATCH", "R1_SCOPE_MATCH", "R5_BOUNDED_PASS"]
    if [row["id"] for row in rules] != expected_order:
        raise RuntimeError("Protocol rule order no longer matches the frozen contract")
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 4.65))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.985)
    ax.set_axis_off()
    panel_heading(ax, 0.010, "(A)", "Route declaration")
    panel_heading(ax, 0.335, "(B)", "Ordered resolution")
    panel_heading(ax, 0.715, "(C)", "Action + responsibility")

    fields = ["Proposed use", "Decision-time information", "Comparator",
              "Evidence boundary", "Consequence / endpoint", "Lifecycle horizon",
              "Source locator", "Decision owner", "Review trigger"]
    box(ax, 0.015, 0.330, 0.270, 0.550, "", "",
        facecolor=COLORS["record"], edgecolor=COLORS["study"])
    ax.text(0.031, 0.835, "Candidate route record", transform=ax.transAxes,
            fontsize=7.2, fontweight="bold", color=COLORS["ink"])
    for i, field in enumerate(fields):
        ax.text(0.033, 0.785 - i * 0.047, field, transform=ax.transAxes,
                fontsize=6.7, color=COLORS["ink"], va="center")
    ax.text(0.150, 0.285, "One proposed use = one route record.",
            transform=ax.transAxes, ha="center", fontsize=6.4, fontweight="bold")
    ax.text(0.150, 0.245,
            "Changed information, comparator,\nconsequence, or lifecycle\nrequires a new route declaration.",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.2,
            color=COLORS["muted"], linespacing=1.4)
    ax.text(0.150, 0.075,
            "Evidence links: requires · supports\n"
            "does not license · blocks deployment",
            transform=ax.transAxes, ha="center", fontsize=6.0, color=COLORS["muted"])
    arrow(ax, (0.290, 0.605), (0.325, 0.605))

    display = [
        ("R4  Withhold retention / transfer", "Unsupported persistence or transfer"),
        ("R2  Retain comparator", "Increment evaluated, not demonstrated"),
        ("R3  Require user study", "Proxy outcome for a consequential use"),
        ("R1  Limit interpretation and use", "Scope exceeds evaluated evidence"),
        ("R5  Bounded permission", "Matched direct evidence + control\nor reversibility; constraints cleared"),
    ]
    ys = [0.770, 0.635, 0.500, 0.365, 0.230]
    for row, y, (title, trigger) in zip(rules, ys, display):
        _, face, edge = ACTION_STYLE[row["action"]]
        box(ax, 0.340, y, 0.340, 0.105, title, trigger,
            facecolor=face, edgecolor=edge, title_size=6.35, body_size=6.0)
    for upper, lower in zip(ys[:-1], ys[1:]):
        arrow(ax, (0.510, upper), (0.510, lower + 0.105))
    ax.text(0.510, 0.160, "R4 → R2 → R3 → R1 → R5",
            transform=ax.transAxes, ha="center", fontsize=7.0, fontweight="bold")
    ax.text(0.510, 0.077,
            "First matching constraint wins; R5 last.\n"
            "No match → unresolved evidence request.",
            transform=ax.transAxes, ha="center", fontsize=6.0, color=COLORS["muted"])
    arrow(ax, (0.683, 0.605), (0.716, 0.605))

    action_order = list(ACTION_STYLE)[:5]
    for action, y in zip(action_order, [0.810, 0.728, 0.646, 0.564, 0.482]):
        label, face, edge = ACTION_STYLE[action]
        box(ax, 0.730, y, 0.245, 0.055, "", "",
            facecolor=face, edgecolor=edge)
        ax.text(0.8525, y + 0.0275, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=6.25, fontweight="bold", color=edge)
    box(ax, 0.730, 0.285, 0.245, 0.145, "Machine checks",
        "Schema · provenance structure\nRule conformance",
        facecolor=COLORS["panel"], title_size=6.8, body_size=6.0)
    box(ax, 0.730, 0.080, 0.245, 0.155, "Human responsibility",
        "Code and justify the evidence\nOwn action + review trigger\nRevisit unresolved judgments",
        facecolor=COLORS["panel"], title_size=6.8, body_size=6.0)
    ax.text(0.5, 0.006, "Machine-checked does not mean substantively validated.",
            transform=ax.transAxes, ha="center", fontsize=6.7,
            fontweight="bold", color=COLORS["muted"])
    save_figure(fig, "protocol_overview")

    rows = [{"panel": "A", "kind": "record_field", "id": field, "priority": "",
             "label": field, "value": "route declaration"} for field in fields]
    rows += [{"panel": "B", "kind": "rule", "id": row["id"],
              "priority": row["priority"], "label": row["title_en"],
              "value": row["action"]} for row in rules]
    rows += [{"panel": "C", "kind": "action", "id": action, "priority": "",
              "label": ACTION_STYLE[action][0],
              "value": "human-reviewed route-specific next step"} for action in action_order]
    write_csv(DATA_DIR / "source_data_protocol_overview.csv",
              ["panel", "kind", "id", "priority", "label", "value"], rows)


def figure_evaluation_design(protocol: dict[str, object]) -> None:
    """Two evaluation layers; explicit links only to the four frozen routes."""
    validation, transfer = protocol["validation"], protocol["external_transfer"]
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 5.55))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.010, top=0.985)
    ax.set_axis_off()
    ax.text(0.015, 0.985, "(A)  Protocol evaluation · RQ1",
            transform=ax.transAxes, fontsize=8.2, fontweight="bold", va="top")
    inputs = [
        (f"{validation['case_count']} frozen worked routes", "Four proposed profile uses"),
        (f"{transfer['case_count']} author-coded external routes",
         f"From {transfer['source_count']} published HCI papers"),
        ("Structural challenges + ablations", "Distinct expected responses"),
    ]
    for y, (title, body) in zip([0.825, 0.710, 0.595], inputs):
        box(ax, 0.020, y, 0.345, 0.087, title, body,
            facecolor=COLORS["record"], edgecolor=COLORS["study"],
            title_size=6.8, body_size=6.2)
        arrow(ax, (0.375, y + 0.043), (0.420, 0.755))
    box(ax, 0.435, 0.655, 0.200, 0.200, "Unchanged",
        "Schema + resolver\nDeclared rule contract",
        facecolor=COLORS["proceed_light"], edgecolor=COLORS["proceed"],
        title_size=7.3, body_size=6.7)
    arrow(ax, (0.640, 0.755), (0.680, 0.755))
    box(ax, 0.695, 0.630, 0.280, 0.250, "Supported claims",
        "Action discrimination\nCross-source representability\nStructural conformance\nRule materiality",
        facecolor=COLORS["panel"], title_size=7.0, body_size=6.4)
    ax.plot([0.015, 0.985], [0.560, 0.560], transform=ax.transAxes,
            color=COLORS["grid"], lw=0.8)
    ax.text(0.015, 0.537, "(B)  Worked-case evidence · RQ2 / RQ3",
            transform=ax.transAxes, fontsize=7.8, fontweight="bold", va="top")
    ax.text(0.575, 0.537, "(C)  Links to frozen records",
            transform=ax.transAxes, fontsize=7.8, fontweight="bold", va="top")
    ax.text(0.025, 0.477,
            "24 participants · 15 released videos\nJoystick reports · EEG/fNIRS · post-trial SAM",
            transform=ax.transAxes, fontsize=6.4, color=COLORS["muted"], linespacing=1.4)
    evidence = [
        ("Profile interpretation · RQ2", "Cross-axis / own–donor / cross-video\nEstimator + reference sensitivity"),
        ("Scalar sensing · RQ3", "q prediction vs video mean"),
        ("Signed calibration · RQ3", "Cross-video reference-proximal correction"),
        ("Lifecycle evidence gap", "No repeated-session / transfer evidence"),
    ]
    routes = ["Trace interpretation", "Optional profile sensing",
              "Signed calibration", "Retention / transfer"]
    rows = []
    for y, (title, body), route in zip([0.350, 0.255, 0.160, 0.065], evidence, routes):
        box(ax, 0.025, y, 0.475, 0.076, title, body,
            facecolor=COLORS["proceed_light"], edgecolor=COLORS["proceed"],
            title_size=6.4, body_size=6.0)
        arrow(ax, (0.505, y + 0.038), (0.563, y + 0.038), color=COLORS["proceed"])
        box(ax, 0.577, y, 0.397, 0.076, route, "Existing frozen route record",
            facecolor=COLORS["record"], edgecolor=COLORS["study"],
            title_size=6.8, body_size=6.0)
        rows.append({"panel": "C", "evaluation_lane": title, "unit": route,
                     "probe": body.replace("\n", "; "), "supports": "formal evidence link",
                     "does_not_establish": "independent reuse or user benefit"})
    # A separate strip has no connector to the route cards or resolver.
    # It is placed outside the formal four-link lane to avoid evidence borrowing.
    fig.text(0.5, 0.026, "Separate dense analyses · RQ3 (no added resolved routes)\n"
             "Metadata / content prior + causal residual: participant × released-video holdout\n"
             "Post-trial SAM recovery: participant holdout, same-video training library",
             ha="center", va="bottom", fontsize=6.1, color=COLORS["gray"],
             bbox={"boxstyle": "round,pad=0.6", "facecolor": COLORS["panel"],
                   "edgecolor": COLORS["grid"], "linewidth": 0.7})
    # Reserve enough space inside the fixed-width export for this footnote.
    fig.subplots_adjust(bottom=0.09)
    save_figure(fig, "evaluation_design")
    rows[:0] = [
        {"panel": "A", "evaluation_lane": title, "unit": body,
         "probe": "unchanged schema and resolver",
         "supports": "RQ1 representation and declared contract",
         "does_not_establish": "coding validity or human decision improvement"}
        for title, body in inputs
    ]
    rows.append({"panel": "B", "evaluation_lane": "additional dense-trajectory analyses",
                 "unit": "RQ3; metadata/content/causal residual/SAM",
                 "probe": "participant + released-video holdout for content/residual; "
                          "participant holdout, known video for SAM",
                 "supports": "information-specific offline increments",
                 "does_not_establish": "additional resolved audit routes"})
    write_csv(DATA_DIR / "source_data_evaluation_design.csv",
              ["panel", "evaluation_lane", "unit", "probe", "supports", "does_not_establish"], rows)


@with_manuscript_fonts
def figure_protocol_evaluation(protocol: dict[str, object]) -> None:
    """Count actions and ablation effects without repeating route rationales."""

    from collections import Counter

    external = protocol["external_case_results"]
    worked = protocol["worked_case_results"]
    action_order = list(ACTION_STYLE)[:5]
    counts = {
        "external": Counter(row["audited_action"] for row in external),
        "worked": Counter(row["audited_action"] for row in worked),
    }
    assert sum(counts["external"].values()) == len(external)
    assert sum(counts["worked"].values()) == len(worked)
    assert not (set(counts["external"]) | set(counts["worked"])) - set(action_order)
    ablation = {
        row["rule_id"].split("_", 1)[0]: row["cases_changed"]
        for row in protocol["rule_ablation"]
    }
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 4.80))
    ax_a = fig.add_axes([0.250, 0.515, 0.280, 0.375])
    ax_b = fig.add_axes([0.725, 0.515, 0.230, 0.375])
    fig.text(0.035, 0.950, "(A)  Actions by record set", fontsize=8.0,
             fontweight="bold", color=COLORS["ink"])
    fig.text(0.620, 0.950, "(B)  Single-rule deletion", fontsize=8.0,
             fontweight="bold", color=COLORS["ink"])
    for index, action in enumerate(action_order):
        for group, shift, face, edge in (
            ("external", -0.16, COLORS["study"], COLORS["study"]),
            ("worked", 0.16, "#FFFFFF", COLORS["comparator"]),
        ):
            value = counts[group][action]
            ax_a.barh(index + shift, value, height=0.27, color=face,
                      edgecolor=edge, linewidth=1.0,
                      label=group.capitalize() if index == 0 else None)
            ax_a.text(value + 0.10, index + shift, str(value), va="center",
                      fontsize=6.9, color=COLORS["ink"])
    ax_a.set_yticks(range(len(action_order)),
                   ["R5 · Permit personalization", "R1 · Bound interpretation",
                    "R2 · Retain comparator", "R3 · Require user study",
                    "R4 · Withhold reuse"], fontsize=6.5)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, max(max(c.values()) for c in counts.values()) + 0.7)
    ax_a.set_xticks(range(6))
    ax_a.set_xlabel("Number of routes", fontsize=6.7)
    rules = ["R1", "R2", "R3", "R4", "R5"]
    for index, rule in enumerate(rules):
        value = ablation[rule]
        ax_b.barh(index, value, height=0.47, color=COLORS["gray"])
        ax_b.text(value + 0.10, index, str(value), va="center",
                  fontsize=6.9, color=COLORS["ink"])
    ax_b.set_yticks(range(len(rules)), rules, fontsize=6.9)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, max(ablation.values()) + 0.7)
    ax_b.set_xticks(range(6))
    ax_b.set_xlabel("Actions changed", fontsize=6.7)
    for ax in (ax_a, ax_b):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(COLORS["grid"])
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=6.2, color=COLORS["grid"])
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=COLORS["grid"], lw=0.5)
    ax_a.legend(loc="lower left", bbox_to_anchor=(-0.25, 1.015),
                ncol=2, fontsize=6.3, frameon=False)
    fig.text(0.630, 0.405, f"Combined benchmark: {len(external) + len(worked)} records",
             fontsize=6.2, color=COLORS["muted"])
    fig.text(0.5, 0.025,
             f"Author-coded: {len(external)} external + {len(worked)} worked routes · "
             "descriptive counts, not prevalence or decision improvement",
             ha="center", fontsize=6.1, color=COLORS["muted"])
    validation = protocol["validation"]
    challenges = [
        ("Order permutations", "Preserve outputs", validation["order_permutations"],
         validation["order_permutations"] if validation["order_invariant"] else 0),
        ("Malformed records", "Reject input", validation["invalid_mutations"],
         validation["invalid_mutations_rejected"]),
        ("Unrelated-field changes", "Preserve focal action", validation["route_locality_probes"],
         validation["route_locality_probes_passed"]),
        ("Permission-boundary failures", "Remove permission", validation["permission_boundary_probes"],
         validation["permission_boundary_probes_passed"]),
        ("Precedence probes", "Return priority action", validation["precedence_probes"],
         validation["precedence_probes_passed"]),
    ]
    fig.text(0.035, 0.373, "(C)  Structural challenges: declared vs observed response",
             fontsize=8.0, fontweight="bold", color=COLORS["ink"])
    ax_c = fig.add_axes([0.035, 0.075, 0.930, 0.270])
    ax_c.set_axis_off()
    table = ax_c.table(
        cellText=[[name, expected, f"{passed}/{total}: {expected.lower()}"]
                  for name, expected, total, passed in challenges],
        colLabels=["Challenge", "Expected response", "Observed response"],
        colWidths=[0.37, 0.28, 0.35], cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["grid"])
        cell.set_linewidth(0.45)
        cell.set_facecolor(COLORS["record"] if row == 0 else
                           (COLORS["panel"] if row % 2 else "white"))
        if row == 0:
            cell.get_text().set_fontweight("bold")
    save_figure(fig, "protocol_evaluation")

    rows = []
    # Preserve every record in Source Data, including the aggregation mapping.
    for case_set, cases in (("external", external), ("worked", worked)):
        for row in cases:
            rows.append({"panel": "A", "kind": "route", "case_set": case_set,
                         "id": row["id"], "label": ACTION_STYLE[row["audited_action"]][0],
                         "rule": row["matched_rule_id"], "action": row["audited_action"],
                         "value": 1, "denominator": len(cases)})
        for action in action_order:
            rows.append({"panel": "A", "kind": "action_count", "case_set": case_set,
                         "id": action, "label": ACTION_STYLE[action][0], "rule": "",
                         "action": action, "value": counts[case_set][action],
                         "denominator": len(cases)})
    for rule, count in ablation.items():
        rows.append({"panel": "B", "kind": "rule_ablation", "case_set": "combined",
                     "id": rule, "label": "actions changed", "rule": rule,
                     "action": "", "value": count,
                     "denominator": len(external) + len(worked)})
    for name, expected, total, passed in challenges:
        rows.append({"panel": "C", "kind": "structural_challenge", "case_set": "challenge",
                     "id": name, "label": expected, "rule": "", "action": expected,
                     "value": passed, "denominator": total})
    write_csv(DATA_DIR / "source_data_protocol_evaluation.csv",
              ["panel", "kind", "case_set", "id", "label", "rule", "action",
               "value", "denominator"], rows)


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source.get("schema_version") != "revision6-publication-source-v1":
        raise RuntimeError("Stale Revision-6 publication source")
    protocol = source["summaries"]["protocol_replay"]
    figure_protocol_overview(protocol)
    figure_evaluation_design(protocol)
    figure_protocol_evaluation(protocol)
    print(
        "Generated protocol overview, evaluation design, and "
        "protocol evaluation figures"
    )


if __name__ == "__main__":
    main()
