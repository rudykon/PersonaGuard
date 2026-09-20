#!/usr/bin/env python3
"""Generate Revision-6 traceability and actionability figures in Python."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from vector_export import save_pdfua_embed
from manuscript_fonts import with_manuscript_fonts

from make_revision4_figures import (
    BLUE,
    DARK,
    FIGURE_DIR,
    GRAY,
    GREEN,
    LIGHT_BLUE,
    LIGHT_GRAY,
    LIGHT_GREEN,
    LIGHT_ORANGE,
    LIGHT_PURPLE,
    LIGHT_RED,
    ORANGE,
    PURPLE,
    RED,
    ROOT,
    box,
    objective_summary,
    panel_label,
    write_source_data,
)


PUBLICATION_SOURCE = ROOT / "results" / "revision6_source.json"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.25,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    }
)


def save_revision5_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    common = {} if stem == "estimator_reference_actionability" else {
        "bbox_inches": "tight", "pad_inches": 0.035
    }
    fig.savefig(FIGURE_DIR / f"{stem}.svg", **common)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", **common)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, **common)
    fig.savefig(FIGURE_DIR / f"{stem}.tiff", dpi=600, **common)
    save_pdfua_embed(fig, FIGURE_DIR / f"{stem}_embed.pdf", common)
    plt.close(fig)


EDGE_STYLE = {
    "requires": {"color": DARK, "linestyle": "-", "linewidth": 1.15},
    "supports": {"color": BLUE, "linestyle": "-", "linewidth": 0.95},
    "does_not_license": {
        "color": ORANGE,
        "linestyle": (0, (4, 2)),
        "linewidth": 0.95,
    },
    "blocks_deployment": {
        "color": RED,
        "linestyle": (0, (1.4, 1.6)),
        "linewidth": 1.15,
    },
}


def typed_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    edge_type: str,
) -> None:
    style = EDGE_STYLE[edge_type]
    path = mpl.path.Path(
        points,
        [mpl.path.Path.MOVETO]
        + [mpl.path.Path.LINETO] * (len(points) - 1),
    )
    ax.add_patch(
        mpl.patches.FancyArrowPatch(
            path=path,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=7.5,
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
            zorder=0,
        )
    )


def status_style(status: str) -> tuple[str, str, str]:
    if status == "AVAILABLE":
        return "AVAILABLE", LIGHT_GREEN, GREEN
    if status == "PARTIAL":
        return "PARTIAL", LIGHT_ORANGE, ORANGE
    if status == "NO_DEMONSTRATED_INCREMENT":
        return "NO DEMONSTRATED\nINCREMENT", LIGHT_RED, RED
    if status == "NOT_TESTED":
        return "NOT TESTED", LIGHT_GRAY, GRAY
    if status == "NOT_JUSTIFIED":
        return "NOT JUSTIFIED", LIGHT_RED, RED
    raise ValueError(f"Unsupported status: {status}")


def figure_measurement_evidence_traceability(
    revision4: dict,
    revision5: dict,
    revision6: dict,
    graph: dict,
) -> list[dict[str, object]]:
    fig, ax = plt.subplots(figsize=(7.17, 5.90))
    ax.set_axis_off()

    ax.text(
        0.006,
        0.985,
        "A",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.035,
        0.985,
        "Fold-safe method: released interaction traces to bounded estimands",
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=DARK,
        va="top",
    )
    method_boxes = [
        (
            0.025,
            0.190,
            "Gated release",
            ["24 participants · 15 videos", "2-D joystick · EEG · fNIRS", "360 participant–video trials"],
            LIGHT_BLUE,
            BLUE,
        ),
        (
            0.245,
            0.205,
            "Split first",
            ["Outer and inner user folds", "Held-out users never define", "their reference or target"],
            LIGHT_GRAY,
            GRAY,
        ),
        (
            0.480,
            0.225,
            "Reference + DTW",
            ["Training-only video reference", "Constrained estimator variants", "Axes/objectives/parameters", "audited"],
            LIGHT_PURPLE,
            PURPLE,
        ),
        (
            0.735,
            0.240,
            "Estimands + decisions",
            [r"$w(t)$ local · $b$ signed · $q$ absolute", "Meaning · shared-reference sensitivity", "Acquisition · utility · governance"],
            LIGHT_GREEN,
            GREEN,
        ),
    ]
    for x, width, title, lines, face, edge in method_boxes:
        box(
            ax,
            x,
            0.792,
            width,
            0.140,
            title,
            lines,
            facecolor=face,
            edgecolor=edge,
            body_size=5.0,
        )
    for left, right in zip(method_boxes[:-1], method_boxes[1:]):
        typed_arrow(
            ax,
            [(left[0] + left[1] + 0.003, 0.866), (right[0] - 0.004, 0.866)],
            "requires",
        )
    ax.text(
        0.5,
        0.767,
        "Reference construction, targets, context, scaling, and model selection remain inside the participant boundary.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.9,
        fontweight="bold",
        color=RED,
    )

    ax.text(
        0.006,
        0.718,
        "B",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.035,
        0.718,
        "Typed evidence traceability graph: route evidence is local and status never propagates automatically",
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=DARK,
        va="top",
    )

    legend_handles = [
        mpl.lines.Line2D(
            [], [], color=spec["color"], linestyle=spec["linestyle"],
            linewidth=spec["linewidth"], label=edge_type.replace("_", " ")
        )
        for edge_type, spec in EDGE_STYLE.items()
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.684),
        ncol=4,
        fontsize=5.6,
        handlelength=2.4,
        columnspacing=1.4,
        borderaxespad=0,
    )

    nodes = {node["id"]: node for node in graph["nodes"]}

    def draw_node(
        node_id: str,
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        lines: list[str],
        facecolor: str,
        edgecolor: str,
        *,
        title_size: float = 6.2,
        body_size: float = 5.0,
    ) -> None:
        status, status_face, status_color = status_style(nodes[node_id]["status"])
        box(
            ax,
            x,
            y,
            width,
            height,
            title,
            lines,
            facecolor=facecolor,
            edgecolor=edgecolor,
            title_size=title_size,
            body_size=body_size,
            status=status,
            status_facecolor=status_face,
            status_color=status_color,
        )

    draw_node(
        "measurement_meaning",
        0.025,
        0.495,
        0.195,
        0.100,
        "Measurement meaning",
        ["Interface-dependent relation", "Small cross-video residual"],
        LIGHT_BLUE,
        BLUE,
    )
    draw_node(
        "estimator_robustness",
        0.025,
        0.350,
        0.195,
        0.100,
        "Estimator robustness",
        ["Five estimands", "Four DTW objectives"],
        LIGHT_PURPLE,
        PURPLE,
    )
    draw_node(
        "empirical_library_resampling_stability",
        0.025,
        0.205,
        0.195,
        0.100,
        "Shared-reference sensitivity",
        ["15 released support points", "Resampling + reference uncertainty"],
        LIGHT_PURPLE,
        PURPLE,
        title_size=5.9,
    )
    draw_node(
        "session_persistence",
        0.025,
        0.065,
        0.195,
        0.090,
        "Session persistence",
        ["No repeat acquisition"],
        LIGHT_GRAY,
        GRAY,
    )
    draw_node(
        "trace_derived_estimation",
        0.300,
        0.495,
        0.190,
        0.100,
        "Trace-derived estimation",
        [r"Compute $w(t)$, $b$, $q$", "after a completed report"],
        LIGHT_BLUE,
        BLUE,
        title_size=5.9,
    )
    draw_node(
        "absolute_q_calibration",
        0.300,
        0.310,
        0.190,
        0.100,
        r"Absolute $q$ calibration",
        ["Predict held-out magnitude", "1, 2, 4, or 8 videos"],
        LIGHT_ORANGE,
        ORANGE,
    )
    draw_node(
        "eeg_fnirs_increment",
        0.300,
        0.125,
        0.190,
        0.100,
        "EEG/fNIRS increment",
        ["Predict without target joystick", "Evaluated offline pipeline"],
        LIGHT_GRAY,
        GRAY,
    )
    draw_node(
        "signed_proximal_correction",
        0.570,
        0.485,
        0.180,
        0.105,
        "Signed proximal correction",
        [r"Constant $b$ shift", "Reference-trace MAE"],
        LIGHT_ORANGE,
        ORANGE,
        title_size=5.8,
    )
    draw_node(
        "downstream_user_utility",
        0.570,
        0.215,
        0.180,
        0.105,
        "Downstream utility",
        ["Benefit · harm · workload", "Meaningful-gain threshold"],
        LIGHT_GRAY,
        GRAY,
    )
    draw_node(
        "reference_representativeness_equity",
        0.820,
        0.165,
        0.155,
        0.105,
        "Reference equity",
        ["Coverage · subgroup error", "Transportability"],
        LIGHT_GRAY,
        GRAY,
        title_size=5.8,
        body_size=4.9,
    )
    draw_node(
        "retention_and_transfer",
        0.820,
        0.345,
        0.155,
        0.105,
        "Retention + transfer",
        ["Retest · interface transfer", "Privacy · access"],
        LIGHT_PURPLE,
        PURPLE,
        title_size=5.8,
        body_size=5.0,
    )

    edge_routes = [
        ("measurement_meaning", "trace_derived_estimation", "supports", [(0.220, 0.545), (0.300, 0.545)]),
        ("measurement_meaning", "absolute_q_calibration", "supports", [(0.220, 0.525), (0.260, 0.525), (0.260, 0.360), (0.300, 0.360)]),
        ("measurement_meaning", "signed_proximal_correction", "supports", [(0.220, 0.570), (0.535, 0.615), (0.570, 0.550)]),
        ("estimator_robustness", "empirical_library_resampling_stability", "requires", [(0.122, 0.355), (0.122, 0.300)]),
        ("empirical_library_resampling_stability", "absolute_q_calibration", "supports", [(0.220, 0.255), (0.270, 0.255), (0.270, 0.345), (0.300, 0.345)]),
        ("empirical_library_resampling_stability", "eeg_fnirs_increment", "supports", [(0.220, 0.235), (0.260, 0.235), (0.260, 0.175), (0.300, 0.175)]),
        ("trace_derived_estimation", "signed_proximal_correction", "requires", [(0.490, 0.545), (0.570, 0.545)]),
        ("signed_proximal_correction", "downstream_user_utility", "does_not_license", [(0.660, 0.490), (0.660, 0.315)]),
        ("absolute_q_calibration", "downstream_user_utility", "does_not_license", [(0.490, 0.360), (0.530, 0.360), (0.530, 0.268), (0.570, 0.268)]),
        ("eeg_fnirs_increment", "trace_derived_estimation", "does_not_license", [(0.490, 0.175), (0.525, 0.175), (0.525, 0.635), (0.285, 0.635), (0.285, 0.545), (0.300, 0.545)]),
        ("session_persistence", "retention_and_transfer", "blocks_deployment", [(0.220, 0.110), (0.790, 0.035), (0.820, 0.365)]),
        ("downstream_user_utility", "retention_and_transfer", "blocks_deployment", [(0.750, 0.268), (0.790, 0.268), (0.790, 0.385), (0.820, 0.385)]),
        ("reference_representativeness_equity", "retention_and_transfer", "blocks_deployment", [(0.898, 0.270), (0.898, 0.345)]),
        ("signed_proximal_correction", "retention_and_transfer", "does_not_license", [(0.750, 0.550), (0.790, 0.550), (0.790, 0.415), (0.820, 0.415)]),
    ]
    for source, target, edge_type, points in edge_routes:
        typed_arrow(ax, points, edge_type)

    ax.text(
        0.500,
        0.012,
        "Current decision: no durable retention, added sensing, or consequential personalization.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.9,
        fontweight="bold",
        color=PURPLE,
    )

    save_revision5_figure(fig, "measurement_evidence_dag_full")

    rows: list[dict[str, object]] = []
    for node in graph["nodes"]:
        rows.append(
            {
                "kind": "node",
                "node": node["id"],
                "status": node["status"],
                "estimand": node["estimand"],
            }
        )
    for edge in graph["edges"]:
        rows.append({"kind": "edge", **edge})

    cluster = revision5["full_pipeline_reference_bootstrap"]["protocols"][
        "cluster_exclusion"
    ]
    consensus_primary = revision4["estimand_robustness"]["consensus"]
    rows.extend(
        [
            {
                "kind": "result",
                "node": "empirical_library_resampling_stability",
                "metric": "descriptive_shared_reference_relative_fraction_with_resampling_interval",
                "value": consensus_primary["relative_g_15"],
                "ci95_low": cluster["relative_g_15"]["percentile_ci95"][0],
                "ci95_high": cluster["relative_g_15"]["percentile_ci95"][1],
            },
            {
                "kind": "result",
                "node": "empirical_library_resampling_stability",
                "metric": "descriptive_shared_reference_absolute_fraction_with_resampling_interval",
                "value": consensus_primary["absolute_phi_15"],
                "ci95_low": cluster["absolute_phi_15"]["percentile_ci95"][0],
                "ci95_high": cluster["absolute_phi_15"]["percentile_ci95"][1],
            },
            {
                "kind": "result",
                "node": "eeg_fnirs_increment",
                "metric": "blockwise_gain_over_video_mean_seconds",
                "value": revision6["primary_antialias_reservation_sensing"][
                    "blockwise_stack_gain_vs_video_mean_seconds"
                ],
            },
        ]
    )
    for budget, values in revision5["signed_calibration_practical_value"][
        "budgets"
    ].items():
        rows.append(
            {
                "kind": "result",
                "node": "signed_proximal_correction",
                "budget_videos": int(budget),
                "metric": "calibrated_gain_vs_video_units",
                "value": values["calibrated_gain_vs_video_units"],
                "ci95_low": values[
                    "calibrated_gain_vs_video_participant_bootstrap_percentile_ci95"
                ][0],
                "ci95_high": values[
                    "calibrated_gain_vs_video_participant_bootstrap_percentile_ci95"
                ][1],
                "relative_gain_percent": values[
                    "calibrated_gain_vs_video_percent_of_video_mae"
                ],
                "annotation_minutes": values["continuous_annotation_burden"][
                    "expected_minutes"
                ],
            }
        )
    write_source_data("measurement_evidence_dag_full", rows)
    return rows


def figure_protocol_replay_overview(protocol: dict[str, object]) -> list[dict[str, object]]:
    """Create the compressed main-paper protocol figure from replay results."""

    fig, ax = plt.subplots(figsize=(7.17, 4.15))
    ax.set_axis_off()
    ax.text(0.006, 0.985, "A", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")
    ax.text(
        0.038,
        0.985,
        "Evidence-traceability audit protocol",
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=DARK,
        va="top",
    )
    stages = [
        (0.025, "1 · Declare claim + use", ["Estimand and proposed action", "Deployment-time inputs"], LIGHT_BLUE, BLUE),
        (0.270, "2 · Record evidence", ["Admissible evidence + boundary", "Burden and unsupported claims"], LIGHT_PURPLE, PURPLE),
        (0.515, "3 · Check structure", ["Schema · hashes · typed edges", "Explicit propagation rules"], LIGHT_GRAY, GRAY),
        (0.760, "4 · Make route action", ["Use · baseline · study", "or withhold retention"], LIGHT_GREEN, GREEN),
    ]
    for x, title, lines, face, edge in stages:
        box(
            ax,
            x,
            0.710,
            0.215,
            0.185,
            title,
            lines,
            facecolor=face,
            edgecolor=edge,
            title_size=6.1,
            body_size=5.2,
        )
    for left, right in zip(stages[:-1], stages[1:]):
        typed_arrow(ax, [(left[0] + 0.217, 0.802), (right[0] - 0.005, 0.802)], "requires")

    ax.text(0.006, 0.615, "B", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")
    ax.text(
        0.038,
        0.615,
        "Naive shortcut → audited route decision",
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=DARK,
        va="top",
    )
    naive = {
        "trace_interpretation": "Stable person profile",
        "optional_sensing": "Add EEG/fNIRS sensors",
        "signed_calibration": "Deploy calibrated correction",
        "retention_transfer": "Retain and transfer profile",
    }
    audited = {
        "trace_interpretation": ("Bound to interface; offline trace only", LIGHT_BLUE, BLUE),
        "optional_sensing": ("Retain no-sensor baseline", LIGHT_GREEN, GREEN),
        "signed_calibration": ("Preregistered user study; no deployment", LIGHT_ORANGE, ORANGE),
        "retention_transfer": ("Withhold retention and transfer", LIGHT_RED, RED),
    }
    labels = {
        "trace_interpretation": "Trace interpretation",
        "optional_sensing": "Optional sensing",
        "signed_calibration": "Signed calibration",
        "retention_transfer": "Retention + transfer",
    }
    order = [
        "trace_interpretation",
        "optional_sensing",
        "signed_calibration",
        "retention_transfer",
    ]
    result_by_id = {row["id"]: row for row in protocol["case_results"]}
    y_positions = [0.455, 0.325, 0.195, 0.065]
    for case_id, y in zip(order, y_positions):
        result = result_by_id[case_id]
        action_text, action_face, action_edge = audited[case_id]
        ax.text(0.025, y + 0.050, labels[case_id], transform=ax.transAxes,
                fontsize=5.7, fontweight="bold", color=DARK, va="center")
        box(
            ax, 0.185, y, 0.275, 0.100, "Naive", [naive[case_id]],
            facecolor=LIGHT_GRAY, edgecolor=GRAY, title_size=5.1, body_size=5.0,
        )
        typed_arrow(ax, [(0.465, y + 0.050), (0.535, y + 0.050)], "does_not_license")
        box(
            ax, 0.540, y, 0.430, 0.100, "Audited", [action_text],
            facecolor=action_face, edgecolor=action_edge, title_size=5.1, body_size=5.0,
        )
        if not result["decision_changed"]:
            raise RuntimeError(f"Expected a changed decision for {case_id}")

    validation = protocol["validation"]
    ax.text(
        0.5,
        0.015,
        f"Structural replay: {validation['case_count']} cases · "
        f"{validation['order_permutations']} order permutations · "
        f"{validation['invalid_mutations_rejected']}/{validation['invalid_mutations']} malformed records rejected · "
        "substantive validity not machine-checked",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.4,
        color=GRAY,
    )
    save_revision5_figure(fig, "measurement_evidence_dag")
    rows = [{"panel": "B", **row} for row in protocol["case_results"]]
    rows.append({"panel": "A", "kind": "validation", **validation})
    write_source_data("measurement_evidence_dag", rows)
    return rows


@with_manuscript_fonts
def figure_estimator_reference_actionability(
    revision4: dict,
    revision5: dict,
    robustness: list[dict[str, str]],
) -> list[dict[str, object]]:
    objective_rows = objective_summary(revision4)
    cluster = revision5["full_pipeline_reference_bootstrap"]["protocols"][
        "cluster_exclusion"
    ]
    category_point = revision4["category_gstudy"]["balanced_dstudy"]["15"]
    calibration = revision5["signed_calibration_practical_value"]["budgets"]

    # Four evidence questions; unlike metrics use separate axes, never dual y axes.
    fig = plt.figure(figsize=(182.1 / 25.4, 5.65))
    ax_a = fig.add_axes([0.195, 0.625, 0.285, 0.245])
    ax_b = fig.add_axes([0.710, 0.625, 0.105, 0.245])
    ax_b_rank = fig.add_axes([0.865, 0.625, 0.115, 0.245])
    ax_c = fig.add_axes([0.180, 0.165, 0.300, 0.275])
    ax_d = fig.add_axes([0.685, 0.300, 0.295, 0.140])
    ax_exposure = fig.add_axes([0.685, 0.150, 0.295, 0.085])
    headings = [(0.025, 0.945, "(A)  Estimand sensitivity"),
                (0.550, 0.945, "(B)  DTW objective sensitivity"),
                (0.025, 0.510, "(C)  Reference sensitivity"),
                (0.550, 0.510, "(D)  Cross-video signed calibration")]
    for x, y, heading in headings:
        fig.text(x, y, heading, fontsize=8.0, fontweight="bold", color=DARK)
    estimand_order = ["consensus", "axis_mean", "multivariate", "valence_only", "arousal_only"]
    labels_a = ["Consensus map", "Axis-mean lag", "Multivariate DTW", "Valence only", "Arousal only"]
    by_name = {row["estimand"]: row for row in robustness}
    if set(by_name) != set(estimand_order):
        raise ValueError("Every frozen estimand must remain represented")
    y = np.arange(len(estimand_order))[::-1]
    ranks = [float(by_name[name]["profile_spearman_vs_consensus"]) for name in estimand_order]
    ax_a.scatter(ranks, y, marker="D", s=20, color=PURPLE)
    for rho, yy in zip(ranks, y):
        ax_a.annotate(f"{rho:.3f}", (rho, yy), xytext=(-6, -10),
                      textcoords="offset points", ha="right", fontsize=6.2, color=GRAY)
    ax_a.set_yticks(y, labels_a)
    ax_a.set_xlim(0.85, 1.015)
    ax_a.set_ylim(-0.6, 4.5)
    ax_a.set_xticks([0.85, 0.90, 0.95, 1.0])
    ax_a.set_xlabel("Profile rank correlation\nwith consensus", fontsize=6.7)

    names_b = ["Direct step", "Mixed norm.", r"$\lambda_s=0$", "Path mean"]
    ordered = {row["objective"]: row for row in objective_rows}
    objectives = [ordered[name] for name in ["direct_step", "mixed_step_mean", "step_zero", "path_mean"]]
    for yy, row in zip(range(3, -1, -1), objectives):
        ax_b.plot(float(row["path_ratio"]), yy, "o", color=BLUE, markersize=4)
        ax_b_rank.plot(float(row["profile_rho"]), yy, "D", color=PURPLE, markersize=3.8)
    ax_b.set_yticks(range(3, -1, -1), names_b)
    ax_b_rank.set_yticks([])
    ax_b.set_xlim(1.05, 2.0)
    ax_b.set_xticks([1.1, 1.5, 1.9])
    ax_b_rank.set_xlim(0.40, 1.06)
    ax_b_rank.set_xticks([0.5, 1.0])
    for ax in (ax_b, ax_b_rank):
        ax.set_ylim(-0.5, 3.5)
    ax_b.set_xlabel("Mean path /\ntrial nodes", fontsize=6.4)
    ax_b_rank.set_xlabel("Profile rank\nvs direct step", fontsize=6.4)

    consensus = by_name["consensus"]
    rows_c = [
        ("Original",
         float(consensus["relative_g_15"]),
         [float(consensus["relative_g_15_ci95_low"]), float(consensus["relative_g_15_ci95_high"])],
         float(consensus["absolute_phi_15"]),
         [float(consensus["absolute_phi_15_ci95_low"]), float(consensus["absolute_phi_15_ci95_high"])]),
        ("Identity-excluding",
         float(consensus["relative_g_15"]), cluster["relative_g_15"]["percentile_ci95"],
         float(consensus["absolute_phi_15"]), cluster["absolute_phi_15"]["percentile_ci95"]),
        ("Category facet\n+ identity-excluding",
         float(category_point["relative_g"]), cluster["category_relative_g_15"]["percentile_ci95"],
         float(category_point["absolute_phi"]), cluster["category_absolute_phi_15"]["percentile_ci95"]),
    ]
    for yy, (_, g, g_ci, phi, phi_ci) in zip([2, 1, 0], rows_c):
        ax_c.plot(g_ci, [yy + .12] * 2, color=BLUE, linewidth=1.2)
        ax_c.plot(g, yy + .12, "o", color=BLUE, markersize=4.0)
        ax_c.plot(phi_ci, [yy - .12] * 2, color=ORANGE, linewidth=1.2)
        ax_c.plot(phi, yy - .12, "s", color=ORANGE, markersize=3.7)
    ax_c.set_yticks([2, 1, 0], [row[0] for row in rows_c])
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(-0.6, 2.5)
    ax_c.set_xticks([0, .25, .5, .75, 1])
    ax_c.set_xlabel("Descriptive shared-reference fraction", fontsize=6.4)
    ax_c.legend(handles=[
        mpl.lines.Line2D([], [], color=BLUE, marker="o", label="Relative"),
        mpl.lines.Line2D([], [], color=ORANGE, marker="s", label="Absolute")],
        loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2, fontsize=6.2)

    budgets = np.asarray([1, 2, 4, 8], dtype=float)
    gains = np.asarray([calibration[str(int(b))]["calibrated_gain_vs_video_units"] for b in budgets])
    intervals = np.asarray([calibration[str(int(b))][
        "calibrated_gain_vs_video_participant_bootstrap_percentile_ci95"] for b in budgets])
    exposure = np.asarray([calibration[str(int(b))]["continuous_annotation_burden"][
        "expected_minutes"] for b in budgets])
    improved = [calibration[str(int(b))]["participants_improved_vs_video"] for b in budgets]
    ax_d.errorbar(budgets, gains,
        yerr=np.vstack([gains - intervals[:, 0], intervals[:, 1] - gains]),
        color=PURPLE, marker="o", markersize=4, capsize=2.5, linewidth=1.2)
    ax_d.axhline(0, color=GRAY, linewidth=.7)
    ax_d.set_ylim(-.01, .12)
    ax_d.set_yticks([0, .05, .10])
    ax_d.set_xticks(budgets, [f"{n}\n/24" for n in improved])
    ax_d.set_ylabel("Gain\n(label units)", fontsize=6.4)
    ax_d.text(.5, 1.05, "Positive participant means shown below",
              transform=ax_d.transAxes, ha="center", fontsize=6.0, color=GRAY)
    ax_exposure.plot(budgets, exposure, color=GRAY, marker="s", linestyle="--",
                     markersize=3.8, linewidth=1.0)
    for b, minutes in zip(budgets, exposure):
        ax_exposure.annotate(f"{minutes:.1f}", (b, minutes), xytext=(0, 5),
                             textcoords="offset points", ha="center", fontsize=6.0)
    ax_exposure.set_ylim(0, 19)
    ax_exposure.set_yticks([0, 15])
    ax_exposure.set_xticks(budgets, [str(int(b)) for b in budgets])
    ax_exposure.set_xlabel("Calibration videos", fontsize=6.7)
    ax_exposure.set_ylabel("Exposure\n(min)", fontsize=6.4)
    for ax in (ax_d, ax_exposure):
        ax.set_xlim(.5, 8.5)
    for ax in (ax_a, ax_b, ax_b_rank, ax_c, ax_d, ax_exposure):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.3)
        ax.grid(axis="y" if ax in (ax_d, ax_exposure) else "x",
                color="#D8DDE3", linewidth=.45)
        ax.set_axisbelow(True)
    fig.text(.5, .028,
             "Reference-relative, interface-contingent, within-session evidence—not a stable trait.\n"
             "Intervals are resampling diagnostics; exposure is estimated, not measured burden.",
             ha="center", fontsize=6.5, color=GRAY, linespacing=1.4)

    save_revision5_figure(fig, "estimator_reference_actionability")

    source_rows: list[dict[str, object]] = []
    for row in robustness:
        source_rows.append({"panel": "A", **row})
    for row in objective_rows:
        source_rows.append({"panel": "B", **row})
    for label, g, g_ci, phi, phi_ci in rows_c:
        source_rows.extend(
            [
                {
                    "panel": "C",
                    "analysis": label,
                    "metric": "descriptive_shared_reference_relative_fraction",
                    "estimate": g,
                    "ci95_low": g_ci[0],
                    "ci95_high": g_ci[1],
                },
                {
                    "panel": "C",
                    "analysis": label,
                    "metric": "descriptive_shared_reference_absolute_fraction",
                    "estimate": phi,
                    "ci95_low": phi_ci[0],
                    "ci95_high": phi_ci[1],
                },
            ]
        )
    for budget in budgets.astype(int):
        values = calibration[str(budget)]
        source_rows.append(
            {
                "panel": "D",
                "budget_videos": budget,
                "comparison": "calibrated_vs_video",
                "estimate": values["calibrated_gain_vs_video_units"],
                "ci95_low": values[
                    "calibrated_gain_vs_video_participant_bootstrap_percentile_ci95"
                ][0],
                "ci95_high": values[
                    "calibrated_gain_vs_video_participant_bootstrap_percentile_ci95"
                ][1],
                "video_only_mae_units": values["video_only_trace_mae_units"],
                "calibrated_mae_units": values["calibrated_trace_mae_units"],
                "relative_gain_percent": values[
                    "calibrated_gain_vs_video_percent_of_video_mae"
                ],
                "oracle_gap_units": values["calibrated_oracle_gap_units"],
                "participants_improved": values["participants_improved_vs_video"],
                "participants_degraded": values["participants_degraded_vs_video"],
                "worst_participant_gain_units": values[
                    "worst_participant_gain_vs_video_units"
                ],
                "expected_annotation_minutes": values[
                    "continuous_annotation_burden"
                ]["expected_minutes"],
            }
        )
    write_source_data("estimator_reference_actionability", source_rows)
    return source_rows


def main() -> None:
    if not PUBLICATION_SOURCE.exists():
        raise FileNotFoundError(
            "Missing results/revision6_source.json; build the canonical source first"
        )
    source = json.loads(PUBLICATION_SOURCE.read_text(encoding="utf-8"))
    if source.get("schema_version") != "revision6-publication-source-v1":
        raise RuntimeError("Stale Revision-6 publication source")
    revision4 = source["summaries"]["revision4"]
    revision5 = source["summaries"]["revision5"]
    revision6 = source["summaries"]["revision6"]
    graph = source["summaries"]["evidence_traceability"]
    robustness = source["tables"]["estimand_robustness"]
    from make_supplementary_figures import figure_graph
    figure_graph(source)
    figure_estimator_reference_actionability(revision4, revision5, robustness)
    print("Generated Revision-6 figures and source-data CSVs.")


if __name__ == "__main__":
    main()
