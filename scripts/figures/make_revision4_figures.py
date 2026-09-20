#!/usr/bin/env python3
"""Generate the revision-4 method/evidence figures from formal artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "artifacts" / "figures"
DATA_DIR = Path(__file__).resolve().parent
REVISION4 = ROOT / "artifacts" / "revision4"

BLUE = "#2C7FB8"
ORANGE = "#D95F0E"
GREEN = "#238B45"
PURPLE = "#756BB1"
RED = "#B33A3A"
DARK = "#263238"
GRAY = "#667085"
LIGHT_GRAY = "#EEF1F4"
LIGHT_BLUE = "#DDEEF8"
LIGHT_ORANGE = "#FBE5D6"
LIGHT_GREEN = "#DDF2E4"
LIGHT_PURPLE = "#E9E4F5"
LIGHT_RED = "#F7DFDF"

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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    common = {"bbox_inches": "tight", "pad_inches": 0.035}
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", **common)
    fig.savefig(FIGURE_DIR / f"{stem}.svg", **common)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, **common)
    fig.savefig(FIGURE_DIR / f"{stem}.tiff", dpi=600, **common)
    plt.close(fig)


def write_source_data(stem: str, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with (DATA_DIR / f"source_data_{stem}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.105,
        1.125,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: list[str],
    *,
    facecolor: str,
    edgecolor: str,
    title_size: float = 6.6,
    body_size: float = 5.25,
    status: str | None = None,
    status_facecolor: str = LIGHT_GRAY,
    status_color: str = DARK,
) -> None:
    patch = mpl.patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.85,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.012,
        y + height - 0.025,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=DARK,
    )
    ax.text(
        x + 0.012,
        y + height - 0.068,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=body_size,
        color=DARK,
        linespacing=1.27,
    )
    if status is not None:
        ax.text(
            x + width - 0.009,
            y + height + 0.004,
            status,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.0,
            fontweight="bold",
            color=status_color,
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": status_facecolor,
                "edgecolor": "none",
            },
        )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GRAY,
    linewidth: float = 0.9,
    style: str = "-|>",
) -> None:
    ax.add_patch(
        mpl.patches.FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle=style,
            mutation_scale=8.0,
            linewidth=linewidth,
            color=color,
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
        )
    )


def routed_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    color: str = GRAY,
    linewidth: float = 0.9,
    style: str = "-|>",
) -> None:
    path = mpl.path.Path(
        points,
        [mpl.path.Path.MOVETO]
        + [mpl.path.Path.LINETO] * (len(points) - 1),
    )
    ax.add_patch(
        mpl.patches.FancyArrowPatch(
            path=path,
            transform=ax.transAxes,
            arrowstyle=style,
            mutation_scale=8.0,
            linewidth=linewidth,
            color=color,
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
        )
    )


def figure_measurement_evidence_dag(summary: dict) -> list[dict[str, object]]:
    fig, ax = plt.subplots(figsize=(7.17, 5.05))
    ax.set_axis_off()

    ax.text(0.006, 0.985, "A", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    ax.text(
        0.035,
        0.985,
        "Fold-safe method: released interaction traces to bounded profile estimands",
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=DARK,
        va="top",
    )
    method_boxes = [
        (0.025, 0.190, "Gated release", ["24 participants · 15 videos", "2-D joystick · EEG · fNIRS", "360 participant–video trials"], LIGHT_BLUE, BLUE),
        (0.245, 0.205, "Split first", ["Outer and inner user folds", "Held-out users never define", "their reference or target"], LIGHT_GRAY, GRAY),
        (0.480, 0.225, "Reference + DTW", ["Training-only video reference", "Constrained alternative estimators", "Axes and objectives audited"], LIGHT_PURPLE, PURPLE),
        (0.735, 0.240, "Objects + decisions", [r"$w(t)$ local · $b$ signed · $q$ absolute", r"Meaning · $G/\Phi$ · acquisition", "Trace correction · governance"], LIGHT_GREEN, GREEN),
    ]
    for x, width, title, lines, face, edge in method_boxes:
        box(ax, x, 0.786, width, 0.142, title, lines, facecolor=face, edgecolor=edge)
    for left, right in zip(method_boxes[:-1], method_boxes[1:]):
        arrow(ax, (left[0] + left[1] + 0.003, 0.857), (right[0] - 0.004, 0.857))
    ax.text(
        0.5,
        0.757,
        "The reference population, target, context, scaling, and model selection are all inside the participant boundary.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.9,
        fontweight="bold",
        color=RED,
    )

    ax.text(0.006, 0.708, "B", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    ax.text(
        0.035,
        0.708,
        "Evidence DAG for a durable HCI profile: acquisition routes are parallel, not validity levels",
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=DARK,
        va="top",
    )

    box(
        ax,
        0.055,
        0.500,
        0.205,
        0.125,
        "Measurement meaning",
        ["Participant-specific structure", "Estimator + interface boundary"],
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
        status="PARTIAL",
        status_facecolor=LIGHT_ORANGE,
        status_color=ORANGE,
    )
    box(
        ax,
        0.055,
        0.315,
        0.205,
        0.125,
        "Generalizability",
        ["Videos + emotion categories", "Reference uncertainty; no retest"],
        facecolor=LIGHT_PURPLE,
        edgecolor=PURPLE,
        status="PARTIAL",
        status_facecolor=LIGHT_ORANGE,
        status_color=ORANGE,
    )
    box(
        ax,
        0.330,
        0.540,
        0.185,
        0.104,
        "Direct joystick",
        ["Observe $b$, $q$, and $w(t)$"],
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
        status="AVAILABLE",
        status_facecolor=LIGHT_GREEN,
        status_color=GREEN,
    )
    box(
        ax,
        0.330,
        0.390,
        0.185,
        0.104,
        "Behavioral calibration",
        ["Estimate held-out $q$ or $b$", "1, 2, 4, or 8 videos"],
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
        status="PROXIMAL SUPPORT",
        status_facecolor=LIGHT_GREEN,
        status_color=GREEN,
    )
    box(
        ax,
        0.330,
        0.240,
        0.185,
        0.104,
        "EEG/fNIRS sensing",
        ["No joystick at use time", "Increment over no sensor"],
        facecolor=LIGHT_GRAY,
        edgecolor=GRAY,
        status="NO INCREMENT",
        status_facecolor=LIGHT_RED,
        status_color=RED,
    )
    box(
        ax,
        0.590,
        0.450,
        0.185,
        0.118,
        "Proximal action test",
        ["Held-out signed shift", "Reference-trace MAE"],
        facecolor=LIGHT_GREEN,
        edgecolor=GREEN,
        status="SUPPORTED",
        status_facecolor=LIGHT_GREEN,
        status_color=GREEN,
    )
    box(
        ax,
        0.590,
        0.270,
        0.185,
        0.118,
        "Downstream utility",
        ["Interaction benefit or harm", "Requires prospective user study"],
        facecolor=LIGHT_GRAY,
        edgecolor=GRAY,
        status="NOT TESTED",
        status_facecolor=LIGHT_RED,
        status_color=RED,
    )
    box(
        ax,
        0.825,
        0.405,
        0.145,
        0.125,
        "Retention + governance",
        ["Retest · transfer", "Privacy + access limits"],
        facecolor=LIGHT_PURPLE,
        edgecolor=PURPLE,
        status="NOT LICENSED",
        status_facecolor=LIGHT_RED,
        status_color=RED,
        title_size=6.1,
        body_size=5.0,
    )

    arrow(ax, (0.260, 0.562), (0.330, 0.592))
    arrow(ax, (0.260, 0.562), (0.330, 0.442))
    arrow(ax, (0.260, 0.377), (0.330, 0.292))
    arrow(ax, (0.260, 0.377), (0.330, 0.442))
    arrow(ax, (0.515, 0.592), (0.590, 0.510))
    arrow(ax, (0.515, 0.442), (0.590, 0.510))
    arrow(ax, (0.515, 0.292), (0.590, 0.329))
    arrow(ax, (0.682, 0.450), (0.682, 0.388))
    arrow(ax, (0.775, 0.329), (0.825, 0.455))
    arrow(ax, (0.775, 0.510), (0.825, 0.455))
    routed_arrow(
        ax,
        [(0.260, 0.350), (0.285, 0.205), (0.800, 0.205), (0.825, 0.425)],
        color=PURPLE,
        linewidth=0.75,
        style="-|>",
    )

    ax.text(
        0.500,
        0.165,
        "Current permitted claim",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=DARK,
    )
    ax.text(
        0.500,
        0.105,
        "Candidate within-session reference-relative profile; estimator-dependent and interface-contingent",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=PURPLE,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": LIGHT_PURPLE,
            "edgecolor": PURPLE,
            "linewidth": 0.85,
        },
    )
    ax.text(
        0.500,
        0.035,
        "Not licensed: persistent trait · cold start · real-time inference · cross-interface transfer · user benefit/harm",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.8,
        color=RED,
    )

    save_figure(fig, "measurement_evidence_dag")

    full = summary["full_pipeline_reference_bootstrap"]
    sensor = summary["no_sensor_baseline"]
    signed = summary["signed_trace_correction"]["budgets"]
    rows: list[dict[str, object]] = [
        {"node": "measurement_meaning", "status": "PARTIAL"},
        {"node": "generalizability", "status": "PARTIAL"},
        {"node": "direct_joystick", "status": "AVAILABLE"},
        {"node": "behavioral_calibration", "status": "PROXIMAL_SUPPORT"},
        {"node": "eeg_fnirs_sensing", "status": "NO_INCREMENT"},
        {"node": "proximal_signed_correction", "status": "SUPPORTED"},
        {"node": "downstream_utility", "status": "NOT_TESTED"},
        {"node": "retention_transfer", "status": "NOT_LICENSED"},
        {
            "node": "full_pipeline_simple_g_above_070",
            "value": full["relative_g_15_above_070_fraction"],
        },
        {
            "node": "full_pipeline_simple_phi_above_070",
            "value": full["absolute_phi_15_above_070_fraction"],
        },
        {
            "node": "sensor_gain_over_context_seconds",
            "value": sensor["sensor_gain_over_context_seconds"],
        },
    ]
    for budget, values in signed.items():
        rows.append(
            {
                "node": "signed_correction_gain_vs_none",
                "budget_videos": int(budget),
                "value": values["mean_calibrated_gain_vs_none_units"],
                "ci95_low": values["calibrated_gain_vs_none_ci95"][0],
                "ci95_high": values["calibrated_gain_vs_none_ci95"][1],
            }
        )
    write_source_data("measurement_evidence_dag", rows)
    return rows


def objective_summary(summary: dict) -> list[dict[str, object]]:
    output = []
    objectives = summary["dtw_objective_audit"]["objectives"]
    labels = {
        "direct_step": "Direct step",
        "mixed_step_mean": "Mixed normalization",
        "step_zero": r"$\lambda_s=0$",
        "path_mean": "Path mean",
    }
    for key, values in objectives.items():
        output.append(
            {
                "objective": key,
                "label": labels[key],
                "path_ratio": values["path_length_to_trial_nodes_ratio"]["mean"],
                "profile_rho": values["profile_spearman_vs_direct"],
                "delta_q": values["mean_q_difference_vs_direct_seconds"],
                "delta_q_low": values["mean_q_difference_ci95"][0],
                "delta_q_high": values["mean_q_difference_ci95"][1],
                "relative_g": values["relative_g_15"],
                "absolute_phi": values["absolute_phi_15"],
            }
        )
    return output


def figure_estimator_reference_actionability(summary: dict) -> list[dict[str, object]]:
    robustness = load_csv(REVISION4 / "estimand_robustness.csv")
    full_rows = load_csv(REVISION4 / "full_pipeline_bootstrap.csv")
    objective_rows = objective_summary(summary)
    signed = summary["signed_trace_correction"]["budgets"]
    category = summary["category_gstudy"]
    full = summary["full_pipeline_reference_bootstrap"]

    # The long panel-A y labels expand a tight bounding box; this canvas width
    # keeps the exported PDF close to a 183-mm double-column figure.
    fig = plt.figure(figsize=(6.70, 5.70))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.985,
        bottom=0.095,
        top=0.895,
        wspace=0.34,
        hspace=0.68,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # A: estimator dependence.
    estimand_order = ["consensus", "axis_mean", "multivariate", "valence_only", "arousal_only"]
    estimand_labels = {
        "consensus": "Consensus map",
        "axis_mean": "Axis-mean lag",
        "multivariate": "Multivariate DTW",
        "valence_only": "Valence only",
        "arousal_only": "Arousal only",
    }
    by_name = {row["estimand"]: row for row in robustness}
    y = np.arange(len(estimand_order))[::-1]
    for position, name in enumerate(estimand_order):
        row = by_name[name]
        yy = y[position]
        g = float(row["relative_g_15"])
        phi = float(row["absolute_phi_15"])
        rho = float(row["profile_spearman_vs_consensus"])
        ax_a.plot(
            [float(row["relative_g_15_ci95_low"]), float(row["relative_g_15_ci95_high"])],
            [yy + 0.12, yy + 0.12],
            color=BLUE,
            linewidth=1.1,
        )
        ax_a.plot(g, yy + 0.12, "o", color=BLUE, markersize=4.0)
        ax_a.plot(
            [float(row["absolute_phi_15_ci95_low"]), float(row["absolute_phi_15_ci95_high"])],
            [yy - 0.12, yy - 0.12],
            color=ORANGE,
            linewidth=1.1,
        )
        ax_a.plot(phi, yy - 0.12, "s", color=ORANGE, markersize=3.6)
        ax_a.plot(rho, yy, "D", color=PURPLE, markersize=3.2)
    ax_a.axvline(0.70, color=GRAY, linestyle="--", linewidth=0.8)
    ax_a.set_yticks(y, [estimand_labels[name] for name in estimand_order])
    ax_a.set_xlim(0.35, 1.03)
    ax_a.set_xlabel("Coefficient / profile rank correlation")
    ax_a.set_title("Estimator choice changes reliability and rank", pad=29)
    ax_a.grid(axis="x", color="#D8DDE3", linewidth=0.45)
    handles = [
        mpl.lines.Line2D([], [], color=BLUE, marker="o", linestyle="-", label=r"$G_{15}$ (95% CI)"),
        mpl.lines.Line2D([], [], color=ORANGE, marker="s", linestyle="-", label=r"$\Phi_{15}$ (95% CI)"),
        mpl.lines.Line2D([], [], color=PURPLE, marker="D", linestyle="none", label=r"Profile $\rho$ vs consensus"),
    ]
    ax_a.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015),
        ncol=3,
        fontsize=5.2,
        columnspacing=0.8,
        handlelength=1.4,
        borderaxespad=0,
    )
    panel_label(ax_a, "A")

    # B: objective geometry.
    offsets = {
        "direct_step": (5, -13),
        "mixed_step_mean": (6, 8),
        "step_zero": (6, -16),
        "path_mean": (-47, 7),
    }
    colors = {
        "direct_step": BLUE,
        "mixed_step_mean": ORANGE,
        "step_zero": GREEN,
        "path_mean": RED,
    }
    display_jitter = {
        "direct_step": (0.0, 0.0),
        "mixed_step_mean": (0.007, 0.008),
        "step_zero": (-0.007, -0.008),
        "path_mean": (0.0, 0.0),
    }
    for row in objective_rows:
        key = str(row["objective"])
        display_x = float(row["path_ratio"]) + display_jitter[key][0]
        display_y = float(row["profile_rho"]) + display_jitter[key][1]
        ax_b.scatter(
            display_x,
            display_y,
            s=34,
            color=colors[key],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        label = str(row["label"])
        if key != "direct_step":
            label += f"\nΔq={float(row['delta_q']):+.3f} s"
        ax_b.annotate(
            label,
            (display_x, display_y),
            xytext=offsets[key],
            textcoords="offset points",
            fontsize=5.6,
            color=DARK,
            ha="left",
            va="center",
        )
    ax_b.axhline(1.0, color="#D8DDE3", linewidth=0.55)
    ax_b.set_xlim(1.08, 1.97)
    ax_b.set_ylim(0.42, 1.055)
    ax_b.set_xlabel("Mean DTW path length / trial nodes")
    ax_b.set_ylabel(r"Profile $\rho$ vs direct-step objective")
    ax_b.set_title("Path-mean normalization changes the estimand")
    ax_b.grid(color="#D8DDE3", linewidth=0.45)
    panel_label(ax_b, "B")

    # C: fast versus reference-aware reliability uncertainty.
    consensus = by_name["consensus"]
    category_fast = category["category_stratified_fast_bootstrap"]
    rows_c = [
        (
            "Fast crossed",
            float(consensus["relative_g_15"]),
            [float(consensus["relative_g_15_ci95_low"]), float(consensus["relative_g_15_ci95_high"])],
            float(consensus["absolute_phi_15"]),
            [float(consensus["absolute_phi_15_ci95_low"]), float(consensus["absolute_phi_15_ci95_high"])],
        ),
        (
            "Category-stratified fast",
            float(consensus["relative_g_15"]),
            category_fast["simple_relative_g_15_ci95"],
            float(consensus["absolute_phi_15"]),
            category_fast["simple_absolute_phi_15_ci95"],
        ),
        (
            "Reference-aware full",
            float(np.median([float(row["relative_g_15"]) for row in full_rows])),
            full["relative_g_15_ci95"],
            float(np.median([float(row["absolute_phi_15"]) for row in full_rows])),
            full["absolute_phi_15_ci95"],
        ),
        (
            "Full + category facet",
            float(np.median([float(row["category_relative_g_15"]) for row in full_rows])),
            full["category_relative_g_15_ci95"],
            float(np.median([float(row["category_absolute_phi_15"]) for row in full_rows])),
            full["category_absolute_phi_15_ci95"],
        ),
    ]
    y_c = np.arange(len(rows_c))[::-1]
    for yy, (_, g, g_ci, phi, phi_ci) in zip(y_c, rows_c):
        ax_c.plot(g_ci, [yy + 0.11, yy + 0.11], color=BLUE, linewidth=1.2)
        ax_c.plot(g, yy + 0.11, "o", color=BLUE, markersize=4.0)
        ax_c.plot(phi_ci, [yy - 0.11, yy - 0.11], color=ORANGE, linewidth=1.2)
        ax_c.plot(phi, yy - 0.11, "s", color=ORANGE, markersize=3.7)
    ax_c.axvline(0.70, color=RED, linestyle="--", linewidth=0.9)
    ax_c.set_yticks(y_c, [row[0] for row in rows_c])
    ax_c.set_xlim(0.20, 0.96)
    ax_c.set_xlabel("Reliability coefficient (95% interval)")
    ax_c.set_title("The .70 decision gate remains uncertain")
    ax_c.grid(axis="x", color="#D8DDE3", linewidth=0.45)
    panel_label(ax_c, "C")

    # D: signed correction.
    budgets = np.asarray([1, 2, 4, 8], dtype=float)
    gain_none = np.asarray([signed[str(int(b))]["mean_calibrated_gain_vs_none_units"] for b in budgets])
    ci_none = np.asarray([signed[str(int(b))]["calibrated_gain_vs_none_ci95"] for b in budgets])
    gain_video = np.asarray([signed[str(int(b))]["mean_calibrated_gain_vs_video_units"] for b in budgets])
    ci_video = np.asarray([signed[str(int(b))]["calibrated_gain_vs_video_ci95"] for b in budgets])
    ax_d.errorbar(
        budgets - 0.10,
        gain_none,
        yerr=np.vstack([gain_none - ci_none[:, 0], ci_none[:, 1] - gain_none]),
        color=GREEN,
        marker="o",
        markersize=4.0,
        capsize=2.4,
        linewidth=1.15,
        label="Calibrated vs no correction",
    )
    ax_d.errorbar(
        budgets + 0.10,
        gain_video,
        yerr=np.vstack([gain_video - ci_video[:, 0], ci_video[:, 1] - gain_video]),
        color=PURPLE,
        marker="s",
        markersize=3.8,
        capsize=2.4,
        linewidth=1.15,
        label="Increment vs video correction",
    )
    ax_d.axhline(0, color=GRAY, linewidth=0.75)
    ax_d.set_xticks(budgets, ["1", "2", "4", "8"])
    ax_d.set_xlabel("Calibration videos")
    ax_d.set_ylabel("Held-out trace-MAE gain (label units)")
    ax_d.set_title("Signed calibration supports proximal correction", pad=29)
    ax_d.grid(axis="y", color="#D8DDE3", linewidth=0.45)
    ax_d.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015),
        ncol=2,
        fontsize=5.5,
        columnspacing=0.9,
        handlelength=1.5,
        borderaxespad=0,
    )
    ax_d.text(
        0.99,
        0.03,
        "Proximal trace endpoint; not user benefit",
        transform=ax_d.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.3,
        color=RED,
    )
    panel_label(ax_d, "D")

    save_figure(fig, "estimator_reference_actionability")

    source_rows: list[dict[str, object]] = []
    for row in robustness:
        source_rows.append({"panel": "A", **row})
    for row in objective_rows:
        source_rows.append({"panel": "B", **row})
    for label, g, g_ci, phi, phi_ci in rows_c:
        source_rows.extend(
            [
                {"panel": "C", "analysis": label, "metric": "G15", "estimate": g, "ci95_low": g_ci[0], "ci95_high": g_ci[1]},
                {"panel": "C", "analysis": label, "metric": "Phi15", "estimate": phi, "ci95_low": phi_ci[0], "ci95_high": phi_ci[1]},
            ]
        )
    for budget in budgets.astype(int):
        values = signed[str(budget)]
        source_rows.extend(
            [
                {
                    "panel": "D",
                    "budget_videos": budget,
                    "comparison": "calibrated_vs_none",
                    "estimate": values["mean_calibrated_gain_vs_none_units"],
                    "ci95_low": values["calibrated_gain_vs_none_ci95"][0],
                    "ci95_high": values["calibrated_gain_vs_none_ci95"][1],
                },
                {
                    "panel": "D",
                    "budget_videos": budget,
                    "comparison": "calibrated_vs_video",
                    "estimate": values["mean_calibrated_gain_vs_video_units"],
                    "ci95_low": values["calibrated_gain_vs_video_ci95"][0],
                    "ci95_high": values["calibrated_gain_vs_video_ci95"][1],
                },
            ]
        )
    write_source_data("estimator_reference_actionability", source_rows)
    return source_rows


def main() -> None:
    summary_path = REVISION4 / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "Formal revision4 analysis is incomplete: artifacts/revision4/summary.json is absent"
        )
    summary = load_json(summary_path)
    figure_measurement_evidence_dag(summary)
    figure_estimator_reference_actionability(summary)
    print("Generated revision-4 figures and source-data CSVs.")


if __name__ == "__main__":
    main()
