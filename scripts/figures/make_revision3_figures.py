#!/usr/bin/env python3
"""Generate the four revision-3 manuscript figures from committed artifacts."""

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
ARTIFACTS = ROOT / "artifacts"

BLUE = "#2C7FB8"
ORANGE = "#D95F0E"
GREEN = "#238B45"
PURPLE = "#756BB1"
RED = "#B33A3A"
DARK = "#374151"
GRAY = "#6B7280"
LIGHT_BLUE = "#D9ECF7"
LIGHT_ORANGE = "#FCE1D3"
LIGHT_GREEN = "#D9F0E1"
LIGHT_PURPLE = "#E7E1F4"
LIGHT_GRAY = "#E5E7EB"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.25,
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
    common = {"bbox_inches": "tight", "pad_inches": 0.03}
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
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def schematic_box(
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
    title_size: float = 6.7,
    body_size: float = 5.15,
) -> None:
    patch = mpl.patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.007,rounding_size=0.012",
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
        y + height - 0.067,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=body_size,
        color=DARK,
        linespacing=1.28,
    )


def schematic_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GRAY,
    linewidth: float = 0.9,
) -> None:
    ax.add_patch(
        mpl.patches.FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=8.0,
            linewidth=linewidth,
            color=color,
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
        )
    )


def figure_measurement_audit_overview() -> None:
    fig, ax = plt.subplots(figsize=(7.17, 4.75))
    ax.set_axis_off()

    ax.text(0.005, 0.98, "A", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    ax.text(
        0.035,
        0.98,
        "Fold-safe construction of a reference-relative temporal profile",
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=DARK,
        va="top",
    )
    top_specs = [
        (0.035, 0.168, "Public arrays", ["24 participants", "15 videos · joystick V/A", "EEG + fNIRS"], LIGHT_BLUE, BLUE),
        (0.220, 0.160, "Outer split", ["5 participant folds", "19/20 training", "4/5 held out"], LIGHT_GRAY, GRAY),
        (0.398, 0.185, "Nested reference", [r"Inner train: $A\setminus p$", r"Inner validation: $A$", "Rebuilt in every split"], LIGHT_PURPLE, PURPLE),
        (0.601, 0.165, "Constrained warp", ["Participant radius 2", "Reference radius 3", "Monotone · 10-s band"], LIGHT_ORANGE, ORANGE),
        (0.784, 0.185, "Common targets", [r"$b_{pv}$: signed bias", r"$q_{pv}$: absolute magnitude", r"$Q_p$: video aggregate"], LIGHT_GREEN, GREEN),
    ]
    for spec in top_specs:
        schematic_box(ax, spec[0], 0.790, spec[1], 0.135, spec[2], spec[3], facecolor=spec[4], edgecolor=spec[5])
    for left, right in zip(top_specs[:-1], top_specs[1:]):
        schematic_arrow(ax, (left[0] + left[1] + 0.004, 0.858), (right[0] - 0.004, 0.858))
    ax.text(
        0.5,
        0.755,
        "Held-out participants never define their own reference, target, context, scaling, or model selection.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.0,
        fontweight="bold",
        color=RED,
    )

    ax.text(0.005, 0.705, "B", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    ax.text(
        0.035,
        0.705,
        "Claim escalation for durable person-level profiling",
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=DARK,
        va="top",
    )
    stage_specs = [
        (0.025, 0.178, "Measurable", ["Cross-axis transfer", "Own vs. donor", "Temporal null"], LIGHT_BLUE, BLUE),
        (0.218, 0.178, "Operational boundary", ["Distinguishability tests", "Known perturbations", "Nine DTW settings", "Geometry controls"], LIGHT_ORANGE, ORANGE),
        (0.411, 0.178, "Generalizable", [r"Unified LOPO $G/\Phi$", "Reference sensitivity", "Declared facets only"], LIGHT_PURPLE, PURPLE),
        (0.604, 0.178, "Predictable", [r"Same target $q_{pv}$", "Nested context baseline", "Modality ablations"], LIGHT_GREEN, GREEN),
        (0.797, 0.178, "Consequential utility", ["Held-out correction", "Meaningful-gain gate", "NOT ESTABLISHED"], LIGHT_GRAY, RED),
    ]
    for spec in stage_specs:
        schematic_box(
            ax,
            spec[0],
            0.430,
            spec[1],
            0.205,
            spec[2],
            spec[3],
            facecolor=spec[4],
            edgecolor=spec[5],
            title_size=5.35,
            body_size=4.95,
        )
    for left, right in zip(stage_specs[:-1], stage_specs[1:]):
        schematic_arrow(ax, (left[0] + left[1] + 0.004, 0.535), (right[0] - 0.004, 0.535), color=DARK)

    session_gate = mpl.patches.FancyBboxPatch(
        (0.417, 0.348),
        0.166,
        0.052,
        boxstyle="round,pad=0.005,rounding_size=0.01",
        transform=ax.transAxes,
        facecolor="#FFF7F7",
        edgecolor=RED,
        linewidth=0.9,
    )
    ax.add_patch(session_gate)
    ax.text(
        0.500,
        0.374,
        "Session / test–retest gate: NOT TESTED",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.25,
        color=RED,
        fontweight="bold",
    )
    schematic_arrow(ax, (0.500, 0.428), (0.500, 0.403), color=RED, linewidth=0.9)

    ax.text(0.005, 0.300, "C", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    ax.text(
        0.035,
        0.300,
        "Evidence-bounded stopping rule",
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=DARK,
        va="top",
    )
    outer = mpl.patches.FancyBboxPatch(
        (0.035, 0.045),
        0.932,
        0.205,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        transform=ax.transAxes,
        facecolor="#FAFAFA",
        edgecolor=LIGHT_GRAY,
        linewidth=0.8,
    )
    ax.add_patch(outer)
    ax.text(
        0.055,
        0.220,
        "If the next gate is absent or fails, stop escalation and change the deployment decision.",
        transform=ax.transAxes,
        fontsize=6.0,
        color=RED,
        fontweight="bold",
        va="center",
    )
    decision_specs = [
        (0.055, 0.155, "STOP", ["Do not promote", "the claim"], LIGHT_ORANGE, RED),
        (0.255, 0.205, "Bound the metric", ["Reference · interface", "· estimator · session"], LIGHT_BLUE, BLUE),
        (0.505, 0.215, "Report consequences", ["Time · hardware · privacy", "· predictive degradation"], LIGHT_PURPLE, PURPLE),
        (0.765, 0.180, "Retain baseline", ["No personalization", "remains valid"], LIGHT_GREEN, GREEN),
    ]
    for spec in decision_specs:
        schematic_box(ax, spec[0], 0.072, spec[1], 0.118, spec[2], spec[3], facecolor=spec[4], edgecolor=spec[5], title_size=5.9, body_size=4.9)
    for left, right in zip(decision_specs[:-1], decision_specs[1:]):
        schematic_arrow(ax, (left[0] + left[1] + 0.006, 0.131), (right[0] - 0.006, 0.131))

    save_figure(fig, "measurement_audit_overview")


def calibration_participant_gains(
    budget: int,
    predictions: np.ndarray,
    methods: list[str],
    target: np.ndarray,
) -> np.ndarray:
    video_mean = predictions[methods.index("video_mean")]
    rows = load_csv(ARTIFACTS / "phase_personalization" / "results.csv")
    selected = [
        row
        for row in rows
        if row["method"] == "additive_fewshot" and int(row["budget"]) == budget
    ]
    gains = []
    for subject in range(1, 25):
        values = []
        for row in selected:
            if int(row["subject"]) != subject:
                continue
            calibration = {int(value) - 1 for value in row["calibration_videos"].split()}
            evaluation = [video for video in range(15) if video not in calibration]
            baseline_mae = np.abs(
                video_mean[subject - 1, evaluation] - target[subject - 1, evaluation]
            ).mean()
            values.append(baseline_mae - float(row["mae_seconds"]))
        gains.append(float(np.mean(values)))
    return np.asarray(gains)


def figure_validity_identifiability(source: dict[str, object]) -> None:
    """Generate the current validity figure from the canonical source only."""
    summaries = source["summaries"]
    revision2 = summaries["revision2"]
    revision4 = summaries["revision4"]
    robust = summaries["measurement_robustness"]
    dynamics = summaries["normative_dynamics"]
    simulation = source["tables"]["simulation_recovery"]

    fig, axes = plt.subplots(2, 2, figsize=(7.17, 4.65), gridspec_kw={"hspace": 0.52, "wspace": 0.40})
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    source_rows: list[dict[str, object]] = []

    directions = [
        ("Arousal → valence", "valence_from_arousal", BLUE, "o"),
        ("Valence → arousal", "arousal_from_valence", ORANGE, "s"),
        ("Combined", "combined", PURPLE, "D"),
    ]
    y_positions = np.arange(len(directions))
    ax_a.axvline(0, color=GRAY, linestyle=":", linewidth=0.8)
    for y, (label, key, color, marker) in zip(y_positions, directions):
        item = revision2["own_other_warp"][key]
        mean = float(item["mean"])
        low, high = map(float, item["participant_video_crossed_ci95"])
        ax_a.errorbar(mean, y, xerr=np.asarray([[mean - low], [high - mean]]), fmt=marker, color=color, markeredgecolor=DARK, markeredgewidth=0.45, markersize=4.7, capsize=2.5)
        source_rows.append({"panel": "A", "analysis": label, "mean": mean, "ci_low": low, "ci_high": high, "unit": "released joystick label units"})
    ax_a.set_yticks(y_positions, [item[0] for item in directions])
    ax_a.invert_yaxis()
    ax_a.set(title="Same-trial own-minus-donor advantage", xlabel="Own-minus-donor advantage (label units)", xlim=(-0.05, 1.75))
    ax_a.text(0.98, 0.06, "19/24 positive\ncombined", transform=ax_a.transAxes, ha="right", va="bottom", color=PURPLE)
    panel_label(ax_a, "A")

    direct_labels = []
    for scenario, color, marker, label in (
        ("pure_temporal", BLUE, "o", "True shift: 4 s"),
        ("pure_amplitude", ORANGE, "s", "Magnitude pulse: lag 0 s"),
    ):
        selected = sorted([row for row in simulation if row["scenario"] == scenario], key=lambda row: float(row["noise_sd"]))
        noise = np.asarray([float(row["noise_sd"]) for row in selected])
        if scenario == "pure_temporal":
            mean = np.asarray([float(row["lag_mae_seconds_mean"]) for row in selected])
            low = np.asarray([float(row["lag_mae_seconds_ci95_low"]) for row in selected])
            high = np.asarray([float(row["lag_mae_seconds_ci95_high"]) for row in selected])
            metric = "lag recovery MAE"
        else:
            mean = np.asarray([float(row["estimated_mean_absolute_lag_seconds_mean"]) for row in selected])
            low = np.asarray([float(row["estimated_mean_absolute_lag_seconds_ci95_low"]) for row in selected])
            high = np.asarray([float(row["estimated_mean_absolute_lag_seconds_ci95_high"]) for row in selected])
            metric = "apparent absolute lag"
        ax_b.errorbar(noise, mean, yerr=[mean - low, high - mean], color=color, marker=marker, markeredgecolor=DARK, markeredgewidth=0.4, capsize=2.2)
        direct_labels.append((mean[-1], label, color))
        for n, value, lo, hi in zip(noise, mean, low, high):
            source_rows.append({"panel": "B", "analysis": label, "noise_sd_label_units": float(n), "metric": metric, "mean": float(value), "ci_low": float(lo), "ci_high": float(hi), "unit": "seconds"})
    ax_b.set(title="Simulation bounds distinguishability", xlabel="Gaussian noise s.d. (label units)", ylabel="Error or false lag (s)", xlim=(-0.2, 7.0))
    ax_b.set_xticks([0, 2, 5])
    for value, label, color in direct_labels:
        ax_b.text(5.25, value, label, color=color, va="center", fontsize=5.5)
    panel_label(ax_b, "B")

    unmatched = dynamics["event_boundary_absolute_lag_effect"]
    matched = robust["matched_high_change"]["lag_effect"]
    c_items = [
        ("High gradient\n(descriptive)", float(unmatched["observed_mean"]), float(unmatched["participant_video_crossed_bootstrap"]["ci95_low"]), float(unmatched["participant_video_crossed_bootstrap"]["ci95_high"]), ORANGE, "//"),
        ("Local peak\n(slope-adjusted)", float(matched["observed_mean"]), float(matched["participant_video_crossed_ci95"][0]), float(matched["participant_video_crossed_ci95"][1]), LIGHT_GRAY, ".."),
    ]
    ax_c.axhline(0, color=GRAY, linestyle=":", linewidth=0.8)
    for x, (label, mean, low, high, color, hatch) in enumerate(c_items):
        ax_c.bar(x, mean, width=0.58, color=color, edgecolor=DARK, linewidth=0.6, hatch=hatch)
        ax_c.errorbar(x, mean, yerr=np.asarray([[mean - low], [high - mean]]), fmt="none", ecolor=DARK, capsize=2.5)
        source_rows.append({"panel": "C", "analysis": label.replace("\n", " "), "mean": mean, "ci_low": low, "ci_high": high, "unit": "seconds"})
    ax_c.set_xticks([0, 1], [item[0] for item in c_items])
    ax_c.set(title="No peak-specific excess after slope matching", ylabel="High-change minus control |lag| (s)")
    ax_c.text(0.98, 0.95, "Matched slope SMD = 0.112", transform=ax_c.transAxes, ha="right", va="top")
    panel_label(ax_c, "C")

    geometry = revision4["estimand_robustness"]["consensus"]
    raw_variance = float(geometry["participant_variance_seconds_squared"])
    adjusted_variance = float(geometry["geometry_adjusted_participant_variance"])
    raw_g = float(geometry["relative_g_15"])
    raw_g_ci = np.asarray(
        [geometry["relative_g_15_ci95_low"], geometry["relative_g_15_ci95_high"]],
        dtype=float,
    )
    adjusted_g = float(geometry["geometry_adjusted_relative_g_15"])
    adjusted_g_ci = np.asarray(
        [
            geometry["geometry_adjusted_relative_g_15_ci95_low"],
            geometry["geometry_adjusted_relative_g_15_ci95_high"],
        ],
        dtype=float,
    )
    labels = ["Participant variance (s²)", r"Descriptive $R^{SR}_{G,15}$"]
    y = np.asarray([1.0, 0.0])
    ax_d.plot([adjusted_variance, raw_variance], [1, 1], color=LIGHT_GRAY, linewidth=2.0, zorder=1)
    ax_d.scatter([raw_variance], [1], color=BLUE, edgecolor=DARK, s=30, zorder=3)
    ax_d.scatter([adjusted_variance], [1], color=ORANGE, edgecolor=DARK, marker="s", s=30, zorder=3)
    ax_d.plot([adjusted_g, raw_g], [0, 0], color=LIGHT_GRAY, linewidth=2.0, zorder=1)
    ax_d.errorbar(raw_g, 0.08, xerr=np.asarray([[raw_g - raw_g_ci[0]], [raw_g_ci[1] - raw_g]]), fmt="o", color=BLUE, markeredgecolor=DARK, markersize=4.8, capsize=2.3)
    ax_d.errorbar(adjusted_g, -0.08, xerr=np.asarray([[adjusted_g - adjusted_g_ci[0]], [adjusted_g_ci[1] - adjusted_g]]), fmt="s", color=ORANGE, markeredgecolor=DARK, markersize=4.8, capsize=2.3)
    ax_d.set_yticks(y, labels)
    ax_d.set(title="Geometry covaries with the profile", xlabel="Estimate", xlim=(0, 0.96), ylim=(-0.48, 1.48))
    ax_d.text(raw_variance + 0.025, 1.13, "raw .184", color=BLUE, fontsize=5.5)
    ax_d.text(adjusted_variance + 0.025, 0.82, "cross-fitted .057", color=ORANGE, fontsize=5.5)
    ax_d.text(
        0.98,
        0.53,
        "Variance retained 30.9%\n95% CI 8.2–177.9%\nprofile rank ρ=.726",
        transform=ax_d.transAxes,
        ha="right",
        va="center",
        fontsize=5.3,
    )
    ax_d.text(raw_g + 0.02, 0.18, "raw .760", color=BLUE, fontsize=5.4)
    ax_d.text(adjusted_g + 0.02, -0.28, "adjusted .465", color=ORANGE, fontsize=5.4)
    source_rows.extend(
        [
            {"panel": "D", "analysis": "Participant variance", "condition": "raw", "estimate": raw_variance, "unit": "seconds squared"},
            {"panel": "D", "analysis": "Participant variance", "condition": "participant-cross-fitted geometry adjusted", "estimate": adjusted_variance, "retained_fraction": float(geometry["geometry_variance_retained_fraction"]), "retained_ci_low": float(geometry["geometry_variance_retained_ci95_low"]), "retained_ci_high": float(geometry["geometry_variance_retained_ci95_high"]), "unit": "seconds squared"},
            {"panel": "D", "analysis": "Descriptive shared-reference RG15", "condition": "raw", "estimate": raw_g, "ci_low": float(raw_g_ci[0]), "ci_high": float(raw_g_ci[1])},
            {"panel": "D", "analysis": "Descriptive shared-reference RG15", "condition": "participant-cross-fitted geometry adjusted", "estimate": adjusted_g, "ci_low": float(adjusted_g_ci[0]), "ci_high": float(adjusted_g_ci[1]), "profile_rank_spearman": float(geometry["geometry_profile_spearman_raw_vs_adjusted"])},
        ]
    )
    panel_label(ax_d, "D")

    write_source_data("validity_identifiability", source_rows)
    save_figure(fig, "validity_identifiability")


def figure_matched_target_comparison() -> None:
    revision3 = load_json(ARTIFACTS / "revision3" / "summary.json")
    nonlinear = load_json(ARTIFACTS / "revision3_nonlinear" / "summary.json")
    calibration = revision3["calibration_uncertainty"]["budgets"]["random"]
    npz = np.load(ARTIFACTS / "revision3" / "matched_target_predictions.npz")
    methods = npz["methods"].tolist()
    predictions = npz["predictions"]
    target = npz["target"]

    fig = plt.figure(figsize=(7.17, 4.05))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.38, 0.95, 1.05], wspace=0.72)
    ax_a, ax_b, ax_c = [fig.add_subplot(grid[0, index]) for index in range(3)]
    source_rows: list[dict[str, object]] = []

    display_methods = [
        ("Video mean", "video_mean", BLUE, "o"),
        ("Context", "context_only", BLUE, "s"),
        ("CBraMod", "cbramod_only", PURPLE, "o"),
        ("Handcrafted EEG", "handcrafted_eeg_only", PURPLE, "s"),
        ("EEG combined", "eeg_only", PURPLE, "D"),
        ("fNIRS HRF-aware", "fnirs_only", ORANGE, "o"),
        ("EEG + fNIRS", "eeg_fnirs_single_penalty", ORANGE, "s"),
        ("Context + EEG", "context_eeg", GREEN, "o"),
        ("Context + fNIRS", "context_fnirs", GREEN, "s"),
        ("Context + EEG + fNIRS", "context_eeg_fnirs_single_penalty", GREEN, "D"),
        ("Blockwise residual", "context_plus_blockwise_residual", RED, "o"),
        ("Compact MLP (off 5/5)", "nonlinear_residual_mlp", GRAY, "X"),
    ]
    method_summary = revision3["nested_matched_target_sensing"]["methods"]
    values = []
    for label, key, color, marker in display_methods:
        if key == "nonlinear_residual_mlp":
            value = float(nonlinear["nonlinear_participant_macro_mae_seconds"])
        else:
            value = float(method_summary[key]["participant_macro_mae_seconds"])
        values.append(value)
    values = np.asarray(values)
    y = np.arange(len(display_methods))
    context_value = float(method_summary["context_only"]["participant_macro_mae_seconds"])
    ax_a.axvline(context_value, color=BLUE, linestyle=":", linewidth=0.9)
    for position, ((label, key, color, marker), value) in enumerate(zip(display_methods, values)):
        ax_a.scatter(value, position, color=color, marker=marker, edgecolor=DARK, linewidth=0.45, s=26, zorder=3)
        ax_a.text(value + 0.0022, position, f"{value:.3f}", va="center", fontsize=5.2)
        source_rows.append({"panel": "A", "analysis": label, "method_key": key, "participant_macro_mae_seconds": float(value)})
    ax_a.set_yticks(y, [item[0] for item in display_methods])
    ax_a.invert_yaxis()
    ax_a.set(title=r"Same-trial prediction of $q_{pv}$", xlabel="Participant-macro MAE (s; lower is better)", xlim=(0.748, 0.895))
    ax_a.text(0.98, 0.02, "Vertical line = context", transform=ax_a.transAxes, ha="right", va="bottom", color=BLUE, fontsize=5.4)
    panel_label(ax_a, "A")

    budgets = [1, 2, 4, 8]
    positions = np.arange(len(budgets))
    means = np.asarray([float(calibration[str(b)]["mean_gain_seconds_scaled_profile_error"]) for b in budgets])
    lows = np.asarray([float(calibration[str(b)]["participant_and_subset_bootstrap_ci95_low"]) for b in budgets])
    highs = np.asarray([float(calibration[str(b)]["participant_and_subset_bootstrap_ci95_high"]) for b in budgets])
    relative = np.asarray([float(calibration[str(b)]["relative_gain_percent"]) for b in budgets])
    ax_b.axhline(0, color=GRAY, linestyle=":", linewidth=0.8)
    ax_b.errorbar(positions, means, yerr=[means - lows, highs - means], color=GREEN, marker="o", markeredgecolor=DARK, markeredgewidth=0.45, capsize=2.5)
    for x, mean, high, rel in zip(positions, means, highs, relative):
        ax_b.text(x, high + 0.006, f"{rel:.1f}%", ha="center", va="bottom", color=GREEN, fontsize=5.4)
    ax_b.set_xticks(positions, [str(b) for b in budgets])
    ax_b.set(title="Random behavioral calibration", xlabel="Calibration videos", ylabel="Aggregate profile-error gain (s)", ylim=(-0.035, 0.125))
    ax_b.text(0.03, 0.03, "Joint participant–subset CI\nnot local millisecond precision", transform=ax_b.transAxes, va="bottom", fontsize=5.2)
    for budget, mean, low, high, rel in zip(budgets, means, lows, highs, relative):
        source_rows.append({"panel": "B", "analysis": "Random calibration", "calibration_videos": budget, "mean_gain_seconds_scaled_profile_error": float(mean), "ci_low": float(low), "ci_high": float(high), "relative_gain_percent": float(rel)})
    panel_label(ax_b, "B")

    colors = [BLUE, PURPLE, ORANGE, GREEN]
    markers = ["o", "s", "D", "^"]
    all_gains = []
    for x, budget, color, marker in zip(positions, budgets, colors, markers):
        gains = calibration_participant_gains(budget, predictions, methods, target)
        all_gains.append(gains)
        expected = calibration[str(budget)]
        assert int(np.sum(gains > 0)) == int(expected["participants_improved"])
        assert np.isclose(np.median(gains), float(expected["participant_gain_median_seconds"]), atol=1e-9)
        assert np.isclose(np.min(gains), float(expected["worst_participant_predictive_degradation_seconds"]), atol=1e-9)
        jitter = np.linspace(-0.13, 0.13, len(gains))
        ax_c.scatter(np.full(len(gains), x) + jitter, gains, color=color, marker=marker, edgecolor="white", linewidth=0.3, s=18, alpha=0.9)
        ax_c.plot([x - 0.18, x + 0.18], [np.median(gains), np.median(gains)], color=DARK, linewidth=1.4)
        for participant, gain in enumerate(gains, start=1):
            source_rows.append({"panel": "C", "analysis": "Participant calibration gain", "calibration_videos": budget, "participant": participant, "gain_seconds_scaled_profile_error": float(gain)})
    ax_c.axhline(0, color=GRAY, linestyle=":", linewidth=0.8)
    ax_c.set_xticks(positions, [str(b) for b in budgets])
    ax_c.set(title="Participant-level heterogeneity", xlabel="Calibration videos", ylabel="Participant gain (s)", xlim=(-0.45, 3.45), ylim=(-0.08, 0.23))
    ax_c.text(0.03, 0.03, "Below zero = predictive degradation", transform=ax_c.transAxes, color=RED, va="bottom", fontsize=5.3)
    panel_label(ax_c, "C")

    write_source_data("matched_target_comparison", source_rows)
    save_figure(fig, "matched_target_comparison")


def balanced_variance_components(matrix: np.ndarray) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64)
    participants, videos = values.shape
    grand = float(values.mean())
    row_mean = values.mean(axis=1)
    column_mean = values.mean(axis=0)
    residual = values - row_mean[:, None] - column_mean[None, :] + grand
    ms_participant = float(videos * np.square(row_mean - grand).sum() / (participants - 1))
    ms_video = float(participants * np.square(column_mean - grand).sum() / (videos - 1))
    ms_residual = float(np.square(residual).sum() / ((participants - 1) * (videos - 1)))
    return {
        "participant": float(max((ms_participant - ms_residual) / videos, 0.0)),
        "video": float(max((ms_video - ms_residual) / participants, 0.0)),
        "residual": float(max(ms_residual, 0.0)),
    }


def reliability_for_k(components: dict[str, float], k: int) -> tuple[float, float]:
    participant = components["participant"]
    video = components["video"]
    residual = components["residual"]
    relative = participant / (participant + residual / k) if participant + residual / k > 0 else 0.0
    absolute = participant / (participant + (video + residual) / k) if participant + (video + residual) / k > 0 else 0.0
    return float(relative), float(absolute)


def unified_lopo_reliability_curves() -> tuple[np.ndarray, ...]:
    rows = load_csv(ARTIFACTS / "revision3" / "unified_lopo_targets.csv")
    matrix = np.full((24, 15), np.nan)
    for row in rows:
        matrix[int(row["subject"]) - 1, int(row["video"]) - 1] = float(row["q_pv_seconds"])
    if not np.isfinite(matrix).all():
        raise RuntimeError("Unified-LOPO target matrix is incomplete")
    videos = np.arange(1, 16)
    components = balanced_variance_components(matrix)
    point = np.asarray([reliability_for_k(components, int(k)) for k in videos])
    rng = np.random.default_rng(20260728 + 200)
    draws = np.empty((5000, 15, 2), dtype=np.float64)
    for repeat in range(5000):
        participant_draw = rng.integers(0, 24, 24)
        video_draw = rng.integers(0, 15, 15)
        draw_components = balanced_variance_components(matrix[np.ix_(participant_draw, video_draw)])
        draws[repeat] = np.asarray([reliability_for_k(draw_components, int(k)) for k in videos])
    low = np.quantile(draws, 0.025, axis=0)
    high = np.quantile(draws, 0.975, axis=0)
    return videos, point[:, 0], point[:, 1], low[:, 0], high[:, 0], low[:, 1], high[:, 1]


def figure_reliability_reference_burden() -> None:
    revision2 = load_json(ARTIFACTS / "revision2" / "summary.json")
    revision3 = load_json(ARTIFACTS / "revision3" / "summary.json")
    gstudy = revision3["unified_lopo_gstudy"]
    calibration = revision3["calibration_uncertainty"]["budgets"]["random"]

    fig, axes = plt.subplots(2, 2, figsize=(7.17, 4.55), gridspec_kw={"hspace": 0.56, "wspace": 0.46})
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    source_rows: list[dict[str, object]] = []

    videos, g, phi, g_low, g_high, phi_low, phi_high = unified_lopo_reliability_curves()
    assert np.isclose(g[-1], float(gstudy["relative_g_15"]), atol=1e-12)
    assert np.isclose(phi[-1], float(gstudy["absolute_phi_15"]), atol=1e-12)
    assert np.allclose([g_low[-1], g_high[-1]], gstudy["relative_g_15_ci95"], atol=1e-12)
    assert np.allclose([phi_low[-1], phi_high[-1]], gstudy["absolute_phi_15_ci95"], atol=1e-12)
    ax_a.fill_between(videos, g_low, g_high, color=LIGHT_BLUE, alpha=0.85)
    ax_a.fill_between(videos, phi_low, phi_high, color=LIGHT_ORANGE, alpha=0.70)
    ax_a.plot(videos, g, color=BLUE, marker="o", markersize=2.6)
    ax_a.plot(videos, phi, color=ORANGE, marker="s", markersize=2.6, linestyle="--")
    ax_a.axhline(0.70, color=GRAY, linestyle=":", linewidth=0.8)
    ax_a.text(15.15, g[-1], "Relative G", color=BLUE, va="center", fontsize=5.6)
    ax_a.text(15.15, phi[-1] - 0.035, "Absolute Φ", color=ORANGE, va="center", fontsize=5.6)
    ax_a.set(title="Unified-LOPO within-session reliability", xlabel="Videos aggregated", ylabel="Coefficient", xlim=(1, 17.5), ylim=(0.02, 1.0))
    ax_a.set_xticks([1, 4, 8, 12, 15])
    for index, video in enumerate(videos):
        source_rows.extend(
            [
                {"panel": "A", "analysis": "Relative G", "videos": int(video), "estimate": float(g[index]), "ci_low": float(g_low[index]), "ci_high": float(g_high[index])},
                {"panel": "A", "analysis": "Absolute Phi", "videos": int(video), "estimate": float(phi[index]), "ci_low": float(phi_low[index]), "ci_high": float(phi_high[index])},
            ]
        )
    panel_label(ax_a, "A")

    b_items = [
        ("Relative G", float(gstudy["g_70_point_crossing_videos"]), gstudy["g_70_bootstrap_crossing"], BLUE, "o", 1),
        ("Absolute Φ", float(gstudy["phi_70_point_crossing_videos"]), gstudy["phi_70_bootstrap_crossing"], ORANGE, "s", 0),
    ]
    ax_b.axvspan(15, 30, color=LIGHT_GRAY, alpha=0.55, zorder=0)
    ax_b.axvline(15, color=GRAY, linestyle=":", linewidth=0.8)
    for label, point_value, boot, color, marker, y in b_items:
        median = float(boot["median"])
        low, high = map(float, boot["ci95"])
        ax_b.errorbar(median, y - 0.08, xerr=np.asarray([[median - low], [high - median]]), color=color, marker=marker, markeredgecolor=DARK, markeredgewidth=0.45, markersize=4.8, capsize=2.5)
        ax_b.scatter([point_value], [y + 0.12], facecolor="white", edgecolor=color, marker=marker, s=32, linewidth=1.0, zorder=4)
        ax_b.text(high + 0.5, y - 0.08, f"{100*float(boot['not_crossed_by_15_fraction']):.1f}% >15", color=color, va="center", fontsize=5.4)
        source_rows.extend(
            [
                {"panel": "B", "analysis": label, "condition": "point estimate", "videos": point_value, "threshold": 0.70},
                {"panel": "B", "analysis": label, "condition": "bootstrap crossing", "videos": median, "ci_low": low, "ci_high": high, "not_crossed_by_15_percent": 100 * float(boot["not_crossed_by_15_fraction"]), "threshold": 0.70},
            ]
        )
    ax_b.set_yticks([1, 0], ["Relative G", "Absolute Φ"])
    ax_b.set(title="Uncertain count at the declared .70 gate", xlabel="Videos needed", xlim=(0, 31), ylim=(-0.45, 1.45))
    ax_b.text(22.5, 1.34, "Extrapolated beyond\n15 observed videos", ha="center", va="top", color=GRAY, fontsize=5.3)
    ax_b.text(0.02, 0.03, "Filled = bootstrap median\nOpen = point count", transform=ax_b.transAxes, va="bottom", fontsize=5.2)
    panel_label(ax_b, "B")

    sensitivity = revision2["reference_composition_sensitivity"]["metrics"]
    sizes = np.asarray([8, 12, 16])
    profile_icc = np.asarray([sensitivity[str(size)]["profile_icc_consistency"]["mean"] for size in sizes])
    icc_low = np.asarray([sensitivity[str(size)]["profile_icc_consistency"]["ci95"][0] for size in sizes])
    icc_high = np.asarray([sensitivity[str(size)]["profile_icc_consistency"]["ci95"][1] for size in sizes])
    profile_shift = np.asarray([sensitivity[str(size)]["profile_mae_seconds"]["mean"] for size in sizes])
    ax_c.errorbar(sizes, profile_icc, yerr=[profile_icc - icc_low, icc_high - profile_icc], color=PURPLE, marker="o", markeredgecolor=DARK, markeredgewidth=0.4, capsize=2.3)
    for size, icc, shift in zip(sizes, profile_icc, profile_shift):
        ax_c.text(size, icc + 0.018, f"shift {shift:.3f} s", ha="center", va="bottom", color=ORANGE, fontsize=5.3)
    ax_c.set(title="Reference samples preserve rank, not level", xlabel="Participants defining reference", ylabel="Profile consistency ICC", ylim=(0.78, 0.99))
    ax_c.set_xticks(sizes)
    for index, size in enumerate(sizes):
        source_rows.append({"panel": "C", "analysis": "Profile consistency ICC", "reference_participants": int(size), "estimate": float(profile_icc[index]), "ci_low": float(icc_low[index]), "ci_high": float(icc_high[index]), "mean_absolute_profile_shift_seconds": float(profile_shift[index])})
    panel_label(ax_c, "C")

    budgets = np.asarray([1, 2, 4, 8], dtype=float)
    relative = np.asarray([float(calibration[str(int(b))]["relative_gain_percent"]) for b in budgets])
    means = np.asarray([float(calibration[str(int(b))]["mean_gain_seconds_scaled_profile_error"]) for b in budgets])
    ci_low_seconds = np.asarray([float(calibration[str(int(b))]["participant_and_subset_bootstrap_ci95_low"]) for b in budgets])
    ci_high_seconds = np.asarray([float(calibration[str(int(b))]["participant_and_subset_bootstrap_ci95_high"]) for b in budgets])
    baseline = means / (relative / 100.0)
    rel_low = 100.0 * ci_low_seconds / baseline
    rel_high = 100.0 * ci_high_seconds / baseline
    ax_d.axhline(0, color=GRAY, linestyle=":", linewidth=0.8)
    ax_d.errorbar(budgets, relative, yerr=[relative - rel_low, rel_high - relative], color=GREEN, marker="D", markeredgecolor=DARK, markeredgewidth=0.45, capsize=2.4)
    for budget, value in zip(budgets, relative):
        ax_d.text(budget, value + 0.55, f"{value:.1f}%", ha="center", va="bottom", color=GREEN, fontsize=5.4)
    ax_d.set(title="Calibration burden versus aggregate gain", xlabel="Calibration videos", ylabel="Relative profile-error gain (%)", xlim=(0.5, 8.5), ylim=(-3.5, 15.0))
    ax_d.set_xticks(budgets)
    mean_video_minutes = 102.4 / 60.0
    top = ax_d.secondary_xaxis("top", functions=(lambda x: x * mean_video_minutes, lambda minutes: minutes / mean_video_minutes))
    top.set_xlabel("Expected reporting time (min)")
    top.set_xticks(budgets * mean_video_minutes, [f"{value:.1f}" for value in budgets * mean_video_minutes])
    ax_d.text(0.03, 0.04, r"Aggregate $q$ prediction; utility not established", transform=ax_d.transAxes, va="bottom", color=RED, fontsize=5.2)
    for index, budget in enumerate(budgets):
        source_rows.append({"panel": "D", "analysis": "Random calibration relative gain", "calibration_videos": int(budget), "expected_reporting_minutes": float(budget * mean_video_minutes), "relative_gain_percent": float(relative[index]), "relative_ci_low_percent": float(rel_low[index]), "relative_ci_high_percent": float(rel_high[index]), "mean_gain_seconds_scaled_profile_error": float(means[index])})
    panel_label(ax_d, "D")

    write_source_data("reliability_reference_burden", source_rows)
    save_figure(fig, "reliability_reference_burden")


def write_dtw_sensitivity_source() -> None:
    rows = load_csv(ARTIFACTS / "measurement_robustness" / "sensitivity.csv")
    write_source_data("dtw_sensitivity", [{key: value for key, value in row.items()} for row in rows])


def main() -> None:
    summary = load_json(ARTIFACTS / "revision3" / "summary.json")
    if summary["target_definition_audit"]["status"] != "pass":
        raise RuntimeError("Revision-3 target audit did not pass")
    source = load_json(ROOT / "results" / "revision6_source.json")
    if source.get("schema_version") != "revision6-publication-source-v1":
        raise RuntimeError("Stale Revision-6 publication source")
    figure_measurement_audit_overview()
    figure_validity_identifiability(source)
    figure_matched_target_comparison()
    figure_reliability_reference_burden()
    write_dtw_sensitivity_source()
    print(
        "generated revision-3 method, validity, prediction, reliability, "
        "and DTW-sensitivity source data"
    )


if __name__ == "__main__":
    main()
