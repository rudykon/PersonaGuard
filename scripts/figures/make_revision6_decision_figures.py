#!/usr/bin/env python3
"""Create the Revision-6 threshold-sensitivity and fold-sensitivity figure.

Publication-facing numerical data are read only from
``results/revision6_source.json``. The curves are aggregate empirical survival
and degradation functions of estimated participant-mean gains. They are
descriptive threshold sensitivities because no meaningful-improvement or
acceptable-degradation threshold was prespecified.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from vector_export import save_pdfua_embed
from manuscript_fonts import with_manuscript_fonts


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "artifacts" / "figures"
DATA_DIR = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "revision6_source.json"
STEM = "decision_boundary_sensitivity"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7A5195"
GRAY = "#5F6368"
LIGHT_GRAY = "#D9DDE3"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.25,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    }
)


def load_source() -> dict[str, object]:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source.get("schema_version") != "revision6-publication-source-v1":
        raise RuntimeError("Stale Revision-6 publication source")
    if source.get("revision") != 6:
        raise RuntimeError("Revision 6 is not the active publication source")
    return source


def select(
    rows: list[dict[str, str]],
    *,
    analysis: str,
    direction: str,
    budget: int | None = None,
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["analysis"] == analysis and row["direction"] == direction
        and (budget is None or int(row["budget"]) == budget)
    ]
    return sorted(selected, key=lambda row: float(row["threshold"]))


def vectors(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([float(row["threshold"]) for row in rows])
    count = np.asarray([int(row["count"]) for row in rows])
    total = np.asarray([int(row["participants"]) for row in rows])
    proportion = np.asarray([float(row["proportion"]) for row in rows])
    if len(x) != 41 or not np.all(np.diff(x) > 0):
        raise RuntimeError("Expected the fixed 41-point threshold grid")
    if not np.all(total == 24) or not np.allclose(proportion, count / total):
        raise RuntimeError("Threshold rows must report exact counts out of 24")
    if any("wilson" in key.lower() for row in rows for key in row):
        raise RuntimeError("Publication-facing threshold rows must contain no bands")
    return x, count


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
    )


def plot_sensing(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    configurations = [
        ("meets_gain_threshold", "Gain ≥ threshold", BLUE, "-"),
        ("exceeds_degradation_threshold", "Gain ≤ −threshold", ORANGE, "--"),
    ]
    for direction, label, color, style in configurations:
        x, count = vectors(
            select(
                rows,
                analysis="sensing_increment_vs_video_mean",
                direction=direction,
            )
        )
        ax.plot(
            x, count, color=color, linestyle=style, drawstyle="steps-post",
            label=label,
        )
    ax.set(
        title="Scalar-sensing gain counts",
        xlabel="Hypothetical MAE-gain threshold (s)",
        ylabel="Number of participants",
        xlim=(0, 0.2),
        ylim=(0, 24.5),
        yticks=(0, 6, 12, 18, 24),
    )
    ax.legend(loc="upper right", fontsize=6.3)
    panel_label(ax, "(A)")


def plot_calibration(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    direction: str,
    title: str,
    panel: str,
) -> None:
    colors = {1: BLUE, 2: GREEN, 4: PURPLE, 8: ORANGE}
    styles = {1: "-", 2: "--", 4: "-.", 8: ":"}
    for budget in (1, 2, 4, 8):
        x, count = vectors(
            select(
                rows,
                analysis="signed_calibration_vs_video_only",
                direction=direction,
                budget=budget,
            )
        )
        ax.plot(
            x,
            count,
            color=colors[budget],
            linestyle=styles[budget],
            drawstyle="steps-post",
            label=f"{budget} video" + ("s" if budget > 1 else ""),
        )
    ax.set(
        title=title,
        xlabel="Hypothetical MAE-gain threshold\n(joystick label units)",
        ylabel="Number of participants",
        xlim=(0, 0.2),
        ylim=(0, 24.5),
        yticks=(0, 6, 12, 18, 24),
    )
    ax.legend(loc="upper right", ncol=2, fontsize=6.3, columnspacing=0.8)
    panel_label(ax, f"({panel})")


def plot_repeated_cv(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    stack = sorted(
        (
            row
            for row in rows
            if row["method"] == "context_plus_blockwise_residual"
        ),
        key=lambda row: int(row["repeat"]),
    )
    if len(stack) < 2:
        raise RuntimeError("Repeated grouped-CV sensitivity is missing")
    repeat = np.asarray([int(row["repeat"]) for row in stack])
    gain = np.asarray([float(row["gain_vs_video_mean_seconds"]) for row in stack])
    low = np.asarray([float(row["gain_vs_video_mean_ci95_low"]) for row in stack])
    high = np.asarray([float(row["gain_vs_video_mean_ci95_high"]) for row in stack])
    assert len(stack) == 5 and list(repeat) == list(range(5))
    for y, x, lo, hi in zip(repeat, gain, low, high):
        ax.errorbar(x, y, xerr=np.asarray([[x-lo], [hi-x]]), fmt="o",
                    color=BLUE, ecolor=GRAY, elinewidth=.9, capsize=2.2,
                    markersize=4, zorder=3)
    ax.axvline(0, color=GRAY, linewidth=.8, linestyle="--", zorder=0)
    ax.set(title="Complete grouped-CV allocations",
           xlabel="Sensor-stack gain over video mean (s)",
           yticks=repeat,
           yticklabels=[f"Allocation {r}" + (" (primary)" if r==0 else "") for r in repeat],
           ylim=(4.6,-.6))
    panel_label(ax, "(D)")


def write_source_data(
    decision_rows: list[dict[str, str]], repeated_rows: list[dict[str, str]]
) -> None:
    path = DATA_DIR / f"source_data_{STEM}.csv"
    fields = [
        "panel", "analysis", "route", "budget", "direction", "threshold",
        "participants", "count", "proportion", "repeat", "method",
        "participant_macro_mae_seconds", "gain_vs_video_mean_seconds",
        "gain_vs_video_mean_ci95_low", "gain_vs_video_mean_ci95_high",
        "partition_sha256",
    ]
    materialized: list[dict[str, object]] = []
    for row in decision_rows:
        panel = "A" if row["analysis"] == "sensing_increment_vs_video_mean" else (
            "B" if row["direction"] == "meets_gain_threshold" else "C"
        )
        materialized.append({"panel": panel, **row})
    for row in repeated_rows:
        if row["method"] == "context_plus_blockwise_residual":
            materialized.append({"panel": "D", **row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def write_qa(source: dict[str, object]) -> None:
    repeated = source["summaries"]["revision6"][
        "primary_antialias_reservation_sensing"
    ]["repeated_grouped_cv_sensitivity"]
    text = f"""# Revision 6 threshold-sensitivity figure contract and QA

- **Core conclusion:** route-level personalization should be withheld when incremental value, stability, user consequences, or governance evidence is insufficient.
- **Evidence archetype:** 2x2 quantitative grid. Aggregate threshold sensitivity is the hero evidence; repeated grouped CV is the stability audit.
- **Panel map:** A, sensing benefit/degradation survival functions; B, calibration benefit survival functions; C, calibration degradation survival functions; D, complete grouped-CV fold-allocation repeats.
- **Evidence hierarchy:** estimated participant-mean gains and exact integer counts are primary; threshold curves are descriptive summaries without uncertainty bands; fold repeats test allocation sensitivity.
- **Numerical source:** `results/revision6_source.json` only; aggregate source CSVs are exported beside the plotting code.
- **Unit and design:** participant (`n=24`); five outer participant folds and four inner participant folds.
- **Fold sensitivity:** {repeated['repeats']} participant partitions; reference, target, scaling, and model selection rebuilt per partition.
- **Uncertainty:** Panels A--C deliberately show no interval because a complete resampling propagation across model fitting, target construction, calibration subsets, shared references, and fold allocation is unavailable. Panel D shows participant-bootstrap percentile intervals for each complete grouped-CV allocation.
- **Decision boundary:** all thresholds are hypothetical because none was prespecified. No panel claims equivalence, practical utility, user benefit, or an acceptable harm rate.
- **Reviewer risk controlled:** step curves expose the discrete 24-participant distribution; units are route-specific; benefit and degradation are separated; uncertainty limits are stated in the caption and SI.
- **Visual and accessibility checks:** no dual y-axis or rainbow palette; color has redundant line style; Panel D has a zero reference; legends avoid dense regions; text remains legible at 7.17-inch double-column width.
- **Exports:** editable text-bearing SVG/PDF masters, a font-free vector PDF/UA embed, 600-dpi PNG/TIFF, and aggregate source-data CSV.
"""
    report_dir = ROOT / "artifacts" / "paper_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "revision6_decision_figure_QA.md").write_text(text, encoding="utf-8")


@with_manuscript_fonts
def main() -> None:
    source = load_source()
    decision_rows = source["tables"]["decision_threshold_curves"]
    repeated_rows = source["tables"]["repeated_grouped_cv"]
    fig, axes = plt.subplots(2, 2, figsize=(7.17, 5.9))
    plot_sensing(axes[0, 0], decision_rows)
    plot_calibration(
        axes[0, 1], decision_rows,
        direction="meets_gain_threshold",
        title="Calibration: estimated gains ≥ δ",
        panel="B",
    )
    plot_calibration(
        axes[1, 0], decision_rows,
        direction="exceeds_degradation_threshold",
        title="Calibration: estimated gains ≤ −δ",
        panel="C",
    )
    plot_repeated_cv(axes[1, 1], repeated_rows)
    for ax in axes.flat:
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.55, alpha=0.65)
        ax.set_axisbelow(True)
    fig.text(
        0.50,
        0.012,
        "A–C: descriptive counts, no uncertainty bands; no deployment threshold was prespecified.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=GRAY,
    )
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.11, top=0.95, wspace=0.62, hspace=0.52)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    common = {}
    fig.savefig(FIGURE_DIR / f"{STEM}.svg", **common)
    fig.savefig(FIGURE_DIR / f"{STEM}.pdf", **common)
    fig.savefig(FIGURE_DIR / f"{STEM}.png", dpi=600, **common)
    fig.savefig(FIGURE_DIR / f"{STEM}.tiff", dpi=600, **common)
    save_pdfua_embed(fig, FIGURE_DIR / f"{STEM}_embed.pdf", common)
    from PIL import Image
    with Image.open(FIGURE_DIR / f"{STEM}.png") as im:
        im.convert("L").save(FIGURE_DIR / f"{STEM}_grayscale.png", dpi=(600, 600))
    plt.close(fig)
    write_source_data(decision_rows, repeated_rows)
    write_qa(source)
    print(f"Wrote {STEM} figure, source data, and QA record")


if __name__ == "__main__":
    main()
