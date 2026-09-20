#!/usr/bin/env python3
"""Plot the three authorized examples from the sole publication-facing source.

Contract: illustrative quantitative grid, not matched-information ranking.
All seconds, both axes, and all five curves are retained for each selected trial.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from vector_export import save_pdfua_embed
from manuscript_fonts import with_manuscript_fonts

ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "artifacts/figures"
DATA_DIR = Path(__file__).resolve().parent
SOURCE = ROOT / "results/revision6_source.json"
STEM = "dense_trajectory_examples"
FIGURE_WIDTH_MM = 182.1
RASTER_DPI = 600

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7.0, "axes.labelsize": 7.0,
    "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "legend.frameon": False,
    "path.simplify": False, "savefig.facecolor": "white",
})

STYLE = {
    "target": ("Observed report", "#172B4D", "-", 1.5),
    "metadata": ("Metadata prior", "#7A818B", (0, (3, 2)), 1.05),
    "content": ("Content prior", "#3D6FB6", "-.", 1.5),
    "residual": ("Full residual candidate", "#D9822B", (0, (6, 2, 1, 2)), 1.05),
    "sam": ("Post-trial SAM", "#756BB1", (0, (5, 2)), 1.35),
}


@with_manuscript_fonts
def make_figure(source):
    release = source["summaries"]["dense_trajectory_examples"]
    examples = release["examples"]
    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH_MM / 25.4, 4.65))
    fig.subplots_adjust(left=.085, right=.985, top=.750, bottom=.220,
                        wspace=.20, hspace=.26)
    handles = [Line2D([], [], label=label, color=color, linestyle=line, linewidth=width)
               for label, color, line, width in STYLE.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .995),
               ncol=3, fontsize=6.6, columnspacing=1.5, handlelength=3.0)
    for col, example in enumerate(examples):
        time = np.asarray(example["time_seconds"])
        axes[0, col].set_title(
            f"({example['label']})  Example {col + 1}\n"
            f"Content-error percentile: {example['percentile']}",
            loc="left", fontsize=7.3, pad=10, fontweight="bold")
        for dimension in (0, 1):
            ax = axes[dimension, col]
            # Target is drawn last so the observed report remains identifiable.
            for name in ("metadata", "content", "residual", "sam", "target"):
                label, color, line, width = STYLE[name]
                values = np.asarray(example["curves"][name], dtype=float)
                if values.shape != (len(time), 2) or not np.isfinite(values).all():
                    raise ValueError("Incomplete figure coordinates")
                ax.plot(time, values[:, dimension], color=color,
                        linestyle=line, linewidth=width, label=label)
            ax.set_ylim(1, 255)
            ax.set_xlim(time[0], time[-1])
            ax.set_yticks([1, 128, 255])
            ax.set_xticks([0, 60, 120] if time[-1] >= 120 else [0, 50, 100])
            ax.grid(axis="y", color="#D9E1E8", linewidth=.45)
            if col:
                ax.tick_params(axis="y", labelleft=False)
            else:
                ax.set_ylabel(("Valence", "Arousal")[dimension] + " (1–255)")
            if dimension:
                ax.set_xlabel("Time within trial (s)")
            else:
                ax.tick_params(axis="x", labelbottom=False)
    fig.text(.53, .130,
             "Metadata: metadata only  |  Content: full video  |  Residual: video + current/past EEG/fNIRS",
             ha="center", fontsize=6.2, color="#5B6573")
    fig.text(.53, .090,
             "SAM: post-trial rating + same-video training library; participant holdout only.",
             ha="center", fontsize=6.3, color="#5B6573")
    fig.text(.53, .032,
             "Different information conditions—not a common ranking. Residual can overlap content.\n"
             "Examples aid interpretation; aggregate evidence and decisions remain in the results table.",
             ha="center", fontsize=6.2, color="#5B6573", linespacing=1.4)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in (".svg", ".pdf", ".png", ".tiff"):
        kwargs = {"dpi": RASTER_DPI} if extension in (".png", ".tiff") else {}
        fig.savefig(FIGURE_DIR / (STEM + extension), **kwargs)
    save_pdfua_embed(fig, FIGURE_DIR / (STEM + "_embed.pdf"), {})
    from PIL import Image
    with Image.open(FIGURE_DIR / (STEM + ".png")) as preview:
        preview.convert("L").save(FIGURE_DIR / (STEM + "_grayscale.png"), dpi=(600, 600))
    plt.close(fig)


def write_coordinates(source):
    fields = ["example", "percentile", "time_seconds", "axis", *STYLE]
    with (DATA_DIR / ("source_data_" + STEM + ".csv")).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for example in source["summaries"]["dense_trajectory_examples"]["examples"]:
            for index, timestamp in enumerate(example["time_seconds"]):
                for dimension, axis in enumerate(("valence", "arousal")):
                    row = {"example": example["label"], "percentile": example["percentile"],
                           "time_seconds": timestamp, "axis": axis}
                    row.update({key: example["curves"][key][index][dimension] for key in STYLE})
                    writer.writerow(row)


def main():
    source = json.loads(SOURCE.read_text())
    make_figure(source)
    write_coordinates(source)
    print("Generated Figure 5 from authorized anonymous examples in the canonical source.")


if __name__ == "__main__":
    main()
