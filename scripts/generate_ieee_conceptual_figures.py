#!/usr/bin/env python3
"""Generate IEEE-style conceptual figures with serif fonts.

The output PDFs are intended for direct inclusion in the IEEE Access LaTeX
template. Text is rendered with Times New Roman when available.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "IEEE_Access" / "ACCESS_latex_template_20240429" / "figures_conceptual"


def configure() -> None:
    font = font_manager.findfont("Times New Roman", fallback_to_default=True)
    font_manager.fontManager.addfont(font)
    family = font_manager.FontProperties(fname=font).get_name()
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.serif": [family, "Times New Roman", "Times", "Nimbus Roman"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.03,
        }
    )


def fig_ax(width: float = 7.16, height: float = 3.6):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax, xy, wh, label=None, fc="#ffffff", ec="#1f4e79", lw=1.1, r=0.018, label_size=8.0, **kw):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle=f"round,pad=0.008,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        **kw,
    )
    ax.add_patch(patch)
    if label:
        ax.text(
            xy[0] + wh[0] / 2,
            xy[1] + wh[1] - 0.025,
            label,
            ha="center",
            va="top",
            fontsize=label_size,
            fontweight="bold",
            color=ec,
        )
    return patch


def arrow(ax, start, end, color="#333333", lw=1.25, ms=10):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def small_network(ax, cx, cy, color):
    pts = np.array(
        [
            [-0.025, 0.018],
            [0.000, 0.032],
            [0.028, 0.018],
            [0.022, -0.014],
            [-0.005, -0.030],
            [-0.032, -0.010],
        ]
    )
    pts[:, 0] += cx
    pts[:, 1] += cy
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4)]
    for i, j in edges:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color=color, lw=0.8, alpha=0.75)
    for x, y in pts:
        ax.add_patch(Circle((x, y), 0.0075, fc=color, ec="white", lw=0.4))


def matrix(ax, xy, rows=4, cols=6, cell=0.012, colors=("#dbe9f6", "#75aadb"), ec="white"):
    x0, y0 = xy
    for r in range(rows):
        for c in range(cols):
            col = colors[(r + 2 * c) % len(colors)]
            ax.add_patch(Rectangle((x0 + c * cell, y0 + r * cell), cell * 0.9, cell * 0.9, fc=col, ec=ec, lw=0.25))


def text_lines(ax, x, y, lines, size=7.5, color="#222222", ha="center", va="top", weight="normal", gap=0.035):
    for i, line in enumerate(lines):
        ax.text(x, y - i * gap, line, ha=ha, va=va, fontsize=size, color=color, fontweight=weight)


def architecture():
    fig, ax = fig_ax(7.16, 4.15)
    ax.text(0.5, 0.965, "Hybrid-PPI architecture", ha="center", va="top", fontsize=14, fontweight="bold")

    panels = [
        (0.018, 0.10, 0.105, 0.74, "Input", "#fff7ec", "#c47a00"),
        (0.148, 0.51, 0.315, 0.33, "1) Bio branch", "#f5fbf1", "#3b7d38"),
        (0.148, 0.10, 0.315, 0.36, "2) ESM branch", "#f1f6ff", "#1e63a8"),
        (0.488, 0.10, 0.120, 0.74, "3) Pair", "#fafafa", "#555555"),
        (0.632, 0.10, 0.120, 0.74, "4) Selection", "#f8f3ff", "#5a3d8a"),
        (0.778, 0.10, 0.120, 0.74, "5) Stacking", "#f3f9ff", "#225b95"),
        (0.925, 0.10, 0.058, 0.74, "Output", "#fffaf0", "#c47a00"),
    ]
    for x, y, w, h, title, fc, ec in panels:
        box(ax, (x, y), (w, h), title, fc=fc, ec=ec)

    # Input
    ax.text(0.070, 0.745, "Protein pair", ha="center", fontsize=7.4)
    small_network(ax, 0.050, 0.650, "#5ea449")
    small_network(ax, 0.091, 0.650, "#4d86d9")
    ax.text(0.071, 0.650, "+", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.050, 0.565, "P1", ha="center", fontsize=7.4)
    ax.text(0.091, 0.565, "P2", ha="center", fontsize=7.4)
    ax.plot([0.033, 0.108], [0.515, 0.515], color="#d0a060", lw=0.6, ls="--")
    text_lines(ax, 0.070, 0.455, ["Sequences", "motif scans"], size=7.2, color="#9a5b00", weight="bold")
    box(ax, (0.036, 0.30), (0.068, 0.055), None, fc="#ffffff", ec="#d9a24a", lw=0.7, r=0.006)
    ax.text(0.072, 0.33, "Protein P1", ha="center", va="center", fontsize=6.7)
    box(ax, (0.036, 0.205), (0.068, 0.055), None, fc="#ffffff", ec="#d9a24a", lw=0.7, r=0.006)
    ax.text(0.072, 0.23, "Protein P2", ha="center", va="center", fontsize=6.7)

    # Biological branch
    feats = [("AAC", "20"), ("DPC", "400"), ("CTD", "147"), ("Moran", "343"), ("PseAAC", "50"), ("ELM", "motifs")]
    xs = np.linspace(0.185, 0.425, len(feats))
    for i, (name, dim) in enumerate(feats):
        x = xs[i]
        ax.text(x, 0.765, name, ha="center", fontsize=6.1)
        matrix(ax, (x - 0.017, 0.665), rows=3, cols=3, cell=0.0115, colors=("#d8ead0", "#6aa84f"))
        ax.text(x, 0.632, dim, ha="center", fontsize=6.8)
    ax.plot([0.18, 0.43], [0.575, 0.575], color="#3b7d38", lw=0.8)
    ax.text(0.305, 0.538, "Concatenated biological vector", ha="center", fontsize=7.0)
    ax.text(0.305, 0.505, "D bio = 960", ha="center", fontsize=7.6, fontweight="bold")

    # Deep branch
    ax.text(0.186, 0.385, "ESM-2", ha="center", fontsize=8.0, fontweight="bold")
    ax.text(0.186, 0.352, "residue matrix", ha="center", fontsize=6.9)
    matrix(ax, (0.169, 0.205), rows=4, cols=5, cell=0.0115)
    arrow(ax, (0.218, 0.300), (0.268, 0.300), color="#1e63a8")
    box(ax, (0.270, 0.300), (0.165, 0.075), None, fc="#ffffff", ec="#5c8fca", lw=0.8, r=0.006)
    ax.text(0.352, 0.347, "Global mean pool", ha="center", fontsize=6.9)
    ax.text(0.352, 0.320, "v global (D)", ha="center", fontsize=7.2)
    box(ax, (0.270, 0.185), (0.165, 0.075), None, fc="#ffffff", ec="#5c8fca", lw=0.8, r=0.006)
    ax.text(0.352, 0.232, "Motif-anchored pool", ha="center", fontsize=6.9)
    ax.text(0.352, 0.205, "v local (2D)", ha="center", fontsize=7.2)
    ax.text(0.305, 0.140, "[v global; v local] = 3D", ha="center", fontsize=7.8, fontweight="bold", color="#173a70")

    # Pair construction
    text_lines(ax, 0.548, 0.760, ["For each", "branch pair"], size=6.9)
    box(ax, (0.510, 0.575), (0.075, 0.095), None, fc="#ffffff", ec="#777777", lw=0.8, r=0.006)
    ax.text(0.548, 0.630, "Hadamard", ha="center", fontsize=6.9)
    ax.text(0.548, 0.602, "x1 * x2", ha="center", fontsize=7.2)
    box(ax, (0.510, 0.405), (0.075, 0.095), None, fc="#ffffff", ec="#777777", lw=0.8, r=0.006)
    ax.text(0.548, 0.460, "Abs diff.", ha="center", fontsize=6.9)
    ax.text(0.548, 0.432, "abs(x1-x2)", ha="center", fontsize=7.2)
    ax.text(0.548, 0.295, "Concatenate", ha="center", fontsize=7.0)
    ax.text(0.548, 0.260, "pair vector", ha="center", fontsize=6.9)

    # Selection
    stages = [("Variance", "remove constants"), ("Importance", "cumulative gain"), ("Correlation", "remove redundancy")]
    y = 0.72
    for title, sub in stages:
        box(ax, (0.652, y - 0.07), (0.086, 0.105), None, fc="#ffffff", ec="#6d4fa0", lw=0.8, r=0.006)
        ax.text(0.695, y - 0.015, title, ha="center", fontsize=6.9, fontweight="bold", color="#4d2a7a")
        ax.text(0.695, y - 0.045, sub, ha="center", fontsize=5.9)
        if y > 0.37:
            arrow(ax, (0.695, y - 0.08), (0.695, y - 0.155), color="#6d4fa0", ms=8)
        y -= 0.22
    ax.text(0.695, 0.18, "inside each CV fold", ha="center", fontsize=6.8, color="#4d2a7a", fontweight="bold")

    # Stacking
    box(ax, (0.795, 0.63), (0.04, 0.12), None, fc="#ffffff", ec="#2c7a3f", lw=0.8, r=0.006)
    box(ax, (0.852, 0.63), (0.04, 0.12), None, fc="#ffffff", ec="#1e63a8", lw=0.8, r=0.006)
    ax.text(0.815, 0.705, "Bio", ha="center", fontsize=7.0, fontweight="bold")
    ax.text(0.815, 0.675, "LGBM", ha="center", fontsize=6.8)
    ax.text(0.872, 0.705, "ESM", ha="center", fontsize=7.0, fontweight="bold")
    ax.text(0.872, 0.675, "LGBM", ha="center", fontsize=6.8)
    arrow(ax, (0.815, 0.62), (0.815, 0.54), color="#2c7a3f")
    arrow(ax, (0.872, 0.62), (0.872, 0.54), color="#1e63a8")
    ax.text(0.843, 0.52, "5-fold OOF", ha="center", fontsize=6.8)
    box(ax, (0.805, 0.37), (0.075, 0.065), None, fc="#ffffff", ec="#777777", lw=0.8, r=0.006)
    matrix(ax, (0.813, 0.39), rows=1, cols=8, cell=0.008, colors=("#76b852", "#78a7e2"))
    arrow(ax, (0.843, 0.36), (0.843, 0.29), color="#555555")
    box(ax, (0.795, 0.19), (0.095, 0.08), None, fc="#ffffff", ec="#225b95", lw=0.8, r=0.006)
    ax.text(0.843, 0.232, "Regularized LR", ha="center", fontsize=7.4)
    ax.text(0.843, 0.205, "meta-learner", ha="center", fontsize=7.2)

    # Output
    ax.text(0.954, 0.70, "Interaction", ha="center", fontsize=6.8)
    ax.text(0.954, 0.665, "probability", ha="center", fontsize=6.8)
    box(ax, (0.943, 0.50), (0.025, 0.13), None, fc="#ffffff", ec="#d8aa5c", lw=0.8, r=0.004)
    ax.add_patch(Rectangle((0.947, 0.505), 0.017, 0.04, fc="#f4bc60", ec="none"))
    ax.add_patch(Rectangle((0.947, 0.545), 0.017, 0.05, fc="#f7d89f", ec="none"))
    ax.text(0.954, 0.42, "threshold", ha="center", fontsize=6.8, fontweight="bold")
    ax.text(0.954, 0.365, "interaction", ha="center", fontsize=6.6)
    ax.text(0.954, 0.330, "label", ha="center", fontsize=6.6)

    for start, end, col in [
        ((0.13, 0.50), (0.155, 0.50), "#c47a00"),
        ((0.455, 0.70), (0.48, 0.68), "#3b7d38"),
        ((0.455, 0.29), (0.48, 0.36), "#1e63a8"),
        ((0.61, 0.50), (0.635, 0.50), "#555555"),
        ((0.755, 0.50), (0.78, 0.50), "#6d4fa0"),
        ((0.90, 0.50), (0.925, 0.50), "#225b95"),
    ]:
        arrow(ax, start, end, color=col, lw=1.35, ms=11)

    fig.savefig(OUT / "hybridppi_architecture_v3_pipeline.pdf")
    fig.savefig(OUT / "hybridppi_architecture_v3_pipeline.png", dpi=300)
    plt.close(fig)


def motif_anchor():
    fig, ax = fig_ax(7.16, 4.25)
    ax.text(0.5, 0.965, "Motif-anchored ESM-2 pooling", ha="center", va="top", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.885, "Local motif context is pooled separately from the global protein context.", ha="center", fontsize=8.8)

    step_marks = [
        (0.035, 0.715, "1", "#1e63a8"),
        (0.205, 0.715, "2", "#b42222"),
        (0.385, 0.715, "3", "#2f7d5c"),
        (0.705, 0.715, "4", "#1e63a8"),
        (0.870, 0.715, "5", "#6d2e8c"),
    ]
    for x, y, lab, col in step_marks:
        ax.add_patch(Circle((x, y), 0.017, fc=col, ec="none"))
        ax.text(x, y, lab, ha="center", va="center", color="white", fontsize=8.0, fontweight="bold")

    # Step 1
    box(ax, (0.025, 0.54), (0.13, 0.17), None, fc="#ffffff", ec="#b8cbe8", lw=0.8, r=0.006)
    text_lines(ax, 0.09, 0.675, ["Protein sequence", "L residues"], size=7.4)
    arrow(ax, (0.09, 0.535), (0.09, 0.47), color="#1e63a8")
    matrix(ax, (0.045, 0.29), rows=6, cols=7, cell=0.013, colors=("#d9e6f5", "#7aa6dc"))
    ax.text(0.09, 0.245, "E matrix (L x D)", ha="center", fontsize=8.4, fontweight="bold", color="#1e63a8")

    # Step 2
    seq_y = [0.66, 0.60, 0.54]
    for y in seq_y:
        ax.text(0.20, y, "MKTIIALSYIFCLVFA...", ha="left", va="center", fontsize=6.8)
        ax.add_patch(Rectangle((0.245, y - 0.014), 0.052, 0.028, fc="#f6c6c6", ec="#b42222", lw=0.6))
        ax.text(0.271, y, "motif", ha="center", va="center", fontsize=6.6, color="#8f1515", fontweight="bold")
    box(ax, (0.205, 0.35), (0.14, 0.10), None, fc="#fffafa", ec="#b42222", lw=0.8, r=0.006)
    text_lines(ax, 0.275, 0.425, ["detected spans", "(start, end)"], size=7.2, color="#8f1515")

    # Step 3 local
    box(ax, (0.385, 0.59), (0.16, 0.15), None, fc="#f8fffb", ec="#2f7d5c", lw=0.9, r=0.006)
    matrix(ax, (0.405, 0.625), rows=4, cols=7, cell=0.0115, colors=("#cce4d6", "#5a9b76"))
    ax.text(0.465, 0.60, "rows from motif spans", ha="center", fontsize=7.1)
    box(ax, (0.385, 0.35), (0.16, 0.15), None, fc="#fff9ed", ec="#b27a12", lw=0.9, r=0.006)
    matrix(ax, (0.405, 0.385), rows=4, cols=7, cell=0.0115, colors=("#f6df9f", "#d39a18"))
    ax.text(0.465, 0.36, "fallback central region", ha="center", fontsize=7.1, color="#8a5b00")
    ax.text(0.465, 0.53, "if motifs exist", ha="center", fontsize=7.4, color="#2f7d5c", fontweight="bold")
    ax.text(0.465, 0.29, "if no motif exists", ha="center", fontsize=7.4, color="#8a5b00", fontweight="bold")
    box(ax, (0.565, 0.50), (0.12, 0.16), None, fc="#ffffff", ec="#2f7d5c", lw=0.9, r=0.006)
    ax.text(0.625, 0.605, "max + mean", ha="center", fontsize=7.5, color="#2f7d5c", fontweight="bold")
    ax.text(0.625, 0.570, "pooling", ha="center", fontsize=7.5, color="#2f7d5c", fontweight="bold")
    ax.text(0.625, 0.525, "v local (2D)", ha="center", fontsize=8.0)

    # Step 4 global
    box(ax, (0.715, 0.47), (0.13, 0.20), None, fc="#f6faff", ec="#1e63a8", lw=0.9, r=0.006)
    matrix(ax, (0.738, 0.555), rows=1, cols=7, cell=0.0125, colors=("#d9e6f5", "#7aa6dc"))
    ax.text(0.78, 0.625, "mean over", ha="center", fontsize=7.3)
    ax.text(0.78, 0.595, "all residues", ha="center", fontsize=7.3)
    ax.text(0.78, 0.485, "v global (D)", ha="center", fontsize=8.0)

    # Step 5 final
    box(ax, (0.885, 0.42), (0.095, 0.26), None, fc="#fffaff", ec="#6d2e8c", lw=0.9, r=0.006)
    matrix(ax, (0.902, 0.58), rows=1, cols=5, cell=0.010, colors=("#7aa6dc", "#d9e6f5"))
    matrix(ax, (0.902, 0.52), rows=1, cols=8, cell=0.010, colors=("#5a9b76", "#d7eadf"))
    ax.text(0.932, 0.47, "[global; local]", ha="center", fontsize=7.4)
    ax.text(0.932, 0.435, "= 3D", ha="center", fontsize=9.2, fontweight="bold", color="#6d2e8c")

    for s, e, c in [
        ((0.155, 0.39), (0.195, 0.58), "#333333"),
        ((0.345, 0.60), (0.385, 0.665), "#b42222"),
        ((0.345, 0.42), (0.385, 0.425), "#b42222"),
        ((0.545, 0.665), (0.565, 0.59), "#2f7d5c"),
        ((0.545, 0.425), (0.565, 0.555), "#b27a12"),
        ((0.685, 0.58), (0.715, 0.57), "#333333"),
        ((0.845, 0.57), (0.885, 0.56), "#333333"),
        ((0.685, 0.55), (0.885, 0.50), "#333333"),
    ]:
        arrow(ax, s, e, color=c, lw=1.0, ms=9)

    box(ax, (0.075, 0.06), (0.85, 0.085), None, fc="#ffffff", ec="#777777", lw=0.8, r=0.006)
    ax.text(
        0.5,
        0.102,
        "Key idea: global pooling summarizes the whole protein; motif pooling preserves local binding-site evidence.",
        ha="center",
        va="center",
        fontsize=8.0,
    )

    fig.savefig(OUT / "motif_anchor.pdf")
    fig.savefig(OUT / "motif_anchor.png", dpi=300)
    plt.close(fig)


def feature_selection():
    fig, ax = fig_ax(7.16, 3.25)
    ax.text(0.5, 0.950, "Fold-local feature selection cascade", ha="center", va="top", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.850, "Feature ranking is fitted only on training folds; selected columns are then applied to the held-out fold.", ha="center", fontsize=8.4)

    # Fold split
    box(ax, (0.035, 0.38), (0.13, 0.30), "CV fold", fc="#f8f8f8", ec="#555555")
    ax.text(0.10, 0.58, "Train folds", ha="center", fontsize=8.2, fontweight="bold")
    for i in range(4):
        ax.add_patch(Rectangle((0.055 + i * 0.023, 0.50), 0.018, 0.055, fc="#8dbf75", ec="#4b8b39", lw=0.5))
    ax.text(0.10, 0.445, "Held-out fold", ha="center", fontsize=8.0)
    ax.add_patch(Rectangle((0.077, 0.40), 0.045, 0.03, fc="#f4c7c3", ec="#b42222", lw=0.5))

    steps = [
        ("1) Variance filter", "remove constant or near-constant features", "#fff6cc", "#c49a00"),
        ("2) LightGBM importance", "retain cumulative-gain top features", "#e4f3df", "#4f8f45"),
        ("3) Correlation pruning", "drop redundant pairs by |rho| threshold", "#e8f1fb", "#3c78b5"),
    ]
    xs = [0.25, 0.49, 0.73]
    for x, (title, sub, fc, ec) in zip(xs, steps):
        box(ax, (x - 0.085, 0.34), (0.17, 0.35), None, fc=fc, ec=ec, lw=1.0, r=0.010)
        ax.text(x, 0.615, title, ha="center", fontsize=8.2, fontweight="bold", color=ec)
    wraps = [
        ["remove constant or", "near-constant features"],
        ["retain cumulative-gain", "top features"],
        ["drop redundant pairs", "by |rho| threshold"],
    ]
    for x, lines in zip(xs, wraps):
        for i, line in enumerate(lines):
            ax.text(x, 0.475 - i * 0.042, line, ha="center", fontsize=7.2)

    box(ax, (0.90, 0.38), (0.075, 0.27), "Selected", fc="#eef8e9", ec="#4f8f45")
    ax.text(0.938, 0.505, "feature set", ha="center", fontsize=8.2, fontweight="bold")

    for s, e in [
        ((0.165, 0.515), (0.165 + 0.06, 0.515)),
        ((0.335, 0.515), (0.405, 0.515)),
        ((0.575, 0.515), (0.645, 0.515)),
        ((0.815, 0.515), (0.90, 0.515)),
    ]:
        arrow(ax, s, e, color="#333333", lw=1.2, ms=10)

    ax.plot([0.10, 0.10, 0.938], [0.35, 0.22, 0.22], color="#b42222", lw=0.9, ls="--")
    arrow(ax, (0.938, 0.22), (0.938, 0.38), color="#b42222", lw=0.9, ms=8)
    ax.text(0.50, 0.17, "No leakage: the held-out fold is never used to fit variance, importance, or correlation thresholds.", ha="center", fontsize=8.4, color="#8f1515", fontweight="bold")

    fig.savefig(OUT / "feature_selection_cascade.pdf")
    fig.savefig(OUT / "feature_selection_cascade.png", dpi=300)
    plt.close(fig)


def same_go():
    fig, ax = fig_ax(7.16, 3.35)
    ax.text(0.5, 0.950, "Same-GO hard-negative dataset", ha="center", va="top", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.850, "Same-GO negative pairs share GO biological context but are not recorded as BioGRID interactions.", ha="center", fontsize=8.6)

    cards = [
        (0.035, "Positive PPI", "#eaf5e4", "#2f6b1f", "label = 1"),
        (0.355, "Random negative", "#fff3e8", "#c15d00", "label = 0"),
        (0.675, "Same-GO negative", "#fff0f0", "#a00000", "label = 0"),
    ]
    for x, title, fc, ec, lab in cards:
        box(ax, (x, 0.22), (0.29, 0.57), None, fc=fc, ec=ec, lw=1.0, r=0.010)
        ax.add_patch(Rectangle((x, 0.72), 0.29, 0.07, fc=ec, ec=ec, lw=0))
        ax.text(x + 0.145, 0.755, title, ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        box(ax, (x + 0.10, 0.245), (0.09, 0.055), None, fc="#ffffff", ec=ec, lw=0.8, r=0.006)
        ax.text(x + 0.145, 0.272, lab, ha="center", va="center", fontsize=8.4, fontweight="bold")

    # Positive
    small_network(ax, 0.105, 0.55, "#4d86d9")
    small_network(ax, 0.245, 0.55, "#6aa84f")
    ax.plot([0.13, 0.22], [0.55, 0.55], color="#2f6b1f", lw=1.4)
    ax.text(0.175, 0.575, "known interaction", ha="center", fontsize=8.0, color="#2f6b1f")
    ax.text(0.105, 0.47, "Protein A", ha="center", fontsize=8.0)
    ax.text(0.245, 0.47, "Protein B", ha="center", fontsize=8.0)
    ax.text(0.175, 0.38, "BioGRID physical interaction", ha="center", fontsize=8.0, color="#2f6b1f", fontweight="bold")

    # Random negative
    small_network(ax, 0.425, 0.55, "#4d86d9")
    small_network(ax, 0.565, 0.55, "#d49a1f")
    ax.plot([0.45, 0.54], [0.55, 0.55], color="#333333", lw=1.1, ls="--")
    ax.text(0.495, 0.55, "X", ha="center", va="center", fontsize=12, color="#a00000", fontweight="bold")
    ax.text(0.425, 0.47, "GO: DNA repair", ha="center", fontsize=7.6, color="#1e63a8")
    ax.text(0.565, 0.47, "GO: metabolism", ha="center", fontsize=7.6, color="#9a5b00")
    ax.text(0.495, 0.38, "different GO context", ha="center", fontsize=8.0, color="#c15d00", fontweight="bold")

    # Same-GO
    small_network(ax, 0.745, 0.55, "#4d86d9")
    small_network(ax, 0.885, 0.55, "#7d4fb3")
    ax.plot([0.77, 0.86], [0.55, 0.55], color="#333333", lw=1.1, ls="--")
    ax.text(0.815, 0.55, "X", ha="center", va="center", fontsize=12, color="#a00000", fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.722, 0.615), 0.185, 0.065, boxstyle="round,pad=0.006,rounding_size=0.03", fc="#ffffff", ec="#7d4fb3", lw=1.0, ls="--"))
    ax.text(0.815, 0.648, "shared BP/MF GO term", ha="center", va="center", fontsize=8.2, color="#6d2e8c", fontweight="bold")
    ax.text(0.815, 0.435, "same GO context", ha="center", fontsize=8.0, color="#6d2e8c", fontweight="bold")
    ax.text(0.815, 0.39, "no BioGRID interaction", ha="center", fontsize=8.0, color="#a00000", fontweight="bold")

    fig.savefig(OUT / "same_go_dataset.pdf")
    fig.savefig(OUT / "same_go_dataset.png", dpi=300)
    plt.close(fig)


def main() -> None:
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    motif_anchor()
    feature_selection()
    same_go()


if __name__ == "__main__":
    main()
