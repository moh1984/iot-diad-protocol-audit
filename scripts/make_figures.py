"""Generate Figures 1-3 from the CSVs in ../results/.

Figure 1  share distortion      <- true_class_inventory.csv
Figure 2  variance strip plot   <- rev1_protocol_20seeds_raw.csv (same run as Table 2)
Figure 3  depth curve           <- rev4_depth_sweep_20seeds_summary.csv

Run:  python scripts/make_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
HERE = os.path.dirname(os.path.abspath(__file__))
U = os.path.join(HERE, "..", "results") + os.sep
OUT = os.path.join(HERE, "..", "figures") + os.sep
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 300, "savefig.bbox": "tight",
})

INK = "#1a1a1a"
C_RANDOM = "#4C72B0"
C_CAPTURE = "#C44E52"

# -------------------------------------------- Figure 1: share distortion
inv = pd.read_csv(U + "true_class_inventory.csv")
inv["file_share"] = 100 * inv.csv_files / inv.csv_files.sum()
inv = inv.sort_values("true_pct", ascending=True)

fig, ax = plt.subplots(figsize=(5.6, 3.1))
ypos = np.arange(len(inv)); h = 0.38
ax.barh(ypos + h/2, inv.true_pct, h, color=INK, label="True share of flows")
ax.barh(ypos - h/2, inv.file_share, h, color="#BFBFBF",
        label="Share of capture files")
ax.set_yticks(ypos); ax.set_yticklabels(inv["class"])
ax.set_xlabel("Percent of corpus")
ax.legend(frameon=False, fontsize=8, loc="lower right")
for i, r in enumerate(inv.itertuples()):
    if r.true_pct > 5 or r.file_share > 5:
        ratio = r.file_share / r.true_pct
        if ratio > 3 or ratio < 0.4:
            ax.annotate(f"{ratio:.0f}x" if ratio > 1 else f"{1/ratio:.1f}x under",
                        xy=(max(r.true_pct, r.file_share) + 1.5, i),
                        fontsize=7, va="center", color=INK)
ax.set_title("Per-file sampling reproduces the file-share column,\n"
             "not the true distribution", fontsize=9.5, loc="left")
fig.savefig(OUT + "fig1_share_distortion.pdf")
fig.savefig(OUT + "fig1_share_distortion.png")
plt.close(fig)

# ----------------------------------------------------- Figure 2: variance
# Same run as Table 2: rev1, binary task, 20 seeds, 300 trees, max_depth=25.
r1 = pd.read_csv(U + "rev1_protocol_20seeds_raw.csv")
b = r1[r1.task == "binary"]
order = ["random record", "capture-aware (file-proportion)",
         "capture-aware (size-matched)"]
labels = ["Random\nrecord split", "Capture-aware\n(file-proportion)",
          "Capture-aware\n(size-matched)"]
cols = ["#4C72B0", "#C44E52", "#55A868"]

fig, ax = plt.subplots(figsize=(5.2, 3.0))
rng = np.random.default_rng(0)
for i, (p, c) in enumerate(zip(order, cols)):
    v = b[b.protocol == p].accuracy.values
    x = np.full(len(v), i) + rng.uniform(-0.09, 0.09, len(v))
    ax.scatter(x, v, s=20, color=c, alpha=0.85, zorder=3,
               edgecolor="white", linewidth=0.5)
    ax.hlines(v.mean(), i - 0.24, i + 0.24, color=INK, lw=1.4, zorder=4)
    ax.annotate(f"$\\sigma$ = {v.std(ddof=1):.4f}", xy=(i, 0.828),
                ha="center", fontsize=8, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(labels)
ax.set_ylabel("Binary accuracy"); ax.set_xlim(-0.5, 2.5); ax.set_ylim(0.82, 1.0)
ax.set_title("Capture-aware estimates are far noisier\n"
             "(binary task, 20 seeds, 300 trees, max_depth = 25)",
             fontsize=9.5, loc="left")
fig.savefig(OUT + "fig2_variance.pdf")
fig.savefig(OUT + "fig2_variance.png")
plt.close(fig)

# ------------------------------------------------- Figure 3: depth curve
# rev4: 20 seeds, three protocols, bootstrap intervals.
s4 = pd.read_csv(U + "rev4_depth_sweep_20seeds_summary.csv")
grid = [6, 10, 12, 20, 25, 30, "None"]
xs = list(range(len(grid)))

def series(proto, col, lo, hi):
    d = s4[s4.protocol == proto].copy()
    d["k"] = d.max_depth.apply(lambda v: "None" if pd.isna(v) else int(v))
    d = d.set_index("k").reindex(grid)
    return d[col].values, d[lo].values, d[hi].values

protos = [("random", "random", "#4C72B0", "o"),
          ("capture-aware (file-proportion)", "capture-aware (file-prop.)",
           "#C44E52", "s"),
          ("capture-aware (size-matched)", "capture-aware (size-matched)",
           "#55A868", "^")]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)
for ax, (col, lo, hi, label) in zip(axes, [
        ("benign_recall_mean", "benign_ci_lo", "benign_ci_hi", "Benign recall"),
        ("f1_macro_mean", "f1_ci_lo", "f1_ci_hi", "Macro-F1")]):
    for p, lab, c, mk in protos:
        m, l, h = series(p, col, lo, hi)
        ax.plot(xs, m, marker=mk, ms=4, lw=1.4, color=c, label=lab, zorder=3)
        ax.fill_between(xs, l, h, color=c, alpha=0.15, lw=0)
    ax.axvline(len(grid) - 1, color=INK, lw=0.8, ls=":", alpha=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([str(g) for g in grid])
    ax.set_xlabel("max_depth"); ax.set_ylabel(label)
axes[0].annotate("library\ndefault", xy=(len(grid) - 1, 0.55),
                 xytext=(len(grid) - 1.4, 0.58), fontsize=7, color=INK,
                 ha="right", va="center")
axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.suptitle("Tree depth governs the minority-class trade-off; "
             "the default minimises benign recall", fontsize=9.5, y=1.04)
fig.savefig(OUT + "fig3_depth_curve.pdf")
fig.savefig(OUT + "fig3_depth_curve.png")
plt.close(fig)

print("written:", sorted(os.listdir(OUT)))
