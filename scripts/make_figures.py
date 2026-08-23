"""Generate Figures 1-3 from the CSVs in ../results/.

Figure 1  share distortion      <- true_class_inventory.csv
Figure 2  variance strip plot   <- split_multiseed_raw.csv (same run as Table 2)
Figure 3  depth curve           <- depth_sweep_multiseed_raw.csv

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
ypos = np.arange(len(inv))
h = 0.38
ax.barh(ypos + h/2, inv.true_pct, h, color=INK, label="True share of flows")
ax.barh(ypos - h/2, inv.file_share, h, color="#BFBFBF",
        label="Share of capture files")
ax.set_yticks(ypos)
ax.set_yticklabels(inv["class"])
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
# Figure 2 must come from the SAME run as Table 2 (300 trees, exp_C2), not from
# the depth sweep (200 trees). The values are close but not identical, and a
# figure that disagrees with its table is exactly the inconsistency this paper
# is about.
sm = pd.read_csv(U + "split_multiseed_raw.csv")
sub = sm[sm.task == "binary"].copy()
sub["protocol"] = sub.protocol.str.replace(" split", "", regex=False)
sub["protocol"] = sub.protocol.replace({"random record": "random"})
fig, ax = plt.subplots(figsize=(4.4, 3.0))
rng = np.random.default_rng(0)
for i, (proto, colour) in enumerate([("random", C_RANDOM),
                                     ("capture-aware", C_CAPTURE)]):
    v = sub[sub.protocol == proto].accuracy.values
    x = np.full(len(v), i) + rng.uniform(-0.07, 0.07, len(v))
    ax.scatter(x, v, s=26, color=colour, alpha=0.85, zorder=3,
               edgecolor="white", linewidth=0.6)
    ax.hlines(v.mean(), i - 0.22, i + 0.22, color=INK, lw=1.4, zorder=4)
    ax.annotate(f"$\\sigma$ = {v.std(ddof=1):.4f}", xy=(i, 0.845),
                ha="center", fontsize=8.5, color=INK)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Random record\nsplit", "Capture-aware\nsplit"])
ax.set_ylabel("Binary accuracy")
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0.835, 0.995)
ax.set_title("Capture-aware estimates are far noisier\n(binary task, 5 seeds, 300 trees, max_depth = 25)",
             fontsize=9.5, loc="left")
fig.savefig(OUT + "fig2_variance.pdf")
fig.savefig(OUT + "fig2_variance.png")
plt.close(fig)

# ------------------------------------------------- Figure 3: depth curve
raw = pd.read_csv(U + "depth_sweep_multiseed_raw.csv")
raw["depth_num"] = raw.max_depth.astype(float).fillna(36)  # NaN = None

valid = raw[~raw.single_class_test]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)

for ax, metric, label in zip(
        axes, ["benign_recall", "f1_macro"],
        ["Benign recall", "Macro-F1"]):
    for proto, colour, marker in [("random", C_RANDOM, "o"),
                                  ("capture-aware", C_CAPTURE, "s")]:
        d = valid[valid.protocol == proto].groupby("depth_num")[metric]
        m, s = d.mean(), d.std()
        ax.plot(m.index, m.values, marker=marker, ms=4, lw=1.4,
                color=colour, label=proto, zorder=3)
        ax.fill_between(m.index, m - s.fillna(0), m + s.fillna(0),
                        color=colour, alpha=0.15, lw=0)
    ax.axvline(36, color=INK, lw=0.8, ls=":", alpha=0.6)
    ax.set_ylabel(label)
    ax.set_xlabel("max_depth")
    ax.set_xticks([6, 10, 16, 20, 25, 30, 36])
    ax.set_xticklabels(["6", "10", "16", "20", "25", "30", "None"])

axes[0].annotate("library\ndefault", xy=(36, 0.5), xytext=(29.5, 0.52),
                 fontsize=7, color=INK, ha="right", va="center")
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("Tree depth governs the minority-class trade-off; "
             "the default minimises benign recall", fontsize=9.5, y=1.04)
fig.savefig(OUT + "fig3_depth_curve.pdf")
fig.savefig(OUT + "fig3_depth_curve.png")
plt.close(fig)

print("written:", sorted(os.listdir(OUT)))
