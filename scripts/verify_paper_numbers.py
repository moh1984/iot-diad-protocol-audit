"""
verify_paper_numbers.py

Cross-checks every headline number in the manuscript against the CSV that
produced it. Runs in under a second and trains nothing.

Two kinds of check appear here. Most compare a value cited in the manuscript
against the cell that produced it. A few check a CLAIM the manuscript makes about
a set of values -- for instance that seven of eight families fell within two
percentage points of their file share. The second kind matters because a
generalisation can be false even when every value it generalises over is correct:
"within two percentage points on every family" does not hold once DoS is included,
and checking the three cited percentages alone would not reveal that. When the
manuscript generalises over numbers, the generalisation is verified too.

The specific hazard this guards against is mixing model configurations: comparing
a binary result obtained at one tree depth against a multiclass result obtained at
another, or a threshold baseline drawn from one fitted model against an operating
point drawn from a second. That is the exact error the manuscript documents
elsewhere -- changing two variables and attributing the effect to one -- and it is
easy to reintroduce whenever an experiment is rerun. Every experiment here is
pinned to max_depth=25 except the depth sweep, and these checks confirm that the
reported values still come from those pinned runs.

Usage:
    python scripts/verify_paper_numbers.py
Exit code 0 if every value matches, 1 otherwise. The number of checks is printed
at the end rather than hard-coded anywhere, so adding a check cannot leave the
documentation stale.
"""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
TOL = 2e-4


def load(name):
    return pd.read_csv(os.path.join(R, name))


checks = []

# ---------- Table 2: protocol comparison ----------
s = load("split_multiseed_summary.csv")
b = s[s.task == "binary"]
m = s[s.task == "multiclass"]
checks += [
    ("T2  binary  random   accuracy",
     b[b.protocol.str.contains("random")].accuracy_mean.iloc[0], 0.9621),
    ("T2  binary  random   macro-F1",
     b[b.protocol.str.contains("random")].f1_macro_mean.iloc[0], 0.6989),
    ("T2  binary  capture  accuracy",
     b[b.protocol.str.contains("capture")].accuracy_mean.iloc[0], 0.9401),
    ("T2  binary  capture  macro-F1",
     b[b.protocol.str.contains("capture")].f1_macro_mean.iloc[0], 0.5621),
    ("T2  8class  random   accuracy",
     m[m.protocol.str.contains("random")].accuracy_mean.iloc[0], 0.9144),
    ("T2  8class  capture  accuracy",
     m[m.protocol.str.contains("capture")].accuracy_mean.iloc[0], 0.7494),
]

sc = load("scored_macro_f1_summary.csv")
checks += [
    ("T2  8class  capture  macro-F1 (all 8)",
     sc[sc.protocol == "capture-aware"].all8_mean.iloc[0], 0.2769),
    ("T2  8class  capture  macro-F1 (scored)",
     sc[sc.protocol == "capture-aware"].scored_mean.iloc[0], 0.4606),
]

# ---------- Table 3: per-family recall, scored draws only ----------
f = load("scored_per_family.csv")
ca = f[(f.protocol == "capture-aware") & (f.scored)]
for fam, exp in [("DOS", 0.9173), ("DDOS", 0.7557), ("Mirai", 0.3256),
                 ("Spoofing", 0.2343), ("Benign", 0.2321), ("Web-Based", 0.1866)]:
    checks.append((f"T3  {fam:<11} mean recall",
                   ca[ca.family == fam].recall.mean(), exp))

# ---------- Table 4: port ablation at depth 25 ----------
a = load("ablation_ports_depth25.csv")
for task, variant, col, exp in [
        ("binary", "with_ports", "accuracy", 0.9615),
        ("binary", "with_ports", "benign_recall", 0.7122),
        ("binary", "no_ports", "accuracy", 0.9650),
        ("binary", "no_ports", "benign_recall", 0.6653),
        ("multiclass", "with_ports", "f1_macro", 0.6120),
        ("multiclass", "no_ports", "f1_macro", 0.5863)]:
    checks.append((f"T4  {task[:6]:<6} {variant:<10} {col}",
                   a[(a.task == task) & (a.variant == variant)][col].iloc[0], exp))

# ---------- Table 5: balancing with size controlled ----------
u = load("undersample_control_depth25.csv")
for arm, exp in [("A_full", 0.6120), ("B_balanced", 0.6115),
                 ("C_size_control", 0.5488)]:
    checks.append((f"T5  {arm:<16} macro-F1",
                   u[u.arm == arm].f1_macro.iloc[0], exp))

# ---------- Table 6: threshold calibration, ONE model (depth 25) ----------
t = load("threshold_sweep_test_descriptive_depth25.csv")
for th, col, exp in [
        (0.50, "accuracy", 0.9623), (0.50, "benign_recall", 0.6592),
        (0.50, "precision_benign", 0.2926), (0.50, "f1_macro", 0.6929),
        (0.38, "accuracy", 0.9689), (0.38, "benign_recall", 0.5551),
        (0.38, "precision_benign", 0.3254), (0.38, "f1_macro", 0.6972),
        (0.98, "benign_recall", 0.9776), (0.98, "f1_macro", 0.6143)]:
    checks.append((f"T6  tau={th:.2f}  {col}",
                   t[t.threshold.round(2) == th][col].iloc[0], exp))

# ---------- Table 7: depth sweep, scored draws only ----------
d = load("depth_sweep_multiseed_raw.csv")
v = d[~d.single_class_test]
for proto, depth, col, exp in [
        ("random", 6.0, "benign_recall", 0.9784),
        ("random", 30.0, "f1_macro", 0.7171),
        ("capture-aware", 6.0, "benign_recall", 0.8875)]:
    sel = v[(v.protocol == proto) & (v.max_depth == depth)]
    checks.append((f"T7  {proto:<13} depth={int(depth):<3} {col}",
                   sel[col].mean(), exp))
for proto, col, exp in [("random", "benign_recall", 0.3245),
                        ("capture-aware", "benign_recall", 0.1391),
                        ("capture-aware", "f1_macro", 0.5878)]:
    sel = v[(v.protocol == proto) & (v.max_depth.isna())]
    checks.append((f"T7  {proto:<13} depth=None {col}", sel[col].mean(), exp))

# ---------- Section 5.4: attribution ranks (both depths, same protocol) ----------
for fname, tag, exp_shap, exp_rf, exp_rho in [
        ("shap_vs_rf_withports_unbounded.csv", "unbounded", 3, 9, 0.8823),
        ("shap_vs_rf_withports_depth25.csv", "depth25", 6, 13, 0.8975)]:
    sh = load(fname)
    row = sh[sh.feature == "Src Port"].iloc[0]
    checks += [
        (f"S5.4 Src Port SHAP rank ({tag})", row.shap_rank, exp_shap),
        (f"S5.4 Src Port impurity rank ({tag})", row.rf_rank, exp_rf),
        (f"S5.4 Spearman SHAP vs impurity ({tag})",
         round(sh.shap_rank.corr(sh.rf_rank, method="spearman"), 4), exp_rho),
    ]

# ---------- Section 4: corpus and preprocessing ----------
inv = load("true_class_inventory.csv")
checks += [
    ("S4  total corpus flows", inv.true_rows.sum(), 19_519_162),
    ("S4  total capture files", inv.csv_files.sum(), 129),
    ("S4  DoS true share (%)",
     round(100 * inv[inv["class"] == "DOS"].true_rows.iloc[0] / inv.true_rows.sum(), 2), 76.09),
]

# ---------- Sections 1 and 5.1: the audited preliminary run ----------
pre = pd.read_csv(os.path.join(HERE, "..", "preliminary", "build_summary.csv"))
tot = pre.rows_after_sampling.sum()
for fam, exp in [("DDOS", 49.19), ("Mirai", 23.39), ("Benign", 3.23)]:
    got = 100 * pre[pre.class_name == fam].rows_after_sampling.iloc[0] / tot
    checks.append((f"S5.1 preliminary {fam} sampled share (%)", round(got, 2), exp))
checks.append(("S5.1 preliminary DoS sampled share (%)",
               round(100 * pre[pre.class_name == "DOS"].rows_after_sampling.iloc[0] / tot, 2), 17.74))
checks.append(("S5.1 preliminary total sampled rows", tot, 124_000))

# The paper makes a claim ABOUT these numbers, not merely a claim OF them:
# "seven of eight families fell within two percentage points of their file share;
# DoS differed by 3.19 points". Checking the cited values alone would not catch a
# false statement over them, so the claim itself is checked here as well.
theo = pd.read_csv(os.path.join(HERE, "..", "preliminary",
                                "theoretical_file_share_expectation.csv"))
pre2 = pre.copy()
pre2["obs"] = 100 * pre2.rows_after_sampling / tot
mg = pre2.merge(theo[["family", "file_share_pct"]],
                left_on="class_name", right_on="family")
dev = (mg.obs - mg.file_share_pct).abs()
checks.append(("S5.1 families within 2 points of file share", int((dev <= 2).sum()), 7))
checks.append(("S5.1 largest deviation from file share (pts)", round(dev.max(), 2), 3.19))

# ---------- report ----------
width = max(len(n) for n, _, _ in checks)
bad = 0
for name, got, exp in checks:
    got = float(got)
    tol = TOL if abs(exp) < 100 else 1.0
    ok = abs(got - exp) < tol
    bad += not ok
    print(f'{"OK      " if ok else "MISMATCH"}  {name:<{width}}  '
          f'csv={got:.4f}  paper={exp}')

print(f"\n{len(checks)} values checked, {bad} mismatches.")
if bad:
    print("\nA mismatch means the manuscript and the released results disagree.")
    print("Fix the manuscript, or rerun the experiment and update results/.")
sys.exit(1 if bad else 0)
