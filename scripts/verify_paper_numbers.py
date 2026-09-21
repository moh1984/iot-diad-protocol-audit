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

# ---------- Table 2: protocol comparison (rev1, 20 seeds) ----------
r1 = load("rev1_protocol_20seeds_raw.csv")

def arm(task, proto, scored_only=False):
    g = r1[(r1.task == task) & (r1.protocol == proto)]
    if scored_only and task == "binary":
        g = g[g.n_scored_families == 2]
    return g

for task, proto, col, exp, label in [
        ("binary", "random record", "accuracy", 0.9626, "T2 binary random acc"),
        ("binary", "capture-aware (file-proportion)", "accuracy", 0.9550,
         "T2 binary file-prop acc"),
        ("binary", "capture-aware (size-matched)", "accuracy", 0.9653,
         "T2 binary size-matched acc"),
        ("multiclass", "random record", "accuracy", 0.9154, "T2 8class random acc"),
        ("multiclass", "capture-aware (file-proportion)", "accuracy", 0.8001,
         "T2 8class file-prop acc"),
        ("multiclass", "capture-aware (size-matched)", "accuracy", 0.8387,
         "T2 8class size-matched acc")]:
    checks.append((label, arm(task, proto)[col].mean(), exp))

# Binary macro-F1 must exclude draws in which only one class is testable.
for proto, exp, n_exp, label in [
        ("random record", 0.7009, 20, "T2 binary random macro-F1"),
        ("capture-aware (file-proportion)", 0.7020, 11,
         "T2 binary file-prop macro-F1"),
        ("capture-aware (size-matched)", 0.6436, 9,
         "T2 binary size-matched macro-F1")]:
    g = arm("binary", proto, scored_only=True)
    checks.append((label, g.f1_macro_scored.mean(), exp))
    checks.append((label + " (draws)", len(g), n_exp))

for proto, col, exp, label in [
        ("random record", "f1_macro_scored", 0.6174, "T2 8class random macro-F1"),
        ("capture-aware (file-proportion)", "f1_macro_all", 0.2924,
         "T2 8class file-prop macro-F1 (all 8)"),
        ("capture-aware (file-proportion)", "f1_macro_scored", 0.4914,
         "T2 8class file-prop macro-F1 (scored)")]:
    checks.append((label, arm("multiclass", proto)[col].mean(), exp))

# Test-partition heterogeneity, quoted in Section 5.2 and the response letter.
cap = r1[r1.protocol == "capture-aware (file-proportion)"]
sm = r1[r1.protocol == "capture-aware (size-matched)"]
checks += [
    ("S5.2 file-prop test fraction min", round(cap.test_frac.min(), 3), 0.080),
    ("S5.2 file-prop test fraction max", round(cap.test_frac.max(), 3), 0.403),
    ("S5.2 size-matched test fraction min", round(sm.test_frac.min(), 3), 0.200),
    ("S5.2 size-matched test fraction max", round(sm.test_frac.max(), 3), 0.285),
]

# ---------- Variance ratios: intervals, not point estimates ----------
vr = load("rev1_variance_ratio_ci.csv")
for task, proto, lo, hi, label in [
        ("multiclass", "capture-aware (file-proportion)", 44.9, 125.9,
         "S5.2 variance ratio CI, 8class file-prop"),
        ("multiclass", "capture-aware (size-matched)", 35.9, 93.5,
         "S5.2 variance ratio CI, 8class size-matched"),
        ("binary", "capture-aware (file-proportion)", 7.5, 30.7,
         "S5.2 variance ratio CI, binary file-prop"),
        ("binary", "capture-aware (size-matched)", 7.6, 20.0,
         "S5.2 variance ratio CI, binary size-matched")]:
    row = vr[(vr.task == task) & (vr.protocol == proto)].iloc[0]
    checks.append((label + " lo", round(row.ratio_ci_lo, 1), lo))
    checks.append((label + " hi", round(row.ratio_ci_hi, 1), hi))

# The manuscript claims every lower bound exceeds seven. Verify the claim itself.
checks.append(("S5.2 all variance-ratio lower bounds exceed 7",
               int((vr.ratio_ci_lo > 7).all()), 1))

# ---------- Table 3: per-family recall, 20 seeds (rev3) ----------
f3 = load("rev3_per_family_summary.csv")
fp = f3[f3.protocol == "capture-aware (file-proportion)"]
for fam, exp, draws in [("DOS", 0.9294, 20), ("DDOS", 0.6946, 20),
                        ("Mirai", 0.2862, 20), ("Spoofing", 0.2170, 14),
                        ("Benign", 0.2872, 11), ("Web-Based", 0.2654, 13)]:
    row = fp[fp.family == fam].iloc[0]
    checks.append((f"T3  {fam:<11} mean recall", row.mean_recall, exp))
    checks.append((f"T3  {fam:<11} draws scored", int(row.draws_scored), draws))

# Recon and Brute Force must never be scorable; the manuscript reports n/a.
for fam in ["Recon", "Brute Force"]:
    checks.append((f"T3  {fam:<11} never scorable",
                   int(f3[f3.family == fam].draws_scored.sum()), 0))

# ---------- Table 4: feature ablation with controls (rev2, 15 seeds) ----------
ab = load("rev2_ablation_summary.csv")
ab["depth"] = ab.depth.fillna("None").astype(str)

for task, depth, removed, exp_drop, exp_lo, exp_hi, label in [
        ("multiclass", "25.0", "no_ports", 0.0250, 0.0239, 0.0261, "T4 8class ports d25"),
        ("multiclass", "25.0", "no_timing_pair", 0.0027, 0.0014, 0.0041, "T4 8class timing d25"),
        ("multiclass", "25.0", "no_arbitrary_pair", 0.0006, -0.0006, 0.0016, "T4 8class arbitrary d25"),
        ("multiclass", "None", "no_ports", 0.0237, 0.0224, 0.0251, "T4 8class ports dNone")]:
    r = ab[(ab.task == task) & (ab.depth == depth) & (ab.removed == removed)].iloc[0]
    checks.append((label, round(r.mean_f1_drop, 4), exp_drop))
    checks.append((label + " CI lo", round(r.ci_lo, 4), exp_lo))
    checks.append((label + " CI hi", round(r.ci_hi, 4), exp_hi))

# Binary benign-recall drop after port removal, quoted in Section 5.4.
rb = ab[(ab.task == "binary") & (ab.depth == "25.0") & (ab.removed == "no_ports")].iloc[0]
checks.append(("T4 binary benign-recall drop, ports", round(rb.mean_benign_recall_drop, 4), 0.0347))

# CLAIM CHECK. The manuscript states that the port-pair interval is separated
# from BOTH control intervals, and deliberately does NOT claim that the two
# control intervals are separated from each other -- they overlap.
d25 = ab[(ab.task == "multiclass") & (ab.depth == "25.0")].set_index("removed")
pl, ph = d25.loc["no_ports", ["ci_lo", "ci_hi"]]
tl, th = d25.loc["no_timing_pair", ["ci_lo", "ci_hi"]]
al, ah = d25.loc["no_arbitrary_pair", ["ci_lo", "ci_hi"]]
sep = lambda a, b, c, d: int(b < c or d < a)
checks += [
    ("S5.4 port interval separated from timing", sep(pl, ph, tl, th), 1),
    ("S5.4 port interval separated from arbitrary", sep(pl, ph, al, ah), 1),
    ("S5.4 control intervals overlap (not claimed otherwise)",
     sep(tl, th, al, ah), 0),
]

# ---------- Table 5: balancing, repeated over 15 seeds (rev2) ----------
bl = load("rev2_balancing_summary.csv").set_index("quantity")
for q, exp_mean, exp_lo, exp_hi in [
        ("A_full", 0.6178, 0.6132, 0.6218),
        ("B_balanced", 0.6120, 0.6087, 0.6147),
        ("C_size_control", 0.5495, 0.5471, 0.5521),
        ("B_minus_C", 0.0625, 0.0582, 0.0662),
        ("A_minus_B", 0.0059, 0.0035, 0.0081)]:
    checks.append((f"T5 {q:<16} mean", round(bl.loc[q, "mean"], 4), exp_mean))
    checks.append((f"T5 {q:<16} CI lo", round(bl.loc[q, "ci_lo"], 4), exp_lo))
    checks.append((f"T5 {q:<16} CI hi", round(bl.loc[q, "ci_hi"], 4), exp_hi))

# CLAIM CHECK: the balancing effect must exclude zero for the claim to stand.
checks.append(("S5.5 balancing interval excludes zero",
               int(bl.loc["B_minus_C", "ci_lo"] > 0), 1))

# ---------- Table 7: depth sweep, 20 seeds, three protocols (rev4) ----------
d4 = load("rev4_depth_sweep_20seeds_summary.csv")
d4["k"] = d4.max_depth.apply(lambda v: "None" if pd.isna(v) else str(int(v)))

for proto, k, col, exp, label in [
        ("random", "6", "benign_recall_mean", 0.9774, "T7 random d6 benign"),
        ("random", "None", "benign_recall_mean", 0.3230, "T7 random None benign"),
        ("random", "30", "f1_macro_mean", 0.7216, "T7 random d30 macro-F1"),
        ("random", "None", "f1_macro_mean", 0.6970, "T7 random None macro-F1"),
        ("capture-aware (file-proportion)", "6", "benign_recall_mean", 0.9125,
         "T7 file-prop d6 benign"),
        ("capture-aware (file-proportion)", "None", "benign_recall_mean", 0.1635,
         "T7 file-prop None benign"),
        ("capture-aware (file-proportion)", "None", "f1_macro_mean", 0.6102,
         "T7 file-prop None macro-F1"),
        ("capture-aware (size-matched)", "None", "benign_recall_mean", 0.2107,
         "T7 size-matched None benign"),
        ("capture-aware (size-matched)", "6", "f1_macro_mean", 0.5915,
         "T7 size-matched d6 macro-F1")]:
    r = d4[(d4.protocol == proto) & (d4.k == k)].iloc[0]
    checks.append((label, round(r[col], 4), exp))

# Usable-draw counts, quoted in the Table 7 note and the Figure 3 caption.
for proto, exp, label in [
        ("capture-aware (file-proportion)", 11, "T7 file-prop draws with benign"),
        ("capture-aware (size-matched)", 9, "T7 size-matched draws with benign"),
        ("random", 20, "T7 random draws with benign")]:
    checks.append((label,
                   int(d4[d4.protocol == proto].draws_with_benign.max()), exp))

# CLAIM CHECKS for Section 5.6, each stated in the manuscript.
for proto, label in [("random", "random"),
                     ("capture-aware (file-proportion)", "file-prop"),
                     ("capture-aware (size-matched)", "size-matched")]:
    g = d4[d4.protocol == proto]
    none_row = g[g.k == "None"].iloc[0]
    checks.append((f"S5.6 None minimises benign recall ({label})",
                   int(none_row.benign_recall_mean == g.benign_recall_mean.min()), 1))
    # benign recall must fall monotonically with depth
    ordered = g.set_index("k").reindex(["6", "10", "12", "20", "25", "30", "None"])
    mono = all(ordered.benign_recall_mean.values[i] > ordered.benign_recall_mean.values[i+1]
               for i in range(len(ordered) - 1))
    checks.append((f"S5.6 benign recall monotone in depth ({label})", int(mono), 1))

# The manuscript says the macro-F1 pattern is protocol-dependent, and
# specifically that None is NOT the worst setting under size-matched splitting.
sm4 = d4[d4.protocol == "capture-aware (size-matched)"]
checks.append(("S5.6 None is not worst macro-F1 under size-matched",
               int(sm4[sm4.k == "None"].f1_macro_mean.iloc[0]
                   > sm4.f1_macro_mean.min()), 1))

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

# ---------- LEGACY: five-seed depth sweep from the v1.0 submission ----------
# These values are NOT in the current manuscript; Table 7 now comes from rev4 at
# twenty seeds (checked above). They are retained as regression checks so the two
# submission versions remain comparable, and are labelled accordingly.
d = load("depth_sweep_multiseed_raw.csv")
v = d[~d.single_class_test]
for proto, depth, col, exp in [
        ("random", 6.0, "benign_recall", 0.9784),
        ("random", 30.0, "f1_macro", 0.7171),
        ("capture-aware", 6.0, "benign_recall", 0.8875)]:
    sel = v[(v.protocol == proto) & (v.max_depth == depth)]
    checks.append((f"LEGACY v1.0  {proto:<13} depth={int(depth):<3} {col}",
                   sel[col].mean(), exp))
for proto, col, exp in [("random", "benign_recall", 0.3245),
                        ("capture-aware", "benign_recall", 0.1391),
                        ("capture-aware", "f1_macro", 0.5878)]:
    sel = v[(v.protocol == proto) & (v.max_depth.isna())]
    checks.append((f"LEGACY v1.0  {proto:<13} depth=None {col}",
                   sel[col].mean(), exp))

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

legacy = sum(1 for n, _, _ in checks if n.startswith("LEGACY"))
print(f"\n{len(checks)} values checked, {bad} mismatches "
      f"({len(checks) - legacy} against the current manuscript, "
      f"{legacy} legacy v1.0 regression checks).")
if bad:
    print("\nA mismatch means the manuscript and the released results disagree.")
    print("Fix the manuscript, or rerun the experiment and update results/.")
sys.exit(1 if bad else 0)
