"""
exp_G2_depth_sweep_multiseed.py

A single-seed sweep suggested that the sklearn default (max_depth=None; realised
depth ~59-61) is the least favourable setting for minority-class recall, but that
run used one group draw, and capture-aware accuracy on this corpus has a standard
deviation near 0.13. The trend needed repeating across draws before any claim
could rest on it.

This repeats the sweep over 5 group draws and reports mean +/- std per depth.

FINDINGS THIS PRODUCED (Section 5.6):
  * The unbounded default gives the lowest benign recall in the grid under BOTH
    protocols -- 0.3245 random, 0.1391 capture-aware -- and less than half the
    next-lowest setting.
  * For macro-F1 the picture is protocol-dependent: the default is worst under
    capture-aware evaluation but mid-range under random splitting, where the
    shallowest trees are worse. The paper therefore claims the minority-recall
    result, not a general macro-F1 result.
  * No optimal depth is claimed. Under capture-aware evaluation the three best
    depths differ by less than a tenth of their standard deviation.

To keep runtime sane the grid is trimmed to the region that matters (the curve
was monotone outside it) and trees are reduced to 200 - depth, not tree count,
is the variable under study.

Runtime: ~40-70 min. Full grid = 5 seeds x 9 depths x 2 protocols = 90 fits.
Set QUICK = True to drop to 3 seeds and 4 depths (~20 min) if you are short.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from common import load, build_xy

# Set the dataset root once, either by editing this line or by exporting
# IOT_DIAD_ROOT in your shell:   export IOT_DIAD_ROOT=/path/to/dataset
ROOT_DIR = os.environ.get("IOT_DIAD_ROOT", r"C:\path\to\CIC-IoT-DIAD-2024")
WORKING_DIR = os.path.join(ROOT_DIR, "_working")

QUICK = False
SEEDS = [42, 7, 123, 2024, 31337] if not QUICK else [42, 123, 2024]
DEPTHS = [6, 8, 10, 12, 16, 20, 25, 30, None] if not QUICK else [8, 12, 20, None]
N_TREES = 200

df = load(WORKING_DIR)
X, y, groups = build_xy(df, "attack_binary")

rows = []


def evaluate(Xtr, Xte, ytr, yte, depth, seed, protocol):
    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=N_TREES, random_state=seed, n_jobs=-1,
            class_weight="balanced_subsample", max_depth=depth)),
    ])
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]
    cm = confusion_matrix(yte, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rf = model.named_steps["rf"]
    # A capture-aware draw can leave the test set single-class (seed 7 puts all
    # benign traffic in train). ROC-AUC is undefined there; record NaN and flag
    # the draw rather than crashing - and report how often it happens, because
    # "1 draw in 5 has no benign test traffic at all" is itself a finding.
    single_class = len(np.unique(yte)) < 2
    return {
        "single_class_test": single_class,
        "protocol": protocol, "seed": seed,
        "max_depth": "None" if depth is None else depth,
        "mean_realised_depth": round(
            float(np.mean([t.get_depth() for t in rf.estimators_])), 1),
        "test_rows": len(yte),
        "accuracy": accuracy_score(yte, yp),
        "f1_macro": f1_score(yte, yp, average="macro", zero_division=0),
        "benign_recall": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "benign_precision": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        "benign_f1": f1_score(yte, yp, pos_label=0, zero_division=0),
        "attack_recall": tp / (tp + fn),
        "roc_auc": np.nan if single_class else roc_auc_score(yte, prob),
        "FP": fp, "FN": fn,
    }


for seed in SEEDS:
    # capture-aware draw
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    itr, ite = next(gss.split(X, y, groups))
    for d in DEPTHS:
        r = evaluate(X.iloc[itr], X.iloc[ite], y.iloc[itr], y.iloc[ite],
                     d, seed, "capture-aware")
        rows.append(r)
        print(f"[capture-aware] seed={seed} depth={str(d):<5} "
              f"benign_recall={r['benign_recall']:.4f} "
              f"macroF1={r['f1_macro']:.4f}")

    # random draw, same seed, for the paired comparison
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    for d in DEPTHS:
        r = evaluate(Xtr, Xte, ytr, yte, d, seed, "random")
        rows.append(r)
        print(f"[random]        seed={seed} depth={str(d):<5} "
              f"benign_recall={r['benign_recall']:.4f} "
              f"macroF1={r['f1_macro']:.4f}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(WORKING_DIR, "depth_sweep_multiseed_raw.csv"),
           index=False, encoding="utf-8-sig")

summary = (res.groupby(["protocol", "max_depth"])
             .agg(realised_depth=("mean_realised_depth", "mean"),
                  acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                  f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
                  benrec_mean=("benign_recall", "mean"),
                  benrec_std=("benign_recall", "std"),
                  benprec_mean=("benign_precision", "mean"))
             .reset_index())
summary.to_csv(os.path.join(WORKING_DIR, "depth_sweep_multiseed_summary.csv"),
               index=False, encoding="utf-8-sig")

bad = res[res.single_class_test].groupby("protocol")["seed"].nunique()
if len(bad):
    print("\n!! Draws with a SINGLE-CLASS test set (no benign traffic at all):")
    print(res[res.single_class_test][["protocol", "seed"]]
          .drop_duplicates().to_string(index=False))
    print("These are excluded from benign metrics. Report the count in the paper.")

print("\n=== DEPTH SWEEP, mean +/- std over seeds ===")
print(summary.to_string(index=False))

print("\n=== ARGMAX macro-F1 per protocol ===")
for p in summary.protocol.unique():
    s = summary[summary.protocol == p].sort_values("f1_mean", ascending=False)
    best = s.iloc[0]
    runner = s.iloc[1]
    print(f"{p:<15} best depth={best.max_depth} "
          f"f1={best.f1_mean:.4f}+/-{best.f1_std:.4f}  "
          f"(runner-up depth={runner.max_depth} f1={runner.f1_mean:.4f})")
print("\nIf best and runner-up overlap within one std, report the optimum as a")
print("RANGE, not a point. Do not claim a precise argmax the variance won't bear.")
