"""
recompute_scored_macro_f1.py

The last open number in the paper.

The capture-aware macro-F1 computed over all eight families (0.2769) averages
over Recon and Brute Force too, whose recall was recorded as 0.0000 in draws
where they had ZERO test support. `zero_division=0` produced those zeros; they
are undefined, not measured, and they drag the macro average down.

This recomputes macro-F1 over families that were actually scorable in each draw
(present in both the training and test partitions), giving 0.4606, and reports
both figures side by side so the paper can state the difference and its cause.

No retraining beyond the five capture-aware fits. ~20-30 min.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from common import load, build_xy

# Set the dataset root once, either by editing this line or by exporting
# IOT_DIAD_ROOT in your shell:   export IOT_DIAD_ROOT=/path/to/dataset
ROOT_DIR = os.environ.get("IOT_DIAD_ROOT", r"C:\path\to\CIC-IoT-DIAD-2024")
WORKING_DIR = os.path.join(ROOT_DIR, "_working")

SEEDS = [42, 7, 123, 2024, 31337]
N_TREES = 300
MAX_DEPTH = 25

df = load(WORKING_DIR)
X, y, groups = build_xy(df, "attack_family")

rows, fam_rows = [], []

for seed in SEEDS:
    for protocol in ["random", "capture-aware"]:
        if protocol == "random":
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y)
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                    random_state=seed)
            itr, ite = next(gss.split(X, y, groups))
            Xtr, Xte, ytr, yte = (X.iloc[itr], X.iloc[ite],
                                  y.iloc[itr], y.iloc[ite])

        model = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=N_TREES, random_state=seed, n_jobs=-1,
                max_depth=MAX_DEPTH, class_weight="balanced_subsample")),
        ]).fit(Xtr, ytr)
        yp = model.predict(Xte)

        # a family is SCORED only if it has training exposure AND test support
        scored = sorted(set(ytr.unique()) & set(yte.unique()))
        unscored = sorted(set(y.unique()) - set(scored))

        f1_all = f1_score(yte, yp, average="macro",
                          labels=sorted(y.unique()), zero_division=0)
        f1_scored = f1_score(yte, yp, average="macro", labels=scored,
                             zero_division=0)

        rows.append({
            "protocol": protocol, "seed": seed,
            "n_scored_families": len(scored),
            "unscored": ",".join(unscored) or "-",
            "accuracy": accuracy_score(yte, yp),
            "f1_macro_all8": f1_all,
            "f1_macro_scored": f1_scored,
            "inflation": f1_scored - f1_all,
        })
        print(f"[{protocol:<14}] seed={seed:<6} scored={len(scored)}/8  "
              f"all8={f1_all:.4f}  scored={f1_scored:.4f}  "
              f"unscored: {rows[-1]['unscored']}")

        rep = classification_report(yte, yp, output_dict=True, zero_division=0)
        for fam in sorted(y.unique()):
            fam_rows.append({
                "protocol": protocol, "seed": seed, "family": fam,
                "scored": fam in scored,
                "recall": rep[fam]["recall"] if fam in scored else np.nan,
                "precision": rep[fam]["precision"] if fam in scored else np.nan,
                "f1": rep[fam]["f1-score"] if fam in scored else np.nan,
                "support": int(rep[fam]["support"]) if fam in rep else 0,
            })

res = pd.DataFrame(rows)
fam = pd.DataFrame(fam_rows)
res.to_csv(os.path.join(WORKING_DIR, "scored_macro_f1_raw.csv"),
           index=False, encoding="utf-8-sig")
fam.to_csv(os.path.join(WORKING_DIR, "scored_per_family.csv"),
           index=False, encoding="utf-8-sig")

summary = (res.groupby("protocol")
             .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                  all8_mean=("f1_macro_all8", "mean"),
                  all8_std=("f1_macro_all8", "std"),
                  scored_mean=("f1_macro_scored", "mean"),
                  scored_std=("f1_macro_scored", "std"))
             .reset_index())
summary.to_csv(os.path.join(WORKING_DIR, "scored_macro_f1_summary.csv"),
               index=False, encoding="utf-8-sig")

print("\n=== MACRO-F1: ALL EIGHT FAMILIES vs SCORED FAMILIES ONLY ===")
print(summary.to_string(index=False))

print("\n=== PER-FAMILY, SCORED DRAWS ONLY (this is the corrected Table 3) ===")
t = (fam[(fam.protocol == "capture-aware") & fam.scored]
     .groupby("family")
     .agg(mean_recall=("recall", "mean"), std=("recall", "std"),
          draws_scored=("recall", "size"), mean_support=("support", "mean"))
     .sort_values("mean_recall", ascending=False))
print(t.round(4).to_string())
print("\nFamilies never scored in any draw must be reported as n/a, not 0.0000.")
