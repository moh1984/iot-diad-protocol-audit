"""
exp_C2_group_split_multiseed.py
Replaces exp_C_group_split_leakage.py.

WHY A REWRITE: the single-seed run gave multiclass accuracy 0.9130 -> 0.5382, but
the diagnostic showed that split was pathological. GroupShuffleSplit partitions
FILES, not rows, so with test_size=0.2 we got 26 of 129 files - which happened to
hold only 10,054 rows (8%, not 20%), and the test distribution bore no relation
to the training one: DoS was 78% of train but 10% of test, while Recon was 0% of
train and 27% of test.

Part of that 37-point drop is therefore covariate shift from one unlucky draw,
not a generalization gap. A single seed cannot separate them.

THIS VERSION:
  * 5 independent group splits, mean +/- std for every metric.
  * Per-seed audit of test size and class composition, so the instability is
    measured and reported rather than hidden.
  * Families absent from training are recorded per seed - their zeros are a
    property of the protocol on single-capture families, not model failure.
  * macro-F1 computed BOTH over all families and over evaluable families only
    (present in both partitions), since scoring a family with zero test support
    drags the macro average down meaninglessly.

Report the mean +/- std. If the std is large, that IS the finding: capture-aware
evaluation on this dataset is high-variance because a few families sit on one or
two capture files.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from common import load, build_xy

# Set the dataset root once, either by editing this line or by exporting
# IOT_DIAD_ROOT in your shell:   export IOT_DIAD_ROOT=/path/to/dataset
ROOT_DIR = os.environ.get("IOT_DIAD_ROOT", r"C:\path\to\CIC-IoT-DIAD-2024")
WORKING_DIR = os.path.join(ROOT_DIR, "_working")

N_SEEDS = 5
SEEDS = [42, 7, 123, 2024, 31337]
N_TREES = 300
MAX_DEPTH = 25          # matches every other experiment; set to None for the unbounded variant


def make_model(seed):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=N_TREES, random_state=seed,
                                      n_jobs=-1, max_depth=MAX_DEPTH,
                                      class_weight="balanced_subsample")),
    ])


df = load(WORKING_DIR)
rows, audit, per_family = [], [], []

for task, target in [("binary", "attack_binary"),
                     ("multiclass", "attack_family")]:
    X, y, groups = build_xy(df, target, verbose=(task == "binary"))
    if groups is None:
        raise SystemExit("source_file column missing - rerun step0 first.")

    for seed in SEEDS:
        # ---------- protocol 1: random record split ----------
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
        m = make_model(seed).fit(Xtr, ytr)
        yp = m.predict(Xte)
        rows.append({"task": task, "protocol": "random record split",
                     "seed": seed, "test_rows": len(yte),
                     "accuracy": accuracy_score(yte, yp),
                     "f1_macro": f1_score(yte, yp, average="macro",
                                          zero_division=0),
                     "f1_macro_evaluable": f1_score(yte, yp, average="macro",
                                                    zero_division=0),
                     "n_absent_from_train": 0})

        # ---------- protocol 2: capture-aware split ----------
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        itr, ite = next(gss.split(X, y, groups))
        ytr2, yte2 = y.iloc[itr], y.iloc[ite]

        m = make_model(seed).fit(X.iloc[itr], ytr2)
        yp2 = m.predict(X.iloc[ite])

        # families present in BOTH partitions can be fairly scored
        in_train = set(ytr2.unique())
        in_test = set(yte2.unique())
        evaluable = sorted(in_train & in_test)
        absent = sorted(in_test - in_train)

        f1_all = f1_score(yte2, yp2, average="macro", zero_division=0)
        f1_eval = f1_score(yte2, yp2, average="macro", labels=evaluable,
                           zero_division=0) if evaluable else np.nan

        rows.append({"task": task, "protocol": "capture-aware split",
                     "seed": seed, "test_rows": len(yte2),
                     "accuracy": accuracy_score(yte2, yp2),
                     "f1_macro": f1_all,
                     "f1_macro_evaluable": f1_eval,
                     "n_absent_from_train": len(absent)})

        # audit of how balanced this particular draw was
        share = yte2.value_counts(normalize=True)
        audit.append({"task": task, "seed": seed,
                      "test_rows": len(yte2),
                      "test_pct_of_total": 100 * len(yte2) / len(y),
                      "test_files": groups.iloc[ite].nunique(),
                      "absent_from_train": ",".join(map(str, absent)) or "-",
                      "largest_test_class_share": round(share.max(), 4)})

        if task == "multiclass":
            rep = classification_report(yte2, yp2, output_dict=True,
                                        zero_division=0)
            for fam in sorted(set(map(str, y.unique()))):
                if fam in rep:
                    per_family.append({
                        "seed": seed, "family": fam,
                        "recall": rep[fam]["recall"],
                        "support": rep[fam]["support"],
                        "in_train": fam in set(map(str, in_train))})

        print(f"[{task}] seed={seed} done "
              f"(capture-aware acc={rows[-1]['accuracy']:.4f})")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(WORKING_DIR, "split_multiseed_raw.csv"),
           index=False, encoding="utf-8-sig")

summary = (res.groupby(["task", "protocol"])
             .agg(accuracy_mean=("accuracy", "mean"),
                  accuracy_std=("accuracy", "std"),
                  f1_macro_mean=("f1_macro", "mean"),
                  f1_macro_std=("f1_macro", "std"),
                  f1_eval_mean=("f1_macro_evaluable", "mean"),
                  f1_eval_std=("f1_macro_evaluable", "std"),
                  test_rows_mean=("test_rows", "mean"))
             .reset_index())
summary.to_csv(os.path.join(WORKING_DIR, "split_multiseed_summary.csv"),
               index=False, encoding="utf-8-sig")

pd.DataFrame(audit).to_csv(
    os.path.join(WORKING_DIR, "split_multiseed_audit.csv"),
    index=False, encoding="utf-8-sig")
if per_family:
    pd.DataFrame(per_family).to_csv(
        os.path.join(WORKING_DIR, "split_multiseed_per_family.csv"),
        index=False, encoding="utf-8-sig")

print("\n=== SUMMARY (mean +/- std over 5 seeds) ===")
print(summary.to_string(index=False))
print("\n=== PER-SEED AUDIT (how unbalanced each capture draw was) ===")
print(pd.DataFrame(audit).to_string(index=False))
print("\nSaved 4 files to", WORKING_DIR)
