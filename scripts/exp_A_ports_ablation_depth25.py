"""
exp_A_ports_ablation_depth25.py

Attribution ranks Src Port among the top few features for attack detection, which
invites the reading that source port encodes meaningful behaviour. It does not
follow: attribution measures what a fitted model uses, not what it needs, and
both port features enter the forest as continuous numbers, so it can split on
"Src Port < 40213.5" -- an ordering with no semantic meaning.

This script settles the question by retraining without them.

RESULT (Table 4 of the paper): removal leaves binary benign recall unchanged to
four decimal places at unbounded depth, and at depth 25 improves binary accuracy
and macro-F1 while reducing binary benign recall and multiclass macro-F1. The
effect is small, mixed in sign, and configuration-dependent, supporting only the
limited conclusion that a highly ranked feature is not thereby indispensable.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, recall_score,
                             roc_auc_score)
from common import load, build_xy, RANDOM_STATE

# Set the dataset root once, either by editing this line or by exporting
# IOT_DIAD_ROOT in your shell:   export IOT_DIAD_ROOT=/path/to/dataset
ROOT_DIR = os.environ.get("IOT_DIAD_ROOT", r"C:\path\to\CIC-IoT-DIAD-2024")
WORKING_DIR = os.path.join(ROOT_DIR, "_working")

# Set to None to regenerate the unbounded-depth variant reported alongside
# the depth-25 results in Table 4.
MAX_DEPTH = 25
TAG = "depth25" if MAX_DEPTH is not None else "unbounded"

df = load(WORKING_DIR)
results = []

for task, target in [("binary", "attack_binary"),
                     ("multiclass", "attack_family")]:
    for tag, drop_ports in [("with_ports", False), ("no_ports", True)]:
        print(f"\n{'='*60}\n{task.upper()} / {tag}\n{'='*60}")
        X, y, _ = build_xy(df, target, drop_ports=drop_ports)

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

        model = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(n_estimators=300,
                                          random_state=RANDOM_STATE,
                                          n_jobs=-1,
                                          class_weight="balanced_subsample",
                                          max_depth=MAX_DEPTH)),
        ])
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)

        row = {
            "task": task, "variant": tag, "n_features": X.shape[1],
            "accuracy": accuracy_score(yte, yp),
            "f1_macro": f1_score(yte, yp, average="macro"),
            "f1_weighted": f1_score(yte, yp, average="weighted"),
        }
        if task == "binary":
            prob = model.predict_proba(Xte)[:, 1]
            tn, fp, fn, tp = confusion_matrix(yte, yp).ravel()
            row.update({"benign_recall": tn / (tn + fp),
                        "attack_recall": tp / (tp + fn),
                        "roc_auc": roc_auc_score(yte, prob)})
        results.append(row)

        print(classification_report(yte, yp, digits=4, zero_division=0))

res = pd.DataFrame(results)
out = os.path.join(WORKING_DIR, f"ablation_ports_{TAG}.csv")
res.to_csv(out, index=False, encoding="utf-8-sig")
print("\n=== PORT ABLATION SUMMARY (goes straight into a paper table) ===")
print(res.to_string(index=False))
print("\nSaved:", out)
