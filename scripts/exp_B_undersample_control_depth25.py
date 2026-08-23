"""
exp_B_undersample_control_depth25.py

Your Experiment 4 changed TWO things at once: the class balance AND the total
training size (DDoS fell from ~48,800 to ~2,400). So the drop from 72.92% to
58.64% cannot be attributed to "balancing destroys diversity" - it may simply be
"we threw away 85% of the training data".

This adds the missing control arm:

  A. full          - all training data, original balance          (your baseline)
  B. balanced      - undersampled to 3x min class                 (your Exp 4)
  C. size_control  - random subsample, SAME TOTAL SIZE as B,
                     but ORIGINAL proportions preserved           (NEW)

If B ~= C, the damage was data loss, not balancing, and the paper's claim must
be softened. If B < C, your claim holds and is now properly evidenced.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score, f1_score,
                             confusion_matrix)
from common import load, build_xy, RANDOM_STATE

# Set the dataset root once, either by editing this line or by exporting
# IOT_DIAD_ROOT in your shell:   export IOT_DIAD_ROOT=/path/to/dataset
ROOT_DIR = os.environ.get("IOT_DIAD_ROOT", r"C:\path\to\CIC-IoT-DIAD-2024")
WORKING_DIR = os.path.join(ROOT_DIR, "_working")

# Set to None to regenerate the unbounded-depth variant reported in Section 5.5.
MAX_DEPTH = 25
TAG = "depth25" if MAX_DEPTH is not None else "unbounded"

df = load(WORKING_DIR)
X, y, _ = build_xy(df, "attack_family", drop_ports=False)

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

tr = Xtr.copy()
tr["_y"] = ytr.values

# --- arm B: aggressive undersampling (3x min class) ---
counts = tr["_y"].value_counts()
cap = counts.min() * 3
parts = [g if len(g) <= cap else g.sample(n=cap, random_state=RANDOM_STATE)
         for _, g in tr.groupby("_y")]
bal = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE)

# --- arm C: same size, original proportions ---
ctrl = tr.sample(n=len(bal), random_state=RANDOM_STATE)

arms = {"A_full": tr, "B_balanced": bal, "C_size_control": ctrl}
rows = []

for name, data in arms.items():
    print(f"\n{'='*60}\n{name}  (n={len(data):,})\n{'='*60}")
    print(data["_y"].value_counts().to_string())
    Xa, ya = data.drop(columns=["_y"]), data["_y"]

    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                      n_jobs=-1,
                                      class_weight="balanced_subsample",
                                      max_depth=MAX_DEPTH)),
    ])
    model.fit(Xa, ya)
    yp = model.predict(Xte)

    rows.append({"arm": name, "train_n": len(data),
                 "accuracy": accuracy_score(yte, yp),
                 "f1_macro": f1_score(yte, yp, average="macro"),
                 "f1_weighted": f1_score(yte, yp, average="weighted")})
    print(classification_report(yte, yp, digits=4, zero_division=0))

    cm = pd.DataFrame(confusion_matrix(yte, yp),
                      index=sorted(y.unique()), columns=sorted(y.unique()))
    cm.to_csv(os.path.join(WORKING_DIR, f"cm_{name}_{TAG}.csv"),
              encoding="utf-8-sig")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(WORKING_DIR, f"undersample_control_{TAG}.csv"),
           index=False, encoding="utf-8-sig")
print("\n=== CONTROLLED COMPARISON ===")
print(res.to_string(index=False))
