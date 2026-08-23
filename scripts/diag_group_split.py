"""
diag_group_split.py

Settles one question before you write it into the paper: WHY do Brute Force and
Recon score 0.0000 recall under the capture-aware split?

Brute Force is obvious - one capture file, so holding it out removes the family
from training entirely. Recon also has one file, yet showed 2,717 test records.
Both cannot be true unless the whole family landed on the test side. This script
prints exactly which families survive in train vs test, so you can state the
mechanism instead of guessing at it.

Runs in seconds. No model training.
"""
import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from common import load, build_xy, RANDOM_STATE

# Set the dataset root once, either by editing this line or by exporting
# IOT_DIAD_ROOT in your shell:   export IOT_DIAD_ROOT=/path/to/dataset
ROOT_DIR = os.environ.get("IOT_DIAD_ROOT", r"C:\path\to\CIC-IoT-DIAD-2024")
WORKING_DIR = os.path.join(ROOT_DIR, "_working")

df = load(WORKING_DIR)
X, y, groups = build_xy(df, "attack_family", verbose=False)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
itr, ite = next(gss.split(X, y, groups))

tr, te = y.iloc[itr], y.iloc[ite]
gtr, gte = groups.iloc[itr], groups.iloc[ite]

rows = []
for fam in sorted(y.unique()):
    n_tr, n_te = (tr == fam).sum(), (te == fam).sum()
    rows.append({
        "family": fam,
        "train_rows": n_tr,
        "test_rows": n_te,
        "train_files": gtr[tr == fam].nunique(),
        "test_files": gte[te == fam].nunique(),
        "total_files": groups[y == fam].nunique(),
        "trainable": "NO - absent from train" if n_tr == 0 else "yes",
    })

out = pd.DataFrame(rows)
print("\n=== FAMILY AVAILABILITY UNDER CAPTURE-AWARE SPLIT ===")
print(out.to_string(index=False))
out.to_csv(os.path.join(WORKING_DIR, "group_split_diagnostic.csv"),
           index=False, encoding="utf-8-sig")

dead = out[out.train_rows == 0]["family"].tolist()
print(f"\nFamilies with ZERO training data: {dead}")
print("These cannot exceed 0.0000 recall by construction - it is a property of")
print("the protocol on single-capture families, not a model failure. State this")
print("explicitly in the paper rather than reporting the zeros unexplained.")

# how many families sit on a single capture file at all
single = out[out.total_files == 1]["family"].tolist()
print(f"\nSingle-capture families (structurally at risk): {single}")
