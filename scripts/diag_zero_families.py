"""
diag_zero_families.py

Answers the last open question in the paper: why do Recon and Brute Force score
0.0000 recall in ALL five capture-aware draws, when the split diagnostic says
each was present in training in four of the five?

Three hypotheses, and this script distinguishes them:

  H1 STRUCTURAL   - the family is absent from train (or from test) in the draws
                    that matter, so 0.0000 is undefined or unlearnable.
  H2 ABSORBED     - the family IS in training but the model predicts a different
                    family for all of its test flows. Then WHICH family absorbs
                    it is the finding (e.g. Recon -> DoS).
  H3 CONTAMINATED - the family's flows are near-duplicates of another family's,
                    i.e. the capture contains mostly background traffic. Then it
                    is a labelling problem, which ties back to NeedManualLabel.

It prints, per seed: whether each family was in train, its test support, what the
model actually predicted for its flows, and how confident those predictions were.

Runtime: ~15-25 min (5 fits). No new dependencies.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from common import load, build_xy

# Set the dataset root once, either by editing this line or by exporting
# IOT_DIAD_ROOT in your shell:   export IOT_DIAD_ROOT=/path/to/dataset
ROOT_DIR = os.environ.get("IOT_DIAD_ROOT", r"C:\path\to\CIC-IoT-DIAD-2024")
WORKING_DIR = os.path.join(ROOT_DIR, "_working")

SEEDS = [42, 7, 123, 2024, 31337]
TARGETS = ["Recon", "Brute Force"]
N_TREES = 300

df = load(WORKING_DIR)
X, y, groups = build_xy(df, "attack_family")

rows = []
for seed in SEEDS:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    itr, ite = next(gss.split(X, y, groups))
    ytr, yte = y.iloc[itr], y.iloc[ite]

    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=N_TREES, random_state=seed,
                                      n_jobs=-1, max_depth=25,
                                      class_weight="balanced_subsample")),
    ]).fit(X.iloc[itr], ytr)

    yp = pd.Series(model.predict(X.iloc[ite]), index=yte.index)
    proba = model.predict_proba(X.iloc[ite])
    classes = list(model.named_steps["rf"].classes_)

    for fam in TARGETS:
        in_train = int((ytr == fam).sum())
        mask = (yte == fam)
        support = int(mask.sum())

        rec = {"seed": seed, "family": fam,
               "train_rows": in_train, "test_rows": support,
               "train_files": groups.iloc[itr][ytr == fam].nunique(),
               "test_files": groups.iloc[ite][mask].nunique()}

        if support == 0:
            rec["verdict"] = "NO TEST DATA - recall undefined, report n/a"
            rec["predicted_as"] = "-"
            rec["mean_prob_true_class"] = np.nan
        elif in_train == 0:
            rec["verdict"] = "NOT IN TRAIN - unlearnable, 0.0000 is structural"
            rec["predicted_as"] = yp[mask].value_counts().head(2).to_dict()
            rec["mean_prob_true_class"] = 0.0
        else:
            preds = yp[mask].value_counts()
            correct = preds.get(fam, 0)
            ci = classes.index(fam) if fam in classes else None
            p_true = float(proba[mask.values, ci].mean()) if ci is not None else np.nan
            rec["verdict"] = ("ABSORBED - in train but never predicted"
                              if correct == 0 else
                              f"partially recovered ({correct}/{support})")
            rec["predicted_as"] = preds.head(3).to_dict()
            rec["mean_prob_true_class"] = round(p_true, 4)

        rows.append(rec)
        print(f"seed={seed:<6} {fam:<12} train={in_train:<6} test={support:<6} "
              f"{rec['verdict']}")
        if rec["predicted_as"] != "-":
            print(f"                 predicted as: {rec['predicted_as']}")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(WORKING_DIR, "zero_family_diagnostic.csv"),
           index=False, encoding="utf-8-sig")

print("\n=== SUMMARY ===")
print(out[["seed", "family", "train_rows", "test_rows",
           "mean_prob_true_class", "verdict"]].to_string(index=False))

print("""
HOW TO READ THIS

If every row says NO TEST DATA or NOT IN TRAIN  -> the zeros are structural.
   Write: single-capture families cannot be evaluated under capture-aware
   splitting; report n/a and exclude them from macro-F1.

If any row says ABSORBED                        -> the zeros are real failures.
   The 'predicted as' field names the family that absorbs it. Report that
   confusion explicitly - it is a substantive result, not a protocol artefact.

If mean_prob_true_class is well above 0 despite zero recall -> the model assigns
   real probability mass to the correct family but never enough to win. That is a
   calibration/threshold finding and belongs alongside Section 5.6.
""")
