"""
rev3_per_family_20seeds.py   -- completes reviewer comment 3 for Table 3

rev1 recorded aggregate metrics only, so Table 3 (per-family recall) still rests
on five draws while every other table has been re-run at twenty. This script
fills that gap: same twenty seeds, same two capture-aware protocols, recording
recall, precision and support for every family in every draw, and marking which
families were scorable.

Runtime: ~2-3 hours (40 fits). Resumable.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from common import load, build_xy

ROOT_DIR = os.environ.get("IOT_DIAD_ROOT", r"C:\path\to\CIC-IoT-DIAD-2024")
WORKING_DIR = os.path.join(ROOT_DIR, "_working")

SEEDS = [42, 7, 123, 2024, 31337, 1, 99, 555, 8, 2718,
         31415, 161803, 271828, 4, 77, 900, 12321, 5150, 60606, 24680]
N_TREES, MAX_DEPTH, TARGET = 300, 25, 0.20
CKPT = os.path.join(WORKING_DIR, "rev3_per_family_raw.csv")


def size_matched_split(groups, y, seed, target=TARGET):
    rng = np.random.default_rng(seed)
    sizes = groups.value_counts()
    order = rng.permutation(sizes.index.to_numpy())
    want = int(round(target * len(groups)))
    test_groups, acc = [], 0
    for g in order:
        if acc >= want:
            break
        test_groups.append(g); acc += sizes[g]
    mask = groups.isin(test_groups).to_numpy()
    return np.where(~mask)[0], np.where(mask)[0]


df = load(WORKING_DIR)
X, y, groups = build_xy(df, "attack_family")
families = sorted(y.unique())

rows, done = [], set()
if os.path.exists(CKPT):
    prev = pd.read_csv(CKPT)
    rows = prev.to_dict("records")
    done = {(r["protocol"], r["seed"]) for r in rows}
    print(f"[resume] {len(done)} runs complete")

for seed in SEEDS:
    for protocol in ["capture-aware (file-proportion)",
                     "capture-aware (size-matched)"]:
        if (protocol, seed) in done:
            continue
        if "file-proportion" in protocol:
            itr, ite = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                              random_state=seed
                                              ).split(X, y, groups))
        else:
            itr, ite = size_matched_split(groups, y, seed)

        ytr, yte = y.iloc[itr], y.iloc[ite]
        m = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(n_estimators=N_TREES,
                                          random_state=seed, n_jobs=-1,
                                          max_depth=MAX_DEPTH,
                                          class_weight="balanced_subsample")),
        ]).fit(X.iloc[itr], ytr)
        yp = m.predict(X.iloc[ite])
        rep = classification_report(yte, yp, output_dict=True, zero_division=0)
        scored = set(ytr.unique()) & set(yte.unique())

        for fam in families:
            rows.append({
                "protocol": protocol, "seed": seed, "family": fam,
                "in_train": fam in set(ytr.unique()),
                "in_test": fam in set(yte.unique()),
                "scored": fam in scored,
                "recall": rep[fam]["recall"] if fam in scored else np.nan,
                "precision": rep[fam]["precision"] if fam in scored else np.nan,
                "support": int(rep[fam]["support"]) if fam in rep else 0,
            })
        pd.DataFrame(rows).to_csv(CKPT, index=False, encoding="utf-8-sig")
        print(f"[{protocol:<32}] seed={seed:<7} scored={len(scored)}/8")

raw = pd.DataFrame(rows)


def boot_ci(v, n_boot=10000, seed=0):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    return tuple(np.percentile(
        [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)],
        [2.5, 97.5]))


out = []
for (proto, fam), g in raw.groupby(["protocol", "family"]):
    s = g[g.scored]
    lo, hi = boot_ci(s.recall)
    out.append({"protocol": proto, "family": fam,
                "draws_scored": len(s), "draws_total": g.seed.nunique(),
                "mean_recall": s.recall.mean() if len(s) else np.nan,
                "sd": s.recall.std(ddof=1) if len(s) > 1 else np.nan,
                "ci_lo": lo, "ci_hi": hi,
                "mean_support": s.support.mean() if len(s) else np.nan})
summ = pd.DataFrame(out).sort_values(["protocol", "mean_recall"],
                                     ascending=[True, False])
summ.to_csv(os.path.join(WORKING_DIR, "rev3_per_family_summary.csv"),
            index=False, encoding="utf-8-sig")

pd.set_option("display.width", 220)
print("\n=== PER-FAMILY RECALL, 20 SEEDS (new Table 3) ===")
print(summ.round(4).to_string(index=False))
print("\nFamilies never scored in any draw must appear as n/a, not 0.0000.")
