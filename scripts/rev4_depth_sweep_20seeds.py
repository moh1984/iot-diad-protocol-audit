"""
rev4_depth_sweep_20seeds.py   -- brings Table 7 into line with Tables 2-5

Table 7 was the last result still resting on five seeds, with its capture-aware
column averaging only the two of those five whose test partitions contained
benign traffic. This repeats the sweep at twenty seeds across all three
protocols, so the depth finding rests on the same evidential basis as everything
else in the paper.

Three changes from the original sweep:

  * Twenty seeds, matching rev1, rather than five.
  * All three protocols, including the size-matched capture-aware protocol added
    for this revision, rather than two.
  * Bootstrap 95% intervals on every reported mean, and an explicit count of the
    draws in which benign traffic was testable, since that count is what limited
    the original version.

The grid is trimmed to the region that matters. The original sweep showed the
curve to be monotone in benign recall outside 6-30, so depths 8 and 16 are
dropped; depth is the variable under study, so tree count stays at 200 as before
to keep the comparison with the published Table 7 exact.

Runtime: roughly 4-7 hours (20 seeds x 7 depths x 3 protocols = 420 fits), which
is longer than the earlier scripts. Two ways to shorten it if needed:
  QUICK = True   -> 10 seeds, 5 depths (~1.5-3 h), still double the current basis
  SKIP_RANDOM    -> omit the random-split arm, which was already stable at five
                    seeds (sigma = 0.0018), and which rev1 re-measured at twenty

Resumable: rerun to skip completed cells.
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

ROOT_DIR = os.environ.get("IOT_DIAD_ROOT", r"C:\path\to\CIC-IoT-DIAD-2024")
WORKING_DIR = os.path.join(ROOT_DIR, "_working")

QUICK = False
SKIP_RANDOM = False

ALL_SEEDS = [42, 7, 123, 2024, 31337, 1, 99, 555, 8, 2718,
             31415, 161803, 271828, 4, 77, 900, 12321, 5150, 60606, 24680]
SEEDS = ALL_SEEDS[:10] if QUICK else ALL_SEEDS
DEPTHS = [6, 12, 20, 30, None] if QUICK else [6, 10, 12, 20, 25, 30, None]
N_TREES = 200
TARGET = 0.20

CKPT = os.path.join(WORKING_DIR, "rev4_depth_sweep_20seeds_raw.csv")


def size_matched_split(groups, y, seed, target=TARGET):
    rng = np.random.default_rng(seed)
    sizes = groups.value_counts()
    order = rng.permutation(sizes.index.to_numpy())
    want = int(round(target * len(groups)))
    test_groups, acc = [], 0
    for g in order:
        if acc >= want:
            break
        test_groups.append(g)
        acc += sizes[g]
    mask = groups.isin(test_groups).to_numpy()
    return np.where(~mask)[0], np.where(mask)[0]


df = load(WORKING_DIR)
X, y, groups = build_xy(df, "attack_binary")

rows, done = [], set()
if os.path.exists(CKPT):
    prev = pd.read_csv(CKPT)
    rows = prev.to_dict("records")
    done = {(r["protocol"], r["seed"], str(r["max_depth"])) for r in rows}
    print(f"[resume] {len(done)} cells already complete")

protocols = ["capture-aware (file-proportion)", "capture-aware (size-matched)"]
if not SKIP_RANDOM:
    protocols.insert(0, "random")

for seed in SEEDS:
    splits = {}
    if not SKIP_RANDOM:
        itr, ite = train_test_split(np.arange(len(y)), test_size=0.2,
                                    random_state=seed, stratify=y)
        splits["random"] = (itr, ite)
    splits["capture-aware (file-proportion)"] = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2,
                          random_state=seed).split(X, y, groups))
    splits["capture-aware (size-matched)"] = size_matched_split(groups, y, seed)

    for protocol in protocols:
        itr, ite = splits[protocol]
        ytr, yte = y.iloc[itr], y.iloc[ite]
        single_class = yte.nunique() < 2

        for d in DEPTHS:
            key = (protocol, seed, str(d))
            if key in done:
                continue
            model = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("rf", RandomForestClassifier(
                    n_estimators=N_TREES, random_state=seed, n_jobs=-1,
                    max_depth=d, class_weight="balanced_subsample")),
            ]).fit(X.iloc[itr], ytr)
            yp = model.predict(X.iloc[ite])
            prob = model.predict_proba(X.iloc[ite])[:, 1]
            tn, fp, fn, tp = confusion_matrix(yte, yp, labels=[0, 1]).ravel()
            rf = model.named_steps["rf"]

            rows.append({
                "protocol": protocol, "seed": seed,
                "max_depth": "None" if d is None else d,
                "single_class_test": single_class,
                "mean_realised_depth": round(float(np.mean(
                    [t.get_depth() for t in rf.estimators_])), 1),
                "test_rows": len(ite),
                "accuracy": accuracy_score(yte, yp),
                "f1_macro": f1_score(yte, yp, average="macro", zero_division=0),
                "benign_recall": (tn / (tn + fp)) if (tn + fp) and not single_class else np.nan,
                "benign_precision": (tn / (tn + fn)) if (tn + fn) and not single_class else np.nan,
                "attack_recall": tp / (tp + fn) if (tp + fn) else np.nan,
                "roc_auc": np.nan if single_class else roc_auc_score(yte, prob),
                "FP": fp, "FN": fn,
            })
            pd.DataFrame(rows).to_csv(CKPT, index=False, encoding="utf-8-sig")
        print(f"[{protocol:<32}] seed={seed:<7} "
              f"{'(single-class test)' if single_class else ''}")

raw = pd.DataFrame(rows)


def boot_ci(v, n_boot=10000, seed=0):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    return tuple(np.percentile(
        [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)],
        [2.5, 97.5]))


out = []
for (proto, d), g in raw.groupby(["protocol", "max_depth"], dropna=False):
    ok = g[~g.single_class_test]
    blo, bhi = boot_ci(ok.benign_recall)
    flo, fhi = boot_ci(ok.f1_macro)
    out.append({
        "protocol": proto, "max_depth": d,
        "realised_depth": g.mean_realised_depth.mean(),
        "draws_total": g.seed.nunique(),
        "draws_with_benign": int(ok.benign_recall.notna().sum()),
        "benign_recall_mean": ok.benign_recall.mean(),
        "benign_recall_sd": ok.benign_recall.std(ddof=1),
        "benign_ci_lo": blo, "benign_ci_hi": bhi,
        "f1_macro_mean": ok.f1_macro.mean(),
        "f1_macro_sd": ok.f1_macro.std(ddof=1),
        "f1_ci_lo": flo, "f1_ci_hi": fhi,
        "accuracy_mean": ok.accuracy.mean(),
    })
summ = pd.DataFrame(out)

order = {str(k): i for i, k in enumerate([6, 10, 12, 20, 25, 30, "None"])}
summ["_o"] = summ.max_depth.astype(str).map(order)
summ = summ.sort_values(["protocol", "_o"]).drop(columns="_o")
summ.to_csv(os.path.join(WORKING_DIR, "rev4_depth_sweep_20seeds_summary.csv"),
            index=False, encoding="utf-8-sig")

pd.set_option("display.width", 240)
print("\n=== DEPTH SWEEP, 20 SEEDS, THREE PROTOCOLS (new Table 7) ===")
print(summ.round(4).to_string(index=False))

print("\n=== ARGMAX macro-F1 per protocol ===")
for p in summ.protocol.unique():
    s = summ[summ.protocol == p].sort_values("f1_macro_mean", ascending=False)
    b, r = s.iloc[0], s.iloc[1]
    overlap = b.f1_ci_lo <= r.f1_ci_hi
    print(f"{p:<34} best={b.max_depth} f1={b.f1_macro_mean:.4f} "
          f"({b.f1_ci_lo:.4f}-{b.f1_ci_hi:.4f}) | runner-up={r.max_depth} "
          f"f1={r.f1_macro_mean:.4f} | intervals overlap: {overlap}")

print("""
HOW TO WRITE THIS UP

  * Replace Table 7 with benign_recall_mean and f1_macro_mean, each with its
    bootstrap interval, and keep the draws_with_benign column: it is the count
    that limited the previous version and the reviewer asked about it directly.
  * The claim to check is whether max_depth=None still gives the lowest benign
    recall under every protocol. If the intervals at the shallowest depth and at
    None are far apart, the claim holds at twenty seeds and the hedging added in
    the previous revision ("indicative only") can be removed.
  * Where the best and runner-up intervals overlap, report an operating region
    rather than an optimum, as the paper already does.
""")
