"""
rev1_protocol_20seeds.py   -- addresses reviewer comments 3, 4 and 7

THREE THINGS THE REVIEWER ASKED FOR, AND WHAT THIS DOES ABOUT EACH.

(3) "Five group splits, test partitions between 8% and 32%, scorable classes
    change across splits -- means over such heterogeneous partitions do not
    represent a stable common test condition."

    Correct, and more seeds alone would NOT fix it: repeating a heterogeneous
    protocol 20 times measures its instability more precisely without making the
    partitions comparable. So this script does two things. It raises the seed
    count to 20, AND it adds a third protocol -- capture-aware splitting with a
    size-matched test partition, where captures are accumulated until ~20% of
    RECORDS are held out rather than 20% of FILES. That gives a capture-aware
    condition whose test size is stable, so the generalisation gap can be read
    without the confound the reviewer identifies.

(4) "'Sixty-seven times noisier' is overstated: a ratio of two SDs from five
    observations, with no confidence interval."

    Also correct. This script bootstraps the variance ratio (10,000 resamples)
    and reports a percentile CI. If the interval is wide, the paper should report
    the interval and drop the point estimate; that is a result either way.

(7) "Conclusions generalise beyond one dataset, one classifier, five seeds."

    The outputs here are what the revised text should be written from: intervals,
    not point estimates.

Runtime: ~3-6 hours on 4+ cores (20 seeds x 3 protocols x 2 tasks = 120 fits).
Set QUICK = True for a 10-seed dry run (~1.5-3 h) before committing to the full run.
Progress is written to disk after every seed, so an interrupted run is resumable
by rerunning: completed seeds are skipped.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from common import load, build_xy

ROOT_DIR = os.environ.get("IOT_DIAD_ROOT", r"C:\path\to\CIC-IoT-DIAD-2024")
WORKING_DIR = os.path.join(ROOT_DIR, "_working")

QUICK = False
N_SEEDS = 10 if QUICK else 20
SEEDS = [42, 7, 123, 2024, 31337, 1, 99, 555, 8, 2718,
         31415, 161803, 271828, 4, 77, 900, 12321, 5150, 60606, 24680][:N_SEEDS]
N_TREES = 300
MAX_DEPTH = 25
TARGET_TEST_FRACTION = 0.20

CKPT = os.path.join(WORKING_DIR, "rev1_protocol_20seeds_raw.csv")


def make_model(seed):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=N_TREES, random_state=seed,
                                      n_jobs=-1, max_depth=MAX_DEPTH,
                                      class_weight="balanced_subsample")),
    ])


def size_matched_capture_split(groups, y, seed, target=TARGET_TEST_FRACTION):
    """Hold out whole captures, but accumulate them until ~target of RECORDS is
    reached. This keeps the unit of independence (the capture) while stabilising
    the test-set size, which file-proportion splitting does not."""
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


def evaluate(X, y, itr, ite, seed, protocol, task):
    ytr, yte = y.iloc[itr], y.iloc[ite]
    model = make_model(seed).fit(X.iloc[itr], ytr)
    yp = model.predict(X.iloc[ite])

    scored = sorted(set(ytr.unique()) & set(yte.unique()))
    row = {
        "task": task, "protocol": protocol, "seed": seed,
        "test_rows": len(ite),
        "test_frac": round(len(ite) / len(y), 4),
        "n_scored_families": len(scored),
        "accuracy": accuracy_score(yte, yp),
        "f1_macro_all": f1_score(yte, yp, average="macro",
                                 labels=sorted(y.unique()), zero_division=0),
        "f1_macro_scored": (f1_score(yte, yp, average="macro", labels=scored,
                                     zero_division=0) if scored else np.nan),
    }
    if task == "binary":
        if yte.nunique() < 2:
            row.update({"benign_recall": np.nan, "benign_precision": np.nan})
        else:
            tn, fp, fn, tp = confusion_matrix(yte, yp, labels=[0, 1]).ravel()
            row["benign_recall"] = tn / (tn + fp) if (tn + fp) else np.nan
            row["benign_precision"] = tn / (tn + fn) if (tn + fn) else np.nan
    return row


df = load(WORKING_DIR)

done = set()
rows = []
if os.path.exists(CKPT):
    prev = pd.read_csv(CKPT)
    rows = prev.to_dict("records")
    done = {(r["task"], r["protocol"], r["seed"]) for r in rows}
    print(f"[resume] {len(done)} runs already complete; skipping those")

for task, target in [("binary", "attack_binary"),
                     ("multiclass", "attack_family")]:
    X, y, groups = build_xy(df, target, verbose=(task == "binary"))
    if groups is None:
        raise SystemExit("source_file column missing - rerun step0 first.")

    for seed in SEEDS:
        plans = {
            "random record":
                lambda s=seed: train_test_split(
                    np.arange(len(y)), test_size=0.2, random_state=s,
                    stratify=y)[:2],
            "capture-aware (file-proportion)":
                lambda s=seed: next(GroupShuffleSplit(
                    n_splits=1, test_size=0.2, random_state=s
                ).split(X, y, groups)),
            "capture-aware (size-matched)":
                lambda s=seed: size_matched_capture_split(groups, y, s),
        }
        for protocol, plan in plans.items():
            if (task, protocol, seed) in done:
                continue
            if protocol == "random record":
                itr, ite = train_test_split(
                    np.arange(len(y)), test_size=0.2, random_state=seed,
                    stratify=y)
            else:
                itr, ite = plan()
            rows.append(evaluate(X, y, itr, ite, seed, protocol, task))
            pd.DataFrame(rows).to_csv(CKPT, index=False, encoding="utf-8-sig")
            r = rows[-1]
            print(f"[{task:<10}] {protocol:<32} seed={seed:<7} "
                  f"test={r['test_frac']:.3f} acc={r['accuracy']:.4f} "
                  f"scored={r['n_scored_families']}/8")

res = pd.DataFrame(rows)

# ---------------- summary with intervals, not point estimates ----------------
def ci(v, lo=2.5, hi=97.5, n_boot=10000, seed=0):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)]
    return (np.percentile(bs, lo), np.percentile(bs, hi))


summ = []
for (task, proto), g in res.groupby(["task", "protocol"]):
    lo, hi = ci(g.accuracy)
    slo, shi = ci(g.f1_macro_scored)
    summ.append({
        "task": task, "protocol": proto, "n_seeds": len(g),
        "test_frac_min": g.test_frac.min(), "test_frac_max": g.test_frac.max(),
        "scored_families_min": g.n_scored_families.min(),
        "scored_families_max": g.n_scored_families.max(),
        "accuracy_mean": g.accuracy.mean(), "accuracy_sd": g.accuracy.std(ddof=1),
        "accuracy_ci_lo": lo, "accuracy_ci_hi": hi,
        "f1_scored_mean": g.f1_macro_scored.mean(),
        "f1_scored_sd": g.f1_macro_scored.std(ddof=1),
        "f1_scored_ci_lo": slo, "f1_scored_ci_hi": shi,
    })
summary = pd.DataFrame(summ)
summary.to_csv(os.path.join(WORKING_DIR, "rev1_protocol_20seeds_summary.csv"),
               index=False, encoding="utf-8-sig")

# ---------------- bootstrap CI on the variance RATIO (comment 4) ----------------
def sd_ratio_ci(a, b, n_boot=10000, seed=0):
    """CI for sd(b)/sd(a) by resampling each group independently."""
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        sa = np.std(rng.choice(a, len(a), replace=True), ddof=1)
        sb = np.std(rng.choice(b, len(b), replace=True), ddof=1)
        if sa > 0:
            out.append(sb / sa)
    return np.percentile(out, 2.5), np.median(out), np.percentile(out, 97.5)


ratios = []
for task in res.task.unique():
    t = res[res.task == task]
    base = t[t.protocol == "random record"].accuracy
    for proto in ["capture-aware (file-proportion)",
                  "capture-aware (size-matched)"]:
        comp = t[t.protocol == proto].accuracy
        lo, med, hi = sd_ratio_ci(base, comp)
        ratios.append({"task": task, "protocol": proto,
                       "sd_random": base.std(ddof=1),
                       "sd_protocol": comp.std(ddof=1),
                       "ratio_point": comp.std(ddof=1) / base.std(ddof=1),
                       "ratio_boot_median": med,
                       "ratio_ci_lo": lo, "ratio_ci_hi": hi})
rat = pd.DataFrame(ratios)
rat.to_csv(os.path.join(WORKING_DIR, "rev1_variance_ratio_ci.csv"),
           index=False, encoding="utf-8-sig")

pd.set_option("display.width", 220)
print("\n=== PROTOCOL SUMMARY (means with bootstrap 95% CIs) ===")
print(summary.round(4).to_string(index=False))
print("\n=== VARIANCE RATIO WITH BOOTSTRAP CI (reviewer comment 4) ===")
print(rat.round(3).to_string(index=False))
print("""
HOW TO WRITE THIS UP
  * Report the CI, not the point ratio. If the interval spans an order of
    magnitude, say so plainly: the direction is established, the magnitude is not.
  * The size-matched protocol is the one that answers comment 3. Compare it
    against random splitting for the generalisation gap, and against the
    file-proportion protocol to show how much of the original instability came
    from varying test size rather than from capture independence itself.
  * Report test_frac_min/max and scored_families_min/max for every protocol.
    Those columns are the evidence that the concern was taken seriously.
""")
