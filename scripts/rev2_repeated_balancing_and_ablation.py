"""
rev2_repeated_balancing_and_ablation.py   -- addresses reviewer comments 5 and 6

(5) "The balancing experiment in Table 5 appears to rely on a single random split
    or seed, and no standard deviations or confidence intervals are reported. A
    macro-F1 difference of 0.0627 cannot establish that balancing consistently
    improves performance without repeated subsampling and repeated model fitting."

    The reviewer is right, and the design flaw is subtler than one seed: arms B
    and C each involve a RANDOM SUBSAMPLE, so their results depend on which rows
    were drawn as well as on which split was used. This repeats both the split
    and the subsample across seeds, so the reported interval covers both sources
    of variation.

(6) "The feature-removal experiment evaluates only the two port variables,
    apparently under one split, and no variability is reported. The unchanged
    benign recall at unbounded depth does not show that a highly ranked feature
    is generally unnecessary, especially because multiclass macro-F1 decreases
    after port removal."

    Two responses. First, repeat the ablation across seeds and report paired
    differences with CIs, so "unchanged" becomes a measured claim rather than a
    coincidence of one split. Second, the reviewer's last clause is a fair hit:
    the multiclass drop is real. This script therefore also ablates two CONTROL
    feature pairs -- a top-ranked timing pair and a random pair -- so the port
    result can be read against how much removing any two features costs. Without
    that baseline the ablation cannot distinguish "ports are dispensable" from
    "any two of 72 features are dispensable".

Runtime: ~2-4 hours on 4+ cores. Resumable: rerun to skip completed cells.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from common import load, build_xy

ROOT_DIR = os.environ.get("IOT_DIAD_ROOT", r"C:\path\to\CIC-IoT-DIAD-2024")
WORKING_DIR = os.path.join(ROOT_DIR, "_working")

QUICK = False
N_SEEDS = 5 if QUICK else 15
SEEDS = [42, 7, 123, 2024, 31337, 1, 99, 555, 8, 2718,
         31415, 161803, 271828, 4, 77][:N_SEEDS]
N_TREES = 300
MAX_DEPTH = 25

# Control pairs for comment 6. The timing pair is the top-ranked pair by SHAP in
# the depth-25 run; the arbitrary pair is a fixed, unremarkable choice. Removing
# each tells us what two-feature removal costs in general.
PORT_PAIR = ["Src Port", "Dst Port"]
TIMING_PAIR = ["Flow IAT Min", "Fwd IAT Min"]
ARBITRARY_PAIR = ["Flow IAT Std", "Fwd Packet Length Mean"]

CKPT_BAL = os.path.join(WORKING_DIR, "rev2_balancing_raw.csv")
CKPT_ABL = os.path.join(WORKING_DIR, "rev2_ablation_raw.csv")


def rf(seed):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=N_TREES, random_state=seed,
                                      n_jobs=-1, max_depth=MAX_DEPTH,
                                      class_weight="balanced_subsample")),
    ])


def boot_ci(v, n_boot=10000, seed=0):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)]
    return (np.percentile(bs, 2.5), np.percentile(bs, 97.5))


df = load(WORKING_DIR)

# =====================================================================
# PART 1 -- balancing with size controlled, repeated (comment 5)
# =====================================================================
X, y, _ = build_xy(df, "attack_family")

rows = []
done = set()
if os.path.exists(CKPT_BAL):
    prev = pd.read_csv(CKPT_BAL)
    rows = prev.to_dict("records")
    done = {(r["seed"], r["arm"]) for r in rows}
    print(f"[resume] balancing: {len(done)} cells done")

for seed in SEEDS:
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    tr = Xtr.copy(); tr["_y"] = ytr.values

    cap = tr["_y"].value_counts().min() * 3
    bal = pd.concat([g if len(g) <= cap else g.sample(n=cap, random_state=seed)
                     for _, g in tr.groupby("_y")]).sample(frac=1,
                                                           random_state=seed)
    ctrl = tr.sample(n=len(bal), random_state=seed)

    for arm, data in [("A_full", tr), ("B_balanced", bal),
                      ("C_size_control", ctrl)]:
        if (seed, arm) in done:
            continue
        m = rf(seed).fit(data.drop(columns=["_y"]), data["_y"])
        yp = m.predict(Xte)
        rows.append({"seed": seed, "arm": arm, "train_n": len(data),
                     "accuracy": accuracy_score(yte, yp),
                     "f1_macro": f1_score(yte, yp, average="macro",
                                          zero_division=0)})
        pd.DataFrame(rows).to_csv(CKPT_BAL, index=False, encoding="utf-8-sig")
        print(f"[balance] seed={seed:<7} {arm:<16} "
              f"f1={rows[-1]['f1_macro']:.4f}")

bal_raw = pd.DataFrame(rows)
piv = bal_raw.pivot(index="seed", columns="arm", values="f1_macro")
piv["B_minus_C"] = piv.B_balanced - piv.C_size_control
piv["A_minus_B"] = piv.A_full - piv.B_balanced
piv.to_csv(os.path.join(WORKING_DIR, "rev2_balancing_paired.csv"),
           encoding="utf-8-sig")

bal_summary = []
for col in ["A_full", "B_balanced", "C_size_control", "B_minus_C", "A_minus_B"]:
    lo, hi = boot_ci(piv[col])
    bal_summary.append({"quantity": col, "mean": piv[col].mean(),
                        "sd": piv[col].std(ddof=1), "ci_lo": lo, "ci_hi": hi,
                        "n_seeds": piv[col].notna().sum()})
bal_summary = pd.DataFrame(bal_summary)
bal_summary.to_csv(os.path.join(WORKING_DIR, "rev2_balancing_summary.csv"),
                   index=False, encoding="utf-8-sig")

# =====================================================================
# PART 2 -- feature removal with controls, repeated (comment 6)
# =====================================================================
arows = []
adone = set()
if os.path.exists(CKPT_ABL):
    prev = pd.read_csv(CKPT_ABL)
    arows = prev.to_dict("records")
    adone = {(r["task"], r["seed"], r["variant"], r["depth"]) for r in arows}
    print(f"[resume] ablation: {len(adone)} cells done")

VARIANTS = {"full": [], "no_ports": PORT_PAIR,
            "no_timing_pair": TIMING_PAIR, "no_arbitrary_pair": ARBITRARY_PAIR}

for task, target in [("binary", "attack_binary"),
                     ("multiclass", "attack_family")]:
    Xf, yf, _ = build_xy(df, target, verbose=False)
    for depth in [MAX_DEPTH, None]:
        for seed in SEEDS:
            itr, ite = train_test_split(np.arange(len(yf)), test_size=0.2,
                                        random_state=seed, stratify=yf)
            for variant, drop in VARIANTS.items():
                key = (task, seed, variant, "None" if depth is None else depth)
                if key in adone:
                    continue
                cols = [c for c in Xf.columns if c not in drop]
                missing = [d for d in drop if d not in Xf.columns]
                if missing:
                    raise SystemExit(f"feature(s) not in X: {missing}")
                Xv = Xf[cols]
                m = Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("rf", RandomForestClassifier(
                        n_estimators=N_TREES, random_state=seed, n_jobs=-1,
                        max_depth=depth,
                        class_weight="balanced_subsample")),
                ]).fit(Xv.iloc[itr], yf.iloc[itr])
                yp = m.predict(Xv.iloc[ite])
                yte = yf.iloc[ite]
                rec = {"task": task, "seed": seed, "variant": variant,
                       "depth": "None" if depth is None else depth,
                       "n_features": Xv.shape[1],
                       "accuracy": accuracy_score(yte, yp),
                       "f1_macro": f1_score(yte, yp, average="macro",
                                            zero_division=0)}
                if task == "binary":
                    tn, fp, fn, tp = confusion_matrix(yte, yp,
                                                      labels=[0, 1]).ravel()
                    rec["benign_recall"] = tn / (tn + fp) if (tn + fp) else np.nan
                arows.append(rec)
                pd.DataFrame(arows).to_csv(CKPT_ABL, index=False,
                                           encoding="utf-8-sig")
            print(f"[ablate] {task:<10} depth={str(depth):<5} seed={seed} done")

abl = pd.DataFrame(arows)

# paired differences against the full feature set, per seed
pairs = []
for (task, depth), g in abl.groupby(["task", "depth"]):
    p = g.pivot(index="seed", columns="variant", values="f1_macro")
    for v in ["no_ports", "no_timing_pair", "no_arbitrary_pair"]:
        d = p["full"] - p[v]
        lo, hi = boot_ci(d)
        rec = {"task": task, "depth": depth, "removed": v,
               "mean_f1_drop": d.mean(), "sd": d.std(ddof=1),
               "ci_lo": lo, "ci_hi": hi, "n_seeds": d.notna().sum()}
        if task == "binary":
            pb = g.pivot(index="seed", columns="variant", values="benign_recall")
            db = pb["full"] - pb[v]
            blo, bhi = boot_ci(db)
            rec.update({"mean_benign_recall_drop": db.mean(),
                        "benign_ci_lo": blo, "benign_ci_hi": bhi})
        pairs.append(rec)
abl_summary = pd.DataFrame(pairs)
abl_summary.to_csv(os.path.join(WORKING_DIR, "rev2_ablation_summary.csv"),
                   index=False, encoding="utf-8-sig")

pd.set_option("display.width", 220)
print("\n=== BALANCING, repeated over seeds (reviewer comment 5) ===")
print(bal_summary.round(4).to_string(index=False))
print("\n=== FEATURE REMOVAL vs CONTROLS (reviewer comment 6) ===")
print(abl_summary.round(4).to_string(index=False))
print("""
HOW TO READ THIS

BALANCING. The row that matters is B_minus_C. If its CI excludes zero, the
claim survives the reviewer's objection and should be reported as a mean with an
interval. If it includes zero, the claim must be withdrawn and stated as
inconclusive -- which is a legitimate outcome and consistent with the paper's
own argument about uncontrolled comparisons.

FEATURE REMOVAL. Compare mean_f1_drop for no_ports against no_timing_pair and
no_arbitrary_pair. Removing the ports is only informative if it costs less than
removing an arbitrary pair; if all three cost about the same, the honest
conclusion is that this forest tolerates the loss of any two of 72 correlated
features, which is a weaker but defensible claim. Report the controls either way
-- they are what turns the reviewer's objection into evidence.
""")
