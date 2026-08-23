"""
exp_F_shap_and_threshold_fixed.py
Replaces step8_shap_binary_rf.py and the tuning half of step4.

TWO PROBLEMS THIS FIXES

1) MODEL MISMATCH. The previous implementation explained a forest that differed
   from the model used for the reported experiment, so the attribution figure
   described a configuration appearing nowhere else. This script computes SHAP
   on exactly the model whose metrics are reported, at whichever MAX_DEPTH is
   set below, so the depth-25 and unbounded rankings are directly comparable.

2) THRESHOLD SELECTED ON THE TEST SET. The earlier sweep scored candidate
   thresholds against y_test and picked the best, which is model selection on
   test. Here the sweep runs on a held-out VALIDATION split, the threshold is
   frozen, and only then applied to test.

The SHAP subsample is 600 (the previous implementation used 300; 2000 proved too
slow for TreeSHAP on this forest).
"""
import os
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             classification_report)
from sklearn.metrics import f1_score as _f1
from common import load, build_xy, RANDOM_STATE

# Set the dataset root once, either by editing this line or by exporting
# IOT_DIAD_ROOT in your shell:   export IOT_DIAD_ROOT=/path/to/dataset
ROOT_DIR = os.environ.get("IOT_DIAD_ROOT", r"C:\path\to\CIC-IoT-DIAD-2024")
WORKING_DIR = os.path.join(ROOT_DIR, "_working")

N_TREES = 300          # must match every other binary experiment
MAX_DEPTH = 25         # set to None for the unbounded variant
TAG = "depth25" if MAX_DEPTH is not None else "unbounded"
SHAP_N = 600            # lowered: TreeSHAP exceeded 1h at 2000
DROP_PORTS = False     # flip to True to get the port-free SHAP ranking too

df = load(WORKING_DIR)
X, y, _ = build_xy(df, "attack_binary", drop_ports=DROP_PORTS)

# --- three-way split: train / validation / test ---
X_tmp, X_te, y_tmp, y_te = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tmp, y_tmp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tmp)
print(f"train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}")

imp = SimpleImputer(strategy="median")
X_tr_i = pd.DataFrame(imp.fit_transform(X_tr), columns=X.columns)
X_va_i = pd.DataFrame(imp.transform(X_va), columns=X.columns)
X_te_i = pd.DataFrame(imp.transform(X_te), columns=X.columns)

rf = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE,
                            n_jobs=-1, class_weight="balanced_subsample",
                            max_depth=MAX_DEPTH)
rf.fit(X_tr_i, y_tr)

# ---------- threshold sweep ON VALIDATION ----------
p_va = rf.predict_proba(X_va_i)[:, 1]
rows = []
for th in np.arange(0.30, 0.995, 0.02):
    yp = (p_va >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_va, yp).ravel()
    rows.append({"threshold": round(th, 2),
                 "accuracy": accuracy_score(y_va, yp),
                 "precision_attack": precision_score(y_va, yp, zero_division=0),
                 "recall_attack": recall_score(y_va, yp, zero_division=0),
                 "f1_attack": f1_score(y_va, yp, zero_division=0),
                 "precision_benign": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
                 "benign_recall": tn / (tn + fp),
                 "f1_benign": _f1(y_va, yp, pos_label=0, zero_division=0),
                 "f1_macro": _f1(y_va, yp, average="macro", zero_division=0),
                 "FP": fp, "FN": fn})
sweep = pd.DataFrame(rows)

# SELECTION CRITERION: macro-F1.
# The earlier 0.7*f1_attack + 0.3*benign_recall was broken: it rewarded benign
# recall with no penalty on benign precision, so it climbed monotonically to the
# edge of the range and selected an operating point where benign precision fell
# to 0.16 and macro-F1 got WORSE. Macro-F1 balances both classes with no
# arbitrary weights and cannot be gamed by trading precision for recall.
sweep["combo"] = sweep["f1_macro"]
best_th = float(sweep.sort_values("f1_macro", ascending=False).iloc[0]["threshold"])
sweep.to_csv(os.path.join(WORKING_DIR, f"threshold_sweep_validation_{TAG}.csv"),
             index=False, encoding="utf-8-sig")
print("\n=== VALIDATION SWEEP ===")
print(sweep.to_string(index=False))
print(f"\nFrozen threshold (chosen on validation): {best_th}")

# ---------- apply frozen threshold to TEST ----------
p_te = rf.predict_proba(X_te_i)[:, 1]
yp_te = (p_te >= best_th).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, yp_te).ravel()
print(f"\n=== TEST @ threshold={best_th} (unbiased operating point) ===")
print(f"Accuracy      : {accuracy_score(y_te, yp_te):.4f}")
print(f"Attack recall : {tp/(tp+fn):.4f}")
print(f"Benign recall : {tn/(tn+fp):.4f}")
print(f"Benign prec.  : {tn/(tn+fn) if (tn+fn)>0 else 0:.4f}")
print(f"ROC-AUC       : {roc_auc_score(y_te, p_te):.4f}")
print(classification_report(y_te, yp_te, digits=4, zero_division=0))

# also report the full test-set sweep for the trade-off figure (descriptive only)
rows = []
for th in np.arange(0.30, 0.995, 0.02):
    yp = (p_te >= th).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_te, yp).ravel()
    rows.append({"threshold": round(th, 2),
                 "accuracy": accuracy_score(y_te, yp),
                 "precision_attack": precision_score(y_te, yp, zero_division=0),
                 "recall_attack": recall_score(y_te, yp, zero_division=0),
                 "f1_attack": f1_score(y_te, yp, zero_division=0),
                 "precision_benign": tn2 / (tn2 + fn2) if (tn2 + fn2) > 0 else 0.0,
                 "benign_recall": tn2 / (tn2 + fp2),
                 "f1_benign": _f1(y_te, yp, pos_label=0, zero_division=0),
                 "f1_macro": _f1(y_te, yp, average="macro", zero_division=0),
                 "FP": fp2, "FN": fn2})
pd.DataFrame(rows).to_csv(
    os.path.join(WORKING_DIR, f"threshold_sweep_test_descriptive_{TAG}.csv"),
    index=False, encoding="utf-8-sig")

# ---------- SHAP on the SAME model ----------
print(f"\nComputing SHAP on {SHAP_N} test records (model: {N_TREES} trees)...")
Xs = X_te_i.sample(n=min(SHAP_N, len(X_te_i)), random_state=RANDOM_STATE)
sv = shap.TreeExplainer(rf).shap_values(Xs, check_additivity=False)

if isinstance(sv, list):
    sv_atk = sv[1] if len(sv) >= 2 else sv[0]
else:
    sv_atk = sv[:, :, 1] if getattr(sv, "ndim", 2) == 3 else sv
sv_atk = np.asarray(sv_atk)

shap_df = pd.DataFrame({
    "feature": Xs.columns,
    "mean_abs_shap": np.abs(sv_atk).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

rf_df = pd.DataFrame({"feature": X_tr_i.columns,
                      "rf_importance": rf.feature_importances_})

cmp = shap_df.merge(rf_df, on="feature")
cmp["shap_rank"] = cmp["mean_abs_shap"].rank(ascending=False).astype(int)
cmp["rf_rank"] = cmp["rf_importance"].rank(ascending=False).astype(int)

tag = "noports" if DROP_PORTS else "withports"
cmp.to_csv(os.path.join(WORKING_DIR, f"shap_vs_rf_{tag}_{TAG}.csv"),
           index=False, encoding="utf-8-sig")

print("\nTop 20 by SHAP:")
print(cmp.head(20).to_string(index=False))

# Spearman agreement gives you a NUMBER instead of "strong alignment"
rho = cmp["shap_rank"].corr(cmp["rf_rank"], method="spearman")
print(f"\nSpearman rank correlation SHAP vs RF importance: {rho:.4f}")
print("(cite this instead of the vague 'strong alignment' in Section 4.6)")
