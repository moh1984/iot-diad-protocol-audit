"""
common.py - Shared preprocessing for all revised experiments.

Fixes applied vs. the original scripts:
  * Src Port / Dst Port can be dropped or one-hot bucketed (they were silently
    kept as continuous numerics before -> shortcut learning).
  * Zero-variance columns are removed and REPORTED (7 exist in the sample).
  * Exact-duplicate flows are removed (they inflate test scores).
  * A `source_file` group column is preserved so we can do capture-aware splits.
"""
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42

ID_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"]
TARGET_COLS = ["attack_family", "attack_binary"]
GROUP_COL = "source_file"
PORT_COLS = ["Src Port", "Dst Port"]


def build_xy(df, target, drop_ports=False, drop_constant=True,
             dedup=True, verbose=True):
    """Return X (numeric DataFrame), y (Series), groups (Series or None)."""
    df = df.copy()

    groups = df[GROUP_COL].copy() if GROUP_COL in df.columns else None

    if dedup:
        feat_cols = [c for c in df.columns
                     if c not in ID_COLS + TARGET_COLS + [GROUP_COL]]
        before = len(df)
        keep = ~df.duplicated(subset=feat_cols, keep="first")
        df = df[keep]
        if groups is not None:
            groups = groups[keep]
        if verbose:
            print(f"[dedup] removed {before - len(df)} duplicate flows "
                  f"({100*(before-len(df))/before:.2f}%)")

    y = df[target]
    drop = [c for c in ID_COLS + TARGET_COLS + [GROUP_COL] if c in df.columns]
    X = df.drop(columns=drop, errors="ignore")

    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    if drop_ports:
        present = [c for c in PORT_COLS if c in X.columns]
        X = X.drop(columns=present)
        if verbose:
            print(f"[ports] dropped: {present}")

    if drop_constant:
        const = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        X = X.drop(columns=const)
        if verbose:
            print(f"[const] dropped {len(const)} zero-variance columns: {const}")

    if verbose:
        print(f"[shape] X={X.shape}  y={y.shape}")
    return X.reset_index(drop=True), y.reset_index(drop=True), (
        groups.reset_index(drop=True) if groups is not None else None)


def load(working_dir, filename="iot_diad_sample_by_class.csv"):
    path = os.path.join(working_dir, filename)
    print("Loading:", path)
    df = pd.read_csv(path, low_memory=False)
    print("Shape:", df.shape)
    return df
