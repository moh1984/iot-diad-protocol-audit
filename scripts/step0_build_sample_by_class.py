"""
step0_build_sample_by_class.py
Replaces step1_test_merge.py.

WHY: the old script took 1,000 rows from EVERY CSV file. DDoS therefore reached
49.2% of the sample only because it happened to have 61 files. The reported
"severe natural imbalance" was an artifact of the folder layout, not of the data.

WHAT THIS DOES:
  1. Counts the TRUE number of rows per class (cheap, chunked read).
  2. Reports the TRUE class proportions -> this is the number your paper must cite.
  3. Draws a sample that PRESERVES those true proportions (proportional mode),
     so the imbalance you analyse is real.
  4. Records `source_file` for every row -> enables capture-aware splitting later.
"""
import os
import glob
import numpy as np
import pandas as pd

# Set the dataset root once, either by editing this line or by exporting
# IOT_DIAD_ROOT in your shell:   export IOT_DIAD_ROOT=/path/to/dataset
ROOT_DIR = os.environ.get("IOT_DIAD_ROOT", r"C:\path\to\CIC-IoT-DIAD-2024")
OUT_DIR = os.path.join(ROOT_DIR, "_working")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_FOLDERS = ["Benign", "Brute Force", "DDOS", "DOS",
                 "Mirai", "Recon", "Spoofing", "Web-Based"]

TOTAL_TARGET = 120_000     # total rows in the working sample
MIN_PER_CLASS = 2_000      # floor so tiny classes stay usable
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


# ---------- pass 1: true row counts (no full load) ----------
print("=== PASS 1: counting true rows per class ===")
inventory = {}
for cls in CLASS_FOLDERS:
    path = os.path.join(ROOT_DIR, cls)
    files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    per_file = {}
    for f in files:
        n = 0
        for chunk in pd.read_csv(f, usecols=[0], chunksize=500_000,
                                 low_memory=False):
            n += len(chunk)
        per_file[f] = n
    inventory[cls] = per_file
    print(f"{cls:<12} files={len(files):>3}  rows={sum(per_file.values()):,}")

true_counts = {c: sum(v.values()) for c, v in inventory.items()}
grand_total = sum(true_counts.values())

inv_rows = [{"class": c, "csv_files": len(v), "true_rows": sum(v.values()),
             "true_pct": 100 * sum(v.values()) / grand_total}
            for c, v in inventory.items()]
pd.DataFrame(inv_rows).to_csv(
    os.path.join(OUT_DIR, "true_class_inventory.csv"),
    index=False, encoding="utf-8-sig")

print("\n>>> TRUE class distribution (put THIS in Table 1 of the paper):")
for r in inv_rows:
    print(f"    {r['class']:<12} {r['true_rows']:>12,}  {r['true_pct']:6.2f}%")


# ---------- allocate quotas proportionally ----------
quota = {}
for c, n in true_counts.items():
    q = int(round(TOTAL_TARGET * n / grand_total))
    quota[c] = max(MIN_PER_CLASS, min(q, n))
print("\nSampling quotas:", quota)


# ---------- pass 2: draw the sample ----------
print("\n=== PASS 2: sampling ===")
frames = []
for cls, per_file in inventory.items():
    cls_total = true_counts[cls]
    want = quota[cls]
    for f, n in per_file.items():
        if n == 0:
            continue
        # each file contributes in proportion to its share OF ITS OWN CLASS
        take = int(round(want * n / cls_total))
        take = min(take, n)
        if take <= 0:
            continue
        df = pd.read_csv(f, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        if len(df) > take:
            df = df.sample(n=take, random_state=RANDOM_STATE)
        df["attack_family"] = cls
        df["attack_binary"] = 0 if cls.lower() == "benign" else 1
        df["source_file"] = os.path.basename(f)
        frames.append(df)
        print(f"  {cls:<12} {os.path.basename(f)[:45]:<45} {take:>6}/{n:>9}")

combined = pd.concat(frames, ignore_index=True, sort=False)
combined = combined.replace([np.inf, -np.inf], np.nan)
combined = combined.dropna(axis=1, how="all")

out = os.path.join(OUT_DIR, "iot_diad_sample_by_class.csv")
combined.to_csv(out, index=False, encoding="utf-8-sig")

print("\n=== DONE ===")
print("Shape:", combined.shape)
print(combined["attack_family"].value_counts())
print("\nSaved:", out)
print("Saved:", os.path.join(OUT_DIR, "true_class_inventory.csv"))
