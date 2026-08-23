import os
import glob
import pandas as pd
import numpy as np

# ==========================================================
# 1) إعدادات المسار
# ==========================================================
ROOT_DIR = os.environ.get("IOT_DIAD_ROOT", r"C:\path\to\CIC-IoT-DIAD-2024")
OUTPUT_DIR = os.path.join(ROOT_DIR, "_working")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# عدد الصفوف المأخوذة من كل ملف (للتجربة الأولية)
# ارفعها لاحقًا (مثلاً 5000 أو 10000) إذا جهازك يتحمل
SAMPLE_PER_FILE = 1000

# إذا True سيقرأ فقط أول ملف من كل فئة (اختبار سريع جدًا)
# إذا False سيقرأ جميع الملفات في كل فئة
READ_ONLY_FIRST_FILE_PER_CLASS = False

# الفئات (كما ظهرت عندك في المجلد)
CLASS_FOLDERS = [
    "Benign",
    "Brute Force",
    "DDOS",
    "DOS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web-Based"
]

# ==========================================================
# 2) دالة تنظيف بسيطة
# ==========================================================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # تنظيف أسماء الأعمدة من الفراغات
    df.columns = [str(c).strip() for c in df.columns]

    # حذف الأعمدة الفارغة بالكامل
    df = df.dropna(axis=1, how="all")

    # استبدال inf بـ NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    return df

# ==========================================================
# 3) قراءة الملفات وتجميع عينة متعددة الفئات
# ==========================================================
all_frames = []
summary_rows = []

print("=== Start building sample dataset ===")
print("ROOT_DIR:", ROOT_DIR)
print("ROOT exists?", os.path.exists(ROOT_DIR))
print("OUTPUT_DIR:", OUTPUT_DIR)
print("SAMPLE_PER_FILE:", SAMPLE_PER_FILE)
print("READ_ONLY_FIRST_FILE_PER_CLASS:", READ_ONLY_FIRST_FILE_PER_CLASS)

for class_name in CLASS_FOLDERS:
    class_path = os.path.join(ROOT_DIR, class_name)

    if not os.path.exists(class_path):
        print(f"\n[WARNING] Folder not found: {class_path}")
        summary_rows.append({
            "class_name": class_name,
            "folder_exists": False,
            "csv_count": 0,
            "files_read": 0,
            "rows_before_sampling": 0,
            "rows_after_sampling": 0,
            "status": "folder_not_found"
        })
        continue

    # recursive=True حتى لو في مجلدات فرعية
    csv_files = glob.glob(os.path.join(class_path, "**", "*.csv"), recursive=True)

    print(f"\n[{class_name}] CSV files found: {len(csv_files)}")
    if len(csv_files) == 0:
        summary_rows.append({
            "class_name": class_name,
            "folder_exists": True,
            "csv_count": 0,
            "files_read": 0,
            "rows_before_sampling": 0,
            "rows_after_sampling": 0,
            "status": "no_csv_found"
        })
        continue

    # ترتيب الملفات (اختياري لكن جيد للتكرار)
    csv_files = sorted(csv_files)

    if READ_ONLY_FIRST_FILE_PER_CLASS:
        csv_files = csv_files[:1]

    class_rows_before = 0
    class_rows_after = 0
    files_read_count = 0

    for file_path in csv_files:
        try:
            print(f"  Reading: {file_path}")
            df = pd.read_csv(file_path, low_memory=False)

            # تنظيف
            df = clean_columns(df)

            # عدد الصفوف الأصلي
            rows_before = len(df)
            class_rows_before += rows_before

            # أخذ عينة بسيطة من الملف (أو كامل إذا أصغر)
            if rows_before > SAMPLE_PER_FILE:
                df = df.sample(n=SAMPLE_PER_FILE, random_state=42)
            else:
                df = df.copy()

            # إضافة الليبلات
            df["attack_family"] = class_name
            df["attack_binary"] = 0 if class_name.lower() == "benign" else 1

            rows_after = len(df)
            class_rows_after += rows_after

            all_frames.append(df)
            files_read_count += 1

        except Exception as e:
            print(f"  [ERROR] Failed to read {file_path}")
            print(f"         Reason: {e}")

    summary_rows.append({
        "class_name": class_name,
        "folder_exists": True,
        "csv_count": len(csv_files),
        "files_read": files_read_count,
        "rows_before_sampling": class_rows_before,
        "rows_after_sampling": class_rows_after,
        "status": "ok" if files_read_count > 0 else "failed_all_files"
    })

# ==========================================================
# 4) التحقق من وجود بيانات
# ==========================================================
if not all_frames:
    raise ValueError("No data was loaded. Please check ROOT_DIR and folder/file structure.")

# دمج آمن (حتى لو اختلفت الأعمدة بين الملفات)
combined = pd.concat(all_frames, ignore_index=True, sort=False)

# تنظيف نهائي سريع
combined = clean_columns(combined)

# ==========================================================
# 5) حفظ الملفات الناتجة
# ==========================================================
# ملف الملخص
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUTPUT_DIR, "build_summary.csv")
summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

# ملف متعدد الفئات
multiclass_csv = os.path.join(OUTPUT_DIR, "iot_diad_multiclass_sample.csv")
combined.to_csv(multiclass_csv, index=False, encoding="utf-8-sig")

# ملف ثنائي (نفس البيانات لكن يمكن الاحتفاظ فقط بالهدف الثنائي إن رغبت)
binary_df = combined.copy()
binary_csv = os.path.join(OUTPUT_DIR, "iot_diad_binary_sample.csv")
binary_df.to_csv(binary_csv, index=False, encoding="utf-8-sig")

# ==========================================================
# 6) طباعة نتائج مهمة
# ==========================================================
print("\n=== DONE ===")
print("Combined shape:", combined.shape)

print("\nColumns count:", len(combined.columns))
print("First 20 columns:")
print(combined.columns.tolist()[:20])

print("\nMulti-class distribution (attack_family):")
print(combined["attack_family"].value_counts(dropna=False))

print("\nBinary distribution (attack_binary):")
print(combined["attack_binary"].value_counts(dropna=False))

print("\nSaved files:")
print("Summary      :", summary_csv)
print("Multi-class  :", multiclass_csv)
print("Binary       :", binary_csv)