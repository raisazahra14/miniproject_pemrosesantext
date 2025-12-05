# ============================================================
# TRAINING MODEL (Logistic Regression + LLM) UNTUK ULASAN ROBLOX
# ------------------------------------------------------------

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import joblib

# ============================================================
# 0. KONFIGURASI
# ============================================================

# Nama file input
MANUAL_FILE = "data_labeled_manual.csv"
UNLABELED_FILE = "data_unlabeled_cleaned.csv"

# Nama file output
DATA_MANUAL_250 = "data_manual_250.csv"
SELF_TRAIN_FILE = "data_self_training.csv"

VECTORIZER_PATH = "tfidf_roblox.pkl"
MODEL_PATH = "logreg_roblox.pkl"

# ============================================================
# (LLM SECTION DI-NONAKTIFKAN SEMENTARA)
# ============================================================
# from openai import OpenAI
# LLM_MODEL_NAME = "gpt-5.1-mini"
#
# if "OPENAI_API_KEY" not in os.environ:
#     raise EnvironmentError(
#         "OPENAI_API_KEY belum diset di environment. "
#         "Set dulu, misalnya di Windows: setx OPENAI_API_KEY \"sk-...\""
#     )
#
# client = OpenAI()
#
# def classify_with_llm_batch(texts, model_name=LLM_MODEL_NAME):
#     """
#     texts: list of strings (ulasan)
#     return: list of label_llm ("positive"/"negative"/"neutral")
#     """
#     results = []
#     for t in texts:
#         prompt = f\"\"\"You are a sentiment classifier...\"\"\"  # dipotong
#         ...
#     return results

# ============================================================
# 3. LOAD DATA MANUAL & STANDARDISASI LABEL
# ============================================================

print(f"📥 Load data manual dari: {MANUAL_FILE}")
df_manual = pd.read_csv(MANUAL_FILE)

print("Kolom di data_manual:", df_manual.columns.tolist())

# Pastikan kolom penting ada
required_cols = ["cleaned", "label_manual"]
for c in required_cols:
    if c not in df_manual.columns:
        raise KeyError(f"Kolom '{c}' tidak ditemukan di {MANUAL_FILE}")

# Rapikan teks & label
df_manual["cleaned"] = df_manual["cleaned"].astype(str).str.strip()
df_manual["label_manual"] = df_manual["label_manual"].fillna("").astype(str).str.strip()

# Mapping label Indonesia/Inggris -> 3 kelas utama
label_map = {
    "positif": "positive",
    "positive": "positive",
    "negatif": "negative",
    "negative": "negative",
    "netral": "neutral",
    "neutral": "neutral",
}
df_manual["label"] = df_manual["label_manual"].str.lower().map(label_map)

# Buang baris yang label/text kosong / tidak termapping
df_manual = df_manual[
    (df_manual["label"].notna()) &
    (df_manual["label"] != "") &
    (df_manual["cleaned"] != "")
].copy()

print("\nDistribusi label setelah dibersihkan (semua data manual):")
print(df_manual["label"].value_counts())
print("Total data manual (usable):", len(df_manual))

# ============================================================
# 4. SAMPLING 250 DATA MANUAL (100 POS, 100 NEG, 50 NETRAL)
# ============================================================

target_counts = {"positive": 100, "negative": 100, "neutral": 50}
dfs_250 = []

print("\n📌 Sampling 250 data manual (target: 100 pos, 100 neg, 50 netral)...")
for lab, n in target_counts.items():
    subset = df_manual[df_manual["label"] == lab]
    count_available = len(subset)
    print(f" - {lab}: tersedia {count_available}, target {n}")
    if count_available < n:
        print(f"   ⚠️ WARNING: data '{lab}' kurang dari target, ambil semua yang ada.")
        dfs_250.append(subset)
    else:
        dfs_250.append(subset.sample(n=n, random_state=42))

df_250 = pd.concat(dfs_250).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDistribusi label di data_manual_250:")
print(df_250["label"].value_counts())
print("Total baris data_manual_250:", len(df_250))

# Simpan file manual 250 (sesuai instruksi tugas)
df_250[["cleaned", "label"]].to_csv(DATA_MANUAL_250, index=False, encoding="utf-8")
print(f"\n✅ File '{DATA_MANUAL_250}' berhasil dibuat.")

# ============================================================
# 5. SPLIT TRAIN/TEST & TRAIN MODEL (TF-IDF + LOGREG)
# ============================================================

print("\n📚 Split train/test 80/20 (stratified)...")

X = df_250["cleaned"]
y = df_250["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Ukuran train:", len(X_train))
print("Ukuran test :", len(X_test))

print("\n🔧 Training TF-IDF + Logistic Regression...")

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)
clf.fit(X_train_tfidf, y_train)

# Evaluasi
y_pred = clf.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy di data test: {acc:.4f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Simpan model & vectorizer
joblib.dump(tfidf, VECTORIZER_PATH)
joblib.dump(clf, MODEL_PATH)
print(f"\n📁 Vectorizer TF-IDF disimpan ke: {VECTORIZER_PATH}")
print(f"📁 Model Logistic Regression disimpan ke: {MODEL_PATH}")

# ============================================================
# 6. LOAD DATA UNLABELED & AMBIL 750 DATA
# ============================================================

print(f"\n Load data unlabeled dari: {UNLABELED_FILE}")
df_unlab = pd.read_csv(UNLABELED_FILE)

if "cleaned" not in df_unlab.columns:
    raise KeyError("Kolom 'cleaned' tidak ditemukan di data_unlabeled_cleaned.csv")

# Ambil 750 data (sample random, reproducible)
n_self = 750
if len(df_unlab) < n_self:
    print(f"⚠️ Data unlabeled hanya {len(df_unlab)}, kurang dari 750. Ambil semuanya.")
    df_unlab_sample = df_unlab.copy()
else:
    df_unlab_sample = df_unlab.sample(n=n_self, random_state=42).reset_index(drop=True)

print("Jumlah data untuk self-training (model + LLM):", len(df_unlab_sample))

# ============================================================
# 7. PREDIKSI label_model UNTUK 750 DATA
# ============================================================

print("\n Prediksi label_model untuk data self-training...")

X_unlab = df_unlab_sample["cleaned"].astype(str)
X_unlab_tfidf = tfidf.transform(X_unlab)

pred_labels = clf.predict(X_unlab_tfidf)

# ============================================================
# 8. (LLM DIMATIKAN) SIAPKAN KOLom label_llm KOSONG
# ============================================================

print("\n Bagian LLM via API dinonaktifkan sementara.")
print("   Kolom 'label_llm' akan dikosongkan (string kosong).")

ulasan_list = X_unlab.tolist()
label_llm_list = [""] * len(ulasan_list)  # nanti bisa kamu isi manual / via ChatGPT UI

# ============================================================
# 9. BENTUK & SIMPAN FILE data_self_training.csv
# ============================================================

df_self = pd.DataFrame({
    "ulasan": ulasan_list,
    "label_model": pred_labels,
    "label_llm": label_llm_list
})

df_self.to_csv(SELF_TRAIN_FILE, index=False, encoding="utf-8")

print(f"\n✅ File '{SELF_TRAIN_FILE}' berhasil dibuat.")
print("   Kolom: ulasan, label_model, label_llm (sementara masih kosong).")

print("\nS E L E S A I  ✅")
print("Sekarang di folder kamu ada:")
print(f" - {DATA_MANUAL_250}   (250 data manual untuk training)")
print(f" - {SELF_TRAIN_FILE}   (750 data: ulasan + label_model + label_llm)")
print(f" - {VECTORIZER_PATH}, {MODEL_PATH} (model & vectorizer)")
