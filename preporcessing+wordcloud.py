# ============================================================
# 1️⃣ IMPORT LIBRARY
# ============================================================
# pip install pandas tqdm Sastrawi nltk wordcloud matplotlib emoji
import os
import pandas as pd
from tqdm import tqdm
import re
import emoji
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ============================================================
# 2️⃣ LOAD DATASET (TANPA SCRAPING)
# ============================================================
DATA_PATH = "roblox_raw.csv"  # ganti sesuai lokasi dataset kamu
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"⚠️ File '{DATA_PATH}' tidak ditemukan. Pastikan dataset sudah ada!")

df = pd.read_csv(DATA_PATH, encoding='utf-8')
print(f"✅ Dataset berhasil dimuat: {len(df)} data")

# Hapus duplikat & kosong
df.drop_duplicates(subset='content', inplace=True)
df.dropna(subset=['content'], inplace=True)
df.reset_index(drop=True, inplace=True)

# ============================================================
# 3️⃣ PREPROCESSING (CLEANING, NORMALISASI, STOPWORDS, BIGRAM/TRIGRAM)
# ============================================================
print("\n🧹 Tahap 1: Preprocessing dimulai...")

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Stopwords dasar dari NLTK
stop_words = set(stopwords.words('indonesian'))

# ============================================================
# Tambahan Stopwords Kustom untuk Bahasa Gaul / Umum Tak Bermakna
# ============================================================
custom_stopwords = {
    'nya','sip','oke','ok','yah','ya','lah','deh','dong','sih','nih','loh',
    'pls','please','ges','bro','guys','bang','anj','anjir','anjay','wkwk',
    'gua','gw','aku','saya','kamu','anda','loe','lu','tpi','tp','pas',
    'gitu','kalo','kalau','kayak','yg','yaudah','udah','dah','aja',
    'buka','nama','pakai','pake','lagi','doang','nih','tuh','bgt','banget',
    'gpp','mantap','mantul','bikin','kan','itu','ini','dong','lah'
}
stop_words = stop_words.union(custom_stopwords)

# Kamus normalisasi slang & koreksi typo
slang_dict = { 
    'aja':'saja','aku':'saya','apknya':'aplikasi','bagu':'bagus','bener':'benar','bgs':'bagus','bgt':'banget',
    'bikin':'buat','blm':'belum','bngt':'banget','dgn':'dengan','dpt':'dapat','ga':'tidak','gaje':'tidak jelas',
    'gak':'tidak','gamenya':'game','gk':'tidak','jele':'jelek','kamu':'anda','km':'kamu','krn':'karena',
    'laggy':'lambat','mainn':'main','makasih':'terima kasih','mksih':'terima kasih','ngelag':'lambat',
    'ngecrash':'crash','ngehang':'macet','nggak':'tidak','nih':'ini','parah':'buruk sekali','ser':'seru',
    'suk':'suka','tdk':'tidak','tp':'tapi','trs':'terus','udh':'sudah','yg':'yang','errornya':'error'
}

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text)

    # 1️⃣ Hilangkan URL, mention, angka, emoji
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = emoji.replace_emoji(text, replace='')

    # 2️⃣ Negasi handling: ubah "tidak bagus" jadi "tidak_bagus"
    text = re.sub(r"tidak (\w+)", r"tidak_\1", text)

    # 3️⃣ Bersihkan simbol dan ubah huruf kecil
    text = re.sub(r"[^a-zA-Z_\s]", " ", text).lower()

    # 4️⃣ Tokenizing
    tokens = word_tokenize(text)

    # 5️⃣ Normalisasi slang
    tokens = [slang_dict.get(w, w) for w in tokens]

    # 6️⃣ Stopword removal
    filtered = [w for w in tokens if w not in stop_words and len(w) > 3]

    # 7️⃣ Stemming
    stemmed = [stemmer.stem(w) for w in filtered]

    # 8️⃣ Bigram & Trigram
    bigrams = ['_'.join(bg) for bg in ngrams(stemmed, 2)] if len(stemmed) >= 2 else []
    trigrams = ['_'.join(tg) for tg in ngrams(stemmed, 3)] if len(stemmed) >= 3 else []
    all_terms = stemmed + bigrams + trigrams

    return " ".join(all_terms)

tqdm.pandas(desc="Cleaning Text")
df["cleaned"] = df["content"].progress_apply(clean_text)
df = df[df["cleaned"].str.strip() != ""]

df.to_csv("roblox_cleaned.csv", index=False, encoding="utf-8")
print("✅ Tahap 1 selesai — data bersih disimpan ke 'roblox_cleaned.csv'")

# ============================================================
# 4️⃣ LABELING SENTIMEN BERDASARKAN RATING
# ============================================================
print("\n🏷️ Tahap 2: Pelabelan Sentimen...")

def label_sentimen(score):
    if score >= 4:
        return "Positif"
    elif score == 3:
        return "Netral"
    else:
        return "Negatif"

df["sentimen"] = df["score"].apply(label_sentimen)
df.to_csv("roblox_labeled.csv", index=False, encoding="utf-8")
print("✅ Labeling selesai — file 'roblox_labeled.csv' tersimpan")

# ============================================================
# 5️⃣ EDA (Exploratory Data Analysis)
# ============================================================
print("\n📊 Tahap 3: EDA...")

# Statistik umum rating
print(df["score"].describe())

# Distribusi rating
plt.figure(figsize=(7,5))
df["score"].value_counts().sort_index().plot(kind="bar", color="skyblue", title="Distribusi Rating Roblox")
plt.xlabel("Rating (1–5)")
plt.ylabel("Jumlah Ulasan")
plt.show()

# Panjang ulasan
df["length"] = df["content"].apply(lambda x: len(str(x).split()))
print("Rata-rata panjang ulasan:", round(df["length"].mean(),2))

# Scatter: panjang ulasan vs rating
plt.figure(figsize=(6,4))
plt.scatter(df["score"], df["length"], alpha=0.3, color="orange")
plt.title("Hubungan Panjang Ulasan vs Rating")
plt.xlabel("Rating (1–5)")
plt.ylabel("Panjang (kata)")
plt.show()

# 🔍 Distribusi Sentimen per Versi Aplikasi
if 'reviewCreatedVersion' in df.columns:
    plt.figure(figsize=(10,6))
    df.groupby('reviewCreatedVersion')['sentimen'].value_counts(normalize=True).unstack().plot(
        kind='bar', stacked=True, color=['salmon','gold','lightgreen'])
    plt.title('Distribusi Sentimen per Versi Aplikasi Roblox')
    plt.xlabel('Versi Aplikasi')
    plt.ylabel('Proporsi Ulasan')
    plt.legend(title='Sentimen', loc='upper right')
    plt.tight_layout()
    plt.show()

# 👍 Korelasi antara Thumbs Up dan Sentimen
if 'thumbsUpCount' in df.columns:
    plt.figure(figsize=(6,4))
    df.groupby('sentimen')['thumbsUpCount'].mean().plot(
        kind='bar', color=['lightcoral','khaki','lightgreen'])
    plt.title('Rata-rata Thumbs Up per Sentimen')
    plt.xlabel('Kategori Sentimen')
    plt.ylabel('Rata-rata Jumlah Likes')
    plt.show()

print("✅ EDA selesai — distribusi & korelasi ditampilkan.")

# ============================================================
# 6️⃣ WORDCLOUD PER SENTIMEN
# ============================================================
print("\n🎨 Tahap 4: WordCloud...")

def generate_wordcloud(text, title):
    plt.figure(figsize=(8,6))
    wc = WordCloud(width=800, height=600, background_color="white", colormap="viridis", max_words=150).generate(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.show()

text_all = " ".join(df["cleaned"])
text_pos = " ".join(df[df["sentimen"]=="Positif"]["cleaned"])
text_net = " ".join(df[df["sentimen"]=="Netral"]["cleaned"])
text_neg = " ".join(df[df["sentimen"]=="Negatif"]["cleaned"])

generate_wordcloud(text_all, "WordCloud Semua Ulasan")
generate_wordcloud(text_pos, "WordCloud Ulasan Positif")
generate_wordcloud(text_net, "WordCloud Ulasan Netral")
generate_wordcloud(text_neg, "WordCloud Ulasan Negatif")

print("\n✅ Semua tahap analisis deskriptif selesai!")
