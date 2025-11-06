# ============================================================
# 🧩 1️⃣ IMPORT LIBRARY
# ============================================================
import os
import re
import emoji
import pandas as pd
from tqdm import tqdm
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
# 🧱 2️⃣ LOAD DATASET (MENTAH ATAU HASIL CLEANING)
# ============================================================
DATA_PATH = "roblox_raw.csv"  # atau ganti sesuai file kamu
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"⚠️ File '{DATA_PATH}' tidak ditemukan!")

df = pd.read_csv(DATA_PATH, encoding='utf-8')
print(f"✅ Dataset dimuat: {len(df)} data")

# Hapus duplikat dan baris kosong
df.drop_duplicates(subset='content', inplace=True)
df.dropna(subset=['content'], inplace=True)
df.reset_index(drop=True, inplace=True)

# ============================================================
# 🧹 3️⃣ KONFIGURASI STEMMER, STOPWORDS, DAN NORMALISASI
# ============================================================
stemmer = StemmerFactory().create_stemmer()

# Stopwords dasar Bahasa Indonesia
stop_words = set(stopwords.words('indonesian'))

# Tambahan stopword (kata gaul, kata umum, filler, domain)
custom_stopwords = {
    'nya','sip','oke','ok','yah','ya','lah','deh','dong','sih','nih','loh','lho','kan','mah','nah',
    'pls','please','ges','bro','guys','bang','sis','wkwk','wk','haha','hehe','hihi','lmao','btw',
    'gua','gw','gue','loe','lu','aku','kamu','km','anda','saya','kita','kami','mereka','dia',
    'iya','yaudah','udah','dah','aja','deh','dong','gpp','bgt','banget','doang','tuh','nih',
    'yang','dan','atau','untuk','dari','pada','ke','di','ini','itu','sebagai','jadi','agar',
    'dengan','tanpa','sama','juga','lagi','udah','baru','udahh',
    # kata domain atau objek (hapus)
    'roblox','rblx','rbx','roblx','robloxnya','gamenya','game','aplikasi','apk','app','robux','devex'
}
stop_words = stop_words.union(custom_stopwords)

# Kamus normalisasi slang dan typo
slang_dict = {
    'aja':'saja','aku':'saya','apknya':'aplikasi','bagu':'bagus','bener':'benar','bgs':'bagus','bgt':'banget',
    'bikin':'buat','blm':'belum','bngt':'banget','dgn':'dengan','dpt':'dapat','ga':'tidak','gaje':'tidak jelas',
    'gak':'tidak','gamenya':'game','gk':'tidak','jele':'jelek','km':'kamu','krn':'karena',
    'laggy':'lambat','mainn':'main','makasih':'terima kasih','mksih':'terima kasih','ngelag':'lambat',
    'ngecrash':'crash','ngehang':'macet','nggak':'tidak','nih':'ini','parah':'buruk sekali',
    'ser':'seru','suk':'suka','tdk':'tidak','tp':'tapi','trs':'terus','udh':'sudah','yg':'yang',
    'errornya':'error','bugnya':'bug','servernya':'server','loginnya':'login'
}

# ============================================================
# ⚙️ 4️⃣ FUNGSI PEMBERSIHAN + BIGRAM + TRIGRAM
# ============================================================
RE_URL = re.compile(r"http\S+|www\S+")
RE_MENTION_HASHTAG = re.compile(r"@\w+|#\w+")
RE_NUM = re.compile(r"\d+")
RE_NONALPHA = re.compile(r"[^a-zA-Z_\s]")
RE_REPEAT = re.compile(r"(.)\1{2,}")

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()

    # 1️⃣ Hapus URL, mention, hashtag, angka, emoji
    text = RE_URL.sub("", text)
    text = RE_MENTION_HASHTAG.sub("", text)
    text = RE_NUM.sub("", text)
    text = emoji.replace_emoji(text, replace='')

    # 2️⃣ Gabungkan negasi (tidak bagus → tidak_bagus)
    text = re.sub(r"\btidak\s+(\w+)", r"tidak_\1", text)

    # 3️⃣ Hapus simbol non huruf
    text = RE_NONALPHA.sub(" ", text)

    # 4️⃣ Tokenisasi
    tokens = word_tokenize(text)

    # 5️⃣ Koreksi huruf berulang + normalisasi slang
    tokens = [RE_REPEAT.sub(r"\1\1", t) for t in tokens]
    tokens = [slang_dict.get(t, t) for t in tokens]

    # 6️⃣ Stopword removal dan token pendek
    filtered = [t for t in tokens if t not in stop_words and len(t) > 2]

    # 7️⃣ Stemming
    stemmed = [stemmer.stem(t) for t in filtered]

    # 8️⃣ Bigram & Trigram
    bigrams = ['_'.join(bg) for bg in ngrams(stemmed, 2)] if len(stemmed) >= 2 else []
    trigrams = ['_'.join(tg) for tg in ngrams(stemmed, 3)] if len(stemmed) >= 3 else []

    # Gabungkan semua
    return " ".join(stemmed + bigrams + trigrams)

# ============================================================
# 🚀 5️⃣ PROSES CLEANING
# ============================================================
print("\n🧹 Tahap 1: Preprocessing dimulai...")
tqdm.pandas(desc="Cleaning Progress")
df["cleaned"] = df["content"].progress_apply(clean_text)

# Hapus hasil kosong
df = df[df["cleaned"].str.strip() != ""]
print("✅ Preprocessing selesai — Jumlah data akhir:", len(df))

# ============================================================
# 📊 6️⃣ LABELING OTOMATIS (BERDASARKAN RATING)
# ============================================================
def label_sentimen(score):
    if score >= 4: return "Positif"
    elif score == 3: return "Netral"
    else: return "Negatif"

df["sentimen"] = df["score"].apply(label_sentimen)

# ============================================================
# 🖥️ 7️⃣ TAMPILKAN CONTOH HASIL
# ============================================================
print("\n📄 Contoh hasil pembersihan:\n")
print(df[["userName", "content", "cleaned", "sentimen"]].sample(10, random_state=42))

# ============================================================
# 💾 8️⃣ (OPSIONAL) SIMPAN HASIL CLEANING
# ============================================================
# df.to_csv("roblox_cleaned_final.csv", index=False, encoding="utf-8")
# print("📁 Hasil disimpan ke roblox_cleaned_final.csv")

# ============================================================
# 🌥️ 9️⃣ WORDCLOUD AMAN (ANTI ERROR)
# ============================================================
print("\n🎨 Membuat WordCloud...")

def wc(text, title):
    if not isinstance(text, str) or not text.strip():
        print(f"⚠️  Lewati WordCloud: '{title}' kosong atau tidak ada teks.")
        return
    words = text.split()
    if len(words) < 3:
        print(f"⚠️  Data '{title}' terlalu sedikit ({len(words)} kata). Dilewati.")
        return
    plt.figure(figsize=(8,6))
    wc_img = WordCloud(width=800, height=600, background_color="white",
                       colormap="viridis", max_words=150).generate(text)
    plt.imshow(wc_img, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Gabungkan teks berdasarkan kategori sentimen
wc(" ".join(df["cleaned"]), "WordCloud Semua Ulasan")
wc(" ".join(df[df["sentimen"]=="Positif"]["cleaned"]), "WordCloud Ulasan Positif")
wc(" ".join(df[df["sentimen"]=="Netral"]["cleaned"]), "WordCloud Ulasan Netral")
wc(" ".join(df[df["sentimen"]=="Negatif"]["cleaned"]), "WordCloud Ulasan Negatif")

print("\n✅ Semua tahap selesai tanpa error dan hasil siap untuk analisis lanjutan!")

