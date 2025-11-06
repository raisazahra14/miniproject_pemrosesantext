# ============================================================
# 1) IMPORTS
# ============================================================
# pip install pandas tqdm Sastrawi nltk wordcloud matplotlib emoji
import os, re, emoji
import pandas as pd
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
nltk.download('punkt'); nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ============================================================
# 2) LOAD DATASET
# ============================================================
DATA_PATH = "roblox_raw.csv"  # ganti sesuai file
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"⚠️ File '{DATA_PATH}' tidak ditemukan!")

df = pd.read_csv(DATA_PATH, encoding='utf-8')
print(f"✅ Dataset dimuat: {len(df)} data")

df.drop_duplicates(subset='content', inplace=True)
df.dropna(subset=['content'], inplace=True)
df.reset_index(drop=True, inplace=True)

# ============================================================
# 3) KONFIGURASI
# ============================================================
stemmer = StemmerFactory().create_stemmer()

# Stopwords dasar
stop_words = set(stopwords.words('indonesian'))

# Stopwords tambahan (gaul/filler/fungsi + domain)
custom_stopwords = {
    # filler/gaul/sapaan
    'nya','nyaa','sip','oke','ok','yah','ya','lah','deh','dong','sih','sihh','nih','loh','lho','kan','mah','nah',
    'pls','please','ges','bro','guys','bang','sis','wkwk','wk','haha','hehe','hihi','lmao','btw','yaa',
    'gua','gw','gue','loe','lu','aku','kamu','km','anda','saya','kita','kami','mereka','dia',
    'iya','yaudah','udah','dah','aja','gpp','bgt','banget','doang','tuh','biar','anj','woi','kek','gin',
    # fungsi/konjungsi/preposisi
    'yang','dan','atau','untuk','dari','pada','ke','di','ini','itu','sebagai','jadi','gitu','agar','dengan','tanpa','sama','juga','lagi','baru',
    # kata yang kamu sebut & variannya
    'bgtt','ken','bsa','kalo','klo','kaya','kayak','kali','tpi','tp','apa','moga','semoga','pas','tau',
    'nge', 'plis','pliss','knp','kenapa','mengapa',
    # domain/objek
    'roblox','rblx','rbx','roblx','robloxnya','aplikasi','apk','app','robux','devex','apknya','game','gamenya','geme','gem'
}
stop_words = stop_words.union(custom_stopwords)

# Normalisasi slang & typo → baku
slang_dict = {
    'bgus':'bagus','habi':'habis','slalu':'selalu','aja':'saja','aku':'saya','bagu':'bagus','bener':'benar','bgs':'bagus','bgt':'banget',
    'bikin':'buat','blm':'belum','bngt':'banget','dgn':'dengan','dpt':'dapat','ga':'tidak','gaje':'tidak jelas',
    'gak':'tidak','gk':'tidak','jele':'jelek','km':'kamu','krn':'karena','moga':'semoga',
    'laggy':'lambat','mainn':'main','makasih':'terima kasih','mksih':'terima kasih','ngelag':'lambat',
    'ngecrash':'crash','ngehang':'macet','nggak':'tidak','nih':'ini','parah':'buruk sekali',
    'ser':'seru','suk':'suka','tdk':'tidak','tpi':'tapi','tp':'tapi','trs':'terus','trus':'terus','udh':'sudah','yg':'yang',
    'errornya':'error','bugnya':'bug','servernya':'server','loginnya':'login','ngelek':'lambat','ngeleg':'lambat',
}

# ============================================================
# 4) UTIL & CLEAN FUNCTION
# ============================================================
RE_URL = re.compile(r"http\S+|www\S+")
RE_MENTION_HASHTAG = re.compile(r"@\w+|#\w+")
RE_NUM = re.compile(r"\d+")
RE_NONALPHA = re.compile(r"[^a-zA-Z_\s]")
RE_REPEAT = re.compile(r"(.)\1{2,}")  # collapse huruf berulang

def clean_text(text: str) -> str:
    if pd.isna(text): 
        return ""
    text = str(text).lower()

    # (1) Buang noise
    text = RE_URL.sub("", text)
    text = RE_MENTION_HASHTAG.sub("", text)
    text = RE_NUM.sub("", text)
    text = emoji.replace_emoji(text, replace='')

    # (2) Negasi: "tidak bagus" → "tidak_bagus"
    text = re.sub(r"\btidak\s+(\w+)", r"tidak_\1", text)

    # (3) Hanya huruf/underscore/space
    text = RE_NONALPHA.sub(" ", text)

    # (4) Token
    tokens = word_tokenize(text)

    # (5) Collapse huruf berulang + normalisasi slang
    tokens = [RE_REPEAT.sub(r"\1\1", t) for t in tokens]
    tokens = [slang_dict.get(t, t) for t in tokens]

    # (6) FILTER I: stopwords & token pendek
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # (7) Stemming
    stemmed = [stemmer.stem(t) for t in tokens]

    # (8) FILTER II (post-stemming): ulangi agar kata yang berubah bentuk tetap terbuang
    stemmed = [w for w in stemmed if w not in stop_words and len(w) > 2]

    # (9) Bigram & Trigram
    bigrams = ['_'.join(bg) for bg in ngrams(stemmed, 2)] if len(stemmed) >= 2 else []
    trigrams = ['_'.join(tg) for tg in ngrams(stemmed, 3)] if len(stemmed) >= 3 else []

    return " ".join(stemmed + bigrams + trigrams)

# ============================================================
# 5) PREPROCESS
# ============================================================
print("\n🧹 Tahap 1: Preprocessing dimulai...")
tqdm.pandas(desc="Cleaning Progress")
df["cleaned"] = df["content"].progress_apply(clean_text)
df = df[df["cleaned"].str.strip() != ""]
print("✅ Preprocessing selesai — Jumlah data akhir:", len(df))

# ============================================================
# 6) LABEL SENTIMEN DARI RATING
# ============================================================
def label_sentimen(score):
    if score >= 4: return "Positif"
    if score == 3: return "Netral"
    return "Negatif"

df["sentimen"] = df["score"].apply(label_sentimen)

# ============================================================
# 7) PREVIEW HASIL
# ============================================================
print("\n📄 Contoh hasil pembersihan:\n")
print(df[["userName", "content", "cleaned", "sentimen"]].sample(10, random_state=42))

# ============================================================
# 8) (OPSIONAL) SIMPAN
# ============================================================
# df.to_csv("roblox_cleaned_final.csv", index=False, encoding="utf-8")
# print("📁 Hasil disimpan ke roblox_cleaned_final.csv")

# ============================================================
# 9) WORDCLOUD ANTI-ERROR
# ============================================================
print("\n🎨 Membuat WordCloud...")

def wc(text, title):
    if not isinstance(text, str) or not text.strip():
        print(f"⚠️  Lewati WordCloud: '{title}' kosong.")
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

wc(" ".join(df["cleaned"]), "WordCloud Semua Ulasan")
wc(" ".join(df[df["sentimen"]=="Positif"]["cleaned"]), "WordCloud Ulasan Positif")
wc(" ".join(df[df["sentimen"]=="Netral"]["cleaned"]),  "WordCloud Ulasan Netral")
wc(" ".join(df[df["sentimen"]=="Negatif"]["cleaned"]), "WordCloud Ulasan Negatif")

print(" hasil WordCloud bersih.")
