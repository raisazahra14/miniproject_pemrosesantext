# ============================================================
# 1️⃣ IMPORT LIBRARY YANG DIBUTUHKAN
# ============================================================
# Jalankan dulu ini kalau belum di-install:
# pip install google-play-scraper pandas tqdm Sastrawi nltk wordcloud matplotlib emoji
from google_play_scraper import reviews, Sort
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
# 2️⃣ SCRAPING DATA DARI GOOGLE PLAY STORE
# ============================================================
APP_ID = 'com.roblox.client'
TOTAL_REVIEWS = 5000
LANG = 'id'
COUNTRY = 'id'

print(f"Mengambil {TOTAL_REVIEWS} ulasan dari Google Play Store...")

all_reviews = []
continuation_token = None
BATCH_SIZE = 200

with tqdm(total=TOTAL_REVIEWS, desc="Scraping Progress") as pbar:
    while len(all_reviews) < TOTAL_REVIEWS:
        result, continuation_token = reviews(
            APP_ID,
            lang=LANG,
            country=COUNTRY,
            sort=Sort.NEWEST,
            count=BATCH_SIZE,
            continuation_token=continuation_token
        )
        all_reviews.extend(result)
        pbar.update(len(result))
        if continuation_token is None or len(result) == 0:
            break

df = pd.DataFrame(all_reviews[:TOTAL_REVIEWS])
df = df[['reviewId', 'userName', 'content', 'score', 'thumbsUpCount', 'reviewCreatedVersion', 'at']]

# 🧹 Hapus duplikat konten
df.drop_duplicates(subset='content', inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv('roblox_raw.csv', index=False, encoding='utf-8')
print(f"\n✅ Tahap 1: Scraping selesai — {len(df)} data disimpan ke 'roblox_raw.csv'")

# ============================================================
# 3️⃣ PREPROCESSING TEKS (CLEANING, NORMALISASI, BIGRAM, TRIGRAM)
# ============================================================
print("\n🧹 Tahap 2: Preprocessing data dimulai...")

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

# Kamus normalisasi kata tidak baku
slang_dict = {
    'gk': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'gak': 'tidak',
    'bgt': 'banget', 'bngt': 'banget', 'bener': 'benar', 'makasih': 'terima kasih',
    'mksih': 'terima kasih', 'kamu': 'anda', 'aku': 'saya', 'km': 'kamu',
    'bikin': 'buat', 'dpt': 'dapat', 'blm': 'belum', 'udh': 'sudah',
    'tdk': 'tidak', 'dgn': 'dengan', 'yg': 'yang', 'aja': 'saja',
    'trs': 'terus', 'tp': 'tapi', 'krn': 'karena', 'nih': 'ini',
    'dong': '', 'deh': '', 'lah': '', 'loh': '',
    'bagu': 'bagus', 'suk': 'suka', 'mainn': 'main', 'ser': 'seru', 'jele': 'jelek'
}

def clean_text(text):
    if pd.isna(text):
        return ""

    # 🔹 Hapus URL, mention, hashtag, angka, dan emoji
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text))  # hapus URL
    text = re.sub(r'@\w+|#\w+', '', text)                     # hapus mention dan hashtag
    text = re.sub(r'\d+', '', text)                           # hapus angka
    text = emoji.replace_emoji(text, replace='')              # hapus emoji

    # 🔹 Cleaning non-huruf dan case folding
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()

    # 🔹 Tokenizing
    tokens = word_tokenize(text)

    # 🔹 Normalisasi slang word
    tokens = [slang_dict.get(word, word) for word in tokens]

    # 🔹 Stopword removal
    filtered = [word for word in tokens if word not in stop_words and len(word) > 2]

    # 🔹 Stemming
    stemmed = [stemmer.stem(word) for word in filtered]

    # 🔹 Bigram & Trigram
    bigrams = ['_'.join(bg) for bg in ngrams(stemmed, 2)] if len(stemmed) >= 2 else []
    trigrams = ['_'.join(tg) for tg in ngrams(stemmed, 3)] if len(stemmed) >= 3 else []
    all_terms = stemmed + bigrams + trigrams

    return ' '.join(all_terms)

tqdm.pandas(desc="Preprocessing Lengkap")
df['cleaned'] = df['content'].progress_apply(clean_text)

df.to_csv('roblox_cleaned.csv', index=False, encoding='utf-8')
print("✅ Tahap 2: Preprocessing selesai — data disimpan ke 'roblox_cleaned.csv'")

# ============================================================
# 🧱 2.5️⃣ EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n📊 Tahap 2.5: Exploratory Data Analysis (EDA)...")

# Statistik umum rating
print("\nStatistik Umum Rating:")
print(df['score'].describe())

# Distribusi rating (bar chart)
plt.figure(figsize=(7,5))
df['score'].value_counts().sort_index().plot(kind='bar', color='skyblue', title='Distribusi Rating Roblox')
plt.xlabel('Rating (1-5)')
plt.ylabel('Jumlah Ulasan')
plt.show()

# Panjang rata-rata teks
df['length'] = df['content'].apply(lambda x: len(str(x).split()))
print("📏 Rata-rata panjang ulasan:", round(df['length'].mean(), 2))

# Korelasi sederhana: panjang teks vs rating
plt.figure(figsize=(6,4))
plt.scatter(df['score'], df['length'], alpha=0.3, color='orange')
plt.title('Hubungan Panjang Ulasan vs Rating')
plt.xlabel('Rating (1-5)')
plt.ylabel('Panjang Ulasan (jumlah kata)')
plt.show()

print("✅ Tahap 2.5: EDA selesai — distribusi rating dan pola teks ditampilkan.")

# ============================================================
# 4️⃣ LABELING SENTIMEN BERDASARKAN RATING
# ============================================================
print("\n🏷️ Tahap 3: Pelabelan sentimen otomatis...")

def label_sentimen(score):
    if score >= 4:
        return 'Positif'
    elif score == 3:
        return 'Netral'
    else:
        return 'Negatif'

df['sentimen'] = df['score'].apply(label_sentimen)
df.to_csv('roblox_labeled.csv', index=False, encoding='utf-8')
print("✅ Tahap 3: Labeling selesai — data disimpan ke 'roblox_labeled.csv'")

# ============================================================
# 5️⃣ VISUALISASI WORDCLOUD
# ============================================================

print("\n🎨 Tahap 4: Membuat WordCloud...")

text_all = ' '.join(df['cleaned'])
text_pos = ' '.join(df[df['sentimen'] == 'Positif']['cleaned'])
text_neg = ' '.join(df[df['sentimen'] == 'Negatif']['cleaned'])
text_net = ' '.join(df[df['sentimen'] == 'Netral']['cleaned'])

def generate_wordcloud(text, title):
    plt.figure(figsize=(8,6))
    wc = WordCloud(width=800, height=600, background_color='white', colormap='viridis',
                   max_words=150).generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.show()

generate_wordcloud(text_all, "WordCloud Semua Ulasan (Unigram + Bigram + Trigram)")
generate_wordcloud(text_pos, "WordCloud Ulasan Positif")
generate_wordcloud(text_neg, "WordCloud Ulasan Negatif")
generate_wordcloud(text_net, "WordCloud Ulasan Netral")

print("\n✅ Tahap 4: WordCloud berhasil ditampilkan!")
print("\n📊 Distribusi Sentimen:")
print(df['sentimen'].value_counts())
print("\nContoh data setelah semua tahap:")
print(df[['userName', 'content', 'cleaned', 'score', 'sentimen']].head(10))
