# Analisis Sentimen

> Tugas NLP kanonik. Sebagian besar hal yang perlu kamu ketahui tentang klasifikasi teks klasik muncul di sini.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 5 · 02 (BoW + TF-IDF), Fase 2 · 14 (Naive Bayes)
**Waktu:** ~75 menit

## Masalah

“Makanannya tidak enak.” Positif atau negatif?

Sentimen terdengar sederhana. Seorang pengulas mengatakan mereka menyukai atau tidak menyukai sesuatu. Beri label pada kalimat tersebut. Alasan mengapa ini menjadi tugas NLP kanonik adalah karena setiap kasus yang tampak mudah menyembunyikan kasus yang sulit. Negasi membalikkan makna. Sarkasme membalikkannya. "Tidak buruk sama sekali" adalah positif meskipun ada dua kata berkode negatif. Emoji membawa lebih banyak sinyal dibandingkan teks di sekitarnya. Kosakata domain penting (`tight` dalam ulasan musik versus `tight` dalam ulasan mode).

Sentimen adalah laboratorium kerja untuk NLP klasik. Jika kamu memahami mengapa setiap garis dasar yang naif memiliki mode kegagalan tertentu, kamu memahami mengapa setiap model yang lebih kaya diciptakan. Lesson ini membangun dasar Naive Bayes dari awal, menambahkan regresi logistik, dan menyebutkan jebakan yang menjadikan sentimen produksi sebagai masalah tingkat kepatuhan.

## Konsep

Sentimen klasik adalah resep dua langkah.

1. **Mewakili.** Ubah teks menjadi vector feature. BoW, TF-IDF, atau n-gram.
2. **Klasifikasi.** Sesuaikan model linier (Naive Bayes, regresi logistik, SVM) pada contoh berlabel.

Naive Bayes adalah model paling bodoh yang berhasil. Asumsikan setiap feature bersifat independen jika diberi label. Perkirakan `P(word | positive)` dan `P(word | negative)` dari hitungan. Pada inference, gandakan probabilitasnya. Asumsi kemerdekaan yang “naif” adalah sebuah kesalahan yang menggelikan, namun hasilnya sangat mengejutkan. Alasannya: dengan feature teks jarang dan data moderat, pengklasifikasi lebih memperhatikan sisi mana yang lebih condong ke setiap kata daripada seberapa banyak.

Regresi logistik memperbaiki asumsi independensi. Ia mempelajari weight per feature, termasuk weight negatif. `not good` sebagai feature bigram mendapat weight negatif. Naive Bayes tidak dapat melakukan hal tersebut untuk bigram yang belum pernah diberi label.

## Build

### Langkah 1: dataset mini yang sebenarnya

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

Sengaja kecil. Pekerjaan nyata menggunakan puluhan ribu contoh (IMDb, SST-2, polaritas Yelp). Matematikanya identik.

### Langkah 2: Naive Bayes multinomial dari awal

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Pemulusan aditif (alpha=1.0) adalah pemulusan Laplace. Tanpanya, sebuah kata yang tidak terlihat di kelas memiliki probabilitas nol dan lognya meledak. `alpha=0.01` adalah hal yang umum dalam praktik. `alpha=1.0` adalah default pengajaran.

### Langkah 3: regresi logistik dari awal

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

Regularisasi L2 penting di sini. Feature teks jarang; tanpa L2 model akan mengingat contoh training. Mulai dari `0.01` dan selaraskan.

### Langkah 4: menangani negasi (mode kegagalan)

Pertimbangkan "tidak baik" dan "tidak buruk". Pengklasifikasi BoW melihat `{not, good}` dan `{not, bad}` dan belajar dari mana pun yang muncul lebih banyak dalam training. Pengklasifikasi bigram melihat `not_good` dan `not_bad` dan mempelajarinya sebagai feature yang berbeda. Biasanya itu sudah cukup.

Perbaikan yang lebih sederhana yang berfungsi saat kamu tidak memiliki bigram: **pelingkupan negasi**. Token awalan setelah kata negasi dengan `NOT_` hingga tanda baca berikutnya.

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```Sekarang `good` dan `NOT_good` merupakan feature yang berbeda. Pengklasifikasi dapat memberikan weight yang berlawanan. Tiga baris pra-pemrosesan, akurasi terukur melonjak pada tolok ukur sentimen.

### Langkah 5: metrik evaluasi yang penting

Akurasi saja bisa menyesatkan jika kelas tidak seimbang. Korpora sentimen riil biasanya 70-80% positif atau 70-80% negatif; pengklasifikasi mayoritas konstan mendapatkan akurasi 80% dan tidak berguna. Laporkan setiap hal berikut:

- **Presisi dan perolehan per kelas.** Satu pasang per kelas. Rata-ratakan secara makro untuk mendapatkan satu angka yang menghormati keseimbangan kelas.
- **Macro-F1 (metrik utama untuk data tidak seimbang).** Rata-rata skor F1 per kelas, diberi weight yang sama. Gunakan ini sebagai pengganti akurasi ketika kelas tidak seimbang.
- **Berbobot-F1 (alternatif).** Sama seperti makro tetapi diberi weight berdasarkan frekuensi kelas. Laporkan bersama makro-F1 ketika ketidakseimbangan itu memiliki arti bisnis.
- **Matrix perplexity.** Jumlah mentah. Selalu periksa sebelum mempercayai metrik scalar apa pun; ini mengungkapkan pasangan kelas mana yang membingungkan model.
- **Contoh kesalahan per kelas.** Dapatkan 5 prediksi salah per kelas. Bacalah. Tidak ada yang menggantikan membaca kesalahan yang sebenarnya.

Untuk data yang sangat tidak seimbang (rasio > 95-5), laporkan **AUROC** dan **AUPRC**, bukan akurasi. AUPRC lebih sensitif terhadap kelas minoritas, yang biasanya kamu pedulikan (spam, penipuan, sentimen langka).

**Bug umum yang harus dihindari.** Melaporkan mikro-F1 dan bukan makro-F1 pada data yang tidak seimbang memberikan angka yang terlihat tinggi karena didominasi oleh kelas mayoritas. Macro-F1 memaksa kamu melihat performa kelas minoritas.

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## Pakai

scikit-learn melakukannya dalam enam baris, dengan benar.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

Tiga hal yang perlu diperhatikan. `stop_words=None` terus melakukan negasi. `ngram_range=(1, 2)` menambahkan bigram sehingga `not_good` menjadi feature. `sublinear_tf=True` meredam kata-kata yang diulang-ulang. Ketiga tanda ini merupakan perbedaan antara garis dasar dengan akurasi 75% dan garis dasar dengan akurasi 85% pada SST-2.

### Kapan harus meraih trafo

- Deteksi sarkasme. Model klasik gagal di sini. Periode.
- Ulasan panjang dimana sentimen bergeser di tengah-tengah dokumen.
- Sentimen berbasis aspek. "Kameranya bagus, tapi baterainya jelek." kamu perlu mengaitkan sentimen dengan aspek. Transformer atau model output terstruktur saja.
- Bahasa non-Inggris dengan sumber daya rendah. BERT multibahasa memberi kamu garis dasar zero-shot secara gratis.

Jika kamu memerlukan salah satu hal di atas, lanjutkan ke fase 7 (penyelaman mendalam pada Transformer). Jika tidak, Naive Bayes atau regresi logistik pada TF-IDF plus bigram plus penanganan negasi adalah dasar produksi tahun 2026 kamu.

### Perangkap reproduktifitas (lagi)

Melatih ulang model sentimen adalah hal yang rutin. Mengevaluasi kembali mereka tidaklah benar. Angka akurasi yang dilaporkan di makalah menggunakan pemisahan spesifik, preprocessing spesifik, tokenizer spesifik. Jika kamu membandingkan model baru dengan garis dasar tanpa menggunakan pipeline yang sama, kamu akan mendapatkan delta yang menyesatkan. Selalu buat ulang garis dasar pada pipeline pipa kamu, bukan nomor kertasnya.

## Kirim

Simpan sebagai `outputs/prompt-sentiment-baseline.md`:

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## Latihan1. **Mudah.** Tambahkan `apply_negation` sebagai langkah preprocessing dalam alur scikit-learn dan ukur delta F1 pada dataset sentimen kecil.
2. **Sedang.** Terapkan regresi logistik berbobot kelas (teruskan `class_weight="balanced"` untuk mempelajari scikit, atau dapatkan sendiri gradiennya). Ukur dampaknya terhadap ketidakseimbangan kelas sintetis 90-10.
3. **Sulit.** Buat pendeteksi sarkasme dengan melatih pengklasifikasi kedua pada sisa model sentimen. Dokumentasikan pengaturan eksperimental kamu. Peringatkan pembaca ketika akurasi kamu di bawah kemungkinan (tingkat peluang pada sarkasme kelas 2 adalah ~50%, dan sebagian besar percobaan pertama berakhir di sana).

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Polaritas | Positif atau negatif | Label biner; terkadang diperluas hingga netral atau berbutir halus (bintang 5). |
| Sentimen berbasis aspek | Polaritas per aspek | Atribusikan sentimen ke entitas atau atribut tertentu yang disebutkan dalam teks. |
| Pelingkupan negasi | Membalikkan token terdekat | Awali token setelah "tidak" dengan `NOT_` hingga tanda baca. |
| Pemulusan Laplace | Menambahkan 1 ke hitungan | Mencegah feature probabilitas nol di Naive Bayes. |
| Regularisasi L2 | Menyusut weight | Menambahkan `lambda * sum(w^2)` ke loss. Penting untuk feature teks jarang. |

## Bacaan Lanjutan

- [Pang dan Lee (2008). Penambangan Opini dan Analisis Sentimen](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-lysis-survey.html) — survei dasar. Panjang, tetapi empat bagian pertama mencakup segala sesuatu yang klasik.
- [Wang dan Manning (2012). Baseline dan Bigram: Sederhana, Sentimen Baik dan Klasifikasi Topik](https://aclanthology.org/P12-2018/) — makalah yang menunjukkan bigram + Naive Bayes sulit dikalahkan dalam teks pendek.
- [dokumen ekstraksi feature teks scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — referensi untuk `CountVectorizer`, `TfidfVectorizer`, dan setiap kenop yang akan kamu sesuaikan.
