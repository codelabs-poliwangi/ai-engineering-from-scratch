# Kantong Kata, TF-IDF, dan Representasi Teks

> Hitung dulu, pikirkan nanti. TF-IDF masih mengungguli embedding pada tugas-tugas yang terdefinisi dengan baik pada tahun 2026.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 5 · 01 (Pemrosesan Teks), Phase 2 · 02 (Regresi Linier dari Awal)
**Waktu:** ~75 menit

## Masalah

Modelnya membutuhkan angka. kamu memiliki string.

Setiap pipeline NLP harus menjawab pertanyaan yang sama. Bagaimana kita mengubah aliran token dengan panjang variabel menjadi vector berukuran tetap yang dapat digunakan oleh pengklasifikasi. Jawaban pertama yang diberikan adalah jawaban paling bodoh yang berhasil. Hitung kata-katanya. Buatlah vector.

Vector tersebut telah membawa lebih banyak NLP produksi dibandingkan model embedding mana pun. Filter spam, pengklasifikasi topik, deteksi anomali log, peringkat pencarian (sebelum BM25), analisis sentimen gelombang pertama, dekade pertama tolok ukur NLP akademik. Praktisi pada tahun 2026 masih meraihnya terlebih dahulu pada tugas klasifikasi sempit. Ini cepat, dapat ditafsirkan, dan sering kali tidak dapat dibedakan dari model embedding parameter 400M pada tugas-tugas yang mengutamakan keberadaan kata.

Lesson ini membangun sekumpulan kata, lalu TF-IDF, dari awal. Kemudian tunjukkan scikit-learn melakukan hal yang sama dalam tiga baris. Lalu beri nama mode kegagalan yang membuat kamu ingin melakukan embedding.

## Konsep

**Bag of Words (BoW)** membuang pesanan. Untuk setiap dokumen, hitung berapa kali setiap kata kosakata muncul. Panjang vector adalah ukuran kosakata. Posisi `i` adalah jumlah kata `i`.

**TF-IDF** mengubah weight BoW. Sebuah kata yang muncul di setiap dokumen tidak informatif, jadi perkecil skalanya. Sebuah kata yang jarang ditemukan di seluruh korpus tetapi sering ditemukan dalam satu dokumen adalah sebuah sinyal, jadi tingkatkanlah kata tersebut.

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

Dimana `TF` adalah frekuensi istilah dalam dokumen, `df` adalah frekuensi dokumen (berapa banyak dokumen yang memuat kata tersebut), `N` adalah total dokumen. `log` menjaga weight tetap terbatas pada kata-kata yang ada di mana-mana.

Properti utama: keduanya menghasilkan vector renggang dengan sumbu yang dapat diinterpretasikan. kamu dapat melihat weight pengklasifikasi terlatih dan membaca kata mana yang mendorong dokumen ke setiap kelas. kamu tidak dapat melakukan ini dengan embedding BERT 768 dimension.

## Build

### Langkah 1: membangun kosakata

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

Input: daftar dokumen yang diberi token (tokenizer tingkat kata apa pun dapat digunakan; `code/main.py` dalam lesson ini menggunakan varian huruf kecil yang disederhanakan). Output: `{word: index}` dikt. Urutan penyisipan stabil berarti indeks kata 0 adalah kata pertama yang terlihat di dokumen pertama. Konvensi bervariasi; scikit-learn mengurutkan berdasarkan abjad.

### Langkah 2: sekantong kata-kata

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

Baris adalah dokumen. Kolom adalah indeks kosakata. Entri `[i][j]` adalah "berapa kali kata `j` muncul di dokumen `i`." Dokumen 1 memiliki `cat` dua kali karena memang demikian. Dokumen 0 memiliki `ran` kali nol karena tidak.

### Langkah 3: frekuensi istilah dan frekuensi dokumen

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

Dua trik penghalusan yang patut disebutkan. `(n+1)/(d+1)` menghindari `log(x/0)`. `+1` di akhir memastikan sebuah kata di setiap dokumen masih memiliki IDF 1 (bukan 0), cocok dengan default scikit-learn. Implementasi lainnya menggunakan `log(N/df)` mentah. Keduanya bekerja; versi yang dihaluskan lebih ramah.

### Langkah 4: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```Tiga dokumen, lima kosakata (`the`, `cat`, `sat`, `dog`, `ran`). `the` muncul di ketiganya, jadi IDF-nya rendah. `dog` muncul di satu, jadi IDF-nya tinggi. Vektornya jarang (sebagian besar entri berukuran kecil) dan kata-kata diskriminatif muncul.

### Langkah 5: L2-normalkan baris

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

Tanpa normalisasi, dokumen yang lebih panjang akan mendapatkan vector yang lebih besar dan mendominasi skor kesamaan. Normalisasi L2 menempatkan setiap dokumen pada unit hipersfer. Kemiripan cosinus antar baris kini hanya berupa perkalian titik.

## Pakai

scikit-learn mengirimkan versi produksi.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` melakukan tokenization, kosakata, dan BoW dalam satu panggilan. `TfidfVectorizer` menambahkan weight IDF dan normalisasi L2. Keduanya mengembalikan matrix renggang. Untuk 100 ribu dokumen, versi padat tidak muat di memori; tetap jarang sampai pengklasifikasi menuntut padat.

Kenop yang mengubah segalanya:

| Arg | Efek |
|-----|--------|
| `ngram_range=(1, 2)` | Sertakan bigram. Biasanya meningkatkan klasifikasi. |
| `min_df=2` | Masukkan kata-kata dalam waktu kurang dari 2 dokumen. Memangkas kosakata pada data yang berisik. |
| `max_df=0.95` | Masukkan kata-kata di lebih dari 95% dokumen. Perkiraan penghapusan stopword tanpa daftar hardcode. |
| `stop_words="english"` | daftar stopword bawaan scikit-learn. Bergantung pada tugas — analisis sentimen *tidak boleh* menghilangkan negasi. |
| `sublinear_tf=True` | Gunakan `1 + log(tf)` alih-alih `tf` mentah. Membantu ketika suatu istilah diulang berkali-kali dalam satu dokumen. |

### Saat TF-IDF masih menang (per 2026)

- Deteksi spam, pelabelan topik, penandaan anomali log. Kehadiran kata adalah yang terpenting; nuansa semantik tidak.
- Rezim data rendah (ratusan contoh berlabel). TF-IDF plus regresi logistik tidak memerlukan biaya pra-training.
- Latensi di mana pun penting. TF-IDF plus model linier menjawab dalam mikrodetik. Embed dokumen melalui Transformer membutuhkan waktu 10-100 ms.
- Sistem yang harus menjelaskan prediksinya. Periksa koefisien pengklasifikasi. Kata-kata positif teratas adalah alasannya.

### Ketika TF-IDF gagal

Kegagalan kebutaan semantik. Pertimbangkan dua dokumen ini:

- "Filmnya tidak bagus sama sekali."
- "Filmnya luar biasa."

Salah satunya adalah ulasan negatif. Yang satu positif. Tumpang tindih TF-IDF mereka persis `{the, movie, was}`. Pengklasifikasi kumpulan kata harus mengingat bahwa kata `not` di dekat `good` membalik label. Ia dapat mempelajari hal ini dengan data yang cukup, tetapi tidak pernah seanggun model yang memahami sintaksis.

Kegagalan lainnya: kata-kata di luar kosa kata pada inference. Model BoW yang dilatih berdasarkan ulasan IMDb tidak tahu apa yang harus dilakukan dengan `Zoomer-approved` jika token tersebut tidak pernah muncul dalam training. Embedding subkata (lesson 04) menangani hal ini. TF-IDF tidak bisa.

### Hibrida: embedding berbobot TF-IDF

Default pragmatis tahun 2026 untuk klasifikasi data menengah: gunakan weight TF-IDF sebagai attention pada embedding kata.

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

kamu mendapatkan kapasitas semantik dari embedding, dan penekanan kata langka dari TF-IDF. Pengklasifikasi berlatih pada vector gabungan. Performanya lebih baik untuk klasifikasi sentimen, topik, dan maksud di bawah sekitar 50 ribu contoh berlabel.

## Kirim

Simpan sebagai `outputs/prompt-vectorization-picker.md`:

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## Latihan1. **Mudah.** Menerapkan `cosine_similarity(doc_vec_a, doc_vec_b)` pada output TF-IDF yang dinormalisasi L2. Verifikasi bahwa dokumen identik mendapat skor 1,0 dan dokumen kosakata terpisah mendapat skor 0,0.
2. **Sedang.** Tambahkan dukungan `n-gram` ke `bag_of_words`. Parameter `n` menghasilkan hitungan lebih dari `n`-gram. Uji apakah `n=2` di `["the", "cat", "sat"]` menghasilkan jumlah bigram untuk `["the cat", "cat sat"]`.
3. **Sulit.** Buat hibrida embedding berbobot TF-IDF di atas menggunakan vector GloVe 100d (unduh sekali, cache). Bandingkan keakuratan klasifikasi dengan TF-IDF biasa dan embedding gabungan rata-rata pada dataset 20 Newsgroup. Laporkan siapa yang menang di mana.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Busur | Vector frekuensi kata | Jumlah kosakata dalam satu dokumen. Membuang pesanan. |
| TF | Frekuensi istilah | Jumlah kata dalam dokumen, secara opsional dinormalisasi berdasarkan panjang dokumen. |
| DF | Frekuensi dokumen | Jumlah dokumen yang mengandung kata tersebut setidaknya satu kali. |
| IDF | Frekuensi dokumen terbalik | `log(N / df)` dihaluskan. Mengurangi kata-kata yang muncul di mana-mana. |
| Vector jarang | Kebanyakan nol | Kosakata biasanya terdiri dari 10 ribu-100 ribu kata; sebagian besar tidak ada dalam dokumen tertentu. |
| Kesamaan kosinus | Sudut vector | Produk titik dari vector yang dinormalisasi L2. 1 identik, 0 ortogonal. |

## Bacaan Lanjutan

- [scikit-learn — ekstraksi feature dari teks](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — referensi API kanonik, ditambah catatan di setiap kenop.
- [Salton, G., & Buckley, C. (1988). Pendekatan weighting istilah dalam pengambilan teks otomatis](https://www.sciencedirect.com/science/article/pii/0306457388900210) — makalah yang menjadikan TF-IDF sebagai default selama satu dekade.
- ["Mengapa TF-IDF Masih Mengalahkan Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 diambil ketika metode lama menang dan alasannya.
