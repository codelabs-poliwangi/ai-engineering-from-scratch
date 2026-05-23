# Embedding Kata — Word2Vec dari Awal

> Sebuah kata adalah perusahaan yang dipegangnya. Latihlah jaring dangkal pada gagasan itu dan geometri akan hilang.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 5 · 02 (BoW + TF-IDF), Fase 3 · 03 (Propagasi Mundur dari Awal)
**Waktu:** ~75 menit

## Masalah

TF-IDF mengetahui `dog` dan `puppy` adalah kata yang berbeda. Ia tidak tahu bahwa maksud mereka hampir sama. Pengklasifikasi yang dilatih pada `dog` tidak dapat menggeneralisasi ulasan tentang `puppy`. kamu dapat mengatasinya dengan mencantumkan sinonim, tetapi gagal pada istilah langka, jargon domain, dan setiap bahasa yang tidak kamu antisipasi.

kamu ingin representasi di mana `dog` dan `puppy` mendarat berdekatan di ruang angkasa. Dimana `king - man + woman` mendarat di dekat `queen`. Saat model dilatih di `dog` mentransfer sejumlah sinyal ke `puppy` secara gratis.

Word2Vec memberi kami ruang itu. Jaringan saraf dua lapis, training triliunan token, diterbitkan pada tahun 2013. Arsitekturnya sangat sederhana. Hasilnya mengubah NLP selama satu dekade.

## Konsep

**Hipotesis distribusi** (Firth, 1957): "kamu akan mengetahui sebuah kata berdasarkan perusahaan yang dipegangnya." Jika dua kata muncul dalam konteks serupa, kemungkinan besar artinya serupa.

Word2Vec hadir dalam dua versi, keduanya memanfaatkan ide tersebut.

- **Lewati-gram.** Diberi kata tengah, prediksi kata di sekitarnya. `cat -> (the, sat, on)` dengan ukuran jendela 2.
- **CBOW (kumpulan kata berkelanjutan).** Berdasarkan kata-kata di sekitarnya, prediksi pusatnya. `(the, sat, on) -> cat`.

Skip-gram lebih lambat untuk dilatih tetapi menangani kata-kata langka dengan lebih baik. Itu menjadi default.

Jaringan memiliki satu layer tersembunyi tanpa nonlinier. Input adalah vector yang sangat menarik dalam kosakata. Outputnya adalah softmax atas kosakata. Setelah training, kamu membuang layer output. Weight layer tersembunyi adalah embeddingsnya.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

Caranya: softmax lebih dari 100 ribu kata itu sangat mahal. Word2Vec menggunakan **pengambilan sample negatif** untuk mengubahnya menjadi tugas klasifikasi biner. Prediksikan "apakah kata konteks ini muncul di dekat kata tengah ini, ya atau tidak". Cicipi beberapa kata negatif (yang tidak muncul bersamaan) per pasangan training alih-alih menghitung softmax pada keseluruhan kosakata.

## Build

### Langkah 1: latih pasangan dari korpus

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

Setiap pasangan (tengah, konteks) di jendela adalah contoh training yang positif.

### Langkah 2: embed tabel

Dua matrix. `W` adalah tabel embedding kata tengah (yang kamu simpan). `W'` adalah tabel kata konteks (sering dibuang, terkadang dirata-ratakan dengan `W`).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

Init acak kecil. Ukuran kosakata 10k dan redup 100 realistis; untuk pengajaran cukup 50 vocab x 16 dim untuk melihat geometri.

### Langkah 3: tujuan pengambilan sample negatif

Untuk setiap pasangan positif `(center, context)`, contohkan `k` kata acak dari kosakata sebagai negatif. Latih model sehingga perkalian titik `W[center] · W'[context]` bernilai tinggi untuk positif dan rendah untuk negatif.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

Rumus ajaibnya: loss logistik pada pasangan positif (ingin sigmoid mendekati 1) ditambah loss logistik pada pasangan negatif (ingin sigmoid mendekati 0). Gradient mengalir ke kedua tabel. Derivasi lengkap ada di makalah asli; telusuri sekali dengan pensil dan kertas jika kamu ingin menempel.

### Langkah 4: melatih korpus mainan

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```Setelah jangka waktu yang cukup lama pada korpus besar, kata-kata yang memiliki konteks yang sama memiliki embedding pusat yang serupa. Pada korpus mainan, kamu melihat efeknya secara samar. Pada miliaran token, kamu melihatnya secara dramatis.

### Langkah 5: trik analogi

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

Pada vector Google Berita 300d yang telah dilatih sebelumnya:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. Bukan karena modelnya tahu apa itu royalti. Karena vector `(king - man)` menangkap sesuatu seperti "kerajaan", dan menambahkannya ke `woman` tanah di dekat wilayah perempuan kerajaan.

## Pakai

Menulis Word2Vec dari awal adalah pengajaran. NLP produksi menggunakan `gensim`.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

Untuk pekerjaan nyata, kamu hampir tidak pernah melatih Word2Vec sendiri. kamu mengunduh vector terlatih.

- **GloVe** — Pendekatan faktorisasi matrix kejadian bersama Stanford. Pos pemeriksaan 50d, 100d, 200d, 300d. Cakupan umum yang bagus. Lesson 04 mencakup GloVe secara khusus.
- **fastText** — Ekstensi Word2Vec Facebook yang embed karakter n-gram. Menangani kata-kata di luar kosakata dengan menyusun subkata. Lesson 04.
- **Word2Vec yang telah dilatih sebelumnya di Google Berita** — 300d, kosakata 3 juta kata, diterbitkan tahun 2013. Masih diunduh setiap hari.

### Saat Word2Vec masih menang di tahun 2026

- Pengambilan khusus domain yang ringan. Latih abstrak medis dalam satu jam di laptop, dapatkan vector khusus yang tidak dapat ditangkap oleh model umum.
- Rekayasa feature bergaya analogi. `gender_vector = mean(man - woman pairs)`. Kurangi dari kata lain untuk mendapatkan sumbu netral gender. Masih digunakan dalam penelitian keadilan.
- Interpretasi. 100d cukup kecil untuk diplot melalui PCA atau t-SNE dan benar-benar melihat bentuk cluster.
- Inference di mana pun harus dijalankan di perangkat tanpa GPU. Pencarian Word2Vec adalah pengambilan baris tunggal.

### Dimana Word2Vec gagal

Dinding polisemi. `bank` memiliki satu vector. `river bank` dan `financial bank` membagikannya. `table` (spreadsheet vs. furnitur) membagikannya. Pengklasifikasi di bagian hilir tidak dapat membedakan indra dari vector.

Embedding kontekstual (ELMo, BERT, setiap Transformer sejak itu) menyelesaikan masalah ini dengan menghasilkan vector yang berbeda untuk setiap kemunculan kata berdasarkan konteks sekitarnya. Itulah lompatan dari Word2Vec ke BERT: dari statis ke kontekstual. Fase 7 mencakup separuh trafo.

Masalah di luar kosakata adalah kegagalan lainnya. Word2Vec belum pernah melihat `Zoomer-approved` jika tidak ada dalam training data. Tidak ada kemunduran. fastText memperbaikinya dengan komposisi subkata (lesson 04).

## Kirim

Simpan sebagai `outputs/skill-embedding-probe.md`:

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Latihan

1. **Mudah.** Jalankan loop training pada korpus kecil (20 kalimat tentang kucing dan anjing). Setelah 200 epoch, verifikasi `nearest(vocab, W, W[vocab["cat"]])` mengembalikan `dog` di 3 teratasnya. Jika tidak, tambah epoch atau kosakata.
2. **Sedang.** Tambahkan subsampling kata-kata yang sering digunakan. Kata-kata dengan frekuensi di atas `10^-5` dikeluarkan dari pasangan training dengan probabilitas sebanding dengan frekuensinya. Ukur pengaruhnya terhadap kesamaan kata langka.
3. **Sulit.** Melatih model pada korpus 20 Newsgroup. Hitung dua sumbu bias: `he - she` dan `doctor - nurse`. Proyeksikan kata-kata pekerjaan ke kedua sumbu. Laporkan pekerjaan mana yang memiliki kesenjangan bias terbesar. Ini adalah jenis penyelidikan keadilan yang digunakan para peneliti.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Embedding kata | Kata sebagai vector | Representasi yang padat dan redup (biasanya 100-300) yang dipelajari dari konteks. |
| Lewati-gram | Trik Word2Vec | Memprediksi kata konteks dari kata tengah. Lebih lambat dari CBOW, lebih baik untuk kata-kata langka. |
| Pengambilan sample negatif | Pintasan training | Ganti softmax pada kosakata lengkap dengan klasifikasi biner terhadap `k` kata-kata acak. |
| Embedding statis | Satu vector per kata | Vector yang sama terlepas dari konteksnya. Gagal dalam polisemi. |
| Embedding kontekstual | Vector peka konteks | Vector berbeda untuk setiap kemunculan berdasarkan kata-kata di sekitarnya. Transformer apa yang dihasilkan. |
| OOV | Kehabisan kosakata | Kata-kata tidak terlihat dalam training. Word2Vec tidak dapat menghasilkan vector untuk ini. |

## Bacaan Lanjutan

- [Mikolov dkk. (2013). Representasi Kata dan Frasa Terdistribusi serta Komposisinya](https://arxiv.org/abs/1310.4546) — makalah pengambilan sample negatif. Singkat dan mudah dibaca.
- [Rong, X. (2014). Penjelasan Pembelajaran Parameter word2vec](https://arxiv.org/abs/1411.2738) — gradient descent yang paling jelas, jika matematika makalah asli terasa padat.
- [tutorial gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) — pengaturan training produksi yang benar-benar berfungsi.
