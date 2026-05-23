# Pembuatan Teks Sebelum Transformers — Model Bahasa N-gram

> Jika sebuah kata mengejutkan, modelnya buruk. Perplexity membuat banyak kejutan. Penghalusan membuatnya tetap terbatas.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 5 · 01 (Pemrosesan Teks), Phase 2 · 14 (Naive Bayes)
**Waktu:** ~45 menit

## Masalah

Sebelum Transformer, sebelum RNN, sebelum embedding kata, model bahasa memperkirakan kata berikutnya dengan menghitung seberapa sering kata tersebut mengikuti kata `n-1` sebelumnya. Hitung "kucing" → "duduk" 47 kali, "kucing" → "melompat" 12 kali, "kucing" → "kulkas" 0 kali. Normalisasikan untuk mendapatkan distribusi probabilitas.

Itu adalah model bahasa n-gram. Ia menjalankan setiap pengenal ucapan, setiap pemeriksa ejaan, dan setiap sistem terjemahan mesin berbasis frasa dari tahun 1980 hingga 2015. Ia masih berjalan saat kamu memerlukan pemodelan bahasa murah di perangkat.

Masalah yang menarik adalah apa yang harus dilakukan terhadap n-gram yang tidak terlihat. Model berbasis penghitungan mentah memberikan probabilitas nol pada apa pun yang belum dilihatnya, yang merupakan bencana besar karena kalimatnya panjang dan hampir setiap kalimat panjang berisi setidaknya satu rangkaian yang tidak terlihat. Penelitian selama lima puluh tahun telah memperbaikinya. Hasilnya adalah pemulusan Kneser-Ney, dan pembelajaran mendalam modern mewarisi tradisi empirisnya.

## Konsep

![Model N-gram: menghitung, menghaluskan, menghasilkan](../assets/ngram.svg)

**Probabilitas N-gram:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. Perbaiki `n` (biasanya 3 untuk trigram, 4 untuk 4 gram). Hitung dari jumlah:

```text
P(w | context) = count(context, w) / count(context)
```

**Masalah penghitungan nol.** Setiap n-gram yang tidak terlihat dalam training mempunyai probabilitas nol. Sebuah studi tahun 2007 tentang korpus Brown menemukan bahwa model 4 gram pun memiliki 30% dari 4 gram yang tidak terlihat dalam training. kamu tidak dapat mengevaluasi teks nyata apa pun tanpa pemulusan.

**Pendekatan yang lebih halus, berdasarkan tingkat kecanggihannya:**

1. **Laplace (tambahkan satu).** Tambahkan 1 pada setiap hitungan. Sederhana, buruk pada kejadian langka.
2. **Good-Turing.** Mengalokasikan kembali massa probabilitas dari peristiwa yang berfrekuensi lebih tinggi ke peristiwa yang tidak terlihat berdasarkan frekuensi frekuensi.
3. **Interpolasi.** Gabungkan perkiraan n-gram, (n-1)-gram, dll. dengan weight yang dapat disetel.
4. **Backoff.** Jika n-gram memiliki hitungan nol, kembali ke (n-1)-gram. Backoff Katz menormalkan ini.
5. **Diskon mutlak.** Kurangi diskon tetap `D` dari semua hitungan, distribusikan kembali ke yang tidak terlihat.
6. **Kneser-Ney.** Diskon mutlak ditambah pilihan cerdas untuk model tingkat rendah: gunakan *probabilitas kelanjutan* (berapa banyak konteks sebuah kata muncul) alih-alih frekuensi mentah.

Wawasan Kneser-Ney sangat mendalam. "San Francisco" adalah bigram yang umum. Unigram "Francisco" kebanyakan muncul setelah "San". Diskon absolut yang naif memberikan probabilitas unigram yang tinggi kepada "Francisco" (karena hitungannya tinggi). Kneser-Ney memperhatikan bahwa "Francisco" hanya muncul dalam satu konteks dan menurunkan kemungkinan kelanjutannya. Hasil: bigram baru yang diakhiri dengan "Francisco" mendapatkan probabilitas rendah yang sesuai.

**Evaluasi: perplexity.** Eksponen rata-rata kemungkinan log negatif per kata pada set pengujian yang diadakan. Lebih rendah lebih baik. Perplexity 100 berarti model tersebut sama bingungnya dengan memilih secara seragam di antara 100 kata.

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## Build

### Langkah 1: jumlah trigram

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

Input adalah daftar kalimat yang diberi token. Outputnya adalah jumlah n-gram dan jumlah konteks. `<s>` dan `</s>` adalah batasan kalimat.

### Langkah 2: Penghalusan Laplace

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```Tambahkan 1 untuk setiap hitungan. Menghaluskan tetapi mengalokasikan massa secara berlebihan ke peristiwa yang tidak terlihat, sehingga merugikan peristiwa yang jarang diketahui juga.

### Langkah 3: Kneser-Ney (bigram, diinterpolasi)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

Tiga bagian yang bergerak. `continuation_prob` menangkap "dalam berapa banyak konteks berbeda kata ini muncul?" (inovasi Kneser-Ney). `lambda_prev` adalah massa yang dibebaskan dengan diskon, digunakan untuk membebani backoff. Probabilitas akhir adalah suku utama yang didiskontokan ditambah suku kelanjutan tertimbang.

### Langkah 4: menghasilkan teks dengan pengambilan sample

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

Pengambilan sample sebanding dengan probabilitas. Selalu memberikan hasil yang berbeda per benih. Untuk output seperti pencarian sinar, pilih argmax di setiap langkah (serakah) dan tambahkan kenop keacakan kecil (suhu).

### Langkah 5: perplexity

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

Lebih rendah lebih baik. Untuk korpus Brown, model KN 4 gram yang disetel dengan baik menghasilkan perplexity sekitar 140. Sebuah Transformer LM mencapai 15-30 pada set pengujian yang sama. Kesenjangannya sekitar 10x. Kesenjangan itulah yang menyebabkan lapangan terus bergerak.

## Pakai

- **Pengajaran NLP klasik.** Paparan paling jelas tentang pemulusan, MLE, dan perplexity yang bisa kamu dapatkan.
- **KenLM.** Pustaka n-gram produksi. Digunakan sebagai pencetak ulang dalam sistem ucapan dan MT yang mengutamakan latensi rendah.
- **Pelengkapan otomatis di perangkat.** Model trigram di keyboard. Tetap.
- **Garis dasar.** Selalu hitung perplexity LM n-gram sebelum menyatakan LM saraf kamu baik. Jika trafo kamu tidak mengalahkan KN dengan selisih yang besar, ada sesuatu yang salah.

## Kirim

Simpan sebagai `outputs/prompt-lm-baseline.md`:

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## Latihan

1. **Mudah.** Latih trigram LM pada korpus Shakespeare yang terdiri dari 1.000 kalimat. Hasilkan 20 kalimat. Pernyataan-pernyataan tersebut mungkin masuk akal secara lokal namun tidak koheren secara global. Ini adalah demo kanonik.
2. **Sedang.** Terapkan perplexity untuk model KN kamu pada perpecahan Shakespeare yang bertahan lama. Bandingkan dengan Laplace. kamu akan melihat KN menurunkan perplexity sebesar 30-50%.
3. **Sulit.** Buat korektor ejaan trigram: jika ada kata yang salah eja dan konteksnya, hasilkan koreksi dan beri peringkat berdasarkan probabilitas konteks di bawah LM. Evaluasi korpus ejaan Birkbeck (publik).

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| N-gram | Urutan kata | Urutan `n` token berturut-turut. |
| Menghaluskan | Menghindari angka nol | Mengalokasikan kembali massa probabilitas sehingga peristiwa yang tidak terlihat mendapatkan probabilitas bukan nol. |
| Perplexity | Metrik kualitas LM | `exp(-average log-prob)` pada data yang disimpan. Lebih rendah lebih baik. |
| Mundur | Penggantian ke konteks yang lebih pendek | Jika jumlah trigram nol, gunakan bigram. Backoff Katz meresmikan hal ini. |
| Kneser-Ney | Penghalusan terbaik untuk n-gram | Diskon mutlak + kemungkinan kelanjutan untuk model tingkat rendah. |
| Probabilitas kelanjutan | Khusus KN | `P(w)` ditimbang berdasarkan jumlah konteks `w` muncul, bukan berdasarkan hitungan mentah. |

## Bacaan Lanjutan- [Jurafsky dan Martin — Pemrosesan Ucapan dan Bahasa, Bab 3 (draf 2026)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — perlakuan kanonik terhadap LM n-gram dan penghalusannya.
- [Chen dan Goodman (1998). Studi Empiris Teknik Penghalusan untuk Pemodelan Bahasa](https://dash.harvard.edu/handle/1/25104739) — makalah yang menetapkan Kneser-Ney sebagai penghalus n-gram terbaik.
- [Kneser dan Ney (1995). Peningkatan Backing-off untuk Pemodelan Bahasa M-gram](https://ieeexplore.ieee.org/document/479394) — makalah KN asli.
- [KenLM](https://kheafield.com/code/kenlm/) — produksi cepat n-gram LM, masih digunakan pada tahun 2026 untuk aplikasi yang sensitif terhadap latensi.
