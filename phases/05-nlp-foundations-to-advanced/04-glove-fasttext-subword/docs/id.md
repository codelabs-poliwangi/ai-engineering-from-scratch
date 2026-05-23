# GloVe, FastText, dan Embedding Subkata

> Word2Vec melatih satu embedding per kata. GloVe memfaktorkan matrix kejadian bersama. FastText embed potongan-potongan itu. BPE dijembatani ke trafo.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec dari Awal)
**Waktu:** ~45 menit

## Masalah

Word2Vec meninggalkan dua pertanyaan terbuka.

Pertama, terdapat penelitian paralel yang memfaktorkan matrix kejadian bersama secara langsung (LSA, HAL) dibandingkan melakukan pembaruan skip-gram secara online. Apakah pendekatan berulang Word2Vec secara fundamental lebih baik, atau apakah perbedaannya merupakan artefak dari cara kedua metode menangani hal tersebut? **GloVe** menjawab bahwa: faktorisasi matrix dengan kekalahan yang dipilih dengan cermat cocok atau mengalahkan Word2Vec, dan biaya training-nya lebih murah.

Kedua, tidak ada metode yang memiliki cerita untuk kata-kata yang belum pernah dilihatnya. `Zoomer-approved`, `dogecoin`, setiap kata benda yang diciptakan minggu lalu, setiap bentuk infleksi dari akar kata langka. **FastText** memperbaikinya dengan embed karakter n-gram: sebuah kata adalah jumlah dari bagian-bagiannya, termasuk morfem, sehingga kata-kata di luar kosakata pun mendapatkan vector yang masuk akal.

Ketiga, begitu trafo tiba, pertanyaannya kembali berubah. Kosakata tingkat kata membatasi sekitar satu juta entri; bahasa sebenarnya lebih terbuka dari itu. **Pengkodean pasangan byte (BPE)** dan kerabatnya memecahkan masalah ini dengan mempelajari kosakata unit subkata yang sering mencakup semuanya. Setiap tokenizer modern untuk setiap LLM modern adalah tokenizer subkata.

Lesson ini membahas ketiganya, lalu menjelaskan mana yang harus dicapai dan kapan.

## Konsep

**GloVe (Vector Global).** Buat matrix kemunculan bersama kata-kata `X` dengan `X[i][j]` adalah seberapa sering kata `j` muncul dalam konteks kata `i`. Latih vector sedemikian rupa sehingga `v_i · v_j + b_i + b_j ≈ log(X[i][j])`. Bobotnya turun sehingga frequent pair tidak mendominasi. Selesai.

**FastText.** Sebuah kata adalah jumlah karakternya n-gram ditambah kata itu sendiri. `where` menjadi `<wh, whe, her, ere, re>, <where>`. Kata vector adalah jumlah dari vector-vector komponen tersebut. Latih sebagai Word2Vec. Manfaat: kata-kata tak terlihat (`whereupon`) disusun dari n-gram yang diketahui.

**BPE (Byte-Pair Encoding).** Mulailah dengan kosakata byte (atau karakter) individual. Hitung setiap pasangan yang berdekatan dalam korpus. Gabungkan pasangan yang paling sering menjadi token baru. Ulangi untuk iterasi `k`. Hasil: kosakata token `k + 256` dengan urutan yang sering (`ing`, `tion`, `the`) adalah token tunggal dan kata-kata langka dipecah menjadi bagian-bagian yang familier. Setiap kalimat diubah menjadi sesuatu.

## Build

### GloVe: memfaktorkan matrix kejadian bersama

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

Dua bagian bergerak yang layak diberi nama. Fungsi weighting `f(x) = (x/x_max)^alpha` menurunkan pasangan yang sangat sering (seperti `(the, and)`) sehingga tidak mendominasi loss. Embedding akhir adalah jumlah tabel `W` (tengah) dan `W_tilde` (konteks). Menjumlahkan keduanya adalah trik yang dipublikasikan yang cenderung lebih baik jika hanya menggunakan satu saja.

### FastText: embedding sub-kata

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

Setiap kata diwakili oleh kumpulan n-gramnya (biasanya 3 hingga 6 karakter). Kata embedding adalah jumlah dari n-gram embedding-nya. Untuk training skip-gram, masukkan ini ke tempat Word2Vec menggunakan satu vector.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```Untuk kata yang tidak terlihat, kamu masih mendapatkan vector selama beberapa n-gramnya diketahui. `whereupon` berbagi `<wh`, `her`, `ere`, dan `<where` dengan `where`, sehingga keduanya mendarat berdekatan.

### BPE: mempelajari kosakata subkata

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

Iterasi pertama menggabungkan pasangan berdekatan yang paling umum. Setelah cukup iterasi, substring yang sering (`low`, `est`, `tion`) menjadi token tunggal dan kata-kata langka terpecahkan dengan rapi.

Tokenizer GPT / BERT / T5 yang sebenarnya mempelajari penggabungan 30k-100k. Hasilnya: teks apa pun diberi token menjadi rangkaian ID yang dikenal dengan panjang terbatas, tidak pernah ada OOV.

## Pakai

Dalam praktiknya, kamu jarang melatih hal-hal ini sendiri. kamu memuat pos pemeriksaan terlatih.

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Untuk tokenization subkata bergaya BPE di era Transformer:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

Awalan `Ġ` menandai batasan kata (konvensi GPT-2). Setiap tokenizer modern adalah varian BPE, WordPiece (BERT), atau SentencePiece (T5, LLaMA).

### Kapan harus memilih yang mana

| Situasi | Pilih |
|-----------|------|
| Vector kata tujuan umum yang telah dilatih sebelumnya, tidak diperlukan toleransi OOV | Sarung Tangan 300d |
| Vector kata tujuan umum yang telah dilatih sebelumnya, harus menangani kesalahan ejaan / neologisme / bahasa yang kaya secara morfologis | Teks Cepat |
| Apa pun yang masuk ke Transformer (training atau inference) | Tokenizer apa pun yang modelnya dikirimkan. Jangan pernah bertukar. |
| Melatih model bahasa kamu sendiri dari awal | Latih tokenizer BPE atau SentencePiece di korpus kamu terlebih dahulu |
| Klasifikasi teks produksi dengan model linier | Masih TF-IDF. Lesson 02. |

## Kirim

Simpan sebagai `outputs/skill-tokenizer-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## Latihan

1. **Mudah.** Jalankan `char_ngrams("playing")` dan `char_ngrams("played")`. Hitung tumpang tindih Jaccard dari dua himpunan n-gram. kamu akan melihat bagian substansial yang dibagikan (`pla`, `lay`, `play`), itulah sebabnya FastText dapat ditransfer dengan baik di seluruh varian morfologi.
2. **Sedang.** Perluas `learn_bpe` untuk melacak pertumbuhan kosakata. Plot token-per-corpus-character sebagai fungsi dari jumlah penggabungan. kamu akan melihat kompresi cepat pada awalnya, tanpa gejala mendekati ~2-3 karakter per token.
3. **Sulit.** Latih BPE gabungan 1k pada karya lengkap Shakespeare. Bandingkan tokenization kata-kata umum vs. kata benda langka. Ukur rata-rata token per kata sebelum dan sesudah. Tuliskan apa yang mengejutkan kamu.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Matrix kejadian bersama | Tabel frekuensi kata-kata | `X[i][j]` = seberapa sering kata `j` muncul di jendela sekitar kata `i`. |
| Subkata | Sepotong kata | Karakter n-gram (FastText) atau token yang dipelajari (BPE/WordPiece/SentencePiece). |
| BPE | Pengkodean pasangan byte | Penggabungan berulang dari pasangan berdekatan yang paling sering terjadi hingga kosakata mencapai ukuran target. |
| OOV | Kehabisan kosakata | Kata yang belum pernah dilihat modelnya. Word2Vec/GloVe gagal. FastText dan BPE menanganinya. |
| BPE tingkat byte | BPE pada byte mentah | Skema GPT-2. Kosakata dimulai dengan 256 byte, jadi tidak ada yang OOV. |

## Bacaan Lanjutan- [Pennington, Socher, Manning (2014). GloVe: Vector Global untuk Representasi Kata](https://nlp.stanford.edu/pubs/glove.pdf) — makalah GloVe, tujuh halaman, masih merupakan turunan terbaik dari loss tersebut.
- [Bojanowski dkk. (2017). Memperkaya Vector Kata dengan Informasi Subkata](https://arxiv.org/abs/1607.04606) — FastText.
- [Sennrich, Haddow, Birch (2016). Terjemahan Mesin Neural Kata Langka dengan Unit Subkata](https://arxiv.org/abs/1508.07909) — makalah yang memperkenalkan BPE ke NLP modern.
- [Ringkasan tokenizer Hugging Face](https://huggingface.co/docs/transformers/tokenizer_summary) — perbedaan praktik BPE, WordPiece, dan SentencePiece.
