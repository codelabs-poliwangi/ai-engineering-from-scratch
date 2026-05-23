# Tokenization Subkata — BPE, WordPiece, Unigram, SentencePiece

> Pembuat token kata tersedak oleh kata-kata yang tidak terlihat. Tokenizer karakter memperbesar panjang urutan. Tokenizer subkata membagi perbedaannya. Setiap LLM modern dikirimkan dalam satu kapal.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 5 · 01 (Pemrosesan Teks), Phase 5 · 04 (GloVe / FastText / Subword)
**Waktu:** ~60 menit

## Masalah

Kosakata kamu memiliki 50.000 kata. Seorang pengguna mengetik "tidak dapat dikenali". Tokenizer kamu mengembalikan `[UNK]`. Model tersebut sekarang tidak memiliki sinyal tentang kata tersebut. Lebih buruk lagi: dokumen persentil ke-90 di korpus kamu memiliki 40 kata langka, yang berarti 40 bit informasi hilang per dokumen.

Tokenization subkata memecahkan masalah ini. Kata-kata umum tetap menjadi satu kesatuan. Kata-kata langka terurai menjadi bagian-bagian yang bermakna: `untokenizable` → `un`, `token`, `izable`. Training data mencakup segalanya karena string apa pun pada akhirnya merupakan urutan byte.

Setiap LLM frontier pada tahun 2026 dikirimkan dengan salah satu dari tiga algoritme (BPE, Unigram, WordPiece), yang dibungkus dalam salah satu dari tiga perpustakaan (tiktoken, SentencePiece, HF Tokenizers). kamu tidak dapat mengirimkan model bahasa tanpa memilihnya.

## Konsep

![BPE vs Unigram vs WordPiece, karakter demi karakter](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).** Mulailah dengan kosakata tingkat karakter. Hitung setiap pasangan yang berdekatan. Gabungkan pasangan yang paling sering menjadi token baru. Ulangi sampai kamu mencapai ukuran kosakata target. Algoritma dominan: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**BPE tingkat byte.** Algoritme yang sama tetapi menggunakan byte mentah (256 token dasar) dan bukan karakter Unicode. Menjamin nol `[UNK]` token — urutan byte apa pun yang dikodekan. GPT-2 menggunakan 50.257 token (256 byte + 50.000 gabungan + 1 khusus).

**Unigram.** Mulailah dengan kosakata yang banyak. Tetapkan setiap token probabilitas unigram. Pangkas token secara berulang yang penghapusannya paling sedikit meningkatkan kemungkinan log korpus. Probabilistik dalam inference: dapat mengambil sample tokenization (berguna untuk augmentasi data melalui regularisasi subkata). Digunakan oleh T5, mBART, ALBERT, XLNet, Gemma.

**WordPiece.** Gabungkan pasangan yang memaksimalkan kemungkinan korpus training, bukan frekuensi mentah. Digunakan oleh BERT, DistilBERT, ELECTRA.

**SentencePiece vs tiktoken.** SentencePiece adalah perpustakaan yang *melatih* kosakata (BPE atau Unigram) langsung pada teks Unicode mentah, menyandikan spasi putih sebagai `▁`. tiktoken adalah *encoder* OpenAI yang cepat terhadap kosakata bawaan; itu tidak melatih.

Aturan praktisnya:

- **Melatih kosakata baru:** SentencePiece (multibahasa, tanpa pra-tokenization) atau HF Tokenizers.
- **Inference cepat terhadap kosakata GPT:** tiktoken (cl100k_base, o200k_base).
- **Keduanya:** HF Tokenizer — satu perpustakaan, training + penyajian.

## Build

### Langkah 1: BPE dari awal

Lihat `code/main.py`. Lingkaran:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

Tiga fakta yang dikodekan oleh algoritma. `</w>` menandai akhir kata sehingga "rendah" (akhiran) dan "lebih rendah" (awalan) tetap berbeda. Weighting frekuensi membuat pasangan frekuensi tinggi menang lebih awal. Daftar gabungan diurutkan — inference menerapkan penggabungan dalam urutan training.

### Langkah 2: enkode dengan gabungan yang dipelajari

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

Naif O(n·|bergabung|). Implementasi produksi (tiktoken, HF Tokenizers) menggunakan pencarian peringkat gabungan dengan antrian prioritas dan dijalankan dalam waktu hampir linier.

### Langkah 3: Latihan Kalimat

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```Pemberitahuan: tidak diperlukan pra-tokenization, ruang dikodekan sebagai `▁`, `character_coverage` mengontrol seberapa agresif karakter langka dipertahankan vs dipetakan ke `<unk>`.

### Langkah 4: tiktoken untuk kosakata yang kompatibel dengan OpenAI

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Hanya pengkodean. Cepat (backend karat). Pencocokan persis dengan tokenization GPT-4/5 untuk penghitungan byte, estimasi biaya, penganggaran jendela konteks.

## Kesalahan yang masih dikirimkan pada tahun 2026

- **Tokenizer drift.** Training vocab A, penerapan terhadap vocab B. ID Token berbeda; model mengeluarkan sampah. Periksa `tokenizer.json` hash di CI.
- **Ambiguitas spasi.** BPE "halo" vs "halo" menghasilkan token yang berbeda. Selalu tentukan `add_special_tokens` dan `add_prefix_space` secara eksplisit.
- **Training multibahasa.** Korpora yang banyak berbahasa Inggris menghasilkan kosakata yang membagi skrip non-Latin menjadi token 5-10x lebih banyak. Permintaan yang sama membutuhkan biaya 5-10x lebih mahal dalam bahasa Jepang/Arab di GPT-3.5. o200k_base memperbaikinya sebagian.
- **Emoji terpecah.** Satu emoji dapat menampung 5 token. Penanganan emoji pos pemeriksaan saat menganggarkan konteks.

## Pakai

Tumpukan tahun 2026:

| Situasi | Pilih |
|-----------|------|
| Melatih model monolingual dari awal | Tokenizer HF (BPE) |
| Melatih model multibahasa | Kalimat Kalimat (Unigram, `character_coverage=0.9995`) |
| Melayani API yang kompatibel dengan OpenAI | tiktoken (`o200k_base` untuk GPT-4+) |
| Kosakata khusus domain (code, matematika, protein) | Latih BPE khusus pada korpus domain, gabungkan dengan kosakata dasar |
| Inference tepi, model kecil | Unigram (kosa kata yang lebih kecil berfungsi lebih baik) |

Ukuran kosakata adalah keputusan penskalaan, bukan suatu hal yang konstan. Heuristik kasar: 32k untuk parameter <1B, 50-100k untuk 1-10B, 200k+ untuk multibahasa/perbatasan.

## Kirim

Simpan sebagai `outputs/skill-tokenizer-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## Latihan

1. **Mudah.** Latih BPE gabungan 500 di korpus kecil `code/main.py`. Menyandikan tiga kata yang diulurkan. Berapa banyak yang menghasilkan tepat 1 token vs >1 token?
2. **Sedang.** Bandingkan jumlah token pada 100 kalimat Wikipedia bahasa Inggris antara `cl100k_base`, `o200k_base`, dan BPE SentencePiece yang kamu latih dengan vocab=32k. Laporkan rasio kompresi masing-masing.
3. **Sulit.** Latih korpus yang sama dengan BPE, Unigram, dan WordPiece. Ukur akurasi downstream saat menggunakan masing-masing pengklasifikasi sentimen kecil. Apakah pilihan tersebut menggerakkan jarum lebih dari 1 poin F1?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| BPE | Pengkodean Pasangan Byte | Penggabungan pasangan karakter yang paling sering dilakukan secara serakah hingga ukuran kosakata target tercapai. |
| BPE tingkat byte | Tidak ada token yang tidak diketahui | BPE lebih dari 256 byte mentah; GPT-2/Llama gunakan ini. |
| Unigram | Tokenizer probabilistik | Memangkas dari kumpulan kandidat besar menggunakan log-likelihood; digunakan oleh T5, Gemma. |
| Potongan Kalimat | Spasi putih | Perpustakaan yang melatih BPE/Unigram tentang teks mentah; ruang yang dikodekan sebagai `▁`. |
| tiktok | Yang cepat | Encoder BPE OpenAI yang didukung Rust untuk kosakata bawaan. Tidak ada training. |
| Gabungkan daftar | Angka ajaib | Daftar pesanan gabungan `(a, b) → ab`; inference berlaku secara berurutan. |
| Cakupan karakter | Seberapa langka dan terlalu langka? | Sebagian karakter dalam korpus training yang harus dicakup oleh tokenizer; ~0,9995 tipikal. |

## Bacaan Lanjutan- [Sennrich, Haddow, Birch (2015). Terjemahan Mesin Neural dari Kata-Kata Langka dengan Unit Subkata](https://arxiv.org/abs/1508.07909) — makalah BPE.
- [Kudo (2018). Regularisasi Subkata dengan Model Bahasa Unigram](https://arxiv.org/abs/1804.10959) — makalah Unigram.
- [Kudo, Richardson (2018). SentencePiece: Tokenizer subkata yang sederhana dan tidak bergantung pada bahasa](https://arxiv.org/abs/1808.06226) — perpustakaan.
- [Hugging Face — Ringkasan tokenizer](https://huggingface.co/docs/transformers/tokenizer_summary) — referensi ringkas.
- [Repo tiktoken OpenAI](https://github.com/openai/tiktoken) — buku masak + daftar pengkodean.
