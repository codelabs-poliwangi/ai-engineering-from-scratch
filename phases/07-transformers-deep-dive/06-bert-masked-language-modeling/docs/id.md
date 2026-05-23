# BERT — Pemodelan Bahasa Terselubung

> GPT memprediksi kata berikutnya. BERT memprediksi kata yang hilang. Satu kalimat yang membedakan - dan setengah dekade dari segala sesuatunya berbentuk embedding.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 7 · 05 (Transformer Penuh), Fase 5 · 02 (Representasi Teks)
**Waktu:** ~45 menit

## Masalah

Pada tahun 2018, setiap tugas NLP - sentimen, NER, QA, keterlibatan - melatih modelnya sendiri dari awal pada data berlabelnya sendiri. Tidak ada pos pemeriksaan "memahami bahasa Inggris" terlatih yang dapat kamu sesuaikan. ELMo (2018) menunjukkan bahwa kamu dapat melatih embedding kontekstual terlebih dahulu dengan LSTM dua arah; itu membantu tetapi tidak menggeneralisasi.

BERT (Devlin dkk. 2018) bertanya: bagaimana jika kita menggunakan encoder Transformer, melatihnya pada setiap kalimat di internet, dan memaksanya untuk memprediksi kata-kata yang hilang dari konteks di kedua sisi? Kemudian kamu menyempurnakan satu tugas pada tugas hilir kamu. Efisiensi parameter adalah sebuah wahyu.

Hasilnya: dalam waktu 18 bulan BERT dan variannya (RoBERTa, ALBERT, ELECTRA) mendominasi setiap leaderboard NLP yang ada. Pada tahun 2020, setiap mesin pencari, pipeline moderasi konten, dan sistem pencarian semantik di dunia memiliki BERT di dalamnya.

Pada tahun 2026, model khusus encoder masih menjadi alat yang tepat untuk klasifikasi, pengambilan, dan ekstraksi terstruktur — model ini berjalan 5–10× lebih cepat per token dibandingkan decoder dan embedding-nya adalah tulang punggung setiap tumpukan pengambilan modern. ModernBERT (Des 2024) mendorong arsitektur ke konteks 8K dengan Flash Attention + RoPE + GeGLU.

## Konsep

![Pemodelan bahasa yang disamarkan: pilih token, tutupi, prediksi yang asli](../assets/bert-mlm.svg)

### Sinyal training

Ambil kalimat: `the quick brown fox jumps over the lazy dog`.

Tutupi 15% token secara acak:

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

Latih model untuk memprediksi token asli pada posisi bertopeng. Karena encoder bersifat dua arah, memprediksi `[MASK]` di posisi 1 dapat menggunakan `brown fox jumps` di posisi 2+. Ini adalah hal yang tidak dapat dilakukan oleh GPT.

### Aturan topeng BERT

Dari 15% token yang dipilih untuk prediksi:

- 80% diganti dengan `[MASK]`.
- 10% diganti dengan token acak.
- 10% dibiarkan tidak berubah.

Mengapa tidak selalu `[MASK]`? Karena `[MASK]` tidak pernah muncul pada waktu inference. Melatih model untuk mengharapkan `[MASK]` pada 100% posisi yang disamarkan akan menciptakan pergeseran distribusi antara pra-training dan penyesuaian. 10% acak + 10% tidak berubah membuat model tetap jujur.

### Prediksi Kalimat Berikutnya (NSP) — dan alasannya dihilangkan

BERT asli juga dilatih di NSP: diberi dua kalimat A dan B, prediksi jika B mengikuti A. RoBERTa (2019) menghapuskannya dan menunjukkan NSP sakit, bukan membantu. Pembuat enkode modern melewatkannya.

### Apa yang berubah pada tahun 2026: ModernBERT

Makalah ModernBERT tahun 2024 membangun kembali blok tersebut dengan primitif tahun 2026:

| Komponen | BERT Asli (2018) | ModernBERT (2024) |
|-----------|----------------------|----------------------------------|
| Posisi | Dipelajari mutlak | Tali |
| Activation | GELU | GeGLU |
| Normalisasi | Norm Layer | RMSNorm pra-norm |
| Attention | Padat penuh | Lokal bergantian (128) + global |
| Panjang konteks | 512 | 8192 |
| Tokenizer | Potongan Kata | BPE |

Dan tidak seperti tumpukan 2018, ini adalah Flash-Attention-native. Inference 2–3× lebih cepat pada panjang urutan 8K dibandingkan DeBERTa-v3 dengan skor GLUE yang lebih baik.

### Kasus penggunaan yang masih memilih encoder pada tahun 2026| Tugas | Mengapa encoder mengalahkan decoder |
|------|---------------------------|
| Pengambilan / embedding pencarian semantik | Konteks dua arah = kualitas embedding per token yang lebih baik |
| Klasifikasi (sentimen, niat, toksisitas) | Satu umpan ke depan; tidak ada overhead pembangkitan |
| Pelabelan NER / token | Output per posisi, awalnya dua arah |
| Keterlibatan tembakan nol (NLI) | Kepala pengklasifikasi di atas pembuat enkode |
| Pemeringkatan ulang untuk RAG | Penilaian lintas-encoder, 10x lebih cepat dari pemeringkat ulang LLM |

## Build

### Langkah 1: menutupi logika

Lihat `code/main.py`. Fungsi `create_mlm_batch` mengambil daftar ID token, ukuran vocab, dan probabilitas mask. Mengembalikan ID input (dengan masker diterapkan) dan label (hanya pada posisi bertopeng, -100 di tempat lain — konvensi indeks abaikan PyTorch).

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### Langkah 2: jalankan prediksi MLM pada korpus kecil

Latih encoder 2 lapis + MLM dengan kosakata 20 kata, 200 kalimat. Tidak ada gradient — kami melakukan pemeriksaan kewarasan forward-pass. Training penuh membutuhkan PyTorch.

### Langkah 3: bandingkan jenis masker

Tunjukkan bagaimana aturan tiga arah menjaga model tetap dapat digunakan tanpa `[MASK]`. Memprediksi kalimat terbuka kedok dan kalimat bertopeng. Keduanya harus menghasilkan distribusi token yang masuk akal karena model melihat kedua pola tersebut dalam training.

### Langkah 4: menyempurnakan kepala

Ganti kepala MLM dengan kepala klasifikasi pada dataset sentimen mainan. Hanya kepala yang berlatih; pembuat enkode dibekukan. Ini adalah pola yang diikuti setiap aplikasi BERT.

## Pakai

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Model embedding adalah BERT yang disempurnakan.** `sentence-transformers` model seperti `all-MiniLM-L6-v2` adalah BERT yang dilatih dengan loss kontrastif. Encodernya sama. Kerugiannya berubah.

**Pemeringkatan ulang lintas-encoder juga merupakan BERT yang disempurnakan.** Klasifikasi berpasangan di `[CLS] query [SEP] doc [SEP]`. Attention dua arah antara kueri dan dokumen inilah yang memberikan keunggulan kualitas bagi pembuat enkode silang dibandingkan biencoder.

**Kapan tidak memilih BERT pada tahun 2026.** Apa pun yang bersifat generatif. Pembuat enkode tidak memiliki cara yang masuk akal untuk menghasilkan token secara otomatis. Juga: parameter apa pun di bawah 1B di mana dekoder kecil dapat mencocokkan kualitas dengan lebih banyak fleksibilitas (Phi-3-Mini, Qwen2-1.5B).

## Kirim

Lihat `outputs/skill-bert-finetuner.md`. Keterampilan ini mencakup penyesuaian BERT (pilihan tulang punggung, spesifikasi kepala, data, evaluasi, penghentian) untuk tugas klasifikasi atau ekstraksi baru.

## Latihan

1. **Mudah.** Jalankan `code/main.py` dan cetak distribusi masker pada 10.000 token. Konfirmasikan ~15% dipilih, dan ~80% di antaranya menjadi `[MASK]`.
2. **Sedang.** Terapkan penyembunyian seluruh kata: jika sebuah kata diberi token menjadi subkata, sembunyikan semua subkata secara bersamaan atau tidak sama sekali. Ukur apakah ini meningkatkan akurasi MLM pada korpus 500 kalimat.
3. **Sulit.** Latih BERT kecil (2 layer, d=64) pada 10.000 kalimat dari dataset publik. Sempurnakan token `[CLS]` untuk sentimen SST-2. Bandingkan dengan garis dasar khusus decoder pada parameter yang cocok — mana yang menang?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| MLM | "Pemodelan bahasa bertopeng" | Sinyal training: ganti 15% token secara acak dengan `[MASK]`, prediksi yang asli. |
| Dua arah | "Terlihat dua arah" | Attention encoder tidak memiliki topeng sebab akibat - setiap posisi melihat setiap posisi lainnya. |
| `[CLS]` | "Token penyatuan" | Token khusus ditambahkan ke setiap urutan; embedding terakhirnya digunakan sebagai representasi tingkat kalimat. |
| `[SEP]` | "Pemisah segmen" | Memisahkan urutan berpasangan (misalnya kueri/dokumen, kalimat A/B). |
| NSP | "Prediksi kalimat berikutnya" | tugas pra-training kedua BERT; terbukti tidak berguna di RoBERTa, dijatuhkan setelah 2019. |
| Penyempurnaan | "Beradaptasi dengan tugas" | Jaga agar sebagian besar encoder tetap beku; latih kepala kecil di atas untuk tugas hilir. |
| Pembuat enkode silang | "Seorang pembuat ulang" | BERT yang menggunakan kueri dan dokumen sebagai input, menghasilkan skor relevansi. |
| ModernBERT | "Penyegaran 2024" | Encoder dibangun kembali dengan RoPE, RMSNorm, GeGLU, attention lokal/global bergantian, konteks 8K. |

## Bacaan Lanjutan

- [Devlin dkk. (2018). BERT: Pra-training Transformer Dua Arah Mendalam untuk Pemahaman Bahasa](https://arxiv.org/abs/1810.04805) — makalah asli.
- [Liu dkk. (2019). RoBERTa: Pendekatan Pra-Training BERT yang Dioptimalkan dengan Kuat](https://arxiv.org/abs/1907.11692) — cara melatih BERT dengan benar; membunuh NSP.
- [Clark dkk. (2020). ELECTRA: Encoder Teks Pra-training sebagai Diskriminator Daripada Generator](https://arxiv.org/abs/2003.10555) — deteksi token yang diganti mengalahkan MLM pada komputasi yang cocok.
- [Warner dkk. (2024). Lebih Cerdas, Lebih Baik, Lebih Cepat, Lebih Lama: Encoder Dua Arah Modern](https://arxiv.org/abs/2412.13663) - Makalah ModernBERT.
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — referensi encoder kanonik.
