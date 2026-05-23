# Transformer Lengkap — Encoder + Decoder

> Attention adalah bintangnya. Segala sesuatu yang lain — residu, normalisasi, umpan maju, attention silang — adalah perancah yang memungkinkan kamu menumpuknya lebih dalam.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 7 · 02 (Attention Diri), Fase 7 · 03 (Attention Multi-Kepala), Fase 7 · 04 (Pengkodean Posisi)
**Waktu:** ~75 menit

## Masalah

Layer attention tunggal adalah pengekstraksi feature, bukan model. Satu matmul per layer tidak cukup untuk kapasitas bahasa. kamu membutuhkan kedalaman - dan kedalaman pecah tanpa pipa ledeng yang tepat.

Makalah Vaswani 2017 mengemas enam keputusan desain yang mengubah satu layer attention menjadi satu blok yang dapat ditumpuk. Setiap Transformer sejak — encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5) — mewarisi kerangka yang sama. Pada tahun 2026 blok telah disempurnakan (RMSNorm, SwiGLU, pre-norm, RoPE) tetapi kerangkanya identik.

Lesson ini adalah kerangkanya. Lesson berikutnya mengkhususkannya — 06 untuk encoder, 07 untuk decoder, 08 untuk encoder-decoder.

## Konsep

![Internal blok encoder dan decoder, berkabel](../assets/full-transformer.svg)

### Enam buah

1. **Embedding + sinyal posisi.** Token → vector. Posisi disuntikkan melalui RoPE (modern) atau sinusoidal (klasik).
2. **Attention pada diri sendiri.** Setiap posisi memperhatikan satu sama lain. Bertopeng dalam decoder.
3. **Jaringan feed-forward (FFN).** MLP dua lapis berdasarkan posisi: `W_2 · activation(W_1 · x)`. Rasio ekspansi 4× secara default.
4. **Sambungan sisa.** `x + sublayer(x)`. Tanpa ini, gradient akan hilang melewati ~6 layer.
5. **Normalisasi layer.** `LayerNorm` atau `RMSNorm` (modern). Menstabilkan aliran sisa.
6. **Attention silang (khusus decoder).** Kueri berasal dari decoder, kunci, dan nilai dari output encoder.

### Blok encoder (digunakan oleh BERT, encoder T5)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Encoder bersifat dua arah. Tanpa penyamaran. Semua posisi melihat semua posisi.

### Blok decoder (digunakan oleh GPT, decoder T5)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Decoder memiliki tiga sublayer per blok. Yang di tengah - attention silang - adalah satu-satunya tempat aliran informasi dari encoder ke decoder. Dalam arsitektur khusus dekoder murni (GPT), attention silang dihilangkan dan kamu hanya menutupi attention diri + FFN.

### Pra-norm vs pasca-norm

Kertas asli: `x + sublayer(LN(x))` vs `LN(x + sublayer(x))`. Pasca-norm tidak lagi disukai sekitar tahun 2019 - lebih sulit untuk berlatih secara mendalam tanpa pemanasan yang cermat. Pra-norm (`LN` *sebelum* sublayer) adalah default tahun 2026: Llama, Qwen, GPT-3+, Mistral semuanya menggunakannya.

### Blok yang dimodernisasi tahun 2026

Vaswani 2017 mengirimkan LayerNorm + ReLU. Tumpukan modern menggantikan keduanya. Seperti apa sebenarnya blok produksi itu:

| Komponen | 2017 | 2026 |
|-----------|------|------|
| Normalisasi | Norm Layer | RMSNorm |
| Activation FFN | ULT | SwiGLU |
| Ekspansi FFN | 4× | 2.6× (SwiGLU menggunakan tiga matrix, total parameter cocok) |
| Posisi | Mutlak sinusoidal | Tali |
| Attention | MHA Penuh | GQA (atau MLA) |
| Istilah bias | Ya | Tidak |

RMSNorm menghilangkan pemusatan rata-rata LayerNorm (satu pengurangan lebih sedikit), yang menghemat komputasi dan secara empiris setidaknya sama stabilnya. SwiGLU (`Swish(W1 x) ⊙ W3 x`) secara konsisten mengungguli ReLU/GELU FFN sebesar ~0,5 poin ppl di makalah Llama, PaLM, dan Qwen.

### Jumlah parameter

Untuk satu blok dengan `d_model = d` dan ekspansi FFN `r`:

- MHA: `4 · d²` (proyeksi Q, K, V, O)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- Norm: dapat diabaikanDi `d = 4096, r = 2.6, layers = 32` (kira-kira Llama 3 8B), total: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B` (ditambah embeddings dan head). Kecocokan yang diterbitkan dihitung.

## Build

### Langkah 1: elemen penyusunnya

Menggunakan kelas kecil `Matrix` dari Lesson 03 (disalin ke file ini untuk kemandirian):

- `layer_norm(x, eps=1e-5)` — kurangi mean, bagi dengan std.
- `rms_norm(x, eps=1e-6)` — dibagi dengan RMS. Tidak ada pengurangan berarti.
- `gelu(x)` dan `silu(x) * W3 x` (SwiGLU).
- `ffn_swiglu(x, W1, W2, W3)`.
- `encoder_block(x, params)` dan `decoder_block(x, enc_out, params)`.

Lihat `code/main.py` untuk pengkabelan selengkapnya.

### Langkah 2: sambungkan encoder 2 lapis dan decoder 2 lapis

Tumpuk mereka. Teruskan output encoder ke setiap attention silang decoder. Tambahkan LN terakhir sebelum proyeksi output.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Langkah 3: jalankan contoh mainan

Berikan sumber 6 token dan target 5 token. Pastikan bentuk keluarannya adalah `(5, vocab)`. Tidak ada training — lesson ini tentang arsitektur, bukan loss.

### Langkah 4: tukar dengan RMSNorm + SwiGLU

Ganti LayerNorm dan ReLU-FFN dengan RMSNorm dan SwiGLU. Konfirmasikan bentuk masih cocok. Inilah modernisasi 2026 dengan substitusi satu fungsi.

## Pakai

Implementasi referensi PyTorch/TF: `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`. Namun sebagian besar code produksi tahun 2026 menjalankan bloknya sendiri karena:

- Flash Attention dipanggil ke dalam attention, bukan melalui `nn.MultiheadAttention`.
- GQA/MLA tidak ada dalam referensi stdlib.
- RoPE, RMSNorm, SwiGLU bukan default PyTorch.

HF `transformers` memiliki blok referensi bersih yang harus kamu baca: `modeling_llama.py` adalah blok khusus dekoder kanonik 2026. Ini ~500 baris dan layak untuk dilalui sekali.

**Encoder vs decoder vs encoder-decoder — kapan harus memilih:**

| Butuh | Pilih | Contoh |
|------|------|---------|
| Klasifikasi, embedding, QA melalui teks | Hanya pembuat enkode | BERT, DeBERTa, ModernBERT |
| Pembuatan teks, obrolan, code, penalaran | Khusus dekoder | GPT, Llama, Claude, Qwen |
| Input terstruktur → output terstruktur (terjemahan, ringkasan) | Encoder-decoder | T5, BART, Bisikan |

Bahasa yang hanya dimenangkan oleh decoder karena skalanya paling bersih dan menangani pemahaman dan pembuatan. Encoder-decoder tetap terbaik bila input memiliki identitas "urutan sumber" yang jelas (terjemahan, pengenalan suara, tugas terstruktur).

## Kirim

Lihat `outputs/skill-transformer-block-reviewer.md`. Keterampilan meninjau implementasi blok Transformer baru terhadap default 2026 dan menandai bagian yang hilang (rasio ekspansi pra-norm, RoPE, RMSNorm, GQA, FFN).

## Latihan

1. **Mudah.** Hitung parameter di encoder_block kamu di `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. Validasi dengan menerapkan blok dan menggunakan `sum(p.numel() for p in block.parameters())`.
2. **Sedang.** Beralih dari pasca-norm ke pra-norm. Inisialisasi keduanya dan ukur norm activation setelah 12 layer ditumpuk pada input acak. Activation pasca-norm akan meledak; pra-norm harus tetap dibatasi.
3. **Sulit.** Menerapkan encoder-decoder 4 lapis pada tugas penyalinan mainan (salin `x` terbalik). Latih 100 langkah. Laporkan loss. Tukar dengan RMSNorm + SwiGLU + RoPE — apakah loss turun?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Blokir | "Satu layer Transformer" | Tumpukan norm + attention + norm + FFN, dibungkus dengan sambungan sisa. |
| Sisa | "Lewati koneksi" | `x + f(x)` output; memungkinkan aliran gradient melalui tumpukan yang dalam. |
| Pra-norm | "Normalisasi sebelum, bukan sesudahnya" | Modern: `x + sublayer(LN(x))`. Berlatih lebih dalam tanpa senam pemanasan. |
| RMSNorm | "LayerNorm tanpa maksud" | Bagi dengan RMS; satu operasi lebih sedikit, stabilitas empiris yang sama. |
| SwiGLU | "FFN semua orang beralih ke" | `Swish(W1 x) ⊙ W3 x → W2`. Mengalahkan ReLU/GELU di LM ppl. |
| Attention silang | "Bagaimana decoder melihat encoder" | MHA dengan Q dari decoder, K/V dari output encoder. |
| Ekspansi FFN | "Seberapa lebar MLP tengah" | Rasio ukuran tersembunyi terhadap d_model, biasanya 4 (LayerNorm) atau 2,6 (SwiGLU). |
| Bebas bias | "Hapus istilah +b" | Tumpukan modern menghilangkan bias pada layer linier; sedikit peningkatan ppl, model lebih kecil. |

## Bacaan Lanjutan

- [Vaswani dkk. (2017). Yang kamu Butuhkan Hanya Attention](https://arxiv.org/abs/1706.03762) — spesifikasi blok asli.
- [Xiong dkk. (2020). Tentang Normalisasi Layer dalam Arsitektur Transformer](https://arxiv.org/abs/2002.04745) — mengapa pra-norm sangat mengalahkan pasca-norm.
- [Zhang, Sennrich (2019). Normalisasi Layer Root Mean Square](https://arxiv.org/abs/1910.07467) — RMSNorm.
- [Shazeer (2020). Varian GLU Meningkatkan Transformer](https://arxiv.org/abs/2002.05202) — makalah SwiGLU.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — blok khusus dekoder kanonik 2026.
