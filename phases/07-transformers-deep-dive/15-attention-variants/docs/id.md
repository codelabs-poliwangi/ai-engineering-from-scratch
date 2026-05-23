# Varian Attention — Jendela Geser, Jarang, Diferensial

> Attention penuh berbentuk lingkaran. Setiap token melihat setiap token, dan memori membayar harganya. Empat varian membengkokkan bentuk lingkaran dan memulihkan separuh biaya.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 7 · 02 (Attention Mandiri), Fase 7 · 03 (Multi-Head), Fase 7 · 12 (KV Cache / Flash Attention)
**Waktu:** ~60 menit

## Masalah

Attention penuh memerlukan memori `O(N²)` dan komputasi `O(N²)` dalam panjang urutan. Untuk Llama 3 70B dengan konteks 128K yaitu 16 miliar entri attention per layer, dikalikan 80 layer. Flash Attention (Lesson 12) menyembunyikan memori activation `O(N²)` tetapi tidak mengubah biaya aritmatika — setiap token tetap memperhatikan setiap token lainnya.

Tiga kelas varian mengubah topologi matrix attention itu sendiri:

1. **Attention jendela geser (SWA).** Setiap token memperhatikan jendela tetangga yang tetap, bukan awalan penuh. Memori dan komputasi turun ke `O(N · W)` dengan `W` adalah jendelanya. Gemma 2/3, layer pertama Mistral 7B, Phi-3-Long.
2. **Attention jarang/blok.** Hanya pasangan terpilih `(i, j)` yang mendapat skor; sisanya dipaksa menjadi nol weight. Trafo sparse Longformer, BigBird, OpenAI.
3. **Attention yang berbeda.** Hitung dua peta attention dengan proyeksi Q/K terpisah, kurangi satu dari yang lain. Membunuh "penyerap attention" yang membebani beberapa token pertama. Transformer DIFF Microsoft (2024).

Ini hidup berdampingan. Model frontier tahun 2026 sering kali memadukannya: sebagian besar layer adalah SWA-1024, seperlima menjadi attention penuh global, dan segelintir merupakan kepala diferensial yang membersihkan pengambilan. Rasio SWA-global 5:1 Gemma 3 adalah standar buku teks saat ini.

## Konsep

### Attention Jendela Geser (SWA)

Setiap permintaan di posisi `i` hanya melayani posisi di `[i - W, i]` (SWA kausal) atau `[i - W/2, i + W/2]` (dua arah). Token di luar jendela mendapatkan `-inf` dalam matrix skor.

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

Untuk `N = 8192` dan `W = 1024`, matrix skor diharapkan memiliki 1024 × 8192 baris bukan nol — pengurangan 8×.

**Cache KV menyusut dengan SWA.** Hanya `W` token K dan V terakhir yang perlu disimpan per layer. Untuk konfigurasi Gemma-3-ish (jendela 1024, konteks 128K), cache KV turun 128×.

**Biaya kualitas.** Transformer khusus SWA mengalami kesulitan dalam pengambilan distance jauh. Cara mengatasinya: sisipkan layer SWA dengan layer attention penuh. Gemma 3 menggunakan SWA 5:1:global. Mistral 7B menggunakan tumpukan SWA kausal di mana informasi "mengalir maju" melalui jendela yang tumpang tindih - setiap layer memperluas bidang reseptif efektif sebesar `W`, dan setelah layer `L` model dapat menghadiri `L × W` token kembali.

### Jarang / Blokir Attention

Pilih pola ketersebaran `N × N` terlebih dahulu. Tiga bentuk kanonik:- **Lokal + melangkah (Transformer renggang OpenAI).** Hadiri token `W` terakhir ditambah setiap token `stride`-th sebelum itu. Merekam baik lokal maupun distance jauh di komputasi `O(N · sqrt(N))`.
- **Longformer / BigBird.** Jendela lokal + sekumpulan kecil token global (misalnya `[CLS]`) yang ditujukan untuk semua orang dan dihadiri oleh semua orang + tautan acak-jarang. Konteks 2× empiris dengan kualitas yang sesuai.
- **Native Sparse Attention (DeepSeek, 2025).** Learn blok mana dari `(Q, K)` yang penting; lewati blok nol di tingkat kernel. Kompatibel dengan FlashAttention.

Sedikit attention adalah kisah rekayasa kernel. Perhitungannya sederhana (menutupi matrix skor); kemenangan datang dari tidak pernah memuat entri nol ke SRAM. FlashAttention-3 dan FlexAttention API 2026 membuat pola renggang khusus menjadi kelas satu di PyTorch.

### Attention Diferensial (DIFF Transformer, 2024)

Attention reguler memiliki masalah "penyerapan attention": softmax memaksa setiap baris berjumlah 1, sehingga token yang tidak ingin memperhatikan apa pun secara khusus akan membuang weight pada token pertama (atau beberapa yang pertama). Ini mencuri kapasitas yang seharusnya digunakan untuk konten sebenarnya.

Attention diferensial memperbaikinya dengan menghitung **dua** peta attention dan mengurangi:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

dimana `λ` adalah scalar yang dipelajari (biasanya 0,5–0,8). A1 menangkap weight konten nyata; A2 menangkap wastafel. Pengurangan membatalkan sink, mengalokasikan kembali weight ke token yang relevan.

Hasil yang dilaporkan (Microsoft 2024): tingkat perplexity 5–10% lebih rendah, konteks efektif 1,5–2× lebih panjang dengan durasi training yang sama, pengambilan jarum di tumpukan jerami lebih tajam.

### Perbandingan Varian

| Varian | Hitung | Cache KV | Kualitas vs penuh | Penggunaan produksi |
|---------|---------|----------|-----------------|----------------|
| Attention penuh | O(N²) | PADA(N) per layer | dasar | layer default setiap model |
| SWA (jendela 1024) | DI(N·W) | O(W) per layer | -0,1 ppl, bagus dengan layer global | Gemma 2/3, Phi-3-Panjang |
| Lokal + melangkah jarang | O(N·√N) | campuran | mirip dengan SWA | Transformer jarang OpenAI, Longformer |
| BigBird (lokal + global + acak) | PADA(N) kira-kira | campuran | cocok penuh pada konteks 2× | BERT konteks panjang awal |
| Jarang Asli (DeepSeek-V3.2) | O(N · pecahan aktif) | PADA(N) | dalam 0,05 orang | DeepSeek-V3.2, 2025 |
| Diferensial | O(2·N²) | O(2N) | -5 hingga -10% orang | DIFF Transformer, model awal 2026 |

## Build

Lihat `code/main.py`. Kami menerapkan komparator topeng kausal yang menunjukkan attention penuh, SWA, lokal+langkah, dan diferensial secara berdampingan pada rangkaian mainan.

### Langkah 1: masker kausal penuh (dasar)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Garis dasar dari Lesson 07. Segitiga bawah; weight nol di atas diagonal.

### Langkah 2: topeng kausal jendela geser

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

Satu parameter — `window`. Untuk `window >= n`, kamu memulihkan attention kausal penuh. Untuk `window = 1`, setiap token hanya melayani dirinya sendiri.

### Langkah 3: masker lokal + masker jarang

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

Jendela lokal yang padat ditambah setiap `stride`-token kembali ke awal urutan. Bidang reseptif tumbuh dalam langkah-langkah log dengan layer tambahan.

### Langkah 4: attention yang berbeda

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

Dua attention berlalu, kurangi dengan koefisien pencampuran yang dipelajari. Dalam code tersebut kita membandingkan heatmap attention-sink tunggal vs diferensial dan melihat sink runtuh.

### Langkah 5: Ukuran cache KVCetak ukuran cache per layer di `N = 131072` untuk setiap varian. SWA dan varian jarang turun 10–100×. Diferensial ganda. Bayar tagihan memori kamu secara sadar.

## Pakai

Pola produksi tahun 2026:

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

FlexAttention di PyTorch 2.5+ menerima fungsi mask:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

Ini dikompilasi ke kernel Triton khusus. Dalam 10% dari kecepatan FlashAttention-3 untuk pola umum, dan fungsi mask dapat dipanggil dengan Python.

**Kapan memilih masing-masing:**

- **Attention penuh murni** — setiap layer hingga ~16 ribu konteks, atau ketika kualitas pengambilan adalah yang terpenting.
- **SWA + campuran global** — konteks panjang (>32K), training dan memori inference terikat. Default 2026 diatas 32K.
- **Attention blok jarang** — kernel khusus, pola khusus. Dicadangkan untuk weight kerja khusus (pengambilan, audio).
- **Attention yang berbeda** — weight kerja apa pun yang menyebabkan kontaminasi pada pipeline attention (RAG konteks panjang, tumpukan jerami).

## Kirim

Lihat `outputs/skill-attention-variant-picker.md`. Keterampilan memilih topologi attention untuk model baru dengan mempertimbangkan panjang konteks target, permintaan pengambilan, dan profil komputasi training/inference.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Verifikasi SWA di `window=4` menghilangkan semua yang ada di luar 4 token terakhir per baris. Pastikan `window=n` mereproduksi attention kausal penuh secara sedikit identik.
2. **Sedang.** Terapkan SWA kausal dengan `window=1024` di atas batu penjuru Lesson 07. Berlatih 1.000 langkah di tinyshakespeare. Berapa kemunduran val loss vs attention penuh? Berapa penurunan memori puncak?
3. **Hard.** Menerapkan campuran layer 5:1 gaya Gemma-3 (5 SWA, 1 global) dalam model batu penjuru. Bandingkan kehilangan, memori, dan kualitas pembangkitan dengan data dasar SWA murni dan global murni pada parameter yang cocok.
4. **Sulit.** Menerapkan attention diferensial dengan `λ` yang dipelajari per kepala. Latih tugas pengambilan sintetik (satu jarum, 2.000 pengecoh). Ukur akurasi pengambilan vs garis dasar attention tunggal pada parameter yang cocok.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Attention jendela geser (SWA) | "Attention lokal" | Setiap kueri menangani token `W` terakhirnya; Cache KV menyusut menjadi `O(W)`. |
| Bidang reseptif yang efektif | "Seberapa jauh model melihat ke belakang" | Dalam tumpukan SWA layer `L` dengan jendela `W`, hingga token `L × W`. |
| Mantan / BigBird | "Lokal + global + acak" | Pola yang jarang dengan beberapa token global yang selalu hadir; pendekatan konteks panjang awal. |
| Attention Jarang Asli | "Trik kernel DeepSeek" | Learn ketersebaran tingkat blok; lewati nol blok di tingkat kernel dengan tetap menjaga kualitas. |
| Perbedaan attention | "Dua peta, satu kurangi" | DIFF Transformer: kurangi `λ` kali peta attention kedua dari peta attention pertama untuk membatalkan pemusatan attention. |
| Penyerap attention | "Berat berdarah hingga token 0" | Normalisasi Softmax memaksa baris berjumlah 1; kueri yang tidak informatif membuang weight pada posisi 0. |
| FleksibelPerhatian | "Topeng-sebagai-Python" | PyTorch 2.5+ API yang mengkompilasi fungsi topeng arbitrer ke dalam kernel berbentuk FlashAttention. |
| Campuran jenis layer | "5:1 SWA-ke-global" | Sisipkan layer dengan attention penuh dan jarang dalam tumpukan untuk menjaga kualitas pada memori yang lebih rendah. |

## Bacaan Lanjutan- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — jendela geser kanonik + kertas token global.
- [Zaheer dkk. (2020). Big Bird: Transformers untuk Urutan yang Lebih Panjang](https://arxiv.org/abs/2007.14062) — lokal + global + acak.
- [Anak dkk. (2019). Menghasilkan Urutan Panjang dengan Transformer Jarang](https://arxiv.org/abs/1904.10509) — Pola lokal+strided OpenAI.
- [Tim Gemma (2024). Permata 2: Meningkatkan Model Bahasa Terbuka pada Ukuran Praktis](https://arxiv.org/abs/2408.00118) — campuran SWA:global 1:1.
- [Tim Gemma (2025). Laporan teknis Gemma 3](https://arxiv.org/abs/2503.19786) — campuran 5:1 dengan window=1024 yang kini menjadi default buku teks.
- [Kamu dkk. (2024). Transformer Diferensial](https://arxiv.org/abs/2410.05258) — Kertas Transformer DIFF.
- [Yuan dkk. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — attention terhadap ketersebaran yang dipelajari DeepSeek-V3.2.
- [PyTorch — blog dan dokumen FlexAttention](https://pytorch.org/blog/flexattention/) — Referensi API untuk pola mask-as-callable di Use It.
