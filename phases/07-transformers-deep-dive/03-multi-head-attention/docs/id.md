# Attention Multi-Kepala

> Satu attention head mempelajari satu hubungan pada satu waktu. Delapan kepala belajar delapan. Kepala gratis. Ambil lebih banyak dari mereka.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 7 · 02 (Attention Diri dari Awal)
**Waktu:** ~75 menit

## Masalah

Satu attention head diri menghitung satu matrix attention. Matrix tersebut menangkap satu jenis hubungan - biasanya hubungan yang meminimalkan loss pada sinyal training apa pun. Jika data kamu memiliki kesepakatan subjek-kata kerja, referensi bersama, wacana jangka panjang, dan pengelompokan sintaksis yang semuanya saling terkait, satu kepala akan menyebarkannya ke dalam satu distribusi soft-max dan kehilangan separuh sinyalnya.

Perbaikan dari makalah Vaswani 2017: menjalankan beberapa fungsi attention secara paralel, masing-masing dengan proyeksi Q, K, V sendiri, dan menggabungkan keluarannya. Setiap kepala beroperasi dalam subruang yang lebih kecil dengan dimension `d_model / n_heads`. Parameter total tetap sama. Kekuatan ekspresif meningkat.

Attention multi-head adalah standar yang disertakan setiap trafo pada tahun 2026. Satu-satunya argumen adalah tentang *berapa banyak* kepala dan apakah kunci dan nilai berbagi proyeksi (Attention Kueri Berkelompok, Attention Multi-Kueri, Attention Laten Multi-kepala).

## Konsep

![Attention multi-head terpecah, hadir, digabungkan](../assets/multi-head-attention.svg)

**Pisahkan.** Ambil `X` dari bentuk `(N, d_model)`. Proyeksikan ke Q, K, V masing-masing bentuk `(N, d_model)`. Bentuk ulang menjadi `(N, n_heads, d_head)` di mana `d_head = d_model / n_heads`. Transpos ke `(n_heads, N, d_head)`.

**Hadiri secara paralel.** Jalankan attention perkalian titik berskala di dalam setiap kepala. Setiap kepala menghasilkan `(N, d_head)`. Kepala beroperasi pada subruang berbeda dari embedding dan tidak pernah berbicara selama penghitungan attention itu sendiri.

**Gabungkan dan proyeksikan.** Tumpukan kembali ke `(N, d_model)` dan kalikan dengan matrix output yang dipelajari `W_o` berbentuk `(d_model, d_model)`. `W_o` adalah tempat berkumpulnya semua orang.

**Mengapa ini berhasil.** Setiap kepala dapat berspesialisasi tanpa bersaing dengan kepala lainnya untuk mendapatkan anggaran keterwakilan. Studi penyelidikan dari tahun 2019–2024 menunjukkan peran kepala yang berbeda: kepala posisional, kepala yang memperhatikan token sebelumnya, kepala salinan, kepala entitas bernama, kepala induksi (yang mendasari pembelajaran dalam konteks).

**Silsilah variasi tahun 2026:**

| Varian | Q kepala | Kepala K/V | Digunakan oleh |
|---------|---------|-----------|---------|
| Multi-kepala (MHA) | tidak | tidak | GPT-2, BERT, T5 |
| Multi-kueri (MQA) | tidak | 1 | PaLM, Elang |
| Kueri yang dikelompokkan (GQA) | tidak | G (misalnya N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Laten multi-kepala (MLA) | tidak | dikompresi ke peringkat rendah | DeepSeek-V2, V3 |

GQA adalah default modern karena memotong memori cache KV dengan faktor `N/G` sambil menjaga kualitas hampir penuh. MLA melangkah lebih jauh dengan mengompresi K/V ke dalam ruang laten, lalu memproyeksikan kembali pada waktu komputasi — memerlukan biaya FLOP, menghemat lebih banyak memori.

## Build

### Langkah 1: pisahkan kepala dari attention satu kepala yang sudah kita miliki

Ambil `SelfAttention` dari Lesson 02 dan bungkus dengan pasangan split/concat. Lihat `code/main.py` untuk implementasi yang numpy; logikanya adalah:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

Satu membentuk kembali dan satu mengubah posisi. Tidak ada putaran. Inilah yang dilakukan PyTorch di bawah `nn.MultiheadAttention`.

### Langkah 2: jalankan attention produk titik-skala per kepala

Setiap kepala mendapat potongan Q, K, V. Attention-nya menjadi matmul yang dikumpulkan:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```Pada perangkat keras sebenarnya `Qh @ Kh.transpose(...)` adalah `bmm`. GPU melihat satu kumpulan matmul dengan bentuk `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`. Menambahkan kepala gratis.

### Langkah 3: Varian Attention Kueri yang Dikelompokkan

Hanya proyeksi kunci dan nilai yang berubah. Q mendapat grup `n_heads`; K dan V mendapatkan grup `n_kv_heads < n_heads` dan diulang untuk mencocokkan:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

Kesimpulannya, ini menghemat memori karena hanya `n_kv_heads` salinan yang ada di cache KV, bukan `n_heads`. Llama 3 70B menggunakan 64 kepala kueri dengan 8 kepala KV — pengurangan cache 8×.

### Langkah 4: selidiki apa yang dipelajari setiap kepala

Jalankan MHA pada kalimat pendek dengan 4 kepala. Untuk setiap kepala, cetak matrix attention `(N, N)`. kamu akan melihat kepala yang berbeda memilih struktur yang berbeda bahkan dengan inisialisasi acak — itu sebagian merupakan sinyal, sebagian lagi simetri rotasi di subruang.

## Pakai

Di PyTorch, versi satu baris:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA pada PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**Berapa banyak kepala?** Aturan praktis dari model produksi pada tahun 2026:

| Ukuran model | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Kecil (~125 juta) | 768 | 12 | 64 |
| Basis (~350M) | 1024 | 16 | 64 |
| Besar (~1 miliar) | 2048 | 16 | 128 |
| Perbatasan (~70B) | 8192 | 64 | 128 |

`d_head` hampir selalu mendarat di 64 atau 128. Ini adalah satuan seberapa banyak kepala dapat "melihat". Turun di bawah 32 dan mulailah melawan faktor penskalaan `sqrt(d_head)`; melampaui 256 dan kamu kehilangan manfaat "banyak spesialis kecil".

## Kirim

Lihat `outputs/skill-mha-configurator.md`. Keterampilan ini merekomendasikan jumlah head, jumlah kv-head, dan strategi proyeksi untuk Transformer baru dengan mempertimbangkan anggaran parameter, panjang urutan, dan target penerapan.

## Latihan

1. **Mudah.** Ambil MHA dari `code/main.py` dan ubah `n_heads` dari 1 menjadi 16 dengan `d_model=64` diperbaiki. Plot hilangnya model satu layer kecil pada tugas penyalinan sintetis. Apakah lebih banyak kepala yang membantu, tidak bergerak, atau terluka?
2. **Sedang.** Menerapkan MQA (satu kepala KV digunakan bersama di semua kepala kueri). Ukur berapa banyak penurunan jumlah parameter vs MHA penuh. Hitung seberapa besar penyusutan ukuran cache KV pada inference untuk N=2048.
3. **Sulit.** Menerapkan versi kecil Attention Laten Multi-kepala: kompres K,V ke laten peringkat-`r`, simpan laten dalam cache KV, dekompresi pada waktu attention. Pada `r` apakah memori cache melampaui 1/8 MHA penuh sementara kualitas tetap dalam 1 bit validasi ppl?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Kepala | "Sirkuit attention tunggal" | Satu proyeksi dimension Q/K/V `d_head = d_model / n_heads` dengan matrix attention-nya sendiri. |
| d_head | "Dimension kepala" | Lebar tersembunyi per kepala; hampir selalu 64 atau 128 dalam produksi. |
| Pisahkan / gabungkan | "Membentuk kembali trik" | `(N, d_model) ↔ (n_heads, N, d_head)` membentuk ulang+mengubah posisi di sekitar attention. |
| A_o | "Proyeksi output" | `(d_model, d_model)` matrix diterapkan setelah menggabungkan kepala; tempat kepala bercampur. |
| MQA | "Satu kepala KV" | Attention Multi-Kueri: proyeksi K/V tunggal yang dibagikan. Cache KV terkecil, beberapa penurunan kualitas. |
| GQA | "Default sejak Llama 2" | Attention Kueri yang Dikelompokkan dengan `n_kv_heads < n_heads`; diulangi untuk mencocokkan Q. |
| MLA | "Trik DeepSeek" | Attention Laten Multi-kepala: K,V dikompresi menjadi laten tingkat rendah, didekompresi pada waktu kehadiran. |
| Kepala induksi | "Sirkuit di balik pembelajaran dalam konteks" | Sepasang kepala yang mendeteksi kejadian sebelumnya dan menyalin kejadian berikutnya. |

## Bacaan Lanjutan

- [Vaswani dkk. (2017). Yang kamu Butuhkan Hanya Attention §3.2.2](https://arxiv.org/abs/1706.03762) — spesifikasi multi-head asli.
- [Shazeer (2019). Decoding Transformer Cepat: Hanya Satu Write-Head yang kamu Butuhkan](https://arxiv.org/abs/1911.02150) — makalah MQA.
- [Ainslie dkk. (2023). GQA: Melatih Model Transformer Multi-Kueri Umum dari Pos Pemeriksaan Multi-Kepala](https://arxiv.org/abs/2305.13245) — cara mengonversi MHA ke GQA setelah training.
- [DeepSeek-AI (2024). Laporan Teknis DeepSeek-V2](https://arxiv.org/abs/2405.04434) — MLA dan alasannya mengalahkan MHA/GQA pada memori cache.
- [Olsson dkk. (2022). Pembelajaran Dalam Konteks dan Kepala Induksi](https://transformer-circirs.pub/2022/in-context-learning-and-induction-heads/index.html) — pandangan mekanistik tentang apa yang sebenarnya dilakukan kepala.
