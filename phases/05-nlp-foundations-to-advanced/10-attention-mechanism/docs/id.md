# Mekanisme Attention — Terobosan

> Dekoder berhenti melihat ringkasan terkompresi dan mulai melihat seluruh sumber. Segala sesuatu setelah ini adalah attention plus rekayasa.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 5 · 09 (Model Urutan ke Urutan)
**Waktu:** ~45 menit

## Masalah

Lesson 09 berakhir dengan kegagalan terukur. Encoder-decoder GRU yang dilatih untuk tugas menyalin mainan berubah dari akurasi 89% pada panjang 5 menjadi hampir kebetulan pada panjang 80. Alasannya bersifat struktural, bukan bug training: setiap bit informasi yang diperoleh encoder harus sesuai dengan satu keadaan tersembunyi berukuran tetap, dan decoder tidak pernah melihat yang lain.

Bahdanau, Cho, dan Bengio menerbitkan perbaikan tiga baris pada tahun 2014. Daripada memberikan decoder hanya status encoder akhir, pertahankan setiap status encoder. Pada setiap langkah decoder, hitung rata-rata tertimbang status encoder yang bobotnya menyatakan "berapa banyak yang dibutuhkan decoder untuk melihat posisi encoder `i` saat ini?" Rata-rata tertimbang tersebut adalah konteksnya, dan itu mengubah setiap langkah decoder.

Itulah keseluruhan gagasannya. Transformers memperluasnya. Attention diri menerapkannya pada satu urutan. Attention multi-kepala menjalankannya secara paralel. Namun versi 2014 telah memecahkan hambatan tersebut, dan setelah kamu memilikinya, poros menuju Transformer adalah rekayasa, bukan konseptual.

## Konsep

![Attention Bahdanau: decoder menanyakan semua status encoder](../assets/attention.svg)

Pada setiap langkah dekoder `t`:

1. Gunakan status tersembunyi dekoder sebelumnya `s_{t-1}` sebagai **kueri**.
2. Skor berdasarkan setiap status tersembunyi pembuat enkode `h_1, ..., h_T`. Satu scalar per posisi encoder.
3. Softmax skor untuk mendapatkan weight attention `α_{t,1}, ..., α_{t,T}` yang berjumlah 1.
4. Vector konteks `c_t = Σ α_{t,i} * h_i`. Rata-rata tertimbang status encoder.
5. Decoder mengambil `c_t` ditambah token output sebelumnya, menghasilkan token berikutnya.

Rata-rata tertimbang adalah intinya. Saat dekoder perlu menerjemahkan "Je" menjadi "I", dekoder akan memberi weight pada status encoder di atas "Je" yang tinggi dan yang lainnya rendah. Ketika dibutuhkan "tidak", bobotnya "pas" tinggi. Vector konteks membentuk kembali setiap langkah.

## Bentuk (sesuatu yang menggigit semua orang)

Di sinilah setiap penerapan attention menjadi salah pada kali pertama. Baca perlahan.

| Hal | Bentuk | Catatan |
|-------|-------|-------|
| Status tersembunyi pembuat enkode `H` | `(T_enc, d_h)` | Jika BiLSTM, `d_h = 2 * d_hidden` |
| Status tersembunyi dekoder `s_{t-1}` | `(d_s,)` | Satu vector |
| Skor attention `e_{t,i}` | scalar | Satu per posisi encoder |
| Berat attention `α_{t,i}` | scalar | Setelah softmax keseluruhan `i` |
| Vector konteks `c_t` | `(d_h,)` | Bentuknya sama dengan status encoder |

**Skor Bahdanau (tambahan).** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`.

- `s_{t-1}` berbentuk `(d_s,)`, `h_i` berbentuk `(d_h,)`.
- `W_a` berbentuk `(d_attn, d_s)`. `U_a` memiliki bentuk `(d_attn, d_h)`.
- Jumlahnya di dalam tanh berbentuk `(d_attn,)`.
- `v_α` berbentuk `(d_attn,)`. Produk dalam dengan `v_α` diciutkan menjadi scalar. **Inilah yang dilakukan `v_α`.** Ini bukan keajaiban. Proyeksi inilah yang mengubah vector redup attention menjadi skor scalar.

**Skor Luong (perkalian).** Tiga varian:- `dot`: `e_{t,i} = s_t^T * h_i`. Memerlukan `d_s == d_h`. Kendala yang sulit. Lewati jika encoder kamu dua arah.
- `general`: `e_{t,i} = s_t^T * W * h_i` dengan bentuk `W` `(d_s, d_h)`. Menghapus batasan sama redupnya.
- `concat`: pada dasarnya bentuk Bahdanau. Jarang dipakai karena dua yang pertama lebih murah.

**Satu Bahdanau / Luong gotcha yang layak diberi nama.** Bahdanau menggunakan `s_{t-1}` (status decoder *sebelum* menghasilkan kata saat ini). Luong menggunakan `s_t` (keadaan *setelah*). Mencampurnya menghasilkan gradient yang agak salah dan sangat sulit untuk di-debug. Pilih satu kertas dan patuhi konvensinya.

## Build

### Langkah 1: attention tambahan (Bahdanau).

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

Periksa bentuk kamu dengan tabel di atas. `encoder_states` memiliki bentuk `(T_enc, d_h)`. `projected_enc` memiliki bentuk `(T_enc, d_attn)`. `projected_dec` memiliki bentuk `(d_attn,)` dan siaran. `combined` memiliki bentuk `(T_enc, d_attn)`. `scores` memiliki bentuk `(T_enc,)`. `weights` memiliki bentuk `(T_enc,)`. `context` memiliki bentuk `(d_h,)`. Kirim itu.

### Langkah 2: Luong titik dan umum

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

Masing-masing tiga baris. Inilah sebabnya mengapa makalah Luong mendarat. Akurasi yang sama pada sebagian besar tugas, lebih sedikit code.

### Langkah 3: contoh numerik yang berhasil

Dengan adanya tiga status encoder (kira-kira "cat", "sat", "mat") dan satu status decoder yang paling sejajar dengan yang pertama, distribusi attention terkonsentrasi pada posisi 0. Jika status decoder bergeser agar sejajar dengan yang terakhir, attention berpindah ke posisi 2. Jalur vector konteks.

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

Baris pertama menang. Kemudian pindahkan status decoder lebih dekat ke status encoder ketiga dan lihat bobotnya bergeser. Hanya itu saja. Attention adalah keselarasan yang eksplisit.

### Langkah 4: mengapa ini menjadi jembatan menuju Transformer

Terjemahkan bahasa di atas ke dalam Q/K/V:

- **Kueri** = status dekoder `s_{t-1}`
- **Kunci** = status encoder (nilai yang kami peroleh)
- **Nilai** = status encoder (apa yang kita timbang dan jumlahkan)

Dalam attention klasik, kunci dan nilai adalah hal yang sama. Attention diri memisahkannya: kamu dapat menanyakan urutan terhadap dirinya sendiri, dengan proyeksi pembelajaran berbeda untuk K dan V. Attention multi-kepala menjalankannya secara paralel dengan proyeksi pembelajaran berbeda. Transformer menumpuk seluruh tahapan berkali-kali dan menjatuhkan RNN.

Matematikanya sama. Bentuknya sama. Lompatan pedagogis dari attention Bahdanau ke attention produk titik berskala sebagian besar berupa notasi.

## Pakai

PyTorch dan TensorFlow mengirimkan attention secara langsung.

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

Itu adalah layer attention Transformer. Kumpulan kueri 5 posisi, kumpulan kunci/nilai 10 posisi, masing-masing 128 redup, 8 kepala. `output` adalah kueri baru yang ditambah konteks. `weights` adalah matrix penyelarasan 5x10 yang dapat kamu visualisasikan.

### Saat attention klasik masih penting

- Pedagogi. Versi single-head, single-layer, berbasis RNN membuat setiap konsep terlihat.
- Tugas urutan pada perangkat yang Transformer-nya tidak sesuai.
- Makalah apa pun dari 2014-2017. kamu akan salah membacanya tanpa mengetahui konvensi Bahdanau.
- Analisis keselarasan terperinci di MT. Weight attention mentah adalah alat interpretasi bahkan pada model Transformer, dan membacanya memerlukan pengetahuan tentang weight attention tersebut.

### Perangkap attention-weight-sebagai-penjelasanBobot attention terlihat dapat ditafsirkan. Itu adalah weight yang berjumlah satu di seluruh posisi; kamu dapat mem-plot-nya; tinggi berarti "melihat ini." Para pengulas menyukainya.

Pernyataan-pernyataan tersebut tidak dapat ditafsirkan sebagaimana kelihatannya. Jain dan Wallace (2019) menunjukkan bahwa distribusi attention dapat diubah dan diganti dengan alternatif yang sewenang-wenang tanpa mengubah prediksi model untuk beberapa tugas. Jangan pernah melaporkan weight attention sebagai bukti penalaran tanpa ablasi atau pemeriksaan kontrafaktual.

## Kirim

Simpan sebagai `outputs/prompt-attention-shapes.md`:

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## Latihan

1. **Mudah.** Terapkan masking `softmax` sehingga padding token di encoder mendapatkan attention yang berbobot nol. Uji pada batch dengan urutan panjang variabel.
2. **Sedang.** Tambahkan attention multi-head ke formulir Luong `general`. Bagi `d_h` menjadi grup `n_heads`, jalankan attention per kepala, gabungkan. Verifikasikan kasus single-head cocok dengan implementasi kamu sebelumnya.
3. **Sulit.** Latih encoder-decoder GRU dengan attention Bahdanau pada tugas menyalin mainan dari lesson 09. Akurasi plot vs panjang urutan. Bandingkan dengan garis dasar tanpa attention. kamu akan melihat kesenjangan melebar seiring bertambahnya panjang, memastikan attention menghilangkan hambatan.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Attention | Melihat sesuatu | Rata-rata tertimbang dari urutan nilai, weight dihitung dari kesamaan kunci kueri. |
| Kueri, Kunci, Nilai | QKV | Tiga proyeksi: Q bertanya, K apa yang cocok, V apa yang dikembalikan. |
| Attention tambahan | Bahdanau | Skor umpan maju: `v^T tanh(W q + U k)`. |
| Attention perkalian | Luong titik / umum | Skornya adalah `q^T k` atau `q^T W k`. Lebih murah, akurasi yang sama pada sebagian besar tugas. |
| Matrix penyelarasan | Gambar cantik | Weight attention sebagai kisi `(T_dec, T_enc)`. Bacalah untuk mengetahui apa yang diperhatikan oleh model tersebut. |

## Bacaan Lanjutan

- [Bahdanau, Cho, Bengio (2014). Terjemahan Mesin Neural dengan Belajar Bersama untuk Menyelaraskan dan Menerjemahkan](https://arxiv.org/abs/1409.0473) - makalah.
- [Luong, Pham, Manning (2015). Pendekatan Efektif untuk Terjemahan Mesin Neural Berbasis Attention](https://arxiv.org/abs/1508.04025) — tiga varian skor dan perbandingannya.
- [Jain dan Wallace (2019). Attention bukanlah Penjelasan](https://arxiv.org/abs/1902.10186) — peringatan interpretasi.
- [Selami Pembelajaran Mendalam — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — panduan yang dapat dijalankan dengan PyTorch.
