# Pengkodean Posisi — Sinusoidal, RoPE, ALiBi

> Attention bersifat invarian permutasi. "Kucing duduk di atas matras" dan "kucing di atas sat kucing" menghasilkan output yang sama tanpa sinyal posisi. Tiga algoritme memperbaikinya — masing-masing dengan taruhan berbeda tentang arti "posisi".

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 7 · 02 (Attention Diri Sendiri), Fase 7 · 03 (Attention Multi-Kepala)
**Waktu:** ~45 menit

## Masalah

Attention terhadap produk titik yang berskala tidak bersifat order-blind. Matrix attention `softmax(Q K^T / √d) V` dihitung dari kesamaan berpasangan. Kocok baris `X`, kocok baris output dengan cara yang sama. Tidak ada attention dari dalam yang peduli dengan posisi.

Itu bukan bug dalam model segudang kata. Untuk bahasa, code, audio, video - segala sesuatu yang keteraturannya mempunyai makna - itu berakibat fatal.

Cara mengatasinya adalah dengan memasukkan posisi ke dalam embeddings. Tiga era jawaban:

1. **sinusoidal absolut** (Vaswani 2017). Tambahkan `sin/cos` posisi ke embedding. Sederhana, tidak dapat dipelajari, mengekstrapolasi dengan buruk melampaui jangka waktu yang dilatih.
2. **Tali — Embedding Posisi Putar** (Su 2021). Putar vector Q dan K dengan sudut yang sebanding dengan posisinya. Mengkodekan posisi *relatif* langsung dalam perkalian titik. Dominan pada tahun 2026.
3. **ALiBi — Attention dengan Bias Linier** (Pers 2022). Lewati embedding seluruhnya; tambahkan penalti linier per kepala ke skor attention berdasarkan distance. Ekstrapolasi panjang yang sangat baik.

Pada tahun 2026, pada dasarnya setiap model terbuka perbatasan menggunakan RoPE: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. Sejumlah model konteks panjang menggunakan ALiBi atau varian modernnya. Sinusoidal absolut bersifat historis.

## Konsep

![Rotasi absolut sinusoidal vs RoPE vs bias distance ALiBi](../assets/positional-encoding.svg)

### Sinusoidal mutlak

Pra-hitung matrix tetap `PE` dengan bentuk `(max_len, d_model)`:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

Kemudian `X' = X + PE[:N]` sebelum attention. Setiap dimension merupakan sinusoidal dengan frekuensi berbeda. Model belajar membaca posisi dari pola fase. Gagal melampaui `max_len`: tidak ada yang memberi tahu model apa yang terjadi pada posisi 2048 ketika model hanya melihat posisi 0–2047.

### Tali

Putar vector Q dan K (bukan embeddings). Untuk sepasang dimension `(2i, 2i+1)`:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

Terapkan rotasi yang sama pada tombol dengan posisi `pos_k`. Produk titik `q'_m · k'_n` menjadi fungsi dari `(m - n)` saja. Artinya: **skor attention hanya bergantung pada distance relatif**, meskipun rotasinya tidak sesuai dengan posisi absolut. Trik yang indah.

Memperluas RoPE: `base` dapat diskalakan (sadar NTK, YaRN, LongRoPE) untuk diekstrapolasi ke konteks yang lebih panjang tanpa training ulang. Llama 3 diperluas dari konteks 8K ke 128K dengan cara ini.

### ALiBi

Lewati trik embedding. Bias skor attention secara langsung:

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

Dimana `m_h` adalah kemiringan khusus kepala (misalnya `1 / 2^(8·h/H)`). Token yang lebih dekat akan ditingkatkan; token jauh mendapat penalti. Tidak ada biaya waktu training. Makalah ini menunjukkan ekstrapolasi panjang mengalahkan sinusoidal dan mencocokkan RoPE dengan panjang aslinya yang dilatih.

### Apa yang harus dipilih pada tahun 2026| Varian | Ekstrapolasi | Biaya training | Digunakan oleh |
|---------|---------------|---------------|---------|
| Sinusoidal mutlak | miskin | gratis | trafo asli, BERT awal |
| Dipelajari mutlak | tidak ada | kecil | GPT-2, GPT-3 |
| Tali | bagus dengan penskalaan | gratis | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| TALI + BENANG | luar biasa | phase menyempurnakan | Qwen2-1M, Llama 3.1 128K |
| AliBi | luar biasa | gratis | MEKAR, MPT, Baichuan |

RoPE menang karena mendapat attention tanpa mengubah arsitektur, mengkodekan posisi relatif, dan hyperparameter `base` memberikan tombol yang bersih untuk penyesuaian konteks panjang.

## Build

### Langkah 1: pengkodean sinusoidal

Lihat `code/main.py`. Perhitungan 4 baris:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

Tambahkan ini ke matrix embedding sebelum layer attention pertama.

### Langkah 2: Tali diterapkan pada Q, K

RoPE beroperasi di tempat pada Q dan K. Untuk setiap pasang dim:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

Penting: terapkan fungsi yang sama ke Q di posisi `m` dan K di posisi `n`. Produk titiknya mengambil faktor `cos((m-n)·θ_i)` pada setiap pasangan koordinat. Attention mempelajari posisi relatif secara gratis.

### Langkah 3: Kemiringan dan bias ALiBi

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

Tambahkan `bias[h]` ke `(seq_len, seq_len)` matrix skor attention kepala `h`, lalu softmax.

### Langkah 4: verifikasi properti distance relatif dari RoPE

Pilih dua vector acak `a, b`. Putar oleh `(pos_a, pos_b)`. Kemudian oleh `(pos_a + k, pos_b + k)`. Kedua perkalian titik harus cocok dengan kesalahan floating-point. Properti tersebut adalah inti dari RoPE — properti ini tidak berubah terhadap offset absolut, yang penting hanyalah kesenjangan relatif.

## Pakai

PyTorch 2.5+ mengirimkan utilitas RoPE di `torch.nn.functional`. Sebagian besar code produksi menggunakan `flash_attn` atau `xformers` di mana RoPE diterapkan di dalam kernel attention.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Trik konteks panjang pada tahun 2026:**

- **Interpolasi sadar NTK.** Ubah skala `base` menjadi `base * (scale_factor)^(d/(d-2))` saat memperluas dari 4K ke 16K+.
- **YaRN.** Interpolasi lebih cerdas yang menjaga entropi attention pada konteks panjang. Llama 3.1 128K menggunakannya.
- **LongRoPE.** Metode Microsoft tahun 2024 yang menggunakan penelusuran evolusioner untuk memilih faktor skala per dimension. Phi-3-Long menggunakannya.
- **Interpolasi posisi + penyesuaian.** Cukup perkecil posisi berdasarkan faktor ekstensi dan sesuaikan untuk 1–5 miliar token. Sangat efektif.

## Kirim

Lihat `outputs/skill-positional-encoding-picker.md`. Keterampilan memilih strategi pengkodean untuk model baru dengan mempertimbangkan panjang konteks target, kebutuhan ekstrapolasi, dan anggaran training.

## Latihan

1. **Mudah.** Plot matrix sinusoidal `PE` sebagai peta panas untuk `max_len=512, d=128`. Konfirmasikan pola "garis menjadi lebih lebar seiring bertambahnya indeks dimension".
2. **Sedang.** Menerapkan penskalaan RoPE yang sadar NTK. Latih LM kecil pada rangkaian panjang 256, lalu uji pada panjang 1024 dengan dan tanpa penskalaan. Ukur perplexity.
3. **Hard.** Menerapkan ALiBi dan RoPE dalam modul attention yang sama. Latih trafo 4 lapis pada tugas penyalinan dengan urutan panjang 512. Ekstrapolasi ke 2048 pada waktu pengujian. Bandingkan degradasi.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Pengkodean posisi | "Memberitahukan attention tentang ketertiban" | Sinyal apa pun yang ditambahkan ke embedding atau attention yang mengkodekan posisi. |
| Sinusoidal | "Yang asli" | `sin/cos` pada frekuensi geometris yang ditambahkan ke embeddings; tidak melakukan ekstrapolasi. |
| Tali | "Embedding putar" | Putar Q, K dengan sudut yang bergantung pada posisi; produk titik mengkodekan distance relatif. |
| AliBi | "Trik bias linier" | Tambahkan `-m·|i-j|` ke skor attention; tidak perlu embedding, ekstrapolasi yang bagus. |
| dasar | "Tombol Tali" | Penskala frekuensi di RoPE; meningkat untuk memperluas konteks pada inference. |
| Sadar NTK | "Trik penskalaan Tali" | Ubah skala `base` sehingga peredupan frekuensi tinggi tidak diperkecil saat konteks diperluas. |
| BENANG | "Yang mewah" | Interpolasi+ekstrapolasi per dimension yang mempertahankan entropi attention. |
| Ekstrapolasi | "Bekerja melebihi panjang terlatih" | Bisakah skema posisi memberikan output yang benar setelah `max_len` yang terlihat dalam training? |

## Bacaan Lanjutan

- [Vaswani dkk. (2017). Yang kamu Butuhkan Hanya Attention §3.5](https://arxiv.org/abs/1706.03762) — sinusoidal asli.
- [Su dkk. (2021). RoFormer: Transformer yang Ditingkatkan dengan Embedding Posisi Putar](https://arxiv.org/abs/2104.09864) — Kertas Tali.
- [Pers, Smith, Lewis (2021). Latihan Singkat, Tes Panjang: Attention dengan Bias Linier Memungkinkan Ekstrapolasi Panjang Input](https://arxiv.org/abs/2108.12409) — ALiBi.
- [Peng dkk. (2023). YaRN: Ekstensi Jendela Konteks yang Efisien untuk Large Language Model](https://arxiv.org/abs/2309.00071) — penskalaan RoPE yang canggih.
- [Chen dkk. (2023). Memperluas Jendela Konteks Large Language Model melalui Interpolasi Posisi](https://arxiv.org/abs/2306.15595) — Makalah konteks panjang Llama 2 Meta.
- [Ding dkk. (2024). LongRoPE: Memperluas Jendela Konteks LLM Melampaui 2 Juta Token](https://arxiv.org/abs/2402.13753) — metode Microsoft yang digunakan oleh Phi-3-Long dan dikutip di bagian Gunakan Ini.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — implementasi tingkat produksi dari setiap skema penskalaan RoPE (default, linier, dinamis, YaRN, LongRoPE, Llama-3).
