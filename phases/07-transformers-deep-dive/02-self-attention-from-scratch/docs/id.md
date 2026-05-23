# Attention Diri dari Awal

> Attention adalah tabel pencarian di mana setiap kata menanyakan "siapa yang penting bagi saya?" - dan mempelajari jawabannya.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 3 (Inti Pembelajaran Mendalam), Fase 5 Lesson 10 (Urutan-ke-Urutan)
**Waktu:** ~90 menit

## Tujuan Pembelajaran

- Menerapkan attention mandiri produk titik berskala dari awal hanya menggunakan NumPy, termasuk proyeksi kueri/kunci/nilai dan jumlah tertimbang softmax
- Membangun layer attention multi-kepala yang memisahkan kepala, menghitung attention paralel, dan menggabungkan hasil
- Lacak bagaimana matrix attention menangkap hubungan token dan jelaskan mengapa penskalaan dengan sqrt(d_k) mencegah saturasi softmax
- Terapkan penyembunyian kausal untuk mengubah attention dua arah menjadi attention autoregresif (gaya dekoder).

## Masalah

RNN memproses urutan satu token pada satu waktu. Pada saat kamu mencapai token 50, informasi dari token 1 telah diperas melalui 50 langkah kompresi. Ketergantungan jangka panjang dihancurkan menjadi keadaan tersembunyi berukuran tetap - hambatan yang tidak dapat diselesaikan sepenuhnya oleh gerbang LSTM mana pun.

Makalah attention Bahdanau tahun 2014 menunjukkan perbaikannya: biarkan decoder melihat kembali setiap posisi encoder dan memutuskan mana yang penting untuk langkah saat ini. Tapi itu masih melesat ke RNN. Makalah "Attention Adalah Yang kamu Butuhkan" tahun 2017 mengajukan pertanyaan yang lebih tajam: bagaimana jika attention adalah *satu-satunya* mekanisme? Tidak ada pengulangan. Tidak ada konvolusi. Hanya attention.

Attention diri memungkinkan setiap posisi dalam suatu urutan memperhatikan setiap posisi lainnya dalam satu langkah paralel. Hal inilah yang menjadikan Transformer cepat, terukur, dan dominan.

## Konsep

### Analogi Pencarian Basis Data

Bayangkan attention sebagai pencarian basis data lunak:

```
Traditional database:
  Query: "capital of France"  -->  exact match  -->  "Paris"

Attention:
  Query: "capital of France"  -->  similarity to ALL keys  -->  weighted blend of ALL values
```

Setiap token menghasilkan tiga vector:
- **Pertanyaan (Q)**: "Apa yang saya cari?"
- **Kunci (K)**: "Apa isi saya?"
- **Nilai (V)**: "Informasi apa yang saya berikan jika terpilih?"

Produk titik antara kueri dan semua kunci menghasilkan skor attention. Skor tinggi berarti "kunci ini cocok dengan kueri saya". Skor tersebut memberi weight pada nilainya. Outputnya adalah jumlah nilai yang tertimbang.

### Perhitungan Q, K, V

Setiap embedding token diproyeksikan melalui tiga matrix weight yang dipelajari:

```
Input embeddings (sequence of n tokens, each d-dimensional):

  X = [x1, x2, x3, ..., xn]       shape: (n, d)

Three weight matrices:

  Wq  shape: (d, dk)
  Wk  shape: (d, dk)
  Wv  shape: (d, dv)

Projections:

  Q = X @ Wq    shape: (n, dk)      each token's query
  K = X @ Wk    shape: (n, dk)      each token's key
  V = X @ Wv    shape: (n, dv)      each token's value
```

Secara visual, untuk satu token:

```
             Wq
  x_i ------[*]------> q_i    "What am I looking for?"
       |
       |     Wk
       +----[*]------> k_i    "What do I contain?"
       |
       |     Wv
       +----[*]------> v_i    "What do I offer?"
```

### Matrix Attention

Setelah kamu memiliki Q, K, V untuk semua token, skor attention membentuk matrix:

```
Scores = Q @ K^T    shape: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- how much q1 attends to each key
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

Each row: one token's attention over the entire sequence
```

### Mengapa Skala?

Perkalian titiknya tumbuh dengan dimension dk. Jika dk = 64, perkalian titik bisa berada di kisaran puluhan, mendorong softmax ke wilayah di mana gradiennya hilang. Cara mengatasinya: bagi dengan sqrt(dk).

```
Scaled scores = (Q @ K^T) / sqrt(dk)
```

Ini menjaga nilai-nilai dalam kisaran di mana softmax menghasilkan gradient yang berguna.

### Softmax Mengubah Skor menjadi Weight

Softmax mengubah skor mentah menjadi distribusi probabilitas di setiap baris:

```
Raw scores for q1:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
Attention weights:   [0.52, 0.09, 0.07, 0.14, 0.08]   (sums to ~1.0)
```

Sekarang setiap token memiliki serangkaian weight yang menyatakan berapa banyak yang harus diperhatikan pada setiap token lainnya.

### Jumlah Nilai Tertimbang

Output akhir untuk setiap token adalah jumlah tertimbang dari semua vector nilai:

```
output_i = sum( attention_weight[i][j] * v_j  for all j )

For token 1:
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### Pipeline Pipa Penuh

```
                    +-------+
  X (input)  ----->|  @ Wq  |-----> Q
                    +-------+
                    +-------+
  X (input)  ----->|  @ Wk  |-----> K
                    +-------+                     +----------+
                    +-------+                     |          |
  X (input)  ----->|  @ Wv  |-----> V ---------->| weighted |----> output
                    +-------+          ^          |   sum    |
                                       |          +----------+
                              +--------+--------+
                              |    softmax      |
                              +---------+-------+
                                        ^
                              +---------+-------+
                              | Q @ K^T / sqrt  |
                              +-----------------+
```

Rumus dalam satu baris:

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

## Build

### Langkah 1: Softmax dari awal

Softmax mengubah logit mentah menjadi probabilitas. Kurangi maks untuk stabilitas numerik.

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### Langkah 2: Meningkatkan attention produk titik

Fungsi inti. Mengambil matrix Q, K, V dan mengembalikan output attention ditambah matrix weight.

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### Langkah 3: Kelas attention diri dengan proyeksi yang dipelajariModul attention mandiri lengkap dengan matrix weight Wq, Wk, Wv yang diinisialisasi dengan penskalaan mirip Xavier.

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### Langkah 4: Jalankan pada sebuah kalimat

Buat embeddings palsu untuk sebuah kalimat dan perhatikan weight attention-nya.

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### Langkah 5: Visualisasikan attention dengan peta panas ASCII

Petakan weight attention pada karakter untuk visual yang cepat.

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## Pakai

`nn.MultiheadAttention` PyTorch melakukan persis seperti yang kami buat, ditambah pemisahan multi-head dan proyeksi output:

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))
```

Perbedaan utamanya: attention multi-kepala menjalankan beberapa fungsi attention secara paralel, masing-masing dengan proyeksi Q, K, V sendiri dengan ukuran dk = d_model / n_heads, lalu menggabungkan hasilnya. Hal ini memungkinkan model menangani berbagai jenis hubungan secara bersamaan.

## Kirim

Lesson ini menghasilkan:
- `outputs/prompt-attention-explainer.md` - prompt untuk menjelaskan attention melalui analogi pencarian database

## Latihan

1. Ubah `scaled_dot_product_attention` untuk menerima matrix mask opsional yang menetapkan posisi tertentu ke tak terhingga negatif sebelum softmax (begitulah cara kerja masking kausal/dekoder)
2. Menerapkan attention multi-head dari awal: pisahkan Q, K, V menjadi `n_heads` potongan, jalankan attention pada masing-masing bagian, gabungkan, dan proyeksikan melalui matrix weight akhir Wo
3. Ambil dua kalimat berbeda dengan panjang yang sama, berikan melalui contoh SelfAttention yang sama, dan bandingkan pola attention-nya. Perubahan apa? Apa yang tetap sama?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|----------------------|
| Kueri (Q) | "Vector pertanyaan" | Proyeksi input yang dipelajari yang mewakili informasi apa yang dicari token ini |
| Kunci (K) | "Vector label" | Proyeksi yang dipelajari yang mewakili informasi apa yang terkandung dalam token ini, dicocokkan dengan kueri |
| Nilai (V) | "Vector konten" | Proyeksi pembelajaran yang membawa informasi aktual yang dikumpulkan berdasarkan skor attention |
| Attention produk titik berskala | "Formula Attention" | softmax(QK^T / sqrt(dk)) @ V - penskalaan mencegah saturasi softmax dalam high-dimensional |
| Attention diri | "Token itu melihat dirinya sendiri dan orang lain" | Attention dimana Q, K, V semuanya berasal dari urutan yang sama, biarkan setiap posisi memperhatikan setiap posisi lainnya |
| Weight attention | "Seberapa fokus" | Distribusi probabilitas atas posisi, dihasilkan oleh softmax atas perkalian titik berskala |
| Attention multi-kepala | "Attention paralel" | Menjalankan beberapa fungsi attention dengan proyeksi berbeda, lalu menggabungkan hasil untuk representasi yang lebih kaya |

## Bacaan Lanjutan

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) - kertas trafo asli
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) - panduan visual terbaik dari arsitektur lengkap
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) - implementasi PyTorch baris demi baris dengan penjelasan
