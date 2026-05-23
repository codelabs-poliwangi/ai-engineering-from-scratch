# Probabilitas dan Distribusi

> Probabilitas adalah bahasa yang digunakan AI untuk menyatakan ketidakpastian.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lesson 01-04
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Menerapkan PMF dan PDF dari awal untuk distribusi Bernoulli, kategorikal, Poisson, seragam, dan normal
- Hitung nilai yang diharapkan, varians, dan gunakan Teorema Limit Pusat untuk menjelaskan mengapa Gaussian mendominasi
- Build fungsi softmax dan log-softmax dengan trik stabilitas numerik (kurangi max logit)
- Hitung loss lintas entropi dari logit dan hubungkan dengan kemungkinan log negatif

## Masalah

Pengklasifikasi menghasilkan `[0.03, 0.91, 0.06]`. Model bahasa memilih kata berikutnya dari 50.000 kandidat. Model difusi menghasilkan gambar dengan mengambil sample dari distribusi yang dipelajari. Semua ini adalah kemungkinan dalam tindakan.

Setiap prediksi yang dibuat model adalah distribusi probabilitas. Setiap loss function mengukur seberapa jauh distribusi prediksi dari distribusi sebenarnya. Setiap langkah training menyesuaikan parameter untuk membuat satu distribusi terlihat lebih mirip dengan distribusi lainnya. Tanpa probabilitas, kamu tidak dapat membaca satu makalah ML pun, men-debug satu model, atau memahami mengapa loss training kamu sebesar NaN.

## Konsep

### Peristiwa, Ruang Sample, dan Probabilitas

Ruang sample S adalah himpunan semua kemungkinan hasil. Suatu peristiwa adalah bagian dari ruang sample. Probabilitas memetakan peristiwa ke angka antara 0 dan 1.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

Tiga aksioma mendefinisikan semua probabilitas:
1. P(A) >= 0 untuk sembarang kejadian A
2. P(S) = 1 (sesuatu selalu terjadi)
3. P(A atau B) = P(A) + P(B) ketika A dan B tidak dapat terjadi keduanya

Segala sesuatu yang lain (teorema Bayes, ekspektasi, distribusi) mengikuti ketiga aturan ini.

### Probabilitas dan Independensi Bersyarat

P(A|B) adalah peluang A jika B terjadi.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

Dua peristiwa bersifat independen ketika mengetahui satu peristiwa tidak memberi tahu kamu apa pun tentang peristiwa lainnya:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

Pembalikan koin bersifat independen. Menggambar kartu tanpa pengembalian tidaklah demikian.

### Fungsi Massa Probabilitas vs Fungsi Kepadatan Probabilitas

Variabel acak diskrit memiliki fungsi massa probabilitas (PMF). Setiap hasil memiliki probabilitas tertentu yang dapat kamu baca secara langsung.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

Variabel acak kontinu memiliki fungsi kepadatan probabilitas (PDF). Kepadatan pada satu titik bukanlah suatu probabilitas. Probabilitas berasal dari pengintegrasian kepadatan dalam suatu interval.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

Perbedaan ini penting dalam ML. Output klasifikasi adalah PMF (pilihan terpisah). Ruang laten VAE menggunakan PDF (berkelanjutan).

### Distribusi Umum

**Bernoulli:** satu percobaan, dua hasil. Model klasifikasi biner.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Kategoris:** satu percobaan, k hasil. Model klasifikasi kelas jamak (output softmax).

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Seragam:** semua hasil memiliki kemungkinan yang sama. Digunakan untuk inisialisasi acak.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian):** kurva lonceng. Diparameterisasi dengan mean (mu) dan varians (sigma^2).

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:** menghitung kejadian langka dalam interval tetap. Memodelkan tingkat kejadian.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### Nilai dan Varians yang Diharapkan

Nilai yang diharapkan adalah hasil rata-rata tertimbang.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

Ukuran varians menyebar di sekitar mean.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

Dalam ML, nilai yang diharapkan muncul sebagai loss function (rata-rata loss sepanjang distribusi data). Varians memberi tahu kamu tentang stabilitas model. Varians gradient yang tinggi berarti training yang berisik.### Distribusi Gabungan dan Marginal

Distribusi gabungan P(X, Y) menggambarkan dua variabel acak secara bersamaan.

Contoh gabungan PMF (X = cuaca, Y = payung):

| | Y=0 (tanpa payung) | Y=1 (payung) | P(X) Marginal |
|---|---|---|---|
| X=0 (matahari) | 0,40 | 0,10 | P(X=0) = 0,50 |
| X=1 (hujan) | 0,05 | 0,45 | P(X=1) = 0,50 |
| **P Marginal(Y)** | P(Y=0) = 0,45 | P(Y=1) = 0,55 | 1,00 |

Distribusi marjinal merangkum variabel lainnya:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

Total baris dan kolom pada tabel di atas adalah marginnya.

### Mengapa Distribusi Normal Muncul Dimana-mana

Teorema Batas Pusat: jumlah (atau rata-rata) dari banyak variabel acak independen menyatu ke distribusi normal, terlepas dari distribusi aslinya.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

Inilah alasannya:
- Kesalahan pengukuran kira-kira normal (banyak sumber independen kecil)
- Inisialisasi weight pada neural network menggunakan distribusi normal
- Kebisingan gradient dalam SGD kira-kira normal (jumlah dari banyak gradient sample)
- Distribusi normal adalah distribusi entropi maksimum untuk mean dan varians tertentu

### Catat Probabilitas

Probabilitas mentah menyebabkan masalah numerik. Mengalikan banyak probabilitas kecil bersama-sama dengan cepat menghasilkan nol.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

Probabilitas log memperbaikinya. Perkalian menjadi penjumlahan.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

Aturan:
- catatan(a * b) = catatan(a) + catatan(b)
- probabilitas log selalu <= 0 (karena 0 < P <= 1)
- Lebih negatif = kecil kemungkinannya
- Loss lintas entropi adalah probabilitas log negatif dari kelas yang benar

### Softmax sebagai Distribusi Probabilitas

Jaringan saraf mengeluarkan skor mentah (logit). Softmax mengubahnya menjadi distribusi probabilitas yang valid.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Trik softmax: kurangi max logit sebelum melakukan eksponensial untuk mencegah overflow.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax menggabungkan softmax dan log untuk stabilitas numerik. PyTorch menggunakan ini secara internal untuk kehilangan lintas entropi.

### Pengambilan sample

Sampling berarti mengambil nilai acak dari suatu distribusi. Dalam ML:
- Dropout secara acak mengambil sample neuron mana yang menjadi nol
- Augmentasi data mengambil sample transformasi acak
- Model bahasa mengambil sample token berikutnya dari distribusi yang diprediksi
- Model difusi mengambil sample kebisingan dan secara bertahap menghilangkan kebisingan

Pengambilan sample dari distribusi arbitrer memerlukan teknik seperti pengambilan sample transformasi terbalik, pengambilan sample penolakan, atau trik reparameterisasi (digunakan dalam VAE).

## Build

### Langkah 1: Dasar-dasar probabilitas

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Langkah 2: PMF dan PDF dari awal

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Langkah 3: Nilai dan varians yang diharapkan

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Langkah 4: Pengambilan sample dari distribusi

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Langkah 5: Softmax dan mencatat probabilitas

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Langkah 6: Demonstrasi Teorema Limit Pusat

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Langkah 7: Visualisasi

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Implementasi penuh dengan semua visualisasi ada di `code/probability.py`.

## Pakai

Dengan NumPy dan SciPy, semua hal di atas adalah satu kalimat:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

kamu membuatnya dari awal. Sekarang kamu tahu apa yang dilakukan panggilan perpustakaan.

## Latihan

1. Menerapkan pengambilan sample transformasi terbalik untuk distribusi eksponensial. Verifikasi dengan mengambil sample 10.000 nilai dan membandingkan histogram dengan PDF sebenarnya.

2. Buatlah tabel distribusi gabungan untuk dua dadu yang dimuati. Hitung distribusi marjinal dan periksa apakah dadunya independen.3. Hitung loss lintas entropi untuk pengklasifikasi 5 kelas yang menampilkan logit `[2.0, 0.5, -1.0, 3.0, 0.1]` ketika kelas yang benar adalah indeks 3. Kemudian verifikasi jawaban kamu dengan `nn.CrossEntropyLoss` PyTorch.

4. Tulis fungsi yang mengambil daftar probabilitas log dan mengembalikan urutan yang paling mungkin, total probabilitas log, dan probabilitas mentah yang setara. Ujilah dengan kalimat 50 kata yang setiap kata memiliki probabilitas 0,01.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|----------------------|
| Ruang sample | "Semua kemungkinan" | Himpunan S dari setiap kemungkinan hasil suatu percobaan |
| PMF | "Fungsi probabilitas" | Suatu fungsi yang memberikan probabilitas pasti dari setiap hasil diskrit, yang dijumlahkan menjadi 1 |
| PDF | "Kurva probabilitas" | Fungsi kepadatan untuk variabel kontinu. Integrasikan dalam suatu interval untuk mendapatkan probabilitas |
| Probabilitas bersyarat | "Probabilitas diberikan sesuatu" | P(A\|B) = P(A dan B) / P(B). Landasan Pemikiran Bayesian dan Teorema Bayes |
| Kemerdekaan | "Mereka tidak saling mempengaruhi" | P(A dan B) = P(A) * P(B). Mengetahui satu peristiwa tidak memberi tahu kamu apa pun tentang peristiwa lainnya |
| Nilai yang diharapkan | "Rata-rata" | Jumlah semua hasil yang diberi weight probabilitas. Loss function adalah nilai yang diharapkan |
| Varians | "Betapa tersebarnya" | Deviasi kuadrat yang diharapkan dari mean. Varians tinggi = perkiraan yang berisik dan tidak stabil |
| Distribusi normal | "Kurva lonceng" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). Muncul dimana-mana karena CLT |
| Teorema Limit Pusat | "Rata-rata menjadi normal" | Rata-rata dari banyak sample independen menyatu ke distribusi normal tanpa memandang sumber |
| Distribusi bersama | "Dua variabel bersama" | P(X,Y) menggambarkan probabilitas setiap kombinasi hasil X dan Y |
| Distribusi marjinal | "Jumlah variabel lainnya" | P(X) = jumlah_y P(X, Y). Memulihkan distribusi satu variabel dari gabungan |
| Log probabilitas | "Log kemungkinan" | catatan P(x). Mengubah produk menjadi jumlah, mencegah kekurangan numerik dalam urutan yang panjang |
| Softmax | "Ubah skor menjadi probabilitas" | softmax(z_i) = exp(z_i) / jumlah(exp(z_j)). Memetakan logit bernilai nyata ke distribusi probabilitas yang valid |
| Entropi silang | "Loss function" | -jumlah(p_benar * log(p_prediksi)). Mengukur seberapa berbedanya dua distribusi. Lebih rendah lebih baik |
| Logit | "Output model mentah" | Skor tidak dinormalisasi sebelum softmax. Dinamakan berdasarkan fungsi logistik |
| Pengambilan sample | "Menggambar nilai acak" | Menghasilkan nilai berdasarkan distribusi probabilitas. Bagaimana model menghasilkan output |

## Bacaan Lanjutan

- [3Blue1Brown: Tapi apa itu Teorema Batas Pusat?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - bukti visual mengapa rata-rata menjadi normal
- [Tinjauan Probabilitas Stanford CS229](https://cs229.stanford.edu/section/cs229-prob.pdf) - referensi ringkas yang mencakup semuanya di sini dan banyak lagi
- [Trik Log-Sum-Exp](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - mengapa stabilitas numerik penting dan cara mencapainya
