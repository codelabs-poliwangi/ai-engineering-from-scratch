# Stabilitas Numerik

> Floating point adalah abstraksi yang bocor. Ia akan menggigit kamu selama latihan, dan kamu tidak akan melihatnya datang.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lesson 01-04
**Waktu:** ~120 menit

## Tujuan Pembelajaran

- Menerapkan softmax dan log-sum-exp yang stabil secara numerik menggunakan trik pengurangan maksimal
- Identifikasi overflow, underflow, dan pembatalan bencana dalam perhitungan floating-point
- Verifikasi gradient analitik terhadap gradient numerik menggunakan perbedaan hingga terpusat
- Jelaskan mengapa bfloat16 lebih disukai daripada float16 untuk training dan bagaimana penskalaan loss mencegah aliran bawah gradient

## Masalah

Model kamu berlatih selama tiga jam, lalu kerugiannya menjadi NaN. kamu menambahkan pernyataan cetak. Logitnya baik-baik saja pada langkah 9.000. Pada langkah 9.001 mereka adalah `inf`. Pada langkah 9.002, setiap gradient menjadi `nan` dan training dihentikan.

Atau: model kamu dilatih hingga selesai tetapi akurasinya 2% lebih buruk daripada yang diklaim di kertas. kamu memeriksa semuanya. Pertandingan arsitektur. Hyperparameter cocok. Datanya cocok. Masalah adalah kertas tersebut menggunakan float32 dan kamu menggunakan float16 tanpa skala yang tepat. Tiga puluh dua bit kesalahan pembulatan yang terakumulasi secara diam-diam memakan akurasi kamu.

Atau: kamu menerapkan loss lintas entropi dari awal. Ini bekerja pada log kecil. Ketika logit melebihi 100, ia mengembalikan `inf`. Softmax meluap karena `exp(100)` lebih besar dari yang dapat diwakili oleh float32. Setiap framework ML menangani hal ini dengan trik dua baris. kamu tidak tahu ada triknya.

Stabilitas numerik bukanlah masalah teoritis. Inilah perbedaan antara latihan lari yang berhasil dan latihan yang gagal secara diam-diam. Setiap bug ML serius yang akan kamu debug pada akhirnya akan menjadi floating point.

## Konsep

### IEEE 754: Bagaimana Komputer Menyimpan Bilangan Nyata

Komputer menyimpan bilangan real sebagai nilai floating point mengikuti standar IEEE 754. Pelampung mempunyai tiga bagian: bit tanda, eksponen, dan mantissa (signifikansi).

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

Mantissa menentukan presisi (berapa angka penting). Eksponen menentukan rentang (seberapa besar atau kecil suatu bilangan).

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 memberi kamu presisi sekitar 7 digit desimal. Artinya ia dapat membedakan 1,0000001 dan 1,0000002, namun tidak dapat membedakan 1,00000001 dan 1,00000002. Setelah 7 digit, semuanya membulat.

float16 memberi kamu sekitar 3 digit. Jumlah terbesar yang dapat diwakilinya adalah 65.504. Jumlah ini sangat kecil untuk ML yang logit, gradient, dan aktivasinya sering melebihi angka ini.

bfloat16 adalah jawaban Google untuk masalah jangkauan float16. Ia memiliki eksponen 8-bit yang sama dengan float32 (kisaran yang sama, hingga 3.4e38) tetapi hanya 7 bit mantissa (kurang presisi dibandingkan float16). Untuk melatih neural network, jangkauan lebih penting daripada presisi, jadi bfloat16 biasanya menang.

### Mengapa 0,1 + 0,2 != 0,3

Angka 0,1 tidak dapat direpresentasikan secara tepat dalam floating point biner. Pada basis 2, ini adalah pecahan berulang:

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 memotongnya menjadi 23 bit mantissa. Nilai yang disimpan kira-kira 0,100000001490116. Demikian pula, 0,2 disimpan sebagai sekitar 0,200000002980232. Jumlahnya adalah 0,300000004470348, bukan 0,3.

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

Ini penting bagi ML karena:1. Perbandingan loss seperti `if loss < threshold` dapat memberikan jawaban yang salah
2. Mengumpulkan banyak nilai kecil (pembaruan gradient dalam ribuan langkah) menyimpang dari jumlah sebenarnya
3. Tes checksum dan reprodusibilitas gagal jika kamu membandingkan float dengan `==`

Cara mengatasinya: jangan pernah membandingkan float dengan `==`. Gunakan `abs(a - b) < epsilon` atau `math.isclose()`.

### Pembatalan Bencana

Saat kamu mengurangkan dua bilangan floating point yang hampir sama, digit signifikannya akan hilang dan kamu akan mendapatkan suara pembulatan yang dipromosikan ke digit terdepan.

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

Itu adalah kesalahan relatif 19% dari satu pengurangan. Di ML, ini terjadi setiap kali kamu:

- Hitung varians data dengan mean yang besar: `E[x^2] - E[x]^2` ketika E[x] besar
- Kurangi probabilitas log yang hampir sama
- Hitung gradient perbedaan hingga dengan epsilon yang terlalu kecil

Cara mengatasinya: susun ulang rumus untuk menghindari pengurangan bilangan yang besar dan hampir sama. Untuk varians, gunakan algoritma Welford atau pusatkan datanya terlebih dahulu. Untuk probabilitas log, kerjakan seluruh ruang log.

### Meluap dan Meluap

Overflow terjadi ketika suatu hasil terlalu besar untuk diwakili. Underflow terjadi jika terlalu kecil (mendekati nol daripada bilangan positif terkecil yang dapat direpresentasikan).

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

Fungsi `exp()` adalah sumber utama overflow di ML:

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

Fungsi `log()` mengarah ke arah lain:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

Dalam ML, `exp()` muncul dalam perhitungan softmax, sigmoid, dan probabilitas. `log()` muncul dalam cross-entropy, log-likelihood, dan KL divergence. Kombinasi `log(exp(x))` adalah ladang ranjau tanpa trik yang tepat.

### Trik Log-Sum-Exp

Menghitung `log(sum(exp(x_i)))` secara numerik berbahaya. Jika ada `x_i` yang berukuran besar, `exp(x_i)` akan meluap. Jika semua `x_i` sangat negatif, setiap `exp(x_i)` underflow ke nol dan `log(0)` adalah `-inf`.

Caranya: kurangi nilai maksimalnya sebelum melakukan eksponensial.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

Mengapa ini berhasil: setelah mengurangkan `max(x)`, eksponen terbesarnya adalah `exp(0) = 1`. Tidak ada luapan yang mungkin terjadi. Setidaknya satu suku dalam penjumlahannya adalah 1, jadi jumlahnya setidaknya 1, dan `log(1) = 0`. Tidak ada aliran bawah ke `-inf` yang mungkin dilakukan.

Bukti:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

Setel `c = max(x)` dan luapan dihilangkan.

Trik ini muncul dimana-mana di ML:
- Normalisasi Softmax
- Perhitungan loss lintas entropi
- Penjumlahan log-probabilitas dalam model urutan
- Campuran Gaussian
- Inference variasi

### Mengapa Softmax Membutuhkan Trik Pengurangan Maks

Softmax mengubah logit menjadi probabilitas:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Tanpa triknya, logit [100, 101, 102] menyebabkan overflow:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

Dengan cara ini kurangi maks(x) = 102:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

Kemungkinannya sama. Perhitungannya aman. Ini bukan optimization. Ini adalah persyaratan untuk kebenaran.

### NaN dan Inf: Deteksi dan Pencegahan

`nan` (Bukan Angka) dan `inf` (tak terhingga) menyebar secara viral melalui komputasi. Satu `nan` dalam pembaruan gradient menghasilkan weight `nan`, yang menghasilkan setiap output berikutnya `nan`. Training mati dalam satu langkah.Bagaimana `inf` muncul:
- `exp()` dengan bilangan positif yang besar
- Pembagian dengan nol: `1.0 / 0.0`
- `float32` akumulasi yang melimpah

Bagaimana `nan` muncul:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` dari bilangan negatif
- `log()` dari bilangan negatif
- Aritmatika apa pun yang melibatkan `nan` yang ada

Deteksi:

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

Strategi pencegahan:

1. Jepit input ke `exp()`: `exp(clamp(x, -80, 80))`
2. Tambahkan epsilon ke penyebut: `x / (y + 1e-8)`
3. Tambahkan epsilon di dalam `log()`: `log(x + 1e-8)`
4. Gunakan implementasi yang stabil (log-sum-exp, stable softmax)
5. Kliping gradient untuk mencegah ledakan berat
6. Periksa `nan`/`inf` setelah setiap forward pass selama debugging

### Pemeriksaan Gradient Numerik

Gradient analitik (dari backpropagation) dapat memiliki bug. Pemeriksaan gradient numerik memverifikasinya dengan menghitung gradient dengan perbedaan terbatas.

Rumus selisih terpusat:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

Ini O(h^2) akurat, jauh lebih baik daripada selisih maju `(f(x+h) - f(x)) / h` yang hanya O(h).

Memilih h: terlalu besar dan perkiraannya salah. Pembatalan yang terlalu kecil dan membawa bencana akan menghancurkan jawabannya. `h = 1e-5` hingga `1e-7` adalah tipikal.

Pemeriksaannya: hitung perbedaan relatif antara gradient analitik dan numerik.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

Aturan praktis:
- relative_error < 1e-7: sempurna, gradient benar
- relative_error < 1e-5: dapat diterima, mungkin benar
- relative_error > 1e-3: ada yang salah
- relative_error > 1: gradient sepenuhnya salah

Selalu periksa gradient saat mengimplementasikan layer baru atau loss function. PyTorch menyediakan `torch.autograd.gradcheck()` untuk ini.

### Training Presisi Campuran

GPU modern memiliki perangkat keras khusus (Tensor Cores) yang menghitung perkalian matrix float16 2-8x lebih cepat dibandingkan float32. Training presisi campuran memanfaatkan ini:

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

Masalah dengan training float16 murni: gradient seringkali sangat kecil (1e-8 atau lebih kecil). Float16 mengalirkan apa pun di bawah ~6e-8 ke nol. Model kamu berhenti belajar karena semua pembaruan gradient adalah nol.

Cara mengatasinya adalah penskalaan loss:

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

Penskalaan loss dinamis menyesuaikan faktor skala secara otomatis. Mulailah dengan nilai yang besar (65536). Jika gradient meluap ke `inf`, bagilah menjadi dua. Jika N langkah berhasil tanpa luapan, gandakan.

### bfloat16 vs float16: Mengapa bfloat16 Menang untuk Training

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 memiliki presisi lebih tinggi (10 bit mantissa vs 7) tetapi jangkauannya terbatas (maks ~65.504). bfloat16 memiliki presisi yang lebih rendah tetapi rentangnya sama dengan float32 (maks ~3.4e38).

Untuk melatih neural network:

- Activation dan logit secara teratur melebihi 65.504 selama lonjakan training. float16 meluap; bfloat16 menanganinya.
- Penskalaan loss diperlukan dengan float16 tetapi biasanya tidak diperlukan dengan bfloat16 karena jangkauannya mencakup spektrum besaran gradient.
- bfloat16 adalah pemotongan sederhana dari float32: jatuhkan 16 bit terbawah mantissa. Konversi itu sepele dan tidak merugikan dalam eksponen.

float16 lebih disukai untuk inference di mana nilai dibatasi dan presisi lebih penting. bfloat16 lebih disukai untuk training yang mengutamakan jangkauan. Inilah sebabnya mengapa TPU dan GPU NVIDIA modern (A100, H100) memiliki dukungan asli bfloat16.

### Kliping GradienMeledaknya gradient terjadi ketika gradient tumbuh secara eksponensial melalui banyak layer (umumnya terjadi pada RNN, jaringan dalam, dan Transformer). Satu gradient besar dapat merusak semua weight dalam satu langkah.

Dua jenis kliping:

**Klip berdasarkan nilai:** menjepit setiap elemen gradient secara terpisah.

```
grad = clamp(grad, -max_val, max_val)
```

Sederhana namun dapat mengubah arah vector gradient.

**Klip menurut norm:** menskalakan seluruh vector gradient sehingga normanya tidak melebihi ambang batas.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

Mempertahankan arah gradient. Inilah yang `torch.nn.utils.clip_grad_norm_()` lakukan. Ini adalah pilihan standar.

Nilai umum: `max_norm=1.0` untuk Transformer, `max_norm=0.5` untuk RL, `max_norm=5.0` untuk jaringan yang lebih sederhana.

Kliping gradient bukanlah peretasan. Ini adalah mekanisme keamanan. Tanpanya, satu kumpulan outlier dapat menghasilkan gradient yang cukup besar sehingga merusak training berminggu-minggu.

### Layer Normalisasi sebagai Penstabil Numerik

Normalisasi batch, normalisasi layer, dan normalisasi RMS biasanya disajikan sebagai pengatur yang membantu konvergensi training. Mereka juga merupakan penstabil numerik.

Tanpa normalisasi, activation dapat bertambah atau berkurang secara eksponensial melalui layer:

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

Normalisasi memusatkan kembali dan mengubah skala activation di setiap layer:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon` (biasanya 1e-5) mencegah pembagian dengan nol ketika semua activation identik. Parameter yang dipelajari `gamma` dan `beta` memungkinkan jaringan memulihkan skala apa pun yang diperlukan.

Hal ini menjaga nilai-nilai dalam rentang yang aman secara numerik di seluruh jaringan, mencegah luapan pada lintasan maju dan ledakan gradient pada lintasan mundur.

### Bug Numerik ML Umum

**Bug: Loss adalah NaN setelah beberapa epoch.**
Penyebab: logit bertambah besar, softmax meluap. Atau learning rate terlalu tinggi dan bobotnya berbeda.
Cara mengatasinya: gunakan softmax stabil (pengurangan maksimal), kurangi learning rate, tambahkan kliping gradient.

**Bug: Loss tertahan di log(num_classes).**
Penyebab: output model memiliki probabilitas yang hampir seragam. Seringkali berarti gradient menghilang atau model tidak belajar sama sekali.
Cara mengatasinya: periksa apakah label data sudah benar, verifikasi loss function, periksa ReLU yang mati.

**Bug: Akurasi validasi 1-3% lebih rendah dari yang diharapkan.**
Penyebab: presisi tercampur tanpa penskalaan loss yang tepat. Aliran bawah gradient secara diam-diam menghilangkan pembaruan kecil.
Perbaiki: aktifkan penskalaan loss dinamis, atau beralih ke bfloat16.

**Bug: Norm gradient adalah 0,0 untuk beberapa layer.**
Penyebab: neuron ReLU mati (semua input negatif), atau float16 underflow.
Cara mengatasinya: gunakan LeakyReLU atau GELU, gunakan penskalaan gradient, periksa inisialisasi weight.

**Bug: Model berfungsi pada satu GPU tetapi memberikan hasil berbeda pada GPU lain.**
Penyebab: tatanan akumulasi floating point non-deterministik. Pengurangan paralel GPU berjumlah dalam urutan berbeda pada perangkat keras berbeda, dan penambahan floating point tidak bersifat asosiatif.
Perbaiki: terima perbedaan kecil (1e-6), atau setel `torch.use_deterministic_algorithms(True)` dan terima penalti kecepatan.

**Bug: `exp()` mengembalikan `inf` dalam perhitungan loss.**
Penyebab: logit mentah diteruskan ke `exp()` tanpa trik pengurangan maksimal.
Perbaiki: gunakan `torch.nn.functional.log_softmax()` yang mengimplementasikan log-sum-exp secara internal.

**Bug: Training menyimpang setelah beralih dari float32 ke float16.**
Penyebab: float16 tidak dapat mewakili besaran gradient di bawah 6e-8 atau activation di atas 65.504.
Cara mengatasinya: gunakan presisi campuran dengan penskalaan loss (AMP), atau gunakan bfloat16 sebagai gantinya.

## Build### Langkah 1: Tunjukkan batas presisi floating point

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Langkah 2: Terapkan softmax yang naif vs stabil

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Langkah 3: Terapkan log-sum-exp yang stabil

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Langkah 4: Terapkan entropi silang yang stabil

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Langkah 5: Pemeriksaan gradient

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Pakai

### Simulasi presisi campuran

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Kliping gradient

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### Deteksi NaN/Inf

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

Lihat `code/numerical.py` untuk implementasi lengkap dengan semua kasus edge yang didemonstrasikan.

## Kirim

Lesson ini menghasilkan:
- `code/numerical.py` dengan softmax stabil, log-sum-exp, cross-entropy, pemeriksaan gradient, dan simulasi presisi campuran
- `outputs/prompt-numerical-debugger.md` untuk mendiagnosis masalah NaN/Inf dan numerik dalam training

Implementasi stabil ini muncul kembali di Fase 3 ketika membangun loop training dan di Fase 4 ketika menerapkan mekanisme attention.

## Latihan

1. **Pembatalan bencana.** Hitung varians [1000000.0, 1000001.0, 1000002.0] menggunakan rumus naif `E[x^2] - E[x]^2` di float32. Kemudian hitung menggunakan algoritma online Welford. Bandingkan kesalahan dengan varian sebenarnya (0,6667).

2. **Perburuan presisi.** Temukan nilai float32 positif terkecil `x` sehingga `1.0 + x == 1.0` dengan Python. Ini adalah mesin epsilon. Verifikasi apakah cocok dengan `numpy.finfo(numpy.float32).eps`.

3. **Kasus tepi log-sum-exp.** Uji fungsi `logsumexp_stable` kamu dengan: (a) semua nilai sama, (b) satu nilai jauh lebih besar daripada nilai lainnya, (c) semua nilai sangat negatif (-1000). Verifikasi bahwa itu memberikan hasil yang benar ketika versi naif gagal.

4. **Pemeriksaan gradient layer jaringan neural.** Mengimplementasikan satu layer linier `y = Wx + b` dan proses backward pass analitisnya. Gunakan `numerical_gradient` untuk memverifikasi kebenaran matrix weight 3x2.

5. **Eksperimen penskalaan loss.** Simulasikan training dengan float16: buat gradient acak dalam rentang [1e-9, 1e-3], konversikan ke float16, dan ukur pecahan yang menjadi nol. Kemudian terapkan penskalaan loss (kalikan dengan 1024), konversikan ke float16, turunkan skalanya, dan ukur kembali pecahan nolnya.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|----------------------|
| IEEE 754 | "Standar pelampung" | Standar internasional yang mendefinisikan format floating point biner, aturan pembulatan, dan nilai khusus (inf, nan). Setiap CPU dan GPU modern mengimplementasikannya. |
| Mesin epsilon | "Batas presisi" | Nilai terkecil e sehingga 1.0 + e != 1.0 dalam format float tertentu. Untuk float32, ini sekitar 1.19e-7. |
| Pembatalan bencana | "Loss presisi dari pengurangan" | Saat mengurangkan bilangan floating point yang hampir sama, angka penting akan hilang dan gangguan pembulatan mendominasi hasilnya. |
| Meluap | "Jumlahnya terlalu besar" | Suatu hasil melebihi nilai maksimum yang dapat direpresentasikan dan menjadi inf. exp(89) meluap float32. |
| Aliran bawah | "Jumlahnya terlalu kecil" | Hasilnya mendekati nol daripada bilangan positif terkecil yang dapat direpresentasikan dan menjadi 0,0. exp(-104) aliran bawah float32. |
| Trik log-sum-exp | "Kurangi maksnya dulu" | Menghitung log(sum(exp(x))) dengan memfaktorkan exp(max(x)) untuk mencegah overflow dan underflow. Digunakan dalam matematika softmax, cross-entropy, dan log-probability. |
| Softmax stabil | "Softmax yang tidak meledak" | Mengurangi max(logits) sebelum eksponensial. Hasil yang identik secara numerik, tidak ada kemungkinan meluap. |
| Pemeriksaan gradient | "Verifikasi backprop kamu" | Membandingkan gradient analitik dari backpropagation dengan gradient numerik dari perbedaan terbatas untuk menangkap bug implementasi. |
| Presisi campuran | "Float16 maju, float32 mundur" | Menggunakan pelampung dengan presisi lebih rendah untuk operasi yang kritis terhadap kecepatan dan pelampung dengan presisi lebih tinggi untuk operasi yang sensitif secara numerik. Percepatan tipikal adalah 2-3x. |
| Skala loss | "Mencegah aliran bawah gradient" | Mengalikan loss dengan konstanta besar sebelum backprop sehingga gradient tetap berada dalam rentang terwakili float16, lalu membaginya dengan konstanta yang sama sebelum weight diperbarui. |
| bfloat16 | "Titik mengambang otak" | Format 16-bit Google dengan 8 bit eksponen (rentang yang sama dengan float32) dan 7 bit mantissa (kurang presisi dibandingkan float16). Lebih disukai untuk training. |
| Kliping gradient | "Batasi norm gradient" | Menskalakan vector gradient sehingga normanya tidak melebihi ambang batas. Mencegah ledakan gradient agar tidak merusak weight. |
| tidak | "Bukan Angka" | Nilai float khusus dari operasi yang tidak ditentukan (0/0, inf-inf, sqrt(-1)). Menyebar melalui semua aritmatika berikutnya. |
| Informasi | "Tak terbatas" | Nilai float khusus dari overflow atau pembagian dengan nol. Dapat digabungkan untuk menghasilkan NaN (inf - inf, inf * 0). |
| Gradient numerik | "Turunan kekerasan" | Mendekati suatu turunan dengan mengevaluasi f(x+h) dan f(x-h) dan membaginya dengan 2h. Lambat tapi dapat diandalkan untuk verifikasi. |

## Bacaan Lanjutan

- [Yang Harus Diketahui Setiap Ilmuwan Komputer Tentang Aritmatika Titik Mengambang (Goldberg 1991)](https://docs.Oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- referensi definitif, padat namun lengkap
- [Training Presisi Campuran (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- makalah NVIDIA yang memperkenalkan penskalaan loss untuk training float16
- [AMP: Presisi Campuran Otomatis (dokumen PyTorch)](https://pytorch.org/docs/stable/amp.html) -- panduan praktis untuk presisi campuran di PyTorch
- [format bfloat16 (dokumen Google Cloud TPU)](https://cloud.google.com/tpu/docs/bfloat16) -- alasan Google memilih format ini untuk TPU
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- algoritma untuk mengurangi kesalahan pembulatan dalam penjumlahan floating point
