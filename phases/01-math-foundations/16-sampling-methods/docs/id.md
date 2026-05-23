# Metode Pengambilan Sample

> Pengambilan sample adalah cara AI mengeksplorasi berbagai kemungkinan.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 1, Lesson 06-07 (Probabilitas, Teorema Bayes)
**Waktu:** ~120 menit

## Tujuan Pembelajaran

- Menerapkan CDF terbalik, penolakan, dan pengambilan sample kepentingan dari awal hanya dengan menggunakan nomor acak yang seragam
- Build pengambilan sample suhu, top-k, dan top-p (inti) untuk pembuatan token model bahasa
- Jelaskan trik reparameterisasi dan mengapa hal ini memungkinkan backpropagation melalui pengambilan sample di VAE
- Jalankan MCMC Metropolis-Hastings untuk mengambil sample dari distribusi target yang tidak dinormalisasi

## Masalah

Model bahasa selesai memproses prompt kamu dan menghasilkan vector 50.000 logit. Satu untuk setiap token dalam kosakatanya. Sekarang ia harus memilih satu. Bagaimana?

Jika selalu memilih token dengan probabilitas tertinggi, setiap respons akan identik. deterministik. Membosankan. Jika ia mengambil secara acak dan seragam, hasilnya adalah omong kosong. Jawabannya ada di antara kedua ekstrem ini, dan hal itu dikendalikan oleh pengambilan sample.

Pengambilan sample tidak terbatas pada pembuatan teks. Pembelajaran penguatan memperkirakan gradient kebijakan dengan mengambil contoh lintasan. VAE mempelajari representasi laten dengan mengambil sample dari distribusi yang dipelajari dan melakukan backpropagation melalui keacakan. Model difusi menghasilkan gambar dengan mengambil sample noise dan melakukan denoising secara berulang. Metode Monte Carlo memperkirakan integral yang tidak memiliki solusi bentuk tertutup. Algoritme MCMC mengeksplorasi distribusi posterior berdimensi tinggi yang tidak mungkin dihitung.

Setiap sistem AI generatif adalah sistem pengambilan sample. Strategi pengambilan sample menentukan kualitas, keragaman, dan pengendalian output. Lesson ini membangun setiap metode pengambilan sample utama dari awal, mulai dari bilangan acak seragam dan diakhiri dengan teknik yang mendukung LLM modern dan model generatif.

## Konsep

### Mengapa Pengambilan Sample Penting

Pengambilan sample muncul dalam empat peran mendasar di AI dan machine learning:

**Generasi.** Model bahasa, model difusi, dan GAN semuanya menghasilkan output melalui pengambilan sample. Algoritme pengambilan sample secara langsung mengontrol kreativitas, koherensi, dan keragaman. Pengambilan sample suhu, top-k, dan inti adalah tombol yang diputar setiap hari oleh para insinyur.

**Training.** Kumpulan mini sample gradient descent stokastik. Dropout mengambil sample neuron untuk dinonaktifkan. Augmentasi data mengambil sample transformasi acak. Pengambilan sample penting menimbang ulang sample untuk mengurangi varian gradient dalam pembelajaran penguatan (PPO, TRPO).

**Estimasi.** Banyak kuantitas dalam ML yang tidak memiliki solusi bentuk tertutup. Loss yang diharapkan pada distribusi data, fungsi partisi model berbasis energi, bukti dalam inference Bayesian. Estimasi Monte Carlo memperkirakan semua ini dengan merata-ratakan sample.

**Eksplorasi.** Algoritma MCMC mengeksplorasi distribusi posterior dalam inference Bayesian. Strategi evolusi mengambil contoh gangguan parameter. Pengambilan sample Thompson menyeimbangkan eksplorasi dan eksploitasi pada bandit.

Tantangan intinya: kamu hanya dapat mengambil sample langsung dari distribusi sederhana (seragam, normal). Untuk hal lainnya, kamu memerlukan metode untuk mengubah sample sederhana menjadi sample dari distribusi target kamu.

### Pengambilan Sample Acak Seragam

Setiap metode pengambilan sample dimulai di sini. Generator bilangan acak seragam menghasilkan nilai dalam [0, 1) dimana setiap sub-interval dengan panjang yang sama memiliki probabilitas yang sama.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```Untuk mengambil sample secara seragam dari kumpulan n item terpisah, buat U dan kembalikan lantai (n * U). Untuk mengambil sample dari rentang kontinu [a, b], hitung a + (b - a) * U.

Wawasan utamanya: satu bilangan acak seragam mengandung jumlah keacakan yang tepat untuk menghasilkan satu sample dari distribusi mana pun. Triknya adalah menemukan transformasi yang tepat.

### Metode CDF Terbalik (Inverse Transform Sampling)

Fungsi distribusi kumulatif (CDF) memetakan nilai ke probabilitas:

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

CDF terbalik memetakan probabilitas kembali ke nilai. Jika U ~ Uniform(0, 1), maka X = F_inverse(U) mengikuti distribusi target.

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Contoh distribusi eksponensial:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

Ini berfungsi sempurna ketika kamu dapat menuliskan F_inverse dalam bentuk tertutup. Untuk distribusi normal, tidak ada CDF inverse bentuk tertutup, jadi kami menggunakan metode lain (Box-Muller, atau pendekatan numerik).

**Versi diskrit:** Untuk distribusi diskrit, buat CDF sebagai jumlah kumulatif, buat U, dan temukan indeks pertama yang jumlah kumulatifnya melebihi U. Beginilah cara kerja `sample_categorical` di Lesson 06.

### Pengambilan Sample Penolakan

Ketika kamu tidak dapat membalikkan CDF tetapi dapat mengevaluasi PDF target hingga konstan, pengambilan sample penolakan berfungsi.

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

Semakin ketat ikatan M, semakin tinggi tingkat penerimaannya. Dalam dimension rendah (1-3), pengambilan sample penolakan bekerja dengan baik. Dalam high-dimensional, tingkat penerimaan turun secara eksponensial karena sebagian besar volume proposal ditolak. Ini adalah curse of dimensionality untuk pengambilan sample penolakan.

**Contoh: pengambilan sample dari normal terpotong.** Gunakan proposal seragam pada rentang terpotong. Amplop M adalah jumlah maksimum PDF normal dalam rentang tersebut.

**Contoh: pengambilan sample dari setengah lingkaran.** Usulkan secara seragam pada persegi panjang pembatas. Terima jika titik tersebut berada di dalam setengah lingkaran. Beginilah cara Monte Carlo menghitung pi: tingkat penerimaan sama dengan rasio luas pi/4.

### Pentingnya Pengambilan Sample

Terkadang kamu tidak memerlukan sample dari distribusi target p(x). kamu perlu memperkirakan ekspektasi di bawah p(x), dan kamu memiliki sample dari distribusi q(x) yang berbeda.

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

Ini sangat penting dalam pembelajaran penguatan. Dalam PPO (Optimization Kebijakan Proksimal), kamu mengumpulkan lintasan berdasarkan kebijakan lama pi_old tetapi ingin mengoptimalkan kebijakan baru pi_new. Weight pentingnya adalah pi_new(a|s) / pi_old(a|s). PPO memotong weight-weight ini untuk mencegah kebijakan baru agar tidak menyimpang terlalu jauh dari kebijakan lama.

Varians penduga pengambilan sample penting bergantung pada seberapa mirip q dengan p. Jika q sangat berbeda dengan p, beberapa sample mendapatkan weight yang sangat besar dan mendominasi estimasi. Pengambilan sample kepentingan yang dinormalisasi sendiri dibagi dengan jumlah weight untuk mengurangi masalah ini:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Estimasi Monte Carlo

Estimasi Monte Carlo memperkirakan integral dengan merata-ratakan sample acak. Hukum bilangan besar menjamin konvergensi.

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

Tingkat kesalahan tidak bergantung pada dimension. Inilah sebabnya mengapa metode Monte Carlo mendominasi dalam high-dimensional di mana integrasi berbasis jaringan tidak mungkin dilakukan.

**Memperkirakan pi:**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**Memperkirakan ekspektasi:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Rantai Markov Monte Carlo (MCMC): Metropolis-Hastings

MCMC membangun rantai Markov yang distribusi stasionernya adalah distribusi target p(x). Setelah langkah yang cukup, sample dari rantai tersebut (kira-kira) adalah sample dari p(x).

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```Untuk proposal simetris (q(x'|x) = q(x|x')), rasionya disederhanakan menjadi p(x')/p(x). Ini adalah algoritma Metropolis yang asli.

**Mengapa berhasil.** Aturan penerimaan memastikan keseimbangan terperinci: probabilitas berada di x dan berpindah ke x' sama dengan probabilitas berada di x' dan berpindah ke x. Keseimbangan terperinci menyiratkan bahwa p(x) adalah distribusi rantai yang stasioner.

**Pertimbangan praktis:**
- Burn-in: membuang sample awal sebelum rantai mencapai keseimbangan
- Thinning: pertahankan setiap sample ke-k untuk mengurangi autokorelasi
- Skala proposal: terlalu kecil dan rantai bergerak lambat (penerimaan tinggi, eksplorasi lambat); terlalu besar dan sebagian besar proposal ditolak (penerimaan rendah, terhenti di tempatnya)
- Tingkat penerimaan optimal untuk proposal Gaussian dalam high-dimensional adalah sekitar 0,234

### Pengambilan Sample Gibbs

Pengambilan sample Gibbs adalah kasus khusus MCMC untuk distribusi multivariat. Daripada mengusulkan perpindahan di semua dimension sekaligus, ia memperbarui satu variabel pada satu waktu dari distribusi kondisionalnya.

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Pengambilan sample Gibbs mengharuskan kamu mengambil sample dari setiap distribusi bersyarat p(x_i | x_{-i}). Ini mudah dilakukan untuk banyak model:
- Jaringan Bayesian: kondisional mengikuti struktur grafik
- Campuran Gaussian: kondisinya Gaussian
- Model Ising: kondisi setiap putaran hanya bergantung pada tetangganya

Tingkat penerimaan selalu 1 (setiap proposal diterima) karena pengambilan sample dari kondisi yang tepat secara otomatis memenuhi saldo terperinci.

**Keterbatasan.** Jika variabel sangat berkorelasi, pengambilan sample Gibbs akan tercampur secara perlahan karena memperbarui satu variabel pada satu waktu tidak dapat membuat pergerakan diagonal yang besar dalam distribusi.

### Pengambilan Sample Suhu (Digunakan di LLM)

Model bahasa mengeluarkan logit z_1, ..., z_V untuk setiap token dalam kosakata. Softmax mengubahnya menjadi probabilitas. Suhu mengubah skala logit sebelum softmax:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**Mengapa cara ini berhasil.** Membagi logit dengan T < 1 akan memperbesar perbedaan antar logit. Jika z_1 = 2 dan z_2 = 1, membaginya dengan T = 0,5 menghasilkan z_1/T = 4 dan z_2/T = 2, sehingga membuat jaraknya lebih besar. Setelah softmax, token dengan logit tertinggi mendapat bagian yang jauh lebih besar.

**Dalam praktiknya:**
- T = 0,0: decoding serakah, paling baik untuk tanya jawab faktual
- T = 0,3-0,7: sedikit kreatif, bagus untuk pembuatan code
- T = 0.7-1.0 : seimbang, baik untuk percakapan umum
- T = 1,0-1,5 : menulis kreatif, brainstorming
- T > 1,5 : semakin acak, jarang berguna

Suhu tidak mengubah token mana yang memungkinkan. Ini mengubah massa probabilitas yang dialokasikan untuk setiap token.

### Pengambilan Sample Top-k

Pengambilan sample k teratas membatasi kumpulan kandidat ke k token dengan probabilitas tertinggi, kemudian melakukan normalisasi ulang dan mengambil sample dari kumpulan terbatas tersebut.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k mencegah model memilih token yang sangat tidak mungkin (kesalahan ketik, omong kosong) yang ada di bagian panjang distribusi kosakata. Masalah: k diperbaiki terlepas dari konteksnya. Ketika model yakin (satu token memiliki probabilitas 95%), k = 40 masih memungkinkan 39 alternatif. Ketika model tidak pasti (probabilitas tersebar di 1000 token), k = 40 memotong opsi yang masuk akal.

### Pengambilan Sample Top-p (Inti).

Pengambilan sample top-p secara dinamis menyesuaikan ukuran kumpulan kandidat. Alih-alih menyimpan sejumlah token, ia menyimpan kumpulan token terkecil yang probabilitas kumulatifnya melebihi p.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```Jika modelnya yakin, pengambilan sample inti hanya menyimpan sedikit token (mungkin 2-3). Jika modelnya tidak pasti, modelnya tetap banyak (mungkin 200). Perilaku adaptif inilah yang menyebabkan pengambilan sample inti umumnya menghasilkan teks yang lebih baik daripada top-k.

**Kombinasi umum:**
- Suhu 0,7 + top-p 0,9: pengaturan tujuan umum yang baik
- Suhu 0,0 (serakah): terbaik untuk tugas deterministik
- Suhu 1,0 + top-k 50: Fan dkk. (2018) pengaturan kertas asli

Top-k dan top-p dapat digabungkan. Terapkan top-k terlebih dahulu, lalu top-p pada set sisanya.

### Trik Reparameterisasi (Digunakan di VAE)

Autoencoder variasional (VAE) belajar dengan menyandikan input ke dalam distribusi di ruang laten, mengambil sample dari distribusi tersebut, dan mendekode kembali sample tersebut. Masalah: kamu tidak dapat melakukan backpropagation melalui operasi pengambilan sample.

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

Trik reparameterisasi memisahkan keacakan dari parameter:

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

Ini berfungsi karena N(mu, sigma^2) memiliki distribusi yang sama dengan mu + sigma * N(0, 1). Wawasan utamanya: pindahkan keacakan ke sumber bebas parameter (epsilon), lalu nyatakan sample sebagai transformasi parameter yang dapat dibedakan.

**Dalam putaran training VAE:**
1. Encoder mengeluarkan mu dan log(sigma^2) untuk setiap input
2. Contoh epsilon ~ N(0, 1)
3. Hitung z = mu + sigma * epsilon
4. Dekode z untuk merekonstruksi input
5. Backpropagation melalui langkah 4, 3, 2, 1 (mungkin karena langkah 3 dapat terdiferensiasi)

Tanpa trik reparameterisasi, VAE tidak dapat dilatih dengan backpropagation standar. Wawasan tunggal ini menjadikan VAE praktis.

### Gumbel-Softmax (Pengambilan Sample Kategoris yang Dapat Dibedakan)

Trik reparameterisasi berfungsi untuk distribusi berkelanjutan (Gaussian). Untuk distribusi kategoris diskrit, kita memerlukan pendekatan yang berbeda. Gumbel-Softmax memberikan perkiraan yang dapat dibedakan untuk pengambilan sample kategorikal.

**Trik Gumbel-Max (tidak dapat dibedakan):**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax (perkiraan yang dapat dibedakan):**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax menghasilkan relaksasi berkelanjutan dari sample diskrit. Outputnya adalah vector probabilitas (soft one-hot) dan bukan hard one-hot. Gradient mengalir melalui softmax. Selama latihan forward pass, kamu dapat menggunakan estimator "straight-through": gunakan hard argmax untuk forward pass tetapi gradient lembut Gumbel-Softmax untuk backward pass.

**Aplikasi:**
- Variabel laten diskrit dalam VAE
- Pencarian arsitektur saraf (memilih operasi diskrit)
- Mekanisme attention yang keras
- Pembelajaran penguatan dengan tindakan terpisah

### Pengambilan Sample Berstrata

Pengambilan sample standar Monte Carlo dapat meninggalkan celah dalam ruang sample secara kebetulan. Pengambilan sample bertingkat memaksa cakupan yang merata dengan membagi ruang menjadi beberapa strata dan mengambil sample dari masing-masing strata.

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

Pengambilan sample bertingkat selalu memiliki varians yang lebih rendah atau sama dibandingkan dengan standar Monte Carlo:

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**Aplikasi:**
- Integrasi numerik (quasi-Monte Carlo)
- Pemisahan training data (memastikan keseimbangan kelas di setiap kelompok)
- Pengambilan sample penting dengan stratifikasi (menggabungkan kedua teknik)
- NeRF (Neural Radiance Fields) menggunakan pengambilan sample bertingkat di sepanjang sinar kamera

### Koneksi ke Model Difusi

Model difusi menghasilkan gambar melalui proses pengambilan sample. Proses maju menambahkan noise Gaussian ke gambar melalui T langkah hingga menjadi noise murni. Proses sebaliknya mempelajari denoise, memulihkan gambar asli selangkah demi selangkah.

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```Kaitannya dengan metode dalam lesson ini:
- Setiap langkah denoising menggunakan trik reparameterisasi (contoh noise, terapkan transformasi deterministik)
- Jadwal kebisingan {alpha_t} mengontrol bentuk anil suhu
- Training menggunakan estimasi Monte Carlo untuk memperkirakan ELBO (batas bawah bukti)
- Pengambilan sample leluhur dalam model difusi adalah rantai Markov (setiap langkah hanya bergantung pada keadaan saat ini)

Seluruh proses pembuatan gambar adalah pengambilan sample berulang: mulai dari noise, dan pada setiap langkah, ambil sample versi yang sedikit lebih sedikit noise yang dikondisikan pada model denoising yang dipelajari.

## Build

### Langkah 1: Pengambilan sample CDF seragam dan terbalik

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

Hasilkan 10.000 sample eksponensial dan verifikasi meannya adalah 1/lambda.

### Langkah 2: Pengambilan sample penolakan

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

Gunakan pengambilan sample penolakan untuk mengambil sample dari distribusi normal terpotong. Verifikasi bentuknya dengan membuat histogram sample.

### Langkah 3: Pentingnya pengambilan sample

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

Perkirakan E[X^2] pada distribusi normal menggunakan proposal seragam. Bandingkan dengan jawaban yang diketahui (mu^2 + sigma^2).

### Langkah 4: Estimasi pi di Monte Carlo

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Langkah 5: MCMC Metropolis-Hastings

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

Sample dari distribusi bimodal (campuran dua Gaussian). Visualisasikan lintasan rantai.

### Langkah 6: Pengambilan sample Gibbs

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Langkah 7: Pengambilan sample suhu

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

Tunjukkan bagaimana suhu mengubah distribusi output untuk sekumpulan logit token.

### Langkah 8: Pengambilan sample top-k dan top-p

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Langkah 9: Trik parameterisasi ulang

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

Tunjukkan bahwa gradient mengalir melalui sample yang diparameterisasi ulang tetapi tidak melalui pengambilan sample langsung.

### Langkah 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

Tunjukkan bagaimana penurunan suhu menyebabkan output mendekati vector satu panas.

Implementasi penuh dengan semua visualisasi ada di `code/sampling.py`.

## Pakai

Dengan NumPy dan SciPy, versi produksinya:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

Untuk MCMC dalam skala besar, gunakan perpustakaan khusus:
- PyMC: pemodelan Bayesian lengkap dengan NUTS (HMC adaptif)
- pembawa acara: sampler MCMC ansambel
- NumPyro/JAX: MCMC dengan akselerasi GPU

kamu membuatnya dari awal. Sekarang kamu tahu apa yang dilakukan panggilan perpustakaan.

## Latihan

1. Menerapkan pengambilan sample CDF terbalik untuk distribusi Cauchy. CDFnya adalah F(x) = 0,5 + arctan(x)/pi. Hasilkan 10.000 sample dan plot histogram dengan PDF yang sebenarnya. Perhatikan ekor yang berat (nilai ekstrim jauh dari pusat).

2. Gunakan pengambilan sample penolakan untuk menghasilkan sample dari distribusi Beta(2, 5) menggunakan proposal Seragam(0, 1). Plot sample yang diterima dengan PDF Beta yang sebenarnya. Berapa tingkat penerimaan teoretisnya?

3. Estimasi integral sin(x) dari 0 sampai pi menggunakan Monte Carlo dengan 1.000, 10.000, dan 100.000 sample. Bandingkan kesalahan di setiap level. Verifikasi bahwa kesalahan berskala sebagai O(1/sqrt(N)).

4. Implementasikan Metropolis-Hastings untuk mengambil sample dari distribusi 2D p(x, y) sebanding dengan exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2). Plot sample dan lintasan rantai. Bereksperimenlah dengan deviasi standar proposal yang berbeda.

5. Buat demo pembuatan teks lengkap: dengan kosakata 10 kata dengan logit, buat urutan 20 token menggunakan (a) serakah, (b) suhu=0,7, (c) top-k=3, (d) top-p=0,9. Bandingkan keragaman output di 5 proses.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|----------------------|
| Pengambilan sample | "Menggambar nilai acak" | Menghasilkan nilai berdasarkan distribusi probabilitas. Mekanisme di balik semua AI generatif |
| Distribusi seragam | "Semua kemungkinannya sama" | Setiap nilai di [a, b] mempunyai kepadatan probabilitas yang sama yaitu 1/(b-a). Titik awal untuk semua metode pengambilan sample |
| CDF terbalik | "Transformasi probabilitas" | F_inverse(U) mengubah sample seragam menjadi sample dari distribusi mana pun dengan CDF yang diketahui. Akurat dan efisien |
| Pengambilan sample penolakan | "Usulkan dan terima/tolak" | Hasilkan dari proposal sederhana, terima dengan probabilitas sebanding dengan rasio target/proposal. Akurat tetapi membuang sample |
| Pentingnya pengambilan sample | "Timbang ulang sample" | Perkirakan ekspektasi di bawah p(x) menggunakan sample dari q(x) dengan memberi weight pada setiap sample sebesar p(x)/q(x). Inti dari PPO di RL |
| Monte Carlo | "Sample acak rata-rata" | Perkiraan integral sebagai rata-rata sample. Kesalahan O(1/sqrt(N)) terlepas dari dimension |
| MCMC | "Jalan acak yang menyatu" | Bangunlah rantai Markov yang distribusi stasionernya menjadi targetnya. Metropolis-Hastings adalah algoritma dasar |
| Metropolis-Hastings | "Terima menanjak, kadang menurun" | Usulkan gerakan, terima berdasarkan rasio kepadatan. Keseimbangan terperinci memastikan konvergensi terhadap distribusi sasaran |
| Pengambilan sample Gibbs | "Satu variabel pada satu waktu" | Perbarui setiap variabel dari distribusi kondisionalnya dengan menjaga variabel lainnya tetap. Tingkat penerimaan 100% |
| Suhu | "Tombol kepercayaan diri" | Membagi logit dengan T sebelum softmax. T<1 menajam (lebih percaya diri), T>1 mendatar (lebih beragam) |
| Pengambilan sample k teratas | "Pertahankan yang terbaik" | Nol semua kecuali k token dengan probabilitas tertinggi, normalisasi ulang, sample. Memperbaiki ukuran kumpulan kandidat |
| Pengambilan sample inti (top-p) | "Simpan yang mungkin" | Pertahankan kumpulan token terkecil yang probabilitas kumulatifnya melebihi p. Ukuran kumpulan kandidat adaptif |
| Trik reparameterisasi | "Pindahkan keacakan ke luar" | Tulis z = mu + sigma * epsilon dimana epsilon ~ N(0,1). Membuat pengambilan sample dapat dibedakan. Penting untuk training VAE |
| Gumbel-Softmax | "Pengambilan sample kategoris lunak" | Perkiraan yang dapat dibedakan untuk pengambilan sample kategorikal menggunakan Gumbel noise + softmax dengan suhu |
| Pengambilan sample bertingkat | "Cakupan paksa" | Bagilah ruang sample menjadi beberapa strata, ambil sample dari masing-masing strata. Variansnya selalu lebih rendah daripada Monte Carlo yang naif |
| Terbakar | "Periode pemanasan" | Sample MCMC awal dibuang sebelum rantai mencapai distribusi stasionernya |
| Saldo terperinci | "Kondisi reversibilitas" | p(x) * T(x->y) = p(y) * T(y->x). Kondisi yang cukup untuk p menjadi distribusi stasioner dari rantai Markov |
| Pengambilan sample difusi | "Penyangkalan berulang" | Hasilkan data dengan memulai dari kebisingan dan menerapkan langkah-langkah menghilangkan kebisingan yang dipelajari. Setiap langkah adalah operasi pengambilan sample bersyarat |

## Bacaan Lanjutan- [Holbrook (2023): Algoritma Metropolis-Hastings](https://arxiv.org/abs/2304.07010) - tutorial mendetail tentang fondasi MCMC
- [Jang, Gu, Poole (2017): Reparameterisasi Kategorikal dengan Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - makalah Gumbel-Softmax asli
- [Holtzman dkk. (2020): Kasus Penasaran Degenerasi Teks Neural](https://arxiv.org/abs/1904.09751) - makalah pengambilan sample inti (top-p)
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Makalah VAE memperkenalkan trik reparameterisasi
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM menghubungkan pengambilan sample dengan pembuatan gambar
