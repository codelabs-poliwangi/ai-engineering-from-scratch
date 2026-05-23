# Model Difusi — DDPM dari Awal

> Ho, Jain, Abbeel (2020) memberikan resep yang tidak dapat dihentikan. Hancurkan data dengan noise dalam ribuan langkah kecil. Latih satu neural network untuk memprediksi kebisingan. Balikkan proses pada inference. Saat ini, setiap model gambar, video, 3D, dan musik mainstream berjalan pada loop ini, mungkin dengan trik pencocokan aliran atau konsistensi di atasnya.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 3 · 02 (Backprop), Fase 8 · 02 (VAE)
**Waktu:** ~75 menit

## Masalah

kamu ingin sampler untuk `p_data(x)`. GAN memainkan permainan minimax yang sering kali menyimpang. VAE menghasilkan sample buram dari dekoder Gaussian. Yang benar-benar kamu inginkan adalah tujuan training yang (a) satu loss stabil (tanpa titik pelana, tanpa minimax), (b) batas bawah pada `log p(x)` (sehingga kamu memiliki kemungkinan), dan (c) sample yang cocok dengan kualitas SOTA.

Sohl-Dickstein dkk. (2015) memiliki jawaban teoretis: tentukan rantai Markov `q(x_t | x_{t-1})` yang secara bertahap menambahkan derau Gaussian, dan latih rantai terbalik `p_θ(x_{t-1} | x_t)` untuk menghilangkan derau. Ho, Jain, Abbeel (2020) menunjukkan loss dapat disederhanakan menjadi satu garis — memprediksi kebisingan — dan menyelesaikan perhitungannya. Pada tahun 2020 ini adalah sebuah keingintahuan. Pada tahun 2021, perusahaan ini menghasilkan sample yang canggih. Pada tahun 2022 menjadi Difusi Stabil. Pada tahun 2026 itu adalah substrat.

## Konsep

![DDPM: derau maju, derau balik](../assets/ddpm.svg)

**Meneruskan proses `q`.** Tambahkan noise Gaussian dalam `T` langkah kecil. Bentuk tertutupnya — alasan matematikanya mudah diatur — adalah karena langkah kumulatifnya juga Gaussian:

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

dimana `α̅_t = ∏_{s=1..t} (1 - β_s)` untuk jadwal `β_t`. Pilih `β_t` dari 1e-4 hingga 0,02 secara linear pada T=1000 langkah dan `x_T` kira-kira `N(0, I)`.

**Proses terbalik `p_θ`.** Learn neural network `ε_θ(x_t, t)` yang memprediksi noise yang ditambahkan. Diberikan `x_t`, ditolak oleh:

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

dimana `σ_t` adalah `sqrt(β_t)` atau varians yang dipelajari. Ekspresinya jelek tetapi hanya aljabar - menyelesaikan `x_{t-1}` mengingat posterior `q(x_{t-1} | x_t, x_0)` dan mengganti `x_0` dengan estimasi prediksi kebisingannya.

**Kehilangan latihan.**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

Contoh `x_0` dari data, pilih `t` secara acak, contoh `ε ~ N(0, I)`, hitung noise `x_t` dalam satu pengambilan gambar melalui bentuk tertutup, dan lakukan regresi pada noise. Satu loss, tidak ada minimax, tidak ada KL, tidak ada trik reparameterisasi.

**Pengambilan sample.** Mulai `x_T ~ N(0, I)`. Ulangi langkah sebaliknya dari `t = T` ke `1`. Selesai.

## Mengapa ini berhasil

Tiga intuisi:

1. **Menyangkal kebisingan itu mudah; menghasilkannya sulit.** Di `t=T`, datanya murni noise — jaringan harus memecahkan masalah sepele. Di `t=0`, internet hanya perlu membersihkan beberapa piksel. Pada tingkat menengah `t`, masalahnya sulit tetapi jaringan memiliki banyak gradient yang mengalir melalui weight yang sama dari setiap tingkat kebisingan.

2. **Pencocokan skor secara terselubung.** Vincent (2011) membuktikan bahwa memprediksi kebisingan sama dengan memperkirakan `∇_x log q(x_t | x_0)`, *skor*. SDE terbalik menggunakan skor ini untuk menaikkan gradient kepadatan — perjalanan acak terpandu menuju wilayah dengan probabilitas tinggi.3. **ELBO direduksi menjadi MSE sederhana.** Batas bawah variasi penuh memiliki istilah KL per langkah waktu. Dengan parameterisasi DDPM, istilah KL tersebut disederhanakan menjadi MSE pada prediksi kebisingan dengan koefisien tertentu; Ho menurunkan koefisien (menyebutnya loss "sederhana") dan kualitas *meningkat*.

## Build

`code/main.py` mengimplementasikan DDPM 1-D. Data adalah campuran dua mode. "Jaring" adalah MLP kecil yang mengambil `(x_t, t)` dan mengeluarkan kebisingan yang diprediksi. Training adalah loss satu baris. Pengambilan sample mengulangi rantai terbalik.

### Langkah 1: jadwal maju (formulir tertutup)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### Langkah 2: sample `x_t` dalam satu kesempatan

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### Langkah 3: satu langkah training

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### Langkah 4: pengambilan sample terbalik

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

Untuk masalah 1-D dengan 40 langkah waktu dan MLP 24 unit, ini mempelajari campuran dua mode dalam ~200 epoch.

## Pengondisian waktu

Internet perlu mengetahui rentang waktu mana yang dikecamnya. Dua opsi standar:

- **Embedding sinusoidal.** Seperti pengkodean posisi Transformer. `embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`. Melewati MLP, disiarkan ke internet.
- **Pengondisian film / norm grup.** Embedding proyek ke skala/bias per pipeline (FiLM) di setiap blok.

Code mainan kami menggunakan sinusoidal → concat. Produksi U-Nets menggunakan FiLM.

## Jebakan

- **Jadwal sangat penting.** Linear `β` adalah default DDPM tetapi jadwal kosinus (Nichol & Dhariwal, 2021) memberikan FID yang lebih baik untuk komputasi yang sama. Ganti jadwal jika kualitas tidak stabil.
- **Embedding timestep bersifat rapuh.** Melewati `t` mentah sebagai pelampung berfungsi untuk mainan 1-D tetapi gagal untuk gambar; selalu gunakan embedding yang tepat.
- **Prediksi V vs prediksi ε.** Untuk rezim sempit (t sangat kecil atau sangat besar), `ε` memiliki signal-to-noise yang buruk. Prediksi V (`v = α·ε - σ·x`) lebih stabil; SDXL, SD3, dan Flux menggunakannya.
- **Panduan bebas pengklasifikasi.** Pada inference, hitung `ε` bersyarat dan tidak bersyarat, lalu `ε_cfg = (1 + w) · ε_cond - w · ε_uncond` dengan `w ≈ 3-7`. Dibahas dalam Lesson 08.
- **1000 langkah itu banyak.** Produksi menggunakan DDIM (20-50 langkah), DPM-Solver (10-20 langkah), atau distilasi (1-4 langkah). Lihat Lesson 12.

## Pakai

| Peran | Tumpukan tipikal pada tahun 2026 |
|------|-----------------------|
| Difusi ruang piksel gambar (kecil, mainan) | DDPM + U-Net |
| Difusi laten gambar | Encoder VAE + U-Net atau DiT (Lesson 07) |
| Difusi laten video | DiT Spatiotemporal (Sora, Veo, WAN) |
| Difusi laten audio | Encodec + Transformer difusi |
| Sains (molekul, protein, fisika) | Difusi ekivalen (EDM, difusi RF, AlphaFold3) |

Difusi adalah tulang punggung generatif universal. Pencocokan aliran (Lesson 13) adalah pesaing 2024-2026 yang biasanya menang dalam kecepatan inference untuk kualitas yang sama.

## Kirim

Simpan `outputs/skill-diffusion-trainer.md`. Keterampilan mengambil dataset + menghitung anggaran dan output: jadwal (linier/kosinus/sigmoid), target prediksi (ε/v/x), jumlah langkah, skala panduan, kelompok sample, dan protokol evaluasi.

## Latihan1. **Mudah.** Ubah T dari 40 menjadi 10 di `code/main.py`. Bagaimana kualitas sample (histogram visual output) menurun? Pada T berapa struktur dua mode runtuh?
2. **Sedang.** Beralih dari prediksi ε ke prediksi v. Turunkan kembali langkah sebaliknya. Bandingkan kualitas sample akhir.
3. **Sulit.** Tambahkan panduan bebas pengklasifikasi. Kondisikan pada label kelas `c ∈ {0, 1}`, hilangkan 10% selama training, dan pada waktu pengambilan sample gunakan `ε = (1+w)·ε_cond - w·ε_uncond`. Ukur tingkat hit mode bersyarat di `w = 0, 1, 3, 7`.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Proses maju | "Menambahkan kebisingan" | Memperbaiki rantai Markov `q(x_t | x_{t-1})` yang merusak data. |
| Proses terbalik | "Mencela" | Rantai terpelajar `p_θ(x_{t-1} | x_t)` yang merekonstruksi data. |
| jadwal β | "Tangga Kebisingan" | Varians per langkah; linier, kosinus, atau sigmoid. |
| α̅ | "Bilah alpha" | Produk kumulatif `∏(1 - β)`; memberikan bentuk tertutup `x_t` dari `x_0`. |
| Loss sederhana | "UMK tentang kebisingan" | `||ε - ε_θ(x_t, t)||²`; semua derivasi variasional runtuh pada hal ini. |
| ε-prediksi | "Memprediksi kebisingan" | Output adalah kebisingan yang ditambahkan; standar DDPM. |
| Prediksi V | "Memprediksi kecepatan" | Keluarannya adalah `α·ε - σ·x`; pengkondisian yang lebih baik di t. |
| DDPM | "Kertas" | Ho dkk. 2020; linier β, 1000 langkah, U-Net. |
| DDIM | "Contoh deterministik" | Sampler non-Markov, 20-50 langkah, tujuan training yang sama. |
| Panduan bebas pengklasifikasi | "CFG" | Gabungkan prediksi kebisingan bersyarat dan tidak bersyarat untuk memperkuat pengondisian. |

## Catatan produksi: inference difusi adalah masalah penghitungan langkah

Makalah DDPM menjalankan T=1000 langkah terbalik. Tidak ada yang mengirimkannya dalam produksi. Setiap tumpukan inference sebenarnya memilih salah satu dari tiga strategi — dan masing-masing memetakan dengan rapi ke kerangka produksi "dari mana latensi berasal":

1. **Sampler lebih cepat, model sama.** DDIM (20-50 langkah), DPM-Solver++ (10-20), UniPC (8-16). Penggantian loop terbalik secara drop-in; weight `ε_θ` yang terlatih tidak tersentuh. Memotong latensi 20-50×.
2. **Distilasi.** Latih siswa untuk mencocokkan guru dalam langkah-langkah yang lebih sedikit: Distilasi Progresif (2 → 1), Model Konsistensi (sewenang-wenang → 1-4), LCM, SDXL-Turbo, SD3-Turbo. Memotong latensi 5-10× lagi, memerlukan training ulang.
3. **Caching dan kompilasi.** `torch.compile(unet, mode="reduce-overhead")`, backend difusi TensorRT-LLM, `xformers`/SDPA attention, weight bf16. Memotong latensi per langkah ~2×. Tumpukan dengan (1) dan (2).

Untuk server difusi produksi, percakapan anggaran sama dengan yang dijelaskan dalam literatur produksi untuk LLM: latensi adalah `num_steps × step_cost + VAE_decode`, throughput adalah `batch_size × (num_steps × step_cost)^-1`. TTFT kecil (satu langkah); Setara dengan TPOT adalah waktu respons penuh karena pembuatan gambar dilakukan "sekaligus" dari sudut pandang pengguna.

## Bacaan Lanjutan- [Sohl-Dickstein dkk. (2015). Pembelajaran Tanpa Pengawasan Mendalam menggunakan Termodinamika Nonequilibrium](https://arxiv.org/abs/1503.03585) — makalah difusi, yang lebih maju dari masanya.
- [Ho, Jain, Abbeel (2020). Menyangkal Model Probabilistik Difusi](https://arxiv.org/abs/2006.11239) — DDPM.
- [Lagu, Meng, Ermon (2021). Menolak Model Implisit Difusi](https://arxiv.org/abs/2010.02502) — DDIM, langkah lebih sedikit.
- [Nichol & Dhariwal (2021). Peningkatan DDPM](https://arxiv.org/abs/2102.09672) — jadwal kosinus, varians yang dipelajari.
- [Dhariwal & Nichol (2021). Model Difusi Mengalahkan GAN dalam Sintesis Gambar](https://arxiv.org/abs/2105.05233) — panduan pengklasifikasi.
- [Ho & Saliman (2022). Panduan Difusi Bebas Pengklasifikasi](https://arxiv.org/abs/2207.12598) — CFG.
- [Karras dkk. (2022). Menjelaskan Ruang Desain Model Generatif Berbasis Difusi (EDM)](https://arxiv.org/abs/2206.00364) — notasi terpadu, resep terbersih.
