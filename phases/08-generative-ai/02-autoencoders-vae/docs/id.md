# Autoencoder & Variational Autoencoder (VAE)

> Autoencoder biasa mengompres lalu merekonstruksi. Itu menghafal. Itu tidak menghasilkan. Tambahkan satu trik — paksa code agar terlihat Gaussian — dan kamu akan mendapatkan sampler. Trik tunggal tersebut, parameterisasi ulang `z = μ + σ·ε`, adalah alasan setiap model gambar difusi laten dan pencocokan aliran yang kamu gunakan pada tahun 2026 memiliki VAE pada inputnya.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 3 · 02 (Backprop), Fase 3 · 07 (CNN), Fase 8 · 01 (Taksonomi)
**Waktu:** ~75 menit

## Masalah

Kompres digit MNIST 784 piksel menjadi code 16 angka, lalu buat ulang. Autoencoder biasa akan berhasil merekonstruksi MSE tetapi ruang kodenya berantakan. Pilih titik acak di ruang code, pecahkan kodenya, dan kamu akan mendapatkan noise. Ia tidak memiliki sampler. Ini adalah model kompresi yang didandani.

Apa yang sebenarnya kamu inginkan adalah: (a) ruang code adalah distribusi yang bersih dan lancar tempat kamu dapat mengambil sample - misalnya Gaussian isotropik `N(0, I)`, (b) mendekode sample apa pun menghasilkan digit yang masuk akal, dan (c) encoder dan decoder masih terkompresi dengan baik. Tiga gol, satu arsitektur, satu kekalahan.

VAE Kingma 2013 memecahkan masalah ini dengan melatih encoder untuk menghasilkan *distribusi* `q(z|x) = N(μ(x), σ(x)²)`, menarik distribusi tersebut ke `N(0, I)` sebelumnya melalui penalti KL, dan kemudian mengambil sample `z` dari `q(z|x)` sebelum mendekode. Pada waktu inference, lepaskan encoder, contoh `z ~ N(0, I)`, decode. Hukuman KL inilah yang memaksa ruang code menjadi terstruktur.

Pada tahun 2026, VAE jarang dikirimkan secara mandiri — VAE telah kalah dalam hal difusi dalam hal kualitas gambar mentah — namun VAE merupakan encoder pilihan untuk setiap model difusi laten (SD 1/2/XL/3, Flux, AudioCraft). Learn VAE dan kamu akan mempelajari layer pertama yang tidak terlihat dari setiap pipeline gambar yang kamu gunakan.

## Konsep

![Autoencoder vs VAE: trik parameterisasi ulang](../assets/vae.svg)

**Autoencoder.** `z = encoder(x)`, `x̂ = decoder(z)`, loss = `||x - x̂||²`. Ruang code tidak terstruktur.

**Encoder VAE.** Menghasilkan dua vector: `μ(x)` dan `log σ²(x)`. Ini mendefinisikan `q(z|x) = N(μ, diag(σ²))`.

**Trik reparameterisasi.** Pengambilan sample dari `q(z|x)` tidak dapat dibedakan. Tulis ulang sample sebagai `z = μ + σ·ε` di mana `ε ~ N(0, I)`. Sekarang `z` adalah fungsi deterministik dari `(μ, σ)` ditambah noise non-parameter — aliran gradient melalui `μ` dan `σ`.

**Loss.** Bukti Batas Bawah (ELBO), dua istilah:

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

Rekonstruksi mendorong `x̂` menuju `x`. KL mendorong `q(z|x)` menuju sebelumnya. Mereka berdagang. β kecil (<1) = sample lebih tajam, ruang code lebih sedikit Gaussian. β besar (>1) = ruang code lebih bersih, sample lebih buram. β-VAE (Higgins 2017) menjadikan kenop ini terkenal dan memulai penelitian penguraian.

**Pengambilan sample.** Saat inference: gambar `z ~ N(0, I)`, teruskan melalui decoder. Satu langkah maju - tidak ada pengambilan sample berulang seperti difusi.

## Build

`code/main.py` mengimplementasikan VAE kecil tanpa numpy atau obor. Input adalah data sintetik 8 dimension yang diambil dari campuran Gaussian 2 komponen dalam 8-D. Encoder dan decoder adalah MLP layer tersembunyi tunggal. Kami menerapkan activation tanh, forward pass, loss, dan backward pass yang ditulis tangan. Bukan produksi — pedagogi.

### Langkah 1: meneruskan pembuat enkode

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
````log σ²` alih-alih `σ` sehingga output jaringan tidak dibatasi (softplus dari σ adalah jebakan — gradient mati pada σ ≈ 0).

### Langkah 2: parameterisasi ulang dan dekode

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### Langkah 3: ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

KL bentuk tertutup yang tepat karena kedua distribusinya Gaussian. Jangan mengintegrasikan secara numerik. Orang-orang masih mengirimkan code dengan perkiraan monte-carlo KL pada tahun 2026 — code ini 3x lebih lambat tanpa alasan.

### Langkah 4: buat

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

Itu adalah model generatif. Lima baris.

## Jebakan

- **Keruntuhan posterior.** Istilah KL mendorong `q(z|x) → N(0, I)` dengan sangat agresif sehingga `z` tidak memberikan informasi tentang `x`. Perbaiki: β-annealing (mulai β=0, ramp ke 1), bebaskan bit, atau lewati KL pada dimension tidak aktif.
- **Sample buram.** Kemungkinan dekoder Gaussian menyiratkan rekonstruksi MSE, yang merupakan Bayes-optimal untuk L2 (rata-rata) — rata-rata dari sekumpulan digit yang masuk akal adalah digit fuzzy. Perbaiki: decoder diskrit (VQ-VAE, NVAE), atau gunakan VAE hanya sebagai encoder dan tumpukan difusi pada laten (inilah yang dilakukan Difusi Stabil).
- **β terlalu besar, terlalu dini.** Lihat keruntuhan posterior. Mulai dari β≈0.01 dan ramp.
- **Peredupan laten terlalu kecil.** 16-D berfungsi untuk MNIST, 256-D untuk ImageNet 256², 2048-D untuk ImageNet 1024². Kompres VAE Difusi Stabil 512×512×3 → 64×64×4 (faktor sample bawah 32x dalam area spasial, 32x dalam pipeline).

## Pakai

Tumpukan VAE 2026:

| Situasi | Pilih |
|-----------|------|
| Encoder gambar-laten untuk difusi | VAE Difusi Stabil (`sd-vae-ft-ema`) atau Fluks VAE |
| Encoder audio-laten | Encodec (Meta), SoundStream, atau DAC (Deskripsi) |
| Video laten | Patch spatiotemporal Sora, Latte VAE, WAN VAE |
| Pembelajaran representasi terurai | β-VAE, FaktorVAE, TCVAE |
| Laten diskrit (untuk pemodelan Transformer) | VQ-VAE, RVQ (SisaVQ) |
| Laten berkelanjutan untuk generasi | VAE biasa, lalu kondisikan model aliran/difusi pada ruang laten |

Model difusi laten adalah VAE dengan model difusi yang berada di antara encoder dan decoder. VAE melakukan kompresi kasar, model difusi melakukan weight berat. Pola yang sama untuk video (VAE + video-diffusion DiT) dan audio (Encodec + MusicGen Transformer).

## Kirim

Simpan `outputs/skill-vae-trainer.md`.

Keahlian yang dibutuhkan: profil dataset + target peredupan laten + penggunaan hilir (rekonstruksi, pengambilan sample, atau input difusi laten) dan output: pilihan arsitektur (polos/β/VQ/RVQ), jadwal β, peredupan laten, kemungkinan dekoder (Gaussian vs kategorikal), dan rencana evaluasi (pengintaian MSE, KL per dim, distance Fréchet antara `q(z|x)` dan `N(0, I)`).

## Latihan

1. **Mudah.** Ubah `β` di `code/main.py` menjadi `0.01`, `0.1`, `1.0`, `5.0`. Catat rekonstruksi akhir UMK dan KL. β manakah yang terbaik untuk Pareto untuk data sintetis kamu?
2. **Medium.** Ganti kemungkinan dekoder Gaussian dengan kemungkinan Bernoulli (loss lintas entropi). Bandingkan kualitas sample pada versi biner dari data sintetis yang sama.
3. **Sulit.** Perluas `code/main.py` menjadi VQ-VAE mini: ganti `z` berkelanjutan dengan pencarian nearest neighbor dalam buku code dengan entri K=32. Bandingkan MSE rekonstruksi dan laporkan berapa banyak entri buku code yang digunakan (keruntuhan buku code memang nyata).

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Pembuat enkode otomatis | Jaringan encode-decode | `x → z → x̂`, pelajari UMK. Bukan generatif. |
| VAE | AE dengan sampler | Encoder mengeluarkan distribusi, penalti KL membentuk ruang code. |
| ELBO | Bukti batas bawah | `log p(x) ≥ recon - KL[q(z|x) \|\| p(z)]`; ketat ketika `q = p(z|x)`. |
| Reparameterisasi | `z = μ + σ·ε` | Menulis ulang simpul stokastik sebagai deterministik + kebisingan murni. Mengaktifkan backprop melalui pengambilan sample. |
| Sebelumnya | `p(z)` | Distribusi target untuk yang laten, biasanya `N(0, I)`. |
| Runtuhnya bagian belakang | "Istilah KL menang" | Encoder mengabaikan `x`, mengeluarkan yang sebelumnya; decoder harus berhalusinasi. |
| β-VAE | Berat KL merdu | `loss = recon + β·KL`. β lebih tinggi = lebih terurai tetapi lebih kabur. |
| VQ-VAE | Diskrit laten | Ganti `z` berkelanjutan dengan vector buku code terdekat; memungkinkan pemodelan Transformer. |

## Catatan produksi: VAE adalah jalur terpanas di server difusi

Dalam pipeline Stable Diffusion / Flux / SD3, VAE dipanggil dua kali per permintaan — sekali untuk menyandikan (jika melakukan img2img / inpainting) dan sekali untuk mendekode. Pada 1024² jalur dekoder sering kali merupakan puncak memori activation terbesar di seluruh pipeline karena meningkatkan sample `128×128×16` secara laten kembali ke `1024×1024×3`. Dua konsekuensi praktis:

- **Potong atau ubinkan dekodenya.** `diffusers` menampilkan `pipe.vae.enable_slicing()` dan `pipe.vae.enable_tiling()`. Ubin menukar artefak jahitan kecil untuk memori `O(tile²)` alih-alih `O(H·W)`. Penting untuk 1024²+ pada GPU konsumen.
- **dekoder bf16, angka fp32 untuk pengubahan ukuran akhir.** SD 1.x VAE dirilis di fp32 dan *secara diam-diam menghasilkan NaN* saat dilemparkan ke fp16 pada 1024²+. SDXL dikirimkan `madebyollin/sdxl-vae-fp16-fix` — selalu lebih memilih varian fp16-fix atau menggunakan bf16.

## Bacaan Lanjutan

- [Kingma & Welling (2013). Bayes Variasi Pengkodean Otomatis](https://arxiv.org/abs/1312.6114) — makalah VAE.
- [Higgins dkk. (2017). β-VAE: Mempelajari Konsep Visual Dasar dengan Kerangka Variasi Terkendali](https://openreview.net/forum?id=Sy2fzU9gl) — menguraikan β-VAE.
- [van den Oord dkk. (2017). Pembelajaran Representasi Diskrit Neural](https://arxiv.org/abs/1711.00937) — VQ-VAE.
- [Vahdat & Kautz (2021). NVAE: Autoencoder Variasi Hierarki Mendalam](https://arxiv.org/abs/2007.03898) — gambar VAE yang canggih.
- [Rombach dkk. (2022). Sintesis Gambar Resolusi Tinggi dengan Model Difusi Laten](https://arxiv.org/abs/2112.10752) — Difusi Stabil; VAE sebagai pembuat enkode.
- [Défossez dkk. (2022). Kompresi Audio Neural Fidelitas Tinggi](https://arxiv.org/abs/2210.13438) — Encodec, standar audio VAE.
