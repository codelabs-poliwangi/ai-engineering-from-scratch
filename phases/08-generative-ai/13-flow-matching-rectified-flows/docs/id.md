# Pencocokan Aliran & Aliran yang Diperbaiki

> Model difusi memerlukan 20-50 langkah pengambilan sample karena model tersebut menempuh jalur melengkung dari noise ke data. Pencocokan aliran (Lipman et al., 2023) dan aliran yang diperbaiki (Liu et al., 2022) melatih jalur lurus. Jalur yang lebih lurus berarti langkah yang lebih sedikit berarti inference yang lebih cepat. Difusi Stabil 3, Flux.1, dan AudioCraft 2 semuanya dialihkan ke pencocokan aliran pada tahun 2024.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 8 · 06 (DDPM), Phase 1 · Kalkulus
**Waktu:** ~45 menit

## Masalah

Proses kebalikan DDPM adalah perjalanan stokastik 1000 langkah dari `N(0, I)` kembali ke distribusi data. DDIM mengecilnya menjadi 20-50 langkah deterministik. kamu menginginkan lebih sedikit langkah — idealnya satu langkah. Penghalangnya adalah ODE yang menyelesaikan proses sebaliknya kaku; jalannya melengkung.

Jika kamu dapat melatih model sedemikian rupa sehingga jalur dari noise ke data adalah *garis lurus*, satu langkah Euler dari `t=1` ke `t=0` akan berhasil. Pencocokan aliran membangun ini secara langsung: tentukan interpolasi garis lurus dari `x_1 ∼ N(0, I)` ke `x_0 ∼ data`, latih bidang vector `v_θ(x, t)` untuk mencocokkan turunan waktunya, integrasikan pada inference.

Aliran yang diperbaiki (Liu 2022) melangkah lebih jauh: meluruskan jalur secara berulang dengan prosedur reflow yang menghasilkan ODE yang semakin mendekati linier. Setelah dua iterasi reflow, sampler 2 langkah cocok dengan kualitas DDPM 50 langkah.

## Konsep

![Pencocokan aliran: interpolasi garis lurus antara noise dan data](../assets/flow-matching.svg)

### Aliran garis lurus

Definisikan:

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

di mana `x_0 ~ data` dan `x_1 ~ N(0, I)`. Turunan waktu sepanjang garis lurus ini adalah konstan:

```
dx_t / dt = x_1 - x_0
```

Tentukan bidang vector saraf `v_θ(x_t, t)` dan latih agar cocok dengan turunan ini:

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

Ini adalah loss **pencocokan aliran bersyarat** (Lipman 2023). Training bebas simulasi: kamu tidak pernah membuka gulungan ODE. Cukup cicipi `(x_0, x_1, t)` dan mundur.

### Pengambilan sample

Pada inference, integrasikan bidang vector yang dipelajari *mundur* dalam waktu:

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

Mulai dari `x_1 ~ N(0, I)`, Euler turun ke `t=0`.

### Aliran diperbaiki (Liu 2022)

Aliran garis lurus berfungsi tetapi jalur yang dipelajari *tidak benar-benar lurus* — jalur tersebut melengkung karena banyak `x_0` yang dapat dipetakan ke `x_1` yang sama. Langkah reflow aliran yang diperbaiki:

1. Latih model aliran v_1 dengan pasangan acak.
2. Contoh N pasangan `(x_1, x_0)` dengan mengintegrasikan v_1 dari `x_1` ke pendaratannya `x_0`.
3. Latih v_2 pada contoh berpasangan tersebut. Karena pasangan-pasangan tersebut sekarang "cocok dengan ODE", interpolasi garis lurus di antara keduanya benar-benar datar.
4. Ulangi.

Dalam praktiknya, 2 iterasi reflow membuat kamu mendekati linier, memungkinkan inference 2-4 langkah. SDXL-Turbo, SD3-Turbo, LCM semuanya merupakan model pencocokan aliran yang disuling.

### Mengapa ini menang untuk gambar pada tahun 2024

Tiga alasan:

1. **Training bebas simulasi** — tidak ada ODE yang dibuka selama training, mudah diterapkan.
2. **Geometri loss yang lebih baik** — jalur lurus memiliki signal-to-noise yang konsisten, sedangkan DDPM ε-loss memiliki SNR yang buruk di tepi jadwal.
3. **Inference lebih cepat** — 4-8 langkah dengan kualitas SDXL-Turbo; 1 langkah dengan distilasi konsistensi.

## Pencocokan aliran vs DDPM — koneksi yang tepatPencocokan aliran dengan jalur bersyarat Gaussian adalah difusi *dengan jadwal kebisingan tertentu*. Pilih jadwal `x_t = α(t) x_0 + σ(t) x_1` dan pencocokan aliran memulihkan difusi yang diformulasi ulang Stratonovich dengan `v = α'·x_0 - σ'·x_1`. Keduanya secara aljabar setara untuk jalur Gaussian.

Apa yang ditambahkan oleh pencocokan aliran: *kejelasan* target (kecepatan biasa), loss yang lebih bersih, dan izin untuk bereksperimen dengan interpolan non-Gaussian.

## Build

`code/main.py` mengimplementasikan pencocokan aliran 1-D pada campuran Gaussian dua mode. Bidang vector `v_θ(x, t)` adalah MLP kecil yang dilatih dengan target garis lurus. Pada inference, integrasikan langkah Euler 1, 2, 4, dan 20 dan bandingkan kualitas sample.

### Langkah 1: kehilangan latihan

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### Langkah 2: inference multi-langkah

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Langkah 3: bandingkan jumlah langkah

Harapkan sampler 4 langkah sudah sesuai dengan kualitas 20 langkah — masalah latensi yang besar.

## Jebakan

- **Parameterisasi waktu.** Pencocokan aliran menggunakan `t ∈ [0, 1]` dengan `t=0` pada data, `t=1` pada kebisingan. DDPM menggunakan `t ∈ [0, T]` dengan `t=0` pada data, `t=T` pada noise. Arahnya sama, skalanya berbeda. Makalah terus-menerus melakukan kesalahan ini.
- **Pilihan jadwal.** Garis lurus aliran yang diperbaiki adalah jadwal pencocokan aliran "", namun kamu dapat menggunakan t-sampling cosinus atau logit-normal (SD3 melakukan ini) untuk cakupan skala yang lebih baik.
- **Biaya reflow.** Pembuatan set data berpasangan untuk reflow adalah proses inference penuh per sample. Lakukan reflow hanya jika kamu benar-benar membutuhkan inference 1-2 langkah.
- **Panduan bebas pengklasifikasi masih berlaku.** Cukup tukar ε dengan v dalam kombinasi linier: `v_cfg = (1+w) v_cond - w v_uncond`.

## Pakai

| Kasus penggunaan | tumpukan 2026 |
|----------|-----------|
| Teks-ke-gambar, kualitas terbaik | Pencocokan aliran: SD3, Flux.1-dev |
| Teks-ke-gambar, 1-4 langkah | Pencocokan aliran suling: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| Inference waktu nyata | Distilasi konsistensi dari basis pencocokan aliran (LCM, PCM) |
| Generasi audio | Pencocokan aliran: Audio Stabil 2.5, AudioCraft 2 |
| Pembuatan video | Pencocokan aliran dicampur dengan difusi (Sora, Veo, Stable Video) |
| Sains/fisika (lintasan partikel, molekul) | Pencocokan aliran + bidang vector ekuivalen |

Setiap kali sebuah makalah mengatakan "lebih cepat dari difusi" pada tahun 2025-2026, hampir selalu merupakan pencocokan aliran + distilasi.

## Kirim

Simpan `outputs/skill-fm-tuner.md`. Keterampilan mengambil spesifikasi model gaya difusi dan mengubahnya menjadi konfigurasi training yang cocok dengan aliran: pilihan jadwal, distribusi pengambilan sample waktu (seragam / logit-normal), optimizer, rencana alur ulang, jumlah langkah target, protokol eval.

## Latihan

1. **Mudah.** Jalankan `code/main.py` dan bandingkan MSE 1 langkah vs 20 langkah vs distribusi data sebenarnya.
2. **Sedang.** Beralih dari pengambilan sample seragam `t` ke logit-normal (konsentrasi pengambilan sample pada pertengahan t). Apakah kualitas model meningkat?
3. **Sulit.** Terapkan satu iterasi reflow: buat pasangan (x_0, x_1) dengan mengintegrasikan model pertama, latih model kedua secara berpasangan, dan bandingkan kualitas sample 1 langkah.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Pencocokan aliran | "Difusi garis lurus" | Latih `v_θ(x, t)` untuk mencocokkan `x_1 - x_0` sepanjang interpolant. |
| Aliran diperbaiki | "Mengalir ulang" | Prosedur berulang yang meluruskan alur yang dipelajari. |
| Bidang kecepatan | "v_θ" | Output model — arah pergerakan `x_t`. |
| Interpolan garis lurus | "Jalan" | `x_t = (1-t)·x_0 + t·x_1`; turunan target yang sepele. |
| Sample Euler | "Pemecah ODE urutan pertama" | Integrator paling sederhana; bekerja dengan baik ketika jalurnya lurus. |
| Logit-normal t | "Pengambilan sample SD3" | Konsentrasikan pengambilan sample `t` ke arah nilai tengah yang gradiennya paling kuat. |
| Distilasi konsistensi | "sample 1 langkah" | Latih siswa untuk memetakan `x_t` langsung ke `x_0`. |
| CFG dengan kecepatan | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`; trik yang sama, variabel baru. |

## Catatan produksi: Flux.1-schnell adalah pencocokan aliran paling cepat

Keunggulan produksi pencocokan aliran adalah Flux.1-schnell — DiT pencocokan aliran yang disaring menjadi 1-4 langkah inference sekaligus menjaga kualitas tingkat pengembangan Flux. Notebook "Jalankan Flux pada mesin 8 GB" Niels adalah resep penerapan referensi: enkode T5 + CLIP, denoise MMDiT terkuantisasi (dalam 4 langkah untuk schnell vs 50 untuk dev), dekode VAE. Akuntansi biaya:

| Varian | Langkah | Latensi pada 1024² di L4 | Total FLOP (relatif) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (mentah) | 50 | ~15 detik | 1,0× |
| Fluks.1-schnell | 4 | ~1,2 dtk | 0,08× (12× lebih cepat) |
| SDXL-basis | 30 | ~4 detik | 0,25× |
| SDXL-Lightning 2 langkah | 2 | ~0,3 dtk | 0,03× |

Aturan produksi: **basis pencocokan aliran + distilasi = default tahun 2026 untuk teks-ke-gambar yang cepat.** Setiap vendor besar mengirimkan kombo ini: SD3-Turbo (SD3 + aliran + distilasi), Flux-schnell (Flux-dev + pelurusan aliran yang diperbaiki), CogView-4-Flash. Basis difusi murni hanya ada untuk pos pemeriksaan lama.

## Bacaan Lanjutan

- [Liu, Gong, Liu (2022). Aliran Lurus dan Cepat: Belajar Menghasilkan dan Mentransfer Data dengan Aliran yang Diperbaiki](https://arxiv.org/abs/2209.03003) — aliran yang diperbaiki.
- [Lipman dkk. (2023). Pencocokan Aliran untuk Pemodelan Generatif](https://arxiv.org/abs/2210.02747) — pencocokan aliran.
- [Esser dkk. (2024). Menskalakan Transformer Aliran yang Diperbaiki untuk Sintesis Gambar Resolusi Tinggi](https://arxiv.org/abs/2403.03206) — SD3, memperbaiki aliran dalam skala besar.
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — kerangka umum yang mencakup difusi FM +.
- [Lagu dkk. (2023). Model Konsistensi](https://arxiv.org/abs/2303.01469) — distilasi / aliran distilasi 1 langkah.
- [Sauer dkk. (2023). Distilasi Difusi Adversarial (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — varian turbo.
- [Laboratorium Hutan Hitam (2024). Model Flux.1](https://blackforestlabs.ai/announcing-black-forest-labs/) — pencocokan aliran dalam produksi.
