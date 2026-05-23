# GAN & Pix2Pix bersyarat

> Peluang besar pertama pada tahun 2014-2017 adalah mengendalikan apa yang dilakukan GAN. Lampirkan label, atau gambar, atau kalimat. Pix2Pix mengerjakan versi gambar dan masih mengalahkan setiap model teks-ke-gambar umum pada tugas gambar-ke-gambar yang sempit.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 8 · 03 (GAN), Fase 4 · 06 (U-Net), Fase 3 · 07 (CNN)
**Waktu:** ~75 menit

## Masalah

GAN tanpa syarat mengambil sample wajah secara sewenang-wenang. Berguna untuk demo, tidak berguna dalam produksi. kamu ingin: *memetakan sketsa ke foto*, *memetakan peta ke foto udara*, *memetakan pemandangan siang hari ke malam hari*, *mewarnai gambar skala abu-abu*. Dalam semua ini, kamu diberikan gambar input `x` dan harus mengeluarkan `y` dengan beberapa korespondensi semantik. Ada banyak `y` yang masuk akal per `x`. Kesalahan kuadrat rata-rata membuat mereka menjadi bubur. Loss yang merugikan tidak terjadi, karena "tampak nyata" itu tajam.

GAN bersyarat (Mirza & Osindero, 2014) menambahkan kondisi `c` sebagai input ke `G` dan `D`. Pix2Pix (Isola et al., 2017) mengkhususkan hal ini: kondisi adalah gambar input penuh, generator adalah U-Net, diskriminator adalah pengklasifikasi *berbasis patch* (PatchGAN), dan loss adalah adversarial + L1. Resep tersebut mengungguli model teks-ke-gambar dari awal pada domain gambar-ke-gambar yang sempit bahkan pada tahun 2026 karena resep tersebut dilatih pada *data berpasangan* — kamu memiliki sinyal yang tepat yang kamu perlukan.

## Konsep

![Pix2Pix: generator U-Net, diskriminator PatchGAN](../assets/pix2pix.svg)

**G Bersyarat** `G(x, z) → y`. Di Pix2Pix, `z` putus sekolah di dalam G (tidak ada derau input — Isola menemukan derau eksplisit diabaikan).

**Kondisi D.** `D(x, y) → [0, 1]`. Inputnya adalah *pasangan* (kondisi, output). Inilah perbedaan utamanya: D harus menilai apakah `y` konsisten dengan `x`, bukan hanya apakah `y` terlihat nyata.

**Generator U-Net.** Encoder-decoder dengan koneksi lewati melintasi kemacetan. Penting untuk tugas-tugas di mana input dan output berbagi struktur tingkat rendah (tepi, siluet). Tanpa lompatan, detail frekuensi tinggi akan hilang.

**Diskriminator PatchGAN.** Daripada mengeluarkan satu skor asli/palsu, D mengeluarkan kisi `N×N` yang mana setiap sel menilai bidang reseptif berukuran ~70×70 piksel. Rata-rata. Ini adalah asumsi lapangan acak Markov: realisme bersifat lokal. Jauh lebih cepat untuk dilatih, lebih sedikit parameter, output lebih tajam.

**Loss.**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

Istilah L1 menstabilkan training dan mendorong G menuju target yang diketahui. L1 memberikan tepi yang lebih tajam dari L2 (median, bukan berarti). `λ = 100` adalah default Pix2Pix.

## CycleGAN — bila kamu tidak memiliki pasangan

Pix2Pix memerlukan data `(x, y)` yang dipasangkan. CycleGAN (Zhu et al., 2017) menghilangkan persyaratan ini dengan mengorbankan loss tambahan: loss *konsistensi siklus*. Dua generator `G: X → Y` dan `F: Y → X`. Latih mereka jadi `F(G(x)) ≈ x` dan `G(F(y)) ≈ y`. Ini memungkinkan kamu menerjemahkan kuda menjadi zebra, musim panas ke musim dingin, tanpa contoh berpasangan.

Pada tahun 2026, gambar-ke-gambar yang tidak berpasangan sebagian besar dilakukan melalui difusi (ControlNet, IP-Adapter) dibandingkan CycleGAN, namun gagasan konsistensi siklus tetap bertahan di hampir setiap makalah adaptasi domain yang tidak berpasangan.

## Build`code/main.py` mengimplementasikan GAN bersyarat kecil pada data 1-D. Kondisi `c` adalah label kelas (0 atau 1). Tugasnya: menghasilkan sample dari distribusi bersyarat untuk kelas tertentu.

### Langkah 1: tambahkan kondisi ke input G dan D

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

Pengkodean one-hot adalah cara paling sederhana. Model yang lebih besar menggunakan embeddings yang dipelajari, modulasi FiLM, atau attention silang.

### Langkah 2: latih bersyarat

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

Generator harus sesuai dengan distribusi sebenarnya *untuk kondisi tertentu*, bukan distribusi marginal.

### Langkah 3: verifikasi output per kelas

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## Jebakan

- **Kondisi diabaikan.** G belajar meminggirkan, D tidak pernah memberikan penalti karena sinyal kondisi lemah. Cara mengatasinya: kondisi D lebih agresif (layer awal, bukan hanya akhir), gunakan diskriminator proyeksi (Miyato & Koyama 2018).
- **Weight L1 terlalu rendah.** G beralih ke output yang tampak nyata, bukan output yang sebenarnya. Mulai λ≈100 untuk tugas bergaya Pix2Pix.
- **Weight L1 terlalu tinggi.** G menghasilkan output buram karena L1 masih merupakan norm L_p. Anil setelah latihan stabil.
- **Kebocoran kebenaran dasar di D.** Gabungkan `(x, y)` sebagai input D, bukan hanya `y`. Tanpa ini D tidak dapat memeriksa konsistensi.
- **Mode penciutan per kelas.** Setiap kelas dapat diciutkan secara mandiri. Jalankan pemeriksaan keragaman bersyarat kelas.

## Pakai

Status tugas gambar-ke-gambar tahun 2026:

| Tugas | Pendekatan terbaik |
|------|---------------|
| Sketsa → foto, domain sama, data berpasangan | Pix2Pix / Pix2PixHD (masih cepat, masih tajam) |
| Sketsa → foto, tidak berpasangan | ControlNet dengan model pengkondisian Scribble |
| Segmen semantik → foto | SPADE / GauGAN2 atau SD + ControlNet-Seg |
| Perpindahan gaya | Difusi dengan IP-Adapter atau LoRA; Metode GAN adalah warisan |
| Kedalaman → foto | Kedalaman ControlNet atas Difusi Stabil |
| Resolusi super | Real-ESRGAN (GAN), ESRGAN-Plus, atau SD-Upscale (difusi) |
| Pewarnaan | ColTran, pewarna berbasis difusi, atau warna Pix2Pix |
| Siang → malam hari, musim, cuaca | CycleGAN atau berbasis ControlNet |

Pix2Pix tetap menjadi alat yang tepat ketika (a) kamu memiliki ribuan contoh berpasangan, (b) tugasnya sempit dan dapat diulang, dan (c) kamu memerlukan inference cepat. Pada tugas domain terbuka umum, difusi menang.

## Kirim

Simpan `outputs/skill-img2img-chooser.md`. Keterampilan mengambil deskripsi tugas, ketersediaan data (berpasangan vs tidak berpasangan, N sample), dan latensi/anggaran kualitas, lalu output: pendekatan (Pix2Pix, CycleGAN, varian ControlNet, SDXL + IP-Adapter), persyaratan training data, biaya inference, dan protokol eval (LPIPS, FID, khusus tugas).

## Latihan

1. **Mudah.** Ubah `code/main.py` untuk menambahkan kelas ketiga. Konfirmasikan G masih memetakan kebisingan setiap kelas ke mode yang benar.
2. **Sedang.** Ganti L1 dengan gaya perseptual yang hilang dalam setelan 1-D (misalnya D beku kecil yang berfungsi sebagai ekstraktor feature). Apakah ini mengubah ketajaman distribusi bersyarat?
3. **Sulit.** Buat sketsa CycleGAN dalam pengaturan 1-D: dua distribusi, dua generator, kehilangan siklus. Tunjukkan bahwa ia belajar memetakan di antara keduanya tanpa data berpasangan.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| GAN Bersyarat | "GAN dengan label" | G(z, c), D(x, c). Kedua jaringan melihat kondisinya. |
| Pix2Pix | "GAN Gambar-ke-gambar" | Memasangkan cGAN dengan loss U-Net G dan PatchGAN D + L1. |
| U-Net | "Encoder-decoder dengan lompatan" | Jaringan konv simetris; lompatan mempertahankan frekuensi tinggi. |
| PatchGAN | "Pengklasifikasi realisme lokal" | D mengeluarkan skor per patch, bukan skor global. |
| SiklusGAN | "Terjemahan gambar tidak berpasangan" | Hilangnya konsistensi siklus dua G +; tidak ada data berpasangan. |
| SPADE | "GauGAN" | Menormalkan activation perantara dengan peta semantik; segmentasi-ke-gambar. |
| Film | "Modulasi linier berdasarkan feature" | Transformasi affine per feature dari kondisi; pengkondisian murah. |

## Catatan produksi: Pix2Pix sebagai garis dasar yang terikat latensi

Saat kamu memasangkan data dan tugas yang sempit (sketsa → render, peta semantik → foto, siang → malam), inference satu jepretan Pix2Pix mengalahkan difusi dengan urutan besarnya latensi. Perbandingan produksi biasanya:

| Jalur | Langkah | Latensi tipikal pada 512² pada satu L4 |
|------|-------|----------------------------------------|
| Pix2Pix (penerusan U-Net) | 1 | ~30 mdtk |
| SD-Inpaint atau SD-Img2Img | 20 | ~1,2 dtk |
| SDXL-Turbo Img2Img | 1-4 | ~0,15-0,35 dtk |
| Basis ControlNet + SDXL | 20-30 | ~3-5 detik |

Pix2Pix menang pada throughput dalam batch statis (setiap permintaan adalah FLOP yang sama). Difusi menang dalam hal kualitas dan generalisasi. Permainan modern sering kali mengirimkan model sulingan gaya Pix2Pix untuk tugas sempit dan fallback difusi untuk input ekor.

## Bacaan Lanjutan

- [Mirza & Osindero (2014). Jaring Adversarial Generatif Bersyarat](https://arxiv.org/abs/1411.1784) - makalah cGAN.
- [Isola dkk. (2017). Terjemahan Gambar-ke-Gambar dengan Jaringan Adversarial Bersyarat](https://arxiv.org/abs/1611.07004) — Pix2Pix.
- [Zhu dkk. (2017). Terjemahan Gambar-ke-Gambar Tidak Berpasangan menggunakan Jaringan Adversarial yang Konsisten Siklus](https://arxiv.org/abs/1703.10593) — CycleGAN.
- [Wang dkk. (2018). Sintesis Gambar Resolusi Tinggi dengan GAN Bersyarat](https://arxiv.org/abs/1711.11585) — Pix2PixHD.
- [Taman dkk. (2019). Sintesis Gambar Semantik dengan Normalisasi Adaptif Spasial](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN.
- [Miyato & Koyama (2018). cGAN dengan Diskriminator Proyeksi](https://arxiv.org/abs/1802.05637) — proyeksi D.
