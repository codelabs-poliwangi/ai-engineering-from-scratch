# Generasi 3D

> 3D adalah modalitas dengan pengaruh 2D ke 3D yang paling kuat. Terobosan tahun 2023 adalah 3D Gaussian Splatting. Difusi multi-tampilan layer dorong generatif 2024-2026 + rekonstruksi 3D di bagian atas untuk menghasilkan objek dan pemandangan dari satu prompt atau foto.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 4 (Visi), Fase 8 · 07 (Difusi Laten)
**Waktu:** ~45 menit

## Masalah

Konten 3D menyakitkan:

- **Representasi.** Jerat, awan titik, kisi voxel, bidang distance bertanda tangan (SDF), bidang pancaran saraf (NeRF), Gaussian 3D. Masing-masing memiliki trade-off.
- **Kelangkaan data.** ImageNet memiliki 14 juta gambar. Dataset 3D bersih terbesar (Objaverse-XL, 2023) memiliki ~10 juta objek, sebagian besar berkualitas rendah.
- **Memori.** Kotak voxel 512³ adalah 128 juta voxel; pemandangan yang berguna NeRF membutuhkan 1 juta sample/sinar. Generasi lebih sulit daripada rekonstruksi.
- **Pengawasan.** Untuk gambar 2D, kamu memiliki piksel. Untuk 3D, kamu biasanya memiliki beberapa tampilan 2D dan harus meningkatkannya ke 3D.

Tumpukan tahun 2026 memisahkan kedua masalah tersebut. Pertama, buat *gambar multi-tampilan 2D* dengan model difusi. Kedua, sesuaikan *representasi 3D* (biasanya percikan Gaussian) ke gambar tersebut.

## Konsep

![Generasi 3D: difusi multi-tampilan + rekonstruksi 3D](../assets/3d-generasi.svg)

### Representasi: Percikan Gaussian 3D (Kerbl et al., 2023)

Merepresentasikan pemandangan sebagai awan ~1 juta Gaussians 3D. Masing-masing memiliki 59 parameter: posisi (3), kovarians (6, atau quaternion 4 + skala 3), opacity (1), warna harmonik bola (48 pada derajat 3, 3 pada derajat 0).

Rendering = proyeksi + pengomposisian alpha. Cepat (~100 fps pada 1080p pada 4090). Dapat dibedakan. Cocokkan berdasarkan gradient descent dengan foto kebenaran dasar. Sebuah adegan dapat ditampung dalam 5-30 menit pada GPU konsumen.

Dua inovasi teratas pada tahun 2023-2024:
- **Percikan Gaussian Generatif.** Model seperti LGM, LRM, InstantMesh memprediksi awan Gaussian langsung dari satu atau beberapa gambar.
- **4D Gaussian Splatting.** Gaussian dengan offset per frame untuk pemandangan dinamis.

### Difusi multi-tampilan

Sempurnakan model difusi gambar yang telah dilatih sebelumnya untuk menghasilkan beberapa tampilan konsisten dari objek yang sama dari prompt teks atau gambar tunggal. Zero123 (Liu et al., 2023), MVDream (Shi et al., 2023), SV3D (Stabilitas, 2024), CAT3D (Google, 2024). Biasanya menghasilkan 4-16 tampilan di sekitar objek, diangkat ke 3D melalui Gaussian splatting atau NeRF.

### Pipeline pipa teks-ke-3D

| Model | Input | Output | Waktu |
|-------|-------|--------|------|
| DreamFusion (2022) | teks | NeRF melalui SDS | ~1 jam per aset |
| Ajaib3D | teks | jala + tekstur | ~40 menit |
| Bentuk-E (OpenAI, 2023) | teks | 3D implisit | ~1 menit |
| SJC / Pemimpi Produktif | teks | NeRF / jaring | ~30 menit |
| LRM (Meta, 2023) | gambar | pesawat tiga | ~5 detik |
| Mesh Instan (2024) | gambar | jala | ~10 detik |
| SV3D (Stabilitas, 2024) | gambar | pandangan baru | ~2 menit |
| CAT3D (Google, 2024) | 1-64 gambar | NeRF 3D | ~1 menit |
| TripoSR (2024) | gambar | jala | ~1 detik |
| Meshy 4 (2025) | teks + gambar | jaring PBR | ~30 detik |
| Rodin Gen-1.5 (2025) | teks + gambar | jaring PBR | ~60 detik |
| Tencent Hunyuan3D 2.0 (2025) | gambar | jala | ~30 detik |

Arah 2025-2026: model direct text-to-mesh dengan material PBR yang cocok untuk mesin game. Langkah perantara difusi multi-tampilan masih merupakan resep dengan kinerja terbaik untuk objek umum.

### NeRF (untuk konteks)Bidang Cahaya Neural (Mildenhall dkk., 2020). MLP kecil membutuhkan `(x, y, z, view direction)` dan menghasilkan `(color, density)`. Render dengan mengintegrasikan sepanjang sinar. Mengalahkan sintesis tampilan novel berbasis mesh dalam kualitas tetapi 100-1000x lebih lambat untuk dirender. Digantikan oleh Gaussian splatting untuk sebagian besar penggunaan waktu nyata tetapi masih dominan dalam penelitian.

## Build

`code/main.py` mengimplementasikan mainan 2D "Gaussian splatting" yang cocok: mewakili gambar target sintetik (gradient halus) sebagai jumlah dari splat Gaussian 2D. Optimalkan posisi, warna, dan kovarians dengan gradient descent agar sesuai dengan target. kamu melihat dua operasi inti: render ke depan (percikan + komposit alpha) dan penyesuaian berdasarkan gradient descent.

### Langkah 1: Percikan Gaussian 2D

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Langkah 2: render dengan menjumlahkan percikan

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

Percikan Gaussian 3D nyata mengurutkan Gaussian berdasarkan kedalaman dan komposit alpha secara berurutan. Mainan 2D kami hanya merangkum.

### Langkah 3: menyesuaikan dengan gradient descent

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## Jebakan

- **Inkonsistensi tampilan.** Jika kamu membuat 4 tampilan secara terpisah dan keduanya berbeda pendapat mengenai struktur objek, kesesuaian 3D akan menjadi buram. Cara mengatasinya: difusi multi-tampilan dengan attention bersama.
- **Halusinasi sisi belakang.** Gambar tunggal → 3D harus menciptakan sisi yang tak terlihat. Kualitas sangat bervariasi.
- **Ledakan percikan Gaussian.** Latihan tanpa batasan bertambah hingga 10 juta percikan dan pakaian luar. Heuristik pemadatan + pemangkasan (dari makalah asli 3D-GS) sangat penting.
- **Masalah topologi.** Jerat dari bidang implisit (SDF) sering kali memiliki lubang atau perpotongan sendiri. Jalankan remesher (misalnya remesh voxel blender) sebelum pengiriman.
- **Lisensi training data.** Objaverse memiliki lisensi campuran; penggunaan komersial bervariasi per model.

## Pakai

| Tugas | pilihan 2026 |
|------|-----------|
| Rekonstruksi pemandangan dari foto | Percikan Gaussian (3DGS, Gsplat, Scaniverse) |
| Objek teks-ke-3D untuk game | Meshy 4 atau Rodin Gen-1.5 (output PBR) |
| Gambar-ke-3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| Sintesis tampilan novel dari beberapa gambar | CAT3D, SV3D |
| Rekonstruksi pemandangan dinamis | Percikan Gaussian 4D |
| Avatar / manusia berpakaian | Avatar Gaussian, PELUKAN |
| Penelitian / SOTA | Apapun yang turun minggu lalu |

Untuk mengirimkan produksi 3D dalam game atau jalur e-commerce: Meshy 4 atau Rodin Gen-1.5 output mesh PBR yang langsung masuk ke Unity / Unreal.

## Kirim

Simpan `outputs/skill-3d-pipeline.md`. Keterampilan mengambil ringkasan 3D (input: teks / satu gambar / beberapa gambar; output: mesh / splat / NeRF; penggunaan: render / game / VR) dan output: pipeline (difusi multi-tampilan + fit, atau model mesh langsung), model dasar, anggaran iterasi, pasca-pemrosesan topologi, pipeline material yang dibutuhkan.

## Latihan

1. **Mudah.** Jalankan `code/main.py` dengan 4, 16, 64 Gaussians. Laporkan UMK final vs target.
2. **Medium.** Perluas ke warna Gaussians (RGB). Pastikan rekonstruksi cocok dengan pola warna target.
3. **Sulit.** Menggunakan gsplat atau Nerfstudio, buat ulang objek nyata dari pengambilan 50 foto. Laporkan waktu fit dan SSIM akhir pada penayangan yang ditunda.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Percikan Gaussian 3D | "3DGS" | Adegan sebagai awan Gaussians 3D; render alpha-komposit yang dapat dibedakan. |
| NeRF | "Bidang pancaran saraf" | MLP yang menghasilkan warna + kepadatan pada titik 3D; render dengan integrasi sinar. |
| Pesawat Tiga | "Tiga bidang 2-D" | Faktorkan 3D menjadi tiga kisi feature sejajar sumbu 2D; lebih murah dibandingkan volumetrik. |
| SD | "Sampling distilasi skor" | Latih model 3D dengan menggunakan skor difusi 2D sebagai gradient semu. |
| Difusi multi-tampilan | "Banyak penayangan sekaligus" | Model difusi yang menghasilkan kumpulan tampilan kamera yang konsisten. |
| PBR | "Render berbasis fisik" | Bahan dengan pipeline albedo, kekasaran, metalik, normal. |
| Densifikasi | "Tumbuhkan percikan" | Heuristik training 3DGS: perpecahan / kloning percikan di wilayah gradient tinggi. |

## Catatan produksi: 3D belum memiliki media bersama

Berbeda dengan gambar (difusi laten + DiT) dan video (DiT spasialtemporal), 3D tidak memiliki waktu proses tunggal yang dominan pada tahun 2026. Pohon keputusan produksi bercabang pada representasi:

- **NeRF / triplane.** Inference adalah ray-marching + penerusan MLP per sample. Render 512² memerlukan jutaan penerusan MLP. Kumpulkan sample sinar secara agresif; SDPA/xformers berlaku.
- **Difusi multi-tampilan + rekonstruksi LRM.** Pipeline pipa dua phase. Phase 1 (multi-view DiT) adalah server difusi seperti Lesson 07. Phase 2 (Transformer LRM) adalah one-shot forward pass atas pandangan. Profil latensi keseluruhannya adalah "difusi + satu kali" — pilih layanan primitif per phase yang sesuai.
- **SDS / DreamFusion.** Optimization per aset, bukan inference. Build pekerjaan, bukan minta penangan.

Untuk sebagian besar produk tahun 2026, jawaban yang tepat adalah "menjalankan model difusi multi-tampilan berdasarkan permintaan, merekonstruksi ke 3DGS secara asinkron, menyajikan 3DGS untuk tampilan waktu nyata". Ini membagi weight kerja dengan rapi antara server inference GPU (cepat) dan optimizer offline (lambat).

## Bacaan Lanjutan

- [Mildenhall dkk. (2020). NeRF: Mewakili Pemandangan sebagai Bidang Cahaya Neural](https://arxiv.org/abs/2003.08934) — NeRF.
- [Kerbl dkk. (2023). Percikan Gaussian 3D untuk Rendering Bidang Cahaya Secara Real-Time](https://arxiv.org/abs/2308.04079) — 3DGS.
- [Poole dkk. (2022). DreamFusion: Teks-ke-3D menggunakan Difusi 2D](https://arxiv.org/abs/2209.14988) — SDS.
- [Liu dkk. (2023). Zero-1-to-3: Zero-shot Satu Gambar ke Objek 3D](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi dkk. (2023). MVDream](https://arxiv.org/abs/2308.16512) — difusi multi-tampilan.
- [Hong dkk. (2023). LRM: Model Rekonstruksi Besar untuk Gambar Tunggal menjadi 3D](https://arxiv.org/abs/2311.04400) — LRM.
- [Gao dkk. (2024). CAT3D: Buat Apa Pun dalam 3D dengan Model Difusi Multi-Tampilan](https://arxiv.org/abs/2405.10314) — CAT3D.
- [Stabilitas AI (2024). Video Stabil 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D.
