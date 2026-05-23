# GAN — Generator vs Diskriminator

> Trik Goodfellow pada tahun 2014 adalah mengabaikan kepadatan sepenuhnya. Dua jaringan. Seseorang membuat palsu. Seseorang menangkap mereka. Mereka berjuang sampai yang palsu tidak bisa dibedakan dari yang asli. Ini seharusnya tidak berhasil. Seringkali tidak. Jika hal ini terjadi, sampelnya masih menjadi yang paling tajam dalam literatur untuk domain sempit.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 3 · 02 (Backprop), Fase 3 · 08 (Optimizer), Fase 8 · 02 (VAE)
**Waktu:** ~75 menit

## Masalah

VAE menghasilkan sample buram karena kehilangan decoder MSE-nya adalah Bayes-optimal untuk gambar *rata-rata* — dan rata-rata dari banyak digit yang masuk akal adalah digit fuzzy. kamu menginginkan loss yang memberi imbalan *masuk akal*, bukan kedekatan piksel dengan satu target. Tidak ada bentuk tertutup untuk masuk akal. kamu harus mempelajarinya.

Ide Goodfellow: latih pengklasifikasi `D(x)` untuk membedakan gambar asli dari gambar palsu. Latih generator `G(z)` untuk menipu `D`. Sinyal loss untuk `G` adalah apa pun yang menurut `D` saat ini membuat sesuatu terlihat nyata. Sinyal ini diperbarui seiring dengan peningkatan `G`, mengejar target bergerak. Jika kedua jaringan bertemu, `G` telah mempelajari distribusi data tanpa pernah menuliskan `log p(x)`.

Ini adalah training permusuhan. Matematika adalah permainan minimax:

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

Pada tahun 2026 GAN tidak lagi menjadi generator SOTA (pencocokan difusi dan aliran memakan mahkota itu). Namun StyleGAN 2/3 tetap menjadi model wajah paling tajam yang pernah dikirimkan, diskriminator GAN digunakan sebagai *loss persepsi* dalam training difusi, dan training adversarial mendukung distilasi 1 langkah cepat (SDXL-Turbo, SD3-Turbo, LCM) yang memungkinkan kamu mengirimkan difusi waktu nyata.

## Konsep

![Training GAN: generator dan diskriminator di minimax](../assets/gan.svg)

**Generator `G(z)`.** Memetakan vector derau `z ~ N(0, I)` ke sample `x̂`. Jaringan berbentuk decoder (konv padat atau dialihkan).

**Diskriminator `D(x)`.** Memetakan sample ke probabilitas scalar (atau skor). Asli → 1, palsu → 0.

**Rugi.** Dua pembaruan bergantian:

- **Latihan `D`:** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`. Entropi silang biner pada nyata=1, palsu=0.
- **Latihan `G`:** `loss_G = -log D(G(z))`. Ini adalah bentuk *tidak jenuh* yang digunakan Goodfellow (asli `log(1 - D(G(z)))` menjenuhkan dan mematikan gradient ketika `D` yakin).

**Lingkaran training.** Satu langkah `D`, satu langkah `G`. Mengulang.

**Mengapa berhasil.** Jika `G` sangat cocok dengan `p_data`, maka `D` tidak bisa lebih baik daripada kebetulan dan menghasilkan 0,5 di semua tempat; `G` tidak lagi mendapat gradient. Keseimbangan.

**Mengapa rusak.** Mode runtuh (`G` menemukan satu mode `D` tidak dapat mengklasifikasikan dan mencetaknya selamanya), gradient hilang (`D` belajar terlalu cepat dan `log D` jenuh), ketidakstabilan training (learning rate, ukuran batch, apa pun).

## Varian yang membuat GAN berfungsi| Tahun | Inovasi | Perbaiki |
|------|------------|-----|
| 2015 | DCGAN | Konv/dekonv, norm batch, LeakyReLU — arsitektur stabil pertama. |
| 2017 | WGAN, WGAN-GP | Gantikan BCE dengan distance Wasserstein + penalti gradient. Memperbaiki gradient yang hilang. |
| 2017 | Normalisasi spektral | Lipschitz mengikat diskriminatornya. Masih digunakan pada tahun 2026 diskriminator. |
| 2018 | GAN Progresif | Latih resolusi rendah terlebih dahulu, tambahkan layer. Hasil megapiksel pertama. |
| 2019 | GayaGAN / GayaGAN2 | Jaringan pemetaan + norm contoh adaptif. Canggih untuk fotorealisme domain tetap. |
| 2021 | GayaGAN3 | Bebas alias, setara terjemahan — masih menjadi standar emas pada tahun 2026. |
| 2022 | GayaGAN-XL | Bersyarat, sadar kelas, skala lebih besar. |
| 2024 | R3GAN | Mengubah merek dengan regularisasi yang lebih kuat; bekerja pada 1024² tanpa trik. |

## Build

`code/main.py` melatih GAN kecil pada data 1-D: campuran dua Gaussian. Generator dan diskriminator adalah MLP layer tunggal yang tersembunyi. Kami menerapkan loop maju, mundur, dan minimax dengan tangan. Tujuannya adalah untuk melihat dua mode kegagalan utama (mode runtuh + gradient hilang) yang terjadi.

### Langkah 1: loss yang tidak jenuh

Loss vanilla Goodfellow `log(1 - D(G(z)))` menjadi 0 ketika D mengklasifikasikan palsu G sebagai palsu dengan keyakinan tinggi. Pada titik ini, gradient untuk G pada dasarnya adalah nol — G tidak dapat ditingkatkan. Bentuk non-jenuh `-log D(G(z))` mempunyai asimtot yang berlawanan: ia meledak ketika D percaya diri, memberikan G sinyal yang kuat.

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### Langkah 2: satu langkah diskriminator per langkah generator

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

Palsu baru untuk G, jika tidak, gradiennya akan basi.

### Langkah 3: perhatikan mode runtuh

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

Gejala kanonik: salah satu dari dua mode nyata berhenti dihasilkan. Diskriminator berhenti mengoreksinya karena tidak pernah dianggap palsu.

## Jebakan

- **Diskriminator terlalu kuat.** Kurangi learning rate D sebanyak 2-5x, atau tambahkan noise pada instance/layer. Jika D mencapai akurasi >95%, G mati.
- **Generator mengingat suatu mode.** Tambahkan noise ke input D, gunakan layer diskriminator minibatch, atau alihkan ke WGAN-GP.
- **Statistik kebocoran norm batch.** Batch asli + batch palsu yang mengalir melalui layer BN yang sama menggabungkan statistiknya. Gunakan norm contoh atau norm spektral sebagai gantinya.
- **Game dengan skor awal.** FID dan IS menimbulkan gangguan pada jumlah sample yang sedikit. Gunakan sample ≥10k pada eval.
- **Pengambilan sample sekali pakai adalah kebohongan untuk tugas bersyarat.** kamu masih memerlukan skala CFG, trik pemotongan, dan pengambilan sample ulang untuk mendapatkan hasil yang dapat digunakan.

## Pakai

Tumpukan GAN 2026:

| Situasi | Pilih |
|-----------|------|
| Wajah manusia fotoreal, pose tetap | StyleGAN3 (paling tajam, terkecil) |
| Anime / wajah bergaya | StyleGAN-XL atau LoRA Difusi Stabil |
| Terjemahan gambar-ke-gambar | Pix2Pix / CycleGAN (Fase 8 · 04) atau ControlNet (Fase 8 · 08) |
| Teks-ke-gambar 1 langkah cepat | Distilasi difusi permusuhan (SDXL-Turbo, SD3-Turbo) |
| Hilangnya persepsi di dalam pelatih difusi | Diskriminator GAN kecil pada pemotongan gambar |
| Segala sesuatu yang multimodal, terbuka | Jangan — gunakan difusi atau pencocokan aliran |

GAN tajam tapi sempit. Setelah domain kamu terbuka — foto, teks acak, video — beralihlah ke difusi. Trik permusuhan hidup sebagai sebuah komponen (kehilangan persepsi, penyulingan), bukan generator yang berdiri sendiri.

## KirimkanSimpan `outputs/skill-gan-debugger.md`. Skill mengambil proses GAN yang gagal (kurva loss, grid sample, ukuran dataset) dan menghasilkan daftar peringkat kemungkinan penyebab, perbaikan satu baris, dan protokol eksekusi ulang.

## Latihan

1. **Mudah.** Jalankan `code/main.py` dengan pengaturan stok. Kemudian atur `D_LR = 5 * G_LR` dan jalankan kembali. Seberapa cepat loss G turun hingga konstan?
2. **Sedang.** Ganti loss Goodfellow BCE dengan loss WGAN: `loss_D = E[D(fake)] - E[D(real)]`, `loss_G = -E[D(fake)]`, dan klip weight D ke `[-0.01, 0.01]`. Apakah training lebih stabil? Bandingkan konvergensi jam dinding.
3. **Sulit.** Perluas contoh 1-D ke data 2-D (campuran 8 Gaussian pada sebuah cincin). Lacak berapa banyak dari 8 mode yang ditangkap generator pada langkah 1k, 5k, 10k. Terapkan diskriminasi minibatch dan pengukuran ulang.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Pembangkit | "G" | Jaringan kebisingan-ke-sample, `G: z → x̂`. |
| Diskriminator | "D" | Pengklasifikasi `D: x → [0, 1]`, asli vs palsu. |
| Minimaks | "Permainan" | `min_G max_D` dari tujuan bersama. |
| Loss tidak jenuh | "Perbaikannya" | Gunakan `-log D(G(z))` untuk G, bukan `log(1 - D(G(z)))`. |
| Mode runtuh | "G hafal satu hal" | Generator menghasilkan sedikit output berbeda meskipun datanya beragam. |
| WGAN | "Wasserstein" | Gantikan BCE dengan distance Penggerak Bumi + penalti gradient; gradient yang lebih halus. |
| Norm spektral | "Trik Lipschitz" | Batasi norm weight D untuk membatasi kemiringannya; menstabilkan training. |
| GayaGAN | "Yang berhasil" | Jaringan pemetaan + AdaIN; terbaik di kelasnya untuk wajah, masih di tahun 2026. |

## Catatan produksi: inference satu kali adalah keuntungan abadi GAN

GAN tidak lagi unggul dalam hal kualitas sample untuk pembuatan domain terbuka, namun mereka tetap unggul dalam biaya inference. Dalam kosakata literatur inference produksi, GAN memiliki:

- **Tanpa pengisian awal, tanpa phase dekode.** Satu tiket ke depan `G(z)`. TTFT ≈ latensi total.
- **Tidak ada tekanan cache KV.** Satu-satunya status adalah weight. Ukuran batch dibatasi oleh memori activation, bukan cache.
- **Pengelompokan berkelanjutan yang sepele.** Karena setiap permintaan menggunakan FLOP tetap yang sama, kumpulan statis pada hunian target server biasanya optimal. Tidak diperlukan penjadwal dalam penerbangan.

Inilah sebabnya mengapa distilasi GAN (SDXL-Turbo, SD3-Turbo, ADD, LCM) adalah teknik dominan untuk teks-ke-gambar cepat pada tahun 2026: teknik ini meruntuhkan jalur pipa difusi 20-50 langkah menjadi 1-4 gaya GAN lintasan maju sambil menjaga distribusi basis difusi. Loss yang merugikan bertahan sebagai tombol waktu training untuk mengubah generator yang lambat menjadi generator yang cepat.

## Bacaan Lanjutan

- [Teman Baik dkk. (2014). Jaring Adversarial Generatif](https://arxiv.org/abs/1406.2661) — makalah GAN asli.
- [Radford dkk. (2015). Pembelajaran Representasi Tanpa Pengawasan dengan DCGAN](https://arxiv.org/abs/1511.06434) — arsitektur stabil pertama.
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN.
- [Miyato dkk. (2018). Normalisasi Spektral untuk GAN](https://arxiv.org/abs/1802.05957) — SN.
- [Karras dkk. (2020). Menganalisis dan Meningkatkan Kualitas Gambar StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras dkk. (2021). Jaringan Adversarial Generatif Bebas Alias](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Sauer dkk. (2023). Distilasi Difusi Adversarial](https://arxiv.org/abs/2311.17042) — SDXL-Turbo.
