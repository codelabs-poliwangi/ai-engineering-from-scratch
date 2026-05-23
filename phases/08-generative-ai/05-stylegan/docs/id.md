#GayaGAN

> Kebanyakan generator mengaduk `z` ke dalam setiap layer secara bersamaan. StyleGAN membaginya: pertama memetakan `z` ke perantara `w`, lalu *menyuntikkan* `w` di setiap tingkat resolusi melalui AdaIN. Perubahan tunggal ini mengurai ruang laten dan menjadikan wajah fotorealistik sebagai masalah terpecahkan selama tujuh tahun berturut-turut.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 8 · 03 (GAN), Fase 4 · 08 (Normalisasi), Fase 3 · 07 (CNN)
**Waktu:** ~45 menit

## Masalah

DCGAN memetakan `z` ke gambar melalui tumpukan konvolusi yang dialihkan. Masalah: `z` mengontrol segalanya — pose, pencahayaan, identitas, latar belakang — saling terkait. Bergerak sepanjang satu sumbu `z`, keempatnya berubah. kamu tidak dapat menanyakan model "orang yang sama, pose berbeda" karena representasinya tidak memperhitungkan hal tersebut.

Karras dkk. (2019, NVIDIA) mengusulkan: berhenti memasukkan `z` langsung ke layer konv. Masukkan tensor `4×4×512` konstan sebagai input jaringan. Learn MLP 8 lapis yang memetakan `z ∈ Z → w ∈ W`. Suntikkan `w` pada setiap resolusi melalui *normalisasi instance adaptif* (AdaIN): normalkan setiap peta feature konv, lalu skalakan dan geser berdasarkan proyeksi affine `w`. Tambahkan noise per layer untuk detail stokastik (pori-pori kulit, helai rambut).

Hasilnya: `W` memiliki sumbu ortogonal untuk "gaya tingkat tinggi" (pose, identitas) vs "gaya halus" (pencahayaan, warna). kamu dapat menukar gaya antara dua gambar dengan menggunakan `w` gambar A untuk tingkat resolusi rendah dan `w` gambar B untuk tingkat resolusi tinggi. Pengeditan tidak terkunci, penataan gaya lintas domain, dan seluruh lini penelitian "Inversi StyleGAN".

## Konsep

![StyleGAN: jaringan pemetaan + AdaIN + noise per layer](../assets/stylegan.svg)

**Pemetaan jaringan.** `f: Z → W`, MLP 8 lapis. `Z = N(0, I)^512`. `W` tidak dipaksa menjadi Gaussian — ia mempelajari bentuk yang diadaptasi dari data.

**Jaringan sintesis.** Dimulai dari konstanta yang dipelajari `4×4×512`. Setiap blok resolusi: `upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`. Resolusi ganda: 4, 8, 16, 32, 64, 128, 256, 512, 1024.

**AdaIN.**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

di mana `y_scale` dan `y_bias` berasal dari proyeksi affine `w`. Normalisasikan per peta feature, lalu ubah gaya. "Gaya" di sini adalah statistik urutan pertama dan kedua dari peta feature.

**Noise per layer.** Noise Gaussian pipeline tunggal ditambahkan ke setiap peta feature, diskalakan berdasarkan faktor per pipeline yang dipelajari. Mengontrol detail stokastik tanpa mempengaruhi struktur global.

**Trik pemotongan.** Pada inference, sample `z`, hitung `w = mapping(z)`, lalu `w' = ŵ + ψ·(w - ŵ)` dengan `ŵ` adalah mean `w` pada banyak sample. `ψ < 1` memperdagangkan keberagaman demi kualitas. Hampir setiap demo StyleGAN menggunakan `ψ ≈ 0.7`.

## GayaGAN 1 → 2 → 3| Versi | Tahun | Inovasi |
|---------|------|------------|
| GayaGAN | 2019 | Jaringan pemetaan + AdaIN + noise + pertumbuhan progresif. |
| GayaGAN2 | 2020 | Demodulasi weight menggantikan AdaIN (memperbaiki artefak tetesan); lewati/arsitektur sisa; regularisasi panjang jalur. |
| GayaGAN3 | 2021 | Konvolusi bebas alias + kernel ekuivalen; menghilangkan tekstur yang menempel pada kisi piksel. |
| GayaGAN-XL | 2022 | Kelas-bersyarat, 1024², ImageNet. |
| R3GAN | 2024 | Mengubah merek dengan reg yang lebih kuat; menutup kesenjangan difusi pada FFHQ-1024 dengan parameter 20x lebih sedikit. |

Pada tahun 2026 StyleGAN3 tetap menjadi default untuk (a) fotorealisme domain sempit pada FPS tinggi, (b) adaptasi domain beberapa foto (melatih dataset baru dengan 100 gambar, pemetaan beku), (c) pengeditan berbasis inversi (temukan `w` yang merekonstruksi foto asli, lalu mengedit `w`). Untuk teks-ke-gambar domain terbuka, ini bukanlah alatnya — difusi adalah alatnya.

## Build

`code/main.py` mengimplementasikan mainan "style-GAN lite" dalam 1-D: MLP pemetaan, fungsi sintesis yang mengambil vector konstanta yang dipelajari dan memodulasinya dengan skala/bias turunan `w`, dan noise per layer. Ini menunjukkan bahwa menyuntikkan `w` melalui modulasi affine cocok atau mengalahkan gabungan `z` ke dalam input generator.

### Langkah 1: memetakan jaringan

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### Langkah 2: normalisasi instance adaptif

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

Skala dan bias per peta feature berasal dari `w` melalui proyeksi linier.

### Langkah 3: kebisingan per layer

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

Sigma per pipeline dapat dipelajari.

## Jebakan

- **Artefak tetesan.** StyleGAN 1 menghasilkan tetesan gumpalan di peta feature karena AdaIN menghilangkan mean. Demodulasi weight StyleGAN 2 memperbaikinya dengan menskalakan weight konvolusi.
- **Tekstur menempel.** Tekstur StyleGAN 1 dan 2 mengikuti koordinat piksel, bukan koordinat objek (terlihat saat interpolasi). Konvolusi bebas alias StyleGAN 3 memperbaikinya dengan filter sinc berjendela.
- **Mode cakupan.** Pemotongan `ψ < 0.7` terlihat bersih namun sampelnya diambil dari kerucut yang sempit; gunakan `ψ = 1.0` jika kamu membutuhkan keberagaman.
- **Pembalikan bersifat lossy.** Pembalikan foto asli menjadi `W` biasanya dilakukan melalui optimization atau encoder (e4e, ReStyle, HyperStyle). Hasil melayang melalui banyak iterasi.

## Pakai

| Kasus penggunaan | Pendekatan |
|----------|----------|
| Wajah manusia fotoreal (anime, produk, sempit) | StyleGAN3 FFHQ / penyesuaian khusus |
| Pengeditan wajah dari foto | inversi e4e + arah StyleSpace / InterFaceGAN |
| Pertukaran wajah / pemeragaan | StyleGAN + encoder + pencampuran |
| Pipeline pipa Avatar | StyleGAN3 dengan ADA untuk penyesuaian data rendah |
| Adaptasi domain dari beberapa gambar | Bekukan jaringan pemetaan, sempurnakan sintesis |
| Generasi multi-modal atau berkondisi teks | Jangan — gunakan difusi |

Untuk demo tingkat produk yang jawabannya adalah "foto wajah seseorang", StyleGAN mengalahkan difusi dalam hal biaya inference (single forward pass, <10ms pada 4090) dan ketajaman untuk bilah kualitas yang sama.

## Kirim

Simpan `outputs/skill-stylegan-inversion.md`. Keterampilan mengambil foto asli dan menghasilkan: metode inversi (e4e / ReStyle / HyperStyle), perkiraan loss laten, anggaran pengeditan (seberapa jauh di `W` kamu dapat berpindah sebelum artefak), dan daftar arahan pengeditan yang diketahui baik (usia, ekspresi, pose).

## Latihan1. **Mudah.** Jalankan `code/main.py` dengan `adain_on=True` dan `adain_on=False`. Bandingkan penyebaran output untuk laten tetap vs laten terganggu.
2. **Sedang.** Menerapkan regularisasi pencampuran: untuk batch training, hitung `w_a`, `w_b`, dan terapkan `w_a` untuk paruh pertama sintesis dan `w_b` untuk paruh kedua. Apakah decoder mempelajari gaya yang terurai?
3. **Sulit.** Ambil model StyleGAN3 FFHQ yang telah dilatih sebelumnya (ffhq-1024.pkl). Temukan arah `w` yang mengontrol "senyuman" dengan melatih SVM pada sample berlabel; laporkan seberapa jauh kamu dapat mendorong sebelum identitas hilang.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Jaringan pemetaan | "MLP" | `f: Z → W`, 8 layer, memisahkan geometri laten dari statistik data. |
| ruang W | "Ruang gaya" | Output jaringan pemetaan; diurai secara kasar. |
| AdaIN | "Norm contoh adaptif" | Normalisasikan peta feature, lalu skalakan + geser sebanyak `w`-proyeksi. |
| Trik pemotongan | "Psi" | `w = mean + ψ·(w - mean)`, ψ<1 memperdagangkan keragaman demi kualitas. |
| Regularisasi panjang jalur | "PL Reg" | Menghukum perubahan besar pada gambar per perubahan unit di `w`; membuat `W` lebih lancar. |
| Demodulasi berat badan | "Perbaikan StyleGAN2" | Normalisasikan weight konv alih-alih activation; membunuh artefak tetesan. |
| Bebas alias | "Trik StyleGAN3" | Filter sin berjendela; menghilangkan tekstur yang menempel pada kisi piksel. |
| Inversi | "Temukan w untuk gambar nyata" | Optimalkan atau encode `x → w` jadi `G(w) ≈ x`. |

## Catatan produksi: mengapa StyleGAN masih dikirimkan pada tahun 2026

StyleGAN3 pada 4090 menghasilkan wajah FFHQ 1024² dalam waktu kurang dari 10 ms — `num_steps = 1`, tanpa dekode VAE, tanpa pass attention silang. Dalam istilah produksi, ini adalah latensi dasar untuk generator gambar apa pun. Pipeline dekode SDXL + VAE 50 langkah pada resolusi yang sama adalah ~3 detik. Ini merupakan selisih **300×**, dan untuk produk dengan domain sempit (layanan avatar, pipeline dokumen ID, pembuatan tampilan saham), produk ini unggul dalam TCO.

Dua konsekuensi operasional:

- **Tanpa penjadwal, tanpa batcher.** Batch statis pada hunian target sudah optimal. Pengelompokan berkelanjutan (penting untuk LLM dan difusi) tidak memberikan manfaat apa pun karena setiap permintaan memerlukan FLOP yang sama.
- **Pemotongan `ψ` adalah kenop pengaman.** `ψ < 0.7` sample dari kerucut sempit jangkauan jaringan pemetaan. Ini adalah satu-satunya pengaruh yang dimiliki layer penyajian terhadap varians sample. Turunkan `ψ` pada weight puncak, naikkan untuk pengguna premium.

## Bacaan Lanjutan

- [Karras dkk. (2019). Arsitektur Generator Berbasis Gaya untuk GAN](https://arxiv.org/abs/1812.04948) — StyleGAN.
- [Karras dkk. (2020). Menganalisis dan Meningkatkan Kualitas Gambar StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras dkk. (2021). Jaringan Adversarial Generatif Bebas Alias](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Tov dkk. (2021). Merancang Encoder untuk Manipulasi Gambar StyleGAN](https://arxiv.org/abs/2102.02766) — inversi e4e.
- [Sauer dkk. (2022). StyleGAN-XL: Menskalakan StyleGAN ke Kumpulan Data Besar yang Beragam](https://arxiv.org/abs/2202.00273) — StyleGAN-XL.
- [Huang dkk. (2024). R3GAN: GAN sudah mati; panjang umur GAN!](https://arxiv.org/abs/2501.05441) — resep GAN minimal modern.
