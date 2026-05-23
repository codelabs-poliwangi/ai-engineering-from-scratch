# Visual Autoregressive Modeling (VAR): Prediksi Skala Berikutnya

> Model difusi mengambil sample secara iteratif dalam waktu (langkah-langkah penolakan). VAR mengambil sample skala secara iteratif - VAR memprediksi token 1x1, lalu 2x2, lalu 4x4, hingga resolusi akhir, setiap skala mengkondisikan skala sebelumnya. Makalah tahun 2024 menunjukkan bahwa VAR cocok dengan hukum penskalaan gaya GPT untuk pembuatan gambar dan mengalahkan DiT pada anggaran komputasi yang sama. Lesson ini membangun mekanisme inti.

**Type:** Build
**Language:** Python (dengan PyTorch)
**Prerequisites:** Fase 7 Lesson 03 (Attention Banyak Kepala), Fase 8 Lesson 06 (DDPM)
**Waktu:** ~90 menit

## Masalah

Generasi autoregresif mendominasi pemodelan bahasa karena skalanya dapat diprediksi: lebih banyak komputasi, lebih banyak parameter, lebih sedikit perplexity, dan output yang lebih baik. Pembuatan gambar memiliki dua upaya AR utama sebelum tahun 2024: PixelRNN/PixelCNN (piksel demi piksel) dan DALL-E 1 / Parti / MuseGAN (token demi token pada code VQ-VAE).

Keduanya menderita masalah urutan generasi. Piksel dan token disusun dalam kisi 2D, namun model AR harus mengunjunginya dalam urutan raster 1D. Piksel sudut awal tidak tahu akan jadi apa gambarnya nanti. Kualitas pembuatan diskalakan lebih buruk dibandingkan GPT-on-text dan tidak pernah mencapai kualitas model difusi pada komputasi yang sesuai.

VAR memperbaiki masalah urutan generasi dengan mengubah apa yang dihasilkan. Daripada memprediksi token gambar satu per satu di ruang angkasa, VAR memprediksi keseluruhan gambar dengan resolusi yang meningkat. Langkah 1: prediksi token 1x1 ("ringkasan" gambar keseluruhan). Langkah 2: prediksi grid token 2x2 (feature lebih kasar). Langkah 3: prediksi grid 4x4. Langkah K: prediksi grid akhir (H/8)x(W/8).

Setiap skala memperhatikan semua skala sebelumnya (secara kausal dalam "urutan skala") dan paralel dalam skalanya sendiri. Masalah urutan hilang: seluruh gambar pada skala k dihasilkan dalam satu lintasan Transformer.

## Konsep

### Tokenizer Multi-Skala VQ-VAE

VAR memerlukan **tokenizer diskrit multi-skala**. Untuk gambar x, ini menghasilkan rangkaian token grid dengan resolusi yang semakin tinggi:

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

Setiap z_k menggunakan buku code yang sama (ukuran tipikal 4096-16384). Tokenization pada setiap skala tidak independen — tokenization dilatih sedemikian rupa sehingga menjumlahkan residu pada setiap skala akan merekonstruksi f:

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

Ini adalah varian **sisa VQ**. Skala k menangkap skala 1..k-1 yang terlewat. Decoder mengambil jumlah semua embedding skala dan menghasilkan gambar.

Tokenizer VQ multi-skala dilatih satu kali (seperti VQGAN) dan kemudian dibekukan. Semua pekerjaan generatif dilakukan oleh model autoregresif di atas.

### Prediksi Skala Berikutnya

Model generatif merupakan Transformer yang melihat token dari semua skala sebelumnya dan memprediksi token pada skala berikutnya.

Struktur urutan input:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Embedding posisi mengkodekan indeks skala dan posisi spasial dalam skala. Attention bersifat kausal dalam urutan skala: token pada skala k, posisi (i, j) dapat memperhatikan semua token pada skala 1..k dan token pada skala k itu sendiri yang datang lebih awal dalam urutan intra-skala apa pun yang digunakan (VAR menggunakan attention posisi tetap tanpa kausalitas intra-skala — semua posisi dalam skala diprediksi secara paralel).

Loss training: pada setiap skala k, prediksi token z_k dengan mempertimbangkan semua token skala sebelumnya. Hilangnya entropi silang pada code VQ diskrit. Strukturnya sama seperti GPT kecuali "urutan" sekarang terstruktur skala.

### Generasi

Pada inference:
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```Untuk skala K = 10, pembangkitannya adalah 10 lintasan maju Transformer. Setiap lintasan menghasilkan seluruh skalanya secara paralel — tidak ada autoregresi per token dalam suatu skala. Untuk gambar 256x256, ini kira-kira 10 lintasan vs 28-50 DiT.

### Mengapa Skala Berikutnya Menang Dibandingkan Token Berikutnya

Tiga kemenangan struktural:
1. **Sejajarkan kasar hingga halus dengan statistik gambar alami.** Persepsi visual manusia dan dataset gambar menunjukkan keteraturan yang bergantung pada skala: struktur frekuensi rendah stabil dan dapat diprediksi; detail frekuensi tinggi bergantung pada konten frekuensi rendah. Prediksi skala berikutnya memanfaatkan hal ini.
2. **Pembuatan paralel dalam skala.** Tidak seperti token AR bergaya GPT, VAR menghasilkan semua token dalam skala besar dalam satu langkah. Panjang pembangkitan efektif adalah skala log, bukan linier.
3. **Tidak ada bias urutan pembuatan.** Token pada skala k lihat semua skala k-1; tidak ada bias "kiri" atau "atas" yang memaksa token awal untuk dikomit sebelum konteks akhir tersedia.

### Hukum Penskalaan

Tian dkk. menunjukkan bahwa VAR mengikuti kurva penskalaan hukum kekuasaan untuk FID di ImageNet — seperti yang dilakukan GPT untuk perplexity. Menggandakan parameter atau menghitung separuh kesalahan dengan andal. Ini adalah model generatif gambar pertama yang menunjukkan perilaku penskalaan seperti model bahasa. Hasilnya adalah prediksi skala VAR dapat diprediksi dari komputasi, bukan tebakan empiris per arsitektur.

### Hubungan dengan Difusi

VAR dan difusi memiliki cerita kompresi data yang sama: keduanya memecah masalah pembangkitan menjadi serangkaian submasalah yang lebih mudah.

- Difusi: tambahkan kebisingan secara bertahap, belajar membatalkan satu langkah.
- VAR: tambahkan resolusi secara bertahap, belajar memprediksi skala berikutnya.

Mereka adalah sumbu yang berbeda dalam mengatasi masalah. Keduanya menghasilkan distribusi kondisional yang dapat diatur. Secara empiris VAR lebih cepat dalam inference (lebih sedikit operan, semuanya paralel dalam skala) dan cocok atau mengalahkan DiT pada ImageNet bersyarat kelas. VAR bersyarat teks (VARclip, HART) adalah arah penelitian aktif.

## Build

Di `code/main.py` kamu akan:
1. Buat **tokenizer VQ multi-skala** kecil pada data "gambar" sintetis (cincin Gaussian 2D).
2. Latih **Transformer gaya VAR** untuk memprediksi token pada skala berikutnya.
3. Sample dengan memanggil trafo sebanyak 4 kali (4 skala) dan decoding.
4. Verifikasi bahwa training berdasarkan skala membuat pembangkitan menjadi paralel dalam suatu skala.

Ini adalah implementasi mainan. Intinya adalah untuk melihat topeng attention terstruktur skala dan generasi paralel dalam skala benar-benar berfungsi.

## Kirim

Lesson ini menghasilkan `outputs/skill-var-tokenizer-designer.md` — keterampilan untuk merancang tokenizer multi-skala: jumlah skala, rasio skala, ukuran buku code, pembagian sisa, arsitektur dekoder.

## Latihan

1. **Ablasi penghitungan skala.** Latih VAR dengan 4, 6, 8, 10 skala. Ukur kualitas rekonstruksi vs jumlah lintasan autoregresif. Lebih banyak skala = residu lebih halus = kualitas lebih baik tetapi lebih banyak lintasan.

2. **Ukuran buku code.** Latih tokenizer dengan ukuran buku code 512, 4096, 16384. Buku code yang lebih besar memberikan rekonstruksi yang lebih baik tetapi prediksi yang lebih sulit. Temukan lututnya.

3. **Pemeriksaan paralel dalam skala.** Untuk VAR terlatih, ukur pola attention secara eksplisit. Dalam skala k, apakah model memperhatikan posisi lintas skala tetapi tidak dalam skala? Verifikasi penerapan masker.

4. **Penskalaan VAR vs DiT.** Untuk tugas kondisi kelas ImageNet yang sama, latih VAR dan DiT dengan anggaran param yang sesuai (misalnya, 33M, 130M, 458M). Plot FID vs komputasi. VAR harus mengungguli DiT pada setiap ukuran — mereproduksi hasil makalah dalam skala kecil.5. **Pengkondisian teks.** Perluas VAR untuk mengambil embedding teks (kumpulan CLIP) sebagai input pengondisian tambahan melalui adaLN. Ini adalah resep HART. Seberapa besar peningkatan FID pada pengambilan sample yang disejajarkan dengan teks?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|----------------------|
| VAR | "Visual AutoRegresif" | Pembuatan gambar dengan prediksi skala berikutnya pada piramida jaringan token VQ |
| Prediksi skala berikutnya | "Memprediksi lebih kasar, lalu lebih halus" | Model ini memprediksi token pada skala resolusi yang meningkat, mengkondisikan semua skala sebelumnya |
| Tokenizer VQ multi-skala | "Sisa VQ" | VQ-VAE yang menghasilkan kisi token K dengan resolusi yang meningkat, dengan decoder yang menjumlahkan semua skala |
| Skala k | "Piramida tingkat k" | Salah satu level resolusi K, dari 1x1 pada k=1 hingga (H/p)x(W/p) pada k=K |
| Skala Paralel | "Satu maju per skala" | Semua token pada skala k diprediksi dalam satu lintasan Transformer, bukan secara autoregresif |
| Kausal-lintas-skala | "Attention berdasarkan skala" | Token pada skala k dapat menangani semua skala 1..k tetapi tidak pada skala k+1..K |
| Sisa VQ | "Tokenization aditif" | Token setiap skala mengkodekan sisa yang ditinggalkan oleh skala yang lebih rendah; decoder menjumlahkan semua embedding skala |
| Hukum penskalaan VAR | "Penskalaan GPT gambar" | FID mengikuti hukum pangkat yang dapat diprediksi dalam komputasi, seperti perplexity model bahasa |
| HART | "VAR hibrida + teks" | Varian VAR bersyarat teks menggabungkan decoding berulang gaya MaskGIT dengan struktur skala VAR |
| Embedding posisi skala | "(skala, baris, kolom) rangkap tiga" | Pengkodean posisi membawa indeks skala dan koordinat spasial dalam skala |

## Bacaan Lanjutan

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — makalah VAR, referensi kanonik
- [Peebles dan Xie, 2022 — "Model Difusi yang Dapat Diskalakan dengan Transformer"](https://arxiv.org/abs/2212.09748) — DiT, dasar perbandingan difusi
- [Esser dkk., 2021 — "Menjinakkan Transformers untuk Sintesis Gambar Resolusi Tinggi"](https://arxiv.org/abs/2012.09841) — VQGAN, rangkaian tokenizer multi-skala VAR yang merupakan keluarga tokenizer
- [van den Oord et al., 2017 — "Pembelajaran Representasi Diskrit Neural"](https://arxiv.org/abs/1711.00937) — VQ-VAE, dasar tokenization gambar diskrit
- [Tang et al., 2024 — "HART: Pembuatan Visual yang Efisien dengan Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — VAR bersyarat teks
