# Model Generatif — Taksonomi & Sejarah

> Setiap model gambar, model teks, model video, dan model 3D cocok dalam satu dari lima keranjang. Pilih ember yang salah dan kamu akan berjuang berhitung selama berminggu-minggu. Pilih yang tepat dan kemajuan bidang ini selama dua belas tahun terakhir akan terpampang rapi di kepala kamu.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 2 (Dasar-dasar ML), Fase 3 (Inti Pembelajaran Mendalam), Fase 7 · 14 (Transformer)
**Waktu:** ~45 menit

## Masalah

Model generatif melakukan satu tugas: dengan memberikan sample training yang diambil dari beberapa distribusi yang tidak diketahui `p_data(x)`, menghasilkan sample baru yang sepertinya berasal dari distribusi yang sama. Wajah, kalimat, file MIDI, struktur protein — semuanya akan menjadi masalah yang sama jika kamu menyipitkan mata.

Masalah adalah `p_data` berada di ruang dengan jutaan dimension (gambar RGB 512x512 memiliki dimension ~786k), sample berada pada manifold tipis di dalam ruang tersebut, dan kamu mungkin hanya memiliki 10 juta contoh. Memaksa kepadatan secara kasar tidak ada harapan. Setiap model generatif adalah kompromi yang menukar satu masalah sulit dengan masalah yang lebih mudah.

Lima keluarga telah bertahan selama dua belas tahun terakhir. Mengetahui kompromi mana yang dibuat setiap keluarga akan memberi tahu kamu mengapa mereka menang dalam beberapa tugas dan gagal dalam tugas lain.

## Konsep

![Lima kelompok model generatif — taksonomi berdasarkan modelnya](../assets/taxonomy.svg)

**1. Kepadatan eksplisit, mudah diatur.** Tulis `log p(x)` sebagai jumlah yang benar-benar dapat kamu evaluasi. Model autoregresif (PixelCNN, WaveNet, GPT) memfaktorkan `p(x) = ∏ p(x_i | x_<i)`. Aliran normalisasi (RealNVP, Glow) membangun `p(x)` sebagai transformasi basis sederhana yang dapat dibalik. Pro: kemungkinan pasti, loss training bersih. Kontra: inference autoregresif bersifat sekuensial (lambat untuk rangkaian yang panjang), aliran memerlukan arsitektur yang dapat dibalik (restriktif secara arsitektural).

**2. Kepadatan eksplisit, perkiraan.** Terikat `log p(x)` dari bawah (ELBO) dan optimalkan pengikatan. VAE (Kingma 2013) menggunakan encoder-decoder dengan posterior variasional. Model difusi (DDPM, Ho 2020) melatih denoiser yang secara implisit mengoptimalkan ELBO berbobot. Difusi adalah tulang punggung gambar, video, dan 3D yang dominan pada tahun 2026.

**3. Kepadatan implisit.** Lewati kepadatan seluruhnya; pelajari generator `G(z)` yang menghasilkan sample dan diskriminator `D(x)` yang membedakan yang asli dan yang palsu. GAN (Goodfellow 2014). Cepat dalam inference (satu umpan ke depan) tetapi terkenal tidak stabil selama latihan. StyleGAN 1/2/3 tetap menjadi yang tercanggih untuk fotorealisme domain tetap (wajah, kamar tidur) bahkan pada tahun 2026.

**4. Berbasis skor / waktu berkelanjutan.** Learn gradient kepadatan log `∇_x log p(x)` (skor) secara langsung. Song & Ermon (2019) menunjukkan pencocokan skor menggeneralisasi difusi ke SDE. Pencocokan aliran (Lipman 2023) adalah yang paling populer di tahun 2024-2026: training bebas simulasi, jalur yang lebih lurus, pengambilan sample 4-10x lebih cepat daripada DDPM. Difusi Stabil 3, Fluks, AudioCraft 2 semuanya menggunakan pencocokan aliran.

**5. Autoregresif berbasis token pada code diskrit.** Kompres data dengan tingkat kecerahan tinggi dengan VQ-VAE atau kuantizer sisa menjadi rangkaian pendek token diskrit, lalu gunakan Transformer untuk memodelkan rangkaian token. Parti, MuseNet, AudioLM, VALL-E, tokenizer patch Sora semuanya menggunakan ini. Ini adalah ember 1 ditambah tokenizer yang dipelajari.

## Sejarah Singkat| Tahun | Model | Mengapa itu penting |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | Model generatif mendalam pertama dengan loss training yang dapat digunakan. |
| 2014 | GAN (Teman baik) | Kepadatan tersirat, tidak ada kemungkinan — sample yang sangat tajam. |
| 2015 | GAMBAR, PixelCNN | Pembuatan gambar berurutan. |
| 2017 | Cahaya, NVP Nyata | Arus yang dapat dibalik; kemungkinan pasti dengan kedalaman. |
| 2017 | GAN Progresif | Wajah megapiksel pertama. |
| 2019 | GayaGAN / GayaGAN2 | Wajah fotorealistik masih sulit dikalahkan untuk domain yang satu itu. |
| 2020 | DDPM (Ho) | Difusi menjadi praktis. |
| 2021 | KLIP, DALL-E 1, VQGAN | Teks-ke-gambar menjadi arus utama. |
| 2022 | Imagen, Difusi Stabil 1, DALL-E 2 | Difusi laten + pengondisian teks = komoditas. |
| 2022 | KontrolNet, LoRA | Kontrol yang baik atas difusi yang telah dilatih sebelumnya. |
| 2023 | SDXL, Midjourney v5, Pencocokan aliran | Skala + dinamika training yang lebih baik. |
| 2024 | Sora, Difusi Stabil 3, Fluks.1 | Difusi video; kemenangan pencocokan aliran. |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Pisang | Video tingkat produksi. |
| 2026 | Konsistensi + Aliran yang Diperbaiki | Pengambilan sample satu langkah dari tulang punggung difusi. |

## Triase lima pertanyaan

Saat makalah model generatif baru dirilis, jawablah lima pertanyaan ini sebelum membaca bagian metode.

1. **Apa yang dimodelkan?** Piksel, laten, token diskrit, Gaussian 3D, jerat, bentuk gelombang?
2. **Apakah kepadatannya eksplisit atau implisit?** Apakah mereka menuliskan `log p(x)`?
3. **Pengambilan sample: satu kali atau berulang?** Iteratif berarti inference yang lebih lambat; one-shot biasanya berarti permusuhan atau sulingan.
4. **Pengkondisian: tanpa syarat, kelas, teks, gambar, pose?** Ini menentukan hilangnya dan perancah arsitektur.
5. **Evaluasi: FID, skor CLIP, IS, preferensi manusia, akurasi tugas?** Masing-masing memiliki mode kegagalan yang diketahui (lihat Lesson 14).

kamu akan menjawab kembali kelima hal ini untuk setiap lesson di fase ini. Pada akhirnya, mereka akan menjadi refleks.

## Build

Code untuk lesson ini adalah visualisasi ringan: paskan campuran Gaussians 1-D dari sample menggunakan tiga pendekatan mainan (kepadatan kernel, histogram diskrit, dan generator "GAN-ish" sample terdekat) sehingga kamu dapat melihat perbedaan antara kepadatan eksplisit dan implisit pada soal yang dapat kamu cetak di satu layar.

Jalankan `code/main.py`. Ia mengambil 2000 sample dari campuran Gaussian dua mode, lalu mencetak:

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

Perhatikan: dua yang pertama memungkinkan kamu bertanya "seberapa besar kemungkinannya?" Yang ketiga tidak bisa. Inilah perbedaan *eksplisit dan implisit* yang penting untuk setiap lesson di masa depan.

## Pakai

Keluarga yang mana, untuk tugas yang mana, pada tahun 2026?

| Tugas | Keluarga terbaik | Mengapa |
|------|-------------|-----|
| Wajah fotoreal, domain sempit | GayaGAN 2/3 | Masih inference paling tajam dan tercepat. |
| Teks-ke-gambar umum | Difusi laten + pencocokan aliran | SD3, Fluks.1, DALL-E 3. |
| Teks-ke-gambar cepat | Aliran diperbaiki + distilasi | SDXL-Turbo, SD3-Turbo, LCM. |
| Teks-ke-video | Transformer Difusi + pencocokan aliran | Sora, Veo 2, Kling. |
| Pidato + musik | AR berbasis token (AudioLM, VALL-E, MusicGen) atau pencocokan aliran (AudioCraft 2) | Token diskrit berskala murah. |
| Adegan 3D | Gaussian Splatting fit, difusi sebelumnya | 3D-GS untuk rekonstruksi, difusi untuk tampilan baru. |
| Estimasi kepadatan (tanpa pengambilan sample) | Arus | Hanya keluarga dengan `log p(x)` yang tepat. |
| Simulasi / Fisika | Pencocokan aliran, skor SDE | Jalur garis lurus, bidang vector halus. |

## Kirim

Simpan sebagai `outputs/skill-model-chooser.md`.Keterampilan ini mengambil deskripsi tugas dan output: (1) kelompok mana yang akan digunakan, (2) daftar peringkat dari tiga opsi terbuka dan tiga opsi yang dihosting, (3) kemungkinan mode kegagalan yang harus kamu waspadai, dan (4) anggaran komputasi/waktu.

## Latihan

1. **Mudah.** Untuk masing-masing dari lima produk ini, identifikasi keluarga dan tulang punggung: Gambar ChatGPT, Midjourney v7, Sora, Runway Gen-3, ElevenLabs. Bukti harus berasal dari laporan teknis publik.
2. **Sedang.** Makalah yang akan kamu baca besok mengklaim pengambilan sample 100x lebih cepat daripada difusi. Tuliskan tiga pertanyaan untuk memeriksa apakah percepatan dapat bertahan dalam kondisi dan resolusi tinggi.
3. **Sulit.** Ambil satu domain yang kamu minati (misalnya struktur protein, CAD, molekul, lintasan). Jawab lima pertanyaan triase untuk model SOTA saat ini di domain tersebut dan buat sketsa model apa yang lebih baik yang akan berubah.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Model generatif | "Itu membuat hal baru" | Mempelajari sampler untuk `p_data(x)`, secara opsional mengekspos `log p(x)`. |
| Kepadatan eksplisit | "kamu dapat mengevaluasinya" | Model menyediakan `log p(x)` bentuk tertutup atau penurut. |
| Kepadatan implisit | "Gaya GAN" | Hanya contoh — tidak ada cara untuk mengevaluasi `p(x)` dari suatu poin tertentu. |
| ELBO | "Bukti batas bawah" | Batas bawah yang dapat diatur di `log p(x)`; VAE dan difusi mengoptimalkannya. |
| Skor | "Gradient kepadatan log" | `∇_x log p(x)`; model difusi dan SDE mempelajari bidang ini. |
| Berbagai hipotesis | "Data ada di permukaan" | Data dengan tingkat kecerahan tinggi terkonsentrasi pada manifold tingkat kecerahan rendah; mengapa dimensionality reduction berhasil. |
| Autoregresif | "Prediksi bagian selanjutnya" | Faktorkan gabungan sebagai hasil kali kondisional. |
| Laten | "Code terkompresi" | Representasi dengan tingkat kecerahan rendah yang dapat digunakan oleh dekoder untuk merekonstruksi input. |

## Catatan produksi: lima keluarga, lima bentuk inference

Setiap keluarga memetakan kurva biaya server inference yang berbeda. literatur inference produksi membingkai inference LLM sebagai prefill + decode; decomposition yang sama berlaku di sini:

- **Autoregressive (bucket 1 dan 5).** Dekode berurutan mendominasi latensi; KV-cache, batching berkelanjutan, dan decoding spekulatif semuanya berlaku secara langsung.
- **VAE / difusi / pencocokan aliran (bucket 2 dan 4).** Tidak ada dekode dalam pengertian LLM. Biaya = `num_steps × step_cost`, dan `step_cost` adalah Transformer atau penerus U-Net pada resolusi laten penuh. Tombol produksinya adalah penghitungan langkah (DDIM / DPM-Solver / distilasi), ukuran batch, dan presisi (bf16 / fp8 / int4).
- **GAN (ember 3).** Satu operan ke depan. Tanpa jadwal, tanpa cache KV. TTFT ≈ latensi total. Inilah sebabnya StyleGAN masih unggul dalam UX domain sempit.

Saat kamu melihat "lebih cepat daripada difusi" dalam abstrak makalah, terjemahkan menjadi "langkah lebih sedikit × biaya langkah yang sama" atau "langkah yang sama × biaya langkah lebih murah". Yang lainnya adalah pemasaran.

## Bacaan Lanjutan- [Teman Baik dkk. (2014). Jaring Adversarial Generatif](https://arxiv.org/abs/1406.2661) - makalah GAN.
- [Kingma & Welling (2013). Bayes Variasi Pengkodean Otomatis](https://arxiv.org/abs/1312.6114) — makalah VAE.
- [Ho, Jain, Abbeel (2020). Menyangkal Model Probabilistik Difusi](https://arxiv.org/abs/2006.11239) - makalah DDPM.
- [Lagu dkk. (2021). Pemodelan Generatif Berbasis Skor melalui SDE](https://arxiv.org/abs/2011.13456) — difusi sebagai SDE.
- [Lipman dkk. (2023). Pencocokan Aliran untuk Pemodelan Generatif](https://arxiv.org/abs/2210.02747) — makalah pencocokan aliran.
- [Esser dkk. (2024). Menskalakan Transformer Aliran yang Diperbaiki untuk Sintesis Gambar Resolusi Tinggi](https://arxiv.org/abs/2403.03206) — Difusi Stabil 3.
