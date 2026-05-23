# Transfusi: Teks Autoregresif + Gambar Difusi dalam Satu Transformer

> Bunglon dan Emu3 mempertaruhkan segalanya pada token terpisah. Mereka berfungsi, tetapi hambatan kuantisasi terlihat - kualitas gambar berada di bawah model difusi ruang kontinu. Transfusi (Meta, Zhou dkk., Agustus 2024) mengambil risiko sebaliknya: menjaga gambar tetap kontinu, menghilangkan VQ-VAE seluruhnya, dan melatih satu Transformer dengan dua loss. Token teks mendapatkan prediksi token berikutnya. Tambalan gambar mengalami kehilangan pencocokan aliran/difusi. Kedua tujuan mengoptimalkan weight yang sama. Arsitektur yang mendasari Stable Diffusion 3 (MMDiT) adalah sepupu dekat. Lesson ini membaca tesis Transfusi, membuat mainan pelatih dua-kehilangan, dan menelusuri topeng attention yang memungkinkan satu trafo melakukan kedua pekerjaan.

**Type:** Build
**Language:** Python (stdlib, pelatih dua-kehilangan pada mainan skala MNIST)
**Prerequisites:** Fase 12 · 11 (Bunglon), Fase 8 (AI Generatif)
**Waktu:** ~180 menit

## Tujuan Pembelajaran

- Hubungkan trafo yang menjalankan dua loss (NTP pada token teks, MSE difusi pada patch gambar) pada satu tulang punggung.
- Jelaskan mengapa attention dua arah pada patch gambar ditambah attention kausal pada token teks adalah pilihan topeng yang tepat.
- Bandingkan gaya Transfusi (gambar kontinu, kehilangan difusi) dengan gaya Bunglon (gambar diskrit, NTP) pada komputasi, kualitas, dan kompleksitas code.
- Sebutkan kontribusi MMDiT: weight modalitas spesifik di setiap blok, attention bersama pada aliran sisa.

## Masalah

Perdebatan token gambar diskrit vs berkelanjutan sudah lebih tua dari LLM. Representasi berkelanjutan (piksel mentah, VAE laten) menjaga detail. Token diskrit (indeks VQ) sesuai dengan kosakata asli Transformer tetapi kehilangan detail pada langkah kuantisasi.

Chameleon / Emu3 menjadi terpisah: satu loss, satu arsitektur, tetapi fidelitas gambar dibatasi oleh kualitas tokenizer.

Model difusi berlangsung terus-menerus: kualitas gambar luar biasa, tetapi model terpisah dari LLM, rekayasa jadwal kebisingan yang rumit, dan tidak ada integrasi yang baik dengan pembuatan teks.

Transfusi bertanya: bisakah kita mendapatkan keduanya? Jaga agar gambar tetap kontinu, tetap latih satu model, gunakan dua loss yang digabungkan menjadi satu langkah gradient.

## Konsep

### Arsitektur dua loss

Sebuah Transformer khusus dekoder memproses rangkaian yang berisi:

- Token teks (diskrit, dari vocab BPE).
- Tambalan gambar (blok terus menerus berukuran 16x16 piksel diproyeksikan ke dalam redup tersembunyi melalui embedding linier — sama seperti input encoder ViT).
- Tag `<image>` dan `</image>` menandai tempat patch berkelanjutan berada.

Pass ke depan berjalan satu kali. Loss mengambil satu dari dua kepala per token:

- Untuk token teks: entropi silang standar pada kepala vocab-logits.
- Untuk patch gambar: hilangnya difusi pada patch berkelanjutan — memprediksi noise yang ditambahkan ke setiap patch.

Gradient mengalir melalui badan Transformer bersama. Kedua loss tersebut meningkatkan weight bersama secara bersamaan.

### Attention mask: teks kausal + gambar dua arah

Token teks harus bersifat sebab-akibat — kamu tidak boleh membiarkan token teks memperhatikan teks berikutnya, atau guru memaksa istirahat. Namun, patch gambar mewakili satu snapshot; mereka harus saling memperhatikan satu sama lain secara dua arah dalam blok gambar yang sama.

Topeng:

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

Diimplementasikan sebagai topeng blok-segitiga pada training dan inference.

### Loss difusi di dalam trafoHilangnya difusi adalah standar: tambahkan noise ke patch gambar, minta model untuk memprediksi noise (atau patch bersih, yang setara). Versi transfusi menggunakan pencocokan aliran — memprediksi bidang kecepatan dari berisik hingga bersih.

Selama training:
1. Untuk setiap patch gambar x0, ambil contoh langkah waktu acak t.
2. Contoh noise ε, hitung xt = (1-t) * x0 + t * ε (interpolasi linier untuk pencocokan aliran).
3. Trafo memprediksi v_theta(xt, t); loss = MSE(v_theta(xt, t), ε - x0).
4. Backprop bersama teks NTP loss dari urutan yang sama.

Kesimpulannya, generasi adalah:
- Token teks: pengambilan sample autoregresif standar.
- Tambalan gambar: loop pengambilan sample difusi (khas 10-30 langkah) dikondisikan pada token teks sebelumnya.

### MMDiT: varian Difusi Stabil 3

Stable Diffusion 3 (Esser et al., Maret 2024) mengirimkan MMDiT (Multimodal Diffusion Transformer) sekitar waktu yang sama dengan Transfusion. Arsitekturnya bersaudara.

Perbedaan utama MMDiT:

- Weight spesifik modalitas per blok. Setiap blok Transformer memiliki weight Q, K, V, dan MLP terpisah untuk token teks vs patch gambar. Attention bersifat bersama (lintas modalitas); segala sesuatu yang lain bersifat spesifik modalitas.
- Training aliran yang diperbaiki. Varian pencocokan aliran tertentu dengan pengambilan sample yang diketahui dan perhitungan yang lebih sederhana daripada DDPM.
- Skala. MMDiT adalah tulang punggung untuk SD3 (varian param 2B dan 8B). Skala kertas transfusi mencapai 7B.

Keduanya menyatu pada gagasan inti yang sama: satu Transformer menjalankan NTP pada teks dan difusi pada representasi gambar berkelanjutan.

### Mengapa ini mengalahkan gaya Bunglon

Kesenjangan kualitas antara difusi kontinu dan NTP diskrit pada pembuatan gambar dapat diukur. Laporan kertas transfusi:

- Pada parameter 7B, kalahkan model gaya Bunglon berukuran sama di FID dengan 3-5 poin.
- Tidak diperlukan training tokenizer — pembuat enkode gambar lebih sederhana (Proyeksi linier ke tersembunyi, sama seperti layer input ViT).
- Inference dapat memparalelkan penolakan patch gambar, tidak seperti token gambar autoregresif.

Kelemahan: Transfusi adalah model loss ganda, membuat dinamika training menjadi lebih rumit. Penurunan berat badan perlu penyesuaian. Ketidaksesuaian jadwal antara NTP dan difusi dapat menyebabkan satu kepala mendominasi.

### Apa yang ada di hilir

Janus-Pro (Lesson 12.15) menyempurnakan ide Transfusi dengan memisahkan encoder visi untuk pemahaman dan pembangkitan — SigLIP untuk satu, VQ untuk yang lain — sambil berbagi badan Transformer. Show-o (Lesson 12.14) menukar difusi dengan difusi diskrit (prediksi terselubung). Keluarga generasi terpadu bercabang dengan cepat setelah Transfusi.

VLM produksi tahun 2026 yang memancarkan gambar — Gemini 3 Pro, GPT-5, jalur pembuatan gambar Claude Opus 4.7 — hampir pasti menggunakan beberapa keturunan dari keluarga ini. Detail adalah hak milik.

## Pakai

`code/main.py` membuat mainan Transfusi berdasarkan masalah kecil seperti MNIST:

- Keterangan teks adalah urutan bilangan bulat pendek yang menggambarkan suatu digit (0-9).
- Gambar berukuran 4x4 grid byte.
- Sepasang proyeksi linier berbobot bersama bertindak sebagai Transformer pengganti; Hilangnya NTP pada teks, hilangnya MSE pada patch yang berisik.
- Loop training mengganti dua loss, topeng attention eksplisit.
- Generasi menghasilkan keterangan teks dan gambar 4x4 dalam satu forward pass.

Transformer adalah mainan. Pipa dua loss, konstruksi attention mask, dan loop inference adalah artefak sebenarnya.

## Kirim

Lesson ini menghasilkan `outputs/skill-two-loss-trainer-designer.md`. Dengan adanya tugas training multimodal baru (teks + gambar, teks + audio, teks + video), tugas ini merancang jadwal dua loss (weight penurunan, bentuk topeng, blok bersama vs blok khusus modalitas) dan menandai risiko penerapan.

## Latihan

1. Model gaya Transfusi melatih 70% token teks dan 30% patch gambar. Kehilangan difusi gambar ~10x besarnya kehilangan NTP teks. Weight penurunan berat badan apa yang menyeimbangkannya?

2. Implementasikan masker blok-segitiga untuk urutan: `[T, T, <image>, P, P, P, P, </image>, T]`. Tandai setiap entri 0 atau 1.

3. MMDiT memiliki weight QKV khusus modalitas. Berapa jumlah parameter overhead yang ditambahkan ini vs trafo yang digunakan bersama sepenuhnya oleh Transfusion? Di parameter 7B, apakah itu layak?

4. Pembuatan: diberi prompt teks, model menjalankan NTP untuk 50 token, lalu menekan `<image>`, lalu menjalankan difusi pada 256 patch dalam 20 langkah denoise. Berapa jumlah total operan ke depan?

5. Baca makalah SD3 Bagian 3. Jelaskan aliran yang diperbaiki dan mengapa aliran tersebut menyatu dalam langkah inference yang lebih sedikit dibandingkan DDPM.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Training dua kekalahan | "NTP + difusi" | Sebuah Transformer tunggal mengoptimalkan entropi silang pada token teks dan MSE pada patch gambar berkelanjutan dalam langkah gradient yang sama |
| Pencocokan aliran | "Aliran yang diperbaiki" | Varian difusi yang memprediksi bidang kecepatan dari noise hingga data bersih; matematika lebih sederhana dari DDPM |
| MMDiT | "DiT Multimoda" | Arsitektur Stable Diffusion 3: attention bersama, MLP dan norm khusus modalitas |
| Masker blok-segitiga | "Teks kausal + gambar dua arah" | Attention mask yang bersifat kausal di seluruh teks tetapi dua arah dalam wilayah gambar |
| Representasi gambar berkelanjutan | "Tidak Ada VQ" | Tambalan gambar sebagai vector bernilai nyata, bukan indeks buku code bilangan bulat |
| Prediksi kecepatan | "v-parameterisasi" | Output jaringan adalah medan kecepatan antara noise dan data, bukan noise itu sendiri |

## Bacaan Lanjutan

- [Zhou dkk. — Transfusi (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser dkk. — Difusi Stabil 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao dkk. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie dkk. — Tampilkan-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
