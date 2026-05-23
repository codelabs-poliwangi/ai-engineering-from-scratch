# Model Terpadu Show-o dan Difusi Diskrit

> Transfusi memadukan representasi kontinu dan diskrit. Show-o (Xie dkk., Agustus 2024) melakukan sebaliknya: token teks menggunakan prediksi kausal token berikutnya, token gambar menggunakan difusi diskrit bertopeng dalam semangat MaskGIT. Keduanya berada di dalam satu trafo dengan attention mask hybrid. Hasilnya menyatukan VQA, text-to-image, inpainting, dan pembuatan modalitas campuran pada satu tulang punggung, satu tokenizer per modalitas, satu formulasi loss (token berikutnya diperluas ke prediksi bertopeng). Lesson ini membahas desain Show-o — mengapa difusi diskrit bertopeng merupakan generator gambar beberapa langkah paralel — dan kontras dengan Transfusion dan Emu3.

**Type:** Learn
**Language:** Python (stdlib, sampler difusi diskrit bertopeng)
**Prerequisites:** Phase 12 · 13 (Transfusi)
**Waktu:** ~120 menit

## Tujuan Pembelajaran

- Jelaskan difusi diskrit bertopeng: jadwal yang menutupi token secara seragam kemudian meminta Transformer untuk memulihkannya.
- Bandingkan decoding gambar paralel (Show-o, MaskGIT) dengan decoding gambar autoregresif (Chameleon, Emu3) dalam hal kecepatan dan kualitas.
- Sebutkan tiga tugas yang ditangani Show-o dalam satu pos pemeriksaan: T2I, VQA, pengecatan gambar.
- Pilih jadwal masking (kosinus, linier, terpotong) dan alasan pengaruhnya terhadap kualitas sample.

## Masalah

Training dua-kehilangan transfusi berhasil tetapi memiliki dinamika yang lebih rumit - kehilangan difusi terus-menerus berada pada skala numerik yang berbeda dari kehilangan NTP yang terpisah. Menyeimbangkan weight penurunan adalah pencarian hyperparameter. Arsitekturnya efektif tetapi kompleks.

Jawaban Show-o: jaga agar kedua modalitas tetap terpisah (seperti Bunglon), tetapi hasilkan gambar secara paralel melalui difusi diskrit bertopeng, bukan secara berurutan. Tujuan training menjadi prediksi token bertopeng tunggal yang menggeneralisasi prediksi token berikutnya secara alami.

## Konsep

### Difusi diskrit bertopeng (MaskGIT)

Chang dkk yang asli. (2022) Trik MaskGIT itu elegan. Mulai dari gambar yang sepenuhnya tertutup (setiap token adalah `<MASK>` id khusus). Pada setiap langkah, prediksi semua token yang disamarkan secara paralel, lalu pertahankan prediksi paling percaya diri K teratas dan tutupi ulang sisanya. Setelah ~8-16 iterasi, semua token terisi. Jadwal berapa banyak token yang akan dibuka kedoknya per langkah disesuaikan — jadwal kosinus berfungsi dengan baik.

Training-nya sederhana: ambil sample rasio masking secara seragam dari [0, 1], terapkan ke token VQ gambar, latih Transformer untuk memulihkan rasio masking. Persis seperti yang dilakukan BERT untuk teks, yang diskalakan hingga menghasilkan gambar.

### Show-o: satu trafo, masker hybrid

Show-o menempatkan MaskGIT di dalam Transformer model bahasa kausal. Attention mask-nya adalah:

- Token teks: kausal (LLM standar).
- Token gambar: dua arah penuh dalam blok gambar (sehingga token bertopeng dapat melihat setiap token gambar lainnya selama prediksi).
- Teks-ke-gambar: teks mengikuti gambar sebelumnya, gambar mengikuti teks sebelumnya.

Training bergantian antara:
1. NTP standar pada urutan teks.
2. Sample T2I: teks → gambar dengan token gambar bertopeng, kehilangan prediksi token bertopeng.
3. Sample VQA: gambar → teks dengan token teks bertopeng (sebenarnya hanya NTP).

Loss terpadu adalah entropi silang pada token `<MASK>`, yang mencakup NTP teks (hanya token terakhir yang "ditutupi") dan difusi bertopeng gambar (subset acak disamarkan).

### Pengambilan sample paralelShow-o menghasilkan gambar dalam ~16 langkah, bukan ~1000 (autoregresif per token) atau ~20 (difusi). Pada setiap langkah, prediksi semua token bertopeng secara paralel; melakukan top-K dengan percaya diri; mengulang.

Bandingkan:
- Chameleon / Emu3 (autoregressive over token): N_tokens forward pass, biasanya 1024-4096 per gambar.
- Transfusi (difusi kontinu): ~20 langkah, masing-masing melewati trafo penuh.
- Show-o (difusi diskrit bertopeng): ~16 langkah, masing-masing melewati trafo penuh.

Show-o lebih cepat daripada Chameleon pada model skala serupa, kira-kira cocok dengan jumlah langkah Transfusi dengan biaya per langkah yang lebih rendah (logit vocab terpisah vs kehilangan MSE berkelanjutan).

### Tugas dalam satu pos pemeriksaan

Show-o mendukung empat tugas pada inference, dipilih berdasarkan format prompt:

- Pembuatan teks: output teks autoregresif standar.
- VQA: gambar masuk, teks keluar.
- T2I: teks masuk, gambar keluar melalui difusi diskrit bertopeng.
- Inpainting: gambar dengan beberapa token bertopeng, isi.

Kemampuan melukis diperoleh secara gratis dari training prediksi bertopeng. Sembunyikan wilayah kisi token VQ, masukkan sisanya ditambah prompt teks, prediksi token yang disamarkan.

### Jadwal penyamaran

Jadwal berapa banyak token yang akan dibuka kedoknya per langkah menentukan kualitas. Show-o merekomendasikan kosinus:

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

Pada langkah 0, semua token ditutup (rasio 1.0). Pada langkah T, tidak ada yang bertopeng. Cosine memusatkan massa pada rasio kisaran menengah di mana prediksi paling informatif. Jadwal linier juga berfungsi tetapi lebih cepat stabil.

### Tampilkan-o2

Show-o2 (tindak lanjut tahun 2025, arXiv 2506.15564) menskalakan Show-o: basis LLM yang lebih besar, tokenizer yang lebih baik, jadwal masker yang lebih baik. Pola arsitektur yang sama.

### Tempat Show-o berada

Dalam taksonomi tahun 2026:

- Token diskrit + NTP: Bunglon, Emu3. Inference sederhana namun lambat.
- Token diskrit + difusi bertopeng: Show-o, MaskGIT, LlamaGen, Muse. Pengambilan sample paralel, masih lossy oleh tokenizer.
- Kontinyu + difusi: Transfusi, MMDiT, DiT. Training dengan kualitas terbaik dan lebih kompleks.
- Pencocokan aliran + berkelanjutan dalam VLM: JanusFlow, InternVL-U. Terbaru.

Pilih berdasarkan tugas: Tampilkan-o bila kamu ingin T2I + inpainting + VQA dalam satu model terbuka dengan kecepatan yang wajar; Transfusi ketika kualitas adalah yang terpenting dan kamu mampu membeli pipa dua loss.

## Pakai

`code/main.py` menyimulasikan pengambilan sample Show-o:

- Kotak mainan berisi 16 token VQ.
- Sebuah tiruan "Transformer" yang memprediksi logit berdasarkan prompt dan token yang saat ini terbuka kedoknya.
- Pengambilan sample bertopeng paralel dalam 8 langkah dengan jadwal kosinus.
- Mencetak status peralihan (evolusi pola topeng) dan token akhir.

Jalankan, lihat maskernya larut selangkah demi selangkah.

## Kirim

Lesson ini menghasilkan `outputs/skill-unified-gen-model-picker.md`. Mengingat produk yang membutuhkan pemahaman (VQA, captioning) dan generasi (T2I, inpainting) dengan batasan weight terbuka, pilihan antara keluarga Show-o, keluarga Transfusion/MMDiT, dan keluarga Emu3 / Chameleon dengan trade-off yang nyata.

## Latihan

1. Sample difusi diskrit tertutup dalam ~16 langkah. Kenapa tidak 1? Apa yang rusak jika kamu membuka kedok semuanya pada langkah 0?

2. Inpainting gratis dengan difusi bertopeng. Usulkan kasus penggunaan produk (nyata atau hipotetis) di mana lukisan Show-o mengalahkan model spesialis.

3. Jadwal kosinus vs jadwal linier: lacak jumlah token yang terbuka kedoknya per langkah untuk T=8. Mana yang lebih seimbang?4. Gambar Show-o berukuran 512x512 berisi 1024 token. Pada vocab K=16384, model mengeluarkan 1024 * log2(16384) = 14,336 bit (~1,75 KiB) data. Output Difusi Stabil 512*512*24 bit = 6.291.456 bit (~768 KiB) piksel mentah. Berapa rasio kompresi dan kualitas apa yang dibelinya?

5. Baca LlamaGen (arXiv:2406.06525). Apa perbedaan model gambar autoregresif bersyarat kelas LlamaGen dengan pendekatan bertopeng Show-o?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Difusi diskrit terselubung | "Gaya MaskGIT" | Training untuk memprediksi token bertopeng; pada inference, buka kedok prediksi paling yakin |
| Jadwal kosinus | "Buka kedok jadwal" | Peluruhan rasio topeng atas langkah-langkah inference; memusatkan pertumbuhan kepercayaan pada kisaran menengah |
| Penguraian code paralel | "Semua token sekaligus" | Setiap langkah memprediksi urutan penuh token bertopeng dalam satu forward pass, lalu melakukan top-K |
| Attention hibrida | "Kausal + dua arah" | Topeng yang bersifat kausal terhadap token teks dan dua arah dalam blok gambar |
| Lukisan | "Generasi pengisian" | Kondisi pada gambar dengan beberapa token yang disamarkan, prediksi yang hilang; bebas dari tujuan training |
| Tingkat komitmen | "K Teratas per langkah" | Berapa banyak token yang dinyatakan "selesai" per iterasi; mengontrol inference vs trade-off kualitas |

## Bacaan Lanjutan

- [Xie dkk. — Tampilkan-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Tampilkan-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang dkk. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Matahari dkk. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang dkk. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
