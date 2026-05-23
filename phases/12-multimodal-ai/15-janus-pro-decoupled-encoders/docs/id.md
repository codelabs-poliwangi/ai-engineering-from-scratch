# Janus-Pro: Encoder Terpisah untuk Model Multimodal Terpadu

> Model multimodal terpadu mempunyai ketegangan yang tidak dapat dihindari. Pemahaman menginginkan feature semantik - vector output SigLIP atau DINOv2 yang kaya dengan informasi tingkat konsep. Generasi menginginkan code yang ramah rekonstruksi — token VQ yang disusun kembali menjadi piksel yang tajam. Kedua tujuan tersebut tidak kompatibel dalam satu pembuat enkode. Janus (DeepSeek, Oktober 2024) dan Janus-Pro (DeepSeek, Januari 2025) berpendapat bahwa solusinya adalah berhenti mencoba: memisahkan kedua encoder. Bagikan badan Transformer antar tugas, tetapi rutekan pemahaman melalui SigLIP dan pembangkitan melalui tokenizer VQ. Pada 7B, Janus-Pro mengalahkan DALL-E 3 di GenEval sambil mencocokkan LLaVA di MMMU. Lesson ini menjelaskan mengapa dua encoder berfungsi ketika salah satunya gagal.

**Type:** Build
**Language:** Python (stdlib, perutean encoder ganda + sinyal badan bersama)
**Prerequisites:** Fase 12 · 13 (Transfusi), Fase 12 · 14 (Show-o)
**Waktu:** ~120 menit

## Tujuan Pembelajaran

- Jelaskan mengapa satu pembuat enkode bersama membahayakan pemahaman atau kualitas pembuatan.
- Jelaskan perutean Janus-Pro: Feature SigLIP di sisi input untuk pemahaman, token VQ pada input dan output untuk pembangkitan.
- Menelusuri penskalaan campuran data yang membuat Janus-Pro berhasil sedangkan Janus tidak.
- Bandingkan arsitektur berpasangan (Janus-Pro), berpasangan-kontinu (Transfusi), dan berpasangan-diskrit (Show-o).

## Masalah

Model terpadu berbagi kerangka Transformer dalam pemahaman dan generasi. Upaya sebelumnya (Bunglon, Show-o, Transfusi) semuanya menggunakan satu tokenizer visual untuk kedua arah. Tokenizer adalah kompromi:

- Dioptimalkan untuk rekonstruksi (generasi): VQ-VAE menangkap detail piksel yang sangat halus tetapi menghasilkan token dengan koherensi semantik yang lemah.
- Dioptimalkan untuk semantik (pemahaman): SigLIP mengelompokkan gambar "kucing" di dekat token "kucing" tetapi tidak memungkinkan rekonstruksi yang baik.

Show-o dan Transfusion membayarnya dengan pajak kualitas yang terlihat pada satu arah. Janus-Pro bertanya: mengapa memerlukan satu tokenizer ketika tugasnya memiliki kebutuhan yang berbeda?

## Konsep

### Pengodean visual yang dipisahkan

Arsitektur Janus-Pro memisahkan dua pembuat enkode:

- Memahami jalur. Gambar input → SigLIP-SO400m → MLP 2 lapis → badan Transformer.
- Jalur generasi. Gambar input (jika pengkondisian pada gambar yang sudah ada) → VQ tokenizer → ID token → badan Transformer.
- Pembangkitan output. Token gambar diprediksi oleh Transformer → dekoder VQ → piksel.

Badan Transformer digunakan bersama. Segala sesuatu di hulu dan hilir tubuh bersifat spesifik tugas.

Input dibedakan berdasarkan format prompt: tag `<understand>` yang dirutekan melalui SigLIP; `<generate>` rute melalui VQ. Atau peruteannya tersirat dari tugas.

### Mengapa ini berhasil

Memahami loss mendapatkan feature SigLIP, yang pra-training gaya CLIP telah disesuaikan untuk kesamaan semantik. Tolok ukur persepsi model meningkat dibandingkan Show-o/Transfusion karena feature input lebih baik untuk tugas tersebut.

Hilangnya generasi mendapatkan token VQ, yang telah disetel oleh pembuat token untuk direkonstruksi. Kualitas gambar meningkat dibandingkan Show-o karena code VQ disusun kembali menjadi piksel dengan rapi.

Badan Transformer bersama melihat dua distribusi input (SigLIP dan VQ) dan belajar bekerja dengan keduanya. Klaimnya: data yang cukup + parameter yang cukup, tubuh menyerap peralihan.

### Penskalaan data — Janus vs Janus-Pro

Janus (asli, arXiv 2410.13848) memperkenalkan decoupling tetapi dalam skala kecil (1,3 miliar parameter, data terbatas). Janus-Pro (arXiv 2501.17811) berskala:- Parameter 7B (vs 1.3B).
- 90 juta pasangan gambar-teks untuk phase 1 (penyelarasan) naik dari 72 juta.
- 72M untuk phase 2 (terpadu) naik dari 26M.
- Menambahkan 200 ribu sample instruksi gen gambar untuk phase 3.

Hasilnya: Janus-Pro-7B menyamai LLaVA di MMMU (60,3 vs ~58) dan mengalahkan DALL-E 3 di GenEval (0,80 vs 0,67). Satu model terbuka, kompetitif di kedua sisi spektrum terpadu.

### JanusFlow — varian aliran yang diperbaiki

JanusFlow (arXiv 2411.07975) menukar jalur pembangkitan VQ dengan jalur pembangkitan aliran yang diperbaiki (kontinu). Perpecahan tersebut menjadi SigLIP-untuk-pemahaman + aliran-untuk-generasi yang diperbaiki. Langit-langit berkualitas semakin terangkat. Arsitekturnya tetap dipisahkan-encoder-bersama-tubuh.

### Tugas badan bersama

Badan Transformer memproses urutan terpadu tetapi dengan dua distribusi input. Tugasnya adalah untuk:

- Untuk pemahaman: gunakan feature SigLIP + token teks → keluarkan teks secara otomatis.
- Untuk pembuatan: gunakan token teks + (token VQ gambar opsional) → pancarkan token VQ gambar secara otomatis.

Badan tidak memiliki weight spesifik modalitas per blok. Ini adalah Transformer gaya teks yang kamu harapkan ditemukan di dalam Qwen atau Llama, ditambah dua adaptor input.

Menariknya, ini berarti tubuh Janus-Pro dapat diinisialisasi dari LLM yang telah dilatih sebelumnya. Janus-Pro melakukan inisialisasi dari DeepSeek-MoE-7B. Pilihan itu penting: LLM menyumbangkan kemampuan penalaran yang sulit dicapai oleh model terpadu yang murni dari awal.

### Dibandingkan dengan InternVL-U

InternVL-U (Lesson 12.10) adalah tindak lanjut tahun 2026. Ini menggabungkan:

- Pra-training multimodal asli (tulang punggung InternVL3).
- Perutean encoder terpisah (SigLIP masuk, VQ + difusi keluar).
- Pemahaman terpadu + generasi + pengeditan.

InternVL-U memasukkan pilihan arsitektur Janus-Pro ke dalam kerangka yang lebih besar. Ide encoder terpisah kini menjadi default untuk model terpadu dalam skala besar.

### Keterbatasan

Encoder yang dipisahkan menambah kompleksitas arsitektur. Dua tokenizer untuk dilatih, dua jalur input untuk dipertahankan, dua set mode gagal. Untuk produk yang tidak memerlukan generasi, Janus-Pro direkayasa secara berlebihan — pilih model pemahaman keluarga LLaVA.

Untuk produk yang tidak memerlukan pemahaman, Janus-Pro terlalu memenuhi syarat — pilih model Stable Diffusion 3 / Flux.

Untuk produk yang membutuhkan keduanya, Janus-Pro kini menjadi referensi arsitektur terbuka.

## Pakai

`code/main.py` menyimulasikan perutean Janus-Pro:

- Dua encoder tiruan: seperti SigLIP (menghasilkan vector semantik 256-dim) dan seperti VQ (menghasilkan code integer).
- Router prompt yang memilih encoder berdasarkan tag tugas.
- Badan bersama (pengganti) yang memproses urutan token terlepas dari pembuat enkode mana yang memproduksinya.
- Peralihan dari jadwal sample tertimbang phase 1 (penyelarasan) ke phase 3 (penyetelan instruksi).

Cetak jalur yang dirutekan untuk 3 contoh: gambar QA, T2I, pengeditan gambar.

## Kirim

Lesson ini menghasilkan `outputs/skill-decoupled-encoder-picker.md`. Mengingat produk yang menginginkan generasi + pemahaman terpadu dengan kualitas terdepan, produk tersebut memilih Janus-Pro, JanusFlow, atau InternVL-U dengan rekomendasi skala data yang konkret.

## Latihan

1. Janus-Pro-7B mengalahkan DALL-E 3 di GenEval. Jelaskan mengapa model terbuka 7B dapat menyamai model kepemilikan terdepan dalam hal generasi tetapi tidak dalam hal pemahaman.

2. Mengimplementasikan fungsi router: diberikan teks prompt, klasifikasikan sebagai `understand` atau `generate`. Bagaimana kamu menangani prompt ambigu seperti "deskripsikan lalu buat sketsa"?

3. JanusFlow menggantikan jalur VQ dengan aliran yang diperbaiki. Apa yang sekarang dihasilkan oleh badan trafo, dan apa perubahan kerugiannya?4. Usulkan tugas keempat yang dapat ditangani oleh arsitektur Janus-Pro dengan satu lagi encoder yang dipisahkan. Contoh: segmentasi gambar (gaya DINO), kedalaman (gaya MiDaS).

5. Baca Janus-Pro Bagian 4.2 tentang penskalaan data. Phase data manakah yang paling berkontribusi terhadap peningkatan kualitas T2I vs Janus?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Pengkodean terpisah | "Dua pembuat enkode visual" | Pisahkan tokenizer atau encoder per arah: semantik untuk pemahaman, rekonstruksi untuk pembuatan |
| Tubuh bersama | "Satu trafo" | Transformer tunggal memproses output pembuat enkode; tidak ada weight khusus modalitas |
| SigLIP untuk pemahaman | "Feature semantik" | Menara visi keluarga CLIP menyediakan feature konseptual yang kaya tetapi rekonstruksi yang buruk |
| VQ untuk generasi | "Code rekonstruksi" | Token terkuantisasi vector yang memecahkan code dengan rapi kembali ke piksel |
| JanusFlow | "Varian aliran yang diperbaiki" | Janus-Pro dengan kepala generasi pencocokan aliran berkelanjutan, bukan VQ |
| Tag perutean | "Tag tugas" | Penanda prompt (`<understand>` / `<generate>`) yang memilih encoder input |

## Bacaan Lanjutan

- [Wu dkk. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen dkk. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma dkk. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong dkk. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
