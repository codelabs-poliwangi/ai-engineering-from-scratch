# Pemahaman Dokumen dan Diagram

> Dokumen bukan foto. PDF, makalah ilmiah, faktur, atau formulir tulisan tangan memiliki tata letak, tabel, diagram, catatan kaki, header, dan struktur semantik yang tidak dapat ditangkap oleh pemahaman gambar biasa. Tumpukan pra-VLM adalah pipeline pipa: Tesseract OCR + LayoutLMv3 + heuristik ekstraksi tabel. Gelombang VLM menggantikannya dengan model bebas OCR — Donut (2022), Nougat (2023), DocLLM (2023) — yang memancarkan markup terstruktur secara langsung. Pada tahun 2026, frontier hanya "memasukkan gambar halaman ke Claude Opus 4.7 dengan resolusi asli 2576 piksel", dan output markup terstruktur tersedia secara gratis. Lesson ini membaca alur tiga era dokumen AI.

**Type:** Build
**Language:** Python (stdlib, kerangka parser dokumen yang peka terhadap tata letak)
**Prerequisites:** Fase 12 · 05 (LLaVA), Fase 5 (NLP)
**Waktu:** ~180 menit

## Tujuan Pembelajaran

- Jelaskan tiga era AI dokumen: pipeline OCR, bebas OCR, asli VLM.
- Jelaskan tiga aliran input LayoutLMv3: teks, tata letak (bbox), patch gambar, dengan masking terpadu.
- Bandingkan Donut (bebas OCR, gambar → markup), Nougat (makalah ilmiah → LaTeX), DocLLM (generatif sadar tata letak), PaliGemma 2 (asli VLM).
- Pilih model dokumen untuk tugas baru (faktur, makalah ilmiah, formulir tulisan tangan, kuitansi berbahasa Mandarin).

## Masalah

"Memahami PDF ini" tampaknya sulit. Informasinya ada di:

- Konten teks (90% sinyal).
- Tata letak (header, catatan kaki, sidebar, format dua kolom).
- Tabel (baris, kolom, sel gabungan).
- Gambar dan diagram.
- Anotasi tulisan tangan.
- Font dan tipografi (judul vs isi).

OCR mentah membuang teks dan kehilangan sisanya. Sistem yang peduli dengan faktur perlu mengetahui bahwa "Total: $1.245" berasal dari kanan bawah, bukan dari catatan kaki.

## Konsep

### Era 1 — Pipeline OCR (pra-2021)

Tumpukan klasik:

1. PDF → gambar per halaman.
2. Tesseract (atau OCR komersial) mengekstrak teks dengan kotak pembatas per kata.
3. Penganalisis tata letak mengidentifikasi blok (header, tabel, paragraf).
4. Pengenal struktur tabel mem-parsing tabel.
5. Aturan domain + bidang ekstrak regex.

Berfungsi untuk teks cetak yang bersih. Kerusakan pada tulisan tangan, pindaian miring, tabel rumit, skrip non-Inggris. Setiap mode kegagalan memerlukan jalur pengecualian khusus.

### TroCR (2021)

TrOCR (Li et al., arXiv:2109.10282) menggantikan CNN-CTC klasik Tesseract dengan encoder-decoder Transformer yang dilatih pada gambar teks sintetik + nyata. Kemenangan bersih pada teks tulisan tangan dan multibahasa. Masih berupa pipeline (detektor lalu TrOCR lalu tata letak), tetapi langkah OCR meningkat secara dramatis.

### Era 2 — Bebas OCR (2022-2023)

Model bebas OCR pertama mengatakan: lewati deteksi sepenuhnya, petakan piksel gambar ke output terstruktur secara langsung.

Donat (Kim dkk., arXiv:2111.15664):
- Transformer encoder-decoder, encodernya adalah Swin-B.
- Outputnya adalah JSON untuk pemahaman formulir, penurunan harga untuk ringkasan, atau skema khusus tugas apa pun.
- Tanpa OCR, tanpa tata letak, tanpa deteksi.

Nougat (Blecher dkk., arXiv:2308.13418):
- Dilatih khusus mengenai karya ilmiah.
- Outputnya adalah LaTeX / penurunan harga.
- Menangani persamaan, tata letak multi-kolom, gambar.
- Model setiap panggilan parser arXiv.

Mereka adalah spesialis, bukan generalis. Donat pada makalah ilmiah gagal; Nougat pada faktur gagal.

### Tata LetakLMv3 (2022)

Trek yang berbeda. LayoutLMv3 (Huang et al., arXiv:2204.08387) mempertahankan OCR tetapi menambahkan pemahaman tata letak:- Tiga aliran input: token teks OCR, kotak pembatas 2D per token, patch gambar.
- Tujuan training bertopeng di ketiga modalitas (teks bertopeng, tambalan bertopeng, tata letak bertopeng).
- Hilir: klasifikasi, ekstraksi entitas, tabel QA.

LayoutLMv3 adalah puncak pemahaman dokumen berbasis OCR. Kuat dalam formulir dan faktur. Membutuhkan OCR di bagian hulu. Akurasi pra-VLM terbaik pada tolok ukur dokumen standar.

### DokumenLLM (2023)

DocLLM (Wang et al., arXiv:2401.00908) adalah saudara generatif LayoutLM. Menghasilkan jawaban bentuk bebas yang dikondisikan pada token tata letak. Lebih baik untuk QA pada dokumen; masih tergantung pada input OCR.

### Era 3 — VLM asli (2024+)

VLM 2024 menjadi cukup baik untuk menggantikan seluruh jalur pipa. Masukkan gambar satu halaman penuh dengan resolusi tinggi ke VLM, ajukan pertanyaan, dapatkan jawaban.

- LLaVA-NeXT 336-tile AnyRes berfungsi untuk dokumen kecil.
- Resolusi dinamis Qwen2.5-VL menangani 2048+ piksel secara asli.
- Claude Opus 4.7 mendukung dokumen 2576px.
- PaliGemma 2 (April 2025) melatih khusus dokumen + tulisan tangan.

Kesenjangan antara pipa asli VLM dan pipa OCR menutup dengan cepat. Pada tahun 2026, VLM-asli menang dalam:

- Teks adegan (tulisan tangan + cetakan, skrip campuran).
- Tabel kompleks dengan sel gabungan.
- Persamaan matematika tertanam dalam teks.
- Gambar dengan anotasi teks.

Pipeline pipa OCR masih unggul dalam hal:

- Weight kerja pemindaian murni dalam skala besar yang mengutamakan latensi per halaman.
- Keandalan pipeline pipa (kegagalan deterministik vs halusinasi VLM).
- Lingkungan teregulasi yang memerlukan output OCR yang dapat diaudit.

### Perbatasan Claude 4.7 / GPT-5

Pada input asli 2576 piksel, VLM frontier melakukan pemahaman dokumen dengan akurasi mendekati manusia. Angka acuan dari awal tahun 2026:

- DocVQA: Claude 4.7 ~95.1, PaliGemma 2 ~88.4, Nougat ~77.3, pipeline LayoutLMv3 ~83.
- BaganQA: Claude 4.7 ~92.2, GPT-4V ~78.
- VisualMRC: Claude 4.7 ~94.

Kesenjangan model tertutup sebagian besar adalah resolusi dan skala LLM dasar. Model terbuka di 7B tertinggal beberapa poin tetapi tetap mengejar ketinggalan.

### Persamaan matematika dan output LaTeX

Makalah ilmiah memerlukan output LaTeX yang tepat untuk persamaan. Nougat dilatih dalam hal ini. VLM yang dilatih dengan target LaTeX (Qwen2.5-VL-Math, turunan Nougat) menghasilkan LaTeX yang dapat digunakan. Tanpa training LaTeX yang eksplisit, VLM menghasilkan transkripsi yang dapat dibaca namun tidak tepat.

Untuk jalur makalah ilmiah pada tahun 2026: rantai Nougat di PDF, lalu VLM di halaman rumit.

### Tulisan tangan

Masih merupakan sub-tugas yang paling sulit. Campuran cetakan + tulisan tangan (catatan dokter, formulir yang diisi) adalah tempat pipeline pipa OCR masih mengalahkan VLM dalam hal biaya. VLM yang hanya ditulis tangan semakin membaik (Claude 4.7, PaliGemma 2).

### resep 2026

Untuk proyek dokumen-AI baru:

- Faktur yang dicetak murni dalam skala besar: aturan LayoutLMv3 +, hemat biaya.
- Dokumen campuran (ilmiah + tulisan tangan + formulir): VLM-asli (PaliGemma 2 atau Qwen2.5-VL).
- Penyerapan arXiv penuh: Nougat untuk matematika, VLM untuk angka.
- Peraturan: Pipeline OCR + validator VLM untuk pemeriksaan silang.

## Pakai

`code/main.py`:

- Tokenizer yang sadar tata letak mainan: pasangan yang diberikan (teks, bbox), menghasilkan input gaya LayoutLMv3.
- Generator skema tugas bergaya Donat: templat JSON untuk formulir.
- Perbandingan anggaran token per halaman di seluruh pipa OCR, Donut, Nougat, dan VLM-asli.

## Kirim

Lesson ini menghasilkan `outputs/skill-document-ai-stack-picker.md`. Mengingat proyek dokumen-AI (domain, skala, kualitas, peraturan), pilihan antara pipeline OCR, spesialis bebas OCR, dan asli VLM.

## Latihan1. Proyek kamu adalah 10 juta faktur per hari. Tumpukan manakah yang meminimalkan biaya per halaman tanpa kehilangan akurasi?

2. Mengapa LayoutLMv3 mengungguli CLIP-VLM murni pada formulir QA namun berkinerja buruk pada teks adegan? Apa yang diberikan oleh aliran bbox?

3. Nougat menghasilkan LaTeX. Usulkan kasus uji di mana output asli VLM mengalahkan Nougat dalam fidelitas LaTeX, dan kasus di mana Nougat menang.

4. Membaca makalah PaliGemma 2 (Google, 2024). Apa penambahan training data utama yang meningkatkan keakuratan dokumen vs PaliGemma 1?

5. Rancang hibrida yang aman terhadap peraturan: pipa OCR sebagai pipa primer, VLM sebagai pemeriksaan silang sekunder. Bagaimana cara menyelesaikan perselisihan?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Pipa OCR | "Gaya Tesseract" | Tumpukan berdasarkan tahapan: deteksi -> OCR -> tata letak -> aturan; deterministik, rapuh |
| Bebas OCR | "Gaya donat" | Transformer gambar-ke-output yang melewatkan OCR eksplisit; model tunggal |
| Sadar tata letak | "Tata LetakLM" | Input mencakup koordinat bbox per token; penyamaran terpadu lintas modalitas |
| VLM-asli | "VLM Perbatasan" | Gambar halaman umpan langsung ke Claude/GPT/Qwen VLM dengan resolusi tinggi; tidak ada pipa |
| DokumenVQA | "Patokan dokumen" | Dokumen standar VQA; skor yang paling banyak dikutip |
| Output markup | "LaTeX / MD" | Format output terstruktur, bukan teks bentuk bebas; memungkinkan otomatisasi hilir |

## Bacaan Lanjutan

- [Li dkk. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher dkk. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang dkk. — Tata LetakLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim dkk. — Donat (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang dkk. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
