# Penyelarasan Ekosistem Penelitian - MATS, Redwood, Apollo, METR

> Lima organisasi menentukan layer penelitian penyelarasan non-lab pada tahun 2026. MATS (ML Alignment & Theory Scholars): 527+ peneliti sejak akhir 2021, 180+ makalah, 10K+ kutipan, h-index 47; kelompok musim panas 2024 tergabung sebagai 501(c)(3) dengan ~90 sarjana dan 40 mentor; 80% alumni pra-2025 bekerja di bidang keselamatan/keamanan dengan 200+ orang di Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo. Redwood Research: laboratorium penyelarasan terapan yang didirikan oleh Buck Shlegeris; memperkenalkan Kontrol AI (Lesson 10); berkolaborasi dengan AISI Inggris dalam kasus keamanan pengendalian. Apollo Research: evaluasi perencanaan pra-penerapan untuk laboratorium perbatasan; menulis Skema Dalam Konteks (Lesson 8) dan Menuju Kasus Keamanan untuk Skema AI. METR (Model Evaluation and Threat Research): evaluasi kemampuan berbasis tugas, studi cakrawala waktu tugas otonom; "Elemen Umum Kebijakan Keamanan AI Frontier" membandingkan framework laboratorium. Eleos AI Research: evaluasi pra-penerapan kesejahteraan model (Lesson 19); melakukan penilaian kesejahteraan Claude Opus 4.

**Type:** Learn
**Language:** tidak ada
**Prerequisites:** Phase 18 · 01-27 (sebelum lesson Phase 18)
**Waktu:** ~45 menit

## Tujuan Pembelajaran

- Identifikasi lima organisasi ekosistem penelitian penyelarasan non-lab dan output intinya.
- Jelaskan skala MATS (sarjana, makalah, h-index) dan perannya sebagai pipeline bakat.
- Jelaskan agenda Pengendalian AI Redwood dan kemitraannya dengan AISI Inggris.
- Jelaskan metodologi evaluasi berbasis tugas METR.

## Masalah

Laboratorium terdepan (Lesson 18) melakukan evaluasi keselamatan secara internal dan mempublikasikan hasil yang dipilih. Ekosistem di luar laboratorium adalah tempat evaluasi divalidasi, tempat modus kegagalan baru pertama kali ditemukan, dan tempat talenta dilatih. Memahami ekosistem membantu menafsirkan temuan penelitian mana yang dipercaya oleh siapa.

## Konsep

### MATS (Penyelarasan ML & Sarjana Teori)

Dimulai akhir tahun 2021. Program bimbingan penelitian; para sarjana menghabiskan 10-12 minggu dengan peneliti senior pada masalah keselarasan tertentu.

Skala (2026):
- 527+ peneliti sejak awal.
- 180+ makalah diterbitkan.
- 10K+ kutipan.
- indeks-h 47.
- Musim Panas 2024: 90 cendekiawan + 40 mentor; dimasukkan sebagai 501(c)(3).

Hasil karir: ~80% alumni pra-2025 bekerja di bidang keselamatan/keamanan. 200+ di Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo.

### Penelitian Kayu Merah

Lab penyelarasan terapan. Didirikan oleh Buck Shlegeris. Memperkenalkan agenda Pengendalian AI (Lesson 10). Berkolaborasi dengan AISI Inggris dalam kasus keamanan pengendalian. Memberikan saran kepada DeepMind dan Anthropic tentang desain evaluasi.

Makalah kanonik: Greenblatt, Shlegeris dkk., "AI Control" (arXiv:2312.06942, ICML 2024); Alignment Faking (Greenblatt, Denison, Wright et al., arXiv:2412.14093, bersama dengan Anthropic).

Gaya: model ancaman spesifik, musuh terburuk, protokol konkret yang dapat diuji tekanannya.

### Penelitian Apollo

Evaluasi skema pra-penerapan untuk laboratorium perbatasan. Skema Dalam Konteks yang Ditulis (Lesson 8, arXiv:2412.04984). Bermitra dalam kolaborasi training anti-penipuan OpenAI 2025. Menghasilkan Kasus Keamanan untuk Perencanaan AI (2024).

Gaya: evaluasi pengaturan agen dimana penipuan dapat muncul; decomposition tiga pilar (ketidakselarasan, pengarahan pada tujuan, kesadaran situasional).

### METR (Evaluasi Model dan Riset Ancaman)Evaluasi kemampuan berbasis tugas. Studi cakrawala waktu penyelesaian tugas otonom. "Elemen Umum Kebijakan Keamanan AI Frontier" (metr.org/common-elements, 2025) membandingkan framework laboratorium.

Rekan penulis sketsa kasus keselamatan AI Scheming dengan Apollo.

Gaya: evaluasi tugas jangka panjang, pengukuran kemampuan empiris, sintesis framework.

### Penelitian AI Eleos

Evaluasi pra-penerapan model kesejahteraan. Melakukan penilaian kesejahteraan Claude Opus 4 yang didokumentasikan dalam bagian 5.3 kartu sistem. Menyediakan metodologi eksternal untuk memeriksa klaim-klaim yang relevan dengan kesejahteraan di Lesson 19.

### Aliran

MATS melatih para peneliti. Lulusan pergi ke Anthropic, DeepMind, OpenAI (tim keamanan lab) atau ke Redwood, Apollo, METR, Eleos (evaluasi eksternal). Evaluator eksternal bermitra dengan laboratorium dan dengan AISI/CAISI Inggris. Publikasi memberikan input bagi ekosistem ke MATS untuk kelompok berikutnya.

### Mengapa layer ini penting

Evaluasi sumber tunggal tidak dapat diandalkan: laboratorium yang mengevaluasi model mereka sendiri mempunyai konflik kepentingan struktural. Evaluator eksternal dapat meningkatkan dan memvalidasi mode kegagalan yang mungkin tidak dilaporkan oleh laboratorium. Makalah Agen Tidur tahun 2024 (Lesson 7) adalah Anthropic + Redwood; Alignment Faking adalah Antropik + Redwood; Skema Dalam Konteks adalah Apollo; Anti-Scheming adalah Apollo + OpenAI. Struktur multi-organisasi adalah pengendalian kualitas.

### Cocok untuk Fase 18

Lesson 7-11 merujuk pada karya Redwood dan Apollo; Lesson 18 merujuk pada perbandingan framework METR; Lesson 19 merujuk pada Eleos. Lesson 28 adalah peta organisasi yang eksplisit untuk ekosistem yang menjadi andalan Fase ini.

## Pakai

Tidak ada code. Bacalah "Elemen Umum Kebijakan Keamanan AI Frontier" METR sebagai contoh bagaimana sintesis eksternal menambah nilai pada kerja kebijakan internal laboratorium.

## Kirim

Lesson ini menghasilkan `outputs/skill-ecosystem-map.md`. Jika ada klaim atau evaluasi yang selaras, maka hal ini akan mengidentifikasi organisasi, tempat publikasi, dan gaya metodologisnya, serta melakukan pemeriksaan silang terhadap organisasi-organisasi yang dikenal sebagai mitra.

## Latihan

1. Pilih satu makalah dari Lesson 7-15 dan identifikasikan organisasi-organisasi yang terlibat. Periksa ulang penulis dengan alumni MATS dan afiliasi ekosistem saat ini.

2. Baca "Elemen Umum Kebijakan Keamanan AI Frontier" METR. Identifikasi tiga konvergensi lintas lab yang mereka tekankan dan dua divergensi terbesar.

3. Hasil karir MATS adalah ~80% keselamatan/keamanan. Perdebatkan apakah tekanan seleksi ini bersifat adaptif (melatih lapangan) atau bias (menyaring posisi heterodoks).

4. Redwood dan Apollo sama-sama melakukan pekerjaan kontrol/perencanaan tetapi dengan gaya yang berbeda. Pilih mode kegagalan dan jelaskan bagaimana masing-masing mode akan menyelidikinya.

5. Eleos AI adalah satu-satunya organisasi model kesejahteraan murni. Rancang organisasi hipotetis kedua yang berfokus pada pertanyaan berbeda yang berkaitan dengan kesejahteraan (kebebasan kognitif, perwujudan robot, dll.) dan jelaskan metodologinya.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| TIKAR | "program bimbingan" | Sarjana Penyelarasan & Teori ML; 527+ peneliti sejak 2021 |
| Penelitian Redwood | "lab kontrol" | Penyelarasan yang diterapkan; Penulis Kontrol AI; Mitra AISI Inggris |
| Penelitian Apollo | "evaluasi licik" | Evaluasi skema pra-penerapan untuk laboratorium perbatasan |
| METR | "penilaian cakrawala tugas" | Evaluasi kemampuan berbasis tugas; sintesis kerangka |
| Eleo AI | "lab kesejahteraan" | Evaluasi pra-penempatan kesejahteraan model |
| Pipeline bakat | "MATS -> laboratorium" | Lulusan MATS mengalir ke Anthropic, DM, OpenAI, Redwood, Apollo, METR |
| Evaluasi eksternal | "pemeriksaan non-lab" | Evaluasi tidak dilakukan oleh produser model; menambah kredibilitas |

## Bacaan Lanjutan

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — program bimbingan
- [Redwood Research](https://www.redwoodresearch.org/) — makalah Kontrol AI
- [Apollo Research](https://www.apolloresearch.ai/) — evaluasi licik
- [METR — Elemen Umum Kebijakan Keamanan AI Frontier](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — perbandingan framework
- [Eleos AI Research](https://www.eleosai.org/research) — model metodologi kesejahteraan
