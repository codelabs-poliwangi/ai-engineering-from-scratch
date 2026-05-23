# Kartu Model, Sistem, dan Kumpulan Data

> Tiga format dokumentasi menyusun transparansi AI. Kartu Model (Mitchell dkk. 2019) — label nutrisi untuk model: training data, analisis terpilah kuantitatif, pertimbangan etis, peringatan; hanya 0,3% kartu model Hugging Face yang mendokumentasikan pertimbangan etis (Oreamuno dkk. 2023). Lembar Data untuk Kumpulan Data (Gebru et al. 2018, CACM) — motivasi, komposisi, proses pengumpulan, pelabelan, distribusi, pemeliharaan; analogi lembar data elektronik. Kartu Data (Pushkarna et al., Google 2022) — detail berlapis modular (teleskopik, periskopik, mikroskopis) sebagai objek pembatas untuk pembaca yang beragam. Perkembangan tahun 2024-2025: pembuatan otomatis melalui LLM (CardGen, Liu dkk. 2024); detail kartu model berkorelasi dengan peningkatan unduhan HF hingga 29% (Liang dkk. 2024); pengesahan yang dapat diverifikasi (Laminator, Duddu dkk. 2024); penambahan pelaporan keberlanjutan untuk karbon/air (Jouneaux et al. Juli 2025); Kartu peraturan UE/ISO bermunculan. Kartu Sistem (Sidhpurwala 2024; Transparansi tingkat sistem meta; "Cetak Biru Kepercayaan" arXiv:2509.20394) — dokumentasi sistem AI menyeluruh yang mencakup kemampuan keamanan, perlindungan injeksi cepat, deteksi penyelundupan data, penyelarasan dengan nilai-nilai kemanusiaan.

**Type:** Build
**Language:** Python (stdlib, kartu model + lembar data + generator kartu sistem)
**Prerequisites:** Fase 18 · 18 (kerangka keselamatan), Fase 18 · 24 (peraturan)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Jelaskan Mitchell dkk yang asli. Kartu model 2019 dan Gebru dkk. lembar data 2018.
- Jelaskan layer teleskopik/periskopis/mikroskopis Kartu Data.
- Jelaskan Kartu Sistem dan cakupan end-to-endnya.
- Sebutkan tiga perkembangan tahun 2024-2025 (pembuatan otomatis, pengesahan yang dapat diverifikasi, pelaporan keberlanjutan).

## Masalah

Kerangka peraturan (Lesson 24) dan kebijakan keselamatan laboratorium (Lesson 18) keduanya memerlukan dokumentasi. Format dokumentasi berevolusi dari khusus model (kartu model) menjadi khusus dataset (lembar data) hingga khusus sistem (kartu sistem). Masing-masing membahas cakupan transparansi yang berbeda. Pekerjaan otomatisasi dan pengesahan yang dapat diverifikasi pada tahun 2024-2025 mengatasi masalah adopsi yang sudah berlangsung lama.

## Konsep

### Kartu Model (Mitchell dkk. 2019)

Bagian:
- Detail model.
- Tujuan penggunaan.
- Faktor (faktor demografi atau lingkungan yang relevan untuk evaluasi).
- Metrik.
- Data evaluasi.
- Training data.
- Analisis kuantitatif (dipilah berdasarkan faktor).
- Pertimbangan etis.
- Peringatan dan rekomendasi.

Masalah adopsi: Oreamuno dkk. Audit kartu model Hugging Face pada tahun 2023 hanya menemukan 0,3% dokumen pertimbangan etis.

### Lembar Data untuk Kumpulan Data (Gebru dkk. 2018)

Analogi lembar data elektronik. Bagian:
- Motivasi (mengapa dataset dibuat).
- Komposisi (apa isinya).
- Proses pengumpulan (bagaimana perakitannya).
- Pelabelan (jika ada).
- Penggunaan (dimaksudkan, dilarang, berisiko).
- Distribusi.
- Pemeliharaan.

Diterbitkan di CACM 2021. Lembar data adalah dokumentasi hulu; kartu model bergantung pada keakuratan lembar data.

### Kartu Data (Pushkarna dkk., Google 2022)

Detail berlapis modular. Tiga tingkat zoom:
- **Teleskopik.** Ringkasan tingkat tinggi untuk non-ahli.
- **Periskopis.** Ikhtisar tingkat menengah untuk praktisi ML.
- **Mikroskopis.** Dokumentasi tingkat feature terperinci untuk auditor.

Pembingkaian objek batas: pembaca yang berbeda mengekstrak informasi berbeda dari dokumen yang sama.

### Kartu SistemCakupan: sistem AI ujung ke ujung termasuk model + tumpukan keamanan + konteks penerapan. Bagian biasanya meliputi:
- Kemampuan keamanan.
- Perlindungan injeksi cepat.
- Deteksi eksfiltrasi data.
- Keselarasan dengan nilai-nilai kemanusiaan yang dinyatakan.
- Respons insiden.

Sidhpurwala 2024 dan transparansi tingkat sistem Meta berfungsi. "Cetak Biru Kepercayaan" (arXiv:2509.20394) meresmikan Kartu Sistem sebagai pelengkap layer penerapan Kartu Model.

### Perkembangan 2024-2025

- **CardGen (Liu et al. 2024).** Pembuatan kartu model otomatis melalui LLM; melaporkan objektivitas yang lebih tinggi daripada banyak kartu yang ditulis manusia di bidang standar Mitchell 2019.
- **Korelasi pengunduhan (Liang dkk. 2024).** Kartu model terperinci berkorelasi dengan tingkat pengunduhan HF yang lebih tinggi hingga 29% — tekanan adopsi kini didorong oleh pasar, bukan hanya didorong oleh kepatuhan.
- **Laminator (Duddu et al. 2024).** Pengesahan yang dapat diverifikasi melalui TEE perangkat keras / tanda tangan kriptografi — memungkinkan kartu model membawa bukti klaim, bukan sekadar klaim.
- **Keberlanjutan (Jouneaux dkk. Juli 2025).** Penambahan jejak karbon, air, dan energi komputasi; standar ISO yang muncul.
- **Kartu peraturan.** EU AI Act (Lesson 24) Bab Transparansi Code Praktik GPAI memerlukan kartu model sebagai artefak kepatuhan.

### Cocok untuk Fase 18

Lesson 24-25 adalah layer regulasi dan CVE. Lesson 26 adalah layer dokumentasi. Lesson 27 adalah training tata kelola data, yang merupakan bagian hulu lembar data. Lesson 28 adalah ekosistem penelitian yang menghasilkan evaluasi yang dirujuk dalam kartu.

## Pakai

`code/main.py` menghasilkan kartu model minimal, lembar data, dan kartu sistem untuk penerapan mainan. Masing-masing mengikuti struktur bagian kanonik. kamu dapat memeriksa format dan membandingkan ketiga cakupan.

## Kirim

Lesson ini menghasilkan `outputs/skill-card-audit.md`. Dengan adanya kartu model, lembar data, atau kartu sistem, kartu tersebut mengaudit cakupan bagian, pemilahan numerik, dan apakah terdapat pengesahan yang dapat diverifikasi.

## Latihan

1. Jalankan `code/main.py`. Periksa kartu yang dihasilkan. Identifikasi bagian-bagian yang lemah (hanya placeholder) dan tentukan bukti apa yang dapat memperkuat bagian-bagian tersebut.

2. Perluas kartu model dengan analisis terpilah kuantitatif pada dua kelompok demografi (Lesson 20).

3. Baca Oreamuno dkk. 2023 dengan tingkat adopsi 0,3%. Usulkan satu perubahan struktural pada spesifikasi kartu model yang akan meningkatkan penerapan pertimbangan etis.

4. Laminator (Duddu dkk. 2024) menggunakan TEE untuk pengesahan yang dapat diverifikasi. Rancang bidang kartu model yang membawa pengesahan kriptografis dari hasil evaluasi dan jelaskan peran pemverifikasi.

5. Tulis Kartu Sistem (Kartu Sistem, bukan Kartu Model) untuk salah satu proyek kamu sebelumnya atau penerapan hipotetis. Identifikasi bagian dengan nilai tertinggi untuk auditor pihak ketiga.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Kartu Model | "kartu Mitchell" | Mitchell dkk. Dokumentasi standar 2019 untuk model ML |
| Lembar Data | "lembar data Gebru" | Gebru dkk. Dokumentasi standar 2018 untuk dataset |
| Kartu Data | "kartu Pushkarna" | Dokumentasi data berlapis modular Google 2022 |
| Kartu Sistem | "kartu penerapan" | Dokumentasi sistem AI menyeluruh termasuk tumpukan keamanan |
| Objek batas | "pembaca berbeda, satu dokumen" | Pembingkaian Kartu Data: dokumen yang sama melayani khalayak yang beragam |
| Pengesahan yang dapat diverifikasi | "pengesahan Laminator" | Bukti kriptografi atau TEE yang dilampirkan pada klaim dokumentasi |
| Bidang keberlanjutan | "jejak karbon/air" | Penambahan akuntansi lingkungan pada tahun 2025 |

## Bacaan Lanjutan

- [Mitchell dkk. — Kartu Model untuk Pelaporan Model (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — kartu model kanonik
- [Gebru dkk. — Lembar Data untuk Kumpulan Data (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — kertas lembar data
- [Pushkarna dkk. — Kartu Data (Google 2022)](https://arxiv.org/abs/2204.01075) — dokumentasi data berlapis
- [Sidhpurwala dkk. — Cetak Biru Kepercayaan (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) — Formalisasi Kartu Sistem
