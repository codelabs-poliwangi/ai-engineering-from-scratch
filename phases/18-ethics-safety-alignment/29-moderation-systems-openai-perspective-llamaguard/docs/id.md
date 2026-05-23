# Sistem Moderasi — OpenAI, Perspektif, Llama Guard

> Sistem moderasi produksi mengoperasionalkan kebijakan keselamatan yang dijelaskan dalam Lesson 12-16. API Moderasi OpenAI: `omni-moderation-latest` (2024) dibangun di atas GPT-4o mengklasifikasikan teks + gambar dalam satu panggilan; 42% lebih baik pada set pengujian multibahasa dibandingkan versi sebelumnya; skema respons mengembalikan 13 kategori boolean — pelecehan, pelecehan/ancaman, kebencian, kebencian/ancaman, terlarang, terlarang/kekerasan, menyakiti diri sendiri, menyakiti diri sendiri/niat, menyakiti diri sendiri/instruksi, seksual, seksual/anak di bawah umur, kekerasan, kekerasan/grafik; gratis untuk sebagian besar pengembang. Pola berlapis: Moderasi input (pra-generasi), Moderasi output (pasca-generasi), Moderasi khusus (aturan domain). Panggilan paralel asinkron menyembunyikan latensi; tanggapan placeholder pada bendera. Llama Guard 3/4 (Lesson 16): 14 bahaya MLCommons, Penyalahgunaan Penerjemah Code, 8 bahasa (v3), multi-gambar (v4). Perspective API (Google Jigsaw): penilaian toksisitas sebelum gelombang LLM sebagai moderator; terutama toksisitas satu dimension dengan varian toksisitas parah/penghinaan/kata-kata kotor; dasar untuk penelitian moderasi konten. Penghentian: Moderator Konten Azure tidak digunakan lagi pada Februari 2024, dihentikan pada Februari 2027, digantikan oleh Keamanan Konten Azure AI.

**Type:** Build
**Language:** Python (stdlib, harness moderasi tiga lapis)
**Prerequisites:** Fase 18 · 16 (Penjaga Llama / Garak / PyRIT)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Jelaskan taksonomi kategori OpenAI Moderation API dan perbedaannya dengan kumpulan MLCommons Llama Guard 3.
- Jelaskan tiga pola layer moderasi (input, output, custom) dan sebutkan satu mode kegagalan masing-masing.
- Jelaskan posisi Perspective API sebagai dasar sebelum era LLM dan mengapa tetap digunakan dalam penelitian.
- Nyatakan garis waktu penghentian Azure.

## Masalah

Lesson 12-16 menjelaskan serangan dan peralatan pertahanan. Lesson 29 mencakup sistem moderasi yang diterapkan yang mengoperasionalkan pertahanan di permukaan tempat pengguna menyentuh produk. Pola tiga lapis adalah konfigurasi default tahun 2026.

## Konsep

### API Moderasi OpenAI

`omni-moderation-latest` (2024). Dibangun di atas GPT-4o. Mengklasifikasikan teks + gambar dalam satu panggilan. Gratis untuk sebagian besar pengembang.

Kategori (13 boolean dalam skema respons):
- pelecehan, pelecehan / ancaman
- benci, benci/mengancam
- menyakiti diri sendiri, menyakiti diri sendiri/niat, menyakiti diri sendiri/instruksi
- seksual, seksual/anak di bawah umur
- kekerasan, kekerasan/grafik
- terlarang, terlarang/kekerasan

Dukungan multimoda berlaku untuk `violence`, `self-harm`, dan `sexual` tetapi tidak pada `sexual/minors`; sisanya hanya berupa teks.

Untuk pemanfaatan code di `code/main.py` kami menggabungkan sub-kategori `/threatening`, `/intent`, `/instructions`, dan `/graphic` ke dalam induk tingkat atas untuk kesederhanaan pedagogis. Code produksi harus menggunakan skema 13 kategori lengkap.

42% lebih baik pada set pengujian multibahasa dibandingkan titik akhir moderasi generasi sebelumnya. Skor per kategori; aplikasi menetapkan ambang batas.

### Penjaga Llama 3/4

Dibahas dalam Lesson 16. 14 kategori bahaya MLCommons (disusun secara berbeda dari 13 boolean skema respons OpenAI). Mendukung 8 bahasa (v3). Llama Guard 4 (April 2025) adalah multimodal asli, 12B.

Taksonomi OpenAI dan Llama Guard tumpang tindih tetapi berbeda. OpenAI memiliki "illicit" sebagai kategori yang luas; Llama Guard memiliki "kejahatan dengan kekerasan" dan "kejahatan tanpa kekerasan" secara terpisah. Penerapan dipilih berdasarkan kesesuaian taksonomi kebijakannya.

### API Perspektif (Google Jigsaw)Sistem penilaian toksisitas sebelum gelombang LLM sebagai moderator (pra-2020). Kategori: TOXICITY, SEVERE_TOXICITY, INSULT, PROFANITY, ANCAMAN, IDENTITY_ATTACK. Skor utama satu dimension (TOXICITY) dengan varian subdimensi.

Banyak digunakan sebagai dasar penelitian moderasi konten karena API ini stabil, terdokumentasi, dan memiliki data kalibrasi bertahun-tahun. Untuk kasus penggunaan modern yang berdekatan dengan LLM, Llama Guard atau OpenAI Moderation biasanya lebih cocok.

### Pola tiga lapis

1. **Moderasi input.** Klasifikasikan prompt pengguna sebelum dibuat. Tolak jika ditandai. Latensi: satu panggilan pengklasifikasi.
2. **Moderasi output.** Klasifikasikan output model sebelum pengiriman. Ganti dengan penolakan jika ditandai. Latensi: satu panggilan pengklasifikasi setelah generasi.
3. **Moderasi khusus.** Aturan khusus domain (regex, daftar yang diizinkan, kebijakan bisnis). Berjalan pada input atau output.

Ketiga layer tersebut dirancang secara berurutan: moderasi input harus selesai sebelum pembangkitan, dan moderasi output dijalankan setelah pembangkitan. Paralelisme berlaku dalam satu layer — menjalankan beberapa pengklasifikasi (misalnya, Moderasi OpenAI + Llama Guard + Perspektif) secara bersamaan pada teks yang sama akan menyembunyikan latensi per pengklasifikasi. Sebagai optimization opsional, respons placeholder ("suatu saat, pemeriksaan...") dapat ditampilkan saat moderasi input selesai dan streaming token-1 ditangguhkan. Perilaku penandaan dapat dikonfigurasi: menolak, membersihkan, meneruskan ke peninjauan manusia.

### Mode kegagalan

- **Hanya input.** Tidak menangkap halusinasi output (Lesson 12-14 serangan pengkodean mengabaikan pengklasifikasi input).
- **Hanya output.** Memungkinkan input apa pun mencapai model; meningkatkan biaya; memunculkan alasan internal kepada penyerang.
- **Khusus khusus.** Tidak kuat di seluruh kategori; regex rapuh.

Berlapis adalah defaultnya. Sabuk dan suspender.

### Penghentian penggunaan Azure

Moderator Konten Azure: tidak digunakan lagi pada Februari 2024, dihentikan pada Februari 2027. Digantikan oleh Azure AI Content Safety, yang berbasis LLM dan terintegrasi dengan Azure OpenAI. Migrasi ini merupakan proyek tingkat lapangan tahun 2024-2027 untuk penerapan Azure.

### Cocok untuk Fase 18

Lesson 16 mencakup perangkat moderasi dalam konteks tim merah. Lesson 29 mencakup moderasi yang diterapkan. Lesson 30 ditutup dengan bukti kemampuan penggunaan ganda saat ini.

## Pakai

`code/main.py` membuat pemanfaatan moderasi tiga lapis: moderator input (skor kata kunci + kategori), moderator output (pengklasifikasi yang sama pada output), moderator khusus (aturan domain). kamu dapat menjalankan input dan mengamati layer mana yang menangkap apa.

## Kirim

Lesson ini menghasilkan `outputs/skill-moderation-stack.md`. Mengingat penerapan, ini merekomendasikan konfigurasi tumpukan moderasi: pengklasifikasi mana pada input, yang mana pada output, aturan khusus mana, dan apa yang menilai kasus edge.

## Latihan

1. Jalankan `code/main.py`. Jalankan input yang tidak berbahaya, berbahaya, dan berbahaya melalui ketiga layer. Laporkan layer mana yang diaktifkan untuk masing-masing layer.

2. Memperluas pemanfaatan dengan penilaian toksisitas ala Perspective-API pada kategori tertentu. Bandingkan perilaku ambang batasnya dengan skor kategori.

3. Baca dokumen OpenAI Moderation API dan daftar kategori Llama Guard 3. Petakan setiap kategori OpenAI ke kategori Llama Guard terdekat. Identifikasi tiga kategori yang tidak dipetakan dengan rapi.

4. Rancang tumpukan moderasi untuk penerapan asisten code (misalnya, GitHub Copilot). Identifikasi kategori yang paling dan paling tidak relevan dan usulkan aturan adat.5. Moderator Konten Azure akan dihentikan pada Februari 2027. Rencanakan migrasi ke Keamanan Konten Azure AI. Identifikasi elemen migrasi dengan risiko tertinggi.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Moderasi OpenAI | "omni-moderasi-terbaru" | Pengklasifikasi 13 kategori (teks) berbasis GPT-4o dengan dukungan multimodal parsial |
| API Perspektif | "Toksisitas Google Jigsaw" | Dasar penilaian toksisitas sebelum era LLM |
| Penjaga Llama | "MLCommons 14 kategori" | Pengklasifikasi bahaya Meta (v3: 8B teks, 8 bahasa; v4: 12B multimodal) |
| Moderasi input | "filter pra-generasi" | Pengklasifikasi pada prompt pengguna sebelum panggilan model |
| Moderasi output | "filter pasca-generasi" | Pengklasifikasi berdasarkan output model sebelum pengiriman |
| Moderasi khusus | "aturan domain" | Aturan khusus penerapan (regex, daftar yang diizinkan, kebijakan) |
| Moderasi berlapis | "ketiga layer" | Pola penerapan produksi standar |

## Bacaan Lanjutan

- [Dokumen API Moderasi OpenAI](https://platform.openai.com/docs/api-reference/moderations) — titik akhir moderasi omni
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Repo Llama Guard
- [API Perspektif Google Jigsaw](https://perspectiveapi.com/) — penilaian toksisitas
- [Keamanan Konten Azure AI](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — pengganti Azure
