# Tanda Air — SynthID, Tanda Tangan Stabil, C2PA

> Tiga teknologi menyusun sumber konten yang dihasilkan AI pada tahun 2026. SynthID (Google DeepMind) — watermarking gambar diluncurkan Agustus 2023, teks+video Mei 2024 (Gemini + Veo), teks bersumber terbuka Oktober 2024 melalui Responsible GenAI Toolkit, detektor multi-media terpadu November 2025 bersama Gemini 3 Pro. Penandaan air teks menyesuaikan probabilitas pengambilan sample token berikutnya secara tidak kentara; tanda air gambar/video bertahan dari kompresi, pemotongan, filter, perubahan kecepatan bingkai. Tanda Tangan Stabil (Fernandez dkk., ICCV 2023, arXiv:2303.15435) — menyempurnakan dekoder difusi laten sehingga setiap output berisi pesan tetap; gambar yang dihasilkan dipotong (10% konten) terdeteksi >90% pada FPR<1e-6. Tindak lanjut "Tanda Tangan Stabil Tidak Stabil" (arXiv:2405.07145, Mei 2024) — penyesuaian menghilangkan tanda air sambil menjaga kualitas. C2PA — standar metadata yang ditandatangani secara kriptografis dan anti kerusakan (C2PA 2.2 Penjelasan 2025). Watermarking dan C2PA saling melengkapi: metadata dapat dihilangkan namun memiliki sumber yang lebih kaya; tanda air tetap ada melalui transcoding tetapi membawa lebih sedikit informasi.

**Type:** Build
**Language:** Python (stdlib, embedding tanda air token + deteksi)
**Prerequisites:** Fase 10 · 04 (sampling), Fase 01 · 09 (teori informasi)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Jelaskan watermarking tingkat token (gaya teks SynthID) dan mekanisme yang dapat mendeteksinya.
- Jelaskan Stable Signature dan serangan penghapusan tahun 2024 yang merusaknya.
- Sebutkan peran C2PA dan alasan C2PA saling melengkapi dengan watermarking.
- Jelaskan batasan utama: sinyal khusus model, ketahanan parafrase, dan serangan yang menjaga makna (arXiv:2508.20228).

## Masalah

Pada tahun 2023-2024, deepfake dan konten buatan AI memasuki konteks politik dan konsumen dalam skala besar. Watermarking adalah sinyal asal teknis yang diusulkan: menandai generasi pada waktu pembuatan, mendeteksinya nanti. Bukti tahun 2025: tidak ada tanda air yang kuat tanpa syarat, namun dilapisi dengan metadata C2PA, kombinasi tersebut memberikan cerita asal yang dapat digunakan.

## Konsep

### Tanda air teks (gaya teks SynthID)

Kirchenbauer dkk. Mekanisme 2023, diproduksi oleh Google:

1. Pada setiap langkah penguraian code, hashkan token K sebelumnya untuk menghasilkan partisi kosakata pseudorandom menjadi kumpulan "hijau" dan "merah".
2. Pengambilan sample bias terhadap himpunan hijau dengan menambahkan δ ke logit hijau.
3. Generasi ini mengandung lebih banyak token hijau daripada yang dihasilkan secara kebetulan.

Deteksi: ulangi setiap awalan, hitung token hijau dalam pembuatan, hitung skor-z. Skor-z >0 untuk teks yang diberi watermark, ~0 untuk teks manusia.

Properti:
- Tidak terlihat oleh pembaca (δ cukup kecil sehingga penurunan kualitasnya kecil).
- Dapat dideteksi dengan akses ke fungsi partisi kosakata.
- Tidak kuat untuk diparafrasekan — menulis ulang teks akan merusak sinyal.

Teks SynthID bersumber terbuka pada Oktober 2024 melalui Perangkat GenAI yang Bertanggung Jawab Google.

### Tanda Tangan Stabil (gambar)

Fernandez dkk. ICCV 2023. Sempurnakan dekoder difusi laten sehingga setiap gambar yang dihasilkan berisi pesan biner tetap yang tertanam dalam representasi laten. Deteksi diterjemahkan dari laten dengan decoder saraf. Gambar yang dipotong (hingga 10% konten) terdeteksi >90% pada FPR<1e-6.

Mei 2024 "Tanda Tangan Stabil Tidak Stabil" (arXiv:2405.07145): menyempurnakan dekoder untuk menghilangkan tanda air sekaligus menjaga kualitas gambar. Penyempurnaan pasca-generasi yang merugikan itu murah; kekuatan permusuhan tanda air tersebut terbatas.### Detektor terpadu SynthID (November 2025)

Bersamaan dengan Gemini 3 Pro: detektor multi-media yang membaca sinyal SynthID dari teks, gambar, audio, dan video dalam satu API. Menyatukan tumpukan asal Google.

### C2PA

Koalisi untuk Asal dan Keaslian Konten. Standar metadata anti rusak yang ditandatangani secara kriptografis. Penjelasan C2PA 2.2 (2025). Manifes C2PA mencatat klaim asal (siapa yang menciptakan, kapan, transformasi apa) yang ditandatangani oleh kunci pencipta.

Pelengkap tanda air:
- Metadata dapat dihapus; tanda air tidak bisa (dengan mudah).
- Metadata kaya (rantai asal penuh); tanda air membawa bit.
- C2PA bergantung pada adopsi platform; tanda air di-embed secara otomatis.

Google mengintegrasikan keduanya dalam Penelusuran, Iklan, dan "Tentang gambar ini".

### Keterbatasan

- **Khusus model.** Generasi tanda air SynthID dari model yang mendukung SynthID. Generasi dari model tanpa SynthID tidak diberi watermark, jadi "tidak ada sinyal SynthID" bukanlah bukti keaslian.
- **Parafrase.** Tanda air teks tidak dapat bertahan jika parafrase mempertahankan makna.
- **Serangan transformasi.** arXiv:2508.20228 (2025) menunjukkan serangan yang mempertahankan makna yang menghancurkan tanda air teks dan banyak tanda air gambar.
- **Penghapusan penyempurnaan.** Sesuai dengan "Tanda Tangan Stabil Tidak Stabil", penyempurnaan pasca-generasi akan menghilangkan tanda air yang tersemat.

### UU AI UE Pasal 50

Code Transparansi untuk pelabelan konten yang dihasilkan AI (draf pertama Desember 2025, draf kedua Maret 2026, diharapkan akhir Juni 2026 per [halaman status Komisi Eropa](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)). Code ini masih dalam rancangan pada April 2026 dan jangka waktunya dapat berubah. Layer regulasi yang memerlukan layer teknis. Deepfake harus diberi label.

### Cocok untuk Fase 18

Lesson 22-23 membahas tentang apa yang dipancarkan model (data pribadi, sinyal asal). Lesson 27 mencakup training tata kelola data. Lesson 24 adalah kerangka peraturan yang memerlukan langkah-langkah teknis ini.

## Pakai

`code/main.py` membuat tanda air teks mainan. Token adalah bilangan bulat 0..N-1; bias pengambilan sample yang diberi tanda air terhadap kumpulan hijau yang ditentukan hash. Detektor menghitung skor z token hijau. kamu dapat mengamati deteksi pada generasi 1000 token, melihat parafrase menghancurkan sinyal, dan mengukur tingkat positif palsu pada teks manusia.

## Kirim

Lesson ini menghasilkan `outputs/skill-provenance-audit.md`. Mengingat penerapan konten dengan klaim asal, maka hal ini akan mengaudit: mekanisme watermark (jika ada), rantai penandatanganan C2PA (jika ada), kekuatan masing-masing watermark, dan cakupan per-modalitas.

## Latihan

1. Jalankan `code/main.py`. Laporkan skor-z untuk pembuatan 1000 token yang diberi watermark vs teks yang ditulis oleh manusia. Identifikasi tingkat positif palsu pada ambang kepercayaan 95%.

2. Menerapkan serangan parafrase yang menggantikan 30% token dengan sinonim. Ukur kembali skor-z.

3. Baca Kirchenbauer dkk. 2023 Bagian 6 tentang ketahanan. Mengapa tanda air teks gagal diparafrasekan tetapi tanda air gambar tetap bertahan saat dipotong?

4. Rancang penerapan yang menggunakan metadata SynthID-teks + C2PA. Jelaskan rantai asal yang dilihat konsumen. Identifikasi satu mode kegagalan setiap komponen.

5. Hasil "Stable Signature is Unstable" tahun 2024 menunjukkan fine-tuning menghilangkan watermark gambar. Rancang kontrol penerapan yang membatasi serangan ini — misalnya, memerlukan rilis pos pemeriksaan yang telah disesuaikan dan ditandatangani.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| SynthID | "Tanda air Google" | Sinyal asal lintas modal; teks, gambar, audio, video |
| Tanda air token | "Gaya Kirchenbauer" | Tanda air teks pengambilan sample bias dapat dideteksi melalui skor z token hijau |
| Tanda Tangan Stabil | "tanda air gambar" | Tanda air dekoder yang disempurnakan; ICCV 2023 |
| C2PA | "standar metadata" | Metadata asal bukti kerusakan yang ditandatangani secara kriptografis |
| Ketahanan parafrase | "apakah penulisan ulang merusaknya" | Properti tanda air teks; saat ini terbatas |
| Penghapusan sempurna | "tanda air permusuhan" | Serangan yang menghilangkan tanda air gambar melalui penyempurnaan dekoder |
| Detektor lintas modal | "SynthID terpadu" | November 2025 API terpadu lintas modalitas |

## Bacaan Lanjutan

- [Kirchenbauer dkk. — Tanda Air untuk Large Language Model (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — mekanisme tanda air token
- [Fernandez dkk. — Tanda Tangan Stabil (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) — kertas tanda air gambar
- ["Tanda Tangan Stabil Tidak Stabil" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145) — serangan penghapusan
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — tanda air lintas modal
- [Penjelasan C2PA 2.2 (2025)](https://c2pa.org/spesifikasi/spesifikasi/2.2/explainer/Explainer.html) — standar metadata
