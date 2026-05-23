# Pengembangan Agen Berbasis Evaluasi

> Panduan Anthropic: "mulai dengan prompt sederhana, optimalkan dengan evaluasi komprehensif, dan tambahkan sistem agen multi-langkah hanya jika diperlukan." Evaluasi bukanlah langkah terakhir. Ini adalah putaran luar yang mendorong setiap pilihan lainnya di Fase 14.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Semua Fase 14.
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Sebutkan tiga layer evaluasi — tolok ukur statis, offline kustom, produksi online — dan kegunaan masing-masing layer tersebut.
- Jelaskan loop ketat evaluator-optimizer.
- Jelaskan praktik terbaik tahun 2026: eval ada di sebelah code, dijalankan di CI, gerbang PR.
- Hubungkan setiap lesson Fase 14 dengan kasus evaluasi yang dihasilkannya.

## Masalah

Agen melewati demo. Mereka gagal dalam produksi dengan cara yang tidak dapat diprediksi oleh demo. Jawaban benchmark “apakah model ini mampu secara luas?” bukan "apakah agen ini mengirimkan patch yang tepat untuk produk saya?" Jawabannya: evaluasi pada tiga layer, berjalan terus menerus, dengan setiap pagar pembatas dan aturan yang dipelajari dipetakan ke kasus eval.

## Konsep

### Tiga layer evaluasi

1. **Tolok ukur statis** — SWE-bench Terverifikasi untuk code (Lesson 19), WebArena/OSWorld untuk penjelajahan / desktop (Lesson 20), GAIA untuk generalis (Lesson 19), BFCL V4 untuk penggunaan alat (Lesson 06). Gunakan untuk perbandingan lintas model dan gerbang regresi. Kontaminasi memang nyata: SWE-bench+ menemukan 32,67% kebocoran larutan. Selalu laporkan skor Terverifikasi / +-diaudit.

2. **Evaluasi offline khusus** — bentuk produk kamu:
   - LLM-sebagai-hakim (Langfuse, Phoenix, Opik — Lesson 24).
   - Berbasis eksekusi (jalankan patch, periksa tes).
   - Berbasis lintasan (bandingkan urutan aksi melawan emas; OSWorld-Human menunjukkan agen teratas 1,4-2,7x dibandingkan emas).

3. **Evaluasi online** — produksi:
   - Pemutaran ulang sesi (Langfuse).
   - Peringatan yang dipicu oleh pagar pembatas (Lesson 16, 21).
   - Pelacakan biaya / latensi per langkah (lesson 23 rentang OTel).

### Evaluator-optimizer (Antropik)

Lingkaran ketat:

1. Pengusul menghasilkan output.
2. Juri penilai.
3. Sempurnakan hingga evaluator lulus.

Ini adalah Penyempurnaan Diri (Lesson 05) yang digeneralisasikan. Alur agen apa pun yang kamu pedulikan dapat digabungkan dengan optimizer-evaluator untuk keandalan.

### Praktik terbaik tahun 2026

- Eval tinggal di sebelah code.
- Jalankan di CI pada setiap PR.
- Penggabungan gerbang pada skor eval (misalnya "tidak ada regresi > 5% vs utama").
- Setiap pagar pembatas dipetakan ke kasus eval.
- Setiap aturan yang dipelajari (Refleksi, aturan pembelajaran pro-alur kerja) dipetakan ke kasus kegagalan.

### Mengikat Fase 14 bersama-sama

Setiap lesson di Fase 14 menghasilkan kasus eval:| Lesson | Kasus eval yang dihasilkannya |
|--------|------------------------|
| 01 Lingkaran Agen | Penjaga putaran tak terbatas yang kehabisan anggaran |
| 02 UlangWOO | Perencana merencanakan ulang dengan benar ketika alat gagal |
| 03 Refleksi | Refleksi yang dipelajari berlaku pada percobaan ulang |
| 05 Memperbaiki Diri/KRITIK | Juri memberikan hasil yang disempurnakan |
| 06 Penggunaan Alat | Pemaksaan argumen berhasil; alat yang tidak diketahui ditolak |
| 07-10 Memori | Pengambilan kutipan sesuai dengan sumber; fakta basi tidak valid |
| 12 Pola Alur Kerja | Setiap pola menghasilkan output yang benar |
| 13 Grafik Lang | Resume mereproduksi status dengan tepat |
| 14 Aktor AutoGen | DLQ menangkap penangan yang mogok |
| 16 SDK Agen OpenAI | Perjalanan pagar pembatas di input kanan |
| 17 SDK Agen Claude | Hasil subagen dikembalikan ke orkestrator |
| 19-20 Tolok Ukur | SWE-bench Skor terverifikasi, tingkat keberhasilan WebArena, efisiensi OSWorld |
| 21 Penggunaan Komputer | Tangkapan pengaman per langkah disuntikkan DOM |
| 23 OTel | Rentang memancarkan atribut yang diperlukan |
| 26 Mode Kegagalan | Detektor menandai kegagalan yang diketahui |
| 27 Injeksi Segera | PVE menolak pengambilan racun |
| 28 Orkestrasi | Supervisor mengarahkan ke spesialis yang tepat |
| 29 Bentuk Waktu Proses | DLQ menangani kegagalan N% |

Jika rangkaian evaluasi kamu memiliki masing-masing kasus, kamu telah membahas Fase 14.

### Ketika pengembangan yang didorong oleh evaluasi gagal

- **Tidak ada garis dasar.** Evaluasi tanpa barang yang terakhir diketahui tidak dapat dibaca. Simpan garis dasar.
- **LLM-hakim tanpa landasan.** Hakim juga berhalusinasi. Pola CRITIC (Lesson 05) - menilai berdasarkan alat eksternal.
- **Over-fitting pada evals.** Mengoptimalkan eval berbeda dari kegunaan produksi. Putar kasus.
- **Eval tidak stabil.** Kasus non-deterministik menyebabkan alarm palsu. Sematkan benih, status snapshot.

## Build

`code/main.py` adalah harness eval stdlib:

- Registri kasus dengan kategori (benchmark, custom, online).
- Agen dengan skrip sedang diuji.
- Putaran optimizer-evaluator: mengusulkan, menilai, menyempurnakan hingga lulus atau putaran maksimal.
- Gerbang CI: tingkat kelulusan agregat + regresi terhadap baseline.

Jalankan:

```
python3 code/main.py
```

Output: lulus/gagal per kasus, tanda regresi, keputusan gerbang CI.

## Pakai

- Tulis kasus eval di repo yang sama dengan code agen kamu.
- Jalankan di setiap PR melalui CI.
- Gagal membangun regresi.
- Lacak tingkat kelulusan dari waktu ke waktu.
- Ikat setiap kegagalan produksi dengan kasus baru.

## Kirim

`outputs/skill-eval-suite.md` membuat rangkaian evaluasi tiga lapis untuk produk agen dengan gerbang CI dan pelacakan regresi.

## Latihan

1. Ambil contoh salah satu kegagalan produksi kamu. Tulis kasus eval yang mereproduksinya. Apakah agen kamu meneruskannya sekarang?
2. Build rubrik juri LLM untuk domain kamu dengan tiga dimension (faktual, nada, cakupan). Skor 50 sesi.
3. Hubungkan rangkaian eval ke CI. Gagal membangun regresi >=5%.
4. Tambahkan metrik efisiensi lintasan: berapa banyak langkah yang diambil agen vs lintasan emas?
5. Petakan setiap lesson Fase 14 ke kasus evaluasi di suite kamu. Ada yang hilang? Itu adalah celah yang harus ditutup.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Tolok ukur statis | "Evaluasi siap pakai" | Bangku SWE, GAIA, AgentBench, WebArena, OSWorld |
| Evaluasi offline khusus | "Evaluasi domain" | LLM-sebagai-hakim / eksekutif / lintasan pada bentuk produk kamu |
| Evaluasi daring | "Evaluasi produksi" | Pemutaran ulang sesi, peringatan pagar pembatas, pelacakan biaya/latensi |
| Optimizer-evaluator | "Usulkan-hakim-perbaiki" | Ulangi hingga juri lulus |
| Gerbang CI | "Gabungkan pemblokir" | Gagal membangun regresi evaluasi |
| Dasar | "Bagus yang terakhir diketahui" | Skor referensi untuk mendeteksi regresi |
| Efisiensi lintasan | "Melangkah di atas emas" | Jumlah langkah agen dibagi minimum pakar manusia |

## Bacaan Lanjutan

- [Antropik, Membangun Agen yang Efektif](https://www.anthropic.com/research/building- Effective-agents) — "mulai dari yang sederhana, optimalkan dengan eval"
- [OpenAI, SWE-bench Terverifikasi](https://openai.com/index/introducing-swe-bench-verified/) — tolok ukur yang dikurasi
- [Papan Peringkat Panggilan Fungsi Berkeley](https://gorilla.cs.berkeley.edu/leaderboard.html) — tolok ukur penggunaan alat
- [Langfuse docs](https://langfuse.com/) — evaluasi + pemutaran ulang sesi dalam latihan
