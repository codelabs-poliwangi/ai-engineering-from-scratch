# Batch API — Diskon 50% sebagai Standar Industri

> Setiap penyedia besar mengirimkan API batch asinkron dengan diskon 50% dan penyelesaian ~24 jam. OpenAI, Anthropic, Google, dan sebagian besar platform inference (tingkat batch Fireworks, batch Together) menerapkan pola yang sama. Tumpukan tumpukan dengan caching cepat dan pipeline semalaman turun hingga ~10% dari biaya yang tidak di-cache secara sinkron. Aturannya sangat sederhana: jika tidak interaktif, maka termasuk dalam batch. Jalur pembuatan konten, klasifikasi dokumen, ekstraksi data, pembuatan laporan, pelabelan massal, penandaan katalog — apa pun yang dapat mentolerir latensi 24 jam hanya akan menyisakan uang hingga dipindahkan ke batch. Pola produksi tahun 2026 adalah melakukan triase setiap weight kerja LLM baru ke dalam tiga jalur: interaktif (sinkron dengan caching), semi-interaktif (antrian async dengan fallback), batch (semalaman, input cache ditumpuk). Weight kerja yang berpura-pura interaktif namun menoleransi latensi bermenit-menit paling banyak terbuang.

**Type:** Learn
**Language:** Python (stdlib, simulator biaya batch-vs-sync mainan)
**Prerequisites:** Fase 17 · 14 (Caching Cepat & Semantik)
**Waktu:** ~45 menit

## Tujuan Pembelajaran

- Sebutkan tiga API batch penyedia (OpenAI, Anthropic, Google) dan diskon umum 50% + jaminan penyelesaian 24 jam.
- Hitung biaya untuk menumpuk batch + input cache pada weight kerja klasifikasi semalam dan membandingkannya dengan garis dasar yang tidak di-cache secara sinkron.
- Triase weight kerja menjadi interaktif / semi-interaktif / batch dan ratakan jalurnya.
- Sebutkan dua jebakan: interaktivitas parsial (pengguna mengharapkan lebih cepat dari 24 jam) dan penyimpangan skema output (format file batch berbeda untuk setiap penyedia).

## Masalah

Tim kamu mengirimkan pipeline pembuatan laporan setiap malam. 50.000 dokumen, masing-masing diringkas, mengelompokkan ringkasan, menyusun ringkasan eksekutif. Berjalan serentak membutuhkan waktu 4 jam dengan biaya $2.000/malam. kamu mendengar tentang API batch.

Batchnya memberi kamu diskon 50%. kamu juga mengaktifkan cache cepat pada system prompt (dibagikan ke seluruh 50 ribu panggilan). Jika digabungkan, tagihannya turun menjadi $180/malam — ~9% dari nilai dasar. Pipeline pipa yang sama, tiga perubahan konfigurasi.

Batch adalah tuas termurah dalam perangkat biaya LLM yang tidak dapat ditarik oleh siapa pun. Alasannya sebagian besar bersifat organisasional: tim berpikir "waktu nyata" padahal SLA sebenarnya adalah "pada pagi hari". Lesson ini adalah tentang tidak meninggalkan 90% tagihan di atas meja.

## Konsep

### Tiga API batch

**OpenAI Batch API**: Unggahan file JSONL dengan daftar permintaan. Perputaran 24 jam yang dijanjikan (biasanya ~2-8 jam dalam praktik). Diskon 50% untuk token input dan output. `/v1/batches` titik akhir. Input yang memenuhi syarat cache juga mendapatkan harga input cache di atas.

**Kumpulan Pesan Antropis**: Unggahan JSONL. Penyelesaian 24 jam. Diskon 50%. Mendukung `cache_control` — penulisan cache bersifat eksplisit, pembacaan terjadi secara otomatis dalam batch.

**Prediksi Batch Google Vertex AI**: input BigQuery atau GCS. Diskon 50% serupa untuk Gemini. Terintegrasi dengan pipeline pipa Vertex.

### Semantik: asinkron, tidak lambat

Batchnya adalah "Saya berjanji akan kembali dalam waktu 24 jam" — bukan "ini akan memakan waktu 24 jam". P50 tipikal adalah 2-6 jam. Penyedia menjadwalkan batch kamu selama periode di luar jam sibuk ketika inventaris GPU kurang dimanfaatkan.

### Tumpukan dengan cache

Peringkasan 50 ribu dokumen dengan system prompt token 4K yang sama:- Sinkronisasi yang tidak di-cache: 50000 × ($input × 4000 + $output × 200) dengan tarif penuh.
- Cache sinkron: prompt sistem di-cache setelah penulisan pertama; sisa 49999 dapatkan input 10x lebih murah.
- Batch cache: semua hal di atas ditambah diskon 50% untuk baca dan tulis.

Tumpukan: batch + cache = ~10% dari sinkronisasi tagihan yang tidak di-cache. Weight kerja apa pun yang berjalan dalam semalam dan memiliki system prompt bersama harus menggunakan ini.

### Triase weight kerja

**Interaktif** — pengguna menunggu respons. TTFT penting. Panggilan sinkron dengan cache cepat. Tidak bisa berkelompok.

**Semi-interaktif** — pengguna mengirimkan tugas, memeriksanya kembali dalam hitungan menit. Antrean asinkron dengan fallback untuk disinkronkan jika batch tidak tersedia. Pikirkan pengindeksan RAG volume sedang.

**Batch** — pengguna mengharapkan hasil "pada pagi hari" atau "jam berikutnya". Pipeline konten, klasifikasi dalam skala besar, analisis offline. Selalu batch, selalu tumpukan cache.

Kesalahan umum: mengklasifikasikan semuanya sebagai interaktif karena jalur pipanya adalah produksi. Produksi bukanlah spesifikasi latensi — SLA adalah spesifikasi latensi.

### Perangkap interaktivitas parsial

Beberapa feature terlihat interaktif tetapi bertahan 5-10 menit. Contoh: laporan kesehatan pelanggan setiap malam dengan tombol "segarkan". Klik pengguna menyegarkan; tunggu 10 menit tidak apa-apa. Tim mengirimkannya secara sinkron. Biaya 50 penyegaran serentak 10x lipat dari biaya batch dan pengiriman melalui email.

Pertanyaan yang diajukan: "Apa arti 24 jam bagi pengguna ini?" Jika jawabannya adalah "mereka tidak akan menyadarinya", kelompokkanlah.

### Perangkap skema output

Format file batch berbeda untuk setiap penyedia:

- OpenAI: JSONL, satu permintaan per baris.
- Antropis: JSONL, satu pesan per baris; format respons tertanam.
- Vertex: Tabel BigQuery atau awalan GCS dengan TFRecord.

Menulis "satu klien batch" di seluruh penyedia berarti code adaptor per penyedia. Gateway yang mengiklankan kumpulan multi-penyedia (Portkey, LiteLLM beberapa tingkatan) masih membungkus format mentahnya dengan tipis.

### Nomor yang harus kamu ingat

- Diskon batch di seluruh penyedia: 50% tetap untuk input + output.
- SLA Turnaround: jaminan 24 jam, P50 tipikal 2-6 jam.
- Batch bertumpuk + input cache: ~10% dari biaya sinkronisasi yang tidak di-cache.
- Aturan triase weight kerja: jika latensi 24 jam dapat diterima, selalu batch.

## Pakai

`code/main.py` menghitung biaya sinkronisasi, sinkronisasi+cache, batch, dan batch+cache untuk weight kerja 50 ribu dokumen. Melaporkan penghematan dalam $ dan persen.

## Kirim

Lesson ini menghasilkan `outputs/skill-batch-triager.md`. Mengingat karakteristik weight kerja, lakukan triase menjadi interaktif/semi/batch dan perkiraan penghematan.

## Latihan

1. Jalankan `code/main.py`. Untuk pipeline 100 ribu dokumen dengan system prompt 3K token dan output 500 token, hitung penghematan tumpukan penuh (batch + cache) vs garis dasar sinkronisasi.
2. Pilih tiga feature dalam produk nyata lho. Triase masing-masing menjadi interaktif/semi/batch.
3. Seorang pengguna mengeluh bahwa laporannya memakan waktu 3 jam. Apakah itu merupakan kesalahan triase batch atau interaktif yang sah? Tuliskan kriteria keputusannya.
4. SLA pengembalian API batch kamu adalah 24 jam tetapi P99 adalah 20 jam. Bagaimana kamu mengomunikasikan hal ini kepada pengguna — bagaimana perilaku sistem downstream pada kasus edge?
5. Hitung titik impas: berapa panjang awalan bersama yang membuat batch + cache menjadi lebih murah daripada berjalan semalaman dengan GPU cadangan kamu sendiri?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| API Batch | "diskon asinkron" | Diskon 50% dengan penyelesaian 24 jam |
| JSONL | "format kumpulan" | Satu permintaan JSON per baris; Standar OpenAI/Antropik |
| Kumpulan Pesan | "Kelompok antropik" | Nama produk API batch Anthropic |
| Prediksi kumpulan | "Kumpulan simpul" | Produk API batch Vertex AI |
| SLA Perputaran | "janji 24 jam" | Jaminan, tidak khas; khasnya adalah 2-6 jam |
| Triase weight kerja | "keputusan interaktivitas" | Keputusan perutean interaktif / semi / batch |
| Skema output | "format respons" | Tata letak JSONL per penyedia; tidak portabel |
| Diskon bertumpuk | "kumpulan + cache" | ~10% dari tagihan sinkronisasi yang tidak di-cache ketika keduanya berlaku |

## Bacaan Lanjutan

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — format JSONL dan semantik `/v1/batches`.
- [Batch Pesan Antropik](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — format batch dan interaksi `cache_control`.
- [Prediksi Batch Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Semantik batch Gemini.
- [Finout — Harga OpenAI vs Anthropic API 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — Perbandingan Biaya API LLM 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
