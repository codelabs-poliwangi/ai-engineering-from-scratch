# Waktu Proses Produksi: Antrian, Acara, Cron

> Agen produksi berjalan dalam enam bentuk waktu proses: respons permintaan, streaming, eksekusi yang tahan lama, latar belakang berbasis antrean, berdasarkan peristiwa, dan terjadwal. Pilih bentuknya sebelum kamu memilih kerangkanya. Observabilitas menahan weight di setiap bentuk.

**Type:** Learn
**Language:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 22 (Suara)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Sebutkan enam bentuk waktu proses produksi dan cocokkan masing-masing dengan framework/pola produk.
- Jelaskan mengapa eksekusi yang tahan lama (LangGraph) penting untuk tugas jangka panjang.
- Jelaskan runtime berdasarkan peristiwa dan kapan Agen Terkelola Claude cocok.
- Jelaskan klaim kemampuan observasi sebagai penahan weight untuk agen multi-langkah.

## Masalah

Agen produksi gagal karena notebook Jupyter tidak muncul: waktu tunggu jaringan habis pada langkah 37, pengguna menutup telepon di tengah panggilan suara, tugas cron mati saat mesin dinyalakan ulang, pekerja latar belakang kehabisan memori. Bentuk runtime menentukan kegagalan mana yang bisa bertahan.

## Konsep

### Permintaan-tanggapan

- HTTP Sinkron. Pengguna menunggu selesai.
- Hanya layak untuk tugas singkat (<30 detik).
- Tumpukan: Agno (Python + FastAPI), Mastra (TypeScript + Express/Hono/Fastify/Koa).
- Observabilitas: log akses HTTP standar + rentang OTel.

### Streaming

- SSE atau WebSocket untuk output progresif.
- LiveKit memperluas ini ke WebRTC untuk suara/video (Lesson 22).
- Tumpukan: framework apa pun dengan dukungan streaming + frontend yang menangani SSE/WS.
- Observabilitas: waktu per bagian, latensi token pertama, latensi ekor.

### Eksekusi yang tahan lama

- Negara pos pemeriksaan setelah setiap langkah; melanjutkan otomatis jika gagal.
- Model aktor AutoGen v0.4 mengisolasi kegagalan pada satu agen (Lesson 14).
- Pembeda inti LangGraph (Lesson 13).
- Penting ketika jumlah langkah tidak diketahui dan biaya pemulihan tinggi.

### Berbasis antrian / latar belakang

- Pekerjaan memasuki antrian, pekerja mengambil, hasil mengalir kembali melalui webhook atau pub/sub.
- Penting untuk agen jangka panjang (puluhan hingga ratusan langkah per tugas, sesuai pengumuman penggunaan komputer Anthropic).
- Tumpukan: Seledri (Python), BullMQ (Node), SQS + Lambda (AWS), custom.
- Observabilitas: kedalaman antrian, distribusi latensi per pekerjaan, ukuran DLQ.

### Didorong oleh peristiwa

- Agen berlangganan pemicu: email baru, PR dibuka, cron fire.
- Agen Terkelola Claude membahas hal ini secara langsung (Lesson 17).
- Arus CrewAI (Lesson 15) menyusun alur kerja deterministik berbasis peristiwa.
- Observabilitas: sumber pemicu, latensi peristiwa-ke-mulai, latensi agen.

### Dijadwalkan

- Agen berbentuk cron yang dijalankan secara berkala.
- Kombinasikan dengan eksekusi yang tahan lama sehingga kegagalan lari malam akan dilanjutkan kembali pada waktu berikutnya.
- Tumpukan: Kubernetes CronJob + framework yang tahan lama; dihosting (Render cron, Vercel cron).

### Pola penerapan 2026

- **Arus CrewAI** untuk produksi berbasis peristiwa.
- **Agno** FastAPI tanpa kewarganegaraan untuk layanan mikro Python.
- Adaptor server **Mastra** (Express, Hono, Fastify, Koa) untuk embedding.
- **Pipecat Cloud / LiveKit Cloud** untuk suara terkelola (Lesson 22).
- **Agen Terkelola Claude** untuk host asinkron yang sudah berjalan lama.

### Observabilitas bersifat menahan weight

Tanpa rentang OpenTelemetry GenAI (Lesson 23) ditambah backend Langfuse/Phoenix/Opik (Lesson 24), kamu tidak dapat men-debug agen multi-langkah yang gagal pada langkah 40. Ini bukan opsional untuk produksi. Inilah perbedaan antara "kami melakukan debug dengan cepat" dan "kami memutar ulang dari awal dengan lebih banyak logging".

### Saat runtime produksi gagal- **Pilihan bentuk salah.** Memilih permintaan-respons untuk tugas 5 menit. Pengguna menutup telepon; pekerja menumpuk; mencoba ulang senyawa.
- **Tidak ada DLQ.** Antrian pekerja tanpa surat mati. Pekerjaan yang gagal lenyap.
- **Pekerjaan latar belakang buram.** Agen latar belakang berjalan tanpa ekspor jejak. Kegagalan tidak terlihat sampai pengguna melaporkannya.
- **Melewati status tahan lama.** Setiap proses yang berlangsung > 30 detik saat kamu tidak mampu memulai ulang memerlukan eksekusi yang tahan lama.

## Build

`code/main.py` adalah demo multi-bentuk stdlib:

- Titik akhir permintaan-respons (fungsi biasa).
- Pengendali streaming (generator).
- Pekerja berbasis antrian dengan DLQ.
- Registri pemicu peristiwa.
- Penjadwal berbentuk cron.

Jalankan:

```bash
python3 code/main.py
```

Output: lima jejak yang menunjukkan perilaku masing-masing bentuk pada tugas yang sama. Logika agen yang sama, kulit terluar berbeda. Eksekusi yang tahan lama (bentuk keenam) sengaja dibahas dalam Lesson 13 dengan pos pemeriksaan LangGraph.

## Pakai

- **Respon-permintaan** untuk UX bergaya obrolan.
- **Streaming** untuk tanggapan progresif.
- **Tahan lama** untuk tugas jangka panjang.
- **Antrian** untuk batch / async / berjalan lama.
- **Acara** untuk reaktivitas agen.
- **Cron** untuk housekeeping (konsolidasi memori, evaluasi, laporan biaya).

## Kirim

`outputs/skill-runtime-shape.md` memilih bentuk runtime untuk suatu tugas dan menghubungkan persyaratan observabilitas.

## Latihan

1. Pindahkan loop ReAct Lesson 01 kamu ke keenam bentuk di tumpukan kamu. Bentuk mana yang cocok dengan permukaan produk mana?
2. Tambahkan DLQ ke demo berbasis antrian. Simulasikan 10% kegagalan pekerjaan; ukuran DLQ permukaan.
3. Tulis agen eval yang dipicu cron yang berjalan setiap malam terhadap 20 jejak teratas kamu pada hari itu.
4. Terapkan streaming dengan tekanan balik: jika klien lambat, jeda agen. Bagaimana hal ini berinteraksi dengan anggaran perputaran?
5. Baca dokumen Agen Terkelola Claude. Kapan kamu akan memindahkan agen jangka panjang yang dihosting sendiri ke agen terkelola?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Permintaan-tanggapan | "Sinkron" | Pengguna menunggu; tugas singkat saja |
| Streaming | "SSE / WS" | Output progresif; UX yang lebih baik; latensi dapat diamati per potongan |
| Eksekusi tahan lama | "Melanjutkan dari kegagalan" | negara bagian yang diperiksa; restart pada langkah terakhir |
| Berbasis antrian | "Pekerjaan latar belakang" | Produser / kumpulan pekerja / DLQ |
| Didorong oleh peristiwa | "Berbasis pemicu" | Agen bereaksi terhadap peristiwa eksternal |
| DLQ | "Antrian surat mati" | Tempat parkir untuk pekerjaan yang gagal |
| Agen yang Dikelola Claude | "Harness yang dihosting" | Async jangka panjang yang dihosting secara antropik dengan caching + pemadatan |

## Bacaan Lanjutan

- [Ikhtisar LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — detail eksekusi yang tahan lama
- [Ikhtisar Agen Terkelola Claude](https://platform.claude.com/docs/en/managed-agents/overview) — menghosting asinkron yang sudah berjalan lama
- [Antropik, Memperkenalkan penggunaan komputer](https://www.anthropic.com/news/3-5-models-and-computer-use) — "puluhan hingga ratusan langkah per tugas"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — isolasi kesalahan model aktor
