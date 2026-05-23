# Agno dan Mastra: Waktu Proses Produksi

> Agno (Python) dan Mastra (TypeScript) adalah pasangan waktu proses produksi tahun 2026. Agno bertujuan untuk instantiasi agen mikrodetik dan backend FastAPI tanpa kewarganegaraan. Mastra mengirimkan agen, alat, alur kerja, perutean model terpadu, dan penyimpanan komposit pada substrat Vercel AI SDK.

**Type:** Learn
**Language:** Python, TypeScript
**Prerequisites:** Fase 14 · 01 (Agent Loop), Fase 14 · 13 (LangGraph)
**Waktu:** ~45 menit

## Tujuan Pembelajaran

- Identifikasi target kinerja Agno dan kapan target tersebut penting.
- Beri nama tiga primitif Mastra — Agen, Alat, Alur Kerja — dan adaptor server yang didukung.
- Jelaskan mengapa backend FastAPI dengan cakupan sesi stateless adalah jalur produksi Agno yang direkomendasikan.
- Pilih Agno vs Mastra untuk tumpukan tertentu (Python-first vs TypeScript-first).

## Masalah

LangGraph, AutoGen, CrewAI memiliki banyak framework. Tim yang menginginkan "hanya perulangan agen, cepat, dalam waktu proses saya" menggunakan Agno (Python) atau Mastra (TypeScript). Keduanya memperdagangkan beberapa primitif yang dimiliki framework untuk kecepatan mentah dan kesesuaian yang lebih ketat dengan tumpukan di sekitarnya.

## Konsep

### Agno

- Waktu proses Python, sebelumnya Phi-data.
- "Tidak ada grafik, rantai, atau pola berbelit-belit — hanya python murni."
- Target kinerja dari dokumen mereka: ~2μs instans agen, ~3,75 memori KiB per agen, ~23 penyedia model.
- Jalur produksi: backend FastAPI dengan cakupan sesi stateless. Setiap permintaan memulai agen baru; status sesi tinggal di DB.
- Multimodal asli (teks, gambar, audio, video, file) dan RAG agen.

Target kecepatan penting ketika kamu memiliki ribuan agen berumur pendek per detik (penggemar obrolan, pipeline evaluasi). Mereka tidak terlalu berarti jika satu agen berjalan selama 10 menit.

### master

- TypeScript, dibangun di atas Vercel AI SDK.
- Tiga primitif: **Agen**, **Alat** (diketik Zod), **Alur Kerja**.
- Router Model Terpadu — 3.300+ model di 94 penyedia (Maret 2026).
- Penyimpanan komposit: memori, alur kerja, kemampuan observasi ke berbagai backend; ClickHouse direkomendasikan untuk observasi dalam skala besar.
- Apache 2.0 dengan direktori `ee/` di bawah lisensi perusahaan yang tersedia sumber.
- Adaptor server untuk Express, Hono, Fastify, Koa; integrasi Next.js dan Astro kelas satu.
- Mengirimkan Mastra Studio (localhost:4111) untuk debugging.
- 22k+ bintang GitHub, 300k+ unduhan npm mingguan pada 1.0 (Jan 2026).

### Pemosisian

Tidak ada yang mencoba menjadi LangGraph. Mereka berkompetisi di:

- **Kesesuaian bahasa.** Agno untuk tim yang mengutamakan Python; Mastra untuk TypeScript-pertama.
- **Ergonomi runtime.** Agno = overhead mendekati nol; Mastra = terintegrasi dengan ekosistem Vercel.
- **Observabilitas.** Keduanya terintegrasi dengan Langfuse/Phoenix/Opik (Lesson 24) tetapi Mastra Studio adalah pihak pertama.

### Kapan harus memilih masing-masing

- **Agno** — Backend Python, banyak agen berumur pendek, persyaratan kinerja yang kuat, toko FastAPI.
- **Mastra** — Backend TypeScript, penerapan Next.js / Vercel, perutean model multi-penyedia terpadu, alat bertipe Zod.
- **LangGraph** (Lesson 13) — ketika status tahan lama dan penalaran grafik eksplisit lebih penting daripada kecepatan mentah.
- **OpenAI / Claude Agent SDK** — bila kamu menginginkan bentuk produksi penyedia (Lesson 16–17).

### Dimana letak kesalahan pola ini- **Perf-for-perf's-demi.** Memilih Agno karena "2μs" terdengar bagus ketika weight kerja adalah satu panggilan agen yang lambat per permintaan. Overhead bukanlah hambatannya.
- **Penguncian ekosistem.** Integrasi rasa Vercel dari Mastra merupakan nilai tambah di Vercel, dan minus di tempat lain.
- **Perplexity lisensi perusahaan.** Direktori `ee/` Mastra tersedia dalam sumber, bukan Apache 2.0. Bacalah lisensinya jika kamu berencana melakukan fork.

## Build

Lesson ini pada dasarnya bersifat komparatif - tidak ada satu pun artefak code yang mampu melakukan keadilan pada kedua framework tersebut. Lihat `code/main.py` untuk mainan berdampingan: alur minimal "jalankan agen, streaming output, pertahankan sesi" yang diterapkan dua kali (sekali berbentuk Agno, sekali berbentuk Mastra).

Jalankan:

```
python3 code/main.py
```

Dua jejak yang berbeda secara struktural tetapi secara fungsional setara.

## Pakai

- **Agno** — Backend Python yang membutuhkan kecepatan dan bentuk FastAPI.
- **Mastra** — Backend TypeScript dengan banyak penyedia dan alur kerja primitif.
- Keduanya mengirimkan kait observasi pihak pertama. Keduanya terintegrasi dengan Langfuse.

## Kirim

`outputs/skill-runtime-picker.md` memilih Agno, Mastra, LangGraph, atau SDK penyedia berdasarkan tumpukan, anggaran latensi, dan bentuk operasional.

## Latihan

1. Baca dokumen Agno. Port loop ReAct stdlib (Lesson 01) ke Agno. Apa yang hilang? Apa yang tersisa?
2. Baca dokumen Mastra. Port loop yang sama ke Mastra. Apa yang berubah dalam pengetikan alat (Zod vs tidak sama sekali)?
3. Tolok ukur: mengukur latensi instantiasi agen di tumpukan kamu. Apakah 2μs Agno penting bagi weight kerja kamu?
4. Rancang migrasi: jika kamu telah menjalankan CrewAI dengan Python, apa salahnya jika kamu pindah ke Agno?
5. Baca persyaratan lisensi `ee/` Mastra. Pembatasan apa yang akan mempengaruhi fork sumber terbuka?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Agno | "Agen Python Cepat" | Waktu proses agen dengan cakupan sesi tanpa status |
| master | "Agen TypeScript di Vercel AI SDK" | Agen + Alat + Alur Kerja + Model Router |
| Router Model Terpadu | "Akses multi-penyedia" | Klien tunggal untuk 3.300+ model di 94 penyedia |
| Penyimpanan komposit | "Beberapa backend" | Memori/alur kerja/observabilitas masing-masing ke penyimpanan yang berbeda |
| Sanggar Master | "Debug lokal" | localhost:4111 UI untuk agen introspeksi |
| Tersedia sumber | "Bukan OSS" | Lisensi mengizinkan pembacaan sumber tetapi membatasi penggunaan komersial |

## Bacaan Lanjutan

- [dokumen Agno Agent Framework](https://www.agno.com/agent-framework) — target kinerja, integrasi FastAPI
- [Dokumen Mastra](https://mastra.ai/docs) — primitif, adaptor server, Model Router
- [Ikhtisar LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — alternatif stateful-graph
- [Comet Opik](https://www.comet.com/site/products/opik/) — perbandingan observabilitas yang dikutip oleh integrasi Mastra
