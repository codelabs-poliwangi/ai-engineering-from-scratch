# Konvensi Semantik OpenTelemetry GenAI

> GenAI SIG OpenTelemetry (diluncurkan April 2024) mendefinisikan skema standar untuk telemetri agen. Nama rentang, atribut, dan aturan pengambilan konten menyatu di seluruh vendor sehingga jejak agen memiliki arti yang sama di Datadog, Grafana, Jaeger, dan Honeycomb.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 13 (LangGraph), Fase 14 · 24 (Platform Observabilitas)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Sebutkan kategori rentang GenAI: model/klien, agen, alat.
- Bedakan `invoke_agent` CLIENT vs rentang INTERNAL dan kapan masing-masing berlaku.
- Cantumkan atribut GenAI tingkat atas: nama penyedia, model permintaan, ID sumber data.
- Jelaskan kontrak pengambilan konten: ikut serta, `OTEL_SEMCONV_STABILITY_OPT_IN`, rekomendasi referensi eksternal.

## Masalah

Setiap vendor menciptakan nama rentangnya sendiri. Tim operasi akhirnya membuat dasbor per framework. GenAI SIG OpenTelemetry memperbaikinya dengan menetapkan satu standar yang ditargetkan oleh seluruh ekosistem.

## Konsep

### Rentang kategori

1. **Rentang model / klien.** Mencakup panggilan LLM mentah. Dipancarkan oleh SDK penyedia (Anthropic, OpenAI, Bedrock) dan adaptor model framework.
2. **Rentang agen.** `create_agent` (saat agen dibuat) dan `invoke_agent` (saat dijalankan).
3. **Rentang alat.** Satu per pemanggilan alat; terhubung ke rentang agen melalui hubungan orang tua-anak.

### Penamaan rentang agen

- Nama rentang: `invoke_agent {gen_ai.agent.name}` jika diberi nama; mundur ke `invoke_agent`.
- Jenis bentang:
  - **KLIEN** — untuk layanan agen distance jauh (OpenAI Assistants API, Bedrock Agents).
  - **INTERNAL** — untuk framework agen yang sedang dalam proses (LangChain, CrewAI, ReAct lokal).

### Atribut utama

- `gen_ai.provider.name` — `anthropic`, `openai`, `aws.bedrock`, `google.vertex`.
- `gen_ai.request.model` — ID model.
- `gen_ai.response.model` — model yang diselesaikan (mungkin berbeda dari permintaan karena perutean).
- `gen_ai.agent.name` — pengenal agen.
- `gen_ai.operation.name` — `chat`, `completion`, `invoke_agent`, `tool_call`.
- `gen_ai.data_source.id` — untuk RAG: korpus atau toko mana yang dikonsultasikan.

Konvensi khusus teknologi ada untuk Anthropic, Azure AI Inference, AWS Bedrock, OpenAI.

### Pengambilan konten

Aturan default: instrumentasi TIDAK BOLEH menangkap input/output secara default. Pengambilan diikutsertakan melalui:

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

Pola produksi yang disarankan: menyimpan konten secara eksternal (S3, penyimpanan log kamu), mencatat referensi pada rentang (ID penunjuk, bukan prosa). Inilah lesson 27 tentang pertahanan terhadap keracunan konten yang dihubungkan dengan kemampuan observasi.

### Stabilitas

Sebagian besar konvensi masih bersifat eksperimental mulai Maret 2026. Ikut serta dalam pratinjau stabil dengan:

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ memetakan atribut GenAI secara asli ke dalam skema Observabilitas LLM-nya. Backend lainnya (Grafana, Honeycomb, Jaeger) mendukung atribut mentah.

### Dimana letak kesalahan pola ini

- **Menangkap prompt lengkap dalam rentang waktu.** PII, rahasia, data pelanggan dalam jejak yang dapat dibaca oleh operasi. Simpan secara eksternal.
- **Tidak `gen_ai.provider.name`.** Dasbor multi-penyedia rusak saat atribusi tidak ada.
- **Rentang tanpa tautan induk.** Rentang alat yatim piatu. Selalu sebarkan konteks.
- **Tidak menyetel keikutsertaan stabilitas.** Atribut kamu mungkin diganti namanya saat peningkatan backend.

## Build`code/main.py` mengimplementasikan konvensi GenAI yang cocok dengan emitor rentang stdlib:

- `Span` dengan skema atribut GenAI.
- `Tracer` dengan `start_span`, konteks bertingkat.
- Proses agen bernaskah yang memancarkan: `create_agent`, `invoke_agent` (INTERNAL), rentang per alat, `chat` rentang untuk panggilan LLM.
- Mode pengambilan konten yang menyimpan prompt secara eksternal dan mencatat ID pada rentang.

Jalankan:

```
python3 code/main.py
```

Output: pohon rentang dengan semua atribut GenAI yang diperlukan, dan "penyimpanan eksternal" yang menampilkan referensi konten keikutsertaan.

## Pakai

- **Datadog LLM Observability** (v1.37+) memetakan atribut secara asli.
- **Langfuse / Phoenix / Opik** (Lesson 24) — melakukan instrumen otomatis pada ekosistem.
- **Jaeger / Honeycomb / Grafana Tempo** — jejak OTel mentah; membangun dasbor dari atribut GenAI.
- **Dihosting sendiri** — menjalankan OTel Collector dengan prosesor GenAI.

## Kirim

`outputs/skill-otel-genai.md` wire OTel GenAI berkembang menjadi agen yang sudah ada dengan default pengambilan konten dan penyimpanan referensi eksternal.

## Latihan

1. Instrumen loop ReAct Lesson 01 kamu dengan `invoke_agent` (INTERNAL) + rentang per alat. Kirim ke instance Jaeger.
2. Tambahkan pengambilan konten dalam mode "hanya referensi": petunjuk ke SQLite, atribut span hanya membawa ID baris.
3. Baca spesifikasi untuk `gen_ai.data_source.id`. Masukkan ke dalam pencarian Lesson 09 Mem0 kamu.
4. Tetapkan `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` dan pastikan atribut kamu tidak diganti namanya oleh kolektor.
5. Buat dasbor: "kesalahan alat mana yang berkorelasi dengan model mana" hanya dari atribut GenAI.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| GenAI SIG | "Grup OpenTelemetry GenAI" | Kelompok kerja Otel mendefinisikan skema |
| panggil_agen | "Rentang agen" | Nama rentang yang mewakili eksekusi agen |
| Rentang KLIEN | "Panggilan distance jauh" | Rentang panggilan ke layanan agen distance jauh |
| Rentang INTERNAL | "Dalam proses" | Rentang untuk menjalankan agen dalam proses |
| gen_ai.penyedia.nama | "Penyedia" | antropik / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "Sumber RAG" | Korpus/penyimpanan mana yang pengambilannya berhasil |
| Pengambilan konten | "Pencatatan cepat" | Keikutsertaan dalam pengambilan pesan; simpan secara eksternal di prod |
| Keikutsertaan stabilitas | "Mode pratinjau" | Env var untuk embed konvensi eksperimental |

## Bacaan Lanjutan

- [Konvensi semantik OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — spesifikasi
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — GenAI mencakup secara default
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — rentang OTel bawaan
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — propagasi konteks jejak W3C
