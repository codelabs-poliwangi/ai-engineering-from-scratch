# OpenTelemetry GenAI — Alat Pelacakan Panggilan End-to-End

> Seorang agen memanggil lima alat, tiga server MCP, dan dua sub-agen. kamu memerlukan satu jejak untuk semuanya. Konvensi semantik OpenTelemetry GenAI (atribut stabil di v1.37 dan lebih tinggi) adalah standar tahun 2026, yang secara asli didukung oleh Datadog, Langfuse, Arize Phoenix, OpenLLMetry, dan AgentOps. Lesson ini memberi nama atribut yang diperlukan, menelusuri hierarki rentang (agen → LLM → alat), dan mengirimkan pemancar rentang stdlib yang dapat kamu sambungkan ke eksportir OTel mana pun.

**Type:** Build
**Language:** Python (stdlib, emitor rentang OTel)
**Prerequisites:** Fase 13 · 07 (server MCP), Fase 13 · 08 (klien MCP)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Sebutkan atribut OTel GenAI yang diperlukan untuk rentang LLM dan rentang eksekusi alat.
- Build hierarki jejak yang mencakup loop agen, panggilan LLM, panggilan alat, dan pengiriman klien MCP.
- Putuskan konten apa yang akan diambil (ikut serta) vs disunting (default).
- Emit menjangkau ke kolektor lokal (Jaeger, Langfuse) tanpa menulis ulang code alat.

## Masalah

Debug dari Februari 2026: pengguna melaporkan "agen saya terkadang memerlukan waktu 30 detik untuk merespons; terkadang 3 detik." Tidak ada jejak. Log menunjukkan panggilan LLM, namun bukan pengiriman alat, bukan server MCP pulang-pergi, bukan sub-agen. kamu dapat menebaknya. Akhirnya kamu menemukan: satu server MCP kadang-kadang hang saat start dingin.

Tanpa penelusuran ujung ke ujung, kamu tidak dapat menemukannya. Otel GenAI memperbaikinya.

Konvensi tersebut diselesaikan pada tahun 2025-2026 di bawah kelompok konvensi semantik OpenTelemetry. Mereka mendefinisikan nama atribut yang stabil sehingga Datadog, Langfuse, Phoenix, OpenLLMetry, dan AgentOps semuanya mengurai rentang yang sama. Instrumen satu kali; kirim ke backend mana pun.

## Konsep

### Rentang hierarki

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

Semuanya bersarang di bawah satu id jejak. Id rentang menghubungkan hubungan orang tua-anak.

### Atribut yang diperlukan

Per semester 2025-2026:

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`.
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"`.
- `gen_ai.request.model` — string model yang diminta (misalnya `"gpt-4o-2024-08-06"`).
- `gen_ai.response.model` — model benar-benar ditayangkan.
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`.
- `gen_ai.response.id` — id respons penyedia untuk korelasi.

Untuk rentang alat:

- `gen_ai.tool.name` — pengidentifikasi alat.
- `gen_ai.tool.call.id` — id panggilan spesifik.
- `gen_ai.tool.description` — deskripsi alat (opsional).

Untuk rentang agen:

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`.

### Jenis rentang

- `SpanKind.CLIENT` untuk panggilan yang melintasi batas proses (penyedia LLM, server MCP).
- `SpanKind.INTERNAL` untuk langkah loop dan eksekusi alat milik agen.

### Ikut serta dalam pengambilan konten

Secara default, rentang membawa metrik dan waktu — bukan prompt atau penyelesaian. Muatan besar dan PII dinonaktifkan secara default. Setel `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` dan env vars pengambilan konten tertentu untuk menyertakan konten. Tinjau dengan cermat sebelum mengaktifkan di prod.

### Acara dalam rentang waktu tertentu

Peristiwa tingkat token dapat ditambahkan sebagai peristiwa rentang:

- `gen_ai.content.prompt` — memasukkan pesan.
- `gen_ai.content.completion` — pesan output.
- `gen_ai.content.tool_call` — panggilan alat seperti yang direkam.

Urutan waktu acara dalam rentang waktu tertentu untuk diputar ulang secara mendetail.

### Eksportir

Otel menjangkau ekspor ke:- **Jaeger / Tempo.** OSS, di lokasi.
- **Langfuse.** Khusus kemampuan observasi LLM; memvisualisasikan penggunaan token.
- **Arize Phoenix.** Gabungan evaluasi + penelusuran.
- **Datadog.** Komersial; secara asli mem-parsing atribut `gen_ai.*`.
- **Honeycomb.** Berorientasi kolom; ramah kueri.

Semua berbicara OTLP, format kawat. Code kamu tidak peduli.

### Propagasi di seluruh MCP

Saat klien MCP memanggil server, masukkan header traceparent W3C ke dalam permintaan. HTTP yang dapat dialirkan mendukung header standar. Stdio tidak membawa header HTTP secara asli; peta jalan spesifikasi tahun 2026 membahas penambahan bidang `_meta.traceparent` pada panggilan JSON-RPC.

Hingga permintaan tersebut dikirimkan: sertakan traceparent di `_meta` setiap permintaan secara manual. Server mencatat id jejak.

### Metrik

Selain rentang, semconv GenAI mendefinisikan metrik:

- `gen_ai.client.token.usage` — histogram.
- `gen_ai.client.operation.duration` — histogram.
- `gen_ai.tool.execution.duration` — histogram.

Gunakan ini untuk dasbor yang tidak memerlukan detail per panggilan.

### Layer AgenOps

AgentOps (didirikan tahun 2024) berspesialisasi dalam observabilitas GenAI. Ini membungkus framework populer (LangGraph, Pydantic AI, CrewAI) untuk memancarkan rentang OTel secara otomatis. Berguna jika tumpukan kamu menggunakan framework yang didukung; gunakan instrumentasi manual sebaliknya.

## Pakai

`code/main.py` memancarkan rentang berbentuk OTel ke stdout (dalam format mirip OTLP-JSON) untuk agen yang memanggil LLM, mengirimkan dua alat, dan melakukan satu perjalanan pulang pergi MCP. Tidak ada eksportir nyata — lesson berfokus pada bentuk rentang dan kumpulan atribut. Tempel hasilnya ke penampil yang kompatibel dengan OTLP atau baca saja.

Apa yang harus dilihat:

- ID jejak dibagikan ke semua rentang.
- Tautan orang tua-anak dikodekan melalui `parentSpanId`.
- Atribut `gen_ai.*` yang diperlukan telah diisi.
- Pengambilan konten dinonaktifkan secara default; satu skenario menyalakannya melalui env var.

## Kirim

Lesson ini menghasilkan `outputs/skill-otel-genai-instrumentation.md`. Dengan adanya basis code agen, keterampilan tersebut menghasilkan rencana instrumentasi: tempat menambahkan rentang, atribut mana yang akan diisi, dan eksportir mana yang akan ditargetkan.

## Latihan

1. Jalankan `code/main.py`. Hitung rentangnya dan identifikasi mana yang KLIEN vs INTERNAL.

2. Aktifkan pengambilan konten (env var) dan konfirmasikan peristiwa `gen_ai.content.prompt` dan `gen_ai.content.completion` muncul. Perhatikan implikasinya terhadap PII.

3. Tambahkan metrik eksekusi alat `gen_ai.tool.execution.duration` dan keluarkan sebagai sample histogram per panggilan.

4. Sebarkan traceparent dari rentang agen induk ke dalam bidang `_meta.traceparent` permintaan MCP. Pastikan server MCP akan melihat id jejak yang sama.

5. Baca spesifikasi semkonv OTel GenAI. Identifikasi satu atribut yang tercantum dalam semconv yang TIDAK dikeluarkan oleh code lesson ini. Tambahkan itu.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Otel | "Telemetri Terbuka" | Standar terbuka untuk jejak, metrik, log |
| Semester GenAI | "Konvensi semantik GenAI" | Nama atribut stabil untuk rentang LLM / alat / agen |
| `gen_ai.*` | "Ruang nama atribut" | Semua atribut GenAI menggunakan awalan ini |
| Rentang | "Operasi berjangka waktu" | Satuan kerja yang mempunyai awal, akhir, dan atribut |
| Jejak | "Keturunan lintas rentang" | Pohon bentang berbagi id jejak |
| SpanKind | "KLIEN/SERVER/INTERNAL" | Petunjuk tentang arah bentang |
| OTLP | "Protokol Jalur OpenTelemetri" | Format kawat untuk eksportir |
| Konten keikutsertaan | "Pengambilan segera / selesai" | Mati secara default; env var untuk mengaktifkan |
| jejak orang tua | "Tajuk W3C" | Menyebarkan konteks penelusuran ke seluruh layanan |
| Eksportir | "Pengirim khusus backend" | Komponen yang mengirimkan span ke Jaeger / Datadog / dll |

## Bacaan Lanjutan

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — konvensi kanonis untuk rentang, metrik, dan peristiwa GenAI
- [OpenTelemetry — rentang GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — daftar atribut rentang LLM dan eksekusi alat
- [OpenTelemetry — rentang agen GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — rentang tingkat agen `invoke_agent`
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — sumber kebenaran yang dihosting GitHub
- [Datadog — konvensi semantik LLM OTel](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — panduan integrasi produksi
