# Capstone 11 — Dasbor Observabilitas & Evaluasi LLM

> Langfuse menjadi open-core. Arize Phoenix menerbitkan pemetaan semester GenAI 2026. Helicone dan Braintrust meningkatkan atribusi biaya per pengguna dua kali lipat. OpenLLMetry Traceloop menjadi instrumentasi SDK de-facto. Bentuk produksinya adalah ClickHouse untuk jejak, Postgres untuk metadata, Next.js untuk UI, dan sejumlah kecil tugas eval (DeepEval, RAGAS, LLM-judge) yang menjalankan jejak sample. Build satu yang dihosting sendiri, serap dari setidaknya empat kelompok SDK, dan tunjukkan penangkapan regresi yang disuntikkan dalam waktu kurang dari lima menit.

**Type:** Batu penjuru
**Language:** TypeScript (UI), Python / TypeScript (penyerapan + evals), SQL (ClickHouse)
**Prerequisites:** Fase 11 (rekayasa LLM), Fase 13 (peralatan), Fase 17 (infrastruktur), Fase 18 (keselamatan)
**Fase yang dilakukan:** P11 · P13 · P17 · P18
**Waktu:** 25 jam

## Masalah

Setiap tim AI yang menjalankan lalu lintas produksi pada tahun 2026 mempertahankan bidang observasi di samping model. Atribusi biaya. Deteksi halusinasi. Pemantauan arus. Sinyal pembobolan penjara. Dasbor SLO. Peringatan kebocoran PII. Referensi sumber terbuka — Langfuse, Phoenix, OpenLLMetry — menyatu pada konvensi semantik OpenTelemetry GenAI sebagai skema penyerapan. kamu sekarang dapat menginstrumentasikan OpenAI, Anthropic, Google, LangChain, LlamaIndex, dan vLLM dengan satu SDK dan mengirimkan rentang yang kompatibel.

kamu akan membuat dasbor yang dihosting sendiri yang menyerap setidaknya empat kelompok SDK, menjalankan serangkaian kecil tugas eval pada sample jejak, mendeteksi penyimpangan, dan memberi peringatan. Bilah pengukuran: jika diberi regresi yang sengaja dimasukkan (permintaan yang mulai menghasilkan PII), dasbor menangkapnya dan memicu peringatan dalam waktu kurang dari lima menit.

## Konsep

Penyerapan adalah OTLP HTTP. SDK menghasilkan rentang semkonv GenAI: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.response.id`, `llm.prompts`, `llm.completions`. Rentang lahan di ClickHouse untuk analisis kolom; metadata (pengguna, sesi, aplikasi) masuk ke Postgres.

Evaluasi dijalankan sebagai tugas batch pada jejak sample. DeepEval menilai kesetiaan, toksisitas, dan relevansi jawaban. RAGAS menilai metrik pengambilan ketika jejak membawa konteks pengambilan. Juri LLM khusus menjalankan pemeriksaan khusus domain (kebocoran PII, respons di luar kebijakan). Eval berjalan menulis kembali ke ClickHouse yang sama dengan rentang eval yang ditautkan ke jejak induk.

Deteksi penyimpangan mengamati distribusi ruang embedding dari waktu ke waktu (divergensi PSI atau KL pada embedding cepat) ditambah tren skor evaluasi. Peringatan memberi makan Prometheus Alertmanager dan kemudian Slack / PagerDuty. UI-nya adalah Next.js 15 dengan Recharts.

## Arsitektur

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## Tumpukan

- Penyerapan: OpenTelemetry SDK + konvensi semantik GenAI; Transportasi HTTP OTLP
- Kolektor: Kolektor OpenTelemetry dengan prosesor pengambilan sample ekor (untuk pengendalian biaya)
- Penyimpanan: ClickHouse untuk span, Postgres untuk metadata, S3 untuk arsip acara mentah
- Evals: DeepEval, RAGAS 0.2, paket evaluator Arize Phoenix, juri LLM khusus
- Drift: PSI / KL pada kumpulan prompt embeddings (Transformer kalimat) setiap minggu
- Peringatan: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 Router Aplikasi + Rechart + tindakan server
- SDK yang didukung langsung: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

## Build

1. **Konfigurasi kolektor.** OpenTelemetry Collector dengan penerima HTTP OTLP, tail-sampler yang menyimpan 100% jejak kesalahan dan 10% keberhasilan, dan mengekspor ke ClickHouse dan S3.2. **Skema ClickHouse.** Tabel `spans` dengan kolom yang mencerminkan semconv GenAI: `gen_ai_system`, `gen_ai_request_model`, `input_tokens`, `output_tokens`, `latency_ms`, `prompt_hash`, `trace_id`, `parent_span_id`, ditambah tas JSON untuk muatan panjang. Tambahkan indeks sekunder berdasarkan user_id dan app_id.

3. **Uji cakupan SDK.** Tulis aplikasi klien kecil menggunakan setiap SDK (OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM) dengan instrumen otomatis OpenLLMetry. Verifikasi setiap menghasilkan rentang GenAI kanonik yang mendarat di ClickHouse.

4. **Pekerjaan evaluasi.** Pekerjaan terjadwal membaca jejak sample 15 menit terakhir dan menjalankan kesetiaan DeepEval, toksisitas, dan relevansi jawaban. Output adalah rentang eval yang ditautkan ke jejak induk.

5. **Hakim LLM khusus.** Juri kebocoran PII: jika diberi tanggapan, panggil penjaga LLM untuk menilai kemungkinan kebocoran PII. Respons dengan skor tinggi masuk ke antrean triase.

6. **Deteksi penyimpangan.** Tugas mingguan menghitung PSI antara embedding cepat yang dikumpulkan minggu ini dan garis dasar 4 minggu berikutnya. Jika PSI di atas ambang batas, waspada.

7. **Dasbor.** Next.js 15 dengan halaman: ikhtisar (rentang/dtk, biaya/pengguna, latensi p95), jejak (penelusuran + air terjun), eval (tren kesetiaan, toksisitas), penyimpangan (PSI seiring waktu), peringatan.

8. **Rantai peringatan.** Eksportir Prometheus membaca agregat skor eval dan persentil latensi; Alertmanager mengarahkan ke Slack untuk mendapatkan peringatan dan PagerDuty untuk pelanggaran kritis.

9. **Pemeriksaan regresi.** Menyuntikkan bug: chatbot yang dievaluasi mulai membocorkan SSN palsu 1% sepanjang waktu. Ukur MTTR: ​​dari bug yang diterapkan hingga peringatan Slack.

## Pakai

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## Kirim

`outputs/skill-llm-observability.md` adalah hasil yang dapat dicapai. Mengingat aplikasi LLM, dasbor menyerap jejaknya, menjalankan eval, memperingatkan penyimpangan, dan menampilkan perincian biaya/pengguna di Next.js.

| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | Cakupan skema jejak | Jumlah kelompok SDK yang memproduksi rentang GenAI kanonik (target: 6+) |
| 20 | Evaluasi kebenaran | Skor DeepEval / RAGAS vs set berlabel tangan |
| 20 | UX Dasbor | MTTR pada regresi yang disuntikkan (di bawah target 5 menit) |
| 20 | Biaya/skala | Penyerapan berkelanjutan pada rentang 1k/dtk tanpa simpanan |
| 15 | Peringatan + deteksi penyimpangan | Rantai Prometheus/Alertmanager dijalankan dari ujung ke ujung |
| **100** | | |

## Latihan

1. Tambahkan instrumentasi khusus untuk kerangka Haystack. Verifikasi rentang kanonik di ClickHouse dengan atribut `gen_ai.*` yang setia.

2. Tukar DeepEval dengan evaluator Phoenix pada jalur yang sama. Ukur penyimpangan skor antara dua mesin evaluasi.

3. Pertajam pendeteksi penyimpangan: hitung PSI per id aplikasi, bukan secara global. Tampilkan jalur drift per aplikasi.

4. Tambahkan halaman "dampak pengguna": biaya per pengguna dan tingkat kegagalan per pengguna dengan grafik mini.

5. Membangun kebijakan pengambilan sample ekor yang mempertahankan 100% jejak dengan toksisitas > 0,5 ditambah 10% sample bertingkat sisanya. Ukur bias pengambilan sample yang diperkenalkan.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Semester GenAI | "Atribut Otel LLM" | Spesifikasi OpenTelemetry 2025 untuk atribut rentang LLM (sistem, model, token) |
| Pengambilan sample ekor | "Sample pasca-pelacakan" | Kolektor memutuskan untuk menyimpan atau menghapus jejak setelah selesai (dapat mengintip kesalahan) |
| PSI | "Indeks stabilitas penduduk" | Metrik drift yang membandingkan dua distribusi; > 0,2 biasanya menandakan penyimpangan yang berarti |
| Hakim LLM | "Eval sebagai model" | Sebuah LLM menilai output LLM lain pada rubrik (kesetiaan, toksisitas, PII) |
| Kebijakan pengambilan sample ekor | "Pertahankan aturan" | Aturan yang memutuskan jejak mana yang akan dipertahankan vs dihilangkan; error + tingkat sample |
| Rentang evaluasi | "Jejak eval tertaut" | Rentang anak membawa skor eval yang ditautkan ke rentang panggilan LLM asli |
| Biaya per pengguna | "Satuan ekonomi" | Biaya dolar dikaitkan ke user_id melalui jendela; metrik produk utama |

## Bacaan Lanjutan

- [Langfuse](https://github.com/langfuse/langfuse) — referensi platform observasi inti terbuka
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — referensi alternatif dengan dukungan drift yang kuat
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — rangkaian SDK instrumentasi otomatis
- [Konvensi semantik OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — skema penyerapan
- [Helicone](https://www.helicone.ai) — kemampuan observasi yang dihosting secara alternatif
- [Braintrust](https://www.braintrust.dev) — platform eval-first alternatif
- [Dokumentasi ClickHouse](https://clickhouse.com/docs) — toko rentang kolom
- [DeepEval](https://github.com/confident-ai/deepeval) — perpustakaan evaluator
