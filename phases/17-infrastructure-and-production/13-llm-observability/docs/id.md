# Pemilihan Tumpukan Observabilitas LLM

> Pasar observasi pada tahun 2026 terbagi menjadi dua kategori. Platform pengembangan (LangSmith, Langfuse, Comet Opik) menggabungkan pemantauan dengan evaluasi, manajemen cepat, pemutaran ulang sesi. Alat gateway/instrumentasi (Helicone, SigNoz, OpenLLMetry, Phoenix) fokus pada telemetri. Langfuse adalah inti berlisensi MIT dengan saldo OSS yang kuat (50 ribu acara/bulan cloud gratis). Phoenix adalah OpenTelemetry-native di bawah Elastic License 2.0 — sangat baik untuk visualisasi drift/RAG, bukan backend produksi persisten. Arize AX menggunakan integrasi Zero-copy Iceberg/Parquet yang diklaim 100x lebih murah dibandingkan observabilitas monolitik. LangSmith memimpin untuk LangChain/LangGraph, $39/pengguna/bln, host mandiri di Perusahaan saja. Helicone berbasis proxy dengan pengaturan 15-30 menit, gratis 100 ribu permintaan/bln, tetapi penelusuran agen kurang mendalam. Pola produksi umum: Gateway (Helicone/Portkey) + platform eval (Phoenix/TruLens) yang direkatkan oleh OpenTelemetry.

**Type:** Learn
**Language:** Python (stdlib, simulator pengambilan sample jejak mainan)
**Prerequisites:** Fase 17 · 08 (Metrik Inference), Fase 14 (Rekayasa Agen)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Membedakan platform pengembangan (dipaketkan: evals + prompt + sesi) dari alat gateway/telemetri (hanya jejak + metrik).
- Memetakan enam alat utama (Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik) ke kasus penggunaan lisensi, harga, dan sweet-spotnya.
- Jelaskan pola lem OpenTelemetry yang memungkinkan kamu menggabungkan alat gateway dengan platform eval terpisah.
- Sebutkan pembeda biaya tahun 2026 (pendekatan zero-copy Arize AX vs penyerapan monolitik) dan nyatakan pengganda kasarnya sebesar 100x.

## Masalah

kamu mengirimkan feature LLM. Ini berhasil. kamu tidak memiliki visibilitas terhadap kegagalan cepat, perulangan alat, regresi latensi, lonjakan biaya, atau tingkat keberhasilan cache cepat. kamu mencari di Google "pengamatan LLM" dan mendapatkan delapan alat yang semuanya mengklaim bahwa alat tersebut memecahkan masalah yang sama pada tiga titik harga yang berbeda.

Mereka tidak memecahkan masalah yang sama. LangSmith menjawab "mengapa proses LangGraph ini gagal?" Phoenix menjawab "apakah pipa RAG saya melayang?" Helicone menjawab "aplikasi mana yang membakar token?" Langfuse menjawab, "bisakah saya menghosting sendiri semuanya?" Alat yang berbeda, audiens yang berbeda.

Pengambilan melibatkan empat sumbu: tumpukan (LangChain? SDK mentah? multi-vendor?), toleransi lisensi (hanya MIT? Elastic OK? komersial baik?), anggaran (tingkat gratis? $100/bln? $1000/bln?), dan self-host (harus? bagus untuk dimiliki? tidak pernah?).

## Konsep

### Dua kategori

**Platform pengembangan** menggabungkan kemampuan observasi dengan evaluasi, manajemen cepat, pembuatan versi set data, pemutaran ulang sesi. kamu menjalankan eksperimen, melihat permintaan mana yang berhasil, regresi dataset, permintaan baru terhadap pemenang lama. LangSmith, Langfuse, Komet Opik.

**Alat gateway/telemetri** panggilan inference instrumen — prompt, respons, token, latensi, model, biaya. Helicone, SigNoz, OpenLLMetry, Phoenix. Minimalis. Dapat digabungkan dengan alat evaluasi terpisah melalui OpenTelemetry.

### Langfuse — saldo OSS

- Inti Apache / MIT berlisensi; host mandiri melalui Docker.
- Tingkat gratis cloud: 50 ribu acara/bulan. Dibayar: $29/bln untuk tim.
- Evaluasi, manajemen cepat, jejak, dataset. Cakupan yang wajar untuk keempat feature platform pengembang.
- Sweet spot: kamu menginginkan feature kelas LangSmith tetapi harus menghosting sendiri atau tetap menggunakan lisensi OSS.

### Phoenix (Arize) — mengutamakan telemetri, asli OpenTelemetri- Lisensi Elastis 2.0; self-host sepele.
- Sangat baik dalam visualisasi RAG dan drift. Plot sebar ruang embedding dikirimkan sebagai kelas satu.
- Tidak dirancang sebagai backend produksi yang persisten — terutama kemampuan observasi waktu pengembangan.
- Sweet spot: pengembangan pipeline RAG, debugging drift, dipasangkan dengan gateway terpisah untuk produksi.

### Arize AXE — permainan skala

- Komersial. Integrasi data lake tanpa salinan melalui Iceberg/Parquet.
- Klaim ~100x lebih murah dibandingkan observabilitas monolitik (kelas Datadog) dalam skala besar. Perhitungannya: kamu menyimpan jejak di Parket kamu sendiri di S3; Arize langsung membaca.
- Sweet spot: >10 juta pelacakan/hari, data lake yang ada, menginginkan dasbor khusus LLM tanpa harga Datadog.

### LangSmith — LangChain/LangGraph terlebih dahulu

- Komersial, $39/pengguna/bulan. Host mandiri hanya di Perusahaan.
- Terbaik di kelasnya untuk tumpukan LangChain dan LangGraph. Jika kamu tidak menggunakan keduanya, itu kurang menarik.
- Sweet spot: tim berkomitmen pada LangChain, bersedia membayar.

### Helicone — minimum yang layak berbasis proxy

- Penyiapan 15-30 menit dengan menukar `OPENAI_API_BASE` kamu ke proxy Helicone.
- berlisensi MIT; 100 ribu permintaan/bln gratis, berbayar $20/bln+.
- Termasuk failover, caching, batas kecepatan — juga berfungsi sebagai gerbang.
- Kurang mendalam pada agen / jejak multi-langkah.
- Sweet spot: mulai cepat, aplikasi tumpukan tunggal, memerlukan gateway + kemampuan observasi dalam satu aplikasi.

### Opik (Komet) — platform pengembang OSS

- Apache 2.0, sepenuhnya OSS.
- Feature serupa diatur ke Langfuse dengan warisan Komet.
- Sweet spot: Tim ML sudah ada di Comet, menginginkan observabilitas LLM di panel yang sama.

### SigNoz — APM penuh pertama OpenTelemetry

-Apache 2.0. Menangani APM umum plus LLM melalui OpenTelemetry.
- Sweet spot: observabilitas terpadu di seluruh layanan dan panggilan LLM.

### Perekatnya: Konvensi semantik OpenTelemetry + GenAI

OpenTelemetry menerbitkan konvensi semantik GenAI pada akhir tahun 2025 (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`). Alat yang mengkonsumsi OTel dapat saling beroperasi. Pola produksi yang muncul:

1. Keluarkan OTel dengan konvensi GenAI dari setiap panggilan LLM.
2. Rute ke gateway (Helicone/Portkey) untuk sehari-hari.
3. Pengiriman ganda ke platform evaluasi (Phoenix / Langfuse) untuk regresi.
4. Arsip di data lake (Iceberg) untuk analisis jangka panjang melalui Arize AX atau DuckDB.

### Jebakan: memasang instrumen pada layer yang salah

Instrumen di dalam kerangka agen kamu (misalnya, menambahkan jejak LangSmith) memasangkan kamu ke kerangka tersebut. Instrumen pada layer HTTP/OpenAI-SDK (melalui OpenLLMetry atau gateway kamu) bersifat portabel.

### Pengambilan sample — kamu tidak bisa menyimpan semuanya

Pada >1 juta permintaan/hari, biaya retensi pelacakan penuh lebih mahal daripada panggilan LLM. Contoh berdasarkan aturan: 100% kesalahan, 100% biaya tinggi, 5% keberhasilan. Selalu jaga agregat; tetap mentah untuk ekor panjang.

### Nomor yang harus kamu ingat

- Cloud gratis Langfuse: 50 ribu acara/bulan.
- LangSmith: $39/pengguna/bulan.
- Bebas helicone: kebutuhan 100 ribu/bulan.
- Klaim Arize AX: ~100x lebih murah dibandingkan monolitik dalam skala besar.
- Konvensi OpenTelemetry GenAI: pengiriman tahun 2025, diadopsi secara luas pada tahun 2026.

## Pakai

`code/main.py` menyimulasikan 1 juta hari pelacakan di seluruh strategi retensi (100% penyerapan, pengambilan sample, pengambilan sample + kesalahan). Melaporkan biaya penyimpanan dan apa yang hilang di masing-masingnya.

## Kirim

Lesson ini menghasilkan `outputs/skill-observability-stack.md`. Berdasarkan tumpukan, skala, anggaran, postur lisensi, pilih alatnya.

## Latihan1. Tim kamu di LangChain menginginkan kemampuan observasi yang dihosting sendiri oleh OSS. Pilih Langfuse atau Opik dan justifikasi.
2. Pada 5 juta jejak/hari dengan penawaran Datadog $150K/bulan, hitung titik impas untuk Arize AX.
3. Rancang atribut OpenTelemetry GenAI yang ditetapkan sebagai pedoman organisasi kamu yang harus diamanatkan pada setiap panggilan LLM.
4. Berdebat apakah Phoenix saja sudah cukup untuk produksi. Kapan itu tidak cukup?
5. Helicone adalah overhead proxy 20ms. Pada P99 TTFT 300 ms, apakah itu dapat diterima? Bagaimana jika SLA 100 ms?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| OpenLLMetry | "OTel untuk LLM" | Instrumentasi OpenTelemetry sumber terbuka untuk LLM |
| Konvensi GenAI | "Atribut Otel" | Nama atribut OTel standar untuk panggilan LLM |
| LangSmith | "Pengamatan LangChain" | Platform komersial yang dibundel dengan ekosistem LangChain |
| Langfuse | "OSS LangSmith" | MIT OSS dengan kumpulan feature serupa |
| Phoenix | "Alat pengembang Arize" | Platform pengembangan/eval asli OpenTelemetry |
| Arize AX | "observabilitas skala" | Observabilitas Gunung Es/Parket tanpa salinan komersial |
| Helikon | "kemampuan observasi proxy" | Proksi HTTP mengumpulkan feature telemetri + gateway LLM |
| Opik | "Komet LLM" | Platform pengembangan Apache 2.0 OSS dari Comet |
| Pemutaran ulang sesi | "jejak jalankan ulang" | Putar ulang sesi agen lengkap dengan panggilan alat |
| Evaluasi | "tes luring" | Menjalankan model kandidat/prompt pada dataset berlabel |

## Bacaan Lanjutan

- [SigNoz — Alat Observabilitas LLM Teratas 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Analisis Alternatif Arize AX](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Menyiapkan Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [Konvensi Semantik OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Dokumen Arize Phoenix](https://docs.arize.com/phoenix)
- [Dokumen helicone](https://docs.helicone.ai/)
