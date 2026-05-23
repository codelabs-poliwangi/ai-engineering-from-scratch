# Observabilitas Agen: Langfuse, Phoenix, Opik

> Tiga platform observabilitas agen sumber terbuka mendominasi tahun 2026. Langfuse (MIT) — 6 juta+ pemasangan/bulan, penelusuran + manajemen cepat + evaluasi + pemutaran ulang sesi. Arize Phoenix (Elastic 2.0) — evaluasi khusus agen yang mendalam, relevansi RAG, instrumentasi otomatis OpenInference. Comet Opik (Apache 2.0) — optimization cepat otomatis, pagar pembatas, deteksi halusinasi juri LLM.

**Type:** Learn
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 23 (OTel GenAI)
**Waktu:** ~45 menit

## Tujuan Pembelajaran

- Sebutkan tiga platform observasi agen sumber terbuka teratas dan lisensinya.
- Bedakan apa yang paling kuat dari masing-masing: Langfuse (sesi mgmt + cepat), Phoenix (RAG + instrumentasi otomatis), Opik (optimization + pagar pembatas).
- Jelaskan mengapa 89% organisasi melaporkan adanya kemampuan observasi agen pada tahun 2026.
- Menerapkan pipeline trace-to-dashboard stdlib dengan evaluasi juri LLM.

## Masalah

OTel GenAI (Lesson 23) memberi kamu skemanya. kamu masih memerlukan platform yang menyerap rentang, menjalankan evaluasi, menyimpan versi cepat, dan menampilkan regresi. Ketiga pesaing tersebut masing-masing menekankan bagian siklus hidup yang berbeda.

## Konsep

### Langfuse (MIT)

- 6 juta+ pemasangan SDK/bulan, 19 ribu+ bintang GitHub.
- Feature: penelusuran, manajemen cepat dengan pembuatan versi + taman bermain, evaluasi (LLM sebagai juri, umpan balik pengguna, kustom), pemutaran ulang sesi.
- Juni 2025: modul yang sebelumnya bersifat komersial (LLM sebagai juri, antrian anotasi, eksperimen cepat, Taman Bermain) bersumber terbuka di bawah MIT.
- Paling kuat untuk: kemampuan observasi ujung ke ujung dengan loop manajemen cepat yang ketat.

### Arize Phoenix (Lisensi Elastis 2.0)

- Evaluasi spesifik agen yang lebih mendalam: pengelompokan jejak, deteksi anomali, relevansi pengambilan untuk RAG.
- Instrumentasi otomatis OpenInference asli.
- Pasangkan dengan Arize AX yang dikelola untuk produksi.
- Tidak ada pembuatan versi cepat — diposisikan sebagai alat penyimpangan/regresi perilaku bersama dengan platform yang lebih luas.
- Terkuat untuk: relevansi RAG, penyimpangan perilaku, deteksi anomali.

### Komet Opik (Apache 2.0)

- Optimization cepat otomatis melalui eksperimen A/B.
- Pagar Pembatas (redaksi PII, kendala topikal).
- Deteksi halusinasi hakim LLM.
- Tolok ukur dari pengukuran Comet sendiri: Opik logs + evals dalam 23,44 detik vs Langfuse 327,15 detik (~14x gap) — gunakan tolok ukur vendor sebagai petunjuk.
- Paling kuat untuk: putaran optimization, eksperimen otomatis, penegakan pagar pembatas.

### Data industri

Per Maxim (analisis lapangan tahun 2026): 89% organisasi memiliki kemampuan observasi agen; permasalahan kualitas merupakan hambatan utama dalam produksi (32% responden menyebutkan permasalahan tersebut).

### Memilih satu

| Butuh | Pilih |
|------|------|
| All-in-one dengan manajemen cepat | Langfuse |
| Evaluasi RAG mendalam + penyimpangan | Phoenix |
| Optimization otomatis + pagar pembatas | Opik |
| Lisensi terbuka, tanpa ELv2 | Langfuse (MIT) atau Opik (Apache 2.0) |
| Integrasi Datadog / Relik Baru | Apa saja — mereka semua mengekspor OTel |

### Dimana letak kesalahan pola ini

- **Tidak ada strategi evaluasi.** Penelusuran tanpa evaluasi hanyalah logging yang mahal.
- **Hakim LLM yang bertindak sendiri tanpa landasan.** Pola KRITIK (Lesson 05) berlaku — hakim memerlukan alat eksternal untuk verifikasi faktual.
- **Versi prompt tidak terikat dengan jejak.** Saat prod mengalami kemunduran, kamu tidak dapat membagi dua ke prompt yang menyebabkannya.

## Build

`code/main.py` mengimplementasikan pengumpul jejak stdlib + evaluator juri LLM:- Menelan bentang berbentuk GenAI.
- Kelompokkan berdasarkan sesi, tandai proses yang gagal (perjalanan pagar pembatas, evaluasi tingkat kepercayaan rendah).
- Juri LLM bernaskah yang menilai tanggapan agen pada rubrik.
- Ringkasan seperti dasbor: tingkat kegagalan, alasan kegagalan teratas, distribusi skor evaluasi.

Jalankan:

```
python3 code/main.py
```

Output: skor evaluasi per sesi dan kategorisasi kegagalan sesuai dengan apa yang akan ditampilkan Langfuse/Phoenix/Opik.

## Pakai

- **Langfuse** dihosting sendiri atau cloud; kawat melalui OTel atau SDK mereka.
- **Arize Phoenix** dihosting sendiri; OpenInference instrumen otomatis.
- **Comet Opik** dihosting sendiri atau cloud; lingkaran optimization otomatis.
- **Datadog LLM Observability** untuk tim operasi campuran+ML yang sudah menjalankan Datadog.

## Kirim

`outputs/skill-obs-platform-wiring.md` mengambil platform dan menyambungkan jejak + evals + versi prompt ke agen yang ada.

## Latihan

1. Ekspor jejak OTel selama seminggu ke cloud Langfuse (tingkat gratis). Sesi mana yang gagal? Mengapa?
2. Tulis rubrik juri LLM untuk domain kamu (kebenaran faktual, nada, kepatuhan cakupan). Uji pada 50 jejak.
3. Bandingkan pembuatan versi cepat Langfuse dengan pengelompokan jejak Phoenix. Mana yang memberitahu kamu apa yang rusak lebih cepat?
4. Baca dokumen pagar pembatas Opik. Hubungkan pagar pembatas redaksi PII ke salah satu proses agen kamu.
5. Patokan ketiganya pada korpus kamu. Abaikan nomor yang diterbitkan vendor; ukur sendiri.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Menelusuri | "Spans kolektor" | Menyerap rentang OTel / SDK; indeks berdasarkan sesi |
| Manajemen yang cepat | "CMS Cepat" | Prompt berversi terkait dengan jejak |
| LLM-sebagai-hakim | "Eval otomatis" | Pisahkan output agen skor LLM dengan rubrik |
| Pemutaran ulang sesi | "Lacak pemutaran" | Telusuri proses sebelumnya untuk debugging |
| Relevansi RAG | "Kualitas pengambilan" | Apakah konteks yang diambil cocok dengan kueri |
| Pengelompokan jejak | "Pengelompokan perilaku" | Cluster berjalan serupa untuk deteksi penyimpangan |
| Penegakan pagar pembatas | "Kebijakan pada waktu log" | Pemeriksaan PII/toksisitas/ruang lingkup pada konten yang dicatat |

## Bacaan Lanjutan

- [Dokumen Langfuse](https://langfuse.com/) — menelusuri, mengevaluasi, mengelola prompt
- [Arize Phoenix docs](https://docs.arize.com/phoenix) — instrumentasi otomatis, drift
- [Comet Opik](https://www.comet.com/site/products/opik/) — optimization + pagar pembatas
- [Konvensi semantik OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — skema yang digunakan ketiganya
