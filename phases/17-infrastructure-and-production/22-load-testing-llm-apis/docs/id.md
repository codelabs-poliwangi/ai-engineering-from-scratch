# Pengujian Weight API LLM — Mengapa k6 dan Locust Lie

> Penguji weight tradisional tidak dirancang untuk respons streaming, panjang output variabel, metrik tingkat token, atau saturasi GPU. Dua jebakan menggigit sebagian besar tim. Perangkap GIL: Pengukuran tingkat token Locust menjalankan tokenization dengan Python GIL, yang bersaing dengan pembuatan permintaan dalam konkurensi berat; simpanan tokenization kemudian meningkatkan latensi antar-token yang dilaporkan — klien kamu adalah penghambatnya, bukan servernya. Perangkap keseragaman prompt: prompt yang identik dalam satu putaran menguji satu titik pada distribusi token; lalu lintas nyata memiliki panjang yang bervariasi dan kecocokan awalan yang beragam. LLMPerf memperbaikinya dengan `--mean-input-tokens` + `--stddev-input-tokens`. Pemetaan alat pada tahun 2026: khusus LLM (GenAI-Perf, LLMPerf, LLM-Locust, guidellm) untuk akurasi tingkat token; **k6 v2026.1.0** + **k6 Operator 1.0 GA (Sep 2025)** — streaming-aware, asli Kubernetes yang didistribusikan melalui TestRun/PrivateLoadZone CRD, paling cocok untuk gerbang CI/CD; Vegeta untuk Go saturasi tingkat konstan; Locust 2.43.3 hanya dengan ekstensi LLM-Locust untuk streaming. Pola weight: kondisi tunak, ramp, spike (uji penskalaan otomatis), rendam (kebocoran memori).

**Type:** Build
**Language:** Python (stdlib, generator realistik mainan + pengumpul latensi)
**Prerequisites:** Fase 17 · 08 (Metrik Inference), Fase 17 · 03 (Penskalaan Otomatis GPU)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Jelaskan dua anti-pola (perangkap GIL, perangkap keseragaman prompt) yang membuat penguji weight umum berbohong untuk API LLM.
- Pilih alat untuk tujuan tertentu: LLMPerf (benchmark run), k6 + streaming extension (CI gate), guidellm (sintetis skala besar), GenAI-Perf (referensi NVIDIA).
- Rancang empat pola weight (stabil, ramp, spike, rendam) dan beri nama mode kegagalan yang ditangkap setiap tangkapan.
- Build distribusi cepat yang realistis menggunakan mean + stddev token input, bukan panjang tetap.

## Masalah

kamu menguji k6 titik akhir LLM kamu pada 500 pengguna secara bersamaan. Itu bertahan. kamu mengirim. Dalam produksi dengan 200 pengguna aktual, layanan ini terhenti — P99 TTFT meledak, GPU di-embed.

Ada dua hal yang terjadi. Pertama, k6 mengirimkan 500 prompt yang identik — penggabungan permintaan dan cache awalan kamu membuatnya tampak seperti kamu menangani 500 dekode bersamaan padahal sebenarnya kamu sedang menanganinya. Kedua, k6 tidak melacak latensi antar-token pada respons streaming seperti yang dialami mata; ia melihat satu koneksi HTTP, bukan 500 token yang tiba pada interval yang berbeda-beda.

Pengujian weight untuk LLM adalah disiplinnya sendiri.

## Konsep

### Perangkap GIL (Belalang)

Locust menggunakan Python dan menjalankan tokenization sisi klien di bawah GIL. Di bawah konkurensi tinggi, tokenizer mengantri di belakang pembuatan permintaan. Latensi antar-token yang dilaporkan mencakup simpanan tokenization sisi klien. kamu mengira servernya lambat; itu adalah alat uji.

Perbaiki: Ekstensi LLM-Locust memindahkan tokenization ke proses terpisah, atau menggunakan harness bahasa yang dikompilasi (k6, LLMPerf menggunakan tokenizers.rs).

### Perangkap keseragaman cepat

Semua penguji weight yang dikenal memungkinkan kamu mengonfigurasi satu prompt. Dalam pengujian loop 10.000 iterasi, prompt yang sama dikirimkan setiap kali. Server melihat awalan yang sama setiap kali — cache awalan mencapai 100%, throughput tampak bagus.

Perbaiki: sample dari distribusi cepat. LLMPerf menggunakan `--mean-input-tokens 500 --stddev-input-tokens 150` — durasi yang beragam, konten yang beragam.

### Empat pola pemuatan1. **Kondisi tunak** — RPS konstan selama 30-60 menit. Hasil tangkapan: regresi kinerja dasar.
2. **Ramp** — meningkatkan RPS secara linear dari 0 ke target selama 15 menit. Hasil tangkapan: titik henti kapasitas, anomali pemanasan.
3. **Spike** — RPS 3-10x secara tiba-tiba selama 2 menit, lalu kembali lagi. Hasil: latensi penskalaan otomatis, saturasi antrean, dampak cold-start.
4. **Rendam** — kondisi stabil selama 4-8 jam. Tangkapan: kebocoran memori, penyimpangan kumpulan koneksi, kelebihan observasi.

### 2026 pemetaan alat

**LLMPerf** (Skala apa pun) — Tokenization yang didukung Python tetapi Rust. Mean/stddev meminta. Sadar streaming. Default terbaik untuk kinerja berjalan.

**NVIDIA GenAI-Perf** — referensi NVIDIA. Menggunakan klien Triton; cakupan metrik yang komprehensif. Perhatikan ITL-nya tidak termasuk TTFT; LLMPerf menyertakannya. Dua alat menghasilkan TPOT berbeda untuk server yang sama.

**LLM-Locust** (TrueFoundry) — Ekstensi belalang yang memperbaiki jebakan GIL. Metrik streaming DSL + Locust yang familier.

**guidellm** — pembandingan sintetis skala besar.

**k6 v2026.1.0** + **k6 Operator 1.0 GA (September 2025)**:
- k6 sendiri (Go, dikompilasi, tanpa GIL) menambahkan metrik streaming-aware.
- Operator k6 menggunakan CRD TestRun / PrivateLoadZone untuk pengujian terdistribusi asli Kubernetes.
- Terbaik untuk gerbang CI/CD dan pengujian SLA.

**Vegeta** — Lebih sederhana dari k6. Saturasi HTTP tingkat konstan. Tidak mengetahui LLM tetapi bagus untuk pengujian gateway/batas kecepatan.

**Locust 2.43.3 stock** — memiliki jebakan GIL untuk LLM. Hanya dengan ekstensi LLM-Locust.

### Gerbang SLA di CI

Jalankan k6 di PR dengan:

- 30-50 iterasi masing-masing pada RPS dasar.
- Gerbang: P50/P95 TTFT, 5xx < 5%, TPOT di bawah ambang batas.
- Hancurkan pembangunan yang melanggar.

### Distribusi cepat yang realistis

Build dari sample lalu lintas nyata (jika kamu memilikinya) atau dari distribusi yang dipublikasikan (misalnya, ShareGPT meminta untuk mengobrol, HumanEval untuk code). Masukkan mean + stddev ke LLMPerf. Hindari loop-with-one-prompt dengan cara apa pun.

### Nomor yang harus kamu ingat

- Operator k6 1.0 GA: September 2025.
- k6 v2026.1.0: metrik yang mendukung streaming.
- Proses LLMPerf yang umum: 100-1000 permintaan pada konkurensi X.
- Gerbang CI tipikal: 30-50 iterasi per PR.
- Empat pola: stabil, ramp, spike, rendam.

## Pakai

`code/main.py` menyimulasikan pengujian weight dengan distribusi prompt yang realistis, mengukur TPOT yang efektif, dan mendemonstrasikan perangkap prompt seragam.

## Kirim

Lesson ini menghasilkan `outputs/skill-load-test-plan.md`. Mengingat weight kerja dan SLA, pilih alat dan desain empat pola weight.

## Latihan

1. Jalankan `code/main.py`. Bandingkan distribusi yang seragam dan realistis — di manakah kesenjangannya?
2. Tulis skrip k6 untuk gerbang CI: TTFT P95 < 800 ms pada 100 konkuren, waktu proses 5 menit.
3. Tes rendam kamu menunjukkan memori bertambah 50 MB/jam. Sebutkan tiga penyebab dan instrumen untuk memilih di antaranya.
4. Tes lonjakan dari 10 RPS menjadi 100 RPS. Berapa perkiraan waktu pemulihan jika tumpukan produksi Karpenter + vLLM sudah ada (Fase 17 · 03 + 18)?
5. GenAI-Perf melaporkan TPOT=6ms; LLMPerf melaporkan TPOT=11ms di server yang sama. Menjelaskan.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| LLMPerf | "harnes LLM" | Alat benchmark skala apa pun, streaming-aware |
| GenAI-Perf | "Alat NVIDIA" | memanfaatkan referensi NVIDIA |
| LLM-Belalang | "Belalang untuk LLM" | Ekstensi belalang memperbaiki perangkap GIL |
| panduan | "patokan sintetis" | Alat Sintetis Skala Besar |
| k6 Operator | "K8s k6" | K6 terdistribusi berbasis CRD |
| perangkap GIL | "Overhead klien Python" | Backlog tokenization meningkatkan latensi yang dilaporkan |
| Perangkap keseragaman cepat | "kebohongan satu kali" | Loop dengan prompt yang sama mengenai cache, meningkatkan throughput |
| Kondisi mapan | "weight konstan" | RPS datar selama N menit |
| Jalan | "berbaris" | 0 untuk menargetkan durasi |
| Lonjakan | "tes meledak" | Pengganda tiba-tiba lalu kembalikan |
| Rendam | "ujian panjang" | Jam untuk deteksi kebocoran |

## Bacaan Lanjutan

- [TianPan — Aplikasi LLM Pengujian Weight](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — LLM Pengujian Weight 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Pengantar Tolok Ukur Inference LLM](https://docs.nvidia.com/nim/large-lingual-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-kinerja)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [Operator k6](https://github.com/grafana/k6-operator)
