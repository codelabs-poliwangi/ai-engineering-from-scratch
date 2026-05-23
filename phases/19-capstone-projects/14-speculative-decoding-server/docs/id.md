# Capstone 14 — Server Inference Decoding Spekulatif

> EAGLE-3 di vLLM 0.7 mengirimkan throughput 2,5-3x pada lalu lintas nyata. P-EAGLE (AWS 2026) mendorong spekulasi paralel lebih jauh lagi. SpecForge SGLang melatih draft head dalam skala besar. Pusat Spekulan Red Hat menerbitkan draf yang selaras untuk model terbuka umum. TensorRT-LLM membuat decoding spekulatif kelas satu di NVIDIA. Tumpukan penyajian produksi tahun 2026 adalah vLLM atau SGLang dengan draf keluarga EAGLE, kuantisasi FP8 atau INT4, dan HPA saat antrean tunggu. Tujuan utama ini adalah untuk melayani dua model terbuka dengan throughput dasar 2,5x+ dengan laporan latensi ekor penuh.

**Type:** Batu penjuru
**Language:** Python (melayani), C++ / CUDA (inspeksi kernel), YAML (konfigurasi)
**Prerequisites:** Fase 3 (pembelajaran mendalam), Fase 7 (Transformer), Fase 10 (LLM dari awal), Fase 17 (infrastruktur)
**Fase yang dilakukan:** P3 · P7 · P10 · P17
**Waktu:** 30 jam

## Masalah

Penguraian code spekulatif menjadi komoditas pada tahun 2026. Kepala rancangan EAGLE-3 melatih status tersembunyi model target dan memprediksi N token di masa depan; model target memverifikasi dalam sekali jalan. Tingkat penerimaan 60-80% berarti 2-3x throughput end-to-end. vLLM 0.7 mengintegrasikan ini secara asli. SGLang + SpecForge memberi kamu alur training. Spekulan Red Hat menerbitkan draf yang selaras untuk Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B.

Kerajinannya ada dalam operasi penyajiannya, bukan modelnya. Tingkat penerimaan berubah seiring dengan distribusi lalu lintas (ShareGPT vs code vs data domain). Latensi ekor pada kondisi penolakan lebih buruk dibandingkan tanpa spekulasi — kamu harus melaporkan p99 pada berbagai ukuran batch, bukan hanya token kondisi stabil/dtk. Biaya per 1 juta token vs Anthropic / OpenAI API adalah faktor penentu kredibilitas.

## Konsep

Penguraian code spekulatif memiliki dua layer. Model **draf** (head EAGLE-3, ngram, atau model selaras target yang lebih kecil) mengusulkan k kandidat token per langkah. Model **target** memverifikasi semua k dalam satu lintasan; awalan apa pun yang diterima menggantikan jalur serakah. Tingkat penerimaan tergantung pada penyelarasan target rancangan dan distribusi input.

EAGLE-3 mengalahkan draf ngram di sebagian besar lalu lintas. P-EAGLE menjalankan spekulasi paralel untuk rancangan pohon yang lebih dalam. Keuntungannya: latensi P99 pada penolakan lebih tinggi karena izin verifikasi lebih besar. Konfigurasi penyajian harus melaporkan latensi dalam jumlah batch untuk menampilkan hal ini.

Penerapannya adalah Kubernetes. vLLM 0.7 menjalankan satu replika per GPU atau pecahan tensor-paralel. HPA melakukan penskalaan otomatis pada antrian tunggu, bukan pada CPU. Kuantitas FP8 (Marlin) dan INT4 (AWQ) menyimpan memori GPU di dalam envelope H100 / H200. Laporan ujung ke ujung adalah throughput, tingkat penerimaan, p50/p99 pada batch 1/8/32, dan token $/1 juta.

## Arsitektur

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## Tumpukan

- Penyajian: vLLM 0.7 atau SGLang 0.4
- Metode spekulatif: draft head EAGLE-3, spekulasi paralel P-EAGLE, ngram fallback
- Training draft: SpecForge (SGLang) atau Red Hat Speculator
- Model target: Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B
- Kuantisasi: FP8 (Marlin), INT4 AWQ
- Penerapan: Kubernetes + plugin perangkat NVIDIA; HPA pada metrik antrian-tunggu
- Eval: ShareGPT, MT-Bench-v2, GSM8K, HumanEval untuk pengukuran penerimaan penyebaran domain
- Referensi: Dekode spekulatif TensorRT-LLM untuk dasar vendor

## Build

1. **Persiapan model target.** Pilih Llama 3.3 70B. Kuantisasi ke FP8 melalui Marlin. Terapkan di bawah vLLM 0.7 pada 1xH100 (atau 2x tensor-paralel).

2. **Sumber draf.** Tarik draft head EAGLE-3 yang sejajar dari Red Hat Speculators (atau latih melalui SpecForge). Muat ke dalam konfigurasi decoding spekulatif vLLM.3. **Angka dasar.** Sebelum spekulasi: token pada batch 1/8/32, latensi p50/p99, penggunaan GPU. Menerbitkan.

4. **Aktifkan EAGLE-3.** Konfigurasi balik; jalankan kembali benchmark yang sama. Kecepatan laporan, tingkat penerimaan, delta latensi ekor p99.

5. **P-EAGLE.** Aktifkan spekulasi paralel; mengukur pohon rancangan yang lebih dalam vs serial EAGLE-3. Laporkan perubahan di mana P-EAGLE membantu vs merugikan.

6. **Lalu lintas domain.** Jalankan lalu lintas ShareGPT vs HumanEval vs khusus domain melalui server yang sama. Ukur tingkat penerimaan per distribusi. Identifikasi kapan draf melayang.

7. **Model target kedua.** Jalankan pipeline yang sama di Qwen3-Coder-30B MoE. Draf lebih rumit (kebisingan perutean MoE). Laporan.

8. **K8s HPA.** Terapkan pada K8 dengan pelacakan HPA `queue_wait_ms`. Peragakan penskalaan keluar saat weight menjadi tiga kali lipat.

9. **Perbandingan biaya.** Hitung token $/1 juta vs Anthropic Claude Sonnet 4.7 dan OpenAI GPT-5.4 pada evaluasi yang sama. Menerbitkan.

## Pakai

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## Kirim

`outputs/skill-inference-server.md` menjelaskan penyampaiannya. Tumpukan penyajian terukur dengan decoding spekulatif, laporan benchmark lengkap, dan penerapan K8.

| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | Kecepatan terukur vs garis dasar | Throughput 2,5x+ dengan kualitas yang sesuai pada dua model |
| 20 | Tingkat penerimaan pada lalu lintas yang realistis | Laporan tingkat penerimaan per distribusi |
| 20 | Disiplin latensi ekor P99 | p99 pada batch 1/8/32 dengan dan tanpa spekulasi |
| 20 | Operasi | Penerapan K8, HPA saat menunggu antrian, peluncuran lancar |
| 15 | Penulisan dan metodologi | Penjelasan yang jelas tentang apa yang berubah dan mengapa |
| **100** | | |

## Latihan

1. Mengukur penurunan tingkat penerimaan ketika rancangan berada satu versi di belakang target (misalnya, Llama 3.3 -> 3.4 drift). Buat peringatan pemantauan.

2. Terapkan ngram-fallback: jika penerimaan EAGLE-3 turun di bawah ambang batas, beralihlah ke ngram draft. Laporkan peningkatan keandalan.

3. Jalankan eksperimen MoE terkontrol: Qwen3-Coder-30B yang sama dengan routing noise yang disuntikkan vs tanpa. Ukur sensitivitas penerimaan draf.

4. Perpanjang hingga H200 (141 GB). Laporkan ruang kepala ukuran model per replika yang diperoleh dan apakah kamu dapat melayani Llama 3.3 70B yang tidak dikuantisasi.

5. Dekode spekulatif TensorRT-LLM pada perangkat keras H100 yang sama. Laporkan di mana ia menang vs vLLM.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Model rancangan | "Spekulator" | Model kecil yang mengusulkan N token untuk diverifikasi target |
| EAGLE-3 | "Draf arsitektur 2026" | Kepala wajib militer dilatih pada kondisi target yang tersembunyi; ~75% penerimaan |
| P-EAGLE | "Spekulasi paralel" | Pohon rancangan cabang diverifikasi dalam satu lintasan sasaran |
| Tingkat penerimaan | "Tingkat klik" | Sebagian kecil token yang dirancang diterima tanpa pengambilan sample ulang |
| Kuantisasi | "FP8/INT4" | Weight dengan presisi lebih rendah untuk memuat lebih banyak model dalam memori GPU |
| Antrian tunggu | "Metrik HPA" | Waktu permintaan menunggu dalam antrian tertunda sebelum inference dimulai |
| Pusat Spekulan | "Draf selaras" | Pusat Red Hat Neural Magic dari draf EAGLE untuk model terbuka umum |

## Bacaan Lanjutan- [dokumentasi vLLM EAGLE dan P-EAGLE](https://docs.vllm.ai) — tumpukan penyajian referensi
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — makalah decoding spekulatif paralel + integrasi
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — pipeline training draft-head
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — hub draf yang selaras
- [Decoding spekulatif TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) — alternatif vendor
- [Arsitektur penyajian Fireworks.ai](https://fireworks.ai/blog) — referensi komersial
- [Makalah EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — makalah metode
- [repositori vLLM](https://github.com/vllm-project/vllm) — code dan tolok ukur
