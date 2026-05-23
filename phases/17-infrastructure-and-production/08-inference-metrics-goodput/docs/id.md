# Metrik Inference — TTFT, TPOT, ITL, Goodput, P99

> Empat metrik menentukan apakah penerapan inference berhasil. TTFT adalah prefill plus antrian plus jaringan. TPOT (setara dengan ITL) adalah biaya dekode terikat memori per token. Latensi end-to-end adalah TTFT ditambah TPOT dikalikan panjang output. Throughput adalah token per detik yang dikumpulkan di seluruh armada. Namun hal yang penting bagi produk adalah hasil yang baik — sebagian kecil dari permintaan yang memenuhi setiap SLO secara bersamaan. Throughput tinggi dengan goodput rendah berarti kamu memproses token yang tidak pernah menjangkau pengguna tepat waktu. Nomor referensi untuk Llama-3.1-8B-Instruksikan TRT-LLM pada tahun 2026: rata-rata TTFT 162 ms, rata-rata TPOT 7,33 ms, rata-rata E2E 1,093 ms. Selalu laporkan P50, P90, P99 — jangan pernah bermaksud jahat. Dan perhatikan jebakan pengukurannya: GenAI-Perf mengecualikan TTFT dari perhitungan ITL, LLMPerf memasukkannya; dua alat tidak setuju pada TPOT untuk proses yang sama.

**Type:** Learn
**Language:** Python (stdlib, kalkulator persentil mainan, dan reporter goodput)
**Prerequisites:** Fase 17 · 04 (vLLM Melayani Internal)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Tentukan TTFT, TPOT, ITL, E2E, throughput, dan goodput dengan tepat dan beri nama komponen yang diukur masing-masing.
- Jelaskan mengapa mean adalah statistik yang salah untuk penayangan LLM dan cara membaca P50/P90/P99.
- Buatlah multi-kendala SLO (misalnya TTFT<500 ms DAN TPOT<15 ms DAN E2E<2 s) dan hitung goodput terhadapnya.
- Sebutkan dua alat benchmark yang tidak sesuai pada TPOT untuk proses yang sama dan jelaskan alasannya.

## Masalah

"Throughput kami adalah 15.000 token per detik." Jadi apa? Jika 40% permintaan gagal melewati 2 detik secara menyeluruh, pengguna akan mengabaikan sesi tersebut. Throughput saja tidak memberi tahu kamu apakah produk tersebut berfungsi.

Inference memiliki beberapa sumbu latensi dan masing-masing sumbu mengalami kegagalan yang berbeda. Pra-pengisian terikat pada komputasi dan berskala dengan panjang yang cepat. Decode terikat pada memori dan diskalakan dengan ukuran batch. Keterlambatan antrian merupakan masalah operasional. Jaringan adalah masalah distance fisik. kamu memerlukan metrik yang berbeda untuk masing-masing metrik, dan kamu memerlukan persentil, dan kamu memerlukan satu gabungan yang menyatakan "apakah pengguna mendapatkan apa yang mereka harapkan" — itu adalah hasil yang bagus.

## Konsep

### TTFT — waktu untuk token pertama

`TTFT = queue_time + network_request + prefill_time`

Prefill mendominasi ketika perintahnya panjang. Pada Llama-3.3-70B FP8 di H100, prompt 32k membutuhkan ~800 ms prefill murni. Waktu antrian adalah perilaku penjadwal yang sedang dimuat. Permintaan jaringan adalah wire time termasuk TLS. TTFT adalah latensi yang dilihat pengguna sebelum sesuatu dialirkan kembali.

### TPOT / ITL — latensi antar token

Banyak nama untuk satu kuantitas. `TPOT` (waktu per token output), `ITL` (latensi antar-token), `decode latency per token` — semuanya sama. Ini adalah waktu antara token yang dialirkan berturut-turut setelah yang pertama.

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

Pada tumpukan Llama-3.3-70B H100 yang sama dengan prefill yang dipotong, rata-rata TPOT ~7 ms. Tanpa pra-pengisian yang terpotong, selama pra-pengisian yang lama pada urutan yang berdekatan, TPOT dapat melonjak hingga 50 ms. Tonton P99, tidak jahat.

### Latensi E2E

`E2E = TTFT + TPOT * output_tokens + network_response`

Untuk output panjang (>500 token), E2E didominasi TPOT. Untuk output pendek dengan petunjuk panjang, E2E didominasi TTFT. Laporkan E2E yang dikondisikan dengan panjang output.

### Hasil

`throughput = total_output_tokens / elapsed_time`

Metrik agregat. Memberi tahu kamu efisiensi armada. Tidak memberi tahu kamu kesehatan permintaan individu.

### Goodput — metrik yang benar-benar kamu pedulikan

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`SLO adalah multi-batasan. Sebuah permintaan dianggap "baik" hanya jika setiap batasan dapat dipenuhi. Goodput adalah bagiannya. Throughput tinggi pada goodput 60% adalah kegagalan. Throughput yang lebih rendah pada goodput 99% adalah targetnya.

Pada tahun 2026, goodput adalah metrik yang digunakan dalam pengiriman MLPerf Inference v6.0 dan pelacakan SLA internal di penyedia platform AI.

### Mengapa mean adalah statistik yang salah

Distribusi latensi LLM condong ke kanan. Kumpulan dekode dengan satu tetangga yang telah diisi sebelumnya dapat mengirimkan 500 token dengan TPOT ~7 mdtk dan 20 token dengan TPOT ~60 mdtk. Berarti TPOT adalah 9 ms. P99 TPOT adalah 65 mdtk. Pengguna menekan P99 secara teratur — itulah sebabnya mereka keluar.

Selalu laporkan triple (P50, P90, P99). Untuk pengalaman pengguna, P99 adalah yang kamu optimalkan.

### Nomor referensi — Llama-3.1-8B-Instruksikan tentang TRT-LLM, 2026

- rata-rata TTFT: 162 ms
- rata-rata TPOT: 7,33 ms
- rata-rata E2E: 1,093 ms
- TPOT P99: bervariasi 10-25 ms tergantung pada konfigurasi pra-pengisian yang dipotong.

Ini adalah titik referensi NVIDIA yang dipublikasikan. Mereka berubah berdasarkan ukuran model (70B akan menampilkan 3-5x), perangkat keras (H100 vs B200 ~3x), dan weight.

### Perangkap pengukuran

Dua alat benchmark yang paling banyak digunakan pada tahun 2026 tidak setuju dengan TPOT untuk proses yang sama:

- **NVIDIA GenAI-Perf**: tidak termasuk TTFT dari perhitungan ITL. ITL dimulai dari token 2.
- **LLMPerf**: termasuk TTFT. ITL dimulai dari token 1.

Untuk permintaan dengan TTFT 500 ms dan 100 token output dalam total dekode 700 ms, GenAI-Perf melaporkan `ITL = 700/99 = 7.07 ms`, LLMPerf melaporkan `ITL = 1200/100 = 12.00 ms`. Pilihan alat mengubah nomornya.

Selalu nyatakan alat yang mana. Selalu publikasikan definisinya.

### Membuat SLO

SLO wajar yang dihadapi konsumen untuk model obrolan 70 miliar pada tahun 2026:

- TTFT P99 <= 800 mdtk.
- TPOT P99 <= 25 mdtk.
- E2E P99 <= 3 detik untuk output <300 token.
- Target produksi >= 99%.

SLO perusahaan memperketat TTFT (200-400 ms) dan melonggarkan E2E. Intinya adalah menuliskannya, mengukur ketiganya, dan melacak hasil yang baik sebagai satu kesatuan.

### Cara mengukur

- Jalankan lalu lintas nyata atau sintetis realistis (LLMPerf dengan `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`).
- Targetkan konkurensi puncak 2x untuk menjalankan benchmark.
- Jalankan 30-50 iterasi, ambil persentil dari sample gabungan.
- Publikasikan dengan nama alat, versi alat, model, perangkat keras, konkurensi, distribusi cepat.

## Pakai

`code/main.py` adalah kalkulator output mainan. Hasilkan distribusi latensi sintetis, terapkan SLO, dan hitung goodput. Juga menunjukkan perbedaan TPOT GenAI-Perf vs LLMPerf pada jejak yang sama.

## Kirim

Lesson ini menghasilkan `outputs/skill-slo-goodput-gate.md`. Mengingat weight kerja dan SLO, ini menghasilkan resep benchmark siap CI/CD yang diterapkan gerbang pada goodput, bukan throughput.

## Latihan

1. Jalankan `code/main.py`. Hasilkan distribusi dengan lonjakan ekor 1%. Bagaimana perubahan goodput saat kamu mengencangkan TPOT P99 dari 30 ms menjadi 15 ms?
2. Vendor mengutip "15.000 tok/s pada Llama 3.3 70B H100". Sebutkan tiga pertanyaan untuk ditanyakan sebelum memercayainya.
3. Mengapa potongan awal yang terpotong melindungi TPOT P99 tetapi tidak berarti TPOT?
4. Buat SLO konsumen untuk asisten suara (token pertama didengar, bukan dibaca). Metrik manakah yang paling terlihat oleh pengguna?
5. Baca README LLMPerf dan dokumen GenAI-Perf. Identifikasi tiga metrik lain yang alatnya tidak sesuai.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| TTFT | "waktunya untuk token pertama" | Antrian + jaringan + isi awal; didominasi oleh prefill pada prompt panjang |
| TPOT | "waktu per token output" | Biaya dekode terikat memori per token setelah |
| ITL | "latensi antar-token" | Sama seperti TPOT di sebagian besar alat (tidak semua — lihat GenAI-Perf) |
| E2E | "ujung ke ujung" | TTFT+TPOT*output_len; jaringan sisi respons di atas |
| Output | "tok/s" | Efisiensi armada; tidak berguna tanpa persentil latensi |
| output bagus | "Tarif terpenuhi SLO" | Sebagian kecil permintaan memenuhi setiap batasan SLO secara bersamaan |
| Hlm99 | "ekor" | 1 dari 100 latensi terburuk; metrik pengalaman pengguna |
| SLO multi-batasan | "gabungan" | DAN dari ketiga batas latensi; permintaan gagal jika ada yang dilanggar |
| GenAI-Perf vs LLMPerf | "perangkap alat" | Alat tidak setuju apakah ITL menyertakan TTFT |

## Bacaan Lanjutan

- [NVIDIA NIM — Metrik Tolok Ukur LLM](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — definisi kanonik TTFT, ITL, TPOT.
- [Anyscale — Metrik Tolok Ukur Penyajian LLM](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — definisi alternatif dan resep pengukuran.
- [BentoML — Metrik Inference LLM](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — menerapkan pengukuran pada penerapan nyata.
- [LLMPerf](https://github.com/ray-project/llmperf) — Tolok ukur sumber terbuka berbasis Ray.
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — alat benchmark NVIDIA.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — tolok ukur berbasis goodput yang diterima industri.
