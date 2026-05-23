# Pelayanan LLM Multi-Wilayah dan Lokalitas Cache KV

> Penyeimbangan weight round-robin secara aktif berbahaya untuk inference LLM yang di-cache. Permintaan yang tidak mendarat di node yang menyimpan awalannya akan membayar biaya pengisian awal secara penuh — kira-kira 800 ms pada P50 pada prompt panjang dibandingkan ~80 ms dengan cache hit. Pada tahun 2026, pola produksinya adalah router yang sadar cache (Router vLLM di Rust, router llm-d) yang menggunakan peristiwa dan rute cache KV pada kecocokan awalan-hash. Penelitian terbaru (GORGO) menjadikan latensi jaringan lintas wilayah sebagai istilah eksplisit dalam tujuan perutean. Penawaran komersial "inference lintas wilayah" (inference lintas wilayah dasar, gateway multi-kluster GKE) memperlakukan inference sebagai sesuatu yang tidak jelas — penawaran tersebut menangani ketersediaan, bukan TTFT. JPMorgan dan Mayo Clinic menjalankan failover us-east-1 pada November 2024 dalam waktu ~22 menit. Kenyataan DR: 32% kegagalan LLM DR disebabkan karena tim mencadangkan weight tetapi lupa file tokenizer atau konfigurasi kuantisasi.

**Type:** Learn
**Language:** Python (stdlib, simulator router sadar-cache-awalan mainan)
**Prerequisites:** Fase 17 · 04 (Pelayanan vLLM), Fase 17 · 06 (SGLang RadixAttention)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Jelaskan mengapa penyeimbangan weight round-robin merusak inference cache dan menghitung penalti TTFT.
- Diagram router yang sadar cache: input (peristiwa KV-cache), algoritma (pencocokan awalan-hash), tie-breaker (pemanfaatan GPU).
- Beri nama driver kegagalan DR 32% untuk LLM (file tokenizer / konfigurasi kuantisasi tidak ada) dan sebutkan daftar periksa DR tiga file.
- Membedakan penawaran komersial lintas wilayah (Bedrock CRI, GKE Multi-Cluster Gateway) dari perutean KV-aware.

## Masalah

Layanan kamu berjalan di us-east-1, us-west-2, dan eu-west-1. kamu menempatkan ALB di depan dengan round-robin. Tingkat pencapaian cache awalan dalam produksi turun menjadi 8%. TTFT P50 tiga kali lipat. Log vLLM kamu menunjukkan bahwa setiap permintaan membayar biaya pengisian awal secara penuh.

Round-robin optimal untuk layanan tanpa kewarganegaraan. Inference LLM dirancang secara stateful — cache KV mengkodekan semua yang telah dilihat model. Routing blind adalah merutekan ke cache yang salah.

Secara terpisah, tim kamu memiliki rencana DR. kamu mencadangkan weight model ke S3 lintas wilayah. Pemadaman regional terjadi; kamu mencoba melakukan failover; replika menolak untuk memulai. kamu lupa tokenizer.json, konfigurasi kuantisasi, dan konfigurasi penskalaan RoPE berada di keranjang terpisah yang tidak kamu sinkronkan.

Pelayanan LLM multi-wilayah merupakan masalah cache, masalah perutean, dan masalah kebersihan DR — bukan masalah penyeimbang weight.

## Konsep

### Perutean sadar cache

Permintaan tiba dengan cepat. Router melakukan hash pada awalan (misalnya, 512 token pertama); ia menanyakan setiap replika "apakah kamu memiliki awalan ini dalam cache?". Replika menerbitkan peristiwa cache KV di pipeline pub/sub saat mereka mengalokasikan dan mengeluarkan blok. Router mengambil replika yang cocok, dan lolos ke tie-breaker berbasis utilitas GPU jika tidak ada yang melakukannya.

**vLLM Router** (Rust, tumpukan produksi 2026): berlangganan acara `kv.cache.block_added`, mempertahankan hash awalan → indeks replika, merutekan dengan pencarian O(1). Gagal mencapai kedalaman antrian paling sedikit ketika tidak ada kecocokan.

**llm-d router**: pola yang sama, asli Kubernetes. Memublikasikan acara melalui ControlPlane API.

**SGLang RadixAttention** (Fase 17 · 06) setara dengan intra-replika. Perutean lintas replika sepenuhnya dilakukan di bagian hulu.

### Angka

TTFT P50 pada prompt 2K-token, Llama 3.3 70B FP8, H100:
- Cache hit (replika yang sama, awalan tetap): ~80 ms.
- Cache hilang (pengisian awal dingin): ~800 ms.kesenjangan 10x. Jika router kamu mencapai 60-80% cache awalan di seluruh replika, kamu memperkirakan performa replika tunggal pada kapasitas N-replika. Jika mencapai 10%, kamu memperkirakan penskalaan yang naif.

### Lintas wilayah memiliki kendala baru — latensi jaringan

RTT antar wilayah:
- us-east-1 ↔ us-west-2: ~65 ms.
- us-east-1 ↔ eu-west-1: ~75 ms.
- us-east-1 ↔ ap-tenggara-1: ~220 ms.

Jika perutean mengambil permintaan dari us-east-1 ke hot prefix di ap-southeast-1, pra-pengisian yang disimpan (800 → 80 ms) akan berkurang sebesar 440 ms pulang pergi. GORGO (penelitian tahun 2026) menyatakan hal ini secara eksplisit — minimalkan `prefill_time + network_latency` secara bersamaan, bukan hanya melakukan pra-pengisian saja. Seringkali jawabannya adalah mempertahankan perutean regional kecuali pada awalan multi-MB yang besar dimana pra-pengisian mendominasi.

### "Inference lintas wilayah" komersial tidak membantu di sini

Inference lintas wilayah AWS Bedrock secara otomatis merutekan permintaan ke wilayah lain selama tekanan kapasitas. Ini mengoptimalkan ketersediaan, bukan TTFT, dan memperlakukan inference sebagai sesuatu yang buram. GKE Multi-Cluster Gateway juga sama — failover tingkat layanan, tidak ada kesadaran akan cache KV.

kamu masih memerlukan router yang sadar cache layer aplikasi bahkan saat menggunakannya. Mereka menangani kasus "us-east-1 is on fire". Perutean sadar cache menangani kasus TTFT.

### Kebersihan DR — 32% masalah file hilang

Statistik tahun 2026 yang banyak dikutip: 32% kegagalan LLM DR terjadi karena tim membuat cadangan weight tetapi lupa:

- `tokenizer.json` atau `tokenizer.model`
- Konfigurasi kuantisasi (`quantize_config.json`, skala AWQ, titik nol GPTQ)
- Konfigurasi khusus model (penskalaan Tali, attention mask, templat obrolan)
- Konfigurasi mesin (`vllm_config.yaml`, pengambilan sample default, manifes adaptor LoRA)

Cara mengatasinya adalah manifes DR minimum tiga file:

1. Semua file di bawah repo model HF (weight + konfigurasi + tokenizer).
2. Konfigurasi penyajian khusus mesin.
3. Manifes penerapan (K8s YAML, Dockerfile, kunci ketergantungan).

Plus: jalankan latihan DR setiap tiga bulan. Latihan JPMorgan us-east-1 mencapai pemulihan 22 menit pada November 2024 hanya karena pedomannya telah dilatih.

### Residensi data bersifat ortogonal

Pelanggan UE, PHI, tidak dapat meninggalkan UE. Jika router cache-aware kamu mengirimkan permintaan yang berasal dari Paris ke us-east-1 untuk pencocokan awalan, kamu telah melanggar GDPR terlepas dari perolehan TTFT. Partisi router berdasarkan batas tempat tinggal sebelum mengoptimalkan cache.

### Nomor yang harus kamu ingat

- Cache hit vs miss TTFT gap: ~10x (80 ms vs 800 ms pada prompt 2K).
- RTT antar wilayah AS-UE: ~75 ms.
- Kegagalan DR: 32% tokenizer/konfigurasi kuantitas hilang.
- Kegagalan JPMorgan us-east-1 Nov 2024: 22 menit (SLA 30 menit).

## Pakai

`code/main.py` menyimulasikan tiga strategi perutean (round-robin, regional yang sadar cache, global yang sadar cache) pada weight kerja multi-wilayah. Melaporkan tingkat cache hit, TTFT P50/P99, dan tagihan lintas wilayah.

## Kirim

Lesson ini menghasilkan `outputs/skill-multi-region-router.md`. Mengingat wilayah, batasan tempat tinggal, dan SLA, rancang rencana perutean.

## Latihan

1. Jalankan `code/main.py`. Berapa lama waktu yang dibutuhkan perutean lintas wilayah mengalahkan perutean lokal saja, dengan RTT 75 mdtk?
2. Tingkat cache hit kamu turun dari 70% menjadi 12%. Diagnosis tiga kemungkinan penyebab dan pengamatan yang dapat mengkonfirmasi masing-masing penyebab.
3. Rancang manifes DR untuk model terkuantisasi AWQ 70B yang disajikan di vLLM dengan 5 adaptor LoRA. Daftar setiap file dan konfigurasi.
4. Memperdebatkan apakah inference lintas wilayah Batuan Dasar “cukup” untuk fintech dengan SLO TTFT yang ketat. Kutip perilaku tertentu.
5. Permintaan asal Paris cocok dengan awalan di us-east-1. Apakah kamu mengarahkannya? Tulis kebijakannya.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Perutean sadar cache | "LB pintar" | Rutekan pada kecocokan awalan-hash ke replika penyimpanan cache KV |
| Acara cache KV | "cache pub-sub" | Replika menerbitkan blok tambah/pengusiran; indeks router |
| Hash awalan | "kunci cache" | Hash dari N token pertama yang digunakan sebagai pencarian router |
| gorgo | "penelitian perutean lintas wilayah" | arXiv 2602.11688; latensi jaringan sebagai istilah eksplisit |
| Inference lintas wilayah | "CRI Batuan Dasar" | produk AWS; failover ketersediaan, bukan kesadaran TTFT |
| Manifes DR | "daftar cadangan" | Setiap file perlu dipulihkan — bukan hanya weight |
| Residensi data | "Batas GDPR" | Kendala hukum di wilayah mana yang melihat data pengguna |
| RTT | "waktu pulang pergi" | Latensi jaringan; 75 mdtk AS-UE, 220 mdtk AS-APAC |
| LB sadar LLM | "LB yang terkena cache" | Router yang sadar cache sebagai kategori produk |

## Bacaan Lanjutan

- [BentoML — Inference multi-cloud dan lintas wilayah](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — penggunaan kembali cache KV lintas wilayah dengan istilah latensi jaringan.
- [TianPan — LLM Melayani Lokalitas Cache Multi-Wilayah](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [Inference Lintas Wilayah Batuan Dasar AWS](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — dokumentasi failover ketersediaan.
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — sumber router yang sadar cache.
