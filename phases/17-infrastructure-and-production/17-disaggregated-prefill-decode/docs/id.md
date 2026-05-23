# Prefill/Decode Terpilah — NVIDIA Dynamo dan llm-d

> Pra-pengisian terikat pada komputasi; dekode terikat memori. Menjalankan keduanya pada GPU yang sama akan membuang satu sumber daya. Disagregasi membaginya ke dalam kumpulan terpisah dan mentransfer cache KV di antara keduanya melalui NIXL (RDMA/InfiniBand atau TCP fallback). NVIDIA Dynamo (diumumkan di GTC 2025, 1.0 GA) berada di atas vLLM/SGLang/TRT-LLM — rasio prefill:decode Planner Profiler + SLA Planner auto-rate-match untuk memenuhi SLO. NVIDIA menerbitkan peningkatan throughput dalam rata-rata ini — developer.nvidia.com (2025-06) menunjukkan peningkatan ~6x untuk DeepSeek-R1 MoE pada GB200 NVL72 + Dynamo dalam rezim latensi menengah, dan halaman produk Dynamo (developer.nvidia.com, tanpa tanggal) mengiklankan throughput hingga 50x MoE pada GB300 NVL72 + Dynamo vs Hopper. Angka "30x" adalah agregat komunitas di seluruh laporan full-stack Blackwell + Dynamo + DeepSeek-R1; kami belum menemukan satu pun sumber utama yang menyatakan secara tepat 30x, jadi perlakukan ini sebagai klaim terarah. llm-d (Red Hat + AWS) adalah asli Kubernetes: prefill / decode / router sebagai Layanan independen dengan HPA per peran. llm-d 0.5 menambahkan pembongkaran KV hierarkis, perutean LoRA yang sadar cache, jaringan UCCL, skala-ke-nol. Ekonomi: penggabungan internal beberapa pengungkapan pelanggan menunjukkan penghematan 30–40% pada pembelanjaan inference kelas sebesar $2 juta (yaitu $600-800 ribu/tahun) ketika beralih dari penayangan bersama ke penayangan terpilah dengan Dynamo pada SLA konstan; angka spesifik $2M→$600-800K adalah gabungan internal, bukan studi kasus tunggal yang dipublikasikan — gunakan sebagai jangkar urutan besarnya, bukan kutipan referensi. Prompt singkat (<512 token, output pendek) tidak membenarkan biaya transfer.

**Type:** Learn
**Language:** Python (stdlib, simulator mainan terpilah vs colokasi)
**Prerequisites:** Fase 17 · 04 (vLLM Melayani Internal), Fase 17 · 08 (Metrik Inference)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Jelaskan mengapa pra-pengisian dan dekode memiliki alokasi GPU optimal yang berbeda dan menghitung pemborosan dalam kolokasi.
- Diagram arsitektur terpilah: kumpulan prefill, kumpulan dekode, transfer KV melalui NIXL, router.
- Sebutkan kondisi ketika disagregasi TIDAK membuahkan hasil (petunjuk singkat, output pendek).
- Bedakan NVIDIA Dynamo (stack-above) dari llm-d (Kubernetes-native) dan cocokkan masing-masing dengan konteks operasional.

## Masalah

kamu menjalankan Llama 3.3 70B pada 8 H100. Di bawah weight kerja campuran (prompt panjang + output pendek), GPU menganggur selama dekode karena sebagian besar komputasi dihabiskan untuk pengisian awal. Di bawah weight kerja yang berbeda (prompt pendek + output panjang), yang terjadi justru sebaliknya. Pra-pengisian + dekode yang ditempatkan di tempat yang sama berarti kamu menyediakan keduanya secara berlebihan.

Dampak anggaran: 20-40% waktu GPU terbuang untuk sumber daya yang salah. kamu membeli komputasi H100 untuk menjalankan dekode terikat memori, atau membeli bandwidth HBM H100 untuk menjalankan pra-pengisian terikat komputasi. Keduanya merupakan limbah yang mahal.

Disagregasi membagi pengisian awal dan dekode ke dalam kumpulan terpisah yang disesuaikan dengan ukuran kemacetan masing-masing. Transfer cache KV dari kumpulan pra-pengisian ke kumpulan dekode melalui interkoneksi bandwidth tinggi.

## Konsep

### Mengapa hambatannya berbeda

**Prefill** — menjalankan trafo melalui prompt input penuh sekaligus. Perkalian matrix mendominasi; terikat komputasi. H100 FP8 memberikan ~2000 TFLOPS throughput yang berguna. Efisiensi batch bagus — satu proses maju memproses banyak token.

**Decode** — menghasilkan satu token dalam satu waktu, membaca weight penuh setiap iterasi. Terikat bandwidth memori. HBM3 memberikan ~3 TB/dtk. Efisiensi batch hanya bagus pada konkurensi tinggi — weight yang dibaca diamortisasi di seluruh batch.Menempatkan keduanya: kamu membeli GPU yang dioptimalkan untuk keduanya. H100 bagus dalam keduanya tetapi harganya sama. Dalam skala besar, kamu ingin kumpulan pra-pengisian pada H100/berat komputasi; kumpulan dekode pada H200 / banyak memori, atau dengan kuantisasi agresif.

### Arsitektur

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL adalah transportasi antar-node NVIDIA. Menggunakan RDMA/InfiniBand bila tersedia, sebaliknya TCP mundur. Latensi transfer nyata — biasanya 20-80 ms untuk cache KV dari prompt token 4K pada 70B FP8. Inilah sebabnya mengapa petunjuk singkat tidak membenarkan pemilahan: pajak transfer melebihi tabungan.

### Dinamo vs llm-d

**NVIDIA Dynamo** (pengumuman GTC 2025, 1.0 GA):
- Duduk di atas vLLM, SGLang, TRT-LLM sebagai orkestrator.
- Planner Profiler mengukur weight kerja, SLA Planner secara otomatis mengonfigurasi rasio pra-pengisian: dekode.
- Inti karat, ekstensibilitas Python.
- Peningkatan throughput: NVIDIA melaporkan 6x untuk DeepSeek-R1 MoE pada GB200 NVL72 + Dynamo dalam rezim latensi menengah (developer.nvidia.com, 2025-06); laporan komunitas "hingga 30x" pada tumpukan penuh Blackwell + Dynamo + DeepSeek-R1 tidak memiliki satu sumber utama dan harus diperlakukan sebagai arahan.
- GB300 NVL72 + Dynamo: throughput MoE hingga 50x vs Hopper per halaman produk Dynamo (developer.nvidia.com, tidak bertanggal).

**llm-d** (Red Hat + AWS, asli Kubernetes):
- Prefill / decode / router sebagai Layanan Kubernetes independen.
- HPA per peran dengan sinyal kedalaman antrian (prefill) / pemanfaatan KV (decode).
- `topologyConstraint packDomain: rack` mengemas klik pra-isi+dekode pada rak yang sama untuk transfer KV bandwidth tinggi.
- llm-d 0.5 (2026): pembongkaran KV hierarkis, perutean LoRA yang sadar cache, jaringan UCCL, skala ke nol.

Gunakan Dynamo jika kamu menginginkan orkestrator tumpukan di atas yang terkelola. Gunakan llm-d jika kamu menginginkan primitif asli Kubernetes dan berkomitmen pada ekosistem CNCF.

### Ekonomi

Gabungan internal (bukan satu studi kasus yang dipublikasikan — jangkar urutan besarnya):

- Pembelanjaan inference sebesar $2 juta/tahun untuk penayangan di lokasi yang sama.
- Beralih ke terpilah dengan Dynamo.
- Volume permintaan yang sama, SLA latensi P99 yang sama.
- Penghematan yang dilaporkan: $600K–$800K/tahun (pengurangan 30–40%).
- Tidak ada perangkat keras baru.

Kami mensintesis angka ini dari berbagai pengungkapan pelanggan, bukan dari satu studi kasus yang dapat dikutip; titik data terdekat yang dipublikasikan adalah TTFT Baseten 2x lebih cepat / throughput 61% lebih tinggi dengan perutean Dynamo KV (baseten.co, 2025-10), dan proyeksi VAST + CoreWeave tentang 60–130% lebih banyak token/$ pada tingkat hit 40–60% KV (vastdata.com, 2025-12). Penghematan berasal dari penyesuaian ukuran setiap kelompok; weight kerja pra-pengisian (RAG dengan awalan 8K+) memberikan manfaat lebih dibandingkan weight kerja seimbang.

### Kapan TIDAK memilah

- Anjuran <512 token dan output <200 token: pajak transfer mendominasi perolehan.
- Cluster kecil (<4 GPU): keragaman kumpulan tidak cukup.
- Tim tidak dapat mengoperasikan dua kumpulan GPU dengan penskalaan per peran: Dynamo membantu tetapi tidak sepele.
- Tanpa kain RDMA: pajak transfer TCP lebih berat.

### Router terintegrasi dengan Fase 17 · 11

Router terpilah sadar akan KV-cache (Fase 17 · 11). Permintaan mendarat di kumpulan dekode yang menyimpan awalannya — jika tidak ada yang cocok, permintaan akan mengalir melalui pra-pengisian → dekode. Tingkat hit dan gabungan disagregasi — router yang sadar cache menentukan apakah prefill baru diperlukan.

### MoE di Blackwell adalah tempat angka sebenarnya berada

GB300 NVL72 + Dynamo menunjukkan throughput 50x MoE di atas garis dasar Hopper. Perutean pakar MoE membutuhkan banyak komputasi pada pra-pengisian, namun banyak memori pada dekode (cache pakar), sehingga pemilahan merupakan keuntungan ganda. Penyajian model perbatasan tahun 2026 dominan di Kementerian Pendidikan (DeepSeek-V3, varian GPT-5 masa depan).### Nomor yang harus kamu ingat

Angka tolok ukur melayang — NVIDIA dan tumpukan inference memposting hasil yang diperbarui setiap kuartal. Periksa kembali sebelum mengutip.

- DeepSeek-R1 pada GB200 NVL72 + Dynamo: ~6x throughput vs baseline dalam rezim latensi menengah (developer.nvidia.com, 2025-06); klaim komunitas "hingga 30x" pada tumpukan penuh Blackwell + Dynamo adalah agregat terarah tanpa satu sumber utama.
- GB300 NVL72 + Dynamo: throughput MoE hingga 50x vs Hopper (developer.nvidia.com, tidak bertanggal).
- Jangkar tabungan (gabungan internal, bukan studi kasus tunggal): diskon $600-800K/tahun untuk pembelanjaan tahunan $2 juta dengan SLA konstan.
- Ambang batas disagregasi: meminta >512 token + output >200 token.
- Transfer KV melalui NIXL: 20-80 ms untuk KV 4K-prompt pada 70B FP8.

## Pakai

`code/main.py` menyimulasikan penayangan bersama vs terpilah. Melaporkan throughput, biaya per permintaan, dan persilangan panjang prompt.

## Kirim

Lesson ini menghasilkan `outputs/skill-disaggregation-decider.md`. Mengingat weight kerja dan cluster, putuskan apakah akan memilah.

## Latihan

1. Jalankan `code/main.py`. Sejauh mana disagregasi mengalahkan kolokasi?
2. Rancang kumpulan pra-pengisian dan kumpulan dekode untuk layanan RAG dengan panjang awalan P99 8K, output 300.
3. Dynamo vs llm-d: pilih satu untuk toko Kubernetes murni tanpa preferensi runtime Python.
4. Hitung biaya transfer KV: isi awal 4K pada 70B FP8 = ~500 MB KV. Pada RDMA 100 GB/dtk, transfer = 5 mdtk. Pada TCP 10 GB/dtk = 50 mdtk. Mana yang penting bagi SLA kamu?
5. Perutean ahli KLH mengubah pola akses KV. Bagaimana perilaku disagregasi dengan MoE yang mengaktifkan pakar berbeda per token?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Penyajian terpilah | "pisahkan pra-isi/dekode" | Pisahkan kumpulan GPU untuk setiap fase |
| NIXL | "Transportasi NVIDIA" | Transfer KV antar-simpul Dynamo (RDMA/TCP) |
| NVIDIA Dinamo | "sang orkestra" | Koordinator tumpukan di atas untuk vLLM/SGLang/TRT-LLM |
| llm-d | "Kubernet asli" | Tumpukan terpilah Red Hat + AWS K8 |
| Profil Perencana | "Konfigurasi otomatis Dynamo" | Mengukur weight kerja, mengonfigurasi rasio kumpulan |
| Perencana SLA | "Kebijakan Dinamo" | Isian awal pencocokan nilai otomatis: dekode untuk memenuhi SLO |
| `packDomain: rack` | "topologi llm-d" | Kemas prefill+decode di rak yang sama untuk KV cepat |
| UCCL | "kolektif bersatu" | llm-d 0,5 layer jaringan untuk skala-ke-nol |
| Perutean ahli MoE | "ahli per token" | Pola DeepSeek-V3; disagregasi membantu |

## Bacaan Lanjutan

- [NVIDIA — Memperkenalkan Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Inference LLM Terpilah di Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [Blog Penyajian Terpilah TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [catatan rilis llm-d 0,5](https://github.com/llm-d/llm-d/releases)
