# Penskalaan Otomatis GPU di Kubernetes — Karpenter, Penjadwal KAI, Penjadwalan Gang

> Tiga layer, bukan satu. Karpenter menyediakan node secara dinamis (kurang dari satu menit, 40% lebih cepat dibandingkan Cluster Autoscaler). KAI Scheduler menangani penjadwalan geng, kesadaran topologi, dan antrean hierarki — ini mencegah jebakan alokasi parsial 7-dari-8 di mana tujuh node menunggu dan membakar satu GPU yang hilang. Penskala otomatis tingkat aplikasi (NVIDIA Dynamo Planner, llm-d Workload Variant Autoscaler) menskalakan sinyal spesifik inference — kedalaman antrean, pemanfaatan cache KV — bukan siklus kerja CPU/DCGM. Perangkap HPA klasik adalah `DCGM_FI_DEV_GPU_UTIL` adalah pengukuran siklus kerja: 100% bisa berupa 10 permintaan atau 100. vLLM melakukan pra-alokasi memori cache KV, sehingga memori tidak pernah memicu penurunan skala. Lesson ini mengajarkan kamu untuk menyusun tiga layer dan menghindari kebijakan default Karpenter `WhenEmptyOrUnderutilized` yang menghentikan menjalankan pekerjaan GPU di tengah inference.

**Type:** Learn
**Language:** Python (stdlib, simulator autoscaler kedalaman antrean mainan)
**Prerequisites:** Fase 17 · 02 (Ekonomi Platform Inference), Fase 17 · 04 (vLLM Melayani Internal)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Buat diagram tiga layer penskalaan otomatis (penyediaan node, penjadwalan geng, tingkat aplikasi) dan beri nama alat yang digunakan di setiap layer.
- Jelaskan mengapa `DCGM_FI_DEV_GPU_UTIL` adalah sinyal HPA yang salah untuk vLLM dan sebutkan dua penggantinya (kedalaman antrian, pemanfaatan cache KV).
- Jelaskan penjadwalan geng dan mode kegagalan alokasi parsial yang dicegah oleh Penjadwal KAI (7 dari 8 GPU menganggur).
- Sebutkan kebijakan konsolidasi Karpenter (`WhenEmptyOrUnderutilized`) yang menghentikan pekerjaan GPU yang sedang berjalan dan sebutkan alternatif aman tahun 2026.

## Masalah

Tim kamu mengirimkan layanan penyajian LLM di Kubernetes. kamu menyiapkan HPA dengan `DCGM_FI_DEV_GPU_UTIL` sebagai sinyalnya. Layanan ini memiliki pemanfaatan 100% selama jam kerja. HPA tidak pernah meningkat - ia mengira kamu sudah kenyang. kamu menambahkan replika secara manual; TTFT turun. HPA masih belum berskala. Sinyalnya berbohong kepada kamu.

Secara terpisah, kamu menggunakan Cluster Autoscaler untuk node. Permintaan token 1M tiba pada jam 2 pagi; cluster menghabiskan waktu 3 menit untuk menyediakan node, dan waktu permintaan habis.

Sekali lagi secara terpisah, kamu menerapkan model 70B yang memerlukan 8 GPU di 2 node. Cluster ini memiliki 7 GPU gratis dan 1 tersebar di 3 node. Cluster Autoscaler menyediakan node untuk 1 GPU yang hilang. Tujuh node menunggu 4 menit untuk menghabiskan uang sementara Kubernetes mengaktifkan GPU terakhir.

Tiga layer, tiga mode kegagalan berbeda. Penskalaan otomatis yang mendukung GPU pada tahun 2026 tidak "mengaktifkan HPA". Ini menyusun penyediaan node, penjadwalan geng, dan penskalaan otomatis sinyal aplikasi.

## Konsep

### Layer 1 — penyediaan simpul (Karpenter)

Karpenter mengawasi pod yang tertunda dan menyediakan node dalam waktu ~45-60 detik (Cluster Autoscaler biasanya membutuhkan waktu 90-120 detik untuk node GPU). Ia mengambil tipe instance secara dinamis sesuai batasan `NodePool` — jika pod kamu memerlukan 8 H100 dan klaster tidak memiliki node yang cocok, Karpenter akan menyediakannya secara langsung alih-alih menskalakan grup yang sudah ada.

**Perangkap konsolidasi**: `consolidationPolicy: WhenEmptyOrUnderutilized` default Karpenter berbahaya untuk kumpulan GPU. Ini akan menghentikan node GPU yang sedang berjalan untuk memigrasikan pod ke instance berukuran lebih murah. Untuk weight kerja inference, itu berarti mengeluarkan permintaan yang berjalan dan memuat ulang model 70B pada node baru. Loss adalah kapasitas menit ditambah kegagalan permintaan.

Pengaturan aman untuk kumpulan GPU:

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```Memungkinkan Karpenter mengkonsolidasikan node yang benar-benar kosong setelah satu jam tetapi tidak pernah mengeluarkan tugas yang sedang berjalan.

### Layer 2 — penjadwalan geng (KAI Scheduler)

KAI Scheduler (proyek "Karp" kemudian diganti namanya) menangani apa yang tidak dilakukan oleh kube-scheduler default:

**Penjadwalan geng** — menjadwalkan semua atau tidak sama sekali. Pod inference terdistribusi yang memerlukan 8 GPU, semuanya dimulai bersamaan atau tidak sama sekali. Tanpa ini, kamu mendapatkan jebakan alokasi parsial: 7 dari 8 pod dimulai, menunggu tanpa batas waktu, menghabiskan uang.

**Kesadaran topologi** — mengetahui GPU mana yang berbagi NVLink, mana yang berada di rak yang sama, dan mana yang memiliki InfiniBand di antaranya. Tempatkan pod sesuai kebutuhan. Weight kerja paralel tensor DeepSeek-V3 67B harus tetap berada di satu domain NVLink; KAI Scheduler menghormati hal itu.

**Antrean hierarki** — beberapa tim bersaing untuk mendapatkan kumpulan GPU yang sama dengan prioritas dan kuota. Kesulitan produksi Tim A didahului oleh tugas training Tim B hanya jika aturan prioritas mengizinkan.

KAI diterapkan bersama kube-scheduler sebagai penjadwal sekunder; kamu memberi anotasi pada weight kerja untuk menggunakannya. Tumpukan produksi Ray dan vLLM keduanya terintegrasi.

### Layer 3 — sinyal tingkat aplikasi

**Perangkap HPA**: `DCGM_FI_DEV_GPU_UTIL` adalah metrik siklus kerja — yang mengukur apakah GPU melakukan pekerjaan pada setiap interval pengambilan sample. Pemanfaatan 100% bisa berarti 10 atau 100 permintaan bersamaan; GPU sedang sibuk. Penskalaan pada siklus tugas adalah penskalaan secara membabi buta.

Lebih buruk lagi, vLLM dan mesin serupa mengalokasikan memori cache KV terlebih dahulu (hingga `--gpu-memory-utilization`). Penggunaan memori tetap mendekati 90% bahkan pada satu permintaan. HPA berbasis memori tidak pernah diturunkan skalanya.

**Sinyal pengganti tahun 2026**:

- Kedalaman antrian (jumlah permintaan menunggu prefill).
- Pemanfaatan cache KV (berapa fraksi blok yang dialokasikan ke urutan aktif).
- Per-replika P99 TTFT (sinyal SLA kamu).
- Goodput (permintaan memenuhi semua SLO per detik).

NVIDIA Dynamo Planner dan llm-d Workload Variant Autoscaler menggunakan sinyal ini dan menskalakan replika. Mereka menggantikan HPA sepenuhnya untuk melayani LLM.

### Kapan menggunakan apa

| Keputusan skala | Alat |
|----------------|------|
| Tambah/hapus node | Tukang Kayu |
| Jadwalkan pekerjaan multi-GPU | Penjadwal KAI |
| Tambah/hapus replika | Dynamo Planner / llm-d WVA (atau HPA khusus pada kedalaman antrian) |
| Pilih jenis GPU | Kolam Node Karpenter |
| Mencegah prioritas rendah | Antrian Penjadwal KAI |

### Pra-pengisian/dekode yang terpilah memperumit segalanya

Jika kamu menjalankan pra-pengisian/dekode terpilah (Fase 17 · 17), kamu memiliki dua kelas pod dengan pemicu penskalaan yang berbeda: pod pra-pengisian skala pada kedalaman antrean, dekode skala pod pada tekanan cache KV. llm-d mengekspos ini sebagai `Services` terpisah dengan HPA per peran. Jangan mencoba untuk menempatkan satu HPA di depan keduanya.

### Cold start juga penting di sini

Mitigasi cold-start (Fase 17 · 10) adalah saat waktu penyediaan node dapat dilihat oleh pengguna. Pemanasan Karpenter selama 45-60 detik ditambah weight model 20 GB ditambah mesin init berarti permintaan dari nol membutuhkan waktu 2-5 menit. Simpan kolam hangat (`min_workers=1`) untuk jalur kritis SLO, atau gunakan pos pemeriksaan bergaya Modal di layer aplikasi.

### Nomor yang harus kamu ingat

- Penyediaan node Karpenter: ~45-60 detik vs Cluster Autoscaler ~90-120 detik (node GPU).
- Penjadwal KAI mencegah alokasi sebagian limbah — perangkap 7-dari-8.
- `DCGM_FI_DEV_GPU_UTIL` sebagai sinyal HPA: rusak; menggunakan kedalaman antrian atau pemanfaatan KV.
- Karpenter `WhenEmptyOrUnderutilized`: menghentikan pekerjaan GPU yang sedang berjalan. Gunakan `WhenEmpty + consolidateAfter: 1h` untuk inference.

## Pakai`code/main.py` menyimulasikan penskala otomatis tiga lapis pada weight kerja GPU yang tinggi. Membandingkan HPA naif (siklus tugas), HPA kedalaman antrean, dan penskalaan terjadwal geng KAI. Melaporkan permintaan yang belum terpenuhi, menit GPU yang menganggur, dan skor gabungan.

## Kirim

Lesson ini menghasilkan `outputs/skill-gpu-autoscaler-plan.md`. Dengan topologi klaster, bentuk weight kerja, dan SLO, rencana penskalaan otomatis tiga lapis dirancang.

## Latihan

1. Jalankan `code/main.py`. Di bawah weight kerja yang meledak-ledak, berapa banyak permintaan yang dijatuhkan oleh HPA siklus tugas naif yang ditangkap oleh HPA kedalaman antrean? Dari manakah perbedaannya?
2. Rancang Karpenter NodePool untuk cluster yang melayani Llama 3.3 70B FP8 di H100 SXM5. Tentukan `capacity-type`, `disruption.consolidationPolicy`, `consolidateAfter`, dan taint yang menjauhkan weight kerja non-GPU dari node ini.
3. Tim kamu melaporkan bahwa penerapan terhenti dalam status Tertunda karena "GPU tersedia tetapi pod tidak dapat dijadwalkan." Diagnosis — apakah ini Karpenter, kube-scheduler, atau KAI Scheduler? Metrik mana yang dikonfirmasi?
4. Pilih sinyal untuk melakukan penskalaan otomatis pada pod prefill terpilah dan sinyal berbeda untuk decode pod. Benarkan keduanya.
5. Hitung biaya jebakan konsolidasi `WhenEmptyOrUnderutilized` pada layanan produksi 24x7 yang rata-rata terjadi 60 kejadian penghentian permintaan/hari pada P99 TTFT > 10 detik.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Tukang Kayu | "penyedia simpul" | Penskala otomatis simpul Kubernetes; penyediaan sub-menit |
| Penskala Otomatis Klaster | "scaler tua" | Pendahulu autoscaler node Kubernetes; lebih lambat, berbasis grup |
| Penjadwal KAI | "penjadwal GPU" | Penjadwal sekunder untuk geng + topologi + antrian |
| Penjadwalan geng | "semua atau tidak sama sekali" | Jadwalkan N pod secara atomik atau tunda semuanya |
| Kesadaran topologi | "sadar rak" | Tempatkan pod berdasarkan penempatan NVLink/IB/rak |
| `DCGM_FI_DEV_GPU_UTIL` | "Pemanfaatan GPU" | Metrik siklus tugas; BUKAN sinyal penskalaan untuk LLM |
| Kedalaman antrian | "menunggu permintaan" | Sinyal HPA yang benar untuk penskalaan terikat pra-pengisian |
| Pemanfaatan cache KV | "tekanan memori" | Sinyal HPA yang benar untuk penskalaan terikat dekode |
| Konsolidasi | "Konsolidasi Karpenter" | Penghentian node ke tipe instans yang lebih murah |
| `WhenEmpty + 1h` | "konsolidasi yang aman" | Kebijakan yang tidak menghapus pekerjaan GPU yang sedang berjalan |

## Bacaan Lanjutan

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — dokumen desain dan contoh konfigurasi.
- [Kontrol Gangguan Karpenter](https://karpenter.sh/docs/concepts/disruption/) — semantik kebijakan konsolidasi dan default yang aman untuk GPU.
- [NVIDIA — Inference LLM Terpilah di Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — sinyal penskalaan Dynamo Planner.
- [Ray docs — KAI Scheduler untuk RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Pola integrasi Ray.
- [Praktik Terbaik Komputasi dan Penskalaan Otomatis AWS EKS](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — panduan khusus Kubernetes yang dikelola.
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Desain Autoscaler Varian Weight Kerja.
