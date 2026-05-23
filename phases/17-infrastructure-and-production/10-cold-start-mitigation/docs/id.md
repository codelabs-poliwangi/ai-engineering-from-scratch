# Mitigasi Cold Start untuk LLM Tanpa Server

> Gambar model 20 GB memerlukan waktu 5-10 menit (7B) hingga 20+ menit (70B) untuk beralih dari dingin ke penyajian. Di dunia yang benar-benar tanpa server, hal ini bukanlah pemanasan — melainkan pemadaman listrik. Mitigasi beroperasi pada lima layer: gambar node yang telah diunggulkan (Bottlerocket di AWS, lengkungan volume ganda), streaming model (NVIDIA Run:ai Model Streamer, asli dalam vLLM), snapshot memori GPU (Modal checkpoint, mulai ulang hingga 10x lebih cepat), kolam hangat (`min_workers=1`), pemuatan berjenjang (NVMe→DRAM→HBM ServerlessLLM, latensi 10-200x pengurangan), dan migrasi langsung yang memindahkan token input (KB) daripada cache KV (GB). Modal menerbitkan 2-4 detik permulaan dingin sebagai lantai; Baseten 5-10 detik default, sub-detik dengan pemanasan awal. Lesson ini mengajarkan kamu untuk mengukur, menganggarkan, dan menyusun lima layer.

**Type:** Learn
**Language:** Python (stdlib, mainan simulator jalur cold-start)
**Prerequisites:** Fase 17 · 02 (Ekonomi Platform Inference), Fase 17 · 03 (Penskalaan Otomatis GPU)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Hitung lima layer mitigasi cold-start dan beri nama satu alat atau pola pada setiap layer.
- Hitung total waktu start dingin sebagai jumlah dari (penyediaan node) + (pengunduhan weight) + (pemuatan weight ke dalam HBM) + (inisiasi mesin) untuk model 70B.
- Jelaskan mengapa migrasi langsung mentransfer token input (KB) bukan cache KV (GB) dan apa penaltinya (penghitungan ulang).
- Sebutkan trade-off kolam hangat (membayar untuk GPU yang menganggur atau menerima ekor start dingin) dan ambang batas SLA di mana `min_workers > 0` menjadi wajib.

## Masalah

Titik akhir LLM tanpa server kamu berskala ke nol dalam semalam. Pukul 8 pagi lalu lintas melonjak. Permintaan pertama menunggu sementara:

1. Karpenter menyediakan node GPU: 45-60 detik.
2. Wadah menarik gambar 30 GB dengan weight: 120-300 detik.
3. Mesin memuat weight ke dalam HBM: 45-120 detik tergantung pada ukuran model dan kecepatan penyimpanan.
4. vLLM atau TRT-LLM menginisialisasi grafik CUDA, kumpulan cache KV, tokenizer: 10-30 detik.

Total: 220-510 detik (kira-kira 3-8 menit) sebelum satu token muncul kembali. SLA kamu adalah 2 detik. kamu mengirimkan kolam hangat (`min_workers=1`) dan masalahnya sepertinya hilang — namun sekarang kamu membayar untuk satu GPU yang menganggur 24x7. Jika layanan kamu memiliki 5 produk yang masing-masing memiliki satu replika hangat, itu berarti 5 × 24 × 30 = 3.600 jam GPU/bulan, baik ada satu pengguna yang menelepon atau tidak.

Mitigasi cold-start adalah cara menjaga perekonomian tanpa server sambil memperkirakan latensi yang selalu aktif.

## Konsep

### Layer 1 — gambar simpul yang telah diunggulkan sebelumnya (Bottlerocket)

Di AWS, arsitektur volume ganda Bottlerocket memisahkan OS dari data. Ambil cuplikan volume data dengan gambar kontainer yang telah ditarik sebelumnya; rujuk ID snapshot di `EC2NodeClass` kamu. Node baru melakukan booting dengan weight yang sudah ada di NVMe lokal — langkah 2 dan sebagian dari 3 hilang. Bekerja dengan Karpenter secara asli. Penghematan umum: 2-4 menit per start dingin untuk model besar.

Setara di GCP: image VM kustom dengan layer container yang telah disiapkan sebelumnya. Di Azure: snapshot disk terkelola dengan pola yang sama.

### Layer 2 — streaming model (Jalankan:ai Model Streamer)

Daripada memuat seluruh file sebelum menjawab permintaan pertama, alirkan weight ke dalam memori GPU layer by layer dan mulai pemrosesan segera setelah blok Transformer pertama terpasang. NVIDIA Run:ai Model Streamer dikirimkan secara asli di vLLM 2026. Bekerja dengan S3, GCS, dan NVMe lokal. Memotong waktu pemuatan weight kira-kira setengahnya untuk model besar dengan tumpang tindih I/O dengan penyiapan komputasi.

### Layer 3 — snapshot memori GPU (Modal)Modal mengambil pos pemeriksaan status GPU (weight, grafik CUDA, wilayah cache KV) setelah pemuatan pertama. Restart berikutnya melakukan deserialisasi langsung ke HBM — 10x lebih cepat daripada inisialisasi ulang. Ini adalah hal yang paling mendekati "boot GPU hangat dalam 2 detik". Trade-off: snapshot adalah topologi per GPU, jadi jika Karpenter memigrasikan kamu ke SKU yang berbeda, kamu harus melakukan pemeriksaan ulang.

### Layer 4 — kolam hangat (min_workers=1)

Mitigasi paling sederhana: selalu siapkan satu replika. Biaya adalah tarif per jam satu GPU 24x7. Aritmatikanya brutal pada model kecil (kamu membayar $0,85-$1,50/jam untuk menghindari start dingin selama 30 detik) dan baik untuk model besar (membayar $4/jam untuk menghindari start dingin 5 menit). Ambang batas SLA yang mewajibkan kolam air hangat: biasanya TTFT P99 < 60 detik pada model 70B+.

### Layer 5 — pemuatan berjenjang (LLM Tanpa Server)

ServerlessLLM memperlakukan penyimpanan sebagai hierarki: NVMe (cepat namun besar), DRAM (sedang namun berjenjang), HBM (kecil namun instan). Weight sudah dimuat sebelumnya ke DRAM; memuat sesuai permintaan ke HBM. Makalah melaporkan pengurangan latensi 10-200x pada weight dingin dibandingkan disk-ke-HBM yang naif. Adopsi produksi masih awal tetapi integrasi dengan vLLM sudah ada.

### Layer 6 — migrasi langsung (pola bonus)

Ketika sebuah node menjadi tidak tersedia (spot eviction, node drain), pola tradisionalnya adalah memulai replika lain dan menguras antrean permintaan. Migrasi langsung memindahkan token input (kilobyte) ke tujuan yang modelnya dimuat dan menghitung ulang cache KV di tujuan. Perhitungan ulang lebih murah daripada mentransfer GB cache KV melalui jaringan. Berlaku untuk penerapan terpilah.

### Matematika kolam air hangat

Untuk layanan dengan P99 TTFT SLA 2s, pertanyaannya bukanlah "kolam hangat ya/tidak" tetapi "berapa banyak replika hangat, dan jalur mana yang mendapatkannya."

- Jalur interaktif bernilai tinggi (obrolan langsung, agen suara): `min_workers=1-2`.
- Jalur batch latar belakang (klasifikasi malam): diterima dari skala ke nol, cold start 5-10 menit dapat ditoleransi.
- Tingkat premium: `min_workers` per penyewa dengan kapasitas khusus.

### Ukur sebelum mengoptimalkan

Anatomi cold-start untuk model 70B pada node baru (ilustratif):

| Fase | Waktu | Mitigasi |
|-------|------|-----------|
| Penyediaan simpul | 50an | Bottlerocket + gambar yang sudah diunggulkan, kolam hangat |
| Tarikan gambar | 180an | Volume data yang diunggulkan sebelumnya (hilangkan) |
| Weight ke HBM | 75-an | Model streamer (setengah); Snapshot GPU (hilangkan) |
| Mesin mulai | 20an | Cache grafik CUDA persisten |
| Penyerang pertama | 3d | Latensi bawaan minimum |
| **Dingin sekali** | **328dtk** | |
| **Total dengan mitigasi** | **~15 detik** | pengurangan 22x |

### Nomor yang harus kamu ingat

- Modal cold start: 2-4 detik (dengan snapshot GPU).
- Mulai dingin default Baseten: 5-10 detik; sub-detik dengan pemanasan awal.
- Start dingin 70B mentah: 3-8 menit.
- Jalankan:ai Model Streamer: ~2x percepatan weight berat.
- Pemuatan berjenjang LLM tanpa server: pengurangan latensi 10-200x (nomor kertas).

## Pakai

`code/main.py` memodelkan jalur cold-start dengan dan tanpa setiap mitigasi. Melaporkan total waktu mulai dingin, biaya kolam hangat, dan tingkat permintaan titik impas yang melebihi jumlah yang dibayar oleh kolam hangat.

## Kirim

Lesson ini menghasilkan `outputs/skill-cold-start-planner.md`. Mengingat SLA, ukuran model, dan bentuk lalu lintas, pilih mitigasi mana yang akan ditumpuk.

## Latihan1. Jalankan `code/main.py`. Hitung tingkat permintaan titik impas di mana replika hangat lebih murah dibandingkan membayar pajak cold-start melalui penurunan permintaan tambahan di SLO.
2. kamu menyebarkan model 13B dengan P99 TTFT SLA 3 detik. Pilih tumpukan mitigasi minimum (layer paling sedikit) yang dapat mencapainya.
3. Pra-seeding Bottlerocket menghilangkan tarikan gambar tetapi weight masih dimuat dari snapshot ke HBM. Hitung jam dinding untuk model 70B jika NVMe yang didukung snapshot terbaca pada 7 GB/dtk.
4. Penyedia tanpa server kamu menawarkan snapshot GPU (Modal) dan tim kamu menolak karena "snapshot membocorkan PII". Perdebatkan kedua belah pihak — apa risiko realistisnya, dan apa mitigasinya (snapshot singkat, enkripsi, isolasi namespace)?
5. Rancang kebijakan kolam hangat berjenjang: berapa banyak replika hangat untuk pengguna berbayar, pengguna uji coba, dan weight kerja batch? Tunjukkan perhitungannya.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Mulai dingin | "jeda besar" | Waktu dari permintaan hingga token pertama pada replika baru |
| Kolam hangat | "selalu aktif minimum" | `min_workers >= 1` untuk menyiapkan setidaknya satu replika |
| Gambar yang sudah diunggulkan | "AMI panggang" | Gambar node dengan weight kontainer pra-residen |
| Roket botol | "OS simpul AWS" | OS yang dioptimalkan untuk kontainer AWS dengan dukungan snapshot volume ganda |
| Streamer model | "muatan streaming" | Tumpang tindih weight I/O dengan pengaturan komputasi |
| cuplikan GPU | "pos pemeriksaan ke HBM" | Membuat serial status GPU pasca-muat; deserialisasi saat restart |
| Pemuatan berjenjang | "NVMe + DRAM + HBM" | Hierarki tingkatan penyimpanan; memuat sesuai permintaan |
| Migrasi langsung | "pindahkan token" | Transfer input (KB), hitung ulang KV di tujuan |
| `min_workers` | "replika hangat" | Jumlah minimum tetap hidup tanpa server |
| Skala-ke-nol | "tanpa server penuh" | Tidak ada biaya saat menganggur; menerima pajak cold-start penuh |

## Bacaan Lanjutan

- [Modal — Performa cold start](https://modal.com/docs/guide/cold-start) — Tolok ukur dan arsitektur pos pemeriksaan yang diterbitkan Modal.
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — pola snapshot volume data yang telah diunggulkan sebelumnya.
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — weight weight tumpang tindih dengan pengaturan komputasi.
- [Baseten — Mitigasi cold-start](https://www.baseten.co/blog/cold-start-mitigation/) — pedoman pra-pemanasan.
- [Makalah LLM Tanpa Server (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — desain pemuatan berjenjang.
- [NVIDIA — Inference LLM Terpilah di Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — migrasi langsung untuk penerapan terpilah.
