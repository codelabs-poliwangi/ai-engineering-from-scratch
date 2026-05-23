# TensorRT-LLM di Blackwell dengan FP8 dan NVFP4

> TensorRT-LLM hanya untuk NVIDIA tetapi unggul di Blackwell. Pada GB200 NVL72 dengan orkestrasi Dynamo, SemiAnalysis InferenceX mengukur $0,012 per juta token pada model 120 miliar pada Q1-Q2 2026, dibandingkan $0,09/M pada H100 + vLLM — kesenjangan ekonomi sebesar 7x. Tumpukannya terdiri dari tiga rezim floating-point yang digabungkan: FP8 tetap penting untuk cache KV dan kernel attention karena memiliki rentang dinamis yang mereka perlukan; NVFP4 (penskalaan mikro 4-bit) menangani weight dan activation; prediksi multi-token (MTP) dan pra-pengisian/dekode terpilah menambahkan 2-3x lagi di atasnya. Dukungan model hari-0 memuat weight FP4 secara langsung tanpa konversi pasca-latihan. Hasil tangkapan bagi tim teknik tahun 2026: TRT-LLM adalah tumpukan NVIDIA tertutup, jadi mengadopsinya akan menukar portabilitas dengan throughput. Jalankan perhitungan pada campuran model dan perangkat keras kamu sebelum melakukan.

**Type:** Learn
**Language:** Python (stdlib, memori mainan FP8/NVFP4, dan kalkulator biaya)
**Prerequisites:** Fase 17 · 04 (vLLM Melayani Internal), Fase 10 · 13 (Kuantisasi)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Jelaskan mengapa FP8 tetap penting untuk cache dan attention KV bahkan ketika weight berada di NVFP4.
- Hitung jejak HBM model perbatasan di bawah BF16, FP8, dan NVFP4 dan alasan dari mana penghematan itu berasal.
- Sebutkan feature khusus Blackwell yang dieksploitasi TRT-LLM (FP4 hari ke-0, MTP, penyajian terpilah, primitif semua-ke-semua).
- Putuskan kapan kunci NVIDIA TRT-LLM sebanding dengan selisih biaya 7x vs vLLM di Hopper.

## Masalah

Batasan inference ekonomi pada tahun 2026 adalah "berapa banyak token per dolar". Jawabannya bergantung pada empat pilihan bertumpuk: pembuatan perangkat keras (Hopper H100/H200 vs Blackwell B200/GB200), presisi (BF16 → FP8 → NVFP4), mesin penyajian (vLLM vs SGLang vs TRT-LLM), dan orkestrasi (biasa vs terpilah vs Dynamo).

Di Hopper dengan vLLM, MoE 120 miliar berjalan pada ~$0,09 per juta token. Di Blackwell dengan TRT-LLM + Dynamo, model yang sama dijual dengan harga ~$0,012 — 7x lebih murah. Beberapa dari kesenjangan itu adalah perangkat keras (Blackwell memiliki throughput LLM 11-15x per GPU vs Hopper). Beberapa di antaranya adalah: weight FP4, draf MTP, pra-pengisian/dekode terpilah, dan NVLink 5 menyeluruh untuk komunikasi pakar Kementerian Lingkungan Hidup.

kamu tidak dapat mereplikasi ini di luar tumpukan NVIDIA. Inilah konsekuensinya – portabilitas bagi perekonomian. Memahami pilihan tumpukan mana yang memberikan bagian kesenjangan yang mana adalah inti dari lesson ini.

## Konsep

### Mengapa FP8 masih menjadi dasar untuk cache KV

Kesalahan umum di tahun 2026: berasumsi NVFP4 berlaku di mana saja. Tidak. Cache KV memerlukan FP8 (floating point 8-bit) karena menyimpan kunci attention dan nilai yang menjangkau rentang dinamis yang luas. Menghitung KV ke FP4 menyebabkan hilangnya akurasi yang sangat besar - ujung distribusi menurun dan skor attention menurun. Bit eksponen FP8 memberikan cache KV rentang yang dibutuhkan.

NVFP4 (2025-2026) berlaku untuk weight dan activation. Penskalaan mikro: setiap blok weight memiliki faktor skalanya sendiri sehingga blok kecil dapat menjangkau rentang dinamis yang berbeda tanpa kehilangan skala per tensor. Untuk activation, FP4 bertahan karena activation dalam distance kecil dalam satu layer.

Konfigurasi khas Blackwell:

- Weight: NVFP4 (skala mikro 4-bit).
- Activation: NVFP4.
-Cache KV: FP8.
- Akumulator attention: FP32 (stabilitas softmax).

### Primitif khusus Blackwell yang digunakan TRT-LLM- **Anak timbangan FP4 hari ke-0**: penyedia model mengirimkan anak timbangan FP4 secara langsung; TRT-LLM dimuat tanpa konversi pasca training. Tidak ada langkah AWQ / GPTQ untuk FP4.
- **Prediksi multi-token (MTP)**: ide yang sama dengan EAGLE (Fase 17 · 05) tetapi terintegrasi ke dalam build TRT-LLM.
- **Pelayanan terpilah**: pra-pengisian dan dekode pada kumpulan GPU terpisah, cache KV ditransfer melalui NVLink atau InfiniBand. Ide yang sama dengan Dynamo (Fase 17 · 20).
- **Komunikasi primitif menyeluruh**: NVLink 5 memotong latensi komunikasi pakar MoE sebesar 3x vs Hopper. Kernel MoE TRT-LLM disetel untuk ini.
- **Penskalaan mikro NVFP4 + MXFP8**: penanganan faktor skala yang dipercepat perangkat keras pada Blackwell Tensor Cores.

### Angka-angka yang harus anda hafal

- HGX B200 seharga $0,02/M token di GPT-OSS-120B melalui TRT-LLM.
- GB200 NVL72 dengan token $0,012/M melalui Dynamo (mengatur TRT-LLM).
- Token H100 + vLLM ≈ $0,09/M pada weight kerja yang sebanding.
- Peningkatan throughput 2,8x dalam tiga bulan pembaruan TRT-LLM (2026).
- Throughput LLM 11-15x per GPU, Blackwell vs Hopper.
- MLPerf Inference v6.0 (April 2026): Blackwell mendominasi setiap tugas yang diserahkan.

### Berapa sebenarnya harga FP4 dalam hal kualitas

NVFP4 agresif. Pada weight kerja yang sangat berat (rantai pemikiran, matematika, pembuatan code dengan konteks yang panjang), weight FP4 terlihat menurun. Kalibrasi per blok mengurangi tetapi tidak menghilangkan. Model penalaran pengiriman tim sering kali menggunakan weight FP8 + activation FP4 sebagai kompromi, atau tetap menggunakan H200 dengan FP8 secara keseluruhan.

Aturannya: selalu validasi kualitas tugas pada set eval kamu sebelum melakukan weight NVFP4.

### Mengapa ini merupakan keputusan kunci NVIDIA

TRT-LLM adalah kernel sumber tertutup C++ + CUDA +. Model perlu dikompilasi untuk SKU GPU tertentu. Tanpa AMD, tanpa Intel, tanpa ARM. Jika strategi infra kamu adalah multi-vendor, TRT-LLM bukan merupakan starter untuk tingkat yang dilayani TRT-LLM — kamu masih dapat melakukan servis dari vLLM pada perangkat keras campuran. Jika kamu hanya menggunakan NVIDIA, celah 7x akan membayar kunci tersebut.

### 2026 resep praktis

Untuk tagihan inference tahunan $100M+, menjalankan Hopper + vLLM menyisakan 7-10x keuntungan. Migrasikan weight kerja dengan biaya dominan ke Blackwell + TRT-LLM + Dynamo. Pertahankan tingkat eksperimen pada H100 + vLLM untuk kecepatan iterasi model. Validasi kualitas pada setiap model yang dikonversi NVFP4 sebelum produksi.

### Bonus disagregasi

Penyajian terpilah TRT-LLM (kumpulan pra-pengisian dan dekode terpisah) dibahas secara mendalam di Fase 17 · 20. Di Blackwell, tumpukan pengali: weight FP4 × percepatan MTP × penempatan terpilah × perutean sadar-cache. Angka 7x mengasumsikan tumpukan penuh ini.

## Pakai

`code/main.py` menghitung jejak HBM, memecahkan code throughput (rezim terikat memori), dan token $/M untuk model di tiga tumpukan: H100 + BF16 + vLLM, H100 + FP8 + vLLM, B200 + NVFP4/FP8 + TRT-LLM. Jalankan untuk melihat efek penggabungan dan bagian kesenjangan yang disumbangkan setiap perubahan.

## Kirim

Lesson ini menghasilkan `outputs/skill-trtllm-blackwell-advisor.md`. Mengingat weight kerja, ukuran model, dan volume token tahunan, hal ini menentukan apakah tumpukan Blackwell + TRT-LLM layak untuk dikunci NVIDIA.

## Latihan1. Jalankan `code/main.py`. Pada MoE 120B dengan parameter aktif 30%, hitung throughput dekode terbatas bandwidth memori pada H100 BF16, H100 FP8, dan B200 NVFP4/FP8. Dari mana datangnya lompatan terbesar?
2. Pelanggan membelanjakan $2 juta/tahun untuk H100 + vLLM. Berapa jumlah impas GPU Blackwell yang perlu mereka beli untuk mengamortisasi migrasi ke TRT-LLM dalam 12 bulan, mengingat kesenjangan ekonomi sebesar 7x?
3. kamu melihat akurasi turun 3 poin pada MATEMATIKA setelah konversi weight NVFP4. Sebutkan dua jalur pemulihan: satu yang mengutamakan kualitas (pertahankan weight FP8), yang satu mengutamakan biaya (kalibrasi dengan data dalam domain).
4. Baca hasil inference MLPerf v6.0. Tugas mana yang memiliki kesenjangan Blackwell-over-Hopper terkecil, dan mengapa?
5. Hitung HBM yang diperlukan untuk model 405B pada weight NVFP4 + cache FP8 KV pada konteks 128k. Apakah cocok pada satu node GB200 NVL72?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| FP8 | "pelampung delapan bit" | titik mengambang 8-bit; digunakan untuk cache KV dan attention karena rentang dinamis |
| NVFP4 | "mikro empat bit" | Format FP mikroskala 4-bit NVIDIA; weight dan activation di Blackwell |
| MXFP8 | "MX delapan" | Varian FP8 skala mikro; akselerasi perangkat keras di Blackwell Tensor Cores |
| Hari ke-0 FP4 | "kirim weight FP4" | Penyedia model sudah merilis weight di FP4; tidak ada langkah konversi pasca kereta api |
| MTP | "prediksi multi-token" | Draf penguraian spekulatif terintegrasi TRT-LLM (Phase 17 · 05) |
| Penyajian terpilah | "pisahkan pra-isi/dekode" | Isi ulang dan dekode pada kumpulan GPU terpisah; KV ditransfer melalui NVLink/IB |
| Semua untuk semua | "Komunitas ahli MoE" | Token perutean pola komunikasi ke GPU ahli; NVLink 5 memotong 3x |
| InferensiX | "Bangku inference SemiAnalisis" | Tolok ukur biaya per token yang diterima industri pada tahun 2026 |

## Bacaan Lanjutan

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — Hasil MLPerf April 2026.
- [NVIDIA — Inference MoE di Blackwell](https://developer.nvidia.com/blog/delivering-massive- Performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — kernel NVLink 5 all-to-all dan MoE.
- [Ikhtisar TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/overview.html) — dokumentasi mesin resmi.
- [NVIDIA — Memperkenalkan Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — orkestrasi terpilah di atas TRT-LLM.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — rangkaian benchmark yang memublikasikan nomor Blackwell.
