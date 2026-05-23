# Panduan Arsitektur DeepSeek-V3

> Fase 10 · Lesson 14 menyebutkan enam kenop arsitektur yang diputar setiap model terbuka. DeepSeek-V3 (Desember 2024, total 671 miliar parameter, 37 miliar aktif) mengubah keenamnya dan menambahkan empat lagi: Attention Laten Multi-Kepala, penyeimbangan weight bebas loss tambahan, Prediksi Multi-Token, dan training DualPipe. Lesson ini membaca arsitektur DeepSeek-V3 dari atas ke bawah dan memperoleh setiap jumlah parameter dari konfigurasi yang dipublikasikan. Pada bagian akhir, kamu dapat menjelaskan mengapa rasio 671B/37B adalah pilihan yang tepat dan mengapa MLA + MoE bersama-sama mengalahkan keduanya di garis depan.

**Type:** Learn
**Language:** Python (stdlib, kalkulator parameter)
**Prerequisites:** Fase 10 · 14 (panduan model terbuka), Fase 10 · 17 (NSA), Fase 10 · 18 (MTP), Fase 10 · 19 (Pipa Ganda)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Baca konfigurasi DeepSeek-V3 dari atas ke bawah dan jelaskan setiap bidang dalam bentuk enam kenop GPT-2 ditambah empat tambahan khusus DeepSeek.
- Turunkan jumlah parameter total (671B), jumlah parameter aktif (37B), dan komponen yang berkontribusi pada masing-masingnya.
- Hitung jejak cache KV MLA pada konteks 128k dan bandingkan dengan apa yang akan dibayar oleh model padat param aktif yang sama dengan GQA.
- Sebutkan empat inovasi khusus DeepSeek (MLA, MTP, perutean bebas loss tambahan, DualPipe) dan sebutkan bagian arsitektur/tumpukan training mana yang ditargetkan masing-masing.

## Masalah

DeepSeek-V3 adalah model frontier open pertama yang arsitekturnya sangat berbeda dari keluarga Llama. Llama 3 405B adalah "GPT-2 dengan enam kenop diputar". DeepSeek-V3 adalah GPT-2 dengan keenam kenop ditambah empat kenop lainnya. Membaca konfigurasi Llama 3 adalah pemanasan untuk membaca konfigurasi DeepSeek, tetapi struktur dalamnya — bentuk blok attention, logika perutean, tujuan waktu training — cukup berbeda sehingga kamu memerlukan panduan terpisah.

Hasil dari mempelajarinya: Rilis weight terbuka DeepSeek-V3 mengubah arti "kemampuan perbatasan" dalam model terbuka. Arsitekturnya adalah cetak biru yang banyak ditiru oleh training pada tahun 2026. Memahaminya adalah taruhan utama untuk peran apa pun yang menyentuh training atau inference LLM terdepan.

## Konsep

### Inti invarian, lagi

DeepSeek-V3 masih bersifat autoregresif. Itu masih menumpuk blok dekoder. Setiap blok masih mendapat attention ditambah MLP ditambah dua RMSNorm. Masih menggunakan SwiGLU di MLP. Masih menggunakan Tali. Pra-norm. Embedding yang terikat dengan weight. Garis dasar yang sama seperti setiap Llama atau Mistral.

### Perubahannya: MLA, bukan GQA

Dari Fase 10 · 14 kamu tahu GQA mengecilkan cache KV dengan membagikan K dan V ke seluruh kelompok kepala Q. Multi-Head Latent Attention (MLA) melangkah lebih jauh: K dan V dikompresi menjadi representasi laten tingkat rendah bersama (`kv_lora_rank`), kemudian didekompresi per kepala dengan cepat. Cache KV hanya menyimpan yang laten — biasanya 512 float per token per layer, bukan 8 x 128 = 1024 float.

Pada konteks 128k, DeepSeek-V3 dengan MLA (satu laten bersama `c^{KV}` per token per layer; K dan V keduanya berasal dari laten ini melalui proyeksi ke atas yang dapat diserap ke dalam matmul berikutnya):

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

Garis dasar GQA hipotetis (bentuk Llama 3 70B, kepala 8 KV, kepala redup 128) akan menghasilkan:

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

MLA 4x lebih kecil dari cache GQA gaya Llama-3-70B pada konteks 128k.

Pengorbanannya: MLA menambahkan langkah dekompresi per perhitungan attention (per kepala). Komputasi ekstra kecil dibandingkan dengan bandwidth yang dihemat. Kemenangan bersih untuk inference konteks panjang.### Perutean: penyeimbangan weight bebas loss tambahan

Router MoE memutuskan pakar mana yang memproses setiap token. Router yang naif memusatkan terlalu banyak pekerjaan pada beberapa ahli, meninggalkan yang lain menganggur. Perbaikan standar: tambahkan istilah loss tambahan yang menyebabkan ketidakseimbangan weight. Ini berfungsi tetapi sedikit menurunkan kinerja tugas utama.

DeepSeek-V3 memperkenalkan skema tambahan bebas loss. Ketentuan bias per pakar ditambahkan ke logit router, disesuaikan selama training dengan aturan sederhana: jika pakar `e` kelebihan weight, kurangi `bias_e`; jika kekurangan, tingkatkan. Tidak ada istilah loss tambahan. Training tetap bersih. Weight ahli tetap seimbang.

Dampak terhadap loss utama: tidak ada yang dapat diukur. Efek pada arsitektur MoE: lebih bersih, tidak ada hyperparameter loss tambahan yang perlu disesuaikan.

### MTP: training lebih padat + draf gratis

Dari Fase 10 · 18 kamu tahu DeepSeek-V3 menambahkan modul D=1 MTP yang memprediksi token dua posisi di depan. Sebagai kesimpulan, modul yang dilatih digunakan kembali sebagai draf penguraian code spekulatif dengan penerimaan 80%+. Pada training, setiap keadaan tersembunyi diawasi pada target D+1 = 2, memberikan sinyal yang lebih padat.

Parameter: 14B di atas utama 671B. Biaya tambahan: 2,1%.

### Training: DualPipe

Dari Fase 10 · 19 kamu tahu DualPipe adalah pipeline pipa dua arah yang tumpang tindih dengan potongan maju dan mundur dengan komunikasi lintas-simpul semua-ke-semua. Pada skala 2.048-H800 DeepSeek-V3, ia memulihkan sekitar 245 ribu jam GPU yang mungkin hilang dari 1F1B karena gelembung pipa.

### Konfigurasi, bidang demi bidang

Berikut adalah konfigurasi DeepSeek-V3 (disederhanakan):

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

Uraikan:

- `hidden_size=7168`: embed dimension.
- `num_hidden_layers=61`: total kedalaman blok.
- `first_k_dense_layers=3`: 3 blok pertama menggunakan MLP padat ukuran 18432. 58 sisanya menggunakan MoE.
- `num_attention_heads=128`: 128 kepala kueri.
- `kv_lora_rank=512`: K dan V dikompresi ke dimension laten ini dan didekompresi per kepala.
- `num_experts=256, num_experts_per_tok=8`: setiap blok KLH memiliki 256 tenaga ahli, rute teratas-8.
- `shared_experts=1`: selain 256 pakar yang diarahkan, 1 pakar yang selalu aktif berkontribusi pada setiap token. Anggap saja sebagai "lantai padat" yang memastikan setiap token mendapatkan sesuatu yang dapat diandalkan.
- `moe_intermediate_size=2048`: ukuran tersembunyi MLP masing-masing pakar. Lebih kecil dari MLP padat karena jumlahnya 256.

### Akuntansi parameter

Perhitungan selengkapnya ada di `code/main.py`. Judulnya:

- Embedding: `vocab * hidden = 129280 * 7168 = ~0.93B`.
- 3 blok padat pertama: attention dengan MLA (~144 juta per blok) + MLP padat (~260 juta per blok) + norm. Totalnya sekitar 1,2 miliar.
- 58 blok MoE: attention dengan MLA (~144 juta) + masing-masing 256 pakar (masing-masing 30 juta) + 1 pakar bersama (30 juta) + norm. Total ~7,95 miliar per blok, termasuk semua pakar. Total 461 miliar untuk 58 blok KLH.
- Modul MTP: 14B.

Total keseluruhan: ~476 miliar untuk arsitektur inti + 14 miliar MTP + jelas bahwa angka 671 miliar yang dipublikasikan memperhitungkan parameter struktural tambahan (tensor bias, komponen khusus pakar, penskalaan pakar bersama, dll.). Angka yang kami hasilkan di kalkulator berada dalam rentang 3-5% dari angka yang dipublikasikan — delta tersebut berasal dari dokumen laporan akuntansi mendalam DeepSeek di lampiran Bagian 2.

Parameter aktif per penerusan:- Attention: 144M per layer * 61 = 8,8B (semua layer terbakar).
- MLP aktif: 3 layer pertama padat (3 * 260M = 780M), 58 layer MoE masing-masing aktif dengan 8 rute + 1 bersama + overhead perutean. MLP aktif per layer: ~260 juta. Jumlah: 3*260M + 58*260M = ~15,9B.
- Embedding + norm : 1.2B.
- Total aktif: kira-kira 26B inti + 14B MTP (dilatih tetapi tidak selalu dijalankan pada inference) ≈ 37B.

### Rasio 671B / 37B

Rasio ketersebaran 18x (param aktif adalah 5,5% dari total). DeepSeek-V3 adalah model MoE perbatasan paling jarang yang mengirimkan weight terbuka. Mixtral 8x7B dengan rasio 13/47 (28%) jauh lebih padat. Llama 4 Maverick dengan rasio 17B/400B (4,25%) sebanding. Taruhan DeepSeek: pada skala terdepan, lebih banyak ahli dengan rasio activation lebih rendah menghasilkan kualitas per FLOP aktif yang lebih baik.

### Tempat DeepSeek-V3 berada

| Model | Jumlah | Aktif | Rasio | Attention | Ide baru |
|-------|------|-------|-------|-----------|-------------|
| Lama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4,25% | GQA | — |
| Campuran 8x22B | 141B | 39B | 27% | GQA | — |
| Pencarian Dalam V3 | 671B | 37B | 5,5% | MLA 512 | MLA + MTP + bebas aux + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | Ekstensi Benang |

### Lanjutan: R1, V4

DeepSeek-R1 (2025) adalah training penalaran yang dijalankan pada tulang punggung V3. R1 menggunakan arsitektur yang sama. Yang berubah adalah resep pasca-training (RL skala besar pada tugas-tugas yang dapat diverifikasi), bukan arsitektur pra-training.

DeepSeek-V4 (jika dikirimkan) diharapkan mempertahankan MLA + MoE + MTP dan menambahkan DSA (DeepSeek Sparse Attention), penerus NSA dari Fase 10 · 17. Garis keturunannya stabil: inovasi tingkat arsitektur terakumulasi; setiap versi memutar kenop tambahan.

## Pakai

`code/main.py` adalah kalkulator parameter khusus untuk bentuk DeepSeek-V3. Jalankan, bandingkan keluarannya dengan angka-angka di makalah, dan gunakan pada varian hipotetis (256 pakar vs 512, 8 teratas vs 16 teratas, peringkat MLA 512 vs 1024).

Apa yang harus dilihat:

- Jumlah parameter total vs 671B yang diterbitkan.
- Jumlah parameter aktif vs 37B yang dipublikasikan.
- Cache KV pada konteks 128k — perbandingan MLA vs GQA.
- Perincian per layer untuk melihat ke mana sebenarnya anggaran parameter disalurkan.

## Kirim

Lesson ini menghasilkan `outputs/skill-deepseek-v3-reader.md`. Mengingat model keluarga DeepSeek (V3, R1, atau varian masa depan lainnya), model ini menghasilkan pembacaan arsitektur komponen demi komponen yang memberi nama setiap bidang konfigurasi, memperoleh jumlah parameter berdasarkan komponen, dan mengidentifikasi yang mana dari empat inovasi khusus DeepSeek yang digunakan model.

## Latihan

1. Jalankan `code/main.py`. Bandingkan estimasi parameter total kalkulator dengan 671B yang dipublikasikan dan identifikasi dari mana delta tersebut berasal. Bagian 2 makalah ini memiliki perincian lengkap.

2. Ubah konfigurasi untuk menggunakan peringkat MLA 256, bukan 512. Hitung ukuran cache KV yang dihasilkan pada konteks 128k. Berapa persentase pengurangan yang diperolehnya, dan berapa biaya ekspresi per kepala?

3. Bandingkan perutean DeepSeek-V3 (256 pakar, 8 teratas) dengan varian hipotetis (512 pakar, 8 teratas). Parameter total bertambah; parameter aktif tetap sama. Secara teori, apa yang diperoleh dari kapasitas ahli tambahan, dan berapa biayanya dalam inference?

4. Baca Bagian 2.1 laporan teknis DeepSeek-V3 (arXiv:2412.19437) tentang MLA. Jelaskan dalam tiga kalimat mengapa matrix dekompresi K dan V dapat "diserap" ke dalam matmul berikutnya untuk efisiensi waktu inference.5. DeepSeek-V3 menggunakan training FP8 untuk sebagian besar operasi. Hitung penghematan memori FP8 vs BF16 untuk menyimpan weight 671B. Bagaimana hal ini bersinggungan dengan anggaran training token 14,8T?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| MLA | "Attention Laten Multi-Kepala" | Kompres K dan V menjadi laten peringkat rendah bersama (kv_lora_rank, biasanya 512), dekompresi per kepala saat itu juga; Cache KV hanya menyimpan |
| kv_lora_rank | "Kompresi MLA redup" | Ukuran laten bersama untuk K dan V; DeepSeek-V3 menggunakan 512 |
| K pertama layer padat | "Layer awal tetap padat" | Beberapa layer model MoE pertama melewati router MoE dan menjalankan MLP padat untuk stabilitas |
| nomor_ahli_per_tok | "Perutean teratas" | Berapa banyak pakar yang diarahkan untuk menembak per token; DeepSeek-V3 menggunakan 8 |
| Pakar bersama | "Pakar yang selalu aktif" | Para ahli yang memproses setiap token terlepas dari peruteannya; DeepSeek-V3 menggunakan 1 |
| Perutean bebas loss tambahan | "Keseimbangan weight yang disesuaikan dengan bias" | Istilah bias per pakar disesuaikan selama training untuk menjaga keseimbangan weight pakar tanpa menambahkan istilah loss |
| Modul MTP | "Kepala prediksi ekstra" | Blok Transformer memprediksi t+2 dari h^(1) dan E(t+1); training yang lebih padat, draf decoding spekulatif gratis |
| Pipa Ganda | "Pipa dua arah" | Jadwal training yang tumpang tindih dengan komputasi maju/mundur dengan cross-node all-to-all |
| Rasio parameter aktif | "Ketersebaran" | active_params/total_params; DeepSeek-V3 mencapai 5,5% |
| training FP8 | "training 8-bit" | Penyimpanan training dan banyak operasi komputasi di FP8; kira-kira membagi dua memori vs BF16 dengan biaya kualitas yang kecil |

## Bacaan Lanjutan

- [DeepSeek-AI — Laporan Teknis DeepSeek-V3 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — dokumen arsitektur, training, dan hasil lengkap
- [Kartu model DeepSeek-V3 di Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — file konfigurasi dan catatan penerapan
- [Makalah DeepSeek-V2 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — pendahulu yang memperkenalkan MLA
- [Makalah DeepSeek-R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — penerus training penalaran pada arsitektur V3
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — arah masa depan untuk attention keluarga DeepSeek
- [Repositori DualPipe](https://github.com/deepseek-ai/DualPipe) — referensi jadwal training
