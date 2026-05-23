# Jamba — Transformer SSM Hibrid

> Model ruang negara (SSM) dan Transformer menginginkan hal yang berbeda. Transformer membeli kualitas melalui attention dengan biaya kuadrat. SSM membeli inference waktu linier dan memori konstan melalui pengulangan tetapi kualitas lag. Jamba AI21 (Maret 2024) dan Jamba 1.5 (Agustus 2024) menempatkannya dalam model yang sama: 1 layer Transformer untuk setiap 7 layer Mamba, MoE di setiap blok lainnya, dan jendela konteks 256k yang muat pada satu GPU 80GB. Mamba-3 (ICLR 2026) memperketat sisi SSM dengan ruang negara bernilai kompleks dan proyeksi MIMO. Lesson ini membaca kedua arsitektur secara menyeluruh dan menjelaskan mengapa resep hibrid dapat bertahan selama tiga tahun penskalaan, sedangkan upaya konteks panjang SSM murni dan Transformer murni tidak.

**Type:** Learn
**Language:** Python (stdlib, kalkulator campuran layer)
**Prerequisites:** Fase 10 · 14 (arsitektur model terbuka), Fase 10 · 17 (attention asli yang jarang)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Jelaskan tiga primitif dalam blok Jamba — layer Transformer, layer Mamba, MoE — dan resep interleaving genap 1:7:.
- Nyatakan seperti apa kekambuhan SSM pada tingkat tinggi dan mengapa hal ini memungkinkan inference memori konstan.
- Hitung jejak cache KV model Jamba pada konteks 256k dan bandingkan dengan apa yang dibutuhkan model Transformer murni.
- Sebutkan tiga inovasi Mamba-3 (diskritisasi trapesium eksponensial, pembaruan status bernilai kompleks, MIMO) dan masalah yang menjadi target masing-masing inovasi tersebut.

## Masalah

Attention berbentuk kuadrat dalam panjang barisan. Model ruang negara bersifat linier. Perbedaan tersebut bertambah: pada 256 ribu token, peta attention Transformer adalah 65 miliar entri per kepala; status berulang SSM berukuran tetap, berapa pun panjang urutannya.

Model SSM murni (Mamba, Mamba-2) cocok dengan perplexity Transformer pada skala kecil namun tertinggal dalam tugas pelacakan status dan gagal pada beberapa kategori pengambilan dalam konteks. Intuisinya: SSM memampatkan riwayat ke keadaan tetap, dan ketika sejarah panjang, informasi bocor. Attention mengingat semuanya dengan tepat tetapi membayar biaya kuadrat.

Perbaikan yang jelas: gunakan keduanya. Letakkan layer Transformer di tempat yang penting untuk mengingat kembali. Gunakan layer SSM di tempat lain. Sesuaikan rasionya. Jamba adalah model tingkat produksi pertama yang mengirimkan resep hybrid ini dalam skala besar (total 52 miliar, aktif 12 miliar, konteks 256 ribu, GPU tunggal 80 GB). Jamba 1.5 menambah jumlah keluarga menjadi 398 miliar / 94 miliar aktif. Mamba-3 (ICLR 2026) adalah baseline SSM murni terbaik saat ini yang dapat dibangun kembali oleh hybrid.

Lesson ini membaca ketiga makalah dan menghasilkan model mental untuk "memilih rasio yang tepat".

## Konsep

### SSM dalam satu halaman

Model ruang negara memproses urutan `x_1, ..., x_N` melalui negara berukuran tetap `h`:

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

Pada setiap langkah, negara berkembang melalui dinamika linier `A`, menerima input `B x_t`, dan mengeluarkan output `C h_t`. `A, B, C` dapat dipelajari. Perhatikan properti penting: komputasi `y_t` hanya memerlukan `h_{t-1}` dan `x_t`, bukan `x` sebelumnya. Memori itu konstan. Inferensinya adalah O(1) per token.Trik untuk kualitas pemodelan adalah struktur `A`. S4 (Gu 2021) menggunakan matrix yang sangat terstruktur yang dapat dievaluasi secara efisien sebagai konvolusi panjang selama training. Mamba (Gu, Dao 2023) mengganti `A, B, C` yang tetap dengan yang bergantung pada data (bagian "selektif"). Mamba-2 (2024) semakin menyederhanakan strukturnya. Mamba-3 (2026) kembali menambahkan kompleksitas di tempat tertentu.

Properti utama: untuk decoder LLM, layer SSM adalah pengganti drop-in untuk layer attention, dengan status per layer ukuran tetap, bukan cache KV yang terus bertambah.

### Blok Jamba

Blok Jamba menyisipkan layer menurut dua angka:

- `l`: rasio attention terhadap Mamba. Jamba menggunakan `l = 8`, artinya 1 layer Transformer untuk setiap 7 layer Mamba (7 Mamba + 1 Attention = 8 layer per grup).
- `e`: frekuensi MoE. Jamba menggunakan `e = 2`, artinya setiap layer lainnya menerapkan MoE.

Urutan layer dalam satu blok:

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

Setiap blok Jamba terdiri dari 8 layer. Pada kedalaman 4 blok (total 32 layer), kamu mendapatkan 28 layer Mamba dan 4 layer Attention. 16 di antaranya menggunakan MoE.

### Mengapa rasio 1:7

AI21 menjalankan ablasi: berapa rasio attention terhadap Mamba yang memberikan perplexity per parameter DAN ingatan dalam konteks terbaik pada evaluasi konteks panjangnya?

- Terlalu banyak attention (1:1): kualitas meningkat tetapi memori dan kecepatan menurun.
- Terlalu sedikit attention (1:15): memori bagus tetapi pengambilan dalam konteks gagal.
- Titik manis: 1:7 atau 1:8.

Intuisinya: layer Transformer menangani penarikan kembali dan pelacakan status. Layer Mamba menangani sebagian besar pemrosesan yang murah.

### Pengkodean posisi

Layer Mamba sendiri sadar akan posisi (melalui pengulangan). Layer attention pada hibrida asli berbasis Mamba tidak menggunakan RoPE — layer SSM menyediakan informasi posisi. Jamba 1.5 menambahkan RoPE ke layer attention untuk generalisasi konteks yang lebih panjang, penyempurnaan post-hoc berdasarkan evaluasi empiris konteks panjang.

### Anggaran memori

Untuk bentuk Jamba-1 (32 layer: 28 Mamba + 4 Attention, tersembunyi 4096, 32 attention head):

- Cache KV (hanya layer attention): `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB` pada 256k BF16. Hanya 4 layer attention yang berkontribusi.
- Status SSM: `28 * hidden * state_size` per awalan token, tetapi ini adalah ukuran tetap per layer, tidak diskalakan dengan panjang urutan. Status Mamba pada umumnya adalah 16 per feature, tersembunyi 4096: `28 * 4096 * 16 * 2 = 3.7 MB` total.

Bandingkan dengan Transformer murni dengan 32 layer, tersembunyi yang sama, MHA penuh dengan 32 kepala: `2 * 32 * 32 * 128 * 256k * 2 = 128 GB` pada 256k BF16. Pengurangan 8x dalam cache KV. Bahkan jika dibandingkan dengan standar GQA(8) yang digunakan sebagian besar model tahun 2024 (`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`), hibrida 1:7 Jamba dengan kapasitas 16 GB masih 2x lebih kecil.

Itulah yang dimaksud AI21 dengan "konteks 256 ribu pada satu GPU 80 GB". Cache KV dari Transformer murni MHA penuh tidak akan muat; bahkan garis dasar GQA tidak memberikan ruang untuk weight dan activation; Jamba melakukannya.

### Mamba-3: garis dasar SSM murni pada tahun 2026

Mamba-3 (ICLR 2026, arXiv:2603.15569) memperkenalkan tiga inovasi pada sisi SSM murni:

1. **Diskritisasi trapesium eksponensial.** Menggantikan diskritisasi metode Euler di Mamba-2 dengan pengulangan yang lebih ekspresif. Operasi mirip konvolusi diterapkan pada input status dalam pengulangan inti, bukan sebagai konvolusi luar pada `x_t`.2. **Pembaruan status bernilai kompleks.** Mamba sebelumnya mereduksi matrix status dari kompleks (S4) menjadi diagonal nyata (Mamba) hingga identitas berskala (Mamba-2). Mamba-3 menambahkan kembali nilai-nilai kompleks — setara dengan embedding putaran yang bergantung pada data pada negara bagian. Hal ini memulihkan kemampuan pelacakan keadaan yang sebelumnya memerlukan biaya penyederhanaan yang bernilai riil.

3. **Proyeksi multi-input multi-output (MIMO).** Daripada menggunakan proyeksi scalar per feature, gunakan proyeksi bernilai matrix. Meningkatkan kekuatan pemodelan dan pemanfaatan perangkat keras waktu inference tanpa meningkatkan latensi dekode.

Pada parameter 1,5 miliar, Mamba-3 meningkatkan rata-rata akurasi hilir sebesar 0,6 poin dibandingkan Gated DeltaNet; varian MIMO menambahkan 1,2 lebih banyak dengan total perolehan 1,8 poin. Pada ukuran negara bagian yang sama, Mamba-3 cocok dengan Mamba-2 dengan separuh negara bagian.

Mamba-3 belum dikirimkan dalam produksi hibrida dalam skala besar — ​​namun merupakan kandidat yang jelas untuk sisi SSM pada model kelas Jamba berikutnya.

### Kapan harus memilih hibrida

Hibrida menang ketika:

- Konteksnya cukup panjang sehingga cache Transformer KV murni menjadi sulit (64k+).
- Tugas menggabungkan struktur jangka pendek (bagus untuk SSM) dengan penarikan jangka panjang (membutuhkan Transformer).
- kamu ingin menerapkan anggaran memori GPU tunggal yang tidak dapat menampung cache Transformer KV saja.

Hibrida kalah ketika:

- Konteksnya pendek (di bawah 16k). Overhead SSM terbuang sia-sia; Transformer murni baik-baik saja.
- Tugas memerlukan attention dari mana saja (penalaran mendalam, referensi silang multi-dokumen). Jarangnya layer attention pada hibrida itu menyakitkan.
- kamu menskalakan ke model perbatasan triliunan parameter. Pure-Transformer + MLA + MoE (gaya DeepSeek-V3) saat ini memenangkan perlombaan kemampuan.

### Lanskap persaingan

| Model | Keluarga | Skala | Klaim unik |
|-------|--------|------|-------------|
| Mamba-2 | SSM murni | 3B | waktu linier, memori konstan |
| Jamba | hibrida | 52B/12B | 256k pada 80GB |
| Jamba 1,5 Besar | hibrida | 398B/94B | konteks panjang tingkat perusahaan |
| Mamba-3 | SSM murni | 1.5B (kertas) | pelacakan negara dipulihkan |
| DeepSeek-V3 | Transformer murni + MoE | 671B/37B | kemampuan perbatasan |

Lanskap tahun 2026: Transformer murni MoE mendominasi wilayah terdepan, namun hibrida memiliki lebih dari 256 ribu konteks. Kemenangan pelacakan negara bagian Mamba-3 dapat mendorong rasio hybrid lebih rendah (lebih banyak SSM, lebih sedikit attention) pada generasi berikutnya.

## Pakai

`code/main.py` adalah kalkulator memori untuk arsitektur hybrid. Mengingat rasio SSM-Transformer dan konfigurasi ukuran tersembunyi/jumlah layer, ini menghitung:

- Cache KV pada konteks target.
- Memori keadaan SSM.
- Total memori pada konteks N untuk berbagai bentuk model.

Kalkulator mendukung:

- Garis dasar Pure-Transformer (cache KV bertambah dengan N).
- Hibrida 1:7 ala Jamba.
- Pure-SSM (tidak ada cache KV sama sekali).

Angka-angka tersebut diambil langsung dari makalah Jamba-1 dan Jamba-1.5 untuk bentuk yang dipublikasikan dan diekstrapolasi untuk varian hipotetis.

Pertimbangan integrasi untuk penerapan nyata:

- Sebagian besar server inference produksi (vLLM, SGLang) mendukung Jamba dan Mamba. Periksa versi spesifiknya.
- Pada konteks 256k, keunggulan memori Jamba muncul dalam throughput permintaan bersamaan. Pada VRAM yang sama kamu memuat lebih banyak rangkaian Jamba daripada rangkaian Transformer.
- Mamba-3 sebagai model mandiri belum dikirimkan dalam phase produksi — pratinjau penelitian pada 1,5 miliar.

## Kirim

Lesson ini menghasilkan `outputs/skill-hybrid-picker.md`. Mengingat spesifikasi weight kerja (profil panjang konteks, campuran tugas, anggaran memori), ini merekomendasikan antara Transformer murni, hibrida gaya Jamba, dan SSM murni, dengan alasan eksplisit tentang memori dan tradeoff kualitas.

## Latihan

1. Jalankan `code/main.py` untuk menghitung cache KV pada konteks 256k untuk Transformer murni 32 lapis (4096 tersembunyi, 32 kepala) dan untuk hibrida Jamba-1 dengan bentuk yang sama. Verifikasi pengurangan memori ~8x seperti yang diklaim makalah AI21.

2. Ubah kalkulator untuk memodelkan hibrida 1:3 (4 Mamba : 1 Attention) dan hibrida 1:15 (14 Mamba : 1 Attention). Plot cache KV vs rasio. Berapa rasio cache KV sama dengan memori status SSM?

3. Baca Bagian 3 makalah Jamba (arXiv:2403.19887). Jelaskan mengapa AI21 menggunakan Mamba-1 dibandingkan Mamba-2 meskipun Mamba-2 lebih cepat. Petunjuk: bagian ablasi hibrid mendokumentasikan hal ini.

4. Hitung parameter overhead MoE-setiap-layer lainnya di Jamba 1.5 Besar (total 398B, aktif 94B). Bandingkan rasio aktif dengan DeepSeek-V3 (37B/671B) dan jelaskan mengapa arsitektur Jamba mendorong rasio aktif lebih tinggi.

5. Baca Bagian 3 makalah Mamba-3 (arXiv:2603.15569). Jelaskan dalam tiga kalimat mengapa pembaruan status bernilai kompleks setara dengan embedding putar yang bergantung pada data. Ikat jawaban ke Fase 7 · Derivasi RoPE Lesson 04.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Model ruang negara (SSM) | "Pengulangan dengan keadaan tetap" | Layer dengan pengulangan yang dipelajari `h_t = A h_{t-1} + B x_t`; memori konstan per token |
| SSM Selektif | "Trik Mamba" | Parameter A, B, C yang bergantung pada data yang memberikan selektivitas seperti gerbang model pada waktu linier |
| Rasio Attention terhadap Mamba | "Berapa banyak layer attention" | Di Jamba, `l = 8` berarti 1 layer attention per 7 layer Mamba |
| Blok Jamba | "Grup 8 lapis" | Satu attention + tujuh Mamba + MoE pada posisi bergantian |
| Status SSM | "Penyangga tersembunyi" | Status per layer dengan ukuran tetap yang menggantikan cache KV untuk layer Mamba |
| konteks 256k | "Nomor andalan Jamba" | Panjang urutan Jamba-1 muat pada satu GPU 80GB; Transformer murni tidak bisa sebesar itu |
| Mamba-3 | "SSM murni 2026" | Arsitektur SSM murni terbaik saat ini dengan keadaan kompleks + MIMO; hibrida dasar dibangun kembali di sekitar |
| MIMO | "Multi-input multi-output" | Inovasi Mamba-3 menggunakan proyeksi bernilai matrix, bukan scalar per feature |
| Diskritisasi eksponensial-trapesium | "Kekambuhan Mamba-3" | Pengulangan yang lebih ekspresif yang mencakup diskritisasi metode Euler Mamba-2 |
| Arsitektur hibrida | "Campurkan attention dan SSM" | Model apa pun yang menyisipkan layer Transformer dan SSM; Jamba adalah pola dasar produksi |

## Bacaan Lanjutan- [Lieber dkk. — Jamba: Model Bahasa Hybrid Transformer-Mamba (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — makalah Jamba asli, ablasi rasio, klaim konteks 256k
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — keluarga yang ditingkatkan, rilis publik 398B/94B dan 12B/52B
- [Gu, Dao — Mamba: Pemodelan Urutan Waktu Linier dengan Ruang Status Selektif (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — makalah SSM selektif yang dibangun Jamba
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — penerus ruang keadaan terstruktur yang disederhanakan
- [Lahoti dkk. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — keadaan bernilai kompleks, MIMO, garis depan SSM murni tahun 2026
- [Gu dkk. — Memodelkan Urutan Panjang secara Efisien dengan Ruang Keadaan Terstruktur (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — makalah S4, titik awal silsilah SSM untuk LLM
