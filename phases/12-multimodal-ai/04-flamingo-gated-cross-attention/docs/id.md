# Flamingo dan Gated Cross-Attention untuk VLM Sedikit Pemotretan

> Flamingo DeepMind (2022) melakukan dua hal sebelum orang lain. Ini menunjukkan bahwa satu model dapat memproses rangkaian gambar, video, dan teks yang disisipkan secara sewenang-wenang. Dan hal ini menunjukkan bahwa VLM dapat belajar dalam konteks - berikan prompt beberapa gambar dengan tiga pasangan contoh (gambar, keterangan) dan model memberi keterangan pada gambar baru tanpa langkah gradient apa pun. Mekanismenya: layer attention silang yang terjaga keamanannya, disisipkan di antara layer LLM yang ada yang dibekukan, dengan gerbang tanh terpelajar yang dimulai dari nol sehingga kemampuan teks LLM dipertahankan pada inisialisasi. Lesson ini membahas resampler Perceiver Flamingo dan arsitektur gerbang attention silang — nenek moyang input interleaved Gemini dan token visual Idefics2.

**Type:** Learn
**Language:** Python (stdlib, gated cross-attention + demo resampler Perceiver)
**Prerequisites:** Phase 12 · 03 (BLIP-2 Q-Mantan)
**Waktu:** ~120 menit

## Tujuan Pembelajaran

- Jelaskan bagaimana attention silang yang terjaga keamanannya mempertahankan kemampuan teks LLM yang dibekukan pada inisialisasi melalui tanh(gerbang) = 0.
- Telusuri resampler Perceiver: N patch gambar → K pertanyaan "laten" yang diperbaiki melalui attention silang.
- Jelaskan bagaimana Flamingo menangani rangkaian gambar-teks yang disisipkan dengan penyembunyian kausal yang menghormati penempatan gambar.
- Mereproduksi struktur prompt multimodal beberapa gambar (3 contoh keterangan gambar kemudian gambar kueri).

## Masalah

BLIP-2 memasukkan 32 token visual ke dalam layer input LLM yang dibekukan. Berfungsi untuk satu gambar per prompt. Namun bagaimana jika kamu ingin memasukkan *banyak* gambar yang disisipkan dengan teks, seperti pada "ini gambar A, beri keterangan; ini gambar B, beri keterangan; sekarang ini gambar C, beri keterangan"? Attention mandiri LLM perlu menangani token gambar dan token teks dalam satu aliran, dan pertanyaan tentang posisi mana yang dapat menangani gambar mana yang menjadi rumit.

Jawaban Flamingo: jangan mengubah aliran input LLM sama sekali. Sisipkan layer attention silang ekstra di antara blok LLM yang ada. Token teks masih mengalir melalui attention diri kausal LLM seperti biasa. Di antara setiap beberapa blok LLM, token teks juga memperhatikan feature gambar melalui layer baru yang terjaga keamanannya. Gerbang (diinisialisasi ke nol) berarti pada langkah nol layer baru tidak dapat dioperasikan — model berperilaku persis seperti LLM yang telah dilatih sebelumnya. Saat training berlangsung, gerbang terbuka dan informasi visual mulai mengalir.

Pertanyaan kedua yang dijawab Flamingo: bagaimana kamu menangani sejumlah gambar yang bervariasi (0, 1, atau banyak) per prompt? Perceiver resampler — modul attention silang kecil yang mengambil berapa pun jumlah patch yang kamu miliki dan menghasilkan sejumlah token laten visual yang tetap. Layer attention silang LLM melihat bentuk yang sama terlepas dari berapa banyak gambar yang ditampilkan.

## Konsep

### LLM beku

Flamingo dimulai dengan Chinchilla 70B LLM beku. Semua weight 70B tidak tersentuh. Teks yang ada self-attention dan FFN beroperasi normal.

### Pengambil sample ulang persepsi

Untuk setiap gambar di prompt, ViT menghasilkan N token patch. Resampler Perceiver memiliki K laten tetap yang dapat dipelajari (Flamingo menggunakan K=64). Setiap blok resampler terdiri dari dua sub-langkah:

1. Attention silang: K laten memperhatikan N token patch (Q dari laten, K/V dari patch).
2. Attention diri + FFN dalam keadaan laten.

Setelah 6 blok resampler, outputnya adalah K=64 token visual redup 1024, terlepas dari berapa banyak patch yang dihasilkan ViT. Gambar 224x224 (196 patch) dan gambar 480x480 (900 patch) keduanya keluar sebagai 64 token resampler.Untuk video, resampler diterapkan secara temporal: patch setiap frame menghasilkan 64 laten, dan pengkodean posisi temporal memungkinkan model membedakan t=0 dari t=N. Video lengkapnya menjadi token visual T*64.

### Gerbang attention silang

Di antara setiap layer M dari LLM yang dibekukan (Flamingo menggunakan M=4), masukkan blok attention silang yang terjaga keamanannya:

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` adalah scalar yang dapat dipelajari dan diinisialisasi ke nol.
- `tanh(0) = 0`, jadi pada init cabang yang terjaga keamanannya memberikan kontribusi nol.
- Saat `alpha` menjauh dari nol, kontribusi lintas attention tumbuh dengan lancar.
- Sambungan sisa berarti bahkan gerbang yang terbuka penuh tidak menimpa representasi teks LLM; itu hanya menambahkan informasi visual di atasnya.

Ini adalah satu-satunya pilihan desain paling penting di Flamingo: pengkondisian visual bersifat aditif, terjaga keamanannya, dan nol pada inisialisasi. Flamingo pada langkah 0 adalah Chinchilla 70B yang sempurna dengan input teks saja.

### Menyembunyikan attention silang untuk input yang disisipkan

Dalam prompt seperti "<image A> caption A <image B> caption B <image C> ?", setiap token teks hanya akan melihat gambar yang muncul sebelumnya secara berurutan. Attention mask silang menerapkan: token teks pada posisi `t` hanya melayani token resampler gambar yang indeks gambarnya `i < i_t` dengan `i_t` adalah gambar terbaru sebelum posisi `t`. "Hanya melihat gambar terakhir sebelumnya" atau "melihat semua gambar sebelumnya" keduanya merupakan pilihan yang valid; Flamingo memilih yang pertama.

### Pembelajaran singkat dalam konteks

Prompt Flamingo terlihat seperti:

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

Model melihat pola penyelesaian dan mengeluarkan "burung" (atau gambar apa pun yang ditampilkan3). Tidak ada langkah gradient. Kemampuan pembelajaran dalam konteks LLM yang dibekukan dilakukan melalui attention silang yang terjaga keamanannya — inilah inti dari makalah ini dan mengapa hal itu penting.

### Training data

Flamingo dilatih pada tiga dataset:

1. MultiModal MassiveWeb (M3W): 43 juta halaman web dengan gambar dan teks yang disisipkan, merekonstruksi urutan bacaan.
2. Pasangan Gambar-Teks (ALIGN + LTIP): 4,4B pasangan.
3. Video-Text Pairs (VTP): 27 juta klip video pendek.

OBELICS (2023) adalah reproduksi terbuka dari korpus web yang disisipkan, yang digunakan oleh Idefics, Idefics2, dan sebagian besar model terbuka "mirip Flamingo".

### OpenFlamingo dan Berang-berang

OpenFlamingo (2023) adalah reproduksi terbuka. Arsitektur identik (Perceiver resampler + gated cross-attention pada LLaMA atau MPT yang dibekukan). Pos pemeriksaan di 3B, 4B, 9B. Kualitas tertinggal dari Flamingo karena basis LLM yang lebih kecil dan data yang lebih sedikit.

Otter (2023) dibangun di atas OpenFlamingo dengan penyetelan instruksi pada MIMIC-IT (dataset instruksi multimodal), yang juga menunjukkan attention silang yang terjaga keamanannya untuk mengikuti instruksi.

### Keturunan

- Idefics / Idefics2 / Idefics3: Garis silsilah attention silang Hugging Face yang terjaga keamanannya, semakin sederhana (Idefics2 menghilangkan sample ulang dan mendukung token patch langsung dengan pengumpulan adaptif).
- Transisi Flamingo ke Bunglon: pada tahun 2024 banyak tim beralih ke fusi awal (Lesson 12.11); Attention silang berpintu gaya Flamingo tetap dalam produksi di mana pembekuan tulang punggung diperlukan.
- Input sisipan Gemini: secara konseptual mewarisi fleksibilitas format sisipan Flamingo, meskipun mekanisme pastinya bersifat eksklusif.

### Perbandingan dengan BLIP-2| | BLIP-2 | Flamingo |
|---|---|---|
| Jembatan visual | Q-Mantan sekali pada input | Gerbang attention silang di setiap layer M |
| Token visual | 32 per gambar | 64 per gambar per layer lintas-attn |
| LLM Beku | Ya | Ya |
| Sedikit gambaran dalam konteks | Lemah | Kuat — inti kertas |
| Input yang disisipkan | Tidak ada dukungan asli | Ya, target desain |
| Training data | 130 juta pasang | 1,3 miliar pasang + 43 juta halaman disisipkan |
| Jumlah parameter | 188M dilatih | ~10B dilatih (layer lintas-attn) |
| Hitung | Hari pada 8 A100 | Berminggu-minggu di ribuan TPUv4 |

Pilih BLIP-2 untuk VQA gambar tunggal dengan anggaran terbatas. Pilih Flamingo/Idefics2 untuk penalaran interleaved, beberapa gambar, atau multi-gambar.

## Pakai

`code/main.py` menunjukkan:

1. Resampler Perceiver pada 36 token patch palsu dengan 8 laten yang dapat dipelajari (attention silang Python murni).
2. Langkah attention silang yang terjaga keamanannya dengan `alpha = 0` → output sama dengan input (LLM tidak berubah), lalu `alpha = 2.0` → kontribusi visual digabungkan.
3. Pembuat topeng berselingan yang menghasilkan topeng attention 2D untuk rangkaian "(gambar 1) (teks 1) (gambar 2) (teks 2)".

## Kirim

Lesson ini menghasilkan `outputs/skill-gated-bridge-diagnostic.md`. Dengan adanya konfigurasi VLM terbuka (resampler Y/N, frekuensi lintas attn, skema gerbang), ini mengidentifikasi elemen garis keturunan Flamingo dan menjelaskan strategi pembekuan. Berguna untuk men-debug mengapa kinerja teks menurun (jawaban: gerbang menjadi terlalu lebar dan terlalu cepat).

## Latihan

1. Hitung jumlah parameter visual Flamingo-9B: 9B LLM + 1,4B layer attention silang yang terjaga keamanannya + 64M resampler. Berapa bagian dari total parameter yang dilatih?

2. Implementasikan sisa yang terjaga keamanannya `y = tanh(alpha) * cross + x` di PyTorch. Tunjukkan secara eksperimental bahwa dengan `alpha=0`, `y==x` tepat di init.

3. Baca OpenFlamingo Bagian 3.2 (arXiv:2308.01390) tentang cara mereka menangani banyak gambar dalam satu batch ketika setiap prompt memiliki jumlah gambar yang berbeda. Jelaskan strategi padding.

4. Mengapa topeng attention silang Flamingo membiarkan token teks memperhatikan *hanya gambar terbaru* yang mendahuluinya, bukan semua gambar sebelumnya? Baca makalah Flamingo Bagian 2.4 dan jelaskan konsekuensinya.

5. Beberapa pengambilan gambar dalam konteks: buat prompt dengan 4 contoh "gambar → warna objek utama" untuk varian Flamingo baru. Jelaskan pola akurasi yang diharapkan saat kamu memvariasikan jumlah contoh dari 0 hingga 8.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Pengambil sample ulang persepsi | "Attention silang laten tetap" | Modul yang menghasilkan K token tetap dari sejumlah variabel patch input |
| Attention silang yang terjaga keamanannya | "Jembatan berpintu Tanh" | Layer sisa `y = tanh(alpha)*cross + x`, alpha yang dapat dipelajari, init 0 |
| Input yang disisipkan | "Urutan campuran" | Format cepat dengan gambar dan teks dicampur secara bebas sesuai urutan bacaan |
| LLM Beku | "Tidak ada gradient LLM" | Weight teks LLM tidak diperbarui; hanya resampler + training layer lintas-attn |
| Sedikit tembakan | "Contoh dalam konteks" | Berikan beberapa pasangan (gambar, jawaban) pada prompt; model menggeneralisasi tanpa menyempurnakan |
| OBELIK | "Korpus web yang disisipkan" | Buka dataset 141 juta halaman web dengan gambar dan teks dalam urutan bacaan |
| chinchilla | "Pangkalan beku 70B" | Teks beku Flamingo LLM, dari makalah Chinchilla DeepMind |
| Jadwal gerbang | "Bagaimana alpha bergerak" | Kecepatan terbukanya gerbang attention silang selama training |
| Frekuensi lintas attn | "Setiap M layer" | Seberapa sering blok attention silang yang terjaga keamanannya dimasukkan; Flamingo menggunakan M=4 |
| BukaFlamingo | "Reproduksi terbuka" | Pos pemeriksaan terbuka MosesML/LAION di 3-9B; arsitektur-identik dengan Flamingo |

## Bacaan Lanjutan

- [Alayrac dkk. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — makalah asli.
- [Awadalla dkk. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — reproduksi terbuka.
- [Laurençon dkk. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — korpus web yang disisipkan.
- [Jaegle dkk. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — arsitektur Perceiver umum.
- [Li dkk. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — keturunan Flamingo yang disetel dengan instruksi.
- [Laurençon dkk. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — penyederhanaan modern dari pendekatan Flamingo.
