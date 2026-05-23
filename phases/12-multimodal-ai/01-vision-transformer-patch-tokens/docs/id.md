# Vision Transformers dan Primitif Patch-Token

> Sebelum melakukan sesuatu yang multimodal, sebuah gambar harus menjadi rangkaian token yang dapat dimakan oleh Transformer. Makalah ViT 2020 menjawab hal ini dengan patch 16x16 piksel, proyeksi linier, dan embedding posisi. Lima tahun kemudian, setiap model perbatasan tahun 2026 (Claude Opus 4.7 dengan resolusi asli 2576 piksel, Gemini 3.1 Pro, Qwen3.5-Omni) masih dimulai dengan cara ini — pembuat enkode diubah dari ViT ke DINOv2 menjadi SigLIP 2, token register ditambahkan, skema posisi menjadi 2D-RoPE, tetapi tetap primitif. Lesson ini membaca pipeline patch-token dari ujung ke ujung dan membangunnya di stdlib Python sehingga sisa Fase 12 memiliki model mental yang konkret untuk "token visual".

**Type:** Learn
**Language:** Python (stdlib, patch tokenizer + kalkulator geometri)
**Prerequisites:** Fase 7 (Transformer), Fase 4 (Computer Vision)
**Waktu:** ~120 menit

## Tujuan Pembelajaran

- Ubah gambar HxWx3 menjadi rangkaian token patch dengan pengkodean posisi yang benar.
- Hitung panjang urutan, jumlah parameter, dan FLOP untuk ViT tertentu (ukuran patch, resolusi, redup tersembunyi, kedalaman).
- Sebutkan tiga peningkatan yang dilakukan ViT dari penelitian tahun 2020 hingga produksi tahun 2026: pra-training yang diawasi mandiri (DINO / MAE), token register, dan pengemasan resolusi asli.
- Pilih antara pengumpulan CLS, pengumpulan rata-rata, dan daftarkan token untuk tugas hilir.

## Masalah

Transformer beroperasi pada urutan vector. Teks sudah berupa urutan (byte atau token). Gambar adalah kisi piksel 2D dengan tiga pipeline warna — bukan urutan. Jika kamu meratakan setiap piksel, gambar RGB 224x224 menjadi 150.528 token, dan attention diri pada panjang tersebut adalah non-starter (panjang urutan kuadrat).

Pendekatan sebelum tahun 2020 memasang ekstraktor feature CNN ke bagian depan: ResNet menghasilkan peta feature 7x7 dari vector 2048 redup, memasukkan 49 token tersebut ke Transformer. Ini berfungsi tetapi mewarisi bias CNN (ekivariansi terjemahan, bidang reseptif lokal) dan kehilangan selera Transformer terhadap skala.

Dosovitskiy dkk. (2020) menanyakan pertanyaan blak-blakan: bagaimana jika kita melewatkan CNN? Pisahkan gambar menjadi patch berukuran tetap (misalnya 16x16 piksel), proyeksikan setiap patch secara linier ke dalam vector, tambahkan embedding posisi, dan masukkan urutannya ke Transformer vanilla. Pada saat itu, hal ini merupakan ajaran sesat – pandangan yang tidak berbelit-belit. Dengan data yang cukup (JFT-300M, lalu LAION) ia mengalahkan ResNet di ImageNet dan terus meningkat.

Pada tahun 2026, primitif ViT menjadi fondasi yang tidak perlu dipertanyakan lagi. Setiap menara visi VLM weight terbuka memiliki beberapa turunan (DINOv2, SigLIP 2, CLIP, EVA, InternViT). Pertanyaannya bukan lagi “haruskah kita menggunakan patch?” tetapi "berapa ukuran patch, apa jadwal resolusi, apa tujuan pra-training, apa pengkodean posisi."

## Konsep

### Tambalan sebagai token

Dengan adanya gambar `x` dengan bentuk `(H, W, 3)` dan ukuran tambalan `P`, kamu mengukir gambar tersebut ke dalam kotak `(H/P) x (W/P)` tambalan yang tidak tumpang tindih. Setiap patch adalah `P x P x 3` kubus piksel. Ratakan setiap kubus menjadi vector `3 P^2`. Terapkan proyeksi linier bersama `W_E` dengan bentuk `(3 P^2, D)` untuk memetakan setiap patch ke dalam dimension tersembunyi model `D`.

Untuk konfigurasi kanonik ViT-B/16:
- Resolusi 224, ukuran patch 16 → grid 14x14 → 196 token patch.
- Setiap patch memiliki nilai piksel `16 x 16 x 3 = 768`, diproyeksikan ke `D = 768`.
- Tambahkan token `[CLS]` → panjang urutan 197.Proyeksi patch secara matematis identik dengan konvolusi 2D dengan ukuran kernel `P`, stride `P`, dan pipeline output `D`. Begitulah cara code produksi mengimplementasikannya — `nn.Conv2d(3, D, kernel_size=P, stride=P)`. Pembingkaian "proyeksi linier" bersifat konseptual; pembingkaian kernel efisien.

### Embedding posisi

Tambalan tidak memiliki urutan yang melekat — Transformer melihatnya sebagai sebuah tas. ViT awal menambahkan embedding posisi 1D yang dapat dipelajari (satu vector 768 redup per posisi, 197 di antaranya). Berfungsi, tetapi mengaitkan model dengan resolusi training: pada inference kamu harus menginterpolasi tabel posisi jika kamu mengubah kisi.

Tulang punggung visi modern menggunakan 2D-RoPE (M-RoPE Qwen2-VL, default SigLIP 2) atau posisi 2D yang difaktorkan. 2D-RoPE memutar kueri dan vector kunci berdasarkan indeks patch (baris, kolom), sehingga model menyimpulkan posisi 2D relatif dari sudut rotasi. Tidak ada tabel posisi. Model ini menangani ukuran grid sewenang-wenang pada inference.

### Token CLS, output gabungan, dan token pendaftaran

Apa yang dimaksud dengan representasi tingkat gambar? Tiga pilihan hidup berdampingan:

1. `[CLS]` token. Tambahkan vector yang dapat dipelajari ke urutan patch. Setelah semua blok Transformer, status tersembunyi token CLS adalah representasi gambar. Diwarisi dari BERT. Digunakan oleh ViT asli, CLIP.
2. Berarti kolam. Rata-rata status tersembunyi output token patch. Digunakan oleh SigLIP, DINOv2, sebagian besar VLM modern.
3. Daftarkan token. Darcet dkk. (2023) mengamati bahwa ViT yang dilatih tanpa sink token yang eksplisit mengembangkan patch "artefak" bernorma tinggi yang membajak attention diri. Menambahkan 4–16 token register yang dapat dipelajari akan menyerap weight ini dan meningkatkan kualitas prediksi padat (segmentasi, kedalaman). DINOv2 dan SigLIP 2 keduanya dikirimkan dengan register.

Pilihannya penting untuk tugas-tugas hilir. CLS baik untuk klasifikasi. Untuk VLM yang memasukkan token patch ke dalam LLM, kamu melewatkan pengumpulan sepenuhnya — setiap patch menjadi token input LLM. Register dibuang sebelum penyerahan (itu adalah perancah, bukan konten).

### Pra-training: diawasi, kontrastif, disamarkan, disuling sendiri

ViT 2020 telah dilatih sebelumnya dengan klasifikasi yang diawasi pada JFT-300M. Dengan cepat digantikan oleh:

- CLIP (2021): teks gambar kontras pada 400 juta pasang. Lesson 12.02.
- MAE (2021, He dkk.): menutupi 75% tambalan, merekonstruksi piksel. Diawasi sendiri, bekerja pada gambar murni.
- DINO (2021) / DINOv2 (2023): penyulingan mandiri dengan siswa-guru, tanpa label, tanpa keterangan. DINOv2 ViT-g/14 2023 adalah tulang punggung visual murni terkuat dan default untuk kasus penggunaan "feature padat".
- SigLIP / SigLIP 2 (2023, 2025): CLIP dengan loss sigmoid dan NaFlex untuk rasio aspek asli. Menara visi yang dominan di VLM terbuka tahun 2026 (Qwen, Idefics2, LLaVA-OneVision).

Pilihan prapelatihan kamu menentukan kegunaan backbone: CLIP/SigLIP untuk pencocokan semantik dengan teks, DINOv2 untuk feature visual padat, MAE sebagai titik awal untuk penyempurnaan hilir.

### Hukum penskalaan

Penskalaan ViT (Zhai et al. 2022) menetapkan bahwa kualitas ViT mematuhi hukum yang dapat diprediksi dalam ukuran model, ukuran data, dan komputasi. Pada komputasi tetap:
- Model lebih besar + lebih banyak data → kualitas lebih baik.
- Ukuran patch adalah pengungkit pada panjang urutan vs fidelitas. Patch 14 (khas untuk DINOv2/SigLIP SO400m) memberikan lebih banyak token per gambar dibandingkan patch 16; lebih baik untuk OCR dan tugas-tugas padat, lebih buruk untuk kecepatan.
- Resolusi adalah faktor besar lainnya. Beralih dari 224 ke 384 ke 512 hampir selalu membantu, dengan biaya kuadrat di FLOP.ViT-g/14 (1B params, patch 14, resolusi 224 → 256 token) dan SigLIP SO400m/14 (400M params, patch 14) adalah dua encoder pekerja keras untuk VLM terbuka tahun 2026.

### Jumlah parameter untuk ViT

Perhitungan lengkapnya ada di `code/main.py`. Untuk ViT-B/16 pada 224:

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

Parkirkan setiap ViT dengan cara ini sebelum kamu memuat pos pemeriksaan. Ukuran tulang punggung menetapkan lantai VRAM kamu di VLM hilir mana pun.

### Konfigurasi produksi 2026

Encoder VLM paling terbuka yang dikirimkan pada tahun 2026 adalah SigLIP 2 SO400m/14 pada resolusi asli (NaFlex). Ini memiliki:
- Parameter 400 juta.
- Ukuran patch 14, resolusi default 384 → 729 token patch per gambar.
- Kumpulan rata-rata untuk tugas tingkat gambar; semua 729 patch mengalir ke LLM untuk VQA.
- 4 token pendaftaran, dibuang sebelum penyerahan LLM.
- 2D-RoPE dengan penskalaan tingkat gambar untuk rasio aspek asli.

Setiap keputusan dalam konfigurasi itu ditelusuri kembali ke makalah yang dapat kamu baca.

## Pakai

`code/main.py` adalah tokenizer patch dan kalkulator geometri. Dibutuhkan (gambar H, W, patch P, D tersembunyi, kedalaman L) dan melaporkan:

- Bentuk grid dan panjang urutan setelah ditambal.
- Urutan token untuk gambar mainan sintetis 8x8 piksel (berjalan melalui jalur proyek + rata).
- Jumlah parameter dipecah berdasarkan embedding tambalan, embedding posisi, blok Transformer, dan kepala.
- FLOP per forward pass pada resolusi target.
- Tabel perbandingan ViT-B/16 @ 224, ViT-L/14 @ 336, DINOv2 ViT-g/14 @ 224, SigLIP SO400m/14 @ 384.

Jalankan. Cocokkan jumlah parameter dengan angka yang dipublikasikan. Mainkan dengan ukuran dan resolusi patch untuk merasakan biaya penghitungan token.

## Kirim

Lesson ini menghasilkan `outputs/skill-patch-geometry-reader.md`. Mengingat konfigurasi ViT (ukuran patch, resolusi, redup tersembunyi, kedalaman), ini menghasilkan jumlah token, jumlah parameter, dan perkiraan VRAM dengan pembenaran. Gunakan keterampilan ini setiap kali kamu memilih tulang punggung visi untuk VLM — ini mencegah kejutan "token meledak dan konteks LLM saya terisi".

## Latihan

1. Hitung panjang urutan token patch untuk Qwen2.5-VL pada input asli 1280x720 dengan ukuran patch 14. Bagaimana cara membandingkannya dengan representasi khusus CLS?

2. Frame 1080p (1920x1080) di patch 14 menghasilkan berapa token? Pada 30 FPS dalam video berdurasi 5 menit, berapa total token visual? Biaya mana yang paling menghemat biaya kamu: pengumpulan, pengambilan sample bingkai, atau penggabungan token?

3. Menerapkan pengumpulan rata-rata pada token patch dengan Python murni. Verifikasi bahwa kumpulan rata-rata lebih dari 196 token output DINOv2 cocok dengan apa yang dikembalikan `forward` model saat kamu meminta embedding gabungan.

4. Baca Bagian 3 dari "Transformer Visi Perlu Register" (arXiv:2309.16588). Jelaskan dalam dua kalimat artefak apa yang diserap register dan mengapa hal itu penting untuk prediksi padat hilir.

5. Ubah `code/main.py` untuk mendukung patch-n'-pack: diberikan daftar gambar dengan resolusi berbeda, menghasilkan satu rangkaian paket dan attention mask blok-diagonal. Verifikasi dengan Lesson 12.06 ketika kamu mencapainya.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Tambalan | "16x16 piksel persegi" | Wilayah gambar input yang tidak tumpang tindih dengan ukuran tetap; menjadi satu token |
| Embedding tambalan | "Proyeksi linier" | Matrix yang dipelajari bersama (atau Konv2d dengan stride=P) memetakan piksel patch yang diratakan ke vector D-dim |
| Token CLS | "Token kelas" | Vector yang dapat dipelajari sebelumnya yang keadaan tersembunyi terakhirnya mewakili keseluruhan gambar; opsional pada tahun 2026 |
| Daftarkan token | "Tenggelamkan token" | Token ekstra yang dapat dipelajari yang menyerap artefak attention norm tinggi yang dikembangkan ViT selama pra-training |
| Embedding posisi | "Info posisi" | Vector atau rotasi per posisi membuat urutan-urutan sadar; 2D-RoPE adalah standar modern |
| Kisi | "Tambalan jaringan" | Rangkaian patch 2D (H/P) x (W/P) untuk resolusi dan ukuran patch tertentu |
| NaFlex | "Resolusi fleksibel asli" | Feature SigLIP 2: model tunggal menyajikan beberapa rasio aspek dan resolusi tanpa training ulang |
| Tulang punggung | "Menara Visi" | Encoder gambar terlatih yang output patch-tokennya memberi makan LLM dalam VLM |
| Pengumpulan | "Ringkasan tingkat gambar" | Strategi untuk mengubah token patch menjadi satu vector: CLS, mean, kumpulan attention, atau berbasis register |
| Tambalan 14 vs 16 | "Kotak yang lebih halus vs yang lebih kasar" | Patch 14 menghasilkan lebih banyak token per gambar, fidelitas lebih baik untuk OCR, lebih lambat; patch 16 adalah default klasik |

## Bacaan Lanjutan

- [Dosovitskiy dkk. — Sebuah Gambar Bernilai 16x16 Kata (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — ViT asli.
- [Dia dkk. — Autoencoder Bertopeng Adalah Pembelajar Visi yang Skalabel (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE, pra-training yang diawasi sendiri.
- [Oquab dkk. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — penyulingan mandiri dalam skala besar, tanpa label.
- [Darcet dkk. — Vision Transformers Membutuhkan Register (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — mendaftarkan token dan analisis artefak.
- [Tschannen dkk. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — menara visi default tahun 2026.
- [Zhai dkk. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — hukum penskalaan empiris.
