# Model Multimodal Bunglon dan Hanya Token Penggabungan Awal

> Setiap VLM yang kita lihat sejauh ini memisahkan gambar dan teks. Token visual berasal dari encoder visi, dialirkan ke proyektor, lalu bertemu teks di dalam LLM. Kosakata visi dan teks tidak pernah tumpang tindih. Bunglon (Meta, Mei 2024) bertanya: bagaimana jika mereka melakukannya? Latih VQ-VAE yang mengubah gambar menjadi rangkaian token terpisah dari kosakata bersama. Setiap dokumen multimodal kini menjadi satu urutan — token teks dan token gambar disisipkan, satu loss autoregresif. Efek samping: model dapat menghasilkan output modalitas campuran — token teks dan gambar bergantian dalam satu panggilan inference. Lesson ini membaca tesis fusi awal dan membuat versi mainan ujung ke ujung.

**Type:** Build
**Language:** Python (stdlib, tokenizer VQ-VAE + decoder disisipkan)
**Prerequisites:** Fase 12 · 05, Fase 8 (AI Generatif)
**Waktu:** ~180 menit

## Tujuan Pembelajaran

- Jelaskan mengapa kosakata bersama + loss tunggal mengubah apa yang dapat dilakukan model.
- Jelaskan bagaimana VQ-VAE memberi token pada gambar menjadi urutan diskrit yang kompatibel dengan tujuan token berikutnya Transformer.
- Sebutkan trik stabilitas training Chameleon: QK-Norm, penempatan dropout, pengurutan LayerNorm.
- Bandingkan pendekatan Q-Former Chameleon vs BLIP-2 dan jelaskan kapan masing-masing pendekatan tersebut merupakan pilihan yang tepat.

## Masalah

VLM berbasis adaptor (LLaVA, BLIP-2, Qwen-VL) memperlakukan teks dan gambar sebagai dua hal yang berbeda. Token teks melewati `embed(text_token)`; sebuah gambar melewati `visual_encoder(image) → projector → ... pseudo_tokens`. Model ini memiliki dua jalur input yang digabungkan sebagian.

Tiga konsekuensi:

1. LLM hanya dapat menggunakan gambar, tidak memancarkannya. Outputnya hanya berupa teks.
2. Dokumen dengan modalitas campuran (paragraf dan gambar bergantian, seperti dalam artikel) terasa janggal - kamu harus mengurai input multimodal di luar model atau pembuatan rantai.
3. Ketidaksesuaian distribusi. Token visual dan token teks berada di berbagai wilayah ruang tersembunyi, sehingga menimbulkan masalah penyelarasan yang tidak kentara.

Bunglon menolak premis tersebut: gambar hanyalah rangkaian token terpisah dari kosakata bersama. Latih model pada dokumen yang disisipkan, satu kehilangan, satu dekoder autoregresif, dan kamu akan membuka pembuatan modalitas campuran secara gratis.

## Konsep

### VQ-VAE sebagai tokenizer gambar

Tokenizer adalah autoencoder variasional terkuantisasi vector. Arsitektur:

- Encoder: CNN + ViT yang memetakan gambar ke peta feature spasial, katakanlah feature 32x32 redup 256.
- Buku Code: kosakata yang dipelajari tentang vector K (Bunglon menggunakan 8192), juga redup 256.
- Kuantisasi: untuk setiap feature spasial, cari entri buku code terdekat berdasarkan distance L2. Ganti feature berkelanjutan dengan indeks integer.
- Decoder: CNN yang mengembalikan feature terkuantisasi ke piksel.

Training: loss rekonstruksi VAE + loss komitmen + loss buku code. Indeks buku code membentuk alfabet terpisah untuk gambar.

Untuk Bunglon: satu gambar menjadi 32*32 = 1024 token yang diambil dari kosakata 8192. Gabungkan dengan token teks (dari kosakata BPE LLM, katakanlah 32000). Kosakata akhir: 40192. Transformer melihat satu rangkaian, satu loss.

### Kosakata yang dibagikan

Kosakata Bunglon menggabungkan token teks, token gambar, dan pemisah modalitas. Setiap token memiliki satu ID. Layer embedding input memetakan setiap ID ke vector tersembunyi D-dim. Proyeksi output dipetakan kembali ke log kosakata. Softmax memilih token berikutnya, apa pun modalitasnya.Pemisah penting: tag `<image>` dan `</image>` mengelompokkan urutan token gambar. Pada waktu pembuatan, jika model memancarkan `<image>`, perangkat lunak hilir mengetahui 1024 token berikutnya adalah indeks VQ yang akan dikirim ke dekoder untuk rendering piksel.

### Generasi modalitas campuran

Inference adalah prediksi token berikutnya dalam kosakata bersama. Contoh prompt: "Gambarlah seekor kucing dan jelaskan." Bunglon mengeluarkan:

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

Model memilih urutan secara mandiri — model dapat menghasilkan gambar lalu teks, teks lalu gambar, atau interleave. Dekoder yang sama, loss yang sama.

Bandingkan dengan VLM adaptor yang pembuatannya hanya berupa teks. Bunglon membuka kembali pertanyaan tentang modalitas output model.

### Stabilitas training — QK-Norm, dropout, pemesanan LayerNorm

Training fusi awal tidak stabil dalam skala besar. Kertas Chameleon mendokumentasikan tiga trik:

- QK-Norm. Terapkan LayerNorm ke kueri dan proyeksi utama di dalam attention, sebelum perkalian titik. Mencegah ledakan magnitudo logit di kedalaman. Digunakan oleh beberapa model besar pasca-2024.
- Penempatan putus sekolah. Dropout setelah setiap penambahan sisa, bukan hanya setelah attention dan MLP. Diperlukan lebih banyak regularisasi ketika gradient dari token gambar dapat mendominasi.
- Pemesanan LayerNorm. Pra-LN pada cabang sisa (standar), ditambah LN tambahan pada sambungan lewati blok terakhir. Menstabilkan aliran gradient layer akhir.

Tanpa trik ini, training Bunglon 34B-param menyimpang di beberapa pos pemeriksaan. Dengan mereka, hal itu menyatu. Resep training memberikan kontribusi yang sama besarnya dengan arsitektur.

### Batas rekonstruksi tokenizer

VQ-VAE merugikan. Pada 8192 entri buku code dan 1024 token per gambar 512x512, PSNR rekonstruksi dibatasi sekitar 26-28 dB. Ini cukup untuk menghasilkan gambar yang dapat dikenali, namun terlihat lebih buruk daripada difusi ruang kontinu (Difusi Stabil 3 mencapai 32+ dB).

Tokenizer adalah penghambatnya. Tokenizer yang lebih baik (MAGVIT-v2, IBQ, SBER-MoVQGAN) mengangkat langit-langit. Emu3 (Lesson 12.12) mencapai generasi kualitas SDXL melalui tokenizer yang lebih baik saja.

### Bunglon vs BLIP-2 / LLaVA

Bunglon (fusi awal, kosakata bersama):
- Satu loss, satu decoder.
- Menghasilkan output modalitas campuran.
- Tokenizer adalah batas atas kualitas.
- Mahal: Dekoder VQ-VAE per gambar yang dihasilkan pada jalur inference.

BLIP-2 / LLaVA (fusi akhir, menara terpisah):
- Visi masuk, teks keluar saja.
- Menggunakan kembali LLM yang telah dilatih sebelumnya.
- Tidak ada hambatan tokenizer untuk pemahaman.
- Murah: umpan maju tunggal.

Pilih berdasarkan tugas. Jika kamu membutuhkan pembuatan citra, keluarga Bunglon. Jika kamu hanya memerlukan pemahaman, adaptor-VLM lebih sederhana dan menggunakan kembali komputasi yang lebih terlatih.

### Fuyu dan AnyGPT

Fuyu (Adept, 2023) adalah pendekatan terkait: lewati encoder visi terpisah sepenuhnya, masukkan patch gambar mentah melalui proyeksi input LLM seolah-olah itu adalah token, bukan tokenizer. Lebih sederhana dari Chameleon, kehilangan generasi output kosakata bersama.

AnyGPT (Zhan et al., 2024) memperluas Chameleon menjadi empat modalitas: teks, gambar, ucapan, musik. Trik VQ-VAE yang sama untuk masing-masing trafo bersama. Generasi mana pun. Dibahas lebih lanjut dalam Lesson 12.16.

## Pakai

`code/main.py` membuat model fusi awal mainan yang menyeluruh:

- Pengukur gaya VQ-VAE kecil yang memetakan patch 8x8 ke indeks buku code (K=16).
- Kosakata bersama (id teks 0..31) + (id gambar 32..47) + (pemisah 48, 49).
- Dekoder autoregresif mainan (tabel bigram) yang dilatih tentang teks sintetis + urutan token gambar.
- Perulangan pengambilan sample yang memancarkan token teks + gambar bergantian jika diberi prompt.Code tersebut sengaja membuat trafo tetap kecil (bigram) sehingga kamu dapat melacak aliran sinyal dari ujung ke ujung.

## Kirim

Lesson ini menghasilkan `outputs/skill-tokenizer-vs-adapter-picker.md`. Mengingat spesifikasi produk (hanya memahami vs memahami + menghasilkan, kualitas gambar yang diperlukan, anggaran biaya), produk ini memilih antara keluarga Chameleon (fusi awal) dan keluarga LLaVA (fusi akhir) dan membenarkannya dengan aturan praktis kuantitatif.

## Latihan

1. Bunglon menggunakan entri buku code K=8192 dan 1024 token per gambar 512x512. Perkirakan rasio kompresi vs gambar RGB 24-bit. Apakah itu merugikan? Seberapa merugikan?

2. Gambar 4K (3840x2160) dengan kepadatan VQ-VAE yang sama menghasilkan berapa banyak token gambar? Bisakah model bergaya Chameleon menghasilkan gambar 4K dalam satu panggilan inference? Apa yang rusak terlebih dahulu — konteks, kualitas tokenizer, atau cache KV?

3. Menerapkan QK-Norm dengan Python murni. Dengan kueri dan kunci 64-redup, tampilkan perkalian titik sebelum dan sesudah LayerNorm. Mengapa pengendalian magnitudo penting di kedalaman?

4. Baca Chameleon Bagian 2.3 tentang stabilitas latihan. Jelaskan mode kegagalan yang tepat yang diamati pada makalah pada 34B tanpa QK-Norm. Apa yang dimaksud dengan “ledakan norm”?

5. Perluas dekoder mainan untuk mengeluarkan respons modalitas campuran dengan prompt hanya teks. Ukur seberapa sering model memilih gambar pertama vs teks pertama dengan distribusi training data 60% teks pertama / 40% gambar pertama.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Fusi awal | "Token terpadu" | Gambar dikonversi ke token diskrit yang berbagi kosakata Transformer dari langkah pertama |
| VQ-VAE | "Tokenizer gambar" | CNN + ViT + buku code yang memetakan gambar ke indeks bilangan bulat yang dapat diprediksi oleh Transformer |
| Kosakata bersama | "Satu kamus" | Ruang ID token tunggal yang mencakup pemisah teks + gambar + modalitas |
| QK-Norm | "Penstabil attention" | LayerNorm diterapkan pada kueri dan kunci sebelum perkalian titiknya, mencegah ledakan norm |
| Generasi modalitas campuran | "Teks + output gambar" | Inference yang secara mandiri menghasilkan token teks dan gambar yang disisipkan dalam satu lintasan |
| Ukuran buku code | "K entri" | Jumlah vector diskrit yang dapat dikuantisasi oleh VQ-VAE; memperdagangkan kompresi untuk kesetiaan |
| Langit-langit Tokenizer | "Batas rekonstruksi" | PSNR terbaik dapat dicapai dengan mendekode token VQ; membatasi kualitas gambar model |

## Bacaan Lanjutan

- [Tim Bunglon — Bunglon: Model Fondasi Penggabungan Awal Modal Campuran (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan dkk. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu dkk. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan dkk. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Mahir — blog Fuyu-8B (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
