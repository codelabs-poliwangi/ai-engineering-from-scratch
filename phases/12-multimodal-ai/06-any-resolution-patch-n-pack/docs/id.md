# Visi Resolusi Apa Pun: Patch-n'-Pack dan NaFlex

> Gambar asli bukan kotak berukuran 224x224. Tanda terimanya adalah 9:16, grafiknya adalah 16:9, pemindaian medisnya mungkin 4096x4096, tangkapan layar selulernya adalah 9:19.5. Jawaban VLM sebelum tahun 2024 — mengubah ukuran semuanya menjadi persegi tetap — menghilangkan sinyal yang membuat OCR, pemahaman dokumen, dan penguraian adegan resolusi tinggi berfungsi. NaViT (Google, 2023) menunjukkan bahwa kamu dapat mengemas patch resolusi variabel ke dalam satu batch Transformer dengan masking blok-diagonal. M-RoPE Qwen2-VL (2024) menghilangkan tabel posisi absolut sepenuhnya. AnyRes LLaVA-NeXT menyusun gambar resolusi tinggi ke dalam gambar dasar + subgambar. Varian NaFlex SigLIP 2 (2025) kini menjadi encoder default untuk VLM terbuka yang menginginkan satu pos pemeriksaan untuk melayani setiap rasio aspek. Lesson ini mengimplementasikan patch-n'-pack secara end to end.

**Type:** Build
**Language:** Python (stdlib, patch packer + masker blok-diagonal)
**Prerequisites:** Fase 12 · 01 (patch ViT), Fase 12 · 05 (LLaVA)
**Waktu:** ~120 menit

## Tujuan Pembelajaran

- Kemas tambalan dari kumpulan gambar beresolusi variabel ke dalam satu urutan dan buat attention mask blok-diagonal.
- Pilih antara ubin AnyRes (LLaVA-NeXT), NaFlex (SigLIP 2), dan M-RoPE (Qwen2-VL) untuk tugas tertentu.
- Hitung anggaran token untuk OCR, grafik, dan fotografi tanpa mengubah ukuran.
- Sebutkan tiga mode kegagalan pengubahan ukuran persegi: teks terjepit, konten terpotong, token terbuang pada padding.

## Masalah

Transformer mengharapkan suatu urutan. Batch adalah tumpukan urutan dengan panjang yang sama. Jika gambar kamu berukuran 224x224, kamu mendapatkan 196 token patch setiap kali, padding tidak diperlukan, pekerjaan selesai. Berlatihlah di 224, simpulkan di 224, jangan pernah memikirkan resolusi lagi.

Dunia tidak bekerja sama. Dokumen berbentuk potret (8,5x11 inci, 2:3-ish). Tangkapan layar bagan berbentuk lanskap (16:9). Tanda terimanya tinggi dan tipis (1:3). Pencitraan medis dikirimkan dengan resolusi 2048x2048 atau lebih besar. Tangkapan layar perangkat seluler berukuran 1170x2532 (0,46:1).

Tiga opsi sebelum tahun 2024 dan mengapa masing-masing opsi gagal:

1. Ubah ukurannya menjadi persegi tetap (224x224 atau 336x336). Squish mendistorsi teks dan wajah. Downscale menghancurkan label grafik dan konten OCR. Praktek standar hingga LLaVA-1.5.
2. Pangkas ke rasio aspek tetap. kamu membuang sebagian besar gambar, dan memilih lokasi pemotongan adalah masalah penglihatannya sendiri.
3. Bantalan ke sisi terpanjang. Memperbaiki distorsi tetapi membuang 50%+ token pada padding untuk gambar potret. Biaya attention kuadrat pada semua token pad tersebut.

Jawaban tahun 2024-2025: biarkan Transformer memakan patch pada resolusi asli gambar, dan cari tahu cara mengemas kumpulan heterogen ke dalam satu urutan tanpa komputasi yang sia-sia.

## Konsep

### NaViT dan patch-n'-pack

NaViT (Dehghani et al., 2023) adalah makalah yang menunjukkan bahwa hal ini berhasil dalam skala besar. Idenya mekanis:

1. Untuk setiap gambar dalam batch, hitung grid patch aslinya pada ukuran patch yang dipilih (misalnya 14).
2. Ratakan setiap potongan gambar ke dalam urutan panjangnya yang dapat diubah-ubah.
3. Gabungkan semua tambalan gambar ke dalam satu urutan panjang untuk kumpulan tersebut.
4. Buat attention mask diagonal blok sehingga tambalan gambar A hanya ada di dalam gambar A.
5. Membawa informasi posisi per patch (Tali 2D atau embedding posisi pecahan).

Kumpulan tiga gambar dengan ukuran 336x336 (576 token), 224x224 (256 token), dan 448x336 (768 token) menjadi satu rangkaian 1600 token dengan topeng diagonal blok 1600x1600. Tidak ada bantalan. Tidak ada komputasi yang sia-sia. Transformer menangani rasio aspek yang berubah-ubah.NaViT juga memperkenalkan penghapusan patch pecahan selama training — menjatuhkan 50% patch secara acak di seluruh batch — yang mengatur dan mempercepat training. SigLIP 2 mewarisi ini.

### AnyRes (LLaVA-NeXT)

AnyRes dari LLaVA-NeXT adalah alternatif pragmatis. Mengingat gambar beresolusi tinggi dan encoder tetap (CLIP atau SigLIP di 336), susun gambar:

1. Pilih tata letak kisi dari kumpulan yang telah ditentukan sebelumnya — (1x1), (1x2), (2x1), (1x3), (3x1), (2x2), dll. — yang paling sesuai dengan rasio aspek gambar.
2. Ubin gambar penuh ke dalam grid; setiap ubin menjadi potongan 336x336.
3. Buat juga thumbnail: seluruh gambar diubah ukurannya menjadi 336x336 sebagai token konteks global.
4. Enkode setiap ubin melalui encoder 336 yang dibekukan. Gabungkan token ubin + token thumbnail.

Untuk gambar berukuran 672x672 pada kisi 2x2 ditambah thumbnail: 4 * 576 + 576 = 2880 token visual. Mahal tapi efektif — LLM melihat detail lokal dan konteks global.

AnyRes adalah rute pilihan ketika encoder kamu dibekukan dan hanya mendukung satu resolusi. Ini meningkatkan jumlah token untuk gambar besar (gambar 1344x1344 pada kisi 4x4 adalah 9216 + 576 ≈ 9800 token, yang memenuhi sebagian besar konteks LLM 8k).

### M-Tali (Qwen2-VL)

Qwen2-VL memperkenalkan Embedding Posisi Putar Multimodal. Alih-alih posisi pecahan NaViT atau ubin dan thumbnail AnyRes, setiap patch membawa posisi 3D (temporal, tinggi, lebar). Rotasi kueri/kunci menangani H, W, dan panjang temporal yang berubah-ubah.

M-RoPE mengirimkan resolusi dinamis asli tanpa training ulang. Sebagai kesimpulan, kamu memasukkan gambar HxW apa pun, penyemat patch menghasilkan token H/14 x W/14, setiap token mendapatkan posisinya (t=0, r=row, c=col), RoPE memutar attention dengan frekuensi yang tepat, selesai. Qwen2.5-VL dan Qwen3-VL melanjutkan ini. V2PE InternVL3 adalah ide yang sama dengan pengkodean variabel per modalitas.

Tidak seperti AnyRes, M-RoPE adalah token O(H x W / P^2) pada resolusi asli — tanpa overhead ubin perkalian. Tidak seperti NaViT, ia masih mengharapkan satu gambar per penerusan. Pengelompokan resolusi masih memerlukan patch-n'-pack di atasnya.

### NaFlex (SigLIP 2)

NaFlex adalah mode fleksibel asli pos pemeriksaan SigLIP 2. Sebuah model tunggal menyajikan beberapa panjang urutan (256, 729, 1024 token) pada inference. Secara internal ia menggunakan patch-n'-pack gaya NaViT selama training dan posisi pecahan absolut per patch. Nilai jualnya: satu pos pemeriksaan, pilih anggaran token kamu berdasarkan inference berdasarkan tugas.

Untuk tugas semantik (klasifikasi, pengambilan), 256 token. Untuk pemahaman OCR atau grafik, 1024 token. Tidak ada training ulang.

### Masker kemasan

Masker blok-diagonal adalah tempat sebagian besar implementasi tersandung. Untuk rangkaian paket dengan panjang `N_total` yang mencakup gambar `i=0..B-1` dengan panjang `n_i`, topeng `M` bentuk `(N_total, N_total)` adalah 1 jika kedua indeks berada dalam blok gambar yang sama, jika tidak 0. kamu dapat membuatnya dari daftar panjang kumulatif:

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

Ini adalah satu baris di PyTorch dengan `torch.block_diag` atau kumpulan eksplisit. Jalur panjang variabel FlashAttention (`cu_seqlens`) melewati masker seluruhnya dan hadir dalam urutan menggunakan tensor panjang kumulatif secara langsung — ~10x lebih cepat daripada masker padat untuk batch biasa.

### Anggaran token

Pilih strategi kamu berdasarkan tugas:- OCR / dokumen: 1024-4096 token. SigLIP 2 NaFlex pada 1024, atau AnyRes 3x3 + thumbnail.
- Grafik dan UI: 729-1024 token pada 384-448 asli. Resolusi dinamis Qwen2.5-VL dengan batas piksel maksimal.
- Foto natural: 256-576 token boleh saja. LLM hilir sudah cukup melihat. Bayar untuk token yang kepadatan kontennya tinggi.
- Video: 64-128 token per frame setelah penggabungan spasial, 2-8 FPS. Lesson 12.17 membahas hal ini.

Aturan produksi tahun 2026: pilih batas maksimal piksel per tugas, enkode pada rasio aspek asli hingga batas tersebut, kemas batch, dan lewati padding. Qwen2.5-VL mengekspos `min_pixels` dan `max_pixels` untuk kenop ini.

## Pakai

`code/main.py` mengimplementasikan patch-n'-pack untuk kumpulan gambar heterogen dengan koordinat piksel bilangan bulat. Itu:

- Mengambil daftar ukuran gambar (H, W).
- Menghitung panjang urutan patch setiap gambar pada ukuran patch 14.
- Kemas menjadi satu urutan dengan panjang total `sum(n_i)`.
- Membangun topeng attention blok-diagonal (padat, untuk kejelasan).
- Membandingkan biaya pengepakan vs pengubahan ukuran persegi dan ubin AnyRes.
- Mencetak tabel anggaran token untuk batch campuran (kwitansi, bagan, tangkapan layar, foto).

Jalankan. Banyaknya angka drop out menjadi alasan setiap open VLM 2026 menggunakan patch-n'-pack.

## Kirim

Lesson ini menghasilkan `outputs/skill-resolution-budget-planner.md`. Mengingat weight kerja rasio aspek campuran (OCR, bagan, foto, bingkai video) dan anggaran total token, strategi ini memilih strategi yang tepat (NaFlex, AnyRes, M-RoPE, atau fixed-square) dan mengeluarkan konfigurasi per permintaan. Gunakan keterampilan ini saat kamu mengukur VLM untuk suatu produk — keterampilan ini mencegah ledakan token 10x secara diam-diam yang mematikan anggaran latensi.

## Latihan

1. Resi berukuran 600x1500 (1:2.5). Pada ukuran patch 14, berapa banyak token resolusi asli? Berapa banyak setelah ukuran persegi diubah menjadi 336? Mana yang lebih kehilangan akurasi OCR dalam praktiknya?

2. Buat topeng blok-diagonal untuk kumpulan empat gambar dengan panjang 256, 576, 729, 1024. Pastikan matrix attention berukuran 2585x2585 dan memiliki entri tepat `256^2 + 576^2 + 729^2 + 1024^2` bukan nol.

3. Untuk gambar 1792x896 di patch 14, bandingkan: (a) ubah ukuran persegi menjadi 336 lalu enkode, (b) AnyRes 2x1 + thumbnail, (c) M-RoPE pada aslinya. Manakah yang menggunakan token paling sedikit? Manakah yang paling mempertahankan detailnya?

4. Menerapkan penurunan patch pecahan: diberikan urutan yang dikemas, jatuhkan 50% token secara seragam dan acak, dan perbarui masker blok-diagonal yang sesuai. Ukur perubahan ketersebaran masker.

5. Baca Bagian 3.2 makalah Qwen2-VL (arXiv:2409.12191). Jelaskan dalam dua kalimat apa yang dikontrol `min_pixels` dan `max_pixels` dan mengapa kedua batasan itu penting.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Patch-n'-pack | "Pengemasan bergaya NaViT" | Gabungkan urutan patch dengan panjang variabel dari gambar berbeda ke dalam satu dimension batch |
| Masker blok-diagonal | "Kemasan Masker" | Attention mask yang membatasi setiap patch gambar hanya untuk dirinya sendiri, bukan tetangga dalam paket |
| Apapun | "Ubin LLaVA-NeXT" | Pisahkan gambar beresolusi tinggi ke dalam kotak ubin berukuran tetap ditambah thumbnail global; menyandikan setiap ubin dengan encoder tetap |
| NaFlex | "SigLIP 2 asli-fleksibel" | Pos pemeriksaan SigLIP 2 tunggal yang menyajikan anggaran token 256/729/1024 pada inference tanpa training ulang |
| M-Tali | "Tali Multimodal" | Pengkodean posisi putar 3D (waktu, baris, kolom) yang menangani H, W, T sembarang tanpa tabel posisi |
| cu_seqlens | "Kemasan FlashAttention" | Tensor panjang kumulatif yang digunakan jalur varlen FlashAttention alih-alih masker diagonal blok padat |
| min_piksel / maks_piksel | "Batas resolusi" | Qwen2.5-VL kenop per permintaan membatasi token mengandalkan input yang sangat kecil atau sangat besar |
| Anggaran token visual | "Berapa banyak token per gambar" | Jumlah kasar token patch yang dikeluarkan per gambar; menetapkan anggaran cepat dan biaya attention LLM |

## Bacaan Lanjutan

- [Dehghani dkk. — Patch dan Paket: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang dkk. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon dkk. — Apa yang penting ketika membangun model bahasa visi? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen dkk. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Tim Qwen — Laporan Teknis Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
