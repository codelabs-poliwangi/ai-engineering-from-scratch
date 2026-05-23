# CLIP dan Pra-Training Bahasa Penglihatan Kontrasif

> CLIP OpenAI (2021) membuktikan satu ide yang cukup besar untuk mendukung lima tahun ke depan: menyelaraskan encoder gambar dan encoder teks dalam ruang vector yang sama hanya menggunakan pasangan teks gambar web yang berisik dan loss kontras. Tidak ada label yang diawasi. 400 juta pasang. Ruang embedding yang dihasilkan melakukan klasifikasi zero-shot, pengambilan gambar-teks, dan dihubungkan ke setiap VLM 2026 sebagai menara visinya. SigLIP 2 (2025) menggantikan softmax dengan sigmoid dan melewati CLIP dengan biaya lebih rendah. Lesson ini menjelaskan matematika dari InfoNCE ke loss berpasangan sigmoid dan membangun langkah training di stdlib Python.

**Type:** Build
**Language:** Python (stdlib, InfoNCE + implementasi loss sigmoid)
**Prerequisites:** Fase 12 · 01 (patch ViT), Fase 7 (Transformer)
**Waktu:** ~180 menit

## Tujuan Pembelajaran

- Turunkan kehilangan InfoNCE dari informasi bersama dan terapkan versi vector yang stabil secara numerik.
- Jelaskan mengapa sigmoid pairwise loss (SigLIP) berskala ke batch 32768+ tanpa tuntutan softmax overhead gabungan.
- Jalankan klasifikasi ImageNet zero-shot dengan membuat templat teks (`a photo of a {class}`) dan mengambil argmax daripada kesamaan kosinus.
- Beri nama empat tuas yang diberikan training pra-training CLIP / SigLIP kepada kamu: ukuran batch, suhu, templat cepat, kualitas data.

## Masalah

Visi pra-CLIP diawasi. Kumpulkan dataset berlabel (ImageNet: 1,2 juta gambar, 1000 kelas), latih CNN, kirimkan. Label itu mahal, label bias terhadap apa yang bisa disetujui oleh pemberi label, dan label tidak akan berpindah ke tugas baru tanpa penyesuaian.

Web keterangan gambar memiliki lebih dari satu miliar pasangan berlabel longgar secara gratis. Gambar seekor anjing jenis Golden Retriever dengan teks alternatif "anjing saya Max di taman" membawa sinyal pengawasan — teks tersebut menjelaskan gambar tersebut. Pertanyaannya: bisakah kamu mengubahnya menjadi training yang bermanfaat?

Jawaban CLIP: perlakukan pasangan keterangan gambar sebagai tugas yang cocok. Diberikan sekumpulan N gambar dan N keterangan, belajar mencocokkan setiap gambar dengan keterangannya sendiri terhadap N-1 pengecoh. Pengawasannya adalah "dua hal ini saling berkaitan; N-1 ini tidak." Tidak ada label kelas. Tidak ada anotasi manusia. Hanya loss yang kontras.

Ruang embedding yang dihasilkan berfungsi lebih dari yang dilatih CLIP. ImageNet zero-shot berfungsi karena "foto kucing" di-embed di dekat gambar kucing yang tidak pernah diberi label kucing secara eksplisit. Ini adalah taruhan yang muncul setiap VLM 2026.

## Konsep

### Pembuat enkode ganda

CLIP memiliki dua menara:

- Encoder gambar `f`: ViT atau ResNet, menghasilkan vector D-dim per gambar.
- Encoder teks `g`: Transformer kecil, menghasilkan vector D-dim per keterangan.

Kedua menara menormalkan keluarannya ke satuan panjang. Kemiripannya adalah `cos(f(x), g(y)) = f(x)^T g(y)` karena keduanya merupakan norm unit.

Untuk kumpulan N pasangan (gambar, keterangan), buat matrix kesamaan `S` dengan bentuk `(N, N)`:

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

di mana `tau` adalah suhu yang dipelajari (CLIP diinisialisasi ke 0,07; dipelajari di ruang log).

### InfoNCE hilang

CLIP menggunakan entropi silang simetris pada baris dan kolom:

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

Ini adalah InfoNCE. Softmax di CE memaksa setiap gambar untuk mencocokkan keterangannya lebih dari setiap keterangan lainnya dalam kumpulan. Yang "negatif" adalah semua item batch lainnya. Batch yang lebih besar = lebih banyak negatif = sinyal lebih kuat. CLIP dilatih pada batch 32k; skala penting.

### Suhu`tau` mengontrol ketajaman softmax. Tau rendah → distribusi tajam, efek penambangan negatif keras. Tau tinggi → lembut, semua sample berkontribusi. CLIP mempelajari log (1/tau), terpotong untuk mencegah keruntuhan. SigLIP 2 memperbaiki tau awal dan menggunakan bias yang dipelajari sebagai gantinya.

### Mengapa skala sigmoid lebih baik (SigLIP)

Softmax membutuhkan seluruh matrix kesamaan yang sinkron. Dalam training terdistribusi, kamu harus mengumpulkan semua embedding ke setiap replika, lalu melakukan softmax. Ini adalah kuadrat ukuran dunia untuk komunikasi.

SigLIP menggantikan softmax dengan sigmoid berdasarkan elemen: untuk setiap pasangan `(i, j)`, kerugiannya adalah klasifikasi biner "apakah ini pasangan yang cocok?" label kelas positif adalah diagonalnya, yang lainnya negatif. Kerugiannya adalah:

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` if `i == j`, else 0. Loss setiap pasangan tidak tergantung. Tidak perlu berkumpul semua. Setiap GPU menghitung blok dan jumlah lokalnya. SigLIP 2 menskalakan ke batch 32k-512k dengan harga murah di mana CLIP memerlukan lebih banyak komunikasi secara proporsional.

### Klasifikasi zero-shot

Diberikan N nama kelas, untuk setiap kelas buat templat teks:

```
"a photo of a {class}"
```

Sematkan setiap templat dengan pembuat enkode teks. Sematkan gambar kamu dengan pembuat enkode gambar. Kesamaan kosinus Argmax = kelas prediksi. Tidak ada training pada kelas sasaran.

Templat yang cepat penting. Makalah asli CLIP menggunakan 80 templat per kelas (polos, artistik, foto, lukisan, dll.) dan membuat rata-rata embedding-nya. +3 poin ImageNet. Penggunaan modern biasanya memilih satu atau dua templat.

### Probe linier dan penyempurnaan

Zero-shot adalah garis dasar. Probe linier (melatih satu layer linier di atas feature CLIP yang dibekukan untuk kelas target kamu) mengalahkan tugas-tugas dalam domain. Penyempurnaan penuh mengalahkan probe linier pada dalam domain tetapi dapat mengganggu transfer zero-shot. Tiga rezim dengan tiga trade-off.

### SigLIP 2: Feature NaFlex dan padat

SigLIP 2 (2025) menambahkan:
- NaFlex: model tunggal menangani rasio aspek dan resolusi variabel.
- Feature padat yang lebih baik untuk segmentasi dan estimasi kedalaman, menargetkan penggunaan sebagai tulang punggung beku di VLM.
- Multibahasa: dilatih dalam 100+ bahasa dengan CLIP hanya dalam bahasa Inggris.
- Skala param 1B di mana CLIP mencapai puncaknya pada 400M.

Pada VLM terbuka tahun 2026, SigLIP 2 SO400m/14 adalah menara vision default. CLIP tetap menjadi default untuk pengambilan gambar-teks murni dengan distribusi training LAION-2B spesifik yang cocok dengan pola kueri kamu.

### SEJARAH, DASAR, OpenCLIP, EVA-CLIP

ALIGN (Google, 2021): ide yang sama seperti CLIP, skala pasangan 1,8B, 90% berisik. Skala data yang terbukti berisik. OpenCLIP (LAION): reproduksi terbuka CLIP pada LAION-400M / 2B, berbagai skala, pos pemeriksaan terbuka. EVA-CLIP: diinisialisasi dari pemodelan gambar bertopeng; tulang punggung yang kuat untuk VLM. DASAR: hibrida CLIP+ALIGN Google. Semua keluarga sama, data dan tuning berbeda.

### Langit-langit zero-shot

Model kelas CLIP membatasi sekitar 76% imageNet zero-shot (CLIP-G, OpenCLIP-G). Beyond memerlukan data yang jauh lebih besar (SigLIP 2 mendapat 80%+) atau perubahan arsitektur (head yang diawasi, lebih banyak parameter). Tolok ukurnya sudah jenuh; nilai sebenarnya adalah ruang embedding yang digunakan oleh VLM hilir.

## Pakai

`code/main.py` mengimplementasikan:1. Mainan encoder ganda (feature gambar berbasis hash, feature karakter teks) sehingga kamu dapat melihat bentuk InfoNCE tanpa numpy.
2. Hilangnya InfoNCE dengan Python murni (stabilitas numerik melalui log-sum-exp).
3. Loss berpasangan sigmoid untuk perbandingan.
4. Rutinitas klasifikasi zero-shot: menghitung kesamaan kosinus terhadap sekumpulan prompt teks, argmax untuk prediksi.

Jalankan dan perhatikan kurva kerugiannya. Angka absolutnya hanyalah mainan; bentuknya cocok dengan apa yang dipancarkan oleh pelatih CLIP asli.

## Kirim

Lesson ini menghasilkan `outputs/skill-clip-zero-shot.md`. Mengingat serangkaian gambar (melalui jalur) dan daftar kelas target, ia membuat prompt teks dengan templat CLIP, embed kedua sisi dengan pos pemeriksaan yang dinyatakan (misalnya, `openai/clip-vit-large-patch14`), dan mengembalikan prediksi 1 teratas / 5 teratas dengan skor kesamaan. Keterampilan menolak untuk membuat klaim tentang kelas yang tidak ada dalam daftar cepat.

## Latihan

1. Implementasikan InfoNCE untuk batch 4 pasang dengan tangan. Buat matrix kesamaan 4x4, jalankan softmax, pilih diagonal, hitung entropi silang. Verifikasi implementasi Python kamu terhadap perhitungan tangan ini.

2. SigLIP menggunakan parameter bias `b` selain suhu: `S'[i,j] = S[i,j]/tau + b`. Peran apa yang dimainkan `b` ketika kumpulan memiliki ketidakseimbangan kelas yang besar (lebih banyak negatif daripada positif per baris)? Baca SigLIP Bagian 3 (arXiv:2303.15343).

3. Buat pengklasifikasi zero-shot untuk kucing vs anjing. Coba dua templat cepat: `a photo of a {class}` dan `a picture of a {class}`. Ukur akurasi pada 100 gambar uji. Apakah ansambel templat mengalahkan yang tunggal?

4. Hitung biaya komunikasi softmax InfoNCE vs sigmoid berpasangan untuk 512-GPU yang dijalankan pada batch 32k. Skala mana yang O(N), mana yang O(N^2)? Kutip SigLIP Bagian 4.

5. Baca makalah hukum penskalaan OpenCLIP (arXiv:2212.07143, Cherti dkk.). Reproduksi kesimpulan mereka untuk penskalaan data dari gambar: pada ukuran model tetap, apa hubungan log-linear antara akurasi zero-shot ImageNet dan ukuran training data?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| InfoNCE | "Loss kontrastif" | Entropi silang pada matrix kesamaan suatu batch; positif setiap item adalah item berpasangannya, negatif adalah yang lainnya |
| Loss sigmoid | "Kehilangan SigLIP" | Entropi silang biner per pasangan; tidak ada softmax, tidak ada all-gather, skala murah dalam training terdistribusi |
| Suhu | "tau" | Scalar yang menskalakan logit sebelum softmax/sigmoid; mengontrol ketajaman distribusi |
| Tembakan nol | "klasifikasi tanpa penyempurnaan" | Gunakan petunjuk teks untuk membuat embedding kelas dan mengklasifikasikannya berdasarkan kesamaan kosinus; tidak ada training tentang kelas sasaran |
| Templat cepat | "foto dari..." | Perancah teks di sekitar nama kelas; mempengaruhi akurasi zero-shot sebesar 1-5 poin |
| Pembuat enkode ganda | "Dua menara" | Satu encoder gambar + satu encoder teks, output dalam ruang D-redup bersama |
| Negatif keras | "Pengalih attention yang tangguh" | Negatif cukup mirip dengan positif sehingga model harus berupaya memisahkannya |
| Pemeriksaan linier | "Beku + satu lapis" | Latih hanya pengklasifikasi linier di atas feature yang dibekukan; mengukur kualitas feature |
| NaFlex | "Resolusi fleksibel asli" | Kemampuan SigLIP 2 untuk menyerap gambar pada rasio aspek dan resolusi apa pun tanpa mengubah ukuran |
| Skala suhu | "tau berparametri log" | CLIP memparametrikan `log(1/tau)` sehingga gradient berperilaku; klip untuk mencegah keruntuhan mendekati nol tau |

## Bacaan Lanjutan- [Radford dkk. — Mempelajari Model Visual yang Dapat Dipindahtangankan Dari Pengawasan Bahasa Alami (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — makalah CLIP.
- [Zhai dkk. — Kehilangan Sigmoid untuk Pra-Training Citra Bahasa (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP.
- [Tschannen dkk. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — multibahasa + NaFlex.
- [Jia dkk. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — menskalakan dengan data web yang berisik.
- [Cheti dkk. — Hukum penskalaan yang dapat direproduksi untuk pembelajaran gambar-bahasa yang kontrastif (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — Hukum penskalaan OpenCLIP.
