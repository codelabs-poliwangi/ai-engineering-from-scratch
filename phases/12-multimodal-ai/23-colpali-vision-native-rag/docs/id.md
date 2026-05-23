# ColPali dan RAG Dokumen Vision-Native

> RAG tradisional mem-parsing PDF menjadi teks, membaginya menjadi beberapa bagian, embed bagian tersebut, menyimpan vector. Setiap langkah kehilangan sinyal: OCR menghapus data grafik, pemotongan baris tabel rusak, embedding teks mengabaikan angka. ColPali (Faysse dkk., Juli 2024) menanyakan pertanyaan sederhana: mengapa mengekstrak teks? Sematkan gambar halaman secara langsung melalui PaliGemma, gunakan interaksi akhir gaya ColBERT untuk pengambilan, dan pertahankan semua tata letak, gambar, font, dan sinyal pemformatan yang dibawa dokumen. Tolok ukur yang dipublikasikan: akurasi menyeluruh 20-40% lebih baik dibandingkan text-RAG pada dokumen yang kaya visual. ColQwen2, ColSmol, dan VisRAG memperluas polanya. Lesson ini membaca tesis RAG vision-native dan membuat pengindeks kecil mirip ColPali.

**Type:** Build
**Language:** Python (stdlib, pengindeks multi-vector + pencetak gol MaxSim)
**Prerequisites:** Fase 11 (Teknik LLM — dasar-dasar RAG), Fase 12 · 05 (LLaVA)
**Waktu:** ~180 menit

## Tujuan Pembelajaran

- Jelaskan perbedaan antara pengambilan bi-encoder (satu vector per dokumen) dan pengambilan interaksi akhir (banyak vector per dokumen).
- Jelaskan operasi MaxSim ColBERT dan bagaimana ColPali menggeneralisasikannya dari token teks ke patch gambar.
- Buat pengindeks kecil seperti ColPali: halaman → embedding patch → MaxSim melalui embedding istilah kueri → halaman k teratas.
- Bandingkan generator ColPali + Qwen2.5-VL vs text-RAG + GPT-4 pada kasus penggunaan faktur/laporan keuangan.

## Masalah

Text-RAG pada PDF membuang sebagian besar dokumen. Pertumbuhan pendapatan Q3 laporan keuangan biasanya disajikan dalam bentuk grafik; temuan laporan medis disajikan dalam gambar beranotasi; blok tanda tangan kontrak hukum adalah fakta tata letak, bukan fakta teks.

Alur teks-RAG:

1. PDF → teks melalui OCR / pdftotext.
2. Teks → 300-500 potongan token.
3. Potongan → embedding bi-encoder (satu vector).
4. Permintaan pengguna → embedding → kesamaan kosinus → potongan top-k.
5. Potongan + kueri → LLM.

Lima langkah yang merugikan. Bagan tidak ditangkap. Tabel dipecah menjadi beberapa bagian. Tata letak multi-kolom menjadi rata. Anotasi gambar hilang.

Perbaikan ColPali: lewati OCR, sematkan gambar halaman secara langsung. Gunakan interaksi terlambat gaya ColBERT untuk pengambilan sehingga model dapat menangani patch terperinci pada waktu kueri.

## Konsep

### ColBERT (2020)

ColBERT (Khattab & Zaharia, arXiv:2004.12832) adalah metode pengambilan teks. Alih-alih satu vector per dokumen, ini menghasilkan satu vector per token. Pada waktu kueri:

- Token kueri mendapatkan embedding-nya sendiri (N_q vector).
- Token dokumen mendapatkan embedding (N_d vector, biasanya di-cache).
- Skor = jumlah token kueri max atas token dokumen kesamaan kosinus: Σ_i max_j cos(q_i, d_j).

Ini adalah operasi MaxSim. Setiap token kueri "mengambil" token dokumen yang paling cocok. Skor akhir adalah jumlah.

Kelebihan: daya ingat yang kuat, menangani semantik tingkat istilah. Kekurangan: N_d vector per dokumen, penyimpanan mahal.

### KolPali

ColPali (Faysse et al., arXiv:2407.01449) menerapkan pola ColBERT pada gambar.

- Setiap halaman dikodekan oleh PaliGemma (bahasa ViT +) ke dalam patch embeddings: N_p vector per halaman.
- Setiap kueri pengguna (teks) dikodekan ke dalam embedding token kueri: N_q vector.
- Skor = Σ_i max_j cos(q_i, p_j), yaitu MaxSim melalui token teks kueri dan tambalan gambar halaman.
- Ambil halaman top-k dengan skor total.

Pada waktu penyerapan dokumen: sematkan setiap halaman dengan PaliGemma, simpan semua embedding patch. Pada waktu kueri: sematkan token kueri, hitung MaxSim terhadap semua embedding halaman yang disimpan, kembalikan k halaman teratas.Kelebihan: end-to-end mengalahkan text-RAG sebesar 20-40% pada dokumen yang kaya visual. Setiap vector patch menangkap tata letak dan konten lokal.

Kekurangan: N_p patch × float 4-byte × D-dim vector per halaman = penyimpanan bertambah dengan cepat. Dimitigasi dengan kuantisasi PQ / OPQ.

### ColQwen2 dan ColSmol

ColQwen2 (illuin-tech, 2024-2025) menukar PaliGemma dengan Qwen2-VL. Encoder dasar yang lebih baik, pengambilan yang lebih baik.

ColSmol adalah varian skala lebih kecil untuk penggunaan lokal/edge. Retriever ColSmol dengan parameter ~1 miliar berjalan pada GPU konsumen.

### VisRAG

VisRAG (Yu et al., arXiv:2410.10594) adalah varian yang berbeda: alih-alih MaxSim pada patch, kumpulkan setiap halaman ke dalam satu vector dengan VLM lalu ambil bi-encoder. Pengindeksan lebih cepat + penyimpanan lebih kecil, penarikan kembali lebih lemah.

Pertukaran kualitas vs biaya: ColPali untuk kualitas, VisRAG untuk skala.

### M3DocRAG

M3DocRAG (Cho et al., arXiv:2411.04952) memperluas pengambilan multi-modal ke penalaran multi-dokumen multi-halaman. Mengambil halaman di seluruh dokumen, menyusun konteks multi-halaman untuk VLM.

### ViDoRe — patokan

Patokan pendamping ColPali. Evaluasi Pengambilan Dokumen Visual. Tugasnya meliputi laporan keuangan, karya ilmiah, dokumen administrasi, rekam medis, manual. Metrik: nDCG@5.

Skor ColPali-v1 ~80% nDCG@5 di ViDoRe; text-RAG pada dokumen yang sama mendapat skor ~50-60%.

### Pipeline pipa RAG ujung ke ujung

Untuk RAG vision-asli:

1. Serap: PDF → gambar halaman → Pengkodean PaliGemma → simpan semua embedding patch.
2. Kueri: teks pengguna → embedding token kueri → MaxSim terhadap semua halaman yang diindeks → halaman k teratas.
3. Hasilkan: gambar halaman k teratas + kueri → VLM (Qwen2.5-VL atau Claude) → jawaban.

Tidak ada OCR di mana pun. Gambar, bagan, font, tata letak semuanya mengalir ke dalam jawabannya.

### Matematika penyimpanan

Laporan keuangan 50 halaman dengan 729 tambalan per halaman dan embedding 128 redup:

- ColPali: 50 * 729 * 128 * 4 byte = ~18 MB mentah, ~4 MB setelah PQ.
- Teks-RAG: 50 potongan * 768-dim * 4 byte = ~150 kB.

ColPali memiliki ~30x lebih banyak penyimpanan per dokumen. Dalam skala besar, OPQ / PQ menurunkannya menjadi ~5-10x, biasanya dapat ditoleransi.

### Saat text-RAG masih menang

- Dokumen teks murni tanpa sinyal tata letak (artikel wiki, log obrolan). Text-RAG lebih sederhana dan lebih murah penyimpanannya.
- Arsip multi-juta halaman yang biaya penyimpanannya mendominasi.
- Persyaratan peraturan yang ketat menuntut teks OCR yang dapat diekstraksi bersamaan dengan pengambilan.

Untuk semua hal lainnya di tahun 2026 — laporan keuangan, makalah ilmiah, kontrak hukum, catatan medis, dokumentasi UX — RAG yang berbasis visi menang.

## Pakai

`code/main.py`:

- Pembuat enkode tambalan mainan: memetakan "halaman" (kisi kecil vector feature) ke serangkaian embedding tambalan.
- Pencetak gol MaxSim: menghitung skor gaya ColBERT antara set embedding token kueri dan set patch halaman.
- Mengindeks 5 halaman mainan, menjalankan 3 kueri, mengembalikan k teratas dengan skor.

## Kirim

Lesson ini menghasilkan `outputs/skill-vision-rag-designer.md`. Diberikan proyek dokumen-RAG, pilih ColPali / ColQwen2 / VisRAG / text-RAG dan ukur penyimpanannya.

## Latihan

1. Laporan tahunan 200 halaman dengan 729 patch per halaman, emb 128-dim, float 4-byte. Hitung penyimpanan mentah dan penyimpanan terkompresi PQ (8x).

2. MaxSim adalah Σ_i max_j cos(q_i, p_j). Apa yang ditangkap oleh penjumlahan ini yang tidak dimiliki oleh kesamaan rata-rata sederhana?

3. ColPali mengindeks halaman sebagai kumpulan tambalan. Apa yang berubah jika kita mengindeks pada tingkat kata (seperti yang dilakukan ColBERT)? Pengorbanan?

4. Rancang alur end-to-end untuk korpus 1 juta halaman dengan anggaran latensi 500 md per kueri. Pilih ColQwen2 / VisRAG dan justifikasi.5. Baca M3DocRAG (arXiv:2411.04952). Jelaskan pola attention multi-halaman dan perbedaannya dengan pengambilan ColPali satu halaman.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Interaksi terlambat | "Gaya ColBERT" | Pengambilan menggunakan embeddings per token atau per patch + MaxSim, bukan vector dokumen tunggal |
| MaxSim | "Max-over-patch" | Untuk setiap token kueri, pilih token dokumen dengan kemiripan tertinggi; jumlah seluruh kueri |
| Bi-encoder | "Vector tunggal" | Satu vector per dokumen; lebih cepat tetapi kehilangan granularitas |
| Multi-vector | "Banyak-vector-per-dokumen" | Menyimpan N_p vector per dokumen/halaman; biaya penyimpanan bertambah tetapi perolehan kembali meningkat |
| Embedding tambalan | "Feature halaman" | Satu vector per patch gambar dari encoder VLM, di-cache per halaman |
| ViDoRe | "Bangku dokumen visi" | Paket benchmark ColPali untuk pengambilan dokumen visual |
| kuantisasi PQ | "Kuantisasi produk" | Kompresi yang mempertahankan kesamaan vector sekaligus menyusutkan penyimpanan ~8x |

## Bacaan Lanjutan

- [Faysse dkk. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu dkk. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho dkk. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
