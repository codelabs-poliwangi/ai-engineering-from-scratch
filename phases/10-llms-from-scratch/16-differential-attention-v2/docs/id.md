# Attention Diferensial (V2)

> Attention Softmax menyebarkan sejumlah kecil kemungkinan ke setiap token yang tidak cocok. Lebih dari 100 ribu token yang menambah kebisingan dan menenggelamkan sinyal. Transformer Diferensial (Ye et al., ICLR 2025) memperbaikinya dengan menghitung attention sebagai perbedaan dua softmax, mengurangi noise floor bersama. DIFF V2 (Microsoft, Januari 2026) adalah penulisan ulang tumpukan produksi: mencocokkan latensi dekode dengan Transformer dasar, tanpa kernel khusus, kompatibel dengan FlashAttention. Lesson ini adalah V1 hingga V2 end-to-end, dengan implementasi mainan yang berfungsi dari operasi perbedaan yang dapat kamu jalankan di stdlib Python.

**Type:** Build
**Language:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (attention diri), Phase 7 · 15 (varian attention), Phase 10 · 14 (panduan arsitektur)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Nyatakan dengan tepat mengapa attention softmax memiliki tingkat kebisingan dan mengapa attention tersebut berkembang seiring dengan panjangnya konteks.
- Turunkan rumus attention diferensial dan jelaskan mengapa pengurangan membatalkan komponen kebisingan bersama sambil mempertahankan sinyal.
- Telusuri perbedaan V1-ke-V2: apa yang menjadi lebih cepat, apa yang menjadi lebih sederhana, apa yang menjadi lebih stabil, dan mengapa setiap perubahan diperlukan untuk pra-training produksi.
- Menerapkan attention diferensial dari awal dengan Python murni dan memverifikasi secara empiris properti peredam bising pada kueri sinyal-plus-kebisingan sintetis.

## Masalah

Attention softmax standar memiliki properti matematika yang berubah menjadi sakit kepala operasional dalam skala besar. Untuk kueri `q`, weight attention-nya adalah `softmax(qK^T / sqrt(d))`. Softmax tidak pernah bisa menghasilkan angka nol yang tepat — setiap token yang tidak cocok mendapat massa positif. Massa sisa tersebut adalah kebisingan, dan skalanya bergantung pada panjang konteks. Dengan 128 ribu token, meskipun setiap token yang tidak cocok hanya mendapat 0,001% probabilitas, 127.999 di antaranya jika digabungkan berkontribusi sekitar 12% dari total. Model harus belajar mengatasi dasar kebisingan yang berkembang seiring dengan konteks.

Secara empiris hal ini muncul sebagai gangguan attention: kutipan berhalusinasi dalam RAG konteks panjang, kegagalan yang hilang di tengah-tengah pada tugas pengambilan 100 ribu token, dan penurunan akurasi yang halus pada tolok ukur needle-in-haystack yang melewati 32 ribu. Makalah Transformer Diferensial (arXiv:2410.05258, ICLR 2025) mengukur kesenjangan tersebut: Transformer DIFF mencapai tingkat perplexity yang lebih rendah, akurasi konteks panjang yang lebih tinggi, dan halusinasi yang lebih sedikit dibandingkan garis dasar berukuran sama.

DIFF V1 memiliki tiga masalah yang membuatnya tidak masuk dalam jalur pipa pra-training. Cache nilainya harus dimuat dua kali per langkah dekode, memerlukan kernel CUDA khusus yang merusak kompatibilitas FlashAttention, dan RMSNorm per kepala mengganggu kestabilan training jangka panjang pada skala 70 miliar lebih. DIFF V2 (Blog Microsoft unilm, 20 Januari 2026) memperbaiki ketiganya. Lesson ini memandu kedua versi, membuat operator pembeda, dan menentukan tolok ukur peredam bising pada kueri mainan.

## Konsep

### Lantai kebisingan softmax

Untuk kueri `q` dan kunci `K = [k_1, ..., k_N]`, weight attention-nya adalah:

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

Tidak `w_i` selalu nol. Jika `k_i` sama sekali tidak terkait dengan `q`, skor `q . k_i` bukan 0 — skor berfluktuasi di sekitar nol dengan varian `||q||^2 / d`. Setelah normalisasi softmax, setiap token yang tidak terkait masih berkontribusi `O(1/N)` ke jumlah tertimbang. Total kontribusi token yang tidak terkait adalah `O((N-1)/N) = O(1)` — bukan jumlah yang kecil.Apa yang diinginkan model adalah sesuatu seperti hard top-k: weight tinggi pada token yang cocok, weight mendekati nol di tempat lain. Softmax terlalu halus untuk melakukannya secara langsung.

### Ide yang berbeda

Bagi proyeksi Q dan K masing-masing kepala menjadi dua: Q = (Q_1, Q_2) dan K = (K_1, K_2). Hitung dua peta attention:

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

Output:

```
DiffAttn = (A_1 - lambda * A_2) V
```

Pengurangan tersebut membatalkan distribusi kebisingan apa pun yang dimiliki kedua peta. Jika kedua peta memiliki weight yang kira-kira seragam pada 127 ribu token yang tidak terkait (yang akan terjadi, pada inisialisasi acak), peta tersebut akan dibatalkan. Sinyal - yang mencapai weight puncak pada beberapa token yang benar-benar relevan - hanya dibatalkan jika muncul di kedua peta dengan magnitudo yang sama, yang tidak akan terjadi setelah model dilatih.

`lambda` adalah scalar per kepala yang dapat dipelajari, diparameterisasi sebagai `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`. Ini bisa menjadi negatif. `lambda_init` defaultnya adalah angka positif kecil seperti 0,8.

### Mengapa ini cocok dengan peredam bising

Bayangkan dua mikrofon berisik yang merekam suara yang sama. Keduanya menangkap speaker ditambah kebisingan latar belakang yang berkorelasi. Kurangi satu dari yang lain dan kebisingan yang dibagikan akan hilang. Suara tersebut bertahan karena kedua sinyal memiliki perbedaan fase atau amplitudo yang cukup untuk mencegah pembatalan penuh. Per-head `lambda` mempelajari keseimbangan ini dengan tepat.

### V1 vs V2: perbedaannya

V1 menjaga jumlah parameter tetap sama dengan Transformer dasar. Untuk mendapatkan dua kueri per kepala, dimension kepala dibelah dua. Hal ini memerlukan ekspresi kepala dan — yang lebih menyakitkan lagi — mengurangi separuh nilai cache per kepala. Dekode harus memuat cache nilai dua kali per langkah (sekali per cabang softmax). Hasil: decode lebih lambat dari baseline meskipun jumlah parameter cocok.

V2 menggandakan jumlah kepala kueri dan menjaga kepala KV tetap sama (meminjam parameter dari proyeksi ke atas). Dimension kepala tetap sama dengan garis dasar. Setelah pengurangan, dimension ekstra diproyeksikan kembali ke bawah agar sesuai dengan proyeksi O_W dasar Transformer. Tiga hal terjadi sekaligus:

1. Kecepatan dekode sesuai dengan kecepatan dasar (cache KV dimuat satu kali).
2. FlashAttention berjalan tidak berubah (tidak ada kernel khusus).
3. Intensitas aritmatika saat dekode meningkat (lebih banyak komputasi per byte yang dimuat dari HBM).

V2 juga menghapus RMSNorm per kepala yang digunakan V1 untuk menstabilkan pengurangan. Pada skala pra-training kelas 70B, RMSNorm tersebut mengganggu kestabilan training yang terlambat. V2 menggantikannya dengan skema inisialisasi sederhana yang menjaga training tetap stabil tanpa modul tambahan.

### Kapan harus meraihnya

| Weight Kerja | Manfaat |
|----------|---------|
| RAG konteks panjang (64k+) | Peta attention yang lebih bersih, lebih sedikit kutipan halusinasi |
| Tolok ukur yang sulit | Akurasi substansial meningkat melewati 32k |
| QA multi-dokumen | Lebih sedikit gangguan lintas dokumen |
| Penyelesaian code pada 8k | Marginal, tidak sebanding dengan perubahan arsitektur |
| Obrolan singkat (< 4k) | Pada dasarnya tidak dapat dibedakan dari baseline |

Nilainya bertambah seiring dengan panjangnya konteks. Pada token 4k, tingkat kebisingannya cukup kecil sehingga attention standar baik-baik saja. Pada 128k itu merugikan kamu.

### Bagaimana cara menumpuknya dengan kenop 2026 lainnya| Feature | Kompatibel dengan DIFF V2? |
|---------|------------------------|
| GQA | Ya (V2 meningkatkan kepala Q, bukan kepala KV) |
| MLA (Pencarian Mendalam) | Ya pada prinsipnya, tidak ada makalah terbitan yang menggabungkan keduanya |
| Kementerian Lingkungan Hidup | Ya (attention tidak bergantung pada blok MLP) |
| Tali | Ya (tidak berubah) |
| YaRN / penskalaan konteks panjang | Ya (persis dimana DIFF paling membantu) |
| FlashPerhatian | Ya di V2 (tidak ada di V1) |
| Penguraian code spekulatif | Ya (perubahan attention tidak terlihat pada loop decode spesifikasi) |

## Build

`code/main.py` mengimplementasikan attention diferensial dengan Python murni. Kueri mainan dengan struktur sinyal-plus-kebisingan yang diketahui memungkinkan kamu mengukur rasio peredam bising secara langsung.

### Langkah 1: attention softmax standar

Operasi matrix Stdlib: daftar daftar, matmul manual, softmax dengan pengurangan stabilitas numerik maks.

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### Langkah 2: bagi Q, K menjadi dua bagian

Gaya V1: membagi dua dimension kepala. Gaya V2: pertahankan dimension kepala dan gandakan jumlah kepala. Implementasi mainan menggunakan V1 untuk kejelasan pedagogis — matematikanya sama, hanya pembukuannya saja yang berbeda.

### Langkah 3: dua cabang softmax + pengurangan

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

Catatan: weight output bisa negatif. Tidak apa-apa — cache nilai masih menangani kontribusi yang ditandatangani. Proyeksi V berikutnya menyerap tanda tersebut.

### Langkah 4: pengukuran peredam bising

Buatlah rangkaian sintetis dengan panjang 1024. Tempatkan token sinyal pada posisi yang diketahui, isi sisanya dengan noise. Hitung (a) weight attention softmax standar pada posisi sinyal dan (b) weight attention diferensial. Ukur rasio sinyal terhadap kebisingan di masing-masingnya. Attention DIFF secara andal menghasilkan rasio signal-to-noise yang lebih tinggi sebanyak 3x-10x, bergantung pada seberapa besar perbedaan kedua cabang yang telah dilatih.

### Langkah 5: Akuntansi parameter V1 vs V2

Diberikan konfigurasi (tersembunyi=4096, kepala=32, d_head=128), cetak:

- Baseline Transformer: Q, K, V masing-masing ukuran `hidden * hidden`, MLP pada 4 * tersembunyi.
- DIFF V1: Q, K setiap ukuran `hidden * hidden`, ukuran V `hidden * hidden` (tidak berubah), kepala redup dibelah dua secara internal. Menambahkan parameter per kepala `lambda` (O(heads * d_head)).
- DIFF V2: ukuran Q `2 * hidden * hidden`, ukuran K `hidden * hidden`, ukuran V `hidden * hidden`. Ekstra redup diproyeksikan kembali ke bawah sebelum O_W. Menambahkan parameter `lambda` yang sama.

Mainan tersebut mengukur biaya parameter tambahan untuk V2 (kira-kira `hidden * hidden` ekstra per blok attention) dan mencetaknya.

## Pakai

DIFF V2 belum dikirimkan di setiap server inference produksi pada April 2026, namun integrasi sedang berlangsung di vLLM dan SGLang. Sementara itu polanya muncul pada:

- Model produksi konteks panjang internal Microsoft.
- Replikasi penelitian di beberapa training model terbuka yang menargetkan 256 ribu lebih konteks.
- Arsitektur hibrid yang menggabungkan attention DIFF dengan attention jendela geser pada layer alternatif.

Kapan kamu akan mencapai ini pada tahun 2026:

- Melatih model baru dari awal dengan menargetkan 64 ribu lebih konteks efektif. Tambahkan attention yang berbeda dari awal; training ulang nanti mahal.
- Menyempurnakan model konteks panjang di mana kegagalan yang terjadi di tengah-tengah mendominasi evaluasi kamu. LoRA pada proyeksi Q dapat memperkirakan struktur DIFF.

Bila kamu tidak mau:

- kamu menayangkan model padat terlatih dengan performa konteks panjang yang stabil. Biaya training ulang jarang sebanding dengan weight yang ada.
- Konteks kamu selalu di bawah 16k. Lantai kebisingan dapat diabaikan.

## Kirim

Lesson ini menghasilkan `outputs/skill-diff-attention-integrator.md`. Dengan adanya arsitektur model, panjang konteks target, profil halusinasi, dan anggaran training, hal ini menghasilkan rencana integrasi untuk menambahkan attention yang berbeda pada proses pra-training baru atau penyempurnaan LoRA.

## Latihan

1. Jalankan `code/main.py`. Pastikan rasio sinyal terhadap kebisingan yang dilaporkan untuk attention diferensial lebih tinggi daripada attention softmax standar pada kueri sintetis. Variasikan amplitudo kebisingan dan tunjukkan titik persilangan di mana attention standar menjadi tidak dapat digunakan.

2. Hitung jumlah parameter delta dari garis dasar ke DIFF V1 dan dari garis dasar ke DIFF V2 untuk model kelas 7B (tersembunyi=4096, kepala=32, d_head=128, 32 layer). Tunjukkan komponen mana yang memperoleh parameter dan mana yang tetap sama.

3. Baca Bagian 3 makalah DIFF V1 (arXiv:2410.05258) dan Bagian 2 blog DIFF V2 Memeluk Wajah. Dalam dua kalimat, jelaskan mengapa RMSNorm per kepala V1 diperlukan dan mengapa V2 dapat menghapusnya tanpa menyebabkan perbedaan training.

4. Terapkan ablasi: hitung attention diferensial dengan `lambda = 0` (softmax pertama murni) dan `lambda = 1` (pengurangan penuh). Pada kueri sintetis, ukur bagaimana perubahan signal-to-noise di seluruh sapuan. Identifikasi `lambda` yang memaksimalkan signal-to-noise.

5. Perluas mainan ke GQA + DIFF V2. Pilih 8 kepala KV dan 32 kepala Q. Tunjukkan bahwa ukuran cache KV cocok dengan model GQA dasar dengan konfigurasi (8, 32) yang sama.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Perbedaan attention | "Dua softmax saling dikurangi" | Pisahkan Q, K menjadi dua bagian, hitung dua peta softmax, kurangi peta kedua (skala lambda) dari peta pertama, lalu kalikan dengan V |
| Lantai kebisingan | "Ekor softmax yang bukan nol" | Softmax weight O(1/N) diterapkan pada setiap token yang tidak terkait, yang berjumlah O(1) dalam konteks panjang |
| lambda | "Skala pengurangan" | Scalar yang dapat dipelajari per kepala diparameterisasi sebagai `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`; bisa negatif |
| DIFF V1 | "Versi ICLR 2025" | Transformer Diferensial Asli; separuh kepala redup untuk mempertahankan jumlah parameter, memerlukan kernel khusus, dekode lebih lambat |
| DIFF V2 | "Perbaikan Januari 2026" | Menggandakan kepala Q menjaga kepala KV; cocok dengan kecepatan dekode dasar dan berfungsi dengan FlashAttention |
| RMSNorm per kepala | "Penstabil V1" | Norm ekstra V1 diterapkan setelah selisih; V2 menghapusnya untuk mencegah ketidakstabilan training terlambat |
| Rasio sinyal terhadap kebisingan | "Berapa banyak attention yang terbuang" | Rasio weight pada posisi sinyal sebenarnya terhadap weight rata-rata pada posisi yang tidak terkait |
| Tersesat di tengah | "Mode kegagalan konteks panjang" | Fenomena empiris dimana akurasi pengambilan dokumen menurun di tengah konteks yang panjang — attention DIFF mengurangi hal ini |
| Intensitas aritmatika | "FLOP per byte dimuat" | Rasio V2 meningkat saat dekode dengan menggandakan kueri per weight KV; penting untuk dekode terikat memori |

## Bacaan Lanjutan- [Kamu dkk. — Transformer Diferensial (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — makalah asli dengan teori peredam bising dan ablasi konteks panjang
- [Microsoft unilm — Differential Transformer V2 (blog Hugging Face, Januari 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — penulisan ulang tumpukan produksi, dekode dasar yang cocok, kompatibel dengan FlashAttention
- [Memahami Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — analisis teoretis tentang mengapa pengurangan memulihkan struktur attention yang telah dilatih sebelumnya
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — varian berbagi parameter
- [Vaswani dkk. — Hanya Attention yang kamu Butuhkan (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — DIFF Transformer dasar dikurangi dari
- [Liu dkk. — Tersesat di Tengah (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — tolok ukur konteks panjang target attention DIFF
