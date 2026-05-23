# Hukum Penskalaan

> Makalah Kaplan tahun 2020 mengatakan: model yang lebih besar, loss yang lebih rendah. Makalah Hoffmann tahun 2022 mengatakan: kamu sedang dalam training. Komputasi dibagi menjadi dua kelompok — parameter dan token — dan pemisahannya tidak terlihat jelas.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 7 · 05 (Trafo Penuh), Fase 7 · 07 (GPT)
**Waktu:** ~45 menit

## Masalah

Saat kamu memiliki komputasi training C FLOP dan menginginkan model terbaik, kamu menghadapi dua masalah:

1. **Berapa banyak parameter (N)?** Model lebih besar, kapasitas lebih tinggi.
2. **Berapa banyak token training (D)?** Lebih banyak data, penggunaan kapasitas lebih baik.

Skala FLOP kira-kira sebesar `6 × N × D`. kamu dapat menekan N ke atas dan D ke bawah, atau D ke atas dan N ke bawah. Mana yang lebih baik?

Sebelum tahun 2022, jawabannya adalah "push N dengan keras". GPT-3 (2020) adalah 175 miliar parameter yang dilatih pada ~300 miliar token. Rasio sekitar 1,7 token per parameter. Undang-undang penskalaan Kaplan mendukung hal ini.

Hoffmann dkk. (2022), yang melatih sekelompok kecil model bernama Chinchilla, menemukan sesuatu yang berbeda: rasio optimal mendekati **20 token per parameter**. GPT-3 10x kurang terlatih. Chinchilla (param 70 miliar, token 1,4 ton) mengalahkan GPT-3 (token 175 miliar, 300 miliar) di setiap benchmark dengan biaya inference 2,5× lebih murah.

Tahun 2026 adalah dunianya Chinchilla — dengan satu perubahan penting. Llama 3 8B dilatih pada 15 triliun token, rasio 1,875 token per parameter. Sembilan puluh empat kali melewati optimal Chinchilla. Biaya inference lebih penting daripada biaya training untuk model yang akan digunakan dalam skala besar, sehingga training berlebihan (di luar Chinchilla) untuk ukuran penerapan yang lebih kecil adalah standar pada tahun 2026.

## Konsep

![Kurva Chinchilla: loss vs komputasi pada berbagai rasio N/D](../assets/scaling-laws.svg)

### Hukum Hoffmann

Dari makalah Chinchilla, kerugiannya sebagai berikut:

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = parameter (tidak tertanam).
- `D` = token training.
- `α ≈ 0.34`, `β ≈ 0.28` (kira-kira simetris).
- `E ≈ 1.69`, batas loss yang tidak dapat direduksi.
- `A ≈ 406`, `B ≈ 411`.

Dua istilah saling diperdagangkan satu sama lain saat kamu menskalakan. Ambil turunan w.r.t. `N` pada komputasi tetap (C = 6ND) dan selesaikan:

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

Komputasi optimal: 20 token per parameter.

### Kenapa latihan berlebihan

Chinchilla-optimal meminimalkan loss training per training FLOP. Tapi kamu membayar biaya training satu kali; biaya inference selamanya.

Untuk chatbot yang melayani satu triliun token per bulan, inference mendominasi total biaya. Pendekatan Llama: berlatih lebih kecil, lebih lama. 8B dengan 15T token sangat dioptimalkan untuk inference:

- Cocok untuk GPU konsumen.
- Latensi adalah sebagian kecil dari optimal 70B Chinchilla.
- Kualitas cukup dekat untuk sebagian besar tugas.

Makalah DeepMind tahun 2024 (“Training berlebihan adalah optimal baru”) meresmikan hal ini. Untuk weight kerja yang didominasi inference, rasio yang tepat mendekati 100–500 token per parameter bergantung pada volume penayangan.

### Kemunculan vs kehalusan

Klaim: kemampuan tertentu (aritmatika, penalaran multi-langkah, mengikuti rangkaian pemikiran) "muncul" secara tiba-tiba pada skala tertentu.

Schaeffer dkk. (2023) berpendapat bahwa ini adalah artefak pengukuran: metrik yang muncul menggunakan penilaian terputus-putus (pencocokan tepat, akurasi pada ambang batas) yang menyembunyikan peningkatan mulus dalam logit yang mendasarinya. Metrik kontinu (entropi silang) menunjukkan kurva mulus.

Pada tahun 2026, konsensusnya adalah: prediksi melalui loss berkelanjutan dapat diandalkan. Lompatan patokan sering kali merupakan artefak pencetak gol. Rencanakan anggaran berdasarkan metrik berkelanjutan.

### Gambar tahun 2026

Undang-undang penskalaan masih berlaku, tetapi:| Faktor | Mengubah cara |
|--------|-------------|
| Kualitas data | Mengkurasi token "baik" (gaya Phi) menggeser kurva sebesar >2× komputasi efektif |
| Kementerian Lingkungan Hidup | Total param dipisahkan dari FLOP aktif; hukum penskalaan per-aktif-FLOP |
| Pasca training | Beberapa kemampuan (mengikuti instruksi, code) bergeser dengan SFT+RLHF lebih dari prapelatihan |
| Multimodalitas | Token gambar + teks berskala bersama; kurva terpisah per modalitas |
| Data sintetis | Model menghasilkan training data; komputasi yang efektif dapat digabungkan |

Optimizer Muon (Kimi Moonlight, 2024) menunjukkan ~2× perolehan komputasi efektif dibandingkan AdamW pada data yang cocok. Beberapa training yang dijalankan pada tahun 2026 menggunakan Muon secara default. Mengubah konstanta absolut dalam hukum penskalaan, bukan bentuknya.

## Build

Lihat `code/main.py`. Kami menerapkan persamaan loss Chinchilla dan memecahkan masalah komputasi optimal `(N, D)` di setiap anggaran komputasi.

### Langkah 1: Kehilangan Chinchilla

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

Plot `L` sebagai kontur di atas `(N, D)` pada `C = 6ND` tetap. Temukan minimumnya.

### Langkah 2: batas komputasi optimal

Untuk menghitung anggaran dari `1e17` hingga `1e25` FLOP, temukan `(N, D)` yang meminimalkan loss sesuai dengan `6ND = C`. Verifikasi rasio `D/N ≈ 20`.

### Langkah 3: biaya training berlebihan

Hitung loss ekstra yang kamu bayarkan untuk melatih model 10× lebih kecil (1/10 N optimal, 10× D optimal). Melaporkan inference penghematan FLOP (sebanding dengan N) sebagai gantinya.

### Langkah 4: bandingkan dengan model sebenarnya

Masukkan pasangan `(N, D)` yang diketahui untuk GPT-3, Chinchilla, Llama 3 8B, DeepSeek-V3 (param aktif), dan bandingkan perkiraan loss dengan loss yang dilaporkan.

## Pakai

kamu tidak mungkin melatih model perbatasan sendiri. Namun undang-undang penskalaan memberi tahu kamu:

1. **Apakah penyempurnaan kamu memiliki cukup data.** Jika data spesifik tugas kamu di bawah 20 token per parameter model dasar, kemungkinan akan terjadi saturasi pada tingkat loss tertentu.
2. **Apakah akan memilih model dasar yang lebih besar.** Jika kamu menghabiskan seluruh anggaran untuk inference, pilihlah model yang lebih kecil dan lebih terlatih.
3. **Saat keuntungan berkurang.** Melampaui 1000× optimal Chinchilla, perubahan kehilangan log menjadi gangguan.

**Jalur penelitian pada tahun 2026:**

- **Rezim dengan keterbatasan data.** Web memiliki jumlah token berkualitas tinggi yang terbatas (~5–10 triliun bahasa Inggris setelah difilter). Pra-training perbatasan mendekati batas atas ini. Data sintetis, multibahasa, multimodal, dan penyesuaian skala RLHF adalah pendorong berikutnya.
- **Trik pengganda komputasi.** Optimizer muon, MoE, kurasi data yang lebih baik — masing-masing menggeser konstanta absolut, bukan asimtot.
- **Menskalakan undang-undang untuk RL.** Pertanyaan terbuka. Bukti awal menunjukkan hukum kekuasaan dalam sample RL tetapi dengan eksponen yang sangat berbeda dibandingkan pra-training.

## Kirim

Lihat `outputs/skill-training-budget-estimator.md`. Keahlian ini memilih `(N, D, hours, GPU)` untuk menjalankan training baru dengan mempertimbangkan anggaran komputasi, batasan penerapan, dan kehilangan target.

## Latihan1. **Mudah.** Jalankan `code/main.py`. Cetak Chinchilla optimal `(N, D)` untuk anggaran komputasi `1e20`, `1e22`, `1e24`. Bandingkan dengan tabel model sebenarnya.
2. **Sedang.** Menerapkan kurva loss sebagai fungsi komputasi Hoffmann. Kehilangan plot vs `log10(C)` untuk batas optimal komputasi. Identifikasi kapan undang-undang memperkirakan kita memerlukan `>10^28` FLOP untuk pengurangan entropi silang sebesar 0,1 berikutnya.
3. **Sulit.** Sesuaikan hukum penskalaan kamu pada 5 model kecil (100 ribu hingga 10 juta parameter) yang dilatih pada dataset yang sama. Perkirakan `α` dan `E`. Seberapa cocok eksponen kamu dengan eksponen yang dipublikasikan?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Parameter (N) | "Ukuran model" | Jumlah berat yang tidak tertanam; menentukan kapasitas. |
| Token (D) | "Training data" | Jumlah token training yang terlihat; menentukan seberapa baik parameter digunakan. |
| Hitung (C) | "FLOP dihabiskan" | Kira-kira `6 × N × D` untuk trafo standar. |
| Chinchilla-optimal | "H/T ≈ 20" | Rasio yang meminimalkan loss per FLOP prapelatihan. |
| Training berlebihan | "Chinchilla Masa Lalu" | Habiskan FLOP training ekstra untuk menyimpan FLOP inference; H/T >> 20. |
| Loss yang tidak dapat direduksi | "Lantai" | Istilah `E` dalam undang-undang penskalaan; entropi data itu sendiri. |
| Kemampuan yang muncul | "Tiba-tiba melompati skala" | Seringkali merupakan artefak pencetak gol; loss terus menerus itu mulus. |
| Komputasi efektif | "Pengganda efisiensi training" | Data/optimizer/arsitektur yang lebih baik melipatgandakan sejauh mana FLOP berjalan. |

## Bacaan Lanjutan

- [Kaplan dkk. (2020). Hukum Penskalaan untuk Model Bahasa Neural](https://arxiv.org/abs/2001.08361) — makalah hukum penskalaan pertama; kurang terlatih.
- [Hoffmann dkk. (2022). Training Large Language Model yang Optimal Komputasi](https://arxiv.org/abs/2203.15556) — Chinchilla.
- [Schaeffer dkk. (2023). Apakah Kemampuan Muncul Large Language Model Sebuah Fatamorgana?](https://arxiv.org/abs/2304.15004) — kemunculan sebagai artefak pengukuran.
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Akuntansi untuk Inference dalam Hukum Penskalaan Model Bahasa](https://arxiv.org/abs/2401.00448) — mengapa training berlebihan Llama tepat untuk weight kerjanya.
- [Yordania dkk. (2024). Muon: Optimizer untuk layer tersembunyi di jaringan neural](https://kellerjordan.github.io/posts/muon/) — pengganda komputasi 2×.
