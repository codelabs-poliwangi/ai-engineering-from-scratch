# Paralelisme Pipa Ganda

> DeepSeek-V3 dilatih pada 2.048 GPU H800 dengan pakar MoE yang tersebar di seluruh node. Komunikasi ahli lintas node memerlukan biaya komunikasi 1 jam GPU untuk setiap 1 jam komputasi GPU. GPU menganggur separuh waktu. DualPipe (DeepSeek, Des 2024) adalah pipeline dua arah yang tumpang tindih dengan komputasi maju dan mundur dengan komunikasi semua-ke-semua yang dipicunya. Penurunan gelembung, peningkatan throughput, dan penyimpanan dua salinan parameter model ("ganda" yang memberi nama) menjadi murah karena Paralelisme Pakar sudah menyebarkan pakar ke seluruh peringkat. Lesson ini adalah panduan tipe Learn tentang apa yang sebenarnya dilakukan DualPipe dan mengapa penyempurnaan DualPipeV di Sea AI Lab menurunkan biaya parameter 2x dengan mengorbankan gelembung yang sedikit lebih ketat.

**Type:** Learn
**Language:** Python (stdlib, simulator jadwal)
**Prerequisites:** Fase 10 · 05 (training terdistribusi, FSDP, DeepSpeed), Fase 10 · 14 (arsitektur model terbuka dan MoE)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Sebutkan empat komponen potongan maju-mundur DualPipe dan mengapa masing-masing komponen memiliki jendela tumpang tindihnya sendiri.
- Jelaskan masalah gelembung pipeline pipa dalam skala besar, dan apa arti "bebas gelembung" dalam praktik versus pemasaran.
- Lacak jadwal DualPipe dengan tangan untuk 8 peringkat PP dan 16 batch mikro dan konfirmasikan aliran maju dan mundur saling mengisi slot menganggur.
- Nyatakan trade-off yang dilakukan DualPipeV (Sea AI Lab, 2025): menghilangkan replikasi parameter 2x dengan mengorbankan gelembung yang sedikit lebih besar saat Paralelisme Pakar tidak aktif.

## Masalah

Melatih model MoE 671B pada GPU 2k H800 mengalami tiga hambatan yang semakin besar:

1. **Tekanan memori.** Setiap GPU menampung sebagian model. Memori activation pada urutan 8k di 61 layer pada 128 kepala sangatlah besar.
2. **Gelembung alur.** Paralelisme alur tradisional (GPipe, 1F1B) membuat GPU menganggur saat menunggu input atau gradient tahapannya. Pada 8 phase, sekitar 12% waktu GPU dapat digelembungkan bahkan dengan penjadwalan 1F1B.
3. **Cross-node all-to-all.** KLH dengan paralelisme pakar menyebarkan pakar ke seluruh node. Setiap forward pass memicu all-to-all untuk mengirimkan token ke ahlinya, dan token lainnya untuk digabungkan. Pada GPU 2k, ini dengan mudah menjadi rasio komputasi-ke-komunikasi 1:1.

Masing-masing memiliki solusi terpisah: pos pemeriksaan gradient untuk memori, Zero Bubble (Sea AI Lab, 2023) untuk gelembung pipa, kernel komunikasi paralel ahli untuk semua-ke-semua. Apa yang dilakukan DualPipe adalah membuat mereka bermain bersama. Jadwal tersebut tumpang tindih dengan komputasi dan komunikasi dalam satu potongan maju-mundur, memasukkan batch mikro dari kedua ujung pipeline secara bersamaan, dan menggunakan jadwal yang dihasilkan untuk menyembunyikan semua-ke-semua di dalam jendela komputasi.

Hasil yang dilaporkan: gelembung pipeline pipa hampir tereliminasi, penggunaan GPU lebih dari 95% dalam training token 14,8T DeepSeek-V3.

## Konsep

### Penyegaran paralelisme pipeline pipa

Pisahkan model layer-N di seluruh perangkat P. Perangkat `i` menampung layer `i * N/P .. (i+1) * N/P - 1`. Batch mikro mengalir maju melalui perangkat 0 ke P-1, lalu mundur dari P-1 ke 0. Setiap perangkat hanya dapat memulai phase majunya ketika perangkat sebelumnya mengirimkan outputnya dan hanya dapat memulai mundur ketika perangkat hilir mengirimkan gradient hulu.GPipe (Huang et al., 2019) menjadwalkan satu micro-batch dalam satu waktu, yang menghabiskan sebagian besar waktu GPU. 1F1B (Narayanan et al., 2021) menyisipkan lintasan maju dan mundur untuk beberapa mikro-batch. Zero Bubble (Qi dkk., 2023) membagi gerakan mundur menjadi dua bagian — gerakan mundur untuk input (B) dan gerakan mundur untuk weight (W) — dan menjadwalkannya untuk mengisi gelembung. Setelah Zero Bubble, jalur pipa hampir rapat.

DualPipe adalah langkah selanjutnya. Itu menambahkan dua ide di atas:

### Ide 1: decomposition bongkahan

Setiap potongan maju dibagi menjadi empat komponen:

- **Attention.** Proyeksi Q/K/V, attention, proyeksi output.
- **Pengiriman menyeluruh.** Komunikasi lintas node yang mengirimkan token ke pakarnya.
- **MLP.** Perhitungan ahli KLH.
- **Penggabungan semua-ke-semua.** Komunikasi lintas-simpul yang menghadirkan kembali output ahli.

Potongan mundur menambahkan versi gradient dari masing-masingnya. DualPipe menjadwalkannya sehingga pengiriman semua-ke-semua terjadi secara paralel dengan komputasi attention pada potongan berikutnya, dan penggabungan semua-ke-semua terjadi secara paralel dengan komputasi MLP pada potongan berikutnya.

### Ide 2: penjadwalan dua arah

Sebagian besar jadwal pipeline pipa memasukkan batch mikro dari phase 0 dan mengalir menuju phase P-1. DualPipe menyuntikkan batch mikro dari KEDUA ujungnya. Phase 0 melihat kumpulan mikro maju yang berasal dari sana; phase P-1 melihat kumpulan mikro ke depan yang berasal dari sana juga. Kedua aliran itu bertemu di tengah.

Agar ini berfungsi, perangkat `i` harus menampung KEDUA layer pipa awal `i` DAN layer pipa akhir `P - 1 - i`. Itu adalah bagian "ganda" dari DualPipe: setiap perangkat menyimpan dua salinan layer model yang perlu dilayani (satu untuk setiap arah). Pada skala DeepSeek-V3, biaya replikasi parameternya 2x lipat. Hal ini terjangkau karena Paralelisme Pakar sudah menyebarkan tenaga ahli di Kementerian Lingkungan Hidup begitu sedikit sehingga mereplikasi layer non-ahli sebanyak dua kali adalah hal yang sulit.

Yang terpenting, aliran maju dalam satu arah dan aliran mundur dalam arah lain saling tumpang tindih tepat di tempat gelembung-gelembung itu berada dalam jadwal satu arah. Gelembungnya hilang.

### Jadwal yang dilacak dengan tangan

Misalkan P = 4 peringkat, 8 mikro-batch, dibagi 4 maju / 4 mundur. Waktu bergerak dari kiri ke kanan; baris adalah peringkat perangkat.

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

Membaca notasi "F4/F5R": peringkat 1 berjalan maju dari mikro-batch 4 (dari kiri ke kanan dalam pipa) DAN maju dari mikro-batch 5 (dari kanan ke kiri) dalam slot waktu yang sama. Itulah yang dimaksud dengan "dua arah" secara operasional.

Pada peringkat 2 aliran silang saling tumpang tindih lebih cepat, pada peringkat 0 dan P-1 aliran silang tersebut tumpang tindih paling lambat. Dalam jadwal fase tengah yang stabil, setiap peringkat berjalan maju dari arah X tumpang tindih dengan mundur dari arah Y. Komputasi sedang sibuk. Pengiriman semua-ke-semua untuk pass maju disembunyikan di dalam komputasi mundur. Kombinasi all-to-all sembunyikan di dalam komputasi maju. Gelembungnya diperas.

### Akuntansi gelembung

Gelembung pipa standar 1F1B (waktu terbuang per peringkat):

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Penyempurnaan Zero Bubble menurunkannya tetapi tidak menjadi nol. DualPipe, dalam fase stabil, memiliki gelembung nol jika jumlah batch mikro habis dibagi 2 kali kedalaman pipa. Di luar fase stabil (pemanasan dan pendinginan), ada beberapa gelembung tetapi tidak bertambah seiring dengan jumlah batch mikro — sebuah properti utama yang disoroti makalah ini.Dalam istilah pemasaran: "bebas gelembung". Dalam istilah teknis: gelembung tidak tumbuh dengan jumlah mikro-batch. Analisis lanjutan Sea AI Lab (DualPipeV / Cut-in-half) menunjukkan zero-bubble penuh hanya jika Paralelisme Pakar bukan penghambatnya; dengan all-to-all yang digerakkan oleh EP, beberapa kompromi penjadwalan selalu ada.

### DualPipeV — penyempurnaan

Sea AI Lab (2025) mengamati bahwa replikasi parameter 2x akan sia-sia jika komunikasi EP tidak tumpang tindih. Jadwal DualPipeV melipat injeksi dua arah menjadi jadwal "bentuk V" yang berjalan pada salinan parameter tunggal. Gelembungnya sedikit lebih besar dibandingkan DualPipe, namun penghematan memori cukup besar. DeepSeek mengadopsi DualPipeV dalam implementasi DualPipe sumber terbuka sebagai mode EP-off.

Pengorbanannya:

| Feature | Pipa Ganda | Pipa GandaV | 1F1B | Nol Gelembung |
|---------|---------|-----------|------|------------|
| Salinan param per perangkat | 2 | 1 | 1 | 1 |
| Gelembung vs mikro-batch | konstan | pertumbuhan kecil | tumbuh | tumbuh |
| Tumpang tindih komunikasi komputasi | penuh | sebagian | minimal | sebagian |
| Gunakan ketika | MoE yang berat EP | padat atau ringan EP | dasar | pipa apa pun |

### Apa artinya menjalankan token 14,8T

Pra-training DeepSeek-V3 menghabiskan 14,8T token pada 2.048 GPU H800 dalam waktu sekitar 2,8 juta jam GPU. Dengan 1F1B yang naif, mereka akan kehilangan 12-15% dari jumlah tersebut karena gelembung pipa — 340-420 ribu jam GPU, cukup untuk melatih model 70B penuh. DualPipe memulihkan sebagian besar dari itu. Sulit untuk mengukur kontribusi secara langsung tanpa log internal, tetapi klaim dalam makalah ini adalah rata-rata penggunaan GPU di atas 95% di seluruh training.

Untuk proses yang lebih kecil (di bawah 1.000 GPU), DualPipe berlebihan — gelembung pipeline lebih kecil dibandingkan total biaya, dan training model yang padat jarang mencapai hambatan yang menyeluruh. Untuk training MoE terdepan pada skala multi-ribu GPU, hal ini diperlukan secara efektif.

### Tempatnya di tumpukan

- Pelengkap **FSDP** (Fase 10 · 05). FSDP membagi parameter model ke seluruh peringkat; DualPipe menjadwalkan komputasi di seluruh peringkat. Mereka menggabungkan.
- Kompatibel dengan pecahan gradient **ZeRO-3**. Pembukuan untuk replikasi dua salinan perlu bekerja sama dengan gradient pecahan ZeRO.
- Membutuhkan **kernel khusus untuk semua** yang disetel untuk topologi cluster tertentu. Kernel sumber terbuka DeepSeek adalah implementasi referensi.

## Pakai

`code/main.py` adalah simulator jadwal pipeline pipa. Dibutuhkan `(P, n_micro_batches, schedule)` dan mencetak pemanfaatan fase stabil untuk masing-masing 1F1B, Zero Bubble, DualPipe, dan DualPipeV. Ini adalah alat pengajaran — angka-angka tersebut sesuai dengan klaim kualitatif di koran, namun bukan merupakan klaim tentang percepatan produksi yang terukur.

Nilai simulator: jalankan dengan jumlah P dan mikro-batch yang berbeda dan perhatikan bagaimana fraksi gelembung tumbuh untuk 1F1B tetapi tidak untuk DualPipe.

Pertimbangan integrasi untuk pelaksanaan training nyata:

- Pilih kedalaman paralel pipa yang terbagi rapi ke dalam jumlah batch mikro kamu.
- Pastikan mesh paralel ahli kamu mendukung all-to-all dua arah. Kernel DeepSeek adalah referensinya.
- Berharap untuk menghabiskan waktu debug selama seminggu pada jadwal itu sendiri untuk pertama kalinya. Pembukuannya rumit.
- Pantau pemanfaatan GPU per peringkat, bukan hanya agregat. Manfaat DualPipe berasal dari pengetatan yang tersesat.

## Kirim

Lesson ini menghasilkan `outputs/skill-dualpipe-planner.md`. Mengingat spesifikasi klaster training (jumlah GPU, topologi, interkoneksi, bentuk model), strategi ini merekomendasikan strategi paralelisme jalur pipa, algoritme penjadwalan yang akan digunakan, dan pecahan gelembung yang diharapkan pada skala target.

## Latihan

1. Jalankan `code/main.py` di `(P=8, micro_batches=16, schedule=dualpipe)` dan `(P=8, micro_batches=16, schedule=1f1b)`. Hitung perbedaan pemanfaatan GPU dan nyatakan sebagai jam GPU yang dipulihkan per juta token training.

2. Buat sketsa tabel jadwal `(P=4, micro_batches=8, schedule=dualpipe)` dengan tangan. Tandai setiap slot waktu dengan ID mikro-batch dan arahnya. Identifikasi slot waktu pertama di mana tidak ada gelembung.

3. Baca Gambar 5 laporan teknis DeepSeek-V3 (arXiv:2412.19437). Identifikasi jendela yang tumpang tindih untuk pengiriman semua ke semua di dalam potongan maju DualPipe. Jelaskan bagaimana jadwal komputasi menyembunyikannya.

4. Hitung overhead parameter 2x DualPipe untuk model padat 70B dengan phase pipa P=8 dan model MoE 671B dengan phase pipa P=16. Tunjukkan mengapa overhead kasus Kementerian Lingkungan Hidup secara proporsional lebih kecil (sebagian besar parameternya adalah para ahli, yang dibagi ke dalam kelompok EP yang besar).

5. Bandingkan DualPipe dengan Chimera (penjadwal dua arah yang bersaing mulai tahun 2021). Identifikasi dua properti spesifik yang ditambahkan DualPipe yang tidak dimiliki Chimera, dengan menggunakan Bagian 3.4 makalah sebagai referensi.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Gelembung pipeline pipa | "Waktu menganggur per peringkat" | Siklus GPU terbuang karena tahapan pipeline menunggu input atau gradiennya |
| 1F1B | "Jadwal pipeline pipa default" | Satu penjadwalan maju / mundur yang disisipkan; DualPipe dasar mengalahkan |
| Nol Gelembung | "Lab AI Laut 2023" | Dibagi mundur menjadi B (gradient input) dan W (gradient weight); hampir sepenuhnya mengencangkan pipa |
| Pipa Ganda | "Jadwal DeepSeek-V3" | Pipeline dua arah + tumpang tindih komunikasi komputasi; gelembung tidak tumbuh dengan jumlah batch mikro |
| Pipa GandaV | "Potong menjadi dua" | Penyempurnaan bentuk V yang menghilangkan replikasi parameter 2x dengan mengorbankan gelembung yang sedikit lebih besar |
| Potongan | "Satuan Kerja Pipa" | Jalur maju atau mundur dari satu mikro-batch melalui satu phase pipeline pipa |
| Pengiriman semua ke semua | "Kirim token ke ahlinya" | Komunikasi lintas node yang merutekan token ke pakar MoE yang ditugaskan |
| Gabungan semua-ke-semua | "Kembalikan output ahli" | Komunikasi lintas node yang mengumpulkan output pakar setelah MLP |
| Paralelisme Pakar (EP) | "Pakar di seluruh GPU" | Pecahan pakar MoE di seluruh peringkat sehingga GPU yang berbeda memiliki pakar yang berbeda |
| Paralelisme Pipeline Pipa (PP) | "Layer di seluruh GPU" | Pecahan memodelkan layer di seluruh peringkat; dimension jadwal DualPipe |
| Fraksi gelembung | "Waktu GPU yang terbuang" | (waktu_gelembung/waktu_total); pecahan yang digerakkan DualPipe menuju nol |

## Bacaan Lanjutan- [DeepSeek-AI — Laporan Teknis DeepSeek-V3 (arXiv:2412.19437), Bagian 3.3.2 dan Gambar 5](https://arxiv.org/abs/2412.19437) — referensi utama DualPipe
- [DeepSeek — repositori DualPipe GitHub](https://github.com/deepseek-ai/DualPipe) — implementasi referensi sumber terbuka, termasuk mode DualPipeV (Cut-in-half)
- [Qi dkk. — Paralelisme Pipeline Pipa Zero Bubble (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — pendahulu Zero Bubble
- [Sea AI Lab — DualPipe bisa lebih baik tanpa Dual](https://sail.sea.com/blog/articles/63) — analisis DualPipeV yang menginformasikan mode EP-off DeepSeek
- [Narayanan dkk. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — jadwal 1F1B yang dibandingkan dengan DualPipe
- [Huang dkk. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — masalah kertas dan gelembung paralelisme pipa asli
