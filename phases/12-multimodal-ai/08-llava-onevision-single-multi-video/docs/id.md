# LLaVA-OneVision: Gambar Tunggal, Multi-Gambar, Video dalam Satu Model

> Sebelum LLaVA-OneVision (Li et al., Agustus 2024) dunia VLM terbuka memiliki garis keturunan yang berbeda: LLaVA-1.5 untuk gambar tunggal, model multi-gambar seperti Mantis dan VILA, model video seperti Video-LLaVA dan Video-LLaMA. Masing-masing memenangkan patokannya dan gagal dalam patokan lainnya. LLaVA-OneVision berpendapat bahwa kurikulum tunggal dapat melatih satu model untuk mendominasi ketiga skenario, dan bahwa efek pengalihan tugas yang muncul (keterampilan gambar tunggal diekspor ke video, penalaran multi-gambar diekspor ke gambar tunggal) mengalahkan jumlah spesialis. Resepnya tampak sederhana: anggaran token visual yang tetap konstan di seluruh skenario, ditambah kurikulum eksplisit yang beralih dari gambar tunggal ke OneVision (multi-gambar) hingga video. Lesson ini membaca anggaran, kurikulum, dan perilaku yang muncul.

**Type:** Build
**Language:** Python (stdlib, pemecah anggaran token + perencana kurikulum)
**Prerequisites:** Fase 12 · 05 (LLaVA), Fase 12 · 06 (resolusi apa pun)
**Waktu:** ~180 menit

## Tujuan Pembelajaran

- Rancang anggaran token visual yang tetap konstan di seluruh input gambar tunggal, multi-gambar, dan video.
- Pesan kurikulum training yang mentransfer keterampilan dari satu gambar ke video tanpa melupakan bencana.
- Jelaskan mengapa satu model mengalahkan spesialis dalam jumlah parameter yang sama ketika kurikulum dilakukan dengan benar.
- Sebutkan tiga kemampuan yang muncul yang dilaporkan oleh LLaVA-OneVision: penalaran multi-kamera, prompt set-of-mark, agen tangkapan layar iPhone.

## Masalah

Gambar, multi-gambar, dan video masing-masing menekankan model secara berbeda.

Gambar tunggal menginginkan token resolusi tinggi (AnyRes, ~2880 token visual) untuk menangkap OCR dan detail halus. Anggaran per sample: satu gambar, 2880 token.

Multi-gambar menginginkan beberapa gambar dengan resolusi sedang (masing-masing ~576 token) sehingga penalaran di seluruh gambar sesuai dengan konteks. Anggaran per sample: 4-8 gambar, masing-masing 576, 2300-4600 token.

Video menginginkan banyak frame dengan resolusi rendah (~196 token per frame setelah pengumpulan) untuk menangkap dinamika temporal. Anggaran per sample: 8-32 frame, masing-masing 196 frame, 1600-6200 token.

Jika kamu melatih model terpisah, kamu memilih satu anggaran. Jika kamu melatih satu model, kamu memerlukan anggaran yang dapat diskalakan secara wajar di seluruh skenario tanpa mengabaikan konteks.

Sebelum OneVision, jawaban defaultnya adalah "latih satu skenario, abaikan skenario lainnya." Video-LLaVA memasang kembali video ke model gambar dengan tahapan training tambahan. LLaVA-NeXT menambahkan dukungan multi-gambar dengan ubin. Tidak ada yang menangani ketiganya dengan bersih.

## Konsep

### Anggaran token OneVision

LLaVA-OneVision memilih anggaran token visual terpadu sekitar 3000-4000 token per sample, yang dialokasikan secara berbeda per skenario:

- Gambar tunggal: AnyRes-9 (ubin 3x3 + thumbnail), setiap ubin berukuran 384 dengan 729 tambalan, pengumpulan bilinear agresif 2x2 → 182 per ubin. Total: 9*182 + 182 = 1820 token. Atau AnyRes-4 dengan 729 per ubin = 2916 + 729.
- Multi-gambar: setiap gambar dengan resolusi sedang (384, tanpa ubin), 729 token tanpa pengumpulan. Anggaran 6 gambar → 4374 token.
- Video: 32 frame pada resolusi 384 dengan kumpulan bilinear 3x3 agresif → 81 token per frame. Total: 32*81 = 2592 token.

Alokasi ini mempertahankan total token yang kira-kira konstan. LLM tidak pernah melihat kelompok yang melanggar konteksnya. Encoder menghasilkan geometri yang berbeda per skenario, namun LLM menggunakan anggaran yang sama.

### Kurikulum tiga phase

LLaVA-OneVision berlatih dalam tiga phase:1. SFT gambar tunggal (phase SI). Semua data adalah gambar tunggal-plus-teks. Berlatih dengan input AnyRes resolusi tinggi. Ini mengajarkan persepsi, OCR, dan pemahaman yang mendalam. Menggunakan data LLaVA-NeXT ditambah data gambar tunggal khusus OneVision.
2. OneVision SFT (phase OV). Campurkan gambar tunggal + multi-gambar + video (bingkai sample seragam). Latih anggaran token terpadu. Hal ini mengajarkan model untuk menangani bentuk batch yang heterogen. Tidak ada pengaturan ulang weight — berlanjut dari phase SI.
3. Pemindahan tugas (phase TT). Lanjutkan dengan campuran tugas target, biasanya lebih berat pada multi-gambar atau video tergantung pada produknya. Penyempurnaan opsional untuk penerapan.

Kritis: urutan kurikulum itu penting. Training video-first atau multi-image-first menghasilkan performa gambar yang lebih buruk dibandingkan single-image-first, bahkan dengan data yang sama. Makalah ini menghapuskan hal ini secara eksplisit.

### Mengapa kurikulum berhasil

Training gambar tunggal membangun basis persepsi. Token patch membawa feature visual yang sangat detail; LLM belajar mengintegrasikannya dengan teks. Multi-gambar dan video menghadirkan tantangan struktural (gambar mana yang mana, apa yang terjadi pertama kali) yang sulit dipelajari tanpa dasar persepsi yang kuat.

Jika kamu melatih semua skenario dari awal secara bersamaan, model tersebut akan menyesuaikan persepsi (data gambar tunggal terbatas per kumpulan) dan menyesuaikan struktur (banyak data multi-gambar/video). Hasil: model yang mengikuti pola penalaran lintas gambar namun secara visual dangkal.

Pengurutan kurikulum memberi kamu kekuatan persepsi dari phase SI, kemudian penalaran komposisi/temporal dari phase OV, tanpa kehilangan keduanya.

### Keterampilan lintas skenario yang muncul

Makalah LLaVA-OneVision melaporkan tiga kemampuan yang muncul:

1. Alasan multi-kamera. Dilatih tentang multi-gambar + video secara terpisah; pada kesimpulan, dimintai alasan tentang adegan mengemudi multi-kamera. Model mengintegrasikan tampilan dengan benar meskipun tidak pernah melihat format persisnya dalam training.
2. Prompt set-of-mark. Pengguna memberi anotasi pada objek dalam gambar dengan tanda bernomor; alasan model tentang "apa yang dilakukan tanda 3 dibandingkan dengan tanda 7." Dilatih baik dalam hal tanda maupun anotasi; dipelajari dari kombinasi landasan spasial + referensi multi-gambar.
3. Agen tangkapan layar iPhone. Pengguna memberikan tangkapan layar layar iPhone dan meminta rencana klik berikutnya. Dilatih tentang tangkapan layar UI, video alur kerja pengguna, dan pasangan multi-gambar sebelum/sesudah. Generalisasi ke kasus penggunaan agen.

Ini bukanlah tugas yang dilatih; mereka muncul dari struktur komposisi kurikulum.

### Pengumpulan token visual

Anggaran token memerlukan pengumpulan. OneVision menggunakan interpolasi bilinear pada kisi patch 2D: 24x24 = 576 patch menjadi 12x12 = 144 (2x faktor) atau 8x8 = 64 (3x faktor). Pengumpulan dilakukan di ruang patch-grid, bukan ruang token, untuk melestarikan lokalitas.

Pilihan faktor pengumpulan per skenario itu sendiri merupakan hyperparameter. Lebih sedikit pengumpulan = lebih banyak token = representasi lebih kaya. Lebih banyak pengumpulan = lebih sedikit token = lebih banyak bingkai/gambar yang pas.

### LLaVA-OneVision-1.5

Tindak lanjut tahun 2025 (LLaVA-OneVision-1.5, arXiv 2509.23661) "terbuka penuh" dalam training data, weight model, dan code. Mencocokkan kesenjangan kepemilikan pada beberapa tolok ukur dan mendemokratisasikan resepnya. Kurikulum yang sama, lebih banyak data, basis LLM yang lebih baik. Tidak ada perubahan arsitektur.

### Kontras dengan Qwen2.5-VLQwen2.5-VL (Lesson 12.09) membuat pilihan yang berbeda. Ia menggunakan M-RoPE dan FPS dinamis, bukan pengumpulan tetap. Anggarannya disesuaikan dengan input — video berdurasi 1 menit menggunakan lebih banyak token daripada video berdurasi 5 detik. LLaVA-OneVision memperbaiki anggaran dan menskalakan pengumpulan. Keduanya bekerja; mereka memperdagangkan kemampuan konfigurasi untuk prediktabilitas.

## Pakai

`code/main.py` adalah perencana kurikulum dan anggaran untuk VLM bergaya OneVision. Mengingat anggaran token per sample dan campuran skenario target (misalnya 40% gambar tunggal, 30% multi-gambar, 30% video), maka:

- Mengalokasikan resolusi, faktor pengumpulan, dan bingkai per skenario.
- Memeriksa apakah setiap skenario sesuai dengan anggaran bersama.
- Melaporkan jumlah token yang diharapkan, FLOP LLM, dan skenario mana yang kurang diberi token.
- Mencetak jadwal training phase demi phase.

Gunakan untuk merencanakan penyempurnaan OneVision atau untuk memeriksa kewarasan biaya per permintaan penerapan VLM.

## Kirim

Lesson ini menghasilkan `outputs/skill-onevision-budget-planner.md`. Mengingat distribusi tugas target dan anggaran per sample, hal ini menghasilkan faktor AnyRes, pengumpulan per frame, jumlah frame video, dan weight tahapan kurikulum. Gunakan ini setiap kali kamu melatih atau menyempurnakan VLM skenario terpadu.

## Latihan

1. Produk kamu mendukung 80% gambar tunggal, 10% multi-gambar (2-4 gambar), 10% video (8-16 frame). Rancang anggaran token. Di mana kamu akan menaruh anggaran ekstra yang kamu hemat karena tidak membuat banyak gambar?

2. Baca LLaVA-OneVision Bagian 4.3 (kemampuan darurat). Usulkan keterampilan baru yang keempat yang mungkin akan dibuka oleh kurikulum tetapi makalah tidak melaporkannya.

3. Tukar urutan kurikulum — latih multi-gambar terlebih dahulu, lalu gambar tunggal, lalu video. Prediksikan tolok ukur mana yang mengalami penurunan dan alasannya.

4. Makalah ini melaporkan benchmark video yang dilatih hanya pada 8 frame per sample. Apakah itu menggeneralisasi video berdurasi 30 detik pada inference? Apa yang harus didahulukan – anggaran token atau alasan temporal?

5. Penggabungan bilinear dari patch 24x24 ke 12x12 adalah pengurangan 4x per dim. Terapkan penggabungan di stdlib Python dan verifikasi bahwa rata-rata pada setiap blok 2x2 cocok dengan output bilinear.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Skenario OneVision | "Gambar tunggal, multi-gambar, atau video" | Salah satu dari tiga input membentuk pegangan VLM terpadu; anggaran tetap konstan di seluruh |
| Anggaran token | "Berapa banyak token per sample" | Total token visual yang dilihat LLM per sample training/inference, biasanya 3000-4000 |
| Kurikulum | "Prompt training" | Pengurutan phase (gambar tunggal → multi-gambar → video) dipilih untuk transfer yang muncul |
| Penggabungan bilinear | "Token menyusut" | Menerapkan interpolasi bilinear ke grid patch (2D) untuk mengurangi jumlah token sambil mempertahankan lokalitas |
| Keterampilan yang muncul | "Tidak dilatih, masih bekerja" | Kemampuan yang muncul pada inference tanpa pencocokan training data, karena komposisi kurikulum |
| AnyRes-k | "penyiapan k-tile" | k sub-ubin dengan resolusi tetap ditambah satu thumbnail, tipikal k ∈ {4, 9} |
| Pemindahan tugas | "Generalisasi lintas skenario" | Keterampilan yang dipelajari pada gambar tunggal yang diterapkan pada video (dan sebaliknya) melalui backbone bersama |

## Bacaan Lanjutan- [Li dkk. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Framework Terbuka Sepenuhnya (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin dkk. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin dkk. — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang dkk. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
