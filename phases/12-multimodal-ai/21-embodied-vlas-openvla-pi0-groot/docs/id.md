# VLA yang diwujudkan: RT-2, OpenVLA, π0, GR00T

> Pertama kali seorang model membaca resep dari situs web dan mengeksekusinya di robot dapur adalah RT-2 (Google DeepMind, Juli 2023). RT-2 mendiskritisasi tindakan sebagai token teks, menyempurnakan VLM pada data web ditambah data tindakan robot, dan membuktikan bahwa pengetahuan bahasa visi skala web ditransfer ke kontrol robot. OpenVLA (Juni 2024) mengirimkan referensi 7B terbuka. Seri π0 Kecerdasan Fisik (2024-2025) menambahkan pakar tindakan pencocokan aliran. GR00T N1 NVIDIA (Maret 2025) menghadirkan kontrol sistem ganda (Sistem 1 / Sistem 2) untuk robot humanoid dalam skala besar. VLA primitif — tindakan-bahasa-visi, model tunggal yang melihat, membaca, dan bertindak — adalah jembatan antara model pemahaman fase ini dan sistem otonom di Fase 15.

**Type:** Learn
**Language:** Python (stdlib, tokenizer tindakan + kerangka inference VLA)
**Prerequisites:** Fase 12 · 05 (LLaVA), Fase 15 (Sistem Otonom, referensi)
**Waktu:** ~180 menit

## Tujuan Pembelajaran

- Jelaskan tokenization tindakan: pengkodean bin diskrit (RT-2), token tindakan efisien CEPAT, tindakan pencocokan aliran berkelanjutan (π0).
- Jelaskan mengapa penyesuaian bersama pada data web + robot mempertahankan transfer pengetahuan umum ke tugas-tugas baru.
- Bandingkan OpenVLA (buka 7B Llama+VLM), π0 (pencocokan aliran), dan GR00T N1 (sistem ganda) pada tugas robot yang sama.
- Beri nama dataset Open X-Embodiment dan perannya sebagai korpus training RT-X.

## Masalah

Robot yang melakukan tugas-tugas berdasarkan instruksi bahasa alami telah menjadi target penelitian sejak tahun 1970-an. Jawaban tahun 2020-an: model visi-bahasa-aksi (VLA). Arsitektur VLM yang sama digunakan untuk VQA, tetapi keluarannya berupa tindakan (torsi gabungan, pose efektor akhir, prompt terpisah) dan bukan teks.

Tantangan khusus untuk VLA:

1. Ruang aksi bersifat kontinu (sudut sambungan, gaya) dan berdimensi tinggi (lengan 7-DOF + gripper 3-DOF = 10 redup pada 30 Hz).
2. Training data khusus robot masih langka. Open X-Embodiment memiliki ~1 juta lintasan; gambar teks web adalah 5B+.
3. Frekuensi kontrol itu penting. Loop kontrol 30 Hz berarti anggaran 33ms per tindakan.
4. Keamanan. Tindakan yang salah akan merusak perangkat keras, manusia, atau properti.

## Konsep

### Tokenization tindakan (RT-2)

Trik RT-2: mewakili setiap target gabungan sebagai token teks terkuantisasi. Diskritisasi rentang [-1, 1] yang dinormalisasi menjadi 256 bin, petakan setiap bin ke ID kosakata. Tindakan 10-DOF menjadi 10 token pada setiap langkah kontrol.

Menyempurnakan bersama VLM PaLM-X pada campuran:

- Pasangan gambar-teks web (captioning, VQA).
- Demonstrasi robot, aksi sebagai token.

Model melihat "ambil kubus merah" (bahasa) → gambar (penglihatan) → urutan tindakan 10 token (target gabungan yang didiskritkan). Pra-training web mempertahankan transfer pengetahuan umum: RT-2 dapat mengikuti "bergerak menuju objek yang bergerak cepat" meskipun "bergerak cepat" tidak ada dalam training data.

Inference pada 3-5 Hz pada makalah RT-2, dibatasi oleh dekode autoregresif VLM.

### OpenVLA — referensi 7B terbuka

OpenVLA (Kim dkk., Juni 2024) adalah setara RT-2 dengan weight terbuka. Tulang punggung Llama 7B, encoder visi ganda DINOv2 + SigLIP, tokenization tindakan melalui 256 bin.

Dilatih pada Open X-Embodiment (970 ribu lintasan di 22 robot). Dikirim dengan dukungan penyesuaian LoRA untuk beradaptasi dengan robot baru.

Inference: 4-5 Hz pada A100 dengan kuantisasi. Cukup cepat untuk manipulasi lambat, bukan untuk kontrol frekuensi tinggi.

### Tokenizer CEPAT — dekode tindakan yang lebih cepatPertsch dkk. (2024) menunjukkan bahwa tokenization diskrit tidak efisien — sebagian besar tindakan mengelompok di wilayah kecil ruang bin. FAST (Frequency-domain Action Sequence Tokenizer) memampatkan urutan tindakan melalui DCT dan mengkuantisasi koefisien.

Lintasan tindakan 30 langkah menjadi ~10 token FAST, bukan 300 token diskrit. Kecepatan inference meningkat 3-5x tanpa kehilangan kualitas.

### π0 dan tindakan pencocokan aliran

π0 dari Kecerdasan Fisik (Black et al., Oktober 2024) menggantikan token tindakan diskrit dengan pakar tindakan pencocokan aliran:

- Trafo aksi kecil membaca status tersembunyi VLM dan menghasilkan rangkaian aksi 50 langkah berkelanjutan melalui aliran yang diperbaiki.
- Kepala aksi berlatih dengan kehilangan pencocokan aliran; Pra-training VLM tetap tidak berubah.
- Inference: rangkaian tindakan penuh yang dipancarkan dalam ~5 langkah denoising, kontrol efektif 50 Hz.

Klaim π0: mengalahkan OpenVLA dan Octo dalam berbagai tugas manipulasi. Formulasi tindakan berkelanjutan menjaga kelancaran yang merusak diskritisasi.

π0.5 dan π0-FAST adalah peningkatan bertahap. π0-FAST menggabungkan tokenization FAST dengan pencocokan aliran.

### GR00T N1 — sistem ganda untuk humanoids

GR00T N1 NVIDIA (Maret 2025) dibuat untuk robot humanoid (>30 DOF, seluruh tubuh):

- Sistem 2: adegan pembacaan VLM besar + instruksi, menghasilkan subtujuan tingkat tinggi pada ~1 Hz.
- Sistem 1: Transformer kepala aksi kecil yang menghasilkan prompt gabungan tingkat rendah 50-100 Hz yang dikondisikan pada subtujuan.

Perpecahan ini mencerminkan pemikiran Kahneman yang cepat dan lambat: Sistem 2 berencana, Sistem 1 bertindak. Manfaat: perencanaan berukuran VLM yang lambat tidak menghalangi kontrol yang cepat; Sistem 1 tetap kecil untuk latensi.

GR00T N1.7 (akhir 2025) meningkatkan penskalaan data. GR00T menyempurnakan data sim-to-real dari Omniverse.

### Buka Perwujudan X

Training data. RT-X (Oktober 2023) mengumpulkan 22 dataset yang mencakup 1 juta lintasan di 22 robot. Open X-Embodiment adalah korpus yang digunakan semua orang:

- Dapur ALOHA / Bridge V2 / Droid / RT-2 / Meja Bahasa.
- Setiap sample: (keadaan robot, tampilan kamera, instruksi, urutan tindakan).
- Training kebersihan: menyatukan ruang tindakan, menormalkan rentang sendi, mengubah ukuran kamera.

OpenVLA dan π0 berlatih di Open X-Embodiment. Kesenjangan domain pada robot tertentu ditutup dengan penyempurnaan LoRA pada 100-1000 demo tugas tertentu.

### Penyempurnaan bersama vs khusus robot

Penyetelan bersama memadukan data VQA web dengan lintasan robot. Rasio itu penting: terlalu banyak VQA dan model melupakan tindakan; terlalu banyak data robot dan model kehilangan pengetahuan umum.

Rasio RT-2: ~1:1. OpenVLA: ~0,5:1 web-ke-robot. π0: serupa. Rasio yang tepat adalah hyperparameter yang harus disesuaikan per ukuran dataset.

Training khusus robot menghasilkan model tugas spesifik yang gagal pada instruksi di luar distribusi. Penyempurnaan bersama adalah perbedaan antara "ambil kubus merah (dalam demo)" dan "ambil objek terbesar ketiga dari kiri (frase baru)".

### Batas keamanan dan tindakan

Setiap VLA produksi dikirimkan dengan:

- Batas sambungan keras (torsi tidak dapat melewati spesifikasi).
- Batas kecepatan (kliping lembut).
- Batas ruang kerja (efektor akhir tidak dapat meninggalkan tabel).
- Persetujuan human-in-the-loop untuk tugas-tugas baru.

Ini berada di luar VLA sebagai pemeriksaan layer kontrol. Output VLA berupa saran, bukan prompt.

## Pakai

`code/main.py`:

- Mengimplementasikan tokenization dan de-tokenization tindakan 256-bin.
- Membuat sketsa tokenizer CEPAT berdasarkan kuantisasi DCT +.
- Membandingkan jumlah token per langkah tindakan (discrete-bin, FAST, continuous-flow).
- Mencetak ringkasan silsilah RT-2 → OpenVLA → π0 → GR00T.## Kirim

Lesson ini menghasilkan `outputs/skill-vla-action-format-picker.md`. Diberikan tugas robot (manipulasi, navigasi, seluruh tubuh humanoid), memilih antara discrete-bin + RT-2, FAST + OpenVLA, pencocokan aliran + π0, atau sistem ganda + GR00T.

## Latihan

1. Lengan 10-DOF pada kecepatan kontrol 30 Hz. Tokenization nampan diskrit pada 256 nampan mengeluarkan berapa banyak token per detik? Bisakah VLM 7B mengimbanginya?

2. Tokenization CEPAT memampatkan lintasan 30 langkah menjadi ~10 token. Apa loss pengguna jika lintasan memiliki gerakan frekuensi tinggi (misalnya, bermain drum)?

3. Kepala pencocokan aliran π0 berbunyi dalam ~5 langkah. Bandingkan throughput dengan dekode autoregresif OpenVLA pada 4-5 Hz.

4. Sistem 1 / Sistem 2 GR00T membagi peta ke Kahneman. Usulkan pemisahan berbeda (Sistem 3?) yang mungkin membantu berjalan bipedal.

5. Baca Open X-Embodiment Bagian 4 tentang kurasi dataset. Sebutkan tiga aturan kurasi yang mencegah kebocoran domain.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| VLA | "Visi-bahasa-aksi" | Model yang mengambil gambar + instruksi dan mengeluarkan prompt tindakan |
| Tokenization tindakan | "Tempat sampah terpisah" | Hitung target gabungan berkelanjutan menjadi 256 bins per dim, masing-masing merupakan vocab ID |
| Tokenizer CEPAT | "Token tindakan frekuensi" | DCT + kuantisasi untuk mengompresi lintasan 30 langkah menjadi ~10 token |
| Sempurnakan bersama | "Campurkan web + robot" | Latih data VQA web bersama demo robot untuk melestarikan pengetahuan umum |
| Kepala tindakan pencocokan aliran | "π0 output berkelanjutan" | Trafo kecil yang menghasilkan rangkaian aksi 50 langkah melalui aliran yang diperbaiki |
| Sistem 1 / Sistem 2 | "Kontrol sistem ganda" | VLM besar merencanakan dengan lambat, kepala tindakan kecil bertindak cepat; Pola GR00T |
| Buka X-Perwujudan | "Dataset RT-X" | dataset lintas robot lintasan 1M; korpus training |

## Bacaan Lanjutan

- [Brohan dkk. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim dkk. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Hitam dkk. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Buka Kolaborasi Perwujudan X — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)
