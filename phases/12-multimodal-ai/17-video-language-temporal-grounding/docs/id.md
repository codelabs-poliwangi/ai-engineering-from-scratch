# Model Bahasa Video: Token Temporal dan Grounding

> Video bukanlah tumpukan foto. Klip berdurasi 5 detik memiliki urutan sebab akibat, kata kerja tindakan, dan waktu peristiwa yang tidak dapat diwakili oleh model gambar. Video-LLaMA (Zhang et al., Juni 2023) mengirimkan video-LLM terbuka pertama dengan landasan audio-visual. VideoChat dan Video-LLaVA menskalakan pola tersebut. Pada tahun 2025, TMRoPE Qwen2.5-VL menutup kesenjangan dengan model eksklusif terdepan. Setiap sistem memecahkan token temporal secara berbeda — Q-former per klip, concat-pool per frame, TMRoPE per token. Lesson ini membaca pola, membuat frame sampler seragam vs dinamis, dan mengevaluasi tugas landasan temporal.

**Type:** Build
**Language:** Python (stdlib, frame sampler + evaluator temporal-grounding)
**Prerequisites:** Fase 12 · 08 (LLaVA-OneVision)
**Waktu:** ~180 menit

## Tujuan Pembelajaran

- Jelaskan mengapa pengkodean posisi temporal mengubah kinerja VLM video secara independen dari encoder visi.
- Bandingkan pengambilan sample frame yang seragam, FPS dinamis, dan berbasis peristiwa pada akurasi token per detik vs grounding.
- Jelaskan desain Q-former-per-klip (Video-LLaMA) vs pooled-per-frame (Video-LLaVA) vs M-RoPE-per-token (Qwen2.5-VL).
- Sebutkan empat tolok ukur video: VideoMME, TempCompass, EgoSchema, Video-MMMU.

## Masalah

Video berdurasi 1 menit pada 30 FPS adalah 1800 frame. Dengan 196 token visual per frame (ViT-B pada 224), itu berarti 352 ribu token — lebih besar dari konteks LLM era 2024 mana pun.

Ada tiga strategi pengurangan:

1. Bingkai subsampel (1-8 FPS tergantung konten).
2. Gabungkan token patch setiap frame secara agresif (kumpulan bilinear 3x3 atau 4x4).
3. Kompres melalui Q-former yang mengambil klip 16 bingkai dan menghasilkan 64 token.

Setiap trade-off berbeda. Subsampling kehilangan detail temporal. Pooling kehilangan detail spasial. Q-mantan kehilangan keduanya sedikit tetapi menyimpan token.

Pengkodean posisi temporal adalah sumbu lainnya: bagaimana model mengetahui frame 5 datang sebelum frame 6? Opsinya mencakup RoPE temporal 1D sederhana (Video-LLaMA), embeddings temporal yang dipelajari (Video-LLaVA), dan TMRoPE (Qwen2.5-VL, 3D penuh).

## Konsep

### Video-LLaMA: Q-former per klip + cabang audio

Video-LLaMA (2023) adalah video-LLM terbuka pertama. Arsitektur:

- Klip 16 bingkai pada 2 FPS (jadi 8 detik).
- Feature ViT per-frame -> Video Q-former yang hadir secara silang di seluruh 16 frame -> 32 kueri yang dipelajari -> LLM.
- Cabang audio paralel: bentuk gelombang -> encoder audio ImageBind -> Audio Q-former -> 32 kueri -> LLM.

Kekuatan: penalaran bersama audio-visual. Kelemahan: panjang klip tetap, tidak ada grounding waktu yang sewenang-wenang.

### Obrolan Video dan Video-LLaVA

VideoChat mempertahankan ide Video-LLaMA tetapi menghilangkan audio dan menyederhanakannya. Video-LLaVA (Lin et al., 2023) melatih encoder visual tunggal pada gambar dan bingkai video ("penyelarasan sebelum proyeksi"), memberikan representasi terpadu. Keduanya adalah encoder CLIP beku + MLP + LLM.

Tidak ada yang menangani video panjang. Keduanya adalah sistem bingkai 8-16.

### Qwen2.5-VL dan TMRoPE

Qwen2.5-VL memperkenalkan TMRoPE — Embedding Posisi Putar Modalitas Temporal. Setiap token patch memiliki posisi (t, h, w) di mana t adalah stempel waktu sebenarnya (bukan indeks bingkai).

Perbedaan utama dari embedding temporal sederhana:

- Waktu absolut, bukan indeks. Model melihat "pada 4,2 detik" bukan "pada frame 15".
- Rotasi per token, bukan per klip. Setiap token visual berputar secara independen berdasarkan stempel waktunya.
- Kompatibel dengan FPS dinamis. Jika kamu mengambil sample pada 2 FPS di sini dan 4 FPS di sana, TMRoPE menangani distance yang tidak rata secara asli.TMRoPE mengaktifkan "pada detik berapa kucing itu melompat?" pertanyaan. Model dapat menghasilkan output "dalam 4,2 detik". Video-LLaMA hanya bisa mengatakan "di awal klip".

### Strategi pengambilan sample bingkai

Seragam: sample N frame secara merata sepanjang durasi. Sederhana, kehilangan puncak gerak.

FPS Dinamis: sample secara adaptif berdasarkan intensitas gerakan. Perbedaan aliran optik atau bingkai mengambil segmen gerak tinggi untuk pengambilan sample yang lebih padat. Qwen2.5-VL melatih ini.

Berbasis peristiwa: jalankan detektor ringan, ambil sample lebih banyak saat tindakan terjadi. Digunakan oleh VideoAgent.

Bingkai utama + konteks: sample pada batas pengambilan gambar + beberapa bingkai yang berdekatan. Digunakan untuk konten sinematik.

### Penggabungan per frame

Pada 1 FPS dan 576 token per frame, klip berdurasi 5 menit menghasilkan 172.800 token. Dapat dilakukan dengan konteks 128k Qwen2.5-VL-72B tetapi mahal.

Kumpulan bilinear 3x3 dikurangi menjadi 64 token per frame -> 19.200 token selama 5 menit. Tempat yang tepat untuk sebagian besar tugas.

Penggabungan lebih agresif (6x6 -> 16 token per frame) untuk alur kerja agen yang detail spasialnya tidak terlalu penting.

### Empat tolok ukur video

- VideoMME: pemahaman video komprehensif, pendek + sedang + panjang.
- TempCompass: penalaran temporal yang terperinci, pertanyaan "sebelum" / "sesudah".
- EgoSchema: video orang pertama dengan cakrawala panjang.
- Video-MMMU: pertanyaan video multimodal multi-disiplin.

Evaluasi video-VLM lengkap mengenai keempatnya. Mereka menekankan sumbu yang berbeda - TempCompass adalah tentang pemesanan, EgoSchema adalah sekitar 3+ menit, VideoMME mencakup durasi.

### Membumikan format output

Format output untuk landasan sementara:

- Teks gratis: "Kucing itu melompat sekitar tanda 4 detik." Mudah diurai tetapi tidak tepat.
- JSON Terstruktur: `{"event": "jump", "start": 4.1, "end": 4.3}`. Qwen2.5-VL melatih ini.
- Berbasis token: token khusus `<time>4.1</time>` disisipkan dengan jawabannya. Format internal Qwen2.5-VL.

Berbasis token paling akurat untuk penggunaan hilir. Format output JSON Qwen2.5-VL diurai secara langsung.

### Praktik terbaik tahun 2026

Untuk VLM video pada tahun 2026:

- Encoder: SigLIP 2 dengan M-RoPE atau TMRoPE (Qwen2.5-VL).
- Pengambilan sample bingkai: FPS dinamis (1-4 tergantung gerakan) dengan batas bingkai maksimal.
- Penggabungan per bingkai: bilinear 3x3.
- Output: JSON terstruktur dengan bidang waktu + acara.
- Tolok ukur: VideoMME + TempCompass untuk umum; EgoSchema untuk cakrawala panjang.

## Pakai

`code/main.py` meliputi:

- Sampler bingkai FPS yang seragam dan dinamis.
- Evaluator landasan waktu mainan: diberi peristiwa "kebenaran dasar" pada waktu T dan output model, skor akurasi dengan toleransi.
- Perbandingan Video-LLaMA (16 frame, Q-former), Video-LLaVA (8 frame, MLP), Qwen2.5-VL (FPS dinamis + TMRoPE).

## Kirim

Lesson ini menghasilkan `outputs/skill-video-vlm-frame-planner.md`. Dengan adanya tugas video (pemantauan, pengenalan tindakan, landasan temporal, peringkasan), ia memilih frame sampler, faktor pengumpulan, format output, dan tingkat akurasi yang diharapkan.

## Latihan

1. Untuk demo memasak 3 menit, pilih FPS seragam vs dinamis. Benarkan dengan jumlah token.

2. TMRoPE menambahkan apa yang secara spesifik tidak dapat dilakukan oleh tabel embedding temporal sederhana?

3. Tulis skema JSON untuk landasan temporal yang dapat dipelajari oleh VLM untuk dipancarkan. Sertakan kasus kesalahan.

4. Baca Video-LLaVA Bagian 3 tentang "Penyelarasan Sebelum Proyeksi". Mengapa ini lebih baik daripada melatih pembuat enkode gambar dan video secara terpisah?

5. Berdasarkan papan peringkat VideoMME, berapa kesenjangan antara model terbuka teratas dan model eksklusif teratas pada tahun 2026? Seberapa besar kesenjangan tersebut disebabkan oleh pengkodean temporal vs skala LLM dasar?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Landasan sementara | "Jawaban yang dilokalkan waktu" | VLM mengeluarkan rentang stempel waktu tertentu ketika suatu peristiwa terjadi |
| TMRoPE | "Tali Multimodal Waktu" | Posisi putar 3D dengan stempel waktu absolut, digunakan oleh Qwen2.5-VL |
| FPS Dinamis | "Pengambilan sample sadar gerakan" | Cicipi lebih banyak frame pada segmen gerak tinggi, lebih sedikit pada segmen statis |
| Penggabungan bingkai | "Kompres spasial per frame" | Kurangi patch per frame dengan interpolasi bilinear sebelum LLM |
| Video Q-mantan | "Klip kompresor" | Pemetaan kemacetan attention silang N frame ke K kueri yang dipelajari |
| VideoMME | "Bangku video" | Tolok ukur video pendek/sedang/panjang yang komprehensif, 2500+ sample |

## Bacaan Lanjutan

- [Zhang dkk. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li dkk. — Obrolan Video (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin dkk. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Tim Qwen — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin dkk. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
