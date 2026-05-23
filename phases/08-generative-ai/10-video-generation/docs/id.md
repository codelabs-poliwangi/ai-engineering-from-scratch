# Pembuatan Video

> Gambar adalah tensor 2-D. Video adalah video 3-D. Teorinya sama; komputasinya 10-100x lebih sulit. Sora OpenAI (Februari 2024) membuktikan hal itu mungkin. Pada tahun 2026 Veo 2, Kling 1.5, Runway Gen-3, Pika 2.0, dan WAN 2.2 mengirimkan video produksi dari teks pada 1080p — dan tumpukan weight terbuka (CogVideoX, HunyuanVideo, Mochi-1, WAN 2.2) tertinggal 12 bulan.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 8 · 07 (Difusi Laten), Fase 7 · 09 (ViT), Fase 8 · 06 (DDPM)
**Waktu:** ~45 menit

## Masalah

Video 1080p 10 detik pada 24fps adalah 240 frame 1920×1080×3 piksel. Itu berarti ~1,5 GB data mentah per klip. Difusi ruang piksel tidak mungkin dilakukan. kamu membutuhkan:

1. **Kompresi spasialtemporal.** VAE yang mengkodekan video, bukan bingkai, ke dalam rangkaian patch spasial-temporal.
2. **Koherensi temporal.** Bingkai perlu berbagi konten, pencahayaan, dan identitas objek dalam hitungan detik. Jaring harus memodelkan gerakan.
3. **Hitung anggaran.** Training video 10-100x lebih mahal dibandingkan gambar untuk ukuran model yang sama.
4. **Pengondisian.** Teks, gambar (frame pertama), audio, atau video lainnya. Kebanyakan model produksi menerima keempatnya.

Arsitektur yang memecahkan masalah ini adalah **Diffusion Transformer (DiT)** yang diterapkan pada patch spatiotemporal, yang dilatih pada dataset besar (prompt, caption, video). Kehilangan difusi yang sama seperti Lesson 06.

## Konsep

![Difusi video: patchify, DiT, decode](../assets/video-generasi.svg)

### Tambalan

Encode video dengan VAE 3D (mempelajari kompresi spatiotemporal). Latennya adalah bentuk `[T_latent, H_latent, W_latent, C_latent]`. Bagi menjadi beberapa bagian dengan ukuran `[t_p, h_p, w_p]`. Untuk model gaya Sora, `t_p = 1` (patch per frame) atau `t_p = 2` (setiap dua frame). Video 1080p berdurasi 10 detik dikompres menjadi ~20.000-100.000 patch.

### DiT Spasialtemporal

Sebuah Transformer memproses rangkaian patch yang datar. Setiap patch memiliki embedding posisi 3D (waktu + y + x). Attention biasanya difaktorkan:

- **Attention spasial** dalam setiap patch frame.
- **Attention sementara** di seluruh bingkai pada lokasi spasial yang sama.
- **Attention 3D penuh** 16-100x lebih mahal; hanya digunakan pada resolusi rendah atau dalam penelitian.

### Pengondisian teks

Attention silang dengan encoder teks besar (T5-XXL untuk Sora, CogVideoX-5B menggunakan T5-XXL). Prompt yang panjang itu penting — set training Sora memiliki teks ulang padat yang dihasilkan GPT dengan rata-rata 200 token per klip.

### Training

Kehilangan difusi standar (prediksi ε atau v) pada laten spatiotemporal. Data: video web + ~100 juta klip hasil kurasi + keterangan teks sintetis. Hitung: 10.000+ jam GPU bahkan untuk penelitian kecil; Skala Sora adalah 100.000+.

## Lanskap produksi tahun 2026| Model | Tanggal | Durasi maksimal | Resolusi maksimal | Buka weight? | Terkenal |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60an | 1080p | Tidak | Model pertama yang menampilkan properti simulator dunia dalam skala besar |
| Sora Turbo | 2024-12 | 20an | 1080p | Tidak | Produksi Sora dengan inference 5x lebih cepat |
| Veo 2 (Google) | 2024-12 | 8 detik | 4K | Tidak | Kualitas + fisika tertinggi pada tahun 2025 |
| Veo 3 | Kuartal 3 2025 | 15 detik | 4K | Tidak | Audio asli dan kontrol kamera yang lebih kuat |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10 detik | 1080p | Tidak | Gerak manusia terbaik di Q1 2025 |
| Landasan Pacu Gen-3 Alpha | 2024-06 | 10 detik | 768p | Tidak | Alat video profesional di atas |
| Pika 2.0 | 2024-10 | 5 detik | 1080p | Tidak | Konsistensi karakter terkuat |
| CogVideoX (THUDM) | 2024 | 10 detik | 720p | Ya (2B, 5B) | Video skala 5B pertama kali dibuka |
| HunyuanVideo (Tencent) | 2024-12 | 5 detik | 720p | Ya (13B) | Buka SOTA akhir 2024 |
| Mochi-1 (Genmo) | 2024-10 | 5.4 detik | 480p | Ya (10B) | Berlisensi paling permisif |
| WAN 2.2 (Alibaba) | 2025-07 | 5 detik | 720p | Ya | Model terbuka terkuat pertengahan tahun 2025 |

Weight terbuka menutup kesenjangan tersebut lebih cepat dibandingkan dengan ruang gambar: HunyuanVideo + WAN 2.2 LoRA telah mendukung sebagian besar alur kerja sumber terbuka pada pertengahan tahun 2026.

## Build

`code/main.py` mensimulasikan ide inti DiT spatiotemporal: menambal video sintetis kecil, menambahkan embedding posisi per tambalan, dan menghilangkan kebisingan seluruh rangkaian dengan attention gaya Transformer pada tambalan. Tidak ada angka; Python murni. Kami menunjukkan bahwa koherensi temporal muncul bahkan dalam 1-D ketika patch bingkai yang berdekatan berbagi denoiser dan embedding posisi.

### Langkah 1: menambal "video" 1-D sintetis

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Langkah 2: embedding posisi per frame

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Langkah 3: denoiser melihat seluruh rangkaian

Daripada menolak setiap frame secara terpisah, jaring kecil kami menggabungkan semua nilai frame + embedding posisinya dan memprediksi noise untuk semua frame secara bersamaan.

### Langkah 4: uji koherensi temporal

Setelah training, sample video. Ukur delta frame-ke-frame. Jika model telah mempelajari struktur temporal, delta akan tetap lebih kecil daripada mengambil sample setiap frame secara independen.

## Jebakan

- **Pengambilan sample per bingkai independen = berkedip.** Jika kamu menjalankan difusi gambar pada setiap bingkai secara terpisah, outputnya akan berkedip karena noise setiap bingkai bersifat independen. Difusi video memperbaikinya dengan menggabungkan frame melalui attention atau kebisingan bersama.
- **Attention 3D naif = OOM.** Attention 3D penuh pada laten 1080p 10 detik berarti ratusan miliar operasi. Faktorkan menjadi spasial + temporal.
- **Teks ​​data lebih penting daripada ukuran.** Peningkatan utama Sora dibandingkan pekerjaan sebelumnya adalah melatih ~10x teks yang lebih detail (klip yang diberi label ulang GPT-4). Laporan teknis OpenAI secara eksplisit mengenai hal ini.
- **Pengondisian frame pertama.** Sebagian besar model produksi juga menerima gambar sebagai frame pertama. Ini adalah mode "gambar-ke-video"; training mencakup varian ini.
- **Fisika melayang.** Klip panjang (>10 detik) mengakumulasi ketidakkonsistenan halus. Pembuatan jendela geser + penahan bingkai utama membantu.

## Pakai| Kasus penggunaan | pilihan 2026 |
|----------|-----------|
| Teks-ke-video kualitas tertinggi, dihosting | Veo 3 atau Sora |
| Sinematik yang dikendalikan kamera | Runway Gen-3 dengan kuas gerak |
| Konsistensi karakter di seluruh klip | Pika 2.0 atau Kling 2.1 |
| Weight terbuka, penyesuaian cepat | WAN 2.2 + LoRA |
| Gambar-ke-video | WAN 2.2-I2V, Kling 2.1 I2V, atau Landasan Pacu |
| Sinkronisasi bibir audio-ke-video | Veo 3 (audio asli) atau model sinkronisasi bibir khusus |
| Pengeditan video | Runway Act-Two, Kling Motion Brush, Flux-Kontext (bingkai diam) |

Biaya per detik video dengan kualitas setara telah turun 20x antara tahun 2024 dan 2026.

## Kirim

Simpan `outputs/skill-video-brief.md`. Keterampilan mengambil ringkasan video (durasi, rasio aspek, gaya, rencana kamera, konsistensi subjek, audio) dan output: model + hosting, perancah cepat (bahasa kamera, deskripsi subjek, deskriptor gerakan), seed + protokol reproduksibilitas, dan daftar periksa QA tingkat bingkai.

## Latihan

1. **Mudah.** Di `code/main.py`, bandingkan delta frame-ke-frame untuk (a) pengambilan sample per-frame independen, (b) pengambilan sample urutan gabungan. Laporkan mean dan varians delta.
2. **Medium.** Tambahkan kondisi frame pertama: sematkan frame 0 ke nilai tertentu dan ambil sample sisanya. Ukur bagaimana nilai yang di-embed menyebar.
3. **Sulit.** Gunakan diffuser HuggingFace untuk menjalankan CogVideoX-2B pada GPU lokal. Waktu 20 langkah inference pada 720p untuk klip 6 detik. Profil attention spatiotemporal untuk mengidentifikasi hambatan.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Video VAE | "VAE 3-D" | Encoder yang memampatkan `(T, H, W, C)` → spatiotemporal laten. |
| Tambalan | "Token" | Blok laten 3-D berukuran tetap; input ke DiT. |
| Attention yang difaktorkan | "Spasial + temporal" | Jalankan attention pada ruang, lalu pada waktu; lewati attention 3-D penuh. |
| Gambar-ke-video (I2V) | "Animasikan foto ini" | Model mengambil gambar + teks, mengeluarkan video yang dimulai darinya. |
| Pengkondisian bingkai utama | "Bingkai jangkar" | Sematkan bingkai tertentu untuk mengontrol alur video. |
| Kuas gerak | "Petunjuk arah" | Input UI tempat pengguna melukiskan vector gerakan ke gambar. |
| Membuat teks ulang | "Teks padat" | Menggunakan LLM untuk memberi label ulang pada klip training dengan petunjuk mendetail. |
| Berkedip | "Artefak sementara" | Inkonsistensi frame-to-frame; diperbaiki dengan denoising berpasangan. |

## Catatan produksi: video laten adalah masalah bandwidth memori

Klip 1080p 10 detik pada 24 fps adalah 240 bingkai × 1920 × 1080 × 3 ≈ 1,5 GB piksel mentah. Setelah kompresi VAE video 4× (`2 × spatial × 2 × temporal`) latennya adalah ~100 MB per permintaan. Jalankan ini melalui DiT spatiotemporal selama 30 langkah pada batch 1 dan kamu memindahkan ~3 GB/langkah melalui HBM — bandwidth memori, bukan FLOP, yang menjadi hambatannya.

Tiga tombol produksi, semuanya langsung dari bab inference literatur inference produksi:- **TP di seluruh DiT.** Model teks-ke-video biasanya memiliki parameter ≥10 miliar. TP=4 pada 4 H100 adalah standar; PP=2 × TP=2 untuk model kelas 405B. Latensi per langkah turun secara linier dengan TP hingga dinding pengurangan semua.
- **Pengelompokan bingkai = pengelompokan berkelanjutan.** Pada waktu pembuatan, video secara konseptual merupakan kumpulan bingkai yang dihubungkan oleh attention. Pengelompokan berkelanjutan (penjadwalan dalam penerbangan) berlaku: mulai merender bingkai `t+1` saat bingkai `t-1` dikembalikan, jika arsitektur model memungkinkan pembuatan jendela geser.
- **Cache pra-pengisian tingkat klip.** Untuk gambar-ke-video, pengkondisian bingkai pertama analog dengan pra-pengisian cepat LLM: hitung sekali, gunakan kembali di seluruh lintasan dekoder temporal. Ini secara efektif merupakan cache KV untuk video.

## Bacaan Lanjutan

- [Brooks dkk. (2024). Model pembuatan video sebagai simulator dunia](https://openai.com/index/video-generasi-models-as-world-simulators/) — Laporan teknis Sora.
- [Yang dkk. (2024). CogVideoX: Model Difusi Teks-ke-Video dengan Transformer Ahli](https://arxiv.org/abs/2408.06072) — CogVideoX.
- [Kong dkk. (2024). HunyuanVideo: Kerangka Sistematis untuk Model Generatif Video Besar](https://arxiv.org/abs/2412.03603) — HunyuanVideo.
- [Genmo (2024). Laporan Teknis Mochi-1](https://www.genmo.ai/blog/mochi) — Mochi-1.
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — membuka SOTA pertengahan tahun 2025.
- [Ho, Salimans, Gritsenko dkk. (2022). Model Difusi Video](https://arxiv.org/abs/2204.03458) — makalah difusi video yang penting.
- [Blattmann dkk. (2023). Sejajarkan Laten kamu (LDM Video)](https://arxiv.org/abs/2304.08818) — Nenek moyang Difusi Video Stabil.
