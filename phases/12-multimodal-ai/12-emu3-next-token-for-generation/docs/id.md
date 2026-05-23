# Emu3: Prediksi Token Berikutnya untuk Pembuatan Gambar dan Video

> Emu3 BAAI (Wang et al., September 2024) adalah hasil tahun 2024 yang seharusnya mengakhiri perdebatan difusi versus autoregresif. Transformer khusus dekoder gaya Llama, yang dilatih hanya pada tujuan prediksi token berikutnya, dalam kosakata terpadu teks + token gambar VQ + token video VQ 3D, mengalahkan SDXL dalam pembuatan gambar dan LLaVA-1.6 dalam persepsi. Tidak ada loss CLIP. Tidak ada jadwal difusi. Panduan tanpa pengklasifikasi digunakan pada inference untuk kualitas, namun tujuan training inti adalah prediksi token berikutnya dengan pemaksaan guru. Diterbitkan di Alam. Lesson ini membaca tesis Emu3 — mengapa kamu hanya membutuhkan tokenizer plus skala yang lebih baik — dan kontras dengan pendekatan difusi.

**Type:** Learn
**Language:** Python (stdlib, matematika tokenizer video 3D + kerangka sampler autoregresif)
**Prerequisites:** Fase 12 · 11 (Bunglon)
**Waktu:** ~120 menit

## Tujuan Pembelajaran

- Jelaskan mengapa objektif single-loss next-token Emu3 berhasil meskipun ada asumsi lama bahwa difusi diperlukan untuk kualitas gambar.
- Jelaskan tokenizer video 3D: seperti apa buku code VQ spatiotemporal, mengapa patch membutuhkan waktu.
- Bandingkan Emu3 vs Stable Diffusion XL pada (komputasi training, biaya inference, batas kualitas).
- Sebutkan tiga peran yang dimainkan oleh model Emu3 yang sama: Emu3-Gen (gen gambar), Emu3-Chat (persepsi), Emu3-Stage2 (gen video).

## Masalah

Kebijaksanaan konvensional hingga tahun 2024: pembuatan citra memerlukan difusi. Argumennya: token gambar diskrit kehilangan terlalu banyak informasi untuk merekonstruksi detail, dan pengambilan sample autoregresif mengakumulasi kesalahan di ribuan token. Difusi Stabil, DALL-E 3, Imagen, Midjourney semuanya menggunakan beberapa bentuk difusi. Chameleon (Lesson 12.11) sebagian menyangkal hal ini dalam skala kecil tetapi tidak menandingi SDXL dalam hal kualitas.

Emu3 menyerang argumen tersebut secara langsung. Klaim: tokenizer visual yang lebih baik + skala yang cukup + kehilangan token berikutnya = pembuatan gambar yang mengalahkan difusi dalam model yang sama yang juga melakukan persepsi.

Taruhan itu kontroversial ketika dipublikasikan. Dua tahun kemudian, rangkaian generasi terpadu sumber terbuka (Emu3, Show-o, Janus-Pro, Transfusion) menjadi jalur default untuk penelitian; model batas produksi tampaknya menggunakan beberapa varian.

## Konsep

### Tokenizer Emu3

Bahan utamanya adalah tokenizer visual. Emu3 melatih tokenizer kelas IBQ khusus (Inverse Bottleneck Quantizer, keluarga SBER-MoVQGAN) dengan pengurangan resolusi 8x8 per token. Gambar berukuran 512x512 menjadi 64x64 = 4096 token pada ukuran buku code 32768.

Ini lebih besar dari 1024 token Chameleon per 512x512 pada K=8192 tetapi lebih murah per token (pencarian buku code lebih kecil, codec lebih sederhana). Metrik utamanya: rekonstruksi PSNR pada 30,5 dB, bersaing dengan ruang laten berkelanjutan Difusi Stabil pada 32 dB.

Untuk video: tokenizer VQ 3D mengkodekan patch spatiotemporal (4x4x4 piksel) menjadi satu bilangan bulat. Klip 4s pada 8 FPS memiliki 32 frame; pada 256x256 dengan pengurangan spasial 4x dan pengurangan temporal 4x, jumlah tokennya adalah (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32.768 token.

Kualitas Tokenizer adalah yang tertinggi. Kontribusi Emu3 sebagian adalah "kami melatih tokenizer yang sangat bagus."

### Training satu kekalahan

Emu3 menggunakan satu tujuan: prediksi token berikutnya pada kosakata bersama di seluruh token teks, token gambar 2D, dan token video 3D. Weight dikalikan dengan faktor modalitas spesifik selama training untuk menyeimbangkan kontribusi, namun loss function-nya identik.Berlatihlah dengan campuran:
- Pembuatan gambar: `<text caption> <image> image_tokens </image>`
- Persepsi gambar: `<image> image_tokens </image> <question> text_tokens`
- Pembuatan video: `<text caption> <video> video_tokens </video>`
- Persepsi video: analog.
- Teks saja: NTP standar.

Model ini mempelajari kapan harus memancarkan token gambar vs token teks dari distribusi data. Generasi muncul dari model yang memprediksi token gambar setelah tag `<image>`.

### Panduan dan suhu bebas pengklasifikasi

Pembuatan gambar autoregresif menjadi jauh lebih baik dengan panduan bebas pengklasifikasi (CFG) pada inference. Emu3 menggunakannya: buat dua kali, sekali dengan teks lengkap, sekali dengan teks kosong, campur logit dengan weight panduan (khas 3.0-7.0). Ini adalah penggunaan difusi trik CFG yang sama, dipinjam ke pengaturan autoregresif.

Suhu penting: terlalu tinggi, artefak; terlalu rendah, mode runtuh. Suhu yang disarankan Emu3 adalah 1,0 untuk persepsi, 0,8 untuk pembuatan gambar.

### Tiga peran, satu model

Emu3 dikirimkan sebagai tiga API yang berbeda secara fungsional tetapi satu kumpulan weight yang mendasarinya:

- Emu3-Gen. Pembuatan gambar. Teks input, token gambar output.
- Emu3-Obrolan. VQA dan teks. Gambar input (token), teks output.
- Emu3-Tahap2. Pembuatan video dan VQA video. Masukkan teks atau video, output teks atau video.

Tidak ada kepala khusus tugas. Hanya templat prompt yang berbeda. Pos pemeriksaan yang sama.

### Tolok ukur

Dari makalah Emu3 (September 2024):

- Pembuatan gambar: mengalahkan SDXL pada MJHQ-30K FID (5,4 vs 5,6), GenEval secara keseluruhan (0,54 vs 0,55 — ikatan statistik), dan komposit Deep-Eval setara.
- Persepsi gambar: mengalahkan LLaVA-1.6 di VQAv2 (75.1 vs 72.4) dan kira-kira cocok di MMMU.
- Pembuatan video: kualitas klip 4 detik pada FVD kompetitif dengan model benchmark publik era Sora.

Jumlahnya tidak selalu menang — Emu3 memperdagangkan satu poin di sini dengan satu poin di sana — tetapi klaim "yang kamu perlukan hanya prediksi token berikutnya" dapat dipertahankan di seluruh modalitas.

### Hitung biaya

Emu3 dilatih pada ~300 miliar token multimodal dengan model parameter 7B. Jam GPU kira-kira sebanding dengan pra-training Llama-2-7B (2k-4k GPU-tahun pada silikon kelas A100). Model difusi seperti Difusi Stabil 3 dilatih dengan anggaran yang sama tetapi memerlukan pembuat enkode teks terpisah dan pipeline yang lebih kompleks.

Sebagai kesimpulan, Emu3 lebih lambat dibandingkan SDXL per gambar: 4096 token gambar pada 30 tok/s adalah ~2 menit per gambar 512x512, vs 2-5 detik untuk SDXL. Penguraian code spekulatif dan optimalisasi cache KV mempersempit kesenjangan tetapi tidak menutupnya. Gen gambar autoregresif membutuhkan banyak komputasi; ini adalah trade-off yang tetap.

### Mengapa itu penting

Kontribusi mendalam Emu3 bersifat konseptual. Jika prediksi token berikutnya berskala agar sesuai dengan difusi pada pembuatan gambar, jalur model terpadu (satu loss, satu tulang punggung, modalitas apa pun) dapat dijalankan. Model masa depan tidak memerlukan encoder teks terpisah, penjadwal difusi terpisah, VAE terpisah. Satu Transformer, satu tokenizer per modalitas, skala.

Show-o, Janus-Pro, dan InternVL-U semuanya membangun atau menantang tesis ini. Laboratorium Tiongkok (BAAI, DeepSeek) melakukan publikasi lebih agresif ke arah ini dibandingkan laboratorium AS pada tahun 2025.

## Pakai

`code/main.py` membuat dua buah mainan:

- Kalkulator jumlah tokenizer VQ 2D vs 3D: diberikan (resolusi, patch, clip_length, FPS), hitung jumlah token untuk gambar vs video.
- Sampler token gambar autoregresif dengan panduan bebas pengklasifikasi pada suhu.

Implementasi CFG cocok dengan resep Emu3 — menggabungkan logit bersyarat dan tidak bersyarat dengan weight panduan.

## Kirim

Lesson ini menghasilkan `outputs/skill-token-gen-cost-analyzer.md`. Berdasarkan spesifikasi produk generasi (gambar atau video, resolusi target, tingkat kualitas, anggaran latensi), ini menghitung jumlah token, biaya inference, dan memilih keluarga Emu3 vs difusi.

## Latihan

1. Emu3 menghasilkan 4096 token per gambar 512x512 dengan pengurangan 8x8. Hitung persamaan untuk 1024x1024 dan 2048x2048. Apa yang terjadi dengan latensi inference?

2. Baca Emu3 Bagian 3.3 pada video tokenizer. Jelaskan bentuk patch VQ 3D dan mengapa 4x4x4 bukan 8x8x1.

3. Panduan bebas pengklasifikasi weight 5.0 vs 3.0: apa efek visualnya? Lacak perhitungannya di `code/main.py`.

4. Hitung FLOP training untuk Emu3-7B dengan token 300B dan bandingkan dengan Difusi Stabil 3. Mana yang lebih mahal untuk dilatih?

5. Emu3 mengalahkan SDXL pada FID tetapi tidak pada VQAv2 vs VLM khusus. Jelaskan mengapa pendekatan loss terpadu menunjukkan kekuatan yang berbeda vs spesialis pada tolok ukur yang berbeda.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Prediksi token berikutnya | "NTP" | Loss autoregresif standar: prediksi token[i+1] diberikan token[0..i]; berfungsi untuk setiap modalitas saat diberi token |
| Tokenizer IQ | "Pengukur kemacetan terbalik" | Kelas VQ-VAE dengan buku code yang lebih besar (32768+) dan rekonstruksi yang lebih baik daripada Chameleon |
| VQ 3D | "Pengukur spasialtemporal" | Buku code diindeks berdasarkan (waktu, baris, kolom); satu token menutupi kubus piksel 4x4x4 |
| Panduan bebas pengklasifikasi | "CFG" | Campurkan log bersyarat dan tidak bersyarat dengan gamma weight; meningkatkan kualitas gambar pada inference |
| Kosakata terpadu | "Token bersama" | Teks + gambar + video semuanya diambil dari ruang bilangan bulat yang sama; model memprediksi modalitas mana yang muncul berikutnya |
| MJHQ-30K | "Patokan gen gambar" | Tolok ukur kualitas tengah perjalanan dengan 30 ribu permintaan; Emu3 melaporkan FID di sini |

## Bacaan Lanjutan

- [Wang dkk. — Emu3: Yang kamu Butuhkan hanyalah Prediksi Token Berikutnya (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Matahari dkk. — Emu: Pra-training Generatif dalam Multimodalitas (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu dkk. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu dkk. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian dkk. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
