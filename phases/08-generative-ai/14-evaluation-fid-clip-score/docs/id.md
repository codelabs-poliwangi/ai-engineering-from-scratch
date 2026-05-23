# Evaluasi — FID, Skor CLIP, Preferensi Manusia

> Setiap papan peringkat model generatif mengutip FID, skor CLIP, dan tingkat kemenangan dari arena preferensi manusia. Setiap angka memiliki mode kegagalan yang dapat dimainkan oleh peneliti yang gigih. Jika kamu tidak mengetahui mode kegagalannya, kamu tidak dapat mengetahui peningkatan nyata dari permainan yang dijalankan.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 8 · 01 (Taksonomi), Phase 2 · 04 (Metrik Evaluasi)
**Waktu:** ~45 menit

## Masalah

Model generatif dinilai berdasarkan *kualitas sample* dan *kepatuhan pengkondisian*. Tidak ada satupun yang mempunyai ukuran tertutup. Model kamu harus merender 10.000 gambar; sesuatu harus memberi mereka nomor; kamu harus memercayai angka-angka di seluruh kelompok model, di seluruh resolusi, di seluruh arsitektur. Tiga metrik selamat dari tantangan tahun 2014-2026:

- **FID (Fréchet Inception Distance).** Distance antara dua distribusi — nyata dan dihasilkan — dalam ruang feature jaringan Inception. Lebih rendah lebih baik.
- **Skor CLIP.** Kemiripan kosinus antara embedding gambar CLIP pada gambar yang dihasilkan dan embedding teks CLIP pada prompt. Lebih tinggi lebih baik. Mengukur kepatuhan yang cepat.
- **Preferensi manusia.** Adu dua model secara langsung pada prompt yang sama, minta manusia (atau model kelas GPT-4) memilih yang lebih baik, lalu gabungkan menjadi skor Elo.

kamu juga akan melihat: IS (skor awal, sebagian besar dihentikan), KID, CMMD, ImageReward, PickScore, HPSv2, MJHQ-30k. Masing-masing mengoreksi satu kegagalan sebelumnya.

## Konsep

![FID, CLIP, dan preferensi: tiga sumbu, mode kegagalan berbeda](../assets/evaluation.svg)

### FID — kualitas sample

Heusel dkk. (2017). Langkah-langkah:

1. Ekstrak feature Inception-v3 (2048-D) untuk N gambar nyata dan N yang dihasilkan.
2. Cocokkan Gaussian ke setiap kumpulan: hitung mean `μ_r, μ_g` dan kovarians `Σ_r, Σ_g`.
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`.

Interpretasi: Distance Fréchet antara dua Gauss multivariat di ruang feature. Lebih rendah = lebih banyak distribusi serupa.

Mode kegagalan:
- **Bias pada N kecil.** FID dikuadratkan pada distribusi feature — N kecil meremehkan kovarians, menghasilkan FID yang sangat rendah. Selalu gunakan N ≥ 10.000.
- **Inception-dependent.** Inception-v3 dilatih di ImageNet. Domain yang jauh dari ImageNet (wajah, seni, gambar teks) menghasilkan FID yang tidak berarti. Gunakan ekstraktor feature khusus domain.
- **Gaming.** Overfitting ke Inception sebelumnya memberikan FID rendah tanpa peningkatan kualitas visual. Kalahkan dengan CMMD (di bawah).

### Skor CLIP — kepatuhan yang cepat

Radford dkk. (2021). Untuk gambar + prompt yang dihasilkan:

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

Rata-rata dari 30 ribu gambar yang dihasilkan → scalar yang sebanding antar model.

Mode kegagalan:
- **Titik buta CLIP sendiri.** CLIP memiliki penalaran komposisi yang lemah ("kubus merah pada bola biru" sering gagal). Model dapat mendapat peringkat yang baik pada skor CLIP tanpa benar-benar mengikuti prompt yang rumit.
- **Bias prompt singkat.** Prompt singkat memiliki lebih banyak kecocokan gambar CLIP di alam liar. Prompt yang lebih panjang memiliki skor CLIP yang lebih rendah secara mekanis.
- **Permainan cepat.** Menyertakan "kualitas tinggi, 4k, mahakarya" dalam prompt akan meningkatkan skor CLIP tanpa meningkatkan pengikatan gambar-teks.

CMMD (Jayasumana dkk., 2024) memperbaiki beberapa hal berikut: menggunakan feature CLIP sebagai pengganti Inception, perbedaan rata-rata maksimum sebagai ganti Fréchet. Lebih baik dalam mendeteksi perbedaan kualitas yang halus.

### Preferensi manusia — kebenaran dasar

Pilih kumpulan petunjuk. Hasilkan dengan model A dan model B. Tunjukkan pasangan kepada manusia (atau juri LLM yang kuat). Kemenangan agregat menjadi skor Elo atau Bradley-Terry. Tolok ukur:- **PartiPrompts (Google)**: 1.600 prompt beragam, 12 kategori.
- **HPSv2**: 107 ribu anotasi manusia, banyak digunakan sebagai proxy otomatis.
- **ImageReward**: 137 ribu pasangan preferensi gambar cepat, berlisensi MIT.
- **PickScore**: dilatih tentang preferensi Pick-a-Pic 2,6 juta.
- **Arena gambar bergaya Chatbot-Arena**: https://imagearena.ai/ dan lainnya.

Mode kegagalan:
- **Variansitas hakim.** Non-ahli memiliki preferensi yang berbeda dengan pakar. Gunakan keduanya.
- **Distribusi cepat.** Prompt pilihan ceri menguntungkan satu keluarga. Selalu dokumentasikan.
- **Hadiah peretasan juri LLM.** Hakim GPT-4 tertipu oleh hasil yang cantik tapi salah. Lakukan triangulasi dengan manusia.

## Gunakan bersama-sama

Laporan evaluasi produksi harus mencakup:

1. FID pada 10-30 ribu sample dibandingkan dengan distribusi riil yang ada (kualitas sample).
2. Skor CLIP / CMMD pada sample yang sama vs petunjuknya (kepatuhan).
3. Tingkat kemenangan di arena buta vs model sebelumnya (preferensi keseluruhan).
4. Analisis mode kegagalan: 50 output sample acak, ditandai untuk masalah umum (anatomi tangan, rendering teks, jumlah objek yang konsisten).

Metrik apa pun adalah sebuah kebohongan. Tiga metrik yang menguatkan + tinjauan kualitatif adalah sebuah klaim.

## Build

`code/main.py` mengimplementasikan agregasi FID, CLIP-score-like, dan Elo pada "vector feature" sintetik (kami menggunakan vector 4-D sebagai pengganti feature Inception). kamu melihat:

- Perhitungan FID pada N kecil dan N besar — bias.
- "Skor CLIP" sebagai kesamaan kosinus antara kumpulan feature.
- Aturan pembaruan Elo dari aliran preferensi sintetis.

### Langkah 1: FID dalam empat baris

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### Langkah 2: Kemiripan kosinus gaya CLIP

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### Langkah 3: Agregasi Elo

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## Jebakan

- **FID pada N=1000.** Heuristik tidak dapat diandalkan pada N=10k. Makalah yang melaporkan FID rendah adalah permainan.
- **Membandingkan FID di seluruh resolusi.** Pengubahan ukuran 299×299 Inception mengubah distribusi feature. Bandingkan pada resolusi yang cocok saja.
- **Melaporkan satu benih.** Jalankan minimal 3 benih. Laporkan std.
- **Inflasi skor CLIP melalui prompt negatif.** Beberapa pipeline pipa meningkatkan CLIP dengan menyesuaikan prompt secara berlebihan. Periksa saturasi visual.
- **Bias Elo dari prompt yang tumpang tindih.** Jika kedua model melihat prompt benchmark selama training, Elo tidak ada artinya. Gunakan set prompt yang ditunda.
- **Kemiringan manusia yang dibayar oleh orang banyak.** Anotator MTurk yang produktif cenderung lebih muda/ramah teknologi. Campurkan dengan pakar seni/desain yang direkrut.

## Pakai

Protokol evaluasi produksi pada tahun 2026:

| Pilar | Minimal | Direkomendasikan |
|--------|---------|-------------|
| Kualitas sample | FID pada 10k vs real yang diadakan | + CMMD pada 5k + FID pada subset per kategori |
| Kepatuhan yang cepat | Skor CLIP pada 30k | + HPSv2 + ImageReward + Jawaban pertanyaan bergaya VQA |
| Preferensi | 200 pasangan buta vs baseline | + 2000 manusia berpasangan + juri LLM + Chatbot Arena |
| Analisis kegagalan | 50 berbendera tangan | 500 pengklasifikasi keamanan berbendera tangan + otomatis |

Keempat pilar dalam satu laporan = klaim. Siapa pun sendirian = pemasaran.

## Kirim

Simpan `outputs/skill-eval-report.md`. Keterampilan mengambil pos pemeriksaan model baru + garis dasar dan menghasilkan rencana evaluasi lengkap: ukuran sample, metrik, pemeriksaan mode kegagalan, kriteria penandatanganan.

## Latihan1. **Mudah.** Jalankan `code/main.py`. Bandingkan FID pada N=100 vs N=1000 pada distribusi sintetik yang sama. Laporkan besarnya bias.
2. **Medium.** Menerapkan CMMD dari feature gaya CLIP sintetis (lihat Jayasumana dkk., 2024 untuk mengetahui rumusnya). Bandingkan sensitivitas terhadap perbedaan kualitas vs FID.
3. **Sulit.** Replikasi penyiapan HPSv2: ambil 1000 pasangan gambar-prompt dari subset Pick-a-Pic, sempurnakan pencetak skor kecil berbasis CLIP pada preferensi, dan ukur kesesuaiannya dengan set yang ditahan.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| FID | "Distance Awal Fréchet" | Distance Fréchet Gaussian cocok dengan feature Inception nyata vs gen. |
| Skor KLIP | "Kesamaan teks-gambar" | Kemiripan kosinus antara gambar CLIP dan embedding teks. |
| CMMD | "Pengganti FID" | MMD feature CLIP; kurang bias, tidak ada asumsi Gaussian. |
| ADALAH | "Skor awal" | Exp KL(p(y|x) || p(y)); berkorelasi buruk dengan model modern, sudah pensiun. |
| HPSv2 / ImageReward / PickScore | "Proksi preferensi yang dipelajari" | Model kecil dilatih berdasarkan preferensi manusia; digunakan sebagai juri otomatis. |
| Elo | "Peringkat catur" | Agregasi kemenangan berpasangan Bradley-Terry. |
| Prompt Parti | "Permintaan patokan ditetapkan" | 1.600 prompt yang dikurasi Google di 12 kategori. |
| FD-DINO | "Penggantian mandiri" | FD menggunakan feature DINOv2; lebih baik untuk domain di luar ImageNet. |

## Catatan produksi: evaluasi juga merupakan weight kerja inference

Menjalankan FID pada 10 ribu sample berarti menghasilkan 10 ribu gambar. Untuk basis SDXL 50 langkah pada 1024² pada satu L4, itu berarti ~11 jam inference permintaan tunggal. Anggaran evaluasi adalah nyata, dan penyusunannya persis seperti skenario inference offline (maksimalkan throughput, abaikan TTFT):

- **Batch keras, lupakan latensi.** Offline eval = pengelompokan statis pada ukuran terbesar yang muat dalam memori. `pipe(...).images` dengan `num_images_per_prompt=8` pada H100 80GB menjalankan jam dinding 4-6× lebih cepat dibandingkan permintaan tunggal.
- **Cache feature sebenarnya.** Ekstraksi feature Inception (FID) atau CLIP (CLIP-score, CMMD) pada kumpulan referensi sebenarnya dijalankan *sekali*, disimpan sebagai `.npz`. Jangan menghitung ulang per eval.

Untuk gerbang CI/regresi: jalankan skor FID + CLIP pada subset 500 sample per PR (~30 menit); jalankan 10rb FID + HPSv2 + Elo penuh setiap malam.

## Bacaan Lanjutan

- [Heusel dkk. (2017). GAN Dilatih dengan Aturan Pembaruan Dua Skala Waktu Menyatu ke Ekuilibrium Nash Lokal (FID)](https://arxiv.org/abs/1706.08500) - Makalah FID.
- [Jayasumana dkk. (2024). Memikirkan Kembali FID: Menuju Metrik Evaluasi yang Lebih Baik untuk Pembuatan Citra (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD.
- [Radford dkk. (2021). Mempelajari Model Visual yang Dapat Dipindahtangankan dari Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP.
- [Wu dkk. (2023). HPSv2: Skor Preferensi Manusia yang Komprehensif](https://arxiv.org/abs/2306.09341) — HPSv2.
- [Xu dkk. (2023). ImageReward: Mempelajari dan Mengevaluasi Preferensi Manusia untuk Pembuatan Teks-ke-Gambar](https://arxiv.org/abs/2304.05977) — ImageReward.
- [Yu dkk. (2023). Menskalakan Model Autoregresif untuk Pembuatan Teks-ke-Gambar yang Kaya Konten (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts.
- [Stein dkk. (2023). Mengungkap kelemahan metrik evaluasi model generatif](https://arxiv.org/abs/2306.04675) — survei mode kegagalan.
