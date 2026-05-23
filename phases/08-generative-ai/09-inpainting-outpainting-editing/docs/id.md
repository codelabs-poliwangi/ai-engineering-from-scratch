# Inpainting, Outpainting & Pengeditan Gambar

> Text-to-image membuat hal-hal baru. Pengecatan ulang memperbaiki yang lama. Dalam produksi, 70% pekerjaan gambar yang dapat ditagih adalah pengeditan — menukar latar belakang, menghapus logo, memperluas kanvas, membuat ulang tangan. Inpainting adalah tempat difusi mendapatkan tempatnya.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 8 · 07 (Difusi Laten), Fase 8 · 08 (ControlNet & LoRA)
**Waktu:** ~75 menit

## Masalah

Seorang klien mengirimkan foto produk yang sempurna dengan tanda yang mengganggu di latar belakang. kamu ingin menghapus tandanya dan membiarkan yang lainnya tetap identik dengan piksel. kamu tidak dapat menjalankan text-to-image dari awal — hasilnya akan memiliki warna berbeda, pencahayaan berbeda, sudut produk berbeda. kamu ingin membuat ulang *hanya* wilayah bertopeng, dan kamu ingin regenerasi menghormati konteks sekitarnya.

Itu adalah melukis. Varian:

- **Inpainting.** Regenerasi di dalam topeng, pertahankan piksel luar.
- **Outpainting.** Regenerasi di luar topeng (atau di luar kanvas), tetap di dalam.
- **Pengeditan gambar.** Buat ulang seluruh gambar tetapi pertahankan ketelitian semantik atau struktural dengan aslinya (SDEdit, InstructPix2Pix).

Setiap jalur pipa difusi pada tahun 2026 mengirimkan mode pengecatan. Flux.1-Fill, Inpaint Difusi Stabil, SDXL-Inpaint, DALL-E 3 Edit. Mereka bekerja dengan prinsip yang sama.

## Konsep

![Inpainting: penolakan terhadap masker dengan injeksi ulang yang menjaga konteks](../assets/inpainting.svg)

### Pendekatan naif (dan mengapa itu salah)

Jalankan text-to-image standar dengan mask. Pada setiap langkah pengambilan sample, ganti wilayah noise laten yang terbuka kedoknya dengan gambar bersih yang tersebar ke depan. Ini berhasil... buruk. Artefak batas akan hilang karena model tidak memiliki informasi tentang apa yang ada di wilayah yang disamarkan.

### Model pengecatan yang tepat

Latih U-Net yang dimodifikasi yang menggunakan 9 pipeline input, bukan 4:

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

Pipeline tambahan adalah salinan gambar sumber yang dikodekan VAE ditambah masker pipeline tunggal. Pada waktu training, kamu secara acak menutupi bagian gambar dan melatih model untuk hanya menghilangkan noise pada bagian yang ditutupi, sementara bagian yang tidak tertutup diberikan sebagai sinyal pengondisian bersih. Pada inference, model dapat "melihat" apa yang mengelilingi wilayah yang disamarkan dan menghasilkan penyelesaian yang koheren.

SD-Inpaint, SDXL-Inpaint, Flux-Fill semuanya menggunakan input 9 pipeline (atau analog) ini. Diffuser `StableDiffusionInpaintPipeline`, `FluxFillPipeline`.

### SDEdit (Meng et al., 2022) — pengeditan gratis

Tambahkan noise ke gambar sumber hingga `t` perantara, lalu jalankan rantai terbalik dari `t` ke 0 dengan prompt baru. Tidak ada training ulang. Pilihan untuk memulai `t` memperdagangkan kesetiaan demi kebebasan berkreasi:

- `t/T = 0.3` → hampir identik dengan sumber, sedikit perubahan gaya
- `t/T = 0.6` → pengeditan moderat, mempertahankan struktur kasar
- `t/T = 0.9` → dihasilkan dari pelestarian sumber yang nyaris derau dan minimal

### InstructPix2Pix (Brooks dkk., 2023)

Sempurnakan model difusi pada `(input_image, instruction, output_image)` tiga kali lipat. Sebagai kesimpulan, kondisikan pada gambar input dan instruksi teks ("buat matahari terbenam", "tambahkan naga"). Dua skala CFG: skala gambar dan skala teks.

### Pengecatan Ulang (Lugmayr dkk., 2022)

Pertahankan model difusi standar tanpa syarat. Pada setiap langkah mundur, ambil sample ulang - sesekali lompat kembali ke kondisi yang lebih berisik dan lakukan regenerasi. Menghindari artefak batas. Digunakan ketika kamu tidak memiliki model melukis yang terlatih.

## Build`code/main.py` mengimplementasikan skema pengecatan 1-D mainan pada data 5 dimension. Kami melatih DDPM pada data campuran 5-D di mana setiap sample terdiri dari 5 float dari salah satu dari dua cluster. Sebagai kesimpulan, kami "menutupi" 2 dari 5 dimension, menyuntikkan versi noise-forward dari tiga dimension yang terbuka kedoknya di setiap langkah, dan hanya membuat ulang dimension yang disamarkan.

### Langkah 1: Data DDPM 5-D

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Langkah 2: latih denoiser pada 5 redup

DDPM Standar. Prediksi kebisingan 5-D output bersih untuk input bising 5-D.

### Langkah 3: pada inference, kebalikan dari kesadaran akan topeng

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

Ini adalah pendekatan naif dan berhasil pada data mainan 1-D. Inpainting gambar nyata menggunakan input 9 pipeline karena koherensi tekstur lebih penting.

### Langkah 4: pengecatan luar

Outpainting adalah pengecatan dengan topeng terbalik: menutupi kanvas baru (yang sebelumnya tidak ada), mengisi sisanya dengan yang asli. Tujuan training yang identik.

## Jebakan

- **Seams.** Pendekatan naif meninggalkan batas yang terlihat karena info gradient tidak mengalir melintasi mask. Cara mengatasinya: melebarkan topeng sebanyak 8-16 piksel, atau menggunakan model pengecatan yang tepat.
- **Kebocoran masker.** Jika bagian gambar pengkondisian yang terbuka kedoknya berkualitas rendah atau berisik, hal ini akan mencemari generasi di dalam masker. Tolak atau buramkan sedikit.
- **CFG berinteraksi dengan ukuran masker.** CFG tinggi pada masker kecil = patch jenuh. Kurangi CFG untuk pengeditan kecil.
- **SDEdit fidelity cliff.** Beralih dari `t/T = 0.5` ke `t/T = 0.6` dapat menghilangkan identitas subjek. Sapu dan pos pemeriksaan.
- **Permintaan tidak cocok.** Prompt harus mendeskripsikan *keseluruhan* gambar, bukan hanya konten baru. "Seekor kucing duduk di kursi" bukan "kucing".

## Pakai

| Tugas | Pipeline pipa |
|------|----------|
| Hapus objek, topeng kecil | SD-Inpaint atau Flux-Fill, prompt standar |
| Ganti langit | SD-Inpaint + "langit biru saat matahari terbenam" |
| Perpanjang kanvas | Mode outpaint SDXL (bulu 8px) atau Flux-Fill dengan outpaint mask |
| Meregenerasi tangan/wajah | SD-Inpaint dengan prompt mendeskripsikan ulang subjek + ControlNet-Openpose |
| Ubah gaya satu wilayah | SDEdit di `t/T=0.5` di wilayah bertopeng |
| "Jadikan matahari terbenam" | InstructPix2Pix atau Flux-Kontext |
| Penggantian latar belakang | Masker SAM → SD-Inpaint |
| Fidelitas ultra-tinggi | Flux-Fill atau GPT-Image (dihosting) untuk kasus tersulit |

SAM (Meta's Segment Anything, 2023) + diffusion inpaint adalah jalur penghapusan latar belakang tahun 2026. SAM 2 (2024) berfungsi pada video.

## Kirim

Simpan `outputs/skill-editing-pipeline.md`. Keterampilan mengambil gambar asli + deskripsi edit + topeng opsional (atau prompt SAM) dan output: pendekatan pembuatan topeng, model dasar, skala CFG (gambar + teks), mode SDEdit-t atau inpainting, dan daftar periksa QA.

## Latihan

1. **Mudah.** Di `code/main.py`, variasikan pecahan dimension yang ditutupi dari 0,2 hingga 0,8. Pada bagian berapakah kualitas cat (sisa dalam keremangan terselubung) setara dengan generasi tanpa syarat?
2. **Sedang.** Terapkan Pengecatan Ulang: pada setiap langkah mundur ke-10, lompat mundur 5 langkah (tambahkan noise) dan deoise ulang. Ukur apakah tindakan tersebut mengurangi sisa batas pada tepi masker.
3. **Hard.** Gunakan Hugging Face diffuser untuk membandingkan: SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill pada 20 tugas regenerasi wajah. Skor menimbulkan kepatuhan dan pelestarian identitas secara terpisah.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Lukisan | "Isi lubangnya" | Regenerasi di dalam topeng; pertahankan piksel luar. |
| Pengecatan | "Perluas kanvas" | Regenerasi di luar kanvas; tetap di dalam. |
| U-Net 9 pipeline | "Model pengecatan yang tepat" | U-Net dengan `noisy | encoded-source | mask` sebagai input. |
| SDSunting | "Img2img dengan tingkat kebisingan" | Kebisingan seiring waktu `t`, hentikan dengan prompt baru. |
| InstruksikanPix2Pix | "Pengeditan hanya teks" | Difusi yang disempurnakan pada (gambar, instruksi, output) tiga kali lipat. |
| Cat Ulang | "Tidak ada training ulang" | Bunyikan kembali secara berkala saat mundur untuk mengurangi jahitan. |
| SAM | "Segmen Apa Saja" | Generator topeng dengan klik atau kotak; berpasangan dengan cat. |
| Fluks-Konteks | "Edit dengan konteks" | Varian Flux yang menerima gambar referensi + instruksi untuk mengedit. |

## Catatan produksi: alur edit sensitif terhadap latensi

Pengguna yang mengedit gambar mengharapkan perjalanan bolak-balik kurang dari 5 detik. SDXL-Inpaint 30 langkah pada 1024² membutuhkan waktu 3-4 detik pada L4, ditambah pembuatan masker SAM (~200 mdtk) dan enkode/dekode VAE (gabungan ~500 mdtk). Dalam pembingkaian produksi, ini terikat TTFT dan bukan terikat throughput — batch 1, konkurensi rendah, minimalkan setiap tahapan:

- **SAM-H adalah yang lambat.** SAM-H pada 1024² berdurasi ~200 mdtk; SAM-ViT-B berdurasi ~40 ms dengan sedikit penurunan kualitas. SAM 2 (video) menambahkan overhead sementara; jangan gunakan untuk pengeditan gambar tunggal.
- **Lewati pengkodean jika memungkinkan.** `pipe.image_processor.preprocess(img)` mengkodekan ke laten. Jika kamu memiliki laten dari generasi sebelumnya (umumnya pada UI edit berulang), teruskan langsung melalui `latents=...` untuk melewati satu enkode VAE.
- **Dilatasi mask juga penting untuk throughput.** Mask kecil berarti sebagian besar forward pass U-Net terbuang (piksel yang terbuka kedoknya tetap dijepit). `diffusers`' `StableDiffusionInpaintPipeline` tetap menjalankan U-Net secara penuh; hanya varian inpaint 9 pipeline yang mengeksploitasi komputasi bertopeng.
- **Flux-Kontext adalah jawaban tahun 2025.** Single forward pass over `(source_image, instruction)` — tanpa mask terpisah, tanpa sapuan noise SDEdit. Pada H100 ia mengirimkan hasil edit dalam ~1,5 detik. Lesson arsitektur: runtuhkan tahapannya.

## Bacaan Lanjutan

- [Lugmayr dkk. (2022). RePaint: Pengecatan Ulang menggunakan Model Probabilistik Difusi Denoising](https://arxiv.org/abs/2201.09865) — pengecatan ulang tanpa training.
- [Meng dkk. (2022). SDEdit: Sintesis dan Pengeditan Gambar Terpandu dengan Persamaan Diferensial Stokastik](https://arxiv.org/abs/2108.01073) — SDEdit.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — pengeditan instruksi teks.
- [Kirillov dkk. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM, sumber topeng.
- [Ravi dkk. (2024). SAM 2: Segmentasikan Apa Pun dalam Gambar dan Video](https://arxiv.org/abs/2408.00714) — video SAM.
- [Hertz dkk. (2022). Pengeditan Gambar Prompt-to-Prompt dengan Kontrol Lintas Attention](https://arxiv.org/abs/2208.01626) — pengeditan tingkat attention.
- [Laboratorium Hutan Hitam (2024). Flux.1-Fill dan Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — perkakas 2024.
