# Difusi Laten & Difusi Stabil

> Difusi ruang piksel pada gambar 512×512 adalah kejahatan perang komputasi. Rombach dkk. (2022) menyadari bahwa kamu tidak memerlukan seluruh dimension 786k untuk menghasilkan gambar — kamu memerlukan dimension yang cukup untuk menangkap struktur semantik, dan dekoder terpisah untuk sisanya. Jalankan difusi di dalam ruang laten VAE. Ide yang satu itu adalah Difusi Stabil.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 8 · 02 (VAE), Fase 8 · 06 (DDPM), Fase 7 · 09 (ViT)
**Waktu:** ~75 menit

## Masalah

Difusi ruang piksel pada 512² berarti U-Net berjalan pada tensor bentuk `[B, 3, 512, 512]`. Setiap langkah pengambilan sample adalah ~100 GFLOPS untuk U-Net 500 juta parameter. Lima puluh langkah adalah 5 TFLOPS per gambar. Melatih satu miliar gambar dan tagihan komputasinya tidak masuk akal.

Sebagian besar FLOP tersebut digunakan untuk mendorong detail-detail yang dianggap tidak penting melalui internet - tekstur frekuensi tinggi yang dapat dikompres oleh VAE yang lossy. Ide Rombach: latih VAE satu kali (*phase pertama*), bekukan, dan jalankan difusi seluruhnya dalam ruang laten 4 pipeline 64×64 (*phase kedua*). U-Net yang sama. 1/16 piksel. ~64x lebih sedikit FLOP untuk kualitas yang sebanding.

Ini adalah resep Difusi Stabil. SD 1.x / 2.x menggunakan U-Net 860M melalui `64×64×4` laten, SDXL menggunakan U-Net 2,6B melalui `128×128×4`, SD3 menukar U-Net dengan Diffusion Transformer (DiT) dengan pencocokan aliran. Flux.1-dev (Black Forest Labs, 2024) mengirimkan DiT-MMDiT 12B-param. Semua dijalankan pada substrat dua phase yang sama.

## Konsep

![Difusi laten: kompresi VAE + difusi dalam ruang laten](../assets/latent-diffusion.svg)

**Dua phase, dilatih secara terpisah.**

1. **Phase 1 — VAE.** Encoder `E(x) → z`, decoder `D(z) → x`. Kompresi target: 8× downsample di setiap sumbu spasial + sesuaikan pipeline sehingga total ukuran laten adalah ~1/16 jumlah piksel. Loss = rekonstruksi (L1 + LPIPS perceptual) + KL (weight kecil jadi `z` tidak dipaksakan terlalu Gaussian, karena tidak perlu pengambilan sample yang tepat dari `z`). Seringkali dilatih dengan kekalahan permusuhan sehingga gambar yang diterjemahkan menjadi tajam.

2. **Phase 2 — difusi pada `z`.** Perlakukan `z = E(x_real)` sebagai data. Latih U-Net (atau DiT) untuk menolak `z_t`. Pada inference: sample `z_0` melalui difusi, lalu `x = D(z_0)`.

**Pengondisian teks.** Dua komponen tambahan. Encoder teks beku (CLIP-L untuk SD 1.x, CLIP-L+OpenCLIP-G untuk SD 2/XL, T5-XXL untuk SD3 dan Flux). Injeksi attention silang: setiap blok U-Net mengambil `[Q = image features, K = V = text tokens]` dan menggabungkannya. Token adalah satu-satunya cara teks memengaruhi gambar.

**Loss function identik dengan Lesson 06.** DDPM / MSE pencocokan aliran yang sama pada kebisingan. kamu tinggal menukar domain datanya.

## Varian arsitektur

| Model | Tahun | Tulang punggung | Bentuk laten | Pembuat enkode teks | Param |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 token) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + pemurni | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | sulingan | 128×128×4 | sama | Pengambilan sample 1-4 langkah |
| SD3 | 2024 | MMDiT (DiT multimodal) | 128×128×16 | T5-XXL + KLIP-L + KLIP-G | 2B / 8B |
| Fluks.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + KLIP-L | 12B |
| Fluks.1-schnell | 2024 | sulingan MMDiT | 128×128×16 | T5-XXL + KLIP-L | 12B, 1-4 langkah |Trennya: ganti U-Net dengan DiT (Transformer di atas patch laten), skalakan encoder teks (T5 mengalahkan CLIP untuk kepatuhan yang cepat), tingkatkan pipeline laten (4 → 16 memberikan ruang kepala yang lebih detail).

## Build

`code/main.py` menumpuk mainan "VAE" 1-D (encoder + decoder identitas, untuk demonstrasi; VAE asli akan menjadi jaringan konv) di atas DDPM dari Lesson 06 dan menambahkan pengondisian kelas dengan panduan bebas pengklasifikasi. Hal ini menunjukkan bahwa loss difusi yang sama berfungsi baik kamu menjalankan nilai 1-D mentah atau nilai yang dikodekan — wawasan utama.

### Langkah 1: pembuat enkode/dekoder

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

VAE sejati telah melatih weight. Untuk pedagogi, peta linier ini cukup untuk menunjukkan bahwa difusi beroperasi pada `z` tanpa mempedulikan ruang data asli.

### Langkah 2: difusi di `z`-space

DDPM yang sama dengan Lesson 06. Data yang dilihat internet adalah `z = E(x)`. Setelah pengambilan sample `z_0`, dekode dengan `D(z_0)`.

### Langkah 3: panduan bebas pengklasifikasi

Selama training, hilangkan label kelas sebanyak 10% (ganti dengan token nol). Pada inference, hitung `ε_cond` dan `ε_uncond`, lalu:

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = tidak ada panduan (keberagaman penuh), `w = 3` = default, `w = 7+` = jenuh / terlalu tajam.

### Langkah 4: pengondisian teks (konsep, bukan code)

Ganti label kelas dengan output encoder teks yang dibekukan. Masukkan teks yang di-embed ke U-Net melalui attention silang:

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

Inilah satu-satunya perbedaan substantif antara model difusi bersyarat kelas dan Difusi Stabil.

## Jebakan

- **Ketidakcocokan skala VAE.** SD 1.x VAE memiliki konstanta penskalaan (`scaling_factor ≈ 0.18215`) yang diterapkan setelah pengkodean. Melupakan hal ini membuat U-Net berlatih secara laten dengan varian yang sangat salah. Setiap pos pemeriksaan mengirimkan satu.
- **Encoder teks salah secara diam-diam.** SD3 memerlukan T5-XXL dengan >=128 token, dan penggantian ke CLIP saja bersifat lossy. Selalu periksa `use_t5=True` atau konfirmasikan kawah fidelitas.
- **Mencampur ruang laten.** SDXL, SD3, Flux semuanya menggunakan VAE yang berbeda. LoRA yang dilatih pada SDXL laten tidak akan berfungsi pada SD3. Hugging Face diffusers 0,30+ menolak memuat pos pemeriksaan yang tidak cocok.
- **CFG terlalu tinggi.** `w > 10` menghasilkan gambar yang jenuh dan berminyak serta terlalu menyesuaikan tampilan dengan mengorbankan keberagaman. Tempat terbaiknya adalah `w = 3-7`.
- **Permintaan negatif bocor.** Prompt negatif kosong menjadi token nol; prompt negatif yang terisi menjadi `ε_uncond`. Ini tidaklah sama; beberapa pipeline pipa diam-diam default ke nol.

## Pakai

Tumpukan produksi pada tahun 2026:

| Sasaran | Tulang punggung yang direkomendasikan |
|--------|----------------------|
| Domain sempit, data berpasangan, melatih model dari awal | SDXL fine-tune (LoRA / penuh) — pengiriman tercepat |
| Teks-ke-gambar domain terbuka, weight terbuka | Flux.1-dev (12B, Apache / non-komersial) atau SD3.5-Large |
| Inference tercepat, weight terbuka | Flux.1-schnell (1-4 langkah, Apache) atau SDXL-Lightning |
| Kepatuhan cepat terbaik, dihosting | GPT-Image / DALL-E 3 (masih), Midjourney v7, Imagen 4 |
| Edit alur kerja | Flux.1-Kontext (Des 2024) — menerima gambar + teks |
| Penelitian, dasar | SD 1.5 — kuno namun dipelajari dengan baik |

## Kirim

Simpan `outputs/skill-sd-prompter.md`. Keterampilan mengambil prompt teks + gaya dan output target: model + pos pemeriksaan, skala CFG, sampler, prompt negatif, resolusi, kombo ControlNet/Adaptor IP opsional, dan daftar periksa QA per langkah.

## Latihan1. **Mudah.** Jalankan `code/main.py` dengan panduan `w ∈ {0, 1, 3, 7, 15}`. Catat sample rata-rata berdasarkan kelas. Pada `w` apa arti kelas menyimpang dari arti data sebenarnya?
2. **Sedang.** Tukar encoder linier mainan dengan pasangan encoder/decoder tanh-MLP yang mengalami loss rekonstruksi. Latih kembali difusi pada laten baru. Apakah kualitas sample berubah?
3. **Sulit.** Siapkan inference Difusi Stabil nyata dengan diffuser: muat `sdxl-base`, jalankan 30 langkah Euler dengan CFG=7, tentukan waktunya. Sekarang beralih ke `sdxl-turbo` dengan 4 langkah dan CFG=0. Subjek yang sama, kualitas berbeda — jelaskan apa yang berubah dan mengapa.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Phase pertama | "VAE" | Pasangan encoder/decoder terlatih; kompres 512² menjadi 64². |
| Phase kedua | "U-Net" | Model difusi pada ruang laten. |
| CFG | "Skala panduan" | `(1+w)·ε_cond - w·ε_uncond`; kekuatan pengkondisi lagu. |
| Token nol | "Embedding prompt kosong" | Embedding tanpa syarat digunakan untuk `ε_uncond`. |
| Attention silang | "Bagaimana teks masuk" | Setiap blok U-Net menangani token teks sebagai K dan V. |
| DiT | "Trafo Difusi" | Ganti U-Net dengan trafo melalui patch laten; skala lebih baik. |
| MMDiT | "DiT Multimodal" | Arsitektur SD3: aliran teks dan gambar dengan attention bersama. |
| Faktor penskalaan VAE | "Nomor ajaib" | Membagi laten dengan ~5,4 sehingga difusi beroperasi dalam ruang unit-varians. |

## Catatan produksi: menjalankan Flux-12B pada GPU konsumen 8 GB

referensi integrasi Flux adalah kanonis "Saya memiliki GPU konsumen, bolehkah saya mengirimkan ini?" resep. Caranya adalah daftar literatur inference produksi resep tiga tombol yang sama yang diterapkan pada DiT difusi:

1. **Pemuatan terhuyung-huyung.** Flux memiliki tiga jaringan yang tidak perlu berdampingan di VRAM: encoder teks T5-XXL (~10 GB di fp32), CLIP-L (kecil), MMDiT 12B, dan VAE. Encode prompt terlebih dahulu, *hapus* pembuat enkode, muat DiT, denoise, *hapus* DiT, muat VAE, dekode. GPU konsumen 8GB hanya cocok untuk satu phase dalam satu waktu.
2. **kuantisasi 4-bit melalui bitsandbytes.** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` pada encoder T5 dan DiT. Memotong memori 8×, penurunan kualitas tidak terlihat untuk teks-ke-gambar sesuai tolok ukur Aritra (ditautkan di notebook).
3. **Pembongkaran CPU.** `pipe.enable_model_cpu_offload()` menukar modul secara otomatis antara CPU dan GPU seiring kemajuan setiap forward pass. Menambahkan latensi 10-20% tetapi membuat pipeline berjalan sama sekali.

Penghitungan memori adalah: `10 GB T5 / 8 = 1.25 GB` terkuantisasi, `12 B params × 0.5 bytes = ~6 GB` DiT terkuantisasi, ditambah activation. Dalam istilah stas00, ini adalah ujung ekstrim dari inference TP=1 — tidak ada paralelisme model, kuantisasi maksimum. Untuk produksi, kamu akan menjalankan TP=2 atau TP=4 pada H100s; untuk satu laptop dev, ini resepnya.

## Bacaan Lanjutan- [Rombach dkk. (2022). Sintesis Gambar Resolusi Tinggi dengan Model Difusi Laten](https://arxiv.org/abs/2112.10752) — Difusi Stabil.
- [Podell dkk. (2023). SDXL: Meningkatkan Model Difusi Laten untuk Sintesis Gambar Resolusi Tinggi](https://arxiv.org/abs/2307.01952) — SDXL.
- [Peebles & Xie (2023). Model Difusi yang Dapat Diskalakan dengan Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT.
- [Esser dkk. (2024). Menskalakan Transformer Aliran yang Diperbaiki untuk Sintesis Gambar Resolusi Tinggi](https://arxiv.org/abs/2403.03206) — SD3, MMDiT.
- [Ho & Saliman (2022). Panduan Difusi Bebas Pengklasifikasi](https://arxiv.org/abs/2207.12598) — CFG.
- [Lab (2024). Flux.1 — Pengumuman Black Forest Labs](https://blackforestlabs.ai/announcing-black-forest-labs/) — keluarga Flux.1.
- [Dokumen Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index) — implementasi referensi untuk setiap pos pemeriksaan di atas.
