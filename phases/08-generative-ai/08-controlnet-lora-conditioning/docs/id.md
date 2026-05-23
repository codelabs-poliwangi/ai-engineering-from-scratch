# ControlNet, LoRA & Pengkondisian

> Teks saja merupakan sinyal kontrol yang kikuk. ControlNet memungkinkan kamu mengkloning model difusi yang telah dilatih sebelumnya dan mengarahkannya dengan peta kedalaman, kerangka pose, coretan, atau gambar tepi. LoRA memungkinkan kamu menyempurnakan model parameter 2B dengan melatih 10 juta parameter. Bersama-sama mereka mengubah Difusi Stabil dari mainan menjadi pipeline gambar tahun 2026 yang dikirimkan ke setiap agensi.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 8 · 07 (Difusi Laten), Fase 10 (LLM dari Awal — untuk fondasi LoRA)
**Waktu:** ~75 menit

## Masalah

Prompt seperti "seorang wanita berpakaian merah sedang mengajak anjing berjalan-jalan di jalan yang sibuk" tidak memberikan informasi kepada model tentang *di mana* anjing tersebut berada, *pose* apa yang dilakukan wanita tersebut, atau *perspektif* jalan. Teks embed sekitar 10% dari apa yang kamu perlukan untuk menentukan gambar. Sisanya bersifat visual dan tidak dapat dijelaskan secara efisien dengan kata-kata.

Melatih model kondisional baru dari awal untuk setiap sinyal (pose, kedalaman, kecerdikan, segmentasi) sangatlah sulit. kamu ingin menjaga tulang punggung SDXL 2,6B-param tetap beku, memasang jaringan samping kecil yang membaca pengondisian, dan mendorongnya ke feature perantara tulang punggung. Itu adalah ControlNet.

kamu juga ingin mengajarkan model konsep baru (wajah kamu, produk kamu, gaya kamu) tanpa melatih ulang model secara keseluruhan. kamu menginginkan delta 100x lebih kecil. Itu adalah LoRA — adaptor tingkat rendah yang dihubungkan ke weight attention yang ada.

ControlNet + LoRA + text = perangkat praktisi tahun 2026. Sebagian besar pipeline gambar produksi melapisi 2-5 LoRA, 1-3 ControlNets, dan Adaptor IP di atas basis SDXL/SD3/Flux.

## Konsep

![ControlNet mengkloning pembuat enkode; LoRA menambahkan delta peringkat rendah](../assets/controlnet-lora.svg)

### ControlNet (Zhang dkk., 2023)

Ambil SD yang sudah dilatih sebelumnya. *Klon* setengah encoder dari U-Net. Bekukan yang asli. Latih klon untuk menerima input pengkondisian tambahan (tepi, kedalaman, pose). Hubungkan klon kembali ke setengah dekoder aslinya dengan *konvolusi nol* lewati koneksi (konvs 1×1 diinisialisasi ke nol — mulai sebagai larangan operasi, pelajari delta).

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

Zero-conv init berarti ControlNet dimulai sebagai identitas — tidak ada salahnya bahkan sebelum training. Berlatih pada 1M (prompt, condition, image) tiga kali lipat dengan loss difusi standar.

ControlNets per modalitas dikirimkan sebagai model sampingan kecil (~360M untuk SDXL, ~70M untuk SD 1.5). kamu dapat menyusunnya pada inference:

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

Untuk setiap layer linier `W ∈ R^{d×d}` dalam model, bekukan `W` dan tambahkan delta peringkat rendah:

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

dengan `r << d`. Peringkat 4-16 adalah standar untuk attention, peringkat 64-128 untuk penyempurnaan berat. Jumlah parameter baru: `2 · d · r` bukannya `d²`. Untuk attention SDXL dengan `d=640`, `r=16`: 20k parameter per adaptor, bukan 410k — pengurangan 20x. Di seluruh model: LoRA biasanya berukuran 20-200MB vs dasar 5GB.

Sebagai kesimpulan, kamu dapat menskalakan LoRA: `W' = W + α · B @ A`. `α = 0.5-1.5` normal. Beberapa LoRA ditumpuk secara aditif (dengan peringatan umum bahwa mereka berinteraksi secara non-linier).

### Adaptor IP (Ye et al., 2023)

Adaptor kecil yang menerima *gambar* sebagai pengondisian (bersama teks). Menggunakan encoder gambar CLIP untuk menghasilkan token gambar, memasukkannya ke dalam attention silang bersama token teks. ~20MB per model dasar. Memungkinkan kamu "menghasilkan gambar dengan gaya referensi ini" tanpa LoRA.

## Matrix komposisi| Alat | Apa yang dikontrolnya | Ukuran | Kapan menggunakan |
|------|------------------|------|-------------|
| KontrolNet | Struktur spasial (pose, kedalaman, tepi) | 70-360MB | Tata letak yang tepat, komposisi |
| LoRA | Gaya, subjek, konsep | 20-200MB | Personalisasi, gaya |
| Adaptor IP | Gaya atau subjek dari gambar referensi | 20MB | Tidak ada teks yang dapat menggambarkan tampilan |
| Inversi Tekstual | Konsep tunggal sebagai token baru | 10KB | Legacy, sebagian besar digantikan oleh LoRA |
| Stan Impian | Penyempurnaan penuh pada suatu subjek | 2-5GB | Identitas kuat, komputasi tinggi |
| Adaptor T2I | Alternatif ControlNet yang lebih ringan | 70MB | Perangkat edge, anggaran inference |

ControlNet ≈ spasial. LoRA ≈ semantik. Gunakan keduanya.

## Build

`code/main.py` menyimulasikan dua mekanisme pada 1-D:

1. **LoRA.** Layer linier terlatih `W`. Bekukan. Latih `B @ A` peringkat rendah sedemikian rupa sehingga `W + BA` cocok dengan layer linier target. Tunjukkan bahwa `r = 1` cukup untuk mempelajari koreksi peringkat-1 dengan sempurna.

2. **ControlNet-lite.** Prediktor "basis beku" dan "jaringan samping" yang membaca sinyal tambahan. Output jaringan samping dilindungi oleh scalar yang dapat dipelajari yang diinisialisasi ke nol (konv-nol versi kami). Latih dan saksikan gerbangnya menanjak.

### Langkah 1: Matematika LoRA

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### Langkah 2: jaringan sisi nol-init

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

Pada langkah 0, keluarannya identik dengan basis. Pembaruan training awal `gate` secara perlahan — tidak ada penyimpangan yang membawa bencana.

## Jebakan

- **LoRA dengan skala berlebihan.** `α = 2` atau `α = 3` adalah peretasan umum "membuatnya lebih kuat" yang menghasilkan output yang terlalu bergaya/rusak. Simpan `α ≤ 1.5`.
- **Konflik weight ControlNet.** Menggunakan Pose ControlNet dengan weight 1.0 dan Depth ControlNet dengan weight 1.0 biasanya overshoot. Jumlah weight ≈ 1,0 adalah default yang aman.
- **LoRA di basis yang salah.** SDXL LoRA diam-diam tidak beroperasi di SD 1.5 karena dimension attention-nya tidak cocok. Diffuser akan memperingatkan di 0,30+.
- **Penyimpangan Inversi Tekstual.** Token yang dilatih di satu pos pemeriksaan mengalami penyimpangan yang buruk di pos pemeriksaan lainnya. LoRA lebih portabel.
- **Penggabungan dan penyimpanan weight LoRA.** kamu dapat memasukkan LoRA ke dalam weight model dasar untuk inference yang lebih cepat (tanpa penambahan waktu proses), namun kamu kehilangan kemampuan untuk menskalakan `α` saat waktu proses. Simpan kedua versi tersebut.

## Pakai

| Sasaran | pipa 2026 |
|------|---------------|
| Mereproduksi gaya seni suatu merek | LoRA dilatih pada ~30 gambar yang dikurasi di peringkat 32 |
| Letakkan wajah saya di gambar yang dihasilkan | DreamBooth atau LoRA + IP-Adaptor-FaceID |
| Pose khusus + prompt | ControlNet-Openpose + SDXL + teks |
| Komposisi sadar kedalaman | ControlNet-Kedalaman + SD3 |
| Referensi + prompt | Adaptor IP + teks |
| Tata letak yang tepat | ControlNet-Scribble atau ControlNet-Canny |
| Penggantian latar belakang | ControlNet-Seg + Inpainting (Lesson 09) |
| Gaya 1 langkah cepat | LCM-LoRA di SDXL-Turbo |

## Kirim

Simpan `outputs/skill-sd-toolkit-composer.md`. Keterampilan mengambil tugas (aset input: prompt, gambar referensi opsional, pose opsional, kedalaman opsional, coretan opsional) dan mengeluarkan tumpukan alat, weight, dan protokol benih yang dapat direproduksi.

## Latihan1. **Mudah.** Di `code/main.py`, variasikan peringkat LoRA `r` dari 1 ke 4. Pada peringkat berapa LoRA sama persis dengan delta target peringkat-2?
2. **Sedang.** Latih dua LoRA terpisah pada dua transformasi target. Gabungkan semuanya dan tunjukkan interaksi aditifnya. Kapan interaksi tersebut memutus linearitas?
3. **Hard.** Gunakan diffuser untuk menumpuk: SDXL-base + Canny-ControlNet (weight 0,8) + gaya LoRA (α 0,8) + IP-Adapter (weight 0,6). Ukur trade-off FID-vs-prompt-adherence karena weight tumpukan bervariasi.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| KontrolNet | "Kontrol spasial" | Encoder yang dikloning + lompatan tanpa konversi; membaca gambar pengkondisian. |
| Konvolusi nol | "Dimulai sebagai identitas" | 1×1 konv diinisialisasi ke nol; ControlNet dimulai sebagai no-op. |
| LoRA | "Adaptor tingkat rendah" | `W + B @ A`, `r << d`; Parameter 100x lebih sedikit dibandingkan penyempurnaan penuh. |
| peringkat r | "Kenop" | kompresi LoRA; 4-16 tipikal, 64+ untuk personalisasi berat. |
| | "Kekuatan LoRA" | Penskalaan waktu proses delta LoRA. |
| Adaptor IP | "Gambar referensi" | Adaptor pengkondisian gambar kecil melalui token CLIP-image. |
| Stan Impian | "Penyempurnaan subjek penuh" | Latih model lengkap pada ~30 gambar subjek. |
| Inversi Tekstual | "Token baru" | Learn embedding kata baru saja; warisan, sebagian besar diganti. |

## Catatan produksi: Pertukaran LoRA, jalur ControlNet, layanan multi-penyewa

SaaS text-to-image yang sebenarnya melayani ratusan LoRA dan selusin ControlNet melalui pos pemeriksaan dasar yang sama. Masalah penyajiannya sangat mirip dengan multi-tenancy LLM (literatur produksi mencakup kasus LLM dalam batching berkelanjutan dan LoRAX / S-LoRA):

- **LoRA hot-swap, jangan digabungkan.** Menggabungkan `W' = W + α·B·A` ke dalam basis memberikan ~3-5% lebih cepat inference per langkah tetapi membekukan `α` dan basis. Jaga agar LoRA tetap panas di VRAM sebagai delta peringkat-r; diffusers mengekspos `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` untuk activation per permintaan. Biaya swap adalah weight `2 · d · r · num_layers` — skala MB, sub-detik.
- **ControlNet sebagai jalur attention kedua.** Encoder yang dikloning berjalan secara paralel dengan basis. Dua ControlNet dengan weight masing-masing 1,0 = dua operan maju ekstra per langkah, bukan satu operan gabungan. Ruang kepala ukuran batch turun secara kuadrat. Anggaran untuk ~1,5× biaya langkah per ControlNet aktif.
- **LoRA terkuantisasi juga.** Jika kamu mengkuantisasi basisnya (lihat Lesson 07, Fluks pada 8GB), delta LoRA juga terkuantisasi dengan rapi menjadi 8-bit atau 4-bit. Pemuatan gaya QLoRA memungkinkan kamu menumpuk 5-10 LoRA di atas basis Flux 4-bit tanpa menghabiskan memori.

Khusus Fluks: Notebook Niels Flux-on-8GB mengkuantisasi basis menjadi 4-bit; menumpuk gaya LoRA (`pipe.load_lora_weights("user/style-lora")`) pada basis terkuantisasi di `weight_name="pytorch_lora_weights.safetensors"` masih berfungsi. Ini adalah resep yang dikirimkan sebagian besar agensi SaaS pada tahun 2026.

## Bacaan Lanjutan- [Zhang, Rao, Agrawala (2023). Menambahkan Kontrol Bersyarat ke Model Difusi Teks-ke-Gambar](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu dkk. (2021). LoRA: Adaptasi Tingkat Rendah dari Large Language Model](https://arxiv.org/abs/2106.09685) — LoRA (awalnya untuk LLM; port ke difusi).
- [Kamu dkk. (2023). Adaptor IP: Adaptor Prompt Gambar yang Kompatibel dengan Teks](https://arxiv.org/abs/2308.06721) — Adaptor IP.
- [Mou dkk. (2023). T2I-Adapter: Adaptor Pembelajaran untuk Menggali Lebih Banyak Kemampuan Terkendali](https://arxiv.org/abs/2302.08453) — alternatif yang lebih ringan untuk ControlNet.
- [Ruiz dkk. (2023). DreamBooth: Menyempurnakan Model Difusi Teks-ke-Gambar untuk Pembuatan Berdasarkan Subjek](https://arxiv.org/abs/2208.12242) — DreamBooth.
- [HuggingFace Diffusers — dokumen ControlNet / LoRA / IP-Adapter](https://huggingface.co/docs/diffusers/training/controlnet) — pipeline pipa referensi.
