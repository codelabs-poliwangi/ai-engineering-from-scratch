# Neural Audio Codec - EnCodec, SNAC, Mimi, DAC dan Semantic-Acoustic Split

> Generasi audio 2026 hampir semuanya token. EnCodec, SNAC, Mimi, dan DAC mengubah bentuk gelombang kontinu menjadi rangkaian diskrit yang dapat diprediksi oleh Transformer. Pemisahan token semantik-vs-akustik - buku code pertama sebagai semantik, sisanya sebagai akustik - adalah perubahan arsitektur paling penting sejak Transformer untuk audio.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 6 · 02 (Spektogram), Fase 10 · 11 (Kuantisasi), Fase 5 · 19 (Tokenization Subkata)
**Waktu:** ~60 menit

## Masalah

Model bahasa bekerja pada token diskrit. Audio berkelanjutan. Jika kamu menginginkan model gaya LLM untuk ucapan/musik — MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus — pertama-tama kamu memerlukan **codec audio saraf**: encoder terpelajar yang mendiskritisasi audio menjadi sejumlah kecil token, dan decoder yang cocok yang merekonstruksi bentuk gelombang.

Dua keluarga telah muncul:

1. **Codec yang pertama direkonstruksi** — EnCodec, DAC. Optimalkan kualitas audio persepsi. Token bersifat "akustik" — token menangkap semuanya termasuk identitas pembicara, warna suara, kebisingan latar belakang.
2. **Codec yang mengutamakan semantik** — Mimi (Kyutai), SpeechTokenizer. Paksa buku code pertama untuk menyandikan konten linguistik/fonetik (seringkali dengan menyaring dari WavLM). Buku code berikutnya adalah detail akustik.

Wawasan 2024-2026: **codec rekonstruksi murni menghasilkan ucapan yang buram saat kamu mencoba membuat dari teks.** LLM melalui token codec harus mempelajari struktur bahasa DAN struktur akustik dalam buku code yang sama, yang tidak berskala. Memisahkan keduanya — buku code semantik 0, buku code akustik 1-N — inilah yang membuat Moshi dan Sesame CSM berfungsi.

## Konsep

![Empat lanskap codec: EnCodec, DAC, SNAC (multi-skala), Mimi (semantik+akustik)](../assets/codec-comparison.svg)

### Trik inti: Residual Vector Quantization (RVQ)

Daripada satu buku code besar (yang memerlukan jutaan code untuk kualitas yang baik), semua codec audio modern menggunakan **RVQ**: rangkaian buku code kecil. Buku code pertama mengkuantisasi output encoder; yang kedua mengkuantifikasi sisa; dll. Setiap buku code berisi 1024 code. 8 buku code = kosakata efektif 1024^8 = 10^24.

Pada waktu inference, decoder menjumlahkan semua code yang dipilih per frame untuk direkonstruksi.

### Empat codec yang penting di tahun 2026

**EnCodec (Meta, 2022).** Garis dasar. Encoder-decoder melalui bentuk gelombang, hambatan RVQ. 24 kHz, 32 buku code dimungkinkan, default 4 buku code @ 1,5 kbps. Menggunakan arsitektur `1D conv + transformer + 1D conv`. Digunakan oleh MusicGen.

**DAC (Deskripsi, 2023).** RVQ dengan buku code yang dinormalisasi L2, fungsi activation berkala, peningkatan loss. Fidelitas rekonstruksi tertinggi dari semua codec terbuka — terkadang tidak dapat dibedakan dari ucapan asli dengan 12 buku code. Pita penuh 44,1 kHz.

**SNAC (Hubert Siuzdak, 2024).** RVQ multiskala — buku code kasar beroperasi pada kecepatan bingkai yang lebih rendah daripada buku code halus. Secara efektif memodelkan audio secara hierarki: "sketsa" kasar pada ~12 Hz ditambah detail pada 50 Hz. Digunakan oleh Orpheus-3B karena struktur hierarkinya dipetakan dengan baik ke generasi berbasis LM.

**Mimi (Kyutai, 2024).** Pengubah permainan tahun 2026. Kecepatan bingkai 12,5 Hz (sangat rendah), 8 buku code @ 4,4 kbps. Buku Code 0 **disuling dari WavLM** — dilatih untuk memprediksi feature konten ucapan WavLM. Buku code 1-7 adalah residu akustik. Perpecahan ini memberi kekuatan pada Moshi (Lesson 15) dan Sesame CSM.

### Kecepatan bingkai penting untuk pemodelan bahasa

Frame rate lebih rendah = urutan lebih pendek = LM lebih cepat.| Kodek | Kecepatan bingkai | 1 detik = N bingkai | Baik untuk |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75Hz | 75 | musik, audio umum |
| DAC-44.1k | 86Hz | 86 | musik dengan ketelitian tinggi |
| SNAC-24k (kasar) | ~12Hz | 12 | AR-LM efisien |
| Mimi | 12,5Hz | 12.5 | streaming pidato |

Pada 12,5 Hz, ucapan 10 detik hanya terdiri dari 125 frame codec — sebuah Transformer dapat dengan mudah memprediksinya.

### Token semantik vs akustik

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Token semantik (buku code 0 di Mimi).** Mengkodekan apa yang diucapkan — fonem, kata, konten. Disaring dari WavLM melalui loss prediksi tambahan.
- **Token akustik (buku code 1-7).** Mengkodekan timbre, identitas pembicara, prosodi, kebisingan latar belakang, detail halus.

AR LM memprediksi token semantik terlebih dahulu (dikondisikan pada teks), kemudian memprediksi token akustik (dikondisikan pada referensi semantik + pembicara). Faktorisasi inilah yang menjadi alasan mengapa TTS modern dapat mengkloning suara secara zero-shot: model semantik menangani konten; model akustik menangani timbre.

### Kualitas rekonstruksi 2026 (bit per detik, bitrate lebih rendah lebih baik)

| Kodek | Kecepatan bit | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6kbps | 3.2 | 3.8 |
| DAC-6kbps | 6kbps | 3,5 | 4.0 |
| SNAC-3kbps | 3kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4,4 kbps | 3.1 | 3.7 |

Codec tradisional seperti Opus masih unggul dalam hal kualitas persepsi. Codec neural unggul dalam **token diskrit** (yang tidak diproduksi Opus) dan **kualitas model generatif** (apa yang dapat dilakukan LM dengan token tersebut).

## Build

### Langkah 1: enkode dengan EnCodec

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

`n_codebooks=8` pada 6 kbps. Setiap code adalah 0-1023 (10-bit).

### Langkah 2: memecahkan code dan mengukur rekonstruksi

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### Langkah 3: pemisahan semantik-akustik (gaya Mimi)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Buku code semantik 0 selaras dengan WavLM. kamu dapat melatih Transformer teks-ke-semantik - kosakata yang jauh lebih sedikit dibandingkan langsung ke audio. Kemudian pisahkan kondisi dekoder akustik-ke-gelombang pada referensi speaker.

### Langkah 4: mengapa AR LM melalui token codec berfungsi

Untuk klip ucapan 10 detik di buku code 12,5 Hz × 8 Mimi:

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 token adalah konteks sepele untuk sebuah Transformer. Transformer dengan parameter 256M dapat menghasilkan 10 detik ucapan dalam milidetik pada GPU modern.

## Pakai

Masalah peta → codec:

| Tugas | Kodek |
|------|-------|
| Generasi musik umum | EnCodec-24k |
| Rekonstruksi dengan ketelitian tertinggi | DAC-44.1k |
| AR LM melalui ucapan (TTS) | SNAC atau Mimi |
| Streaming pidato dupleks penuh | Mimi (12,5 Hz) |
| Perpustakaan efek suara dengan teks | Kondisi EnCodec + T5 |
| Pengeditan audio yang mendetail | DAC + dalam lukisan |

Aturan praktisnya: **jika kamu membuat model generatif, mulailah dengan Mimi atau SNAC. Jika kamu sedang membuat pipeline kompresi, gunakan Opus.**

## Jebakan

- **Terlalu banyak buku code.** Menambahkan buku code akan meningkatkan fidelitas secara linier, namun panjang urutan LM juga meningkat secara linier. Berhenti jam 8-12.
- **Ketidakcocokan kecepatan bingkai.** Melatih LM pada 12,5 Hz Mimi lalu menyempurnakan EnCodec 50 Hz gagal secara diam-diam.
- **Dengan asumsi semua buku code sama.** Di Mimi, buku code 0 membawa konten; kehilangannya menghancurkan kejelasan. Kehilangan codebook 7 hampir tidak terlihat.
- **Menggunakan kualitas rekonstruksi sebagai satu-satunya metrik.** Codec dapat memiliki rekonstruksi yang bagus namun tidak berguna untuk pembuatan berbasis LM jika struktur semantiknya buruk.

## Kirim

Simpan sebagai `outputs/skill-codec-picker.md`. Pilih codec untuk tugas generatif atau kompresi tertentu.

## Latihan1. **Mudah.** Jalankan `code/main.py`. Ini mengimplementasikan scalar mainan + kuantizer sisa dan mengukur reconstruction error saat kamu menambahkan buku code.
2. **Sedang.** Instal `encodec` dan bandingkan 1, 4, 8, 32 buku code pada klip ucapan yang diluruskan. Plot PESQ atau MSE vs bitrate.
3. **Sulit.** Muat Mimi. Menyandikan klip. Ganti buku code 0 dengan bilangan bulat acak; membaca sandi. Kemudian ganti codebook 7 dengan cara yang sama. Bandingkan kedua kerusakan tersebut - korupsi buku code 0 harus menghancurkan kejelasan; korupsi codebook 7 seharusnya tidak mengubah apa pun.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| RVQ | Kuantisasi sisa | Rangkaian buku code kecil; masing-masing mengkuantifikasi sisa sebelumnya. |
| Kecepatan bingkai | Kecepatan codec | Berapa banyak token-frame per detik. Lebih rendah = LM lebih cepat. |
| Buku code semantik | Buku Code 0 (Mimi) | Buku code disaring dari feature SSL; mengkodekan konten. |
| Buku code akustik | Segala sesuatu yang lain | Timbre, prosodi, noise, detail halus. |
| PESQ / ViSQOL | Kualitas persepsi | Metrik obyektif yang berkorelasi dengan MOS. |
| Enkodek | Kodek meta | Garis dasar RVQ; digunakan oleh MusicGen. |
| Mimi | Kodek Kyutai | kecepatan bingkai 12,5 Hz; pemisahan semantik-akustik; kekuatan Moshi. |

## Bacaan Lanjutan

- [Défossez dkk. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — garis dasar RVQ.
- [Kumar dkk. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — terbuka dengan fidelitas tertinggi.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — RVQ multi-skala.
- [Kyutai (2024). Codec Mimi](https://kyutai.org/codec-explainer) — pemisahan semantik-akustik, distilasi WavLM.
- [Borsos dkk. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — paradigma semantik/akustik dua phase.
- [Zeghidour dkk. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — codec RVQ asli yang dapat dialirkan.
