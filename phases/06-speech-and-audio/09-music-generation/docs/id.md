# Generasi Musik — MusicGen, Audio Stabil, Suno, dan Gempa Lisensi

> Generasi musik 2026: Suno v5 dan Udio v4 mendominasi iklan; MusicGen, Stable Audio Open, dan sumber terbuka utama ACE-Step. Masalah teknis sebagian besar sudah teratasi. Masalah hukum (penyelesaian Warner Music $500 juta, penyelesaian UMG) mengubah bidang ini pada tahun 2025-2026.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 6 · 02 (Spektogram), Fase 4 · 10 (Model Difusi)
**Waktu:** ~75 menit

## Masalah

Teks → klip musik berdurasi 30 detik hingga 4 menit, dengan lirik, vokal, dan struktur. Tiga sub-masalah:

1. **Pembuatan instrumental.** Teks seperti "drum hip-hop lo-fi dengan kunci hangat" → audio. MusicGen, Audio Stabil, AudioLDM.
2. **Pembuatan lagu (dengan vokal + lirik).** "Lagu country tentang malam hujan di Texas" → lagu lengkap. Suno, Udio, YuE, ACE-Langkah.
3. **Bersyarat/dapat dikontrol.** Memperluas klip yang ada, membuat ulang jembatan, menukar genre, memisahkan batang, atau mengecat. Pengecatan + pemisahan batang Udio adalah feature tahun 2026 yang cocok.

## Konsep

![Pembuatan musik: token-LM vs difusi, peta model 2026](../assets/music-generasi.svg)

### Token LM melalui token neural-codec

**MusicGen** Meta (2023, MIT) dan banyak turunannya: mengkondisikan embedding teks/melodi, memprediksi token EnCodec secara otomatis (32 kHz, 4 buku code), mendekode dengan EnCodec. 300M - 3,3B parameter. Garis dasar yang kuat; berjuang melewati 30 detik.

**ACE-Step** (sumber terbuka, 4B XL dirilis April 2026) memperluas ini untuk generasi yang dilengkapi lirik lagu lengkap. Komunitas terbuka adalah hal yang paling dekat dengan Suno.

### Difusi melalui mels atau laten

**Stable Audio (2023)** dan **Stable Audio Open (2024)**: difusi laten pada audio terkompresi. Unggul dalam loop, desain suara, tekstur sekitar. Tidak bagus dalam lagu lengkap yang terstruktur.

**AudioLDM / AudioLDM2**: teks-ke-audio melalui difusi laten gaya T2I, digeneralisasikan ke musik, efek suara, ucapan.

### Hibrida (produksi) — Suno, Udio, Lyria

Weight tertutup. Kemungkinan kodek AR LM + vocoder berbasis difusi dengan kepala suara/drum/melodi khusus. Suno v5 (2026) adalah pemimpin kualitas ELO 1293. Udio v4 menambahkan inpainting + pemisahan batang (bass, drum, vokal unduhan terpisah).

### Evaluasi

- **FAD (Fréchet Audio Distance).** Distance tingkat embedding antara distribusi audio yang dihasilkan vs distribusi audio nyata menggunakan feature VGGish atau PANNs. Lebih rendah lebih baik. MusicGen kecil: 4,5 FAD di MusicCaps; SOTA ~3.0.
- **Musikalitas (subyektif).** Preferensi manusia. Suno v5 ELO 1293 memimpin.
- **Penyelarasan teks-audio.** Skor CLAP antara prompt dan output.
- **Artefak musikalitas.** Transisi yang tidak sesuai irama, penyimpangan frasa vokal, hilangnya struktur setelah 30 detik.

## Peta model 2026

| Model | Param | Panjang | Vokal | Lisensi |
|-------|--------|--------|--------|---------|
| MusicGen-besar | 3.3B | 30 detik | tidak | MIT |
| Audio Stabil Terbuka | 1.2B | 47 detik | tidak | Stabilitas non-komersial |
| ACE-Langkah XL (Apr 2026) | 4B | > 2 menit | ya | Apache-2.0 |
| YuE | 7B | > 2 menit | ya, multibahasa | Apache-2.0 |
| Suno v5 (ditutup) | ? | 4 menit | ya, ELO 1293 | komersial |
| Udio v4 (ditutup) | ? | 4 menit | ya + batang | komersial |
| Google Lyria 3 (ditutup) | ? | waktu nyata | ya | komersial |
| MiniMax Musik 2.5 | ? | 4 menit | ya | API komersial |

## Lanskap hukum (2025-2026)- **Penyelesaian Warner Music vs Suno.** $500 juta. WMG kini mengawasi kemiripan AI, hak musik, dan lagu buatan pengguna di Suno. Penyelesaian UMG serupa di Udio.
- **EU AI Act** + **California SB 942**: Musik yang dihasilkan AI harus diungkapkan.
- **Riffusion / MusicGen** di bawah MIT tidak memiliki weight kepatuhan tetapi juga tidak memiliki vokal komersial.

Pola pengiriman yang aman:

1. Hasilkan instrumental saja (MusicGen, Stable Audio Open, output MIT/CC0).
2. Gunakan API komersial (Suno, Udio, ElevenLabs Music) dengan lisensi per generasi.
3. Melatih katalog yang dimiliki atau berlisensi (sebagian besar perusahaan berakhir di sini).
4. Tandai generasi dengan tanda air + metadata.

## Build

### Langkah 1: buat dengan MusicGen

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

Tiga ukuran: `small` (300M, cepat), `medium` (1,5B), `large` (3,3B). Kecil sudah cukup untuk "apakah idenya berhasil".

### Langkah 2: pengondisian melodi

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

Melodi MusicGen mengambil kromagram dan mempertahankan nada sambil menukar timbre. Berguna untuk "berikan saya melodi ini sebagai kuartet gesek".

### Langkah 3: Evaluasi FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

Menghitung distance embedding VGGish. Berguna untuk tes regresi tingkat genre; bukan pengganti pendengar manusia.

### Langkah 4: menambah alur kerja musik LLM

Gabungkan dengan ide-ide dari Lesson 7-8:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## Pakai

| Sasaran | Tumpukan |
|------|-------|
| Desain suara instrumental | Audio Stabil Terbuka |
| Permainan / musik adaptif | Google Lyria RealTime (ditutup) |
| Lagu lengkap dengan vokal (komersial) | Suno v5 atau Udio v4 dengan lisensi eksplisit |
| Lagu lengkap dengan vokal (terbuka) | ACE-Langkah XL atau YuE |
| Jingle iklan pendek | Melodi MusicGen dikondisikan pada referensi bersenandung |
| Latar belakang video musik | MusicGen + Difusi Video Stabil |

## Kesalahan yang masih dikirimkan pada tahun 2026

- **Prompt pencucian hak cipta.** "Lagu dengan gaya Taylor Swift" — Suno/Udio komersial memfilternya sekarang, model terbuka tidak. Tambahkan daftar filter kamu sendiri.
- **Pengulangan / penyimpangan melewati 30 detik.** Loop model AR. Crossfade beberapa generasi, atau gunakan ACE-Step untuk koherensi struktural.
- **Tempo drift.** Model menyimpang dari BPM. Gunakan tag BPM di prompt dan post-filter dengan `beat_track` librosa.
- **Kejelasan vokal.** Suno luar biasa; model terbuka sering kali tidak bisa berkata-kata. Jika lirik penting, gunakan API komersial atau sempurnakan.
- **Output mono.** Model terbuka menghasilkan mono atau stereo palsu. Tingkatkan dengan rekonstruksi stereo yang tepat (ezst, difusi stereo Cartesia).

## Kirim

Simpan sebagai `outputs/skill-music-designer.md`. Pilih model, strategi lisensi, rencana panjang/struktur, dan metadata pengungkapan untuk penerapan gen musik.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Ini menghasilkan progresi akord "generatif" + pola drum sebagai simbol ASCII — kartun gen musik. Putar ulang melalui penyaji MIDI mana pun jika kamu mau.
2. **Medium.** Instal `audiocraft`, buat klip berdurasi 10 detik dalam 4 genre dengan MusicGen-small, ukur FAD berdasarkan kumpulan genre referensi.
3. **Hard.** Menggunakan ACE-Step (atau melodi MusicGen), buat tiga variasi lagu yang sama dengan prompt timbre berbeda. Hitung kesamaan CLAP dengan prompt untuk memverifikasi keselarasan.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| mode | Audio FID | Distance Fréchet antara embedding distribusi nyata vs yang dihasilkan. |
| Kromagram | Melodi sebagai nada | vector 12-redup per bingkai; input ke pengkondisian melodi. |
| Batang | Trek instrumen | Bass / drum / vokal / melodi terpisah sebagai WAV. |
| Lukisan | Regen bagian | Menutupi jendela waktu; model meregenerasi hal itu. |
| TEPUK | KLIP teks-audio | Embedding teks audio yang kontras; evaluasi perataan teks-audio. |
| Enkodek | Kodek musik | Codec saraf Meta yang digunakan oleh MusicGen; 32 kHz, 4 buku code. |

## Bacaan Lanjutan

- [Copet dkk. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — tolok ukur autoregresif terbuka.
- [Evans dkk. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — default desain suara.
- [ACE-Step](https://github.com/ace-step/ACE-Step) — membuka generator lagu lengkap 4B, April 2026.
- [Suno v5 platform docs](https://suno.com) — pemimpin kualitas komersial.
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — difusi laten untuk musik + efek suara.
- [Cakupan penyelesaian WMG-Suno](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — preseden November 2025.
