# Generasi Audio

> Audio adalah sinyal 1-D pada 16-48 kHz. Klip lima detik adalah 80-240 ribu sample. Tidak ada Transformer yang menangani urutan itu secara langsung. Solusi untuk setiap model audio produksi pada tahun 2026 adalah sama: codec saraf (Encodec, SoundStream, DAC) mengompresi audio menjadi token diskrit pada 50-75 Hz, dan model Transformer atau difusi menghasilkan token.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 6 · 02 (Feature Audio), Fase 6 · 04 (ASR), Fase 8 · 06 (DDPM)
**Waktu:** ~45 menit

## Masalah

Tiga tugas pembuatan audio:

1. **Text-to-speech.** Teks yang diberikan, menghasilkan ucapan. Ucapan yang bersih bersifat pita sempit dan memiliki struktur fonetik yang kuat — diselesaikan dengan baik dengan Transformer-over-tokens. VALL-E (Microsoft), NaturalSpeech 3, ElevenLabs, OpenAI TTS.
2. **Pembuatan musik.** Dengan adanya prompt (teks, melodi, progresi akord, genre), hasilkan musik. Distribusinya jauh lebih luas. MusicGen (Meta), Audio Stabil 2.5, Suno v4, Udio, Riffusion.
3. **Efek audio / desain suara.** Jika diberi prompt, hasilkan suara sekitar atau Foley. AudioGen, AudioLDM 2, Audio Terbuka Stabil.

Ketiganya berjalan pada substrat yang sama: codec audio neural + token-AR atau generator difusi.

## Konsep

![Pembuatan audio: token codec + Transformer atau difusi](../assets/audio-generasi.svg)

### Codec audio saraf

Encodec (Meta, 2022), SoundStream (Google, 2021), Descript Audio Codec (DAC, 2023). Encoder konvolusional memampatkan bentuk gelombang menjadi vector per langkah waktu; kuantisasi vector sisa (RVQ) mengubah setiap vector menjadi rangkaian indeks buku code K. Decoder membalikkannya. Audio 24 kHz pada 2 kbps menggunakan 8 buku code RVQ pada 75 Hz = 600 token/detik.

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### Dua paradigma generatif di atas

**Token-autoregresif.** Ratakan token RVQ menjadi berurutan, jalankan Transformer khusus dekoder. MusicGen menggunakan "paralel tertunda" untuk memancarkan aliran buku code K secara paralel dengan offset per aliran. VALL-E menghasilkan token ucapan dari prompt teks + sample suara 3 detik.

**Difusi laten.** Kemas token codec sebagai laten berkelanjutan atau buat model dengan difusi kategorikal. Stable Audio 2.5 menggunakan pencocokan aliran pada audio laten berkelanjutan. AudioLDM 2 menggunakan difusi teks-ke-mel-ke-audio.

Tren 2024-2026: pencocokan aliran lebih unggul dalam hal musik (inference lebih cepat, sample lebih bersih) sementara token-AR masih mendominasi ucapan karena bersifat kausal alami dan dialirkan dengan baik.

## Lanskap produksi

| Sistem | Tugas | Tulang punggung | Latensi |
|--------|------|----------|---------|
| SebelasLabs V3 | TTS | Token-AR + vocoder saraf | ~300ms token pertama |
| Audio OpenAI GPT-4o | Pidato dupleks penuh | AR multimodal ujung ke ujung | ~200 md |
| Pidato Alami 3 | TTS | Pencocokan aliran laten | Non-streaming |
| Audio Stabil 2.5 | Musik / SFX | Pencocokan aliran DiT + pada audio laten | ~10 detik untuk klip 1 menit |
| Suno v4 | Lagu lengkap | Tidak diungkapkan; token-AR dicurigai | ~30 detik per lagu |
| Udio v1.5 | Lagu lengkap | Tidak diungkapkan | ~30 detik per lagu |
| MusikGen 3.3B | Musik | Token-AR pada Encodec 32kHz | Waktu nyata |
| Kerajinan Audio 2 | Musik + SFX | Pencocokan aliran | ~5 detik untuk klip 5 detik |
| Rifusi v2 | Musik | Difusi spektogram | ~10 detik |

## Build

`code/main.py` mensimulasikan ide inti: melatih trafo token berikutnya yang kecil pada rangkaian "token audio" sintetis yang dihasilkan dari dua "gaya" yang berbeda (bergantian token rendah dan tinggi untuk gaya A, jalan monoton untuk gaya B). Kondisi pada gaya dan sample.

### Langkah 1: token audio sintetis

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### Langkah 2: latih prediktor token kecilPrediktor gaya bigram dikondisikan pada gaya. Intinya polanya: token codec → training lintas entropi → pengambilan sample autoregresif.

### Langkah 3: sample secara kondisional

Mengingat token gaya dan token awal, ambil contoh token berikutnya dari distribusi yang diprediksi. Lanjutkan untuk 20-40 token.

## Jebakan

- **Kualitas codec membatasi kualitas output.** Jika codec tidak dapat mewakili suara dengan tepat, kualitas generator apa pun tidak akan membantu. DAC adalah yang terbaik terbuka saat ini.
- **Akumulasi kesalahan RVQ.** Setiap layer RVQ memodelkan sisa dari layer sebelumnya. Kesalahan pada layer 1 menyebar. Pengambilan sample dengan suhu 0 pada layer yang lebih tinggi membantu.
- **Struktur musik.** Token berdurasi 30 detik setara dengan 20 ribu+ token pada 75 Hz. Sulit untuk Transformer. MusicGen menggunakan jendela geser + kelanjutan cepat; Audio Stabil menggunakan klip yang lebih pendek + crossfading.
- **Artefak pada batasnya.** Crossfading antar klip yang dihasilkan memerlukan penambahan tumpang tindih secara hati-hati.
- **Selera data bersih.** Generator musik memerlukan puluhan ribu jam musik berlisensi. Gugatan Suno/Udio RIAA (2024) mengemuka.
- **Etika kloning suara.** Sample 3 detik ditambah prompt teks sudah cukup bagi VALL-E / XTTS / ElevenLabs untuk mengkloning suara. Setiap model produksi memerlukan deteksi penyalahgunaan + daftar pilihan untuk tidak ikut serta.

## Pakai

| Tugas | tumpukan 2026 |
|------|------------|
| TTS Komersial | ElevenLabs, OpenAI TTS, atau Azure Neural |
| Kloning suara (diverifikasi persetujuan) | XTTS v2 (terbuka) atau ElevenLabs Pro |
| Musik latar, cepat | API Audio Stabil 2.5, Suno, atau Udio |
| Musik dengan lirik | Suno v4 atau Udio v1.5 |
| Efek suara / Foley | AudioCraft 2, ElevenLabs SFX, atau Audio Stabil Terbuka |
| Agen suara waktu nyata | GPT-4o waktu nyata atau Gemini Live |
| Penelitian musik weight terbuka | MusicGen 3.3B, Audio Stabil Terbuka 1.0, AudioLDM 2 |
| Sulih suara/terjemahan | HeyGen, Sulih Suara ElevenLabs |

## Kirim

Simpan `outputs/skill-audio-brief.md`. Keterampilan mengambil ringkasan audio (tugas, durasi, gaya, suara, lisensi) dan output: model + hosting, format prompt (tag genre, deskriptor gaya, penanda struktural), codec + generator + rantai vocoder, protokol seed, dan rencana evaluasi (skor MOS / CLAP / CER untuk TTS / pengguna A/B).

## Latihan

1. **Mudah.** Jalankan `code/main.py` dan atur gaya secara eksplisit. Verifikasi urutan yang dihasilkan cocok dengan pola gaya.
2. **Sedang.** Tambahkan dekode paralel tertunda: simulasikan 2 aliran token yang harus tetap diimbangi sebanyak 1 langkah. Latih prediktor bersama.
3. **Sulit.** Gunakan Transformer HuggingFace untuk menjalankan MusicGen-small secara lokal. Hasilkan klip 10 detik dengan tiga prompt berbeda; A/B untuk kepatuhan gaya.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Kodek | "Kompresi saraf" | Encoder / decoder untuk audio; output tipikal adalah token 50-75 Hz. |
| RVQ | "Sisa VQ" | Rangkaian kuantizer K; masing-masing memodelkan sisa dari sebelumnya. |
| Tanda | "Satu simbol codec" | Indeks diskrit ke dalam buku code; 1024 atau 2048 tipikal. |
| Paralel tertunda | "Buku code offset" | Memancarkan aliran token K dengan offset terhuyung-huyung untuk mengurangi panjang urutan. |
| Pencocokan aliran | "Kemenangan tahun 2024 untuk audio" | Alternatif jalur yang lebih lurus terhadap difusi; pengambilan sample lebih cepat. |
| Prompt suara | "sample 3 detik" | Embedding speaker atau awalan token yang mengarahkan suara kloning. |
| Spektogram mel | "Visualnya" | Spektogram persepsi magnitudo log; digunakan oleh banyak sistem TTS. |
| Vocoder | "Mel melambai" | Komponen saraf yang mengubah spektogram mel kembali menjadi audio. |## Catatan produksi: audio adalah masalah streaming

Audio adalah satu-satunya modalitas output yang diharapkan pengguna *saat dihasilkan*, tidak sekaligus. Dalam istilah produksi, hal ini berarti TPOT (Time Per Output Token) penting karena kecepatan mendengarkan pengguna adalah target throughput — bukan kecepatan membaca mereka. Untuk audio 16kHz yang diberi token pada ~75 token/detik (Encodec), server harus menghasilkan ≥75 token/detik per pengguna agar pemutaran tetap lancar.

Dua konsekuensi arsitektur:

- **Model audio pencocokan aliran tidak dapat melakukan streaming dengan mudah.** Stable Audio 2.5 dan AudioCraft 2 merender panjang klip tetap dalam satu pass. Untuk melakukan streaming, kamu memotong klip dan tumpang tindih batas — bayangkan difusi jendela geser — menambahkan overhead latensi 100-300 md vs model codec AR.

Jika produknya adalah "obrolan suara langsung" atau "sambungan musik real-time", pilih jalur codec AR. Jika "merender klip 30 detik saat dikirimkan", pencocokan aliran menang dalam hal kualitas dan latensi total.

## Bacaan Lanjutan

- [Défossez dkk. (2022). Encodec: Kompresi Audio Neural Fidelitas Tinggi](https://arxiv.org/abs/2210.13438) — standar codec.
- [Zeghidour dkk. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — codec audio neural pertama yang banyak digunakan.
- [Kumar dkk. (2023). Kompresi Audio Fidelitas Tinggi dengan Peningkatan RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC.
- [Wang dkk. (2023). Model Bahasa Neural Codec adalah Zero-Shot Text to Speech Synthesizer (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E.
- [Copet dkk. (2023). Generasi Musik yang Sederhana dan Terkendali (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen.
- [Liu dkk. (2023). AudioLDM 2: Mempelajari Pembuatan Audio Holistik dengan Pra-training yang Diawasi Sendiri](https://arxiv.org/abs/2308.05734) — AudioLDM 2.
- [Stabilitas AI (2024). Audio Stabil 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — text-to-music 2025 dengan pencocokan aliran.
