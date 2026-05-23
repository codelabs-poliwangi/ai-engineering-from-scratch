# Text-to-Speech (TTS) — Dari Tacotron hingga F5 dan Kokoro

> ASR membalikkan ucapan menjadi teks; TTS membalikkan teks menjadi ucapan. Tumpukan 2026 terdiri dari tiga bagian: teks → token, token → mel, mel → bentuk gelombang. Setiap bagiannya mempunyai model bawaan yang pas di laptop.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 6 · 02 (Spektogram & Mel), Fase 5 · 09 (Seq2Seq), Fase 7 · 05 (Transformer Penuh)
**Waktu:** ~75 menit

## Masalah

kamu mempunyai kalimat: "Tolong ingatkan saya untuk menyirami tanaman pada jam 6 sore." kamu memerlukan klip audio berdurasi 3 detik yang terdengar alami, memiliki prosodi yang benar (jeda, stres), mengucapkan "tanaman" dengan vokal yang tepat, dan berjalan dalam waktu kurang dari 300 ms pada CPU untuk asisten suara langsung. kamu juga perlu bertukar suara, menangani input alih code ("ingatkan saya jam 6 sore, daijoubu?"), dan tidak mempermalukan diri sendiri dengan nama.

Pipeline pipa TTS modern terlihat seperti ini:

1. **Teks frontend.** Menormalkan teks (tanggal, angka, email), mengkonversi ke fonem atau token subkata, memprediksi feature prosodi.
2. **Model akustik.** Teks → spektogram mel. Tacotron 2 (2017), FastSpeech 2 (2020), VITS (2021), F5-TTS (2024), Kokoro (2024).
3. **Vocoder.** Mel → bentuk gelombang. WaveNet (2016), WaveRNN, HiFi-GAN (2020), BigVGAN (2022), vocoder codec saraf pada tahun 2024+.

Pada tahun 2026, akustik + vocoder membagi blur dengan difusi ujung ke ujung dan model pencocokan aliran. Namun model mental dari tiga bagian masih berlaku untuk debugging.

## Konsep

![Tacotron, FastSpeech, VITS, F5/Kokoro berdampingan](../assets/tts.svg)

**Tacotron 2 (2017).** Seq2seq: embedding karakter → encoder BiLSTM → attention peka lokasi → decoder LSTM autoregresif memancarkan bingkai mel. Lambat (AR), goyah pada teks panjang. Masih dikutip sebagai garis dasar.

**FastSpeech 2 (2020).** Non-autoregresif. Prediktor durasi menghasilkan berapa banyak bingkai mel yang didapat setiap fonem. 1-pass, 10× lebih cepat dari Tacotron. Kehilangan beberapa kealamian (penyelarasan monotonik) tetapi dikirimkan ke mana-mana.

**VITS (2021).** Melatih encoder + durasi berbasis aliran + vocoder HiFi-GAN secara menyeluruh dengan inference variasional. Kualitas tinggi, model tunggal. TTS sumber terbuka yang dominan 2022–2024. Varian: YourTTS (multi-speaker zero-shot), XTTS v2 (2024, Coqui).

**F5-TTS (2024).** Trafo difusi melalui pencocokan aliran. Prosodi alami, kloning suara zero-shot dengan audio referensi 5 detik. Papan peringkat TTS sumber terbuka teratas tahun 2026. 335 juta parameter.

**Kokoro (2024).** TTS Bahasa Inggris kecil (82M), dapat dijalankan dengan CPU, dan terbaik di kelasnya untuk penggunaan waktu nyata. Kosakata tertutup khusus bahasa Inggris, Apache-2.0.

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3.** Komersial canggih. Tag emosi ElevenLabs v2.5 ("[berbisik]", "[tertawa]") dan suara karakter mendominasi produksi buku audio pada tahun 2026.

### Evolusi Vocoder

| Zaman | Vocoder | Latensi | Kualitas |
|-----|---------|---------|---------|
| 2016 | GelombangNet | hanya offline | SOTA saat rilis |
| 2018 | GelombangRNN | ~waktu nyata | bagus |
| 2020 | HiFi-GAN | 100× waktu nyata | hampir manusia |
| 2022 | VGAN Besar | 50× waktu nyata | menggeneralisasi seluruh penutur/bahasa |
| 2024 | SNAC, DAC (codec saraf) | terintegrasi dengan model AR | token diskrit, hemat bit |

Pada tahun 2026 sebagian besar model "TTS" bersifat end-to-end dari teks ke bentuk gelombang; spektogram mel adalah representasi internal.

### Evaluasi- **MOS (Mean Opinion Score).** Skala 1–5, bersumber dari banyak orang. Masih standar emas; sangat lambat.
- **CMOS (MOS Komparatif).** Preferensi A-vs-B. Interval kepercayaan yang lebih ketat per anotasi.
- **UTMOS, DNSMOS.** Prediktor MOS saraf bebas referensi. Digunakan untuk papan peringkat.
- **CER (Character Error Rate) melalui ASR.** Jalankan output TTS melalui Whisper, hitung CER terhadap teks input. Proksi untuk kejelasan.
- **SECS (Speaker Embedding Cosine Kemiripan).** Kualitas kloning suara.

Nomor 2026 pada tes LibriTTS-bersih:

| Model | UTMOS | CER (melalui Bisikan) | Ukuran |
|-------|-------|-------------------|------|
| Kebenaran dasar | 4.08 | 1,2% | — |
| F5-TTS | 3,95 | 2,1% | 335M |
| XTTS v2 | 3.81 | 3,5% | 470M |
| VIT | 3.62 | 3,1% | 25M |
| Kokoro v0.19 | 3.87 | 1,8% | 82M |
| Parler-TTS Besar | 3.76 | 2,8% | 2.3B |

## Build

### Langkah 1: input fonemisasi

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Fonem adalah jembatan universal. Hindari memasukkan teks mentah ke konten apa pun di bawah kualitas tingkat VITS.

### Langkah 2: jalankan Kokoro (default CPU 2026)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

Berjalan offline, file tunggal, 82 juta parameter.

### Langkah 3: jalankan F5-TTS dengan kloning suara

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

Berikan klip referensi berdurasi 5 detik + transkripnya; F5 mengkloning prosodi dan timbre.

### Langkah 4: Vocoder HiFi-GAN dari awal

Terlalu besar untuk dimasukkan ke dalam skrip tutorial, tetapi bentuknya adalah:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

Training: permusuhan (diskriminator pada jendela pendek) + loss rekonstruksi mel-spektogram + loss pencocokan feature. Dikomoditisasi — gunakan pos pemeriksaan terlatih dari `hifi-gan` repo atau nvidia-NeMo.

### Langkah 5: pipeline lengkap (pseudocode)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## Pakai

Tumpukan tahun 2026:

| Situasi | Pilih |
|-----------|------|
| Asisten suara bahasa Inggris waktu nyata | Kokoro (CPU) atau XTTS v2 (GPU) |
| Kloning suara dari referensi 5 detik | F5-TTS |
| Suara karakter komersial | ElevenLabs v2.5 |
| Narasi buku audio | ElevenLabs v2.5 atau XTTS v2 + penyempurnaan |
| Bahasa dengan sumber daya rendah | Latih VITS pada data target-lang 5–20 jam |
| Tag ekspresif / emosi | Penyempurnaan ElevenLabs v2.5 atau StyleTTS 2 |

Pemimpin sumber terbuka pada tahun 2026: **F5-TTS untuk kualitas, Kokoro untuk efisiensi**. Jangan meraih Tacotron kecuali kamu seorang sejarawan.

## Jebakan

- **Tidak ada penormal teks.** "Dr. Smith" dibaca sebagai "Dokter" atau "Drive"? "2026" sebagai "dua puluh dua puluh enam" atau "dua nol dua enam"? Normalisasikan SEBELUM phonemizer.
- **OOV kata benda yang tepat.** "Ghumare" → "ghyu-mair"? Kirim model grafem-ke-fonem cadangan untuk token yang tidak diketahui.
- **Kliping.** Output vocoder jarang terklip, tetapi ketidakcocokan penskalaan mel pada inference dapat melampaui ±1,0. Selalu `np.clip(wav, -1, 1)`.
- **Ketidakcocokan laju sample.** Kokoro menghasilkan 24 kHz; pipeline hilir kamu mengharapkan 16 kHz → sample ulang atau dapatkan aliasing.

## Kirim

Simpan sebagai `outputs/skill-tts-designer.md`. Rancang pipeline TTS untuk target suara, latensi, dan bahasa tertentu.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Membuat kamus fonem dari kosakata mainan, memperkirakan durasi per fonem, dan mencetak jadwal "mel" palsu.
2. **Sedang.** Instal Kokoro, sintesis kalimat yang sama di suara `af_bella` dan `am_adam`. Bandingkan durasi audio dan kualitas subjektif.
3. **Sulit.** Rekam klip referensi diri kamu berdurasi 5 detik. Gunakan F5-TTS untuk mengkloningnya. Laporkan SECS antara referensi dan output kloning.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Fonem | Satuan suara | Kelas suara abstrak; 39 dalam bahasa Inggris (ARPABet). |
| Prediktor durasi | Berapa lama setiap fonem bertahan | Output model non-AR; bingkai bilangan bulat per fonem. |
| Vocoder | Mel → bentuk gelombang | Pemetaan neural network mel-spec ke sample mentah. |
| HiFi-GAN | Vocoder standar | berbasis GAN; dominan 2020–2024. |
| MOS | Kualitas subyektif | 1–5 skor opini rata-rata dari penilai manusia. |
| DETIK | Metrik klon suara | Kesamaan kosinus antara embedding speaker target dan output. |
| F5-TTS | SOTA sumber terbuka 2024 | Difusi pencocokan aliran; kloning tanpa tembakan. |
| Kokoro | Pemimpin CPU Bahasa Inggris | Model 82M-param, Apache 2.0. |

## Bacaan Lanjutan

- [Shen dkk. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — garis dasar seq2seq.
- [Kim, Kong, Nak (2021). VITS](https://arxiv.org/abs/2106.06103) — berbasis aliran end-to-end.
- [Chen dkk. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — SOTA sumber terbuka saat ini.
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — vocoder yang masih dikirimkan pada tahun 2026.
- [Kokoro-82M di HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — TTS bahasa Inggris ramah CPU 2024.
