# Dasar-Dasar Audio — Bentuk Gelombang, Pengambilan Sample, Transformasi Fourier

> Bentuk gelombang adalah sinyal mentah. Spektogram adalah representasinya. Feature Mel adalah bentuknya yang ramah ML. Setiap jalur pipa ASR dan TTS modern melewati tangga ini, dan anak tangga pertama adalah memahami pengambilan sample dan Fourier.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 1 · 06 (Vector & Matrix), Fase 1 · 14 (Distribusi Probabilitas)
**Waktu:** ~45 menit

## Masalah

Mikrofon menghasilkan sinyal tekanan vs waktu. Jaringan saraf kamu menggunakan tensor. Diantaranya terdapat setumpuk konvensi yang, jika dilanggar, akan menghasilkan bug diam: model berlatih dengan baik tetapi WER berlipat ganda, atau TTS mengeluarkan desisan, atau sistem kloning suara mengingat mikrofon, bukan pembicara.

Setiap bug dalam sistem ucapan dapat ditelusuri kembali ke salah satu dari tiga pertanyaan:

1. Pada tingkat sample berapa data dicatat, dan apa yang diharapkan oleh model?
2. Apakah sinyalnya alias?
3. Apakah kamu mengoperasikan sample mentah atau representasi frekuensi?

Lakukan ini dengan benar dan sisa Fase 6 dapat dilakukan. Salahkan mereka dan bahkan Whisper-Large-v4 menghasilkan sampah.

## Konsep

![Bentuk gelombang, pengambilan sample, DFT, dan wadah frekuensi divisualisasikan](../assets/audio-fundamentals.svg)

**Bentuk gelombang.** Array satu dimension yang mengapung di `[-1.0, 1.0]`. Diindeks berdasarkan nomor sample. Untuk mengonversi ke detik, bagi dengan laju sample: `t = n / sr`. Klip berdurasi 10 detik pada 16 kHz adalah rangkaian 160.000 float.

**Kecepatan pengambilan sample (sr).** Berapa banyak sample per detik. Tarif umum pada tahun 2026:

| Nilai | Gunakan |
|------|-----|
| 8kHz | Telepon, VOIP lama. Nyquist pada 4 kHz membunuh konsonan. Hindari untuk ASR. |
| 16kHz | standar ASR. Whisper, Parkit, SeamlessM4T v2 semuanya mengkonsumsi 16 kHz. |
| 22,05kHz | Training vocoder TTS untuk model lama. |
| 24kHz | TTS modern (Kokoro, F5-TTS, xTTS v2). |
| 44,1 kHz | Audio CD, musik. |
| 48kHz | Film, audio pro, TTS fidelitas tinggi (VALL-E 2, NaturalSpeech 3). |

**Nyquist-Shannon.** Tingkat sample `sr` dapat dengan jelas mewakili frekuensi hingga `sr/2`. Batas `sr/2` adalah *frekuensi Nyquist*. Energi di atas Nyquist menjadi *alias* — dilipat ke frekuensi yang lebih rendah — dan merusak sinyal. Selalu filter low-pass sebelum melakukan downsampling.

**Kedalaman bit.** PCM 16-bit (bertanda int16, rentang ±32.767) adalah format pertukaran universal. 24-bit untuk musik, float 32-bit untuk DSP internal. Perpustakaan seperti `soundfile` membaca int16 tetapi mengekspos array float32 di `[-1, 1]`.

**Transformasi Fourier.** Sinyal berhingga apa pun merupakan penjumlahan sinusoidal pada frekuensi berbeda. Transformasi Fourier Diskrit (DFT) menghitung, untuk sample `N`, `N` koefisien kompleks — satu per nampan frekuensi. `bin k` memetakan ke frekuensi `k · sr / N` Hz. Besaran adalah amplitudo pada frekuensi itu, sudut adalah fase.

**FFT.** Fast Fourier Transform: algoritma `O(N log N)` untuk DFT ketika `N` adalah pangkat 2. Setiap pustaka audio menggunakan FFT. FFT sample 1024 pada 16 kHz menghasilkan 512 nampan frekuensi yang dapat digunakan yang mencakup 0–8 kHz pada resolusi 15,6 Hz.

**Pembingkaian + jendela.** Kami tidak melakukan FFT seluruh klip. Kami memotongnya menjadi *bingkai* yang tumpang tindih (biasanya 25 mdtk dengan lompatan 10 mdtk), mengalikan setiap bingkai dengan fungsi jendela (Hann, Hamming) untuk mematikan diskontinuitas tepi, lalu FFT setiap bingkai. Ini adalah Transformasi Fourier Waktu Pendek (STFT). Lesson 02 diambil dari sini.

## Build### Langkah 1: baca klip dan plot bentuk gelombangnya

`code/main.py` hanya menggunakan modul stdlib `wave` untuk menjaga demo bebas ketergantungan. Untuk produksi, kamu akan menggunakan `soundfile` atau `torchaudio.load` (keduanya mengembalikan tupel `(waveform, sr)`):

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### Langkah 2: sintesis gelombang sinus dari prinsip pertama

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

Sinus 440 Hz (konser A) pada 16 kHz selama 1 detik adalah 16.000 float. Menulis dengan `wave.open(..., "wb")` menggunakan pengkodean PCM 16-bit.

### Langkah 3: hitung DFT dengan tangan

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — baik untuk `N=256` untuk mengonfirmasi kebenarannya, tidak berguna untuk audio sebenarnya. Panggilan code asli `numpy.fft.rfft` atau `torch.fft.rfft`.

### Langkah 4: temukan frekuensi dominan

Indeks puncak magnitudo `k_star` dipetakan ke frekuensi `k_star * sr / N`. Menjalankan ini pada sinus 440 Hz akan menghasilkan puncak di bin `440 * N / sr`.

### Langkah 5: mendemonstrasikan aliasing

Contoh sinus 7 kHz pada 10 kHz (Nyquist = 5 kHz). Nada 7 kHz berada di atas Nyquist dan dilipat menjadi `10 − 7 = 3 kHz`. Puncak FFT muncul pada 3 kHz. Ini adalah demo aliasing klasik dan alasan mengapa setiap DAC/ADC dikirimkan dengan filter low-pass berdinding bata.

## Pakai

Tumpukan yang sebenarnya akan kamu kirimkan pada tahun 2026:

| Tugas | Perpustakaan | Mengapa |
|------|---------|-----|
| Baca/tulis WAV/FLAC/OGG | `soundfile` (pembungkus file libsnd) | Tercepat, stabil, mengembalikan float32. |
| Contoh ulang | `torchaudio.transforms.Resample` atau `librosa.resample` | Anti-aliasing bawaan yang benar. |
| STFT / Mel | `torchaudio` atau `librosa` | Ramah GPU; Ekosistem PyTorch. |
| Streaming waktu nyata | `sounddevice` atau `pyaudio` | Pengikatan PortAudio lintas platform. |
| Periksa file | `ffprobe` atau `soxi` | CLI, cepat, laporkan sr/pipeline/codec. |

Aturan pengambilan keputusan: **cocokkan rasio sample sebelum kamu mencocokkan yang lain**. Whisper mengharapkan 16 kHz mono float32. Berikan stereo 44,1 kHz dan kamu akan mendapatkan sampah yang terlihat seperti bug model.

## Kirim

Simpan sebagai `outputs/skill-audio-loader.md`. Keterampilan ini membantu kamu memeriksa apakah input audio sesuai dengan ekspektasi model downstream dan mengambil sample ulang dengan benar jika tidak.

## Latihan

1. **Mudah.** Sintesis campuran 1 detik 220 Hz + 440 Hz + 880 Hz pada 16 kHz. Jalankan DFT. Konfirmasikan tiga puncak pada tempat sampah yang diharapkan.
2. **Sedang.** Rekam WAV suara kamu selama 3 detik pada 48 kHz. Turunkan sample ke 16 kHz menggunakan `torchaudio.transforms.Resample` (dengan anti-aliasing), lalu ke 16 kHz menggunakan penipisan naif (setiap sample ketiga). FFT keduanya. Di mana aliasingnya muncul?
3. **Sulit.** Buat STFT dari awal hanya menggunakan `math` dan DFT dari Langkah 3. Ukuran bingkai 400, hop 160, jendela Hann. Plot besaran dengan `matplotlib.pyplot.imshow`. Ini adalah spektogram Lesson 02.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Tingkat sample | Berapa banyak sample per detik | Frekuensi dalam Hz di mana ADC mengukur sinyal. |
| Nyquist | Frekuensi maksimal yang dapat kamu wakili | `sr/2`; energi di atasnya alias kembali turun. |
| Kedalaman bit | Resolusi setiap sample | `int16` = 65.536 level; `float32` = presisi 24-bit di `[-1, 1]`. |
| DFT | Transformasi Fourier untuk barisan | `N` sample → `N` koefisien frekuensi kompleks. |
| FFT | DFT cepat | Algoritma `O(N log N)` membutuhkan `N` = pangkat 2. |
| Tempat Sampah | Kolom frekuensi | `k · sr / N` Hz; resolusi = `sr / N`. |
| STFT | Spektogram di bawah tenda | FFT berbingkai + berjendela seiring waktu. |
| Mengasingkan | Hantu frekuensi aneh | Energi di atas Nyquist dipantulkan ke wadah yang lebih rendah. |

## Bacaan Lanjutan

- [Shannon (1949). Komunikasi di Tengah Kebisingan](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — makalah di balik teorema pengambilan sample.
- [Smith — Panduan Ilmuwan dan Insinyur untuk Pemrosesan Sinyal Digital](https://www.dspguide.com/ch8.htm) — buku teks DSP kanonik gratis.
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — panduan praktis dengan code.
- [Heinrich Kuttruff — Room Acoustics (edisi ke-6)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — referensi mengapa audio di dunia nyata bukanlah sinusoid yang bersih.
- [Steve Eddins — buku catatan Interpretasi FFT](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — intuisi bin frekuensi terhapus dalam 10 menit.
