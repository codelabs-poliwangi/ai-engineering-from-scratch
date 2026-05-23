# Spektogram, Skala Mel & Feature Audio

> Jaringan saraf tidak menggunakan bentuk gelombang mentah dengan baik. Mereka mengkonsumsi spektogram. Mereka mengonsumsi spektogram mel dengan lebih baik. Setiap ASR, TTS, dan pengklasifikasi audio pada tahun 2026 hidup atau mati karena satu pilihan preprocessing ini.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 6 · 01 (Dasar-Dasar Audio)
**Waktu:** ~45 menit

## Masalah

Ambil klip 16 kHz berdurasi 10 detik. Itu berarti 160.000 kendaraan hias, semuanya di `[-1, 1]`, hampir tidak berkorelasi sama sekali dengan label "anjing menggonggong" atau "kata kucing". Bentuk gelombang mentah memiliki informasi tetapi dalam bentuk yang tidak dapat diekstraksi dengan mudah oleh model. Dua fonem identik yang diucapkan dengan distance 100 ms memiliki sample mentah yang sangat berbeda.

Spektogram memperbaikinya. Ini meruntuhkan detail temporal di mana persepsi manusia mengabaikannya (jitter mikrodetik) dan mempertahankan struktur tempat persepsi hadir (yang frekuensinya energik, dalam rentang waktu ~10–25 ms).

Spektogram Mel mendorong lebih jauh. Manusia merasakan nada secara logaritmik: 100 Hz vs 200 Hz terdengar "distance yang sama" dengan 1000 Hz vs 2000 Hz. Skala mel membengkokkan sumbu frekuensi agar sesuai. Spektogram berskala mel adalah satu-satunya feature terpenting dalam pidato ML dari tahun 2010 hingga 2026.

## Konsep

![Bentuk gelombang ke STFT ke mel spektogram ke tangga MFCC](../assets/mel-features.svg)

**STFT (Short-Time Fourier Transform).** Membagi bentuk gelombang menjadi bingkai yang tumpang tindih (umumnya: jendela 25 mdtk, hop 10 mdtk = 400 sample / 160 sample pada 16 kHz). Lipat gandakan setiap frame dengan fungsi jendela (Hann adalah default; Hamming tradeoff yang sedikit berbeda). FFT setiap frame. Susun spektrum magnitudo ke dalam matrix berbentuk `(n_frames, n_freq_bins)`. Itu adalah spektogram kamu.

**Log-magnitudo.** Besaran mentah berkisar antara 5-6 kali lipat. Ambil `log(|X| + 1e-6)` atau `20 * log10(|X|)` untuk mengompresi rentang dinamis. Setiap jalur produksi menggunakan besaran log, bukan besaran mentah.

**Skala mel.** Frekuensi `f` dalam Hz dipetakan ke mel `m` oleh `m = 2595 * log10(1 + f / 700)`. Pemetaannya kira-kira linier di bawah 1 kHz dan kira-kira logaritmik di atas. Wadah 80 mel yang mencakup 0–8 kHz adalah input ASR standar.

**Mel filterbank.** Seperangkat filter segitiga dengan distance yang sama pada skala mel. Setiap filter adalah jumlah tertimbang dari nampan FFT yang berdekatan. Mengalikan besaran STFT dengan matrix filterbank menghasilkan spektogram mel dalam satu matmul.

**Spektogram log-mel.** `log(mel_spec + 1e-10)`. Input bisikan. Input Parkit. Input SeamlessM4T. Tampilan depan audio universal 2026.

**MFCCs.** Ambil spektogram log-mel, terapkan DCT (tipe II), pertahankan 13 koefisien pertama. Dekorasi feature dan kompres lebih lanjut. Feature dominan hingga sekitar tahun 2015 ketika CNN/Transformers pada log-mel mentah menyusul. Masih digunakan dalam pengenalan speaker (x-vectors, ECAPA).

**Pertukaran resolusi.** FFT yang lebih besar = resolusi frekuensi yang lebih baik tetapi resolusi waktu yang lebih buruk. 25 mdtk / 10 mdtk adalah default audio-ML; 50 mdtk / 12,5 mdtk untuk musik; 5 ms / 2 ms untuk deteksi sementara (drum hit, plosif).

## Build

### Langkah 1: bingkai bentuk gelombang

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

Klip 16 kHz 10 detik dengan `frame_len=400, hop=160` menghasilkan 998 frame.

### Langkah 2: Jendela Hann

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

Kalikan berdasarkan elemen sebelum FFT. Menghapus kebocoran spektral yang disebabkan oleh pemotongan pada titik akhir yang bukan nol.

### Langkah 3: Besaran STFT

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```Produksi menggunakan `torch.stft` atau `librosa.stft` (didukung FFT, di-vector-kan). Lingkaran di sini bersifat pedagogis; itu berjalan pada klip pendek di `code/main.py`.

### Langkah 4: mel filterbank

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

80 mel mencakup 0–8 kHz dengan `n_fft=400` menghasilkan matrix `(80, 201)`. Kalikan magnitudo `(n_frames, 201)` STFT dengan transpose untuk mendapatkan spektogram `(n_frames, 80)` mel.

### Langkah 5: log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

Alternatif umum: `librosa.power_to_db` (dB yang dinormalisasi referensi), `10 * log10(power + eps)`. Whisper menggunakan klip yang lebih terlibat + rutinitas normalisasi (lihat `log_mel_spectrogram` Whisper).

### Langkah 6: MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

Terapkan DCT ke setiap frame log-mel, pertahankan 13 koefisien pertama. Itu adalah matrix MFCC kamu. Koefisien pertama biasanya dihilangkan (mengkodekan energi keseluruhan).

## Pakai

Tumpukan tahun 2026:

| Tugas | Feature |
|------|----------|
| ASR (Berbisik, Parkit, SeamlessM4T) | 80 log-mel, hop 10 mdtk, jendela 25 mdtk |
| Model akustik TTS (VITS, F5-TTS, Kokoro) | 80 mels, lompatan 5–12 ms untuk kontrol temporal yang baik |
| Klasifikasi audio (AST, PANN, BEAT) | 128 log-mel, 10 ms hop |
| Embedding speaker (ECAPA-TDNN, WavLM) | 80 log-mel atau SSL bentuk gelombang mentah |
| Musik (MusicGen, Audio Stabil 2) | Token diskrit EnCodec (bukan mels) |
| Pencarian kata kunci | 40 MFCC untuk perangkat kecil |

Aturan praktisnya: **jika kamu tidak mengerjakan musik, mulailah dengan 80 log-mel.** Weight pembuktian ada pada penyimpangan apa pun.

## Kesalahan yang masih dikirimkan pada tahun 2026

- **Ketidakcocokan jumlah mel.** Latihan dengan 80 mels, inference dengan 128 mels. Kegagalan diam-diam. Catat bentuk feature di kedua ujungnya.
- **Ketidakcocokan laju sample di bagian hulu.** Mel yang dihitung pada 22,05 kHz terlihat berbeda dari 16 kHz. Perbaiki SR *sebelum* fiturisasi.
- **dB vs log.** Whisper mengharapkan log-mel, bukan dB-mel. Beberapa pipeline pipa HF mendeteksi secara otomatis; code khusus kamu tidak akan melakukannya.
- **Penyimpangan normalisasi.** Normalisasi perucapan selama training, normalisasi global selama inference. Bug produksi yang menggandakan WER.
- **Kebocoran dari padding.** Zero-padding pada ujung klip menghasilkan spektrum datar pada frame tambahan. Pad secara simetris atau meniru.

## Kirim

Simpan sebagai `outputs/skill-feature-extractor.md`. Jenis feature pengambilan keterampilan, jumlah mel, frame/hop, dan normalisasi untuk target model tertentu.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Ini mensintesis kicauan (frekuensi menyapu 200 → 4000 Hz) dan mencetak argmax mel bin per frame. Plot (opsional) dan pastikan cocok dengan sapuan.
2. **Sedang.** Jalankan ulang dengan `n_mels` di `{40, 80, 128}` dan `frame_len` di `{200, 400, 800}`. Ukur bandwidth puncak tajam melintasi sumbu waktu. Kombo manakah yang paling mampu menyelesaikan kicauan tersebut?
3. **Sulit.** Implementasikan `power_to_db` dan bandingkan akurasi ASR pengklasifikasi CNN kecil di AudioMNIST menggunakan (a) log-mel mentah, (b) dB-mel dengan `ref=max`, (c) MFCC-13 + delta + delta-delta. Laporkan akurasi peringkat 1 teratas.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Bingkai | Sepotong | Potongan bentuk gelombang 25 ms diumpankan ke satu FFT. |
| Lompatan | Langkah | Sample di antara frame yang berurutan; 10 ms adalah default ASR. |
| Jendela | Hal Hann/Hamming | Pengganda titik yang memperkecil tepi bingkai menjadi nol. |
| STFT | Generator spektogram | FFT berbingkai + berjendela; menghasilkan matrix waktu × frekuensi. |
| Mel | Frekuensi melengkung | Skala log-persepsi; `m = 2595·log10(1 + f/700)`. |
| Bank Filter | Matrix | Filter segitiga yang memproyeksikan STFT ke wadah mel. |
| Log-mel | Input bisikan | `log(mel_spec + eps)`; distandarisasi pada tahun 2026. |
| MFCC | Feature jadul | DCT dari log-mel; 13 kopi, berhubungan dengan dekorasi. |

## Bacaan Lanjutan

- [Davis, Mermelstein (1980). Perbandingan representasi parametrik untuk pengenalan kata bersuku kata satu](https://ieeexplore.ieee.org/document/1163420) — makalah MFCC.
- [Stevens, Volkmann, Newman (1937). Skala Pengukuran Nada Besaran Psikologis](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — skala mel asli.
- [OpenAI — Sumber Whisper, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — baca implementasi referensi.
- [dokumen ekstraksi feature librosa](https://librosa.org/doc/main/feature.html) — referensi untuk `mfcc`, `melspectrogram`, dan hop/window.
- [NVIDIA NeMo — preprocessing audio](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — pipeline skala produksi untuk model Parkit + Canary.
