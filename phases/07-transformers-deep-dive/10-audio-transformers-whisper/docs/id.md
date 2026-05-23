# Audio Transformers — Arsitektur Bisikan

> Audio adalah gambar frekuensi dari waktu ke waktu. Whisper adalah ViT yang memakan spektogram mel dan membalasnya.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 7 · 05 (Trafo Penuh), Fase 7 · 08 (Encoder-Decoder), Fase 7 · 09 (ViT)
**Waktu:** ~45 menit

## Masalah

Sebelum Whisper (OpenAI, Radford et al. 2022), pengenalan ucapan otomatis (ASR) yang canggih berarti wav2vec 2.0 dan HuBERT — ekstraktor feature yang diawasi sendiri ditambah kepala yang disetel dengan baik. Pipeline data berkualitas tinggi dan mahal, domain rapuh. Pengenalan ucapan multibahasa memerlukan model terpisah per rumpun bahasa.

Whisper membuat tiga taruhan:

1. **Latih semuanya.** 680.000 jam audio berlabel lemah yang diambil dari internet dalam 97 bahasa. Tidak ada korpus akademik yang bersih. Tidak ada label fonem.
2. **Model tunggal multi-tugas.** Satu decoder dilatih bersama tentang transkripsi, terjemahan, deteksi aktivitas suara, ID bahasa, dan stempel waktu melalui token tugas.
3. **Transformer encoder-decoder standar.** Encoder menggunakan spektogram log-mel. Decoder menghasilkan token teks secara otomatis. Tanpa vocoder, tanpa CTC, tanpa HMM.

Hasilnya: Whisper large-v3 kuat dalam aksen, noise, dan bahasa yang tidak memiliki data berlabel bersih. Ini adalah antarmuka ucapan default untuk setiap asisten suara sumber terbuka dan sebagian besar asisten suara komersial pada tahun 2026.

## Konsep

![Pipa bisikan: audio → mel → encoder → decoder → teks](../assets/whisper.svg)

### Langkah 1 — sample ulang + jendela

Audio pada 16 kHz. Klip/pad hingga 30 detik. Hitung spektogram log-mel: 80 mel bin, langkah 10 ms → ~3.000 frame × 80 feature. Ini adalah "gambar input" yang dilihat Whisper.

### Langkah 2 — batang konvolusional

Dua layer Conv1D dengan kernel 3 dan langkah 2 mengurangi 3.000 frame menjadi 1.500. Membelah dua panjang urutan tanpa menambahkan banyak parameter.

### Langkah 3 — pembuat enkode

Encoder Transformer 24 lapis (untuk ukuran besar) dengan lebih dari 1.500 langkah waktu. Pengkodean posisi sinusoidal, attention diri, GELU FFN. Menghasilkan 1.500 × 1.280 status tersembunyi.

### Langkah 4 — dekoder

Dekoder Transformer 24 lapis. Ini secara otomatis menghasilkan token dari kosakata BPE yang merupakan superset dari GPT-2 dengan beberapa token khusus audio tertentu.

### Langkah 5 — token tugas

Prompt decoder dimulai dengan token kontrol yang memberi tahu model apa yang harus dilakukan:

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

atau

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

Model tersebut dilatih pada konvensi ini. kamu mengontrol tugas berdasarkan awalan. Setara dengan penyetelan instruksi pada tahun 2026, tetapi diterapkan pada ucapan.

### Langkah 6 — output

Pencarian berkas (lebar 5) dengan ambang log-prob. Stempel waktu diprediksi setiap 0,02 detik audio ketika token `<|notimestamps|>` tidak ada.

### Ukuran bisikan

| Model | Param | Layer | d_model | Kepala | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Kecil | 39M | 4 | 384 | 6 | ~1 GB |
| Basis | 74M | 6 | 512 | 8 | ~1 GB |
| Kecil | 244M | 12 | 768 | 12 | ~2 GB |
| Sedang | 769M | 24 | 1024 | 16 | ~5 GB |
| Besar | 1550M | 32 | 1280 | 20 | ~10 GB |
| Besar-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Besar-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (dekoder 4 lapis) |

Large-v3-turbo (2024) memotong dekoder dari 32 layer menjadi 4. Dekode 8× lebih cepat dengan regresi <1 titik WER. Pembukaan kecepatan dekode itulah yang menjadi alasan Whisper-turbo menjadi default untuk agen suara real-time pada tahun 2026.

### Apa yang tidak dilakukan Whisper- Tidak ada diarisasi (siapa yang berbicara). Sandingkan dengan pyannote untuk itu.
- Tidak ada streaming real-time secara asli — jendela 30 detik sudah diperbaiki. Pembungkus modern (`faster-whisper`, `WhisperX`) langsung streaming melalui VAD + tumpang tindih.
- Tidak ada konteks bentuk panjang yang melebihi 30 detik tanpa pengelompokan eksternal. Berfungsi dengan baik dalam praktiknya karena ucapan manusia jarang memerlukan konteks jangka panjang untuk transkripsi.

### Pemandangan 2026

| Tugas | Model | Catatan |
|------|-------|-------|
| Bahasa Inggris ASR | Bisikan-turbo, Moonshine | Moonshine 4× lebih cepat di edge |
| ASR multibahasa | Bisikan-besar-v3 | 97 bahasa |
| Streaming ASR | bisikan lebih cepat + VAD | Target latensi 150 ms dapat dicapai |
| TTS | Piper, XTTS-v2, Kokoro | Pola encoder-decoder, namun berbentuk Bisikan |
| Audio + bahasa | AudioLM, MulusM4T | Token teks + token audio dalam satu trafo |

## Build

Lihat `code/main.py`. Kami tidak melatih Whisper — kami membuat pipeline spektogram log-mel + pemformat prompt token tugas. Itu adalah bagian yang sebenarnya kamu sentuh dalam produksi.

### Langkah 1: sintesis audio

Hasilkan gelombang sinus 1 detik pada 440 Hz dengan sample pada 16 kHz. 16.000 sample.

### Langkah 2: spektogram log-mel (disederhanakan)

Spektogram mel penuh membutuhkan FFT. Kami melakukan pembingkaian sederhana + versi energi per bingkai yang menunjukkan alur tanpa memerlukan `librosa`:

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Bingkai = 25 mdtk, lompatan = 10 mdtk. Cocok dengan jendela Whisper. Energi per-frame menggantikan mel bin untuk pedagogi.

### Langkah 3: lanjutkan hingga 30 detik

Whisper selalu memproses potongan 30 detik. Pad (atau klip) spektogram ke 3.000 frame.

### Langkah 4: buat token prompt

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

Itulah keseluruhan permukaan kendali tugas. Awalan 4 token.

## Pakai

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

Lebih cepat, kompatibel dengan OpenAI:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**Kapan memilih Whisper pada tahun 2026:**

- ASR multibahasa dengan satu model.
- Transkripsi audio yang berisik dan beragam.
- Penelitian / prototipe ASR — titik awal tercepat.

**Kapan memilih yang lain:**

- Streaming latensi sangat rendah di edge — Moonshine mengalahkan Whisper dengan kualitas yang sesuai.
- AI percakapan real-time yang membutuhkan <200 ms — streaming ASR khusus.
- Diarisasi pembicara — Whisper tidak melakukan ini; baut pada pyannote.

## Kirim

Lihat `outputs/skill-asr-configurator.md`. Keterampilan ini memilih model ASR, mendekode parameter, dan alur pra-pemrosesan untuk aplikasi ucapan baru.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Konfirmasikan jumlah frame untuk sinyal 1 detik pada 16 kHz dengan hop 10 ms adalah ~100 frame. Selama 30 detik: ~3.000 bingkai.
2. **Medium.** Buat spektogram log-mel lengkap menggunakan `numpy.fft`. Verifikasi wadah 80 mel cocok dengan `librosa.feature.melspectrogram(n_mels=80)` dalam kesalahan numerik.
3. **Sulit.** Menerapkan inference streaming: membagi audio ke dalam jendela berdurasi 10 detik dengan tumpang tindih selama 2 detik, menjalankan Whisper pada setiap potongan, menggabungkan transkrip. Ukur tingkat kesalahan kata vs kesalahan single-pass pada sample podcast 5 menit.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Spektogram mel | "Gambar audio" | Representasi 2D: nampan frekuensi di satu sumbu, kerangka waktu di sumbu lainnya; energi berskala log per sel. |
| Log-mel | "Apa yang Bisikan lihat" | Spektogram mel melewati log; mendekati persepsi manusia tentang kenyaringan. |
| Bingkai | "Satu kali irisan" | Jendela sample 25 ms; tumpang tindih dengan langkah 10 ms. |
| Token tugas | "Awalan cepat untuk pidato" | Token khusus seperti `<|transcribe|>` / `<|translate|>` di prompt decoder. |
| Deteksi aktivitas suara (VAD) | "Temukan pidatonya" | Gerbang yang menghilangkan keheningan sebelum ASR; pemotongan memakan biaya besar-besaran. |
| CTC | "Klasifikasi Temporal Koneksionis" | Loss ASR klasik untuk training tanpa penyelarasan; Bisikan TIDAK menggunakannya. |
| Bisikan-turbo | "Decoder kecil, encoder lengkap" | encoder v3 besar + decoder 4 lapis; Dekode 8× lebih cepat. |
| Bisikan lebih cepat | "Pembungkus produksi" | implementasi ulang CTranslate2; kuantisasi int8; 4× lebih cepat dari referensi OpenAI. |

## Bacaan Lanjutan

- [Radford dkk. (2022). Pengenalan Ucapan yang Kuat melalui Pengawasan Lemah Berskala Besar](https://arxiv.org/abs/2212.04356) — Kertas bisikan.
- [Repo OpenAI Whisper](https://github.com/openai/whisper) — code referensi + weight model. Baca `whisper/model.py` untuk melihat batang Conv1D + encoder + decoder dari atas ke bawah dalam ~400 baris.
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — logika beam-search + task-token yang dijelaskan pada Langkah 5–6 ada di sini; 500 baris, dapat dibaca sepenuhnya.
- [Baevski dkk. (2020). wav2vec 2.0: Framework untuk Pembelajaran Representasi Ucapan yang Diawasi Sendiri](https://arxiv.org/abs/2006.11477) — pendahulu; masih feature SOTA di beberapa pengaturan.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — pembungkus produksi, 4× lebih cepat dari referensi.
- [Jia dkk. (2024). Moonshine: Pengenalan Ucapan untuk Transkripsi Langsung dan Prompt Suara](https://arxiv.org/abs/2410.15608) — ASR ramah tepi tahun 2024, berbentuk Bisikan namun lebih kecil.
- [Blog HuggingFace — "Sempurnakan Bisikan Untuk ASR Multibahasa dengan 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — resep penyempurnaan kanonik termasuk praprosesor spektogram mel dan penanganan stempel waktu token.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — implementasi penuh (encoder, decoder, cross-attention, generation) yang mencerminkan diagram arsitektur lesson.
