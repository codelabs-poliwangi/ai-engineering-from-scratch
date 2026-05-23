# Whisper — Arsitektur & Penyempurnaan

> Whisper adalah encoder-decoder Transformer jendela berdurasi 30 detik, dilatih pada pasangan audio-teks multibahasa dengan pengawasan lemah selama 680 ribu jam. Satu arsitektur, banyak tugas, tangguh dalam 99 bahasa. Referensi ASR tahun 2026.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 6 · 04 (ASR), Fase 5 · 10 (Attention), Fase 7 · 05 (Trafo Penuh)
**Waktu:** ~75 menit

## Masalah

Whisper, dirilis oleh OpenAI pada September 2022, adalah model ASR pertama yang dikirimkan sebagai komoditas: menempelkan audio, mendapatkan teks, 99 bahasa, tahan terhadap kebisingan, dan dijalankan di laptop. Pada tahun 2024 OpenAI telah mengirimkan varian Large-v3 dan Turbo; pada tahun 2026, Whisper menjadi dasar default untuk segala hal mulai dari transkripsi podcast, asisten suara, hingga subtitle YouTube.

Namun Whisper bukanlah pipeline pipa yang bisa kamu perlakukan sebagai kotak hitam selamanya. Pergeseran domain mematikannya — jargon teknis, aksen pembicara, kata benda, klip pendek, keheningan. kamu perlu tahu:

1. Apa yang sebenarnya ada di dalamnya.
2. Bagaimana cara memberikan audio yang dipotong, streaming, atau berdurasi panjang dengan benar.
3. Kapan melakukan penyesuaian dan bagaimana caranya.

## Konsep

![Whisper encoder-decoder, tugas, inference terpotong, penyempurnaan](../assets/whisper.svg)

**Arsitektur.** Encoder-decoder trafo standar.

- Input: spektogram log-mel 30 detik, 80 mels, hop 10 ms → 3000 frame. Klip yang lebih pendek tidak memiliki bantalan, klip yang lebih panjang akan dipotong-potong.
- Encoder: conv-downsample (langkah 2) + `N` blok Transformer. Untuk Large-v3: 32 layer, 1280-redup, 20 kepala.
- Decoder: `N` blok Transformer dengan self-attn kausal + cross-attn ke output encoder. Ukurannya sama dengan pembuat enkode.
- Output: token BPE melalui vocab 51.865 token.

Large-v3 memiliki 1,55 miliar parameter. Turbo menggunakan dekoder 4 lapis (dari 32), memotong latensi 8× dengan hit WER <1%.

**Format prompt.** Whisper adalah model multitask yang dikendalikan oleh token khusus dalam prompt dekoder:

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — tag bahasa; memaksa perilaku terjemahan-vs-transkripsi.
- `<|transcribe|>` atau `<|translate|>` — menerjemahkan output bahasa Inggris dari input bahasa apa pun, atau kata demi kata.
- `<|notimestamps|>` — melewati stempel waktu tingkat kata (lebih cepat).

Perintahnya memungkinkan satu model melakukan banyak tugas. Ubah `<|en|>` menjadi `<|fr|>` dan itu akan mentranskripsikan bahasa Prancis.

**Jendela 30 detik.** Semuanya di-embed ke 30 detik. Klip yang lebih panjang perlu dipotong; klip yang lebih pendek diberi bantalan. Windows tidak dialirkan secara asli — inilah alasan mengapa WhisperX, Whisper-Streaming, dan fast-whisper ada.

**Normalisasi log-mel.** `(log_mel - mean) / std` yang statistiknya berasal dari korpus training Whisper sendiri. kamu *harus* menggunakan preprocessing Whisper (`whisper.audio.log_mel_spectrogram`), bukan `librosa.feature.melspectrogram`.

### Varian pada tahun 2026

| Varian | Param | Latensi (A100) | WER (LibriSpeech-bersih) |
|---------|--------|----------------|------------------------|
| Kecil | 39M | 1× waktu nyata | 5,4% |
| Basis | 74M | 1× | 4,1% |
| Kecil | 244M | 1× | 3,0% |
| Sedang | 769M | 1× | 2,7% |
| Besar-v3 | 1,55B | 2× | 1,8% |
| Besar-v3-turbo | 809M | 8× | 1,58% |
| Streaming Bisikan (2024) | 1,55B | streaming | 2,0% |

### Penyempurnaan

Alur kerja kanonik pada tahun 2026:1. Kumpulkan audio domain target berdurasi 10–100 jam dengan transkrip yang selaras.
2. Jalankan `transformers.Seq2SeqTrainer` dengan panggilan balik `generate_with_loss`.
3. Hemat parameter: LoRA pada `q_proj`, `k_proj`, `v_proj` layer attention mengurangi memori GPU 4× dengan biaya <0,3 WER.
4. Bekukan encoder jika kamu memiliki waktu <10 jam. Hanya setel dekodernya.
5. Gunakan tokenizer dan format prompt milik Whisper; jangan pernah menukar tokenizer.

Hasil komunitas: penyempurnaan Medium dengan dikte medis selama 20 jam menurunkan WER dari 12% menjadi 4,5% pada kosakata medis. Penyempurnaan Turbo pada 4 jam bahasa Islandia menurunkan WER dari 18% menjadi 6%.

## Build

### Langkah 1: jalankan Whisper di luar kotak

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

Default kunci yang harus selalu kamu timpa: `temperature=0.0` (default pengambilan sample ke 0.0 → 0.2 → 0.4 … rantai fallback), `condition_on_previous_text=False` (mencegah masalah halusinasi berjenjang), dan `no_speech_threshold=0.6` (deteksi senyap).

### Langkah 2: dipotong-potong dalam bentuk panjang

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX menambahkan (1) Silero VAD gating, (2) penyelarasan tingkat kata melalui wav2vec 2.0, (3) diarisasi melalui `pyannote.audio`. Pekerja keras tahun 2026 untuk transkripsi produksi.

### Langkah 3: menyempurnakan dengan LoRA

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

Kemudian loop Pelatih standar. Pos pemeriksaan setiap 1000 langkah. Evaluasi dengan WER jika ditunda.

### Langkah 4: periksa apa yang dipelajari setiap layer

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

Visualisasikan dengan peta panas — kamu akan melihat perataan diagonal saat langkah dekoder memindai bingkai encoder. Diagonal itu adalah gagasan Whisper tentang cap waktu kata.

## Pakai

Tumpukan tahun 2026:

| Situasi | Pilih |
|-----------|------|
| Bahasa Inggris Umum, offline | Turbov3 besar melalui `whisperx` |
| Seluler / tepi | Bisikan-Tiny terkuantisasi (int8) atau Moonshine |
| Bentuk panjang multibahasa | Besar-v3 melalui `whisperx` + diarisasi |
| Bahasa dengan sumber daya rendah | Sempurnakan Medium atau Turbo dengan LoRA |
| Streaming (latensi 2 detik) | Streaming Bisikan atau Parkit-TDT |
| Stempel waktu tingkat kata | WhisperX (penyelarasan paksa melalui wav2vec 2.0) |

`faster-whisper` (backend CTranslate2) adalah runtime inference CPU+GPU tercepat pada tahun 2026 — 4× lebih cepat dibandingkan vanilla dengan output yang sama.

## Kesalahan yang masih dikirimkan pada tahun 2026

- **Teks halusinasi dalam keheningan.** Bisikan yang dilatih pada teks mencakup "Terima kasih telah menonton!", "Berlangganan!", lirik lagu. Selalu gerbang VAD sebelum menelepon.
- **`condition_on_previous_text` cascade.** Satu halusinasi mencemari jendela berikutnya. Setel `False` kecuali kamu memerlukan kelancaran antar bagian.
- **Padding klip pendek.** Klip berdurasi 2 detik yang diisi hingga 30 detik dapat membuat halusinasi dalam keheningan. Gunakan `pad=False` atau gerbang VAD.
- **Statistik mel salah.** Menggunakan mel librosa alih-alih Whisper menghasilkan output yang hampir acak. Gunakan `whisper.audio.log_mel_spectrogram`.

## Kirim

Simpan sebagai `outputs/skill-whisper-tuner.md`. Rancang alur penyempurnaan atau inference Whisper untuk domain tertentu.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Ini memberi token pada prompt gaya Whisper, menghitung anggaran bentuk yang didekodekan, dan mencetak jadwal potongan untuk klip 10 menit.
2. **Medium.** Instal `faster-whisper`, transkripsikan podcast 10 menit, bandingkan WER dengan transkrip manusia. Coba `language="auto"` vs dipaksa `language="en"`.
3. **Sulit.** Menggunakan HF `datasets`, pilih bahasa yang sulit digunakan Whisper (misalnya Urdu), sempurnakan Medium dengan LoRA selama 2 epoch dalam 2 jam, dan laporkan delta WER.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| jendela 30 detik | Batas bisikan | Tutup input keras; potong audio yang lebih panjang. |
| JADI | Awal transkrip | `<|startoftranscript|>` memulai prompt decoder. |
| Token stempel waktu | Penyelarasan temporal | Setiap offset 0,02 detik adalah token khusus dalam vocab 51k. |
| turbo | Varian cepat | Layer 4-decoder, 8× lebih cepat, regresi WER <1%. |
| BisikanX | Pembungkus bentuk panjang | VAD + Whisper + penyelarasan wav2vec + diarisasi. |
| Penyempurnaan LoRA | Penyetelan yang efisien | Tambahkan adaptor tingkat rendah ke attention; latih ~0,3% param. |
| Halusinasi | Kegagalan diam-diam | Whisper menghasilkan bahasa Inggris yang fasih dari kebisingan/keheningan. |

## Bacaan Lanjutan

- [Radford dkk. (2022). Kertas bisikan](https://arxiv.org/abs/2212.04356) — arsitektur asli dan resep training.
- [OpenAI (2024). Rilis Whisper Large-v3-turbo](https://github.com/openai/whisper/discussions/2363) — dekoder 4 lapis, percepatan 8×.
- [Bain dkk. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — bentuk panjang, selaras dengan kata, diarialisasi.
- [Systran — repo fast-whisper](https://github.com/SYSTRAN/faster-whisper) — didukung CTranslate2, 4× lebih cepat.
- [HuggingFace — Tutorial menyempurnakan Whisper](https://huggingface.co/blog/fine-tune-whisper) — panduan LoRA kanonik / full-FT.
