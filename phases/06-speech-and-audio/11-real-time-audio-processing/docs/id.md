# Pemrosesan Audio Waktu Nyata

> Pipeline pipa batch memproses file. Pipeline real-time memproses 20 milidetik berikutnya sebelum 20 milidetik berikutnya tiba. Setiap AI percakapan, studio siaran, dan bot telepon hidup dan mati oleh anggaran latensi ini.

**Type:** Build
**Language:** Python, Rust
**Prerequisites:** Fase 6 · 02 (Spektogram), Fase 6 · 04 (ASR), Fase 6 · 07 (TTS)
**Waktu:** ~75 menit

## Masalah

kamu menginginkan asisten suara yang terasa hidup. Latensi giliran percakapan manusia adalah ~230 ms (diam hingga merespons). Apa pun di atas 500 ms terasa seperti robot; di atas 1500 ms terasa rusak. Anggaran untuk siklus penuh **mendengar → memahami → merespons → berbicara** pada tahun 2026 adalah:

| Phase | Anggaran |
|-------|--------|
| Mikrofon → penyangga | 20 ms |
| VAD | 10 ms |
| ASR (pengaliran) | 150 ms |
| LLM (token pertama) | 100 ms |
| TTS (potongan pertama) | 100 ms |
| Render → pembicara | 20 ms |
| **Jumlah** | **~400 ms** |

Moshi (Kyutai, 2024) mencatat waktu dupleks penuh 200 ms. Jam GPT-4o-realtime (2024) ~320 mdtk. Jaringan pipa bertingkat pada tahun 2022 dikirimkan dengan kecepatan 2500 ms. Peningkatan 10× berasal dari tiga teknik: (1) streaming di mana-mana, (2) pipeline asinkron dengan hasil parsial, (3) pembangkitan yang dapat diinterupsi.

## Konsep

![Streaming pipeline audio dengan ring buffer, gerbang VAD, interupsi](../assets/real-time.svg)

**Bingkai / potongan / jendela.** Audio real-time mengalir sebagai blok berukuran tetap. Pilihan umum: 20 ms (320 sample pada 16 kHz). Segala sesuatu di hilir harus mengikuti irama ini.

**Buffer cincin.** Buffer melingkar berukuran tetap. Thread produsen menulis frame baru, thread konsumen membaca. Mencegah alokasi di jalur panas. Ukuran latensi maksimum × laju sample; cincin 16 kHz 2 detik = 32.000 sample.

**VAD (Deteksi Aktivitas Suara).** Gerbang hilir berfungsi saat tidak ada orang yang berbicara. Silero VAD 4.0 (2024) berjalan <1 ms per 30 ms frame pada CPU. `webrtcvad` adalah alternatif lama.

**Streaming ASR.** Model yang mengeluarkan sebagian transkrip saat audio tiba. Parkit-CTC-0.6B dalam mode streaming (NeMo, 2024) menghasilkan 2–5% WER pada latensi 320 ms. Whisper-Streaming (Macháček dkk., 2023) membagi Whisper untuk streaming dekat dengan latensi ~2 detik.

**Interupsi.** Saat pengguna berbicara saat asisten berbicara, kamu harus (a) mendeteksi masuknya tongkang, (b) menghentikan TTS, (c) membuang sisa output LLM. Semua dalam 100 ms, atau pengguna merasakan asisten tunarungu.

**Transportasi WebRTC Opus.** Bingkai 20 ms, 48 ​​kHz, kecepatan bit adaptif 8–128 kbps. Standar untuk browser dan seluler. LiveKit, Daily.co, Pion adalah tumpukan tahun 2026 untuk membuat aplikasi suara.

**Jitter buffer.** Paket jaringan tiba rusak/terlambat. Buffer jitter menyusun ulang dan menghaluskan; terlalu kecil → celah yang terdengar, terlalu besar → latensi. Biasanya 60–80 ms.

### Masalah umum

- **Pertentangan thread.** Model GIL + berat Python dapat membuat thread audio tidak berfungsi. Gunakan pustaka audio C-callback (sounddevice, PortAudio) dan jauhkan Python dari jalur populer.
- **Latensi konversi laju sample.** Pengambilan sample ulang di dalam pipeline menambah waktu 5–20 ms. Ambil sample ulang terlebih dahulu atau gunakan resampler latensi nol (PolyPhase, `soxr_hq`).
- **TTS priming.** Bahkan TTS cepat seperti Kokoro memiliki pemanasan 100–200 mdtk berdasarkan permintaan pertama. Model cache + hangatkan dengan dummy run sebelum giliran nyata pertama.
- **Pembatalan gema.** Tanpa AEC, output TTS masuk kembali ke mikrofon dan memicu ASR pada suara bot itu sendiri. WebRTC AEC3 adalah sumber terbuka default.

## Build

### Langkah 1: penyangga cincin

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```Kapasitas menentukan latensi buffering maksimal. 32.000 sample pada 16 kHz = 2 detik.

### Langkah 2: Gerbang VAD

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Ganti dengan Silero VAD dalam produksi:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Langkah 3: streaming ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Langkah 4: penangan interupsi

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

Bergantung pada I/O asinkron dan streaming TTS yang dapat dibatalkan. WebRTC peerconnection.stop() pada trek audio adalah cara kanonik.

## Pakai

Tumpukan tahun 2026:

| Layer | Pilih |
|-------|------|
| Transportasi | LiveKit (WebRTC) atau Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parkit-CTC-0.6B atau Streaming Bisikan |
| Token pertama LLM | Groq, Cerebras, streaming vLLM |
| Streaming TTS | Kokoro atau ElevenLabs Turbo v2.5 |
| Gema batal | WebRTC AEC3 |
| Asli ujung ke ujung | OpenAI Realtime API atau Moshi |

## Jebakan

- **Buffer 500 ms agar aman.** Buffer *adalah* tingkat latensi kamu. Kecilkan.
- **Tidak embed thread.** Callback audio pada thread dengan prioritas lebih rendah dari UI = gangguan saat dimuat.
- **Bagian TTS terlalu kecil.** Potongan di bawah 200 ms membuat artefak vocoder terdengar. Potongan 320 ms adalah titik terbaiknya.
- **Tidak ada buffer jitter.** Jaringan sebenarnya gelisah; tanpa menghaluskan kamu akan mendapat pop.
- **Penanganan error satu kali.** Pipeline audio harus anti error. Satu pengecualian mematikan sesi tersebut.

## Kirim

Simpan sebagai `outputs/skill-realtime-designer.md`. Rancang pipeline audio real-time dengan anggaran latensi nyata per phase.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Mensimulasikan buffer cincin + VAD energi; mencetak latensi phase untuk streaming 10 detik palsu.
2. **Medium.** Menggunakan `sounddevice`, buat loop passthrough yang memproses mikrofon kamu dalam frame 20 ms dan mencetak status VAD di setiap frame.
3. **Sulit.** Buat pengujian gema dupleks penuh dengan `aiortc`: browser → WebRTC → Python → WebRTC → browser. Ukur latensi kaca-ke-kaca dengan pulsa 1 kHz.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Penyangga dering | Antrian melingkar | FIFO ukuran tetap, bebas kunci (atau dikunci SPSC) untuk bingkai audio. |
| VAD | Gerbang keheningan | Model atau penandaan heuristik ucapan vs non-ucapan. |
| Streaming ASR | STT waktu nyata | Memancarkan sebagian teks saat audio tiba; melihat ke depan yang dibatasi. |
| Penyangga jitter | Jaringan lebih lancar | Antrian pemesanan ulang paket yang rusak; Biasanya 60–80 ms. |
| MEA | Pembatalan gema | Mengurangi jalur umpan balik speaker-ke-mikrofon. |
| Tongkang masuk | Interupsi pengguna | Sistem mendeteksi ucapan pengguna di tengah TTS; harus membatalkan pemutaran. |
| Dupleks penuh | Kedua arah secara bersamaan | Pengguna dan bot dapat berbicara secara bersamaan; Moshi adalah dupleks penuh. |

## Bacaan Lanjutan

- [Macháček dkk. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — Whisper yang hampir streaming terpotong.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — latensi dupleks penuh 200 ms.
- [Framework Agen LiveKit (2024)](https://docs.livekit.io/agents/) — orkestrasi agen audio produksi.
- [Repo Silero VAD](https://github.com/snakers4/silero-vad) — kurang dari 1 ms VAD, Apache 2.0.
- [makalah WebRTC AEC3](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — pembatalan gema pada sumber terbuka.
