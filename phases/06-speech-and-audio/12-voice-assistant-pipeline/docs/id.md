# Membangun Pipeline Asisten Suara — Batu Penjuru Fase 6

> Semuanya dari lesson 01-11, digabungkan menjadi satu. Build asisten suara yang mendengarkan, memberi alasan, dan membalas. Pada tahun 2026, hal ini merupakan masalah teknik yang terpecahkan, bukan masalah penelitian – namun rincian integrasi menentukan apakah produk tersebut dapat dikirimkan.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Fase 11 · 09 (Pemanggilan Fungsi); Fase 14 · 01 (Agen Loop)
**Waktu:** ~120 menit

## Masalah

Build asisten menyeluruh:

1. Menangkap input mikrofon (mono 16 kHz).
2. Mendeteksi awal/akhir ucapan pengguna.
3. Mentranskripsikan streaming.
4. Melewati transkrip ke LLM yang dapat memanggil alat (timer, cuaca, kalender).
5. Mengalirkan teks LLM ke TTS.
6. Memutar audio kembali ke pengguna.
7. Berhenti jika pengguna menyela di tengah respons.

Target latensi: byte audio TTS pertama dalam waktu 800 ms setelah pengguna menyelesaikan ucapannya di CPU laptop. Target kualitas: tidak ada kata yang terlewat, tidak ada subtitle berhalusinasi saat diam, tidak ada kebocoran kloning suara, tidak ada keberhasilan injeksi yang cepat.

## Konsep

![Pipeline asisten suara: mikrofon → VAD → STT → LLM+alat → TTS → speaker](../assets/voice-assistant.svg)

### Tujuh komponen

1. **Pengambilan audio.** Mikrofon → mono 16 kHz → potongan 20 ms. Biasanya `sounddevice` dengan Python atau AudioUnit/ALSA/WASAPI asli dalam produksi.
2. **VAD (Lesson 11).** Silero VAD @ ambang batas 0,5, ucapan minimum 250 mdtk, keheningan hang-over 500 mdtk. Sinyal "mulai" dan "berakhir".
3. **Streaming STT (Lesson 4-5).** Whisper-streaming, Parkit-TDT, atau Deepgram Nova-3 (API). Transkrip sebagian + akhir.
4. **LLM dengan pemanggilan alat.** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. Skema JSON untuk alat. Token streaming.
5. **Streaming TTS (Lesson 7).** Kokoro-82M (pembukaan tercepat) atau Cartesia Sonic (komersial). Mulai TTS setelah 20 token LLM.
6. **Pemutaran.** Speaker keluar; opus-encode untuk jaringan bandwidth rendah.
7. **Penanganan interupsi.** Jika VAD aktif selama pemutaran TTS, hentikan pemutaran, batalkan LLM, mulai ulang STT.

### Tiga mode kegagalan yang akan kamu alami

1. **Klip kata pertama.** VAD terlambat memulai irama. Kata "hei" dari pengguna tidak ada. Mulai ambang batas pada 0,3, bukan 0,5.
2. **Perplexity interupsi respons tengah.** LLM terus muncul setelah interupsi pengguna; asisten berbicara tentang pengguna. Kawat VAD → batalkan-LLM.
3. **Halusinasi senyap.** Bisikan mengeluarkan suara "Terima kasih telah menonton" pada bingkai pemanasan senyap. Selalu gerbang VAD.

### Tumpukan referensi produksi 2026

| Tumpukan | Latensi | Lisensi | Catatan |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | API komersial | Default industri 2026 |
| Pipecat + Streaming Bisikan + GPT-4o + Kokoro | 500-800 ms | sebagian besar terbuka | Ramah DIY |
| Moshi (dupleks penuh) | 200-300 ms | CC-BY 4.0 | Model tunggal; arsitektur berbeda, lesson 15 |
| Vapi / Ceritakan Kembali (dikelola) | 300-500 ms | komersial | Tercepat untuk diluncurkan; kustomisasi terbatas |
| Bisikan.cpp + llama.cpp + Kokoro-ONNX | luring | buka | Privasi / tepi |

## Build

### Langkah 1: pengambilan mikrofon dengan chunking (pseudocode)

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Langkah 2: Pengambilan giliran dengan gerbang VAD

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Langkah 3: streaming STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Langkah 4: pemanggilan alat di dalam loop LLM

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Langkah 5: penanganan interupsi

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## Pakai

Lihat `code/main.py` untuk simulasi runnable yang menyambungkan ketujuh komponen dengan model stub, sehingga kamu dapat melihat bentuk pipa bahkan tanpa perangkat keras. Untuk implementasi nyata, tukar stub dengan:

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` atau `openai-whisper`
- `openai` (`gpt-4o`) atau `anthropic`
- `kokoro` atau `cartesia`
- `sounddevice` untuk I/O

## Jebakan

- **Mencatat PII selamanya.** Audio putaran penuh adalah PII di sebagian besar wilayah hukum. Retensi 30 hari, dienkripsi saat tidak digunakan.
- **Tidak boleh masuk dengan tongkang.** Pengguna akan menyela. Asisten kamu harus berhenti bicara.
- **TTS yang memblokir.** TTS Sinkron memblokir loop peristiwa. Gunakan async atau utas terpisah.
- **Tidak ada penanganan kesalahan panggilan alat.** Alat gagal. LLM harus mengembalikan kesalahan + coba lagi sekali, lalu menurunkannya dengan baik.
- **Filter halusinasi berlebihan.** Filter berlebihan dan asisten mengulangi "Saya tidak bisa membantu dengan itu." Di bawah filter dan ia mengatakan apa pun. Kalibrasi pada set yang ditahan.
- **Tidak ada opsi kata bangun.** Selalu mendengarkan adalah tanggung jawab privasi. Tambahkan gerbang kata bangun (Porcupine atau openWakeWord).

## Kirim

Simpan sebagai `outputs/skill-voice-assistant-architect.md`. Mengingat batasan anggaran + skala + bahasa + kepatuhan, buatlah spesifikasi tumpukan penuh.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Ini mensimulasikan satu putaran penuh ujung ke ujung dengan modul rintisan dan mencetak latensi per phase.
2. **Medium.** Ganti stub STT dengan model Whisper asli di `.wav` yang telah direkam sebelumnya. Ukur WER dan latensi ujung ke ujung.
3. **Sulit.** Tambahkan pemanggilan alat: implementasi `get_weather` (API apa pun) dan `set_timer`. Rutekan LLM melalui alat dan verifikasi bahwa ketika pengguna mengatakan "setel pengatur waktu 5 menit" fungsi yang tepat akan diaktifkan dan balasan lisan mengonfirmasinya.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Putar | Perjalanan pulang pergi pengguna + asisten | Satu ucapan pengguna yang dibatasi VAD + satu respons LLM-TTS. |
| Tongkang masuk | Interupsi | Pengguna berbicara sementara asisten berbicara; asisten berhenti. |
| Build kata | "Hai asisten" | Detektor kata kunci pendek; Landak, Anak Salju, openWakeWord. |
| Menunjuk akhir | Putar akhir | VAD + diamkan keputusan yang telah selesai dilakukan pengguna. |
| Pra-putar | Buffer pra-ucapan | Pertahankan audio 200-400 ms sebelum VAD diaktifkan untuk menghindari klip kata pertama. |
| Panggilan alat | Pemanggilan fungsi | LLM memancarkan JSON; pengiriman waktu proses; result diumpankan kembali dalam loop. |

## Bacaan Lanjutan

- [LiveKit — panduan memulai agen suara](https://docs.livekit.io/agents/) — referensi tingkat produksi.
- [Pipecat — contoh agen suara](https://github.com/pipecat-ai/pipecat) — framework yang ramah DIY.
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — jalur suara asli yang dikelola.
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — referensi dupleks penuh (Lesson 15).
- [Kata bangun landak](https://picovoice.ai/products/porcupine/) — gerbang kata bangun.
- [Antropik — panduan penggunaan alat](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — pemanggilan fungsi LLM.
