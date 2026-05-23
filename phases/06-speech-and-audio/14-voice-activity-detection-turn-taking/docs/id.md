# Deteksi Aktivitas Suara & Pengambilan Giliran — Silero, Cobra, dan Trik Flush

> Setiap agen suara hidup atau mati karena dua keputusan: apakah pengguna berbicara sekarang, dan apakah sudah selesai? VAD menjawab yang pertama. Deteksi giliran (VAD + silence-hangover + model titik akhir semantik) menjawab pertanyaan kedua. Jika kamu salah, asisten kamu akan memutus pengguna atau tidak pernah tutup mulut.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 6 · 11 (Audio Real-Time), Fase 6 · 12 (Asisten Suara)
**Waktu:** ~45 menit

## Masalah

Tiga keputusan berbeda yang diambil agen suara pada setiap potongan 20 ms:

1. **Apakah ini pidato berbingkai?** — VAD. Biner, per frame.
2. **Apakah pengguna sudah memulai ucapan baru?** — deteksi permulaan.
3. **Apakah pengguna sudah selesai?** — penunjuk akhir (turn-end).

Jawaban naif (ambang energi) gagal pada kebisingan apa pun - lalu lintas, keyboard, ocehan orang banyak. Jawaban tahun 2026: Silero VAD (terbuka, dipelajari secara mendalam) + model deteksi giliran (titik akhir semantik) + mabuk keheningan yang dikalibrasi VAD.

## Konsep

![Kaskade VAD: energi → Silero → detektor putaran → trik siram](../assets/vad-turn-taking.svg)

### Kaskade VAD tiga tingkat

**Tingkat 1: gerbang energi.** Termurah. Ambang batas RMS pada -40 dBFS. Menyaring kesunyian yang terlihat jelas, tetapi memicu kebisingan apa pun di atas ambang batas.

**Tingkat 2: Silero VAD** (2020-2026, MIT). parameter 1 juta. Dilatih dalam 6000+ bahasa. Berjalan dalam ~1 mdtk per potongan 30 mdtk pada satu thread CPU. 87,7% TPR pada 5% FPR. Default sumber terbuka.

**Tingkat 3: detektor belokan semantik.** Model deteksi belokan LiveKit (2024-2026) atau pengklasifikasi kecil kamu sendiri. Membedakan "jeda di tengah kalimat" dari "selesai berbicara". Menggunakan konteks kebahasaan (intonasi + kata terkini), bukan sekedar diam.

### Parameter kunci dan defaultnya

- **Threshold.** Silero mengeluarkan probabilitas; mengklasifikasikan ucapan pada > 0,5 (default) atau > 0,3 (sensitif). Ambang batas bawah = lebih sedikit klip kata pertama, lebih banyak kesalahan positif.
- **Durasi bicara minimum.** Tolak ucapan yang berdurasi kurang dari 250 mdtk — biasanya berupa batuk atau suara kursi.
- **Membungkam mabuk (menunjuk akhir).** Setelah VAD kembali ke 0, tunggu 500-800 ms sebelum menyatakan akhir giliran. Terlalu pendek → mengganggu pengguna. Terlalu lama → terasa lamban.
- **Buffer pra-putar.** Pertahankan audio 300-500 ms sebelum VAD diaktifkan. Mencegah "hei" terpotong.

### Trik siram (Kyutai 2025)

Model streaming STT memiliki penundaan di depan (500 mdtk untuk Kyutai STT-1B, 2,5 dtk untuk STT-2.6B). Biasanya kamu akan menunggu selama itu setelah pidato berakhir untuk mendapatkan transkripnya. Trik flush: ketika VAD mengaktifkan end-of-speech, **kirim sinyal flush ke STT** yang memaksa output langsung. STT memproses pada ~4× waktu nyata, sehingga buffer 500 mdtk selesai dalam ~125 mdtk.

End-to-end: 125 ms VAD + flush STT = latensi percakapan.

### Perbandingan VAD 2026

| VAD | TPR @ 5% FPR | Latensi | Lisensi |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50,0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87,7% | ~1 mdtk | MIT |
| Cobra VAD (Picovoice) | 98,9% | ~1 mdtk | komersial |
| segmentasi catatan pyan | 95% | ~10 mdtk | MIT-ish |

Silero adalah default yang tepat. Cobra adalah peningkatan kepatuhan / akurasi. VAD khusus energi tidak memiliki tempat dalam produksi tahun 2026.

## Build

### Langkah 1: gerbang energi

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Langkah 2: Silero VAD dengan Python

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Langkah 3: mesin keadaan turn-end

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Langkah 4: kerangka trik siram

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```STT (Kyutai, Deepgram, AssemblyAI) harus mendukung flush agar ini berfungsi. Streaming bisikan tidak — ini berbasis blok dan selalu menunggu potongan.

## Pakai

| Situasi | Pilihan VAD |
|-----------|-----------|
| Terbuka, cepat, umum | Silero VAD |
| Pusat panggilan komersial | Kobra VAD |
| Di perangkat (ponsel) | Silero VAD ONNX |
| Penelitian / diarisasi | segmentasi catatan pyan |
| Penggantian tanpa ketergantungan | WebRTC VAD (warisan) |
| Butuh kualitas turn-ending | Silero + Detektor giliran LiveKit berlapis |

Aturan praktisnya: jangan pernah mengirimkan VAD yang hanya menggunakan energi kecuali kamu benar-benar tidak punya pilihan lain.

## Jebakan

- **Ambang batas tetap.** Bekerja dalam keadaan senyap, gagal dalam keadaan bising. Kalibrasi pada perangkat atau beralih ke Silero.
- **Mabuk keheningan yang terlalu singkat.** Agen menyela di tengah kalimat. 500-800 ms adalah titik terbaik untuk percakapan.
- **Mabuk terlalu lama.** Terasa lesu. Tes A/B dengan pengguna target.
- **Tidak ada buffer pra-putar.** Audio pengguna 200-300 ms pertama hilang. Selalu pertahankan pra-putar yang bergulir.
- **Mengabaikan titik akhir semantik.** "Hmm, biarkan aku berpikir..." berisi jeda panjang. Pengguna tidak suka diputus di tengah-tengah pemikiran. Gunakan detektor giliran LiveKit atau sejenisnya.

## Kirim

Simpan sebagai `outputs/skill-vad-tuner.md`. Pilih model VAD, ambang batas, hangover, pre-roll, dan strategi deteksi giliran untuk weight kerja.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Ini mensimulasikan urutan ucapan + keheningan + ucapan + batuk dan menguji tiga tingkatan VAD.
2. **Sedang.** Instal `silero-vad`, proses rekaman 5 menit, setel ambang batas untuk meminimalkan klip kata pertama dan pemicu palsu. Laporkan presisi/recall.
3. **Sulit.** Buat detektor putaran mini: Silero VAD + MLP 3 lapis pada embedding 10 kata terakhir (gunakan pengubah kalimat). Latihlah dataset turn-end yang diberi label tangan. Kalahkan Silero hanya dengan 10% F1.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| VAD | Detektor suara | Biner per frame: apakah ini pidato? |
| Putar deteksi | Menunjuk akhir | VAD + keheningan-mabuk + titik akhir semantik. |
| Diam mabuk | Tunggu setelah pidato | Saatnya menunggu sebelum menyatakan giliran berakhir; 500-800 ms. |
| Pra-putar | Buffer pra-ucapan | Pertahankan audio 300-500 ms sebelum VAD diaktifkan. |
| Trik siram | Retasan Kyutai | VAD → flush-STT → 125 mdtk, bukan penundaan 500 mdtk. |
| Titik akhir semantik | "Apakah mereka bermaksud berhenti?" | Pengklasifikasi ML yang melihat kata-kata, bukan hanya diam. |
| TPR @ FPR 5% | Titik ROC | Tolok ukur VAD standar; 87,7% untuk Silero, 50% WebRTC. |

## Bacaan Lanjutan

- [Silero VAD](https://github.com/snakers4/silero-vad) — referensi VAD terbuka.
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — pemimpin akurasi komersial.
- [Kyutai — Suarakan + trik flush](https://kyutai.org/stt) — trik teknik sub-200 ms.
- [LiveKit — deteksi belokan](https://docs.livekit.io/agents/logic/turns/) — titik akhir semantik dalam produksi.
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — garis dasar lama.
- [segmentasi piannote](https://github.com/pyannote/pyannote-audio) — segmentasi tingkat diarisasi.
