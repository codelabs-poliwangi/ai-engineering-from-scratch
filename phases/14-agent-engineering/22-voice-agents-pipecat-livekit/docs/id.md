# Agen Suara: Pipecat dan LiveKit

> Agen suara adalah kategori produksi kelas satu pada tahun 2026. Pipecat memberi kamu pipeline berbasis frame Python (VAD → STT → LLM → TTS → transport). Agen LiveKit menjembatani model AI dengan pengguna melalui WebRTC. Latensi produksi menargetkan 450–600 md secara end-to-end untuk tumpukan premium.

**Type:** Learn
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 01 (Agent Loop), Fase 14 · 12 (Pola Alur Kerja)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Jelaskan pipeline berbasis frame Pipecat: DOWNSTREAM (sumber→sink) dan UPSTREAM (kontrol).
- Beri nama tahapan pipeline suara kanonik dan yang mengangkut dukungan Pipecat.
- Jelaskan dua kelas agen suara Agen LiveKit (MultimodalAgent, VoicePipelineAgent) dan kapan masing-masing kelas cocok.
- Meringkas ekspektasi latensi produksi tahun 2026 dan cara ekspektasi tersebut mendorong pilihan arsitektur.

## Masalah

Agen suara bukanlah loop teks dengan TTS yang terpasang. Anggaran latensi sangat besar (~600ms), audio parsial adalah defaultnya, deteksi belokan adalah modelnya, dan jangkauan transportasi dari SIP telepon ke WebRTC. Entah kamu membangun pipeline berbasis bingkai (Pipecat) atau bersandar pada platform (LiveKit).

## Konsep

### Pipecat (pipecat-ai/pipecat)

- Kerangka pipa berbasis bingkai Python.
- `Frame` → `FrameProcessor` rantai.
- Dua arah aliran:
  - **DOWNSTREAM** — sumber → sink (audio masuk, TTS keluar).
  - **UPSTREAM** — umpan balik dan kontrol (pembatalan, metrik, tongkang masuk).
- `PipelineTask` mengelola siklus hidup dengan acara (`on_pipeline_started`, `on_pipeline_finished`, `on_idle_timeout`) dan pengamat untuk metrik/pelacakan/RTVI.

Pipeline pipa yang umum:

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

Transportasi: Harian, LiveKit, SmallWebRTCTransport, FastAPI WebSocket, WhatsApp.

Pipecat Flows menambahkan percakapan terstruktur (mesin negara). Pipecat Cloud adalah runtime terkelola.

### Agen LiveKit (livekit/agen)

- Menjembatani model AI ke pengguna melalui WebRTC.
- Konsep utama: `Agent`, `AgentSession`, `entrypoint`, `AgentServer`.
- Dua kelas agen suara:
  - **MultimodalAgent** — audio langsung melalui OpenAI Realtime atau setara.
  - **VoicePipelineAgent** — STT → LLM → TTS kaskade; memberikan kontrol tingkat teks.
- Deteksi belokan semantik melalui model Transformer.
- Integrasi MCP asli.
- Telepon melalui SIP.
- 50+ model tanpa kunci API melalui LiveKit Inference; 200+ lebih banyak melalui plugin.

### Platform komersial

Vapi (~450–600ms pada tumpukan premium yang dioptimalkan) dan Retell (~600ms end-to-end di 180 panggilan pengujian) dibangun di atasnya. Pilih platform jika kamu menginginkan tumpukan suara terkelola tanpa tim WebRTC.

### Dimana letak kesalahan pola ini

- **Tidak ada penanganan tongkang.** Interupsi pengguna; agen terus berbicara. Memerlukan pembatalan frame UPSTREAM di Pipecat, setara dengan LiveKit.
- **Kepercayaan STT diabaikan.** Transkrip berkeyakinan rendah dimasukkan ke LLM seolah-olah Injil. Gerbang kepercayaan atau minta konfirmasi.
- **Pemotongan tengah kalimat TTS.** Saat pipeline membatalkan ucapan di tengah, TTS perlu mengetahui atau memotong audio.
- **Anggaran latensi diabaikan.** Setiap komponen menambahkan 50–200 md. Jumlahkan rantai kamu sebelum pengiriman.

### Latensi umum tahun 2026

- VAD: 20–60 md
- STT parsial: 100–250 ms
- Token pertama LLM: 150–400 md
- TTS audio pertama: 100–200 md
- RTT Transportasi: 30–80 md

450–600 md ujung ke ujung adalah premium. 800–1200ms adalah hal biasa. Apa pun > 1500ms terasa rusak.

## Build

`code/main.py` adalah rangkaian mainan berbasis bingkai dengan:- Jenis `Frame` (audio, transkrip, teks, tts_audio, kontrol).
- Antarmuka `Processor` dengan `process(frame)`.
- Pipa lima phase (VAD → STT → LLM → TTS → transport) sebagai prosesor bernaskah.
- Bingkai pembatalan UPSTREAM untuk menunjukkan tongkang masuk.

Jalankan:

```
python3 code/main.py
```

Jejak menunjukkan aliran normal dan pembatalan tongkang yang menghentikan ucapan TTS.

## Pakai

- **Pipecat** untuk kontrol penuh — prosesor khusus, penyedia pluggable yang mengutamakan Python.
- **Agen LiveKit** untuk penerapan dan telepon yang mengutamakan WebRTC.
- **Vapi / Retell** untuk agen suara yang dihosting tanpa tim WebRTC.
- **OpenAI Realtime / Gemini Live** untuk audio-in/audio-out langsung (MultimodalAgent).

## Kirim

`outputs/skill-voice-pipeline.md` merancah pipa suara berbentuk Pipecat dengan VAD + STT + LLM + TTS + transportasi ditambah penanganan tongkang.

## Latihan

1. Tambahkan pengamat metrik ke pipeline mainan kamu: hitung frame per phase per detik. Di mana latensi terakumulasi?
2. Terapkan STT dengan tingkat kepercayaan: di bawah ambang batas, mintalah "bisakah kamu mengulanginya?"
3. Tambahkan deteksi giliran semantik: aturan sederhana — jika transkrip diakhiri dengan "?", akhir giliran.
4. Baca dokumen transportasi Pipecat. Tukar transportasi stdlib dengan konfigurasi SmallWebRTCTransport (rintisan).
5. Ukur kaskade OpenAI Realtime vs STT+LLM+TTS pada kueri yang sama. Berapa biaya latensi yang ditanggung oleh kontrol tingkat teks?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Bingkai | "Acara" | Unit data yang diketik dalam pipeline (audio, transkrip, teks, kontrol) |
| Prosesor | "Phase pipeline pipa" | Penangan dengan proses(bingkai) |
| HILIR | "Aliran maju" | Sumber untuk tenggelam: audio masuk, ucapan keluar |
| HULU | "Aliran umpan balik" | Kontrol: pembatalan, metrik, tongkang masuk |
| VAD | "Deteksi aktivitas suara" | Mendeteksi saat pengguna berbicara |
| Deteksi giliran semantik | "Akhir belokan yang cerdas" | Keputusan berdasarkan model yang dilakukan pengguna |
| Agen Multimodal | "Agen audio langsung" | Audio masuk, audio keluar; tidak ada teks di tengah |
| Agen Pipeline Suara | "Agen kaskade" | STT+LLM+TTS; kontrol tingkat teks |

## Bacaan Lanjutan

- [Dokumen Pipecat](https://docs.pipecat.ai/getting-started/introduction) — pipeline, prosesor, transportasi berbasis frame
- [Dokumen Agen LiveKit](https://docs.livekit.io/agents/) — WebRTC + suara primitif
- [Vapi](https://vapi.ai/) — platform suara terkelola
- [Retell AI](https://www.retellai.com/) — suara terkelola, tolok ukur latensi
