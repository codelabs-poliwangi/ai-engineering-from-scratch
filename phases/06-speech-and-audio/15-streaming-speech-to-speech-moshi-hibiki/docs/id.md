# Streaming Speech-to-Speech — Moshi, Hibiki, dan Dialog Dupleks Penuh

> 2024-2026 mendefinisikan ulang AI suara. Moshi mengirimkan model tunggal yang mendengarkan dan berbicara secara bersamaan dengan latensi 200 ms. Hibiki melakukan terjemahan ucapan-ke-ucapan sedikit demi sedikit. Keduanya meninggalkan pipa ASR → LLM → TTS untuk arsitektur dupleks penuh terpadu melalui token codec Mimi. Ini adalah desain referensi baru.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 6 · 13 (Codec Audio Neural), Fase 6 · 11 (Audio Real-Time), Fase 7 · 05 (Transformer Penuh)
**Waktu:** ~75 menit

## Masalah

Setiap agen suara yang dibuat dari Lesson 11 + 12 memiliki dasar latensi dasar sekitar 300-500 ms: kebakaran VAD, proses STT, alasan LLM, pembangkitan TTS. Setiap phase memiliki latensi minimumnya sendiri. kamu dapat menyetel dan memparalelkan, tetapi bentuk pipa membatasi kamu.

Moshi (Kyutai, 2024-2026) mengajukan pertanyaan berbeda: bagaimana jika tidak ada pipeline pipa? Bagaimana jika satu model menerima audio masuk dan mengeluarkan audio secara langsung, terus-menerus, dengan teks sebagai "monolog batin" perantara dan bukan tahapan yang diperlukan?

Jawabannya adalah **ucapan-ke-ucapan dupleks penuh**. Latensi teoretis 160 ms (bingkai Mimi 80 ms + penundaan akustik 80 ms). Latensi praktis 200 ms pada satu GPU L4. Itu adalah setengah dari apa yang dicapai oleh agen suara pipeline pipa terbaik di kelasnya.

## Konsep

![Arsitektur Moshi: dua aliran Mimi paralel + teks monolog dalam](../assets/moshi-hibiki.svg)

### Arsitektur Moshi

**Input.** Dua aliran codec Mimi, keduanya pada 12,5 Hz × 8 buku code:

- Streaming 1: audio pengguna (dikodekan Mimi, terus berdatangan)
- Streaming 2: Audio milik Moshi sendiri (dihasilkan oleh Moshi)

**Transformer.** Transformer Temporal dengan parameter 7B memproses aliran dan aliran teks "monolog dalam". Pada setiap langkah 80 ms, ini:

1. Menggunakan token Mimi pengguna terbaru (8 buku code).
2. Menggunakan token Moshi Mimi terbaru (8 buku code, sesuai produksi).
3. Menghasilkan token teks Moshi berikutnya (monolog batin).
4. Menghasilkan token Moshi Mimi berikutnya (8 buku code melalui Depth Transformer kecil).

Ketiga aliran — audio pengguna, audio Moshi, teks Moshi — berjalan secara paralel. Moshi dapat mendengar pengguna saat berbicara; dapat menginterupsi dirinya sendiri ketika pengguna menginterupsi; dapat melakukan back-channel ("mhm") tanpa merusak ucapan utamanya.

**Transformer kedalaman.** Dalam sebuah frame, 8 buku code tidak diprediksi secara paralel — mereka memiliki ketergantungan antar-buku code. Sebuah "Transformer kedalaman" 2 layer kecil memprediksinya secara berurutan dalam waktu 80 ms. Ini adalah faktorisasi standar untuk LM codec AR (juga digunakan oleh VALL-E, VibeVoice).

### Mengapa teks monolog batin membantu

Tanpa teks eksplisit, model harus memodelkan bahasa secara implisit dalam aliran akustiknya. Wawasan Moshi: memaksanya untuk mengeluarkan token teks bersama audio. Aliran teks pada dasarnya adalah transkrip dari apa yang dikatakan Moshi. Hal ini meningkatkan koherensi semantik, mempermudah penggantian kepala model bahasa, dan memberi kamu transkrip secara gratis.

### Hibiki: streaming terjemahan ucapan-ke-ucapan

Arsitektur yang sama, dilatih tentang pasangan terjemahan. Sumber audio masuk, audio keluar bahasa target, terus menerus. Hibiki-Zero (Februari 2026) menghilangkan kebutuhan akan training data yang selaras di tingkat kata — menggunakan data tingkat kalimat + pembelajaran penguatan GRPO untuk optimization latensi.

Empat pasangan bahasa didukung pada awalnya; dapat diadaptasi ke bahasa baru dengan ≈1000 jam.

### Tumpukan Kyutai yang lebih luas (2026)- **Moshi** — dialog dupleks penuh (Perancis pertama, bahasa Inggris didukung dengan baik)
- **Hibiki / Hibiki-Zero** — terjemahan ucapan simultan
- **Kyutai STT** — streaming ASR (500 mdtk atau 2,5 dtk)
- **Kyutai Pocket TTS** — TTS 100M-param berjalan pada CPU (Jan 2026)
- **Suarakan** — alur lengkap yang menggabungkan ini di server publik

Throughput pada GPU L40S: 64 sesi bersamaan pada 3× real-time.

### Wijen CSM — sepupu

Sesame CSM (2025) menggunakan ide serupa — tulang punggung Llama-3 dengan kepala codec Mimi. Namun CSM bersifat satu arah (mengambil konteks + teks, menghasilkan ucapan) dan bukan dupleks penuh. Ini adalah TTS "kehadiran suara" terbaik di pasaran; tidak persis sama dengan kemampuan full-duplex Moshi.

### Angka kinerja 2026

| Model | Latensi | Kasus penggunaan | Lisensi |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | Dialog bahasa Inggris / Prancis dupleks penuh | CC-BY 4.0 |
| Hibiki | Kecepatan bingkai 12,5 Hz | Terjemahan streaming bahasa Prancis ↔ bahasa Inggris | CC-BY 4.0 |
| Hibiki-Nol | sama | 5 pasangan bahasa, tidak ada data yang selaras | CC-BY 4.0 |
| Wijen CSM-1B | TTFA 200 mdtk | TTS berkondisi konteks | Apache-2.0 |
| GPT-4o Waktu Nyata | ~300 mdtk | ditutup, OpenAI API | komersial |
| Gemini 2.5 Langsung | ~350 mdtk | ditutup, Google API | komersial |

## Build

### Langkah 1: antarmuka

Moshi mengekspos server WebSocket yang mengambil 80 ms potongan audio yang dikodekan Mimi dan mengembalikan 80 ms potongan audio yang dikodekan Mimi. Kedua cara tersebut. Selalu.

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### Langkah 2: loop dupleks penuh

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

Kedua arah berjalan secara bersamaan. Python asyncio atau Rust futures adalah transportasi standar.

### Langkah 3: tujuan training (konseptual)

Untuk setiap bingkai 80 ms `t`:

- Input: `user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- Prediksi: `moshi_text[t]`, lalu `moshi_mimi[t, codebook_0..7]`

Teks diprediksi sebelum audio (monolog batin); audio diprediksi sekuensial buku code dalam Transformer kedalaman.

### Langkah 4: dimana Moshi menang dan mana yang tidak

Moshi menang:

- Sub-250 ms end-to-end pada perangkat keras murah.
- Pipeline belakang dan interupsi alami.
- Tidak ada code lem pipa.

Moshi tidak menang:

- Panggilan alat (tidak dilatih untuk itu; kamu memerlukan jalur LLM terpisah).
- Alasan panjang (Moshi adalah model dialog 8B, bukan Claude/GPT-4).
- Akurasi faktual pada topik khusus.
- Sebagian besar kasus penggunaan perusahaan produksi (masih menggunakan jaringan pipa pada tahun 2026).

## Pakai

| Situasi | Pilih |
|-----------|------|
| Pendamping suara dengan latensi terendah | Moshi |
| Panggilan terjemahan langsung | Hibiki |
| Demo / penelitian suara | Moshi, CSM |
| Agen perusahaan dengan alat | Pipeline (Lesson 12), bukan Moshi |
| TTS suara khusus dalam konteks | CSM wijen |
| Pidato-ke-ucapan, bahasa apa saja | GPT-4o Realtime atau Gemini 2.5 Live (komersial) |

## Jebakan

- **Pemanggilan alat terbatas.** Moshi adalah model dialog, bukan kerangka agen. Kombinasikan dengan pipa untuk alat.
- **Pengondisian suara khusus.** Moshi menggunakan satu persona terlatih; kloning adalah training yang dijalankan secara terpisah.
- **Cakupan bahasa.** Bahasa Prancis + Inggris sangat bagus; lainnya terbatas. Hibiki-Zero membantu, tetapi kamu masih memerlukan training data.
- **Biaya sumber daya.** Sesi Moshi penuh memiliki slot GPU; bukan pola penerapan penyewa bersama yang murah.

## Kirim

Simpan sebagai `outputs/skill-duplex-pipeline.md`. Pilih arsitektur pipeline vs full-duplex untuk weight kerja agen suara, dengan alasan yang masuk akal.

## Latihan1. **Mudah.** Jalankan `code/main.py`. Ini mensimulasikan arsitektur dua aliran + monolog dalam secara simbolis.
2. **Sedang.** Tarik Moshi dari HuggingFace, jalankan server, uji satu percakapan. Ukur latensi jam dinding dari akhir ucapan pengguna hingga awal respons Moshi.
3. **Sulit.** Ambil agen pipeline Lesson 12 kamu dan bandingkan latensi P50 vs Moshi pada 20 ucapan pengujian yang cocok. Tuliskan ketika pipeline pipa menang secara arsitektural.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Dupleks penuh | Mendengar-dan-berbicara sekaligus | Dua aliran audio aktif secara bersamaan pada model yang sama. |
| Monolog batin | Aliran teks model | Moshi mengeluarkan token teks bersama output audionya. |
| Transformer kedalaman | Prediktor antar-buku code | Transformer kecil yang memprediksi 8 buku code dalam satu frame 80 ms. |
| Mimi | Kodek Kyutai | 12,5 Hz × 8 buku code; semantik+akustik; kekuatan Moshi. |
| Streaming S2S | Audio → audio langsung | Terjemahan/dialog sepotong demi sepotong, tanpa tahapan pipeline. |
| Penyaluran kembali | Reaksi "Mhm" | Moshi dapat mengeluarkan ucapan terima kasih kecil tanpa menghentikan gilirannya. |

## Bacaan Lanjutan

- [Défossez dkk. (2024). Moshi — model dasar teks ucapan](https://arxiv.org/html/2410.00037v2) — makalah.
- [Laboratorium Kyutai (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — terjemahan streaming tanpa data selaras.
- [Wijen (2025). Menyeberangi lembah suara yang luar biasa](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — spesifikasi CSM.
- [Kyutai — repo Moshi](https://github.com/kyutai-labs/moshi) — instal + server.
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — rekan komersial tertutup.
- [Kyutai — Pemodelan Aliran Tertunda](https://github.com/kyutai-labs/delayed-streams-modeling) — framework STT/TTS di balik terpal.
