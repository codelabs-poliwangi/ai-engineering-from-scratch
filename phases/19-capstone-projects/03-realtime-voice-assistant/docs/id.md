# Capstone 03 — Asisten Suara Real-Time (ASR ke LLM ke TTS)

> Agen suara yang dirasa tepat memiliki latensi end-to-end di bawah 800 ms, mengetahui kapan kamu berhenti berbicara, menangani menerobos masuk, dan dapat memanggil alat tanpa terhenti. Retell, Vapi, LiveKit Agents, dan Pipecat semuanya mencapai standar ini pada tahun 2026. Mereka melakukannya dengan bentuk yang sama: streaming ASR, turn-detector, streaming LLM, dan streaming TTS, semuanya dihubungkan melalui WebRTC dengan anggaran latensi yang agresif di setiap hop. Build satu, ukur WER dan MOS serta tingkat batas palsu, dan jalankan dalam kondisi kehilangan paket.

**Type:** Batu penjuru
**Language:** Python (agen + pipeline), TypeScript (klien web)
**Prerequisites:** Fase 6 (ucapan dan audio), Fase 7 (Transformer), Fase 11 (rekayasa LLM), Fase 13 (peralatan), Fase 14 (agen), Fase 17 (infrastruktur)
**Fase yang dilakukan:** P6 · P7 · P11 · P13 · P14 · P17
**Waktu:** 30 jam

## Masalah

Voice telah menjadi kategori AI UX dengan pergerakan tercepat pada tahun 2025-2026. Batasan teknis turun setiap kuartal. OpenAI Realtime API, Gemini 2.5 Live, Cartesia Sonic-2, ElevenLabs Flash v3, LiveKit Agents 1.0, dan Pipecat 0.0.70 semuanya menempatkan audio-out pertama di bawah 800ms dalam jangkauan. Bilahnya bukan hanya latensi saja. Ini adalah nuansa interaksinya: tidak memotong pengguna, tidak terputus, memulihkan diri dari interupsi di tengah kalimat, memanggil alat di tengah percakapan tanpa menghentikan audio, bertahan dalam jaringan seluler yang gelisah.

kamu tidak dapat mencapainya dengan menggabungkan tiga panggilan REST. Arsitekturnya disalurkan secara streaming dari ujung ke ujung. Bangunlah dan mode kegagalan akan terlihat: VAD yang disetel untuk audio ponsel yang menyala di TV latar belakang, detektor putaran yang menunggu tanda baca yang tidak pernah muncul, TTS yang melakukan buffering 400 ms sebelum dipancarkan. Puncaknya adalah memperbaikinya satu per satu saat sedang dimuat dan menerbitkan laporan latensi dan kualitas.

## Konsep

Pipeline ini memiliki lima phase streaming: **audio masuk** (WebRTC dari browser atau PSTN), **ASR** (streaming transkrip sebagian dari Deepgram Nova-3 atau bisikan yang lebih cepat), **deteksi belokan** (VAD plus model pendeteksi belokan kecil yang membaca sebagian transkrip untuk isyarat penyelesaian), **LLM** (streaming token segera setelah giliran dinilai selesai), **TTS** (streaming audio keluar dalam waktu ~200 md dari token LLM pertama).

Tiga permasalahan lintas sektoral. **Barge-in**: ketika pengguna mulai berbicara saat agen berbicara, TTS dibatalkan dan ASR segera menjawabnya. **Penggunaan alat**: panggilan fungsi di tengah percakapan (cuaca, kalender) harus dijalankan di pipeline samping tanpa menghentikan audio; agen mengisi token pengakuan ("satu detik...") terlebih dahulu jika latensi melebihi 300 md. **Tekanan balik**: saat paket hilang, sebagian transkrip ditahan, VAD menaikkan ambang batas gerbang bicara, dan agen menghindari pembicaraan melalui pesan yang tidak diakui.

Bilah pengukuran bersifat kuantitatif. WER di bawah 8% pada benchmark Hamming VAD pada SNR 15 dB. P50 output audio pertama di bawah 800 ms pada 100 panggilan terukur. Tingkat false-cutoff di bawah 3%. MOS diatas 4.2 pada TTS. 50 panggilan bersamaan pada satu g5.xlarge. Angka-angka ini adalah hasil yang bisa dicapai.

## Arsitektur

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## Tumpukan- Transportasi: Agen LiveKit 1.0 (WebRTC) ditambah gateway Twilio PSTN; Pipecat 0.0.70 sebagai kerangka alternatif
- ASR: Deepgram Nova-3 (streaming, sub-300ms parsial pertama) atau lebih cepat-bisikan Whisper-v3-turbo yang dihosting sendiri
- VAD: Silero VAD v5 plus turn-detector LiveKit (Transformer kecil yang membaca sebagian transkrip)
- LLM: OpenAI GPT-4o-realtime untuk integrasi yang erat, Gemini 2.5 Flash Live, atau cascaded Claude Haiku 4.5 (penyelesaian streaming, jalur audio terpisah)
- TTS: Cartesia Sonic-2 (byte pertama terendah), ElevenLabs Flash v3, atau Orpheus sumber terbuka untuk host mandiri
- Alat: Pipeline samping FastMCP untuk cuaca/kalender/pemesanan; agen memancarkan pengisi terlebih dahulu jika alat memerlukan waktu >300ms
- Observabilitas: Rentang suara OpenTelemetry, jejak suara Langfuse dengan pemutaran ulang audio
- Penerapan: g5.xlarge tunggal (VRAM 24GB) untuk Whisper + Orpheus yang dihosting sendiri; API yang dihosting untuk latensi terendah

## Build

1. **Sesi WebRTC.** Siapkan ruang LiveKit dan klien web yang mengalirkan audio mikrofon. Di server, lampirkan agen pekerja yang bergabung dalam ruangan.

2. **ASR streaming.** Mengumpankan frame PCM 20 md ke Deepgram Nova-3 (atau lebih cepat di GPU). Berlangganan transkrip sebagian dan akhir. Catat latensi per parsial.

3. **VAD dan putar detektor.** Jalankan Silero VAD v5 pada aliran bingkai. Pada acara akhir ucapan, aktifkan detektor giliran LiveKit terhadap transkrip parsial terbaru. Hanya berkomitmen untuk "menyelesaikan" ketika VAD mengatakan diam selama 500 md dan detektor giliran mendapat skor penyelesaian > 0,6.

4. **Aliran LLM.** Saat giliran selesai, mulai panggilan LLM dengan percakapan yang sedang berjalan ditambah transkrip akhir. Streaming token keluar. Pada token pertama serahkan ke TTS.

5. **Aliran TTS.** Cartesia Sonic-2 mengalirkan kembali potongan audio. Potongan pertama harus meninggalkan server dalam waktu 200 md dari token LLM pertama. Keluarkan bongkahan ke ruang LiveKit; klien memutar melalui buffer jitter WebRTC.

6. **Barge-in.** Saat VAD mendeteksi ucapan pengguna baru saat TTS diputar, segera batalkan streaming TTS, hapus sisa output LLM, dan persenjatai kembali ASR. Publikasikan rentang `tts_canceled`.

7. **Pipeline samping alat.** Daftarkan cuaca dan kalender sebagai alat pemanggil fungsi. Saat dipanggil, aktifkan panggilan secara bersamaan; jika tidak terselesaikan dalam waktu 300 ms, minta LLM mengeluarkan "satu detik, izinkan saya memeriksa" sebagai pengisi; lanjutkan setelah alat kembali.

8. **Eval harness.** Rekam 100 panggilan. Hitung WER (terhadap transkrip yang ditahan), tingkat batas palsu (TTS dibatalkan saat pengguna berada di tengah kalimat), p50 output audio pertama, TTS MOS (manusia atau NISQA), dan uji jitter-loss (jatuhkan 3% paket).

9. **Uji weight.** Dorong 50 panggilan serentak pada satu g5.xlarge dengan pemanggil sintetis. Ukur output audio pertama yang berkelanjutan p95.

## Pakai

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## Kirim

`outputs/skill-voice-agent.md` adalah hasil yang dapat dicapai. Dengan adanya domain (dukungan pelanggan, penjadwalan, atau kios), agen LiveKit akan berdiri dengan alur ASR/VAD/LLM/TTS yang disetel ke bilah pengukuran. Rubrik:

| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | Latensi ujung ke ujung | p50 audio-out pertama di bawah 800ms di 100 panggilan yang direkam |
| 20 | Kualitas pengambilan giliran | Tingkat false-cutoff di bawah 3% pada benchmark Hamming VAD |
| 20 | Kebenaran penggunaan alat | Panggilan alat di tengah percakapan yang mengembalikan data yang benar tanpa menghentikan audio |
| 20 | Keandalan dalam kehilangan paket | WER dan stabilitas pengambilan giliran dengan 3% packet drop yang disuntikkan |
| 15 | Evaluasi kelengkapan harness | Pengukuran yang dapat direproduksi dengan konfigurasi publik |
| **100** | | |

## Latihan1. Tukar Deepgram Nova-3 dengan turbo v3 yang lebih cepat di g5.xlarge. Ukur latensi dan kesenjangan WER. Identifikasi pentingnya keputusan CPU-vs-GPU.

2. Tambahkan kebijakan arbitrase interupsi: apa yang dilakukan agen ketika pengguna menerobos masuk selama panggilan alat? Bandingkan tiga kebijakan (pembatalan keras, selesaikan alat lalu berhenti, antrian giliran berikutnya).

3. Jalankan tes pendeteksi giliran permusuhan: berikan jeda panjang di tengah kalimat kepada pengguna. Sesuaikan ambang batas keheningan VAD dan ambang batas skor detektor belokan untuk batas palsu terendah tanpa melampaui 900 mdtk.

4. Sebarkan agen yang sama di PSTN melalui Twilio. Bandingkan output audio pertama PSTN dengan WebRTC. Jelaskan perbedaan jitter-buffer dan codec.

5. Tambahkan deteksi aktivitas suara untuk bahasa non-Inggris (Jepang, Spanyol). Ukur tingkat pemicu palsu Silero VAD v5 versus penyesuaian khusus bahasa.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Putar deteksi | "Akhir ucapan" | Pengklasifikasi yang, dengan keheningan VAD dan sebagian transkrip, memutuskan bahwa pengguna sudah selesai berbicara |
| Tongkang masuk | "Penanganan interupsi" | Membatalkan pemutaran TTS di tengah-tengah ketika VAD mendeteksi ucapan pengguna baru |
| Output audio pertama | "Latensi" | Waktu sejak pengguna berhenti berbicara hingga paket audio pertama meninggalkan server |
| VAD | "Gerbang Pidato" | Model yang mengklasifikasikan bingkai audio sebagai ucapan vs keheningan; Silero VAD v5 adalah default tahun 2026 |
| Penyangga jitter | "Perataan audio" | Buffer sisi klien yang menampung paket sebentar untuk menyerap varian jaringan |
| Pengisi | "Token pengakuan" | Frasa pendek yang dikeluarkan agen untuk menghindari keheningan saat alat lambat |
| MOS | "Skor opini rata-rata" | Peringkat kualitas ucapan persepsi; NISQA adalah proxy otomatis |

## Bacaan Lanjutan

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — referensi framework agen WebRTC
- [Pipecat](https://github.com/pipecat-ai/pipecat) — framework agen streaming alternatif Python-first
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — referensi untuk model ucapan terintegrasi
- [Dokumentasi Deepgram Nova-3](https://developers.deepgram.com/docs) — referensi streaming ASR
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — model referensi VAD
- [Cartesia Sonic-2](https://docs.cartesia.ai) — referensi TTS latensi rendah
- [Menceritakan kembali arsitektur AI](https://docs.retellai.com) — arsitektur agen suara produksi
- [tumpukan produksi Vapi.ai](https://docs.vapi.ai) — referensi produksi alternatif
