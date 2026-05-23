# Evaluasi Audio — WER, MOS, UTMOS, MMAU, FAD, dan Papan Peringkat Terbuka

> kamu tidak dapat mengirimkan apa yang tidak dapat kamu ukur. Lesson ini menyebutkan metrik 2026 untuk setiap tugas audio: ASR (WER, CER, RTFx), TTS (MOS, UTMOS, SECS, WER-on-ASR-round-trip), bahasa audio (MMAU, LongAudioBench), musik (FAD, CLAP), dan speaker (EER). Ditambah papan peringkat tempat kamu membandingkan.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Fase 2 · 09 (Evaluasi Model)
**Waktu:** ~60 menit

## Masalah

Setiap tugas audio memiliki beberapa metrik, masing-masing mengukur sumbu berbeda. Menggunakan metrik yang salah adalah cara kamu mengirimkan model yang tampak bagus di dasbor kamu dan sangat buruk dalam produksi. Daftar kanonik tahun 2026:

| Tugas | Utama | Sekunder |
|------|---------|-----------|
| ASR | ADA | CER · RTFx · latensi token pertama |
| TTS | MOS/UTMOS | SECS · WER-on-ASR-pulang pergi · CER · TTFA |
| Kloning suara | SECS (kosinus ECAPA) | MOS · CER |
| Verifikasi pembicara | EER | minDCF · FAR / FRR pada titik operasi |
| Diarisasi | DER | JER · perplexity pembicara |
| Klasifikasi audio | 1 teratas · peta | makro F1 · pemanggilan kembali per kelas |
| Generasi musik | mode | CLAP · panel mendengarkan MOS |
| Model bahasa audio | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Streaming S2S | latensi P50/P95 | ADA · MOS |

## Konsep

![Matrix evaluasi audio — metrik vs tugas vs papan peringkat 2026](../assets/eval-landscape.svg)

### metrik ASR

**WER (Tingkat Kesalahan Kata).** `(S + D + I) / N`. Huruf kecil, hapus tanda baca, normalkan angka sebelum mencetak gol. Gunakan `jiwer` atau `whisper_normalizer` OpenAI. < 5% = ucapan baca dengan paritas manusia.

**CER (Tingkat Kesalahan Karakter).** Rumus yang sama, tingkat karakter. Digunakan untuk bahasa nada (Mandarin, Kanton) yang segmentasi katanya ambigu.

**RTFx (faktor real-time terbalik).** Detik audio diproses per detik jam dinding. Lebih tinggi lebih baik. Parkit-TDT mencapai 3380×. Bisikan-besar-v3 adalah ~30×.

**Latensi token pertama.** Jam dinding dari input audio ke token transkrip pertama. Penting untuk streaming. Deepgram Nova-3: ~150 mdtk.

### Metrik TTS

**MOS (Mean Opinion Score).** 1-5 penilaian manusia. Standar emas tapi lambat. Kumpulkan 20+ pendengar per sample, 100+ sample per model.

**UTMOS (2022-2026).** Mempelajari prediktor MOS. Berkorelasi ~0,9 dengan MOS manusia pada tolok ukur standar. F5-TTS: UTMOS 3.95; kebenaran dasar: 4.08.

**SECS (Speaker Encoder Cosine Kemiripan).** Untuk kloning suara. ECAPA embed kosinus antara referensi dan output kloning. > 0,75 = klon yang dapat dikenali.

**WER-on-ASR-pulang-pergi.** Jalankan Whisper pada output TTS, hitung WER terhadap teks input. Menangkap regresi kejelasan. SOTA 2026: <2% CER.

**TTFA (waktu-ke-audio-pertama).** Latensi jam dinding. Kokoro-82M: ~100 mdtk; F5-TTS: ~1 dtk.

### Khusus kloning suara

**SECS + MOS + CER** sebagai triple. Kloning yang mendapat skor SECS tinggi tetapi MOS rendah berarti timbre-benar-tetapi-tidak wajar; kebalikannya berarti suara alami tetapi pembicara salah.

### Verifikasi pembicara

**EER (Equal Error Rate).** Ambang batas ketika Tingkat Penerimaan Palsu sama dengan Tingkat Penolakan Palsu. ECAPA di VoxCeleb1-O: 0,87%.

**minDCF (Biaya Deteksi min).** Biaya tertimbang pada titik operasi yang dipilih (seringkali FAR=0,01). Lebih relevan dengan produksi dibandingkan EER.

### Diarisasi**DER (Tingkat Kesalahan Diarisasi).** `(FA + Miss + Confusion) / total_speaker_time`. Ucapan tak terjawab + ucapan alarm palsu + perplexity pembicara, masing-masing sebagai pecahan. Pertemuan AMI: DER ~10-20% realistis. pyannote 3.1 + Iklan Precision-2: <10% DER pada audio yang direkam dengan baik.

**JER (Jaccard Error Rate).** Alternatif untuk DER, kuat terhadap bias segmen pendek.

### Klasifikasi audio

Multi-label: **mAP (rata-rata Presisi)** di semua kelas. AudioSet: 0,548 peta untuk BEATs-iter3.

Eksklusif multi-kelas: **akurasi 1 teratas, 5 teratas**. Prompt Ucapan v2: 99,0% teratas-1 (Audio-MAE).

Tidak seimbang: **makro F1** + **penarikan kembali per kelas**. Laporan per kelas — akurasi agregat menyembunyikan kelas mana yang gagal.

### Generasi musik

**FAD (Fréchet Audio Distance).** Distance antara distribusi audio nyata vs yang dihasilkan yang di-embed VGGish. MusicGen-kecil di MusicCaps: 4.5. MusikLM: 4.0. Lebih rendah lebih baik.

**Skor CLAP.** Skor penyelarasan teks-audio menggunakan embedding CLAP. > 0,3 = keselarasan wajar.

**Panel pendengaran MOS.** Masih menjadi kata terakhir untuk musik kelas konsumen. Suno v5 ELO 1293 di TTS Arena (dari preferensi manusia berpasangan).

### Tolok ukur bahasa audio

**MMAU (Pemahaman Multi-Audio Masif).** 10 ribu pasangan audio-QA.

**MMAU-Pro.** 1800 item keras, empat kategori: ucapan / suara / musik / multi-audio. Peluang acak 25% pada 4 arah. Gemini 2.5 Pro secara keseluruhan ~60%; multi-audio ~22% di semua model.

**LongAudioBench.** Klip multi-menit dengan kueri semantik. Audio Flamingo Berikutnya mengalahkan Gemini 2.5 Pro.

**AudioCaps / Clotho.** Tolok ukur teks. Metrik SPICE, CIDEr, FENSE.

### Streaming ucapan-ke-ucapan

**Latensi P50 / P95 / P99.** Jam dinding dari ucapan pengguna akhir hingga respons suara pertama. Moshi: 200 mdtk; GPT-4o Waktu Nyata: 300 mdtk.

**WER / MOS** pada output.

**Responsif yang masuk.** Waktu mulai dari interupsi pengguna hingga senyapnya asisten. Target <150 mdtk.

### Papan peringkat tahun 2026

| Papan Peringkat | Trek | URL |
|------------|--------|-----|
| Buka Papan Peringkat ASR (HF) | Bahasa Inggris + multibahasa + bentuk panjang | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | TTS Bahasa Inggris | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Pidato Analisis Buatan | TTS + STT, ELO dari suara berpasangan | `artificialanalysis.ai/speech` |
| MMAU-Pro | Alasan LALM | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | Pengenalan pembicara | `voxsrc.github.io` |
| Subset musik MMAU | Musik LALM | (dalam MMAU) |
| MENDENGAR patokan | Audio yang diawasi sendiri | `hearbenchmark.com` |

## Build

### Langkah 1: WER dengan normalisasi

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Langkah 2: TTS pulang pergi WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Langkah 3: SECS untuk kloning suara

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Langkah 4: FAD untuk generasi musik

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Langkah 5: EER untuk verifikasi pembicara (code yang sama seperti Lesson 6)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## Pakai

Pasangkan setiap penerapan dengan eval harness tetap yang berjalan pada setiap pembaruan model. Tiga aturan utama:

1. **Normalisasi sebelum memberi skor.** Huruf kecil, strip tanda baca, perluasan angka. Laporkan aturan normalisasi.
2. **Laporkan distribusi, bukan rata-rata.** P50/P95/P99 untuk latensi. Penarikan kembali per kelas untuk klasifikasi. Per kategori untuk MMAU.
3. **Jalankan satu tolok ukur publik kanonik.** Meskipun data produksi kamu berbeda, pelaporan di Open ASR / TTS Arena / MMAU memungkinkan pengulas membandingkan hal yang sama.

## Jebakan- **Ekstrapolasi UTMOS.** Dilatih tentang ucapan bersih ala VCTK; skor audio yang berisik / kloning / emosional buruk.
- **Bias panel MOS.** 20 pekerja Amazon Mechanical Turk ≠ 20 pengguna target. Bayar panel domain jika taruhannya tinggi.
- **FAD bergantung pada kumpulan referensi.** Bandingkan dengan distribusi referensi yang sama di seluruh model.
- **WER Agregat.** Secara keseluruhan 5% WER dapat menyembunyikan 30% WER pada ucapan beraksen. Laporan berdasarkan bagian demografi.
- **Kejenuhan tolok ukur publik.** Sebagian besar model frontier mendekati batas atas tolok ukur standar. Build rangkaian internal yang mencerminkan lalu lintas kamu.

## Kirim

Simpan sebagai `outputs/skill-audio-evaluator.md`. Pilih metrik, tolok ukur, dan format pelaporan untuk rilis model audio apa pun.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Hitung WER / CER / EER / SECS / FAD-ish / MMAU-ish pada input mainan.
2. **Sedang.** Buat harness WER pulang-pergi TTS. Jalankan output Kokoro atau F5-TTS kamu melalui Whisper. Hitung WER lebih dari 50 prompt. Tandai prompt dengan WER > 10%.
3. **Sulit.** Nilai pilihan LALM Lesson 10 kamu pada pidato MMAU-Pro + subset multi-audio (masing-masing 50 item). Laporkan keakuratan per kategori dan bandingkan dengan jumlah yang dipublikasikan.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| ADA | Skor ASR | `(S+D+I)/N` pada tingkat kata setelah normalisasi. |
| CE | Karakter WER | Untuk bahasa nada atau sistem level karakter. |
| MOS | Pendapat manusia | peringkat 1-5; 20+ pendengar × 100 sample. |
| UTMOS | Prediktor ML MOS | Model yang dipelajari; berkorelasi ~0,9 dengan MOS manusia. |
| DETIK | Kesamaan klon suara | ECAPA kosinus antara referensi dan klon. |
| EER | Skor verifikasi pembicara | Ambang batas dimana FAR = FRR. |
| DER | Skor diarisasi | (FA + Nona + Perplexity) / total. |
| mode | Kualitas generasi musik | Distance Fréchet pada embeddings VGGish. |
| RTFx | Output | Detik audio per detik jam dinding. |

## Bacaan Lanjutan

- [jiwer](https://github.com/jitsi/jiwer) — Pustaka WER/CER dengan utilitas normalisasi.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — mempelajari prediktor MOS.
- [Fréchet Audio Distance (Kilgour dkk. 2019)](https://arxiv.org/abs/1812.08466) — standar gen musik.
- [Buka Papan Peringkat ASR](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — peringkat langsung tahun 2026.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — papan peringkat TTS dengan suara manusia.
- [Patokan MMAU-Pro](https://mmaubenchmark.github.io/) — Papan peringkat penalaran LALM.
- [Tolok ukur HEAR](https://hearbenchmark.com/) — tolok ukur SSL audio.
