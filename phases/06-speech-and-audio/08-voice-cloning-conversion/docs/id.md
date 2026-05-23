# Kloning Suara & Konversi Suara

> Kloning suara membaca teks kamu dengan suara orang lain. Konversi suara menulis ulang suara kamu menjadi suara orang lain sambil mempertahankan apa yang kamu katakan. Keduanya berpegang pada prinsip primitif yang sama: memisahkan identitas pembicara dari konten.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 6 · 06 (Pengenalan Pembicara), Phase 6 · 07 (TTS)
**Waktu:** ~75 menit

## Masalah

Pada tahun 2026, klip audio berdurasi 5 detik sudah cukup untuk menghasilkan tiruan suara siapa pun yang berkualitas tinggi dengan GPU konsumen. ElevenLabs, F5-TTS, OpenVoice v2, VoiceBox semuanya mengirimkan kloning zero-shot atau beberapa-shot. Teknologi adalah berkah (aksesibilitas TTS, sulih suara, suara bantu) dan senjata (panggilan penipuan, deepfake politik, pencurian IP).

Dua tugas yang terkait erat:

- **Kloning suara (sisi TTS):** teks + suara referensi 5 detik → audio dalam suara itu.
- **Konversi suara (sisi ucapan):** sumber audio (orang A mengucapkan X) + suara referensi orang B → audio B mengucapkan X.

Keduanya memasukkan bentuk gelombang (konten, pembicara, prosodi) dan menggabungkan kembali konten dari satu sumber dengan pembicara dari sumber lain.

Kendala utama yang kini kamu hadapi pada tahun 2026: **watermarking dan gerbang persetujuan diwajibkan secara hukum di UE (AI Act, berlaku Agustus 2026) dan di California (AB 2905, efektif 2025)**. Pipeline pipa kamu harus mengeluarkan tanda air yang tidak terdengar dan menolak kloning non-konsensual.

## Konsep

![Kloning suara vs konversi: faktorisasi, tukar speaker, gabungkan kembali](../assets/voice-cloning.svg)

**Kloning zero-shot.** Berikan klip berdurasi 5 detik ke model yang telah dilatih pada ribuan speaker. Encoder speaker memetakan klip ke speaker yang terpasang; kondisi decoder TTS pada embedding itu ditambah teks.

Digunakan oleh: F5-TTS (2024), YourTTS (2022), XTTS v2 (2024), OpenVoice v2 (2024).

**Penyempurnaan beberapa jepretan.** Rekam suara target selama 5-30 menit. LoRA menyempurnakan model dasar selama satu jam. Kualitas melonjak dari “oke” menjadi “tidak dapat dibedakan”. Coqui dan ElevenLabs mendukung pola ini; komunitas menggunakannya dengan F5-TTS.

**Konversi suara (VC).** Dua kelompok:

- **Sintesis pengenalan.** Jalankan model mirip ASR untuk mengekstrak representasi konten (misalnya, posterior fonem lunak, PPG), lalu sintesis ulang dengan embedding speaker target. Kuat dalam bahasa dan aksen. Digunakan oleh KNN-VC (2023), Diff-HierVC (2023).
- **Disentanglement.** Latih autoencoder yang memisahkan konten, speaker, dan prosodi dalam ruang laten di titik kemacetan. Tukar embedding speaker pada inference. Kualitas lebih rendah tetapi lebih cepat. Digunakan oleh AutoVC (2019), varian VITS-VC.

**Kloning berbasis codec neural (2024+).** VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox — memperlakukan audio sebagai token terpisah dari SoundStream / EnCodec, melatih model autoregresif besar atau pencocokan aliran melalui token codec. Kualitas sebanding dengan ElevenLabs dalam waktu singkat.

### Etikanya sedikit, bukan langsung

**Watermarking.** PerTh (Perth) dan SilentCipher (2024) embed ID ~16-32 bit tanpa terlihat di audio. Bertahan dari pengkodean ulang, streaming, dan pengeditan umum. Sumber terbuka siap produksi.

**Gerbang izin.** Harus memasangkan setiap output yang dikloning dengan data izin yang dapat diverifikasi. "Saya, Rohit, pada 22-04-2026, mengizinkan suara ini untuk tujuan X." Simpan dalam log anti rusak.

**Deteksi.** AASIST, RawNet2, dan Wav2Vec2-AASIST dikirimkan sebagai detektor. Tantangan ASVspoof 2025 menerbitkan EER sebesar 0,8–2,3% untuk detektor canggih terhadap output ElevenLabs, VALL-E 2, dan Bark.

### Angka (2026)| Model | Pukulan nol? | SECS (sim target) | WER (intel.) | Param |
|-------|-----------|-----|--------------|--------|
| F5-TTS | Ya | 0,72 | 2,1% | 335M |
| XTTS v2 | Ya | 0,65 | 3,5% | 470M |
| OpenVoice v2 | Ya | 0,70 | 2,8% | 220M |
| VALL-E 2 | Ya | 0,77 | 2,4% | 370M |
| Kotak Suara | Ya | 0,78 | 2,1% | 330M |

SECS > 0,70 umumnya tidak dapat dibedakan dari target sebagian besar pendengar.

## Build

### Langkah 1: decomposition dengan sintesis pengenalan (demo khusus code di main.py)

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

Secara konseptual sederhana; massa implementasi ada di `tts_model` dan speaker encoder.

### Langkah 2: klon zero-shot dengan F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

Transkrip referensi harus sama persis dengan audionya; ketidakcocokan merusak keselarasan.

### Langkah 3: konversi suara dengan KNN-VC

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC menjalankan WavLM untuk mengekstrak embeddings per frame untuk kumpulan sumber dan target, lalu mengganti setiap frame sumber dengan tetangga terdekatnya di kumpulan tersebut. Non-parametrik, bekerja dengan satu menit pidato yang ditargetkan.

### Langkah 4: sematkan tanda air

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

~32 bit payload, dapat dideteksi setelah pengkodean ulang MP3 dan noise ringan.

### Langkah 5: gerbang persetujuan

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## Pakai

Tumpukan tahun 2026:

| Situasi | Pilih |
|-----------|------|
| Klon zero-shot 5 detik, sumber terbuka | F5-TTS atau OpenVoice v2 |
| Kloning produksi komersial | Klon Suara Instan ElevenLabs v2.5 |
| Konversi suara (menulis ulang) | KNN-VC atau Diff-HierVC |
| Penyempurnaan banyak speaker | StyleTTS 2 + adaptor speaker |
| Kloning lintas bahasa | XTTS v2 atau VALL-E X |
| Deteksi deepfake | Wav2Vec2-AASIST |

## Jebakan

- **Transkrip referensi tidak selaras.** F5-TTS dan sejenisnya memerlukan teks referensi untuk sama persis dengan audio referensi, termasuk tanda baca.
- **Referensi bergema.** Echo membunuh klon tersebut. Rekam kering, tutup mikrofon.
- **Ketidakcocokan emosional.** Referensi training "ceria" menghasilkan klon ceria dari segalanya. Cocokkan emosi referensi dengan penggunaan target.
- **Kebocoran bahasa.** Mengkloning penutur bahasa Inggris lalu meminta modelnya berbicara bahasa Prancis sering kali mengandung aksen; menggunakan model lintas bahasa (XTTS, VALL-E X).
- **Tanpa tanda air.** Secara hukum tidak dapat dikirimkan di UE mulai Agustus 2026.

## Kirim

Simpan sebagai `outputs/skill-voice-cloner.md`. Rancang pipeline kloning atau konversi dengan gerbang persetujuan + tanda air + target kualitas.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Mendemonstrasikan pertukaran embedding speaker dengan menghitung kosinus antara dua "speaker" sebelum dan sesudah pertukaran.
2. **Medium.** Gunakan OpenVoice v2 untuk mengkloning suara kamu sendiri. Ukur SECS antara referensi dan klon. Ukur CER melalui Whisper.
3. **Hard.** Terapkan tanda air SilentCipher ke 20 klon, jalankan melalui enkode+dekode MP3 128 kbps, deteksi muatannya. Laporkan akurasi bit.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Klon tembakan nol | 5 detik sudah cukup | Model terlatih + embedding speaker; tidak ada training. |
| PPG | Posteriorgram fonetik | Posterior ASR per bingkai digunakan sebagai representasi konten tanpa bahasa. |
| KNN-VC | Konversi nearest neighbor | Ganti setiap frame sumber dengan frame kumpulan target terdekat. |
| Kodek saraf TTS | Gaya VALL-E | Model AR melalui token EnCodec/SoundStream. |
| Tanda Air | Tanda tangan tidak terdengar | Bit yang tertanam dalam audio, tetap dikodekan ulang. |
| DETIK | Kesetiaan kloning | Kosinus antara embedding speaker target dan kloning. |
| AASIST | Detektor deepfake | Model anti-spoof; mendeteksi ucapan yang disintesis. |

## Bacaan Lanjutan

- [Chen dkk. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — kloning zero-shot SOTA sumber terbuka.
- [Baevski dkk. /Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) dan [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — TTS kodek saraf.
- [Qian dkk. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — konversi suara berbasis penguraian.
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — VC berbasis pengambilan.
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — watermark audio 32-bit siap produksi.
- [Hasil ASVspoof 2025](https://www.asvspoof.org/) — perlombaan senjata detektor vs synthesizer, diperbarui pada tahun 2026.
