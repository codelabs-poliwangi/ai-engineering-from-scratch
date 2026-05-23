# Pengenalan & Verifikasi Pembicara

> ASR bertanya "apa yang mereka katakan?" Pengenalan pembicara menanyakan "siapa yang mengatakannya?" Perhitungannya terlihat sama — embedding ditambah kosinus — tetapi setiap keputusan produksi bergantung pada satu nomor EER.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 6 · 02 (Spektogram & Mel), Fase 5 · 22 (Model Embedding)
**Waktu:** ~45 menit

## Masalah

Seorang pengguna mengucapkan frasa sandi. kamu ingin tahu: apakah ini orang yang mereka klaim (*verifikasi*, 1:1), atau orang pertama di bank pendaftaran kamu (*identifikasi*, 1:N)? Atau bukan keduanya — apakah ini speaker yang tidak dikenal (*set terbuka*)?

Pra-2018: GMM-UBM + i-vector. EER yang masuk akal tetapi rapuh terhadap perpindahan pipeline (ponsel vs laptop) dan emosi. 2018–2022: vector-x (tulang punggung TDNN dilatih dengan margin sudut). 2022+: Embedding besar ECAPA-TDNN dan WavLM. Pada tahun 2026, bidang ini didominasi oleh tiga model dan satu metrik.

Metriknya adalah **EER** — Tingkat Kesalahan yang Sama. Tetapkan ambang batas keputusan kamu sehingga Tingkat Penerimaan Palsu = Tingkat Penolakan Palsu. Persilangannya adalah EER. Digunakan di setiap makalah, setiap papan peringkat, setiap panggilan pengadaan.

## Konsep

![Pendaftaran + jalur verifikasi dengan embedding + kosinus + EER](../assets/speaker-verification.svg)

**Pipeline pipa.** Pendaftaran: rekam 5–30 detik pembicara target; menghitung embedding dimension tetap (192-d untuk ECAPA-TDNN, 256-d untuk WavLM-besar). Verifikasi: dapatkan embedding ucapan tes; menghitung kesamaan kosinus; bandingkan dengan ambang batas.

**ECAPA-TDNN (2020, masih dominan 2026).** Menekankan Attention Pipeline, Propagasi, dan Agregasi - Jaringan Syaraf Tunda Waktu. Blok konv 1D dengan eksitasi pemerasan, pengumpulan attention multi-kepala, diikuti oleh layer linier hingga 192-d. Dilatih di VoxCeleb 1+2 (2.700 pembicara, 1,1 juta ucapan) dengan loss Additive Angular Margin (AAM-softmax).

**WavLM-SV (2022+).** Menyempurnakan backbone SSL besar WavLM yang telah dilatih sebelumnya dengan kehilangan AAM. Kualitas lebih tinggi tetapi lebih lambat — 300+ MB vs 15 MB.

**x-vector (dasar).** TDNN + pengumpulan statistik. Klasik; masih berguna di CPU/edge.

**AAM-softmax.** Softmax standar dengan margin tambahan `m` di ruang sudut: `cos(θ + m)` untuk kelas yang benar. Memaksa pemisahan sudut antar kelas. Khas `m=0.2`, skala `s=30`.

### Mencetak gol

- **Kosinus** antara pendaftaran dan embedding pengujian. Keputusan berdasarkan ambang batas.
- **PLDA (LDA Probabilistik).** Embedding proyek ke dalam ruang laten di mana penutur yang sama vs penutur berbeda memiliki rasio kemungkinan bentuk tertutup. Ditambahkan di atas kosinus untuk pengurangan EER +10–20%. Standar sebelum tahun 2020; sekarang hanya digunakan dalam pengaturan set tertutup.
- **Normalisasi skor.** `S-norm` atau `AS-norm`: menormalkan setiap skor terhadap kelompok sarana dan standar palsu. Penting untuk evaluasi lintas domain.

### Angka yang harus kamu ketahui (2026)

| Model | VoxCeleb1-O EER | Param | Hasil (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (klasik) | 3,10% | 5 M | 400× RT |
| ECAPA-TDNN | 0,87% | 15 M | 200× RT |
| WavLM-SV besar | 0,42% | 316 M | 20× RT |
| Segmentasi Pyannote 3.1 + embedding | 0,65% | 6 M | 100× RT |
| ReDimNet (2024) | 0,39% | 24 M | 100× RT |

### Diarisasi"Siapa yang berbicara kapan" dalam klip multi-speaker. Pipeline: VAD → segmen → sematkan setiap segmen → cluster (aglomeratif atau spektral) → batas halus. Tumpukan modern: `pyannote.audio` 3.1, yang menggabungkan segmentasi speaker + embedding + pengelompokan di belakang satu panggilan. SOTA DER pada AMI tahun 2026 adalah ~15% (turun dari 23% pada tahun 2022).

## Build

### Langkah 1: embedding mainan dari statistik MFCC

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

Bukan SOTA sejauh satu mil — hanya untuk mengajar. `code/main.py` menggunakan ini sebagai bukti konsep pada data speaker sintetis.

### Langkah 2: kesamaan kosinus + ambang batas

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Langkah 3: EER dari pasangan kesamaan

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

Pengembalian (eer,threshold_at_eer). Laporkan keduanya.

### Langkah 4: produksi dengan SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### Langkah 5: membuat catatan harian dengan pyannote

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## Pakai

Tumpukan tahun 2026:

| Situasi | Pilih |
|-----------|------|
| Verifikasi 1:1 set tertutup, edge | ECAPA-TDNN + ambang kosinus |
| Verifikasi set terbuka, cloud | WavLM-SV + AS-norm |
| Diarisasi (rapat, podcast) | `pyannote/speaker-diarization-3.1` |
| Anti-spoofing (replay / deteksi deepfake) | AASIST atau RawNet2 |
| Tertanam kecil (KWS + pendaftaran) | Titanet-Kecil (NeMo) |

## Jebakan

- **Ketidakcocokan pipeline.** Model dilatih di VoxCeleb (video web) ≠ audio panggilan telepon. Selalu evaluasi pipeline target.
- **Ucapan singkat.** EER menurun tajam di bawah 3 detik audio pengujian.
- **Pendaftaran dengan kebisingan.** Satu pendaftaran yang berisik akan meracuni pembawa berita. Gunakan ≥3 sample bersih dan rata-rata.
- **Ambang batas tetap di seluruh kondisi.** Selalu sesuaikan ambang batas pada set pengembang yang ditunda dari domain target.
- **Cosinus pada embeddings yang tidak dinormalisasi.** L2-normalkan terlebih dahulu; jika tidak, besarnya akan mendominasi.

## Kirim

Simpan sebagai `outputs/skill-speaker-verifier.md`. Pilih model, protokol pendaftaran, rencana penyesuaian ambang batas, dan perlindungan penipuan.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Membuat "speaker" sintetis (profil nada berbeda), mendaftar, menghitung EER pada daftar uji coba 100 pasang.
2. **Sedang.** Gunakan SpeechBrain ECAPA pada 30 ucapan VoxCeleb1 (masing-masing 5 pembicara × 6). Hitung EER dengan kosinus vs PLCA.
3. **Sulit.** Build pendaftaran lengkap → buat catatan harian → verifikasi pipeline dengan `pyannote.audio`. Evaluasi DER pada set pengembang AMI.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| EER | Metrik judul | Ambang batas di mana Salah Terima = Salah Tolak. |
| Verifikasi | 1:1 | "Apakah ini Alice?" |
| Identifikasi | 1:N | “Siapa yang berbicara?” |
| Set terbuka | Tidak diketahui mungkin | Set pengujian dapat berisi speaker yang belum terdaftar. |
| Pendaftaran | Mendaftar | Menghitung embedding referensi pembicara. |
| AAM-softmax | Loss | Softmax dengan margin sudut tambahan; memaksa pemisahan cluster. |
| PDAM | Penilaian klasik | LDA probabilistik; penilaian rasio kemungkinan di atas embedding. |
| DER | Metrik diarisasi | Tingkat Kesalahan Diarisasi — salah + alarm palsu + perplexity. |

## Bacaan Lanjutan- [Snyder dkk. (2018). X-Vectors: Embedding DNN yang Kuat untuk Pengenalan Pembicara](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — makalah embedding dalam yang klasik.
- [Desplanques dkk. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — arsitektur dominan 2020–2026.
- [Chen dkk. (2022). WavLM: Pra-Training dengan Pengawasan Mandiri Berskala Besar untuk Pemrosesan Pidato Full Stack](https://arxiv.org/abs/2110.13900) — Tulang punggung SSL untuk SV dan diarisasi.
- [Bredin dkk. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — diarisasi produksi + tumpukan embedding.
- [Papan peringkat VoxCeleb (diperbarui 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — kedudukan EER saat ini di seluruh model.
