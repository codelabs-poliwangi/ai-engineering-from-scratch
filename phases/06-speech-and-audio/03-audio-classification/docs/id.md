# Klasifikasi Audio — Dari k-NN di MFCC hingga AST dan BEAT

> Segala sesuatu mulai dari "gonggongan anjing vs sirene" hingga "bahasa apa ini" adalah klasifikasi audio. Feature-fiturnya mels. Arsitekturnya bergerak setiap dekade. Evaluasinya tetap AUC, F1, dan recall per kelas.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 6 · 02 (Spektogram & Mel), Fase 3 · 06 (CNN), Fase 5 · 08 (CNN & RNN untuk Teks)
**Waktu:** ~75 menit

## Masalah

kamu mendapatkan klip 10 detik. kamu ingin tahu: "apa itu?" Suara perkotaan (sirene, bor, anjing), prompt ucapan (ya/tidak/berhenti), ID bahasa (en/es/ar), emosi pembicara (marah/netral), atau suara lingkungan (dalam/luar ruangan, celoteh). Semua ini adalah *klasifikasi audio*, dan pada tahun 2026 arsitektur dasarnya sudah matang: log-mel → CNN atau Transformer → softmax.

Kesulitan intinya bukanlah pada jaringan. Itu adalah data. Dataset audio memiliki ketidakseimbangan kelas yang parah, pergeseran domain yang kuat (bersih vs berisik), dan kebisingan label (siapa yang memutuskan "celoteh perkotaan" vs "kebisingan restoran"?). 80% masalahnya adalah kurasi, augmentasi, dan evaluasi, bukan menukar CNN dengan Transformer.

## Konsep

![Tangga klasifikasi audio: k-NN pada MFCC ke AST ke BEAT](../assets/audio-classification.svg)

**k-NN pada MFCC (dasar tahun 1990-an).** Ratakan MFCC per klip, hitung kemiripan kosinus dengan bank berlabel, kembalikan suara mayoritas pada K teratas. Sangat kuat pada dataset yang bersih dan kecil (Speech Commands, ESC-50). Berjalan tanpa GPU.

**CNN 2D di log-mel (2015-2019).** Perlakukan `(T, n_mels)` log-mel sebagai gambar. Terapkan gaya ResNet-18 atau VGG. Rata-rata global mengumpulkan sumbu waktu. Softmax atas kelas. Masih menjadi baseline di sebagian besar kompetisi kaggle tahun 2026.

**Audio Spectrogram Transformer, AST (2021-2024).** Tambal log-mel (misalnya tambalan 16×16), tambahkan embedding posisi, masukkan ke ViT. Canggih di AudioSet (mAP 0,485) untuk pembelajaran yang diawasi.

**BEATs dan basis WavLM (2024-2026).** Pra-training yang diawasi sendiri selama jutaan jam. Sempurnakan tugas kamu dengan 1-10% data yang diawasi yang kamu perlukan. Pada tahun 2026, ini adalah titik awal default untuk audio non-ucapan. BEATs-iter3 mengalahkan AST dengan 1-2 mAP di AudioSet saat menggunakan 1/4 komputasi.

**Whisper-encoder sebagai tulang punggung yang dibekukan (2024).** Ambil encoder Whisper, lepaskan decoder, lampirkan pengklasifikasi linier. Hampir SOTA pada ID bahasa dan klasifikasi peristiwa sederhana tanpa augmentasi audio. Garis dasar "makan siang gratis".

### Ketidakseimbangan kelas adalah tantangan sebenarnya

ESC-50: 50 kelas, masing-masing 40 klip — seimbang, mudah. UrbanSound8K: 10 kelas, tidak seimbang 10:1. AudioSet: 632 kelas dengan panjang ekor 100.000:1. Teknik yang berhasil:

- Pengambilan sample yang seimbang selama training (bukan dalam evaluasi).
- Mixup: interpolasi linear dua klip (dan labelnya) sebagai augmentasi.
- SpecAugment: menutupi pita waktu dan frekuensi acak. Sederhana; kritis.

### Evaluasi

- Eksklusif multikelas (Prompt Ucapan): akurasi 1 teratas, akurasi 5 teratas.
- Multi-label multikelas (AudioSet, gaya UrbanSound): presisi rata-rata rata-rata (mAP).
- Sangat tidak seimbang: penarikan kembali per kelas + makro F1.

Nomor 2026 yang perlu kamu ketahui:

| Tolok Ukur | Dasar | SOTA 2026 | Sumber |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97,0% (BEATs-iter3) | Makalah BEAT (2024) |
| Peta Set Audio | 0,485 (AST) | 0,548 (BEAT-iter3) | MENDENGAR papan peringkat 2026 |
| Prompt Ucapan v2 | 98% (CNN) | 99,0% (Audio-MAE) | MENDENGAR hasil v2 |

## Build

### Langkah 1: tampilkan

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```### Langkah 2: ringkasan dengan panjang tetap

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

Sederhana namun kuat: rata-rata + varians sepanjang waktu menghasilkan embedding tetap 26-dim untuk MFCC 13-koef. Berjalan secara instan. Kalahkan baseline NN yang canggih pada ESC-50 pada tahun 2017.

### Langkah 3: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Langkah 4: tingkatkan ke CNN di log-mels

Di PyTorch:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

parameter 3M. Berlatih dalam ~10 menit di ESC-50 dengan satu RTX 4090. Akurasi 80%+.

### Langkah 5: default tahun 2026 — menyempurnakan BEAT

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

Untuk BEAT, gunakan `microsoft/BEATs-base` melalui perpustakaan `beats`; API Transformer memiliki bentuk yang sama.

## Pakai

Tumpukan tahun 2026:

| Situasi | Mulai dengan |
|-----------|-----------|
| Dataset kecil (<1000 klip) | k-NN di MFCC berarti (dasar kamu) + augmentasi audio |
| Dataset sedang (1K–100K) | Penyempurnaan BEAT atau AST |
| Dataset besar (>100K) | Latih dari awal atau sempurnakan pembuat enkode Bisikan |
| Waktu nyata, tepi | CNN 40-MFCC, dikuantisasi menjadi int8 (gaya KWS) |
| Multi-label (AudioSet) | BEATs-iter3 dengan loss BCE + campur aduk + SpecAugment |
| ID Bahasa | MMS-LID, garis dasar SpeechBrain VoxLingua107 |

Aturan pengambilan keputusan: **mulai dengan tulang punggung yang beku, bukan model baru**. Menyempurnakan kepala BEAT memberi kamu 95% SOTA dalam hitungan jam, bukan minggu.

## Kirim

Simpan sebagai `outputs/skill-classifier-designer.md`. Pilih arsitektur, augmentasi, strategi keseimbangan kelas, dan metrik evaluasi untuk tugas klasifikasi audio tertentu.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Ini melatih garis dasar k-NN MFCC pada dataset sintetis 4 kelas (nada murni pada nada berbeda). Laporkan matrix perplexity.
2. **Sedang.** Ganti `summarize` dengan [mean, var, skew, kurtosis]. Apakah pengumpulan 4 momen mengalahkan mean+var pada dataset sintetis yang sama?
3. **Sulit.** Menggunakan `torchaudio`, latih CNN 2D pada lipatan ESC-50 1. Laporkan akurasi validasi silang 5 kali lipat. Tambahkan SpecAugment (time mask = 20, freq mask = 10) dan laporkan delta.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Kumpulan Audio | ImageNet audio | Dataset YouTube berlabel lemah kelas 632 milik Google klip 2M. |
| ESC-50 | Tolok ukur klasifikasi kecil | 50 kelas × 40 klip suara lingkungan. |
| AST | Transformer Spektogram Audio | ViT pada patch log-mel; SOTA 2021. |
| MENGALAHKAN | Audio yang diawasi sendiri | Model Microsoft, iter3 memimpin AudioSet pada tahun 2026. |
| Campuran | Augmentasi pasangan | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| Penambahan Spesifikasi | Augmentasi berbasis masker | Menghilangkan pita waktu dan frekuensi acak dari spektogram. |
| peta | Metrik multi-label utama | Rata-rata presisi rata-rata di seluruh kelas dan ambang batas. |

## Bacaan Lanjutan

- [Gong, Chung, Kaca (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — arsitektur rekaman dari tahun 2021–2024.
- [Chen dkk. (2022, edisi 2024). BEATs: Pra-Training Audio dengan Tokenizer Akustik](https://arxiv.org/abs/2212.09058) — default tahun 2024+.
- [Taman dkk. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — augmentasi audio yang dominan.
- [Piczak (2015). Dataset ESC-50](https://github.com/karolpiczak/ESC-50) — tolok ukur 50 kelas yang masih ada.
- [Gemmeke dkk. (2017). AudioSet](https://research.google.com/audioset/) — taksonomi YouTube kelas 632; masih menjadi standar emas.
