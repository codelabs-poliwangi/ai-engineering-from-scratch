# CNN dan RNN untuk Teks

> Konvolusi mempelajari n-gram. Ingat kekambuhan. Keduanya digantikan oleh attention. Keduanya masih penting pada perangkat keras yang terbatas.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 3 · 11 (Intro PyTorch), Fase 5 · 03 (Embedding Kata), Fase 4 · 02 (Konvolusi dari Awal)
**Waktu:** ~75 menit

## Masalah

TF-IDF dan Word2Vec menghasilkan vector datar yang mengabaikan urutan kata. Pengklasifikasi yang dibangun di atasnya tidak dapat membedakan `dog bites man` dari `man bites dog`. Urutan kata terkadang membawa sinyal.

Dua kelompok arsitektur mengisi kesenjangan tersebut sebelum trafo hadir.

**Jaring konvolusional untuk teks (TextCNN).** Terapkan konvolusi 1D pada rangkaian embedding kata. Filter dengan lebar 3 adalah pendeteksi trigram yang dapat dipelajari: filter ini mencakup tiga kata dan menghasilkan skor. Tumpuk lebar berbeda (2, 3, 4, 5) untuk mendeteksi pola multiskala. Kumpulan maksimum ke representasi ukuran tetap. Datar, paralel, cepat.

**Jaring berulang (RNN, LSTM, GRU).** Memproses token satu per satu, mempertahankan status tersembunyi yang meneruskan informasi. Panjang input yang berurutan, mengandung memori, dan fleksibel. Pemodelan sequence mendominasi dari tahun 2014 hingga 2017, kemudian terjadi attention.

Lesson ini membangun keduanya, lalu menyebutkan kegagalan yang memotivasi attention.

## Konsep

**TeksCNN** (Kim, 2014). Token tertanam. Konvolusi 1D lebar-`k` menggeser filter ke `k`-gram embedding berturut-turut, menghasilkan peta feature. Pengumpulan maksimal global pada peta tersebut memilih activation terkuat. Gabungkan output kumpulan maksimal dari beberapa lebar filter. Umpan ke kepala pengklasifikasi.

Mengapa ini berhasil. Filter adalah n-gram yang dapat dipelajari. Pengumpulan maksimum adalah invarian posisi, jadi "tidak bagus" mengaktifkan feature yang sama di awal atau tengah tinjauan. Tiga lebar filter dengan masing-masing 100 filter memberi kamu 300 detektor n-gram yang dipelajari. Training bersifat paralel; tidak ada ketergantungan berurutan.

**RNN.** Pada setiap langkah waktu `t`, status tersembunyi `h_t = f(W * x_t + U * h_{t-1} + b)`. Bagikan `W`, `U`, `b` sepanjang waktu. Keadaan tersembunyi pada waktu `T` adalah ringkasan dari keseluruhan awalan. Untuk klasifikasi, kumpulkan seluruh `h_1 ... h_T` (maks, rata-rata, atau terakhir).

RNN biasa mengalami vanishing gradient. **LSTM** menambahkan gerbang yang memutuskan apa yang harus dilupakan, apa yang disimpan, dan apa yang dihasilkan, menstabilkan gradient melalui rangkaian panjang. **GRU** menyederhanakan LSTM menjadi dua gerbang; melakukan hal yang sama dengan parameter yang lebih sedikit.

**RNN dua arah** menjalankan satu RNN maju dan RNN lainnya mundur, menggabungkan status tersembunyi. Representasi setiap token melihat konteks kiri dan kanan. Penting untuk menandai tugas.

## Build

### Langkah 1: TextCNN di PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` membentuk ulang `[batch, seq_len, embed_dim]` menjadi `[batch, embed_dim, seq_len]` karena `nn.Conv1d` memperlakukan sumbu tengah sebagai pipeline. Output yang dikumpulkan berukuran tetap terlepas dari panjang input.

### Langkah 2: Pengklasifikasi LSTM

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

Kumpulan maksimum di atas urutan, bukan kumpulan keadaan terakhir. Untuk klasifikasi, max-pooling biasanya mengalahkan pengambilan keadaan tersembunyi terakhir karena informasi di akhir rangkaian panjang cenderung mendominasi keadaan terakhir.

### Langkah 3: demo gradient hilang (intuisi)RNN biasa tanpa gating tidak dapat mempelajari dependensi jangka panjang. Pertimbangkan tugas mainan: memprediksi apakah token `A` muncul di mana saja secara berurutan. Jika `A` berada di posisi 1 dan urutannya memiliki panjang 100 token, gradient dari loss harus mengalir kembali melalui 99 perkalian weight berulang. Jika bobotnya kurang dari 1, gradiennya hilang. Jika lebih dari 1 maka akan meledak.

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM memperbaikinya dengan **status sel** yang berjalan melalui jaringan hanya dengan interaksi aditif (gerbang lupa menskalakannya secara multiplikatif, namun gradient tetap mengalir di sepanjang "jalan raya"). GRU melakukan hal serupa dengan parameter yang lebih sedikit. Keduanya memberi kamu training stabil melalui 100+ urutan langkah.

### Langkah 4: mengapa ini masih belum cukup

Tiga masalah tetap ada bahkan dengan LSTM.

1. **Kemacetan berurutan.** Melatih RNN pada urutan dengan panjang 1000 memerlukan 1000 langkah maju/mundur berurutan. Tidak dapat memparalelkan lintas waktu.
2. **Vector konteks berukuran tetap dalam penyiapan encoder-decoder.** Decoder hanya melihat status akhir encoder yang tersembunyi, yang dikompresi pada seluruh input. Input yang panjang kehilangan detail. Lesson 09 membahas hal ini secara langsung.
3. **Batas akurasi ketergantungan distance jauh.** LSTM mengungguli RNN biasa namun masih kesulitan menyebarkan informasi spesifik ke lebih dari 200 langkah.

Attention menyelesaikan ketiganya. Transformers menghilangkan kekambuhan sepenuhnya. Lesson 10 adalah porosnya.

## Pakai

`nn.LSTM` PyTorch, `nn.GRU`, dan `nn.Conv1d` sudah siap produksi. Code training adalah standar.

Hugging Face mengirimkan embeddings terlatih yang kamu colokkan sebagai layer input:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

Daftar periksa penggunaan ketika sesuai dengan batasan.

- **Inference tepi/pada perangkat.** TextCNN dengan embedding GloVe 10-100x lebih kecil dari Transformer. Jika target penerapan kamu adalah telepon, ini adalah tumpukannya.
- **Streaming / klasifikasi online.** RNN memproses satu token dalam satu waktu; Transformer memerlukan urutan penuh. Untuk teks masuk real-time, LSTM tetap menang.
- **Model kecil untuk garis dasar.** Iterasi cepat pada tugas baru. Latih TextCNN dalam 5 menit pada CPU.
- **Pelabelan urutan dengan data terbatas.** BiLSTM-CRF (lesson 06) masih merupakan arsitektur NER tingkat produksi untuk kalimat berlabel 1k-10k.

Segala sesuatu yang lain masuk ke trafo.

## Kirim

Simpan sebagai `outputs/prompt-text-encoder-picker.md`:

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## Latihan

1. **Mudah.** Latih TextCNN pada dataset mainan 3 kelas (kamu yang menciptakan datanya). Verifikasi bahwa lebar filter (2, 3, 4) mengungguli lebar tunggal (3) rata-rata F1.
2. **Medium.** Menerapkan pengumpulan max-pool, mean-pool, dan last-state untuk pengklasifikasi LSTM. Bandingkan pada dataset kecil; dokumentasikan penggabungan mana yang menang dan buat hipotesis alasannya.
3. **Sulit.** Membuat penanda NER BiLSTM-CRF (gabungkan lesson 06 dan lesson ini). Berlatih di CoNLL-2003. Bandingkan dengan garis dasar CRF saja dari lesson 06 dan penyempurnaan BERT. Laporkan waktu training, memori, dan F1.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| TeksCNN | CNN untuk teks | Tumpukan konvolusi 1D pada embedding kata dengan kumpulan maksimal global. Kim (2014). |
| RNN | Jaring berulang | Status tersembunyi diperbarui pada setiap langkah waktu: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | RNN berpagar | Menambahkan gerbang input/lupa/output + status sel. Berlatih secara stabil melalui urutan yang panjang. |
| GRU | LSTM yang lebih sederhana | Dua gerbang, bukan tiga. Akurasi serupa, parameter lebih sedikit. |
| Dua arah | Kedua arah | RNN maju + mundur digabungkan. Setiap token melihat kedua sisi konteksnya. |
| Gradient menghilang | Sinyal latihan mati | Perkalian berulang dengan <1 weight di RNN biasa membuat gradient langkah awal menjadi nol secara efektif. |

## Bacaan Lanjutan

- [Kim, Y. (2014). Jaringan Neural Konvolusional untuk Klasifikasi Kalimat](https://arxiv.org/abs/1408.5882) - makalah TextCNN. Delapan halaman. Dapat dibaca.
- [Hochreiter, S. dan Schmidhuber, J. (1997). Memori Jangka Pendek Panjang](https://www.bioinf.jku.at/publications/older/2604.pdf) — makalah LSTM. Jelas sekali.
- [Olah, C. (2015). Memahami Jaringan LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — diagram yang membuat LSTM dapat diakses oleh semua orang.
