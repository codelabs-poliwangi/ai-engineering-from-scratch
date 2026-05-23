# Model Urutan-ke-Urutan

> Dua RNN berpura-pura menjadi penerjemah. Kemacetan yang mereka alami adalah alasan adanya attention.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 5 · 08 (CNN + RNN untuk Teks), Fase 3 · 11 (Intro PyTorch)
**Waktu:** ~75 menit

## Masalah

Klasifikasi memetakan urutan panjang variabel ke satu label. Terjemahan memetakan urutan panjang variabel ke urutan panjang variabel lainnya. Input dan output berada dalam kosakata berbeda, mungkin bahasa berbeda, tanpa jaminan paritas panjang.

Arsitektur seq2seq (Sutskever, Vinyals, Le, 2014) memecahkannya dengan resep yang sengaja dibuat sederhana. Dua RNN. Seseorang membaca kalimat sumber dan menghasilkan vector konteks berukuran tetap. Yang lain membaca vector itu dan menghasilkan token kalimat target demi token. Code yang sama yang kamu tulis untuk lesson 08, direkatkan secara berbeda.

Hal ini layak dipelajari karena dua alasan. Pertama, hambatan vector konteks adalah kegagalan yang paling berguna secara pedagogis dalam NLP. Ini memotivasi semua attention dan keahlian Transformer. Kedua, resep training (pemaksaan guru, pengambilan sample terjadwal, pencarian berkas di inference) masih berlaku untuk setiap sistem generasi modern termasuk LLM.

## Konsep

**Encoder.** RNN yang membaca kalimat sumber. Status tersembunyi terakhirnya adalah **vector konteks** — ringkasan berukuran tetap dari seluruh input. Seharusnya tidak kehilangan apa pun kecuali sumbernya.

**Decoder.** RNN lain diinisialisasi dari vector konteks. Pada setiap langkah, dibutuhkan token yang dihasilkan sebelumnya sebagai input dan menghasilkan distribusi pada kosakata target. Sample atau argmax untuk memilih token berikutnya. Masukkan kembali. Ulangi hingga token `<EOS>` dihasilkan atau panjang maksimum tercapai.

**Training:** Kehilangan entropi silang pada setiap langkah dekoder, dijumlahkan berdasarkan urutan. Backprop standar sepanjang waktu melalui kedua jaringan.

**Paksaan guru.** Selama training, input decoder pada langkah `t` adalah token *kebenaran dasar* pada posisi `t-1`, bukan prediksi decoder sebelumnya. Ini menstabilkan training; tanpanya, kesalahan awal akan terjadi dan model tidak akan pernah belajar. Saat melakukan inference, kamu harus menggunakan prediksi model itu sendiri, sehingga selalu ada kesenjangan distribusi kereta/inference. Kesenjangan tersebut disebut **bias eksposur**.

**Kemacetan.** Segala sesuatu yang dipelajari pembuat enkode tentang sumber harus dimasukkan ke dalam satu vector konteks tersebut. Kalimat yang panjang kehilangan detail. Kata-kata langka menjadi kabur. Penyusunan ulang (chat noir vs kucing hitam) harus dihafal, bukan dihitung.

Attention (lesson 10) memperbaikinya dengan membiarkan decoder melihat *setiap* status tersembunyi encoder, bukan hanya yang terakhir. Itulah keseluruhan nadanya.

## Build

### Langkah 1: pembuat enkode

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` memiliki bentuk `[batch, seq_len, hidden_dim]` — satu status tersembunyi per posisi input. `hidden` memiliki bentuk `[1, batch, hidden_dim]` — langkah terakhir. Lesson 08 mengatakan "menggabungkan output untuk klasifikasi." Di sini kita menyimpan status tersembunyi terakhir sebagai vector konteks, dan mengabaikan output per langkah.

### Langkah 2: dekoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Decoder disebut selangkah demi selangkah. Input: kumpulan token tunggal dan status tersembunyi saat ini. Output: log kosakata untuk token berikutnya dan status tersembunyi yang diperbarui.

### Langkah 3: putaran training dengan pemaksaan guru

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```Dua kenop yang layak diberi nama. `ignore_index=0` melewatkan loss pada token padding. `teacher_forcing_ratio` adalah probabilitas penggunaan token sebenarnya vs. prediksi model pada setiap langkah. Mulai dari 1,0 (pemaksaan guru penuh) dan turunkan hingga ~0,5 selama training untuk menutup kesenjangan bias eksposur.

### Langkah 4: loop inference (serakah)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Dekode serakah memilih token dengan probabilitas tertinggi di setiap langkah. Itu bisa hilang: begitu kamu berkomitmen pada sebuah token, kamu tidak bisa membatalkannya. **Penelusuran berkas** menjaga urutan parsial teratas-`k` tetap hidup dan memilih urutan lengkap dengan skor tertinggi di akhir. Lebar balok 3-5 adalah standar.

### Langkah 5: hambatan, ditunjukkan

Latih model pada tugas menyalin mainan: sumber `[a, b, c, d, e]`, target `[a, b, c, d, e]`. Meningkatkan panjang urutan. Amati keakuratannya.

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

Satu status tersembunyi GRU tidak dapat mengingat input 40 token tanpa kehilangan. Informasinya ada di setiap langkah encoder, tetapi decoder hanya melihat keadaan terakhir. Attention memperbaikinya secara langsung.

## Pakai

PyTorch memiliki templat seq2seq berbasis `nn.Transformer` dan `nn.LSTM`. Pustaka `transformers` Hugging Face mengirimkan model encoder-decoder lengkap (BART, T5, mBART, NLLB) yang dilatih pada miliaran token.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

Encoder-decoder modern menghilangkan RNN untuk Transformer. Bentuk tingkat tinggi (encoder, decoder, generate-token-by-token) identik dengan kertas seq2seq 2014. Mekanisme di dalam setiap blok berbeda-beda.

### Kapan masih menggunakan seq2seq berbasis RNN

Hampir tidak pernah, untuk proyek baru. Pengecualian khusus:

- Terjemahan streaming di mana kamu menggunakan input satu token pada satu waktu dengan memori terbatas.
- Pembuatan teks pada perangkat yang biaya memori Transformer-nya mahal.
- Pedagogi. Memahami hambatan encoder-decoder adalah cara tercepat untuk memahami mengapa Transformer menang.

### Bias paparan dan mitigasinya

- **Pengambilan sample terjadwal.** Anneal rasio pemaksaan guru selama training sehingga model belajar untuk pulih dari kesalahannya sendiri.
- **Training risiko minimum.** Berlatihlah dengan skor BLEU tingkat kalimat, bukan entropi lintas tingkat token. Lebih dekat dengan apa yang sebenarnya kamu inginkan.
- **Penyempurnaan pembelajaran penguatan.** Hadiahi pembuat urutan dengan metrik. Digunakan di LLM RLHF modern.

Ketiganya masih berlaku untuk pembangkitan berbasis trafo.

## Kirim

Simpan sebagai `outputs/prompt-seq2seq-design.md`:

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## Latihan

1. **Mudah.** Terapkan tugas menyalin mainan. Latih seq2seq GRU pada pasangan input-output yang targetnya sama dengan sumbernya. Ukur akurasi pada panjang 5, 10, 20. Reproduksi kemacetan.
2. **Sedang.** Tambahkan decoding penelusuran berkas dengan lebar berkas 3. Ukur BLEU pada korpus paralel kecil terhadap serakah. Dokumentasikan di mana pencarian sinar menang (biasanya token terakhir) dan di mana tidak ada bedanya.
3. **Sulit.** Sempurnakan `facebook/bart-base` pada dataset parafrase 10 ribu pasangan. Bandingkan output beam-4 model yang telah disetel dengan baik dengan input ditahan model dasar. Laporkan BLEU dan pilih 10 contoh kualitatif.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Pembuat enkode | Masukkan RNN | Membaca sumber. Menghasilkan status tersembunyi per langkah dan vector konteks akhir. |
| Dekoder | Output RNN | Diinisialisasi dari vector konteks. Menghasilkan token target satu per satu. |
| Vector konteks | Ringkasan | Status tersembunyi encoder terakhir. Ukuran tetap. Attention yang menghambat terpecahkan. |
| Guru memaksa | Gunakan token yang sebenarnya | Berikan token kebenaran dasar sebelumnya pada waktu training. Menstabilkan pembelajaran. |
| Bias eksposur | Kesenjangan training/ujian | Model yang dilatih menggunakan token sebenarnya tidak pernah berlatih memulihkan kesalahannya sendiri. |
| Pencarian balok | Penguraian code yang lebih baik | Pertahankan urutan parsial top-k tetap hidup di setiap langkah alih-alih melakukan dengan rakus. |

## Bacaan Lanjutan

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning dengan Neural Networks](https://arxiv.org/abs/1409.3215) — makalah seq2seq asli. Empat halaman.
- [Cho dkk. (2014). Mempelajari Representasi Frasa menggunakan RNN Encoder-Decoder untuk Terjemahan Mesin Statistik](https://arxiv.org/abs/1406.1078) — memperkenalkan GRU dan framing encoder-decoder.
- [Bahdanau, Cho, Bengio (2014). Terjemahan Mesin Neural dengan Belajar Bersama untuk Menyelaraskan dan Menerjemahkan](https://arxiv.org/abs/1409.0473) — makalah attention. Baca segera setelah lesson ini.
- [Tutorial PyTorch NLP dari Awal](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — seq2seq + code attention yang dapat dibangun.
