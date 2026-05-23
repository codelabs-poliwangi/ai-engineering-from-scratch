# Build Transformer dari Awal — Batu Penjuru

> Tiga belas lesson. Satu model. Tidak ada jalan pintas.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 7 · 01 hingga 13. Jangan lewati.
**Waktu:** ~120 menit

## Masalah

kamu telah membaca setiap makalah. kamu telah menerapkan attention, pemisahan multi-head, pengkodean posisi, blok encoder dan decoder, loss BERT dan GPT, MoE, cache KV. Sekarang buat mereka bekerja sama dalam tugas nyata.

Batu penjuru: latih Transformer kecil khusus dekoder ujung ke ujung pada tugas pemodelan bahasa tingkat karakter. Bunyinya Shakespeare. Ini menghasilkan Shakespeare baru. Cukup kecil untuk berlatih di laptop dalam waktu kurang dari 10 menit. Memang benar bahwa menukar dataset yang lebih besar dan training yang lebih lama akan memberi kamu LM yang sesungguhnya.

Ini adalah kursus "nanoGPT". Ini tidak asli — Tutorial nanoGPT Karpathy 2023 adalah implementasi referensi yang ditulis setiap siswa setidaknya satu kali. Kami mengangkat bentuknya dan melengkapinya kembali di sekitar apa yang telah kami bahas.

## Konsep

![Diagram blok Transformer dari awal](../assets/capstone.svg)

Arsitekturnya, dijelaskan:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### Apa yang kami kirimkan

- `GPTConfig` — satu tempat untuk mengonfigurasi semua hyperparameter.
- `MultiHeadAttention` — kausal, batch, dengan jalur gaya Flash opsional (`scaled_dot_product_attention` PyTorch).
- `SwiGLUFFN` — FFN modern.
- `Block` — attention pra-norm dan terbungkus sisa + FFN.
- `GPT` — embedding, blok bertumpuk, kepala LM, hasilkan().
- Training loop dengan AdamW, cosine LR, kliping gradient.
- Tokenizer tingkat char pada teks Shakespeare.

### Apa yang tidak kami kirimkan

- RoPE — diimplementasikan secara konseptual di Lesson 04. Di sini kami menggunakan embedding posisi yang dipelajari untuk kesederhanaan. Latihan ini meminta kamu untuk menukar RoPE.
- Cache KV selama pembuatan — setiap langkah pembuatan menghitung ulang attention pada awalan penuh. Lebih lambat tapi lebih sederhana. Latihan meminta kamu untuk menambahkan cache KV.
- Attention Flash — Pengiriman otomatis PyTorch 2.0+ jika inputnya cocok; kami menggunakan `F.scaled_dot_product_attention`.
- MoE — FFN tunggal per blok. kamu melihat MoE di Lesson 11.

### Metrik target

Pada laptop Mac M2, d_model=128 GPT 4 lapis, 4 kepala, dilatih untuk 2.000 langkah di `tinyshakespeare.txt`:

- Loss training menyatu dari ~4,2 (acak) menjadi ~1,5 dalam waktu sekitar 6 menit.
- Contoh output tampak berbentuk Shakespeare: kata-kata kuno, jeda baris, nama diri seperti "ROMEO:" muncul.
- Val loss (10% teks terakhir yang dibagikan) melacak loss training dengan cermat; tidak ada overfitting pada ukuran/anggaran ini.

## Build

Lesson ini menggunakan PyTorch. Instal `torch` (CPU build baik-baik saja). Lihat `code/main.py`. Skrip menangani:

- Mengunduh `tinyshakespeare.txt` jika hilang (atau membaca salinan lokal).
- Tokenizer karakter tingkat byte.
- Pembagian kereta/val pada 90/10.
- Loop training dengan autocast bf16 pada perangkat keras yang didukung.
- Pengambilan sample setelah training selesai.

### Langkah 1: data

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 karakter unik. Kosakata kecil. Sesuai dengan vocab_size 4-byte. Tanpa BPE, tanpa drama tokenizer.

### Langkah 2: model

Lihat `code/main.py`. Blok ini adalah buku teks dari Lesson 05 - pra-norm, RMSNorm, SwiGLU, MHA kausal. Jumlah parameter untuk 4/4/128: ~800K.

### Langkah 3: putaran training

Dapatkan kumpulan acak jendela token dengan panjang 256. Maju. Pergeseran demi satu entropi silang. Ke belakang. langkah AdamW. Catatan. Mengulang.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Langkah 4: sample

Diberikan contoh prompt, teruskan berulang kali, dari logit top-p, tambahkan, dan lanjutkan. Berhenti setelah 500 token.

### Langkah 5: baca hasilnyaSetelah 2.000 langkah:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

Bukan Shakespeare. Tapi berbentuk Shakespeare. Kemenangan yang jelas untuk ~800 ribu parameter dan 6 menit di laptop.

## Pakai

Batu penjuru ini adalah arsitektur referensi. Tiga ekstensi untuk mengirimkannya ke sesuatu yang nyata:

1. **Tukar tokenizer.** Gunakan BPE (misalnya `tiktoken.get_encoding("cl100k_base")`). Ukuran kosakata melonjak dari 65 menjadi ~50.000. Kapasitas model perlu ditingkatkan untuk mengimbanginya.
2. **Berlatih dengan korpus yang lebih besar.** Gunakan `OpenWebText` atau `fineweb-edu` (HuggingFace). Token 10 miliar pada satu A100 membutuhkan ~24 jam untuk GPT 125 juta parameter.
3. **Tambahkan RoPE + cache KV + Flash Attention.** Latihan di bawah ini memandu kamu melalui masing-masing latihan.

Ini menghasilkan GPT dengan parameter 125 juta yang menghasilkan bahasa Inggris yang lancar. Bukan model perbatasan. Namun jalur code yang sama — hanya saja lebih besar — ​​adalah yang digunakan Karpathy, EleutherAI, dan Allen Institute untuk melatih pos pemeriksaan penelitian pada tahun 2026.

## Kirim

Lihat `outputs/skill-transformer-review.md`. Keterampilan ini meninjau implementasi Transformer dari awal untuk mengetahui kebenarannya di seluruh 13 lesson sebelumnya.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Verifikasi bahwa loss validasi langkah terakhir model terlatih kamu berada di bawah 2.0. Ubah `max_steps` dari 2.000 menjadi 5.000 — apakah nilai loss terus meningkat?
2. **Medium.** Ganti embedding posisi yang dipelajari dengan RoPE. Terapkan rotasi ke Q dan K di dalam `MultiHeadAttention`. Latih dan verifikasi loss val setidaknya sama rendahnya.
3. **Medium.** Mengimplementasikan cache KV dalam loop pengambilan sample. Hasilkan 500 token dengan dan tanpa cache. Jam dinding akan meningkat 5–20× di laptop.
4. **Hard.** Tambahkan kepala kedua ke model yang memprediksi token berikutnya-plus-satu (MTP — Prediksi Multi-Token dari DeepSeek-V3). Berlatih bersama. Apakah itu membantu?
5. **Hard.** Ganti FFN tunggal per blok dengan MoE 4 ahli. Router + perutean 2 teratas. Lihat bagaimana perubahan val loss pada parameter aktif yang cocok.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| nanoGPT | "Repo tutorial Karpathy" | Code training trafo khusus dekoder minimal, ~300 LOC; referensi kanonik. |
| tinyshakespeare | "Korpus mainan standar" | ~1,1 MB teks; setiap tutorial karakter-LM sejak 2015 menggunakannya. |
| Embedding terikat | "Bagikan matrix input/output" | Weight kepala LM = transpos matrix embedding token; menghemat parameter, meningkatkan kualitas. |
| transmisi otomatis bf16 | "Melatih trik presisi" | Jalankan maju/mundur di bf16, pertahankan status optimizer di fp32; standar sejak tahun 2021. |
| Kliping gradient | "Menghentikan lonjakan" | Batasi norm lulusan global pada 1,0; mencegah ledakan training. |
| Jadwal Cosinus LR | "Default 2020+" | LR meningkat secara linier (pemanasan) kemudian meluruh secara kosinus hingga 10% dari puncak. |
| MFU | "Pemanfaatan Model FLOP" | Mencapai FLOP/puncak teoritis; 40% padat, 30% MoE kuat di tahun 2026. |
| Loss Val | "Loss yang ditahan" | Entropi silang pada data yang tidak pernah dilihat oleh model; detektor pakaian berlebih. |

## Bacaan Lanjutan

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — implementasi beranotasi klasik.
