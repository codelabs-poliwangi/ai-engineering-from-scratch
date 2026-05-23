# T5, BART — Model Encoder-Decoder

> Pembuat enkode memahami. Decoder menghasilkan. Gabungkan kembali semuanya dan kamu akan mendapatkan model yang dibuat untuk tugas input → output: menerjemahkan, meringkas, menulis ulang, menyalin.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 7 · 05 (Trafo Penuh), Fase 7 · 06 (BERT), Fase 7 · 07 (GPT)
**Waktu:** ~45 menit

## Masalah

GPT khusus decoder dan BERT khusus encoder masing-masing menghapus arsitektur 2017 untuk tujuan yang berbeda. Tetapi banyak tugas yang secara alami merupakan input-output:

- Terjemahan: Inggris → Prancis.
- Ringkasan: artikel 5.000 token → ringkasan 200 token.
- Pengenalan ucapan: token audio → token teks.
- Ekstraksi terstruktur: prosa → JSON.

Untuk ini, encoder-decoder adalah pilihan yang paling tepat. Encoder menghasilkan representasi sumber yang padat. Decoder menghasilkan output, memperhatikan representasi tersebut di setiap langkah. Training dilakukan secara shift per satu di sisi output. Kerugiannya sama seperti GPT, hanya dikondisikan pada output encoder.

Dua makalah mendefinisikan pedoman modern:

1. **T5** (Raffel dkk. 2019). "Transformer Transfer Teks-ke-Teks." Setiap tugas NLP dibingkai ulang sebagai text-in, text-out. Arsitektur tunggal, kosakata tunggal, loss tunggal. Telah dilatih sebelumnya tentang prediksi rentang bertopeng (rentang yang rusak di input, dekodekan di output).
2. **BART** (Lewis dkk. 2019). "Trafo Dua Arah dan Regresif Otomatis." Menyangkal autoencoder: input rusak dalam berbagai cara (acak, tutupi, hapus, putar), minta decoder untuk merekonstruksi aslinya.

Pada tahun 2026, format encoder-decoder tetap ada di mana struktur input penting:

- Bisikan (ucapan → teks).
- Tumpukan terjemahan Google.
- Beberapa model penyelesaian code/perbaikan yang memiliki struktur konteks dan pengeditan yang berbeda.
- Flan-T5 dan varian untuk tugas penalaran terstruktur.

Hanya decoder yang mendapat sorotan, tetapi encoder-decoder tidak pernah hilang.

## Konsep

![Encoder-decoder dengan attention silang](../assets/encoder-decoder.svg)

### Putaran ke depan

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

Yang terpenting, encoder berjalan satu kali per input. Decoder berjalan secara autoregresif tetapi memperhatikan output encoder yang *sama* di setiap langkah. Menyimpan output encoder dalam cache adalah percepatan gratis untuk input panjang.

### Prapelatihan T5 — rentang korupsi

Pilih rentang input secara acak (panjang rata-rata 3 token, total 15%). Ganti setiap span dengan sentinel unik: `<extra_id_0>`, `<extra_id_1>`, dll. Dekoder hanya mengeluarkan span yang rusak dengan awalan sentinelnya:

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

Sinyal lebih murah daripada memprediksi seluruh rangkaian. Kompetitif dengan MLM (BERT) dan awalan-LM (UniLM) dalam ablasi kertas T5.

### Pra-training BART — penghilangan multi-kebisingan

BART mencoba lima fungsi noise:

1. Penyembunyian token.
2. Penghapusan token.
3. Pengisian teks (menutupi rentang, dekoder menyisipkan panjang yang tepat).
4. Permutasi kalimat.
5. Rotasi dokumen.

Menggabungkan pengisian teks + permutasi kalimat menghasilkan angka hilir terbaik. Decoder selalu merekonstruksi aslinya. Output BART adalah urutan penuh, bukan hanya rentang yang rusak — sehingga komputasi pra-training lebih tinggi dari T5.

### Inference

Generasi autoregresif yang sama seperti GPT. Pengambilan sample serakah / balok / top-p berlaku. Pencarian berkas (lebar 4–5) merupakan standar untuk penerjemahan dan peringkasan karena distribusi keluarannya lebih sempit daripada obrolan.

### Kapan harus memilih setiap varian pada tahun 2026| Tugas | Encoder-decoder? | Mengapa |
|------|------------------|-----|
| Terjemahan | Ya, biasanya | Hapus urutan sumber; distribusi output tetap; pencarian sinar berfungsi |
| Pidato-ke-teks | Ya (Berbisik) | Modalitas input berbeda dengan output; feature audio bentuk encoder |
| Obrolan / penalaran | Tidak, hanya dekoder | Tidak ada "input" yang terus-menerus — percakapannya adalah urutan |
| Penyelesaian code | Biasanya tidak | Hanya decoder dengan konteks panjang yang menang; model code seperti Qwen 2.5 Coder hanya untuk decoder |
| Ringkasan | Keduanya berfungsi | BART, PEGASUS mengalahkan baseline khusus dekoder sebelumnya; LLM khusus decoder modern yang cocok dengan mereka |
| Ekstraksi terstruktur | Baik | T5 bersih karena "teks → teks" menyerap format output apa pun |

Tren sejak ~2022: khusus dekoder mengambil alih tugas yang biasanya dimiliki oleh dekoder-enkode karena (a) LLM khusus dekoder yang disetel dengan instruksi menggeneralisasi apa pun melalui prompt, (b) satu arsitektur berskala lebih mudah daripada dua, (c) RLHF mengasumsikan dekoder. Encoder-decoder berlaku jika modalitas input berbeda (ucapan, gambar) atau jika kualitas penelusuran berkas penting.

## Build

Lihat `code/main.py`. Kami menerapkan korupsi rentang gaya T5 untuk korpus mainan — bagian yang paling berguna dari lesson ini karena muncul di setiap resep pra-training encoder-decoder sejak saat itu.

### Langkah 1: rentangkan korupsi

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

Format targetnya adalah konvensi T5: `<sent0> span0 <sent1> span1 ...`. Input yang rusak menyisipkan token yang tidak berubah dengan token sentinel di lokasi rentang.

### Langkah 2: verifikasi pulang pergi

Mengingat input dan target yang rusak, rekonstruksi kalimat aslinya. Jika korupsi kamu dapat dibalikkan, maka dampaknya sudah jelas. Ini adalah pemeriksaan kewarasan - training sebenarnya tidak pernah melakukan hal ini, tetapi tes ini murah dan mendeteksi satu per satu bug dalam pembukuan rentang waktu kamu.

### Langkah 3: BART mengeluarkan suara

Lima fungsi: `token_mask`, `token_delete`, `text_infill`, `sentence_permute`, `document_rotate`. Buatlah dua diantaranya dan tunjukkan hasilnya.

## Pakai

Referensi HuggingFace:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

Trik T5: nama tugas dimasukkan ke dalam teks input. Model yang sama menangani lusinan tugas karena setiap tugas berupa teks masuk dan keluar teks. Pada tahun 2026, pola ini telah digeneralisasikan oleh model khusus dekoder yang disesuaikan dengan instruksi, tetapi T5 mengkodifikasikannya terlebih dahulu.

## Kirim

Lihat `outputs/skill-seq2seq-picker.md`. Keterampilan memilih antara encoder-decoder dan decoder saja untuk tugas baru dengan mempertimbangkan struktur input-output, latensi, dan target kualitas.

## Latihan

1. **Mudah.** Jalankan `code/main.py`, terapkan korupsi rentang pada kalimat 30 token, verifikasi bahwa penggabungan token sumber non-sentinel dengan rentang target yang didekodekan akan mereproduksi yang asli.
2. **Medium.** Menerapkan derau `text_infill` BART: ganti span acak dengan satu token `<mask>`, dan decoder harus menyimpulkan panjang span yang benar serta kontennya. Tunjukkan satu contoh.
3. **Sulit.** Sempurnakan `flan-t5-small` dalam bahasa Inggris kecil → korpus babi-Latin (200 pasang). Ukur BLEU pada set 50 pasang yang dipegang. Bandingkan dengan penyempurnaan `Llama-3.2-1B` pada data yang sama dengan komputasi yang sama.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Encoder-decoder | "Transformer Seq2seq" | Dua tumpukan: encoder dua arah untuk input, decoder kausal dengan attention silang untuk output. |
| Attention silang | "Di mana sumber pembicaraan untuk dituju" | Q × K/V pembuat enkode. Satu-satunya tempat informasi encoder masuk ke decoder. |
| Rentang korupsi | "Trik pra-training T5" | Ganti rentang acak dengan token sentinel; decoder mengeluarkan rentang. |
| Mencela tujuan | "Permainan BART" | Terapkan fungsi noise ke input, latih decoder untuk merekonstruksi urutan bersih. |
| Token penjaga | "Placeholder `<extra_id_N>`" | Token khusus yang menandai rentang yang rusak di sumber dan memberi tag ulang di target. |
| flan | "T5 yang disesuaikan dengan instruksi" | T5 menyempurnakan >1.800 tugas; membuat encoder-decoder kompetitif dalam mengikuti instruksi. |
| Pencarian balok | "Strategi penguraian code" | Pertahankan urutan parsial top-k di setiap langkah; standar untuk penerjemahan/ringkasan. |
| Guru memaksa | "Input waktu training" | Selama training, masukkan token output sebelumnya yang sebenarnya ke dekoder, bukan token sample. |

## Bacaan Lanjutan

- [Raffel dkk. (2019). Menjelajahi Batasan Pembelajaran Transfer dengan Transformer Teks-ke-Teks Terpadu](https://arxiv.org/abs/1910.10683) — T5.
- [Lewis dkk. (2019). BART: Pra-training Denoising Sequence-to-Sequence untuk Pembuatan, Terjemahan, dan Pemahaman Bahasa Alami](https://arxiv.org/abs/1910.13461) — BART.
- [Chung dkk. (2022). Model Bahasa yang Diselaraskan dengan Instruksi Penskalaan](https://arxiv.org/abs/2210.11416) — Flan-T5.
- [Radford dkk. (2022). Pengenalan Ucapan yang Kuat melalui Pengawasan Lemah Berskala Besar](https://arxiv.org/abs/2212.04356) — Whisper, encoder-decoder kanonikal 2026.
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — implementasi referensi.
