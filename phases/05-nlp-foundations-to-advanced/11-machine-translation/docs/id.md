# Terjemahan Mesin

> Penerjemahan adalah tugas yang membayar penelitian NLP selama tiga puluh tahun dan terus membayar hingga saat ini.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 5 · 10 (Mekanisme Attention), Fase 5 · 04 (GloVe, FastText, Subword)
**Waktu:** ~75 menit

## Masalah

Seorang model membaca kalimat dalam satu bahasa dan menghasilkan kalimat dalam bahasa lain. Panjangnya bervariasi. Urutan kata bervariasi. Beberapa kata sumber dipetakan ke beberapa kata target dan sebaliknya. Idiom menolak pemetaan satu-ke-satu. "Aku merindukanmu" dalam bahasa Prancis adalah "tu me manques" yang secara harafiah berarti "kamu kekurangan bagiku." Tidak ada penyelarasan tingkat kata yang dapat bertahan.

Terjemahan mesin adalah tugas yang memaksa NLP untuk menciptakan encoder-decoder, attention, Transformer, dan akhirnya paradigma LLM secara keseluruhan. Setiap langkah maju tercapai karena kualitas terjemahan dapat diukur dan kesenjangan antara manusia dan mesin sangat besar.

Lesson ini melewatkan lesson sejarah dan mengajarkan alur kerja tahun 2026: encoder-decoder multibahasa yang telah dilatih sebelumnya (NLLB-200 atau mBART), tokenization subkata, pencarian berkas, evaluasi BLEU dan chrF, dan beberapa mode kegagalan yang masih dikirim ke produksi tanpa tertangkap.

## Konsep

![Pipa MT: tokenize → encode → decode dengan attention → detokenize](../assets/mt-pipeline.svg)

MT modern adalah encoder-decoder Transformer yang dilatih pada teks paralel. Pembuat enkode membaca sumber dalam tokenization bahasanya. Decoder menghasilkan target, satu subkata pada satu waktu, menggunakan output encoder melalui attention silang (lesson 10). Decoding menggunakan pencarian berkas untuk menghindari jebakan decoding serakah. Outputnya didetokenisasi, didetruecasing, dan dinilai berdasarkan referensi.

Tiga pilihan operasional mendorong kualitas MT dunia nyata.

- **Tokenizer.** SentencePiece BPE dilatih pada korpus bahasa campuran. Kosakata bersama antar bahasa inilah yang memungkinkan zero-shot pair di NLLB.
- **Ukuran model.** NLLB-200 suling 600M muat di laptop. NLLB-200 3.3B adalah standar produksi yang dipublikasikan. 54.5B adalah batas atas penelitian.
- **Decoding.** Lebar balok 4-5 untuk konten umum. Penalti panjang untuk menghindari output terlalu pendek. Penguraian code yang dibatasi ketika kamu membutuhkan konsistensi terminologi.

## Build

### Langkah 1: panggilan MT yang telah dilatih sebelumnya

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

Ada tiga hal penting di sini. `src_lang` memberi tahu tokenizer skrip dan segmentasi mana yang akan diterapkan. `forced_bos_token_id` memberi tahu dekoder bahasa mana yang akan dihasilkan. Keduanya merupakan trik khusus NLLB; mBART dan M2M-100 menggunakan konvensinya sendiri dan tidak dapat dipertukarkan.

### Langkah 2: BLEU dan chrF

BLEU mengukur tumpang tindih n-gram antara output dan referensi. Empat ukuran referensi n-gram (1-4), rata-rata presisi geometrik, penalti singkat untuk output yang terlalu pendek. Skornya ada di [0, 100]. Biasa digunakan. Frustrasi untuk menafsirkan: 30 BLEU "dapat digunakan"; 40 adalah "baik"; 50 adalah "luar biasa"; perbedaan di bawah 1 BLEU adalah kebisingan.

chrF mengukur F-score tingkat karakter. Lebih sensitif terhadap bahasa yang kaya secara morfologis di mana jumlah BLEU kurang cocok. Sering dilaporkan bersama BLEU.

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

Selalu gunakan `sacrebleu`. Ini menormalkan tokenization sehingga skornya sebanding di seluruh makalah. Menggulirkan perhitungan BLEU kamu sendiri adalah penyebab terjadinya tolok ukur yang menyesatkan.

### Hierarki evaluasi tiga tingkat (2026)

Evaluasi MT modern menggunakan tiga kelompok metrik yang saling melengkapi. Kirim dengan setidaknya dua.- **Heuristik** (BLEU, chrF). Cepat, berbasis referensi, dapat ditafsirkan, tidak peka terhadap parafrase. Gunakan untuk perbandingan lama dan deteksi regresi.
- **Dipelajari** (COMET, BLEURT, BERTScore). Model saraf dilatih berdasarkan penilaian manusia; membandingkan kesamaan semantik terjemahan dengan sumber dan referensi. COMET memiliki hubungan tertinggi dengan penelitian MT sejak tahun 2023 dan merupakan standar produksi tahun 2026 yang mengutamakan kualitas.
- **LLM-sebagai-hakim** (bebas referensi). Anjurkan model besar untuk menilai terjemahan berdasarkan kelancaran, kecukupan, nada, dan kesesuaian budaya. GPT-4-sebagai-hakim sesuai dengan kesepakatan manusia ~80% jika rubrik dirancang dengan baik. Gunakan untuk konten terbuka yang tidak memiliki referensi.

Tumpukan praktis tahun 2026: `sacrebleu` untuk BLEU dan chrF, `unbabel-comet` untuk COMET, dan LLM yang diminta untuk sinyal terakhir yang menghadap manusia. Kalibrasi setiap metrik terhadap 50-100 contoh yang diberi label manusia sebelum mempercayainya pada data produksi.

Metrik bebas referensi (COMET-QE, BLEURT-QE, LLM-as-judge) memungkinkan kamu mengevaluasi terjemahan tanpa referensi, yang penting untuk pasangan bahasa long-tail yang tidak memiliki terjemahan referensi.

### Langkah 3: apa yang rusak dalam produksi

Pipeline yang berfungsi di atas akan menerjemahkan dengan lancar 80% dari waktu dan secara diam-diam gagal dalam 20% sisanya. Mode kegagalan yang diberi nama:

- **Halusinasi.** Model menciptakan konten yang tidak ada dalam sumbernya. Umum dalam kosakata domain asing. Gejala: output lancar tetapi mengklaim fakta yang tidak disebutkan oleh sumber. Mitigasi: penguraian code terbatas pada istilah domain, tinjauan manusia terhadap konten yang diatur, pemantauan output lebih lama daripada input.
- **Pembuatan di luar target.** Model diterjemahkan ke dalam bahasa yang salah. NLLB secara mengejutkan rentan terhadap hal ini pada pasangan bahasa yang langka. Mitigasi: verifikasi `forced_bos_token_id` dan selalu dekode dengan pemeriksaan model ID bahasa pada output.
- **Penyimpangan terminologi.** "Daftar" menjadi "s'inscrire" di dokumen 1 dan "créer un compte" di dokumen 2. Untuk teks UI dan string yang dapat dilihat pengguna, konsistensi lebih penting daripada kualitas mentah. Mitigasi: decoding yang dibatasi glosarium atau kamus pasca-edit.
- **Ketidakcocokan formalitas.** Bahasa Prancis "tu" vs "vous", tingkat kesopanan Jepang. Model memilih bentuk mana pun yang lebih umum dalam training. Untuk konten yang berhubungan dengan pelanggan, hal ini biasanya salah. Mitigasi: awalan cepat dengan token formalitas jika model mendukungnya, atau menyempurnakan model kecil pada corpora formal saja.
- **Ledakan panjang pada input pendek.** Kalimat input yang sangat pendek sering kali menghasilkan terjemahan yang terlalu panjang karena hukuman panjangnya jauh di bawah ~5 token sumber. Mitigasi: tutup keras dengan panjang maksimal sebanding dengan panjang sumber.

### Langkah 4: menyempurnakan domain

Model yang telah dilatih sebelumnya adalah model yang bersifat generalis. Terjemahan hukum, medis, atau dialog permainan mendapat manfaat yang signifikan dari penyempurnaan data paralel domain. Resepnya tidak eksotik:

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

Beberapa ribu contoh paralel berkualitas tinggi mengalahkan beberapa ratus ribu contoh web yang berisik. Kualitas training data adalah pendorong produksi terbesar.

## Pakai

Tumpukan produksi tahun 2026 untuk MT:| Kasus penggunaan | Titik awal yang disarankan |
|---------|---------------------------|
| Apa saja, apa saja, 200 bahasa | `facebook/nllb-200-distilled-600M` (laptop) atau `nllb-200-3.3B` (produksi) |
| Berpusat pada bahasa Inggris, berkualitas tinggi, 50 bahasa | `facebook/mbart-large-50-many-to-many-mmt` |
| Jangka pendek, inference murah, Inggris-Prancis/Jerman/Spanyol | Model Helsinki-NLP / Marian |
| Sisi browser yang kritis terhadap latensi | Marian terkuantisasi ONNX (~50 MB) |
| Kualitas maksimal, bersedia bayar | GPT-4 / Claude / Gemini dengan petunjuk terjemahan |

LLM sekarang mengungguli model MT khusus pada beberapa pasangan bahasa pada tahun 2026, khususnya pada konten idiomatik dan konteks panjang. Pengorbanannya adalah biaya per token dan latensi. Pilih LLM ketika panjang konteks, konsistensi gaya, atau adaptasi domain melalui dorongan lebih penting daripada throughput.

## Kirim

Simpan sebagai `outputs/skill-mt-evaluator.md`:

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## Latihan

1. **Mudah.** Terjemahkan paragraf 5 kalimat bahasa Inggris ke bahasa Prancis dan kembali ke bahasa Inggris menggunakan `nllb-200-distilled-600M`. Ukur seberapa dekat perjalanan pulang pergi dengan aslinya. kamu akan melihat pelestarian semantik dengan penyimpangan pilihan kata.
2. **Medium.** Terapkan pemeriksaan ID bahasa pada output terjemahan menggunakan `fasttext lid.176` atau `langdetect`. Integrasikan ke dalam panggilan MT sehingga generasi yang tidak sesuai target dapat ditangkap sebelum kembali lagi.
3. **Sulit.** Sempurnakan `nllb-200-distilled-600M` pada korpus domain 5.000 pasang pilihan kamu. Ukur BLEU pada set yang ditahan sebelum dan sesudah fine-tuning. Laporkan jenis kalimat mana yang ditingkatkan dan mana yang mengalami kemunduran.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| BIRU | Skor terjemahan | Presisi N-gram dengan penalti singkat. [0, 100]. |
| chrF | Skor F karakter | Skor F tingkat karakter. Lebih sensitif untuk bahasa yang kaya secara morfologi. |
| NMT | MT saraf | Encoder-decoder Transformer dilatih pada teks paralel. Standar 2017+. |
| NLLB | Tidak Ada Bahasa Tertinggal | Keluarga model MT 200 bahasa Meta. |
| Penguraian code terbatas | Output terkendali | Memaksa token atau n-gram tertentu untuk muncul/tidak muncul di output. |
| Halusinasi | Konten yang diciptakan | Output model yang tidak didukung oleh sumbernya. |

## Bacaan Lanjutan

- [Costa-jussà dkk. (2022). Tidak Ada Bahasa yang Tertinggal: Menskalakan Terjemahan Mesin yang Berpusat pada Manusia](https://arxiv.org/abs/2207.04672) — makalah NLLB.
- [Posting (2018). Seruan untuk Kejelasan dalam Pelaporan Skor BLEU](https://aclanthology.org/W18-6319/) — mengapa `sacrebleu` adalah satu-satunya cara yang benar untuk melaporkan BLEU.
- [Popović (2015). chrF: karakter n-gram F-score untuk evaluasi MT otomatis](https://aclanthology.org/W15-3049/) — makalah chrF.
- [Panduan Hugging Face MT](https://huggingface.co/docs/transformers/tasks/translation) — panduan praktis penyesuaian.
