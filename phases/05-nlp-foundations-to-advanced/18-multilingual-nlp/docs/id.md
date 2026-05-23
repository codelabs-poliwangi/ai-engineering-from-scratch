# NLP multibahasa

> Satu model, 100+ bahasa, tidak ada training data untuk sebagian besar bahasa. Transfer lintas bahasa adalah keajaiban praktis di tahun 2020-an.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 5 · 04 (GloVe, FastText, Subword), Fase 5 · 11 (Terjemahan Mesin)
**Waktu:** ~45 menit

## Masalah

Bahasa Inggris memiliki miliaran contoh berlabel. Bahasa Urdu memiliki ribuan. Maithili hampir tidak memilikinya. Sistem NLP praktis apa pun yang melayani audiens global harus bekerja pada bahasa-bahasa yang tidak memiliki training data khusus tugas.

Model multibahasa mengatasi masalah ini dengan melatih satu model dalam banyak bahasa secara bersamaan. Representasi bersama memungkinkan model mentransfer keterampilan yang dipelajari dalam bahasa dengan sumber daya tinggi ke bahasa dengan sumber daya rendah. Sempurnakan model analisis sentimen bahasa Inggris, dan model ini menghasilkan prediksi sentimen yang sangat bagus dalam bahasa Urdu. Itu adalah transfer lintas bahasa yang zero-shot, dan ini telah mengubah cara NLP dikirimkan ke seluruh dunia.

Lesson ini menyebutkan tradeoff, model kanonik, dan satu keputusan yang membuat tim baru dalam pekerjaan multibahasa: memilih bahasa sumber untuk transfer.

## Konsep

![Transfer lintas bahasa melalui ruang embedding multibahasa bersama](../assets/multilingual.svg)

**Kosakata bersama.** Model multibahasa menggunakan tokenizer SentencePiece atau WordPiece yang dilatih pada teks dari semua bahasa target. Kosakata digunakan bersama: unit subkata yang sama mewakili morfem yang sama dalam bahasa terkait. `anti-` dalam bahasa Inggris dan Italia mendapatkan token yang sama.

**Representasi bersama.** Sebuah Transformer yang telah dilatih sebelumnya tentang pemodelan bahasa bertopeng di banyak bahasa akan mempelajari bahwa kalimat yang serupa secara semantik dalam bahasa berbeda menghasilkan keadaan tersembunyi yang serupa. mBERT, XLM-R, dan NLLB semuanya menunjukkan hal ini. Embedding untuk "cat" dalam bahasa Inggris dikelompokkan di dekat "chat" dalam bahasa Prancis dan "gato" dalam bahasa Spanyol, begitu pula embedding kalimat lengkap.

**Transfer zero-shot.** Menyempurnakan model pada data berlabel dalam satu bahasa (biasanya bahasa Inggris). Sebagai kesimpulan, jalankan model tersebut dalam bahasa lain yang didukung model. Tidak diperlukan label bahasa target. Hasil yang diperoleh kuat untuk bahasa-bahasa yang tipologisnya berkaitan dan lemah untuk bahasa-bahasa yang berjauhan.

**Penyempurnaan beberapa langkah.** Tambahkan 100-500 contoh berlabel dalam bahasa target. Akurasi melonjak hingga 95-98% dari dasar bahasa Inggris pada tugas klasifikasi. Ini adalah satu-satunya tuas yang paling hemat biaya dalam NLP multibahasa.

## Modelnya

| Model | Tahun | Cakupan | Catatan |
|-------|------|----------|-------|
| mBERT | 2018 | 104 bahasa | Dilatih di Wikipedia. LM multibahasa praktis pertama. Lemah pada sumber daya yang rendah. |
| XLM-R | 2019 | 100 bahasa | Dilatih di CommonCrawl (jauh lebih besar dari Wikipedia). Menetapkan garis dasar lintas bahasa. Basis 270M, Besar 550M. |
| XLM-V | 2023 | 100 bahasa | XLM-R dengan kosakata token 1 juta (vs 250k). Lebih baik dengan sumber daya rendah. |
| mT5 | 2020 | 101 bahasa | Arsitektur T5 untuk generasi multibahasa. |
| NLLB-200 | 2022 | 200 bahasa | model terjemahan Meta; mencakup 55 bahasa sumber daya rendah. |
| MEKAR | 2022 | 46 bahasa + 13 pemrograman | Buka 176B LLM dilatih multibahasa. |
| Ayat-23 | 2024 | 23 bahasa | LLM multibahasa Cohere. Kuat dalam bahasa Arab, Hindi, Swahili. |

Pilih berdasarkan kasus penggunaan. Klasifikasi berfungsi baik dengan basis XLM-R sebagai default yang masuk akal. Tugas pembuatan memerlukan mT5 atau NLLB bergantung pada terjemahan vs pembuatan terbuka. Pasangan kerja gaya LLM dengan Aya-23 atau Claude menggunakan prompt multibahasa yang eksplisit.## Keputusan bahasa sumber (penelitian tahun 2026)

Sebagian besar tim menggunakan bahasa Inggris sebagai sumber penyesuaian. Penelitian terbaru (2026) menunjukkan bahwa hal ini seringkali salah.

Kesamaan bahasa memprediksi kualitas transfer lebih baik daripada ukuran korpus mentah. Untuk target Slavia, bahasa Jerman atau Rusia sering mengalahkan bahasa Inggris. Untuk target India, bahasa Hindi seringkali mengalahkan bahasa Inggris. Metrik kesamaan **qWALS** (2026, berdasarkan feature World Atlas of Language Structures) mengukur hal ini. **LANGRANK** (Lin et al., ACL 2019) adalah metode terpisah dan terdahulu yang mengurutkan kandidat bahasa sumber berdasarkan kombinasi kemiripan linguistik, ukuran korpus, dan keterkaitan genetik.

Aturan praktisnya: jika bahasa target kamu memiliki tipologi yang mirip dengan sumber daya tinggi, cobalah menyempurnakan bahasa tersebut terlebih dahulu, lalu bandingkan dengan bahasa Inggris fine-tune.

## Build

### Langkah 1: klasifikasi lintas bahasa zero-shot

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

Satu model, tiga bahasa, API yang sama. XLM-R dilatih tentang transfer data NLI dengan baik ke klasifikasi melalui trik keterlibatan.

### Langkah 2: embedding ruang multibahasa

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

Terjemahan mendarat dekat di ruang embedding. Kalimat bahasa Inggris yang berbeda muncul lebih jauh. Inilah yang membuat pengambilan, pengelompokan, dan kesamaan lintas bahasa berhasil.

### Langkah 3: strategi penyempurnaan beberapa langkah

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

Untuk 100-500 contoh bahasa target, `num_train_epochs=5` dan `learning_rate=2e-5` adalah default yang aman. Learning rate yang lebih tinggi menyebabkan penyelarasan multibahasa gagal dan kamu mendapatkan model hanya dalam bahasa Inggris.

## Evaluasi yang benar-benar berhasil

- **Akurasi per bahasa pada set yang ditunda.** Tidak diagregasi. Agregat menyembunyikan ekor panjang.
- **Tolok ukur terhadap dasar monolingual.** Untuk bahasa dengan data yang cukup, model monolingual yang dilatih dari awal terkadang mengalahkan model multibahasa. Tes.
- **Tes tingkat entitas.** Entitas yang diberi nama dalam bahasa target. Model multibahasa sering kali memiliki tokenization yang lemah untuk skrip yang jauh dari bahasa Latin.
- **Konsistensi lintas bahasa.** Arti yang sama dalam dua bahasa seharusnya menghasilkan prediksi yang sama. Ukur kesenjangannya.

## Pakai

Tumpukan tahun 2026:

| Tugas | Direkomendasikan |
|-----|-------------|
| Klasifikasi, 100 bahasa | Basis XLM-R (~270M) disetel dengan baik |
| Klasifikasi teks zero-shot | `joeddav/xlm-roberta-large-xnli` |
| Embedding kalimat multibahasa | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Terjemahan, 200 bahasa | `facebook/nllb-200-distilled-600M` (lihat lesson 11) |
| Multibahasa generatif | Claude, GPT-4, Aya-23, mT5-XXL |
| NLP bahasa sumber daya rendah | XLM-V atau penyempurnaan khusus domain pada bahasa sumber daya tinggi terkait |

Selalu anggarkan anggaran untuk menyempurnakan bahasa target jika kinerja penting. Zero-shot adalah titik awal, bukan jawaban akhir.

### Pajak tokenization (apa yang salah dengan bahasa dengan sumber daya rendah)

Model multibahasa berbagi satu tokenizer di semua bahasanya. Kosakata tersebut dilatih pada korpus yang didominasi oleh bahasa Inggris, Perancis, Spanyol, Cina, Jerman. Untuk bahasa apa pun di luar kelompok dominan, ada tiga pajak yang digabungkan secara diam-diam:- **Pajak kesuburan.** Teks bahasa dengan sumber daya rendah ditoken menjadi lebih banyak token per kata dibandingkan bahasa Inggris. Kalimat bahasa Hindi memerlukan 3-5x token kalimat bahasa Inggris yang setara. Itu 3-5x memakan jendela konteks, efisiensi training, dan latensi kamu.
- **Pajak pemulihan varian.** Setiap kesalahan ketik, varian diakritik, ketidakcocokan normalisasi Unicode, atau variasi huruf menjadi urutan cold-start yang tidak terkait dalam ruang embedding. Model tidak dapat mempelajari korespondensi ortografis yang dianggap jelas oleh penutur asli.
- **Pajak limpahan kapasitas.** Pajak 1 dan 2 menggunakan posisi konteks, kedalaman layer, dan dimension embedding. Apa yang tersisa untuk penalaran sebenarnya secara sistematis lebih kecil dibandingkan dengan apa yang diperoleh bahasa dengan sumber daya tinggi dari model yang sama.

Gejala praktisnya: model kamu berlatih secara normal dalam bahasa Hindi, kurva loss terlihat benar, perplexity evaluasi terlihat masuk akal, dan output produksi sedikit salah. Morfologi runtuh di tengah kalimat. Perubahan yang jarang terjadi tetap tidak dapat dipulihkan. **kamu tidak dapat menskalakan data untuk keluar dari tokenizer yang rusak.**

Mitigasi: pilih tokenizer dengan cakupan yang baik untuk bahasa target kamu (kosa kata 1 juta token XLM-V adalah perbaikan langsung); memverifikasi kesuburan tokenization pada teks target yang disimpan sebelum training; gunakan fallback tingkat byte (SentencePiece `byte_fallback=True`, BPE tingkat byte gaya GPT-2) untuk skrip berekor panjang sehingga tidak ada yang OOV.

## Kirim

Simpan sebagai `outputs/skill-multilingual-picker.md`:

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## Latihan

1. **Mudah.** Jalankan alur klasifikasi zero-shot pada 10 kalimat per bahasa dalam bahasa Inggris, Prancis, Hindi, dan Arab. Laporkan keakuratan masing-masing. kamu akan melihat bahasa Prancis yang kuat, bahasa Hindi yang baik, dan bahasa Arab yang bervariasi.
2. **Medium.** Gunakan `paraphrase-multilingual-MiniLM-L12-v2` untuk membuat retriever lintas bahasa pada korpus kecil bahasa campuran. Kueri dalam bahasa Inggris, ambil dokumen dalam bahasa apa pun. Ukur penarikan @5.
3. **Sulit.** Bandingkan penyesuaian sumber bahasa Inggris dan sumber bahasa Hindi untuk tugas klasifikasi bahasa Hindi. Gunakan 500 contoh bahasa target untuk beberapa penyesuaian pada kedua rezim. Laporkan sumber mana yang menghasilkan akurasi bahasa Hindi yang lebih baik dan seberapa banyak. Ini adalah tesis LANGRANK dalam bentuk mini.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Model multibahasa | Satu model, banyak bahasa | Kosakata dan parameter bersama lintas bahasa. |
| Transfer lintas bahasa | Berlatih dalam satu bahasa, jalankan dalam bahasa lain | Sempurnakan sumbernya, evaluasi sesuai target tanpa label bahasa target. |
| Tembakan nol | Tidak ada label bahasa target | Transfer tanpa menyempurnakan bahasa target. |
| Sedikit tembakan | Label target kecil | 100-500 contoh bahasa target yang digunakan untuk penyesuaian. |
| mBERT | LM multibahasa pertama | BERT 104 bahasa dilatih sebelumnya di Wikipedia. |
| XLM-R | Garis dasar lintas bahasa standar | RoBERTa 100 bahasa telah dilatih sebelumnya di CommonCrawl. |
| NLLB | MT 200 bahasa Meta | Tidak Ada Bahasa yang Tertinggal. Termasuk 55 bahasa sumber daya rendah. |

## Bacaan Lanjutan- [Conneau dkk. (2019). Pembelajaran Representasi Lintas Bahasa dalam Skala Besar Tanpa Pengawasan](https://arxiv.org/abs/1911.02116) — makalah XLM-R.
- [Pires, Schlinger, Garrette (2019). Seberapa Multibahasa BERT Multibahasa?](https://arxiv.org/abs/1906.01502) — makalah analisis yang memulai jalur penelitian transfer lintas bahasa.
- [Costa-jussà dkk. (2022). Tidak Ada Bahasa yang Tertinggal](https://arxiv.org/abs/2207.04672) — Makalah NLLB-200.
- [Üstün dkk. (2024). Model Aya: Model Bahasa Multibahasa Akses Terbuka yang Disempurnakan dengan Instruksi](https://arxiv.org/abs/2402.07827) — Aya, LLM multibahasa Cohere.
- [Kesamaan Bahasa Memprediksi Kinerja Pembelajaran Transfer Bahasa (2026)](https://www.mdpi.com/2504-4990/8/3/65) — makalah bahasa sumber qWALS / LANGRANK.
