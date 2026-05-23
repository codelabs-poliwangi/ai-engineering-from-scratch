# Pengakuan Entitas Bernama

> Tarik keluar namanya. Kedengarannya mudah sampai kamu menangani batasan yang ambigu, entitas bersarang, dan jargon domain.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 5 · 02 (BoW + TF-IDF), Fase 5 · 03 (Word Embeddings)
**Waktu:** ~75 menit

## Masalah

"Apple menggugat Google atas kesepakatan pencarian iPhone di AS." Lima entitas: Apple (ORG), Google (ORG), iPhone (PRODUK), kesepakatan pencarian (mungkin), AS (GPE). Sistem NER yang baik mengekstrak semuanya dengan tipe yang benar. Yang buruk merindukan iPhone, mengacaukan Apple sebagai buah dengan Apple sebagai perusahaannya, dan memberi label "KAMI" sebagai ORANG.

NER adalah pekerja keras di bawah setiap jalur ekstraksi terstruktur. Penguraian resume, pemindaian log kepatuhan, anonimisasi rekam medis, pemahaman kueri penelusuran, landasan respons chatbot, ekstraksi kontrak hukum. kamu tidak pernah melihatnya; kamu selalu bergantung padanya.

Lesson ini mengarahkan jalur klasik (berbasis aturan, HMM, CRF) ke jalur modern (BiLSTM-CRF, lalu Transformer). Setiap langkah memecahkan batasan spesifik dari langkah sebelumnya. Polanya adalah pelajarannya.

## Konsep

**Penandaan BIO** (atau BILOU) mengubah ekstraksi entitas menjadi masalah pelabelan urutan. Labeli setiap token dengan `B-TYPE` (awal entitas), `I-TYPE` (di dalam entitas), atau `O` (di luar entitas mana pun).

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

Rantai entitas multi-token: `New B-GPE`, `York I-GPE`, `City I-GPE`. Model yang memahami BIO dapat mengekstraksi rentang yang berubah-ubah.

Perkembangan arsitektur:

- **Berbasis aturan.** Pencarian regex + gazetteer. Presisi tinggi pada entitas yang dikenal, tidak ada cakupan pada entitas baru.
- **HMM.** Model Markov Tersembunyi. Probabilitas emisi dari tag yang diberikan token, probabilitas transisi dari tag-ke-tag. Dekode Viterbi. Dilatih pada data berlabel.
- **CRF.** Bidang Acak Bersyarat. Seperti HMM tetapi diskriminatif, sehingga kamu dapat mencampurkan feature-feature yang sewenang-wenang (bentuk kata, kapitalisasi, kata-kata yang bertetangga). Masih menjadi pekerja keras produksi klasik pada tahun 2026 untuk penerapan sumber daya rendah.
- **BiLSTM-CRF.** Feature saraf, bukan buatan tangan. LSTM membaca kalimat dua arah, layer CRF di atas menerapkan urutan tag yang konsisten.
- **Berbasis Transformer.** Sempurnakan BERT dengan kepala klasifikasi token. Akurasi terbaik. Kebanyakan menghitung.

## Build

### Langkah 1: Pembantu penandaan BIO

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### Langkah 2: feature kerajinan tangan

Untuk NER klasik (non-neural), feature adalah permainannya. Yang berguna:

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` mengembalikan `xXxxxx`. `word_shape("USA-2024")` mengembalikan `XXX-dddd`. Pola kapitalisasi merupakan sinyal penting untuk kata benda yang tepat.

### Langkah 3: dasar aturan + kamus sederhana

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

Lembaran produksi memiliki jutaan entri yang diambil dari Wikipedia dan DBpedia. Cakupannya bagus. Disambiguasi (`Apple` perusahaan vs buahnya) sangat buruk. Itulah sebabnya model statistik menang.

### Langkah 4: langkah CRF (sketsa, bukan impl penuh)

CRF lengkap dari awal dalam 50 baris tidak akan mencerahkan tanpa landasan teori probabilitas. Gunakan `sklearn-crfsuite` sebagai gantinya:

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` dan `c2` adalah regularisasi L1 dan L2. `all_possible_transitions=True` memungkinkan model mempelajari urutan ilegal (misalnya, `I-ORG` setelah `O`) kecil kemungkinannya, begitulah cara CRF menerapkan konsistensi BIO tanpa kamu menulis batasannya.### Langkah 5: apa yang ditambahkan BiLSTM-CRF

Feature menjadi dipelajari. Input: embedding token (GloVe atau fastText). LSTM membaca dari kiri ke kanan dan kanan ke kiri. Status tersembunyi yang digabungkan melewati layer output CRF. CRF masih menerapkan konsistensi urutan tag; LSTM menggantikan feature buatan tangan dengan feature yang dipelajari.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

Untuk layer CRF, gunakan `torchcrf.CRF` (pip install pytorch-crf). Keuntungan dibandingkan CRF buatan tangan dapat diukur tetapi lebih kecil dari yang kamu harapkan kecuali kamu memiliki puluhan ribu kalimat berlabel.

## Pakai

spaCy mengirimkan NER tingkat produksi secara langsung.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

Perhatikan `iPhone` berlabel `ORG` daripada `PRODUCT` — model kecil spaCy memiliki cakupan entitas produk yang lemah. Model besar (`en_core_web_lg`) bekerja lebih baik. Model trafo (`en_core_web_trf`) masih lebih baik lagi.

Memeluk Wajah untuk NER berbasis BERT:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` menggabungkan token BX dan IX yang berdekatan ke dalam suatu rentang. Tanpanya, kamu mendapatkan label tingkat token dan harus menggabungkan diri kamu sendiri.

### NER berbasis LLM (opsi 2026)

LLM NER zero-shot dan some-shot kini mampu bersaing dengan model yang telah disesuaikan di banyak domain, dan jauh lebih baik ketika data berlabel langka.

- **Permintaan zero-shot.** Berikan LLM daftar jenis entitas dan contoh skema. Minta output JSON. Bekerja di luar kotak; akurasinya moderat pada domain baru.
- **Permintaan gaya ZeroTuneBio.** Uraikan tugas menjadi ekstraksi kandidat → penjelasan makna → penilaian → periksa ulang. Prompt multi-phase (bukan satu kali) meningkatkan akurasi secara signifikan pada NER biomedis. Pola yang sama berlaku untuk bidang hukum, keuangan, dan ilmiah.
- **Permintaan dinamis dengan RAG.** Ambil contoh berlabel yang paling mirip dari kumpulan benih kecil yang diberi anotasi untuk setiap panggilan inference; buat prompt beberapa langkah dengan cepat. Pada tolok ukur tahun 2026, hal ini meningkatkan NER F1 biomedis GPT-4 sebesar 11-12% dibandingkan dorongan statis.
- **Decomposition tipe per-entitas.** Untuk dokumen panjang, satu panggilan yang mengekstrak semua tipe entitas sekaligus akan kehilangan ingatan seiring bertambahnya panjang. Jalankan satu jalur ekstraksi per jenis entitas. Biaya inference lebih tinggi, akurasi jauh lebih tinggi. Ini adalah pola standar untuk catatan klinis dan kontrak hukum.

Rekomendasi produksi pada tahun 2026: mulailah dengan garis dasar zero-shot LLM sebelum kamu mengumpulkan training data. Seringkali F1 sudah cukup bagus sehingga kamu tidak perlu menyempurnakannya.

### Dimana NER klasik masih menang

Bahkan dengan LLM yang tersedia, NER klasik menang ketika:

- Anggaran latensi di bawah 50 ms.
- kamu memiliki ribuan contoh berlabel dan membutuhkan 98%+ F1.
- Domain memiliki ontologi yang stabil di mana CRF atau BiLSTM yang telah dilatih sebelumnya dapat ditransfer dengan baik.
- Kendala peraturan memerlukan model non-generatif di lokasi.

### Dimana berantakan

- **Pergeseran domain.** Performa NER yang dilatih CoNLL dalam kontrak hukum lebih buruk daripada Gazetteer. Sempurnakan domain kamu.
- **Entitas bersarang.** "Bank of America Tower" sekaligus merupakan ORG dan FASILITAS. BIO standar tidak dapat mewakili rentang yang tumpang tindih. kamu memerlukan NER bersarang (model multi-pass atau berbasis rentang).
- **Entitas panjang.** "Perusahaan Penjamin Simpanan Federal Amerika Serikat". Model tingkat token terkadang memisahkan hal ini. Gunakan `aggregation_strategy` atau pasca-proses.
- **Jenis jarang.** Label NER medis seperti DRUG_BRAND, ADVERSE_EVENT, DOSE. Model tujuan umum tidak tahu. Scispacy dan BioBERT adalah titik awal di sana.## Kirim

Simpan sebagai `outputs/skill-ner-picker.md`:

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## Latihan

1. **Mudah.** Terapkan `bio_to_spans` (kebalikan dari `spans_to_bio`) dan verifikasi konsistensi bolak-balik pada 10 kalimat.
2. **Sedang.** Latih CRF sklearn-crfsuite di atas pada dataset NER Bahasa Inggris CoNLL-2003. Laporkan per entitas F1 menggunakan `seqeval`. Hasil umum: ~84 F1.
3. **Sulit.** Menyempurnakan `distilbert-base-cased` pada dataset NER khusus domain (medis, hukum, atau keuangan). Bandingkan dengan model kecil spaCy. Dokumentasikan pemeriksaan kebocoran data dan tulis apa yang mengejutkan kamu.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| AMAN | Ekstrak nama | Label token mencakup tipe (PERSON, ORG, GPE, DATE, ...). |
| BIO | Skema penandaan | `B-X` dimulai, `I-X` berlanjut, `O` di luar. |
| BILOU | BIO yang lebih baik | Tambahkan `L-X` (terakhir), `U-X` (unit) untuk batas yang lebih bersih. |
| CRF | Pengklasifikasi terstruktur | Model transisi antar label, bukan hanya emisi. Menerapkan urutan yang valid. |
| NER Bersarang | Entitas yang tumpang tindih | Satu rentang adalah entitas yang berbeda dari sub-rentangnya. BIO tidak bisa mengungkapkan hal ini. |
| F1 tingkat entitas | Metrik NER yang tepat | Rentang yang diprediksi harus sama persis dengan rentang sebenarnya. F1 tingkat token melebih-lebihkan akurasi. |

## Bacaan Lanjutan

- [Lample dkk. (2016). Arsitektur Neural untuk Pengenalan Entitas Bernama](https://arxiv.org/abs/1603.01360) — makalah BiLSTM-CRF. Resmi.
- [Devlin dkk. (2018). BERT: Pra-training Transformer Dua Arah Dalam](https://arxiv.org/abs/1810.04805) — memperkenalkan pola klasifikasi token yang menjadi standar.
- [feature linguistik spaCy — entitas bernama](https://spacy.io/usage/linguistic-features#named-entities) — referensi praktis untuk setiap atribut di `Doc.ents` dan `Span`.
- [seqeval](https://github.com/chakki-works/seqeval) — pustaka metrik yang benar. Gunakan selalu.
