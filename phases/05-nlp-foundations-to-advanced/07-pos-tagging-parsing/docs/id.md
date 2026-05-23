# Penandaan POS dan Parsing Sintaksis

> Tata bahasa sudah ketinggalan zaman untuk sementara waktu. Kemudian setiap jalur pipa LLM perlu memvalidasi ekstraksi terstruktur, dan jalur tersebut kembali lagi.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 5 · 01 (Pemrosesan Teks), Phase 2 · 14 (Naive Bayes)
**Waktu:** ~45 menit

## Masalah

Lesson 01 berjanji bahwa lemmatisasi memerlukan tag part-of-speech. Tanpa mengetahui `running` adalah kata kerja, lemmatizer tidak dapat mereduksinya menjadi `run`. Tanpa mengetahui `better` adalah kata sifat, maka tidak dapat disingkat menjadi `good`.

Janji itu menyembunyikan keseluruhan subbidang. Penandaan part-of-speech memberikan kategori tata bahasa. Penguraian sintaksis memulihkan struktur pohon kalimat: kata mana yang mengubah kata mana, kata kerja mana yang mengatur argumen mana. NLP klasik menghabiskan dua puluh tahun untuk menyempurnakan keduanya. Kemudian pembelajaran mendalam memecahnya menjadi tugas klasifikasi token di atas Transformer yang telah dilatih sebelumnya, dan komunitas riset melanjutkan.

Bukan komunitas terapan. Setiap pipeline ekstraksi terstruktur masih menggunakan POS dan pohon ketergantungan. JSON yang dihasilkan LLM divalidasi berdasarkan batasan tata bahasa. Sistem penjawab pertanyaan menguraikan kueri menggunakan penguraian ketergantungan. Penilai kualitas terjemahan mesin memeriksa keselarasan pohon parse.

Perlu diketahui. Lesson ini memperkenalkan kumpulan tag, garis dasar, dan titik di mana kamu berhenti menerapkan dari awal dan memanggil spaCy.

## Konsep

**Penandaan POS** memberi label pada setiap token dengan kategori tata bahasa. Taget **Penn Treebank (PTB)** adalah default bahasa Inggris. 36 tag dengan perbedaan yang dianggap rewel oleh pembaca biasa: `NN` kata benda tunggal, `NNS` kata benda jamak, `NNP` kata benda yang tepat tunggal, `VBD` kata kerja bentuk lampau, `VBZ` kata kerja orang ketiga tunggal saat ini, dan seterusnya. Taget **Universal Dependencies (UD)** lebih kasar (17 tag) dan tidak bergantung pada bahasa; ini menjadi standar untuk pekerjaan lintas bahasa.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**Penguraian sintaksis** menghasilkan pohon. Dua gaya utama:

- **Penguraian konstituensi.** Frasa kata benda, frasa kata kerja, dan frasa preposisi bersarang di dalam satu sama lain. Outputnya berupa pohon kategori non terminal (NP, VP, PP) dengan kata-kata sebagai daunnya.
- **Penguraian ketergantungan.** Setiap kata memiliki satu kata utama yang bergantung padanya, diberi label dengan hubungan tata bahasa. Outputnya adalah sebuah pohon yang setiap sisinya merupakan tripel (kepala, dependen, relasi).

Penguraian ketergantungan menang pada tahun 2010-an karena dapat digeneralisasi dengan rapi di seluruh bahasa, terutama bahasa dengan susunan kata bebas.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## Build

### Langkah 1: garis dasar tag yang paling sering

Pemberi tag POS paling bodoh yang berfungsi. Untuk setiap kata, prediksi tag yang paling sering dimilikinya dalam training.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

Pada korpus Brown, garis dasar ini mencapai akurasi ~85%. Tidak bagus, tapi lantai di bawahnya tidak ada model serius yang jatuh.

### Langkah 2: penanda bigram HMM

Modelkan probabilitas gabungan dari barisan tersebut:

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

Dua tabel: probabilitas transisi (tag diberikan tag sebelumnya), probabilitas emisi (tag diberikan kata). Perkirakan keduanya dari hitungan dengan pemulusan Laplace. Dekode dengan Viterbi (pemrograman dinamis melalui kisi tag).

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Bigram HMM di Brown mencapai akurasi ~93%. Lonjakan dari 85% ke 93% sebagian besar merupakan probabilitas transisi — model pembelajaran `DET NOUN` adalah hal yang umum dan `NOUN DET` jarang terjadi.

### Langkah 3: mengapa pemberi tag modern mengalahkan iniProbabilitas transisi + emisi bersifat lokal. Mereka tidak dapat memahami bahwa `saw` adalah kata benda dalam "Saya membeli gergaji" tetapi merupakan kata kerja dalam "Saya melihat filmnya". CRF dengan feature arbitrer (akhiran, bentuk kata, kata sebelum dan sesudah, kata itu sendiri) mencapai ~97%. BiLSTM-CRF atau Transformer mencapai ~98%+.

Batas atas tugas ini ditentukan oleh ketidaksepakatan anotator. Anotator manusia 97% setuju dengan Penn Treebank. Model yang melebihi 98% mungkin melakukan overfitting pada set pengujian.

### Langkah 4: sketsa penguraian ketergantungan

Penguraian ketergantungan penuh dari awal berada di luar cakupan; perlakuan buku teks kanonik ada di Jurafsky dan Martin. Dua keluarga klasik yang perlu diketahui:

- Pengurai **berbasis transisi** (arc-eager, arc-standard) bertindak seperti pengurai pengurangan shift: membaca token, memindahkannya ke tumpukan, dan menerapkan tindakan pengurangan yang menghasilkan busur. Penguraian code serakah itu cepat. Implementasi klasiknya adalah MaltParser. Versi saraf modern: parser berbasis transisi Chen dan Manning.
- Pengurai **berbasis grafik** (algoritme Eisner, biaffine Dozat-Manning) menilai setiap kemungkinan tepi yang bergantung pada kepala dan memilih pohon rentang maksimum. Lebih lambat namun lebih akurat.

Untuk sebagian besar pekerjaan yang diterapkan, hubungi spaCy:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

Baca kolom `dep` dari bawah ke atas dan struktur tata bahasa kalimatnya akan hilang.

## Pakai

Setiap perpustakaan NLP produksi mengirimkan POS dan parser ketergantungan sebagai bagian dari pipeline standar.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). Cepat, akurat, terintegrasi dengan tokenization + NER + lemmatisasi. `token.tag_` (Penn), `token.pos_` (UD), `token.dep_` (hubungan ketergantungan).
- **Stanford NLP (bait)**. Penerus Stanford untuk CoreNLP. Tercanggih dalam 60+ bahasa.
- **trankit**. Berbasis Transformer, akurasi UD bagus.
- **NLTK**. `pos_tag`. Dapat digunakan, lambat, lebih tua. Baik untuk mengajar.

### Dimana hal ini masih penting pada tahun 2026

- **Lematisasi.** Lesson 01 memerlukan POS untuk melakukan lemmatisasi dengan benar. Selalu.
- **Ekstraksi terstruktur dari output LLM.** Validasi bahwa kalimat yang dihasilkan mematuhi batasan tata bahasa (misalnya, kesepakatan subjek-kata kerja, pengubah yang diperlukan).
- **Sentimen berbasis aspek.** Penguraian ketergantungan memberi tahu kamu kata sifat mana yang mengubah kata benda mana.
- **Pemahaman kueri.** "film yang disutradarai oleh Wes Anderson yang dibintangi oleh Bill Murray" diuraikan menjadi batasan terstruktur melalui penguraian.
- **Transfer lintas bahasa.** Tag UD dan hubungan ketergantungan tidak bergantung pada bahasa, sehingga memungkinkan analisis terstruktur zero-shot pada bahasa baru.
- **Pipa komputasi rendah.** Jika kamu tidak dapat mengirimkan trafo, POS + penguraian ketergantungan + gazetteer akan membawa kamu jauh lebih baik.

## Kirim

Simpan sebagai `outputs/skill-grammar-pipeline.md`:

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## Latihan

1. **Mudah.** Dengan menggunakan garis dasar tag yang paling sering pada korpus kecil yang diberi tag (misalnya, subset Brown NLTK), ukur akurasi pada kalimat yang diluruskan. Verifikasi hasil ~85%.
2. **Medium.** Latih bigram HMM di atas dan laporkan presisi/penarikan per tag. Tag mana yang paling membingungkan HMM?
3. **Sulit.** Gunakan penguraian ketergantungan spaCy untuk mengekstrak tiga kali lipat subjek-kata kerja-objek dari sample 1000 kalimat. Evaluasi pada 50 tripel yang diberi label secara manual. Dokumentasikan saat ekstraksi gagal (sering kali pasif, koordinasi, dan subjek yang dihilangkan).

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Label POS | Tipe kata | Kategori tata bahasa. PTB memiliki 36; UD memiliki 17. |
| Bank Pohon Penn | Kumpulan tag standar | Khusus bahasa Inggris. Bentuk kata kerja terperinci dan nomor kata benda. |
| Ketergantungan Universal | Kumpulan tag multibahasa | Lebih kasar dari PTB; netral bahasa; default untuk pekerjaan lintas bahasa. |
| Penguraian ketergantungan | Pohon kalimat | Setiap kata memiliki satu kepala, setiap sisi memiliki hubungan tata bahasa. |
| Viterbi | Pemrograman dinamis | Menemukan urutan tag dengan probabilitas tertinggi berdasarkan emisi dan transisi. |

## Bacaan Lanjutan

- [Jurafsky dan Martin — Pemrosesan Pidato dan Bahasa, bab 8 dan 18](https://web.stanford.edu/~jurafsky/slp3/) — pembahasan buku teks kanonik tentang POS dan penguraian.
- [Proyek Ketergantungan Universal](https://universaldependencies.org/) — kumpulan tag dan treebank lintas bahasa yang digunakan oleh setiap parser multibahasa.
- [panduan feature linguistik spaCy](https://spacy.io/usage/linguistic-features) — referensi praktis untuk setiap atribut yang diekspos di `Token`.
- [Chen dan Manning (2014). Parser Ketergantungan yang Cepat dan Akurat menggunakan Jaringan Neural](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — makalah yang membawa neural parser ke dalam arus utama.
