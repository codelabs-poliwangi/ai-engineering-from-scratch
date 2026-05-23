# Pemrosesan Teks — Tokenization, Stemming, Lemmatisasi

> Bahasa bersifat berkelanjutan. Model bersifat diskrit. Preprocessing adalah jembatannya.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Waktu:** ~45 menit

## Masalah

Seorang model tidak bisa membaca "Kucing-kucing itu berlari". Bunyinya bilangan bulat.

Setiap sistem NLP dibuka dengan tiga pertanyaan yang sama. Di mana sebuah kata dimulai. Apa akar kata tersebut. Bagaimana kita memperlakukan "lari", "berlari", "berlari" sebagai hal yang sama jika membantu, dan sebagai hal yang berbeda jika tidak membantu.

Jika tokenization salah, model akan belajar dari sampah. Jika tokenizer kamu memperlakukan `don't` sebagai satu token tetapi `do n't` sebagai dua, distribusi training akan terpecah. Jika stemmer kamu runtuh `organization` dan `organ` ke stem yang sama, pemodelan topik akan mati. Jika lemmatizer kamu memerlukan konteks part-of-speech tetapi kamu tidak meneruskannya, kata kerja akan diperlakukan sebagai kata benda.

Lesson ini membangun tiga primitif pra-pemrosesan dari awal, lalu menunjukkan bagaimana NLTK dan spaCy melakukan pekerjaan yang sama sehingga kamu dapat melihat keuntungannya.

## Konsep

Tiga operasi. Masing-masing memiliki pekerjaan dan mode kegagalan.

**Tokenization** membagi string menjadi token. "Token" sengaja dibuat tidak jelas karena perincian yang tepat bergantung pada tugas. Tingkat kata untuk NLP klasik. Subkata untuk Transformer. Karakter untuk bahasa tanpa spasi.

**Membendung** sufiks daging dengan aturan. Cepat, agresif, bodoh. `running -> run`. `organization -> organ`. Yang kedua adalah mode kegagalan.

**Lematisasi** mereduksi sebuah kata ke bentuk kamusnya menggunakan pengetahuan tata bahasa. Lebih lambat, akurat, memerlukan tabel pencarian atau penganalisis morfologi. `ran -> run` (perlu diketahui "ran" adalah bentuk lampau dari "run"). `better -> good` (perlu mengetahui bentuk perbandingan).

Aturan praktis. Stem ketika kecepatan penting dan kamu dapat mentolerir kebisingan (pengindeksan pencarian, klasifikasi kasar). Lemmatisasi ketika makna itu penting (menjawab pertanyaan, pencarian semantik, apa pun yang akan dibaca pengguna).

## Build

### Langkah 1: tokenizer kata regex

Tokenizer berguna yang paling sederhana dibagi menjadi karakter non-alfanumerik sambil mempertahankan tanda baca sebagai tokennya sendiri. Belum sempurna, belum final, namun berjalan dalam satu baris.

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

Tiga pola dalam urutan prioritas. Kata-kata dengan tanda kutip dalam opsional (`don't`, `it's`). Angka murni. Setiap karakter non-alfanumerik non-spasi putih sebagai token mandiri (tanda baca).

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

Mode kegagalan yang perlu diperhatikan. `3pm` dibagi menjadi `['3', 'pm']` karena kami bergantian antara penulisan huruf dan penulisan angka. Cukup baik untuk sebagian besar tugas. URL, email, hashtag semuanya rusak. Untuk produksi, tambahkan pola sebelum pola umum.

### Langkah 2: Porter stemmer (hanya langkah 1a)

Algoritma Porter lengkap memiliki lima fase aturan. Langkah 1a sendiri mencakup sufiks bahasa Inggris yang paling sering digunakan dan mengajarkan polanya.

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

Baca aturannya dari atas ke bawah. Aturan `ies -> i` adalah alasannya `ponies -> poni`, bukan `pony`. Real Porter memiliki langkah 1b yang akan memperbaikinya. Aturan bersaing. Aturan sebelumnya menang. Prompt itu lebih penting daripada aturan apa pun.

### Langkah 3: lemmatizer berbasis pencarian

Lemmatisasi yang tepat membutuhkan morfologi. Versi pengajaran yang mudah diatur menggunakan tabel lemma kecil dan fallback.

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```Kasus terakhir adalah momen pengajaran yang penting. `watched` tidak ada dalam tabel kami dan fallback kami hanya menangani `ing`. Lemmatisasi nyata mencakup `ed`, kata kerja tidak beraturan, kata sifat komparatif, bentuk jamak dengan perubahan bunyi (`children -> child`). Inilah sebabnya mengapa sistem produksi menggunakan WordNet, pembuat morfologi spaCy, atau penganalisis morfologi lengkap.

### Langkah 4: satukan keduanya

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

Bagian yang hilang adalah penanda POS. Phase 5 · 07 (POS Tagging) membangun satu. Untuk saat ini, default semuanya ke `NOUN` dan akui batasannya.

## Pakai

NLTK dan spaCy mengirimkan versi produksi. Masing-masing beberapa baris.

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` menangani kontraksi, Unicode, kasus tepi yang terlewatkan oleh regex kamu. `PorterStemmer` menjalankan kelima fase. `WordNetLemmatizer` memerlukan tag POS yang diterjemahkan dari skema Penn Treebank NLTK ke kumpulan singkatan WordNet. Kabel terjemahan di atas adalah bagian yang dilewati sebagian besar tutorial.

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy menyembunyikan seluruh pipeline di belakang `nlp(text)`. Tokenization, penandaan POS, dan lemmatisasi semuanya berjalan. Lebih cepat dari NLTK dalam skala besar. Lebih akurat di luar kotak. Kerugiannya adalah kamu tidak dapat dengan mudah menukar komponen satu per satu.

### Kapan harus memilih yang mana

| Situasi | Pilih |
|-----------|------|
| Mengajar, meneliti, menukar komponen | NLTK |
| Produksi, multi-bahasa, kecepatan penting | spaCy |
| Pipa Transformer (kamu tetap akan melakukan tokenization dengan tokenizer model) | Gunakan `tokenizers` / `transformers` dan lewati preprocessing klasik |

### Dua mode kegagalan yang tidak diperingatkan oleh siapa pun

Kebanyakan tutorial mengajarkan algoritma dan berhenti. Ada dua hal yang akan mengganggu proses pra-pemrosesan yang sebenarnya, dan keduanya hampir tidak pernah tercakup.

**Penyimpangan reproduksibilitas.** NLTK dan spaCy mengubah perilaku tokenization dan lemmatizer antar-versi. Apa yang dihasilkan `['do', "n't"]` di spaCy 2.x dapat menghasilkan `["don't"]` di 3.x. Model kamu dilatih pada satu distribusi. Inference sekarang berjalan pada inference yang berbeda. Akurasi diam-diam menurun dan tidak ada yang tahu alasannya. Sematkan versi perpustakaan di `requirements.txt`. Tulis uji regresi preprocessing yang membekukan tokenization yang diharapkan dari 20 contoh kalimat. Jalankan di setiap peningkatan.

**Training / ketidakcocokan inference.** Berlatih dengan preprocessing yang agresif (huruf kecil, penghapusan stopword, stemming), terapkan pada input pengguna mentah, perhatikan penurunan performa. Ini adalah kegagalan produksi NLP yang paling umum. Jika kamu melakukan praproses selama training, kamu harus menjalankan fungsi yang sama selama inference. Kirim preprocessing sebagai fungsi di dalam paket model, bukan sebagai sel buku catatan yang ditulis ulang oleh tim yang melayani.

## Kirim

Prompt yang dapat digunakan kembali yang membantu para insinyur memilih strategi pra-pemrosesan tanpa membaca tiga buku teks.

Simpan sebagai `outputs/prompt-preprocessing-advisor.md`:

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## Latihan1. **Mudah.** Perluas `tokenize` untuk menyimpan URL sebagai token tunggal. Pengujian: `tokenize("Visit https://example.com today.")` harus menghasilkan satu token URL.
2. **Sedang.** Menerapkan Porter langkah 1b. Jika sebuah kata mengandung vokal dan diakhiri dengan `ed` atau `ing`, hapus kata tersebut. Tangani aturan konsonan ganda (`hopping -> hop`, bukan `hopp`).
3. **Hard.** Buat lemmatizer yang menggunakan WordNet sebagai tabel pencarian tetapi kembali ke stemmer Porter kamu ketika WordNet tidak memiliki entri. Ukur akurasi pada korpus yang diberi tag terhadap WordNet biasa dan Porter biasa.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Tanda | Sebuah kata | Berapa pun unit yang dikonsumsi model. Bisa berupa kata, subkata, karakter, atau byte. |
| Batang | Akar kata | Hasil pengupasan sufiks berbasis aturan. Tidak selalu merupakan kata yang nyata. |
| Lemma | Bentuk kamus | Formulir yang kamu cari. Membutuhkan konteks tata bahasa untuk menghitung dengan benar. |
| Label POS | Bagian dari pidato | Kategori seperti KATA BENDA, KATA KERJA, ADJ. Diperlukan lemmatisasi secara akurat. |
| Morfologi | Aturan Bentuk Kata | Bagaimana suatu kata berubah bentuk berdasarkan tense, number, case. Lemmatisasi bergantung padanya. |

## Bacaan Lanjutan

- [Porter, MF (1980). Algoritme untuk pengupasan sufiks](https://tartarus.org/martin/PorterStemmer/def.txt) — makalah asli, lima halaman, masih merupakan penjelasan paling jelas.
- [spaCy 101 — feature linguistik](https://spacy.io/usage/linguistic-features) — cara pipeline sebenarnya dihubungkan.
- [Buku NLTK, bab 3](https://www.nltk.org/book/ch03.html) — kasus tepi tokenization yang belum terpikirkan oleh kamu.
