# Penautan & Disambiguasi Entitas

> NER menemukan "Paris". Penautan entitas memutuskan: Paris, Prancis? Paris Hilton? Paris, Texas? Paris (pangeran Troya)? Tanpa menghubungkan, grafik pengetahuan kamu tetap ambigu.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 5 · 06 (NER), Fase 5 · 24 (Resolusi Koreferensi)
**Waktu:** ~60 menit

## Masalah

Sebuah kalimat berbunyi: "Yordania mengalahkan pers." NER kamu menandai "Jordan" sebagai PERSON. Bagus. Tapi *yang mana* Jordan?

- Michael Jordan (basket)?
- Michael B.Jordan (aktor)?
- Michael I. Jordan (profesor Berkeley ML — ya, perplexity ini nyata dalam makalah ML)?
- Yordania (negara)?
- Jordan (nama depan Ibrani)?

Penautan entitas (EL) menyelesaikan setiap penyebutan ke entri unik di basis pengetahuan: Wikidata, Wikipedia, DBpedia, atau KB domain kamu. Dua subtugas:

1. **Generasi kandidat.** Mengingat "Jordan", entri KB manakah yang masuk akal?
2. **Disambiguasi.** Berdasarkan konteksnya, kandidat mana yang tepat?

Kedua langkah tersebut dapat dipelajari. Keduanya dijadikan acuan. Pipeline pipa gabungan telah stabil selama satu dekade — yang berubah adalah kualitas disambiguatornya.

## Konsep

![Pipa penghubung entitas: sebutkan → kandidat → entitas yang tidak ambigu](../assets/entity-linking.svg)

**Pembuatan kandidat.** Berdasarkan bentuk penyebutan (“Jordan”), cari kandidat dalam indeks alias. Kamus alias Wikipedia mencakup sebagian besar entitas dengan nama: "JFK" → John F. Kennedy, Jacqueline Kennedy, bandara JFK, JFK (film). Indeks tipikal menghasilkan 10-30 kandidat per penyebutan.

**Disambiguasi: tiga pendekatan.**

1. **Sebelumnya + konteks (Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)`. Bekerja dengan baik, cepat, tanpa training.
2. **Berbasis embedding (ESS / REL / Blink).** Encode penyebutan + konteks. Encode deskripsi masing-masing kandidat. Pilih kosinus maks. Default 2020-2024.
3. **Generatif (GENRE, 2021; berbasis LLM, 2023+).** Dekode nama kanonik entitas token demi token. Dibatasi pada percobaan nama entitas yang valid sehingga output dijamin berupa id KB yang valid.

**End-to-end vs pipeline.** Model modern (ELQ, BLINK, ExtEnD, GENRE) menjalankan NER + pembuatan kandidat + disambiguasi dalam satu proses. Sistem pipeline masih mendominasi produksi karena kamu dapat menukar komponen.

### Dua pengukuran

- **Sebutkan recall (gen kandidat).** Fraksi emas menyebutkan di mana entri KB yang benar muncul dalam daftar kandidat. Lantai untuk seluruh pipa.
- **Akurasi disambiguasi / F1.** Jika kandidat benar, seberapa sering kandidat teratas benar.

Selalu laporkan keduanya. Sebuah sistem dengan disambiguasi 99% pada 80% penarikan kandidat adalah 80% pipeline.

## Build

### Langkah 1: buat indeks alias dari pengalihan Wikipedia

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Data alias Wikipedia: ~18 juta pasangan (alias, entitas). Unduh dari dump Wikidata. Simpan sebagai indeks terbalik.

### Langkah 2: disambiguasi berbasis konteks

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Tumpang tindih Jaccard adalah mainan. Ganti dengan kesamaan kosinus pada embeddings (lihat `code/main.py` langkah-2 untuk versi Transformer).

### Langkah 3: berbasis embedding (gaya BLINK)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

Pada waktu indeks, sematkan setiap entitas KB satu kali. Pada waktu kueri, sematkan penyebutan + konteks satu kali, produk titik pada kumpulan kandidat, pilih maks.

### Langkah 4: penautan entitas generatif (konsep)GENRE menerjemahkan judul Wikipedia entitas karakter demi karakter. Penguraian code yang dibatasi (lihat lesson 20) memastikan hanya judul yang valid yang dapat dihasilkan. Integrasi yang erat dengan percobaan yang didukung KB. Keturunan modernnya adalah EL yang dipicu oleh REL-GEN dan LLM dengan output terstruktur.

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

Dikombinasikan dengan daftar putih (Garis Besar `choice`), ini adalah pipeline EL paling sederhana yang dikirimkan pada tahun 2026.

### Langkah 5: evaluasi AIDA-CoNLL

AIDA-CoNLL adalah tolok ukur EL standar: 1.393 artikel Reuters, 34 ribu sebutan, entitas Wikipedia. Laporkan keakuratan dalam KB (`P@1`) dan tingkat deteksi NIL di luar KB.

## Jebakan

- **Penanganan NIL.** Beberapa penyebutan tidak ada di KB (entitas baru, orang tidak jelas). Sistem harus memprediksi NIL daripada menebak entitas yang salah. Diukur secara terpisah.
- **Sebutkan kesalahan batas.** NER hulu tidak memenuhi sebagian rentang ("Bank of America" ​​hanya ditandai sebagai "Bank"). Penarikan kembali EL menurun.
- **Bias popularitas.** Sistem terlatih memprediksi entitas yang sering terjadi secara berlebihan. Penyebutan "Michael I. Jordan" pada makalah ML sering kali dikaitkan dengan bola basket Jordan.
- **EL lintas bahasa.** Memetakan penyebutan dalam teks berbahasa Mandarin ke entitas Wikipedia bahasa Inggris. Memerlukan encoder multibahasa atau langkah terjemahan.
- **KB basi.** Perusahaan, acara, orang-orang baru tidak ada dalam dump Wikipedia tahun lalu. Jalur produksi memerlukan putaran penyegaran.

## Pakai

Tumpukan tahun 2026:

| Situasi | Pilih |
|-----------|------|
| Bahasa Inggris Tujuan Umum + Wikipedia | BLINK atau REL |
| Lintas bahasa, KB = Wikipedia | mGENRE |
| Ramah LLM, sedikit penyebutan/hari | Prompt Claude/GPT-4 dengan daftar kandidat + JSON terbatas |
| KB khusus domain (medis, legal) | BERT khusus dengan pengambilan sadar KB + penyempurnaan pada kumpulan gaya AIDA domain |
| Latensi sangat rendah | Hanya pertandingan persis sebelumnya (garis dasar Milne-Witten) |
| Penelitian SOTA | GENRE / ExtEnD / generatif LLM-EL |

Pola produksi yang dikirimkan pada tahun 2026: NER → coref → EL pada setiap penyebutan → ciutkan cluster menjadi satu entitas kanonik per cluster. Output: satu id KB per entitas dalam dokumen, bukan satu id per penyebutan.

## Kirim

Simpan sebagai `outputs/skill-entity-linker.md`:

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## Latihan

1. **Mudah.** Menerapkan disambiguator+konteks sebelumnya di `code/main.py` pada 10 penyebutan ambigu (Paris, Jordan, Apple). Beri label tangan pada entitas yang benar. Akurasi pengukuran.
2. **Sedang.** Enkode 50 penyebutan ambigu dengan pengubah kalimat. Cantumkan deskripsi masing-masing kandidat. Bandingkan disambiguasi berbasis embedding dengan konteks Jaccard yang tumpang tindih.
3. **Sulit.** Buat KB domain entitas 1k (misalnya karyawan + produk di perusahaan kamu). Menerapkan NER + EL ujung ke ujung. Ukur presisi dan ingat 100 kalimat yang diluruskan.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Penautan entitas (EL) | Tautan ke Wikipedia | Memetakan penyebutan ke entri KB unik. |
| Generasi kandidat | Siapa itu? | Kembalikan daftar entri KB yang masuk akal untuk disebutkan. |
| Disambiguasi | Pilih yang tepat | Skor kandidat menggunakan konteks, pilih pemenangnya. |
| Indeks alias | Tabel pencarian | Petakan dari bentuk permukaan → entitas kandidat. |
| nihil | Tidak di KB | Prediksi eksplisit bahwa tidak ada entri KB yang cocok. |
| KB | Basis pengetahuan | Wikidata, Wikipedia, DBpedia, atau KB domain kamu. |
| AIDA-CoNLL | Tolok ukur | 1.393 artikel Reuters dengan tautan entitas emas. |

## Bacaan Lanjutan- [Milne, Witten (2008). Belajar Menghubungkan dengan Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — pendekatan prior+konteks dasar.
- [Wu dkk. (2020). Penautan Entitas Zero-shot dengan Pengambilan Entitas Padat (BLINK)](https://arxiv.org/abs/1911.03814) — pekerja keras berbasis embedding.
- [De Cao dkk. (2021). Pengambilan Entitas Autoregresif (GENRE)](https://arxiv.org/abs/2010.00904) — EL generatif dengan decoding terbatas.
- [Hoffart dkk. (2011). Disambiguasi Kuat dari Entitas Bernama dalam Teks (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — makalah benchmark.
- [REL: An Entity Linker Berdiri di Bahu Raksasa (2020)](https://arxiv.org/abs/2006.01969) — tumpukan produksi terbuka.
