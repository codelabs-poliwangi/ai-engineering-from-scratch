# Resolusi Intiferensi

> "Dia meneleponnya. Dia tidak menjawab. Dokter sedang makan siang." Tiga referensi untuk dua orang dan tidak ada yang disebutkan namanya. Resolusi inti menunjukkan siapa adalah siapa.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**Waktu:** ~60 menit

## Masalah

Ekstrak setiap penyebutan Apple Inc. dari artikel 300 kata. Mudah jika artikelnya mengatakan "Apple". Sulit jika tertulis "perusahaan", "mereka", "raksasa teknologi Cupertino", atau "perusahaan Jobs". Tanpa menyelesaikan penyebutan ini ke entitas yang sama, pipeline NER kamu akan kehilangan 60-80% penyebutan.

Resolusi inti menghubungkan setiap ekspresi yang merujuk pada entitas dunia nyata yang sama ke dalam satu cluster. Ini adalah perekat antara NLP tingkat permukaan (NER, parsing) dan semantik hilir (IE, QA, summarization, KG).

Mengapa ini penting di tahun 2026:

- Ringkasan: "CEO mengumumkan..." vs "Tim Cook mengumumkan..." — ringkasan harus menyebutkan nama CEO.
- Menjawab pertanyaan: "Siapa yang dia telepon?" membutuhkan penyelesaian "dia".
- Ekstraksi informasi: grafik pengetahuan dengan "PER1 mendirikan Apple" dan "Pekerjaan mendirikan Apple" sebagai entri terpisah adalah salah.
- IE multi-dokumen: menggabungkan penyebutan di seluruh artikel tentang peristiwa yang sama adalah inti referensi lintas dokumen.

## Konsep

![Pengelompokan inti referensi: sebutan → entitas](../assets/coref.svg)

**Tugas.** Input: dokumen. Output: pengelompokan penyebutan (rentang) dimana setiap cluster mengacu pada satu entitas.

**Jenis penyebutan.**

- **Entitas bernama.** "Tim Cook"
- **Nominal.** "CEO", "perusahaan"
- **Kata ganti.** "dia", "dia", "mereka", "itu"
- **Appositif.** "Tim Cook, CEO Apple,"

**Arsitektur.**

1. **Berbasis aturan (Hobbs, 1978).** Resolusi kata ganti berbasis pohon sintaksis menggunakan aturan tata bahasa. Dasar yang bagus. Sangat sulit dikalahkan dalam kata ganti.
2. **Pengklasifikasi pasangan sebutan.** Untuk setiap pasangan sebutan (m_i, m_j), prediksikan apakah pasangan tersebut merujuk. Cluster berdasarkan penutupan transitif. Standar sebelum tahun 2016.
3. **Peringkat penyebutan.** Untuk setiap penyebutan, rangking kandidat yang mendahuluinya (termasuk "tidak ada pendahulunya"). Pilih yang paling atas.
4. **Encoder Transformer berbasis rentang (Lee et al., 2017).** Hitung semua rentang kandidat hingga batas panjangnya. Memprediksi skor penyebutan. Prediksi probabilitas anteseden untuk setiap rentang. Berkelompok dengan rakus. Standar modern.
5. **Generatif (2024+).** Perintahkan LLM: "Cantumkan setiap kata ganti dalam teks ini dan pendahulunya." Bekerja dengan baik pada kasus-kasus mudah, kesulitan pada dokumen panjang dan referensi langka.

**Metrik evaluasi.** Lima metrik standar (MUC, B³, CEAF, BLANC, LEA) karena tidak ada satu metrik pun yang menangkap kualitas pengelompokan. Laporkan rata-rata dari tiga yang pertama sebagai CoNLL F1. Tercanggih pada tahun 2026 di CoNLL-2012: ~83 F1.

**Kasus sulit yang diketahui.**

- Deskripsi pasti mengacu pada entitas yang diperkenalkan halaman sebelumnya.
- Menjembatani anafora ("roda" → mobil yang disebutkan sebelumnya).
- Nol anafora dalam bahasa seperti Cina dan Jepang.
- Cataphora (kata ganti sebelum rujukan): "Saat **dia** masuk, Mary tersenyum."

## Build

### Langkah 1: inti saraf terlatih (AllenNLP / spaCy-experimental)

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

Pada dokumen yang lebih panjang, kamu mendapatkan sesuatu seperti:
- Cluster 1: [Apple, Perusahaan, mereka]
- Klaster 2: [produk baru]

### Langkah 2: penyelesai kata ganti berbasis aturan (pengajaran)

Lihat `code/main.py` untuk implementasi khusus stdlib:1. Ekstrak menyebutkan: entitas bernama (bentang huruf kapital), kata ganti (pencarian dict), deskripsi pasti ("X").
2. Untuk setiap kata ganti, lihat penyebutan K sebelumnya dan beri skor berdasarkan:
   - kesepakatan gender/nomor (heuristik)
   - kekinian (mendekati kemenangan)
   - peran sintaksis (subjek lebih disukai)
3. Tautkan anteseden dengan skor tertinggi.

Tidak kompetitif dengan model saraf. Namun hal ini menunjukkan ruang pencarian dan keputusan yang harus diambil oleh model end-to-end.

### Langkah 3: menggunakan LLM untuk referensi

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

Dua mode kegagalan yang harus diperhatikan. Pertama, LLM digabungkan secara berlebihan ("dia" dan "dia" mengacu pada dua orang yang berbeda). Kedua, LLM secara diam-diam menghilangkan penyebutan dalam dokumen yang panjang. Selalu verifikasi dengan pemeriksaan span-offset.

### Langkah 4: evaluasi

Skrip conll-2012 standar menghitung MUC, B³, CEAF-φ4 dan melaporkan rata-rata. Untuk evaluasi internal, mulailah dengan presisi tingkat rentang dan ingat set pengujian beranotasi kamu, lalu tambahkan F1 yang menghubungkan penyebutan.

## Jebakan

- **Ledakan tunggal.** Beberapa sistem melaporkan setiap penyebutan sebagai klusternya sendiri. B³ toleran. MUC menghukumnya. Selalu periksa ketiga metrik.
- **Kata ganti dalam konteks panjang.** Performa turun ~15 F1 pada dokumen dengan lebih dari 2.000 token. Potong dengan hati-hati.
- **Asumsi gender.** Aturan gender yang kaku tidak berlaku pada referensi non-biner, organisasi, dan hewan. Gunakan model yang dipelajari atau penilaian netral.
- **LLM melayang pada dokumen yang panjang.** Satu panggilan API tidak dapat mengelompokkan penyebutan lebih dari 50 paragraf dengan andal. Gunakan jendela geser + gabung.

## Pakai

Tumpukan tahun 2026:

| Situasi | Pilih |
|-----------|------|
| Bahasa Inggris, dokumen tunggal | `en_coreference_web_trf` (eksperimental spaCy) atau inti saraf AllenNLP |
| Multibahasa | SpanBERT / XLM-R dilatih tentang OntoNotes atau CoNLL Multibahasa |
| Inti acara lintas dokumen | Model end-to-end khusus (SOTA 2025–26) |
| Dasar LLM cepat | GPT-4o / Claude dengan coref prompt output terstruktur |
| Sistem dialog produksi | Penggantian berbasis aturan + peninjauan primer neural + manual untuk slot kritis |

Pola integrasi yang dikirimkan pada tahun 2026: jalankan NER terlebih dahulu, jalankan coref, gabungkan cluster coref menjadi entitas NER. Tugas hilir melihat satu entitas per klaster, bukan satu entitas per penyebutan.

## Kirim

Simpan sebagai `outputs/skill-coref-picker.md`:

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## Latihan

1. **Mudah.** Jalankan penyelesai berbasis aturan di `code/main.py` pada 5 paragraf buatan tangan. Ukur keakuratan tautan penyebutan terhadap kebenaran dasar.
2. **Medium.** Gunakan model inti saraf terlatih pada artikel berita. Bandingkan cluster dengan anotasi manual kamu sendiri. Dimana kegagalannya?
3. **Hard.** Build pipeline NER yang disempurnakan dengan coref: NER terlebih dahulu, lalu gabungkan melalui cluster coref. Ukur peningkatan cakupan entitas vs hanya NER pada 100 artikel.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Sebutkan | Referensi | Rentang teks yang merujuk pada suatu entitas (nama, kata ganti, frasa kata benda). |
| Anteseden | Yang dimaksud dengan "itu" | Penyebutan sebelumnya mengacu pada penyebutan selanjutnya. |
| Gugus | Penyebutan entitas | Kumpulan penyebutan yang semuanya merujuk pada entitas dunia nyata yang sama. |
| Anafora | Referensi mundur | Penyebutan selanjutnya mengacu pada sebelumnya ("dia" → "John"). |
| Katafora | Referensi maju | Penyebutan sebelumnya mengacu pada kemudian ("Ketika dia tiba, John..."). |
| Menjembatani | Referensi implisit | "Saya membeli mobil. Rodanya jelek." (roda mobil ITU.) |
| CoNLL F1 | Nomor di papan peringkat | Rata-rata skor MUC, B³, CEAF-φ4 F1. |

## Bacaan Lanjutan- [Jurafsky & Martin, SLP3 Bab. 26 — Resolusi Inti dan Penautan Entitas](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — bab buku teks kanonik.
- [Lee dkk. (2017). Resolusi Inti Neural ujung ke ujung](https://arxiv.org/abs/1707.07045) — ujung ke ujung berbasis rentang.
- [Joshi dkk. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — training awal yang meningkatkan coref.
- [Pradhan dkk. (2012). Tugas Bersama CoNLL-2012](https://aclanthology.org/W12-4501/) — tolok ukurnya.
- [Hobbs (1978). Referensi Penyelesaian Kata Ganti](https://www.sciencedirect.com/science/article/pii/0024384178900064) — klasik berbasis aturan.
