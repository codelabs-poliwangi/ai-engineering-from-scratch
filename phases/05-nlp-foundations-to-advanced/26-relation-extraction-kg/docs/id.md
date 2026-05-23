# Ekstraksi Relasi & Konstruksi Grafik Pengetahuan

> NER menemukan entitas. Tautan entitas menambatkannya. Ekstraksi relasi menemukan tepi di antara keduanya. Grafik pengetahuan adalah jumlah node, tepi, dan asalnya.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 5 · 06 (NER), Fase 5 · 25 (Entity Linking)
**Waktu:** ~60 menit

## Masalah

Seorang analis membaca: "Tim Cook menjadi CEO Apple pada tahun 2011." Empat fakta:

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

Ekstraksi Relasi (RE) mengubah teks bebas menjadi tiga kali lipat terstruktur `(subject, relation, object)`. Gabungkan seluruh korpus dan kamu memiliki grafik pengetahuan. Agregat dan buat kueri dan kamu memiliki dasar pemikiran untuk RAG, analitik, atau audit kepatuhan.

Masalah tahun 2026: LLM mengekstraksi hubungan dengan antusias. Terlalu antusias. Mereka berhalusinasi tiga kali lipat yang tidak didukung oleh teks sumber. Tanpa asal usulnya, kamu tidak bisa membedakan tripel nyata dari fiksi yang masuk akal. Jawaban pada tahun 2026 adalah pipeline pipa jangkar dan verifikasi bergaya AEVS.

## Konsep

![Teks → tiga kali lipat → grafik pengetahuan](../assets/relation-extraction.svg)

**Bentuk rangkap tiga.** `(subject_entity, relation_type, object_entity)`. Relasi berasal dari ontologi tertutup (properti Wikidata, FIBO, UMLS) atau himpunan terbuka (gaya OpenIE, apa saja).

**Tiga pendekatan ekstraksi.**

1. **Berbasis aturan / pola.** Pola pendengaran: "X seperti Y" → `(Y, isA, X)`. Ditambah regex buatan tangan. Rapuh, tepat, dapat dijelaskan.
2. **Pengklasifikasi yang diawasi.** Dengan adanya dua penyebutan entitas dalam sebuah kalimat, prediksi relasinya dari himpunan tetap. Dilatih di TACRED, ACE, KBP. Standar 2015–2022.
3. **LLM Generatif.** Minta model mengeluarkan tiga kali lipat. Bekerja di luar kotak. Membutuhkan asal usulnya, atau berhalusinasi sampah yang tampak masuk akal.

**AEVS (Anchor-Extraction-Verification-Supplement, 2026).** Kerangka kerja mitigasi halusinasi saat ini:

- **Anchor.** Identifikasi setiap rentang entitas dan rentang frasa relasi dengan posisi yang tepat.
- **Ekstrak.** Hasilkan tiga kali lipat yang ditautkan ke bentang jangkar.
- **Verifikasi.** Cocokkan setiap elemen rangkap tiga kembali ke teks sumber; menolak apa pun yang tidak didukung.
- **Tambahan.** Tiket cakupan memastikan tidak ada bentang jangkar yang terjatuh.

Halusinasi menurun tajam. Membutuhkan lebih banyak komputasi tetapi dapat diaudit.

**Pertukaran terbuka vs tertutup.**

- **Ontologi tertutup.** Daftar properti tetap (misalnya, 11.000+ properti Wikidata). Dapat diprediksi. Dapat ditanyakan. Sulit untuk diciptakan.
- **Buka IE.** Frasa verbal apa pun menjadi relasi. Ingatan tinggi. Presisi rendah. Berantakan untuk ditanyakan.

KG Produksi biasanya menggabungkan: membuka IE untuk penemuan, kemudian mengkanonikalisasi relasi ke dalam ontologi tertutup sebelum digabungkan ke dalam grafik utama.

## Build

### Langkah 1: ekstraksi berbasis pola

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

Lihat `code/main.py` untuk ekstraktor mainan lengkap. Pola Hearst masih dikirimkan dalam alur khusus domain karena dapat di-debug.

### Langkah 2: klasifikasi relasi yang diawasi

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL adalah ekstraktor relasi seq2seq: teks masuk, tiga kali lipat, sudah ada dalam id properti Wikidata. Menyempurnakan data pengawasan distance jauh. Garis dasar weight terbuka standar.

### Langkah 3: Ekstraksi yang diminta LLM dengan penahan

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

Verifikasi setiap rentang yang dikembalikan terhadap sumbernya. Tolak apa pun di mana `text[start:end] != triple_entity`. Ini adalah langkah "verifikasi" AEVS dalam bentuk minimalnya.

### Langkah 4: kanonikalisasi ke dalam ontologi tertutup

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

Kanonikalisasi seringkali merupakan 60-80% dari pekerjaan teknik. Anggaran untuk itu.### Langkah 5: buat grafik dan kueri kecil

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

Ini adalah atom dari setiap sistem RAG-over-KG. Skalakan dengan penyimpanan rangkap tiga RDF (Blazegraph, Virtuoso), grafik properti (Neo4j), atau penyimpanan grafik yang ditambah vector.

## Jebakan

- **Coreference sebelum RE.** "Dia mendirikan Apple" — RE perlu mengetahui siapa "dia". Jalankan coref terlebih dahulu (lesson 24).
- **Kanonikalisasi entitas.** "Apple Inc" dan "Apple" harus diselesaikan ke node yang sama. Penautan entitas terlebih dahulu (lesson 25).
- **Halusinasi tiga kali lipat.** LLM mengeluarkan tiga kali lipat yang teksnya tidak didukung. Terapkan verifikasi rentang.
- **Penyimpangan kanonikalisasi relasi.** Relasi IE terbuka tidak konsisten ("lahir di", "berasal dari", "adalah penduduk asli"). Ciutkan ke id kanonik atau grafik tidak dapat dikueri.
- **Kesalahan sementara.** "Tim Cook adalah CEO Apple" — benar saat ini, salah pada tahun 2005. Banyak hubungan yang terikat waktu. Gunakan kualifikasi (`P580` waktu mulai, `P582` waktu berakhir di Wikidata).
- **Ketidakcocokan domain.** REBEL dilatih di Wikipedia. Teks hukum, medis, dan ilmiah sering kali memerlukan model RE yang disesuaikan dengan domain.

## Pakai

Tumpukan tahun 2026:

| Situasi | Pilih |
|-----------|------|
| Produksi cepat, domain umum | REBEL atau LlamaPred dengan kanonikalisasi Wikidata |
| Khusus domain (biomed, legal) | Penyempurnaan domain bergaya SciREX + ontologi khusus |
| Output yang diminta dan diaudit oleh LLM | Pipa AEVS: jangkar → ekstrak → verifikasi → suplemen |
| Berita bervolume tinggi IE | Hibrida berbasis pola + diawasi |
| Membangun KG dari awal | Buka IE + pass kanonikalisasi manual |
| KG Sementara | Ekstrak dengan kualifikasi (waktu mulai/berakhir, waktu tertentu) |

Pola integrasi: NER → coref → penghubung entitas → ekstraksi relasi → pemetaan ontologi → pemuatan grafik. Setiap tahapan merupakan gerbang kualitas yang potensial.

## Kirim

Simpan sebagai `outputs/skill-re-designer.md`:

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## Latihan

1. **Mudah.** Jalankan ekstraktor pola di `code/main.py` pada 5 kalimat artikel berita. Ketepatan pemeriksaan tangan.
2. **Sedang.** Gunakan REBEL (atau LLM kecil) pada kalimat yang sama. Bandingkan tiga kali lipat. Ekstraktor mana yang memiliki presisi lebih tinggi? Ingatan lebih tinggi?
3. **Sulit.** Build alur AEVS: ekstrak dengan LLM + verifikasi rentang terhadap sumber. Ukur tingkat halusinasi sebelum dan sesudah langkah verifikasi pada 50 kalimat bergaya Wikipedia.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Tiga kali lipat | Subjek-relasi-objek | `(s, r, o)` tupel yang merupakan satuan atom KG. |
| Buka IE | Ekstrak apa saja | Frase hubungan kosakata terbuka; recall tinggi, presisi rendah. |
| Ontologi tertutup | Skema tetap | Kumpulan tipe relasi yang dibatasi (Wikidata, UMLS, FIBO). |
| Kanonikalisasi | Normalisasikan semuanya | Memetakan nama permukaan/hubungan dengan id kanonik. |
| AEVS | Ekstraksi beralas | Pipa Anchor-Ekstraksi-Verifikasi-Suplemen (2026). |
| Asal | Tautan sumber kebenaran | Setiap triple membawa id dokumen + char-span ke sumbernya. |
| Pengawasan distance jauh | Label murah | Sejajarkan teks dengan KG yang ada untuk membuat training data. |

## Bacaan Lanjutan- [Mintz dkk. (2009). Pengawasan distance jauh untuk ekstraksi relasi tanpa data berlabel](https://www.aclweb.org/anthology/P09-1113.pdf) — makalah pengawasan distance jauh.
- [Huguet Cabot, Navigli (2021). REBEL: Ekstraksi Relasi Berdasarkan Pembuatan Bahasa Ujung-ke-ujung](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE pekerja keras.
- [Wadden dkk. (2019). Ekstraksi Entitas, Relasi, dan Peristiwa dengan Representasi Rentang Kontekstual (DyGIE++)](https://arxiv.org/abs/1909.03546) — IE bersama.
- [AEVS — Framework Anchor-Extraction-Verification-Supplement](https://www.mdpi.com/2073-431X/15/3/178) — Desain mitigasi halusinasi tahun 2026.
- [tutorial Wikidata SPARQL](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — kueri grafik kanonik.
