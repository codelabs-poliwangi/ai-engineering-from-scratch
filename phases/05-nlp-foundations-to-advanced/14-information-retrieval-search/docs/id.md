# Pengambilan dan Pencarian Informasi

> BM25 presisi namun rapuh. Dense memberikan jaring yang luas tetapi melewatkan kata kunci. Hibrida adalah default tahun 2026. Segala sesuatu yang lain sedang disetel.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 5 · 02 (BoW + TF-IDF), Fase 5 · 04 (GloVe, FastText, Subword)
**Waktu:** ~75 menit

## Masalah

Pengguna mengetik "apa yang terjadi jika seseorang berbohong untuk mendapatkan uang" dan berharap menemukan undang-undang yang sebenarnya mencakup hal tersebut: "Pasal 420 IPC". Pencarian kata kunci melewatkannya sepenuhnya (tidak ada kosakata bersama). Penelusuran semantik akan melewatkannya jika embedding tidak dilatih pada teks hukum. Pencarian nyata harus menangani keduanya.

IR adalah jalur pipa di bawah setiap sistem RAG, setiap bilah pencarian, setiap pencarian fuzzy di setiap situs dokumen. Arsitektur 2026 yang berfungsi dalam produksi bukanlah metode tunggal. Ini adalah rangkaian metode yang saling melengkapi, masing-masing menangkap kegagalan metode sebelumnya.

Lesson ini membangun setiap bagian dan memberi nama kegagalan yang ditangkap setiap orang.

## Konsep

![Pengambilan hibrid: BM25 + padat + RRF + pemeringkatan ulang lintas-encoder](../assets/retrieval.svg)

Empat layer. Pilih yang kamu butuhkan.

1. **Pengambilan jarang (BM25).** Cepat, tepat dalam pencocokan tepat, buruk dalam semantik. Jalankan indeks terbalik. Sub-10 md per kueri pada jutaan dokumen. Memberi kamu referensi undang-undang, code produk, pesan kesalahan, nama entitas dengan benar.
2. **Pengambilan padat.** Mengkodekan kueri dan dokumen ke dalam vector. Pencarian nearest neighbor. Menangkap parafrase dan kesamaan semantik. Melewatkan pencocokan kata kunci persis yang berbeda satu karakter. 50-200 md per kueri dengan FAISS atau DB vector.
3. **Fusion.** Menggabungkan daftar peringkat dari jarang dan padat. Reciprocal Rank Fusion (RRF) adalah default yang mudah karena mengabaikan skor mentah (yang berada dalam skala berbeda) dan hanya menggunakan posisi peringkat. Penggabungan tertimbang adalah opsi ketika kamu mengetahui satu sinyal mendominasi domain kamu.
4. **Pemeringkatan ulang lintas-encoder.** Ambil 30 teratas dari fusion. Jalankan cross-encoder (kueri + dokumen secara bersamaan, beri skor pada setiap pasangan). Pertahankan 5 teratas. Cross-encoder lebih lambat per pasangnya dibandingkan bi-encoder tetapi jauh lebih akurat. kamu mengamortisasinya dengan hanya menjalankannya di 30 teratas.

Pengambilan tiga arah (BM25 + padat + renggang yang dipelajari seperti SPLADE) mengungguli kinerja dua arah pada tolok ukur tahun 2026 tetapi memerlukan infrastruktur untuk indeks renggang yang dipelajari. Bagi sebagian besar tim, pemeringkatan ulang dua arah plus lintas-encoder adalah pilihan yang tepat.

## Build

### Langkah 1: BM25 dari awal

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

Dua parameter yang perlu diketahui. `k1=1.5` mengontrol saturasi term-frekuensi; lebih tinggi berarti lebih banyak weight pada pengulangan jangka waktu. `b=0.75` mengontrol normalisasi panjang; 0 mengabaikan panjang dokumen, 1 menormalkan sepenuhnya. Standarnya adalah rekomendasi Robertson dari makalah asli dan jarang memerlukan penyetelan.

### Langkah 2: pengambilan padat dengan bi-encoder

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2-normalkan embedding sehingga perkalian titik sama dengan kosinus. `all-MiniLM-L6-v2` memiliki resolusi 384-redup, cepat, dan cukup kuat untuk sebagian besar pengambilan bahasa Inggris. Untuk pekerjaan multibahasa, gunakan `paraphrase-multilingual-MiniLM-L12-v2`. Untuk akurasi tertinggi, `bge-large-en-v1.5` atau `e5-large-v2`.

### Langkah 3: Penggabungan Peringkat Timbal Balik

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

Konstanta `k=60` berasal dari kertas RRF asli. `k` yang lebih tinggi meratakan kontribusi perbedaan peringkat; `k` yang lebih rendah membuat peringkat teratas mendominasi. 60 adalah default yang diterbitkan dan jarang memerlukan penyetelan.

### Langkah 4: pencarian hibrid + peringkat ulang

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```Tiga phase disusun. BM25 menemukan kecocokan leksikal. Dense menemukan kecocokan semantik. RRF menggabungkan kedua peringkat tersebut tanpa memerlukan kalibrasi skor. Cross-encoder menilai ulang 30 teratas menggunakan pasangan dokumen kueri secara bersamaan, yang menangkap relevansi terperinci yang terlewatkan oleh bi-encoder. Pertahankan posisi 5 teratas.

### Langkah 5: evaluasi

| Metrik | Arti |
|--------|---------|
| Ingat@k | Dari pertanyaan di mana dokumen yang benar ada, seberapa sering dokumen tersebut ada di top-k? |
| MRR (Mean Reciprocal Rank) | Rata-rata 1/peringkat dokumen relevan pertama. |
| nDCG@k | Memperhitungkan gradasi relevansi, bukan hanya relevan/tidak biner. |

Khusus untuk RAG, **Recall@k** ​​dari retriever adalah nomor yang paling penting. Pembaca kamu tidak dapat menjawab jika bagian yang benar tidak ada dalam kumpulan yang diambil.

Kiat debug: untuk kueri yang gagal, bedakan peringkat yang jarang dan padat. Jika yang satu menemukan dokumen yang tepat dan yang lainnya tidak, kamu mengalami ketidakcocokan kosakata (perbaikan: tambahkan separuh yang hilang) atau ambiguitas semantik (perbaikan: embedding yang lebih baik atau pemeringkatan ulang).

## Pakai

Tumpukan tahun 2026:

| Skala | Tumpukan |
|-------|-------|
| 1rb-100rb dokumen | BM25 dalam memori + `all-MiniLM-L6-v2` embedding + RRF. Tidak ada DB terpisah. |
| 100rb-10 juta dokumen | FAISS atau pgvector untuk padat + Elasticsearch / OpenSearch untuk BM25. Jalankan secara paralel. |
| 10 juta+ dokumen | Qdrant / Weaviate / Vespa / Milvus dengan dukungan hybrid. Peringkat ulang lintas-encoder berada di peringkat 30 teratas. |
| Perbatasan kualitas terbaik | Pemeringkatan tiga arah (BM25 + padat + SPLADE) + interaksi akhir ColBERT |

Apa pun yang kamu pilih, anggarkan untuk evaluasi. Pengambilan kembali benchmark sebelum melakukan benchmarking akurasi RAG end-to-end. Seorang pembaca tidak dapat memperbaiki apa yang terlewatkan oleh retriever.

### Lesson yang diperoleh dengan susah payah dari produksi RAG tahun 2026

- **80% kegagalan RAG disebabkan oleh penyerapan dan pengelompokan, bukan modelnya.** Tim menghabiskan waktu berminggu-minggu untuk menukar LLM dan prompt penyetelan sementara pengambilan secara diam-diam mengembalikan konteks yang salah setiap kueri ketiga. Perbaiki potongannya terlebih dahulu.
- **Strategi pemotongan lebih penting daripada ukuran potongan.** Pemisahan tabel, code, dan header bertingkat dengan ukuran tetap. Sadar kalimat adalah defaultnya; chunking berbasis semantik atau LLM bermanfaat untuk dokumen teknis dan manual produk.
- **Pola dokumen induk.** Ambil potongan "anak" kecil agar presisi. Ketika beberapa anak dari bagian induk yang sama muncul, tukar di blok induk untuk mempertahankan konteks. Hal ini secara konsisten meningkatkan kualitas jawaban tanpa training ulang.
- **k_rerank=3 biasanya optimal.** Setiap potongan ekstra yang menambah biaya token dan latensi pembuatan tanpa meningkatkan kualitas jawaban. Jika k=8 masih lebih baik daripada k=3 bagi kamu, reranker berkinerja buruk.
- **HyDE / perluasan kueri.** Hasilkan jawaban hipotetis dari kueri, sematkan, ambil. Menjembatani kesenjangan ungkapan antara pertanyaan pendek dan dokumen panjang. Pengangkatan presisi gratis tanpa training.
- **Konteks anggaran di bawah 8 ribu token.** Penetrasi yang konsisten pada batas tersebut berarti ambang batas reranker terlalu longgar.
- **Versi semuanya.** Prompt, aturan pemotongan, embedding model, reranker. Penyimpangan apa pun secara diam-diam merusak kualitas jawaban. CI mengandalkan kesetiaan, ketepatan konteks, dan regresi blok tingkat pertanyaan tak terjawab sebelum pengguna melihatnya.
- **Pengambilan tiga arah (BM25 + padat + jarang dipelajari seperti SPLADE) mengungguli kinerja dua arah** pada tolok ukur tahun 2026, terutama untuk kueri yang menggabungkan kata benda dengan semantik. Kirim ketika infrastruktur mendukung indeks SPLADE.Desain pengambilan yang tepat mengurangi halusinasi sebesar 70-90% menurut pengukuran industri tahun 2026. Sebagian besar peningkatan kinerja RAG berasal dari pengambilan yang lebih baik, bukan penyesuaian model.

## Kirim

Simpan sebagai `outputs/skill-retrieval-picker.md`:

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## Latihan

1. **Mudah.** Terapkan `hybrid_search` di atas pada korpus 500 dokumen. Uji 20 kueri. Bandingkan penarikan di 5 antara BM25 saja, padat saja, dan hibrida.
2. **Sedang.** Tambahkan perhitungan MRR. Untuk setiap kueri pengujian dengan dokumen yang diketahui benar, temukan peringkat dokumen yang benar dalam peringkat BM25, padat, dan hibrid. Laporkan MRR untuk masing-masing.
3. **Sulit.** Sempurnakan encoder padat di domain kamu menggunakan MultipleNegativesRankingLoss (Sentence Transformers). Buat set training dari 500 pasangan dokumen kueri. Bandingkan penarikan kembali sebelum dan sesudah penyempurnaan.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| BM25 | Pencarian kata kunci | Okapi BM25. Mencetak dokumen berdasarkan frekuensi istilah, IDF, dan panjangnya. |
| Pengambilan padat | Pencarian vector | Enkode kueri + dokumen ke dalam vector, temukan nearest neighbor. |
| Bi-encoder | Menanamkan model | Mengkodekan kueri dan dokumen secara independen. Cepat pada waktu permintaan. |
| Pembuat enkode silang | Model pemeringkatan ulang | Mengkodekan kueri + dokumen secara bersamaan. Lambat tapi akurat. |
| RRF | Penggabungan peringkat | Gabungkan dua peringkat dengan menjumlahkan `1/(k + rank)`. |
| Ingat@k | Metrik pengambilan | Sebagian kecil kueri yang dokumennya relevan ada di k teratas. |

## Bacaan Lanjutan

- [Robertson dan Zaragoza (2009). Kerangka Relevansi Probabilistik: BM25 dan Selanjutnya](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — pengobatan definitif BM25.
- [Karpukhin dkk. (2020). Pengambilan Bagian Padat untuk QA Domain Terbuka](https://arxiv.org/abs/2004.04906) — DPR, bi-encoder kanonik.
- [Formal dkk. (2021). SPLADE: Model Leksikal dan Ekspansi Jarang](https://arxiv.org/abs/2107.05720) — anjing retriever yang belajar-jarang yang menutup kesenjangan dengan padat.
- [Cormack, Clarke, Buttcher (2009). Penggabungan Peringkat Timbal Balik mengungguli Metode Pembelajaran Peringkat Condorcet dan individu](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - makalah RRF.
- [Khattab dan Zaharia (2020). ColBERT: Pencarian Bagian yang Efisien dan Efektif](https://arxiv.org/abs/2004.12832) — pengambilan interaksi yang terlambat.
