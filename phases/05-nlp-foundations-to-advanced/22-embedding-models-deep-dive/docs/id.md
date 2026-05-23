# Embed Model — Penyelaman Mendalam 2026

> Word2Vec memberi kamu vector per kata. Model embedding modern memberi kamu vector per bagian, lintas bahasa, dengan tampilan jarang, padat, dan multi-vector, yang ukurannya sesuai dengan indeks kamu. Pilih yang salah dan RAG kamu mengambil hal yang salah.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 5 · 03 (Word2Vec), Fase 5 · 14 (Pengambilan Informasi)
**Waktu:** ~60 menit

## Masalah

Sistem RAG kamu mengambil bagian yang salah sebanyak 40%. Pelakunya jarang sekali adalah database vector atau prompt. Ini adalah model embedding.

Memilih embedding pada tahun 2026 berarti memilih lima sumbu:

1. **Padat vs jarang vs multi-vector.** Satu vector per bagian, atau satu per token, atau sekumpulan kata berbobot jarang.
2. **Cakupan bahasa.** Model bahasa Inggris monolingual tetap unggul dalam tugas bahasa Inggris saja. Model multibahasa menang jika corpora digabungkan.
3. **Panjang konteks.** 512 token vs 8.192 vs 32.768 — dan kapasitas efektif sebenarnya sering kali mencapai 60-70% dari kapasitas maksimum yang diiklankan.
4. **Anggaran dimension.** 3.072 float dengan presisi penuh = 12 KB per vector. Pada 100 juta vector, penyimpanannya adalah $1.300/bulan. Pemotongan Matryoshka memotong 4× ini.
5. **Terbuka vs dihosting.** Weight terbuka berarti kamu mengontrol tumpukan dan data. Dihosting berarti kamu menukar kendali dengan yang selalu terbaru.

Lesson ini menyebutkan tradeoff sehingga kamu dapat memilih berdasarkan bukti, bukan berdasarkan apa pun yang populer pada kuartal terakhir.

## Konsep

![Embedding padat, jarang, dan multi-vector](../assets/embedding-modes.svg)

**Sematan padat.** Satu vector per bagian (biasanya 384-3.072 dimension). Kesamaan kosinus memberi peringkat pada bagian-bagian berdasarkan kedekatan semantik. OpenAI `text-embedding-3-large`, mode padat BGE-M3, Voyage-3. Pilihan bawaan.

**Sematan yang jarang.** Bergaya SPLADE. Sebuah Transformer memprediksi weight untuk setiap token vocab, lalu menghilangkan sebagian besar bobotnya. Hasilnya adalah vector renggang dengan ukuran |vocab|. Menangkap pencocokan leksikal (seperti BM25) tetapi dengan weight istilah yang dipelajari. Kuat pada kueri yang banyak kata kunci.

**Multi-vector (interaksi terlambat).** ColBERTv2, Jina-ColBERT. Satu vector per token. Penilaian dengan MaxSim: untuk setiap token kueri, temukan token dokumen yang paling mirip, jumlahkan skornya. Lebih mahal untuk disimpan dan dinilai, tetapi unggul dalam kueri panjang dan corpora khusus domain.

**BGE-M3: ketiganya sekaligus.** Model tunggal menghasilkan representasi yang padat, jarang, dan multi-vector secara bersamaan. Masing-masing dapat ditanyakan secara independen; skor menyatu melalui jumlah tertimbang. Default tahun 2026 ketika kamu menginginkan fleksibilitas dari satu pos pemeriksaan.

**Pembelajaran Representasi Matryoshka.** Dilatih agar N dimension pertama vector membentuk embedding mandiri yang berguna. Pangkas vector 1.536 redup menjadi 256 redup dan bayar akurasi ~1% untuk penghematan penyimpanan 6×. Didukung oleh OpenAI text-3, Cohere v4, Voyage-4, Jina v5, Gemini Embedding 2, Nomic v1.5+.

### Papan peringkat MTEB menceritakan sebagian cerita

Tolok Ukur Embedding Teks Besar — 56 tugas di 8 jenis tugas saat peluncuran (2022), diperluas hingga 100+ tugas di MTEB v2. Pada awal tahun 2026, Gemini Embedding 2 pengambilan puncak (67,71 MTEB-R). Cohere embed-v4 memimpin umum (65,2 MTEB). BGE-M3 mengungguli multibahasa kelas terbuka (63.0). Papan peringkat diperlukan tetapi tidak cukup — selalu gunakan tolok ukur pada domain kamu.

### Pola tiga tingkat

| Kasus penggunaan | Pola |
|----------|---------|
| Lintasan pertama yang cepat | Bi-encoder padat (BGE-M3, teks-3-kecil) |
| Peningkatan ingat | Jarang (SPLADE, BGE-M3 jarang) + sekering RRF |
| Presisi di 50 teratas | Pemeringkatan ulang multi-vector (ColBERTv2) atau lintas-encoder |

Sebagian besar tumpukan produksi menggunakan ketiganya.## Build

### Langkah 1: garis dasar — embedding padat dengan Sentence-BERT

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` membuat perkalian titik sama dengan kesamaan kosinus. Selalu atur itu.

### Langkah 2: Pemotongan Matryoshka

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

Normalisasi ulang setelah pemotongan. Nomic v1.5, OpenAI text-3, dan Voyage-4 dilatih sehingga tidak ada loss untuk beberapa level pertama. Model non-Matryoshka (Sentence-BERT asli) menurun tajam saat dipotong.

### Langkah 3: Multifungsi BGE-M3

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

Tiga indeks, satu panggilan inference. Penggabungan skor:

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

Sesuaikan weight pada domain kamu.

### Langkah 4: Evaluasi MTEB pada tugas khusus

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

Jalankan model kandidat kamu pada subset *perwakilan*. Jangan percaya hanya pada peringkat papan peringkat — domain kamu penting.

### Langkah 5: kosinus linting tangan dari awal

Lihat `code/main.py`. Embedding Trik Hashing rata-rata (khusus stdlib). Tidak bersaing dengan embedding Transformer, tetapi menunjukkan bentuk: tokenize → vector → normalisasi → perkalian titik.

## Jebakan

- **Model yang sama untuk kueri dan dokumen.** Beberapa model (Voyage, Jina-ColBERT) menggunakan pengkodean asimetris — kueri dan dokumen melewati jalur yang berbeda. Selalu periksa kartu model.
- **Awalan tidak ada.** `bge-*` model perlu `"Represent this sentence for searching relevant passages: "` ditambahkan ke kueri. Kesenjangan recall 3-5 poin jika kamu lupa.
- **Matryoshka yang terlalu dipangkas.** 1.536 → 256 biasanya aman. 1.536 → 64 bukan. Validasi set evaluasi kamu.
- **Pemotongan konteks.** Sebagian besar model memotong input secara diam-diam melebihi panjang maksimalnya. Dokumen yang panjang perlu dipecah (lihat lesson 23).
- **Mengabaikan ekor latensi.** Skor MTEB menyembunyikan latensi p99. Model 600M mungkin mengalahkan model 335M dengan 2 poin tetapi biayanya 3x lebih mahal per kueri.

## Pakai

Tumpukan tahun 2026:

| Situasi | Pilih |
|-----------|------|
| Hanya dalam bahasa Inggris, cepat, API | `text-embedding-3-large` atau `voyage-3-large` |
| Kelas terbuka, Bahasa Inggris | `BAAI/bge-large-en-v1.5` |
| Kelas terbuka, multibahasa | `BAAI/bge-m3` atau `Qwen3-Embedding-8B` |
| Konteks panjang (32rb+) | Voyage-3-besar, Cohere semat-v4, Qwen3-Embedding-8B |
| Penerapan khusus CPU | Nomic Embed v2 (137 juta parameter, MoE) |
| Penyimpanan terbatas | Kuantisasi Matryoshka-terpotong + int8 |
| Kueri dengan banyak kata kunci | Tambahkan SPLADE renggang, sekering RRF dengan padat |

Pola 2026: mulai dengan BGE-M3 atau text-3-large, evaluasi domain kamu dengan MTEB, tukar jika model khusus domain menang lebih dari 3 poin.

## Kirim

Simpan sebagai `outputs/skill-embedding-picker.md`:

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## Latihan

1. **Mudah.** Enkode 100 kalimat dengan `bge-small-en-v1.5` pada redup penuh (384), lalu pada Matryoshka 128. Ukur penurunan MRR pada 10 kueri.
2. **Sedang.** Bandingkan BGE-M3 padat, jarang, dan colbert pada 500 bagian dari domain kamu. Yang mana yang menang di recall@10? Apakah fusi RRF mengalahkan mode tunggal terbaik?
3. **Sulit.** Jalankan MTEB pada tiga model kandidat di 2 tugas domain teratas kamu. Laporkan skor MTEB, latensi p99 pada kumpulan 100 kueri, dan $/1 juta kueri. Pilih yang optimal Pareto.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Embedding padat | Vector | Satu vector berukuran tetap per teks. Kesamaan kosinus untuk pemeringkatan. |
| Embedding jarang | Mempelajari BM25 | Satu weight per token vocab; kebanyakan nol; dilatih secara end-to-end. |
| Multi-vector | Gaya ColBERT | Satu vector per token; penilaian MaxSim; indeks lebih besar, daya ingat lebih baik. |
| Matryoshka | Trik boneka rusia | N peredupan pertama adalah embedding kecil yang valid. |
| MTEB | Tolok ukur | Tolok Ukur Embedding Teks Besar — ​​56 tugas saat diluncurkan, 100+ di v2. |
| BEIR | Tolok ukur pengambilan | 18 tugas pengambilan gambar nol; sering dikutip karena ketahanan lintas domain. |
| Pengkodean asimetris | Kueri ≠ jalur dokumen | Model menggunakan proyeksi berbeda untuk kueri dan dokumen. |

## Bacaan Lanjutan

- [Reimers, Gurevych (2019). Kalimat-BERT](https://arxiv.org/abs/1908.10084) — makalah bi-encoder.
- [Muennighoff dkk. (2022). MTEB: Tolok Ukur Embedding Teks Besar-besaran](https://arxiv.org/abs/2210.07316) — makalah papan peringkat.
- [Chen dkk. (2024). BGE-M3: Multi-bahasa, Multi-fungsi, Multi-perincian](https://arxiv.org/abs/2402.03216) — model tiga mode terpadu.
- [Kusupati dkk. (2022). Pembelajaran Representasi Matryoshka](https://arxiv.org/abs/2205.13147) — tujuan training tangga dimension.
- [Santhanam dkk. (2022). ColBERTv2: Pengambilan yang Efektif dan Efisien melalui Interaksi Terlambat Ringan](https://arxiv.org/abs/2112.01488) — interaksi yang terlambat dalam produksi.
- [Papan peringkat MTEB di Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — peringkat langsung.
