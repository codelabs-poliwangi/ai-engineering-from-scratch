# Pemodelan Topik — LDA dan BERTopik

> LDA: dokumen adalah campuran topik, topik adalah distribusi kata-kata. BERTopic: cluster dokumen dalam ruang embedding, cluster adalah topik. Tujuan yang sama, primitif yang berbeda.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 5 · 02 (BoW + TF-IDF), Fase 5 · 03 (Word2Vec)
**Waktu:** ~45 menit

## Masalah

kamu memiliki 10.000 tiket dukungan pelanggan, 50.000 artikel berita, atau 200.000 tweet. kamu perlu mengetahui isi koleksi tersebut tanpa membacanya. kamu tidak memiliki kategori yang diberi label. kamu bahkan tidak tahu berapa banyak kategori yang ada.

Pemodelan topik menjawabnya tanpa pengawasan. Berikan korpusnya, dapatkan kembali sekumpulan kecil topik yang koheren dan, untuk setiap dokumen, distribusikan topik-topik tersebut.

Dua keluarga algoritmik mendominasi. LDA (2003) memperlakukan setiap dokumen sebagai campuran topik laten dan setiap topik sebagai distribusi kata-kata. Inferensinya adalah Bayesian. Itu masih dikirimkan dalam produksi di mana kamu memerlukan penetapan topik keanggotaan campuran dan distribusi probabilitas tingkat kata yang dapat dijelaskan.

BERTopic (2020) mengkodekan dokumen dengan BERT, mengurangi dimension dengan UMAP, mengelompokkan dengan HDBSCAN, dan mengekstrak kata topik melalui TF-IDF berbasis kelas. Ini menang dalam teks pendek, media sosial, dan apa pun yang mengutamakan kesamaan semantik daripada tumpang tindih kata. Satu dokumen mendapat satu topik, yang merupakan batasan untuk konten berdurasi panjang.

Lesson ini membangun intuisi untuk keduanya dan menyebutkan mana yang harus dipilih untuk korpus tertentu.

## Konsep

![Model campuran LDA vs pengelompokan BERTopic](../assets/topic-modeling.svg)

**Kisah generatif LDA.** Setiap topik merupakan distribusi kata. Setiap dokumen merupakan campuran topik. Untuk menghasilkan kata dalam dokumen, ambil sample topik dari campuran dokumen, lalu ambil sample kata dari distribusi topik tersebut. Inference membalikkan hal ini: berdasarkan kata-kata yang diamati, simpulkan distribusi topik per dokumen dan distribusi kata per topik. Pengambilan sample Gibbs yang runtuh atau variasi Bayes yang menghitungnya.

Output LDA utama:

- `doc_topic`: matrix `(n_docs, n_topics)`, setiap baris berjumlah 1 (campuran topik dokumen).
- `topic_word`: matrix `(n_topics, vocab_size)`, setiap baris berjumlah 1 (distribusi kata topik).

**Pipa BERTopic.**

1. Enkode setiap dokumen dengan pengubah kalimat (misalnya `all-MiniLM-L6-v2`). vector 384-redup.
2. Kurangi dimension dengan UMAP menjadi ~5 dimension. Embedding BERT terlalu redup untuk pengelompokan.
3. Cluster dengan HDBSCAN. Berbasis kepadatan, menghasilkan cluster berukuran variabel dan label "outlier".
4. Untuk setiap cluster, hitung TF-IDF berbasis kelas pada dokumen cluster untuk mengekstrak kata-kata teratas.

Outputnya adalah satu topik per dokumen (ditambah label outlier -1). Opsional, keanggotaan lunak melalui vector probabilitas HDBSCAN.

## Build

### Langkah 1: LDA melalui scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

Pemberitahuan: stopwords dihapus, min_df dan max_df memfilter istilah yang jarang dan ada di mana-mana, CountVectorizer (bukan TfidfVectorizer) karena LDA mengharapkan penghitungan mentah.

### Langkah 2: BERTopic (produksi)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

Filter pada `Topic != -1` menghilangkan keranjang outlier BERTopic (dokumen yang tidak dapat dikelompokkan oleh HDBSCAN). `min_topic_size` mengontrol ukuran cluster minimum HDBSCAN; Default perpustakaan BERTopic adalah 10. Contoh ini menyetelnya ke 15 secara eksplisit untuk skala lesson. Untuk kumpulan lebih dari 10.000 dokumen, tambah menjadi 50 atau 100.

### Langkah 3: evaluasi

Kedua metode menghasilkan kata topik. Pertanyaannya adalah apakah kata-kata itu selaras.- **Koherensi topik (c_v).** Menggabungkan NPMI (informasi timbal balik titik yang dinormalisasi) dari pasangan kata teratas dalam konteks jendela geser, menggabungkan skor ke dalam vector topik, dan membandingkan vector tersebut melalui kesamaan kosinus. Lebih tinggi lebih baik. Gunakan `gensim.models.CoherenceModel` dengan `coherence="c_v"`.
- **Keberagaman topik.** Fraksi kata unik di seluruh kata teratas semua topik. Lebih tinggi lebih baik (topik tidak tumpang tindih).
- **Pemeriksaan kualitatif.** Baca kata-kata teratas dari setiap topik. Apakah mereka menyebutkan hal yang nyata? Penghakiman manusia masih menjadi garis pertahanan terakhir.

## Kapan harus memilih yang mana

| Situasi | Pilih |
|-----------|------|
| Teks pendek (tweet, review, headline) | BERTopik |
| Dokumen panjang dengan campuran topik | LDA |
| Tanpa GPU / komputasi terbatas | LDA atau NMF |
| Membutuhkan distribusi multi-topik tingkat dokumen | LDA |
| Integrasi LLM untuk pelabelan topik | BERTopic (dukungan langsung) |
| Penyebaran edge dengan sumber daya terbatas | LDA |
| Koherensi semantik maksimal | BERTopik |

Pertimbangan praktis terbesar adalah panjang dokumen. Embedding BERT terpotong; LDA menghitung pekerjaan berapa pun panjangnya. Untuk dokumen yang lebih panjang dari konteks model embedding, gunakan potongan + agregat atau gunakan LDA.

## Pakai

Tumpukan tahun 2026:

- **BERTopic.** Default untuk teks pendek dan apa pun yang mementingkan semantik.
- **`gensim.models.LdaModel`.** LDA klasik untuk produksi, matang, teruji dalam pertempuran.
- **`sklearn.decomposition.LatentDirichletAllocation`.** LDA yang mudah untuk eksperimen.
- **NMF.** Faktorisasi matrix non-negatif. Alternatif cepat untuk LDA, kualitas sebanding pada teks pendek.
- **Top2Vec.** Desain serupa dengan BERTopic. Komunitas yang lebih kecil tetapi bagus dalam beberapa tolok ukur.
- **FASTopic.** Lebih baru, lebih cepat dari BERTopic pada corpora yang sangat besar.
- **Pelabelan berbasis LLM.** Jalankan pengelompokan apa pun, lalu minta model memberi nama setiap kluster.

## Kirim

Simpan sebagai `outputs/skill-topic-picker.md`:

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## Latihan

1. **Mudah.** Sesuaikan LDA dengan 5 topik pada 20 dataset Newsgroup. Cetak 10 kata teratas per topik. Beri label setiap topik dengan tangan. Apakah algoritme menemukan kategori sebenarnya?
2. **Medium.** Sesuaikan BERTopic pada 20 subset Newsgroup yang sama. Bandingkan jumlah topik yang ditemukan, kata teratas, dan koherensi kualitatif dengan LDA. Manakah yang menampilkan kategori sebenarnya dengan lebih rapi?
3. **Hard.** Hitung koherensi c_v untuk LDA dan BERTopic di korpus kamu. Jalankan masing-masing dengan 5, 10, 20, 50 topik. Koherensi plot vs jumlah topik. Laporkan metode mana yang lebih stabil di seluruh jumlah topik.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Topik | Sesuatu tentang korpus | Distribusi probabilitas atas kata-kata (LDA) atau sekelompok dokumen serupa (BERTopic). |
| Keanggotaan campuran | Dokumen berisi banyak topik | LDA menugaskan setiap dokumen distribusi atas semua topik. |
| UMAP | Dimensionality Reduction | Berbagai pembelajaran yang melestarikan struktur lokal; digunakan dalam BERTopic. |
| HDBSCAN | Pengelompokan kepadatan | Menemukan cluster berukuran variabel; menghasilkan label "noise" (-1) untuk outlier. |
| c_v koherensi | Metrik kualitas topik | Rata-rata informasi timbal balik yang tepat dari kata-kata topik teratas dalam jendela geser. |

## Bacaan Lanjutan- [Blei, Ng, Jordan (2003). Alokasi Dirichlet Laten](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — makalah LDA.
- [Grootendorst (2022). BERTopic: Pemodelan topik saraf dengan prosedur TF-IDF berbasis kelas](https://arxiv.org/abs/2203.05794) - makalah BERTopic.
- [Röder, Keduanya, Hinneburg (2015). Menjelajahi Ruang Ukuran Koherensi Topik](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — makalah yang memperkenalkan c_v dan kawan-kawan.
- [Dokumentasi BERTopic](https://maartengr.github.io/BERTopic/) — referensi produksi. Contoh yang bagus.
