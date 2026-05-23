# Sistem Penjawab Pertanyaan

> Tiga sistem berbentuk QA modern. Rentang ditemukan ekstraktif. Retrieval-augmented mendasarkannya pada dokumen. Jawaban yang dihasilkan secara generatif. Setiap asisten AI modern adalah campuran dari ketiganya.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 5 · 11 (Terjemahan Mesin), Fase 5 · 10 (Mekanisme Attention)
**Waktu:** ~75 menit

## Masalah

Seorang pengguna mengetik "Kapan iPhone pertama diluncurkan?" dan mengharapkan "29 Juni 2007." Bukan "Sejarah Apple panjang dan beragam." Bukan "2007" yang duduk terisolasi tanpa kalimat. Jawaban yang langsung, membumi, dan benar.

Tiga arsitektur telah mendominasi QA selama dekade terakhir.

- **QA Ekstraktif.** Diberikan pertanyaan dan bagian yang diketahui berisi jawabannya, temukan indeks awal dan akhir rentang jawaban di bagian tersebut. SQuAD adalah tolok ukur kanonik.
- **QA domain terbuka.** Bagian ini tidak diberikan. Ambil bagian yang relevan terlebih dahulu, lalu ekstrak atau buat jawabannya. Ini adalah landasan dari setiap pipeline pipa RAG saat ini.
- **QA Generatif / Buku Tertutup.** Model bahasa berukuran besar menjawab dari memori parametriknya. Tidak ada pengambilan. Paling cepat dalam mengambil kesimpulan, paling tidak bisa diandalkan berdasarkan fakta.

Tren pada tahun 2026 bersifat hibrid: ambil beberapa bagian terbaik, lalu gunakan model generatif untuk menjawab berdasarkan bagian tersebut. Itu adalah RAG, dan lesson 14 mencakup separuh pengambilan secara mendalam. Lesson ini membangun setengah QA.

## Konsep

![Arsitektur QA: ekstraktif, augmented pengambilan, generatif](../assets/qa.svg)

**Ekstraktif.** Menyandikan pertanyaan dan bagian bersama dengan Transformer (keluarga BERT). Latih dua kepala yang memprediksi indeks token awal dan akhir dari jawabannya. Loss adalah entropi silang atas posisi valid. Output adalah rentang dari bagian tersebut. Tidak pernah berhalusinasi (berdasarkan konstruksi), tidak pernah menangani pertanyaan yang tidak dapat dijawab oleh bagian tersebut (berdasarkan konstruksi).

**Retrieval-augmented (RAG).** Dua phase. Pertama, retriever menemukan bagian teratas-`k` dari sebuah korpus. Kedua, seorang pembaca (ekstraktif atau generatif) menghasilkan jawaban menggunakan bagian-bagian tersebut. Perpecahan retriever-reader memungkinkan masing-masing dilatih dan dievaluasi secara independen. RAG modern sering menambahkan reranker di antara mereka.

**Generatif.** LLM khusus decoder (GPT, Claude, Llama) menjawab dari weight yang dipelajari. Tidak ada langkah pengambilan. Sangat baik dalam hal pengetahuan umum, sangat buruk dalam hal fakta langka atau terkini. Tingkat halusinasi berkorelasi terbalik dengan frekuensi fakta dalam data pra-training.

## Build

### Langkah 1: QA ekstraktif dengan model yang telah dilatih sebelumnya

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` dilatih tentang SQuAD 2.0, yang mencakup pertanyaan yang tidak dapat dijawab. Secara default, pipeline `question-answering` mengembalikan rentang skor tertinggi bahkan ketika skor nol model menang — ia *tidak* secara otomatis mengembalikan jawaban kosong. Untuk mendapatkan perilaku "tidak ada jawaban" yang eksplisit, teruskan `handle_impossible_answer=True` ke panggilan pipeline pipa: pipeline pipa kemudian mengembalikan jawaban kosong hanya ketika skor nol melebihi setiap skor rentang. Selalu periksa bidang `score`.

### Langkah 2: pipeline yang ditambah pengambilan (sketsa)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

Pipa dua phase. Dense retriever (Sentence-BERT) menemukan bagian yang relevan berdasarkan kesamaan semantik. Pembaca ekstraktif (RoBERTa-SQuAD) mengambil rentang jawaban dari gabungan bagian teratas. Bekerja pada corpora kecil. Untuk korpus sejuta dokumen, gunakan FAISS atau database vector.

### Langkah 3: generatif dengan RAG

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```Pola cepat itu penting. Memberi tahu model secara eksplisit agar sesuai dengan konteksnya dan menampilkan "Saya tidak tahu" ketika konteksnya tidak cukup akan mengurangi tingkat halusinasi sebesar 40-60% dibandingkan dengan dorongan yang naif. Pola yang lebih rumit menambahkan kutipan, skor kepercayaan, dan ekstraksi terstruktur.

### Langkah 4: evaluasi yang mencerminkan dunia nyata

SQuAD menggunakan **Pencocokan Akurat (EM)** dan **F1 tingkat token**. EM adalah pencocokan ketat setelah normalisasi (huruf kecil, hapus tanda baca, hapus artikel) — baik prediksinya sama persis atau skornya 0. F1 dihitung berdasarkan token yang tumpang tindih antara prediksi dan referensi dan memberikan kredit parsial. Kedua parafrase di bawah kredit: "29 Juni 2007" vs "29 Juni 2007" biasanya mendapat 0 EM (normalisasi jeda ordinal) tetapi masih menghasilkan F1 yang besar dari token yang tumpang tindih.

Untuk QA produksi:

- **Akurasi jawaban** (dinilai oleh LLM atau dinilai oleh manusia, karena metrik tidak menangkap kesetaraan semantik).
- **Keakuratan kutipan.** Apakah bagian yang dikutip benar-benar mendukung jawabannya? Sepele untuk memeriksa secara otomatis dengan kecocokan string antara kutipan yang dihasilkan dan bagian yang diambil.
- **Penolakan kalibrasi.** Jika jawabannya tidak ada dalam bagian yang diambil, apakah sistem mengatakan "Saya tidak tahu" dengan benar? Ukur tingkat kepercayaan palsu.
- **Retrieval recall.** Sebelum mengevaluasi pembaca, ukur apakah retriever mendapatkan bagian yang tepat ke atas-`k`. Seorang pembaca tidak dapat memperbaiki bagian yang hilang.

### RAGAS: kerangka evaluasi produksi tahun 2026

`RAGAS` dibuat khusus untuk sistem RAG dan merupakan default pengiriman pada tahun 2026. Sistem ini mencetak empat dimension tanpa memerlukan referensi emas:

- **Kesetiaan.** Apakah setiap klaim dalam jawaban berasal dari konteks yang diambil? Diukur dengan keterlibatan berbasis NLI. Metrik halusinasi utama kamu.
- **Relevansi jawaban.** Apakah jawaban menjawab pertanyaan? Diukur dengan menghasilkan pertanyaan hipotetis dari jawaban dan membandingkannya dengan pertanyaan sebenarnya.
- **Ketepatan konteks.** Dari potongan yang diambil, pecahan manakah yang benar-benar relevan? Presisi rendah = kebisingan saat diminta.
- **Pengembalian konteks.** Apakah kumpulan yang diambil berisi semua informasi yang diperlukan? Ingatan rendah = pembaca tidak dapat berhasil.

Penilaian bebas referensi memungkinkan kamu mengevaluasi lalu lintas produksi langsung tanpa jawaban emas yang dikurasi. Lapisi LLM sebagai juri di atas untuk pertanyaan terbuka di mana metrik pencocokan tepat tidak berguna.

`pip install ragas`. Pasang retriever + pembaca kamu. Dapatkan empat scalar per kueri. Peringatan tentang regresi.

## Pakai

Tumpukan tahun 2026.

| Kasus penggunaan | Direkomendasikan |
|---------|-------------|
| Diberikan bagian, temukan rentang jawaban | `deepset/roberta-base-squad2` |
| Pada korpus tetap, buku tertutup tidak dapat diterima | RAG: retriever padat + pembaca LLM |
| Real-time melalui penyimpanan dokumen | RAG dengan retriever hybrid (BM25 + padat) + reranker (lesson 14) |
| QA Percakapan (pertanyaan lanjutan) | LLM dengan riwayat percakapan + RAG di setiap giliran |
| Domain yang sangat faktual dan teregulasi | Ekstraktif atas korpus yang berwenang; tidak pernah generatif sendirian |

QA ekstraktif sudah ketinggalan zaman pada tahun 2026 karena RAG dengan LLM menangani lebih banyak kasus. Laporan ini tetap dikirimkan dalam konteks yang memerlukan kutipan literal: penelitian hukum, kepatuhan terhadap peraturan, alat audit.

## Kirim

Simpan sebagai `outputs/skill-qa-architect.md`:

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## Latihan1. **Mudah.** Siapkan jalur ekstraktif SQuAD di atas pada 10 bagian Wikipedia. Kerajinan tangan 10 pertanyaan. Ukur seberapa sering jawabannya benar. kamu akan melihat 7-9 benar jika bagian dan pertanyaannya bersih.
2. **Sedang.** Tambahkan pengklasifikasi penolakan. Ketika skor pengambilan teratas berada di bawah ambang batas (katakanlah 0,3 kosinus), kembalikan "Saya tidak tahu" alih-alih menelepon pembaca. Setel ambang batas pada set yang ditahan.
3. **Sulit.** Build pipeline RAG pada korpus 10.000 dokumen pilihan kamu. Terapkan pengambilan hibrid (BM25 + padat) dengan fusi RRF (lihat lesson 14). Ukur keakuratan jawaban dengan dan tanpa langkah hibrid. Dokumentasikan jenis pertanyaan mana yang paling bermanfaat.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| QA Ekstraktif | Temukan rentang jawabannya | Memprediksi indeks awal dan akhir dari jawaban dalam bagian tertentu. |
| QA domain terbuka | QA atas korpus | Tidak ada bagian tertentu; harus mengambil lalu menjawab. |
| RAG | Ambil lalu hasilkan | Generasi yang ditambah pengambilan. Retriever + pipeline pembaca. |
| Pasukan | Tolok ukur kanonik | Kumpulan Data Menjawab Pertanyaan Stanford. Metrik EM + F1. |
| Halusinasi | Jawaban yang dibuat-buat | Output pembaca tidak didukung oleh konteks yang diambil. |
| Penolakan kalibrasi | Tahu kapan harus tutup mulut | Sistem dengan benar mengatakan "Saya tidak tahu" ketika tidak dapat menjawab. |

## Bacaan Lanjutan

- [Rajpurkar dkk. (2016). SQuAD: 100.000+ Pertanyaan untuk Pemahaman Teks oleh Mesin](https://arxiv.org/abs/1606.05250) — makalah benchmark.
- [Karpukhin dkk. (2020). Pengambilan Bagian Padat untuk QA Domain Terbuka](https://arxiv.org/abs/2004.04906) — DPR, pengambilan padat kanonik untuk QA.
- [Lewis dkk. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - makalah yang diberi nama RAG.
- [Gao dkk. (2023). Retrieval-Augmented Generation untuk Large Language Model: Sebuah Survei](https://arxiv.org/abs/2312.10997) — survei RAG yang komprehensif.
