# Peringkasan Teks

> Sistem ekstraktif memberi tahu kamu apa yang tertulis dalam dokumen tersebut. Sistem abstraktif memberi tahu kamu apa yang dimaksudkan penulis. Tugas yang berbeda, jebakan yang berbeda.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 5 · 02 (BoW + TF-IDF), Fase 5 · 11 (Terjemahan Mesin)
**Waktu:** ~75 menit

## Masalah

Artikel berita sepanjang 2.000 kata masuk ke feed kamu. kamu membutuhkan 120 kata yang dapat menangkapnya. kamu dapat memilih tiga kalimat terpenting dari artikel (ekstraktif) atau menulis ulang konten dengan kata-kata kamu sendiri (abstraksi). Keduanya disebut ringkasan. Itu adalah masalah yang sangat berbeda.

Peringkasan ekstraktif adalah masalah pemeringkatan. Skor setiap kalimat, kembalikan yang teratas-`k`. Outputnya selalu gramatikal karena diangkat kata demi kata. Risikonya adalah hilangnya konten yang didistribusikan ke seluruh artikel.

Peringkasan abstraktif adalah masalah generasi. Sebuah Transformer menghasilkan teks baru yang dikondisikan pada input. Keluarannya lancar dan padat, namun mungkin berhalusinasi tentang fakta yang tidak disebutkan dalam sumbernya. Risikonya adalah pemalsuan yang penuh percaya diri.

Lesson ini membangun keduanya, dengan mode kegagalan yang dimiliki masing-masing.

## Konsep

![TextRank Ekstraktif vs Transformer abstraktif](../assets/summarization.svg)

**Ekstraktif.** Perlakukan artikel sebagai grafik yang simpulnya adalah kalimat dan sisinya adalah persamaan. Jalankan PageRank (atau semacamnya) pada grafik untuk menilai kalimat berdasarkan seberapa terhubungnya kalimat tersebut dengan hal lainnya. Kalimat dengan skor tertinggi adalah ringkasan. Implementasi kanoniknya adalah **TextRank** (Mihalcea dan Tarau, 2004).

**Abstraksi.** Menyempurnakan encoder-decoder Transformer (BART, T5, Pegasus) pada pasangan ringkasan dokumen. Pada inference, model membaca dokumen dan menghasilkan ringkasan token demi token melalui attention silang. Pegasus khususnya menggunakan tujuan pra-training kalimat-celah yang membuatnya sangat baik dalam meringkas tanpa banyak penyesuaian.

Evaluasi dengan **ROUGE** (Siswa Berorientasi Ingatan untuk Evaluasi Gisting). ROUGE-1 dan ROUGE-2 mendapat skor unigram dan bigram tumpang tindih. ROUGE-L mendapat skor urutan umum terpanjang. Lebih tinggi lebih baik tetapi 40 ROUGE-L "bagus" dan 50 "luar biasa". Setiap surat kabar melaporkan ketiganya. Gunakan paket `rouge-score`.

## Build

### Langkah 1: TextRank (ekstraktif)

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

Dua hal yang patut disebutkan. Fungsi kesamaan menggunakan tumpang tindih kata yang dinormalisasi log, yang merupakan varian TextRank asli. Kosinus vector TF-IDF juga berfungsi. Faktor redaman 0,85 dan jumlah iterasi adalah default PageRank.

### Langkah 2: abstraksi dengan BART

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN disesuaikan dengan korpus CNN/DailyMail. Ini menghasilkan ringkasan gaya berita yang out of the box. Untuk domain lain (makalah ilmiah, dialog, hukum), gunakan pos pemeriksaan Pegasus yang sesuai atau sesuaikan data target kamu.

### Langkah 3: Evaluasi ROUGE

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

Selalu gunakan stemming. Tanpanya, "running" dan "run" dihitung sebagai kata yang berbeda dan ROUGE dihitung lebih rendah.

### Beyond ROUGE (evaluasi ringkasan tahun 2026)

ROUGE telah menjadi metrik peringkasan yang dominan selama dua puluh tahun dan tidak cukup untuk digunakan pada tahun 2026. Sebuah meta-analisis skala besar dari makalah NLG menunjukkan:- **BERTScore** (kesamaan embedding kontekstual) mulai berkembang hingga tahun 2023 dan kini dilaporkan bersama ROUGE di sebagian besar makalah ringkasan.
- **BARTScore** memperlakukan evaluasi sebagai pembuatan: nilai ringkasan berdasarkan seberapa besar kemungkinan BART yang sudah terlatih menugaskannya berdasarkan sumbernya.
- **MoverScore** (Distance Penggerak Bumi atas embedding kontekstual) mencapai posisi teratas pada tolok ukur ringkasan tahun 2025 karena mampu menangkap tumpang tindih semantik lebih baik daripada ROUGE.
- **FactCC** dan **kesetiaan berbasis QA** adalah hal yang umum pada tahun 2021-2023, kini sering digantikan oleh **G-Eval** (rantai cepat GPT-4 yang menilai koherensi, konsistensi, kelancaran, relevansi dengan penalaran rantai pemikiran).
- **G-Eval** dan pendekatan juri LLM serupa cocok dengan penilaian manusia ~80% ketika rubrik dirancang dengan baik.

Rekomendasi produksi: laporkan ROUGE-L untuk perbandingan warisan, BERTScore untuk tumpang tindih semantik, G-Eval untuk koherensi dan faktualitas. Kalibrasi dengan 50-100 ringkasan berlabel manusia.

### Langkah 4: masalah faktualitas

Ringkasan abstrak rentan terhadap halusinasi. Ringkasan ekstraktif memiliki risiko halusinasi yang jauh lebih rendah karena keluarannya diambil secara verbatim dari sumbernya, meskipun ringkasan tersebut masih dapat menyesatkan jika kalimat sumber didekontekstualisasikan, ketinggalan jaman, atau dikutip tidak berurutan. Inilah satu-satunya alasan terbesar mengapa sistem produksi masih memilih metode ekstraktif untuk konten yang berkaitan dengan kepatuhan.

Jenis halusinasi yang perlu disebutkan:

- **Pertukaran entitas.** Sumber mengatakan "John Smith." Ringkasan mengatakan "John Brown."
- **Penyimpangan angka.** Sumber mengatakan "25.000." Ringkasannya mengatakan "25 juta."
- **Pembalikan polaritas.** Sumber mengatakan "menolak tawaran". Ringkasan mengatakan "menerima tawaran itu."
- **Penemuan fakta.** Sumber tidak menyebutkan CEO. Ringkasan mengatakan CEO menyetujuinya.

Pendekatan evaluasi yang berhasil:

- **FactCC.** Pengklasifikasi biner yang dilatih tentang keterlibatan antara kalimat sumber dan kalimat ringkasan. Memprediksi faktual/tidak faktual.
- **Faktualitas berbasis QA.** Ajukan pertanyaan model QA yang jawabannya ada di sumbernya. Jika ringkasan mendukung jawaban yang berbeda, tandai.
- **F1 tingkat entitas.** Bandingkan entitas bernama dalam sumber vs ringkasan. Entitas yang hanya ada dalam ringkasan patut dicurigai.

Untuk segala hal yang berhubungan dengan pengguna yang mengutamakan faktualitas (berita, medis, hukum, keuangan), ekstraktif adalah pilihan yang lebih aman. Abstraksi membutuhkan pemeriksaan faktualitas dalam loop.

## Pakai

Tumpukan tahun 2026:

| Kasus penggunaan | Direkomendasikan |
|---------|-------------|
| Berita, ringkasan 3-5 kalimat, Bahasa Inggris | `facebook/bart-large-cnn` |
| Karya Ilmiah | `google/pegasus-pubmed` atau T5 | yang disetel
| Multi-dokumen, bentuk panjang | LLM apa pun dengan konteks 32k+, diminta |
| Ringkasan dialog | `philschmid/bart-large-cnn-samsum` |
| Ekstraktif, risiko halusinasi rendah berdasarkan konstruksi | TextRank atau LSA / LexRank `sumy` |

LLM dengan konteks panjang sering kali mengalahkan model khusus pada tahun 2026 ketika komputasi tidak menjadi kendala. Pengorbanannya adalah biaya dan reproduktifitas; model khusus memberikan output yang lebih konsisten.

## Kirim

Simpan sebagai `outputs/skill-summary-picker.md`:

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## Latihan1. **Mudah.** Jalankan TextRank pada 5 artikel berita. Bandingkan 3 kalimat teratas dengan ringkasan referensi. Ukur ROUGE-L. kamu akan melihat 30-45 ROUGE-L di artikel bergaya CNN/DailyMail.
2. **Medium.** Menerapkan faktualitas tingkat entitas: mengekstrak entitas bernama dari sumber dan ringkasan (spaCy), menghitung penarikan kembali entitas sumber dalam ringkasan, dan ketepatan entitas ringkasan terhadap sumber. Presisi tinggi dan perolehan rendah berarti aman namun singkat; presisi rendah berarti entitas berhalusinasi.
3. **Sulit.** Bandingkan BART-large-CNN dengan LLM (Claude atau GPT-4) di 50 artikel CNN/DailyMail. Laporkan ROUGE-L, faktualitas (menurut entitas F1), dan biaya per ringkasan. Dokumentasikan di mana masing-masing pihak menang.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Ekstraktif | Pilih kalimat | Mengembalikan kalimat kata demi kata dari sumbernya. Tidak pernah berhalusinasi. |
| Abstraksi | Tulis ulang | Hasilkan teks baru yang dikondisikan pada sumbernya. Bisa berhalusinasi. |
| PEMERAH | Metrik ringkasan | N-gram/LCS tumpang tindih antara output sistem dan referensi. |
| Peringkat Teks | Ekstraktif berbasis grafik | PageRank atas grafik kesamaan kalimat. |
| Faktualitas | Apakah benar | Apakah ringkasan klaim didukung oleh sumbernya. |
| Halusinasi | Konten buatan | Konten dalam ringkasan yang tidak didukung sumbernya. |

## Bacaan Lanjutan

- [Mihalcea dan Tarau (2004). TextRank: Membawa Keteraturan ke dalam Teks](https://aclanthology.org/W04-3252/) — makalah kanonik ekstraktif.
- [Lewis dkk. (2019). BART: Pra-training Denoising Sequence-to-Sequence](https://arxiv.org/abs/1910.13461) - makalah BART.
- [Zhang dkk. (2019). PEGASUS: Pra-training dengan Kalimat Celah yang Diekstraksi](https://arxiv.org/abs/1912.08777) — Pegasus dan tujuan kalimat celah.
- [Lin (2004). ROUGE: Paket untuk Evaluasi Ringkasan Otomatis](https://aclanthology.org/W04-1013/) — Makalah ROUGE.
- [Maynez dkk. (2020). Tentang Kesetiaan dan Faktualitas dalam Peringkasan Abstraktif](https://arxiv.org/abs/2005.00661) - makalah lanskap faktualitas.
