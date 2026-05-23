# Inference Bahasa Alami — Keterikatan Tekstual

> "t memerlukan h" berarti pembacaan manusia t akan menyimpulkan h benar. NLI bertugas meramalkan keterlibatan/kontradiksi/netral. Membosankan di permukaan, menahan weight dalam produksi.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 5 · 05 (Analisis Sentimen), Fase 5 · 13 (Menjawab Pertanyaan)
**Waktu:** ~60 menit

## Masalah

kamu membuat ringkasan. Ini menghasilkan ringkasan. Bagaimana kamu tahu ringkasannya tidak mengandung halusinasi?

kamu membuat chatbot. Ia menjawab "ya." Bagaimana kamu tahu bahwa jawabannya didukung oleh bagian yang diambil?

kamu perlu mengklasifikasikan 10.000 artikel berita berdasarkan topik. kamu tidak memiliki label training. Bisakah kamu menggunakan kembali model?

Ketiga masalah tersebut direduksi menjadi Inference Bahasa Alami. NLI bertanya: jika diberi premis `t` dan hipotesis `h`, apakah `h` termasuk dalam `t`, bertentangan, atau netral (tidak berhubungan)?

- **Pemeriksaan halusinasi:** `t` = dokumen sumber, `h` = ringkasan klaim. Bukan keterikatan = halusinasi.
- **QA Beralas:** `t` = bagian yang diambil, `h` = jawaban yang dihasilkan. Bukan keharusan = fabrikasi.
- **Klasifikasi zero-shot:** `t` = dokumen, `h` = label verbal ("Ini tentang olahraga"). Entailment = label prediksi.

Satu tugas, tiga penggunaan produksi. Inilah sebabnya mengapa setiap kerangka evaluasi RAG menyertakan model NLI.

## Konsep

![NLI: klasifikasi tiga arah, premis vs hipotesis](../assets/nli.svg)

**Tiga label.**

- **Penerimaan.** `t` → `h`. "Kucing itu di atas matras" berarti "Ada seekor kucing."
- **Kontradiksi.** `t` → ¬`h`. "Kucing di atas matras" bertentangan dengan "Tidak ada kucing".
- **Netral.** Tidak ada kesimpulan apa pun. Kata "Kucing di atas matras" adalah netral dengan "Kucing itu lapar".

**Bukan persyaratan logis.** NLI adalah inference bahasa *alami* — yang akan disimpulkan oleh pembaca manusia pada umumnya, bukan logika ketat. "John mengajak anjingnya jalan-jalan" berarti "John punya anjing" di NLI, tetapi logika tingkat pertama yang ketat hanya akan mengakuinya jika kamu melakukan aksioma kepemilikan.

**Kumpulan Data.**

- **SNLI** (2015). 570 ribu pasangan beranotasi manusia, keterangan gambar sebagai premis. Domain sempit.
- **MultiNLI** (2017). 433 ribu pasang di 10 genre. Korpus training standar pada tahun 2026.
- **ANLI** (2019). NLI yang bermusuhan. Manusia menulis contoh yang dirancang khusus untuk mendobrak model yang sudah ada. Lebih sulit.
- **DocNLI, KONTROL** (2020–21). Tempat sepanjang dokumen. Menguji inference multi-hop dan distance jauh.

**Arsitekturnya.** Encoder Transformer (BERT, RoBERTa, DeBERTa) berbunyi `[CLS] premise [SEP] hypothesis [SEP]`. Representasi `[CLS]` memberikan softmax 3 arah. Berlatihlah di MNLI, evaluasi berdasarkan tolok ukur yang ada, dapatkan akurasi 90%+ pada pasangan dalam distribusi.

**Zero-shot melalui NLI.** Dengan adanya dokumen dan label kandidat, ubah setiap label menjadi hipotesis ("Teks ini tentang olahraga"). Hitung probabilitas keterlibatan untuk masing-masing. Pilih maks. Inilah mekanisme di balik pipeline `zero-shot-classification` Hugging Face.

## Build

### Langkah 1: jalankan model NLI yang telah dilatih sebelumnya

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

Untuk NLI produksi, `facebook/bart-large-mnli` dan `microsoft/deberta-v3-large-mnli` adalah default terbuka. DeBERTa-v3 menduduki puncak papan peringkat.

### Langkah 2: klasifikasi zero-shot

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```Templatnya adalah "Contoh ini tentang {label}." secara default. Sesuaikan dengan `hypothesis_template`. Tidak diperlukan training data. Tidak ada penyesuaian. Bekerja di luar kotak.

### Langkah 3: pemeriksaan kesetiaan RAG

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

Inilah inti kesetiaan RAGAS. Pisahkan jawaban yang dihasilkan menjadi klaim atom. Periksa setiap klaim terhadap konteks yang diambil. Laporkan pecahan yang diperlukan.

### Langkah 4: pengklasifikasi NLI linting tangan (konseptual)

Lihat `code/main.py` untuk mainan khusus stdlib: premis dan hipotesis dibandingkan melalui tumpang tindih leksikal + deteksi negasi. Tidak kompetitif dengan model Transformer — tetapi model ini menunjukkan bentuk tugasnya: dua teks masuk, label keluar 3 arah, loss = entropi silang pada `{entail, contradict, neutral}`.

## Jebakan

- **Pintasan khusus hipotesis.** Model dapat memprediksi label dari hipotesis saja sebesar ~60% di SNLI karena "tidak", "tidak ada", "tidak pernah" berkorelasi dengan kontradiksi. Dasar yang kuat untuk mendeteksi kebocoran label.
- **Heuristik tumpang tindih leksikal.** Heuristik berikutnya ("setiap urutan diperlukan") lolos SNLI tetapi gagal HANS/ANLI. Gunakan tolok ukur yang berlawanan.
- **Degradasi panjang dokumen.** Model NLI satu kalimat turun 20+ F1 pada premis panjang dokumen. Gunakan model yang dilatih DocNLI untuk konteks yang panjang.
- **Sensitivitas template zero-shot.** "Contoh ini tentang {label}" vs "{label}" vs "Topiknya adalah {label}" dapat meningkatkan akurasi sebesar 10+ poin. Sesuaikan templatnya.
- **Ketidakcocokan domain.** MNLI melatih bahasa Inggris umum. Teks hukum, medis, dan ilmiah memerlukan model NLI khusus domain (misalnya, SciNLI, MedNLI).

## Pakai

Tumpukan tahun 2026:

| Kasus penggunaan | Model |
|---------|-------|
| NLI tujuan umum | `microsoft/deberta-v3-large-mnli` |
| Cepat / tepi | `cross-encoder/nli-deberta-v3-base` |
| Klasifikasi zero-shot (ringan) | `facebook/bart-large-mnli` |
| NLI tingkat dokumen | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Multibahasa | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| Deteksi halusinasi di RAG | Layer NLI di dalam RAGAS/DeepEval |

Pola meta 2026: NLI adalah lakban pemahaman teks. Kapan pun kamu membutuhkan "apakah A mendukung B?" atau "apakah A bertentangan dengan B?" — raih NLI sebelum kamu melakukan panggilan LLM lainnya.

## Kirim

Simpan sebagai `outputs/skill-nli-picker.md`:

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## Latihan

1. **Mudah.** Jalankan `facebook/bart-large-mnli` pada 20 tripel buatan tangan (premis, hipotesis, label) yang mencakup ketiga kelas. Akurasi pengukuran. Tambahkan jebakan "heuristik berikutnya" yang bermusuhan ("Saya tidak memakan kuenya" vs "Saya memakan kuenya") dan lihat apakah jebakannya rusak.
2. **Sedang.** Bandingkan templat zero-shot `"This text is about {label}"` dengan `"The topic is {label}"` dan `"{label}"` di 100 berita utama AG News. Laporan akurasi berayun.
3. **Sulit.** Buat pemeriksa kesetiaan RAG: decomposition klaim atom + NLI per klaim. Evaluasi 50 jawaban yang dihasilkan RAG dengan konteks emas. Ukur angka positif palsu dan negatif palsu vs label tangan.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| NLI | Inference Bahasa Alami | Klasifikasi 3 arah hubungan premis-hipotesis. |
| RTE | Mengenali Keterikatan Tekstual | Nama lama untuk NLI; tugas yang sama. |
| Keterikatan | "t menyiratkan h" | Pembaca pada umumnya akan menyimpulkan h benar jika t. |
| Kontradiksi | "t mengesampingkan h" | Pembaca pada umumnya akan menyimpulkan h salah mengingat t. |
| Netral | "ragu-ragu" | Tidak ada kesimpulan dari t ke h. |
| Klasifikasi tembakan nol | NLI sebagai pengklasifikasi | Nyatakan label sebagai hipotesis, pilih keterlibatan maksimal. |
| Kesetiaan | Apakah jawabannya didukung? | NLI over (konteks yang diambil, jawaban yang dihasilkan). |

## Bacaan Lanjutan

- [Bowman dkk. (2015). Korpus beranotasi besar untuk mempelajari inference bahasa alami](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). Korpus Tantangan Cakupan Luas untuk Pemahaman Kalimat melalui Inference](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie dkk. (2019). NLI Adversarial](https://arxiv.org/abs/1910.14599) — tolok ukur ANLI.
- [Yin, Hay, Roth (2019). Membandingkan Klasifikasi Teks Zero-shot](https://arxiv.org/abs/1909.00161) — NLI-as-classifier.
- [Dia dkk. (2021). DeBERTa: BERT yang disempurnakan dengan decoding dengan Attention Terurai](https://arxiv.org/abs/2006.03654) — pekerja keras NLI 2026.
