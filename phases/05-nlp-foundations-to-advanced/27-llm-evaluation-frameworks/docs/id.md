# Evaluasi LLM — RAGAS, DeepEval, G-Eval

> Pencocokan tepat dan F1 kehilangan kesetaraan semantik. Tinjauan manusia tidak berskala. LLM sebagai juri adalah jawaban produksi — dengan kalibrasi yang cukup untuk mempercayai nomor tersebut.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 5 · 13 (Menjawab Pertanyaan), Phase 5 · 14 (Pengambilan Informasi)
**Waktu:** ~75 menit

## Masalah

Sistem RAG kamu menjawab: "29 Juni 2007."
Referensi emasnya adalah: "29 Juni 2007."
Skor Pertandingan Akurat 0. Skor F1 ~75%. Manusia akan mendapat skor 100%.

Sekarang kalikan dengan 10.000 kasus uji. Kalikan lagi dengan setiap perubahan pada retriever, chunking, prompt, atau model. kamu memerlukan evaluator yang memahami maknanya, bekerja dengan murah dalam skala besar, tidak berbohong tentang regresi, dan menampilkan mode kegagalan yang tepat.

Tahun 2026 memiliki tiga framework yang mengatasi masalah ini.

- **RAGAS.** Penilaian Pembuatan Augmentasi Pengambilan. Empat metrik RAG (kesetiaan, relevansi jawaban, presisi konteks, ingatan konteks) dengan backend juri NLI + LLM. Didukung penelitian, ringan.
- **DeepEval.** Pytest untuk LLM. G-Eval, penyelesaian tugas, halusinasi, metrik bias. CI/CD-asli.
- **G-Eval.** Metode (dan metrik DeepEval): LLM sebagai juri dengan rangkaian pemikiran, kriteria khusus, skor 0-1.

Ketiganya bersandar pada LLM sebagai juri. Lesson ini membangun intuisi untuk metode dan layer kepercayaan di sekitarnya.

## Konsep

![Empat dimension evaluasi, arsitektur LLM sebagai juri](../assets/llm-evaluation.svg)

**LLM-sebagai-hakim.** Ganti metrik statis dengan LLM yang menilai output berdasarkan rubrik. Diberikan `(query, context, answer)`, meminta juri LLM: "Skor 0-1 untuk kesetiaan." Kembalikan skornya.

Mengapa ini berhasil: LLM memperkirakan penilaian manusia dengan biaya yang sangat kecil. GPT-4o-mini dengan harga ~$0,003 per kasus yang diberi skor memungkinkan proses evaluasi regresi 1000 sample dengan biaya di bawah $5.

Mengapa gagal secara diam-diam:

1. **Bias hakim.** Juri lebih memilih jawaban yang lebih panjang, jawaban dari kelompok model mereka sendiri, jawaban yang sesuai dengan gaya cepat.
2. **Kegagalan penguraian JSON.** JSON buruk → skor NaN → diam-diam dikecualikan dari agregat. Pengguna RAGAS mengetahui penderitaan ini. Gerbang dengan mode coba/kecuali + kegagalan eksplisit.
3. **Beralih ke versi model.** Mengupgrade juri akan mengubah setiap metrik. Model + versi juri beku.

**RAG empat.**

| Metrik | Pertanyaan | Bagian Belakang |
|--------|----------|---------|
| Kesetiaan | Apakah setiap klaim dalam jawaban berasal dari konteks yang diambil? | Keterlibatan berbasis NLI |
| Relevansi jawaban | Apakah jawabannya menjawab pertanyaan tersebut? | Hasilkan pertanyaan hipotetis dari jawaban; bandingkan dengan pertanyaan sebenarnya |
| Ketepatan konteks | Dari potongan yang diambil, pecahan manakah yang relevan? | Hakim LLM |
| Ingatan konteks | Apakah pengambilan mengembalikan semua yang dibutuhkan? | Hakim LLM menentang jawaban emas |

**G-Eval.** Tentukan kriteria khusus: "Apakah jawaban mengutip sumber yang benar?" Kerangka kerja tersebut secara otomatis diperluas ke langkah-langkah evaluasi rangkaian pemikiran, lalu mendapat skor 0-1. Baik untuk dimension kualitas spesifik domain yang tidak dicakup RAGAS.

**Kalibrasi.** Jangan pernah mempercayai skor juri mentah sampai kamu memiliki korelasi dengan label manusia. Jalankan 100 contoh berlabel tangan. Hakim plot vs manusia. Hitung Spearman rho. Jika rho < 0,7, rubrik juri kamu perlu diperbaiki.

## Build

### Langkah 1: kesetiaan dengan NLI (gaya RAGAS)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

Uraikan jawabannya menjadi klaim atom. NLI-periksa setiap klaim terhadap konteks yang diambil. Kesetiaan = sebagian kecil didukung.

### Langkah 2: relevansi jawaban

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```Jika jawabannya menyiratkan pertanyaan yang berbeda dari yang ditanyakan, relevansinya akan menurun.

### Langkah 3: Metrik khusus G-Eval

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

Langkah-langkah evaluasinya adalah rubrik. Langkah-langkah eksplisit lebih stabil daripada prompt implisit "skor 0-1".

### Langkah 4: Gerbang CI

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

Kirim sebagai file pytest. Jalankan di setiap PR. Blokir penggabungan pada regresi.

### Langkah 5: evaluasi mainan dari awal

Lihat `code/main.py`. Perkiraan kesetiaan khusus Stdlib (tumpang tindih klaim jawaban dengan konteks) dan relevansi (tumpang tindih token jawaban dengan token pertanyaan). Bukan produksi. Menunjukkan bentuknya.

## Jebakan

- **Tidak ada kalibrasi.** Juri dengan korelasi 0,3 terhadap label manusia adalah kebisingan. Memerlukan proses kalibrasi sebelum pengiriman.
- **Evaluasi diri.** Menggunakan LLM yang sama untuk menghasilkan dan menilai meningkatkan skor sebesar 10-20%. Gunakan keluarga teladan yang berbeda untuk hakim.
- **Bias posisi dalam penilaian berpasangan.** Juri lebih memilih opsi pertama yang disajikan. Selalu acak urutannya dan jalankan keduanya.
- **Agregat mentah menyembunyikan kegagalan.** Skor rata-rata 0,85 sering kali menyembunyikan 5% kegagalan besar. Selalu periksa kuantil bawah.
- **Kebusukan dataset emas.** Kumpulan evaluasi tak berversi yang melayang seiring waktu mematahkan perbandingan longitudinal. Tandai dataset dengan setiap perubahan.
- **Biaya LLM.** Dalam skala besar, keputusan hakim mendominasi biaya. Gunakan model termurah yang memenuhi ambang kalibrasi. GPT-4o-mini, Claude Haiku, Mistral-kecil.

## Pakai

Tumpukan tahun 2026:

| Kasus penggunaan | Kerangka |
|---------|-----------|
| Pemantauan kualitas RAG | RAGAS (4 metrik) |
| Gerbang regresi CI/CD | DeepEval + pytest |
| Kriteria domain khusus | G-Eval dalam DeepEval |
| Pemantauan lalu lintas langsung online | RAGAS dengan mode bebas referensi |
| Pemeriksaan langsung manusia | LangSmith atau Phoenix dengan UI anotasi |
| Evaluasi tim merah / keselamatan | Promptfoo + DeepEval |

Tumpukan umum: RAGAS untuk pemantauan, DeepEval untuk CI, G-Eval untuk dimension baru. Jalankan ketiganya; mereka tidak setuju.

## Kirim

Simpan sebagai `outputs/skill-eval-architect.md`:

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## Latihan

1. **Mudah.** Gunakan RAGAS pada 10 contoh RAG dengan halusinasi yang diketahui. Verifikasi metrik kesetiaan yang ditangkap masing-masing.
2. **Sedang.** Beri label tangan pada 50 jawaban QA 0-1 untuk mengetahui kebenarannya. Skor dengan G-Eval. Ukur spearman rho antara hakim dan manusia.
3. **Sulit.** Build gerbang CI tersulit dengan DeepEval. Dengan sengaja membuat kemunduran retriever. Pastikan gerbang gagal. Tambahkan peringatan kuantil terbawah melalui pemeriksaan ambang batas pada 10% terendah.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| LLM-sebagai-hakim | Mencetak gol dengan LLM | Anjurkan model juri untuk memberi skor pada output 0-1 berdasarkan rubrik. |
| RAGA | Pustaka metrik RAG | Kerangka kerja evaluasi sumber terbuka dengan 4 metrik RAG bebas referensi. |
| Kesetiaan | Apakah jawabannya beralasan? | Sebagian dari klaim jawaban yang disyaratkan oleh konteks yang diambil. |
| Ketepatan konteks | Apakah potongan yang diambil relevan? | Bagian dari potongan K teratas yang benar-benar penting. |
| Ingatan konteks | Apakah pengambilan menemukan segalanya? | Sebagian kecil klaim jawaban emas didukung oleh potongan yang diambil. |
| G-Eval | Juri LLM khusus | Rubrik + langkah evaluasi rantai pemikiran + skor 0-1. |
| Kalibrasi | Percaya tapi verifikasi | Korelasi Spearman antara skor juri dan skor manusia. |

## Bacaan Lanjutan- [Es dkk. (2023). RAGAS: Evaluasi Otomatis dari Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — makalah RAGAS.
- [Liu dkk. (2023). G-Eval: Evaluasi NLG menggunakan GPT-4 dengan Better Human Alignment](https://arxiv.org/abs/2303.16634) — makalah G-Eval.
- [Dokumen DeepEval](https://deepeval.com/docs/metrics-introduction) — membuka tumpukan produksi.
- [Zheng dkk. (2023). Menilai LLM sebagai Hakim dengan MT-Bench dan Chatbot Arena](https://arxiv.org/abs/2306.05685) — bias, kalibrasi, batasan.
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — framework pemersatu yang mengintegrasikan RAGAS, DeepEval, Phoenix.
