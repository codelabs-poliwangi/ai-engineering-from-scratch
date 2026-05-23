# Output Terstruktur & Decoding Terbatas

> Tanyakan LLM untuk JSON. Dapatkan JSON sebagian besar waktu. Dalam produksi, "sebagian besar" adalah masalahnya. Penguraian code yang dibatasi mengubah "sebagian besar" menjadi "selalu" dengan mengedit logit sebelum pengambilan sample.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 5 · 17 (Chatbots), Fase 5 · 19 (Tokenization Subkata)
**Waktu:** ~60 menit

## Masalah

Pengklasifikasi meminta LLM: "Kembalikan salah satu dari {positif, negatif, netral}." Model tersebut mengembalikan "Sentimennya positif — ulasan ini sangat disukai karena pelanggan secara eksplisit menyatakan bahwa mereka ...". Pengurai kamu mogok. F1 pengklasifikasi kamu adalah 0,0.

Generasi bentuk bebas bukanlah sebuah kontrak. Itu adalah sebuah saran. Sebuah sistem produksi membutuhkan kontrak.

Ada tiga layer pada tahun 2026.

1. **Mendorong.** Meminta dengan baik. "Kembalikan hanya objek JSON." Bekerja ~80% pada model frontier, lebih sedikit pada model yang lebih kecil.
2. **API output terstruktur asli.** OpenAI `response_format`, penggunaan alat antropik, mode Gemini JSON. Dapat diandalkan pada skema yang didukung. Terkunci oleh vendor.
3. **Decoding yang dibatasi.** Ubah logit pada setiap langkah pembuatan sehingga model *tidak dapat* mengeluarkan token yang tidak valid. 100% valid berdasarkan konstruksi. Bekerja pada model lokal apa pun.

Lesson ini membangun intuisi untuk ketiganya dan menyebutkan kapan harus mencapai yang mana.

## Konsep

![Dekode terbatas yang menutupi token tidak valid di setiap langkah](../assets/constrained-decoding.svg)

**Cara kerja decoding terbatas.** Pada setiap langkah pembuatan, LLM menghasilkan vector logit pada seluruh kosakata (~100 ribu token). *Prosesor logit* berada di antara model dan sampler. Ini menghitung token mana yang valid berdasarkan posisi saat ini dalam tata bahasa target — Skema JSON, regex, tata bahasa bebas konteks — dan menetapkan logit dari semua token yang tidak valid ke tak terhingga negatif. Softmax atas logit yang tersisa menempatkan massa probabilitas hanya pada kelanjutan yang valid.

Implementasi pada tahun 2026:

- **Garis Besar.** Mengompilasi Skema JSON atau regex ke dalam mesin keadaan terbatas. Setiap token mendapat pencarian O(1) valid-next-token. Berbasis FSM, jadi skema rekursif perlu diratakan.
- **XGrammar / llguidance.** Mesin tata bahasa bebas konteks. Menangani Skema JSON rekursif. Overhead decoding mendekati nol. OpenAI memberikan penghargaan atas panduan dalam implementasi output terstruktur mereka pada tahun 2025.
- **decoding terpandu vLLM.** `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` bawaan melalui Outlines, XGrammar, atau backend penegak format lm.
- **Instruktur.** Pembungkus berbasis Pydantic di LLM mana pun. Percobaan ulang pada kegagalan validasi. Lintas penyedia, tetapi tidak mengubah logit — ini bergantung pada percobaan ulang + prompt sadar output terstruktur.

### Hasil yang berlawanan dengan intuisi

Penguraian code yang dibatasi seringkali *lebih cepat* dibandingkan pembuatan code yang tidak dibatasi. Dua alasan. Pertama, ini memperkecil ruang pencarian token berikutnya. Kedua, implementasi cerdas melewatkan pembuatan token sepenuhnya untuk token yang dipaksakan (perancah seperti `{"name": "` — setiap byte ditentukan).

### Jebakan yang merugikan kamu

Urutan lapangan penting. Letakkan `answer` sebelum `reasoning`, dan model berkomitmen pada sebuah jawaban sebelum berpikir. JSON valid. Jawabannya salah. Tidak ada validasi yang menangkapnya.

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Urutan bidang skema adalah logika, bukan pemformatan.

## Build

### Langkah 1: pembuatan dengan batasan regex dari awal

Lihat `code/main.py` untuk penerapan FSM mandiri. Ide inti dalam 30 baris:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```FSM melacak bagian tata bahasa mana yang telah kami penuhi sejauh ini. `valid_tokens(state, tokenizer)` menghitung token kosakata mana yang dapat memajukan FSM tanpa meninggalkan jalur penerimaan.

### Langkah 2: Garis Besar Skema JSON

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

Tidak ada kesalahan validasi. Pernah. FSM membuat output yang tidak valid tidak dapat dijangkau.

### Langkah 3: Instruktur untuk Pydantic penyedia-agnostik

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

Mekanisme yang berbeda. Instruktur tidak menyentuh logit. Ini memformat skema ke dalam prompt, mem-parsing output, dan mencoba lagi jika validasi gagal (default 3 kali). Bekerja dengan penyedia mana pun. Percobaan ulang menambah latensi dan biaya. Portabilitas lintas penyedia adalah nilai jualnya.

### Langkah 4: API vendor asli

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

Dekode terbatas sisi server. Kesetaraan keandalan dengan Garis Besar untuk skema yang didukung. Tidak ada manajemen model lokal. Mengunci kamu ke vendor.

## Jebakan

- **Skema rekursif.** Garis besar meratakan rekursi hingga kedalaman tetap. Output terstruktur pohon (komentar bersarang, AST) memerlukan XGrammar atau panduan (berbasis CFG).
- **Enum besar.** Enum 10.000 opsi dikompilasi dengan lambat atau habis waktu. Beralih ke retriever: prediksi kandidat teratas terlebih dahulu, batasi pada kandidat tersebut.
- **Tata bahasa terlalu ketat.** Paksa `date: "YYYY-MM-DD"` regex dan model tidak dapat menampilkan `"unknown"` untuk tanggal yang hilang. Model memberikan kompensasi dengan menentukan tanggal. Izinkan `null` atau penjaga.
- **Komitmen prematur.** Lihat kendala pesanan lapangan di atas. Selalu utamakan alasan.
- **Mode JSON Vendor tanpa skema.** Mode JSON murni hanya menjamin JSON yang valid, tidak valid *untuk kasus penggunaan kamu*. Selalu berikan skema lengkap.

## Pakai

Tumpukan tahun 2026:

| Situasi | Pilih |
|-----------|------|
| Model OpenAI/Anthropic/Google, skema sederhana | Output terstruktur vendor asli |
| Penyedia mana pun, alur kerja Pydantic, dapat mentolerir percobaan ulang | Instruktur |
| Model lokal, memerlukan validitas 100%, skema datar | Garis Besar (FSM) |
| Model lokal, skema rekursif | XTata Bahasa atau Panduan |
| Server inference yang dihosting sendiri | dekode terpandu vLLM |
| Pemrosesan batch dengan percobaan ulang dapat diterima | Instruktur + model termurah |

## Kirim

Simpan sebagai `outputs/skill-structured-output-picker.md`:

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## Latihan

1. **Mudah.** Minta model weight terbuka kecil (misalnya, Llama-3.2-3B) tanpa batasan decoding untuk `Review(sentiment, confidence, evidence_span)`. Ukur pecahan yang diurai sebagai JSON valid pada 100 ulasan.
2. **Medium.** Korpus yang sama dengan mode Outlines JSON. Bandingkan tingkat kepatuhan, latensi, dan akurasi semantik.
3. **Sulit.** Menerapkan decoder dengan batasan regex dari awal untuk nomor telepon (`\d{3}-\d{3}-\d{4}`). Verifikasi 0 output tidak valid pada 1000 sample.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Penguraian code terbatas | Paksa output yang valid | Menyembunyikan logit token yang tidak valid pada setiap langkah pembuatan. |
| Prosesor logit | Hal yang menjadi kendala | Fungsi: `(logits, state) -> masked_logits`. |
| FSM | Mesin keadaan terbatas | Representasi tata bahasa yang dikompilasi; O(1) pencarian token berikutnya yang valid. |
| CFG | Tata bahasa bebas konteks | Tata bahasa yang menangani rekursi; lebih lambat tetapi lebih ekspresif dibandingkan FSM. |
| Urutan bidang skema | Apakah itu penting? | Ya — bidang pertama dilakukan; selalu mengutamakan alasan sebelum jawaban. |
| Penguraian code terpandu | nama vLLM untuk itu | Konsep yang sama, terintegrasi ke dalam server inference. |
| Modus JSON | Versi awal OpenAI | Menjamin sintaksis JSON; TIDAK menjamin kecocokan skema. |

## Bacaan Lanjutan- [Willard, Louf (2023). Generasi Terpandu yang Efisien untuk LLM](https://arxiv.org/abs/2307.09702) — makalah Garis Besar.
- [Makalah XGrammar (2024)](https://arxiv.org/abs/2411.15100) — decoding terbatas cepat berbasis CFG.
- [vLLM — Output Terstruktur](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — integrasi server inference.
- [OpenAI — Panduan Output Terstruktur](https://platform.openai.com/docs/guides/structured-outputs) — Referensi API + gotcha.
- [Perpustakaan instruktur](https://python.useinstructor.com/) — Pydantic + percobaan ulang di seluruh penyedia.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — melakukan benchmark pada 6 framework decoding yang dibatasi.
