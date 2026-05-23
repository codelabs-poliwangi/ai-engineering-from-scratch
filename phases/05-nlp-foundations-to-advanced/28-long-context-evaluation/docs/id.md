# Evaluasi Konteks Panjang — NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro mengiklankan 10 juta token konteks. Pada 1 juta token, MRCR 8 jarum turun menjadi 26,3%. Diiklankan ≠ dapat digunakan. Evaluasi konteks panjang memberi tahu kamu kapasitas sebenarnya dari model yang kamu gunakan.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 5 · 13 (Menjawab Pertanyaan), Fase 5 · 23 (Strategi Chunking)
**Waktu:** ~60 menit

## Masalah

kamu memiliki kontrak 200 halaman. Model ini mengklaim konteks token 1 juta. kamu menempelkan kontrak dan bertanya: "Apa klausul penghentiannya?" Model menjawab — tetapi jawaban dari halaman sampul karena klausa penghentian berada pada kedalaman 120 ribu token, melampaui tempat model sebenarnya hadir.

Inilah kesenjangan konteks-kapasitas pada tahun 2026. Lembar spesifikasi mengatakan 1M atau 10M. Kenyataan mengatakan 60-70% darinya dapat digunakan, dan "dapat digunakan" bergantung pada tugasnya.

- **Pengambilan (jarum tunggal di tumpukan jerami):** hampir sempurna hingga batas maksimum yang diiklankan pada model perbatasan.
- **Multi-hop / agregasi:** menurun tajam hingga melewati ~128k pada sebagian besar model.
- **Menimbang fakta yang tersebar:** tugas pertama yang gagal.

Evaluasi konteks panjang mengukur sumbu-sumbu ini. Lesson ini menyebutkan tolok ukur, apa yang sebenarnya diukur, dan cara membuat pengujian jarum khusus untuk domain kamu.

## Konsep

![Dasar NIAH, multitugas RULER, holistik LongBench](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023).** Tempatkan fakta ("kata ajaibnya adalah nanas") pada kedalaman yang terkendali dalam konteks yang panjang. Minta model untuk mengambilnya. Kedalaman sapuan × panjang. Tolok ukur konteks panjang yang asli. Model Frontier kini memenuhi hal ini; ini merupakan dasar yang diperlukan namun tidak cukup.

**RULER (Nvidia, 2024).** 13 jenis tugas dalam 4 kategori: pengambilan (tunggal / multi-kunci / multi-nilai), penelusuran multi-hop (pelacakan variabel), agregasi (frekuensi kata umum), QA. Panjang konteks yang dapat dikonfigurasi (4k hingga 128k+). Mengungkapkan model yang memenuhi NIAH namun gagal pada multi-hop. Pada rilis tahun 2024, hanya setengah dari 17 model yang mengklaim konteks 32k+ mempertahankan kualitas pada 32k.

**LongBench v2 (2024).** 503 pertanyaan pilihan ganda, konteks kata 8k-2 juta, enam kategori tugas: QA dokumen tunggal, QA multi-dokumen, pembelajaran dalam konteks panjang, dialog panjang, repo code, data terstruktur panjang. Tolok ukur produksi untuk perilaku konteks panjang di dunia nyata.

**MRCR (Resolusi Inti Multi-Putaran).** Inti referensi multi-putaran dalam skala besar. Varian 8 jarum, 24 jarum, 100 jarum. Mengungkapkan berapa banyak fakta yang dapat disulap oleh seorang model sebelum attention-nya menurun.

**NoLiMa.** "Jarum non-leksikal." Jarum dan kueri tidak saling tumpang tindih; pengambilan memerlukan satu langkah penalaran semantik. Lebih sulit dari NIAH.

**HELM.** Menggabungkan banyak dokumen, mengajukan pertanyaan kepada siapa pun. Menguji attention selektif.

**BABILong.** Embed rantai penalaran bAbI di dalam tumpukan jerami yang tidak relevan. Menguji penalaran di tumpukan jerami, bukan hanya pengambilan kembali.

### Apa yang sebenarnya harus dilaporkan

- **Jendela konteks yang diiklankan.** Nomor lembar spesifikasi.
- **Panjang pengambilan yang efektif.** NIAH lolos pada batas tertentu (misalnya, 90%).
- **Panjang penalaran efektif.** Multi-hop atau agregasi lolos pada ambang batas tersebut.
- **Kurva degradasi.** Akurasi vs panjang konteks, diplot per jenis tugas.

Dua angka untuk lembar spesifikasi kamu: efektif dalam pengambilan dan efektif dalam penalaran. Biasanya alasan yang efektif adalah 25-50% dari jendela yang diiklankan.

## Build

### Langkah 1: NIAH khusus untuk domain kamu

Lihat `code/main.py`. Kerangka:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```Sapu `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}. Plot peta panasnya. Itu adalah kartu NIAH untuk model target kamu.

### Langkah 2: varian multi-jarum

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

Pertanyaan seperti "Apa tiga kata ajaib itu?" memerlukan pengambilan ketiganya. Keberhasilan satu jarum tidak memprediksi keberhasilan multi-jarum.

### Langkah 3: penelusuran variabel multi-hop (gaya RULER)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

Jawabannya membutuhkan rangkaian tiga tugas. Model Frontier pada 128k sering kali turun hingga akurasi 50-70% di sini.

### Langkah 4: LongBench v2 di tumpukan kamu

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

Laporkan keakuratan per kategori. Skor agregat menyembunyikan perbedaan tingkat tugas yang besar.

## Jebakan

- **Evaluasi khusus NIAH.** Melewati NIAH dengan 1 juta token tidak berarti apa-apa tentang multi-hop. Selalu jalankan RULER atau pengujian multi-hop khusus.
- **Pengambilan sample kedalaman seragam.** Banyak implementasi yang hanya menguji kedalaman=0,5. Kedalaman pengujian=0, 0,25, 0,5, 0,75, 1,0 — efek "hilang di tengah" adalah nyata.
- **Leksikal tumpang tindih dengan pengisi.** Jika jarum berbagi kata kunci dengan pengisi, pengambilan menjadi hal yang sepele. Gunakan jarum yang tidak tumpang tindih model NoLiMa.
- **Mengabaikan latensi.** Prompt 1 juta token memerlukan waktu 30-120 detik untuk diisi terlebih dahulu. Ukur waktu-ke-pertama-token bersama dengan akurasi.
- **Nomor yang dilaporkan sendiri oleh vendor.** OpenAI, Google, Anthropic semuanya mempublikasikan skor mereka sendiri. Selalu jalankan kembali secara mandiri pada kasus penggunaan kamu.

## Pakai

Tumpukan tahun 2026:

| Situasi | Tolok Ukur |
|-----------|-----------|
| Pemeriksaan kewarasan cepat | NIAH kustom pada 3 kedalaman × 3 panjang |
| Pemilihan model untuk produksi | RULER (13 tugas) sesuai panjang target kamu |
| Kualitas QA dunia nyata | Subset QA dokumen tunggal LongBench v2 |
| Penalaran multi-hop | BABILong atau penelusuran variabel khusus |
| Percakapan/dialog | MRCR 8-jarum sesuai panjang target kamu |
| Regresi peningkatan model | Memperbaiki harness NIAH + RULER internal, dijalankan pada setiap model baru |

Aturan praktis untuk produksi: jangan pernah mempercayai jendela konteks sampai kamu memiliki tugas penalaran NIAH + 1 sesuai panjang yang kamu inginkan.

## Kirim

Simpan sebagai `outputs/skill-long-context-eval.md`:

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## Latihan

1. **Mudah.** Buat NIAH dengan 3 kedalaman (0,25, 0,5, 0,75) × 3 panjang (1k, 4k, 16k). Jalankan pada model apa pun. Plot tingkat kelulusan sebagai peta panas 3×3.
2. **Medium.** Tambahkan varian 3 jarum. Ukur pengambilan ketiganya pada setiap panjang. Bandingkan dengan tingkat lintasan jarum tunggal pada panjang yang sama.
3. **Sulit.** Buat tugas penelusuran variabel (X1 → X2 → X3, dengan 3 lompatan) yang tertanam dalam 64k pengisi. Ukur akurasi di 3 model frontier. Laporkan panjang penalaran efektif per model.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| NIAH | Jarum di tumpukan jerami | Tanam fakta di pengisi, minta model untuk mengambilnya kembali. |
| PENGUASA | NIAH tentang steroid | 13 jenis tugas di pengambilan / multi-hop / agregasi / QA. |
| Konteks efektif | Kapasitas sebenarnya | Panjang yang akurasinya masih berada di atas ambang batas. |
| Tersesat di tengah | Bias kedalaman | Model kurang memperhatikan konten di tengah input yang panjang. |
| Multi-jarum | Banyak fakta sekaligus | Banyak tanaman; menguji juggling attention, bukan pengambilan saja. |
| MRCR | Inti multi-bulat | inti referensi 8, 24, atau 100 jarum; memperlihatkan saturasi attention. |
| Tidak AdaLiMa | Jarum non-leksikal | Jarum dan kueri tidak memiliki token literal; memerlukan penalaran. |

## Bacaan Lanjutan- [Kamradt (2023). Analisis jarum dalam tumpukan jerami](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — repo NIAH asli.
- [Hsieh dkk. (2024). RULER: Berapa Ukuran Konteks Sebenarnya dari LM Konteks Panjang kamu?](https://arxiv.org/abs/2404.06654) — tolok ukur multitugas.
- [Bai dkk. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — evaluasi konteks panjang dunia nyata.
- [Modarressi dkk. (2024). NoLiMa: Jarum non-leksikal](https://arxiv.org/abs/2404.06666) — jarum yang lebih keras.
- [Kuratov dkk. (2024). BABILong](https://arxiv.org/abs/2406.10149) — penalaran di tumpukan jerami.
- [Liu dkk. (2024). Tersesat di Tengah: Bagaimana Model Bahasa Menggunakan Konteks Panjang](https://arxiv.org/abs/2307.03172) - makalah bias mendalam.
