# Strategi Pemotongan untuk RAG

> Konfigurasi chunking memengaruhi kualitas pengambilan seperti halnya pilihan model embedding (Vectara NAACL 2025). Lakukan kesalahan dalam pengelompokan dan tidak ada pemeringkatan ulang yang dapat menyelamatkan kamu.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 5 · 14 (Pengambilan Informasi), Fase 5 · 22 (Model Embedding)
**Waktu:** ~60 menit

## Masalah

kamu memasukkan kontrak 50 halaman ke dalam sistem RAG. Pengguna bertanya: "Apa klausul penghentiannya?" Retriever mengembalikan halaman sampul. Mengapa? Karena model dilatih pada potongan 512 token dan klausa penghentian terdiri dari 20 halaman, dibagi menjadi hentian halaman, tanpa kata kunci lokal yang mengikatnya ke kueri.

Cara mengatasinya bukanlah "beli model embedding yang lebih baik". Perbaikannya terpotong-potong. Seberapa besar? Tumpang tindih? Di mana harus berpisah? Dengan konteks sekitar?

Tolok ukur bulan Februari 2026 menunjukkan hasil yang mengejutkan:

- Studi Vectara tahun 2026: pemotongan 512 token rekursif mengalahkan pemotongan semantik 69% → akurasi 54%.
- SPLADE + Mistral-8B pada Pertanyaan Alami: tumpang tindih tidak memberikan manfaat yang terukur.
- Tebing konteks: kualitas respons turun tajam sekitar 2.500 token konteks.

Jawaban yang "jelas" (pengelompokan semantik, 20% tumpang tindih, 1000 token) sering kali salah. Lesson ini membangun intuisi untuk enam strategi dan memberi tahu kamu kapan harus mencapai strategi mana.

## Konsep

![Enam strategi pemotongan divisualisasikan dalam satu bagian](../assets/chunking.svg)

**Memperbaiki chunking.** Pisahkan setiap N karakter atau token. Dasar yang paling sederhana. Istirahat di tengah kalimat. Kompresi bagus, koherensi buruk.

**Rekursif.** `RecursiveCharacterTextSplitter` LangChain. Coba pisahkan pada `\n\n` terlebih dahulu, lalu `\n`, lalu `.`, lalu spasi. Jatuh kembali dengan bersih. Default tahun 2026.

**Semantik.** Sematkan setiap kalimat. Hitung kemiripan kosinus antara kalimat-kalimat yang berdekatan. Pisahkan ketika kesamaan berada di bawah ambang batas. Mempertahankan koherensi topik. Lebih lambat; terkadang menghasilkan fragmen kecil sebanyak 40 token yang mengganggu pengambilan.

**Kalimat.** Pisahkan batas kalimat. Satu kalimat per potongan atau jendela N kalimat. Mencocokkan potongan semantik hingga ~5 ribu token dengan biaya yang lebih murah.

**Dokumen induk.** Simpan potongan anak kecil untuk pengambilan *dan* potongan induk yang lebih besar untuk konteks. Ambil oleh anak; orang tua kembali. Menurun dengan baik: potongan anak yang buruk masih mengembalikan orang tua yang berakal sehat.

**Pembagian yang terlambat (2024).** Sematkan seluruh dokumen pada tingkat token terlebih dahulu, lalu kumpulkan embedding token ke dalam embedding potongan. Mempertahankan konteks lintas bagian. Bekerja dengan embedding konteks panjang (BGE-M3, Jina v3). Komputasi yang lebih tinggi.

**Pengambilan kontekstual (Anthropic, 2024).** Tambahkan setiap potongan dengan ringkasan posisinya dalam dokumen yang dihasilkan LLM ("Bagian ini adalah bagian 3.2 dari klausul penghentian..."). Peningkatan pengambilan 35-50% pada benchmark Anthropic sendiri. Mahal untuk diindeks.

### Aturan yang mengalahkan setiap default

Cocokkan ukuran potongan dengan jenis kueri:

| Jenis kueri | Ukuran potongan |
|------------|-----------|
| Factoid (“siapa nama CEO?”) | 256-512 token |
| Analitis / multi-hop | 512-1024 token |
| Pemahaman seluruh bagian | Token 1024-2048 |

Tolok ukur NVIDIA tahun 2026. Potongan tersebut harus cukup besar untuk memuat jawaban ditambah konteks lokal, cukup kecil sehingga top-K retriever akan fokus pada jawaban, bukan gangguan konteks.

## Build

### Langkah 1: pemotongan tetap dan rekursif

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### Langkah 2: pengelompokan semantik

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```Sesuaikan `threshold` di domain kamu. Terlalu tinggi → pecahan. Terlalu rendah → satu bongkahan raksasa.

### Langkah 3: dokumen orang tua

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

Wawasan utama: orang tua yang menipu. Beberapa anak dapat dipetakan ke orang tua yang sama; mengembalikan semua akan menyia-nyiakan konteks.

### Langkah 4: pengambilan kontekstual (Pola antropis)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

Indeks bagian yang dikontekstualisasikan. Pada waktu kueri, pengambilan mendapat manfaat dari sinyal tambahan di sekitarnya.

### Langkah 5: evaluasi

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

Selalu menjadi patokan. Strategi "terbaik" untuk korpus kamu mungkin tidak cocok dengan postingan blog mana pun.

## Jebakan

- **Pembagian hanya dievaluasi berdasarkan kueri factoid.** Kueri multi-hop menunjukkan pemenang yang sangat berbeda. Gunakan kumpulan evaluasi bertingkat tipe kueri.
- **Pembagian semantik tanpa ukuran minimum.** Menghasilkan 40 fragmen token yang mengganggu pengambilan. Selalu terapkan `min_tokens`.
- **Tumpang tindih sebagai kultus kargo.** Studi pada tahun 2026 menemukan bahwa tumpang tindih sering kali tidak memberikan manfaat apa pun dan melipatgandakan biaya indeks. Ukur, jangan berasumsi.
- **Tidak ada penerapan min/maks.** Potongan 5 token atau 5.000 token keduanya menghentikan pengambilan. Penjepit.
- **Pengelompokan lintas dokumen.** Jangan biarkan satu pun mencakup dua dokumen. Selalu potong per dokumen, lalu gabungkan.

## Pakai

Tumpukan tahun 2026:

| Situasi | Strategi |
|-----------|----------|
| Pembuatan pertama, korpus tidak diketahui | Rekursif, 512 token, tidak tumpang tindih |
| QA Fakta | Rekursif, 256-512 token |
| Analitis / multi-hop | Rekursif, token 512-1024 + dokumen induk |
| Referensi silang yang berat (kontrak, makalah) | Pengambilan potongan atau pengambilan kontekstual yang terlambat |
| Korpus Percakapan/Dialog | Potongan tingkat giliran + metadata pembicara |
| Ucapan pendek (tweet, review) | Satu dokumen = satu potongan |

Mulailah dengan 512 rekursif. Ukur recall@5 pada set eval 50 kueri. Dengarkan dari sana.

## Kirim

Simpan sebagai `outputs/skill-chunker.md`:

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## Latihan

1. **Mudah.** Potong satu dokumen setebal 20 halaman dengan tetap (512, 0), rekursif (512, 0), dan rekursif (512, 100). Bandingkan jumlah potongan dan kualitas batas.
2. **Sedang.** Buat kumpulan evaluasi 30 kueri pada 5 dokumen. Ukur recall@5 untuk dokumen rekursif, semantik, dan induk. Yang mana yang menang? Apakah itu cocok dengan postingan blog?
3. **Sulit.** Menerapkan pengambilan kontekstual. Ukur peningkatan MRR dibandingkan rekursif dasar. Laporkan biaya indeks (panggilan LLM) vs perolehan akurasi.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Potongan | Sepotong dokumen | Unit sub-dokumen yang di-embed, diindeks, dan diambil. |
| Tumpang tindih | Margin keamanan | N token dibagikan di antara bongkahan yang berdekatan; seringkali tidak berguna pada benchmark tahun 2026. |
| Pemotongan semantik | Pemotongan cerdas | Pisahkan di mana kesamaan embedding kalimat yang berdekatan menurun. |
| Dokumen induk | Pengambilan dua tingkat | Ambil kembali anak kecil, kembalikan orang tua yang lebih besar. |
| Pemotongan terlambat | Potongan setelah embedding | Sematkan dokumen lengkap di tingkat token, gabungkan ke dalam vector potongan. |
| Pengambilan kontekstual | Trik Antropik | Ringkasan yang dihasilkan LLM ditambahkan ke setiap bagian sebelum diindeks. |
| Tebing konteks | Dinding 2500 token | Penurunan kualitas diamati di sekitar 2,5 ribu token konteks di RAG (Jan 2026). |

## Bacaan Lanjutan- [Yepes dkk. / LangChain — dokumen Pemisahan Karakter Rekursif](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — default dalam produksi.
- [Vektara (2024, NAACL 2025). Analisis konfigurasi pengelompokan](https://arxiv.org/abs/2410.13070) — pengelompokan sama pentingnya dengan pilihan embedding.
- [Jina AI — Pemotongan Akhir dalam Model Embedding Konteks Panjang (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — makalah pemotongan terakhir.
- [Antropik — Pengambilan Kontekstual](https://www.anthropic.com/news/contextual-retrieval) — peningkatan pengambilan 35-50% dengan awalan konteks yang dihasilkan LLM.
- [Patokan ukuran bongkahan NVIDIA 2026 — Ringkasan Premai](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — ukuran bongkahan berdasarkan jenis kueri.
