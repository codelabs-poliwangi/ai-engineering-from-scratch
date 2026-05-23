# Capstone 04 — QA Dokumen Multimodal (PDF Visi-Pertama, Tabel, Bagan)

> Batasan dokumen-QA pada tahun 2026 beralih dari OCR-lalu-teks dan menuju interaksi akhir yang mengutamakan visi. ColPali, ColQwen2.5, dan ColQwen3-omni memperlakukan setiap halaman PDF sebagai gambar, menyematkannya dengan interaksi akhir multi-vector, dan membiarkan kueri menangani patch secara langsung. Pada keuangan 10-K, makalah ilmiah, dan catatan tulisan tangan, pola ini mengalahkan OCR dengan selisih yang besar. Build alur dari ujung ke ujung pada 10 ribu halaman dan publikasikan secara berdampingan dengan OCR lalu teks.

**Type:** Batu penjuru
**Language:** Python (pipeline), TypeScript (UI penampil)
**Prerequisites:** Fase 4 (computer vision), Fase 5 (NLP), Fase 7 (Transformer), Fase 11 (rekayasa LLM), Fase 12 (multimodal), Fase 17 (infrastruktur)
**Fase yang dilakukan:** P4 · P5 · P7 · P11 · P12 · P17
**Waktu:** 30 jam

## Masalah

Perusahaan-perusahaan hanya mengandalkan PDF yang dirusak oleh jaringan pipa OCR: pemindaian 10-K dengan tabel yang diputar, makalah ilmiah yang penuh dengan persamaan, bagan yang hanya masuk akal sebagai gambar, anotasi tulisan tangan. Memperlakukan ini sebagai text-first berarti kehilangan separuh sinyal. Jawaban tahun 2026 adalah pengambilan multi-vector interaksi akhir pada gambar halaman mentah. ColPali (Illuin Tech) memperkenalkannya; ColQwen2.5-v0.2 dan ColQwen3-omni mendorong akurasi. Pada ViDoRe v3, perolehan vision-first mendapat skor di atas OCR-kemudian-teks dengan margin yang berarti — dan kesenjangannya semakin lebar pada bagan, tabel, dan tulisan tangan.

Imbalannya adalah penyimpanan dan latensi. Embedding ColQwen adalah ~2048 vector patch per halaman, bukan satu vector 1024 redup. Balon penyimpanan mentah. DocPruner (2026) menghasilkan pemangkasan 50% tanpa kehilangan akurasi yang terukur. kamu akan mengindeks 10 ribu halaman, mengukur ViDoRe v3 nDCG@5, menyajikan jawaban di bawah 2 detik, dan membandingkan langsung dengan garis dasar OCR lalu teks.

## Konsep

Interaksi yang terlambat berarti setiap token kueri mendapat skor terhadap setiap token patch, dan skor maksimum per token kueri dijumlahkan. kamu mendapatkan pencocokan terperinci tanpa memerlukan satu pun vector gabungan. Indeks multi-vector (Vespa, multi-vector Qdrant, atau AstraDB) menyimpan embeddings per patch dan menjalankan MaxSim pada waktu pengambilan.

Penjawab adalah model bahasa visi yang mengambil kueri ditambah halaman teratas yang diambil sebagai gambar dan menulis jawaban dengan wilayah bukti (kotak pembatas atau referensi halaman). Qwen3-VL-30B, Gemini 2.5 Pro, dan InternVL3 adalah pilihan terdepan tahun 2026. Untuk persamaan dan notasi ilmiah, fallback OCR (Nougat, dot.ocr) digabungkan sebagai pipeline teks opsional.

Evaluasi adalah matrix dua dimension. Satu sumbu: tipe konten (paragraf teks biasa, tabel padat, diagram batang/garis, catatan tulisan tangan, persamaan). Sumbu lainnya: pendekatan pengambilan (interaksi akhir penglihatan-pertama vs OCR-lalu-teks vs hibrida). Setiap sel mendapatkan nDCG@5 dan akurasi jawaban. Laporan adalah hasil yang dapat disampaikan.

## Arsitektur

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## Tumpukan

- Render halaman: PyMuPDF (fitz) pada 180 DPI, potret-normal
- Model interaksi terlambat: ColQwen2.5-v0.2 atau ColQwen3-omni (tim video di Hugging Face)
- Indeks: Vespa dengan bidang multi-vector, atau Qdrant multi-vector, atau AstraDB dengan MaxSim
- Pemangkasan: Kebijakan DocPruner 2026 (pertahankan patch varian tinggi, kompresi 50% dengan kehilangan akurasi <0,5%)
- Penggantian OCR (persamaan / tabel padat): titik.ocr atau Nougat
- Penjawab VLM: Qwen3-VL-30B dihosting sendiri atau dihosting Gemini 2.5 Pro; InternVL3 sebagai cadangan
- Evaluasi: benchmark ViDoRe v3, M3DocVQA untuk penalaran multi-halaman
- UI Penampil: Next.js 15 dengan hamparan kanvas untuk wilayah bukti

## Bangun1. **Penyerapan.** Jelajahi kumpulan 10 ribu halaman PDF dalam 10-K, makalah ilmiah, dan dokumen yang dipindai. Render setiap halaman menjadi PNG 1536x2048. Bertahan `{doc_id, page_num, image_path}`.

2. **Sematkan.** Jalankan ColQwen2.5-v0.2 pada setiap gambar halaman. Bentuk output ~2048 embedding tambalan redup 128. Terapkan DocPruner untuk mempertahankan separuh sinyal tertinggi. Tulis ke kolom multi-vector Vespa atau multi-vector Qdrant.

3. **Kueri.** Untuk setiap kueri masuk, sematkan dengan menara kueri (embedding tingkat token). Jalankan MaxSim terhadap indeks: untuk setiap token kueri, ambil produk titik maksimal di atas embedding patch halaman, jumlah. Kembalikan halaman k teratas.

4. **Sintesiskan.** Panggil Qwen3-VL-30B dengan kueri dan gambar 5 halaman teratas. Prompt: "Jawab hanya dengan menggunakan halaman yang disediakan. Kutip setiap klaim berdasarkan (doc_id, halaman) dan beri nama wilayahnya (gambar, tabel, paragraf)."

5. **Wilayah bukti.** Pasca-proses jawaban untuk mengekstrak wilayah yang dikutip. Jika VLM mengeluarkan kotak pembatas (Qwen3-VL melakukannya), render kotak tersebut sebagai overlay di penampil.

6. **Penggantian OCR.** Untuk laman yang teridentifikasi padat persamaan (heuristik pada varian gambar), jalankan Nougat atau dot.ocr dan teruskan teks OCR sebagai pipeline tambahan di samping gambar.

7. **Eval.** Jalankan ViDoRe v3 (pengambilan nDCG@5) dan M3DocVQA (akurasi QA multi-halaman). Jalankan juga pipeline OCR-lalu-teks pada korpus yang sama dengan synthesizer yang sama. Menghasilkan matrix pendekatan tipe konten ×.

8. **UI.** Prototipe streamlit terlebih dahulu; Penampil produksi Next.js 15 dengan hamparan wilayah bukti halaman demi halaman.

## Pakai

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## Kirim

`outputs/skill-doc-qa.md` menjelaskan hasil yang dapat dicapai: sistem QA dokumen multimodal vision-first yang disetel ke korpus tertentu dan dievaluasi berdasarkan garis dasar OCR lalu teks di ViDoRe v3.

| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | Akurasi ViDoRe v3 / M3DocVQA | Nomor tolok ukur vs garis dasar teks OCR dan papan peringkat yang dipublikasikan |
| 20 | Landasan wilayah bukti | Bagian dari wilayah yang dikutip yang sebenarnya memuat rentang jawaban |
| 20 | Rekayasa penyimpanan dan latensi | Rasio kompresi DocPruner, indeks p95, jawaban p95 |
| 20 | Penalaran multi-halaman | Akurasi pada kumpulan 100 pertanyaan multi-halaman berlabel tangan |
| 15 | UX inspeksi sumber | Kejelasan pemirsa, fidelitas overlay, alat perbandingan berdampingan |
| **100** | | |

## Latihan

1. Ukur ColQwen2.5-v0.2 vs ColQwen3-omni pada korpus yang sama. Halaman manakah yang benar dan halaman lainnya terlewat? Tambahkan tag "kelas konten" ke indeks untuk merutekan berdasarkan jenis.

2. Pangkas tanaman yang ditanam secara agresif (75%, 90%). Temukan tebing kompresi: titik di mana ViDoRe nDCG@5 turun di bawah garis dasar OCR.

3. Build hybrid: jalankan OCR-lalu-teks dan ColQwen secara paralel, gabungkan dengan RRF, rangking ulang dengan cross-encoder. Apakah hibrida bisa mengalahkannya sendirian? Di bagian manakah hal ini paling membantu?

4. Tukar Qwen3-VL-30B dengan VLM yang lebih kecil (Qwen2.5-VL-7B). Ukur kurva akurasi per dolar.

5. Tambahkan dukungan catatan tulisan tangan. Render korpus tulisan tangan, sematkan dengan ColQwen, ukur pengambilannya. Bandingkan dengan pipeline OCR tulisan tangan.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Interaksi terlambat | "Pengambilan gaya ColPali" | Token kueri mencetak skor terhadap tambalan halaman secara independen; Agregat MaxSim |
| Multi-vector | "Embedding per patch" | Setiap dokumen mempunyai banyak vector, bukan satu vector gabungan |
| MaxSim | "Skor interaksi terlambat" | Untuk setiap token kueri, ambil kesamaan maksimal pada vector dokumen; jumlah |
| DocPruner | "Patch kompresi" | Pemangkasan tahun 2026 yang menjaga 50% tambalan dengan kehilangan akurasi yang dapat diabaikan |
| ViDoRe v3 | "Patokan pengambilan dokumen" | Standar tahun 2026 untuk mengukur pengambilan dokumen visual |
| Wilayah bukti | "Kotak pembatas yang dikutip" | Sebuah bbox di halaman sumber yang melokalkan rentang jawaban |
| Penggantian OCR | "Pipeline persamaan" | Pipeline teks digunakan bersama visi untuk halaman berisi persamaan atau tabel |

## Bacaan Lanjutan

- [Repositori ColPali (Illuin Tech)](https://github.com/illuin-tech/colpali) — referensi pengambilan dokumen interaksi akhir
- [Makalah ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — makalah metode dasar
- [Keluarga ColQwen di Hugging Face](https://huggingface.co/vidore) — pos pemeriksaan siap produksi
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — garis dasar RAG multimodal multi-halaman
- [Tutorial multi-vector Vespa](https://docs.vespa.ai/en/colpali.html) — tumpukan penyajian referensi
- [Dukungan multi-vector Qdrant](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — indeks alternatif
- [multi-vector AstraDB](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — indeks terkelola alternatif
- [Nougat OCR](https://github.com/facebookresearch/nougat) — penggantian OCR berkemampuan persamaan
