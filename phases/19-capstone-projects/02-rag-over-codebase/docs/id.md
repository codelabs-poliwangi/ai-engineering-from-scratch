# Capstone 02 — RAG over Codebase (Pencarian Semantik Lintas-Repo)

> Setiap organisasi teknik yang serius pada tahun 2026 menjalankan pencarian code internal yang memahami makna, bukan hanya string. Sourcegraph Amp, jawaban basis code Cursor, grafik perusahaan Augment, repomap Aider, MCP internal Pinterest — bentuknya sama. Serap banyak repo, uraikan dengan pengasuh pohon, sematkan potongan tingkat fungsi dan kelas, pencarian hibrid, rangking ulang, jawab dengan kutipan. Batu penjuru ini meminta kamu untuk membuat yang menangani 2 juta baris code di 10 repo dan bertahan dari pengindeksan ulang tambahan pada setiap git push.

**Type:** Batu penjuru
**Language:** Python (penyerapan), TypeScript (API + UI)
**Prerequisites:** Fase 5 (fondasi NLP), Fase 7 (Transformer), Fase 11 (rekayasa LLM), Fase 13 (peralatan), Fase 17 (infrastruktur)
**Fase yang dilakukan:** P5 · P7 · P11 · P13 · P17
**Waktu:** 30 jam

## Masalah

Pada tahun 2026, setiap agen pengkodean frontier dikirimkan dengan layer pengambilan basis code karena jendela konteks saja tidak menyelesaikan pertanyaan lintas repo. Konteks 1M-token Claude membantu; itu tidak menghilangkan kebutuhan untuk pengambilan peringkat. Pencarian kosinus yang naif pada potongan mentah menghasilkan code yang dihasilkan, duplikasi monorepo, dan simbol yang jarang diimpor. Jawaban produksinya adalah penelusuran hibrid (padat + BM25) pada potongan yang sadar AST dengan pemeringkatan ulang, yang didukung oleh grafik referensi simbol.

kamu mempelajarinya dengan mengindeks armada nyata — bukan satu repo tutorial — dan mengukur MRR@10, kesetiaan kutipan, dan kesegaran tambahan. Mode kegagalan bersifat infrastruktur: monorepo file 100k, dorongan yang memperbaiki separuh file, kueri yang perlu melewati empat repo untuk menjawab dengan benar.

## Konsep

Pipeline penyerapan yang sadar AST mem-parsing setiap file dengan tree-sitter, mengekstrak node fungsi dan kelas, serta potongan pada batas node, bukan pada jendela token tetap. Setiap potongan mendapat tiga representasi: embedding padat (code Voyage-3 atau code embedding nomic), istilah BM25 yang jarang, dan ringkasan bahasa alami yang singkat. Ringkasan tersebut menambahkan modalitas ketiga yang dapat diambil — pengguna bertanya "bagaimana X diotorisasi" dan ringkasan tersebut menyebutkan "authz", meskipun code tersebut hanya memiliki `check_permission`.

Pengambilan adalah hibrida. Kueri memicu penelusuran padat dan BM25, menggabungkan top-k, dan menyerahkan gabungan tersebut ke pemeringkat ulang lintas-encoder (Cohere rerank-3 atau bge-reranker-v2-gemma-2b). Daftar yang diberi peringkat ulang ditujukan ke synthesizer konteks panjang (Claude Sonnet 4.7 dengan cache cepat, atau Llama 3.3 70B yang dihosting sendiri) dengan instruksi untuk mengutip setiap klaim berdasarkan rentang file dan baris. Jawaban tanpa kutipan ditolak oleh post-filter.

Kesegaran tambahan adalah masalah infrastruktur. Git push memicu perbedaan: file mana yang berubah, simbol mana yang berubah. Hanya potongan yang terkena dampak yang disisipkan kembali. Tepi simbol lintas file yang terpengaruh (impor, pemanggilan metode) dihitung ulang. Indeks tetap konsisten tanpa memproses ulang 2 juta baris setiap penerapan.

## Arsitektur

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## Tumpukan- Parsing: tree-sitter dengan 17 tata bahasa (Python, TS, Rust, Go, Java, C++, dll.)
- Embedding padat: Voyage-code-3 (dihosting) atau nomic-embed-code-v1.5 (self-host), fallback bge-code-v1
- Indeks renggang: Tantivy (Rust) dengan BM25F, weighting bidang pada nama simbol vs badan
- Vector DB: Qdrant 1.12 dengan pencarian hybrid, atau pgvector + pgvectorscale untuk tim di bawah 50 juta vector
- Model ringkasan potongan: Claude Haiku 4.5 atau Gemini 2.5 Flash, di-cache secara cepat
- Re-ranker: Cohere rerank-3 atau bge-reranker-v2-gemma-2b dihosting sendiri
- Orkestrasi: Alur Kerja LlamaIndex untuk penyerapan, LangGraph untuk agen kueri
- Synthesizer: Claude Sonnet 4.7 (konteks 1 juta) dengan caching cepat
- Grafik simbol: Neo4j (dikelola) atau kuzu (tertanam) untuk impor dan panggilan tepi
- Observabilitas: Rentang Langfuse per langkah pengambilan + sintesis

## Build

1. **Ingestion walker.** Ulangi riwayat git di setiap push hook. Kumpulkan file yang diubah. Untuk setiap file, parsing dengan tree-sitter, ekstrak fungsi dan node kelas dengan rentang sumber penuhnya. Keluarkan rekaman potongan `{repo, path, start_line, end_line, symbol, body}`.

2. **Peringkasan potongan.** Kumpulan potongan ke dalam panggilan Haiku 4.5 dengan cache cepat pada pembukaan sistem. Prompt: "Ringkaslah fungsi ini dalam satu kalimat, sebutkan kontrak publik dan efek sampingnya." Simpan ringkasan di samping potongan.

3. **Kumpulan embedding.** Dua antrean paralel: padat (code Voyage-3 batch 128) dan ringkasan (model yang sama, tetapi pada string ringkasan). Tulis vector ke Qdrant dengan payload `{repo, path, start_line, end_line, symbol, kind}`.

4. **Indeks BM25.** Indeks Tantivy berbobot bidang: weight nama simbol 4, weight isi simbol 1, weight ringkasan 2. Mengaktifkan kueri "temukan fungsi bernama X" di samping "temukan fungsi yang melakukan X".

5. **Grafik simbol.** Untuk setiap potongan, tepi rekaman: impor (file ini menggunakan simbol Y dari repo Z), panggilan (fungsi ini memanggil metode M pada kelas C), pewarisan. Simpan di kuzu. Digunakan pada waktu kueri untuk memperluas pengambilan melintasi batas repo.

6. **Agen kueri.** LangGraph dengan tiga node. `retrieve` menembakkan padat + BM25 secara paralel, menghapus duplikat berdasarkan (repo, jalur, simbol). `rerank` menjalankan cross-encoder di 50 teratas dan mempertahankan 10 teratas. `synth` memanggil Claude Sonnet 4.7 dengan potongan yang diberi peringkat ulang sesuai konteks, menyimpan prompt sistem dalam cache, memerlukan kutipan file:line.

7. **Penerapan kutipan.** Mengurai output model; klaim apa pun tanpa jangkar `(repo/path:start-end)` akan ditandai untuk ditanyakan ulang atau dibatalkan. Kembalikan jawaban yang hanya dikutip kepada pengguna.

8. **Indeks ulang bertahap.** Di setiap webhook, hitung perbedaan tingkat simbol. Hanya sematkan kembali potongan yang teksnya berubah. Hitung ulang tepi simbol untuk potongan yang impornya berubah. Ukur: push 50 file yang diindeks ulang dalam waktu kurang dari 60 detik untuk armada 2M-LOC.

9. **Eval.** Beri label 100 pertanyaan lintas repo dengan file emas: jawaban baris. Ukur MRR@10, nDCG@10, kesetiaan kutipan (sebagian klaim dengan jangkar yang dapat diverifikasi), dan latensi p50/p99.

## Pakai

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## Kirim

Keterampilan yang dapat disampaikan `outputs/skill-codebase-rag.md`. Dengan adanya korpus repo, ia akan menjalankan jalur penyerapan, indeks hibrid, dan agen kueri, serta mengembalikan jawaban yang dikutip untuk pertanyaan lintas-repo apa pun. Rubrik:| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | Kualitas pengambilan | MRR@10 dan nDCG@10 pada set 100 pertanyaan |
| 20 | Kesetiaan kutipan | Sebagian dari klaim jawaban dengan file yang dapat diverifikasi: jangkar baris |
| 20 | Latensi dan skala | latensi kueri p95 pada 10k QPS pada ukuran korpus yang diindeks |
| 20 | Kebenaran pengindeksan tambahan | Waktu dari git push hingga dapat dicari pada komit 50 file |
| 15 | UX dan pemformatan jawaban | Kemampuan kutipan untuk diklik, pratinjau cuplikan, keterjangkauan tindak lanjut |
| **100** | | |

## Latihan

1. Tukar Voyage-code-3 dengan code nomic-embed yang dihosting sendiri. Ukur delta MRR@10. Laporkan apakah kesenjangan tersebut dapat diatasi dengan mengaktifkan pemeringkatan ulang.

2. Suntikkan 20% code yang dihasilkan (boilerplate yang diproduksi LLM) ke dalam korpus dan evaluasi ulang. Amati keracunan pengambilan. Tambahkan tanda "yang dihasilkan" ke payload dan kurangi weight serangan tersebut.

3. Tolok ukur pencarian hibrid Qdrant vs pgvector + pgvectorscale pada ukuran korpus kamu. Laporkan p99 pada ukuran batch 1.

4. Tambahkan pemeriksaan penyimpangan berbasis sample: setiap minggu, jalankan kembali evaluasi 100 pertanyaan. Peringatan pada MRR@10 penurunan > 5%.

5. Perluas ke resolusi simbol lintas bahasa: fungsi Python yang memanggil layanan Go melalui gRPC. Gunakan grafik simbol untuk menghubungkannya.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Pengelompokan sadar AST | "Pemisahan tingkat fungsi" | Memotong code pada batas simpul pengasuh pohon alih-alih jendela token tetap |
| Pencarian hibrida | "Padat + jarang" | Jalankan pencarian BM25 dan vector secara paralel, gabungkan top-k, rangking ulang |
| Pemeringkatan ulang lintas-encoder | "Peringkat phase kedua" | Model yang menilai setiap pasangan (kueri, kandidat) secara bersamaan, lebih akurat daripada kosinus |
| Caching cepat | "Permintaan sistem dalam cache" | Feature Claude / OpenAI 2026 yang mendiskon token awalan berulang hingga 90% |
| Grafik simbol | "Grafik code" | Tepian untuk impor, panggilan, pewarisan seluruh file dan repo |
| Kesetiaan kutipan | "Tingkat jawaban yang membumi" | Sebagian klaim yang dapat diverifikasi pengguna dengan mengeklik jangkar dan membaca rentang yang direferensikan |
| Indeks ulang tambahan | "Waktu push-to-search" | Jam dinding dari git push hingga simbol yang diubah dapat ditanyakan |

## Bacaan Lanjutan

- [Sourcegraph Amp](https://ampcode.com) — kecerdasan code lintas-repo produksi
- [Arsitektur Sourcegraph Cody RAG](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — referensi mendalam untuk batu penjuru ini
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tampilan repo peringkat tree-sitter
- [Grafik perusahaan Augment Code](https://www.augmentcode.com) — grafik simbol komersial RAG
- [Dokumen pencarian hibrid Qdrant](https://qdrant.tech/documentation/concepts/hybrid-queries/) — implementasi referensi
- [Embedding code Voyage AI](https://docs.voyageai.com/docs/embeddings) — Detail code Voyage-3
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — referensi lintas-encoder
- [Pencarian internal Pinterest MCP](https://medium.com/pinterest-engineering) — referensi platform internal
