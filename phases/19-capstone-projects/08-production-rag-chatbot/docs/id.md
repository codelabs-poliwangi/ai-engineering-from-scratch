# Capstone 08 — Chatbot RAG Produksi untuk Vertikal Teregulasi

> Harvey, Glean, Mendable, dan LlamaCloud semuanya menjalankan bentuk produksi yang sama pada tahun 2026. Serap dengan docling atau Unstructured dan ColPali untuk visual. Pencarian hibrida. Rangking ulang dengan bge-reranker-v2-gemma. Sintesis dengan Claude Sonnet 4.7 menggunakan cache cepat pada tingkat hit 60-80%. Jaga dengan Llama Guard 4 dan Pagar Pembatas NeMo. Tonton dengan Langfuse dan Phoenix. Nilai dengan RAGAS pada set emas 200 pertanyaan. Build satu di domain yang diatur (hukum, klinis, asuransi), dan puncaknya adalah melewati set emas, tim merah, dan dasbor drift.

**Type:** Batu penjuru
**Language:** Python (pipa + API), TypeScript (UI obrolan)
**Prerequisites:** Fase 5 (NLP), Fase 7 (Transformer), Fase 11 (rekayasa LLM), Fase 12 (multimoda), Fase 17 (infrastruktur), Fase 18 (keselamatan)
**Fase yang dilakukan:** P5 · P7 · P11 · P12 · P17 · P18
**Waktu:** 30 jam

## Masalah

RAG domain teregulasi (kontrak hukum, protokol uji klinis, polis asuransi) adalah bentuk produksi yang paling banyak dikirim pada tahun 2026 karena ROI-nya jelas dan taruhannya nyata. Harvey (Allen & Overy) membangunnya untuk legal. Mendable mengirimkan cita rasa dokumen pengembang. Glean mencakup pencarian perusahaan. Polanya adalah: menyerap fidelitas tinggi, mengambil hibrida dengan pemeringkatan ulang, mensintesis dengan penegakan kutipan dan cache cepat, menjaga dengan beberapa layer keamanan, dan memantau penyimpangan secara terus menerus.

Bagian yang sulit bukanlah modelnya. Hal tersebut adalah kepatuhan yang sadar yurisdiksi (HIPAA, GDPR, SOC2), kemampuan audit tingkat kutipan, pengendalian biaya (caching cepat membeli diskon 60-90% ketika tingkat hit tinggi), deteksi halusinasi melalui kesetiaan RAGAS, dan deteksi penyimpangan ketika dokumen sumber diperbarui tanpa indeks menyusul. Batu penjuru ini meminta kamu untuk mengirimkan semuanya dalam set emas 200 pertanyaan dengan suite tim merah di sampingnya.

## Konsep

Pipa tersebut memiliki dua sisi. **Penyerapan**: docling atau Parsing tidak terstruktur dokumen terstruktur; ColPali menangani yang kaya secara visual; potongan mendapatkan ringkasan, tag, dan label akses berbasis peran. Vector masuk ke pgvector + pgvectorscale (di bawah 50 juta vector) atau Qdrant Cloud; BM25 yang jarang berjalan di sampingnya. **Percakapan**: LangGraph menangani memori dan multi-putaran; setiap kueri menjalankan pengambilan hibrid, mengurutkan ulang dengan bge-reranker-v2-gemma-2b, mensintesis dengan Claude Sonnet 4.7 (prompt-cache), meneruskan output melalui Llama Guard 4 dan NeMo Guardrails, dan mengeluarkan respons berlabuh kutipan.

Tumpukan eval memiliki empat layer. **Set emas** (200 Q/A berlabel dengan kutipan) untuk kebenarannya. **Tim merah** (jailbreak, upaya ekstraksi PII, pertanyaan di luar domain) demi keamanan. **RAGAS** untuk kesetiaan / relevansi jawaban / presisi konteks secara otomatis per giliran. **Drift dashboard** (Arize Phoenix) mengamati kualitas pengambilan dan skor halusinasi setiap minggu.

Caching yang cepat adalah pengungkit biaya. Claude 4.5+ dan GPT-5+ mendukung system prompt caching + konteks yang diambil. Pada tingkat keberhasilan 60-80%, biaya per kueri turun 3-5x. Pipeline harus dirancang untuk awalan yang stabil (system prompt + konteks yang diberi peringkat ulang terlebih dahulu) untuk mencapai tingkat cache hit yang tinggi.

## Arsitektur

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## Tumpukan- Penyerapan: Unstructured.io atau docling untuk dokumen terstruktur; ColPali untuk PDF yang kaya visual
- DB Vector: pgvector + pgvectorscale di bawah 50 juta vector; Qdrant Cloud sebaliknya
- Jarang: Tantivy BM25 dengan weight lapangan
- Orkestrasi: Alur Kerja LlamaIndex (penyerapan) + LangGraph (percakapan)
- Re-ranker: bge-reranker-v2-gemma-2b dihosting sendiri atau Voyage rerank-2 dihosting
- LLM: Claude Sonnet 4.7 dengan cache cepat; fallback Llama 3.3 70B dihosting sendiri
- Eval: RAGAS 0.2 online, DeepEval untuk rangkaian halusinasi dan jailbreak
- Observabilitas: Langfuse dihosting sendiri dengan antrian anotasi; Arize Phoenix untuk melayang
- Pagar Pembatas: Pengklasifikasi input/output Llama Guard 4, kebijakan NeMo Guardrails v0.12, scrub Presidio PII
- Kepatuhan: label akses berbasis peran pada bongkahan; tag yurisdiksi untuk GDPR/HIPAA

## Build

1. **Penyerapan.** Parsing korpus kamu (1000-10000 dokumen untuk bangunan serius) dengan Tidak Terstruktur atau docling. Untuk halaman yang dipindai / visual yang berat, rutekan melalui ColPali. Menghasilkan potongan dengan ringkasan, label peran, tag yurisdiksi.

2. **Indeks.** Embedding padat (Voyage-3 atau Nomic-embed-v2) ke dalam pgvector + pgvectorscale. Indeks samping BM25 melalui Tantivy. Filter peran dan yurisdiksi sebagai payload.

3. **Pengambilan hibrid.** Filter berdasarkan peran+yurisdiksi terlebih dahulu; lalu paralel padat + BM25; bergabung dengan fusi peringkat timbal balik; 20 teratas untuk melakukan reranker; 5 teratas untuk synth.

4. **Sintesis dengan caching cepat.** System prompt + kebijakan statis di header cache; mengubah peringkat konteks sebagai ekstensi cache; pertanyaan pengguna sebagai akhiran yang tidak di-cache. Targetkan tingkat pencapaian cache 60-80% dalam kondisi stabil.

5. **Pagar Pembatas.** Llama Guard 4 di input; Rel Pagar Pembatas NeMo memblokir pertanyaan di luar domain atau topik yang dilarang oleh kebijakan; Presidio menghapus PII yang tidak disengaja dalam output; pasca-filter penegakan kutipan.

6. **Set emas.** 200 pasangan Tanya Jawab diberi label oleh pakar domain dengan (jawaban, kutipan). Agen skor pada kecocokan kutipan tepat, kebenaran jawaban, kesetiaan (RAGAS).

7. **Tim merah.** 50 petunjuk permusuhan: jailbreak (PAIR, TAP), upaya eksfiltrasi PII, kebocoran di luar domain, lintas yurisdiksi. Skor dengan lulus/gagal dan tingkat keparahan.

8. **Drift dashboard.** Arize Phoenix melacak kualitas pengambilan (nDCG, kesetiaan kutipan) setiap minggu. Peringatan penurunan 5%.

9. **Laporan biaya.** Langfuse: tingkat hit cache cepat, token per kueri, perincian $/kueri berdasarkan phase.

## Pakai

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## Kirim

`outputs/skill-production-rag.md` menjelaskan penyampaiannya. Chatbot dengan domain teregulasi diterapkan dengan label kepatuhan, melewati rubrik, dan diamati dengan pemantauan penyimpangan langsung.

| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | Kesetiaan RAGAS + relevansi jawaban | Skor online pada set emas (200 Q/A) |
| 20 | Kebenaran kutipan | Sebagian jawaban dengan jangkar sumber yang dapat diverifikasi |
| 20 | Cakupan pagar pembatas | Tingkat kelulusan Llama Guard 4 + hasil suite jailbreak |
| 20 | Rekayasa biaya / latensi | Tingkat hit cache cepat, latensi p95, $/query |
| 15 | Dasbor pemantauan drift | Dasbor langsung Phoenix dengan tren kualitas pengambilan mingguan |
| **100** | | |

## Latihan

1. Membangun bagian korpus kedua di bawah yurisdiksi yang berbeda (misalnya, HIPAA bersama GDPR). Tunjukkan pemfilteran peran+yurisdiksi yang mencegah kebocoran silang pada penyelidikan lintas yurisdiksi yang terdiri dari 20 pertanyaan.

2. Ukur tingkat keberhasilan cache cepat selama satu minggu lalu lintas produksi. Identifikasi kueri mana yang merusak awalan cache. Restrukturisasi.3. Tambahkan memori multi-putaran dengan buffer ringkasan 10 ribu token. Ukur apakah kesetiaan menurun seiring berkembangnya percakapan.

4. Tukar Claude Sonnet 4.7 dengan Llama 3.3 70B yang dihosting sendiri. Ukur $/query dan delta kesetiaan.

5. Tambahkan mode "tidak yakin": jika skor peringkat ulang teratas berada di bawah ambang batas, agen akan mengatakan "Saya tidak memiliki kutipan percaya diri" alih-alih menjawab. Ukur pengurangan kepercayaan palsu.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Caching cepat | "Sistem cache + konteks" | Feature Claude/OpenAI: token awalan yang di-cache didiskon 60-90% saat hit |
| RAGA | "Penilai RAG" | Penilaian otomatis atas kesetiaan, relevansi jawaban, ketepatan konteks |
| Set emas | "Berlabel eval" | 200+ Tanya Jawab berlabel ahli dengan kutipan; kebenaran dasar |
| Tag yurisdiksi | "Label kepatuhan" | Cakupan GDPR/HIPAA/SOC2 melekat pada bagian; diberlakukan oleh filter pengambilan |
| Kesetiaan kutipan | "Tingkat jawaban yang membumi" | Sebagian kecil klaim yang didukung oleh rentang sumber yang dapat diambil |
| Melayang | "Pembusukan kualitas pengambilan" | Perubahan mingguan pada nDCG atau skor kutipan; ambang peringatan 5% |
| Tim Merah | "Eval permusuhan" | Jailbreak pra-rilis, ekstraksi PII, pemeriksaan di luar domain |

## Bacaan Lanjutan

- [Harvey AI](https://www.harvey.ai) — referensi tumpukan produksi legal
- [Glean enterprise search](https://www.glean.com) — referensi RAG pada skala perusahaan
- [Dokumentasi yang dapat diperbaiki](https://mendable.ai) — referensi RAG dokumen pengembang
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — penyerapan terkelola
- [Caching prompt antropik](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — referensi pengungkit biaya
- [Dokumentasi RAGAS 0.2](https://docs.ragas.io/) — framework evaluasi RAG kanonik
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — referensi pengamatan penyimpangan
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — pengklasifikasi keamanan 2026
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — kerangka rel kebijakan
