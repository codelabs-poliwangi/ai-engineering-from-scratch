# Capstone 05 — Agen Penelitian Otonom (Kelas Ilmuwan AI)

> AI-Scientist-v2 Sakana menerbitkan makalah lengkap. Agen Laboratorium menjalankan eksperimen. Allen AI membagikan jejaknya. Bentuk tahun 2026 adalah penelusuran pohon rencana-eksekusi-verifikasi atas eksperimen, biaya yang dianggarkan, eksekusi code dalam kotak pasir, penulis LaTeX umpan balik visi, dan ansambel peninjau gaya NeurIPS otomatis. Batu penjurunya adalah membuatnya, menjalankannya dari ujung ke ujung dengan biaya $30 per kertas, dan bertahan dari tim merah pelarian kotak pasir yang didokumentasikan Sakana.

**Type:** Batu penjuru
**Language:** Python (agen + sandbox), LaTeX (output)
**Prerequisites:** Fase 2 (ML), Fase 3 (pembelajaran mendalam), Fase 7 (Transformer), Fase 10 (LLM dari awal), Fase 14 (agen), Fase 15 (otonom), Fase 16 (multi-agen), Fase 18 (keamanan)
**Fase yang dilakukan:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**Waktu:** 40 jam

## Masalah

Agen penelitian otonom melewati ambang batas pada tahun 2026. AI-Scientist-v2 Sakana AI diterbitkan di Nature dengan makalah yang dihasilkan dan lolos tinjauan sejawat lokakarya. ShinkaEvolve (ICLR 2026) memperluas jangkauan hipotesis yang terus berkembang. Laboratorium Agen AMD mengirimkan jejak yang dapat direproduksi. Agen-agen tersebut tidaklah ajaib — mereka adalah putaran rencana-eksekusi-verifikasi yang berjalan pada pohon kandidat eksperimen, dengan batasan biaya, kotak pasir yang terikat pada benih, dan peninjauan otomatis. Kerajinan itu ada dalam lingkaran, anggaran, dan kisah keselamatan.

kamu mempelajari perulangan dengan menerapkannya terhadap ide awal dalam domain sempit (misalnya, ablasi ketersebaran attention pada Transformer parameter 100M). Nilainya bukanlah menemukan sesuatu yang baru pada percobaan pertama. Nilainya ada pada infrastruktur: penelusuran pohon, kotak pasir eksperimen, putaran penulis-peninjau, laporan tim merah. Tim Sakana mendokumentasikan kegagalan pelarian sandbox; agen kamu harus melewati tim merah yang sama.

## Konsep

Agen adalah pencarian pohon terbaik pertama. Node adalah spesifikasi eksperimen: (hipotesis, konfigurasi, code, hasil yang diharapkan). Langkah perluasan mengusulkan anak-anak dengan pengeditan kecil (optimizer swap, pergeseran ukuran batch, ablasi komponen). Setiap anak berjalan di kotak pasir baru dengan batasan sumber daya keras. Hasil diumpankan kembali ke dalam fungsi penilaian yang memberi peringkat node berdasarkan (kebaruan × kualitas × sisa anggaran). Pohon itu tumbuh sampai anggaran habis, barulah dituliskan cabang terbaik.

Penulisnya multimodal. Ini menghasilkan draf LaTeX, mengkompilasinya, merender gambar, dan memasukkan PDF yang dirender kembali ke mode visi Claude Opus 4.7 untuk kritik terhadap tata letak, keterbacaan gambar, dan penyelarasan bukti klaim. Ansambel pengulas yang terdiri dari lima juri LLM mengeluarkan skor gaya NeurIPS (kebaruan, ketelitian, kejelasan, reproduktifitas, dampak); jika rata-rata turun di bawah ambang batas, makalah akan dikembalikan ke penulis dengan kritik.

Keamanan adalah penahan weight. Setiap eksperimen berjalan di sandbox E2B atau Daytona tanpa jalan keluar jaringan, jam dinding terbatas, dan batasan sumber daya yang dipasangi pin. Langkah pembuatan code agen melewati layer kebijakan yang memblokir panggilan sistem yang keluar dari kotak pasir. Laporan tim merah mereproduksi permukaan serangan yang didokumentasikan Sakana (bom garpu, sistem file yang lolos, panggilan jaringan yang ditulis LLM).

## Arsitektur

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## Tumpukan- Orkestrasi: LangGraph dengan pos pemeriksaan dan gerbang persetujuan manusia
- Pencarian pohon: kustom terbaik-pertama melalui node eksperimen (gaya AB-MCTS dari Sakana v2)
- Sandbox: E2B per eksperimen, fallback Docker-in-Docker; batasan sumber daya melalui cgroups
- Sastra: API Grafik Cendekiawan Semantik + OpenAlex + cache abstrak FAISS lokal
- Penulis: Templat LaTeX + Claude Opus 4.7 (mode visi) untuk kritik gambar dan tata letak
- Reviewer: ansambel 5 juri (Opus 4.7, GPT-5.4, Gemini 3 Pro, DeepSeek R1, Qwen3-Max) dengan agregasi tertimbang
- Kerangka eksperimen: PyTorch 2.5 untuk eksperimen fisik, W&B untuk logging
- Observabilitas: Langfuse untuk jejak agen, anggaran keras $30 per kertas

## Build

1. **Pelingkupan benih dan domain.** Ambil ide awal (misalnya, "selidiki pola ketersebaran dalam peta attention Transformer sub-1B"). Tentukan ruang pencarian: model, dataset, anggaran komputasi.

2. **Lulus Sastra.** Kueri Semantic Scholar + OpenAlex untuk 50 makalah relevan yang paling banyak dikutip; cache abstrak secara lokal; menghasilkan intisari domain 1 halaman.

3. **Perancah pohon.** Inisialisasi akar dengan hipotesis benih. Terapkan `expand(node) -> children` dengan proposal pengeditan kecil (satu perubahan konfigurasi per turunan). Terapkan `score(node)` sebagai hal baru × kualitas × anggaran.

4. **Pembungkusan Sandbox.** Setiap eksperimen berjalan `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only` (atau kebijakan E2B yang setara). Benih ditulis ke kotak pasir; output dipasang kembali hanya-baca.

5. **Perulangan rencana-eksekusi-verifikasi.** `plan` mengusulkan turunan. `execute` menjalankan sandbox, menangkap log dan metrik. `verify` menjalankan pemeriksaan unit pada metrik (apakah kerugiannya berkurang? apakah ablasi mengisolasi efeknya?). Node yang gagal mendapatkan alasan kegagalan yang disimpan di pohon.

6. **Penulis.** Setelah anggaran, pilih cabang terbaik. Render angka dengan matplotlib. Hasilkan draf LaTeX melalui Claude Opus 4.7 dengan jejak cabang dalam konteks. Menyusun. Masukkan PDF yang telah dikompilasi kembali ke visi Opus 4.7 untuk dikritik. Ulangi.

7. **Ansambel pengulas.** Lima juri menilai draf tersebut (kebaruan, ketelitian, kejelasan, reproduktifitas, dampak) dengan rubrik bergaya NeurIPS. Jika mean < 4.0/5, kembali ke penulis dengan kritik. Berhenti sulit setelah 3 kali penulisan ulang.

8. **Tim merah.** Build atau integrasikan serangkaian tugas musuh yang menargetkan kotak pasir: bom fork, upaya eksfiltrasi jaringan, pelolosan sistem file, metakarakter shell yang ditulis LLM. Konfirmasikan semua diblokir. Tuliskan temuan.

9. **Reprodusibilitas.** Setiap makalah dikirimkan dengan jejak penelusuran pohon JSON, seed, tautan proses W&B, konfigurasi sandbox, dan README yang mereproduksinya secara menyeluruh.

## Pakai

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## Kirim

`outputs/skill-ai-scientist.md` adalah hasil yang dapat dicapai. Dengan adanya ide awal + domain + anggaran $30, ide tersebut menjalankan keseluruhan alur dan menghasilkan makalah yang dapat ditinjau ditambah paket reproduktifitas.

| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | Kualitas kertas | Tinjauan rubrik buta terhadap makalah lokakarya yang diterbitkan |
| 20 | Ketelitian eksperimental | Garis dasar, benih, ablasi; setiap klaim didukung oleh sel di tabel hasil |
| 20 | Disiplin biaya dan komputasi | $30/pagu kertas diberlakukan, penelusuran Langfuse |
| 20 | Keamanan | Umpan tim merah kotak pasir; kebijakan jaringan dan tombol pemutus terverifikasi |
| 15 | Reproduksibilitas | Pemutaran ulang satu prompt dengan benih yang identik mereproduksi kertas |
| **100** | | |

## Latihan1. Jalankan alur terhadap tiga gagasan awal yang berbeda dalam domain yang sama. Bandingkan bagian pencarian pohon mana yang tumpang tindih. Identifikasi duplikat komputasi terbuang.

2. Tambahkan gerbang human-in-the-loop sebelum eksekusi eksperimen untuk node yang diperkirakan di atas $5. Ukur berapa total biaya yang turun.

3. Tukar ansambel pengulas dengan satu juri. Ukur tingkat penerimaan palsu pada sekumpulan makalah yang diketahui jelek.

4. Memperkenalkan pengujian tim merah eksfiltrasi jaringan: agen menulis code yang mencoba ke `curl` alamat eksternal. Konfirmasikan pemblokiran kebijakan `--network=none`. Catat upaya tersebut.

5. Bandingkan penelusuran pohon kamu dengan baseline acak datar (anggaran sama, tanpa strategi ekspansi). Laporkan hal baru × peningkatan kualitas.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Pencarian pohon | "Ekspansi gaya AB-MCTS" | Eksplorasi terbaik pertama pada node eksperimen dengan skor kebaruan×kualitas×anggaran |
| Kotak Pasir | "Eksperimen isolasi" | Kontainer tanpa jaringan, CPU/memori terbatas, seed yang di-embed, input read-only |
| Kritik visi | "Render-lalu-baca" | Kompilasi makalah ke PDF, masukkan PDF kembali ke VLM untuk tata letak dan kritik bukti klaim |
| Ansambel pengulas | "Tinjauan sejawat otomatis" | Beberapa juri LLM menilai makalah dengan rubrik NeurIPS; agregat tertimbang gerbang pipa |
| Skor baru | "Apakah ini baru?" | Heuristik yang menghukum kedekatan dengan cache literatur 50 makalah |
| Plafon biaya | "$anggaran" | Batasan tegas pada total pembelanjaan per makalah; Penghitung Langfuse + perkiraan pra-jalankan |
| Tim Merah | "Audit pelarian kotak pasir" | Tugas permusuhan yang akan lolos dari kotak pasir jika kebijakannya salah |

## Bacaan Lanjutan

- [Repositori Sakana AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) — agen penelitian produksi referensi
- [Makalah Sakana AI-Scientist-v1 (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — metodologi asli
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — ekstensi evolusioner
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — framework laboratorium penelitian multi-peran
- [Dokumentasi LangGraph](https://langchain-ai.github.io/langgraph/) — layer orkestrasi referensi
- [API Grafik Cendekiawan Semantik](https://api.semanticscholar.org/) — pencarian literatur
- [Kotak pasir E2B](https://e2b.dev) — isolasi eksperimen referensi
- [Pedoman pengulas NeurIPS](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — rubrik yang dikodekan oleh ansambel pengulas
