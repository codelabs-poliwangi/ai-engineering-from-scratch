# Capstone 10 — Tim Rekayasa Perangkat Lunak Multi-Agen

> Arsitektur pabrik SWE-AF, dorongan berbasis peran MetaGPT, grafik aktor yang diketik AutoGen 0.4, Devin dari Cognition, dan Droid Pabrik semuanya menyatu pada bentuk tahun 2026 yang sama: seorang arsitek membuat rencana, N pembuat code bekerja di pohon kerja paralel, gerbang peninjau, dan penguji memverifikasi. Pohon kerja paralel mengubah jam dinding menjadi throughput. Protokol status dan handoff bersama menjadi permukaan kegagalan. Puncaknya adalah membangun tim, mengevaluasi SWE-bench Pro, dan melaporkan handoff mana yang gagal dan seberapa sering.

**Type:** Batu penjuru
**Language:** Python / TypeScript (agen), Shell (skrip pohon kerja)
**Prerequisites:** Fase 11 (rekayasa LLM), Fase 13 (peralatan), Fase 14 (agen), Fase 15 (otonom), Fase 16 (multi-agen), Fase 17 (infrastruktur)
**Fase yang dilakukan:** P11 · P13 · P14 · P15 · P16 · P17
**Waktu:** 40 jam

## Masalah

Memanfaatkan pengkodean agen tunggal mencapai puncaknya pada tugas-tugas besar. Bukan karena setiap agen lemah, namun karena konteks 200 ribu token tidak dapat menampung rencana arsitektur ditambah empat irisan basis code paralel ditambah komentar pengulas ditambah hasil pengujian. Pabrik multi-agen membagi masalahnya: arsitek memiliki rencana, pembuat code memiliki implementasi di pohon kerja paralel, gerbang peninjau, penguji memverifikasi. Arsitektur "pabrik" SWE-AF, peran MetaGPT, grafik aktor yang diketik AutoGen - ketiga framing menggambarkan bentuk yang sama.

Permukaan kegagalan adalah handoff. Arsitek merencanakan sesuatu yang tidak dapat dilaksanakan oleh pembuat code. Pembuat code menghasilkan perbedaan yang saling bertentangan. Peninjau menyetujui perbaikan halusinasi. Penguji balapan dengan pembuat code yang masih menulis. kamu akan membangun salah satu dari tim ini, menjalankannya pada 50 isu SWE-bench Pro, melacak setiap handoff, dan mempublikasikan post-mortem.

## Konsep

Peran adalah agen yang diketik. **Arsitek** (Claude Opus 4.7) membaca masalah, menulis rencana, dan memecahnya menjadi subtugas dengan antarmuka eksplisit. **Coder** (Claude Sonnet 4.7, N instance paralel, masing-masing dalam `git worktree` + Daytona sandbox) mengimplementasikan subtugas secara independen. **Reviewer** (GPT-5.4) membaca perbedaan yang digabungkan dan menyetujui atau meminta perubahan tertentu. **Penguji** (Gemini 2.5 Pro) menjalankan rangkaian pengujian secara terpisah dan melaporkan lulus/gagal dengan artefak.

Komunikasi dilakukan melalui papan tugas bersama (didukung file atau Redis). Setiap peran menggunakan tugas yang diizinkan untuk ditangani. Handoff adalah pesan bertipe protokol A2A. Masalah koordinasi: resolusi konflik penggabungan (peran koordinator atau penggabungan tiga arah otomatis), sinkronisasi negara bersama (rencana dibekukan setelah pembuat code memulai; perencanaan ulang adalah peristiwa terpisah), dan penjaga gerbang peninjau (peninjau tidak dapat menyetujui perubahannya sendiri atau perubahan yang diusulkannya).

Amplifikasi token adalah biaya tersembunyi. Setiap batasan peran menambahkan petunjuk ringkasan dan konteks penyerahan. Perjalanan agen tunggal sebanyak 40 putaran menjadi total 160 putaran dalam empat peran. Rubrik ini secara khusus mempertimbangkan efisiensi token vs dasar agen tunggal karena pertanyaannya bukan “apakah multi-agen berfungsi” tetapi “apakah ia menang per dolar.”

## Arsitektur

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## Tumpukan- Orkestrasi: LangGraph dengan status bersama + sub-grafik per agen
- Pesan: Protokol A2A (Google 2025) untuk pesan antar agen yang diketik
- Model: Opus 4.7 (arsitek), Sonnet 4.7 (coder), GPT-5.4 (reviewer), Gemini 2.5 Pro (tester)
- Isolasi Worktree: `git worktree add` per pembuat code + kotak pasir Daytona
- Koordinator penggabungan: penggabungan tiga arah khusus + resolusi konflik yang dimediasi LLM
- Eval: SWE-bench Pro (50 edisi), skenario SWE-AF, HumanEval++ untuk pengujian unit
- Observabilitas: Langfuse dengan rentang yang diberi tag peran, akuntansi token per agen
- Deployment: K8 dengan masing-masing peran sebagai Deployment + HPA terpisah di backlog

## Build

1. **Papan tugas.** JSONL yang didukung file dengan pesan yang diketik: `plan_request`, `subtask`, `diff_ready`, `review_needed`, `test_needed`, `approved`, `rejected`, `replan_needed`. Agen berlangganan tag.

2. **Architect.** Membaca masalah GitHub, menjalankan Opus 4.7 dengan templat rencana yang memerlukan antarmuka subtugas eksplisit (file disentuh, fungsi publik, dampak pengujian). Memancarkan satu `plan_request` dengan subtugas DAG.

3. **Coders.** N pekerja paralel, masing-masing mengklaim satu subtugas dari dewan. Masing-masing memunculkan cabang `git worktree add` baru ditambah kotak pasir Daytona. Mengimplementasikan subtugas. Memancarkan `diff_ready` dengan patch + delta pengujian.

4. **Gabungkan koordinator.** Setelah semua pembuat code selesai, tiga arah menggabungkan N cabang menjadi cabang pementasan. Penyelesaian konflik yang dimediasi LLM hanya jika terjadi tumpang tindih tingkat file.

5. **Reviewer.** GPT-5.4 membaca perbedaan yang digabungkan. Tidak dapat menyetujui perbedaan yang dibuatnya. Memancarkan `approved` (tanpa pengoperasian) atau `review_feedback` dengan permintaan perubahan spesifik yang dialihkan kembali ke pembuat code yang relevan.

6. **Penguji.** Gemini 2.5 Pro menjalankan rangkaian pengujian di sandbox yang bersih. Menangkap artefak. Memancarkan `test_passed` atau `test_failed` dengan stacktraces. Pengujian yang gagal diulang kembali ke pembuat code yang memiliki subtugas yang gagal.

7. **Akuntansi handoff.** Setiap pesan yang melintasi batas peran akan mendapat rentang di Langfuse dengan ukuran payload dan model yang digunakan. Hitung amplifikasi token per subtugas (coder_tokens + reviewer_tokens + tester_tokens + Architect_share / coder_tokens).

8. **Eval.** Jalankan pada 50 edisi SWE-bench Pro. Bandingkan pass@1 dan $-per-solved-issue dengan baseline agen tunggal (satu Sonnet 4.7 dalam satu pohon kerja).

9. **Post-mortem.** Untuk setiap terbitan yang gagal, identifikasi handoff yang gagal (rencana terlalu kabur, konflik gabungan, peninjau memberikan persetujuan palsu, penguji gagal). Menghasilkan histogram kegagalan handoff.

## Pakai

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## Kirim

`outputs/skill-multi-agent-team.md` adalah hasil yang dapat dicapai. Mengingat URL masalah dan tingkat paralelisme, tim menghasilkan PR yang siap digabungkan dengan penghitungan token per peran.

| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | Tiket Pro bangku SWE@1 | Subset 50 terbitan yang cocok, pass@1 |
| 20 | Percepatan paralel | Garis dasar jam dinding vs agen tunggal |
| 20 | Kualitas ulasan | Tingkat persetujuan palsu pada pemeriksaan bug yang disuntikkan |
| 20 | Efisiensi token | Total token per masalah terpecahkan vs agen tunggal |
| 15 | Teknik Koordinasi | Resolusi konflik gabungan, histogram kegagalan handoff |
| **100** | | |

## Latihan

1. Masukkan bug yang jelas ke dalam perbedaan di tengah proses (ekstra `return None` sebelum bagian utama). Ukur tingkat persetujuan palsu dari pengulas. Sesuaikan permintaan pengulas hingga persetujuan palsu di bawah 5%.2. Kurangi menjadi dua pembuat code (arsitek + pembuat code + reviewer + penguji, pembuat code menjalankan dua subtugas secara berurutan). Bandingkan jam dinding dan tingkat kelulusan.

3. Ganti koordinator penggabungan dengan batasan penulis tunggal (subtugas menyentuh kumpulan file yang terpisah). Ukur weight perencanaan pada arsitek.

4. Tukar pengulas dari GPT-5.4 ke Claude Opus 4.7. Ukur tingkat persetujuan palsu dan delta biaya token.

5. Tambahkan peran kelima: dokumenter (Haiku 4.5). Setelah ditinjau, ini menghasilkan entri log perubahan. Ukur apakah kualitas dokumentasi membenarkan pengeluaran token ekstra.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Pohon kerja paralel | "Cabang terisolasi" | `git worktree add` menghasilkan pohon kerja baru per pembuat code |
| Papan tugas | "Bus pesan bersama" | File atau Redis menyimpan agen pesan yang diketik berlangganan |
| Penyerahan | "Batas peran" | Pesan apa pun yang berpindah dari konteks satu peran ke konteks peran lainnya |
| Amplifikasi token | "Overhead multi-agen" | Total token di seluruh peran/token agen tunggal untuk tugas yang sama |
| Protokol A2A | "Agen-ke-agen" | Spesifikasi Google tahun 2025 untuk pesan antar agen yang diketik |
| Gabungkan koordinator | "Integrator" | Komponen yang menjalankan penggabungan tiga arah dan memediasi konflik |
| Persetujuan palsu | "Halusinasi pengulas" | Peninjau menyetujui perbedaan dengan bug yang diketahui |

## Bacaan Lanjutan

- [Arsitektur pabrik SWE-AF](https://github.com/Agent-Field/SWE-AF) — referensi pabrik multi-agen 2026
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — framework multi-agen berbasis peran
- [AutoGen v0.4](https://github.com/microsoft/autogen) — framework aktor yang diketik Microsoft
- [Cognition AI (Devin)](https://cognition.ai) — produk referensi
- [Factory Droids](https://www.factory.ai) — produk referensi alternatif
- [Protokol Google A2A](https://developers.google.com/agent-to-agent) — spesifikasi perpesanan antar-agen
- [dokumentasi git worktree](https://git-scm.com/docs/git-worktree) — media isolasi
- [SWE-bench Pro](https://www.swebench.com) — target evaluasi
