# Capstone 01 — Agen Pengkodean Terminal-Native

> Pada tahun 2026, bentuk agen pengkodean sudah ditetapkan. Tali pengaman TUI, rencana yang megah, permukaan alat yang dikotak pasir, lingkaran yang merencanakan, bertindak, mengamati, memulihkan. Claude Code, Cursor 3, dan OpenCode semuanya terlihat sama dari distance 50 kaki. Batu penjuru ini meminta kamu untuk membangun satu ujung ke ujung - CLI masuk, menarik permintaan keluar - dan mengukurnya terhadap mini-swe-agent dan Live-SWE-agent di SWE-bench Pro. kamu akan mempelajari mengapa bagian yang sulit bukanlah pemanggilan model tetapi putaran alat, kotak pasir, dan batas biaya pada putaran 50 putaran.

**Type:** Batu penjuru
**Language:** TypeScript / Bun (harness), Python (skrip eval)
**Prerequisites:** Fase 11 (rekayasa LLM), Fase 13 (alat dan protokol), Fase 14 (agen), Fase 15 (sistem otonom), Fase 17 (infrastruktur)
**Fase yang dilakukan:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**Waktu:** 35 jam

## Masalah

Agen pengkodean menjadi kategori aplikasi AI yang dominan pada tahun 2026. Claude Code (Anthropic), Cursor 3 dengan Composer 2 dan Agent Tabs (Cursor), Amp (Sourcegraph), OpenCode (112k bintang), Factory Droids, dan Google Jules semuanya mengirimkan variasi arsitektur yang sama: terminal harness, permukaan alat yang diizinkan, kotak pasir, dan loop plan-act-observe yang dibangun di sekitar model perbatasan. Perbatasannya sempit - Agen Live-SWE mencapai 79,2% di bangku SWE Terverifikasi dengan Opus 4.5 - tetapi bidang tekniknya luas. Kebanyakan mode kegagalan bukanlah kesalahan model. Hal tersebut adalah ketidakstabilan loop alat, keracunan konteks, biaya token yang tidak terkendali, dan operasi sistem file yang merusak.

kamu tidak dapat memikirkan agen-agen ini dari luar. kamu harus membuatnya, melihat loop crash di tikungan 47 ketika ripgrep mengembalikan 8MB kecocokan, dan membangun kembali layer pemotongan. Itulah inti dari batu penjuru ini.

## Konsep

Harness memiliki empat permukaan. **Plan** mempertahankan objek status bergaya TodoWrite yang ditulis ulang oleh model setiap giliran. **Act** mengirimkan panggilan alat (baca, edit, jalankan, cari, git). **Amati** menangkap code stdout / stderr / keluar, memotong, dan memasukkan kembali ringkasannya. **Pulihkan** menangani kesalahan alat tanpa merusak jendela konteks atau mengulang selamanya. Bentuk tahun 2026 menambahkan satu hal lagi: **kait**. `PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `Notification`, `Stop`, dan `PreCompact` — ekstensi yang dapat dikonfigurasi titik di mana operator menyuntikkan kebijakan, telemetri, dan pagar pembatas.

Kotak pasirnya adalah E2B atau Daytona. Setiap tugas berjalan di devcontainer baru dengan git worktree yang terpasang baca-tulis. Harness tidak pernah menyentuh sistem file host. Pohon kerja dirobohkan karena keberhasilan atau kegagalan. Pengendalian biaya diterapkan pada tiga layer: batas atas token per putaran, anggaran dolar per sesi, dan batas putaran keras (biasanya 50). Layer observabilitas adalah rentang OpenTelemetry dengan konvensi semantik GenAI, dikirimkan ke Langfuse yang dihosting sendiri.

## Arsitektur

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## Tumpukan- Waktu proses Harness: Bun 1.2 + Tinta 5 (React-in-terminal)
- Akses model: OpenRouter API terpadu dengan Claude Sonnet 4.7, GPT-5.4-Codex, Gemini 3 Pro, Opus 4.5 (untuk tugas tersulit)
- Transportasi alat: Model Context Protocol StreamableHTTP (revisi MCP 2026)
- Kotak Pasir: Kotak pasir E2B (JS SDK) atau wadah pengembang Daytona
- Pencarian code: subproses ripgrep, parser pengasuh pohon untuk 17 bahasa (dikompilasi sebelumnya)
- Isolasi: `git worktree add` per tugas, pembersihan jika berhasil/gagal
- Eval harness: SWE-bench Pro (subset terverifikasi) + Terminal-Bench 2.0 + 30 tugas kamu sendiri
- Observabilitas: OpenTelemetry SDK dengan `gen_ai.*` semconv → Langfuse yang dihosting sendiri
- Posting PR: Aplikasi GitHub dengan token terperinci, cakupannya terbatas pada repo target

## Build

1. **TUI dan loop prompt.** Perancah proyek Bun dengan Tinta. Terima `agent run <repo> "<task>"`. Cetak tampilan terpisah: panel rencana (atas), aliran panggilan alat (tengah), anggaran token (bawah). Tambahkan pembatalan pada Ctrl-C yang mengaktifkan kait `SessionEnd` sebelum keluar.

2. **Keadaan rencana.** Tentukan skema TodoWrite yang diketik (item tertunda/dalam proses/selesai dengan catatan). Model menulis ulang status penuh setiap putaran sebagai panggilan alat — jangan biarkan model bermutasi secara bertahap. Pertahankan rencana ke `.agent/state.json` sehingga kerusakan dapat berlanjut.

3. **Permukaan alat.** Tentukan enam alat: `read_file`, `edit_file` (dengan pratinjau berbeda), `ripgrep`, `tree_sitter_symbols`, `run_shell` (dengan batas waktu), `git` (status / diff / komit / dorong). Ekspos melalui MCP StreamableHTTP sehingga harness bersifat transport-agnostic. Setiap alat mengembalikan output yang terpotong (dibatasi hingga 4 ribu token per panggilan).

4. **Pembungkusan sandbox.** Setiap tugas memunculkan sandbox E2B. `git worktree add -b agent/$TASK_ID` cabang baru. Semua panggilan alat dijalankan di dalam kotak pasir. Sistem file host tidak dapat dijangkau.

5. **Hooks.** Menerapkan kedelapan jenis hook 2026. Hubungkan setidaknya empat hook yang dibuat oleh pengguna: (a) `PreToolUse` penjaga prompt destruktif yang memblokir `rm -rf` di luar pohon kerja, (b) `PostToolUse` penghitungan token, (c) `SessionStart` inisialisasi anggaran, (d) `Stop` menulis final bundel jejak.

6. **Eval loop.** Mengkloning subset SWE-bench Pro Python yang berisi 30 edisi. Jalankan harness kamu pada masing-masingnya. Bandingkan dengan mini-swe-agent (garis dasar minimal) pada pass@1, turn-per-task, dan $-per-task. Tulis hasilnya ke `eval/results.jsonl`.

7. **Pengendalian biaya.** Batasan sulit: 50 putaran, konteks 200 ribu, $5 per tugas. `PreCompact` hook merangkum belokan lama menjadi blok keadaan sebelumnya pada angka 150k, memberikan ruang untuk pengamatan baru tanpa kehilangan rencana.

8. **Postingan PR.** Jika berhasil, langkah terakhir adalah `git push` + panggilan API GitHub yang membuka PR dengan rencana dan ringkasan perbedaan di bagian isi.

## Pakai

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## Kirim

Keterampilan yang dapat disampaikan ada di `outputs/skill-terminal-coding-agent.md`. Dengan adanya jalur repo dan deskripsi tugas, ia menjalankan perulangan plan-act-observe secara penuh di sandbox dan mengembalikan URL PR plus bundel jejak. Rubrik batu penjuru ini:| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs baseline | Harness kamu vs mini-swe-agent pada 30 tugas Python yang cocok |
| 20 | Kejelasan arsitektur | Merencanakan/bertindak/mengamati pemisahan, permukaan kait, skema alat — ditinjau berdasarkan tata letak agen Live-SWE |
| 20 | Keamanan | Tes pelarian kotak pasir, permintaan izin, penjaga prompt destruktif melewati tim merah |
| 20 | Observabilitas | Kelengkapan penelusuran (100% panggilan alat terentang), penghitungan token per giliran |
| 15 | Pengembang UX | Mulai dingin < 2 detik, pemulihan kerusakan melanjutkan rencana, Ctrl-C membatalkan alat tengah dengan bersih |
| **100** | | |

## Latihan

1. Tukar model pendukung dari Claude Sonnet 4.7 ke Qwen3-Coder-30B yang disajikan di vLLM. Bandingkan pass@1 dan $-per-tugas. Laporkan kinerja model terbuka yang buruk.

2. Tambahkan sub-agen `reviewer` yang membaca perbedaan sebelum posting PR dan dapat meminta putaran revisi. Ukur apakah ulasan positif palsu menurunkan tingkat kelulusan SWE di bawah garis dasar agen tunggal (petunjuk: biasanya ya).

3. Uji stres kotak pasir: tulis tugas yang mencoba `curl` URL eksternal dan tugas yang menulis di luar pohon kerja. Konfirmasikan keduanya diblokir oleh kait PreToolUse. Catat upayanya.

4. Menerapkan ringkasan `PreCompact` dengan model yang lebih kecil (Haiku 4.5). Ukur berapa banyak ketelitian rencana yang hilang pada pemadatan 3x.

5. Tukar transport MCP StreamableHTTP dengan stdio. Tolok ukur latensi cold-start dan per panggilan. Pilih pemenang untuk penggunaan lokal saja.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Memanfaatkan | "Lingkaran agen" | Code yang mengelilingi model yang mengirimkan alat, mempertahankan status rencana, dan menerapkan anggaran |
| Kait | "Agen pendengar acara" | Skrip buatan pengguna dijalankan pada salah satu dari delapan peristiwa siklus hidup dengan memanfaatkan |
| Pohon Kerja | "Kotak pasir Git" | Checkout git tertaut di jalur terpisah; sekali pakai tanpa menyentuh klon utama |
| TodoWrite | "Rencana negara" | Daftar item yang tertunda/sedang berlangsung/selesai yang diketik yang ditulis ulang oleh model setiap giliran |
| StreamableHTTP | "Transportasi MCP" | Revisi MCP 2026: koneksi HTTP berumur panjang dengan streaming dua arah; menggantikan SSE |
| Plafon token | "Konteks anggaran" | Batasan per putaran atau per sesi pada token input+output; memicu pemadatan atau penghentian |
| lulus@1 | "Tingkat kelulusan percobaan tunggal" | Sebagian kecil tugas bangku SWE diselesaikan saat pertama kali dijalankan tanpa percobaan ulang atau pengintipan set pengujian |

## Bacaan Lanjutan

- [Dokumentasi Code Claude](https://docs.anthropic.com/en/docs/claude-code) — memanfaatkan referensi dari Anthropic
- [Cursor 3 changelog](https://cursor.com/changelog) — Tab Agen dan catatan produk Komposer 2
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — garis dasar minimal untuk perbandingan harness bangku SWE
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 79,2% SWE-bench Diverifikasi dengan Opus 4.5
- [OpenCode](https://opencode.ai) — tali pengaman terbuka, 112 ribu bintang
- [SWE-bench Pro leaderboard](https://www.swebench.com) — evaluasi target batu penjuru ini
- [Peta jalan Model Context Protocol 2026](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, metadata kemampuan
- [Konvensi semantik OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — skema rentang untuk panggilan alat dan penggunaan token
