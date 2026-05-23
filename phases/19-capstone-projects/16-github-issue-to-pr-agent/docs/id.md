# Capstone 16 - Agen Otonom Penerbitan ke PR GitHub

> Agen AWS Remote SWE, Agen Latar Belakang Kursor, Cloud OpenAI Codex, dan Google Jules semuanya mengirimkan bentuk produk tahun 2026 yang sama: beri label masalah, dapatkan PR. Jalankan agen di cloud sandbox, verifikasi kelulusan tes, dan posting PR yang siap ditinjau beserta alasannya. Bagian tersulitnya adalah mereproduksi lingkungan build repo secara otomatis, mencegah kebocoran kredensial, menerapkan anggaran per-repo, dan memastikan agen tidak dapat melakukan push paksa. Batu penjuru ini membuat versi yang dihosting sendiri dan membandingkannya dalam hal biaya dan tingkat kelulusan dengan alternatif yang dihosting.

**Type:** Batu penjuru
**Language:** Python (agen), TypeScript (Aplikasi GitHub), YAML (Tindakan)
**Prerequisites:** Fase 11 (rekayasa LLM), Fase 13 (peralatan), Fase 14 (agen), Fase 15 (otonom), Fase 17 (infrastruktur)
**Fase yang dilakukan:** P11 · P13 · P14 · P15 · P17
**Waktu:** 30 jam

## Masalah

Agen pengkodean cloud async adalah kategori produk terpisah dari agen pengkodean interaktif (batu penjuru 01). UX adalah label GitHub. kamu memberi label masalah `@agent fix this`, pekerja menjalankannya di cloud sandbox, mengkloning repo, menjalankan pengujian, mengedit file, memverifikasi, dan membuka PR dengan alasan agen di isi. Tidak ada loop interaktif, tidak ada terminal. Agen AWS Remote SWE, Agen Latar Belakang Kursor, cloud OpenAI Codex, Google Jules, dan Factory Droids semuanya menyatu dalam hal ini.

Tantangan tekniknya sangat nyata: reproduksi lingkungan (agen harus membuat repo dari awal tanpa gambar pengembang yang di-cache), pengujian yang tidak stabil (harus dijalankan ulang atau diisolasi), pelingkupan kredensial (Aplikasi GitHub dengan izin minimal), penegakan anggaran per repo per hari, dan kebijakan tanpa paksaan. Batu penjuru mengukur tingkat kelulusan, biaya, dan keamanan vs alternatif yang dihosting.

## Konsep

Pemicunya adalah webhook GitHub (label terbitan atau komentar PR). Petugas operator memasukkan pekerjaan ke ECS Fargate atau Lambda. Pekerja menarik repo ke dalam sandbox Daytona atau E2B dengan Dockerfile generik yang disimpulkan dari repo (bahasa, framework). Agen menjalankan loop mini-swe-agent atau SWE-agent v2 terhadap Claude Opus 4.7 atau GPT-5.4-Codex. Ini mengulangi: membaca code, mengusulkan perbaikan, menerapkan patch, menjalankan tes.

Verifikasi adalah langkah gerbang. CI penuh harus masuk ke sandbox sebelum PR dibuka. Delta cakupan dihitung; jika negatif melampaui ambang batas, PR terbuka tetapi diberi label `needs-review`. Agen memposting alasannya sebagai deskripsi PR ditambah rangkaian `@agent` yang dapat di-ping oleh pengulas untuk tindak lanjut.

Keamanan dicakup melalui dua permukaan GitHub yang berbeda: Aplikasi menyediakan token instalasi berumur pendek dengan `workflows: read` dan cakupan isi repo/PR yang sempit; perlindungan cabang (bukan izin aplikasi) menerapkan "tidak ada penulisan langsung ke `main`" dan "tidak ada dorongan paksa" — aplikasi tidak pernah ditambahkan ke daftar bypass. Akses baca-saja dengan cakupan jalur ke `.github/workflows` bukanlah aplikasi GitHub yang primitif, sehingga daftar izin agen pada pengeditan file harus menerapkan hal tersebut pada pekerja. Plafon anggaran per repo per hari diberlakukan di petugas operator (misalnya, maksimal 5 PR per repo per hari, $20 per PR).

## Arsitektur

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## Tumpukan- Pemicu: Aplikasi GitHub dengan token terperinci; penerima webhook melalui Lambda atau Fly.io
- Pekerja: tugas ECS Fargate (atau runner yang dihosting sendiri oleh GitHub Actions)
- Sandbox: Daytona devcontainer atau sandbox E2B per tugas
- Agent loop: baseline mini-swe-agent atau SWE-agent v2 melalui Claude Opus 4.7 / GPT-5.4-Codex
- Pengambilan: peta repo pengasuh pohon + ripgrep
- Verifikasi: CI penuh dalam kotak pasir + gerbang delta cakupan
- Observabilitas: Langfuse dengan arsip jejak per-PR yang ditautkan dari badan PR
- Anggaran: plafon dolar harian per-repo; PR maks per repo per hari

## Build

1. **Aplikasi GitHub.** Token instalasi yang terperinci: masalah baca+tulis, pull_requests tulis, baca+tulis konten, baca alur kerja. Perlindungan cabang (satu-satunya permukaan yang dapat melakukan ini) menerapkan "tidak ada dorongan langsung ke `main`" dan "tidak ada dorongan paksa"; aplikasi tidak ada dalam daftar bypass. Pekerja menerapkan "tidak ada penulisan di bawah `.github/workflows`" sebagai pemeriksaan daftar yang diizinkan pada perbedaan yang diusulkan, karena izin Aplikasi GitHub tidak tercakup dalam jalur.

2. **Penerima webhook.** Fungsi Lambda menerima webhook label terbitan / komentar PR. Filter menurut label `@agent fix this`. Antrean ke SQS.

3. **Dispatcher.** Memunculkan tugas dari SQS. Menerapkan anggaran per repo per hari. Menjalankan tugas ECS Fargate dengan URL repo, isi penerbitan, dan sandbox Daytona baru.

4. **Inference lingkungan.** Mendeteksi bahasa (Python, Node, Go, Rust) dan manajer paket (uv, pnpm, go mod, cargo). Hasilkan Dockerfile dengan cepat jika tidak ada.

5. **Agent loop.** mini-swe-agent atau SWE-agent v2 dengan Claude Opus 4.7. Alat: ripgrep, peta repo pengasuh pohon, read_file, edit_file, run_tests, git. Batasan keras: biaya $20, jam dinding 30 menit, 30 pergantian agen.

6. **Verifikasi.** Setelah loop selesai, jalankan rangkaian pengujian lengkap di sandbox. Hitung delta cakupan melalui jacoco / coverage.py. Jika CI merah: berhenti, jangan buka PR. Jika cakupan turun lebih dari 2%: buka PR dengan label `needs-review`.

7. **Posting PR.** Dorong cabang agen. Buka PR melalui GitHub API dengan: judul, alasan, ringkasan perbedaan, URL jejak, biaya, putaran.

8. **Kebersihan kredensial.** Pekerja berjalan dengan token instalasi Aplikasi GitHub yang berumur pendek. Log dibersihkan untuk mencari rahasia sebelum diarsipkan.

9. **Eval.** 30 isu internal unggulan dengan tingkat kesulitan yang berbeda-beda. Ukur tingkat kelulusan, kualitas PR (ukuran berbeda, gaya, cakupan), biaya, latensi. Bandingkan dengan Agen Latar Belakang Kursor dan Agen SWE Distance Jauh AWS pada masalah yang sama.

## Pakai

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## Kirim

`outputs/skill-issue-to-pr.md` adalah hasil yang dapat dicapai. Aplikasi GitHub + pekerja cloud async yang mengubah isu berlabel menjadi PR yang siap ditinjau dengan biaya terbatas dan kredensial yang tercakup.

| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | Tingkat kelulusan pada 30 masalah | Kesuksesan menyeluruh (CI hijau + cakupan OK) |
| 20 | kualitas PR | Ukuran perbedaan, delta cakupan, kesesuaian gaya |
| 20 | Biaya dan latensi per masalah terselesaikan | $ dan jam dinding per PR |
| 20 | Keamanan | Token tercakup, anggaran per repo, tanpa paksaan, kebersihan kredensial |
| 15 | Operator UX | Komentar alasan, coba lagi keterjangkauan, @-sebutkan tindak lanjut |
| **100** | | |

## Latihan

1. Tambahkan mode "perbaiki pengujian yang tidak stabil": label `@agent stabilize-flake TestX` menjalankan pengujian 50 kali dalam kotak pasir dan mengusulkan perubahan minimal yang menstabilkannya.

2. Bandingkan biaya vs Agen Latar Belakang Kursor pada tiga masalah bersama. Laporkan alat mana yang menang di mana.

3. Menerapkan dasbor anggaran: biaya per repo per hari, biaya per pengguna. Waspada terhadap anomali.4. Build mode "dry-run" yang membuka draf PR tanpa menjalankan CI, sehingga pengulas dapat memeriksa rencana tersebut dengan murah.

5. Tambahkan kebijakan penyimpanan: Cabang PR yang lebih lama dari 7 hari tanpa penggabungan akan dihapus secara otomatis.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Aplikasi GitHub | "Identitas bot tercakup" | Aplikasi dengan izin terperinci + token instalasi berumur pendek |
| Agen cloud asinkron | "Agen latar belakang" | Pekerja non-interaktif yang berjalan di cloud sandbox, bukan terminal |
| Inference lingkungan | "Sintesis Dockerfile" | Deteksi bahasa + manajer paket, buat Dockerfile jika tidak ada |
| Verifikasi | "CI-dalam-kotak pasir" | Jalankan rangkaian pengujian lengkap di dalam pekerja sebelum membuka PR |
| Delta cakupan | "Pelestarian Cakupan" | Perubahan cakupan tes % dari basis ke cabang agen |
| Anggaran per-repo | "Plafon harian" | Batas penghitungan dolar dan PR diberlakukan di petugas operator |
| Dasar Pemikiran | "Penjelasan badan PR" | Ringkasan agen tentang apa yang berubah dan alasannya; dibutuhkan di badan PR |

## Bacaan Lanjutan

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — referensi agen cloud async kanonik
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — referensi CLI
- [Agen Latar Belakang Kursor](https://docs.cursor.com/background-agent) — alternatif komersial
- [OpenAI Codex (cloud)](https://openai.com/codex) — yang dihosting pesaing
- [Google Jules](https://jules.google) — versi yang dihosting Google
- [Factory Droids](https://www.factory.ai) — referensi komersial alternatif
- [Dokumentasi Aplikasi GitHub](https://docs.github.com/en/apps) — identitas bot yang tercakup
- [Kotak pasir awan Daytona](https://daytona.io) — kotak pasir referensi
