# Capstone 09 — Agen Migrasi Code (Bahasa Tingkat Repo/Peningkatan Waktu Proses)

> MigrationBench Amazon (Java 8 hingga 17) dan migrator Py2-to-Py3 App Engine Google menetapkan standar tahun 2026. OpenRewrite Moderne melakukan penulisan ulang AST deterministik dalam skala besar. Grit menargetkan masalah yang sama dengan DSL bergaya codemod. Pola produksi menggabungkan keduanya: substrat deterministik untuk penulisan ulang yang aman ditambah layer agen untuk kasus ambigu, kotak pasir untuk pembuatan per cabang, dan rangkaian pengujian yang berubah warna menjadi hijau sebelum PR dibuka. Batu penjurunya adalah memigrasikan 50 repo nyata dan mempublikasikan tingkat kelulusan dengan taksonomi kegagalan.

**Type:** Batu penjuru
**Language:** Python (agen), Java / Python (target), TypeScript (dasbor)
**Prerequisites:** Fase 5 (NLP), Fase 7 (Transformer), Fase 11 (rekayasa LLM), Fase 13 (peralatan), Fase 14 (agen), Fase 15 (otonom), Fase 17 (infrastruktur)
**Fase yang dilakukan:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**Waktu:** 30 jam

## Masalah

Migrasi code skala besar adalah salah satu aplikasi produksi paling bersih dari agen pengkodean tahun 2026. Kebenaran dasarnya sudah jelas (apakah rangkaian pengujian lolos setelah migrasi?), imbalannya nyata (migrasi armada Java-8 adalah proyek berskala jumlah karyawan), dan tolok ukurnya bersifat publik (subset MigrationBench 50-repo). OpenRewrite Moderne menangani sisi deterministik. Layer agen menangani segala hal yang tidak dapat dilakukan oleh resep OpenRewrite: penulisan ulang yang ambigu, penyimpangan sistem pembangunan, sintaksis ekor panjang, kerusakan ketergantungan transitif.

kamu akan membangun agen yang menggunakan repo Java 8 (atau repo Python 2) dan menghasilkan cabang migrasi CI hijau. kamu akan mengukur tingkat kelulusan, pelestarian cakupan pengujian, biaya per repo, dan membuat taksonomi kegagalan. Perbandingan dengan garis dasar deterministik saja memberi tahu kamu di mana nilai agen sebenarnya berada.

## Konsep

Pipa tersebut memiliki dua layer. **Substrat deterministik** (OpenRewrite untuk Java, libcst untuk Python) menjalankan sebagian besar penulisan ulang mekanis dengan aman: impor, tanda tangan metode, pengeditan keamanan null, sumber daya coba-coba, penggantian API yang tidak digunakan lagi. Ini cepat dan menghasilkan perbedaan yang dapat diaudit. **Layer agen** (OpenAI Agents SDK atau LangGraph pada Claude Opus 4.7 dan GPT-5.4-Codex) menangani kasus-kasus yang tidak dapat ditangani oleh resep: peningkatan versi file build (Maven/Gradle/pyproject), konflik ketergantungan transitif, pengujian serpih, anotasi khusus.

Setiap repo mendapat sandbox Daytona dengan target runtime yang sudah diinstal sebelumnya. Agen mengulangi: menjalankan build, mengklasifikasikan kegagalan, menerapkan perbaikan, menjalankan kembali. Batas keras: 30 menit per repo, $8 per repo, 20 pergantian agen. Jika semua tes lulus dan delta cakupan tidak negatif, cabang membuka PR. Jika tidak, repo akan diajukan di bawah kelas kegagalan dengan bukti.

Taksonomi kegagalan adalah hasil yang dapat dicapai. Dari 50 repo, apa yang rusak? Departemen transitif? Anotasi khusus? Buat versi alat? Serpihan pengujian tidak terkait dengan migrasi? Setiap kelas mendapat hitungan dan perbedaan contoh. Penulis resep di masa depan dapat menargetkan tiga teratas.

## Arsitektur

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## Tumpukan- Substrat deterministik: OpenRewrite (Java) atau libcst (Python)
- Agen: OpenAI Agents SDK atau LangGraph melalui Claude Opus 4.7 + GPT-5.4-Codex
- Sandbox: Daytona devcontainer per cabang, target runtime yang sudah diinstal sebelumnya (Java 17 / Python 3.12)
- Membangun sistem: Maven, Gradle, uv (Python)
- Tolok ukur: subset 50 repo Amazon MigrationBench (Java 8 hingga 17), repo Google App Engine Py2-to-Py3
- Test harness: pelari paralel, cakupan melalui Jacoco (Java) atau coverage.py (Python)
- Observabilitas: Langfuse + trace bundle per repo dengan setiap potongan yang berbeda
- Dasbor: dasbor taksonomi kegagalan dengan jumlah per kelas dan perbedaan contoh

## Build

1. **Recipe pass.** Jalankan resep OpenRewrite (Java) atau libcst (Python) terlebih dahulu. Tangkap 70-80% migrasi yang bersifat mekanis. Berkomitmen sebagai komit "resep".

2. **Build trial.** Daytona sandbox: instal runtime target, jalankan build. Jika hijau, lewati ke pengujian. Jika merah, serahkan ke agen.

3. **Lingkaran agen.** LangGraph dengan alat: `run_build`, `read_file`, `edit_file`, `run_test`, `git_diff`. Agen mengklasifikasikan kegagalan (dep, sintaksis, pengujian, alat build) dan menerapkan perbaikan yang ditargetkan. Memutarkan lagi.

4. **Batas anggaran.** Jam dinding 30 menit per repo, biaya $8, 20 pergantian agen. Pelanggaran apa pun terhenti dan file di bawah "budget_exhausted" dengan perbedaan saat ini.

5. **Pengujian + gerbang cakupan.** Setelah build menjadi ramah lingkungan, jalankan rangkaian pengujian. Bandingkan cakupan dengan repo dasar. Jika cakupan turun lebih dari 2%, simpan di bawah "coverage_regression".

6. **PR terbuka.** Jika berhasil, dorong cabang, buka PR dengan perbedaan dan ringkasan resep mana yang diterapkan dan komitmen mana yang dibuat oleh agen.

7. **Taksonomi kegagalan.** Untuk setiap repo yang gagal, beri tag dengan kelas: `dep_upgrade_required`, `build_tool_drift`, `custom_annotation`, `test_flake`, `syntax_edge_case`, `budget_exhausted`. Build dasbor.

8. **50-repo dijalankan.** Jalankan di seluruh subset MigrationBench. Laporkan tingkat kelulusan per kelas, biaya per repo, pelestarian cakupan, dan garis dasar perbandingan vs deterministik saja.

## Pakai

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## Kirim

`outputs/skill-migration-agent.md` adalah hasil yang dapat dicapai. Jika ada repo, ia mengeksekusi resep deterministik kemudian loop agen untuk menghasilkan cabang migrasi hijau, atau mengarsipkan repo di bawah kelas taksonomi.

| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | Tingkat kelulusan MigrationBench | Subset pass 50-repo@1 |
| 20 | Pelestarian cakupan tes | Rata-rata cakupan delta vs basis |
| 20 | Biaya per repo yang dimigrasi | $/repo saat melewati proses |
| 20 | Integrasi agen/alat deterministik | Sebagian perbaikan yang ditangani OpenRewrite vs yang dibuat oleh agen |
| 15 | Tulisan analisis kegagalan | Kelengkapan taksonomi dengan contoh |
| **100** | | |

## Latihan

1. Jalankan pipeline migrasi hanya dengan OpenRewrite (tanpa agen). Bandingkan tingkat kelulusan dengan alur penuh. Identifikasi kasus-kasus di mana hanya agen yang menjadi pembeda.

2. Terapkan pemeriksaan "lint-clean": setelah migrasi, jalankan gaya linter (tidak bernoda untuk Java, ruff untuk Python). Gagalkan PR jika muncul error lint baru. Ukur tingkat cakupan yang dipertahankan tetapi gayanya mengalami kemunduran.

3. Tambahkan optimizer "minimal-diff": setelah cabang agen lulus pengujian, pangkas perubahan yang tidak diperlukan dengan proses kedua. Laporkan pengurangan ukuran perbedaan.

4. Perluas ke migrasi ketiga: Node 18 ke Node 22. Gunakan kembali pembungkus sandbox; tukar layer resep dengan codemod khusus.

5. Ukur time-to-first-green-build (TTFGB) sebagai metrik UX. Target: p50 di bawah 10 menit.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Substrat deterministik | "Mesin resep" | OpenRewrite / libcst: penulisan ulang AST deklaratif dengan jaminan keamanan |
| Kodemod | "Program pengubah code" | Aturan penulisan ulang yang mengubah code sumber secara mekanis |
| Build penyimpangan | "Versi alat miring" | Perubahan perilaku Maven/Gradle/uv yang halus di antara versi utama |
| Kelas kegagalan | "Ember Taksonomi" | Alasan berlabel mengapa repo tidak dimigrasi: dep, sintaksis, pengujian, alat pembangunan, anggaran |
| Delta cakupan | "Pelestarian Cakupan" | Perubahan % cakupan pengujian dari cabang dasar ke cabang yang dimigrasi |
| Giliran agen | "Putaran panggilan alat" | Satu rencana -> tindakan -> siklus pengamatan di loop agen |
| Kehabisan anggaran | "Pukul langit-langit" | Repo menggunakan batas putaran 30 menit / $8 / 20 tanpa melewati |

## Bacaan Lanjutan

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — tolok ukur kanonik tahun 2026
- [Platform OpenRewrite Moderne.io](https://www.moderne.io) — referensi substrat deterministik
- [Dokumentasi OpenRewrite](https://docs.openrewrite.org) — pembuatan resep
- [Grit.io](https://www.grit.io) — kodemod DSL alternatif
- [Buku masak migrasi sandbox OpenAI](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — referensi Agents SDK
- [Migrator Google App Engine Py2 ke Py3](https://cloud.google.com/appengine) — tolok ukur migrasi alternatif
- [libcst](https://github.com/Instagram/LibCST) — Substrat deterministik Python
- [Kotak pasir Daytona](https://daytona.io) — kotak pasir referensi per cabang
