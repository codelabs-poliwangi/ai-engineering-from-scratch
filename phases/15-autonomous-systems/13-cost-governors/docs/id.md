# Anggaran Tindakan, Batasan Iterasi, dan Pengatur Biaya

> Biaya LLM bulanan agen e-niaga skala menengah melonjak dari $1.200 menjadi $4.800 setelah timnya mengaktifkan keterampilan "pelacakan pesanan". Itu bukan kesalahan penetapan harga. Itu adalah agen yang menemukan lingkaran baru dan terus mengeluarkan uang di dalamnya. Toolkit Tata Kelola Agen Microsoft (2 April 2026) mengkodifikasikan pertahanan terhadap kelas ini: per permintaan `max_tokens`, token per tugas dan anggaran dolar, batas per hari/bulan, batas iterasi, perutean model berjenjang, cache cepat, jendela konteks, pos pemeriksaan HITL pada tindakan yang mahal, tombol pemutus pada pelanggaran anggaran. SDK Agen Code Claude Anthropic mengirimkan primitif yang sama dengan nama berbeda. Batasan kecepatan finansial — mis. memotong akses >$50 dalam 10 menit — menangkap loop lebih cepat daripada batas bulanan.

**Type:** Learn
**Language:** Python (stdlib, simulator gubernur biaya berlapis)
**Prerequisites:** Phase 15 · 10 (Mode izin), Phase 15 · 12 (Eksekusi tahan lama)
**Waktu:** ~60 menit

## Masalah

Agen otonom mengeluarkan uang nyata di setiap kesempatan. Output chatbot yang buruk adalah balasan yang buruk; lingkaran buruk seorang agen adalah sebuah tagihan. Istilah yang terdokumentasi dalam industri untuk mode kegagalan adalah "Penolakan Dompet" - agen terus berpikir, terus menggunakan alat, terus melakukan penagihan, dan tidak ada yang menghentikannya karena tidak ada yang dirancang untuk itu.

Perbaikannya bukan pada satu angka. Ini adalah kumpulan batasan pada skala waktu dan rincian yang berbeda: per permintaan, per tugas, per jam, per hari, per bulan. Tumpukan yang dirancang dengan baik menangkap loop yang tidak terkendali dalam hitungan menit, kebocoran lambat dalam hitungan jam, dan rilis buruk dalam sehari. Tumpukan yang sama menyimpan anggaran sama sekali ketika agen bersifat jangka panjang dan otonom.

Ini adalah lesson teknik: matematika itu sepele, disiplin adalah penyebab kegagalan tim. Daftar batasan di bawah semuanya disebutkan dalam Microsoft Agent Governance Toolkit atau dokumen Anthropic Claude Code Agent SDK.

## Konsep

### Tumpukan pengatur biaya

1. **`max_tokens` per permintaan.** Sederhana. Mencegah satu panggilan menghasilkan penyelesaian tanpa batas.
2. **Anggaran token per tugas.** Secara keseluruhan, jangan melebihi N token. Berhenti keras di tutupnya.
3. **Anggaran dolar per tugas.** Sama seperti token tetapi dalam mata uang. `max_budget_usd` dalam Code Claude.
4. **Batas panggilan per alat.** Tidak lebih dari N `WebFetch` panggilan, N `shell_exec` panggilan, dll.
5. **Batas iterasi (`max_turns`).** Total iterasi loop agen; mencegah putaran penalaran yang tak terbatas.
6. **Batas per menit / per jam / per hari / per bulan.** Jendela bergulir. Menangkap kebocoran pada skala waktu yang berbeda.
7. **Batas kecepatan finansial.** Misalnya, "jika pembelanjaan melebihi $50 dalam 10 menit, potong akses". Menangkap luka bakar berbasis loop sebelum batas api bulanan.
8. **Perutean model berjenjang.** Default untuk model yang lebih kecil; meningkat ke tingkat yang lebih besar hanya ketika pengklasifikasi menilai tugas tersebut memerlukannya.
9. **Caching cepat.** Prompt sistem dan konteks stabil disimpan dalam cache penyedia; biaya token untuk mengirim ulang mendekati nol.
10. **Penjendelaan konteks.** Pemadatan/peringkasan untuk menjaga konteks aktif di bawah ambang batas; pengurangan biaya token secara langsung.
11. **Pos pemeriksaan HITL pada tindakan yang mahal.** Sebelum suatu tindakan diketahui mahal (panggilan alat yang lama, pengunduhan yang besar, peningkatan model yang mahal), memerlukan ketukan manusia.
12. **Matikan tombol jika ada pelanggaran anggaran.** Sesi dibatalkan jika ada batasan yang diaktifkan. Batas dicatat; memerlukan jalur pengaktifan kembali yang terpisah.

### Mengapa tumpukan, bukan satu tutupBatas bulanan tunggal menangkap agen yang melarikan diri hanya setelah dompetnya hilang. Batasan per permintaan tunggal tidak menghasilkan apa pun di tingkat sesi. Mode kegagalan yang berbeda memerlukan skala waktu yang berbeda:

- **Runaway loop** (agen terjebak dalam percobaan ulang 5 detik): ditangkap oleh batas kecepatan.
- **Kebocoran lambat** (agen melakukan ~2x pekerjaan yang diharapkan per tugas): ditentukan berdasarkan batas harian.
- **Rilis buruk** (versi baru menggunakan 5x token): dibatasi oleh batasan mingguan/bulanan.
- **Lonjakan yang sah** (permintaan nyata, bukan bug): ditangkap berdasarkan batas jam/hari dengan log yang jelas.

### Permukaan anggaran Claude Code

SDK Agen Code Claude memaparkan (dokumen publik):

- `max_turns` — batas iterasi.
- `max_budget_usd` — batas dolar; sesi dibatalkan karena pelanggaran.
- `allowed_tools` / `disallowed_tools` — daftar alat yang diizinkan dan daftar yang ditolak.
- Poin kaitan sebelum penggunaan alat untuk akuntansi biaya khusus.

Gabungkan dengan tangga mode izin (Lesson 10). Sesi `autoMode` tanpa `max_budget_usd` adalah otonomi yang tidak dapat diatur. Anthropic secara eksplisit membingkai Mode Otomatis sebagai memerlukan kontrol anggaran; pengklasifikasinya ortogonal terhadap biaya.

### UU AI UE, 10 Besar Agen OWASP

Perangkat Tata Kelola Agen Microsoft mencakup persyaratan 10 Besar Agen OWASP dan Pasal 14 UU AI UE (pengawasan manusia). Untuk produksi di UE, penebangan dan penegakan pembatasan bukanlah suatu pilihan.

### Kasus $1.200 yang diamati → $4.800

Kasus sebenarnya dalam dokumen Microsoft: agen e-niaga yang biaya bulanannya meningkat tiga kali lipat setelah alat baru ditambahkan. Alat ini memungkinkan agen untuk melakukan polling status pesanan selama setiap sesi. Tidak ada deteksi loop. Tidak ada tutup per alat. Tidak ada peringatan mengenai pertumbuhan dari minggu ke minggu. Perbaikannya adalah batasan per alat ditambah peringatan pertumbuhan harian. Ini adalah sebuah templat: setiap permukaan pahat baru merupakan lingkaran potensial baru; setiap alat baru memerlukan batas dan peringatannya sendiri.

## Pakai

`code/main.py` menyimulasikan agen yang dijalankan dengan dan tanpa tumpukan cost-governor berlapis. Agen yang disimulasikan masuk ke dalam lingkaran pemungutan suara setelah beberapa putaran; tumpukan berlapis menangkapnya dalam jendela kecepatan sementara batas bulanan tunggal tidak akan diaktifkan hingga beberapa hari kemudian.

## Kirim

`outputs/skill-agent-budget-audit.md` mengaudit tumpukan pengatur biaya penerapan agen yang diusulkan dan menandai layer yang hilang.

## Latihan

1. Jalankan `code/main.py`. Konfirmasikan batas kecepatan yang diaktifkan sebelum batas iterasi pada lintasan polling loop. Sekarang nonaktifkan batas kecepatan dan ukur berapa banyak yang "dibelanjakan" agen sebelum batas iterasi menangkapnya.

2. Rancang set batasan per alat untuk agen browser (Lesson 11). Alat mana yang memerlukan penutup paling rapat? Alat manakah yang dapat berjalan tanpa batas tanpa risiko?

3. Baca dokumen Perangkat Tata Kelola Agen Microsoft. Cantumkan setiap huruf kapital dan nama toolkitnya. Petakan masing-masing ke salah satu mode kegagalan (runaway loop, slow leak, bad release, surge).

4. Hargai pengoperasian semalaman tanpa pengawasan untuk tugas yang realistis (misalnya, "triase 50 masalah dalam repo"). Tetapkan `max_budget_usd` pada 2x perkiraan poin kamu. Benarkan 2x.

5. `max_budget_usd` Claude Code diaktifkan pada biaya agregat sesi. Rancang batas kecepatan pelengkap yang akan kamu terapkan secara eksternal. Apa yang memicu penghentian, dan seperti apa pengaktifan kembali?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| Penolakan Dompet | "Tagihan pelarian" | Pembelanjaan yang menghasilkan loop agen tanpa batas untuk menghentikannya |
| max_tokens | "Batas per permintaan" | Plafon pada ukuran penyelesaian tunggal |
| putaran_maks | "Batas iterasi" | Batas atas iterasi loop agen dalam satu sesi |
| max_budget_usd | "Tombol mematikan dolar" | Batas biaya sesi; dibatalkan karena pelanggaran |
| Batas kecepatan | "Batas tarif" | Batas pembelanjaan per jendela pendek (mis., $50 / 10 mnt) |
| Perutean berjenjang | "Model kecil dulu" | Standar model murah; meningkat hanya jika pengklasifikasi menjamin |
| Caching cepat | "Permintaan sistem dalam cache" | Cache sisi penyedia mengurangi biaya pengiriman ulang token hingga mendekati nol |
| Pos pemeriksaan HITL | "Gerbang persetujuan manusia" | Diperlukan ketukan manusia sebelum tindakan mahal |

## Bacaan Lanjutan

- [SDK Agen Code Claude Antropis — loop agen dan anggaran](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`, `max_budget_usd`, daftar alat yang diizinkan.
- [Microsoft Agent Framework — human-in-the-loop dan tata kelola](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — pos pemeriksaan pengatur biaya.
- [Antropis — Ikhtisar Agen Terkelola Claude](https://platform.claude.com/docs/en/managed-agents/overview) — pengendalian biaya di sisi penyedia.
- [Antropis — Prompt caching (dokumen Claude API)](https://platform.claude.com/docs/en/prompt-caching) — mekanisme caching.
- [Anthropic — Mengukur otonomi agen dalam praktiknya](https://www.anthropic.com/research/measuring-agent-autonomy) — profil biaya untuk agen jangka panjang.
