# Human-in-the-Loop: Usulkan-Lalu-Berkomitmen

> Konsensus tahun 2026 mengenai HITL bersifat spesifik. Ini bukan "agen bertanya, pengguna mengklik Setuju". Ini adalah usulan-lalu-komit: tindakan yang diusulkan disimpan ke penyimpanan tahan lama dengan kunci idempotensi; muncul ke peninjau dengan maksud, silsilah data, izin yang disentuh, radius ledakan, dan rencana pengembalian; dilakukan hanya setelah pengakuan positif; diverifikasi setelah eksekusi untuk memastikan efek samping benar-benar terjadi. `interrupt()` LangGraph ditambah pos pemeriksaan PostgreSQL, `RequestInfoEvent` dari Microsoft Agent Framework, dan `waitForApproval()` dari Cloudflare semuanya mengimplementasikan bentuk yang sama. Modus kegagalan kanonik adalah persetujuan stempel: "Setujui?" diklik tanpa ulasan. Mitigasi yang terdokumentasi adalah tantangan dan respons dengan daftar periksa yang jelas.

**Type:** Learn
**Language:** Python (stdlib, mesin status usulkan lalu komit dengan idempotensi)
**Prerequisites:** Fase 15 · 12 (Eksekusi tahan lama), Fase 15 · 14 (Tripwires)
**Waktu:** ~60 menit

## Masalah

Seorang agen mengambil tindakan. Pengguna harus memutuskan: menyetujui atau tidak. Jika keputusannya instan, mungkin itu bukan peninjauan kembali. Jika keputusannya terstruktur, maka keputusan tersebut lambat namun dapat dipercaya. Pertanyaan teknisnya adalah bagaimana membuat tinjauan terstruktur menjadi jalur yang paling sedikit hambatannya.

Pola HITL era 2023 adalah prompt sinkron: "Agen ingin mengirim email ke X dengan isi Y — setujui?" Pengguna mengklik Setujui. Semua orang merasa sistemnya aman. Dalam praktiknya, permukaan ini sangat kaku: pengguna menyetujui dengan cepat, persetujuan hanya memprediksi sedikit, dan ketika agen melakukan kesalahan, jejak audit menunjukkan sejarah panjang persetujuan yang tidak dapat diingat oleh pengguna.

Pola tahun 2026 — usulkan lalu komit — memindahkan HITL ke substrat yang tahan lama, melampirkan metadata terstruktur, dan memerlukan komitmen positif. Setiap SDK agen terkelola mengirimkan versi: LangGraph `interrupt()`, Microsoft Agent Framework `RequestInfoEvent`, Cloudflare `waitForApproval()`. Nama API berbeda; bentuknya tidak.

## Konsep

### Mesin negara usulkan lalu komit

1. **Mengusulkan.** Agen menghasilkan tindakan yang diusulkan. Bertahan di toko yang tahan lama (PostgreSQL, Redis, Durable Object). Termasuk:
   - niat (mengapa agen melakukan ini)
   - silsilah data (sumber apa yang menyebabkan proposal ini)
   - izin disentuh (cakupan / file / titik akhir mana)
   - radius ledakan (apa kasus terburuknya)
   - rencana rollback (jika dilakukan, bagaimana cara kita membatalkannya)
   - kunci idempotensi (unik per proposal; pengiriman ulang mengembalikan catatan yang sama)
2. **Permukaan.** Peninjau melihat proposal dengan semua metadata. Reviewer adalah orang (bukan agen yang mereview dirinya sendiri).
3. **Berkomitmen.** Pengakuan positif. Tindakan tersebut dijalankan.
4. **Verifikasi.** Setelah eksekusi, efek samping dibaca kembali dan dikonfirmasi. Jika langkah verifikasi gagal, sistem diketahui berada dalam kondisi buruk dan peringatan akan aktif.

### Kunci idempotensi

Tanpa kunci idempotensi, percobaan ulang setelah kegagalan sementara dapat mengeksekusi dua kali tindakan yang disetujui. Contoh nyata: pengguna menyetujui "transfer $100 dari A ke B." Gangguan jaringan. Percobaan ulang alur kerja. Pengguna telah menyetujui satu kali tetapi transfer dilakukan dua kali. Kunci idempotensi mengikat persetujuan pada satu efek samping yang unik; eksekusi kedua tidak boleh dilakukan.

Ini adalah pola idempotensi yang sama yang digunakan Stripe dan API AWS. Menggunakannya kembali untuk persetujuan agen dijelaskan secara eksplisit dalam dokumen Microsoft Agent Framework.

### Daya Tahan: mengapa persetujuan bertahan lebih lama dari prosesRuang tunggu persetujuan adalah bagian negara yang tidak dimiliki agen. Alur kerja dijeda (Lesson 12). Ketika persetujuan tiba, alur kerja dilanjutkan dari titik tersebut. Inilah sebabnya mengapa LangGraph memasangkan `interrupt()` dengan pos pemeriksaan PostgreSQL dan bukan hanya status dalam memori — persetujuan dua hari kemudian masih membuat alur kerja tetap utuh.

### Persetujuan stempel dan mitigasi tantangan dan respons

UI default untuk HITL (tombol "Setujui" / "Tolak") menghasilkan persetujuan cepat tanpa tinjauan asli. Mitigasi terdokumentasi: daftar periksa tantangan dan respons yang memerlukan jawaban positif terhadap pertanyaan spesifik sebelum tombol Setuju diaktifkan. Bentuk beton:

- "Apakah kamu memahami sumber daya apa yang disentuhnya? [ ]"
- "Sudahkah kamu memverifikasi bahwa radius ledakan dapat diterima? [ ]"
- "Apakah kamu memiliki rencana pengembalian jika gagal? [ ]"

Bukan birokrasi untuk kepentingannya sendiri – sebuah fungsi yang memaksa. Peninjau yang tidak dapat mencentang kotak akan meminta klarifikasi (eskalasi) atau menolak (default aman). Penelitian keamanan agen Anthropic secara eksplisit mengutip HITL yang didorong oleh daftar periksa sebagai mitigasi terhadap pola persetujuan stempel.

### Apa yang dianggap sebagai konsekuensial

Tidak setiap tindakan perlu diusulkan lalu dilakukan. Panduan tahun 2026:

- **Tindakan konsekuensial** (selalu HITL): penulisan yang tidak dapat diubah, transaksi keuangan, komunikasi keluar, perubahan database produksi, operasi sistem file yang merusak.
- **Tindakan yang dapat dibalik** (terkadang HITL): pengeditan pada file lokal, perubahan staging-env, penulisan yang dapat dibalik dengan rollback yang jelas.
- **Pembacaan dan inspeksi** (tidak pernah HITL): membaca file, mencantumkan sumber daya, memanggil API hanya-baca.

### Verifikasi pasca tindakan

"Komitmen yang dijalankan" tidak sama dengan "efek samping yang terjadi". Kondisi partisi jaringan dan balapan dapat menghasilkan alur kerja yang dianggap berhasil sementara backend tidak bertahan. Langkah verifikasi membaca ulang sumber daya target setelah berkomitmen untuk mengonfirmasi. Pola ini sama dengan transaksi database dengan klausa `RETURNING` atau AWS `GetObject` setelah `PutObject`.

### UU AI UE Pasal 14

Pasal 14 mengamanatkan pengawasan manusia yang efektif terhadap sistem AI yang berisiko tinggi di UE. "Efektif" bukanlah hiasan. Bahasa peraturan secara khusus mengecualikan pola stempel. Usulan-lalu-komitmen dengan tantangan-dan-tanggapan adalah bentuk yang tetap lolos dari pengawasan Pasal 14 dalam dokumen kepatuhan Perangkat Tata Kelola Agen Microsoft.

## Pakai

`code/main.py` mengimplementasikan mesin status usulkan lalu komit di stdlib Python. Penyimpanan tahan lama adalah file JSON. Kunci idempotensi adalah hash dari (thread_id, action_signature). Penggeraknya menyimulasikan tiga kasus: alur persetujuan yang bersih, percobaan ulang setelah kegagalan sementara (yang tidak boleh dijalankan ganda), dan default stempel karet versus alur tantangan-dan-respons.

## Kirim

`outputs/skill-hitl-design.md` meninjau alur kerja HITL yang diusulkan untuk bentuk usulan-lalu-komit dan menandai layer metadata, idempotensi, verifikasi, atau tantangan-dan-respons yang hilang.

## Latihan

1. Jalankan `code/main.py`. Konfirmasikan bahwa percobaan ulang proposal yang disetujui menggunakan catatan tahan lama dan tidak dijalankan ulang. Sekarang ubah kunci idempotensi untuk menyertakan stempel waktu dan menampilkan percobaan ulang yang dijalankan ganda.

2. Perluas catatan proposal dengan bidang `rollback`. Simulasikan eksekusi yang langkah verifikasinya gagal. Tampilkan rollback yang diaktifkan secara otomatis.3. Baca dokumen `RequestInfoEvent` Microsoft Agent Framework. Identifikasi satu bidang metadata yang disertakan API yang mesin mainannya tidak ada. Tambahkan dan jelaskan apa yang dilindunginya.

4. Rancang daftar periksa tantangan dan respons untuk tindakan tertentu (misalnya, "posting ke akun Twitter publik"). Tiga pertanyaan apa yang harus dijawab oleh pengulas? Mengapa ketiganya?

5. Pilih satu kasus di mana pesan "Setujui?" prompt sudah cukup (tidak perlu toko tahan lama). Jelaskan alasannya, dan sebutkan kelas risiko yang kamu terima.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| Usulkan-lalu-komit | "Persetujuan dua fase" | Proposal yang bertahan + komitmen positif + verifikasi |
| Kunci idempotensi | "Coba lagi token aman" | Unik per proposal; eksekusi kedua tanpa operasi |
| Silsilah data | "Dari mana asalnya" | Konten sumber spesifik yang mengarah pada proposal |
| Radius ledakan | "Kasus terburuk" | Lingkup akibat jika tindakan salah |
| Stempel karet | "Persetujuan cepat" | "Setujui" diklik tanpa ulasan asli |
| Tantangan-dan-respon | "Memaksa daftar periksa" | Reviewer harus menjawab secara positif pertanyaan spesifik |
| PermintaanInfoAcara | "MS Agent Framework primitif" | Permintaan HITL yang tahan lama dengan metadata terstruktur |
| `interrupt()` / `waitForApproval()` | "Kerangka primitif" | LangGraph / Cloudflare setara dengan bentuk yang sama |

## Bacaan Lanjutan

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`, persetujuan yang tahan lama.
- [Agen Cloudflare — Manusia dalam lingkaran](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` dan Objek Tahan Lama.
- [Anthropic — Mengukur otonomi agen dalam praktiknya](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL sebagai mitigasi risiko jangka panjang.
- [EU AI Act — Pasal 14: Pengawasan manusia](https://artificialintelligenceact.eu/article/14/) — dasar peraturan untuk sistem berisiko tinggi.
- [Anthropic — Konstitusi Claude (Januari 2026)](https://www.anthropic.com/news/claudes-constitution) — kerangka konstitusional seputar pengawasan.
