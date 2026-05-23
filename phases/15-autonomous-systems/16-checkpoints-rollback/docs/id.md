# Pos pemeriksaan dan Kembalikan

> Setiap transisi keadaan grafik tetap ada. Ketika seorang pekerja mengalami kecelakaan, masa sewanya berakhir dan pekerja lain mengambil di pos pemeriksaan terakhir. Objek Tahan Lama Cloudflare mempertahankan status selama berjam-jam atau berminggu-minggu. Propose-then-commit (Lesson 15) mendefinisikan rencana rollback per tindakan. Verifikasi pasca tindakan menutup loop. UU AI Uni Eropa Pasal 14 mewajibkan pengawasan manusia yang efektif pada sistem berisiko tinggi — dalam praktiknya hal ini berarti pos pemeriksaan harus dapat ditanyakan, rollback harus dilakukan berulang kali, dan jejak audit harus tetap bertahan saat diterapkan. Mode kegagalan tajam: tanpa kunci idempotensi dan pemeriksaan prasyarat, percobaan ulang setelah kegagalan sementara dapat mengeksekusi dua kali tindakan yang sudah disetujui. Verifikasi pasca-tindakan adalah yang menariknya.

**Type:** Learn
**Language:** Python (stdlib, pos pemeriksaan, dan mesin status rollback)
**Prerequisites:** Fase 15 · 12 (Eksekusi yang tahan lama), Fase 15 · 15 (Usulkan lalu lakukan)
**Waktu:** ~60 menit

## Masalah

Eksekusi yang tahan lama (Lesson 12) membuat agen yang mogok dapat dilanjutkan. Usulkan-lalu-lakukan (Lesson 15) membuat tindakan yang disetujui dapat diaudit. Lesson ini menyatukan mereka: apa yang terjadi ketika tindakan yang disetujui dijalankan sebagian, terhenti, dan dilanjutkan? Kapan rollback dijalankan, dan pada kondisi apa?

Sistem nyata menghubungkannya secara berbeda:

- **LangGraph** memeriksa setiap transisi status grafik ke PostgreSQL. Jika pekerja mengalami kecelakaan, sewa akan dilepaskan dan pekerja lain akan melanjutkan pekerjaan di pos pemeriksaan terakhir. Alur kerja dijeda pada `interrupt()`, dan hal ini akan terus berlanjut.
- **Objek Tahan Lama Cloudflare** mempertahankan status per kunci selama berjam-jam atau berminggu-minggu. Tempatkan komputasi bersama dengan penyimpanan untuk tindakan yang disetujui.
- **Microsoft Agent Framework** mengekspos `Checkpoint` primitif di API alur kerja; pemutaran ulang plus idempotensi mencakup percobaan ulang.

Dalam setiap kasus, kombinasi yang benar-benar berfungsi adalah: kunci idempotensi (mencegah eksekusi ganda) + pemeriksaan prasyarat (status masih sesuai dengan yang kami setujui) + verifikasi pasca-tindakan (efek samping benar-benar terjadi) + rollback pada verifikasi-gagal.

## Konsep

### Setiap transisi tetap ada

Transisi keadaan grafik adalah setiap langkah yang memindahkan alur kerja dari satu keadaan bernama ke keadaan lainnya. Implementasi yang naif hanya bertahan pada titik penerapan tertentu; implementasi produksi tetap ada di setiap transisi. Biaya (beberapa penulisan tambahan) relatif kecil dibandingkan dengan perolehan keandalan (pemutaran ulang dilakukan di mana saja, pemulihan sewa tepat).

### Pemulihan sewa

Ketika seorang pekerja mengalami crash, alur kerja tidak hilang; sewa (klaim berumur pendek bahwa pekerja ini melaksanakan proses ini) akan habis masa berlakunya. Pekerja lain mengambil pos pemeriksaan terakhir dan melanjutkan. Mekanisme sewa inilah yang memungkinkan sistem produksi bertahan dalam penerapan berkelanjutan tanpa kehilangan pekerjaan selama penerbangan.

### Idempotensi plus prasyarat

Idempotensi saja tidak cukup. Pertimbangkan: alur kerja disetujui untuk "mentransfer $100 dari A ke B ketika saldo > $1000." Alur kerja telah diterapkan, mengalami error di tengah eksekusi, dan dilanjutkan. Jika hanya kunci idempotensi yang dicentang, dan eksekusi dilanjutkan, transfer berjalan satu kali (benar). Namun pertimbangkan bahwa antara crash dan resume, saldo A turun menjadi $500 melalui alur kerja yang berbeda. Pemeriksaan idempotensi masih lolos; prasyaratnya tidak. Tanpa pemeriksaan prasyarat, kami mengirimkan cerukan.

Setiap tindakan konsekuensial memerlukan keduanya:

- **Kunci idempotensi**: mencegah eksekusi ganda.
- **Pemeriksaan prasyarat**: memastikan status masih sesuai dengan yang disetujui.

### Verifikasi pasca tindakan"Alat mengembalikan 200" bukan verifikasi. Verifikasi nyata membaca kembali keadaan target dan memastikan efek samping benar-benar terjadi. Pola:

- Pembaruan basis data: `UPDATE ... RETURNING *` lalu nyatakan baris yang dikembalikan cocok dengan status yang diinginkan.
- Pengiriman email: periksa folder terkirim untuk ID pesan setelah pengiriman.
- Penulisan file: membaca kembali file dan melakukan hash.
- Panggilan API: tindak lanjut `GET` pada sumber daya target.

Jika verifikasi gagal, alur kerja berada dalam kondisi yang diketahui buruk. Rollback diaktifkan.

### Rencana pengembalian

Setiap tindakan konsekuensial dalam mengusulkan-lalu-melakukan (Lesson 15) membawa rencana pengembalian. Jenis:

- **In-band rollback**: membalikkan efek samping secara langsung (`DELETE` setelah `INSERT`, `Send-correction-email` setelah pengiriman).
- **Transaksi kompensasi**: tindakan baru yang menetralkan tindakan asli (pola SAGA standar).
- **Kembalikan di luar jalur**: memperingatkan manusia, menjeda alur kerja, meninggalkan keadaan buruk untuk diselidiki.

Rollback tanpa operasi ("kami tidak dapat membatalkan ini") harus disebutkan dalam proposal. Tindakan tanpa rollback memerlukan HITL yang lebih kuat pada waktu penerapan (Lesson 15 tantangan-dan-respons).

### EU AI Act Pasal 14 pembacaan operasional

Pasal 14 mensyaratkan “pengawasan manusia yang efektif” untuk sistem berisiko tinggi. Secara operasional, pelaksana membacanya sebagai:

- Pos pemeriksaan dapat ditanyakan oleh auditor.
- Rollback dilatih (diuji ujung ke ujung setidaknya sekali).
- Jejak audit bertahan saat penerapan (backend pos pemeriksaan tidak bersifat sementara).
- Verifikasi yang gagal diperingatkan, bukan dicatat secara diam-diam.

Alur kerja yang mengalami error saat melakukan, melanjutkan, dan menyelesaikan efek samping tanpa jalur verifikasi + rollback tidak akan bertahan dalam pengujian Pasal 14.

### Mode kegagalan tajam: eksekusi ganda

Insiden produksi paling umum di bidang ini:

1. Tindakan disetujui, kunci idempotensi k.
2. Komit memulai, mengeksekusi, mengembalikan 200.
3. Alur kerja terhenti sebelum mempertahankan status "berkomitmen".
4. Alur kerja dilanjutkan; melihat "disetujui tetapi tidak dilakukan"; dijalankan kembali.
5. Efek samping muncul dua kali.

Mitigasi: pertahankan niat "dalam penerbangan" sebelum eksekusi, jalankan dengan kunci idempotensi, lalu tandai "berkomitmen" hanya setelah verifikasi pasca-tindakan berhasil. Jika tindakan diaktifkan dan penulisan status gagal, kamu tahu untuk memverifikasi dan (jika perlu) mengaktifkan kembali. Jika penulisan status berhasil dan tindakan gagal, kamu memverifikasi dan mengaktifkan tepat satu kali melalui jalur pemulihan.

## Pakai

`code/main.py` mengimplementasikan alur kerja yang diperiksa dengan idempotensi, prasyarat, verifikasi, dan rollback. Driver menyimulasikan empat skenario: proses bersih, coba lagi setelah crash (idempotensi tertangkap), prasyarat gagal (alur kerja dibatalkan tanpa pengaktifan), verifikasi gagal (pengaktifan rollback).

## Kirim

`outputs/skill-rollback-rehearsal.md` merancang pengujian rollback-rehearsal untuk alur kerja yang diusulkan dan mengaudit backend pos pemeriksaan untuk mengetahui persistensi jejak audit.

## Latihan

1. Jalankan `code/main.py`. Verifikasi empat skenario. Untuk kasus crash-selama-komit, konfirmasikan bahwa tindakan tersebut diaktifkan tepat satu kali pada percobaan ulang.

2. Ubah pola "tandai sebagai selesai terlebih dahulu, lalu lakukan" sehingga status tulis diaktifkan setelah tindakan. Jalankan kembali skenario kerusakan. Ukur berapa banyak tindakan duplikat yang diaktifkan.

3. Rancang rencana pengembalian untuk tindakan produksi tertentu (misalnya, "posting ke pipeline Slack"). Klasifikasikan sebagai in-band, kompensasi, atau out-of-band. Benarkan pilihannya.4. Ambil satu alur kerja lho. Identifikasi setiap transisi negara. Tandai masing-masing dengan syarat ketahanannya (bertahan/tidak bertahan). Hitunglah yang saat ini tidak kamu pertahankan.

5. Uji rollback yang telah dilatih: merancang pengujian end-to-end yang menjalankan alur kerja nyata, menghentikan alur kerja tersebut, dan mengonfirmasi jalur rollback diaktifkan. Apa yang ditegaskan tes ini?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| Pos pemeriksaan | "Simpan poin" | Setiap transisi keadaan grafik tetap ada ke penyimpanan tahan lama |
| Sewa | "Klaim pekerja" | Klaim berumur pendek bahwa seorang pekerja sedang menjalankan lari; kedaluwarsa saat mogok |
| Prasyarat | "Gerbang negara" | Penegasan negara masih konsisten dengan tindakan yang disetujui |
| Verifikasi pasca tindakan | "Baca ulang cek" | Konfirmasikan efek samping yang benar-benar terjadi pada sistem target |
| Kembalikan dalam band | "Langsung membatalkan" | Membalikkan efek samping dengan operasi kebalikan |
| Transaksi kompensasi | "SAGA dibatalkan" | Tindakan baru yang menetralkan |
| Tandai-sebagai-selesai-pertama | "Pesanan tulis status" | Pertahankan status komit sebelum kembali dari komit |
| Pasal 14 | "EU AI Act pengawasan manusia" | Operasional: pos pemeriksaan yang dapat ditanyakan, rollback yang telah dilatih, jejak yang dapat diaudit |

## Bacaan Lanjutan

- [Microsoft Agent Framework — Checkpointing dan HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — primitif checkpoint dan pemulihan sewa.
- [Agen Cloudflare — Manusia dalam lingkaran](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — Objek Tahan Lama sebagai substrat keadaan.
- [EU AI Act — Pasal 14: Pengawasan manusia](https://artificialintelligenceact.eu/article/14/) — garis dasar peraturan.
- [Anthropic — Mengukur otonomi agen dalam praktiknya](https://www.anthropic.com/research/measuring-agent-autonomy) — kerangka keandalan untuk alur kerja jangka panjang.
- [Antropik — SDK Agen Code Claude: loop agen](https://code.claude.com/docs/en/agent-sdk/agent-loop) — bentuk alur kerja untuk Rutinitas Code Claude.
