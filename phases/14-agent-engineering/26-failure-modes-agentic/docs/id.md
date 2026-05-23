# Mode Kegagalan: Mengapa Agen Rusak

> MASFT (Berkeley, 2025) mengkatalogkan 14 mode kegagalan multi-agen dalam 3 kategori. Taksonomi Microsoft mendokumentasikan bagaimana kegagalan AI yang ada semakin besar dalam pengaturan agen. Data lapangan industri menyatu dalam lima mode yang berulang: tindakan halusinasi, perluasan cakupan, kesalahan berjenjang, kehilangan konteks, penyalahgunaan alat.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 05 (Self-Refine dan CRITIC), Fase 14 · 24 (Observability)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Sebutkan tiga kategori kegagalan MASFT dan setidaknya empat mode spesifik di masing-masing kategori.
- Jelaskan mengapa kegagalan agen memperkuat mode kegagalan AI yang ada (bias, halusinasi).
- Jelaskan lima mode yang sering terjadi di industri dan mitigasinya.
- Menerapkan detektor stdlib yang menandai jejak agen dengan label mode kegagalan.

## Masalah

Tim mengirimkan agen yang mengerjakan 90% pelacakan. Kegagalan 10% bukanlah gangguan acak — kegagalan tersebut termasuk dalam sejumlah kecil kategori yang berulang. Setelah kamu dapat memberi nama, kamu dapat memantau dan memperbaikinya.

## Konsep

### MASFT (Berkeley, arXiv:2503.13657)

Taksonomi Kegagalan Sistem Multi-Agen. 14 mode kegagalan dikelompokkan menjadi 3 kategori. Kappa 0.88 antar-annotator Cohen - kategori-kategori tersebut dapat dibedakan dengan andal.

Klaim utama: kegagalan adalah kelemahan desain mendasar dalam sistem multi-agen, bukan batasan LLM yang harus diperbaiki dengan model dasar yang lebih baik.

### Taksonomi Microsoft tentang Mode Kegagalan dalam Sistem AI Agentik

- Kegagalan AI yang ada (bias, halusinasi, kebocoran data) diperkuat dalam pengaturan agenik.
- Kegagalan baru muncul dari otonomi: tindakan yang tidak disengaja dalam skala besar, penyalahgunaan alat, penyimpangan misi.
- Whitepaper adalah daftar risiko untuk produk agen.

### Mengkarakterisasi Kesalahan pada AI Agentik (arXiv:2603.06847)

- Kegagalan muncul dari orkestrasi, evolusi keadaan internal, dan interaksi lingkungan.
- Bukan hanya "code buruk" atau "output model buruk".

### Survei Halusinasi Agen LLM (arXiv:2509.18970)

Dua manifestasi utama:

1. **Penyimpangan mengikuti instruksi** — agen tidak mengikuti system prompt.
2. **Penyalahgunaan Kontekstual Jangka Panjang** — agen lupa atau salah menerapkan konteks dari putaran sebelumnya.

Kesalahan sub-intensi: Omission (langkah terlewat), Redundancy (langkah berulang), Disorder (langkah tidak berurutan).

### Lima mode yang berulang di industri

Analisis lapangan Arize, Galileo, NimbleBrain 2024-2026 menyatu pada:

1. **Tindakan halusinasi.** Agen memanggil alat yang tidak ada atau mengarang argumen.
2. **Scope creep.** Agen memperluas tugas melebihi permintaan pengguna (membuat PR tambahan, mengirim email tambahan).
3. **Kesalahan berjenjang.** Satu panggilan yang salah akan memicu efek hilir. Halusinasi SKU hantu memicu empat panggilan API — sebuah insiden multi-sistem.
4. **Kehilangan konteks.** Tugas jangka panjang melupakan batasan awal.
5. **Penyalahgunaan alat.** Memanggil alat yang benar dengan argumen yang salah, atau alat yang salah seluruhnya.

Bertingkat adalah pembunuhnya. Agen tidak dapat membedakan "Saya gagal" dari "tugas tidak mungkin" dan sering berhalusinasi pesan sukses pada 400 kesalahan untuk menutup loop.

### Mitigasi: gerbang di setiap langkah

Gerbang verifikasi otomatis di setiap langkah rantai penalaran, memeriksa landasan faktual terhadap keadaan lingkungan. Secara konkret:

- Pengklasifikasi keamanan per langkah (Lesson 21).
- Validasi argumen pemanggilan alat (Lesson 06).
- Periksa ulang konten yang diambil berdasarkan fakta yang diketahui (Lesson 05, CRITIC).
- Deteksi halusinasi keberhasilan dengan memeriksa ulang keadaan (apakah file benar-benar dibuat?).### Ketika pemantauan kegagalan tidak berjalan dengan baik

- **Hanya pemberian tag yang mogok.** Sebagian besar kegagalan agen menghasilkan output yang tampak valid. Perlu pemeriksaan tingkat konten.
- **Tidak ada garis dasar.** Deteksi penyimpangan memerlukan barang yang terakhir diketahui; tanpanya kamu tidak bisa mengatakan "ini semakin buruk".
- **Peringatan berlebih.** Setiap kegagalan menghasilkan satu halaman. Klaster dan batas kecepatan.

## Build

`code/main.py` mengimplementasikan penanda mode kegagalan stdlib:

- Dataset jejak sintetis yang mencakup lima mode.
- Fungsi detektor per mode (pola tanda tangan pada panggilan alat, output, tindakan berulang).
- Pemberi tag yang memberi label pada setiap jejak dan distribusi mode laporan.

Jalankan:

```
python3 code/main.py
```

Output: label per jejak + distribusi agregat, reproduksi murah dari permukaan pengelompokan jejak Phoenix.

## Pakai

- **Phoenix** untuk pengelompokan penyimpangan produksi (Lesson 24).
- **Langfuse** untuk pemutaran ulang sesi + anotasi.
- **Khusus** untuk tanda tangan khusus domain yang tidak dapat dideteksi oleh platform observasi kamu.

## Kirim

`outputs/skill-failure-detector.md` menghasilkan detektor mode kegagalan yang disesuaikan dengan domain kamu, dihubungkan ke penyimpanan jejak.

## Latihan

1. Tambahkan detektor untuk "halusinasi sukses": agen mengembalikan kesuksesan tetapi status target tidak berubah.
2. Tandai 100 jejak nyata dari produk yang kamu buat. Modus mana yang mendominasi? Berapa biaya perbaikannya?
3. Menerapkan metrik "radius kaskade": jika terjadi kegagalan pada langkah N, berapa banyak langkah hilir yang terpengaruh?
4. Baca 14 mode kegagalan MASFT. Pilih tiga yang sesuai dengan produk kamu. Tulis detektor.
5. Hubungkan satu detektor ke tugas CI: gagalkan build jika >=5% jejak menandai mode.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| MASFT | "Taksonomi kegagalan multi-agen" | Kategorisasi 14 mode Berkeley |
| Kesalahan bertingkat | "Kegagalan riak" | Satu kesalahan awal menyebar melalui N langkah |
| Hilangnya konteks | "Lupa batasannya" | Pergantian cakrawala panjang menjatuhkan fakta putaran awal |
| Penyalahgunaan alat | "Alat salah / argumen salah" | Panggilan valid, pemanggilan salah |
| Halusinasi sukses | "Penyelesaian palsu" | Agen mengklaim sukses pada 400; keadaan tidak berubah |
| Ruang lingkup merayap | "Melampaui jangkauan" | Agen melakukan lebih dari yang diminta |
| Penyimpangan mengikuti instruksi | "Ketidaktaatan" | Mengabaikan system prompt atau batasan pengguna |
| Kesalahan sub-niat | "Rencana bug" | Kelalaian, redundansi, gangguan dalam pelaksanaan rencana |

## Bacaan Lanjutan

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 mode kegagalan, 3 kategori
- [Microsoft, Taksonomi Mode Kegagalan dalam Sistem AI Agentik](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — daftar risiko
- [Arize Phoenix](https://docs.arize.com/phoenix) — praktik pengelompokan drift
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building- Effective-agents) — ketika pola yang lebih sederhana menghindari mode sepenuhnya
