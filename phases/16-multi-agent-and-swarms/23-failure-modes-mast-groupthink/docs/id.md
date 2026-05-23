# Mode Kegagalan - MAST, Groupthink, Monoculture, Cascading Error

> Taksonomi referensi untuk tahun 2026 adalah **MAST** (Cemri et al., NeurIPS 2025, arXiv:2503.13657), yang berasal dari 1.642 jejak eksekusi di 7 MAS sumber terbuka canggih yang menunjukkan **tingkat kegagalan 41–86,7%**. Tiga kategori utama: **Masalah Spesifikasi** (41,77%) — ambiguitas peran, definisi tugas yang tidak jelas; **Kegagalan Koordinasi** (36,94%) — gangguan komunikasi, ketidaksinkronan status; **Kesenjangan Verifikasi** (21,30%) — validasi tidak ada, pemeriksaan kualitas tidak ada. Kelompok **Groupthink** (arXiv:2508.05687) menambahkan: keruntuhan monokultur (model dasar yang sama → kegagalan yang berkorelasi), bias konformitas (agen saling memperkuat kesalahan satu sama lain), teori pikiran yang kurang, dinamika motif campuran, kegagalan keandalan yang berjenjang. Contoh bertingkat: badai percobaan ulang di mana kegagalan pembayaran memicu percobaan ulang pesanan, yang memicu percobaan ulang inventaris, sehingga membebani layanan inventaris (pemuatan 10x dalam hitungan detik — memerlukan pemutus sirkuit). Keracunan memori: halusinasi salah satu agen memasuki memori bersama, agen hilir memperlakukannya sebagai fakta; akurasi menurun secara bertahap, membuat diagnosis akar penyebab menjadi sulit. **STRATUS** (NeurIPS 2025) melaporkan peningkatan keberhasilan mitigasi 1,5x melalui agen deteksi/diagnosis/validasi khusus. Lesson ini memperlakukan mode kegagalan sebagai target teknik kelas satu.

**Type:** Learn
**Language:** Python (stdlib)
**Prerequisites:** Fase 16 · 13 (Memori Bersama), Fase 16 · 14 (Konsensus dan BFT), Fase 16 · 15 (Topologi Voting dan Debat)
**Waktu:** ~75 menit

## Masalah

Sistem multi-agen mengalami kegagalan 41-86,7% pada tugas nyata (Cemri dkk. 2025 mengukur hal ini di 7 MAS sumber terbuka). Itu tidak dapat di-debug dengan "tambahkan saja lebih banyak agen". Kegagalan ini mempunyai penyebab struktural. Taksonomi MAST memberi kamu kategori. Lesson ini memetakan setiap kategori ke pola deteksi, diagnosis, dan mitigasi yang konkrit sehingga jumlahnya tidak lagi terlihat sembarangan.

Praktik produksi tahun 2026 adalah memperlakukan mode kegagalan sebagai input desain. Arsitektur kamu belum "cukup baik" sampai kamu dapat menunjuk ke setiap kategori MAST dan memberi nama mitigasi yang kamu terapkan.

## Konsep

### kategori TIANG

**Masalah Spesifikasi (41,77% kegagalan).** Tugas agen tidak didefinisikan dengan cukup ketat. Contoh:

- Ketidakjelasan peran: dua agen mengira mereka adalah peninjau.
- Tugas tidak ditentukan: "meringkas ini" ketika pengguna menginginkan sudut tertentu.
- Kriteria keberhasilan tersirat: agen tidak dapat mengetahui apakah berhasil.

Mitigasi:
- Tulis kontrak peran yang eksplisit. Prompt setiap agen menyatakan apa yang dilakukannya *dan apa yang tidak dilakukannya*.
- Tes penerimaan per tugas. Sebelum agen memulai, tentukan "selesai terlihat seperti X."
- Pemeriksaan spesifikasi sebelum penerbangan: agen terpisah meninjau ketentuan tugas sebelum pengiriman.

**Kegagalan Koordinasi (36,94%).** Gangguan komunikasi atau keadaan.

Contoh:
- Dua agen memperbarui status bersama tanpa sinkronisasi.
- Pesan hilang antar agen (kegagalan antrian, batas waktu).
- Penyimpangan keadaan: agen A menganggap tugasnya telah selesai; agen B masih mengeksekusi.

Mitigasi:
- Status bersama berversi dengan konkurensi optimis.
- Pengakuan eksplisit untuk pesan penting (coba lagi sampai diterima).
- Pos pemeriksaan sinkronisasi negara secara berkala; mendeteksi penyimpangan sejak dini.

**Kesenjangan Verifikasi (21,30%).** Tidak ada pemeriksaan independen terhadap output.

Contoh:
- Salah satu agen mengklaim sukses; tidak ada yang memverifikasi.
- Rantai agen masing-masing mempercayai output prior.
- Cakupan tes hilang pada perilaku tenang yang muncul.Mitigasi:
- Agen verifikator independen (Lesson 13). Akses sumber independen dan hanya baca.
- Kontrak serah terima eksplisit: "Output A harus melewati pemeriksa C sebelum B dimulai."
- Pencatatan hasil untuk analisis post-hoc.

### Keluarga Groupthink (arXiv:2508.05687)

Lima kegagalan terkait ketika agen melakukan homogenisasi atau meniru satu sama lain:

**Keruntuhan monokultur.** Model dasar atau training data yang sama → kesalahan yang berkorelasi. Ketika tiga agen berbagi LLM, mereka berbagi halusinasinya.

**Bias konformitas.** Agen menyesuaikan diri dengan rekan yang paling keras atau paling percaya diri, bahkan ketika mereka salah.

**Kekurangan ToM.** Agen gagal mencontohkan keyakinan masing-masing; koordinasi menjadi berantakan (Lesson 18).

**Dinamika motif campuran.** Agen dengan insentif yang sebagian selaras beralih ke titik tengah kompromi, dan hal ini tidak memuaskan siapa pun.

**Kegagalan keandalan bertingkat.** Pola kesalahan satu komponen memicu pola kesalahan pada komponen dependen.

### Contoh bertingkat — coba lagi badai

Pola insiden klasik tahun 2026:

```
payment service fails 10% of requests
   ↓
order agent retries payment (exponential backoff but naive)
   ↓
each retry is a new order-inventory check
   ↓
inventory service sees 2x normal load
   ↓
inventory service starts timing out
   ↓
every order retries inventory check
   ↓
inventory service sees 10x normal load
   ↓
cluster goes down
```

Cara mengatasinya klasik: **pemutus sirkuit**. Ketika tingkat kesalahan hilir melebihi ambang batas, terjadi hubungan pendek dengan hasil cache atau default. Ditambah anggaran percobaan ulang yang dibatasi per permintaan.

Pemutus sirkuit adalah salah satu dari sedikit mitigasi kegagalan multi-agen yang kamu pinjam langsung dari sistem terdistribusi tanpa modifikasi.

### Keracunan memori (ditinjau kembali)

Dari Lesson 13: halusinasi seorang agen menjadi fakta ingatan bersama; alasan agen hilir pada fakta keracunan. Dalam istilah MAST, ini adalah celah verifikasi pada layer memori bersama.

Gejalanya adalah penurunan akurasi secara bertahap. kamu tidak mengalami kecelakaan; kamu mendapatkan penyimpangan lambat yang sulit untuk di-root-cause.

Mitigasi: log tambahan saja, asal, pemverifikasi yang tidak dapat ditulis. Sudah dibahas di Lesson 13.

### STRATUS — agen khusus untuk deteksi kegagalan

STRATUS (NeurIPS 2025) melaporkan peningkatan keberhasilan mitigasi 1,5x saat kamu menerapkan:

- **Agen pendeteksi.** Mengawasi pola gejala (ketidaksepakatan tinggi, lonjakan percobaan ulang, penyimpangan akurasi).
- **Agen diagnosis.** Dengan adanya gejala, simpulkan kemungkinan penyebab utama dari taksonomi MAST.
- **Agen validasi.** Setelah mitigasi diterapkan, periksa apakah gejala sudah jelas.

Ini adalah respons insiden bergaya SRE, yang diterapkan pada sistem agen. Ketiga peran tersebut semuanya dapat menjadi agen LLM dengan petunjuk khusus.

### Audit mode kegagalan

Praktik terbaik tahun 2026 adalah audit mode kegagalan tahunan (atau per rilis besar):

1. **Contoh jejak.** Kumpulkan ~1000 jejak eksekusi nyata.
2. **Kategorikan.** Untuk setiap kegagalan pelacakan, petakan ke kategori MAST + Groupthink.
3. **Hitung tingkat kegagalan berdasarkan kategori.** Kategori manakah yang mendominasi sistem kamu?
4. **Mitigasi peringkat.** Perbaikan mana yang paling banyak menghilangkan kegagalan?
5. **Pilih 2-3 mitigasi.** Terapkan; mengaudit ulang kuartal berikutnya.

Disiplin lebih penting daripada pilihan spesifik. Tanpa audit, kegagalan akan bercampur menjadi kebisingan dan tidak akan pernah ditangani secara sistematis.

### Ketika sistem gagal secara diam-diam

Kategori kegagalan yang paling berbahaya adalah kegagalan koreksi diam-diam. Suatu sistem yang mengalami kegagalan keras (crash, Exception, alert) dapat dimonitor. Sistem yang menghasilkan output yang masuk akal namun salah tidak dapat dideteksi oleh log pengecualian. Inilah sebabnya mengapa kesenjangan verifikasi adalah kategori yang paling mahal per kegagalan meskipun hanya 21,30% jika dihitung.

Investasikan pada:
- Tinjauan manusia berdasarkan sample.
- Uji regresi dataset emas.
- Pemeriksaan silang lintas agen pada output penting.

### Kegagalan vs kegagalan lambatBeberapa kegagalan terjadi secara langsung; ada pula yang lambat. Kegagalan langsung (batas waktu, ketidakcocokan skema, kesalahan autentikasi) mudah dideteksi. Kegagalan yang lambat (keracunan memori, penyimpangan monokultur, ambiguitas peran) membutuhkan biaya yang mahal untuk dideteksi dan dicegah.

Langkah rekayasa tahun 2026: instrumen proxy kegagalan lambat sehingga kamu dapat menangkap penyimpangan sebelum menjadi kesalahan yang terlihat. Tingkat persetujuan, tingkat percobaan ulang, distribusi panjang output, dan distance edit antara versi agen yang berurutan semuanya merupakan proksi yang berguna.

## Build

`code/main.py` mengimplementasikan:

- `FailureTaxonomy` — mengkategorikan insiden simulasi ke dalam kategori MAST + Groupthink.
- `CircuitBreaker` — pola klasik; terbuka ketika tingkat kesalahan melebihi ambang batas.
- `RetryStormSimulator` — menunjukkan kegagalan berjenjang; mengaktifkan/menonaktifkan pemutus sirkuit.
- `DetectionAgent` — pencocokan gejala bergaya STRATUS dengan skrip.

Jalankan:

```
python3 code/main.py
```

Hasil yang diharapkan:
- coba lagi badai tanpa pemutus arus: kesalahan inventaris meledak (disimulasikan).
- dengan pemutus arus: tutup pada ambang batas; respons mode terdegradasi disajikan.
- agen pendeteksi menandai pola dan memberi nama kategori MAST.

## Pakai

`outputs/skill-mast-auditor.md` menjalankan audit mode kegagalan gaya MAST pada sistem multi-agen. Jejak → kategorisasi → peringkat mitigasi.

## Kirim

Disiplin mode kegagalan dalam produksi:

- **Audit MAST per kuartal.** Bukan tahunan. Kategori bergeser seiring pertumbuhan sistem kamu.
- **Pemutus sirkuit di mana-mana.** Setiap panggilan keluar ke layanan yang bergantung. Ambang batas terbuka default pada tingkat kesalahan 5-10%.
- **Dataset emas.** Kecil, berkualitas tinggi, dan telah diaudit secara manual. Uji regresi terhadap mereka setiap minggu.
- **STRATUS trio.** Deteksi + Diagnosis + Agen validasi yang memantau produksi. Mulailah dengan agen pendeteksi saja; tambahkan diagnosis ketika gejalanya berisik.
- **Anggaran kegagalan.** SLO eksplisit untuk tingkat kegagalan berdasarkan kategori. Melebihi anggaran memicu percakapan penghentian pengiriman.

## Latihan

1. Jalankan `code/main.py`. Pastikan pemutus sirkuit menutup badai percobaan ulang. Variasikan ambang kegagalan dan amati tradeoffnya.
2. Menerapkan **proksi kegagalan lambat**: tingkat kesepakatan di 3 agen paralel. Saat turun tajam, picu peringatan. Simulasikan penyimpangan monokultur dengan mengkorelasikan output agen secara bertahap.
3. Baca Cemri dkk. (arXiv:2503.13657). Pilih salah satu dari 7 sistem MAS mereka dan petakan 3 kategori kegagalan teratasnya. Bagaimana perbandingannya dengan prediksi MAST?
4. Baca makalah Groupthink (arXiv:2508.05687). Identifikasi mana dari lima pola yang paling sulit dideteksi dalam produksi. Usulkan metrik proksi.
5. Rancang trio deteksi-diagnosis-validasi gaya STRATUS untuk sistem multi-agen tertentu yang kamu tahu. Gejala apa saja yang perlu diperhatikan dalam deteksi? Mitigasi apa yang direkomendasikan oleh diagnosis? Bagaimana validasi memastikan cara kerjanya?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| TIANG | "Taksonomi 2026" | Cemri 2025; 3 kategori akar + 14 subtipe kegagalan. |
| Masalah Spesifikasi | "Ambiguitas peran" | Tugas atau peran kurang jelas; agen tidak tahu harus berbuat apa. |
| Kegagalan Koordinasi | "Pergeseran negara bagian" | Gangguan komunikasi atau sinkronisasi antar agen. |
| Kesenjangan Verifikasi | "Tidak ada yang memeriksa" | Output diterima tanpa validasi independen. |
| Keluarga pemikir kelompok | "Kegagalan homogenitas" | Monokultur, konformitas, defisiensi ToM, motif campuran, berjenjang. |
| Runtuhnya monokultur | "Model yang sama, halusinasi yang sama" | Kesalahan yang berkorelasi dari model dasar atau training data bersama. |
| Coba lagi badai | "Amplifikasi kesalahan berjenjang" | Satu kegagalan memicu percobaan ulang yang memperkuat weight di bagian hilir. |
| Pemutus arus | "Gagal dengan cepat pada tingkat kesalahan" | Buka ketika tingkat kesalahan melebihi ambang batas; hubungan pendek dengan default. |
| STRATUS | "Trio respons insiden" | Deteksi + diagnosis + agen validasi. 1,5x keberhasilan mitigasi. |
| Keracunan memori | "Halusinasi menyebar" | Fakta memori bersama ternoda; alasan agen hilir tentang racun. |

## Bacaan Lanjutan

- [Cemri dkk. — Mengapa Sistem LLM Multi-Agen Gagal?](https://arxiv.org/abs/2503.13657) — Taksonomi MAST, NeurIPS 2025
- [Kegagalan pemikiran kelompok dalam LLM multi-agen](https://arxiv.org/abs/2508.05687) — monokultur, konformitas, dan taksonomi lima keluarga
- [STRATUS — agen khusus untuk respons insiden MAS](https://neurips.cc/) — Entri proses NeurIPS 2025 (deteksi + diagnosis + validasi)
- [Lepaskan! — pola stabilitas (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/) — referensi pemutus sirkuit kanonik
- [Antropik — Sistem penelitian multi-agen](https://www.anthropic.com/engineering/multi-agent-research-system) — catatan mode kegagalan produksi
