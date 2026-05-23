# Desain Peningkatan Diri Terikat

> Penelitian telah menyatukan empat cara primitif untuk membatasi lingkaran perbaikan diri. Invarian formal yang harus berlaku di setiap pengeditan. Jangkar penyelarasan yang tidak dapat dimodifikasi. Kendala multi-tujuan dimana setiap dimension (keselamatan, keadilan, ketahanan) harus dipenuhi, bukan hanya kinerja. Deteksi regresi yang menghentikan perulangan ketika metrik historis menunjukkan hilangnya kemampuan. Tidak ada satupun yang merupakan bukti keamanan — hasil teori informasi (kompleksitas Kolmogorov, teorema Lob) mengikat apa yang dapat dibuktikan oleh sistem mana pun tentang penerusnya. Ini adalah mitigasi yang meningkatkan dampak kegagalan diam-diam.

**Type:** Learn
**Language:** Python (stdlib, loop terbatas dengan pemeriksaan invarian)
**Prerequisites:** Fase 15 · 07 (RSI), Fase 15 · 04 (DGM)
**Waktu:** ~60 menit

## Masalah

Simulator balapan pada Lesson 7 menunjukkan bahwa perbedaan kecepatan yang kecil akan menyebabkan kesenjangan yang besar. Studi kasus DGM pada Lesson 4 menunjukkan bahwa loop dapat secara aktif mempermainkan evaluatornya sendiri. Kedua hasil tersebut mengarah pada pertanyaan teknik yang sama: batasan apa yang dapat kamu berikan pada putaran perbaikan diri sehingga batasan tersebut tidak dapat dilemahkan secara diam-diam oleh putaran itu sendiri?

Ringkasan Lokakarya RSI ICLR 2026 (openreview.net/pdf?id=OsPQ6zTQXV) mengidentifikasi empat primitif tersebut. RSP v3.0 Anthropic (Lesson 19) dan FSF v3 DeepMind (Lesson 20) keduanya mereferensikannya dalam ambang batas kemampuan. Kerangka kerja Meta HyperAgents dan komunitas seperti SAHOO (Maret 2026) mengimplementasikan subset dalam produksi.

Pembingkaian yang jujur: ini adalah mitigasi. Hasil teori informasi membatasi apa yang dapat dibuktikan oleh sistem apa pun tentang penggantinya, dan tidak ada rancangan saat ini yang bisa menyelesaikan masalah secara formal. Lingkaran yang berbatas baik lebih aman daripada lingkaran yang tidak terikat, dan tidak aman secara absolut.

## Konsep

### Primitif 1: invarian formal

Invarian adalah properti yang harus dimiliki sebelum dan sesudah setiap modifikasi diri. Contoh:

- Distribusi output dikondisikan pada header konstitusi yang tetap (Lesson 17).
- Tidak ada panggilan alat yang masuk ke titik akhir yang tidak sah.
- Penulisan memori melalui jalur yang dicatat dan ditandatangani.
- Hash modul evaluator cocok dengan versi yang disetujui.

Invarian diperiksa dengan code eksternal yang tidak dapat diedit oleh loop. Jika modifikasi yang diusulkan melanggar invarian, modifikasi tersebut ditolak. Ini menangkap mode gangguan evaluator gaya DGM (Lesson 4).

Bagian tersulitnya adalah memilih invarian yang diperlukan untuk keamanan dan dapat dihitung dalam waktu yang wajar. Kemajuan yang terlalu spesifik terhenti; penentuan yang terlalu rendah memungkinkan perulangan melewati properti-properti penting.

### Primitif 2: jangkar penyelarasan

Jangkar penyelarasan adalah representasi tujuan inti perulangan yang tidak dapat diubah, yang di-embed di luar permukaan edit perulangan. Contoh:

- Teks konstitusi (Lesson 17) yang dimuat dari lokasi read-only pada setiap doa.
- Seorang evaluator di repositori terpisah dengan kontrol akses independen.
- Manifes izin alat yang ditandatangani oleh manusia dan diverifikasi ulang setiap siklus.

Peran jangkar adalah untuk mencegah penyimpangan objektif. Perulangan ini dapat meningkatkan cara mencapai tujuan, namun tidak dapat mengubah tujuan.

Mode kegagalan yang halus: sebuah jangkar yang tidak dapat diedit oleh loop masih dapat diinterpretasikan ulang oleh sebuah loop yang mengikuti cara ia membaca jangkar. AI Konstitusional (Lesson 17) secara eksplisit didasarkan pada alasan untuk menangani situasi baru; layer penalaran itulah yang menjadi tempat terjadinya penyimpangan interpretasi. Jangkar itu perlu, tidak cukup.

### Primitif 3: batasan multi-tujuanSebuah loop yang mengoptimalkan skor scalar tunggal akan menemukan jalan pintas. Sebuah loop yang secara bersamaan harus memenuhi beberapa batasan keras memiliki lebih sedikit pintasan yang tersedia. Sumbu khas:

- Kinerja (patokan tingkat tugas)
- Keamanan (evaluasi tim merah, tingkat penolakan pada yang diketahui-buruk)
- Kewajaran (batas dampak yang berbeda pada subkelompok sensitif)
- Kekokohan (set pengujian OOD, penanganan input yang berlawanan)

Modifikasi diterima hanya jika setiap batasan terpenuhi. Pengatur biaya Lesson 13 menumpukkan hal ini dengan kendala keuangan. Llama Guard Lesson 18 dihubungkan sebagai poros pengaman.

### Primitif 4: deteksi regresi

Skor setiap siklus dibandingkan dengan distribusi historis. Penurunan yang melampaui toleransi yang dikonfigurasikan akan menjeda loop. Ini menangkap hilangnya kemampuan senyap yang seharusnya diserap ke dalam rata-rata berjalan saat loop "meningkat melewatinya".

Implementasi praktis: menyimpan skor per tugas N siklus terakhir. Setiap siklus baru menghitung delta per tugas. Jika ada delta yang turun di bawah ambang batas, siklusnya ditolak dan dilakukan peninjauan oleh manusia.

### Batasan teori informasi

Kompleksitas Kolmogorov dan teorema Lob menetapkan batasan atas apa yang dapat dibuktikan oleh suatu sistem tentang dirinya sendiri. Mesin Godel formal Schmidhuber (Lesson 4) ditujukan untuk batas tertinggi; tidak ada yang menyelesaikan bukti non-sepele. Hasil Lob mengatakan: jika suatu sistem terbukti percaya "Saya akan melakukan X jika saya membuktikan bahwa saya harus melakukan X," sistem akan melakukan X tanpa membuktikan bahwa hal itu seharusnya dilakukan, sebuah kegagalan referensi mandiri yang terkenal.

Implikasinya bagi primitif kita: mereka tidak bisa menutup masalah keamanan. Mereka membuat kegagalan diam-diam menjadi lebih mahal. Lingkaran berbahaya atau menyimpang yang secara diam-diam akan melewati pemeriksaan yang hilang kini harus secara aktif melemahkan pemeriksaan eksplisit, yang merupakan tanda tangan yang lebih dapat dideteksi.

### Contoh yang berhasil

Misalkan seorang agen mengusulkan pengeditan. Tumpukan gerbang:

1. Pemeriksaan invarian: hash modul, manifes izin alat, header konstitusional.
2. Pemeriksaan jangkar: pernyataan objektif cocok dengan versi yang disetujui (berdasarkan byte atau semantik).
3. Evaluasi multi-tujuan: sumbu kinerja, keselamatan, keadilan, ketahanan.
4. Deteksi regresi: tidak ada sumbu yang turun lebih dari toleransi.

Keempatnya harus lolos agar pengeditan dapat dilakukan. Setiap kegagalan akan menghentikan perulangan.

## Pakai

`code/main.py` menjalankan putaran pengembangan diri terbatas pada mainan gaya DGM dari Lesson 4, tetapi dengan empat primitif berlapis di atasnya. Setiap primitif dapat diaktifkan atau dinonaktifkan satu per satu. Demonstrasinya adalah setiap primitif menangkap kelas kegagalan tertentu, dan menghapus salah satu dari kelas tersebut akan membiarkan kelas kegagalan tersebut lewat.

## Kirim

`outputs/skill-bounded-loop-review.md` mengaudit loop terbatas yang diusulkan dan menilai mana dari empat primitif yang benar-benar diterapkan versus yang diklaimnya.

## Latihan

1. Jalankan `code/main.py` dengan semua primitif diaktifkan. Konfirmasikan bahwa loop masih meningkatkan metrik utama tanpa membiarkan peretasan menang.

2. Nonaktifkan deteksi regresi. Buatlah input yang menyebabkan hilangnya kemampuan senyap diterima.

3. Nonaktifkan batasan multi-tujuan. Tunjukkan loop menyatu pada sumbu kinerja sementara sumbu keselamatan turun.

4. Rancang jangkar penyelarasan untuk agen pengkodean. Teks apa, disimpan di mana, diperiksa caranya?

5. Baca ringkasan Workshop RSI ICLR 2026. Pilih salah satu dari empat primitif dan usulkan perbaikan nyata terhadap keadaan seni saat ini.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| Invarian | "Properti yang selalu benar" | Properti diperiksa oleh code eksternal sebelum dan sesudah setiap pengeditan |
| Jangkar penyelarasan | "Tujuan yang di-embed" | Representasi tujuan inti yang tidak dapat diubah di luar permukaan edit loop |
| Batasan multi-tujuan | "Semua sumbu harus menahan" | Performa, keamanan, keadilan, ketahanan — semuanya diperlukan |
| Deteksi regresi | "Jeda saat dijatuhkan" | Jeda loop ketika delta metrik historis menunjukkan hilangnya kemampuan |
| Kolmogorov terikat | "Batas teori informasi" | Membatasi apa yang dapat dibuktikan oleh suatu sistem tentang penerusnya |
| Teorema Lob | "Perangkap referensi mandiri" | Sistem dapat bertindak berdasarkan "Saya harus" tanpa membuktikan bahwa sistem seharusnya |
| Tumpukan gerbang | "Cek berlapis" | Gabungan beberapa primitif; setiap kegagalan menolak edit |
| Peningkatan terbatas | "Mitigasi, bukan pembuktian" | Meningkatkan biaya kegagalan diam-diam; tidak menutup masalah keamanan |

## Bacaan Lanjutan

- [Ringkasan Lokakarya RSI ICLR 2026 (OpenReview)](https://openreview.net/pdf?id=OsPQ6zTQXV) — konvergensi empat primitif.
- [Kebijakan Penskalaan Bertanggung Jawab Antropik v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — ambang batas kemampuan multi-tujuan.
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — pemantauan penyelarasan yang menipu sebagai primitif invarian.
- [Schmidhuber (2003). Mesin Godel](https://people.idsia.ch/~juergen/goedelmachine.html) — nenek moyang primitif yang tahan formal ini.
- [Anthropic — Konstitusi Claude (Januari 2026)](https://www.anthropic.com/news/claudes-constitution) — jangkar penyelarasan berbasis alasan.
