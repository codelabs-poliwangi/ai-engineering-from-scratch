# Penelitian Penyelarasan Otomatis (Anthropic AAR)

> Anthropic menjalankan tim paralel Claude Opus 4.6 Autonomous Alignment Peneliti di kotak pasir independen, berkoordinasi melalui forum bersama yang lognya berada di luar kotak pasir mana pun (sehingga agen tidak dapat menghapus catatan mereka sendiri). Dalam permasalahan training lemah hingga kuat, AAR mengungguli peneliti manusia. Tanda ringkasan Anthropic yang menentukan alur kerja sering kali membatasi fleksibilitas AAR dan menurunkan kinerja. Mengotomatiskan penelitian penyelarasan adalah langkah kompresi yang memampatkan garis waktu hingga risiko ketidakselarasan yang tepat yang ingin dideteksi oleh RSP.

**Type:** Learn
**Language:** Python (stdlib, simulator forum penelitian paralel)
**Prerequisites:** Fase 15 · 05 (Ilmuwan AI v2), Fase 15 · 04 (DGM)
**Waktu:** ~60 menit

## Masalah

Penelitian keselarasan mahal dalam waktu manusia-peneliti. Masalah seperti pengawasan yang terukur, spesifikasi imbalan, atau training lemah hingga kuat memerlukan eksperimen yang memerlukan waktu berminggu-minggu untuk setiap iterasinya. Seiring dengan kemajuan kemampuan terdepan, weight kerja penyelarasan tumbuh lebih cepat dibandingkan penyediaan peneliti yang berkualifikasi.

Automated Alignment Research (AAR) menanyakan apakah model frontier yang sama yang kemampuannya melebihi keselarasan dapat berkontribusi untuk menutup kesenjangan. Laporan Anthropic tahun 2026 tentang studi training lemah hingga kuat yang dijalankan AAR (alignment.anthropic.com/2026/automated-w2s-researcher/) adalah salah satu hasil publik pertama dari sistem yang diterapkan pada kelas ini.

Hasilnya benar-benar positif: AAR memecahkan masalah penelitian lebih baik daripada data dasar manusia. Hasilnya juga terkonsentrasi, dalam satu sistem, setiap attention yang telah dikembangkan pada fase ini. Jika penelitian penyelarasan dapat diotomatisasi, maka bagian-bagian yang membahayakan upaya perlindungan juga dapat diotomatisasi. Ambang batas RSP untuk kemampuan Penelitian dan Pengembangan AI ditulis dengan mempertimbangkan hal ini.

## Konsep

### Arsitektur (seperti yang dijelaskan secara publik)

- **Agen paralel.** Beberapa instance Claude Opus 4.6 dijalankan sebagai AAR, masing-masing dalam sandbox independen. Sandbox membatasi sistem file, jaringan, dan permukaan proses.
- **Forum bersama.** Temuan dipublikasikan ke forum bersama yang backend penyimpanannya berada di luar setiap sandbox. Agen dapat membaca forum; agen tidak dapat menghapus atau mengedit catatan forum dari dalam kotak pasir mereka sendiri.
- **Alokasi tugas.** Antrean tugas yang dirancang manusia menetapkan masalah penelitian ke AAR. AAR mengusulkan decomposition, menjalankan eksperimen, menulis temuan.
- **Fokus training lemah ke kuat.** Studi yang dipublikasikan membahas training lemah ke kuat — menggunakan model pengawas yang lebih lemah untuk melatih model siswa yang lebih kuat. AAR mengungguli peneliti manusia dalam tugas ini.

### Mengapa log out-of-sandbox penting

Properti integritas log adalah perbedaan antara eksperimen terkontrol dan eksperimen yang berpotensi disusupi. Jika AAR dapat mengedit lognya sendiri, mereka dapat menutupi eksperimen yang gagal atau meragukan. Anthropic menandai hal ini secara langsung: properti adalah kondisi yang diperlukan untuk mempercayai hasil penelitian, dan verifikasinya bukanlah hal yang sepele.

Analog dalam istilah infrastruktur adalah log tambahan saja dengan penulisan ke penyimpanan eksternal (misalnya kunci objek S3, log Datadog yang tidak dapat diubah). Agen tidak memiliki kredensial untuk mengubah catatan masa lalu. Ini adalah pola yang sama yang digunakan tim keamanan perusahaan pada tahun 2026 untuk membuat jejak audit yang tahan terhadap ransomware.

### Pengorbanan alur kerja yang ditentukanLaporan Anthropic mencatat bahwa alur kerja yang ditentukan sering kali membatasi fleksibilitas AAR dan menurunkan kinerja. Dengan kata lain: jika perancang manusia menulis rencana penelitian langkah demi langkah, kinerja AAR akan lebih buruk dibandingkan jika diberi tujuan tingkat tinggi dan kebebasan untuk menguraikannya. Hal ini konsisten dengan hasil penelitian agen (LangGraph, Kerangka Agen Microsoft) — perancah yang terlalu ditentukan akan merugikan agen yang mampu.

Implikasi keamanannya tidak kentara. Fleksibilitas yang meningkatkan kinerja AAR juga merupakan fleksibilitas yang memungkinkan decomposition tanpa izin, penggunaan alat baru, dan penyimpangan sasaran. Pengorbanannya adalah keputusan ruang desain: berapa banyak fleksibilitas yang dapat membeli kemampuan yang cukup untuk mengalahkan manusia, tanpa memberikan ruang kepada agen untuk menggeneralisasi tujuan di luar distribusi?

### Risiko kompresi

RSP v3.0 (Lesson 19) memperkenalkan ambang batas kemampuan Penelitian dan Pengembangan AI: kemampuan untuk sepenuhnya mengotomatisasi alur Penelitian dan Pengembangan AI dengan biaya yang kompetitif dibandingkan dengan alat manusia + AI. FSF v3 DeepMind menyertakan tingkat otonomi Litbang ML yang serupa. Kedua framework tersebut memperlakukan ambang batas ini sebagai pemicu peningkatan kontrol.

AAR hanya satu langkah lebih pendek dari ambang batas: AAR mengotomatiskan bagian dari alur (penelitian penyelarasan pada tugas-tugas spesifik dan tercakup dengan baik) namun tidak mengotomatiskan putaran pengembangan kemampuan end-to-end. Pertanyaan garis waktunya adalah seberapa cepat kesenjangan tersebut tertutup.

Garis waktu yang padat adalah masalah yang semakin memperparah kegagalan. Jika penelitian penyelarasan dan penelitian kemampuan digabungkan pada tingkat yang sama, permukaan risiko ketidakselarasan akan tumbuh setidaknya secepat kemampuan. Jika kemampuan meningkat lebih cepat (tren historis), kesenjangan akan semakin lebar. Inilah argumen yang menjadikan AAR sebagai barang yang memenuhi syarat: setiap hasil penyelarasan tambahan mengurangi kesenjangan jika dan hanya jika proses penelitian dapat dipercaya.

### Apa yang tidak digantikan oleh AAR

Peneliti manusia mengatur antrian tugas, meninjau hasil, dan memegang kewenangan konstitusional. AAR mempercepat bagian tengah pipa, bukan bagian ujungnya. Output Anthropic yang dipublikasikan mencakup kontribusi AAR dan penilaian manusia-peneliti tentang apa yang harus dipublikasikan, apa yang harus ditarik kembali, dan apa yang harus disempurnakan.

Hal ini sesuai dengan pola usulan lalu komitmen dari Lesson 15 yang diterapkan pada penelitian itu sendiri: AAR mengusulkan; manusia berkomitmen.

## Pakai

`code/main.py` mensimulasikan forum penelitian multi-AAR. Tiga "agen penelitian" independen berjalan secara paralel, memposting temuan ke forum khusus tambahan dengan upaya perusakan opsional.

Tonton:

- Bagaimana logging out-of-sandbox menangkap upaya perusakan log.
- Bagaimana alokasi tugas dengan decomposition bebas mengungguli alur kerja tetap pada masalah yang lebih sulit.
- Bagaimana satu AAR yang disusupi dapat mengkontaminasi sinyal forum jika log tidak dapat dirusak.

## Kirim

`outputs/skill-aar-deployment-review.md` adalah tinjauan pra-penerapan dari jalur penelitian penyelarasan otomatis.

## Latihan

1. Jalankan `code/main.py`. Bandingkan pengaturan "alur kerja tetap" vs "decomposition bebas". Apakah decomposition bebas selalu menang, atau adakah kelas masalah yang alur kerjanya tetap kompetitif?

2. Ubah simulator sehingga salah satu agen mencoba merusak log. Konfirmasikan bahwa log hanya tambahan mendeteksinya. Tulis satu paragraf yang menjelaskan dengan tepat seperti apa tanda tangan deteksi di log.

3. Baca laporan AAR lemah hingga kuat dari Anthropic. Identifikasi sub-tugas spesifik yang dilakukan AAR terhadap peneliti manusia. Apa yang membuatnya bisa menerima otomatisasi?

4. Rancang kebijakan alokasi antrian tugas yang menyeimbangkan fleksibilitas AAR (hasil yang lebih baik) terhadap batasan alur kerja yang ditentukan (audit lebih mudah). Jelaskan bagaimana kamu akan menguji keduanya secara A/B.5. Baca ambang R&D-4 AI RSP v3.0. Dalam satu paragraf, jelaskan apa yang menurut kamu akan melanggar peraturan yang saat ini tidak dimiliki oleh AAR.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| AAR | "Peneliti Penyelarasan Otomatis" | Contoh Claude Opus 4.6 beroperasi secara mandiri pada masalah penyelarasan |
| Latihan lemah ke kuat | "Melatih model yang lebih kuat dengan supervisor yang lebih lemah" | Tolok ukur pengawasan skalabel klasik, AAR, mengungguli manusia pada |
| Forum bersama | "Di mana agen mempublikasikan temuan" | Penyimpanan khusus tambahan di luar sandbox |
| Log di luar kotak pasir | "Agen tidak dapat mengedit catatannya sendiri" | Tulisan anti kerusakan ke penyimpanan eksternal |
| Alur kerja yang ditentukan | "Rencana langkah demi langkah dari desainer manusia" | Membatasi AAR; sering menurunkan kinerja vs decomposition bebas |
| Decomposition bebas | "Agen memutuskan bagaimana memecahkan tugas" | Lebih mampu, lebih sulit untuk diaudit |
| Ambang batas penelitian dan pengembangan AI | "Tingkat kemampuan RSP/FSF" | Otomatisasi penuh jalur penelitian dan pengembangan dengan biaya kompetitif |
| Garis waktu terkompresi | "Perlombaan keselarasan vs kemampuan" | Jika kemampuan bertambah lebih cepat daripada penyelarasan, risiko misalignment meningkat |

## Bacaan Lanjutan

- [Antropik — Peneliti Lemah-ke-Kuat Otomatis](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — sumber utama.
- [Kebijakan Anthropic Responsible Scaling v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — Pembingkaian ambang batas Penelitian dan Pengembangan AI.
- [Anthropic — Mengukur otonomi agen AI](https://www.anthropic.com/research/measuring-agent-autonomy) — kerangka otonomi agen yang lebih luas.
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — Tingkat otonomi Litbang ML sejajar dengan RSP.
- [Burns dkk. (2023). Generalisasi Lemah-ke-Kuat (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) — masalah mendasar yang diserang AAR.
