# Kebijakan Penskalaan Bertanggung Jawab Antropis v3.0

> RSP v3.0 mulai berlaku 24 Februari 2026, menggantikan kebijakan tahun 2023. Mitigasi dua tingkat: apa yang akan dilakukan Anthropic secara sepihak vs apa yang dibingkai sebagai rekomendasi industri (termasuk standar keamanan RAND SL-4). Menambahkan Peta Jalan Keselamatan Frontier dan Laporan Risiko sebagai dokumen tetap, bukan hanya sekedar hasil kerja. Mencabut komitmen jeda tahun 2023. Memperkenalkan ambang batas R&D-4 AI: setelah terlampaui, Anthropic harus menerbitkan kasus afirmatif yang mengidentifikasi risiko dan mitigasi ketidakselarasan. Claude Opus 4.6 tidak melewatinya. Anthropic menyatakan dalam pengumuman v3.0 bahwa "dengan yakin mengesampingkan hal ini menjadi sulit." SaferAI menilai RSP 2023 sebesar 2,2; mereka menurunkan versi v3.0 menjadi 1.9, menempatkan Anthropic dalam kategori RSP "lemah" bersama OpenAI dan DeepMind. Ambang batas kualitatif menggantikan komitmen kuantitatif tahun 2023; menghapus klausa jeda adalah kemunduran paling tajam.

**Type:** Learn
**Language:** Python (stdlib, mesin keputusan ambang batas RSP)
**Prerequisites:** Fase 15 · 06 (AAR), Fase 15 · 07 (RSI)
**Waktu:** ~45 menit

## Masalah

Laboratorium Frontier menerbitkan kebijakan penskalaan yang sebagian berupa dokumen teknis, sebagian dokumen tata kelola, dan sebagian lagi merupakan sinyal kepada regulator. RSP v3.0 adalah dokumen Antropik terkini. Membacanya dengan cermat bukanlah hal yang penting karena kepatuhan terhadapnya bersifat mengikat (walaupun tidak mengikat), namun karena kerangka tersebut membentuk cara laboratorium memahami risiko bencana dan cara mereka mengomunikasikan dampaknya kepada publik.

Perbedaan v3.0 vs v2.0 adalah unit yang berguna. Apa yang ditambahkan: Peta Jalan Keselamatan Frontier, Laporan Risiko, ambang batas R&D-4 AI. Yang dihapus: komitmen jeda tahun 2023. Apa yang diubah: jadwal mitigasi dua tingkat yang dibagi antara rekomendasi Antropik-unilateral dan rekomendasi industri. Tinjauan eksternal — SaferAI — menurunkan skor dari 2.2 (v2) menjadi 1.9 (v3.0). Inilah yang membuat kebijakan penskalaan menjadi tidak terlalu ketat namun terlihat lebih sempurna.

## Konsep

### Jadwal mitigasi dua tingkat

- **Tindakan sepihak Anthropic**: apa yang akan dilakukan Anthropic terlepas dari apa yang dilakukan laboratorium lain. Training berhenti di atas ambang batas, langkah-langkah keamanan khusus, gerbang penerapan khusus.
- **Rekomendasi seluruh industri**: apa yang menurut Anthropic harus dilakukan secara kolektif oleh industri. Termasuk standar keamanan RAND SL-4. Ini bukanlah komitmen dari pihak Anthropic; mereka adalah advokasi kebijakan.

Struktur dua tingkat tidak ada di v2. Artinya, pembaca perlu melihat di kolom mana setiap komitmen berada. Tindakan pengamanan di kolom "rekomendasi industri" bukanlah janji Anthropic; itu adalah harapan Anthropic.

### Ambang batas Penelitian dan Pengembangan AI-4

Ini adalah nama tingkat kemampuan RSP v3.0 sebagai ambang batas penting berikutnya. Khususnya: model yang dapat mengotomatisasi sebagian besar penelitian AI dengan biaya yang kompetitif. Setelah Anthropic yakin bahwa suatu model berhasil melewatinya, mereka harus menerbitkan kasus afirmatif yang mengidentifikasi risiko dan mitigasi ketidakselarasan sebelum melanjutkan penskalaan.

Claude Opus 4.6 tidak melewatinya sesuai pengumuman v3.0. Dokumen tersebut menambahkan: "dengan yakin mengesampingkan hal ini menjadi sulit." Ungkapan itu penting; ia mengakui bahwa ambang batas tersebut cukup dekat untuk menjadi attention nyata, bukan batas spekulatif.

Lesson 6 (Penelitian Penyelarasan Otomatis) dan Lesson 7 (Peningkatan Diri Rekursif) dimasukkan langsung ke dalam ambang batas ini. Peneliti penyelarasan otomatis yang melintasi batas kualitas penelitian adalah bukti bahwa ambang batas R&D-4 AI semakin dekat.### Peta Jalan Keselamatan Frontier dan Laporan Risiko

v3.0 meningkatkan dua jenis artefak menjadi dokumen tetap:

- **Frontier Safety Roadmap**: dokumen berwawasan ke depan yang menjelaskan rencana kerja keselamatan, ekspektasi kemampuan, dan penelitian mitigasi.
- **Laporan Risiko**: dokumen retrospektif pada model tertentu setelah rilis, yang menjelaskan kemampuan yang diamati dan risiko sisa.

Keduanya bersifat publik. Keduanya diperbarui dengan irama yang dinyatakan. Kegunaannya adalah: pembaca dapat melacak bagaimana apa yang menurut Anthropic akan mereka lakukan dalam Peta Jalan dibandingkan dengan apa yang mereka laporkan dalam Laporan Risiko.

### Menghapus klausa jeda

RSP 2023 mencakup komitmen jeda yang eksplisit: jika suatu model melewati ambang batas kemampuan tertentu, training akan dijeda hingga mitigasi dilakukan. v3.0 menggantikan jeda eksplisit dengan formulasi yang lebih lembut (publikasikan kasus afirmatif, lanjutkan jika mitigasi sudah memadai). SaferAI dan analis lainnya menyebut hal ini secara langsung sebagai regresi terkuat dalam dokumen baru.

Argumen kebijakan yang mendukung perubahan ini: ambang batas kuantitatif pada tahun 2023 ternyata tidak dapat dicapai oleh tolok ukur kemampuan era tahun 2026 karena tolok ukur tersebut diubah skalanya. Argumen tandingannya: klausul jeda dalam kebijakan penskalaan adalah perangkat komitmen; menghapusnya berarti menghilangkan kredibilitas kebijakan tersebut.

### Penurunan versi SaferAI

SaferAI adalah organisasi independen yang menilai dokumen bergaya RSP. Peringkat publik mereka: RSP Antropik 2023 mendapat skor 2,2 (dari skala di mana 4,0 adalah RSP terbaik saat ini dan 1,0 adalah nominal). v3.0 mencetak 1,9. Hal ini mengubah Anthropic dari "moderat" menjadi "lemah", bergabung dengan OpenAI dan DeepMind dalam kategori lemah.

Faktor penurunan versi menurut SaferAI:
- Ambang batas kualitatif menggantikan ambang batas kuantitatif.
- Jeda komitmen dihapus.
- Mitigasi ambang batas R&D-4 AI digambarkan sebagai “kasus afirmatif” dan bukan tindakan spesifik.
- Mekanisme peninjauan bergantung pada Kelompok Penasihat Keselamatan Anthropic, dengan pengawasan independen yang terbatas.

### Apa yang bukan lesson ini

Ini bukanlah lesson tentang kepatuhan. RSP v3.0 bukanlah peraturan; tidak ada yang memaksa Anthropic untuk mengikutinya. Pelajarannya adalah membaca dokumen dengan kekhususan dan skeptisisme yang layak. Kebijakan penskalaan adalah sinyal publik utama yang dikeluarkan oleh laboratorium terdepan mengenai postur risiko bencana. Membacanya dengan baik adalah keterampilan praktis bagi siapa saja yang pekerjaannya bergantung pada kemampuan terdepan.

## Pakai

`code/main.py` mengimplementasikan mesin keputusan kecil yang mencerminkan bentuk evaluasi ambang batas RSP: berdasarkan model kandidat dan serangkaian pengukuran kemampuan, kembalikan apakah ambang batas R&D-4 AI terlampaui, bagian kasus afirmatif yang diperlukan, dan apakah penerapan dapat dilanjutkan. Ini sengaja dibuat sederhana; intinya adalah membuat logika dokumen menjadi eksplisit.

## Kirim

`outputs/skill-scaling-policy-review.md` meninjau kebijakan penskalaan (Anthropic, OpenAI, DeepMind, atau internal) terhadap referensi v3.0: struktur dua tingkat, ambang batas, komitmen jeda, tinjauan independen.

## Latihan

1. Jalankan `code/main.py`. Umpan dalam tiga model sintetis pada tingkat kemampuan berbeda. Konfirmasikan bahwa penilai ambang batas berperilaku seperti yang diharapkan dan menghasilkan templat kasus afirmatif yang tepat.

2. Baca RSP v3.0 selengkapnya (32 halaman). Identifikasi setiap komitmen yang ada di tingkat "rekomendasi seluruh industri". Komitmen manakah yang bersifat "Antropik sepihak" di v2?3. Baca metodologi penilaian RSP SaferAI. Reproduksi skor 1,9 mereka untuk v3.0 dengan menerapkan rubrik mereka pada dokumen. Baris rubrik manakah yang paling banyak mendorong penurunan peringkat?

4. Komitmen jeda tahun 2023 dihilangkan. Usulkan komitmen pengganti yang menjaga kredibilitas kebijakan sekaligus mengakui masalah penskalaan ulang tolok ukur pada tahun 2026.

5. Bandingkan RSP v3.0 dengan OpenAI Preparedness Framework v2 (Lesson 20). Pilih satu area di mana v3.0 lebih kuat. Pilih satu bidang dimana Kerangka Kesiapsiagaan lebih kuat.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| RSP | "Kebijakan penskalaan antropik" | Kebijakan Penskalaan yang Bertanggung Jawab; v3.0 efektif 24 Februari 2026 |
| Penelitian dan Pengembangan AI-4 | "Ambang batas otomatisasi penelitian" | Kemampuan untuk mengotomatiskan penelitian AI yang substansial dengan biaya yang kompetitif |
| Kasus afirmatif | "Pembenaran keamanan" | Argumen yang dipublikasikan bahwa risiko telah diidentifikasi dan mitigasinya memadai |
| Peta Jalan Keamanan Perbatasan | "Rencana ke depan" | Dokumen tetap tentang pekerjaan keselamatan yang direncanakan dan kemampuan yang diharapkan |
| Laporan Risiko | "Retrospektif pada suatu model" | Dokumen tetap tentang kemampuan yang diamati dan risiko sisa setelah rilis |
| Mitigasi dua tingkat | "Sepihak vs industri" | Komitmen antropis vs rekomendasi industri, dipisahkan |
| Jeda komitmen | "Klausul 2023" | Janji yang jelas untuk menghentikan training; dihapus di v3.0 |
| Peringkat AI yang lebih aman | "Kelas RSP independen" | Rubrik pihak ketiga; v3.0 mencetak 1,9 (v2 adalah 2.2) |

## Bacaan Lanjutan

- [Anthropic — Kebijakan Penskalaan yang Bertanggung Jawab v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — kebijakan lengkap sepanjang 32 halaman.
- [Anthropic — pengumuman RSP v3.0](https://www.anthropic.com/news/responsible-scaling-policy-v3) — ringkasan perubahan dari v2.
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — dokumen tetap yang ditautkan dari RSP v3.0.
- [Anthropic — Laporan Risiko: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — retrospektif pada model perbatasan saat ini.
- [Anthropic — Mengukur otonomi agen dalam praktiknya](https://www.anthropic.com/research/measuring-agent-autonomy) — menghubungkan R&D-4 AI dengan otonomi terukur.
