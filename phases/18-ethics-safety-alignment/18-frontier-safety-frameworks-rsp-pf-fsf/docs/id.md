# Kerangka Keamanan Frontier — RSP, PF, FSF

> Tiga framework laboratorium utama menentukan tata kelola industri kemampuan frontier pada tahun 2026. Kebijakan Penskalaan Bertanggung Jawab Antropik v3.0 (Februari 2026) memperkenalkan Tingkat Keamanan AI berjenjang (ASL-1 hingga ASL-5+), yang dimodelkan berdasarkan tingkat keamanan hayati, dengan ASL-3 diaktifkan pada Mei 2025 untuk model yang relevan dengan CBRN. OpenAI Preparedness Framework v2 (April 2025) menetapkan lima kriteria untuk kemampuan yang dilacak dan memisahkan Laporan Kemampuan dari Laporan Perlindungan. Framework Keamanan DeepMind Frontier v3.0 (September 2025) memperkenalkan Tingkat Kemampuan Kritis termasuk CCL Manipulasi Berbahaya yang baru. Ketiganya kini menyertakan klausul penyesuaian pesaing yang memungkinkan penangguhan jika laboratorium sejawat dikirimkan tanpa perlindungan yang sebanding. Penyelarasan lintas laboratorium tetap bersifat struktural, bukan terminologis: "Ambang Batas Kemampuan", "Ambang Batas Kemampuan Tinggi", dan "Tingkat Kemampuan Kritis" menunjukkan konstruksi analog.

**Type:** Learn
**Language:** tidak ada
**Prerequisites:** Fase 18 · 17 (WMDP), Fase 18 · 07-09 (kegagalan penipuan)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Jelaskan struktur tingkat ASL Anthropic dan apa yang mengaktifkan ASL-3.
- Sebutkan lima kriteria OpenAI Preparedness Framework v2 untuk kemampuan yang dilacak.
- Jelaskan struktur Tingkat Kemampuan Kritis DeepMind dan CCL Manipulasi Berbahaya.
- Jelaskan klausul penyesuaian peserta dan mengapa klausul tersebut penting bagi dinamika balapan.
- Tentukan kasus keselamatan dan jelaskan struktur tiga pilar (pemantauan, ketidakterbacaan, ketidakmampuan).

## Masalah

Lesson 7-17 menetapkan bahwa penipuan mungkin terjadi, ada kemampuan penggunaan ganda, dan evaluasi ada batasnya. Lab dengan model berkemampuan frontier memerlukan struktur tata kelola internal yang:
- Menetapkan ambang batas kapan upaya perlindungan baru diperlukan.
- Mendefinisikan evaluasi yang diperlukan sebelum penskalaan.
- Menjelaskan seperti apa kasus keselamatan itu.
- Menangani masalah dinamika balapan (jika pesaing mengirim tanpa pengamanan, apa yang kamu lakukan?).

Ketiga framework 2025-2026 adalah yang paling canggih – tidak sempurna, berkembang, dan cukup selaras di seluruh laboratorium sehingga pertanyaan tata kelolanya sekarang adalah apakah framework tersebut memadai, bukan apakah kerangka tersebut ada.

## Konsep

### Kebijakan Penskalaan Bertanggung Jawab Antropik v3.0 (Februari 2026)

Struktur ASL:
- ASL-1: bukan model frontier (digabungkan dengan baseline yang lebih lemah dari frontier).
- ASL-2: garis dasar perbatasan saat ini; dikerahkan dengan pengamanan biasa.
- ASL-3: risiko penyalahgunaan bencana yang jauh lebih tinggi; Kemampuan yang relevan dengan CBRN. Diaktifkan Mei 2025.
- ASL-4: ambang batas penelitian dan pengembangan AI-2; model yang dapat mengotomatiskan penelitian AI tingkat pemula.
- ASL-5+: Penelitian dan Pengembangan AI tingkat lanjut; model yang secara dramatis mempercepat penskalaan yang efektif.

Baru di v3.0:
- Peta Jalan Keamanan Frontier (publik dalam bentuk yang telah disunting).
- Laporan Risiko (triwulanan, beberapa ditinjau secara eksternal).
- Penelitian dan pengembangan AI dipilah menjadi penelitian dan pengembangan AI-2 dan penelitian dan pengembangan AI-4.
- Setelah R&D-4 AI berhasil diterapkan, diperlukan kasus keselamatan afirmatif, yang mengidentifikasi risiko ketidakselarasan dari model yang mengejar tujuan yang tidak selaras.

### Kerangka Kesiapsiagaan OpenAI v2 (15 April 2025)

Lima kriteria untuk kemampuan yang dilacak:
- **Masuk akal.** Ada model ancaman yang masuk akal.
- **Dapat diukur.** Evaluasi empiris dapat dilakukan.
- **Parah.** Bahayanya besar.
- **Net-new.** Bukan peningkatan risiko yang sudah ada sebelumnya.
- **Seketika atau tidak dapat diperbaiki.** Kerusakan terjadi dengan cepat atau tidak dapat diperbaiki.

Kemampuan yang memenuhi kelimanya dilacak. Yang lainnya tidak.Struktur PF v2 lainnya:
- Laporan Kemampuan yang terpisah (apa yang dapat dilakukan model) dari Laporan Perlindungan (kontrol apa yang ada).
- Tinjauan Kelompok Penasihat Keamanan.
- Pimpinan menyetujui; Komite Keselamatan & Keamanan Dewan mengawasi.
- "Klausul penyesuaian": OpenAI dapat mengurangi persyaratan jika lab lain dikirimkan tanpa perlindungan yang sebanding.

### Kerangka Keamanan DeepMind Frontier v3.0 (September 2025)

Tingkat Kemampuan Kritis (CCL) berdasarkan domain:
- Peningkatan Senjata Biologis
- Peningkatan Dunia Maya
- Akselerasi Litbang ML
- Manipulasi Berbahaya (baru di v3.0): model yang secara substansial dapat mengubah keyakinan/perilaku dalam konteks berisiko tinggi.

v2.0 (Februari 2025) menambahkan bagian Penyelarasan yang Menipu dan tingkat keamanan yang lebih tinggi untuk CCL Litbang ML.

### Penyelarasan lintas lab

- "Ambang Kemampuan" antropis.
- DeepMind "Tingkat Kemampuan Kritis."
- OpenAI "Ambang Batas Kemampuan Tinggi."

Tidak ada terminologi standar industri. Selaras secara struktural: tiga tingkatan kemampuan terdepan dengan kriteria evaluasi yang dipublikasikan. Ketiganya mencakup klausul penyesuaian pesaing (2025 dan seterusnya).

### Kasus keselamatan

Kasus keselamatan adalah argumen tertulis bahwa penerapan aman berdasarkan asumsi kasus terburuk. Struktur standar menargetkan tiga pilar:

- **Pemantauan.** Bisakah kami mendeteksi perilaku buruk jika terjadi?
- **Tidak terbaca.** Apakah model tidak memiliki kemampuan untuk melaksanakan rencana yang koheren sehingga dapat menimbulkan loss?
- **Ketidakmampuan.** Apakah model tidak memiliki kemampuan untuk menyebabkan loss yang dimaksud?

Kasus keselamatan yang berbeda menargetkan pilar yang berbeda. Untuk kasus CBRN ASL-3, ketidakmampuan (melalui pelepasan pembelajaran) adalah target utamanya. Untuk penyelarasan yang menipu, pemantauan dan ketidakterbacaan adalah targetnya. Untuk peningkatan dunia maya, ketiganya relevan.

### Masalah dinamika balapan

Klausul penyesuaian pesaing masih kontroversial. Kritikus berpendapat bahwa hal ini menciptakan perlombaan menuju titik terbawah: jika ketiga laboratorium akan mengurangi persyaratan ketika pesaing melakukan cacat, maka keseimbangan akan bergeser ke arah pembelotan. Para pembela berpendapat bahwa alternatif (pengamanan sepihak) akan memberikan hasil yang lebih buruk jika laboratorium yang membelot kurang sadar akan keselamatan.

AISI Inggris, CAISI AS, dan Kantor AI UE (Lesson 24) merupakan mitra tata kelola eksternal. Kerangka kerja laboratorium bersifat sukarela; kerangka peraturan mulai muncul.

### Cocok untuk Fase 18

Lesson 17-18 adalah layer pengukuran dan tata kelola di atas penipuan dan analisis tim merah. Lesson 19-24 mencakup kesejahteraan, bias, privasi, watermarking, dan struktur peraturan. Lesson 28 memetakan ekosistem penelitian (MATS, Redwood, Apollo, METR) yang mengoperasionalkan evaluasi.

## Pakai

Tidak ada code untuk lesson ini. Baca tiga sumber utama: RSP v3.0, PF v2, FSF v3.0. Petakan struktur tingkat setiap lab satu sama lain dan identifikasi satu ambang batas yang ditentukan setiap lab yang tidak ditentukan oleh lab lain.

## Kirim

Lesson ini menghasilkan `outputs/skill-framework-diff.md`. Dengan adanya kerangka keselamatan atau catatan rilis, framework ini membandingkan definisi ambang batas framework, evaluasi yang diperlukan, dan struktur kasus keselamatan dengan RSP v3.0, PF v2, FSF v3.0, dan menandai kesenjangan lintas lab.

## Latihan

1. Baca RSP v3.0, PF v2, dan FSF v3.0. Kompilasi tabel ambang batas CBRN masing-masing lab, ambang batas Litbang AI masing-masing, dan evaluasi pra-penerapan yang diperlukan untuk masing-masing lab.

2. Klausul penyesuaian kompetitor ada di ketiga kerangka tersebut (2025+). Tulislah satu paragraf yang mendukungnya; tulis satu paragraf yang menentang. Identifikasi asumsi yang menjadi dasar setiap posisi.3. Rancang kotak pengaman untuk model yang melewati ambang batas R&D-4 AI Anthropic. Sebutkan bukti yang diperlukan oleh ketiga pilar (pemantauan, ketidakterbacaan, ketidakmampuan).

4. FSF v3.0 DeepMind memperkenalkan CCL Manipulasi Berbahaya. Usulkan tiga pengukuran empiris yang menunjukkan bahwa suatu model telah melewati ambang batas ini.

5. Baca "Elemen Umum Kebijakan Keamanan AI Frontier" METR (2025). Sebutkan tiga konvergensi lintas lab terkuat dan dua divergensi terbesar.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| RSP | "Kerangka Antropis" | Kebijakan Penskalaan yang Bertanggung Jawab; tingkatan ASL; v3.0 Februari 2026 |
| PF | "Kerangka OpenAI" | Kerangka Kesiapsiagaan; lima kriteria; v2 April 2025 |
| FSF | "Kerangka DeepMind" | Kerangka Keamanan Perbatasan; CCL; v3.0 September 2025 |
| ASL-3 | "keamanan hayati tingkat 3-analog" | Tingkat antropik untuk kemampuan yang relevan dengan CBRN; diaktifkan Mei 2025 |
| CCL | "tingkat kemampuan kritis" | Konstruksi ambang batas DeepMind; per domain |
| Kasus keamanan | "argumen formal" | Argumen tertulis bahwa penerapan cukup aman dalam kasus terburuk U |
| Klausul penyesuaian | "tunjangan pembelotan pesaing" | Ketentuan framework untuk mengurangi persyaratan jika pesaing melakukan pengiriman tanpa perlindungan yang sebanding |

## Bacaan Lanjutan

- [Anthropic — Responsible Scaling Policy v3.0 (Februari 2026)](https://www.anthropic.com/responsible-scaling-policy) — tingkatan ASL, peta jalan, disagregasi Penelitian dan Pengembangan AI
- [OpenAI — Memperbarui Kerangka Kesiapsiagaan (15 April 2025)](https://openai.com/index/updating-our-preparedness-framework/) — lima kriteria, klausul penyesuaian
- [DeepMind — Memperkuat Kerangka Keamanan Frontier kami (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0, Manipulasi Berbahaya
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — perbandingan lintas lab
