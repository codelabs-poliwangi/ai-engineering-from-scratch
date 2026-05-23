# Kontrol AI — Keamanan Meskipun Subversi

> Greenblatt, Shlegeris, Sachan, Roger (Penelitian Redwood, arXiv:2312.06942, ICML 2024). Pengendalian membingkai ulang pertanyaan keselamatan: mengingat model U yang kuat dan tidak dapat dipercaya yang mungkin melakukan optimalisasi terhadap kamu, protokol apa yang memungkinkan kamu mengambil pekerjaan yang berguna sambil membatasi kemungkinan hasil yang tidak dapat diterima? Sumber daya adalah model T yang lebih lemah dan tepercaya, anggaran tenaga kerja manusia yang tepercaya H, dan kemampuan untuk mengedit output U. Dalam bingkai ini, Pengeditan Tepercaya Pareto mendominasi Pemantauan Tidak Tepercaya pada tugas-tugas pintu belakang code. Kerangka ini mewarisi asumsi kasus terburuk dari rekayasa keamanan: keselamatan adalah probabilitas bahwa U yang dipilih secara musuh tidak dapat menyebabkan hasil nyata yang tidak dapat diterima, diukur berdasarkan tekanan tim merah yang menggantikan U.

**Type:** Learn
**Language:** Python (stdlib, batas keamanan vs kegunaan pada tugas code pintu belakang)
**Prerequisites:** Fase 18 · 07 (agen tidur), Fase 18 · 08 (penipuan), Fase 18 · 09 (penyelarasan palsu)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Tentukan model ancaman Kontrol AI: model U yang kuat dan tidak tepercaya, model T yang lemah dan tepercaya, anggaran manusia H yang tepercaya.
- Jelaskan protokol Pengeditan Tepercaya dan mengapa Pareto mendominasi Pemantauan Tidak Tepercaya pada tugas pintu belakang code ICML 2024.
- Jelaskan perbedaan antara keselarasan (U dapat dipercaya) dan kontrol (U tidak dapat dipercaya dan kita aman pula).
- Nyatakan tiga batas kendali: manusia tidak bisa berkembang, model lemah yang dipercaya akhirnya tertinggal, kendali tidak menghasilkan model kuat yang bisa dipercaya.

## Masalah

Lesson 7-9 menetapkan bahwa model frontier mungkin selaras, licik, atau memalsukan keselarasan. Pertanyaan keamanannya dibingkai ulang: dapatkah kita mengambil manfaat dari model yang tidak kita percayai, tanpa memberinya peluang untuk menimbulkan hasil yang tidak dapat diterima? Ini adalah masalah pengendalian, dan ini terstruktur seperti keamanan klasik: asumsikan musuh dalam kasus terburuk, rancang protokol, ukur tingkat keberhasilan musuh di bawah tekanan tim merah.

## Konsep

### Model ancaman

Kontrol mengasumsikan:
- U — model kuat yang tidak dapat dipercaya. Mampu. Mungkin bersifat permusuhan (evaluasi memperlakukannya seolah-olah memang demikian).
- T — model yang lebih lemah dan tepercaya. Kurang mampu, tapi dikenal tidak bermusuhan.
- H — tenaga kerja manusia yang dipercaya. Anggaran kecil; tidak berskala.

Sistem harus menghasilkan output yang berguna dengan menggunakan U. Musuh mencoba menyebabkan hasil nyata yang tidak dapat diterima (misalnya, pintu belakang dalam code yang dikirimkan) sambil mempertahankan sebagian kecil dari pekerjaan berguna yang sah untuk melewati ambang batas kegunaan.

Kontrol menyerah dalam mempercayai AS. Ia menanyakan protokol apa yang menjaga keamanan jika AS tidak memiliki kepercayaan.

### Tugas pintu belakang ICML 2024

Anjurannya adalah masalah pengkodean tingkat kompetisi. U menghasilkan solusi kandidat. Oracle yang tersembunyi menandai jika solusinya berisi pintu belakang (input spesifik yang membuat code menghasilkan output yang salah). Tim merah mengontrol U dan diizinkan melihat semua internal protokol. Protokol harus memaksimalkan tingkat penyelesaian masalah sambil menjaga tingkat pengiriman pintu belakang di bawah ambang batas.

### Empat protokol dibandingkan

- **Dasar:** mengirimkan output U secara langsung. Kegunaannya maksimal. Keamanan minimal.
- **Audit dimuka:** gunakan H pada semua output. Keamanan maksimal. Anggaran cepat habis.
- **Pemantauan Tidak Tepercaya:** instance kedua dari U memberi skor pada output instance pertama; Audit H menandai output. Resiko: dua instance U dapat berkolusi.
- **Pengeditan Tepercaya:** T mengedit output U pada setiap kueri. Jika T tidak percaya diri, lanjutkan ke H.Greenblatt dkk. temukan Pengeditan Tepercaya Pemantauan Tidak Tepercaya yang mendominasi Pareto: kegunaan lebih tinggi di setiap tingkat keamanan.

### Mengapa Pengeditan Tepercaya menang

Pemantauan Tidak Tepercaya mengharuskan instance kedua U tidak dapat berkoordinasi dengan instance pertama. Kolusi dimungkinkan melalui isyarat steganografi atau training bersama sebelumnya. Protokol harus menginvestasikan anggaran dalam ketahanan kolusi (parafrase, perplexity) yang mengurangi sinyal monitor. Pengeditan Tepercaya menghindari kolusi sepenuhnya karena T dipercaya — setiap pengeditan yang diusulkan oleh T, berdasarkan asumsi, tidak bersifat permusuhan. Dampaknya adalah T kurang mampu, sehingga mungkin terlalu menolak atau melewatkan pintu belakang yang tidak kentara.

### Keempat sumbu protokol

1. Peran apa yang dimainkan U: produser saja, monitor, atau keduanya.
2. Peran apa yang dimainkan T: editor, monitor, atau resampler.
3. Bagaimana H dialokasikan: audit seragam, audit acak, atau audit yang diarahkan pada bendera.
4. Berapa banyak komputasi yang dikeluarkan protokol untuk ketahanan terhadap kolusi.

### Penyelarasan vs kontrol

- Alignment: "apakah kamu aman untuk dipercaya?" Lesson 2, 4, 5, 6, 7, 8, 9 semuanya membahas hal ini.
- Kontrol: "bisakah kita menggunakan U dengan aman tanpa memercayainya?" Lesson 10.

Penyelarasan adalah properti model. Kontrol adalah properti dari protokol penerapan. Pengendalian tidak menghasilkan model yang selaras; ini menghasilkan penggunaan model yang mungkin tidak selaras secara aman. Kedua agenda tersebut diperlukan.

### Tiga batasan mendasar

- Manusia tidak berskala. Anggaran H dibatasi oleh jumlah manusia yang bersedia mengaudit output AI yang sulit.
- Model lemah yang tepercaya pada akhirnya tertinggal. Ketika T jauh lebih lemah dari U, sinyal pengeditan T menjadi noise.
- Kontrol tidak selaras. Protokol yang berfungsi di ASL-3 (Lesson 18) mungkin tidak berfungsi di ASL-4 — jika U dapat memanipulasi T atau beralasan tentang distribusi audit H, pengendalian gagal.

### Perkembangan tahun 2025

AISI Inggris dan Redwood berkolaborasi dalam Kasus Keamanan Kontrol (metr.org / redwoodresearch.org 2025): kasus keamanan kontrol berpendapat bahwa dalam asumsi kasus terburuk tentang U, protokol membatasi kemungkinan hasil yang tidak dapat diterima. Kasus keselamatan (Lesson 18) mewarisi kerangka kendali.

### Cocok untuk Fase 18

Lesson 6-9 menggambarkan ancaman (penipuan mungkin terjadi). Lesson 10 menjelaskan paradigma defensif dengan asumsi bahwa ancaman itu nyata. Lesson 11 menjelaskan paradigma ofensif yang saling melengkapi (membuat kamu dapat dipercaya melalui pengawasan yang terukur).

## Pakai

`code/main.py` menyimulasikan empat protokol pada tugas pintu belakang mainan. U menghasilkan output; sebagian kecil berada di pintu belakang (kebijakan tim merah). T menyetujui atau mengedit. H dipanggil berdasarkan anggaran. kamu dapat menyapu anggaran dan parameter kolusi untuk mereproduksi garis depan Pareto di Greenblatt dkk. 2024 Gambar 2 (bentuk, bukan angka).

## Kirim

Lesson ini menghasilkan `outputs/skill-control-protocol-audit.md`. Mengingat deskripsi protokol penerapan, ini mengidentifikasi peran U/T/H, memberi nama vector kolusi, dan memperkirakan posisi protokol pada batas kegunaan-keamanan di bawah tekanan tim merah.

## Latihan

1. Jalankan `code/main.py`. Mereproduksi batasan kegunaan-keamanan untuk Pengeditan Tepercaya vs Pemantauan Tidak Tepercaya. Identifikasi rezim anggaran di mana asumsi kolusi Untrusted Monitoring menjadi sangat penting.

2. Tambahkan protokol baru: "T mengambil sample ulang dengan suhu rendah, lalu U mengedit output T." Ukur posisinya di perbatasan. Mengapa kegagalan ini lebih parah dibandingkan Pengeditan Tepercaya?

3. Makalah tahun 2024 mengasumsikan kamu dapat melihat protokolnya. Rancang sebuah protokol yang menyembunyikan dari U output mana yang sedang diaudit. Apa yang dibutuhkan protokol ini dari infrastruktur penerapan?4. Baca Greenblatt dkk. 2024 Pasal 7 (kolusi). Usulkan tindakan anti-kolusi dan jelaskan perkiraan biaya kegunaannya.

5. Mengartikulasikan kasus keamanan kontrol untuk penerapan yang menggunakan kelas U = GPT-4, kelas T = GPT-3.5, H = 1% dari output yang diaudit. Identifikasi asumsi terkuat yang mendasari kasus ini.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Kontrol AI | "keamanan di bawah subversi" | Desain protokol yang membatasi hasil yang tidak dapat diterima dalam kasus terburuk U |
| Model tidak tepercaya (U) | "model yang kuat" | Model mumpuni yang ingin kami ekstrak pekerjaannya tetapi tidak percaya |
| Model tepercaya (T) | "model yang lemah" | Model yang lebih lemah, yang diketahui tidak bersifat permusuhan, digunakan sebagai editor/monitor |
| Manusia terpercaya (H) | "anggaran audit" | Tenaga kerja ahli yang dibatasi; mahal; sinyal kebenaran dasar |
| Pengeditan Tepercaya | "T mengedit U" | Protokol dimana T adalah gerbang output U; H meminta output yang ambigu |
| Pemantauan Tidak Tepercaya | "Kamu memantau Kamu" | Protokol dimana instans U kedua mencetak output; tahan kolusi hanya dengan investasi ekstra |
| Kontrol kasus keamanan | "argumen formal" | Argumen tertulis bahwa protokol membatasi keberhasilan musuh dalam kasus terburuk U |

## Bacaan Lanjutan

- [Greenblatt, Shlegeris, Sachan, Roger — AI Control (arXiv:2312.06942)](https://arxiv.org/abs/2312.06942) — makalah kanonik ICML 2024
- [Redwood Research — Indeks makalah Kontrol AI terbaru](https://redwoodresearch.substack.com/) — tindak lanjut tahun 2024-2025
- [UK AISI + METR — Kontrol kasus keselamatan](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — operasionalisasi untuk laboratorium perbatasan
- [Hubinger dkk. — Agen Tidur (Lesson 7, arXiv:2401.05566)](https://arxiv.org/abs/2401.05566) — asumsi pengendalian model ancaman
