# Kerangka Kesiapsiagaan OpenAI dan Kerangka Keamanan DeepMind Frontier

> OpenAI Preparedness Framework v2 (April 2025) memperkenalkan Kategori Penelitian — Otonomi Jangka Panjang, Sandbagging, Replikasi dan Adaptasi Otonom, Merusak Perlindungan — yang berbeda dari Kategori Terlacak. Kategori yang Dilacak memicu Laporan Kemampuan ditambah Laporan Perlindungan yang ditinjau oleh Kelompok Penasihat Keselamatan. FSF v3 DeepMind (September 2025, dengan Tingkat Kemampuan Terlacak ditambahkan pada 17 April 2026) melipatgandakan otonomi ke dalam domain Litbang dan Cyber ​​​​ML (otonomi Litbang ML tingkat 1 = sepenuhnya mengotomatisasi alur Litbang AI dengan biaya kompetitif vs alat manusia + AI). FSF v3 secara eksplisit mengatasi penyelarasan yang menipu melalui pemantauan otomatis untuk penyalahgunaan penalaran instrumental. Catatan jujurnya: Kategori Penelitian di PF v2 (termasuk Otonomi Jangka Panjang) tidak secara otomatis memicu mitigasi; bahasa kebijakannya adalah "potensi". DeepMind sendiri mengatakan pemantauan otomatis "tidak akan cukup dalam jangka panjang" jika alasan instrumental diperkuat.

**Type:** Learn
**Language:** Python (stdlib, alat diff tabel keputusan tiga kerangka)
**Prerequisites:** Fase 15 · 19 (RSP Antropis)
**Waktu:** ~45 menit

## Masalah

Lesson 19 membaca kebijakan penskalaan Anthropic dengan cermat. Lesson ini melengkapi gambarannya dengan membaca OpenAI dan DeepMind. Ketiga dokumen tersebut merupakan artefak sepupu yang menjawab pertanyaan yang sama – kapan laboratorium perbatasan harus menghentikan sementara atau membatasi model – dan ketiga dokumen tersebut menyatu dalam sekumpulan kecil kategori dan berbeda di tempat-tempat tertentu yang penting.

Konvergensi: ketiganya menyebut otonomi jangka panjang sebagai kelas kemampuan yang patut dilacak. Ketiganya mengakui perilaku menipu (penyelarasan palsu, karung pasir) sebagai kelompok risiko tertentu. Ketiganya memiliki badan peninjau internal. Perbedaannya: OpenAI membagi kategori menjadi "Terlacak" (mitigasi wajib) dan "Penelitian" (tidak ada pemicu otomatis). DeepMind membagi otonomi menjadi dua domain daripada menamainya secara terpisah. Nama labnya adalah Tracked vs Research, atau Critical vs Moderate, atau Tier-1 vs Tier-2; konsekuensi operasional di mana suatu kemampuan berada berbeda-beda di berbagai laboratorium.

Membacanya bersama-sama adalah latihan yang bermanfaat. Kemampuan yang sama dapat berupa "mitigasi wajib" di Anthropic, "dipantau tetapi tidak dipicu" di OpenAI, dan "dilacak dalam domain tertentu" di DeepMind. Postur kebijakan itu penting.

## Konsep

### Kerangka Kesiapsiagaan OpenAI v2 (April 2025)

Struktur:

- **Kategori yang Dilacak**: memicu Laporan Kemampuan (apa yang dapat dilakukan model) ditambah Laporan Perlindungan (mitigasi apa yang ada). Ditinjau oleh Kelompok Penasihat Keamanan sebelum penerapan.
- **Kategori Penelitian**: kemampuan model yang dilacak lab tetapi belum melakukan mitigasi tertentu. Mencakup Otonomi Jangka Panjang, Karung Pasir, Replikasi dan Adaptasi Otonom, Merusak Upaya Perlindungan.

Pembingkaian kunci di v2: Kategori Penelitian tidak secara otomatis memicu mitigasi. Bahasa kebijakannya adalah mitigasi yang “potensial”. Ini adalah delta spesifik yang harus diperhatikan: lokasi kategori di Tracked vs Research adalah titik temu antara "gated" dan "observed".

### DeepMind Frontier Safety Framework v3 (September 2025; Tingkat Kemampuan Terlacak ditambahkan 17 April 2026)

Struktur:- **Tingkat Kemampuan Kritis (CCL)**: ambang batas kemampuan di lima domain: Cyber, Bio, ML R&D, CBRN, Autonomy (digabungkan menjadi ML R&D dan Cyber).
- **Tingkat Kemampuan yang Dilacak**: perincian tambahan ditambahkan pada bulan April 2026. Contoh nyata: otonomi Litbang ML tingkat 1 = sepenuhnya mengotomatisasi alur Litbang AI dengan biaya kompetitif vs alat manusia + AI.
- **Pemantauan penyelarasan yang menipu**: komitmen eksplisit terhadap pemantauan otomatis untuk penyalahgunaan penalaran instrumental.

Kerangka otonomi berbeda dari OpenAI. DeepMind tidak menjadikan "Otonomi" sebagai domain tingkat atas; hal ini dimasukkan ke dalam domain di mana otonomi dapat menimbulkan loss (Litbang ML dan Cyber). Argumennya adalah otonomi tanpa domain adalah kemampuan tanpa risiko; argumen tandingannya adalah bahwa otonomi lintas domain merupakan risiko meta yang harus disebutkan dalam framework tersebut.

### Apa yang ketiganya sepakati

- Kelompok Penasihat Keamanan Internal (bernama Anthropic SAG, OpenAI SAG, komite internal DeepMind). Tinjau sebelum penerapan untuk model berkemampuan tinggi.
- Penyebutan secara eksplisit penyelarasan yang menipu / pemalsuan penyelarasan sebagai kelas risiko.
- Artefak berdiri pada irama yang dinyatakan (Anthropic: Frontier Safety Roadmap, Risk Report; OpenAI: Capabilities and Safeguards Reports; DeepMind: siklus pembaruan FSF).
- Pengakuan bahwa pertahanan yang hanya bersifat pemantauan saja mempunyai batas tertinggi. DeepMind secara eksplisit menyatakan: "pemantauan otomatis tidak akan cukup dalam jangka panjang."

### Dimana mereka berbeda

- **Antropik**: komitmen jeda dihapus di v3.0; Ambang batas AI R&D-4 adalah gerbang berikutnya.
- **OpenAI**: Pembagian Terlacak vs Penelitian; Kategori Penelitian (termasuk Otonomi Jangka Panjang) tidak otomatis masuk dalam gerbang.
- **DeepMind**: otonomi diterapkan ke domain lain; Tingkat Kemampuan yang Dilacak menambah rincian pada bulan April 2026.

### Sandbagging: kemampuan khusus yang memperumit ketiganya

Sandbagging (model yang secara strategis berkinerja buruk dalam evaluasi) ada dalam Kategori Penelitian OpenAI. RSP v3.0 Anthropic mengatasinya melalui kesenjangan konteks evaluasi (Lesson 1). DeepMind mengatasinya melalui pemantauan penyelarasan yang menipu di FSF v3.

Jika suatu model tidak melakukan evaluasi, ambang batas kemampuan setiap framework akan diremehkan. Kerangka kerja ini hanya berfungsi jika pengukuran berhasil. Inilah sebabnya mengapa pengukuran eksternal (Lesson 21, METR) dan evaluasi permusuhan diperlukan selain penilaian mandiri di laboratorium.

### Keterampilan membaca kebijakan

- Temukan: setiap kemampuan yang kamu minati harus dapat ditemukan dalam polis. Jika tidak dapat ditemukan, kebijakan tidak mencakup hal tersebut.
- Klasifikasi: apakah Dilacak (memicu mitigasi) atau Riset (dilacak tetapi tidak memicu)? OpenAI menamakannya; Anthropic dan DeepMind memiliki padanannya masing-masing.
- Irama: apakah kebijakan diperbarui pada jadwal yang dinyatakan, atau hanya setelah kejadian tertentu? Irama yang dinyatakan lebih kuat.
- Independensi: apakah tinjauan eksternal bersifat wajib atau opsional? Anthropic bermitra dengan Apollo dan Institut Keamanan AI AS; OpenAI dengan METR; DeepMind dengan SAG internal terutama.

## Pakai

`code/main.py` mengimplementasikan alat pembeda tabel keputusan kecil. Mengingat suatu kemampuan (otonomi, penyelarasan yang menipu, otomatisasi penelitian dan pengembangan, peningkatan dunia maya, dll.), kebijakan ini menunjukkan bagaimana ketiga kebijakan tersebut mengklasifikasikan kemampuan tersebut, dan mitigasi apa yang dipicu. Ini adalah alat bantu membaca, bukan alat kebijakan.

## Kirim

`outputs/skill-cross-policy-diff.md` menghasilkan perbandingan lintas kebijakan untuk kemampuan tertentu, dengan menggunakan tiga framework sebagai referensi.

## Latihan1. Jalankan `code/main.py`. Konfirmasikan output alat diff cocok dengan kebijakan untuk setidaknya dua kemampuan yang dapat kamu verifikasi terhadap dokumen sumber.

2. Baca OpenAI Preparedness Framework v2 secara lengkap. Identifikasi setiap Kategori Penelitian. Untuk masing-masing, tuliskan satu kalimat tentang mengapa hal tersebut ada di Riset, bukan Dilacak.

3. Baca DeepMind FSF v3 secara lengkap, ditambah pembaruan Tingkat Kemampuan Terlacak April 2026. Identifikasi kriteria evaluasi khusus otonomi Litbang ML tingkat 1. Bagaimana kamu mengukurnya secara eksternal?

4. Sandbagging ada dalam Kategori Penelitian OpenAI. Rancang evaluasi yang akan memaksa model karung pasir untuk mengungkapkan kemampuan sebenarnya. Referensi diskusi permainan eval-konteks Lesson 1.

5. Bandingkan ketiga kebijakan tersebut pada kemampuan tertentu (pilihan kamu). Sebutkan klasifikasi kebijakan mana yang menurut kamu paling ketat dan mana yang paling tidak ketat. Justifikasi dengan teks sumber.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| Kerangka Kesiapsiagaan | "Kebijakan penskalaan OpenAI" | PF v2 (April 2025); Kategori Terlacak vs Penelitian |
| Kategori yang Dilacak | "Mitigasi wajib" | Kemampuan Pemicu + Laporan Perlindungan; Ulasan SAG |
| Kategori Penelitian | "Hanya dipantau" | Terlacak tetapi tidak ada mitigasi otomatis; termasuk Otonomi Jangka Panjang |
| Kerangka Keamanan Perbatasan | "Kebijakan penskalaan DeepMind" | FSF v3 (September 2025) + Tingkat Kemampuan Terlacak (Apr 2026) |
| CCL | "Tingkat Kemampuan Kritis" | Ambang batas DeepMind per domain (Cyber, Bio, ML R&D, CBRN) |
| Otonomi Litbang ML tingkat 1 | "Otomasi penelitian dan pengembangan" | Mengotomatiskan sepenuhnya jalur penelitian dan pengembangan AI dengan biaya kompetitif |
| Karung Pasir | "Kinerja buruk yang strategis" | Model berkinerja buruk pada evaluasi; dalam Kategori Penelitian OpenAI |
| Penalaran instrumental | "Penalaran sarana-tujuan" | Penalaran tentang bagaimana mencapai tujuan; target pemantauan DeepMind |

## Bacaan Lanjutan

- [OpenAI — Memperbarui Kerangka Kesiapsiagaan kami](https://openai.com/index/updating-our-preparedness-framework/) — pengumuman v2.
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — dokumen lengkap.
- [DeepMind — Memperkuat Kerangka Keamanan Frontier kami](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — pengumuman FSF v3.
- [DeepMind — Memperbarui Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — Penambahan Tingkat Kemampuan Terlacak.
- [Laporan FSF Gemini 3 Pro](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — contoh Laporan Risiko berformat FSF.
