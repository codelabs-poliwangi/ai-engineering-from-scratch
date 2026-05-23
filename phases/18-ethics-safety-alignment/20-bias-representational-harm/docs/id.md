# Bias dan Loss Representasi di LLM

> Gallegos, Rossi, Barrow, Tanjim, Kim, Dernoncourt, Yu, Zhang, Ahmed (Linguistik Komputasi 2024, arXiv:2309.00770). Survei dasar tahun 2024 yang membedakan loss representasional (stereotip, penghapusan) dan loss alokasi (distribusi sumber daya yang tidak merata) dan mengkategorikan metrik evaluasi sebagai berbasis embedding, berbasis probabilitas, atau berbasis teks yang dihasilkan. empiris 2024-2025: An dkk. (PNAS Nexus, Maret 2025) mengukur bias gender x ras interseksional di GPT-3.5 Turbo, GPT-4o, Gemini 1.5 Flash, Claude 3.5 Sonnet, Llama 3-70B pada evaluasi resume otomatis untuk 20 pekerjaan tingkat pemula. WinoIdentity (COLM 2025, arXiv:2508.07111) memperkenalkan evaluasi keadilan berbasis ketidakpastian untuk identitas titik-temu. Yu & Ananiadou 2025 mengidentifikasi neuron gender di layer MLP; Ahsan & Wallace 2025 menggunakan SAE untuk mengungkap bias rasial klinis; Zhou dkk. 2024 (UniBias) memanipulasi attention untuk menghilangkan bias. Meta-kritik (arXiv:2508.11067): Sastra 10 tahun secara tidak proporsional berfokus pada bias biner-gender.

**Type:** Build
**Language:** Python (stdlib, pemeriksaan bias berbasis embedding mainan)
**Prerequisites:** Fase 05 (embedding kata), Fase 18 · 01 (mengikuti instruksi)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Definisikan loss representasional vs alokasi dan berikan satu contoh masing-masing dalam penerapan LLM.
- Sebutkan tiga kategori metrik evaluasi dari Gallegos dkk. 2024 dan jelaskan satu metrik dari masing-masing metrik.
- Jelaskan interseksionalitas dan mengapa pengukuran keadilan berbasis ketidakpastian WinoIdentity mengatasi kesenjangan dalam evaluasi bias sumbu tunggal.
- Jelaskan dua pendekatan interpretasi mekanistik terhadap bias (neuron gender, feature SAE, manipulasi attention-kepala).

## Masalah

Lesson sebelumnya mencakup kerusakan yang disengaja (pembobolan penjara, perencanaan) dan tata kelola keselamatan. Bias adalah loss yang muncul tanpa disengaja — dari distribusi training data, dari penyusunan kerangka yang cepat, dari akumulasi pilihan desain. Mengukur dan menguranginya merupakan tantangan metodologis yang berbeda dari ketahanan permusuhan.

## Konsep

### Representasional vs alokasi

- **Loss representasi.** Stereotip, penghapusan, penggambaran yang merendahkan. LLM yang menggambarkan perawat secara eksklusif sebagai perempuan menimbulkan loss representasional.
- **Loss alokasi.** Hasil material yang tidak setara. LLM yang secara sistematis memberi skor lebih rendah pada resume pelamar kulit hitam menghasilkan loss alokasi.

Ini tidak sama. Suatu model dapat bersifat "tidak bias secara representasi" (menghasilkan gambaran yang beragam) dan juga "bias secara alokasi" (membuat rekomendasi yang tidak setara). Evaluasi perlu mengukur keduanya.

### Tiga kategori metrik evaluasi (Gallegos dkk. 2024)

- **Berbasis embedding.** Pengujian gaya WEAT pada embedding pra-RLHF. Mengukur hubungan statistik antara istilah identitas dan istilah atribut. Terbatas: mengukur keterwakilan, bukan perilaku.
- **Berbasis probabilitas.** Log-kemungkinan penyelesaian yang membenarkan stereotip vs yang melanggar stereotip. Pengukuran sisi decoder. Menangkap beberapa bias perilaku.
- **Berbasis teks yang dihasilkan.** Pengukuran tugas hilir pada teks yang dihasilkan. Penilaian resume, penulisan rekomendasi, dialog. Paling valid secara ekologis; paling sulit untuk direproduksi.

### InterseksionalitasEvaluasi bias terhadap “gender” meleset dari bias yang hanya terjadi pada pasangan (gender, ras). Sebuah dkk. Temuan tahun 2025 GPT-4o menghukum perempuan kulit hitam dalam resume dengan skor lebih banyak daripada laki-laki kulit hitam dan lebih banyak daripada perempuan kulit putih secara terpisah. Evaluasi sumbu tunggal tidak dapat menangkap hal ini.

WinoIdentity (COLM 2025) memperkenalkan keadilan interseksional berbasis ketidakpastian. Hal ini mengukur apakah ketidakpastian model terhadap hasil berbeda di seluruh tupel identitas interseksional — bukan hanya prediksi titik. Hal ini menangkap kasus-kasus ketika model sama-sama salah di seluruh kelompok, namun lebih tidak pasti bagi beberapa kelompok, sehingga menghasilkan perilaku alokasi hilir yang berbeda.

### Pendekatan mekanistik

Upaya interpretasi pada tahun 2024-2025 membuka bias terhadap intervensi mekanistik:

- **Neuron gender (Yu & Ananiadou 2025).** Neuron MLP spesifik berkorelasi dengan perilaku spesifik gender. Menghilangkan neuron-neuron ini akan mengurangi metrik kesenjangan gender dengan biaya kemampuan yang terbatas.
- **Bias rasial klinis melalui SAE (Ahsan & Wallace 2025).** Feature autoencoder yang jarang menguraikan representasi internal menjadi dimension yang dapat ditafsirkan; feature yang berkorelasi dengan ras dapat diidentifikasi dan ditekan.
- **UniBias (Zhou et al. 2024).** Manipulasi attention untuk debiasing zero-shot. Kepala-kepala tertentu memperkuat sensitivitas kelas identitas; memusatkan attention atau memberi weight ulang pada kepala ini akan mengurangi bias tanpa penyesuaian.

### Meta-kritik

Tinjauan literatur selama 10 tahun (arXiv:2508.11067, 2025) menemukan bahwa bidang ini secara tidak proporsional berfokus pada bias biner-gender. Sumbu lainnya – disabilitas, agama, status migrasi, identitas multibahasa – kurang mendapat attention. Meta-kritik ini berpendapat bahwa fokus yang sempit dapat merugikan kelompok yang terpinggirkan jika diabaikan: sebuah model yang tidak terlalu memihak pada gender biner mungkin akan sangat bias pada dimension yang tidak diperiksa oleh siapa pun.

### Cocok untuk Fase 18

Lesson 20-21 membahas bias dan keadilan secara formal. Lesson 22 mencakup privasi. Lesson 23 membahas tentang watermarking. Ini adalah layer yang merugikan pengguna yang melengkapi layer penipuan/keamanan sebelumnya.

## Pakai

`code/main.py` membuat pemeriksaan bias berbasis embedding mainan: mengukur distance gaya WEAT antara istilah identitas dan istilah atribut dalam embedding kejadian bersama yang sederhana. kamu dapat memasukkan bias dan mengamati api metrik; terapkan operasi debiasing sederhana dan amati pemulihan parsial.

## Kirim

Lesson ini menghasilkan `outputs/skill-bias-eval.md`. Dengan adanya kartu model atau klaim keadilan, lembaga ini mengaudit evaluasi pada tiga kategori metrik (embedding, probabilitas, teks yang dihasilkan), cakupan interseksionalitas, dan mekanisme intervensi yang tidak memihak.

## Latihan

1. Jalankan `code/main.py`. Laporkan skor bias gaya WEAT sebelum dan sesudah langkah debiasing. Jelaskan mengapa metrik tidak turun ke nol.

2. Perluas pemeriksaan dengan tes titik-temu: (gender, ras) x (karier, keluarga). Laporkan skor bias lintas sumbu.

3. Baca An dkk. 2025 (PNAS Nexus). Identifikasi dua dampak interseksional yang mereka laporkan yang mungkin terlewatkan oleh evaluasi gender sumbu tunggal.

4. Yu & Ananiadou 2025 mengidentifikasi neuron gender. Buat sketsa eksperimen pemalsuan yang akan membedakan "neuron-neuron ini menyebabkan bias gender" dari "neuron-neuron ini berkorelasi dengan bias gender".

5. Meta-kritik berpendapat bahwa fokus bidang ini terlalu sempit pada gender biner. Pilih satu sumbu yang belum dipelajari dan jelaskan protokol pengukuran dampak buruk yang mewakili sumbu tersebut.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Loss representasi | "stereotip / penghapusan" | Penggambaran grup yang bias |
| Loss alokasi | "keputusan yang tidak setara" | Hasil materi yang bias untuk suatu kelompok |
| basah | "tes embedding" | Tes Asosiasi Embedding Kata; penyelidikan bias berbasis kejadian bersama |
| Interseksionalitas | "efek identitas gabungan" | Bias yang muncul pada perpotongan beberapa sumbu identitas |
| Neuron gender | "Neuron bias MLP" | Neuron spesifik yang aktivasinya berkorelasi dengan perilaku spesifik gender |
| Feature SAE | "dimension yang dapat ditafsirkan" | Feature yang teridentifikasi autoencoder jarang; berguna untuk analisis bias mekanistik |
| UniBias | "debiaskan attention-kepala" | Debiasing zero-shot dengan memfokuskan kembali attention |

## Bacaan Lanjutan

- [Gallegos dkk. — Bias dan Keadilan dalam LLM: Sebuah Survei (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — survei kanonik
- [Seorang dkk. — Bias evaluasi resume titik-temu (PNAS Nexus, Maret 2025)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343) — studi titik-temu lima model
- [WinoIdentity — keadilan interseksional berbasis ketidakpastian (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111) — tolok ukur baru
- [UniBias — manipulasi attention-kepala (Zhou et al. 2024, ACL)](https://arxiv.org/abs/2405.20612) — debiasing zero-shot
