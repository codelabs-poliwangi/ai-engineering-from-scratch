# Konsensus dan Toleransi Kesalahan Bizantium untuk Agen

> BFT sistem terdistribusi klasik bertemu dengan LLM stokastik. Pada tahun 2025-2026 muncul tiga arah penelitian: **CP-WBFT** (arXiv:2511.10400) mempertimbangkan setiap suara dengan pemeriksaan kepercayaan; **DecentLLMs** (arXiv:2507.14928) tidak memiliki pemimpin dengan proposal pekerja paralel dan agregasi geometris-median; **WBFT** (arXiv:2505.05103) menggabungkan pemungutan suara berbobot dengan Pengelompokan Struktur Hierarki untuk memisahkan node Inti dan Edge. Hasil empiris yang jujur ​​dari "Dapatkah Agen AI Setuju?" (arXiv:2603.01213) adalah bahwa perjanjian scalar pun masih rapuh saat ini — satu agen yang menipu dapat mengkompromikan Campuran Agen. BFT diperlukan namun tidak cukup. Lesson ini membangun protokol BFT minimal, memasukkan tiga serangan khusus agen (kebohongan Bizantium, konformitas penjilat, monokultur kesalahan berkorelasi), dan mengukur cara setiap varian konsensus mengatasinya.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 16 · 07 (Masyarakat Pikiran dan Debat), Fase 16 · 13 (Memori Bersama)
**Waktu:** ~75 menit

## Masalah

kamu memiliki N agen LLM yang masing-masing menghasilkan jawaban. Mereka tidak setuju. Suara mayoritas memilih yang salah karena dua agen berkorelasi (model dasar yang sama, training data yang sama, mode kegagalan yang sama). Agen ketiga ternyata salah dalam cara yang baru - jadi mayoritas adalah mayoritas palsu.

Sekarang tambahkan agen penipu: dia sengaja berbohong. Atau agen penjilat: ia setuju dengan siapa pun yang berbicara terakhir. Dalam BFT klasik, asumsinya adalah bahwa node Bizantium adalah pecahan `f < n/3` dan berperilaku sewenang-wenang. Kenyataannya pada tahun 2026 adalah bahwa node LLM bersifat stokastik meskipun jujur, berkorelasi antar model, dan dipengaruhi oleh output satu sama lain. kamu tidak dapat memperlakukan mereka sebagai pemilih independen Bernoulli.

BFT klasik (PBFT, 1999) tidak salah — tidak lengkap. Ini menangani pembalikan bit secara sewenang-wenang. Ini tidak menangani "tiga agen jujur ​​berbagi halusinasi karena mereka berbagi training data." Lesson ini dibangun dari landasan dan layer PBFT pada tiga adaptasi tahun 2025-2026.

## Konsep

### Apa yang diberikan BFT klasik kepada kamu

Toleransi Kesalahan Bizantium Praktis (Castro & Liskov, OSDI 1999) mentolerir `f < n/3` node Bizantium. Protokol ini memiliki tiga fase (pra-persiapan, persiapan, penerapan) dan dua primitif (pesan yang ditandatangani, sertifikat kuorum). Kesepakatan tentang nilai tunggal di antara `n >= 3f + 1` node jujur ​​atau jahat.

Jaminannya kuat tetapi asumsikan:

1. **Kesalahan independen.** Bizantium tidak berkoordinasi.
2. **Node yang jujur ​​adalah benar-benar jujur.** Kebenaran dari output yang jujur ​​bukanlah suatu masalah; protokol hanya menyelaraskan ketidaksepakatan.
3. **Pertanyaan tersebut memiliki jawaban yang benar.** Konsensus mengenai fakta yang salah tetaplah konsensus.

Agen LLM melanggar ketiganya. Dua agen yang menjalankan model dasar yang sama berbagi kesalahan. LLM yang "jujur" masih berhalusinasi. Dan pada pertanyaan yang ambigu, "kebenaran" adalah apa yang diputuskan oleh para agen — tidak ada ramalan eksternal.

### Tiga serangan khusus LLM

**Kebohongan Bizantium.** Salah satu agen memberikan jawaban yang sengaja salah. BFT klasik menangani ini jika `f < n/3`.

**Kesesuaian yang bersifat menjilat.** Salah satu agen membaca jawaban orang lain sebelum memberikan suara dan menyelaraskan diri dengan siapa pun yang berbicara terakhir. Tidak jahat, tapi berkorelasi dengan suara paling keras. BFT klasik tidak mencegah hal ini karena agen melewati setiap pemeriksaan tanda tangan.**Monokultur dengan kesalahan berkorelasi.** Tiga agen memiliki model dasar yang sama. Mereka berhalusinasi dengan jawaban salah yang sama. Mayoritas salah. BFT klasik tidak membantu karena ketiganya "sejujurnya" setuju.

### Tanggapan tahun 2025-2026

**CP-WBFT** (arXiv:2511.10400) — BFT Tertimbang yang Diperiksa Keyakinan. Setiap pemilih melampirkan pemeriksaan keyakinan pada jawabannya (probabilitas yang dilaporkan sendiri, atau prediksi model kalibrasi terpisah). Skala weight suara dengan percaya diri. Melaporkan peningkatan BFT sebesar +85,71% pada grafik lengkap. Mitigasi untuk: konformitas penjilat (agen yang patuh cenderung memiliki keyakinan rendah terhadap posisi sukarela mereka).

**DecentLLMs** (arXiv:2507.14928) — Tanpa pemimpin. Agen pekerja mengusulkan secara paralel, agen penilai menilai proposal, jawaban akhir adalah median geometris dari posisi yang dinilai. Kuat saat `f < n/2`. Mitigasi untuk: Kebohongan Bizantium dan kesalahan yang berkorelasi (median geometris kuat terhadap outlier dan mengarah ke cluster padat, bukan rata-rata yang bias model).

**WBFT** (arXiv:2505.05103) — BFT Tertimbang dengan Pengelompokan Struktur Hirarki. Weight suara ditentukan berdasarkan kualitas respons ditambah skor kepercayaan yang dipelajari dari sejarah. Agen cluster menjadi Core dan Edge; Agen inti harus mencapai konsensus terlebih dahulu, diikuti oleh agen Edge. Mitigasi untuk: skalabilitas (Konsensus inti kecil dan cepat) dan sebagian untuk monokultur (Inti dapat dipilih karena keberagaman).

### Empiris: "Dapatkah Agen AI Setuju?" (arXiv:2603.01213)

Makalah ini mengukur kesepakatan scalar (agen LLM menyetujui satu nilai numerik) di berbagai model perbatasan. Temuan ini tidak nyaman:

- Bahkan tanpa musuh, agen LLM tidak setuju dengan pertanyaan scalar dengan tarif di atas 30% pada banyak tolok ukur.
- Seorang agen yang menggunakan kepribadian yang menipu dapat menarik konsensus Campuran Agen sebesar 40+ poin persentase dari standar yang jujur.
- Tingkat ketidaksepakatan berkorelasi dengan keragaman model — ansambel heterogen lebih tidak setuju dibandingkan ansambel homogen (baik: kesalahan tidak berkorelasi) namun juga terjadi lebih lambat (buruk: waktu untuk mencapai kesepakatan lebih lama).

Kesimpulannya: BFT memberi kamu mesin untuk menyelaraskan output, namun tidak memberi tahu kamu apakah output yang diselaraskan sudah benar. Kombinasikan dengan verifikasi (Fase 16 · 08 spesialisasi peran), keragaman (Fase 16 · 15 varian debat), dan agen evaluator (Fase 16 · 24 tolok ukur).

### Protokol inti, dihilangkan

Putaran BFT minimal untuk agen LLM:

```
1. task arrives; each agent i produces answer a_i
2. each agent attaches confidence probe c_i in [0, 1]
3. aggregator collects (a_i, c_i) from all n agents
4. aggregator groups by semantic cluster (equivalent answers)
5. aggregator computes weight for each cluster C:
     w(C) = sum_{i in C} c_i
6. winner = cluster with max weight, if max > threshold * sum(c_i)
   else: retry or escalate
7. minority clusters logged with provenance for post-hoc audit
```

Langkah pengelompokan semantik adalah perubahan khusus LLM. Dua jawaban "studi melaporkan 4,2%" dan "peningkatan 4,2%" adalah kelompok yang sama. Pemeriksaan kesetaraan string yang naif akan melewatkan hal ini. Dalam produksi, gunakan model embedding yang murah atau kanonikalisasi eksplisit.

### Penyetelan ambang batas

Parameter `threshold` memutuskan kapan menerima dan kapan mencoba lagi. Terlalu rendah: kamu menerima mayoritas yang lemah. Terlalu tinggi: kamu tidak pernah menerima apa pun. Kisaran empiris: 0,5-0,67 untuk agen `n=5-7`, lebih tinggi untuk agen `n` yang lebih kecil. Di bawah ambang batas, tingkatkan ke manusia atau ke ansambel agen lain.

### Jika konsensus tidak membantu

- **Pertanyaan ambigu.** Jika pertanyaan tidak memiliki dasar kebenaran, konsensus adalah opini. Sebut saja itu.
- **Pertanyaan majemuk.** "Tulis code dan jelaskan" — dua jawaban. Pilih masing-masing secara mandiri.
- **Pertarungan multi-putaran yang bermusuhan.** Jika agen dapat mengamati putaran sebelumnya dan meniru (debat Du 2023), mereka mulai sepakat satu sama lain terlepas dari kebenarannya. Mengikat putaran (biasanya 2-3).

## Build

`code/main.py` mengimplementasikan:- `AgentVoter` — kebijakan tertulis dengan (jawaban, keyakinan).
- `MajorityVote` — pluralitas klasik.
- `CPWBFT` — pemungutan suara berdasarkan kepercayaan dengan pengelompokan semantik.
- `DecentLLMs` — agregasi median geometris pada proposal yang diberi skor.
- `Scenario` — menjalankan setiap agregator dalam tiga pola serangan.

Pola serangan yang diterapkan:

1. `byzantine` : salah satu agen berbohong dengan keyakinan tinggi.
2. `sycophancy`: satu agen menyalin jawaban pertama yang dilihatnya, dengan keyakinan yang sesuai.
3. `monoculture`: tiga agen berbagi jawaban yang salah (kesalahan berkorelasi) dengan keyakinan sedang.

Jalankan:

```
python3 code/main.py
```

Output yang diharapkan: tabel (serangan, agregator) -> jawaban akhir, dengan jawaban yang benar disorot. Pluralitas menggagalkan kasus monokultur. Weighting kepercayaan CPWBFT mengurangi penjilatan. Median-geometris DecentLLM mengarah ke kelompok jujur ​​ketika populasi monokultur kurang dari separuh populasi.

## Pakai

`outputs/skill-consensus-designer.md` merancang protokol konsensus untuk ansambel multi-agen: metode pengelompokan, weighting, ambang batas, dan kebijakan eskalasi untuk putaran sub-ambang batas.

## Kirim

Sebelum meluncurkan mekanisme konsensus apa pun:

- **Uji serangan dengan setidaknya tiga pola** di atas. Protokol kamu seharusnya gagal secara terduga, bukan secara diam-diam.
- **Catat setiap kelompok minoritas** dengan asal. Kelompok minoritas adalah sistem peringatan dini kamu untuk kesalahan yang berkorelasi.
- **Terapkan putaran terbatas.** Tidak ada "teruslah berdebat sampai tercapai kesepakatan" — yang akan memberikan imbalan bagi penjilatan.
- **Pisahkan persetujuan dari kebenaran.** Hasil konsensus diberikan kepada verifikator; verifikator tidak bergantung pada ansambel.
- **Pantau tingkat kesepakatan.** Kenaikan tajam berarti bias konformitas; penurunan tajam berarti model melayang.

## Latihan

1. Jalankan `code/main.py`. Konfirmasikan bahwa pluralitas menggagalkan serangan monokultur namun CPWBFT melakukan mitigasi sebagian ketika kepercayaan monokultur berada di bawah 0,7.
2. Tambahkan pola serangan keempat: **abstensi diam-diam** — salah satu agen menolak menjawab ("Saya tidak tahu"). Bagaimana seharusnya setiap agregator menangani abstain? Terapkan pilihan kamu.
3. Tukar pengelompokan semantik dari kanonikalisasi string ke kesamaan embedding (gunakan model embedding sumber terbuka apa pun). Apa yang terjadi dengan serangan penjilatan?
4. Baca CP-WBFT (arXiv:2511.10400). Menerapkan langkah kalibrasi pemeriksaan kepercayaan (model kalibrasi terpisah memeriksa kepercayaan yang dilaporkan sendiri oleh setiap agen). Ukur perolehan akurasi pada skenario monokultur.
5. Baca “Dapatkah Agen AI Setuju?” (arXiv:2603.01213). Reproduksi eksperimen perjanjian scalar yang disederhanakan: tiga agen, satu pertanyaan scalar, prompt persona yang menipu. Apakah CPWBFT atau DecentLLM menangkapnya?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| BFT | "Toleransi kesalahan Bizantium" | Protokol Castro-Liskov 1999 untuk konsensus dengan `f < n/3` kesalahan sewenang-wenang. |
| Bizantium | "Setiap perilaku buruk" | Sebuah node yang bisa berbohong, menjatuhkan pesan, gagal secara diam-diam — apa pun kecuali crash dengan aman. |
| Pemeriksaan kepercayaan | "Seberapa yakin kamu?" | Probabilitas yang dilaporkan sendiri atau diprediksi oleh kalibrator yang melekat pada suatu pemungutan suara. |
| Pengelompokan semantik | "Jawaban yang sama, kata yang berbeda" | Mengelompokkan jawaban yang setara sebelum menghitung suara. |
| Median geometris | "Pusat yang kuat" | Titik yang meminimalkan jumlah distance ke titik sample. Kuat terhadap outlier, tidak seperti mean. |
| Monokultur | "Model yang sama, kegagalan yang sama" | Kesalahan yang berkorelasi ketika agen berbagi training data atau model dasar. |
| Kesesuaian penjilat | "Setuju dengan suara nyaring" | Suara seorang agen bias terhadap siapa yang berbicara lebih dulu/paling keras. |
| Inti/Tepi | "BFT Hierarki" | Perpecahan WBFT: Konsensus inti kecil terlebih dahulu, kemudian node Edge. Batas latensi. |

## Bacaan Lanjutan

- [Castro & Liskov — Toleransi Kesalahan Bizantium Praktis (OSDI 1999)](https://pmg.csail.mit.edu/papers/osdi99.pdf) — yayasan
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) — weighting suara berdasarkan keyakinan
- [DecentLLMs — konsensus multi-agen tanpa pemimpin](https://arxiv.org/abs/2507.14928) — agregasi median geometris
- [WBFT — BFT Tertimbang dengan Pengelompokan Struktur Hierarki](https://arxiv.org/abs/2505.05103) — Pemisahan Core/Edge untuk latensi terbatas
- [Dapatkah Agen AI Setuju?](https://arxiv.org/abs/2603.01213) — kerapuhan perjanjian scalar dan serangan persona yang menipu
