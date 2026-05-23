# Feature LLM Pengujian A/B — Buku Pertumbuhan, Statsig, dan Masalah Getaran

> Pengujian A/B tradisional tidak dibuat untuk LLM non-deterministik. Perbedaan penting: evals menjawab "dapatkah model melakukan tugasnya?" Tes A/B menjawab "apakah pengguna peduli?" Keduanya diperlukan; pengiriman pada pemeriksaan getaran selesai. Apa yang harus diuji pada tahun 2026: rekayasa cepat (kata-kata), pemilihan model (GPT-4 vs GPT-3.5 vs OSS; akurasi vs biaya vs latensi), parameter pembangkitan (suhu, top-p). Kasus nyata: varian model hadiah chatbot menghasilkan +70% durasi percakapan dan +30% retensi; Eksperimen baris subjek AI Nextdoor menghasilkan RKT +1% setelah penyempurnaan fungsi hadiah; Khan Academy Khanmigo melakukan iterasi pada sumbu latensi vs akurasi matematika. Pemisahan platform: **Statsig** (diakuisisi oleh OpenAI seharga $1,1 miliar pada September 2025) — pengujian berurutan, CUPED, all-in-one. **Buku Pertumbuhan** — sumber terbuka, asli gudang, mesin Bayesian + frequentist + sekuensial, CUPED, pemeriksaan SRM, koreksi Benjamini-Hochberg + Bonferroni. kamu memilih berdasarkan preferensi gudang-SQL dan apakah "diakuisisi oleh OpenAI" penting bagi organisasi kamu.

**Type:** Learn
**Language:** Python (stdlib, simulator pengujian sekuensial mainan)
**Prerequisites:** Fase 17 · 13 (Kemampuan Observabilitas), Fase 17 · 20 (Penerapan Progresif)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Membedakan eval ("dapatkah model melakukan tugasnya") dari pengujian A/B ("apakah pengguna peduli").
- Hitung tiga sumbu yang dapat diuji (prompt, model, parameter) dan pilih metrik untuk masing-masing sumbu.
- Menjelaskan CUPED, pengujian sekuensial, dan koreksi perbandingan berganda Benjamini-Hochberg.
- Pilih Statsig atau GrowthBook berdasarkan postur gudang-SQL dan sikap akuisisi perusahaan.

## Masalah

kamu menyetel sendiri prompt sistem. Rasanya lebih baik. kamu mengirimkannya. Konversi berubah karena kebisingan. kamu menyalahkan metriknya. Atau kamu mengirimkan model baru dan konversinya tidak berubah — apakah modelnya mengalami penurunan atau perubahannya terlalu kecil untuk dideteksi? kamu tidak tahu, karena kamu mengirim tanpa A/B.

Evals menjawab apakah model dapat melakukan tugas pada himpunan berlabel. Mereka tidak menjawab apakah pengguna lebih menyukai keluarannya. Hanya eksperimen online terkontrol yang dapat menjawab hal tersebut, dan hanya jika eksperimen tersebut memiliki kekuatan yang cukup, mengontrol non-determinisme, dan mengoreksi beberapa perbandingan.

## Konsep

### Evaluasi vs pengujian A/B

**Evals** — offline, set berlabel, juri (rubrik atau LLM sebagai juri atau manusia). Jawaban: “Apakah keluarannya benar/bermanfaat/aman pada distribusi tetap ini?”

**Uji A/B** — online, pengguna langsung, diacak. Jawaban: "Apakah varian baru ini memindahkan metrik tingkat pengguna yang penting?"

Keduanya diperlukan. Evaluasi menangkap regresi sebelum paparan; A/B mengonfirmasi dampak produk setelahnya.

### Apa yang harus diuji

1. **Rekayasa cepat** — susunan kata, struktur cepat sistem, contoh. Metrik: keberhasilan tugas, retensi pengguna, biaya/permintaan.
2. **Pemilihan model** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS. Metrik: akurasi (tugas) + biaya/permintaan + latensi P99. Multi-tujuan.
3. **Parameter pembangkitan** — suhu, top-p, max_tokens. Metrik: spesifik tugas (keberagaman output vs determinisme).

### CUPED — pengurangan varians

Eksperimen Terkendali Menggunakan Data Pra-Eksperimen. Regresi varians sebelum periode sebelum membandingkan pasca-periode. Pengurangan varians tipikal: 30-70%. Ukuran sample efektif bertambah secara gratis.

Implementasi: implementasi Statsig dan GrowthBook.

### Pengujian berurutanA/B klasik mengasumsikan ukuran sample tetap. Tes berurutan ("intip-dan-putuskan") mengontrol tingkat positif palsu jika dilihat berulang kali. Prosedur sekuensial yang selalu valid (mSPRT, rangkaian kepercayaan Howard) memungkinkan kamu berhenti lebih awal pada pemenang yang jelas.

### Koreksi perbandingan ganda

Menjalankan 20 pengujian A/B dengan keyakinan 95% menghasilkan satu positif palsu secara kebetulan. Koreksi Bonferroni memperketat α per pengujian; Benjamini-Hochberg mengontrol tingkat penemuan palsu. GrowthBook mengimplementasikan keduanya.

### SRM — ketidakcocokan rasio sample

Hash penugasan mengacak pengguna ke varian. Jika pembagian 50/50 menghasilkan 47/53, ada sesuatu yang rusak — pemeriksaan SRM menandainya. Kedua platform menerapkan.

### Statsig vs Buku Pertumbuhan

**Statistik**:
- Diakuisisi oleh OpenAI seharga $1,1 miliar (September 2025). Dihosting, SaaS.
- Pengujian berurutan, CUPED, populasi yang diadakan.
- All-in-one: tanda feature + eksperimen + kemampuan observasi.
- Paling sesuai: tim sudah menginginkan produk yang dibundel, tidak peduli dengan kepemilikan OpenAI.

**Buku Pertumbuhan**:
- Sumber terbuka (MIT); warehouse-native (dibaca langsung dari Snowflake/BigQuery/Redshift).
- Beberapa mesin: Bayesian, frequentist, sequential.
- Koreksi CUPED, SRM, Bonferroni, BH.
- Host mandiri atau cloud terkelola.
- Paling cocok: toko gudang-SQL, tim data mengontrol layer metrik, menginginkan OSS.

### Non-determinisme mempersulit kekuasaan

Prompt yang sama menghasilkan output yang bervariasi. Perhitungan daya tradisional mengasumsikan observasi IID. Dengan non-determinisme LLM, ukuran sample efektif lebih rendah dari nominal. Kalikan ukuran sample yang diperlukan sebesar ~1,3-1,5x sebagai margin keamanan.

### Hasil kasus nyata

- Varian model hadiah Chatbot: +70% durasi percakapan, +30% retensi.
- Baris subjek berikutnya: +1% RKT setelah penyempurnaan fungsi hadiah.
- Khan Academy Khanmigo: tradeoff latensi-vs-akurasi matematika berulang.

### Anti-pola: pengiriman dengan getaran

Setiap teknisi senior dapat menyebutkan feature yang dikirimkan karena "terasa lebih baik" tanpa A/B. Kebanyakan dari mereka mengalami kemunduran pada metrik produk yang tidak diperhatikan oleh tim selama berbulan-bulan. A/B adalah fungsi pemaksaan.

### Nomor yang harus kamu ingat

- Statsig diakuisisi oleh OpenAI: $1,1 miliar, September 2025.
- GrowthBook: MIT sumber terbuka; Bayesian + frequentist + berurutan.
- Pengurangan varians CUPED: 30-70%.
- Non-determinisme LLM → +30-50% buffer ukuran sample.

## Pakai

`code/main.py` menyimulasikan pengujian A/B berurutan dengan batasan tetap dan berurutan. Menunjukkan bagaimana sekuensial memungkinkan kamu berhenti lebih awal.

## Kirim

Lesson ini menghasilkan `outputs/skill-ab-plan.md`. Mengingat perubahan feature, weight kerja, garis dasar, platform pengambilan, gerbang, ukuran sample.

## Latihan

1. Jalankan `code/main.py`. Untuk peningkatan yang diharapkan sebesar 5% dengan konversi dasar sebesar 3%, berapa ukuran sample hingga kekuatan 80%?
2. Pilih Statsig atau GrowthBook untuk pelanggan lokal yang diatur oleh layanan kesehatan.
3. Rancang A/B yang menguji GPT-4 vs GPT-3.5 berdasarkan biaya per tiket yang diselesaikan. Apa metrik utama, metrik pagar pembatas, dan metrik sekunder?
4. Canary kamu lolos tetapi A/B menunjukkan konversi -1,2%. Apakah kamu mengirim? Tulis kriteria eskalasi.
5. Terapkan CUPED pada pra-periode dengan 60% varians pasca. Hitung peningkatan ukuran sample efektif.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Evaluasi | "tes luring" | Evaluasi kemampuan model berlabel |
| Tes A/B | "percobaan" | Perbandingan acak langsung pada pengguna |
| DIKUMPULKAN | "pengurangan varians" | Regresi pra-periode untuk mengurangi varians |
| Tes berurutan | "tes mengintip-ok" | Prosedur yang selalu valid mengizinkan penghentian lebih awal |
| Perbandingan berganda | "kesalahan keluarga" | Menjalankan banyak tes meningkatkan hasil positif palsu |
| Bonferroni | "koreksi ketat" | Bagilah α dengan jumlah tes |
| Benyamini-Hochberg | "BH FDR" | Kontrol tingkat penemuan palsu, kurang konservatif |
| SRM | "perpecahan buruk" | ketidakcocokan rasio sample; bug tugas |
| Statistik | "Dimiliki OpenAI" | Komersial all-in-one, diakuisisi tahun 2025 |
| Buku Pertumbuhan | "yang OSS" | Platform asli gudang MIT |
| mSPRT | "uji rasio probabilitas berurutan" | Prosedur sekuensial klasik |

## Bacaan Lanjutan

- [GrowthBook — Cara Menguji AI A/B](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Melampaui Prompt: Optimization LLM Berbasis Data](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Perbandingan Statsig vs GrowthBook](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng dkk. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Urutan Keyakinan](https://arxiv.org/abs/1810.08240)
