# Model Perutean sebagai Primitif Pengurangan Biaya

> Broker dinamis mengevaluasi setiap permintaan (jenis tugas, panjang token, kesamaan embedding, keyakinan) dan mengirimkan kueri sederhana ke model murah, meningkatkan kueri kompleks ke model frontier. Juga disebut model berjenjang. Studi kasus produksi menunjukkan pengurangan biaya sebesar 20-60% pada kualitas iso di seluruh penerapan di AS/Inggris/UE; peningkatan efisiensi perutean sebesar 30% pada SaaS volume tinggi menghasilkan penghematan tahunan sebesar enam digit. Konteks tahun 2026 adalah harga inference LLM turun ~10x per tahun — token kelas GPT-4 naik dari $20/M menjadi ~$0,40/M dari akhir tahun 2022 hingga 2026. Sebagian besar penurunan tersebut disebabkan oleh tumpukan layanan yang lebih baik (Fase 17 · 04-09), bukan perangkat keras. Perutean adalah cara kamu mengubah penurunan harga menjadi margin tanpa regresi produk. Mode kegagalannya adalah penyimpangan model murah: rute mendorong 40% ke model yang lebih lemah, kualitas turun 3-5% pada tugas-tugas penalaran, tidak ada yang memperhatikan selama seperempat. Gerbang rute berdasarkan metrik kualitas online, bukan hanya kumpulan evaluasi offline.

**Type:** Learn
**Language:** Python (stdlib, simulator router berjenjang mainan)
**Prerequisites:** Fase 17 · 01 (Platform LLM Terkelola), Fase 17 · 19 (AI Gateways)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Jelaskan model berjenjang: murah terlebih dahulu dengan pemeriksaan keyakinan, tingkatkan pada keyakinan rendah.
- Menghitung empat sinyal perutean (klasifikasi tugas, panjang prompt, embedding kesamaan dengan himpunan yang diketahui-sulit, kepercayaan diri dari lintasan pertama).
- Hitung biaya campuran yang diharapkan pada pembagian rute target dan toleransi loss kualitas.
- Sebutkan metrik pemantauan penyimpangan (gerbang kualitas online) yang menangkap creep model murah.

## Masalah

Biaya layanan kamu $80k/bulan pada GPT-5. Analisis kamu menunjukkan 70% pertanyaannya sederhana: "jam berapa sekarang di Paris?" "ulangi kalimat ini." Model kelas Haiku menanganinya dengan sempurna dengan biaya 3%. 30% memerlukan pemikiran GPT-5 — pengkodean, matematika, perencanaan multi-langkah.

Jika kamu mengarahkan 70% ke murah dan 30% ke mahal, tagihan kamu turun ~65% dengan kualitas produk yang sama. Ini adalah perutean. Triknya adalah membangun broker tanpa menurunkan kualitas.

## Konsep

### Empat sinyal perutean

1. **Klasifikasi tugas**: sederhana/kompleks/codegen/matematika/obrolan. Dapat berupa pengklasifikasi berbasis aturan, LLM kecil (kelas Haiku seharga $0,25/M), atau embed kemiripan dengan keranjang berlabel. Output: rute = murah / seimbang / terdepan.

2. **Panjang prompt**: prompt >4K token sering kali membutuhkan batas agar koherensi. Anjuran <500 token biasanya tidak.

3. **Menanamkan kemiripan dengan himpunan yang diketahui sulit**: jika kuerinya dekat (kosinus > 0,88) dengan keranjang yang diketahui sulit, langsung eskalasi ke perbatasan.

4. **Kepercayaan diri sejak first-pass**: kirim ke murah; jika masalah log model menunjukkan keyakinan rendah ATAU menolak ATAU mengeluarkan bahasa lindung nilai, coba lagi di perbatasan. Menambahkan latensi P95 pada ~10% lalu lintas tetapi menghemat 50%+ pada 90% lainnya.

### Tiga pola

**Pra-rute** (pengklasifikasi di depan): ~latensi 5-10 ms ditambahkan; tercepat secara keseluruhan.

**Cascade** (murah dulu, eskalasi dengan keyakinan rendah): ~1,2x latensi median (jalankan murah plus verifikasi), ~2x saat eskalasi. Lantai kualitas terbaik.

**Rute ansambel** (berjalan murah dan terdepan secara paralel untuk sample, pemilihan model hadiah): kualitas tertinggi, biaya tertinggi; gunakan hanya untuk A/B kritis.

### Implementasi

Gerbang AI (Fase 17 · 19) mengekspos perutean. LiteLLM memiliki konfigurasi `router` dengan fallback dan perutean biaya. Portkey memiliki penjaga + perutean. Kong AI Gateway memiliki perutean berbasis plugin. Pasar model OpenRouter memperlihatkan API rekomendasi.Sumber terbuka: RouteLLM (LMSYS), Bukan Diamond (komersial), Prompt Mule.

### Kurva harga tahun 2026

| Kelas model | Akhir 2022 | 2026 | Ubah |
|-------------|-----------|------|--------|
| Kualitas tingkat GPT-4 | ~$20/M | ~$0,40/M | 50x lebih murah |
| Perbatasan (GPT-5, Claude 4) | — | ~$3-10/M | tingkat baru |

Sebagian besar peningkatannya adalah efisiensi pelayanan — pembelajaran inti di Fase 17 · 04-09 berubah menjadi penurunan biaya di sisi penyedia layanan. Perutean memungkinkan kamu memperoleh keuntungan tersebut di layer aplikasi alih-alih menunggu semua pengguna bermigrasi ke tingkat murah.

### Melayang adalah risiko sebenarnya

Rute kamu mengirimkan 40% ke model murah. Selama enam bulan, pembagian tugas bergeser (pengguna menjadi lebih canggih, mengajukan pertanyaan lebih panjang). Router tidak menyadarinya karena pengklasifikasinya dilatih pada data Q1. Kualitas turun secara diam-diam. Tidak ada yang mengeluh cukup keras. kamu mengetahui dalam tolok ukur pesaing bahwa kamu kalah.

Rute gerbang berdasarkan metrik kualitas online:

- Pengguna jempol ke atas / jempol ke bawah per rute.
- Juri LLM otomatis pada sample yang dibagikan (5%) per rute.
- Tingkat eskalasi: jika kaskade naik pada rute >30%, model yang murah mengalami rute yang berlebihan.
- Tingkat penolakan per rute.

### Nomor yang harus kamu ingat

- Penghematan perutean pada tahun 2026 dengan kualitas iso: 20-60% studi kasus.
- Penurunan harga LLM 2022-2026: agregat ~10x per tahun.
- Tingkat GPT-4 2022 vs 2026: ~$20/M → ~$0,40/M.
- Dampak latensi bertingkat: ~1,2x median, ~2x meningkat (~10% lalu lintas).

## Pakai

`code/main.py` menyimulasikan pra-rute, kaskade, dan ansambel pada weight kerja campuran. Melaporkan biaya campuran, kehilangan kualitas, dan tingkat eskalasi.

## Kirim

Lesson ini menghasilkan `outputs/skill-router-plan.md`. Mengingat weight kerja dan anggaran kualitas, pilih pola perutean dan sinyal.

## Latihan

1. Jalankan `code/main.py`. Pada tingkat akurasi berapa cascade mengalahkan pra-rute?
2. Basis pengguna kamu adalah 30% perusahaan (kueri kompleks), 70% tingkat gratis (sederhana). Rancang pemisahan perutean. Metrik online apa yang menjadi gerbangnya?
3. Sebuah rute menurunkan kualitas sebesar 2% namun menghemat 40%. Apakah itu sebuah kapal? Tergantung pada produknya — bantah keduanya.
4. Terapkan pemeriksaan kepercayaan menggunakan logprob dari OpenAI/Anthropic API. Berapa ambang batas yang kamu gunakan untuk memulai?
5. Selama enam bulan, tingkat eskalasi meningkat dari 8% menjadi 22%. Diagnosis tiga penyebab dan perbaikan untuk masing-masing penyebab.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Perutean model | "broker biaya" | Pilihan model yang dinamis per permintaan |
| Kaskade model | "eskalasi murah-pertama" | Berlari dengan harga murah, gagal mencapai garis depan dengan kepercayaan diri yang rendah |
| Pra-rute | "klasifikasikan dulu" | Pengklasifikasi di depan; tidak ada jalankan ulang |
| Rute ansambel | "pilihan paralel" | Jalankan beberapa pilihan model hadiah terbaik |
| Tingkat eskalasi | "dicabut%" | Sebagian kecil permintaan berjenjang yang meningkat |
| RuteLLM | "Router LMSYS" | perpustakaan router OSS |
| Bukan Berlian | "router komersial" | Produk perutean model SaaS |
| Melayang | "orang murahan" | Pergeseran distribusi tanpa sepengetahuan router |
| Gerbang kualitas online | "cek langsung" | Lalu lintas langsung pengambilan sample juri LLM otomatis |

## Bacaan Lanjutan- [AbhyashSuchi — Praktik Terbaik Perutean Model LLM 2026](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Bangkitnya Optimization Inference 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [Kertas / code RouteLLM](https://github.com/lm-sys/RouteLLM)
- [Bukan Berlian — perutean model](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — gateway multi-model dengan routing primitif.
