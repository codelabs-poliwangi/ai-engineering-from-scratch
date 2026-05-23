# Tolok Ukur Evaluasi dan Koordinasi

> Lima tolok ukur tahun 2025-2026 mencakup ruang evaluasi multi-agen. **MultiAgentBench / MARBLE** (ACL 2025, arXiv:2503.01935) mengevaluasi topologi star/chain/tree/graph dengan KPI pencapaian; **grafik paling baik untuk penelitian**, perencanaan kognitif menambahkan ~3% pencapaian pencapaian. **COMMA** mengevaluasi koordinasi informasi asimetris multimodal; model tercanggih termasuk GPT-4o berjuang untuk mengalahkan baseline acak. **MedAgentBoard** (arXiv:2505.12371) mencakup empat kategori tugas medis dan sering kali ditemukan bahwa multi-agen tidak mendominasi LLM tunggal. **AgentArch** (arXiv:2509.10769) mengukur arsitektur agen perusahaan yang menggabungkan penggunaan alat + memori + orkestrasi. **SWE-bench Pro** ([arXiv:2509.16941](https://arxiv.org/abs/2509.16941)) memiliki 1.865 masalah di 41 repo yang mencakup aplikasi bisnis, layanan B2B, dan alat pengembang; model frontier mendapat skor ~23% di Pro vs 70%+ di Terverifikasi — pemeriksaan realitas terhadap kontaminasi. Claude Opus 4.7 (April 2026) dilaporkan pada **64,3%** di Pro dengan koordinasi tim agen yang eksplisit (belum ada sumber utama Anthropic yang dipublikasikan — dianggap sebagai awal); Verdent (agen scaffold) mencapai **76,1% pass@1** pada Terverifikasi ([Laporan teknis Verdent](https://www.verdent.ai/blog/swe-bench-verified-technical-report)). **AAAI 2026 Bridge Program WMAC** (https://multiagents.org/2026/) adalah titik fokus komunitas tahun 2026. Lesson ini dibangun berdasarkan metrik MARBLE, menjalankan penyisiran topologi vs metrik, dan embed aturan "hanya melewati bangku SWE Terverifikasi bukanlah bukti generalisasi".

**Type:** Learn
**Language:** Python (stdlib)
**Prerequisites:** Phase 16 · 15 (Topologi Pemungutan Suara dan Debat), Phase 16 · 23 (Mode Kegagalan)
**Waktu:** ~75 menit

## Masalah

Ketika sebuah makalah mengklaim “sistem multi-agen kami lebih baik,” pertanyaannya adalah: lebih baik dari apa, dalam hal apa, diukur dengan cara apa? Era evaluasi multi-agen pada tahun 2023-2024 sangat kacau — setiap orang memilih metriknya sendiri, garis dasarnya sendiri, dan rangkaian tugasnya sendiri. Struktur acuan yang diberlakukan 2025-2026.

Tanpa tolok ukur bersama, kamu tidak dapat membandingkan dua sistem multi-agen secara bermakna. Yang lebih buruk lagi, tanpa adanya tolok ukur yang kuat, model-model terdepan bisa saja terkontaminasi. Bangku SWE Terverifikasi menjadi terkontaminasi sebagian di korpora training pada pertengahan tahun 2025; skor perbatasan meningkat; Pro dirancang sebagai pemeriksaan realitas yang tidak terkontaminasi.

Lesson ini menyebutkan lima tolok ukur kanonik tahun 2026, menyebutkan ukuran masing-masing tolok ukur, dan mengajarkan kamu untuk membaca klaim tolok ukur secara skeptis.

## Konsep

### MultiAgentBench (MARMER) — ACL 2025

arXiv:2503.01935. Mengevaluasi empat topologi koordinasi (bintang, rantai, pohon, grafik) pada tugas penelitian, pengkodean, dan perencanaan. KPI berbasis pencapaian melacak sebagian kemajuan, bukan hanya keberhasilan akhir.

Hasil yang diukur:

- **Grafik** topologi terbaik untuk skenario penelitian; mendukung kritik apa pun.
- **Rantai** terbaik untuk pengkodean penyempurnaan bertahap.
- **Bintang** terbaik untuk konsolidasi faktual cepat.
- **Pajak koordinasi** muncul melewati ~4 agen pada grafik.
- **Perencanaan kognitif** menambahkan ~3% pencapaian pencapaian di seluruh topologi.

Gunakan ketika: kamu ingin membandingkan topologi koordinasi apple-to-apples. Repo MARBLE (https://github.com/ulab-uiuc/MARBLE) menyediakan evaluator.

### COMMA — informasi asimetris multimodalMencakup tugas-tugas di mana agen memiliki modalitas observasi yang berbeda dan harus berkoordinasi tanpa berbagi informasi penuh. Hasil yang dilaporkan tidak menyenangkan: model frontier termasuk GPT-4o kesulitan untuk mengalahkan **dasar acak** pada kolaborasi agen-agen di COMMA. Sinyalnya adalah bahwa modalitas multi-agen kurang terlatih dan kurang dievaluasi — LLM menangani kerjasama modalitas tunggal secara wajar; koordinasi multi-modalitas runtuh.

Gunakan ketika: sistem kamu memiliki koordinasi informasi multimodal atau asimetris. Hasil nol dari COMMA merupakan peringatan untuk mengukur sebelum mengklaim.

### MedAgentBoard — uji stres domain

arXiv:2505.12371. Empat kategori tugas medis: diagnosis, perencanaan perawatan, pembuatan laporan, komunikasi pasien. Membandingkan sistem berbasis aturan multi-agen vs LLM tunggal vs konvensional.

Temuan: multi-agen TIDAK mendominasi LLM tunggal di sebagian besar kategori. Keuntungan multi-agennya sempit — penguraian tugas membantu ketika subtugas dapat dipisahkan dengan jelas (diagnosis + pengobatan); akan merugikan bila overhead koordinasi melebihi perolehan spesialisasi (pembuatan laporan).

Gunakan ketika: domain kamu memiliki baseline LLM tunggal yang jelas. Jika lesson MedAgentBoard digeneralisasikan, banyak sistem multi-agen yang diusulkan direkayasa secara berlebihan.

### AgentArch — arsitektur perusahaan

arXiv:2509.10769. Pengaturan perusahaan dengan penggunaan alat, memori, dan orkestrasi yang digabungkan menjadi satu. Benchmark mengisolasi kontribusi setiap layer: seberapa besar bantuan penambahan alat? Menambah memori? Menambahkan orkestrasi multi-agen?

Gunakan ketika: kamu merancang tumpukan agen perusahaan dan perlu membenarkan setiap layer. AgentArch membantu menghindari pembelian feature yang nilainya tidak dapat kamu ukur.

### SWE-bench Pro — pemeriksaan realitas

arXiv:2509.16941. Masalah 1865 di 41 repositori yang mencakup aplikasi bisnis, layanan B2B, dan alat pengembang. Dirancang agar **tidak terkontaminasi** dengan penghentian training berikutnya. Model Frontier mendapat skor ~23% di Pro vs 70%+ di Terverifikasi. Kesenjangan tersebut merupakan sinyal kontaminasi.

Skor April 2026:
- Claude Opus 4.7 di Pro: **64,3%** (dilaporkan dengan koordinasi tim agen yang eksplisit; belum ada sumber utama Anthropic yang dipublikasikan — dianggap sebagai awal).
- Verdent (agen scaffold) di Terverifikasi: **76.1% pass@1** ([laporan teknis](https://www.verdent.ai/blog/swe-bench-verified-technical-report)).
- Skor mentah terdepan di Pro tanpa perancah agen: ~23-35% ([makalah Pro SWE-bench](https://arxiv.org/abs/2509.16941)).

Kesimpulannya: "kami mengalahkan SWE-bench Verified" tidak lagi menjadi bukti kemampuan. Pro adalah tes gerbang saat ini. Perancah tim agen menghasilkan keuntungan terukur pada Pro (~30-40 titik delta), yang merupakan salah satu argumen empiris terkuat untuk koordinasi multi-agen pada tahun 2026.

### AAAI 2026 WMAC

Program Jembatan AAAI 2026 — Lokakarya Koordinasi Multi-Agen (https://multiagents.org/2026/). Titik fokus komunitas tahun 2026 untuk penelitian AI multi-agen. Makalah yang diterima dan proses lokakarya adalah tempat kanonik untuk mengevaluasi metode baru; tunduk pada klaim yang diterima WMAC atas pracetak arXiv untuk keputusan produksi.

### Baca klaim benchmark dengan skeptis — daftar periksa tahun 2026

Ketika seseorang mengklaim hasil multi-agen:1. **Benchmark yang mana, yang terbagi mana?** SWE-bench Terverifikasi vs Pro sangat berarti. Sejumlah laporan mengenai pemisahan yang salah tidak ada gunanya.
2. **Pemeriksaan kontaminasi.** Apakah tolok ukur dirilis setelah batas waktu training model? Jika tidak, obati dengan hati-hati.
3. **Perbandingan dasar.** Vs dasar LLM tunggal, vs acak, vs pekerjaan multi-agen sebelumnya. Bukan "vs versi sistem yang sama yang tidak disetel".
4. **Signifikansi statistik.** N percobaan, nilai p, interval kepercayaan. Model Frontier memiliki varian tinggi; single run menyesatkan.
5. **Keberagaman tugas.** Satu tugas atau banyak? Generalisasi penting untuk produksi.
6. **Pengungkapan biaya.** Token per tugas, jam dinding. Solusi 90% dengan biaya 20x lipat adalah keputusan bisnis, bukan klaim kemampuan.

### Apa yang tidak diukur dengan baik oleh tolok ukur mana pun

- **Koordinasi cakrawala panjang.** Hari interaksi jam dinding. Semua tolok ukur saat ini tidak mencukupi.
- **Ketahanan permusuhan.** Apa yang terjadi jika salah satu agen jahat atau disusupi?
- **Pergeseran dalam penerapan.** Tolok ukur bersifat statis; pergeseran distribusi produksi.
- **Kinerja yang dinormalisasi biaya.** Sebagian besar tolok ukur melaporkan akurasi mentah, bukan akurasi per dolar.

Membangun tolok ukur internal kamu sendiri untuk poros yang benar-benar kamu pedulikan seringkali merupakan langkah yang tepat.

## Build

`code/main.py` adalah panduan non-interaktif:

- Mensimulasikan 3 sistem multi-agen pada tugas mainan.
- Menghitung metrik pencapaian bergaya MARMER untuk masing-masingnya.
- Menjalankan pemeriksaan kontaminasi dengan menahan tugas dari set "training".
- Membandingkan dengan baseline acak secara eksplisit.
- Mencetak kartu skor klaim benchmark.

Jalankan:

```bash
python3 code/main.py
```

Output yang diharapkan: kartu skor sistem dengan akurasi mentah, pencapaian pencapaian, biaya per tugas, delta dasar vs-acak, dan catatan pemeriksaan kontaminasi.

## Pakai

`outputs/skill-benchmark-reader.md` membaca klaim tolok ukur multi-agen dan menerapkan daftar periksa pengawasan. Output: nilai dan peringatan.

## Kirim

Disiplin evaluasi produksi:

- **Build tolok ukur internal** yang mencerminkan distribusi produksi kamu yang sebenarnya. Tolok ukur publik memberi informasi namun tidak menggantikan.
- **Sertakan dasar acak** dalam setiap perbandingan. Jika kamu tidak dapat mengalahkan tugas koordinasi secara acak dengan selisih yang besar, maka tugas tersebut mungkin tidak tepat sasaran.
- **Laporkan biaya beserta keakuratannya.** Biaya token dan jam dinding. Tim operasi membutuhkan keduanya.
- **Membangun kembali tolok ukur setiap triwulan.** Pergeseran distribusi produksi; tolok ukur basi menyesatkan.
- **Hindari overfitting tolok ukur yang dipublikasikan.** Jika tim kamu mengoptimalkan secara khusus untuk nomor SWE-bench Pro, kamu akan mengalami kemunduran dalam produksi.

## Latihan

1. Jalankan `code/main.py`. Identifikasi mana dari tiga sistem simulasi yang memiliki biaya per pencapaian terbaik. Apakah ini cocok dengan sistem akurasi mentah tertinggi?
2. Baca MultiAgentBench (arXiv:2503.01935). Untuk domain tugas kamu sendiri, putuskan mana dari empat topologi yang MARBLE rekomendasikan. Justifikasi dari hasil makalah.
3. Baca makalah SWE-bench Pro. Apa yang secara khusus membuatnya tahan terhadap kontaminasi? Bisakah teknik yang sama diterapkan pada tolok ukur lain yang kamu pedulikan?
4. Baca temuan COMMA tentang koordinasi multimodal. Rancang tugas koordinasi multimoda sederhana yang dapat kamu tambahkan ke tolok ukur internal kamu. Apa yang dianggap sebagai sinyal yang berguna?
5. Terapkan daftar periksa klaim tolok ukur pada hasil judul salah satu makalah multi-agen terbaru. Berapa nilai yang akan kamu berikan untuk klaim tersebut?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| MARMER | "Bench MultiAgen" | ACL 2025; topologi bintang/rantai/pohon/grafik dengan KPI pencapaian. |
| KOMA | "Patokan multimoda" | Koordinasi informasi asimetris multimodal; model perbatasan berjuang vs acak. |
| Papan Agen Med | "Uji stres domain" | Empat kategori medis; sering kali ditemukan multi-agen tidak mendominasi LLM tunggal. |
| AgenArch | "Tolok ukur perusahaan" | Alat + memori + orkestrasi berlapis. |
| Bangku SWE Pro | "Tahan kontaminasi" | Soal 1865, 41 repo; ~23% vs 70%+ pada Terverifikasi (sinyal kontaminasi). |
| Pencapaian tonggak sejarah | "Kredit parsial" | Tolok ukur yang menghargai kemajuan, bukan hanya kesuksesan akhir. |
| Kontaminasi | "Tolok ukur bocor ke dalam training" | Pasca rilis, tolok ukur beralih ke korpora training; skor meningkat. |
| WMAC | "Program Jembatan AAAI 2026" | Lokakarya Koordinasi Multi-Agen; titik fokus masyarakat. |

## Bacaan Lanjutan

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — tolok ukur topologi dengan KPI pencapaian
- [Repositori MARMER](https://github.com/ulab-uiuc/MARBLE) — implementasi referensi
- [MedAgentBoard](https://arxiv.org/abs/2505.12371) — uji stres domain; multi-agen seringkali tidak mendominasi
- [AgentArch](https://arxiv.org/abs/2509.10769) — arsitektur agen perusahaan
- [Papan peringkat SWE-bench](https://www.swebench.com/) — Skor Terverifikasi dan Pro untuk model frontier
- [AAAI 2026 WMAC](https://multiagents.org/2026/) — titik fokus komunitas tahun 2026
