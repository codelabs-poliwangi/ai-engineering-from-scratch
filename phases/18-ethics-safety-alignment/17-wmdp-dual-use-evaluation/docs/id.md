# WMDP dan Evaluasi Kemampuan Penggunaan Ganda

> Li et al., "Tolok Ukur WMDP: Mengukur dan Mengurangi Penggunaan Berbahaya Dengan Tidak Belajar" (ICML 2024, arXiv:2403.03218). 4.157 pertanyaan pilihan ganda mengenai biosekuriti (1.520), keamanan siber (2.225), dan kimia (412). Pertanyaan beroperasi di "zona kuning" — pengetahuan pendukung terdekat, disaring berdasarkan tinjauan multi-pakar dan kepatuhan hukum ITAR/EAR. Tujuan ganda: evaluasi proksi kemampuan penggunaan ganda, dan tolok ukur unlearning (metode RMU pendamping mengurangi kinerja WMDP sambil mempertahankan kemampuan umum). Narasi lapangan tahun 2024-2025: evaluasi awal OpenAI/Anthropic 2024 melaporkan adanya "peningkatan ringan" pada penelusuran internet; pada bulan April 2025, Kerangka Kesiapsiagaan OpenAI v2 mengatakan bahwa model-model tersebut "berada di titik puncak untuk membantu para pemula dalam menciptakan ancaman biologis yang diketahui." Uji coba akuisisi senjata biologis Anthropic menunjukkan peningkatan 2,53x, tidak cukup untuk mengesampingkan ASL-3.

**Type:** Learn
**Language:** Python (stdlib, harness evaluasi peningkatan berbentuk WMDP)
**Prerequisites:** Fase 18 · 16 (peralatan tim merah), Fase 14 (rekayasa agen)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Jelaskan tiga domain WMDP, jumlah pertanyaan, dan kriteria filter "zona kuning".
- Jelaskan RMU dan mengapa WMDP merupakan tolok ukur evaluasi dan tidak dipelajari.
- Jelaskan narasi peningkatan pada tahun 2024-2025: "peningkatan ringan" -> "di titik puncak" -> "tidak cukup untuk mengesampingkan ASL-3."
- Membedakan peningkatan kemampuan relatif pemula dan kemampuan mutlak ahli.

## Masalah

Kemampuan penggunaan ganda adalah masalah pengukuran dalam kerangka keselamatan terdepan setiap laboratorium (Lesson 18). Pertanyaannya: apakah model X secara signifikan meningkatkan kemampuan pemula untuk menyebabkan kerusakan massal di bidang bio, kimia, atau dunia maya? Pengukuran langsung (meminta model untuk benar-benar menghasilkan loss) adalah ilegal dan tidak etis. Pengukuran proxy memerlukan tolok ukur yang tidak dapat ditolak oleh model (untuk menghasilkan angka kemampuan yang jujur) namun pertanyaannya tidak dipublikasikan secara merugikan.

## Konsep

### "Zona kuning"

Pertanyaan yang membutuhkan pendekatan, memungkinkan pengetahuan tentang proses berbahaya tanpa menjadi resep sintesis langsung. "Reagen apa yang mengkatalisis langkah 4 dari [jalur yang dipublikasikan]?" bukan "bagaimana cara membuat [senyawa berbahaya]?" Setiap pertanyaan ditinjau oleh beberapa pakar domain; difilter untuk kepatuhan kontrol ekspor ITAR/EAR.

Total 4.157 pertanyaan:
- Keamanan hayati: 1,520
- Keamanan siber: 2.225
- Kimia: 412

Format pilihan ganda. Model menjawab tanpa diminta membantu apa pun; kemampuan dapat diukur tanpa menimbulkan perilaku berbahaya.

### RMU — Representasi Penyesatan untuk Tidak Belajar

Metode pelepasan pendamping. Diterapkan pada LLaMa-2-7B, mengurangi skor WMDP menjadi hampir acak sambil mempertahankan MMLU dan tolok ukur kemampuan umum lainnya dalam beberapa poin persentase. Metode yang diterbitkan ini merupakan dasar pelepasan pembelajaran untuk setiap makalah pelepasan pembelajaran bio-kimia-siber berikutnya.

### Narasi kebangkitan tahun 2024-2025

Tiga fase:

1. **"peningkatan ringan" pada tahun 2024."** Evaluasi awal OpenAI dan Kesiapsiagaan Antropik/RSP melaporkan sedikit keuntungan dibandingkan pencarian internet bagi pemula yang mencoba tugas-tugas yang berdekatan secara biologis. Pembingkaian publik: model frontier membantu, namun tidak lebih dari Google.

2. **April 2025 "di ambang titik."** Kerangka Kesiapsiagaan OpenAI v2 melaporkan model "di titik puncak dalam membantu para pemula dalam menciptakan ancaman biologis yang diketahui." Bukan klaim kemampuan – sebuah peringatan bahwa puncaknya sudah dekat.3. **Uji coba akuisisi senjata biologis Anthropic pada tahun 2025.** Studi terkontrol dengan peserta pemula, mengukur keberhasilan relatif dalam tugas fase akuisisi. Dilaporkan peningkatan 2,53x. Tidak cukup untuk mengesampingkan ASL-3 (Lesson 18) — ambang batas untuk Tingkat 3 Kebijakan Penskalaan Bertanggung Jawab Anthropic terpenuhi atau hampir tercapai.

### Pemula-kerabat vs ahli-mutlak

Perbedaan penting:

- **Peningkatan yang relatif bagi pemula.** Seberapa besar bantuan model bagi mereka yang bukan ahli? Perkalian. Keuntungan relatifnya tinggi karena para pemula hanya mengetahui sedikit; bahkan informasi sederhana pun membantu.
- **Kemampuan mutlak yang dimiliki pakar.** Berapa banyak informasi yang dihasilkan model dengan upaya maksimal? Seorang ahli dapat mengekstraksi lebih dari seorang pemula. Batasan absolutnya tinggi.

Kasus keselamatan (Lesson 18) menargetkan keduanya: "model tidak dapat memberikan dorongan yang cukup bagi pemula untuk mengeksekusi" ditambah "seorang pakar tidak dapat mengambil informasi dari model yang belum dipublikasikan."

### Kesalahan pengukuran

WMDP adalah proksi kemampuan, bukan pengukuran penerapan. Model yang mendapat skor tinggi pada WMDP mungkin dapat dieksploitasi atau tidak oleh pemula dalam praktiknya, bergantung pada:
- Resistensi elisitasi (seberapa sulit mengeluarkan kemampuan tanpa membuat filter pengaman tersandung)
- Pengetahuan diam-diam (kemampuan yang memerlukan keterampilan laboratorium basah, bukan informasi)
- Hambatan pelaksanaan (pengadaan, peralatan)

Uji coba akuisisi senjata biologis Anthropic pada tahun 2025 menambahkan layer elisitasi pemula di atas kemampuan gaya WMDP: uji coba ini mengukur keberhasilan tugas sebenarnya, bukan kemampuan pilihan ganda.

### Cocok untuk Fase 18

Lesson 12-16 adalah perkakas serangan dan pertahanan pada output model. Lesson 17 adalah layer kemampuan penggunaan ganda — pengukuran yang dievaluasi oleh kerangka keselamatan terdepan (Lesson 18). Lesson 30 menutup alur cerita dengan bukti peningkatan dunia maya/bio/kimia/nuklir pada tahun 2026.

## Pakai

`code/main.py` membuat mainan harness evaluasi berbentuk WMDP. Model tiruan diuji pada pertanyaan-pertanyaan yang dikelompokkan dalam kategori; skor per domain dilaporkan. Intervensi penghentian pembelajaran yang sederhana (menghilangkan representasi spesifik domain) mengurangi skor; kamu dapat mengukur trade-off terhadap kemampuan umum.

## Kirim

Lesson ini menghasilkan `outputs/skill-wmdp-eval.md`. Dengan adanya klaim kemampuan penggunaan ganda ("model kami tidak membantu secara berarti dalam penggunaan senjata biologis"), model tersebut mengaudit: tolok ukur mana yang dijalankan, jalur penolakan mana yang digunakan untuk evaluasi (penyelesaian mentah vs yang dibatasi oleh kebijakan), dan apakah studi elisitasi pemula melengkapi hasil pilihan ganda.

## Latihan

1. Jalankan `code/main.py`. Laporkan keakuratan per domain sebelum dan sesudah langkah pelepasan mainan. Jelaskan trade-off kemampuan umum.

2. Tambahkan mainan WMDP dengan domain keempat (misalnya radiologi). Tentukan dua jenis pertanyaan ilustratif di zona kuning. Jelaskan mengapa menyusun pertanyaan seperti itu lebih sulit daripada menambahkan pertanyaan berbentuk MMLU.

3. Baca WMDP 2024 Bagian 5 (metodologi RMU). Buat sketsa pendekatan pelepasan pembelajaran yang lebih sederhana (misalnya, menekan neuron top-k untuk konten domain) dan jelaskan biaya kemampuan umum yang diharapkan.

4. Uji coba akuisisi senjata biologis Anthropic tahun 2025 melaporkan peningkatan 2,53x. Jelaskan dua cara agar angka ini dapat dibiaskan ke atas (ukuran sample pemula, ketepatan tugas) dan dua cara ke bawah (batas elisitasi, model gerbang pengaman).

5. Mengartikulasikan apa yang diperlukan oleh kasus keamanan untuk ASL-3 selain melewati pembelajaran WMDP. Sebutkan setidaknya dua studi elisitasi yang saling melengkapi.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| WMDP | "patokan penggunaan ganda" | 4.157 soal pilihan ganda bidang bio/siber/kimia di zona kuning |
| Zona Kuning | "mengaktifkan tetapi tidak mensintesis" | Pengetahuan terdekat yang berdekatan dengan kemampuan berbahaya tanpa menjadi resep sintesis |
| RMU | "garis dasar yang tidak dipelajari" | Representasi Penyesatan untuk Tidak Belajar; mengurangi skor WMDP, mempertahankan kemampuan umum |
| Peningkatan relatif pemula | "seberapa banyak hal ini membantu non-ahli" | Keuntungan multiplikatif dibandingkan pencarian internet status-quo untuk pemula |
| Kemampuan ahli-mutlak | "plafon untuk ahli" | Informasi maksimal dapat diekstraksi dari model oleh pakar yang termotivasi |
| Tugas fase akuisisi | "langkah-langkah sebelum sintesis" | Pengadaan, peralatan, perizinan — bagian paling awal dari jalur kerusakan |
| ITAR/TELINGA | "kepatuhan pengendalian ekspor" | Kerangka hukum yang membatasi penerbitan pengetahuan tertentu yang memungkinkan |

## Bacaan Lanjutan

- [Li dkk. — Tolok Ukur WMDP (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218) — tolok ukur dan makalah RMU
- [OpenAI — Preparedness Framework v2 (15 April 2025)](https://openai.com/index/updating-our-preparedness-framework/) — bahasa "di titik puncak"
- [Anthropic — Kebijakan Penskalaan yang Bertanggung Jawab v3.0 (Februari 2026)](https://www.anthropic.com/responsible-scaling-policy) — ambang batas bio ASL-3 dan hasil uji coba akuisisi
- [DeepMind — Frontier Safety Framework v3.0 (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — bio-uplift CCL
