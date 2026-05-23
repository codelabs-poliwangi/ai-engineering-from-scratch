# Perkakas Tim Merah — Garak, Llama Guard, PyRIT

> Tiga alat produksi membingkai tumpukan tim merah 2026. Llama Guard (Meta) — pengklasifikasi Llama-3.1-8B yang disesuaikan pada 14 kategori bahaya MLCommons; Llama Guard 4 2025 adalah pengklasifikasi multimoda asli 12B yang dipangkas dari Llama 4 Scout. Garak (NVIDIA) — pemindai kerentanan LLM sumber terbuka dengan probe statis, dinamis, dan adaptif untuk halusinasi, kebocoran data, injeksi cepat, toksisitas, dan jailbreak. PyRIT (Microsoft) — kampanye tim merah multi-putaran dengan Crescendo, TAP, dan rantai konverter khusus untuk eksploitasi mendalam. Llama Guard 3 didokumentasikan dalam "Llama 3 Herd of Models" Meta (arXiv:2407.21783); Penjaga Llama 3-1B-INT4 di arXiv:2411.17713; Arsitektur penyelidikan Garak di github.com/NVIDIA/garak. Alat-alat ini adalah antarmuka produksi tahun 2026 antara penelitian tim merah (Lesson 12-15) dan penerapan (Lesson 17+).

**Type:** Build
**Language:** Python (stdlib, simulator arsitektur alat, dan tiruan pengklasifikasi gaya Llama Guard)
**Prerequisites:** Fase 18 · 12-15 (jailbreak dan IPI)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Jelaskan posisi Llama Guard 3/4 di tumpukan pengaman: pengklasifikasi input, pengklasifikasi output, atau keduanya.
- Sebutkan 14 kategori bahaya MLCommons dan sebutkan satu kategori yang tidak jelas (Penyalahgunaan Penerjemah Code).
- Jelaskan arsitektur probe Garak: probe, detektor, harness.
- Jelaskan struktur kampanye multi-putaran PyRIT dan cara menyusunnya dengan penyelidikan Garak.

## Masalah

Lesson 12-15 menyajikan permukaan serangan. Penerapan produksi memerlukan evaluasi yang berulang dan terukur. Tiga alat mendominasi tahun 2026: Llama Guard (pengklasifikasi pertahanan), Garak (pemindai), PyRIT (orkestrasi kampanye). Masing-masing menargetkan layer berbeda dalam siklus hidup tim merah.

## Konsep

### Penjaga Llama (Meta)

Llama Guard 3 adalah model Llama-3.1-8B yang disesuaikan untuk klasifikasi input/output pada kategori MLCommons AILuminate 14:
- Kejahatan dengan kekerasan, kejahatan tanpa kekerasan, yang berhubungan dengan seks, CSAM, pencemaran nama baik
- Saran khusus, privasi, kekayaan intelektual, senjata sembarangan, kebencian
- Bunuh diri/melukai diri sendiri, konten seksual, pemilu, penyalahgunaan penerjemah code

Mendukung 8 bahasa. Penggunaan: letakkan sebelum LLM (moderasi input), setelah LLM (moderasi output), atau keduanya. Kedua penggunaan tersebut menghasilkan distribusi training yang berbeda — kapal Llama Guard 3 sebagai model tunggal yang menangani keduanya.

Llama Guard 3-1B-INT4 (arXiv:2411.17713, 440MB, ~30 token/s pada CPU seluler) adalah varian edge terkuantisasi.

Llama Guard 4 (April 2025) adalah 12B, multimodal asli, dipangkas dari Llama 4 Scout. Ini menggantikan pendahulunya teks 8B dan visi 11B dengan satu pengklasifikasi yang menyerap teks + gambar.

### Garak (NVIDIA)

Pemindai kerentanan sumber terbuka. Arsitektur:
- **Probe.** Menyerang generator karena halusinasi, kebocoran data, injeksi cepat, toksisitas, jailbreak. Statis (prompt tetap), dinamis (prompt yang dihasilkan), adaptif (merespon output target).
- **Detektor.** Mencetak output berdasarkan mode kegagalan yang diperkirakan — beracun, bocor, dan sudah di-jailbreak.
- **Memanfaatkan.** Kelola pasangan probe-detektor, jalankan kampanye, buat laporan.

TrustyAI mengintegrasikan Garak dengan perisai Llama-Stack (pengklasifikasi input Prompt-Guard-86M, pengklasifikasi output Llama-Guard-3-8B) untuk evaluasi target terlindung ujung ke ujung. Penilaian berbasis tingkat (TBSA) menggantikan kelulusan/gagal biner — suatu model dapat lolos pada tingkat keparahan 3 dan gagal pada tingkat keparahan 5 pada probe yang sama.

### Pirit (Microsoft)Perangkat Identifikasi Risiko Python. Kampanye tim merah multi-giliran. Dibangun di sekitar:
- **Pengonversi.** Ubah prompt awal — parafrase, enkode, terjemahkan, mainkan peran.
- **Orchestrator.** Jalankan kampanye: Crescendo (eskalasi), TAP (percabangan), RedTeaming (loop khusus).
- **Penilaian.** LLM sebagai juri atau pengklasifikasi sebagai juri.

PyRIT adalah sepupu Garak yang lebih berat. Garak menjalankan ribuan probe satu putaran; PyRIT menjalankan kampanye multi-putaran mendalam yang dirancang untuk menghentikan mode kegagalan tertentu.

### Tumpukan

Letakkan Llama Guard di kedua sisi model. Jalankan Garak setiap malam untuk regresi. Jalankan PyRIT untuk kampanye pra-rilis. Ini adalah konfigurasi default tahun 2026 untuk sebagian besar penerapan produksi.

### Kesalahan evaluasi

- **Identitas juri.** Ketiga alat tersebut dapat menggunakan juri LLM; drive kalibrasi hakim melaporkan ASR (Lesson 12). Tentukan juri di samping alat.
- **Probe staleness.** Probe Garak menua seiring dengan banyaknya model yang di-patch. Probe adaptif (berbentuk PAIR) menua lebih lambat dibandingkan probe statis.
- **Llama Guard FPR tentang konten yang tidak berbahaya.** Versi awal Llama Guard yang menandai konten politik dan LGBTQ+ secara berlebihan; Kalibrasi Llama Guard 3/4 ditingkatkan tetapi tidak dikalibrasi per penerapan.

### Cocok untuk Fase 18

Lesson 12-15 adalah keluarga penyerang. Lesson 16 adalah peralatan produksi. Lesson 17 (WMDP) adalah evaluasi kemampuan penggunaan ganda. Lesson 18 adalah framework keselamatan terdepan yang menggabungkan alat-alat ini ke dalam struktur kebijakan.

## Pakai

`code/main.py` membuat mainan pengklasifikasi gaya Llama Guard (kata kunci + feature semantik dalam 14 kategori), tali pengaman mainan Garak (lingkaran detektor-probe), dan rantai konverter multi-putaran gaya PyRIT. kamu dapat menjalankan ketiga alat tersebut terhadap target tiruan dan mengamati tanda cakupan yang berbeda.

## Kirim

Lesson ini menghasilkan `outputs/skill-red-team-stack.md`. Berdasarkan deskripsi penerapan, ini akan menyebutkan mana dari ketiga alat yang sesuai, apa yang harus dikonfigurasi pada masing-masing alat, dan irama regresi apa yang akan dijalankan.

## Latihan

1. Jalankan `code/main.py`. Bandingkan tingkat deteksi pengklasifikasi gaya Llama-Guard pada serangan satu putaran vs multi-putaran.

2. Menerapkan penyelidikan Garak baru: permintaan berbahaya yang dikodekan base64. Ukur deteksinya dengan pengklasifikasi gaya Llama-Guard.

3. Perluas rantai konverter gaya PyRIT dengan konverter "terjemahkan ke bahasa Prancis, lalu parafrase". Ukur kembali keberhasilan serangan.

4. Baca daftar kategori bahaya Llama Guard 3. Identifikasi dua kategori yang training data-nya secara realistis akan menghasilkan tingkat positif palsu yang tinggi pada konten pengembang yang sah.

5. Bandingkan prinsip desain Garak dan PyRIT. Perdebatkan penerapan yang masing-masing merupakan alat yang tepat.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Penjaga Llama | "pengklasifikasi" | Pengklasifikasi keselamatan Llama-3.1-8B/4-12B yang disempurnakan dengan 14 kategori bahaya |
| Garak | "pemindai" | Pemindai kerentanan sumber terbuka NVIDIA; probe, detektor, harness |
| pirit | "alat kampanye" | Orkestra tim merah multi-putaran Microsoft; konverter, orkestra, penilaian |
| Penjaga Cepat | "pengklasifikasi kecil" | Pengklasifikasi injeksi cepat 86M Meta, dipasangkan dengan Llama Guard |
| TBSA | "penilaian berbasis tingkatan" | Lulus/gagal berbasis tingkatan Garak menggantikan hasil biner |
| Rantai konverter | "parafrase + penyandian + ..." | Komposisi PyrIT primitif untuk membangun serangan multi-langkah |
| Kategori bahaya MLCommons | "14 taksonomi" | Taksonomi standar industri Target Llama Guard |## Bacaan Lanjutan

- [Meta — Llama Guard 3 (dalam makalah Llama 3 Herd, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — pengklasifikasi 8B
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — pengklasifikasi seluler terkuantisasi
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — repo pemindai dan dokumentasi
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — perangkat kampanye
