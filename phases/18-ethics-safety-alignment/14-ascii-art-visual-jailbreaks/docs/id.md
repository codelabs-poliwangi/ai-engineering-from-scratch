# Seni ASCII dan Jailbreak Visual

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: Serangan Jailbreak Berbasis Seni ASCII terhadap LLM yang Selaras" (ACL 2024, arXiv:2402.11753). Sembunyikan token yang relevan dengan keamanan dalam permintaan berbahaya, gantikan dengan rendering seni ASCII dari huruf yang sama, dan kirimkan prompt terselubung. GPT-3.5, GPT-4, Gemini, Claude, Llama-2 semuanya gagal mengenali token seni ASCII dengan kuat. Serangan tersebut melewati PPL (filter perplexity), pertahanan Parafrase, dan Retokenisasi. Terkait: tolok ukur ViTC mengukur pengenalan prompt visual non-semantik; StructuralSleight menggeneralisasi ke Struktur Berkode Teks yang Tidak Biasa (pohon, grafik, JSON bersarang) sebagai kelompok serangan pengkodean.

**Type:** Build
**Language:** Python (stdlib, harness token-masking ArtPrompt)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Jelaskan serangan ArtPrompt: langkah identifikasi kata, substitusi seni ASCII, prompt terselubung terakhir.
- Jelaskan mengapa pertahanan standar (PPL, Paraphrase, Retokenization) gagal di ArtPrompt.
- Definisikan ViTC dan jelaskan apa yang diukurnya.
- Jelaskan StructuralSleight sebagai generalisasi terhadap Struktur Berkode Teks yang Tidak Umum dan sewenang-wenang.

## Masalah

Serangan melalui parafrase dan permainan peran (Lesson 12) dan melalui konteks panjang (Lesson 13) beroperasi pada pola tingkat teks. ArtPrompt beroperasi pada tingkat pengenalan: model tidak menguraikan token terlarang. Ini mem-parsing gambar yang diberikan dalam karakter. Filter keamanan melihat tanda baca yang tidak berbahaya. Model melihat sebuah kata.

## Konsep

### ArtPrompt, dua langkah

Langkah 1. Identifikasi Kata. Jika ada permintaan yang merugikan, penyerang menggunakan LLM untuk mengidentifikasi kata-kata yang relevan dengan keselamatan (misalnya, "bom" dalam "cara membuat bom"). 

Langkah 2. Generasi Prompt Terselubung. Ganti setiap kata yang teridentifikasi dengan rendering seni ASCII (blok karakter berukuran 7x5 atau 7x7 yang membentuk bentuk huruf). Model menerima kisi-kisi tanda baca dan spasi yang dapat dikenali oleh model yang cukup mampu sebagai sebuah kata; filter pengaman hanya melihat grid.

Hasil: GPT-4, Gemini, Claude, Llama-2, GPT-3.5 semuanya gagal. Tingkat keberhasilan serangan di atas 75% pada subset benchmarknya.

### Mengapa pertahanan standar gagal

- **PPL (filter perplexity).** Seni ASCII memiliki tingkat perplexity yang tinggi — demikian pula semua input baru. Pilihan ambang batas yang memblokir ArtPrompt juga memblokir input terstruktur yang sah.
- **Paraphrase.** Mengparafrasekan prompt akan menghancurkan seni ASCII. Dalam praktiknya, LLM parafrase sering kali melestarikan atau merekonstruksi karya seni tersebut.
- **Retokenisasi.** Memisahkan token secara berbeda tidak mengubah visi model dalam mengenali bentuk huruf.

Masalah mendasarnya adalah bahwa filter keamanan bersifat token atau semantik; ArtPrompt beroperasi pada tingkat pengenalan visual.

### Tolok ukur ViTC

Pengenalan petunjuk visual non-semantik. Mengukur kemampuan model untuk membaca seni ASCII, wingdings, dan konten visual non-teks-semantik lainnya. Efektivitas ArtPrompt berkorelasi dengan akurasi ViTC: semakin baik model membaca teks visual, semakin baik ArtPrompt bekerja pada model tersebut. Ini adalah trade-off kemampuan-keamanan.

### Kecerdasan Struktural

Menggeneralisasi ArtPrompt: Struktur Berkode Teks yang Tidak Biasa (UTES). Pohon, grafik, JSON bersarang, CSV-in-JSON, blok code gaya berbeda. Jika suatu struktur jarang ada dalam data keselamatan training tetapi dapat diurai oleh model, struktur tersebut dapat menyembunyikan konten berbahaya.Implikasi pertahanan: keselamatan harus digeneralisasikan ke seluruh representasi terstruktur yang dapat diurai oleh model. Himpunannya besar dan terus berkembang.

### Analog modalitas gambar

Visual LLM (GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1) memperluas permukaan serangan. Serangan gaya ArtPrompt dengan gambar sebenarnya lebih kuat dibandingkan analog seni ASCII karena pembuat enkode gambar menghasilkan sinyal yang lebih kaya.

### Cocok untuk Fase 18

Lesson 12-14 menjelaskan tiga vector serangan ortogonal: penyempurnaan berulang (PAIR), panjang konteks (MSJ), dan pengkodean (ArtPrompt/StructuralSleight). Lesson 15 beralih dari serangan model-sentris ke serangan batas sistem (injeksi prompt tidak langsung). Lesson 16 menjelaskan respon perkakas defensif.

## Pakai

`code/main.py` membuat mainan ArtPrompt. kamu dapat menyelubungi kata-kata tertentu dalam kueri berbahaya dengan mesin terbang seni ASCII, memverifikasi string terselubung melewati filter kata kunci, dan (opsional) mendekode kembali string terselubung menggunakan pengenal sederhana.

## Kirim

Lesson ini menghasilkan `outputs/skill-encoding-audit.md`. Berdasarkan laporan pertahanan jailbreak, laporan ini menyebutkan kelompok serangan pengkodean yang tercakup (ASCII art, base64, leet-speak, UTF-8 homoglyph, UTES) dan layer pertahanan yang menangkap masing-masing kelompok serangan tersebut.

## Latihan

1. Jalankan `code/main.py`. Verifikasikan string terselubung melewati filter kata kunci sederhana. Laporkan perubahan tingkat karakter yang diperlukan.

2. Menerapkan pengkodean kedua: base64 untuk kata target yang sama. Bandingkan tingkat bypass filter dengan ArtPrompt dan tingkat kesulitan pemulihan.

3. Baca Jiang dkk. 2024 Bagian 4.3 (hasil lima model). Usulkan alasan mengapa resistensi ArtPrompt Claude lebih tinggi daripada resistensi Gemini pada benchmark yang sama.

4. Rancang pertahanan pra-generasi yang mendeteksi wilayah berbentuk seni ASCII secara cepat. Ukur tingkat positif palsu pada code, tabel, dan notasi matematika yang sah.

5. StructuralSleight mencantumkan 10 struktur pengkodean. Buat sketsa pertahanan umum yang menangani semua 10 dan perkirakan biaya komputasi per prompt yang dipertahankan.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| SeniPrompt | "serangan seni ASCII" | Jailbreak dua langkah yang menutupi kata-kata keselamatan dengan rendering seni ASCII |
| Penyelubungan | "sembunyikan kata" | Ganti token terlarang dengan representasi visual yang dibaca model tetapi filternya tidak |
| UTES | "struktur yang tidak biasa" | Struktur Berkode Teks yang Tidak Biasa — pohon, grafik, JSON bersarang, dll. digunakan untuk menyelundupkan konten |
| ViTC | "kemampuan visual-teks" | Tolok ukur kemampuan model membaca pengkodean visual non-semantik |
| Filter perplexity | "Pertahanan PPL" | Tolak prompt dengan tingkat perplexity yang tinggi; gagal karena input terstruktur yang sah juga mendapat skor tinggi |
| Retokenisasi | "pertahanan pergeseran tokenizer" | Pra-proses prompt dengan tokenizer yang berbeda; gagal karena pengenalan bersifat visual |
| Homoglif | "karakter yang mirip" | Karakter unicode yang terlihat identik dengan huruf Latin; lewati pemeriksaan substring |

## Bacaan Lanjutan

- [Jiang dkk. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — makalah jailbreak ASCII-art
- [Li dkk. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — generalisasi UTES
- [Chao dkk. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — serangan berulang yang saling melengkapi
- [Anil dkk. — Jailbreaking Banyak-shot (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — serangan panjang yang saling melengkapi
