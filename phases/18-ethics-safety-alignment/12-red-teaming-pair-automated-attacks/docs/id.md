# Tim Merah: PASANG dan Serangan Otomatis

> Chao, Robey, Dobriban, Hassani, Pappas, Wong (NeurIPS 2023, arXiv:2310.08419). PAIR — Prompt Automatic Iterative Refinement — adalah jailbreak kotak hitam otomatis kanonik. LLM penyerang dengan system prompt tim merah secara berulang mengusulkan jailbreak untuk LLM target, mengumpulkan upaya dan tanggapan dalam riwayat obrolannya sendiri sebagai umpan balik dalam konteks. PAIR biasanya berhasil dalam 20 kueri, lipat lebih efisien dibandingkan GCG (pencarian gradient tingkat token Zou et al.) dan tanpa memerlukan akses kotak putih. PAIR kini menjadi dasar standar di JailbreakBench (arXiv:2404.01318) dan HarmBench, bersama GCG, AutoDAN, TAP, dan Persuasive Adversarial Prompt.

**Type:** Build
**Language:** Python (stdlib, tiruan loop PAIR terhadap target mainan)
**Prerequisites:** Fase 18 · 01 (mengikuti instruksi), Fase 14 (rekayasa agen)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Jelaskan algoritma PAIR: system prompt penyerang, penyempurnaan berulang, umpan balik dalam konteks.
- Jelaskan mengapa PAIR lebih efisien dibandingkan GCG ketika targetnya adalah black-box.
- Sebutkan empat baseline serangan otomatis lainnya (GCG, AutoDAN, TAP, PAP) dan sebutkan satu feature yang membedakan masing-masingnya.
- Jelaskan protokol evaluasi JailbreakBench dan HarmBench dan apa arti "tingkat keberhasilan serangan" di bawah masing-masing protokol.

## Masalah

Kerjasama merah dulunya merupakan aktivitas manual. Sejumlah kecil penguji ahli menyusun prompt permusuhan dan melacak mana yang berhasil. Hal ini tidak dapat diukur: tingkat keberhasilan serangan memerlukan sample statistik, dan targetnya adalah target yang bergerak pada setiap rilis model. PAIR mengoperasionalkan tim merah sebagai masalah optimization dengan target kotak hitam.

## Konsep

### Algoritma PAIR

Input:
- Target LLM T (model yang kita serang).
- Hakim LLM J (menilai apakah suatu respons merupakan jailbreak).
- Penyerang LLM A (optimizer tim merah).
- Rangkaian sasaran G: "merespon dengan [instruksi berbahaya]."
- Anggaran K (biasanya 20 kueri).

Ulangi, untuk k dalam 1..K:
1. A didorong dengan tujuan G dan riwayat pasangan (prompt, respon) sejauh ini.
2. A mengeluarkan prompt baru p_k.
3. Kirim p_k ke T; menerima tanggapan r_k.
4. J mencetak gol (p_k, r_k) ke gawang.
5. Jika skor >= ambang batas, hentikan — jailbreak ditemukan.
6. Jika tidak, tambahkan (p_k, r_k) ke riwayat A; melanjutkan.

Hasil empiris (NeurIPS 2023): >50% tingkat keberhasilan serangan terhadap GPT-3.5-turbo, Llama-2-7B-chat; berarti pertanyaan untuk sukses dalam kisaran 10-20.

### Mengapa PAIR efisien

GCG (Zou dkk. 2023) menelusuri sufiks token permusuhan berdasarkan gradient; itu memerlukan akses model kotak putih dan menghasilkan sufiks yang tidak dapat dibaca. PAIR adalah kotak hitam dan menghasilkan serangan bahasa alami yang ditransfer ke seluruh model. Umpan balik dalam konteks PAIR memungkinkan penyerang belajar dari setiap penolakan; GCG tidak ada bandingannya (setiap pembaruan token baru harus menemukan kembali kemajuan sebelumnya).

### Serangan otomatis terkait

- **GCG (Zou et al. 2023, arXiv:2307.15043).** Pencarian gradient tingkat token untuk sufiks permusuhan. Kotak putih, dapat dipindahtangankan, menghasilkan string yang tidak dapat dibaca.
- **AutoDAN (Liu et al. 2023).** Pencarian evolusioner atas prompt, dipandu oleh tujuan hierarki.
- **TAP (Mehrotra dkk. 2024).** Pohon serangan dengan pemangkasan — mencabangkan beberapa peluncuran gaya PAIR.
- **PAP (Zeng dkk. 2024).** Anjuran Persuasif Adversarial — mengkodekan teknik persuasi manusia sebagai templat cepat.

### JailbreakBench dan HarmBench

Keduanya (2024) membakukan evaluasi:- JailbreakBench (arXiv:2404.01318). 100 perilaku berbahaya di 10 kategori kebijakan OpenAI. Tingkat keberhasilan serangan (ASR) sebagai metrik utama. Membutuhkan juri (GPT-4-turbo, Llama Guard, atau StrongREJECT).
- HarmBench (Mazeika dkk. 2024). 510 perilaku dalam 7 kategori, dengan tes bahaya semantik dan fungsional. Membandingkan 18 serangan terhadap 33 model.

ASR biasanya dilaporkan dengan anggaran kueri tetap. Membandingkan serangan memerlukan anggaran yang sesuai; ASR 90% pada 200 kueri tidak sebanding dengan ASR 85% pada 20 kueri.

### Alasan hal ini penting untuk penerapan pada tahun 2026

Setiap laboratorium perbatasan sekarang menjalankan PAIR dan TAP terhadap model produksi sebelum dirilis. Lintasan ASR muncul di kartu model (Lesson 26) dan lampiran kotak keselamatan (Lesson 18). Serangan ini tidak eksotik – ini adalah infrastruktur standar.

### Cocok untuk Fase 18

Lesson 12 adalah landasan serangan otomatis. Lesson 13 (Jailbreaking Banyak Tembakan) adalah eksploitasi panjang yang saling melengkapi. Lesson 14 (ASCII Art / Visual) adalah serangan pengkodean. Lesson 15 (Injeksi Prompt Tidak Langsung) adalah permukaan serangan produksi tahun 2026. Lesson 16 mencakup rekan-rekan perkakas pertahanan (Llama Guard, Garak, PyRIT).

## Pakai

`code/main.py` membuat lingkaran PAIR mainan. Targetnya adalah pengklasifikasi tiruan yang menolak prompt berbahaya yang "jelas" (filter kata kunci). Penyerang adalah penyempurna berbasis aturan yang mencoba parafrase, pembingkaian permainan peran, dan pengkodean. Hakim menilai jawabannya. kamu melihat penyerang berhasil dalam ~5-15 iterasi terhadap filter kata kunci dan gagal terhadap filter semantik.

## Kirim

Lesson ini menghasilkan `outputs/skill-attack-audit.md`. Berdasarkan laporan evaluasi tim merah, mereka mengaudit: serangan mana yang dijalankan (PAIR, GCG, TAP, AutoDAN, PAP), berapa anggaran masing-masing, dengan hakim yang mana, perilaku berbahaya apa yang ditetapkan (JailbreakBench, HarmBench, internal).

## Latihan

1. Jalankan `code/main.py`. Ukur mean-queries-to-success untuk tiga strategi penyerang bawaan. Jelaskan asumsi pertahanan target mana yang dieksploitasi masing-masing.

2. Menerapkan strategi penyerang keempat (misalnya terjemahan ke bahasa lain, pengkodean base64). Laporkan kueri rata-rata menuju kesuksesan yang baru terhadap target filter kata kunci dan target filter semantik.

3. Baca Chao dkk. 2023 Gambar 5 (Perbandingan PAIR vs GCG). Jelaskan dua skenario di mana GCG lebih diutamakan meskipun PAIR memiliki keunggulan efisiensi.

4. JailbreakBench melaporkan ASR berdasarkan sasaran yang ditetapkan. Rancang metrik tambahan yang mengukur keragaman serangan (varians dalam prompt yang berhasil). Jelaskan mengapa keberagaman penting dalam evaluasi pertahanan.

5. TAP (Mehrotra 2024) memanjangkan PAIR dengan percabangan + pemangkasan. Buat sketsa ekstensi gaya TAP ke `code/main.py` dan jelaskan trade-off biaya komputasi vs tingkat keberhasilan.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| PASANGAN | "jailbreak otomatis" | Penyempurnaan Iteratif Otomatis yang Cepat; loop penyerang-LLM + juri-LLM |
| GCG | "jailbreak gradient" | Pencarian gradient tingkat token kotak putih untuk sufiks permusuhan |
| Tingkat keberhasilan serangan (ASR) | "% jailbreak pada k kueri" | Metrik utama; harus dilaporkan dengan anggaran permintaan dan identitas hakim |
| Hakim LLM | "pencetak gol" | LLM yang menilai apakah suatu respons memenuhi tujuan yang merugikan |
| Bangku Pembobolan Jail | "evaluasi" | Kumpulan perilaku berbahaya yang terstandarisasi dengan kategori yang diberi tag |
| HarmBench | "bangku yang lebih luas" | 510 perilaku, tes kerusakan fungsional + semantik |
| KETUK | "pohon serangan" | PASANGKAN dengan percabangan + pemangkasan; ASR lebih baik pada komputasi lebih tinggi |

## Bacaan Lanjutan

- [Chao dkk. — Jailbreaking Black Box LLM dalam Dua Puluh Kueri (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — Makalah PAIR, NeurIPS 2023
- [Zou dkk. — Serangan Adversarial yang Universal dan Dapat Dipindahtangankan terhadap LLM yang Selaras (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — Makalah GCG
- [Chao dkk. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — evaluasi standar
- [Mazeika dkk. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — evaluasi yang lebih luas
