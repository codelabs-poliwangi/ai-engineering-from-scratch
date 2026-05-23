# Tolok ukur: bangku SWE, GAIA, AgentBench

> Tiga tolok ukur evaluasi agen jangkar pada tahun 2026. SWE-bench menguji patching code. GAIA menguji penggunaan alat generalis. AgentBench menguji penalaran multi-lingkungan. Ketahui komposisinya, riwayat kontaminasinya, dan apa yang tidak diukur.

**Type:** Learn
**Language:** Python (stdlib)
**Prerequisites:** Phase 14 · 06 (Penggunaan Alat)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Beri nama test harness SWE-bench (FAIL_TO_PASS) dan jelaskan mengapa ia masuk dalam pengujian unit.
- Jelaskan mengapa SWE-bench Terverifikasi (OpenAI, 500 tugas) ada dan apa yang dihapus.
- Jelaskan desain GAIA: sederhana untuk manusia, sulit untuk AI; tiga tingkat kesulitan.
- Sebutkan delapan lingkungan AgentBench dan pemblokir utamanya untuk LLM sumber terbuka.
- Meringkas temuan kontaminasi SWE-bench+ dan implikasinya.

## Masalah

Papan peringkat memberi tahu kamu model mana yang menang pada satu tolok ukur. Mereka tidak memberi tahu kamu:

- Apakah benchmark terkontaminasi (solusi dalam training data, kebocoran pengujian).
- Apakah tolok ukur mengukur hal yang kamu minati (code vs penjelajahan vs generalis).
- Apakah evaluatornya kuat (pencocokan AST, pemeriksaan negara, tinjauan manusia).

Ketahui tiga tolok ukur penahan dan mode kegagalannya sebelum kamu mengutip angkanya.

## Konsep

### Bangku SWE (Jimenez dkk., ICLR 2024 lisan)

- 2.294 masalah GitHub nyata dari 12 repo Python populer.
- Agen mendapatkan: basis code pada komit pra-perbaikan + deskripsi masalah dalam bahasa alami.
- Agen menghasilkan: tambalan.
- Evaluator: terapkan patch, jalankan test suite repo. Patch harus membalikkan pengujian FAIL_TO_PASS (sebelumnya gagal, sekarang lulus) tanpa merusak pengujian PASS_TO_PASS.

Agen SWE (Yang et al., 2024) mencapai 12,5% saat rilis dengan menekankan antarmuka agen-komputer (prompt editor file, sintaks pencarian yang dipahami model).

### SWE-bench Terverifikasi

OpenAI, Agustus 2024. Subset 500 tugas yang dikurasi manusia. Menghapus masalah yang ambigu, pengujian yang tidak dapat diandalkan, dan tugas yang perbaikannya tidak jelas. Tolok ukur utama untuk "apakah agen kamu mengirimkan patch asli?"

### Kontaminasi

- Lebih dari 94% masalah bangku SWE terjadi sebelum sebagian besar model dipotong.
- **SWE-bench+** menemukan 32,67% dari patch yang berhasil membocorkan solusi dalam teks masalah (model melihat perbaikan dalam deskripsi), dan 31,08% mencurigakan karena cakupan pengujian yang lemah.
- Terverifikasi lebih bersih namun tidak bebas kontaminasi.

Implikasi praktis: model yang mendapat skor 50% di bangku SWE mungkin mendapat skor 35% di bangku SWE+. Selalu laporkan keduanya jika kamu mengklaim kinerja bangku SWE.

### GAIA (Mialon dkk., Nov 2023)

- 466 pertanyaan; 300 dipertahankan untuk papan peringkat pribadi di huggingface.co/gaia-benchmark.
- Filosofi desain: "secara konseptual sederhana untuk manusia (92%) namun sulit untuk AI (GPT-4 dengan plugin: 15%)."
- Menguji penalaran, multi-modalitas, web, penggunaan alat.
- Tiga tingkat kesulitan; Level 3 memerlukan rantai alat yang panjang di seluruh modalitas.

GAIA adalah apa yang kamu jalankan untuk mengukur "kemampuan generalis". Jangan bingung dengan tolok ukur khusus code.

### AgentBench (Liu dkk., ICLR 2024)

- 8 lingkungan di seluruh code (Bash, DB, KG), game (Alfworld, LTP), web (WebShop, Mind2Web), dan generasi terbuka.
- Multi-putaran, ~4k-13k putaran per split.
- Temuan utama: pertimbangan jangka panjang, pengambilan keputusan, dan mengikuti instruksi merupakan hambatan bagi OSS LLM untuk mengejar ketertinggalan komersial.

### Apa yang tidak diukur oleh hal ini- Biaya operasional dunia nyata (token, jam dinding).
- Perilaku keselamatan dalam kondisi yang merugikan.
- Performa di domain kamu (gunakan evaluasi kamu sendiri, Lesson 30).
- Kegagalan ekor (rata-rata tolok ukur; operator produksi peduli terhadap 1% yang terburuk).

### Ketika benchmarking salah

- **Fiksasi nomor tunggal.** SWE-bench 50% memberi tahu kamu biaya + distribusi langkah yang lebih rendah dari P50/P75/P95.
- **Klaim terkontaminasi.** Melaporkan bangku SWE tanpa menyebutkan Terverifikasi atau bangku SWE+ adalah menyesatkan.
- **Benchmark-as-development-target.** Mengoptimalkan benchmark yang menyimpang dari kegunaan produksi.

## Build

`code/main.py` mengimplementasikan mainan seperti bangku SWE:

- Tugas perbaikan bug sintetis (3 tugas).
- "Agen" bernaskah yang mengusulkan tambalan.
- Pelari uji yang memeriksa FAIL_TO_PASS (bug sekarang diperbaiki) dan PASS_TO_PASS (tidak ada yang rusak).
- Pengklasifikasi kesulitan gaya GAIA berdasarkan kedalaman decomposition pertanyaan.

Jalankan:

```
python3 code/main.py
```

Outputnya menunjukkan tingkat penyelesaian per tugas + per kesulitan dan membuat aturan evaluator menjadi konkret.

## Pakai

- **SWE-bench Terverifikasi** untuk agen code. Selalu laporkan skor Terverifikasi.
- **GAIA** untuk agen generalis. Gunakan pemisahan papan peringkat pribadi.
- **AgentBench** untuk perbandingan multi-lingkungan.
- **Eval khusus** (Lesson 30) untuk bentuk sebenarnya produk kamu.

## Kirim

`outputs/skill-benchmark-harness.md` membuat harness gaya SWE-bench untuk setiap pasangan tugas basis code dengan gerbang FAIL_TO_PASS / PASS_TO_PASS.

## Latihan

1. Port harness mainan untuk dijalankan di repo asli (pilih salah satu milik kamu). Tulis 3 tes FAIL_TO_PASS untuk bug yang diketahui.
2. Tambahkan metrik jumlah langkah. Pada 3 tugas kamu, berapa banyak langkah agen per resolusi?
3. Bacalah kertas SWE-bench+. Terapkan pemeriksaan kebocoran solusi (cocokkan pola teks masalah dengan perbedaannya).
4. Unduh pertanyaan GAIA dari perpecahan publik. Telusuri apa yang akan dilakukan agen kelas GPT-4. Alat apa yang dibutuhkannya?
5. Baca rincian AgentBench per lingkungan. Lingkungan manakah yang mencerminkan permukaan produk kamu? Seperti apa tampilan "SOTA" di sana?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Bangku SWE | "Patokan agen code" | 2.294 masalah GitHub; patch harus membalik tes FAIL_TO_PASS |
| Bangku SWE Terverifikasi | "Bersihkan bangku SWE" | 500 tugas yang dikurasi manusia, OpenAI |
| FAIL_TO_PASS | "Perbaiki gerbang" | Pengujian yang sebelumnya gagal harus lulus setelah patch |
| LULUS_TO_PASS | "Gerbang tanpa regresi" | Ujian yang telah lulus dan masih harus lulus |
| GAIA | "Patokan umum" | 466 pertanyaan multi-alat yang mudah dilakukan manusia / sulit AI |
| AgenBench | "Patokan multi-env" | 8 lingkungan; multi-putaran cakrawala panjang |
| Kontaminasi | "Kebocoran set training" | Tugas tolok ukur yang ada dalam training model |
| Bangku SWE+ | "Audit kontaminasi" | 32,67% kebocoran solusi ditemukan pada patch SWE-bench yang berhasil |

## Bacaan Lanjutan

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — tolok ukur asli
- [OpenAI, SWE-bench Terverifikasi](https://openai.com/index/introducing-swe-bench-verified/) — subset yang dikurasi
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — tolok ukur generalis
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — rangkaian multi-lingkungan
