# Masyarakat Pikiran dan Debat Multi-Agen

> Premis Minsky tahun 1986 – intelijen adalah masyarakat spesialis – ditemukan kembali setiap dekade. Pada tahun 2023 Du dkk. mengubahnya menjadi algoritma konkret: beberapa contoh LLM mengajukan jawaban, membaca jawaban satu sama lain, mengkritik, dan memperbarui. Selama N putaran mereka berkumpul pada konsensus yang mengalahkan CoT zero-shot dan refleksi pada enam tugas penalaran dan faktualitas. Ada dua temuan penting: **banyak agen** dan **beberapa putaran** berkontribusi secara independen. Masyarakat mengalahkan monolog agen tunggal; pertukaran multi-putaran mengalahkan pemungutan suara satu kali.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 16 · 04 (Model Primitif)
**Waktu:** ~60 menit

## Masalah

Konsistensi diri - contoh satu model berkali-kali dan ambil jawaban mayoritas - adalah peningkatan penalaran termurah yang dapat kamu lakukan. Ini berfungsi, tetapi cepat jenuh. kamu dapat menggandakan sample kamu dan tidak melihat lompatan berarti lainnya.

Perdebatan memecah kejenuhan. Daripada menggunakan N sample independen dari satu model, N agen saling membaca alasan masing-masing dan merevisinya. Korelasi antara sample turun (tidak lagi i.i.d.), dan titik konvergensi sering kali benar jika i.i.d. pemungutan suara jelas-jelas salah.

## Konsep

### Du dkk. Algoritma 2023

Dari arXiv:2305.14325 (ICML 2024):

1. Masing-masing dari N agen menghasilkan jawaban awal atas pertanyaan tersebut.
2. Untuk putaran r = 2..R: setiap agen diperlihatkan jawaban putaran r-1 agen lainnya dan ditanya "dengan mempertimbangkan hal ini, berikan jawaban terbaru kamu."
3. Setelah putaran R, lakukan pemungutan suara terbanyak untuk jawaban akhir.

Makalah ini menguji MMLU, GSM8K, biografi, MATEMATIKA, dan tolok ukur faktualitas. Debat secara konsisten mengalahkan CoT dan Refleksi Diri.

### Dua kenop independen

Ablasi dari kertas yang sama:

- **Agen dihitung sendiri** (1 putaran, suara mayoritas N) mengalahkan agen tunggal dalam sebagian besar tugas, namun tetap stabil.
- **Penghitungan putaran saja** (1 agen melihat alasannya sendiri sebelumnya) hampir tidak membantu — kelemahan refleksi yang diketahui.
- **Keduanya bersama-sama** menghasilkan lompatan besar. Pertukaran multi-putaran antara banyak agen mendorong keuntungan.

### Mengapa ini berhasil

Dua mekanisme:

1. **Paparan terhadap ketidaksepakatan.** Ketika seorang agen melihat rantai alasan agen lain dengan kesimpulan yang berbeda, ia harus membenarkan atau memperbarui. Apa pun yang terjadi, konteks untuk putaran r+1 lebih kaya daripada putaran r.
2. **Pengurangan kesalahan yang berkorelasi.** Dalam konsistensi mandiri, semua sample berasal dari model yang sama, sehingga kesalahannya berkorelasi — kamu membuat rata-rata jawaban yang benar-benar salah. Beda model atau beda dekorasi benihnya. *Pandangan yang diperdebatkan* yang berbeda akan dihubungan lebih jauh.

### Perdebatan yang heterogen

A-HMAD dan tindak lanjut terkait menggunakan *model dasar yang berbeda* untuk agen yang berbeda. Perdebatan Llama + Claude + GPT mengurangi keruntuhan monokultur (Lesson 26) karena kesalahan yang berkorelasi dari satu keluarga model tidak dialami oleh keluarga model lainnya.

Kelemahannya: model lemah yang berpartisipasi dalam perdebatan dapat menyeret konsensus ke arah jawaban yang salah (lihat "Haruskah kita menjadi MAD?", arXiv:2311.17371).

### NLSOM — ekstensi 129 agen

Zhuge dkk. (“Mindstorms in Natural Language-Based Societies of Mind,” arXiv:2305.17066) memperluas gagasan ini ke masyarakat yang beranggotakan 129 orang. Hasilnya: spesialisasi dan pengorganisasian mandiri muncul dalam skala besar, dan sistem ini mengungguli agen tunggal dalam tugas-tugas seperti menjawab pertanyaan visual.

### Mode kegagalan- **Sycophancy cascade.** Semua agen tunduk pada agen mana pun yang terdengar paling percaya diri. Perdebatan berakhir dengan suara yang paling keras. Mendorong peran yang berlawanan ("satu agen harus memperdebatkan posisi tandingannya") membantu.
- **Pergeseran topik.** Perdebatan dalam beberapa putaran menyimpang dari pertanyaan awal. Mitigasi: masukkan kembali pertanyaan setiap putaran.
- **Hitung ledakan.** N agen × R putaran = N·R panggilan LLM, masing-masing dengan konteks yang terus bertambah. Debat 5 agen dan 5 putaran berarti 25 seruan dalam konteks yang berkembang. Biaya per pertanyaan dapat melebihi 10× satu panggilan CoT.

## Build

`code/main.py` menjalankan debat 3 agen × 3 putaran mengenai pertanyaan matematika di mana setiap agen memulai dengan jawaban yang berbeda (mungkin salah). Agen diberi skrip - masing-masing "memperbarui" dengan merata-ratakan jawaban tetangga yang diberi weight berdasarkan keyakinan yang tertulis. Konvergensi terlihat pada log putaran demi putaran.

Demo ini menunjukkan dua efek utama:

- Satu putaran pertukaran membuat agen semakin dekat dengan jawaban yang benar.
- Putaran tambahan setelah putaran 2 menunjukkan hasil yang semakin berkurang (cocok dengan dataran tinggi Du et al.).

Jalankan:

```
python3 code/main.py
```

## Pakai

`outputs/skill-debate-configurator.md` mengonfigurasikan debat untuk tugas baru: jumlah agen, jumlah putaran, heterogenitas (model yang sama vs campuran), penetapan peran (simetris vs satu musuh). Ini juga memperkirakan biaya token sebelum kamu menjalankannya.

## Kirim

Jika kamu mengirimkan debat:

- **Batas putaran pada pukul 3.** Du dkk. tampilkan 3 putaran menangkap sebagian besar keuntungan. Yang lebih penting adalah biaya, bukan kualitas.
- **Batas agen di 5.** Di atas 5, konteks membengkak dan biaya mendominasi.
- **Heterogen secara default.** Setidaknya ada dua model dasar berbeda dalam kumpulan.
- **Slot permusuhan.** Namun salah satu agen diminta untuk tidak setuju. Mematahkan penjilatan.
- **Catat setiap putaran.** Sistem debat yang menyembunyikan putaran perantara tidak dapat di-debug atau diaudit.

## Latihan

1. Jalankan `code/main.py`, lalu atur hitungan putaran menjadi 5 dan lihat hasil yang semakin berkurang. Pada putaran manakah konvergensi tambahan berhenti?
2. Tambahkan agen keempat dengan peran yang berlawanan: selalu tidak setuju dengan mayoritas saat ini. Apakah hal ini merusak atau meningkatkan konvergensi?
3. Plot (cetak) skor kesepakatan per putaran (fraksi agen pada jawaban mayoritas). Kapan mencapai 1.0 dan apakah itu setara dengan "benar"?
4. Baca Du dkk. Bagian 4 ablasi. Replikasi hasil "khusus agen" vs "khusus putaran" vs "keduanya" menggunakan code ini.
5. Baca “Haruskah kita menjadi GILA?” (arXiv:2311.17371) dan sebutkan dua varian debat di luar sistem round-robin — misalnya, dipimpin hakim, rantai perdebatan, dan permusuhan.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Masyarakat Pikiran | "Ide Minsky" | Intelijen sebagai spesialis yang berinteraksi; Pembingkaian tahun 1986 sekarang dioperasionalkan melalui debat LLM. |
| Debat multi-agen | "Agen berdebat" | N agen mengusulkan, saling mengkritik, merevisi putaran R, suara terbanyak. |
| Konsensus | "Mereka setuju" | Bukan kebenaran epistemik - hanya jawaban sebagian kecil dari mayoritas. Bisa saja salah. |
| Putaran | "Langkah pertukaran" | Satu putaran = setiap agen membaca yang lain dan memperbarui satu kali. |
| Perdebatan heterogen | "Campurkan keluarga model" | Menggunakan model dasar yang berbeda untuk mengatasi kesalahan. |
| Kaskade penjilatan | "Semua orang setuju dengan yang keras" | Kegagalan debat ketika agen tunduk pada agen yang paling percaya diri tanpa mempedulikan kebenarannya. |
| NLSOM | "Masyarakat 129 agen" | Masyarakat pikiran berbahasa alami; Versi skala Zhuge dkk. |
| Kesalahan berkorelasi | "Model yang sama, bug yang sama" | Mengapa konsistensi diri menjadi jenuh; perdebatan antar pandangan yang berbeda saling berhubungan. |

## Bacaan Lanjutan

- [Du dkk. — Meningkatkan Faktualitas dan Penalaran dalam Model Bahasa melalui Debat Multiagen](https://arxiv.org/abs/2305.14325) — makalah referensi, ICML 2024
- [Zhuge dkk. — Badai Pikiran dalam Masyarakat Pikiran Berbasis Bahasa Alami](https://arxiv.org/abs/2305.17066) — 129-agen NLSOM
- [Haruskah kita menjadi GILA? Sekilas tentang Strategi Debat Multi-Agen untuk LLM](https://arxiv.org/abs/2311.17371) — membuat tolok ukur varian debat
- [Halaman proyek debat](https://composable-models.github.io/llm_debate/) — Code, demo, dan detail ablasi Du et al.
