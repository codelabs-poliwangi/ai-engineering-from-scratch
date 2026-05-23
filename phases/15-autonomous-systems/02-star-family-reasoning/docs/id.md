# STaR, V-STAR, Quiet-STAR — Penalaran yang Diajarkan Sendiri

> Lingkaran pengembangan diri sekecil mungkin ada di dalam dasar pemikirannya. Sebuah model menghasilkan rantai pemikiran, mempertahankan jawaban yang benar, dan menyempurnakannya. Itu adalah STAR. V-STAR menambahkan verifier sehingga pemilihan waktu inference menjadi lebih baik. Quiet-STAR menekankan alasan tersebut hingga ke setiap token. Ketiganya berfungsi. Tak satu pun dari mereka yang ajaib — loop mempertahankan pintasan apa pun yang terjadi untuk mencapai jawaban yang benar.

**Type:** Learn
**Language:** Python (stdlib, simulator bootstrap-loop)
**Prerequisites:** Fase 13 · 01-03 (Penalaran dan CoT), Fase 15 · 01 (pembingkaian cakrawala panjang)
**Waktu:** ~60 menit

## Masalah

Cara langsung untuk mengajarkan model bernalar adalah dengan mengumpulkan jejak penalaran yang ditulis oleh manusia. Hal ini mahal, lambat, dan dibatasi oleh seberapa banyak rantai pemikiran berkualitas tinggi yang bersedia ditulis oleh manusia.

STaR (Self-Taught Reasoner, Zelikman et al., 2022) bertanya: bagaimana jika model menulis alasannya sendiri dan menilainya berdasarkan jawaban yang diketahui? Lingkarannya adalah:

1. Contoh penelusuran penalaran plus jawaban.
2. Jika jawaban akhir benar, simpanlah jejaknya.
3. Sempurnakan jejak yang disimpan.
4. Ulangi.

Ini berhasil. GSM8K dan CommonsenseQA keduanya ditingkatkan tanpa anotasi manusia yang baru. Namun loop tersebut memiliki bias bawaan: alasan apa pun yang menghasilkan jawaban yang benar tetap dipertahankan, terlepas dari apakah alasan itu masuk akal. V-STAR (Hosseini et al., 2024) menambalnya dengan verifikator yang dipelajari; Quiet-STAR (Zelikman et al., 2024) menggeneralisasi gagasan tersebut menjadi alasan internal per-token.

## Konsep

### STAR: bootstrap tentang apa yang berhasil

Mulailah dari model dasar dengan kemampuan penalaran yang lemah. Pada setiap soal training, contohlah alasan dan jawabannya. Jika jawabannya sesuai dengan label, pertahankan (masalah, alasan, jawaban) tetap tiga kali lipat. Sempurnakan model pada set yang disimpan. Mengulang.

Satu perubahan penting. Jika model tidak pernah dapat menyelesaikan masalah dengan benar, perulangan tidak dapat mempelajarinya. STaR menambahkan **rasionalisasi**: jika ada masalah yang modelnya gagal, berikan jawaban yang benar sebagai petunjuk dan minta kembali model untuk menghasilkan alasan yang mengarah ke sana. Alasan yang dirasionalisasikan ditambahkan ke set training.

Hasil dalam makalah asli (Zelikman et al., 2022): model dasar GPT-J ditingkatkan pada GSM8K dari 5,8% menjadi 10,7% melalui putaran STaR berulang dengan rasionalisasi — absolut sekitar 5 poin persentase. Di CommonsenseQA, GPT-J 6B yang dilatih STaR mencapai 72,5%, sebanding dengan GPT-3 175B yang telah disempurnakan (~73%) — model yang kira-kira 30x lebih besar yang dilatih berdasarkan alasan yang dianotasi dengan tangan.

### V-STAR: melatih verifikator dengan DPO

STaR membuang alasan yang salah. Hosseini dkk. (2024) mengamati itu juga data: setiap pasangan (alasannya, "apakah ini benar") dapat melatih verifikator. Mereka menggunakan Optimization Preferensi Langsung atas solusi yang benar dan salah untuk membangun peringkat. Pada waktu inference, ambil contoh N alasan dan pilih pilihan utama pemverifikasi.

Delta yang dilaporkan: +4 hingga +17 poin persentase dibandingkan baseline pengembangan mandiri sebelumnya pada GSM8K dan MATH, dengan sebagian besar keuntungan berasal dari penggunaan verifikator untuk pemilihan waktu inference, bukan untuk penyempurnaan generator tambahan.

### Quiet-STAR: alasan internal per tokenZelikman dkk. (2024) bertanya: bagaimana jika model belajar menghasilkan alasan internal yang singkat di setiap posisi token, tidak hanya antara masalah dan jawaban? Quiet-STAR melatih model untuk memancarkan "pikiran" tersembunyi sebelum setiap token yang diprediksi, lalu menggabungkan prediksi berdasarkan pemikiran dengan prediksi dasar melalui weight yang dipelajari.

Hasil: Mistral 7B memperoleh peningkatan zero-shot mutlak pada GSM8K dari 5,9% menjadi 10,9% dan CommonsenseQA dari 36,3% menjadi 47,2% tanpa penyesuaian khusus tugas. Model tersebut mempelajari "kapan harus berpikir" - token keras mendapatkan alasan internal yang lebih panjang; yang mudah hampir tidak mendapatkan apa-apa.

### Mengapa ketiganya mempunyai kepedulian yang sama terhadap keselamatan

Ketiga metode tersebut menggunakan jawaban akhir sebagai sinyal gradient. Alasan yang mencapai jawaban yang benar melalui penalaran yang salah - mengeksploitasi jalan pintas, menebak, atau menggunakan pola yang tidak menggeneralisasi - akan diperkuat secara positif. Pada masalah dalam distribusi, pintasan berfungsi. Pada masalah di luar distribusi, ia rusak secara diam-diam.

Verifikator V-STAR melakukan mitigasi dengan belajar menentukan peringkat alasan, namun verifikator dilatih pada kumpulan label yang sama. Ia dapat belajar untuk lebih memilih penalaran salah yang diformat dengan baik daripada ketidakpastian yang jujur. Desain yang lebih aman adalah menggabungkan data gaya STaR dengan (a) model penghargaan yang diawasi proses (memberi penghargaan pada langkah-langkah perantara, bukan hanya jawaban) dan (b) evaluasi OOD yang dilakukan secara terus-menerus yang memecah jalan pintas sederhana.

### Perbandingan

| Metode | Sinyal training | Biaya inference | Pemborosan data | Mode kegagalan yang diketahui |
|---|---|---|---|---|
| BINTANG | pertahankan (alasan, jawaban) jika benar | 1x | membuang semua alasan yang salah | alasan pintas |
| STAR + rasionalisasi | di atas + jawaban yang benar mengisyaratkan percobaan ulang | 1x | kurang | alasan yang dirasionalisasi mungkin tidak masuk akal |
| V-STAR | Verifikator STAR + DPO dari kedua kelas | Nx (terbaik dari N) | minimal | verifikator dapat memperkuat kesalahan yang diyakini |
| Tenang-STAR | alasan per token + weight pencampuran | 1,5-3x | minimal | masih gradient yang dikondisikan jawaban |

### Dimana ini berada di tumpukan 2026

STAR sudah tua. Namun pola tersebut muncul kembali di mana-mana pada tahun 2025-2026. RL pada soal matematika yang dapat diverifikasi (DeepSeek-R1, Kimi-k1.5, o1) adalah sinyal gradient yang dikondisikan jawaban STaR, yang ditingkatkan skalanya. Model penghargaan proses (Lightman et al., 2023; "Mari kita verifikasi langkah demi langkah" dari OpenAI) adalah alternatif yang diawasi oleh proses. AlphaEvolve (Lesson 3) adalah STaR untuk code, dengan evaluator program, bukan label. Mesin Darwin Godel (Lesson 4) adalah STaR untuk agen scaffolding itu sendiri.

Memahami STaR membuat semua ini klik. Ini adalah lingkaran perbaikan diri minimum yang dapat dilakukan.

## Pakai

`code/main.py` menjalankan simulasi loop STaR pada tugas aritmatika mainan. kamu dapat menonton:

- Bagaimana akurasinya melebihi putaran bootstrap.
- Bagaimana jalan pintas menyelinap: simulator menyertakan kelas pemikiran "malas" yang mendapat jawaban benar 40% dari waktu tetapi menggeneralisasi dengan buruk. Perhatikan apakah STaR menyimpannya.
- Bagaimana verifikator (gaya V-STAR) membantu dalam inference tetapi tidak dapat sepenuhnya memangkas pintasan yang diperkenalkan selama training.

## Kirim

`outputs/skill-star-loop-reviewer.md` membantu kamu mengaudit usulan alur penalaran otodidak sebelum kamu melatihnya.

## Latihan

1. Jalankan simulatornya. Atur frekuensi pintasan ke nol, lalu ke 0,4. Berapa perbedaan akurasi akhir antara kedua proses, meskipun keduanya mencapai >90% pada distribusi training?

2. Tambahkan tes OOD yang ditahan ke simulator. Gambarkan masalah dari distribusi yang berbeda dan evaluasi model bootstrap pada set dalam distribusi dan OOD. Hitung kesenjangannya.3. Baca makalah Quiet-STAR (arXiv:2403.09629) Bagian 3. Jelaskan token "akhir pemikiran" dan kepala pemberat pencampuran masing-masing dalam tiga kalimat.

4. Bandingkan filter simpan-jika-benar STaR dengan alternatif yang diawasi proses yang memberi penghargaan pada setiap langkah rasional secara independen. Identifikasi perbedaan biaya pelabelan dan perbedaan kualitas yang masuk akal.

5. Rancang satu evaluasi yang dapat menangkap alasan-alasan pintas dalam model yang diterapkan. Ini tidak harus sempurna - ia harus memutus pintasan paling sederhana yang akan diperkuat oleh loop STAR.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| BINTANG | "Penalaran yang Diajarkan Sendiri" | Menyempurnakan alasan yang dihasilkan model yang menghasilkan jawaban yang benar; ulangi |
| Rasionalisasi | "Diisyaratkan coba lagi" | Masukkan jawaban yang benar dan minta kembali alasan atas masalah yang gagal dalam model dasar |
| V-STAR | "Pemverifikasi STAR" | DPO melatih pemverifikasi tentang alasan yang benar dan salah, menggunakannya untuk pemilihan waktu inference |
| Tenang-STAR | "Alasan per-token" | Hasilkan pemikiran tersembunyi di setiap posisi token; campur dengan prediksi dasar |
| Gradient yang dikondisikan jawaban | "Sinyal berbasis hasil" | Lingkaran training memberi penghargaan pada jawaban akhir, bukan langkah penalaran |
| Model imbalan proses | "Pemverifikasi tingkat langkah" | Model penghargaan dilatih berdasarkan kebenaran per langkah, bukan hasil — berbeda dengan STaR |
| Alasan jalan pintas | "Jawaban benar, alasan salah" | Alasan yang mencapai label melalui pola non-generalisasi; STAR menyimpan |

## Bacaan Lanjutan

- [Zelikman dkk. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — makalah asli.
- [Hosseini dkk. (2024). V-STAR: Pemverifikasi Training untuk Alasan yang Diajarkan Sendiri](https://arxiv.org/abs/2402.06457) — menambahkan pemverifikasi DPO untuk pemilihan waktu inference.
- [Zelikman dkk. (2024). Quiet-STAR: Model Bahasa Dapat Mengajari Diri Sendiri untuk Berpikir Sebelum Berbicara](https://arxiv.org/abs/2403.09629) — alasan internal per token.
- [Lightman dkk. (2023). Mari Verifikasi Langkah demi Langkah](https://arxiv.org/abs/2305.20050) — memproses model imbalan, sinyal gradient alternatif.
- [Makalah DeepSeek-R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — RL pada tugas yang dapat diverifikasi, STaR ditingkatkan ke training perbatasan.
