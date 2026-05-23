# Pemahaman Video Panjang dalam Konteks Sejuta Token

> Video 4K berdurasi 1 jam pada 24 FPS, ditambal dan di-embed, menghasilkan sekitar 60 juta token. Episode podcast 2 jam yang ditranskripsikan adalah 30.000 token. Film feature Blu-ray lengkap, bahkan dikompresi dengan pengumpulan agresif, bernilai ratusan ribu token. Google Gemini 1.5 (Maret 2024) membuka era ini dengan konteks 10 juta token, melakukan recall yang andal dalam video berdurasi satu jam. LWM (Liu et al., Februari 2024) menunjukkan jalur penskalaan attention cincin. LongVILA dan Video-XL meningkatkan penyerapan lebih lanjut. VideoAgent menukar konteks mentah untuk pengambilan agen. Setiap pendekatan memiliki trade-off yang berbeda dalam hal komputasi, penarikan kembali, dan kompleksitas teknik. Lesson ini membacanya secara berdampingan.

**Type:** Build
**Language:** Python (stdlib, simulator jarum di tumpukan jerami + router pengambilan agen)
**Prerequisites:** Fase 12 · 17 (token temporal video)
**Waktu:** ~180 menit

## Tujuan Pembelajaran

- Hitung total jumlah token visual untuk video berdurasi panjang pada berbagai FPS dan pengumpulan.
- Jelaskan tiga jalur penskalaan: konteks kasar (Gemini 1.5), attention cincin (LWM), kompresi token (LongVILA / Video-XL).
- Bandingkan VLM video konteks mentah vs VLM video pengambilan agen (VideoAgent) dalam hal akurasi dan latensi.
- Rancang tes jarum di tumpukan jerami untuk video berdurasi 30 menit dan ukur ingatan pada menit tertentu.

## Masalah

Satu frame patch berukuran Qwen2.5-VL pada resolusi asli 384 adalah ~729 token. Pada pengumpulan 3x3, itu berarti 81 token per frame. Klip berdurasi 30 menit pada 1 FPS = 1800 frame = 145.800 token. Dapat dilakukan pada tahun 2025 dengan VLM terbuka, ketat. Pada 2 FPS, 291.600 token — hanya konteks terbesar yang cocok.

Film berdurasi 2 jam pada 1 FPS berharga 583 ribu token. Melampaui sebagian besar model terbuka tahun 2026; membutuhkan Gemini 2.5 Pro atau pooling yang lebih agresif.

Tiga jalur penskalaan muncul.

## Konsep

### Jalur 1: Konteks kasar (Gemini 1.5, Claude Opus)

Lemparkan perangkat keras pada masalahnya. Skalakan konteks ke jutaan token, proses semuanya dalam satu langkah maju.

Gemini 1.5 Pro diluncurkan dengan 1 juta token; Gemini 1,5 Ultra hingga 10M; Gemini 2.5 Pro pada tahun 2026 mampu memutar video berjam-jam dengan andal. Makalah (arXiv:2403.05530) mendokumentasikan penarikan kembali jarum di tumpukan jerami sebesar 99,7% hingga ~9,5 juta token.

Rekayasa: implementasi attention khusus dengan hierarki memori (lokal + global + jarang) ditambah perutean ahli MoE untuk efisiensi konteks panjang. Tidak dipublikasikan secara lengkap. Bukan sumber terbuka.

### Jalur 2: Deringkan attention (LWM, LongVILA)

Attention dering mendistribusikan urutan panjang ke seluruh perangkat dalam "cincin" tempat setiap perangkat memegang sepotong. Attention di seluruh rangkaian terjadi dengan masing-masing perangkat mengirimkan potongannya ke perangkat berikutnya dalam pola cincin, menghitung attention parsial, dan melakukan agregasi.

LWM (Liu et al., 2024) melatih model konteks token 1M dengan cara ini. Training komputasi berskala secara linear dengan konteks, bukan secara kuadratik — pencapaian kuadrat pada attention diamortisasi ke seluruh perangkat ring.

LongVILA (arXiv:2408.10188) mengadaptasi pola tersebut ke VLM. Video 1400 bingkai dengan 192 token per bingkai = 268 ribu konteks, dilatih dengan attention penuh pada paralelisme 8 arah.

### Jalur 3: Kompresi token (Video-XL, LongVA)

Lebih murah daripada konteks kasar: kompres secara agresif sebelum LLM melihat urutannya.

Video-XL (arXiv:2409.14485) menggunakan token ringkasan visual: setiap klip dari N frame menghasilkan satu token "ringkasan" yang hadir di atas N. Pada inference, LLM melihat satu token ringkasan per klip, sehingga secara drastis mengecilkan konteksnya.LongVA memperluas konteks LLM dari 200 ribu menjadi 2 juta dengan teknik "transfer konteks panjang". Latih teks konteks panjang, transfer ke video konteks panjang melalui representasi bersama.

Kompresi token mengorbankan penarikan kembali pada stempel waktu tertentu untuk skalabilitas. Model mengetahui secara umum apa yang terjadi tetapi terkadang melewatkan frame yang tepat.

### Jalur 4: Pengambilan agen (VideoAgent)

Jangan memasukkan video lengkap ke LLM. Sebaliknya, perlakukan video tersebut sebagai database dan gunakan LLM untuk menanyakannya.

Agen Video (arXiv:2403.10517):

1. LLM membacakan pertanyaan.
2. LLM meminta alat pengambilan klip yang relevan ("tunjukkan segmen dengan kucing").
3. Alat mengembalikan stempel waktu klip yang cocok.
4. LLM membaca klip tersebut melalui VLM.
5. LLM menyusun jawaban atau menanyakan pertanyaan lanjutan.

Ini adalah pola LLM sebagai agen yang diterapkan pada video panjang. Inference yang lebih murah (hanya klip relevan yang dikodekan), rekayasa yang lebih sulit (kualitas pengambilan menjadi penghambat).

### Tolok ukur yang sulit dicapai

Tes konteks panjang standar: masukkan penanda visual atau tekstual unik pada titik acak dalam video, lalu ajukan pertanyaan yang perlu diingat.

Metrik: Recall@k sepanjang durasi video dan posisi penanda.

Skor Gemini 2.5 Pro >99% recall pada video berdurasi hingga 90 menit. Model terbuka 72B (Qwen2.5-VL-72B, InternVL3-78B) mendapat skor ~85-90% pada 30 menit dan menurun setelah 60 menit.

VideoAgent dapat mencocokkan atau mengalahkan model konteks mentah dalam waktu 2+ jam karena pengambilan akan sangat tepat jika alatnya bagus.

### Jalur mana yang harus dipilih

Untuk klip berdurasi 15 menit dengan akurasi frontier: konteks asli terbuka 72B + biasanya berfungsi. Pilih Qwen2.5-VL-72B.

Untuk konten berdurasi 30 menit hingga 1 jam: LongVILA atau Video-XL untuk konten terbuka; Gemini 2.5 Pro untuk ditutup. Batasan kualitas penting — batas menjadi tertutup.

Untuk konten 2+ jam: VideoAgent atau pola pengambilan serupa. Alternatifnya, rangkum menjadi bagian-bagian yang lebih kecil dan berikan ringkasan hierarki.

### Pola produksi 2026

Dalam praktiknya, pipeline produksi video panjang bersifat hibrid:

1. Jalankan pengambilan sample FPS dinamis + penggabungan agresif di seluruh video (dapatkan representasi global 100 ribu token).
2. Lulus ke VLM 72B untuk ringkasan global.
3. Jika pengguna mengajukan pertanyaan mendetail, jalankan pengambilan agen menggunakan ringkasan sebagai indeks.

Ini menggabungkan konteks kasar untuk pemahaman global dan pengambilan detail lokal.

## Pakai

`code/main.py`:

- Menghitung anggaran token untuk video dari 1 menit hingga 3 jam pada berbagai pengumpulan FPS +.
- Mensimulasikan lari jarum-di-tumpukan jerami: menyuntikkan penanda pada stempel waktu acak, mengajukan pertanyaan, mencetak skor ingat.
- Termasuk simulator router pengambilan agen yang mengambil klip tertentu untuk diumpankan ke VLM hilir.

Jalankan tabel anggaran dan rasakan kesenjangan skalanya.

## Kirim

Lesson ini menghasilkan `outputs/skill-long-video-strategy-planner.md`. Mengingat durasi video dan kompleksitas kueri, ia memilih antara konteks kasar, kompresi, dan pengambilan agen, serta menghitung latensi + ekspektasi kualitas.

## Latihan

1. Kuliah 45 menit pada 1 FPS, 81 token per frame. Jumlah tokennya? Cocok dengan konteks model yang mana?

2. Rancang tes jarum di tumpukan jerami: pada menit berapa kamu memasukkan penanda, dan apa format kueri sebenarnya?

3. Bandingkan konteks kasar Qwen2.5-VL-72B (konteks 80k) dengan VideoAgent (Claude 3.5 + pengambilan) pada video berdurasi 1 jam. Mana yang menang jika ditarik kembali? Manakah yang menang dalam hal latensi?

4. Biaya memori dering attention berskala linier dalam panjang urutan dan linier dalam jumlah perangkat. Jelaskan mengapa dan apa yang gagal jika kamu membatalkan fase rotasi cincin.5. Baca Gemini 1.5 Bagian 5 tentang jarum di tumpukan jerami. Apa yang ditemukan makalah ini tentang penarikan kembali pada batas token 1 juta vs 10 juta?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Konteks kasar | "Hanya lebih banyak token" | Menskalakan konteks LLM hingga jutaan token; memproses semuanya dalam satu kali jalan |
| Deringkan attention | "Paralel gaya LWM" | Pola attention terdistribusi di mana setiap perangkat memegang sepotong dan memutar |
| Kompresi token | "Ringkasan token" | Kurangi token per klip melalui kompresor yang dipelajari sebelum LLM |
| Jarum di tumpukan jerami | "Tes NIH" | Masukkan penanda unik pada titik acak, minta model untuk mengingatnya pada waktu pengujian |
| Pengambilan agen | "LLM sebagai perencana kueri" | LLM meminta alat pengambilan klip yang relevan, membacanya melalui VLM, menyusun jawaban |
| Agen Video | "Pola pengambilan video" | Desain pengambilan agen kanonik: pertanyaan -> alat -> klip -> jawaban |

## Bacaan Lanjutan

- [Tim Gemini — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu dkk. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue dkk. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu dkk. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang dkk. — Agen Video (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
