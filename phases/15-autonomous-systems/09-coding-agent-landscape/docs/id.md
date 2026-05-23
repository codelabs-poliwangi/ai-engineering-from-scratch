# Lanskap Agen Pengkodean Otonom (2026)

> SWE-bench Verified meningkat dari 4% menjadi 80,9% dalam waktu kurang dari tiga tahun. Claude Sonnet 4.5 yang sama mencetak 43,2% pada agen SWE v1 dan 59,8% pada otonom Cline — perancah di sekitar model kini sama pentingnya dengan model itu sendiri. OpenHands (sebelumnya OpenDevin) adalah platform berlisensi MIT yang paling aktif dan loop CodeAct-nya mengeksekusi tindakan Python secara langsung di sandbox, bukan panggilan alat JSON. Angka-angka utama menyembunyikan masalah metodologis: 161 dari 500 tugas SWE-bench Terverifikasi hanya memerlukan 1–2 perubahan baris, dan SWE-bench Pro (10+ tugas baris) berada pada 23–59% untuk model frontier yang sama.

**Type:** Learn
**Language:** Python (stdlib, CodeAct vs perbandingan panggilan alat JSON)
**Prerequisites:** Fase 14 · 07 (Penggunaan alat), Fase 15 · 01 (Agen cakrawala panjang)
**Waktu:** ~45 menit

## Masalah

"Agen pengkodean mana yang terbaik" adalah pertanyaan yang salah. Pertanyaan yang tepat adalah: pada distribusi tugas yang sesuai dengan pekerjaan saya, dengan scaffolding yang akan saya jalankan dalam produksi, keandalan end-to-end apa yang saya dapatkan?

Antara tahun 2022 dan 2026, bidang ini mempelajari bahwa scaffolding — layer pengambilan, perencana, kotak pasir, loop edit-verifikasi, format umpan balik — bersifat menahan weight. Claude Sonnet 4.5 di SWE-agent v1 mencetak 43,2% di SWE-bench Terverifikasi; model yang sama di dalam perancah otonom Klein memperoleh skor 59,8%. Perbedaan 16,6 poin absolut, weight yang sama. Model dasar adalah sebuah komponen; loop adalah produknya.

Masalah yang menyertainya adalah saturasi benchmark menyembunyikan regresi. Bangku SWE Terverifikasi hampir jenuh, dan tugas yang mudah (161 dari 500 tugas yang membutuhkan ≤2 baris) meningkatkan skor tertinggi. Kualitas dunia nyata lebih baik diukur pada distribusi seperti SWE-bench Pro (10+ perubahan lini), dengan pemimpin yang sama masih berada di angka 23–59%.

## Konsep

### Bangku SWE, satu paragraf

SWE-bench (Jimenez dkk.) menangani masalah GitHub yang sebenarnya dengan patch kebenaran dasar dan meminta agen untuk membuat patch yang membuat rangkaian pengujian lolos. SWE-bench Verified (OpenAI, 2024) adalah subset 500 tugas yang dikurasi manusia dengan menghapus tugas yang ambigu dan rusak. SWE-bench Pro adalah penerus yang lebih sulit - tugas yang memerlukan 10+ jalur perubahan, dengan jumlah agen perbatasan saat ini berada di angka 23–59%.

### Apa yang sebenarnya ditunjukkan oleh kurva 2022 → 2026

- **2022**: model penelitian sebesar ~4% pada bangku SWE mentah.
- **2024**: GPT-4 + Scaffolding bergaya Devin sebesar ~14%; Agen SWE di ~12%.
- **2025**: Claude 3.5/3.7 Sonnet di dalam Aider dan agen SWE mendorong ke kisaran 40–55%.
- **2026**: Claude Sonnet 4.5 dan kompetitor terdepan dengan skor 70–80%+ di SWE-bench Terverifikasi. Papan peringkat Epoch AI melacak ini secara langsung.

Kemiringan tersebut berasal dari tiga sumber gabungan: model dasar yang lebih baik, scaffolding yang lebih baik (CodeAct, refleksi, loop verifikasi), dan tolok ukur yang lebih baik (Terverifikasi menghilangkan noise).

### Panggilan alat CodeAct vs JSON

OpenHands (All-Hands-AI, arXiv:2407.16741, sebelumnya OpenDevin) mengambil taruhan arsitektur tertentu: alih-alih model yang mengeluarkan panggilan alat JSON yang didekodekan dan dijalankan oleh host, model tersebut memancarkan code Python dan kernel bergaya Jupyter menjalankannya di kotak pasir. Agen dapat mengulang file, merangkai alat, dan menangkap pengecualiannya sendiri dalam satu tindakan.

Pertukarannya:- **Panggilan alat JSON**: setiap tindakan adalah satu putaran; mudah untuk diaudit; komposisionalitas terbatas; aman secara default karena setiap panggilan melewati validator eksplisit.
- **CodeAct**: satu tindakan dapat menjadi keseluruhan program; komposisi; memerlukan kotak pasir yang diperkeras (OpenHands menggunakan isolasi Docker); mode kegagalan mencakup apa pun yang diizinkan oleh runtime sandbox.

Kedua arsitektur sedang dalam produksi. CodeAct dominan di platform terbuka (OpenHands, smolagents). Panggilan alat JSON tetap dominan dalam layanan terkelola (Agen Terkelola Antropik, Asisten OpenAI) di mana penyedia mengontrol pelaksananya.

### Perancah di lanskap tahun 2026

| Perancah | Lisensi | Model eksekusi | Properti terkenal |
|---|---|---|---|
| Tangan Terbuka (OpenDevin) | MIT | CodeAct di Docker | Platform terbuka paling aktif; aliran acara dapat diputar ulang |
| Agen SWE | MIT | Antarmuka Agen-Komputer (ACI) | Perancah bangku SWE ujung ke ujung pertama |
| Pembantu | Apache-2 | edit-via-diff di repo lokal | Perancah minimal, stabilitas regresi kuat |
| Klinik | Apache-2 | Agen VS Code dengan kebijakan alat | Perancah terbuka dengan skor tertinggi di Sonnet 4.5 |
| Devin (Kognisi) | Kepemilikan | VM + perencana terkelola | Kategori produk "insinyur perangkat lunak AI" pertama |
| Code Claude | Kepemilikan | Mode izin + rutinitas | Lesson 10 membahas loop agen secara detail |

### Mengapa scaffolding mendominasi

Proses pengkodean adalah lintasan cakrawala yang panjang (Lesson 1). Keandalan bertambah di seluruh langkah. Tiga tempat di mana scaffolding membeli poin:

1. **Pengambilan**: menemukan file yang tepat untuk dibaca adalah hambatan terbesar. ACI agen SWE, indeks file OpenHands, dan peta repo Aider semuanya menyerang ini.
2. **Perulangan pemverifikasi**: menjalankan pengujian, membaca jejak tumpukan, dan mencoba ulang adalah delta 10+ poin di bangku SWE.
3. **Penahanan kegagalan**: kotak pasir yang mengembalikan kesalahan mencegah kerusakan yang bertambah parah. Model yang sama dengan dan tanpa loop verifikator tampak seperti dua produk berbeda.

### Saturasi patokan dan distribusi sebenarnya

Penulis OpenHands dan Epoch AI sama-sama menyatakan bahwa SWE-bench Verified memiliki penyelesaian yang mudah: 161 dari 500 tugas hanya memerlukan 1–2 baris perubahan. Skor tinggi sebagian didorong oleh hal ini. SWE-bench Pro membatasi hingga 10+ perubahan garis dan mengembalikan skor dalam kisaran 23–59% bahkan untuk sistem frontier. Distribusi produksi kamu hampir pasti lebih mendekati Pro daripada Terverifikasi.

Implikasi dalam memilih agen: jalankan subset mirip Pro dari bug backlog kamu sendiri. Skor yang penting adalah skor tugas yang mewakili apa yang kamu kirimkan.

## Pakai

`code/main.py` membandingkan dua perancah agen mainan pada distribusi tugas mini tetap:

1. Perancah **panggilan alat JSON** yang mengambil satu tindakan per giliran.
2. Scaffold **CodeAct** yang dapat mengeluarkan cuplikan Python kecil per tindakan.

Keduanya menggunakan "model" rintisan (aturan deterministik) sehingga perbandingan mengisolasi perancah dari kualitas model. Outputnya menunjukkan perancah CodeAct menyelesaikan lebih banyak tugas dalam putaran yang lebih sedikit dengan mengorbankan radius ledakan per tindakan yang lebih besar.

## Kirim

`outputs/skill-scaffold-audit.md` membantu kamu mengaudit scaffold agen pengkodean yang diusulkan sebelum penerapan: kualitas pengambilan, keberadaan pemverifikasi, isolasi sandbox, dan kesesuaian benchmark-ke-distribusi.

## Latihan

1. Jalankan `code/main.py`. Berapa banyak putaran yang dilakukan setiap perancah pada rangkaian tugas yang sama? Berapa radius ledakan per tindakan masing-masing?2. Baca makalah OpenHands (arXiv:2407.16741). Makalah ini berpendapat bahwa CodeAct mengalahkan panggilan alat JSON pada tugas-tugas kompleks. Identifikasi satu mode kegagalan yang diketahui makalah tersebut dan tuliskan satu kalimat kapan mode tersebut akan mendominasi produksi.

3. Pilih satu tugas dari bug backlog kamu yang memerlukan 10+ baris perubahan pada dua file. Perkirakan probabilitas keberhasilan end-to-end untuk model frontier berdasarkan (a) panggilan alat JSON dan (b) CodeAct. Membenarkan kesenjangan tersebut.

4. SWE-bench Verified memiliki 161 tugas file tunggal, 1–2 baris. Buatlah skor yang mengecualikan mereka. Bagaimana cara pengacakan papan peringkat?

5. Baca "Memperkenalkan SWE-bench Terverifikasi" (OpenAI). Jelaskan metodologi spesifik yang digunakan untuk menghilangkan tugas-tugas ambigu, dan sebutkan satu kategori yang mungkin terlewatkan oleh kurasi.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| Bangku SWE | "Patokan pengkodean" | Masalah nyata GitHub dengan patch kebenaran dasar dan rangkaian pengujian |
| Bangku SWE Terverifikasi | "Subset yang dibersihkan" | 500 tugas yang dikurasi manusia, hadiah yang lebih mudah |
| Bangku SWE Pro | "Subset yang lebih sulit" | 10+ perubahan baris; perbatasan berada di 23–59% |
| Code Bertindak | "Code-sebagai-tindakan" | Agen mengeluarkan Python; Kernel bergaya Jupyter dijalankan di sandbox |
| Panggilan alat JSON | "Panggilan fungsi" | Setiap tindakan adalah payload JSON terstruktur yang divalidasi sebelum eksekusi |
| Perancah | "Kerangka agen" | Pengambilan + perencana + pelaksana + putaran pemverifikasi di sekitar model dasar |
| ACI (Antarmuka Agen-Komputer) | "Format agen SWE" | Set prompt dirancang untuk ergonomi LLM, bukan cangkang manusia |
| Lingkaran pemverifikasi | "Uji-dan-coba lagi" | Jalankan tes, baca output, revisi tambalan; perolehan keandalan non-model terbesar |

## Bacaan Lanjutan

- [Jimenez dkk. — SWE-bench](https://www.swebench.com/) — tolok ukur dan metodologi asli.
- [OpenAI — Memperkenalkan SWE-bench Terverifikasi](https://openai.com/index/introducing-swe-bench-verified/) — bagaimana subset yang dikurasi dibuat.
- [Wang dkk. — OpenHands: Platform Terbuka untuk Pengembang Perangkat Lunak AI](https://arxiv.org/abs/2407.16741) — Arsitektur CodeAct dan desain aliran acara.
- [Epoch AI — papan peringkat SWE-bench](https://epoch.ai/benchmarks) — skor yang dilacak secara langsung.
- [Antropik — Mengukur otonomi agen](https://www.anthropic.com/research/measuring-agent-autonomy) — pembingkaian keandalan agen pengkodean cakrawala panjang.
