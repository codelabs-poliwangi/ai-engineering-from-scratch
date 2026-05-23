# Spesialisasi Peran — Perencana, Kritikus, Pelaksana, Verifikator

> Decomposition multi-agen yang paling umum pada tahun 2026: satu agen merencanakan, satu mengeksekusi, satu mengkritik atau memverifikasi. MetaGPT (arXiv:2308.00352) meresmikan ini sebagai SOP yang dikodekan ke dalam prompt peran — Manajer Produk, Arsitek, Manajer Proyek, Insinyur, Insinyur QA — mengikuti `Code = SOP(Team)`. ChatDev (arXiv:2307.07924) perancang rantai, pemrogram, pengulas, penguji melalui "rantai obrolan" dengan "dehalusinasi komunikatif" (agen secara eksplisit meminta detail yang hilang). Verifikatornya menahan weight: Cemri dkk. (MAST, arXiv:2503.13657) menunjukkan setiap kegagalan multi-agen dapat ditelusuri ke verifikasi yang hilang atau rusak. PwC melaporkan peningkatan akurasi 7× (10% → 70%) dari loop validasi terstruktur di CrewAI.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 16 · 04 (Model Primitif), Fase 16 · 05 (Supervisor)
**Waktu:** ~60 menit

## Masalah

Sistem multi-agen generik menghasilkan output generik. Tiga pembuat code dalam obrolan grup menulis tiga jenis code biasa-biasa saja yang sama. kamu dapat menambahkan lebih banyak agen, menambahkan lebih banyak putaran, dan tetap tidak melewati ambang batas kualitas.

Perbaikannya bukan pada lebih banyak agen — melainkan agen yang *berbeda*. Tetapkan peran yang berbeda. Berikan alat kritik yang tidak dimiliki perencana. Berikan rangkaian pengujian objektif kepada verifikator. Sekarang sistem memiliki ketidaksepakatan internal dengan koreksi beralasan, bukan hanya tebakan paralel.

## Konsep

### Empat peran kanonik

**Perencana.** Membaca sasaran, menghasilkan daftar langkah atau spesifikasi. Alat: pengambilan pengetahuan, dokumen. Output: rencana terstruktur.

**Pelaksana.** Membaca rencana selangkah demi selangkah, menghasilkan artefak. Alat: alat kerja sebenarnya (kompiler code, shell, klien API). Output: artefak.

**Kritikus.** Membaca output eksekutor dibandingkan dengan maksud perencana. Alat: akses hanya baca ke artefak, analisis statis. Output : menerima/menolak disertai alasannya.

**Pemverifikasi.** Membaca artefak dan menjalankan pemeriksaan deterministik. Alat: test runner, pemeriksa tipe, validator skema. Output: lulus/gagal dengan bukti.

Kritikus bersifat subyektif, berpendirian keras, seringkali berbasis LLM. Verifikator bersifat obyektif, deterministik, dan seringkali berbasis code. Peran mereka tidak sama.

### Pola SOP MetaGPT

MetaGPT (arXiv:2308.00352) mengkodekan SOP rekayasa perangkat lunak sebagai prompt peran:

- **Manajer Produk** menulis PRD.
- **Arsitek** menghasilkan desain sistem.
- **Manajer Proyek** membagi tugas.
- **Insinyur** mengimplementasikan.
- **QA Engineer** menjalankan pengujian.

Setiap peran memiliki skema input/output yang ketat. Prompt peran menyatakan apa *peran itu* dan apa yang *harus dihasilkannya*. Formulasi `Code = SOP(Team)` — SOP deterministik mengubah tim LLM menjadi jalur yang dapat diprediksi.

### Dehalusinasi komunikatif ChatDev

ChatDev menambahkan langkah penting: ketika pelaksana memerlukan detail spesifik yang tidak ada dalam rencana, ia secara eksplisit menanyakan kepada perancang sebelum melanjutkan. Hal ini mencegah kegagalan LLM klasik dalam menciptakan detail secara masuk akal.

Implementasi: prompt peran mencakup "ketika kamu memerlukan informasi spesifik yang tidak diberikan kepada kamu, tanyakan nama peran yang relevan sebelum menghasilkan output."

### Mengapa pemverifikasi paling penting

Cemri dkk. (MAST) menelusuri 1.642 kegagalan eksekusi multi-agen. 21,3% merupakan kesenjangan verifikasi — sistem mengirimkan jawaban yang belum diperiksa oleh siapa pun. 79% sisanya sering kali ditelusuri kembali ke "ada pemeriksaan yang gagal secara diam-diam atau tidak pernah dijalankan". Verifikasi adalah peran penahan weight.PwC melaporkan (penerapan CrewAI, 2025) bahwa penambahan loop validasi terstruktur meningkatkan akurasi dari 10% menjadi 70%. 7× keuntungan dari satu peran.

### Kritikus vs pemverifikasi

- Kritikus adalah LLM yang meninjau kualitas artefak. Subyektif. Bisa tertipu oleh prosa yang masuk akal.
- Verifikator adalah program deterministik yang berjalan pada artefak. Tujuan. Memberikan lulus/gagal dengan bukti.

Gunakan keduanya. Kritikus menangkap permasalahan selera yang tidak dapat diungkapkan oleh verifikator. Verifikator menangkap bug yang tidak dapat dilihat oleh kritikus karena bug tersebut hanya muncul saat runtime.

### Anti-pola

Setiap peran di sistem kamu adalah LLM dan output setiap peran "tampak bagus bagi saya". Mode kegagalan MAST klasik. Tambahkan setidaknya satu pemverifikasi yang kelulusan/kegagalannya ditentukan oleh code, bukan oleh LLM.

### Pemetaan framework

- **CrewAI** — `Agent(role, goal, backstory)` adalah permukaan spesialisasi buku teks.
- **LangGraph** — node dapat memiliki prompt khusus; tepi menegakkan pipa.
- **AutoGen** — ConversableAgents khusus peran dengan nama satu kata di GroupChat.
- **OpenAI Agents SDK** — alat serah terima antara Agen dengan peran khusus.

## Build

`code/main.py` mengimplementasikan pipeline 4 peran yang membangun fungsi Python sederhana:

- **Perencana** menghasilkan spesifikasi.
- **Pelaksana** menghasilkan string code.
- **Kritikus** (simulasi LLM) menandai masalah yang jelas.
- **Pemverifikasi** menjalankan code yang dihasilkan di kotak pasir (`exec`) terhadap kasus uji.

Demo berjalan dua kali: sekali ketika eksekutor menghasilkan code yang benar (kritikus + verifikator keduanya lulus), sekali ketika eksekutor menghasilkan code di luar spesifikasi (kritikus melewatkan bug karena terlihat masuk akal, verifikator menangkapnya karena pengujian gagal).

Jalankan:

```
python3 code/main.py
```

## Pakai

`outputs/skill-role-designer.md` mengambil tugas dan menghasilkan daftar peran (3-5 peran), skema input/output per peran, dan pemeriksaan verifikasi. Gunakan ini sebelum memasukkan agen ke dalam framework.

## Kirim

Daftar periksa:

- **Setidaknya satu pemverifikasi deterministik.** Tidak pernah semuanya-LLM.
- **Skema I/O eksplisit per peran.** Perencana mengembalikan spesifikasi, bukan prosa; pelaksana membaca skema itu.
- **Dehalusinasi komunikatif.** Pelaksana harus bertanya kepada perencana jika ada informasi yang hilang; jangan pernah menciptakannya.
- **Pengurutan kritik/verifikator.** Jalankan kritik terlebih dahulu (murah, mengatasi masalah desain), verifikasi kedua (lambat, menangkap bug).
- **Loop budget.** Maksimum 2 putaran revisi kritikus-pelaksana sebelum ditingkatkan ke manusia.

## Latihan

1. Jalankan `code/main.py` dan amati bagaimana verifikator menangkap bug yang terlewatkan oleh kritikus. Tambahkan pemeriksaan analisis statis (hitung kemunculan `return`) sebagai pemverifikasi tambahan. Apa yang menarik dari tes runtime yang terlewatkan?
2. Tambahkan peran ke-5: "analis persyaratan" yang menerjemahkan keinginan pengguna menjadi spesifikasi siap perencana. Permintaan dehalusinasi komunikatif apa yang harus diajukan?
3. Baca MetaGPT Bagian 3 ("Agen"). Buat daftar skema input/output dari masing-masing 5 peran MetaGPT.
4. Baca diagram rantai obrolan ChatDev (arXiv:2307.07924 Gambar 3). Identifikasi di mana dehalusinasi komunikatif memutus lingkaran yang seharusnya tidak ada habisnya.
5. Peningkatan akurasi 7× PwC berasal dari putaran verifikasi. Buat hipotesis tiga tugas di mana penambahan verifikator tidak akan membantu — di mana pemeriksaan kebenaran secara deterministik tidak mungkin dilakukan atau sangat mahal.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Spesialisasi peran | "Agen berbeda, pekerjaan berbeda" | System prompt yang berbeda disesuaikan untuk peran perencana/pelaksana/kritikus/verifikator. |
| Pola SOP | "Prosedur operasi standar yang dikodekan" | Pembingkaian MetaGPT: skema I/O yang ketat per peran mengubah tim menjadi pipeline. |
| Dehalusinasi komunikatif | "Bertanyalah sebelum menciptakan" | Pola ChatDev: pelaksana bertanya kepada perencana ketika ada detail yang hilang daripada mengada-ada. |
| Kritikus | "Pengulas LLM" | Pengulas yang subyektif dan berpendirian keras. Menangkap masalah rasa. Bisa tertipu oleh prosa yang masuk akal. |
| Verifikator | "Pemeriksaan deterministik" | Lulus/gagal berbasis code. Pelari pengujian, pemeriksa tipe, validator skema. Tidak bisa dibodohi. |
| Kesenjangan verifikasi | "Tidak ada yang memeriksa" | 21,3% kegagalan MAST. Jawaban dikirimkan tanpa cek yang dapat mendeteksi bug. |
| Lingkaran revisi | "Kritikus mengirimkannya kembali" | Penolakan kritik memicu pelaksana menjalankan kembali dengan umpan balik. Membutuhkan anggaran. |
| Anti-pola All-LLM | "Terlihat bagus untukku" | Setiap peran adalah LLM, tidak ada pemeriksaan deterministik. Kegagalan MAST klasik. |

## Bacaan Lanjutan

- [Hong dkk. — MetaGPT: Pemrograman Meta untuk Kolaborasi Multi-Agen](https://arxiv.org/abs/2308.00352) — makalah referensi SOP-as-role-prompt
- [Qian dkk. — Agen Komunikatif untuk Pengembangan Perangkat Lunak (ChatDev)](https://arxiv.org/abs/2307.07924) — rantai obrolan + dehalusinasi komunikatif
- [Cemri dkk. — Mengapa Sistem LLM Multi-Agen Gagal?](https://arxiv.org/abs/2503.13657) — Taksonomi MAST; kesenjangan verifikasi adalah 21,3% dari kegagalan
- [Dokumen CrewAI — Peran agen](https://docs.crewai.com/en/introduction) — permukaan spesifikasi peran produksi
