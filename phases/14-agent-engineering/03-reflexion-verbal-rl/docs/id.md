# Refleksi: Pembelajaran Penguatan Verbal

> RL berbasis gradient memerlukan ribuan uji coba dan cluster GPU untuk memperbaiki mode kegagalan. Refleksi (Shinn et al., NeurIPS 2023) melakukannya dalam bahasa alami: setelah setiap uji coba gagal, agen menulis refleksi, menyimpannya dalam memori episodik, dan mengkondisikan uji coba berikutnya pada memori tersebut. Ini adalah pola di balik penghitungan waktu tidur Letta, pembelajaran CLAUDE.md dari Claude Code, dan aturan pembelajaran pro-alur kerja.

**Type:** Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 01 (Agent Loop), Fase 14 · 02 (ReWOO)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Sebutkan tiga komponen Refleksi (Aktor, Evaluator, Self-Reflector) dan peran memori episodik.
- Menerapkan loop Refleksi stdlib dengan evaluator biner, buffer refleksi, dan upaya ulang baru.
- Pilih antara sumber umpan balik scalar, heuristik, dan evaluasi diri untuk tugas tertentu.
- Jelaskan mengapa penguatan verbal menemukan kesalahan sehingga RL berbasis gradient memerlukan ribuan percobaan untuk memperbaikinya.

## Masalah

Seorang agen gagal dalam tugasnya. Dalam RL standar kamu akan menjalankan ribuan uji coba lagi, menghitung gradient, memperbarui weight. Mahal, lambat, dan sebagian besar agen produksi tidak memiliki anggaran training untuk setiap kegagalan.

Reflexion (Shinn et al., arXiv:2303.11366) menanyakan pertanyaan yang berbeda: bagaimana jika agen hanya memikirkan mengapa ia gagal dan mencoba lagi dengan pemikiran tersebut dalam promptnya? Tidak ada pembaruan berat badan. Tidak ada gradient. Hanya bahasa alami yang disimpan di antara percobaan.

Hasilnya: di ALFWorld ia mengalahkan ReAct dan baseline lain yang belum disesuaikan. Di HotpotQA, ini lebih baik dibandingkan ReAct. Pada pembuatan code (HumanEval/MBPP) ini menetapkan yang tercanggih pada saat itu. Semua tanpa satu langkah gradient pun.

## Konsep

### Ketiga komponen

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

Ditambah satu struktur data:

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

Satu percobaan menjalankan Aktor. Evaluator menilainya. Jika skornya rendah, Self-Reflector menghasilkan refleksi ("Saya memilih alat yang salah karena saya salah membaca pertanyaan menanyakan tentang X padahal menanyakan tentang Y"). Refleksi masuk ke dalam memori episodik. Uji coba berikutnya dimulai dari awal tetapi melihat refleksinya.

### Tiga tipe evaluator

1. **Scalar** — sinyal biner eksternal. ALFWorld berhasil atau gagal. Tes HumanEval lulus atau gagal. Paling sederhana, sinyal tertinggi.
2. **Heuristik** — tanda tangan kegagalan yang telah ditentukan sebelumnya. "Jika agen melakukan tindakan yang sama dua kali berturut-turut, tandai sebagai macet." “Jika lintasannya melebihi 50 langkah, tandai sebagai tidak efisien.”
3. **Dievaluasi sendiri** — LLM menilai lintasannya sendiri. Dibutuhkan ketika tidak ada kebenaran dasar yang tersedia. Sinyal lebih lemah; berpasangan dengan baik dengan verifikasi berbasis alat (Lesson 05 - CRITIC).

Default tahun 2026 adalah campuran: scalar bila tersedia, evaluasi mandiri jika tidak, heuristik sebagai rel pengaman.

### Mengapa ini menggeneralisasi

Refleksi bukanlah algoritma baru, melainkan sebuah pola bernama. Hampir setiap agen produksi "penyembuhan diri" menjalankan beberapa varian:

- Perhitungan waktu tidur Letta (Lesson 08): agen terpisah merefleksikan percakapan masa lalu dan menulis ke blok memori.
- Pola `CLAUDE.md` / "simpan memori" Claude Code: refleksi ditangkap sebagai pembelajaran, sebelum sesi berikutnya.
- prompt `/learn-rule` pro-alur kerja: koreksi dicatat sebagai aturan eksplisit.
- Node refleksi LangGraph: node yang menilai output dan rute untuk disempurnakan jika diperlukan.

Semua berasal dari wawasan yang sama: bahasa alami adalah media yang cukup kaya untuk membawa "apa yang saya pelajari dari kegagalan" di antara proses.

### Kapan berhasil dan kapan tidakRefleksi bekerja ketika:

- Terdapat sinyal kegagalan yang jelas (kegagalan pengujian, kesalahan alat, jawaban salah).
- Kelas tugas dapat direproduksi (jenis pertanyaan yang sama dapat ditanyakan lagi).
- Refleksi mempunyai ruang untuk memperbaiki arah (anggaran tindakan yang cukup).

Refleksi tidak membantu ketika:

- Agen sudah berhasil pada percobaan pertama.
- Kegagalan bersifat eksternal (jaringan mati, alat rusak) — refleksi pada "jaringan mati" tidak membantu proses selanjutnya.
- Refleksinya berubah menjadi takhayul - menyimpan narasi tentang kegagalan yang terjadi satu kali.

Jebakan tahun 2026: pembusukan memori. Refleksi menumpuk; beberapa sudah usang atau salah; berjalan kembali menjadi lebih lambat seiring bertambahnya buffer episodik. Mitigasi: pemadatan berkala (Lesson 06), TTL pada refleksi, atau agen pembersih waktu tidur terpisah (Letta).

## Build

`code/main.py` mengimplementasikan Refleksi pada puzzle mainan: menghasilkan daftar 3 elemen yang berjumlah target. Aktor mengeluarkan daftar kandidat; Evaluator memeriksa jumlahnya; Reflektor Diri menulis baris tentang apa yang salah. Refleksi tersebut masuk ke dalam memori episodik untuk percobaan berikutnya.

Komponen:

- `Actor` — kebijakan tertulis yang menjadi lebih baik ketika melihat refleksi.
- `Evaluator.binary()` — lulus/gagal pada jumlah target.
- `SelfReflector` — menghasilkan diagnosis kegagalan satu baris.
- `EpisodicMemory` — daftar terbatas dengan semantik TTL.

Jalankan:

```
python3 code/main.py
```

Jejaknya menunjukkan tiga percobaan. Percobaan 1 gagal, refleksi disimpan, percobaan 2 melihat refleksi dan membaik tetapi masih gagal, percobaan 3 berhasil. Bandingkan dengan baseline run (tidak ada refleksi) - ia tetap terhenti pada jawaban percobaan 1.

## Pakai

LangGraph mengirimkan refleksi sebagai pola simpul. Prompt `/memory` dari Claude Code dan `/learn-rule` pro-alur kerja mengeksternalisasi buffer episodik sebagai file penurunan harga. Komputasi waktu tidur Letta menjalankan Self-Reflector pada waktu henti sehingga agen utama tetap terikat latensi. OpenAI Agents SDK tidak mengirimkan Reflexion secara langsung; kamu membangunnya dengan Pagar Pembatas khusus yang menolak lintasan berdasarkan skor dan memori `Session` yang bertahan sepanjang perjalanan.

## Kirim

`outputs/skill-reflexion-buffer.md` membuat dan memelihara buffer episodik dengan penangkapan refleksi, TTL, dan deduplikasi. Mengingat kelas tugas dan kegagalan, ini memancarkan refleksi yang benar-benar membantu percobaan berikutnya (bukan kata umum "lebih berhati-hati").

## Latihan

1. Beralih dari evaluator biner ke scalar yang mengembalikan metrik distance (seberapa jauh dari target). Apakah itu menyatu lebih cepat?
2. Tambahkan TTL 10 percobaan pada refleksi. Apakah refleksi lama menyakitkan atau membantu setelah titik itu?
3. Menerapkan evaluator heuristik: menandai uji coba terhenti jika tindakan yang sama diulangi. Bagaimana cara ini berinteraksi dengan Self-Reflector?
4. Jalankan Refleksi dengan Aktor musuh yang mengabaikan refleksi. Apa rekayasa prompt refleksi minimum yang memaksa Aktor untuk memperhatikannya?
5. Baca Bagian 4 makalah Refleksi di AlfWorld. Mereproduksi peningkatan tingkat keberhasilan 130% secara konseptual: apa kunci delta vs vanilla ReAct?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Refleksi | "Koreksi diri" | Shinn dkk. 2023 — Aktor, Evaluator, Reflektor Diri ditambah memori episodik |
| Penguatan verbal | "Belajar tanpa gradient" | Refleksi bahasa alami sebelum prompt sidang berikutnya |
| Memori episodik | "Refleksi per tugas" | Buffer terbatas dari refleksi sebelumnya untuk satu kelas tugas |
| Penilai scalar | "Sinyal kesuksesan biner" | Lulus/gagal atau skor numerik dari kebenaran dasar |
| Penilai heuristik | "Detektor berbasis pola" | Tanda tangan kegagalan yang telah ditentukan sebelumnya (misalnya loop macet, terlalu banyak langkah) |
| Penilai diri | "LLM-sebagai-hakim atas jejaknya sendiri" | Penggantian sinyal yang lebih rendah ketika tidak ada kebenaran dasar — ​​berpasangan dengan verifikasi berbasis alat |
| Busuk memori | "Refleksi basi" | Buffer episodik terisi dengan entri usang; perbaiki dengan pemadatan/TTL |
| Refleksi waktu tidur | "Refleksi diri asinkron" | Jalankan Self-Reflector keluar dari jalur panas sehingga agen utama tetap cepat |

## Bacaan Lanjutan

- [Shinn dkk., Refleksi: Agen Bahasa dengan Pembelajaran Penguatan Verbal (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — makalah kanonik
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — refleksi asinkron dalam produksi
- [Antropik, Rekayasa konteks yang efektif untuk agen AI](https://www.anthropic.com/engineering/ Effective-context-engineering-for-ai-agents) — mengelola buffer episodik sebagai bagian dari konteks
- [Ikhtisar LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — pola simpul refleksi
