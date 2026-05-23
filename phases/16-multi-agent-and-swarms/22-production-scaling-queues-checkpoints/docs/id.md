# Penskalaan Produksi — Antrian, Pos Pemeriksaan, Daya Tahan

> Menskalakan sistem multi-agen ke ribuan proses bersamaan memerlukan **eksekusi yang tahan lama**. Runtime LangGraph menulis pos pemeriksaan setelah setiap langkah super dikunci oleh `thread_id` (Postgres secara default); pekerja mogok melepaskan sewa dan pekerja lain melanjutkan. Agen bisa tidur tanpa batas menunggu input manusia. **MegaAgent** (arXiv:2408.09955) menjalankan antrean produsen-konsumen per agen dengan tiga status (Idle / Processing / Response) dan koordinasi dua layer (obrolan intra-grup + obrolan admin antar-grup). **Fiber/async** mengalahkan thread per pekerjaan untuk streaming LLM: thread tidak digunakan selama 99% dari waktu menunggu token, fiber secara kooperatif menghasilkan I/O. Counterpoint: "Perangkat Lunak Agen Penskalaan" Ashpreet Bedi mendukung **FastAPI + Postgres + tidak ada yang lain** hingga muatan membuktikan sebaliknya — arsitektur sederhana melangkah lebih jauh dari yang diharapkan. Lesson ini membuat log pos pemeriksaan yang tahan lama, antrean kerja per agen dengan transisi status, demo async-vs-thread, dan menerapkan aturan pragmatis "mulai dari yang sederhana".

**Type:** Learn + Build
**Language:** Python (stdlib, `asyncio`, `sqlite3`)
**Prerequisites:** Fase 16 · 09 (Jaringan Swarm Paralel), Fase 16 · 13 (Memori Bersama)
**Waktu:** ~75 menit

## Masalah

Prototipe sistem multi-agen bekerja pada satu laptop dengan tiga agen dalam loop peristiwa dalam memori. kamu pindah ke produksi:

- Agen kadang-kadang berjalan berjam-jam (penelitian panjang, menunggu manusia dalam lingkaran).
- Proses pekerja macet. Memulai ulang akan kehilangan status.
- Weight puncak rata-rata 10x; kamu memerlukan penskalaan horizontal.
- Pengguna membayar per agen yang dijalankan; kamu memerlukan semantik tepat satu kali untuk mengisi daya.

Perulangan peristiwa dalam memori tidak melakukan semua ini. kamu memerlukan layer eksekusi yang tahan lama di bawahnya. Opsi kanonik tahun 2026 adalah:

1. Mesin alur kerja dengan pos pemeriksaan (Waktu proses Temporal, LangGraph).
2. Antrian pesan dengan penyimpanan negara (Postgres + SQS/RabbitMQ).
3. Kerangka model aktor (produsen-konsumen MegaAgent per agen).
4. FastAPI + Postgres linting tangan (argumen Bedi).

Lesson ini membangun miniatur masing-masing.

## Konsep

### Eksekusi tahan lama, polanya

Mesin eksekusi yang tahan lama mempertahankan status program penuh setelah setiap "langkah" (langkah super, dalam bahasa LangGraph). Saat terjadi kecelakaan:

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

Persyaratan agar ini berfungsi:

- **Status yang dapat diserialisasikan.** Semua status agen harus dapat dipertahankan. Penutupan fungsi dengan koneksi database langsung tidak dapat bertahan.
- **Resume deterministik.** Dengan status dan input yang sama, agen melakukan tindakan yang sama (atau tunduk pada oracle deterministik eksternal untuk panggilan LLM).
- **Efek samping idempoten.** Panggilan eksternal (panggilan alat, pembayaran) harus idempoten atau menggunakan kunci deduplikasi.

LangGraph menulis pos pemeriksaan setelah setiap langkah super; Penulisan sementara setelah setiap aktivitas; Pernyataan ulang menggunakan jurnal yang bersumber dari peristiwa. Ketiganya menerapkan pola yang sama.

### Waktu proses LangGraph

Setiap agen memiliki `thread_id`; negara bagian adalah dikte yang diketik; setiap langkah super menulis satu baris ke tabel pos pemeriksaan. Saat dilanjutkan, runtime diputar ulang dari pos pemeriksaan terakhir, bukan dari awal. Agen dapat `interrupt()` menunggu input dari manusia; runtime tetap ada dan melepaskan pekerja. Ketika input tiba, pekerja mana pun dapat melanjutkan.

Ini adalah referensi desain produksi pada bulan April 2026.

### Antrian per agen MegaAgent

arXiv:2408.09955 menjelaskan eksperimen skala: ribuan agen secara bersamaan dalam satu cluster. Arsitektur:

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```Koordinasi dua lapis memungkinkan percakapan intra-grup terjadi secara padat sementara antar-grup tetap jarang — pola yang digunakan untuk menjaga biaya tetap linier di ribuan agen.

### Async vs thread per pekerjaan

Panggilan LLM terikat pada I/O. Thread yang menunggu token berikutnya 99% menganggur. Thread masing-masing berharga ~1MB RAM; pada 10.000 panggilan bersamaan, itu berarti 10 GB hanya untuk tumpukan.

Serat (Python `asyncio`, Go goroutines, Rust `tokio`) secara kooperatif menghasilkan I/O. 10.000 panggilan yang sama cocok untuk diproses. Pada skala agen LLM, async bukanlah optimization — melainkan arsitektur.

Pengecualian: pasca-pemrosesan yang terikat CPU (embedding, trik tokenizer) masih menginginkan thread atau proses. Pisahkan layer I/O kamu dari layer CPU kamu.

### Tandingan Bedi

"Scaling Agentic Software" (Ashpreet Bedi, 2026) berpendapat bahwa sebagian besar tim melakukan rekayasa berlebihan sebelum mereka mengukur weight. Default pragmatis:

- FastAPI + Postgres.
- Setiap agen yang dijalankan adalah satu baris; negara diperbarui di tempat dengan konkurensi optimis.
- Pekerjaan latar belakang melalui `pg_notify` atau pekerja Seledri sederhana.
- Coba lagi kebijakan dalam code aplikasi.

Untuk memuat di bawah ~100 agen yang dijalankan secara bersamaan pada tugas yang dapat dikelola, sering kali hanya inilah yang kamu perlukan. Tingkatkan versi saat kamu mengukurnya gagal.

Aturannya: terapkan framework eksekusi yang tahan lama ketika kamu menghadapi masalah nyata yang tidak dapat diselesaikan oleh arsitektur sederhana. Adopsi dini menghabiskan waktu pada upacara yang tidak membuahkan hasil.

### Semantik tepat sekali

Untuk menjalankan agen berbayar, kamu memerlukan "efektif sekali" (setidaknya sekali pengiriman + konsumen idempoten). Gerakan rekayasa:

- **Kunci Dedup per proses.** Sertakan dalam setiap panggilan efek samping.
- **Pola kotak keluar.** Efek samping ditulis ke tabel terlebih dahulu, lalu proses terpisah mengeksekusinya. Kedua langkah tersebut idempoten.
- **Transaksi kompensasi.** Jika efek samping berhasil namun penulisan pelacakannya gagal, jadwalkan kompensasi.

Ini adalah pola rekayasa basis data, bukan khusus LLM. Pajak LLM hanya panggilan LLM yang lambat; yang lainnya adalah sistem terdistribusi standar.

### Penerapan pelangi

Sistem penelitian multi-agen Anthropic menggunakan "penerapan pelangi": beberapa versi waktu proses agen dijalankan secara bersamaan sehingga agen yang sudah berjalan lama tidak harus dimatikan pada setiap penerapan code. Canary versi baru pada sepotong lalu lintas; pensiunkan versi lama ketika agen mereka selesai.

Ini adalah standar untuk sistem stateful yang sudah berjalan lama; adaptasi tahun 2026 adalah agen dapat hidup berjam-jam, sehingga siklus penerapan harus mengakomodasi.

### Daftar periksa produksi kanonik

- Status tahan lama (pos pemeriksaan, snapshot, atau kotak keluar + log yang dapat diputar ulang).
- Efek samping idempoten.
- Layer I/O Async untuk panggilan LLM.
- Pengiriman setidaknya sekali dengan dedup.
- Penerapan pelangi/canary untuk weight kerja stateful.
- Observabilitas: jejak per agen, audit langkah super, penghitung percobaan ulang.

## Build

`code/main.py` mengimplementasikan:

- `CheckpointStore` — Log pos pemeriksaan yang didukung SQLite dengan kunci thread-id. Setiap langkah super menambahkan satu baris.
- `run_with_checkpoint(agent, thread_id)` — menyimulasikan crash saat dijalankan; pekerja kedua melanjutkan dari pos pemeriksaan terakhir.
- `AgentQueue` — mesin status Idle / Processing / Response per agen dengan antrian kerja kecil.
- `demo_async_vs_threads()` — menjalankan 500 simulasi "panggilan LLM" secara bersamaan melalui asyncio dan melalui thread; melaporkan jam dinding dan memori puncak (perkiraan).

Jalankan:

```
python3 code/main.py
```Output yang diharapkan: pemeriksaan kembali berhasil setelah simulasi kerusakan; versi async menangani 500 panggilan bersamaan dalam < 1 detik; versi thread membutuhkan waktu beberapa detik dan menggunakan memori lipat lebih banyak per unit bersamaan.

## Pakai

`outputs/skill-scaling-advisor.md` memberikan saran mengenai pilihan eksekusi yang tahan lama: FastAPI + Postgres, runtime LangGraph, Temporal, atau kustom. Dikalibrasi berdasarkan weight, kebutuhan retensi status, dan frekuensi penerapan.

## Kirim

Pengerasan produksi kanonik:

- **Mulai dari yang sederhana (aturan Bedi).** FastAPI + Postgres hingga kamu mengukurnya dengan gagal.
- **Instrumenkan semuanya sebelum mengoptimalkan.** Histogram latensi per proses, waktu per langkah, jumlah percobaan ulang, kategorisasi kegagalan.
- **Pola kotak keluar untuk efek samping.** Terutama pembayaran dan panggilan API eksternal.
- **Rainbow deploy.** Jangan pernah mematikan proses agen dalam penerbangan selama deployment.
- **Mengadopsi mesin eksekusi yang tahan lama (Temporal / LangGraph / Restate) ketika** kamu mengalami masalah tertentu: menunggu selama satu jam, koordinasi lintas wilayah, kebijakan percobaan ulang/kompensasi yang rumit.
- **Async untuk layer I/O.** Thread hanya untuk pasca-pemrosesan yang terikat CPU.

## Latihan

1. Jalankan `code/main.py`. Konfirmasikan resume pos pemeriksaan berfungsi; mengukur perbedaan konkurensi async vs thread.
2. Implementasikan tabel **kotak keluar**: setiap pemanggilan alat menulis ke kotak keluar terlebih dahulu, lalu goroutine/tugas terpisah dijalankan. Verifikasi idempotensi dengan menjalankan panggilan alat dua kali.
3. Simulasikan **penyebaran pelangi**: dua versi runtime secara bersamaan; rutekan setengah dari thread_ids baru ke masing-masing; konfirmasikan bahwa thread dalam penerbangan pada versi lama tidak terputus.
4. Baca dokumen runtime LangGraph (tertaut di bawah). Identifikasi feature runtime mana yang membutuhkan waktu paling lama untuk direplikasi dalam versi FastAPI + Postgres. Apakah itu alasan untuk mengadopsi, atau bisakah kamu menundanya?
5. Baca MegaAgent (arXiv:2408.09955) Bagian 3. Koordinasi dua lapis (obrolan admin intra-grup + antar-grup) bersifat eksplisit. Buat sketsa bagaimana kamu akan memetakannya ke antrean pesan dengan dua kelompok antrean.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Eksekusi tahan lama | "Pertahankan status program" | Mesin menulis status setelah setiap langkah super; pemulihan kecelakaan bersifat deterministik. |
| Langkah super | "Batas transaksional" | Unit kerja antar pos pemeriksaan. Istilah LangGraph. |
| thread_id | "Pengidentifikasi menjalankan agen" | Kunci yang mengikat pos pemeriksaan dan melanjutkan logika. |
| Idempotensi | "Aman untuk mencoba lagi" | Mengulangi efek samping menghasilkan hasil yang sama seperti satu kali percobaan. |
| Pola kotak keluar | "Pisahkan efek samping" | Tulis maksud ke tabel; pelaksana terpisah melakukan dan menandai selesai. |
| Pengiriman setidaknya sekali | "Kemungkinan duplikat" | Semantik antrian pesan; kunci dedup membuat konsumen efektif sekali. |
| Penyebaran pelangi | "Versi yang tumpang tindih" | Beberapa versi runtime secara bersamaan selama weight kerja yang berjalan lama. |
| Serat asinkron | "Koperasi menghasilkan" | Konkurensi mode pengguna; murah dibandingkan dengan thread untuk weight terikat I/O. |
| Pos pemeriksaan | "Status cuplikan" | Status berseri pada batas super-langkah; kunci untuk melanjutkan. |

## Bacaan Lanjutan- [LangChain — Waktu proses di balik agen dalam produksi](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — Desain waktu proses LangGraph
- [MegaAgent](https://arxiv.org/abs/2408.09955) — antrian produsen-konsumen per agen; koordinasi dua lapis di ribuan agen secara bersamaan
- [Matrix](https://arxiv.org/abs/2511.21686) — framework terdesentralisasi dengan antrian pesan sebagai substrat koordinasi
- [Dokumen sementara](https://docs.temporal.io/) — mesin alur kerja referensi untuk eksekusi yang tahan lama
- [Anthropic — Sistem penelitian multi-agen](https://www.anthropic.com/engineering/multi-agent-research-system) — lesson produksi termasuk penerapan pelangi
