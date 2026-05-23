# Penggunaan Alat dan Pemanggilan Fungsi

> Toolformer (Schick et al., 2023) memulai anotasi alat yang diawasi sendiri. Berkeley Function Calling Leaderboard V4 (Patil dkk., 2025) menetapkan standar tahun 2026: 40% agen, 30% multi-putaran, 10% siaran langsung, 10% tidak langsung, 10% halusinasi. Putaran tunggal terpecahkan. Memori, pengambilan keputusan yang dinamis, dan rantai alat yang memiliki cakrawala panjang tidak demikian.

**Type:** Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 01 (Agent Loop), Fase 13 · 01 (Pemanggilan Fungsi Lebih Dalam)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Jelaskan sinyal training yang diawasi sendiri oleh Toolformer: pertahankan anotasi alat hanya ketika eksekusi mengurangi kehilangan token berikutnya.
- Sebutkan lima kategori evaluasi BFCL V4 dan apa yang diukur masing-masing kategori tersebut.
- Menerapkan registri alat stdlib dengan validasi skema, pemaksaan argumen, dan sandbox eksekusi.
- Mendiagnosis tiga masalah terbuka tahun 2026: rangkaian alat cakrawala panjang, pengambilan keputusan dinamis, dan memori.

## Masalah

Penggunaan alat awal ditanyakan: dapatkah model memprediksi pemanggilan fungsi yang benar? Penggunaan alat modern bertanya: dapatkah alat rantai model melintasi 40 langkah, dengan memori, dengan kemampuan observasi parsial, dengan pemulihan dari kegagalan alat, tanpa berhalusinasi alat yang tidak ada?

Toolformer menetapkan dasar: model dapat mempelajari kapan harus memanggil alat dengan pengawasan mandiri. BFCL V4 menetapkan target evaluasi tahun 2026. Kesenjangan di antara mereka adalah tempat tinggal agen produksi luar angkasa.

## Konsep

### Pembentuk Alat (Schick dkk., NeurIPS 2023)

Ide: biarkan model memberi anotasi pada korpus prapelatihannya sendiri dengan kandidat panggilan API. Untuk setiap kandidat, jalankan. Simpan anotasi hanya jika menyertakan hasil alat mengurangi loss pada token berikutnya. Sempurnakan korpus yang difilter.

Alat yang dibahas: kalkulator, sistem QA, mesin pencari, penerjemah, kalender. Sinyal pengawasan mandiri murni tentang apakah alat tersebut membantu memprediksi teks — tidak ada label manusia.

Hasil skala: penggunaan alat muncul dalam skala besar. Model yang lebih kecil dirugikan oleh anotasi alat; keuntungan model yang lebih besar. Inilah sebabnya mengapa model frontier 2026 memiliki penggunaan alat yang kuat sementara sebagian besar model 7B memerlukan penyempurnaan penggunaan alat secara eksplisit agar dapat diandalkan.

### Papan Peringkat Pemanggil Fungsi Berkeley V4 (Patil dkk., ICML 2025)

BFCL adalah evaluasi de facto tahun 2026. Komposisi V4:

- **Agentik (40%)** — lintasan agen lengkap: memori, multi-putaran, keputusan dinamis.
- **Multi-Putaran (30%)** — percakapan interaktif dengan rantai alat.
- **Langsung (10%)** — prompt nyata yang dikirimkan pengguna (distribusi lebih sulit).
- **Non-Live (10%)** — kasus uji sintetis.
- **Halusinasi (10%)** — mendeteksi saat tidak ada alat yang harus dipanggil.

V3 memperkenalkan evaluasi berbasis negara: setelah urutan alat, periksa keadaan API yang sebenarnya (misalnya, "apakah file dibuat?") daripada mencocokkan AST dari panggilan alat. V4 menambahkan kategori pencarian web, memori, dan sensitivitas format.

Temuan penting tahun 2026: pemanggilan fungsi satu putaran hampir terselesaikan. Kegagalan terkonsentrasi pada memori (membawa konteks melintasi putaran), pengambilan keputusan dinamis (memilih alat berdasarkan hasil sebelumnya), rantai cakrawala panjang (melayang setelah 20+ langkah), dan deteksi halusinasi (menolak menelepon ketika tidak ada alat yang cocok).

### Skema alat

Setiap penyedia memiliki skema. Detailnya berbeda tetapi memiliki bentuk yang sama:

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```Antropik menggunakan `input_schema` secara langsung. OpenAI menggunakan `function.parameters`. Keduanya menerima Skema JSON. Deskripsi bersifat menahan weight — model membacanya untuk memilih alat yang tepat. Deskripsi alat yang buruk adalah penyebab utama kegagalan pemilihan alat yang salah.

### Validasi argumen

Jangan percaya pada panggilan alat. Validasi:

1. **Jenis paksaan.** Model dapat mengembalikan string "5" yang skemanya menyatakan int. Paksaan jika tidak ambigu; tolak jika tidak.
2. **Validasi enum.** Jika skema menyatakan `status in {"open", "closed"}` dan model menampilkan `"in_progress"`, tolak dengan error deskriptif.
3. **Kolom yang wajib diisi.** Bidang wajib tidak ada -> observasi kesalahan langsung kembali ke model, bukan kerusakan.
4. **Validasi format.** Tanggal, email, URL — validasi dengan parser konkret, bukan regex.

Setiap kegagalan validasi harus mengembalikan observasi terstruktur sehingga model dapat mencoba kembali dengan bentuk yang benar.

### Panggilan alat paralel

Penyedia modern mendukung panggilan alat paralel dalam satu giliran asisten. Lingkaran:

1. Model mengeluarkan 3 panggilan alat dengan `tool_use_id`s yang berbeda.
2. Runtime mengeksekusinya (secara paralel jika independen).
3. Setiap hasil dikembalikan sebagai blok `tool_result` yang dikorelasikan dengan `tool_use_id`.

Aturan teknik: perlakukan ID korelasi sebagai penahan weight. Tukarkan keduanya dan kamu akan mendapatkan perutean alat yang salah ke hasil yang salah.

### Kotak Pasir

Eksekusi alat adalah batas kotak pasir. Lihat Lesson 09 untuk detailnya. Versi singkat: setiap alat harus menentukan permukaan baca/tulis, akses jaringan, batas waktu, batas memori. Generik `run_shell(cmd)` adalah tanda bahaya; spesifik `git_status()` lebih aman.

## Build

`code/main.py` mengimplementasikan registri alat berbentuk produksi:

- Validator subset Skema JSON (hanya stdlib).
- Pendaftaran alat dengan deskripsi, skema input, batas waktu, dan pelaksana.
- Pemaksaan argumen dan validasi enum.
- Pengiriman alat paralel dengan ID korelasi.
- Pengamatan kesalahan sebagai string terstruktur.

Jalankan:

```
python3 code/main.py
```

Jejak tersebut menunjukkan agen mini memanggil tiga alat dalam satu putaran, dengan satu panggilan yang sengaja diubah formatnya namun ditolak dengan kesalahan deskriptif yang dapat ditindaklanjuti oleh model.

## Pakai

Setiap penyedia memiliki skema alatnya sendiri — Anthropic, OpenAI, Gemini, Bedrock. Gunakan layer terjemahan (OpenAI Agents SDK, Vercel AI SDK, adaptor alat LangChain) jika kamu memerlukan multi-penyedia. BFCL adalah tolok ukur referensi — jalankan terhadap agen kamu sebelum pengiriman jika penggunaan alat adalah hal yang penting dalam produk.

## Kirim

`outputs/skill-tool-registry.md` menghasilkan katalog alat, skema, dan registri untuk domain tugas tertentu. Termasuk pemeriksaan kualitas deskripsi (apakah deskripsi setiap alat memberi tahu model kapan harus menggunakannya?).

## Latihan

1. Tambahkan alat "no-op" yang memungkinkan model secara eksplisit menolak menggunakan alat lainnya. Ukur dengan tes halusinasi seperti BFCL.
2. Menerapkan pemaksaan argumen untuk int-as-string dan float-as-string. Di mana paksaan mulai menyembunyikan bug yang sebenarnya?
3. Tambahkan batas waktu per alat dan pemutus sirkuit (tolak alat selama 60 detik setelah 3 kegagalan berturut-turut). Apa perubahan yang terjadi pada pemulihan model?
4. Baca deskripsi BFCL V4. Pilih satu kategori (misalnya "multi-putaran") dan jalankan 10 contoh prompt melalui agen kamu. Laporkan tingkat kelulusan.
5. Port validator stdlib ke Pydantic atau Zod. Apa yang ditangkap Pydantic/Zod yang terlewatkan oleh mainan itu?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Pemanggilan fungsi | "Penggunaan alat" | Pemanggilan alat output terstruktur dengan skema tervalidasi |
| Pembentuk alat | "Anotasi alat yang diawasi sendiri" | Schick 2023 — pertahankan panggilan alat yang hasilnya mengurangi kehilangan token berikutnya |
| BFCL | "Papan Peringkat Panggilan Fungsi Berkeley" | Tolok ukur tahun 2026: 40% agen, 30% multi-putaran, 10% hidup, 10% tidak hidup, 10% halusinasi |
| Skema alat | "Tanda fungsi untuk model" | nama, deskripsi, Skema argumen JSON |
| alat_use_id | "ID Korelasi" | Mengikat panggilan alat ke hasilnya; penting untuk pengiriman paralel |
| Deteksi halusinasi | "Ketahui kapan tidak menelepon" | Kategori V4: menolak menelepon ketika tidak ada alat yang cocok |
| Pemaksaan argumen | "Perbaikan string-ke-int" | Perbaikan sempit untuk ketidakcocokan skema yang dapat diprediksi; tolak jika ambigu |
| Kotak Pasir | "Batas eksekusi alat" | Permukaan baca/tulis per alat, jaringan, batas waktu, batas memori |

## Bacaan Lanjutan

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — anotasi alat yang diawasi sendiri
- [Papan Peringkat Panggilan Fungsi Berkeley (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — tolok ukur evaluasi 2026
- [Antropik, Dokumentasi penggunaan alat](https://platform.claude.com/docs/en/agent-sdk/overview) — skema alat produksi di Claude Agent SDK
- [Dokumen SDK Agen OpenAI](https://openai.github.io/openai-agents-python/) — jenis alat fungsi dan Pagar Pembatas
