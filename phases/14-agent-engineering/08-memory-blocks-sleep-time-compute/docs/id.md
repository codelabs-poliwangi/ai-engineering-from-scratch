# Blok Memori dan Perhitungan Waktu Tidur (Letta)

> MemGPT menjadi Letta pada tahun 2024. Evolusi tahun 2026 menambahkan dua ide: blok memori fungsional diskrit yang dapat diedit secara langsung oleh model, dan agen waktu tidur yang mengkonsolidasikan memori secara asinkron saat agen utama tidak aktif. Inilah cara kamu meningkatkan memori lebih dari satu percakapan.

**Type:** Build
**Language:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Sebutkan tiga tingkatan memori yang digunakan Letta (inti, penarikan kembali, arsip) dan peran masing-masing.
- Jelaskan pola blok memori: Blok manusia, blok Persona, dan blok yang ditentukan pengguna sebagai objek yang diketik kelas satu.
- Jelaskan apa yang dimaksud dengan komputasi waktu tidur, mengapa komputasi berada di luar jalur kritis, dan mengapa komputasi dapat menjalankan model yang lebih kuat daripada agen utama.
- Menerapkan loop dua agen bernaskah di mana agen utama memberikan respons dan agen waktu tidur menggabungkan blok antar putaran.

## Masalah

MemGPT (Lesson 07) memecahkan aliran kontrol memori virtual. Tiga masalah produksi muncul:

1. **Latensi.** Setiap operasi memori berada di jalur kritis. Jika agen harus memangkas, meringkas, atau merekonsiliasi sementara pengguna menunggu, latensi ekor akan meningkat.
2. **Memori membusuk.** Tulisan menumpuk. Fakta-fakta yang bertentangan tetap ada. Pengambilan tenggelam dalam konten basi.
3. **Kehilangan struktur.** Penyimpanan arsip datar tidak dapat menyatakan "blok Manusia selalu ada di prompt; blok Persona selalu ada di prompt; blok Tugas bertukar per sesi."

Letta (letta.com) adalah penulisan ulang tahun 2026. Blok memori membuat struktur menjadi eksplisit; komputasi waktu tidur memindahkan konsolidasi keluar dari jalur kritis.

## Konsep

### Tiga tingkatan

| Tingkat | Ruang Lingkup | Tempat tinggalnya | Ditulis oleh |
|------|-------|----------------|------------|
| Inti | Selalu terlihat | Di dalam prompt utama | Panggilan alat agen + penulisan ulang waktu tidur |
| Ingat | Riwayat percakapan | Dapat diambil | Pencatatan giliran otomatis |
| Arsip | Fakta sewenang-wenang | Vector + KV + grafik | Panggilan alat agen + penyerapan waktu tidur |

Inti adalah inti MemGPT. Penarikan kembali adalah penyangga percakapan dengan ekornya yang dikeluarkan. Arsip adalah penyimpanan eksternal. Perpecahan ini membersihkan kelebihan weight dua tingkat MemGPT.

### Blok memori

Blok adalah bagian tingkat inti yang diketik, persisten, dan dapat diedit. Makalah MemGPT asli mendefinisikan dua:

- **Blok manusia** — fakta tentang pengguna (nama, peran, preferensi, tujuan).
- **Blok Persona** — konsep diri agen (identitas, nada, batasan).

Letta menggeneralisasi ke blok sewenang-wenang yang ditentukan pengguna: blok `Task` untuk tujuan saat ini, blok `Project` untuk fakta basis code, blok `Safety` untuk batasan sulit. Setiap blok memiliki `id`, `label`, `value`, `limit` (batas karakter), `description` (sehingga model mengetahui kapan harus mengeditnya).

Blok dapat diedit melalui permukaan alat:

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` — memadatkan blok yang mendekati batasnya.

### Komputasi waktu tidur

Penambahan Letta 2025: menjalankan agen kedua di latar belakang, keluar dari jalur kritis. Agen waktu tidur memproses transkrip percakapan dan konteks basis code, menulis `learned_context` ke dalam blok bersama, dan mengkonsolidasikan atau membatalkan catatan arsip.

Properti yang rontok:- **Tanpa biaya latensi.** Respons primer tidak menunggu operasi memori.
- **Model yang lebih kuat diperbolehkan.** Agen waktu tidur bisa menjadi model yang lebih mahal dan lebih lambat karena tidak dibatasi latensi.
- **Jendela konsolidasi alami.** Dedup, rangkum, batalkan fakta yang bertentangan saat pengguna tidak menunggu.

Bentuknya sesuai dengan cara manusia bekerja: kamu mengerjakan tugas, kamu tidur di atasnya, ingatan jangka panjang menetap dalam semalam.

### Letta V1 dan penalaran asli

Letta V1 (`letta_v1_agent`, 2026) tidak lagi menggunakan `send_message`/detak jantung dan token `Thought:` sebaris demi mendukung penalaran asli. Responses API (OpenAI) dan Messages API dengan pemikiran yang diperluas (Anthropic) memancarkan penalaran pada pipeline terpisah, melewati putaran (dienkripsi di seluruh penyedia dalam produksi). Loop kontrol masih ReAct. Jejak pemikiran bersifat struktural, bukan berbentuk cepat.

### Dimana letak kesalahan pola ini

- **Blokir kembung.** Infinite `block_append` mencapai batas dengan cepat. Hubungkan peringkas blok sebelum penulisan yang melewati tutupnya.
- **Penyimpangan senyap.** Agen waktu tidur menulis ulang sebuah blok dan agen utama tidak pernah menyadarinya. Blok versi dan perbedaan permukaan dalam jejak.
- **Konsolidasi beracun.** Agen waktu tidur memproses konten yang dapat dijangkau penyerang menjadi inti. Lesson 27 juga berlaku pada permukaan waktu tidur.

## Build

`code/main.py` mengimplementasikan:

- `Block` — id, label, nilai, batas, deskripsi.
- `BlockStore` — CRUD + `near_limit(label)` pembantu.
- Dua agen bernaskah — `PrimaryAgent` melakukan satu giliran, `SleepTimeAgent` melakukan konsolidasi antar putaran.
- Jejak yang menunjukkan percakapan tiga putaran dengan penulisan blok, ditambah waktu tidur yang merangkum blok dan membatalkan fakta yang sudah usang.

Jalankan:

```
python3 code/main.py
```

Transkrip menunjukkan perpecahan: putaran primer cepat dan menghasilkan tulisan mentah; tiket tidur memadat dan membersihkan.

## Pakai

- **Letta** (letta.com) untuk implementasi referensi. Cloud yang dihosting sendiri atau dikelola.
- **Keterampilan SDK Agen Claude** sebagai pengetahuan berbentuk blok — keterampilan adalah blok instruksi yang diberi nama, diberi versi, dan dapat diambil yang dimuat oleh agen sesuai permintaan.
- **Buatan khusus** untuk tim yang ingin mengontrol backend penyimpanan. Gunakan kontrak Letta API agar kamu dapat bermigrasi nanti.

## Kirim

`outputs/skill-memory-blocks.md` menghasilkan sistem blok berbentuk Letta dengan kait waktu tidur untuk runtime apa pun, termasuk peraturan keselamatan dan kabel kutipan.

## Latihan

1. Tambahkan alat `block_summarize` yang menggantikan nilai blok dengan ringkasan yang dihasilkan model ketika `near_limit` mengembalikan nilai true. Ambang pemicu manakah yang meminimalkan panggilan ringkasan dan luapan blok?
2. Menerapkan dedup waktu tidur pada arsip: dua catatan yang teksnya memiliki >90% token tumpang tindih diciutkan menjadi satu. Lakukan hanya pada jalur tidur, jangan pernah pada jalur kritis.
3. Blok versi. Pada setiap penulisan, catat nilai lama dan selisihnya. Ekspos `block_history(label)` sehingga operator dapat melakukan debug "mengapa agen melupakan X."
4. Perlakukan agen waktu tidur sebagai penulis yang tidak dapat dipercaya. Ketika mereka menyentuh blok Persona atau Keamanan, mintalah peninjauan agen kedua sebelum melakukan.
5. Porting contoh untuk menggunakan Letta API (`letta_v1_agent`). Apa yang berubah dalam skema blok, dan bagaimana penalaran asli mengubah bentuk jejak?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Blok memori | "Bagian prompt yang dapat diedit" | Segmen memori inti yang diketik, persisten, dan dapat diedit LLM |
| Blok manusia | "Memori pengguna" | Fakta tentang pengguna, di-embed di inti |
| Blok persona | "Identitas Agen" | Konsep Diri, Nada, Kendala, di-embed pada inti |
| Komputasi waktu tidur | "Memori asinkron berfungsi" | Agen kedua melakukan konsolidasi keluar jalur kritis |
| Inti / Penarikan / Arsip | "Tingkatan" | Pemisahan memori tiga lapis: selalu terlihat / percakapan / eksternal |
| Batas blok | "Tutup" | Batas karakter per blok; ringkasan kekuatan |
| Penalaran asli | "Pipeline Berpikir" | Output penalaran tingkat penyedia, bukan tingkat prompt `Thought:` |
| Konteks yang dipelajari | "Output tidur" | Fakta yang ditulis agen waktu tidur ke dalam blok bersama |

## Bacaan Lanjutan

- [Letta, blog Memory Blocks](https://www.letta.com/blog/memory-blocks) — pola blok
- [Letta, blog Komputasi Waktu Tidur](https://www.letta.com/blog/sleep-time-compute) — konsolidasi asinkron
- [Letta, Merancang Ulang Lingkaran Agen](https://www.letta.com/blog/letta-v1-agent) — penulisan ulang penalaran asli
- [Packer dkk., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — asal
