# Memori Hibrid: Vector + Grafik + KV (Mem0)

> Mem0 (Chhikara et al., 2025) memperlakukan memori sebagai tiga penyimpanan secara paralel — vector untuk kesamaan semantik, KV untuk pencarian fakta cepat, grafik untuk penalaran hubungan entitas. Layer penilaian menggabungkan ketiganya pada pengambilan. Ini adalah standar produksi tahun 2026 untuk memori eksternal.

**Type:** Build
**Language:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Blok Letta)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Jelaskan mengapa penyimpanan tunggal (hanya vector, grafik saja, KV saja) tidak cukup untuk memori agen.
- Sebutkan tiga penyimpanan paralel Mem0 dan apa yang dioptimalkan masing-masingnya.
- Jelaskan penilaian fusi Mem0 — relevansi, kepentingan, kekinian — dan mengapa ini merupakan jumlah tertimbang, bukan hierarki.
- Mengimplementasikan memori tiga penyimpanan mainan di stdlib dengan `add()` yang menulis ke ketiganya dan `search()` yang menggabungkan hasil.

## Masalah

Satu penyimpanan salah untuk salah satu dari tiga kelas kueri:

- **Kesamaan semantik** — "apa yang kita diskusikan tentang penyimpangan agen minggu lalu?" Vector menang; KV dan grafik meleset.
- **Pencarian fakta** — "berapa nomor telepon pengguna?" KV menang; vector itu boros, grafiknya berlebihan.
- **Alasan hubungan** — "pelanggan mana yang berbagi entitas penagihan yang sama?" Grafik menang; vector dan KV tidak bisa menjawab.

Agen produksi mengeluarkan ketiganya dalam satu sesi. Memori penyimpanan tunggal selalu salah untuk keduanya. Kontribusi Mem0 adalah menghubungkan ketiganya di belakang satu permukaan `add`/`search` dengan fungsi penilaian yang menggabungkannya.

## Konsep

### Tiga toko secara paralel

Mem0 (arXiv:2504.19413, April 2025) di `add(text, user_id, metadata)`:

1. Ekstrak fakta kandidat dari teks (langkah yang didorong oleh LLM).
2. Tulis setiap fakta ke penyimpanan vector (embedding) untuk pencarian semantik.
3. Tulis setiap fakta ke penyimpanan KV dengan kunci (user_id, fact_type, entitas) untuk pencarian O(1).
4. Tulis setiap fakta ke penyimpanan grafik (Mem0g) sebagai tepi yang diketik untuk kueri hubungan.

Di `search(query, user_id)`:

1. Penyimpanan vector mengembalikan top-k dengan embed kosinus.
2. Penyimpanan KV mengembalikan klik langsung yang dikunci berdasarkan kueri (id_pengguna, jenis, entitas).
3. Penyimpanan grafik mengembalikan subgraf yang dapat dijangkau dari entitas kueri.
4. Layer penilaian menggabungkan ketiganya.

### Penilaian fusi

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Relevansi** — kosinus vector, pencocokan tepat KV, weight jalur grafik.
- **Pentingnya** — ditandai pada waktu penulisan atau dipelajari (beberapa fakta lebih penting: nama, ID, kebijakan).
- **Recency** — peluruhan eksponensial dari waktu ke waktu sejak terakhir kali ditulis atau dibaca.

Weight disetel per produk. `w_recency` lebih tinggi untuk agen obrolan; `w_importance` yang lebih tinggi untuk agen kepatuhan; lebih tinggi `w_relevance` untuk agen pengambilan.

### Mem0g dan penalaran sementara

Mem0g menambahkan pendeteksi konflik. Ketika sebuah fakta baru bertentangan dengan sisi yang sudah ada, sisi yang ada akan ditandai sebagai tidak sah namun tidak dihapus. Kueri temporal ("apa kota pengguna pada bulan Maret?") melintasi subgraf valid-at-time.

Ini adalah perilaku tingkat kepatuhan yang digeneralisasikan oleh pola pembatalan Letta.

### Nomor patokan

Laporan makalah Mem0 (2025):

- **LoCoMo** (memori percakapan jangka panjang): 91.6
- **LongMemEval** (memori episodik cakrawala panjang): 93.4
- **BEAM 1M** (patokan memori token 1M): 64.1

Garis dasar perbandingan (LLM 128k konteks penuh, penyimpanan vector datar, KV datar) semuanya kalah 10+ poin. Tolok ukur saja tidak membenarkan pilihan – bentuk operasionalnya – tetapi angka-angka menunjukkan bahwa desain fusi bukanlah kesalahan pembulatan.### Taksonomi cakupan

Mem0 membagi memori berdasarkan cakupan:

- **Memori pengguna** — tetap ada di seluruh sesi, dikunci pada `user_id`.
- **Memori sesi** — tetap ada dalam satu thread.
- **Memori agen** — status instans per agen.

Setiap penulisan mengambil satu cakupan. Pengambilan dapat melakukan kueri di seluruh cakupan dengan weight per cakupan. Mencampur cakupan tanpa berpikir adalah cara kamu mendapatkan insiden "asisten memberi tahu Alice tentang proyek Bob".

### Dimana letak kesalahan pola ini

- **Embed penyimpangan.** Hasil vector yang terlihat tepat pada seratus kueri pertama menurun seiring bertambahnya korpus. Tambahkan embedding ulang secara berkala dari N catatan yang paling banyak digunakan.
- **Skema KV merayap.** `(user_id, type, entity)` terlihat sederhana sampai setiap tim menambahkan `type` mereka sendiri. Audit jenis yang ditetapkan setiap tiga bulan.
- **Ledakan grafik.** Satu ekstraktor berisik menambahkan 50 tepi per pesan. Penulisan grafik batas per `add` panggilan; jatuhkan tepi kepercayaan rendah.

## Build

`code/main.py` mengimplementasikan pola tiga toko di stdlib:

- `VectorStore` — kesamaan token-overlap yang naif sebagai pengganti embedding.
- `KVStore` — dikte dimasukkan pada `(user_id, fact_type, entity)`.
- `GraphStore` — tepian yang diketik (subjek, relasi, objek, valid).
- `Mem0` — fasad tingkat atas dengan `add()`, `search()`, penilaian fusi, dan pengambilan sadar cakupan.
- Jejak yang berhasil pada percakapan multi-pengguna dan multi-sesi.

Jalankan:

```
python3 code/main.py
```

Outputnya menunjukkan tiga jalur penarikan terpisah ditambah top-k yang menyatu. Balikkan weight penilaian di bagian atas `main()` dan lihat perubahan peringkatnya.

## Pakai

- **Mem0 (Apache 2.0)** — siap produksi. Host mandiri dengan Postgres + Qdrant + Neo4j, atau gunakan cloud terkelola.
- **Letta** — inti/penarikan/arsip tiga tingkat; bawa backend vector dan grafik kamu sendiri.
- **Zep** — alternatif komersial dengan KG temporal dan ekstraksi fakta.
- **Buatan khusus** — saat kamu memerlukan kontrol yang tepat atas ekstraktor (kepatuhan) atau weight fusi (agen suara yang didominasi keterkinian).

## Kirim

`outputs/skill-hybrid-memory.md` menghasilkan perancah memori tiga penyimpanan dengan pencetak gol fusi, taksonomi cakupan, dan pembatalan sementara yang terhubung.

## Latihan

1. Ganti kemiripan vector mainan dengan model embedding nyata (Transformer kalimat, Ollama, embedding OpenAI). Ukur recall@10 pada percakapan panjang sintetis. Apakah peringkatnya melayang lebih dari 1000 penulisan?
2. Tambahkan kueri sementara: `search(query, as_of=timestamp)`. Kembalikan hanya catatan yang valid pada atau sebelum waktu tersebut. Toko mana yang paling membutuhkan pekerjaan?
3. Menerapkan pendeteksi konflik: jika fakta yang masuk bertentangan dengan tepi grafik, batalkan tepi yang lama dan catat keduanya. Uji pada "pengguna tinggal di Berlin" -> "pengguna tinggal di Lisbon."
4. Port pencetak angka fusi untuk menyertakan dimension `user_feedback` (yang disukai pada catatan yang diambil). Bagaimana kamu mencegah permainan (agen hanya mengembalikan catatan yang sudah disukainya)?
5. Baca dokumen Mem0 (`docs.mem0.ai`). Pindahkan mainan ke `mem0` panggilan klien. Bandingkan kualitas pengambilan pada 20 kueri pengujian yang sama.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Memori hibrida | "Vector ditambah grafik ditambah KV" | Tiga penyimpanan ditulis secara paralel, digabungkan pada pengambilan |
| Ekstraksi fakta | "Penyerapan memori" | Langkah LLM yang memecah teks menjadi tupel (entitas, relasi, fakta) |
| Penilaian fusi | "Peringkat relevansi" | Jumlah tertimbang dari relevansi, kepentingan, kekinian |
| Ruang Lingkup | "Ruang nama memori" | pengguna / sesi / agen — menentukan siapa yang melihat apa |
| anggota | "Grafik memori" | Tepi yang diketik dengan validitas temporal untuk kueri hubungan |
| Pembatalan sementara | "Hapus sementara" | Tandai tepi yang bertentangan tidak valid; jangan pernah menghapus |
| Menanamkan penyimpangan | "Pengambilan busuk" | Kualitas vector menurun seiring dengan pertumbuhan korpus; sematkan kembali secara berkala |

## Bacaan Lanjutan

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — makalah asli
- [Dokumen Mem0](https://docs.mem0.ai/platform/overview) — API produksi, SDK, cloud terkelola
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — pendahulunya dalam konteks virtual
- [Letta, blog Memory Blocks](https://www.letta.com/blog/memory-blocks) — desain saudara tiga tingkat
