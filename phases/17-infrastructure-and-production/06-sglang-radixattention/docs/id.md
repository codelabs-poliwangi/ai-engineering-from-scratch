# SGLang dan RadixAttention untuk Weight Kerja Berat Awalan

> SGLang memperlakukan cache KV sebagai sumber daya kelas satu yang dapat digunakan kembali dan disimpan dalam pohon radix. Jika jadwal vLLM meminta FCFS (siapa cepat dia dapat), penjadwal cache-aware SGLang memprioritaskan permintaan dengan awalan bersama yang lebih panjang — yang secara efektif merupakan traversal radix yang mengutamakan kedalaman sehingga cabang-cabang panas tetap berada di HBM. Pada Llama 3.1 8B dengan prompt 1K seperti ShareGPT, SGLang mencapai ~16.200 tok/s hingga ~12.500 vLLM, keunggulan ~29%. Pada weight kerja RAG dengan awalan yang berat, keunggulannya mencapai 6,4x. Pada weight kerja berbentuk kloning suara, tingkat cache hit terhapus 86%. Diterapkan pada 400.000+ GPU pada tahun 2026 di xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS. Masalah adalah angka 6,4x menguap ketika pengurutan awalan tidak konsisten — pengurutan adalah tuas insinyur.

**Type:** Learn
**Language:** Python (stdlib, mainan radix-tree cache + penjadwal cache-aware)
**Prerequisites:** Fase 17 · 04 (vLLM Melayani Internal), Fase 14 (Agentic RAG)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Diagram RadixAttention: bagaimana prefiks disimpan dalam pohon radix dan bagaimana blok KV dibagikan ke seluruh rangkaian yang berakar pada cabang yang sama.
- Jelaskan penjadwalan cache-aware dan mengapa FCFS salah untuk lalu lintas yang banyak awalannya.
- Hitung kecepatan yang diharapkan untuk weight kerja berdasarkan tingkat hit cache awalan dan distribusi panjang prompt.
- Sebutkan disiplin pengurutan cepat yang menjadikan angka 6,4x nyata vs keuntungan yang hilang.

## Masalah

Penyajian klasik memperlakukan prompt setiap permintaan sebagai tidak jelas. Bahkan ketika 5.000 permintaan RAG semuanya dimulai dengan system prompt 2.000 token yang sama ditambah pembukaan pengambilan yang sama, vLLM mengisi awalan 2.000 token tersebut sebanyak 5.000 kali. GPU melakukan pekerjaan yang sama berulang kali.

Pengamatan: prompt dalam weight kerja agen dan RAG hampir selalu memiliki awalan yang panjang. System prompt, skema alat, contoh beberapa contoh, header pengambilan, riwayat percakapan — semuanya berulang di seluruh permintaan. Jika kamu menyimpan cache KV untuk awalan tersebut satu kali dan menggunakannya kembali, kamu tidak akan mengisinya lagi.

RadixAttention melakukan hal ini. Token diindeks di pohon radix; setiap node memiliki blok KV untuk urutan token di jalurnya dari root. Permintaan baru berjalan di pohon: setiap node yang tokennya cocok akan menggunakan kembali blok KV node tersebut. Biaya pengisian awal menjadi sebanding dengan akhiran "baru", bukan permintaan penuh.

Tantangannya adalah penjadwalan. Jika dua permintaan berbagi awalan 2.000 token dan permintaan ketiga hanya berbagi 200 token dengan awalan yang sama, kamu ingin melayani dua permintaan yang telah lama dibagikan secara bersamaan sehingga awalan panjang tetap berada di HBM. FCFS melakukan yang sebaliknya - ia melayani siapa pun yang datang lebih dulu, berpotensi mengeluarkan cabang panas sebelum permintaan awalan panjang berikutnya muncul.

## Konsep

### Pohon radix sebagai indeks KV

Pohon radix (trie kompak) menyimpan urutan token. Setiap node memiliki rentang token dan blok KV dihitung untuk rentang tersebut. Anak-anak memperluas urutan satu atau lebih token.

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

Permintaan baru masuk dengan prompt sistem + "Konteks: <doc A>" + "Pertanyaan: Carol". Penjadwal berjalan: awalan sistem cocok (124 blok digunakan kembali), cabang doc-A cocok (31 blok digunakan kembali), kemudian mengalokasikan blok baru hanya untuk "Pertanyaan: Carol" (4 blok). Biaya pengisian awal: 4 blok token baru. Tanpa pohon: 160 blok. ~40x penghematan pada pengisian awal.

### Penjadwalan yang sadar cache

Penggunaan kembali yang didukung pohon Radix tidak ada gunanya jika cache berputar. Dua kebijakan utama:1. **Pengiriman yang mengutamakan kedalaman**. Saat memilih permintaan berikutnya dari antrian, pilihlah permintaan yang di-root di cabang yang sama dengan set yang sedang berjalan. Ini membuat cabang panas tetap terpasang.
2. **LRU di tingkat cabang, bukan di tingkat blok**. Hapus seluruh cabang (mulai dari daun terpendek yang digunakan) daripada blok individual, sehingga bentuk cache cocok dengan bentuk radix.

FCFS melanggar keduanya. Permintaan berbagi 2.000 token berada di belakang permintaan berbagi 50, lalu cabang dengan 2.000 token akan diusir untuk menerima cabang dengan 50 token.

### Angka patokan yang harus kamu hafal

- Prompt Llama 3.1 8B, H100, ShareGPT 1K: SGLang ~16,200 tok/s vs vLLM ~12,500 (~29% edge).
- RAG dengan banyak awalan (sistem yang sama + dokumen yang sama, pertanyaan yang berbeda-beda): hingga 6,4x di SGLang.
- Weight kerja kloning suara: tingkat hit cache awalan 86,4%.
- Tingkat keberhasilan produksi di seluruh pelanggan SGLang: 50-99% tergantung pada disiplin yang cepat.
- Dikerahkan pada 400.000+ GPU pada tahun 2026.

### Pesanannya Gotcha

Angka 6,4x bergantung pada pengurutan template cepat yang konsisten. Jika klien kamu membuat prompt sebagai `[system, tools, context, history, question]` di beberapa permintaan dan `[system, context, tools, history, question]` di permintaan lainnya, pohon tidak dapat menemukan awalan bersama. Apa yang tampak seperti awalan bersama bagi manusia adalah dua rangkaian berbeda pada pohon radix.

Tuas insinyur: templat prompt kamu adalah kunci cache. Perbaiki pesanannya. Tempatkan segala sesuatu yang tidak dapat diubah (sistem, alat, skema) terlebih dahulu. Letakkan konteks pengambilan berikutnya. Tempatkan pertanyaan pengguna di urutan terakhir. Jangan menyisipkan konten dinamis ke dalam awalan.

Kasus nyata dari penelitian ini: memindahkan konten dinamis dari awalan yang dapat di-cache membutuhkan satu penerapan dari 7% menjadi 74% tingkat cache hit dalam satu perubahan.

### Dimana RadixAttention menang dan kalah

Menang:
- RAG (pembukaan pengambilan yang sama, pertanyaan berbeda-beda).
- Agen (skema alat yang sama, kueri yang berbeda-beda).
- Mengobrol dengan prompt sistem yang panjang.
- Weight kerja suara / penglihatan dengan pembukaan berulang.

Loss (kembali ke throughput tingkat vLLM):
- Pembuatan single-shot dengan petunjuk unik (penyelesaian code, obrolan terbuka tanpa system prompt).
- Prompt dinamis di mana setiap permintaan menyisipkan konten unik ke dalam awalan.

### Mengapa ini merupakan masalah penjadwal, bukan hanya masalah kernel

kamu dapat menerapkan penggunaan kembali KV sebagai trik kernel. Wawasan SGLang adalah bahwa penggunaan kembali hanya bermanfaat jika penjadwal tetap menyimpan cabang panas. Kebijakan "penggunaan kembali jika tersedia" yang naif akan membuat cache berada dalam weight campuran. Penjadwal yang diindeks pohon radix inilah yang mengubah trik kernel menjadi keunggulan produksi 29%.

### Interaksi dengan vLLM

Kedua sistem ini bukanlah pesaing yang ketat. Pada tahun 2026 vLLM menambahkan cache awalan (`--enable-prefix-caching`) dan router yang sadar cache (vLLM Router in Rust). Kesenjangannya tertutup tetapi tidak hilang sepenuhnya — seluruh tumpukan SGLang adalah radix-first; vLLM mencangkokkannya. Untuk weight kerja yang didominasi oleh penggunaan kembali prefiks, SGLang tetap menjadi default. Untuk penyajian tujuan umum tanpa pola awalan yang kuat, vLLM tetap sama atau lebih baik.

## Pakai

`code/main.py` mengimplementasikan cache KV mainan radix-tree ditambah penjadwal dengan dua kebijakan: FCFS dan cache-aware. Menjalankan weight kerja yang sama melalui keduanya, melaporkan tingkat hit cache awalan dan delta throughput. Kemudian jalankan weight kerja "pengurutan acak" untuk menunjukkan keruntuhan 6,4x.

## Kirim

Lesson ini menghasilkan `outputs/skill-radix-scheduler-advisor.md`. Mengingat deskripsi weight kerja (bentuk template cepat, pola pengambilan, jumlah penyewa secara bersamaan), ini menghasilkan resep pemesanan cepat dan persetujuan untuk adopsi SGLang.

## Latihan1. Jalankan `code/main.py`. Bandingkan FCFS dan cache-aware pada weight kerja yang sama. Dari mana datangnya delta — penghematan pengisian awal, penghematan dekode, atau penundaan antrian?
2. Ubah weight kerja sehingga perintahnya diubah secara acak `[system, tools, context]`. Memutarkan lagi. Apa yang terjadi dengan tingkat pencapaian? Mengapa?
3. Hitung biaya HBM untuk menjaga sistem 2.000 token tetap berada sebagai satu cabang radix di Llama 3.1 8B. Bandingkan dengan biaya batch 16 urutan tanpa penggunaan kembali awalan.
4. Bacalah makalah SGLang RadixAttention. Jelaskan dalam tiga kalimat mengapa penggusuran LRU berbentuk pohon mengalahkan LRU berbentuk balok di bawah weight awalan berat.
5. Seorang pelanggan melaporkan hanya 8% tingkat cache hit. Sebutkan tiga kemungkinan penyebab dan diagnostik yang akan kamu jalankan untuk masing-masing penyebab.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| RadixPerhatian | "hal SGLang" | Cache KV diindeks sebagai pohon radix sehingga awalan bersama menggunakan kembali blok |
| Pohon Radix | "coba kompak" | Pohon di mana setiap node memiliki rentang token dan blok KV-nya |
| Penjadwal yang sadar cache | "cabang-panas-pertama" | Penjadwal yang lebih memilih permintaan berbagi cabang residen |
| Tingkat hit cache awalan | "berapa banyak prompt kamu yang gratis" | Sebagian kecil token cepat disajikan dari blok KV yang digunakan kembali |
| FCFS | "siapa cepat dia dapat" | Penjadwalan default yang merusak lokalitas awalan |
| LRU tingkat cabang | "mengusir daun" | Kebijakan penggusuran disesuaikan dengan bentuk radix |
| Pemesanan template cepat | "kunci cache" | Urutan komponen prompt menentukan apa yang dapat dibagikan oleh pohon |
| Embedding system prompt | "awalan penduduk" | Tetap sematkan bagian sistem yang tidak dapat diubah untuk menghindari penggusuran thrash |

## Bacaan Lanjutan

- [SGLang GitHub](https://github.com/sgl-project/sglang) — sumber dan dokumen.
- [Dokumentasi SGLang](https://sgl-project.github.io/) — RadixAttention dan detail penjadwalan.
- [Makalah SGLang — Memprogram Large Language Model Secara Efisien (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — referensi desain.
- [Blog LMSYS — SGLang dengan RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — angka tolok ukur dan alasan penjadwal.
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — implementasi mirip radix milik vLLM, sebagai perbandingan.
