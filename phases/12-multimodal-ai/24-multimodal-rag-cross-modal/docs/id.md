# Multimodal RAG dan Pengambilan Lintas Modal

> RAG dokumen vision-asli adalah satu bagian. RAG multimodal produksi berjalan lebih luas — pengambilan teks, gambar, audio, dan video untuk alur kerja seperti perencanaan perjalanan ("temukan saya makan siang vegan yang tenang dengan cahaya alami"), triase medis ("cedera apa yang cocok dengan foto ini + catatan ini"), e-commerce ("pakaian yang mirip dengan selfie ini, dalam ukuran saya"), dan layanan lapangan ("diagnosis suara mesin ini ditambah foto bagiannya"). Tiga survei tahun 2025 — Abootorabi dkk., Mei dkk., Zhao dkk. — mengkodifikasikan sub-masalah: pengambilan lintas modal, penggabungan pengambilan, landasan pembangkitan, evaluasi multimodal. Lesson ini membaca survei dan merancang jalur produksi.

**Type:** Build
**Language:** Python (stdlib, cross-modal retriever dengan fusion + generator ground)
**Prerequisites:** Fase 12 · 23 (ColPali), Fase 11 (dasar-dasar RAG)
**Waktu:** ~180 menit

## Tujuan Pembelajaran

- Desain pengambilan lintas modal: teks → gambar, gambar → teks, audio → video, dll.
- Bandingkan tiga strategi fusi: fusi skor, fusi berbasis attention, fusi MoE.
- Jelaskan landasan generasi: seperti apa "kutip sumber kamu" ketika sumber merupakan campuran modalitas.
- Sebutkan tiga survei RAG multimoda kanonik tahun 2025 dan taksonomi submasalahnya.

## Masalah

RAG modalitas tunggal adalah pola yang diselesaikan: embed kueri, embed potongan, mengambil, memasukkan ke dalam LLM. RAG multimodal membutuhkan:

1. Beberapa kepala pengambilan (setiap modalitas memerlukan embedding di ruang yang kompatibel).
2. Penggabungan hasil pengambilan antar modalitas.
3. Generasi landasan yang mengutip sumber-sumber di seluruh modalitas.
4. Metrik evaluasi yang mencakup sinyal lintas modal.

Semua survei tahun 2025 mempunyai taksonomi yang sama.

## Konsep

### Pengambilan lintas modal

Ambil dokumen modalitas B dengan kueri modalitas A. Tiga pola:

1. Ruang embedding bersama. CLIP dan CLAP menghasilkan embedding teks + gambar / teks + audio di ruang bersama. Kesamaan kosinus antar modalitas bekerja secara langsung. Terbatas pada pasangan terlatih CLIP.

2. Encoder per modalitas + terjemahan. Encoder teks + encoder gambar + modul penerjemah kecil yang memetakan antar spasi. Sen2Sen oleh Gupta dkk. dan desain 2024 lainnya. Fleksibel tetapi menambah kompleksitas.

3. VLM sebagai pembuat enkode. Gunakan status tersembunyi VLM sebagai representasi pengambilan. Modalitas apa pun yang didukung VLM berfungsi. Kualitas lebih tinggi, lebih mahal.

Pilihan: CLIP / SigLIP 2 untuk teks+gambar; CLAP untuk teks+audio; Status tersembunyi VLM untuk lintas modal dengan kualitas perbatasan.

### Strategi fusi

kamu mengambil 10 hasil: 5 gambar, 3 bagian teks, 2 klip audio. Bagaimana cara menggabungkannya?

Skor fusi (termurah). Setiap modalitas memiliki retrievernya sendiri, masing-masing mengembalikan skor. Normalisasikan skor dalam modalitas lalu jumlahkan. Sederhana, sering kali berhasil.

Fusi berbasis attention. Gabungkan semua item yang diambil, biarkan jaringan attention kecil memberi weight pada item tersebut. Perlu training.

fusi MoE. Rute jaringan gerbang ke pakar khusus modalitas. Jenis kueri yang berbeda memiliki rute yang berbeda — pertanyaan visual memberi weight lebih tinggi pada gambar.

Default produksi: fusi skor dengan sedikit bias terhadap modalitas dominan kueri. Tingkatkan ke MoE jika A/B menunjukkan kemenangan yang jelas pada domain kamu.

### Generasi landasan

LLM harus menyebutkan item mana yang diambil yang mendorong setiap klaim. Untuk multimodal:

- Sumber teks: kutipan standar `[1]`.
- Sumber gambar: `[img 3]` dengan keterangan singkat.
- Audio: `[audio 2 at 0:34]`.Latih generator dengan data grounding-aware: setiap klaim dalam target training ditandai dengan indeks sumber. Pada inference, model secara alami mengeluarkan kutipan.

### Survei tahun 2025

Abootorabi dkk. (arXiv:2502.08826, "Tanyakan dalam Modalitas Apa Pun"): taksonomi untuk RAG multimodal. Meliputi pengambilan, fusi, generasi. Cakupan terluas.

Mei dkk. (arXiv:2504.08748, "Survei RAG Multimodal"): berfokus pada tolok ukur subtugas dan mode kegagalan. Berguna untuk desain evaluasi.

Zhao dkk. (arXiv:2503.18016): survei yang berfokus pada visi. Kuat dalam pekerjaan keluarga ColPali.

Membaca ketiganya memberi kamu informasi terkini pada musim semi 2025. Sebagian besar submasalah masih terbuka.

### MuRAG — makalah dasar

MuRAG (Chen et al., 2022) adalah RAG multimodal pertama. Gambar + teks yang diambil dari KB multimodal, menghasilkan jawaban. Menunjukkan kelayakan sebelum gelombang VLM. Sistem modern (REACT, VisRAG, M3DocRAG) dibangun di atasnya.

### Contoh perencana perjalanan produksi

Pertanyaan: "carikan saya makan siang vegan yang tenang dengan cahaya alami."

Pipeline pipa:

1. Decomposition kueri. "tenang" → kata kunci audio/ulasan; "makan siang vegan" → item menu; "cahaya alami" → feature gambar.
2. Ambil per modalitas:
   - Pengambilan teks pada ulasan: "brunch vegan, suasana tenang."
   - Pengambilan gambar pada foto restoran: "cahaya alami, lapang."
   - Pengambilan audio pada klip suara sekitar: "desibel rendah, tanpa musik."
3. Skor sekering. Setiap restoran memiliki skor gabungan.
4. Restoran top-k → Generator VLM dengan semua bukti → jawab dengan kutipan.

Ini jauh melampaui teks-RAG. Setiap modalitas menambah sinyal bahwa teks saja tidak ada.

### RAG multimodal agen

Multi-hop: jika pengambilan pertama tidak menghasilkan jawaban berkeyakinan tinggi, LLM akan memformulasi ulang dan mengambil kembali. Pola RAG agen dari Fase 14 berlaku di sini. Contoh:

- Ambil top-10 awal → LLM menanyakan "terlalu berisik, filter <40 dB" → ambil ulang.
- Ambil gambar → LLM melihat seseorang memiliki menu → mengambil teks menu → jawaban.

Menambah kerumitan namun menangani kueri yang tidak dapat dilakukan oleh pengambilan satu kali pengambilan.

### Evaluasi

Evaluasi lintas modal masih belum matang. Proksi umum:

- Ingat@k per modalitas.
- Akurasi top-k menyatu.
- Kepuasan ujung ke ujung yang dinilai manusia.
- Khusus tugas (pemesanan selesai, pembelian dilakukan).

Tidak ada tolak ukur standar yang mencakup semua modalitas. Sebagian besar makalah mengevaluasi tugas-tugas khusus domain.

## Pakai

`code/main.py`:

- Tiga mock retriever (teks, gambar, audio) yang beroperasi di korpus restoran bersama.
- Penggabungan skor yang menggabungkan skor modalitas dengan weight yang dapat dikonfigurasi.
- Sebuah rintisan generator yang mengeluarkan jawaban akhir dengan kutipan.
- Perulangan agen sederhana yang memformulasi ulang kueri jika keyakinannya rendah.

## Kirim

Lesson ini menghasilkan `outputs/skill-multimodal-rag-designer.md`. Diberikan spesifikasi produk dengan aliran kueri multimodal, desain retriever, fusi, generator, dan evaluasi.

## Latihan

1. Usulkan RAG multimodal triase medis: kueri = foto cedera + gejala teks. Modalitas apa yang diambil dari KB apa?

2. Penggabungan skor adalah penjumlahan tertimbang sederhana. Mode kegagalan apa yang dapat dihindari oleh fusi MoE?

3. Baca taksonomi Abootorabi et al. (Bagian 3). Apa saja tiga sub-masalah kanonik dan bagaimana pemetaannya ke produk pilihan kamu?

4. Rancang spesifikasi evaluasi untuk RAG multimoda perencana perjalanan. Metrik apa yang mencakup ingatan gambar, ingatan audio, dan kebenaran komposit?

5. RAG multi-hop agen memiliki pajak latensi per perjalanan pulang pergi. Pada tingkat kesulitan kueri apa perolehan akurasi membenarkan latensi?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Pengambilan lintas modal | "Permintaan satu modalitas, ambil yang lain" | Kueri teks mengambil gambar; permintaan gambar mengambil teks; membutuhkan ruang bersama atau penerjemah |
| Penggabungan skor | "Gabungkan skor" | Jumlah tertimbang dari skor pengambilan per modalitas; fusi paling sederhana |
| Fusi MoE | "Pakar yang mengarahkan modalitas" | Jaringan Gating memilih skor modalitas mana yang dapat dipercaya per kueri |
| Generasi membumi | "Kutip sumber kamu" | Setiap klaim dalam jawaban ditandai dengan indeks sumber |
| MuRAG | "RAG multimoda pertama" | Makalah tahun 2022 yang menetapkan pola RAG multimodal |
| Multi-hop agen | "Formulasi ulang dan coba lagi" | LLM mengkueri ulang pengambilan ketika kepercayaan lintasan pertama rendah |

## Bacaan Lanjutan

- [Abootorabi dkk. — Tanyakan dalam Modalitas Apa Pun (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei dkk. — Survei RAG Multimodal (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao dkk. — Survei Vision RAG (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen dkk. — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu dkk. — REAKSI (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)
