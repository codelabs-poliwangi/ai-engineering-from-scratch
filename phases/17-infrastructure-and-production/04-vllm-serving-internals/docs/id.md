# vLLM Melayani Internal: PagedAttention, Continuous Batching, Chunked Prefill

> Dominasi vLLM pada tahun 2026 bertumpu pada tiga default gabungan, bukan satu trik. PagedAttention selalu aktif. Pengelompokan berkelanjutan memasukkan permintaan baru ke dalam kumpulan aktif di antara iterasi dekode. Potongan-potongan pra-pengisian yang dipotong-potong meminta petunjuk yang panjang sehingga token yang didekode tidak akan pernah kelaparan. Nyalakan ketiganya dan Llama 3.3 70B FP8 pada satu H100 SXM5 mendorong 2.200-2.400 tok/s pada 128 secara bersamaan — kira-kira 25% di atas default vLLM sendiri dan 3-4x loop PyTorch yang naif. Lesson ini membaca kernel penjadwal dan attention pada tingkat yang dapat kamu diagram, dan diakhiri dengan mainan batcher berkelanjutan di `code/main.py` yang menjadwalkan pra-pengisian dan dekode seperti yang dilakukan vLLM.

**Type:** Learn
**Language:** Python (stdlib, penjadwal batching berkelanjutan mainan)
**Prerequisites:** Fase 17 · 01 (Pelayanan Model), Fase 11 (Teknik LLM)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Jelaskan PagedAttention sebagai pengalokasi cache KV: blok, tabel blok, dan mengapa fragmentasi tetap di bawah 4% pada weight produksi.
- Diagram pengelompokan berkelanjutan pada tingkat iterasi: bagaimana urutan yang sudah selesai meninggalkan kumpulan dan yang baru bergabung tanpa terkuras.
- Jelaskan potongan isi awal dalam satu kalimat dan beri nama metrik latensi mana yang dilindungi (petunjuk: ini adalah ekor TTFT, bukan berarti throughput).
- Beri nama gotcha vLLM v0.18.0 2026 yang menggigit tim yang memungkinkan setiap optimization sekaligus.

## Masalah

Loop servis PyTorch yang naif menjalankan satu permintaan pada satu waktu: tokenize, prefill, decode hingga EOS, return. Pada satu pengguna ini berfungsi. Di angka seratus, itu adalah antrian orang-orang yang sabar. Perbaikan yang jelas - pengelompokan statis - memasukkan setiap permintaan ke prompt terpanjang di jendela, memasukkan setiap dekode ke output terpanjang yang diharapkan, dan menghentikan seluruh kumpulan pada urutan paling lambat. kamu membayar untuk padding yang tidak pernah kamu gunakan, dan permintaan cepat menunggu permintaan lambat.

vLLM memecahkan tiga masalah sekaligus. PagedAttention menghentikan fragmentasi cache KV yang memakan 60-80% memori GPU seperti yang dilakukan alokasi berdekatan klasik. Pengelompokan berkelanjutan memungkinkan permintaan bergabung dan meninggalkan kumpulan di antara setiap iterasi dekode, sehingga kumpulan selalu penuh dengan pekerjaan nyata. Pra-pengisian yang terpotong memecah prompt 32 ribu token menjadi ~512 token yang disisipkan dengan dekode, sehingga prompt yang panjang tidak membekukan setiap token dekode pada GPU.

Default produksi tahun 2026 adalah ketiganya aktif. kamu perlu memahami apa yang dilakukan masing-masing mode kegagalan karena semua mode kegagalan ada pada penjadwal, bukan model.

## Konsep

### PagedAttention sebagai sistem memori virtual

Cache KV adalah `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element` per urutan. Untuk Llama 3.3 70B dengan 8192 token, itu kira-kira 1,25 GB per urutan di BF16. Jika kamu melakukan pra-reservasi 8192 slot untuk setiap permintaan tetapi rata-rata permintaan hanya menggunakan 1500 token, kamu membuang sekitar 82% HBM yang kamu pesan. Pengelompokan klasik menghasilkan pemborosan ini.

PagedAttention meminjam ide dari memori virtual OS. Cache KV tidak bersebelahan per urutan. Itu dialokasikan dalam blok berukuran tetap (default 16 token). Setiap urutan memiliki tabel blok yang memetakan posisi token logisnya ke ID blok fisik. Ketika sebuah sequence berkembang melampaui blok yang dialokasikan, satu blok lagi ditambahkan. Ketika selesai, balok-baloknya kembali ke kolam.Fragmentasi turun dari 60-80% (klasik) menjadi di bawah 4% (PagedAttention). kamu tidak mengaktifkan PagedAttention dengan sebuah tanda — ini adalah satu-satunya pengalokasi vLLM yang dikirimkan. Kenopnya adalah `--gpu-memory-utilization` (default 0.9), yang memberi tahu vLLM berapa banyak HBM yang harus dicadangkan untuk blok KV setelah memuat weight dan activation.

### Pengelompokan berkelanjutan pada tingkat iterasi

"Pengelompokan dinamis" yang lama menunggu jendela (misalnya 10 mdtk) untuk mengisi kumpulan, lalu menjalankan pra-pengisian + dekode + dekode + dekode hingga setiap urutan selesai. Urutan cepat ditinggalkan lebih awal dan diam sementara GPU menyelesaikan urutan lambat.

Pengelompokan berkelanjutan beroperasi di antara setiap langkah dekode. Panggil kumpulan urutan yang berjalan sebagai daftar `RUNNING`. Pada setiap iterasi:

1. Urutan apa pun di `RUNNING` yang baru saja mencapai EOS atau max_tokens akan dihapus.
2. Penjadwal melihat antrian tunggu. Jika ada blok KV gratis, ia menerima urutan baru (pengisian awal atau dilanjutkan).
3. Forward pass berjalan pada apa pun yang sekarang ada di `RUNNING`, memancarkan satu token baru per urutan.

Ukuran batch tidak pernah diisi ke angka tetap. Urutan pada posisi berbeda dalam outputnya berbagi satu kesatuan yang menyatu ke depan. Pada vLLM 2026 ini disebut `V1 scheduler`. Invarian kunci: penjadwal berjalan satu kali per iterasi dekode, bukan satu kali per permintaan.

### Pra-pengisian yang terpotong melindungi ekor TTFT

Pra-pengisian terikat pada komputasi. Prompt 32k-token pada Llama 3.3 70B membutuhkan ~800 ms prefill murni pada satu H100. Saat pra-pengisian berjalan, dekode token untuk setiap urutan lainnya dalam menunggu batch. Dalam loop penyajian, latensi token pertama (TTFT) dari satu prompt panjang menjadi blip latensi antar-token (ITL) untuk lusinan pengguna lainnya.

Pra-pengisian yang dipecah membagi pra-pengisian menjadi potongan-potongan berukuran tetap (token 512 default) dan menjadwalkan setiap potongan sebagai satu unit. Di antara potongan-potongan, penjadwal dapat memajukan urutan dekode dengan satu token. kamu memperdagangkan hit latensi pra-pengisian absolut yang kecil (beberapa ms per bagian) untuk jitter waktu dekode yang jauh lebih rendah. P99 ITL pada weight campuran turun dari ~50 mdtk menjadi ~15 mdtk dalam tolok ukur yang dipublikasikan.

### Ketiga default berinteraksi

Ketiga feature tersebut mengasumsikan satu sama lain. PagedAttention memberi penjadwal sumber daya KV yang terperinci untuk diperdagangkan. Pengelompokan berkelanjutan memerlukan sumber daya yang terperinci sehingga menerima urutan baru tidak memaksa perombakan global. Pra-pengisian yang dipisah adalah keputusan yang dibuat oleh penjadwal pada daftar `RUNNING` yang sama — ini merupakan satu lagi kebijakan penjadwal, bukan sistem terpisah.

kamu tidak perlu mengetahui setiap bendera. kamu perlu mengetahui apa yang dioptimalkan oleh penjadwal: hasil yang baik di bawah anggaran blok KV, tergantung pada pemotongan pra-pengisian yang dipotong.

### Gotcha 2026 v0.18.0

Di vLLM v0.18.0 kamu tidak dapat menggabungkan `--enable-chunked-prefill` dengan decoding spekulatif model draf (`--speculative-model`). Pengecualian yang terdokumentasi adalah decoding spekulatif GPU N-gram di penjadwal V1. Tim yang mengaktifkan setiap flag tanpa membaca catatan rilis akan mendapatkan error run-time saat startup, bukan regresi ringan. Jika keuntungan spekulatif kamu layak untuk mengaktifkan pra-pengisian terpotong, tinjau kembali pilihan tersebut — jawaban yang tepat pada tahun 2026 sering kali adalah EAGLE-3 tanpa pra-pengisian terpotong, bukan model draf ditambah prefill terpotong yang tidak dapat dikompilasi.

### Nomor yang harus kamu ingat- Llama 3.3 70B FP8, H100 SXM5, 128 bersamaan, ketiganya aktif: 2.200-2.400 tok/s.
- Model yang sama, vLLM default (tanpa potongan awal): ~1.800 tok/dtk.
- Model yang sama, loop maju PyTorch yang naif: ~600 tok/s.
- Limbah fragmentasi KV di bawah PagedAttention pada weight produksi: <4%.
- P99 ITL dalam weight campuran: ~15 mdtk dengan pra-pengisian terpotong, ~50 mdtk tanpa.

### Seperti apa penjadwalnya

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` persis seperti loop ini di stdlib Python dengan jumlah token palsu dan latensi penerusan palsu. Menjalankannya menunjukkan bagaimana pra-pengisian yang dipotong membuat urutan dekode tetap hidup selama pra-pengisian yang lama.

## Pakai

`code/main.py` menyimulasikan penjadwal bergaya vLLM dengan feature yang dapat diubah. Jalankan untuk melihat:

- Mode `NAIVE`: satu permintaan dalam satu waktu, tanpa pengelompokan.
- Mode `STATIC`: pad dan tunggu, pengelompokan klasik.
- Mode `CONTINUOUS`: penerimaan dan rilis tingkat iterasi.
- Mode `CONTINUOUS + CHUNKED`: irisan pra-isi disisipkan dengan dekode.

Outputnya menunjukkan total throughput (token per detik virtual), rata-rata TTFT, dan P99 ITL. Baris `CONTINUOUS + CHUNKED` harus mendominasi pada lalu lintas campuran.

## Kirim

Lesson ini menghasilkan `outputs/skill-vllm-scheduler-reader.md`. Dengan adanya konfigurasi penyajian (ukuran batch, pemanfaatan memori KV, ukuran pra-pengisian yang dipotong, konfigurasi spekulatif), ini menghasilkan diagnosis penjadwal yang menyebutkan mana dari tiga default yang mengalami kemacetan dan apa yang harus disesuaikan.

## Latihan

1. Jalankan `code/main.py`. Bandingkan `STATIC` dengan `CONTINUOUS` pada weight kerja dengan permintaan campuran pendek dan panjang. Dari mana asal kesenjangan throughput — efisiensi pengisian awal, efisiensi dekode, atau latensi ekor?
2. Ubah penjadwal mainan untuk menambahkan `--max-num-batched-tokens`. Berapa nilai yang tepat untuk H100 yang menjalankan Llama 3.3 70B FP8? (Petunjuk: ini adalah fungsi dari ukuran blok KV dan jumlah blok bebas, bukan HBM mentah.)
3. Baca kembali catatan rilis vLLM v0.18.0. Kombinasi bendera manakah yang saling eksklusif? Daftarkan mereka.
4. Hitung pemborosan fragmentasi cache KV untuk jejak 1.000 permintaan dengan rata-rata 1.500 token output, std 600 token, dalam (a) alokasi per permintaan yang berdekatan pada maksimal 8192, (b) PagedAttention dengan blok 16 token.
5. Jelaskan dalam satu paragraf mengapa potongan awal membantu P99 ITL tetapi tidak membantu throughput secara terpisah. Dari mana keuntungan throughput dalam praktiknya berasal?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| PagedPerhatian | "trik KV" | Pengalokasi blok berukuran tetap untuk cache KV; fragmentasi <4% |
| Blokir tabel | "tabel halaman" | Peta per-urutan dari posisi token logis ke blok KV fisik |
| Pengelompokan berkelanjutan | "pengelompokan dinamis, tapi benar" | Keputusan akui/rilis dibuat setiap iterasi decode |
| Isi awal yang dipotong | "pemisahan pra-pengisian" | Pecahkan isi awal yang panjang menjadi irisan 512 token yang disisipkan dengan decode |
| TTFT | "waktu token pertama" | Isi awal + antrian + jaringan; didominasi oleh prefill pada prompt panjang |
| ITL | "latensi antar-token" | Waktu antara token dekode berturut-turut; didominasi oleh ukuran batch |
| output bagus | "throughput yang memenuhi SLO" | Token/detik dimana setiap permintaan masih mencapai target TTFT dan ITL |
| Penjadwal V1 | "penjadwal baru" | penjadwal vLLM tahun 2026; Dekode spesifikasi N-gram adalah jalur yang kompatibel dengan prefill chunked |
| `--gpu-memory-utilization` | "kenop memori" | Fraksi HBM yang dicadangkan untuk blok KV setelah weight dan activation |

## Bacaan Lanjutan- [dokumentasi vLLM — Decoding Spekulatif](https://docs.vllm.ai/en/latest/features/spec_decode/) — sumber resmi tentang kompatibilitas chunked-prefill dan speculative-decoding.
- [Catatan Rilis vLLM (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — irama rilis 2026 dan perilaku spesifik versi.
- [Blog vLLM — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — tulisan asli yang masih menjelaskan cara berpikir tentang pengalokasi.
- [Makalah PagedAttention (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — analisis fragmentasi dan desain penjadwal.
- [Aleksa Gordic — Di dalam vLLM](https://www.aleksagordic.com/blog/vllm) — panduan penjadwal V1 mendetail dengan grafik api.
