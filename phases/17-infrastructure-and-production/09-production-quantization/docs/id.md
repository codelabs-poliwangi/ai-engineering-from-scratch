# Kuantisasi Produksi — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4

> Format kuantisasi bukanlah pilihan universal — ini adalah fungsi perangkat keras, mesin penyajian, dan weight kerja. GGUF Q4_K_M atau Q5_K_M memiliki CPU dan edge, dikirimkan melalui llama.cpp dan Ollama. GPTQ menang di dalam vLLM ketika kamu membutuhkan multi-LoRA di basis yang sama. AWQ dengan kernel Marlin-AWQ menghasilkan ~741 tok/s pada model kelas 7B dengan Pass@1 terbaik di INT4 — default tahun 2026 untuk produksi pusat data. FP8 tetap menjadi titik tengah bagi Hopper, Ada, dan Blackwell — nyaris tanpa loss dan didukung secara luas. NVFP4 dan MXFP4 (Blackwell microscaling) bersifat agresif dan memerlukan validasi per blok. Dua jebakan menggigit tim: dataset kalibrasi harus cocok dengan domain penerapan, dan cache KV terpisah dari kuantisasi weight — lesson AWQ "model saya sekarang 4 GB" melupakan cache KV 10-30 GB pada ukuran batch produksi.

**Type:** Learn
**Language:** Python (stdlib, memori mainan, dan perbandingan throughput lintas format)
**Prerequisites:** Fase 10 · 13 (Dasar kuantisasi), Fase 17 · 04 (vLLM Melayani Internal)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Sebutkan enam format kuantisasi produksi dan sweet spotnya pada tahun 2026.
- Pilih format perangkat keras tertentu (CPU vs GPU, Hopper vs Blackwell), mesin (vLLM, TRT-LLM, llama.cpp), dan weight kerja (obrolan rutin, penalaran, multi-LoRA).
- Hitung berat memori yang disimpan dan cache KV yang tidak tersentuh untuk format yang dipilih.
- Sebutkan kendala himpunan data kalibrasi yang menurunkan model terkuantisasi pada lalu lintas domain.

## Masalah

Kuantisasi mengurangi memori dan bandwidth HBM, yang merupakan kebutuhan dekode. Model FP16 70B memiliki weight 140 GB. Kuantifikasi weight ke INT4 (AWQ atau GPTQ) dan modelnya adalah 35 GB — muat dalam satu H100 dengan ruang untuk cache KV, yang penting karena pada 128 rangkaian bersamaan dengan konteks 2k, cache KV saja adalah 20-30 GB.

Namun kuantisasi tidak gratis. Kuantisasi yang agresif menurunkan kualitas, terutama pada tugas-tugas yang berat. Format berbeda bekerja dengan mesin berbeda. Perangkat keras yang berbeda mendukung presisi yang berbeda pula. Kebun binatang format 2026 adalah nyata dan kamu tidak dapat menyalin pilihan orang lain — kamu harus memilih berdasarkan tumpukan kamu.

## Konsep

### Enam format

| Format | Bit | Titik manis | Mesin |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU, edge, laptop | llama.cpp, Ollama |
| GPTQ | 4-8 | Multi-LoRA di vLLM | vLLM, TGI |
| AWQ | 4 | Produksi GPU pusat data | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Pusat data Hopper/Ada/Blackwell | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell multi-pengguna | TRT-LLM |
| NVFP4 | 4 | Blackwell multi-pengguna | TRT-LLM |

### GGUF — default CPU/edge

GGUF adalah format file, bukan skema kuantisasi — GGUF menggabungkan varian K-quant (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0) dalam satu wadah. Q4_K_M dan Q5_K_M adalah default produksi — kualitas mendekati BF16 pada 4-5 bit. Pilihan terbaik untuk CPU atau edge serve karena llama.cpp sejauh ini merupakan mesin inference CPU tercepat.

Penalti throughput di vLLM: ~93 tok/s pada 7B — formatnya tidak dioptimalkan untuk kernel GPU. Gunakan GGUF ketika target penerapannya adalah CPU/edge. Bukan sebaliknya.

### GPTQ — multi-LoRA di vLLM

GPTQ adalah algoritma kuantisasi pasca training dengan izin kalibrasi. Kernel Marlin membuatnya cepat pada GPU (percepatan 2,6x vs GPTQ non-Marlin). ~712 tok/dtk di 7B.Kemenangan uniknya: GPTQ-Int4 mendukung adaptor LoRA di vLLM. Jika kamu menyajikan model dasar ditambah 10-50 varian yang telah disesuaikan (masing-masing sebagai LoRA), GPTQ adalah jalur kamu. NVFP4 belum mendukung LoRA pada awal tahun 2026.

### AWQ — default GPU pusat data

Kuantisasi Berat yang Sadar Activation. Melindungi ~1% weight paling menonjol selama kuantisasi. Kernel Marlin-AWQ: kecepatan 10,9x vs naif. ~741 tok/dtk pada 7B, Pass@1 terbaik di antara format INT4.

Pilih AWQ untuk penyajian GPU baru kecuali kamu memerlukan multi-LoRA (GPTQ) atau Blackwell FP4 (NVFP4) yang agresif.

### FP8 — lini tengah yang andal

Titik mengambang 8-bit. Hampir tanpa loss. Didukung secara luas. Hopper Tensor Cores mempercepat FP8 secara asli. Blackwell mewarisi. FP8 adalah default aman tahun 2026 ketika kualitas tidak dapat dinegosiasikan (penalaran, medis, gen code). Penghematan memori adalah setengah dari INT4 tetapi risiko kualitas jauh lebih rendah.

### MXFP4 / NVFP4 — Blackwell agresif

Skala mikro FP4. Setiap blok weight memiliki faktor skalanya sendiri. Agresif namun memiliki akselerasi perangkat keras di Blackwell Tensor Cores. Mengurangi separuh byte per token versus FP8 — kemenangan ekonomi di Fase 17 · 07.

Peringatan:
- Belum ada dukungan LoRA (awal 2026).
- Penurunan kualitas terlihat pada weight kerja yang berat.
- Validasi set evaluasi kamu per model.

### Perangkap kalibrasi

AWQ dan GPTQ memerlukan dataset kalibrasi — biasanya C4 atau WikiText. Untuk model domain (code, medis, legal), kalibrasi pada teks web umum memungkinkan algoritme membuat keputusan yang salah tentang weight mana yang harus dilindungi. Pass@1 pada HumanEval dapat menurunkan beberapa poin.

Cara mengatasinya: kalibrasi pada data dalam domain. Ratusan sample domain biasanya cukup. Uji pada set eval sebelum pengiriman.

### Perangkap cache KV

AWQ mengecilkan weight menjadi 4 bit. Cache KV terpisah dan tetap di FP16/FP8. Untuk model 70B dengan AWQ:

- Weight: ~35 GB (INT4 dari 140 GB).
- Cache KV pada 128 konteks bersamaan × 2k: ~20 GB.
- Activation: ~5 GB.
- Total: ~60 GB — muat di H100 80GB.

Secara naif "Saya mengkuantisasi model saya menjadi 4 GB" melupakan 30-50 GB lainnya. Anggaran HBM secara holistik.

Secara terpisah, kuantisasi cache KV (FP8 KV atau INT8 KV) adalah pilihan yang berbeda dengan konsekuensinya sendiri — hal ini memengaruhi akurasi attention secara langsung dan bukan merupakan kemenangan gratis.

### AWQ INT4 berbahaya untuk alasan

Rantai pemikiran, matematika, code-gen dengan konteks yang panjang - semuanya ini jelas mengalami kuantisasi yang agresif. AWQ INT4 kehilangan ~3-5 poin pada MATEMATIKA. Untuk weight kerja yang sangat berat, kirimkan FP8 atau BF16; menerima biaya memori.

### Panduan memilih 2026

- Pelayanan CPU/edge: GGUF Q4_K_M. Selesai.
- Layanan GPU, obrolan rutin, tanpa LoRA: AWQ.
- Penyajian GPU, multi-LoRA: GPTQ dengan Marlin.
- Weight kerja penalaran: FP8.
- Pusat data Blackwell, kualitas tervalidasi: NVFP4 + FP8 KV.
- Ambigu: jalankan evaluasi 1.000 sample pada setiap format kandidat.

## Pakai

`code/main.py` menghitung jejak memori (weight + KV + activation) dan throughput relatif di enam format untuk berbagai ukuran model. Menunjukkan di mana cache KV mendominasi, di mana kompresi berat memberikan manfaat, dan di mana FP8 adalah pilihan yang aman.

## Kirim

Lesson ini menghasilkan `outputs/skill-quantization-picker.md`. Dengan mempertimbangkan perangkat keras, ukuran model, jenis weight kerja, dan toleransi kualitas, pilih format dan buat rencana kalibrasi/validasi.

## Latihan1. Jalankan `code/main.py`. Untuk model 70B pada 128 bersamaan dengan konteks 2k, hitung total HBM untuk setiap format. Format mana yang memungkinkan kamu memuat satu H100 80GB?
2. kamu memiliki model pengkodean 7B. Pilih format dan justifikasi. Jika kamu salah mengenai toleransi kualitas, bagaimana jalur pemulihannya?
3. Hitung ukuran dataset kalibrasi yang diperlukan untuk mengkalibrasi AWQ untuk model domain medis. Mengapa lebih banyak data tidak selalu lebih baik?
4. Baca makalah kernel Marlin-AWQ atau catatan rilis. Jelaskan dalam tiga kalimat mengapa AWQ mencapai 741 tok/s pada 7B sementara GPTQ mentah mencapai ~712.
5. Kapan masuk akal untuk menggabungkan weight AWQ dengan cache FP8 KV vs menjaga KV di BF16?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| GGUF | "format llama.cpp" | Format file yang menggabungkan varian K-quant; CPU/tepi default |
| Q4_K_M | "Q4 K M" | media K-kuant 4-bit; default GGUF produksi |
| GPTQ | "wah kencing tee q" | INT4 pasca-latihan dengan kalibrasi; mendukung LoRA di vLLM |
| AWQ | "a w q" | INT4 yang sadar akan activation; kernel Marlin; Lulus@1 terbaik di INT4 |
| Kernel Marlin | "kernel INT4 cepat" | Kernel CUDA khusus untuk INT4 di Hopper; percepatan 10x |
| FP8 | "pelampung delapan bit" | Default presisi aman di Hopper/Ada/Blackwell |
| MXFP4 / NVFP4 | "skala mikro empat" | Blackwell FP 4-bit dengan faktor skala per blok |
| Dataset kalibrasi | "data kal" | Teks input yang digunakan untuk memilih parameter kuantisasi; harus cocok dengan domain |
| Kuantisasi cache KV | "KV INT8" | Pisahkan pilihan dari weight; mempengaruhi akurasi attention |

## Bacaan Lanjutan

- [Teknologi VRLA — Kuantisasi LLM 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — tolok ukur komparatif.
- [Jarvis Labs — Panduan Lengkap Kuantisasi vLLM](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — angka throughput berdasarkan format.
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — pemilihan format demi format.
- [dokumen vLLM — Kuantisasi](https://docs.vllm.ai/en/latest/features/quantization/index.html) — format dan tanda yang didukung.
- [Makalah AWQ (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — formulasi AWQ asli.
- [Makalah GPTQ (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — formulasi GPTQ asli.
