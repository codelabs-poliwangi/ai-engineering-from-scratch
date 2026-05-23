# Pilihan Penyajian yang Dihosting Sendiri — llama.cpp, Ollama, TGI, vLLM, SGLang

> Empat mesin mendominasi inference yang dihosting sendiri pada tahun 2026. Pilih berdasarkan perangkat keras, skala, dan ekosistem. **llama.cpp** adalah yang tercepat di CPU — dukungan model terluas, kontrol penuh atas kuantisasi dan threading. **Ollama** adalah instalasi satu prompt dev-laptop, ~15-30% lebih lambat dibandingkan llama.cpp (Go + CGo + serialisasi HTTP), kesenjangan throughput 3x pada weight seperti prod. **TGI memasuki mode pemeliharaan pada 11 Desember 2025** — hanya perbaikan bug, throughput mentah ~10% lebih lambat dibandingkan vLLM tetapi secara historis memiliki observabilitas teratas dan integrasi ekosistem HF. Status pemeliharaan tersebut menjadikannya taruhan jangka panjang yang berisiko — SGLang atau vLLM adalah default yang lebih aman untuk proyek baru. **vLLM** adalah default produksi tujuan umum — v0.15.1 (Februari 2026) menambahkan PyTorch 2.10, RTX Blackwell SM120, optimization H200. **SGLang** adalah spesialis multi-putaran/prefiks yang agenik — 400.000+ GPU sedang diproduksi (xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS). Kendala perangkat keras: Hanya CPU → llama.cpp saja. AMD / non-NVIDIA → hanya vLLM (TRT-LLM dikunci oleh NVIDIA). Pola pipeline 2026: dev = Ollama, staging = llama.cpp, prod = vLLM atau SGLang. Weight GGUF/HF sama secara keseluruhan.

**Type:** Learn
**Language:** Python (stdlib, walker pohon keputusan mesin)
**Prerequisites:** Semua lesson Fase 17 yang mencakup mesin (04, 06, 07, 09, 18)
**Waktu:** ~45 menit

## Tujuan Pembelajaran

- Pilih mesin berdasarkan perangkat keras (CPU / AMD / NVIDIA Hopper / Blackwell), skala (1 pengguna / 100 / 10.000), dan weight kerja (obrolan umum / agen / konteks panjang).
- Sebutkan status mode pemeliharaan TGI 2026 (11 Desember 2025) dan alasannya membiaskan proyek baru ke arah vLLM atau SGLang.
- Jelaskan pipeline dev/staging/prod menggunakan weight GGUF atau HF yang sama secara keseluruhan.
- Jelaskan mengapa "CPU only" memaksa llama.cpp dan "AMD" mengecualikan TRT-LLM.

## Masalah

Tim kamu memulai proyek LLM baru yang dihosting sendiri. Seorang insinyur mengatakan Ollama, yang lain mengatakan vLLM, yang ketiga mengatakan "bukankah TGI langsung berfungsi?" Ketiganya tepat untuk konteks yang berbeda. Tidak ada yang cocok untuk semua.

Pada tahun 2026, pohon pilihan menjadi hal yang penting: perangkat keras diutamakan, skala kedua, weight kerja ketiga. Dan satu peristiwa spesifik tahun 2025 — TGI memasuki mode pemeliharaan pada 11 Desember — mengubah default untuk proyek baru.

## Konsep

### Lima mesin

| Mesin | Terbaik untuk | Catatan |
|--------|----------|-------|
| **llama.cpp** | CPU / edge / deps minimal / dukungan model terluas | Tercepat di CPU, kontrol penuh |
| **Olama** | Laptop pengembang, pengguna tunggal, instalasi satu prompt | 15-30% lebih lambat dari llama.cpp; kesenjangan throughput produk 3x |
| **TGI** | Ekosistem HF, industri yang diatur | **Mode pemeliharaan 11 Des 2025** |
| **vLLM** | Produksi tujuan umum, 100+ pengguna | Gagal produksi yang luas; v0.15.1 Februari 2026 |
| **SGLang** | Weight kerja multi-putaran, banyak awalan yang agenik | 400.000+ GPU dalam produksi |

### Keputusan yang mengutamakan perangkat keras

**Khusus CPU** → llama.cpp. Ollama juga berfungsi tetapi lebih lambat. Tidak ada mesin lain yang bersaing pada CPU.

**AMD GPU** → vLLM (dukungan AMD ROCm). SGLang juga berfungsi. TRT-LLM dikunci oleh NVIDIA, jadi tidak ada.

**NVIDIA Hopper (H100 / H200)** → vLLM atau SGLang atau TRT-LLM. Ketiganya merupakan pemain papan atas.

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM adalah pemimpin throughput (Fase 17 · 07). vLLM dan SGLang mengikuti dari dekat.

**Apple Silicon (Seri M)** → llama.cpp (Logam). Ollama membungkus ini.

### Keputusan skala kedua

**1 pengguna / pengembang lokal** → Ollama. Satu prompt, token pertama dalam hitungan detik.

**10-100 pengguna / tim kecil** → vLLM GPU tunggal.**100-10 ribu pengguna/produksi** → tumpukan produksi vLLM (Fase 17 · 18) atau SGLang.

**10k+ pengguna / perusahaan** → tumpukan produksi vLLM + terpilah (Fase 17 · 17) + LMCache (Fase 17 · 18).

### Keputusan weight kerja ketiga

**Obrolan umum / Tanya Jawab** → vLLM menang secara default luas.

**Agentic multi-turn (alat, perencanaan, memori)** → RadixAttention SGLang (Fase 17 · 06) mendominasi.

**RAG dengan penggunaan kembali awalan yang berat** → SGLang.

**Pembuatan code** → vLLM baik; SGLang sedikit lebih baik pada cache.

**Konteks panjang (128K+)** → vLLM + potongan awal; SGLang + KV berjenjang.

### Perangkap pemeliharaan TGI

Memeluk Wajah TGI memasuki mode pemeliharaan pada 11 Desember 2025 — hanya perbaikan bug yang akan dilakukan. Secara historis: kemampuan observasi tingkat atas, integrasi ekosistem HF terbaik di kelasnya (kartu model, alat keselamatan), sedikit di belakang vLLM pada throughput mentah.

Untuk proyek baru di tahun 2026: default jauh dari TGI. Penerapan TGI yang ada dapat dilanjutkan tetapi pada akhirnya akan bermigrasi. SGLang dan vLLM adalah default yang lebih aman.

### Pola pipeline pipa

Dev (Ollama) → pementasan (llama.cpp) → prod (vLLM). Weight GGUF atau HF sama secara keseluruhan. Insinyur melakukan iterasi dengan cepat pada laptop; pementasan mencerminkan kuantisasi produksi; prod adalah target penayangan.

### Peringatan Ollama

Ollama sangat bagus untuk pengembang. Ini tidak bagus untuk produksi bersama: serialisasi Go HTTP menambah overhead, manajemen konkurensi lebih sederhana daripada vLLM, dukungan OpenTelemetry lambat. Gunakan Ollama di tempat yang tepat — satu pengguna, satu prompt — dan beralih ke vLLM untuk dibagikan.

### Dihosting sendiri vs dikelola adalah keputusan terpisah

Fase 17 · 01 (hiperscaler terkelola), · 02 (platform inference) penutup terkelola. Lesson ini mengasumsikan kamu telah memutuskan untuk menjadi tuan rumah mandiri. Alasan melakukan self-host: residensi data, penyesuaian khusus, total kepemilikan biaya dalam skala besar, model domain tidak tersedia di host.

### Nomor yang harus kamu ingat

- Mode pemeliharaan TGI: 11 Desember 2025.
- vLLM v0.15.1: Februari 2026; PyTorch 2.10; Dukungan Blackwell SM120.
- Jejak produksi SGLang: 400.000+ GPU.
- Kesenjangan throughput Ollama vs llama.cpp: 15-30% lebih lambat; 3x di bawah weight produksi.

## Pakai

`code/main.py` adalah alat bantu pengambilan keputusan: berdasarkan perangkat keras + skala + weight kerja, memilih mesin dan menjelaskan alasannya.

## Kirim

Lesson ini menghasilkan `outputs/skill-engine-picker.md`. Mengingat kendala, pilih mesin dan tulis rencana migrasi.

## Latihan

1. Jalankan `code/main.py` dengan perangkat keras/skala/weight kerja kamu. Apakah hasilnya sesuai dengan intuisi kamu?
2. Infra kamu adalah 12 H100s dan 8 MI300X AMD. Mesin apa? Mengapa TRT-LLM tidak diperhitungkan?
3. Sebuah tim ingin menggunakan TGI pada tahun 2026 karena "itulah yang kami ketahui". Perdebatkan kasus migrasi.
4. Ollama dev to vLLM prod: perubahan apa dalam kuantisasi, konfigurasi, dan observabilitas?
5. Produk RAG dengan panjang awalan P99 8K dan penggunaan kembali yang tinggi di seluruh penyewa. Pilih mesin dan susun dengan Fase 17 · 11 + 18.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| llama.cpp | "yang CPU" | Dukungan model terluas, tercepat pada CPU |
| Ollama | "yang laptop" | Penginstalan satu prompt, throughput tingkat pengembangan |
| TGI | "Penyajian HF" | Mode pemeliharaan sejak Des 2025 |
| vLLM | "standar" | Baseline produksi luas 2026 |
| SGLang | "yang agen" | Awalan-berat, RadixAttention |
| TRT-LLM | "Terkunci NVIDIA" | Pemimpin throughput Blackwell, khusus NVIDIA |
| GGUF | "format llama.cpp" | Varian K-quant yang dibundel |
| Tumpukan produksi | "vLLM K8s" | Fase 17 · 18 penerapan referensi |
| Pola pipa | "dev→phase→prod" | Ollama → llama.cpp → vLLM dengan weight yang sama |

## Bacaan Lanjutan

- [Alat Buatan AI — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morf — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Perbandingan Mesin Inference LLM Komprehensif](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Alternatif vLLM Terbaik 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [Pengumuman pemeliharaan TGI](https://github.com/huggingface/text-generasi-inference) — catatan rilis.
- [catatan rilis vLLM v0.15.1](https://github.com/vllm-project/vllm/releases)
