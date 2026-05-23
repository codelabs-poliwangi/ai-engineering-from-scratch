# Inference Tepi — Mesin Neural Apple, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

> Batasan inti inti adalah bandwidth memori, bukan komputasi. DRAM Seluler berada pada kecepatan 50-90 GB/dtk; pusat data HBM3 menyelesaikan 2-3 TB/dtk — selisih 30-50x. Decode terikat pada memori sehingga kesenjangannya sangat menentukan. Pada tahun 2026, lanskap terbagi menjadi empat arah. Mesin Neural Apple M4/A18 mencapai puncaknya pada 38 TOPS dengan memori terpadu (tanpa salinan CPU↔NPU). Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon mencapai 45 TOPS. WebGPU + WebLLM menjalankan Llama 3.1 8B (Q4) pada ~41 tok/s pada M3 Max (kira-kira 70-80% asli); 17,6 ribu bintang GitHub, API yang kompatibel dengan OpenAI, ~70-75% cakupan seluler. NVIDIA Jetson Orin Nano Super (8GB) cocok untuk Llama 3.2 3B / Phi-3; AGX Orin menjalankan gpt-oss-20b melalui vLLM pada ~40 tok/s; Jetson T4000 (JetPack 7.1) adalah 2x AGX Orin. TensorRT Edge-LLM mendukung EAGLE-3, NVFP4, potongan prefill — ditampilkan di CES 2026 oleh Bosch, ThunderSoft, MediaTek.

**Type:** Learn
**Language:** Python (stdlib, simulator dekode terikat bandwidth mainan)
**Prerequisites:** Phase 17 · 04 (vLLM Melayani Internal), Phase 17 · 09 (Kuantisasi Produksi)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Jelaskan mengapa inference LLM seluler terikat pada bandwidth memori dan komputasi bersifat sekunder.
- Hitung empat target edge (Apple ANE, Qualcomm Hexagon, WebGPU/WebLLM, NVIDIA Jetson) dan cocokkan masing-masing dengan kasus penggunaan.
- Sebutkan kesenjangan cakupan WebGPU 2026 (Firefox Android menyusul) dan pendaratan Safari iOS 26.
- Pilih format kuantisasi per target (Core ML INT4 + FP16 untuk ANE, QNN INT8/INT4 untuk Hexagon, WebGPU Q4 untuk browser, NVFP4 untuk Jetson Thor).

## Masalah

Pelanggan menginginkan chatbot di perangkat: mengutamakan suara, bersifat pribadi secara default, dan berfungsi secara offline. Di MacBook Pro M3 Max, Llama 3.1 8B Q4 berjalan pada ~55 tok/s — bagus. Di iPhone 16 Pro, model yang sama berjalan pada 3 tok/s — tidak bagus. Pada Android kelas menengah dengan Snapdragon 8 Gen 3, 7 tok/s. Di browser melalui WebGPU di Chrome Android v121+, 4-8 tok/s tergantung perangkatnya.

Varians throughput bukanlah masalah porting. Ini adalah kesenjangan bandwidth dikalikan format kuantisasi dikalikan apakah NPU dapat diakses dari ruang pengguna. Inference tepi pada tahun 2026 adalah empat masalah berbeda dengan empat solusi berbeda.

## Konsep

### Bandwidth adalah batas tertinggi sebenarnya

Dekode membaca kumpulan weight lengkap untuk setiap token. Satu model 7B di Q4 berukuran 3,5 GB. Membaca 3,5 GB pada 50 GB/dtk memerlukan waktu 70 mdtk — batas maksimum teoritis ~14 tok/dtk. Pada 90 GB/dtk (DRAM seluler kelas atas), batas maksimumnya berpindah ke ~25 tok/dtk. Tidak ada jumlah komputasi yang membantu di bawah angka ini.

Pusat data HBM3 pada 3 TB/dtk membersihkan 3,5 GB yang sama dalam 1,2 mdtk — batas maksimumnya adalah 830 tok/dtk. Model yang sama, weight yang sama. Subsistem memori yang berbeda.

### Mesin Syaraf Apple (M4 / A18)

- Hingga 38 TOPS. Memori terpadu (CPU dan ANE berbagi kumpulan yang sama) — tanpa overhead penyalinan.
- Akses melalui model terkompilasi Core ML + `.mlmodel`, atau melalui Metal Performance Shaders (MPS) melalui PyTorch.
- Backend logam Llama.cpp menggunakan MPS, bukan ANE secara langsung; ANE asli memerlukan konversi Core ML.
- Jalur praktis terbaik untuk aplikasi iOS pada tahun 2026: Core ML dengan weight INT4 + activation FP16.

### Qualcomm Segi Enam (Snapdragon X Elite / 8 Gen 4)

- Hingga 45 TOPS. Terintegrasi dengan CPU dan GPU di SoC tetapi domain memori terpisah.
- QNN (Qualcomm Neural Network) SDK dan AI Hub menyediakan konversi dari PyTorch/ONNX.
- Templat obrolan, Llama 3.2, Phi-3 semuanya dikirimkan sebagai artefak kelas satu di AI Hub.

### Intel / AMD NPU (Lunar Lake, Ryzen AI 300)- 40-50 ATAS. Perangkat lunak tertinggal dari Apple/Qualcomm; OpenVINO meningkat tetapi masih khusus.
- Terbaik untuk aplikasi kopilot Windows ARM; asli pada desktop AMD/Intel untuk lokal-pertama.

### WebGPU + WebLLM

- Jalankan model di browser melalui shader komputasi WebGPU; tidak ada instalasi.
- Llama 3.1 8B Q4 pada ~41 tok/s di M3 Max — sekitar 70-80% asli melalui backend yang sama.
- 17,6 ribu bintang GitHub di WebLLM; API JS yang kompatibel dengan OpenAI; Apache 2.0.
- Cakupan 2026: Chrome Android v121+, Safari iOS 26 GA, Firefox Android masih menyusul. Secara keseluruhan ~70-75% jangkauan seluler.

### Keluarga NVIDIA Jetson

- Orin Nano Super (8GB): cocok untuk Llama 3.2 3B, Phi-3 dengan kecepatan yang baik.
- AGX Orin: menjalankan gpt-oss-20b melalui vLLM pada ~40 tok/s.
- Thor / T4000 (JetPack 7.1): Performa 2x AGX Orin, didukung EAGLE-3 dan NVFP4.
- TensorRT Edge-LLM (2026) mendukung decoding spekulatif EAGLE-3, weight NVFP4, prefill yang dipotong — optimization pusat data yang di-porting ke edge.

### Pilihan kuantisasi per target

| Sasaran | Format | Catatan |
|--------|--------|-------|
| Apple ANE | Weight INT4 + activation FP16 | Jalur konversi inti ML |
| Qualcomm Segi Enam | QNN INT8 / INT4 | Konverter AI Hub |
| WebGPU / WebLLM | MLC Q4 (q4f16_1) | Gunakan `mlc_llm convert_weight` + dikompilasi `.wasm`; GGUF tidak didukung |
| Jetson Orin Nano | Q4 GGUF atau TRT-LLM INT4 | Terikat memori |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Jalur Edge-LLM |

### Jebakan konteks panjang di ujung tanduk

Konteks 128K Llama 3.1 adalah feature pusat data. Pada ponsel dengan RAM 8 GB, model 4 GB + cache KV 2 GB untuk token 32K + overhead OS = OOM. Penerapan edge menjaga konteks pada 4K-8K kecuali kuantisasi KV agresif (Q4 KV) diterima.

### Suara adalah aplikasi pembunuh

Agen suara peka terhadap latensi (token pertama <500 ms). Inference lokal menghilangkan latensi jaringan sepenuhnya. Kombinasikan dengan ucapan-ke-teks (varian Whisper Turbo berjalan di edge) dan inference edge menjadi loop suara berkualitas produksi.

### Nomor yang harus kamu ingat

- Apple M4 / A18 ANE: 38 ATAS.
- Qualcomm Hexagon SD X Elite: 45 TOPS.
- WebLLM M3 Maks: ~41 tok/dtk di Llama 3.1 8B Q4.
- AGX Orin: ~40 tok/dtk di gpt-oss-20b melalui vLLM.
- Kesenjangan bandwidth tepi pusat data: 30-50x.
- Cakupan seluler WebGPU: ~70-75% (Firefox Android tertinggal).

## Pakai

`code/main.py` menghitung batas atas throughput dekode teoretis dari matematika terikat bandwidth di seluruh target edge. Bandingkan dengan tolok ukur dan sorotan yang diamati di mana bandwidth, bukan komputasi, yang menjadi hambatannya.

## Kirim

Lesson ini menghasilkan `outputs/skill-edge-target-picker.md`. Platform tertentu (iOS/Android/browser/Jetson), model, dan latensi/anggaran memori, memilih format kuantisasi dan jalur konversi.

## Latihan

1. Jalankan `code/main.py`. Untuk model 7B di Q4 dengan Snapdragon 8 Gen 3 (bandwidth ~77 GB/dtk), hitung batas atas dekode. Bandingkan dengan 6-8 tok/s yang diamati — apakah waktu pengoperasiannya efisien?
2. WebGPU di Android memerlukan Chrome v121+. Rancang penggantian untuk browser lama — di sisi server melalui API yang sama dan kompatibel dengan OpenAI.
3. Aplikasi iOS kamu memerlukan streaming konteks 4K. Kombinasi model/format manakah yang memungkinkan kamu mempertahankan memori aktif di bawah 4 GB pada iPhone 16?
4. Jetson AGX Orin menjalankan gpt-oss-20b dengan kecepatan 40 tok/s. Jetson Nano hanya cocok untuk 3B. Jika produk kamu menargetkan keduanya, bagaimana kamu menyatukan tumpukan inference?
5. Berdebat apakah "WebLLM siap produksi pada tahun 2026." Kutip cakupan, kinerja, dan kesenjangan Firefox Android.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| ANE | "Mesin saraf Apple" | NPU pada perangkat dalam seri M dan seri A; memori terpadu |
| segi enam | "Qualcomm NPU" | NPU Snapdragon; QNN SDK untuk akses |
| WebGPU | "GPU peramban" | API GPU browser berstandar W3C; Chrome/Safari 2026 |
| WebLLM | "waktu proses LLM browser" | proyek MLC-LLM; Apache 2.0; JS yang kompatibel dengan OpenAI |
| Jetson | "Tepi NVIDIA" | Keluarga Orin Nano / AGX / Thor / T4000 |
| TRT Edge-LLM | "tepi TensorRT" | Pelabuhan tepi TensorRT-LLM tahun 2026; EAGLE-3 + NVFP4 |
| Memori terpadu | "kolam bersama" | CPU dan NPU melihat RAM yang sama; tidak ada salinan overhead |
| Terikat bandwidth | "memori terbatas" | Dekode yang dibatasi oleh weight pembacaan byte/detik |
| Inti ML | "Konversi Apple" | Kerangka kerja Apple untuk model asli ANE |
| QNN | "tumpukan Qualcomm" | SDK Jaringan Syaraf Qualcomm |

## Bacaan Lanjutan

- [State of the Union LLM Pada Perangkat 2026](https://v-chandra.github.io/on-device-llms/) — lanskap dan tolok ukur.
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor.
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — pengumuman port edge tahun 2026.
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — desain dan tolok ukur.
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — Konversi asli ANE.
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — model pra-konversi untuk Hexagon.
