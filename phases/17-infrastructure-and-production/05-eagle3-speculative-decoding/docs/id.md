# EAGLE-3 Decoding Spekulatif dalam Produksi

> Penguraian code spekulatif memasangkan model draf cepat dengan model target. Draf tersebut mengusulkan token K; target memverifikasi dalam satu penerusan; token yang diterima gratis. Pada tahun 2026, EAGLE-3 adalah varian tingkat produksi — ia melatih draft head pada status tersembunyi model target, bukan pada token mentah, sehingga mendorong tingkat penerimaan alpha ke kisaran 0,6-0,8 pada obrolan umum. Pertanyaan yang tepat bukanlah "seberapa cepat drafnya" tetapi "berapakah alpha pada lalu lintas saya?" Jika alpha turun di bawah ~0,55, decoding spekulatif menjadi negatif bersih pada konkurensi tinggi karena setiap draf yang ditolak membutuhkan target forward pass kedua. Lesson ini mengajarkan kamu untuk mengukur alpha terlebih dahulu dan membalik bendera kedua.

**Type:** Learn
**Language:** Python (stdlib, simulator tingkat penerimaan mainan)
**Prerequisites:** Fase 17 · 04 (vLLM Melayani Internal), Fase 10 · 18 (Prediksi Multi-Token)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Sebutkan tiga generasi penguraian code spekulatif dan jelaskan apa yang berubah dari EAGLE-3 dari EAGLE-2 dan dari model rancangan klasik.
- Tentukan tingkat penerimaan alpha, hitung percepatan yang diharapkan dari alpha dan K (panjang draf), dan identifikasi alpha impas untuk konkurensi target kamu.
- Jelaskan mengapa decoding spekulatif disertakan (bukan default) di vLLM 2026 dan mengapa menyalakannya tanpa mengukur alpha merupakan anti-pola produksi.
- Tulis rencana pengukuran: tolok ukur mana, distribusi prompt mana, titik konkurensi mana, metrik mana yang akan digunakan.

## Masalah

Decode terikat pada memori. Pada H100 yang menjalankan Llama 3.3 70B FP8, setiap token yang didekodekan membaca weight ~140 GB/dtk dan memancarkan satu token. Komputasi GPU hampir menganggur selama dekode — hambatannya adalah bandwidth HBM, bukan throughput matmul.

Penguraian code spekulatif mengeksploitasi kesenjangan tersebut. Hasilkan K kandidat token dengan model draf yang murah, lalu minta model target untuk memverifikasi semua K dalam satu forward pass. Setiap token yang terverifikasi secara efektif gratis (diamortisasi ke dalam batch-of-K forward yang harus dilakukan oleh target).

Pendekatan model draf klasik menggunakan model yang lebih kecil dari kelompok yang sama (draf Llama 3.2 1B untuk Llama 3.3 70B). Ini berfungsi tetapi tingkat penerimaannya biasa-biasa saja — distribusi model yang lebih kecil menyimpang dari target. EAGLE, lalu EAGLE-2, lalu EAGLE-3 melatih draft head ringan langsung pada status internal model target, sehingga distribusi draft melacak target dengan lebih dekat. Itulah sebabnya alpha berubah dari 0,4 dengan model draft menjadi 0,6-0,8 dengan EAGLE-3.

Tangkapannya: EAGLE-3 ikut serta dalam vLLM 2026. `speculative_config` harus disetel secara eksplisit. Tidak ada bendera, tidak ada akselerasi. Tim yang mengaktifkannya tanpa mengukur alpha pada lalu lintas sebenarnya sering kali melihat latensi ekor menjadi lebih buruk, bukan lebih baik.

## Konsep

### Apa yang sebenarnya dibeli oleh decoding spekulatif

Tanpa dekode spesifikasi, biaya per token hanya satu target ke depan. Dengan dekode spesifikasi pada panjang draf K dan alpha penerimaan, token yang diharapkan per target penerusan adalah `1 + K * alpha`. Percepatannya adalah `(1 + K * alpha) / (1 + epsilon)` di mana epsilon adalah overhead draft-plus-verify. Untuk K=5, alpha=0,7: `(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`. Angka di dunia nyata berjumlah sekitar 2-3x karena alpha jarang setinggi itu pada lalu lintas produksi dan epsilon tumbuh pada ukuran batch yang tinggi.

### Mengapa alpha adalah satu-satunya metrik yang pentingToken yang ditolak tidak hilang — token tersebut memaksakan target kedua ke depan untuk token pertama yang ditolak. Pada weight kerja di mana alpha turun menjadi 0,4, kamu membayar overhead draf ditambah verifikasi ditambah pemutaran ulang. Pada konkurensi tinggi (katakanlah 256 bersamaan), kumpulan dekode sudah cukup besar sehingga kesenjangan bandwidth memori antara "target saja" dan "target dengan verifikasi" menyusut. Di bawah alpha 0,55 pada sebagian besar perangkat keras tahun 2026, dekode spesifikasi adalah negatif bersih.

Alpha bervariasi berdasarkan weight kerja. Pada obrolan umum bergaya ShareGPT, EAGLE-3 yang dilatih tentang ShareGPT mencapai 0,6-0,8. Pada lalu lintas khusus domain (code, medis, hukum), kepala wajib militer yang dilatih tentang data umum turun menjadi 0,4-0,6. Melatih kepala draf khusus domain memulihkan alpha — ini adalah tugas training yang ringan dan cepat dibandingkan dengan penyempurnaan target.

### Sekilas tentang generasi EAGLE

- **Model draf klasik**: model kecil dari keluarga yang sama. Alpha 0,3-0,5. Infrastruktur sederhana - dua model dimuat, draft berjalan K ke depan per target ke depan.
- **EAGLE-1 (2024)**: draft head tunggal dilatih pada status target tersembunyi (layer terakhir). Alpha ~0,5-0,6. Param kecil di atas target.
- **EAGLE-2 (2025)**: panjang draf adaptif dan draf berbasis pohon (verifikasi beberapa cabang dalam satu lintasan target). Alpha ~0,6-0,7. Draf penjadwal yang lebih kompleks.
- **EAGLE-3 (2025-2026)**: draft head dilatih pada beberapa layer target (bukan hanya yang terakhir), keselarasan yang lebih baik. Alpha ~0.6-0.8 pada obrolan umum.

### Resep produksi 2026

1. Model sasaran kapal polos. Ukur TTFT dasar, ITL, throughput pada konkurensi target.
2. Aktifkan draf EAGLE-3 melalui vLLM `speculative_config`. Jalankan kembali benchmark.
3. Catat tingkat penerimaan alpha. vLLM V1 melaporkan ini sebagai `spec_decode_metrics.accepted_tokens_per_request`. Bagilah dengan panjang draf yang diminta untuk mendapatkan alpha.
4. Jika alpha < 0,55 pada distribusi lalu lintas produksi, nonaktifkan dekode spesifikasi atau latih draf EAGLE-3 khusus domain.
5. Pada konkurensi produksi, jalankan kembali. Konfirmasikan P99 ITL tidak bertambah buruk.

### Jebakan produksi: P99 tail

Berarti ITL turun dengan dekode spesifikasi. P99 bisa menjadi lebih buruk jika kamu tidak menyetelnya. Draf yang ditolak memicu urutan dua langkah (draft + verifikasi-gagal + reroll). Dalam batch penuh, kedua pass tersebut diserialkan. Tonton P99 ITL, bukan P50.

### Dimana EAGLE-3 sudah dikerahkan

Google menerapkan decoding spekulatif dalam Ikhtisar AI pada tahun 2025 (kualitas sama, respons lebih cepat). vLLM V1 dikirimkan `speculative_config` sebagai antarmuka yang terdokumentasi; Decoding spekulatif GPU N-gram di V1 adalah varian yang kompatibel dengan prefill yang dipotong. SGLang mendukung EAGLE-3 sebagai jalur draf yang direkomendasikan untuk weight kerja dengan awalan yang berat.

### Matematika titik impas dalam satu baris

Kecepatan yang diharapkan: `S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`. Pengaturan `S = 1` memecahkan alpha: `alpha_breakeven = verify_overhead / K`. Untuk verifikasi_overhead tipikal ~0,15 dan K=5: `alpha_breakeven = 0.03`. Tapi itu adalah matematika decode mentah. Pada konkurensi tinggi, overhead verifikasi meningkat dan kumpulan dekode telah mengamortisasi pembacaan memori di seluruh urutan, sehingga alpha_breakeven yang efektif naik hingga ~0,45-0,55 dalam praktiknya.

### Kapan tidak menggunakan decoding spekulatif

- Generasi offline Batch-1 di mana latensi tidak menjadi masalah. Gunakan target biasa.
- Output sangat singkat (di bawah 50 token). Draf overhead dan verifikasi biaya mendominasi.
- Domain khusus tanpa draft head yang terlatih dalam domain. Alpha terlalu rendah.
- vLLM v0.18.0 ditambah dekode spesifikasi model draf plus `--enable-chunked-prefill`. Kombinasi ini tidak dapat dikompilasi. Pengecualian yang terdokumentasi adalah dekode spesifikasi GPU N-gram di V1.

## Pakai`code/main.py` menyimulasikan loop dekode dengan dan tanpa dekode spekulatif pada rentang nilai alpha dan panjang draf K. Ini mencetak alpha impas, kecepatan terukur, dan perilaku ekor. Jalankan pada beberapa kombinasi (alpha, K) untuk melihat dengan tepat di mana decoding spekulatif berhenti membuahkan hasil.

## Kirim

Lesson ini menghasilkan `outputs/skill-eagle3-rollout.md`. Dengan adanya model target, deskripsi distribusi lalu lintas, dan target konkurensi, ini menghasilkan rencana peluncuran EAGLE-3 bertahap — garis dasar benchmark, konfigurasi aktifkan, pengukuran alpha, gerbang alpha >= 0,55, tonton P99 ITL.

## Latihan

1. Jalankan `code/main.py`. Pada K=5, alpha apa yang kamu perlukan untuk percepatan 2x? Untuk percepatan 3x? Seberapa sensitifnya terhadap verifikasi_overhead?
2. Bayangkan lalu lintas produksi membagi 70% obrolan umum, 30% code. Obrolan umum mencapai alpha 0,7 dengan EAGLE-3 yang dilatih di ShareGPT; code mencapai alpha 0,4. Apa itu alpha campuran dan apakah decode spesifikasi net-positif?
3. Baca dokumentasi vLLM `speculative_config`. Sebutkan tiga mode (model draf, EAGLE, N-gram) dan mana yang kompatibel dengan pra-pengisian terpotong.
4. kamu melihat rata-rata ITL turun 25% setelah mengaktifkan EAGLE-3 tetapi ITL P99 naik 15%. Mendiagnosis dan mengusulkan mitigasi.
5. Hitung biaya memori draft head EAGLE-3 untuk Llama 3.3 70B. Bagaimana cara membandingkannya dengan menjalankan Llama 3.2 1B sebagai draf klasik?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Penguraian code spekulatif | "draf plus verifikasi" | Usulkan token K dengan model yang murah, verifikasi semua K dalam satu target ke depan |
| Tingkat penerimaan alpha | "spesifikasi tingkat penerimaan" | Bagian dari rancangan token yang diterima oleh target; satu-satunya metrik yang penting |
| Panjang draf K | "spesifikasi k" | Berapa banyak token yang diusulkan oleh rancangan tersebut per target ke depan; tipikal 4-8 |
| Verifikasi overhead epsilon | "spesifikasi overhead" | Biaya tambahan untuk memverifikasi-dan-memutar ulang vs target maju yang jelas; tumbuh dengan batch |
| EAGLE-3 | "ELANG terbaru" | varian 2025-2026; melatih draft head pada beberapa layer target; alpha 0.6-0.8 pada obrolan umum |
| `speculative_config` | "konfigurasi spesifikasi vLLM" | Keikutsertaan eksplisit dalam vLLM V1; tidak ada default berarti tidak ada akselerasi |
| Dekode spesifikasi N-gram | "Draf N-gram" | Draf sisi GPU menggunakan pencarian N-gram di prompt; kompatibel dengan potongan-prefill |
| Alpha titik impas | "tidak ada operasi alpha" | Alpha yang dekode spesifikasinya memberikan percepatan nol; tonton ini di konkurensi produksi |
| Draf dua lintasan yang ditolak | "biaya pemutaran ulang" | Dua target ke depan ketika draf ditolak; menggerakkan ekor P99 |

## Bacaan Lanjutan

- [vLLM — dokumen Decoding Spekulatif](https://docs.vllm.ai/en/latest/features/spec_decode/) — sumber resmi di `speculative_config` dan kompatibilitas chunked-prefill di V1.
- [API Konfigurasi Spekulatif vLLM](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — kumpulan kolom yang tepat.
- [Makalah EAGLE (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — formulasi kepala draf EAGLE asli.
- [makalah EAGLE-2 (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — draf dan pohon adaptif.
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — sistem LLM yang efisien dengan decoding spekulatif.
- [BentoML — Decoding Spekulatif](https://bentoml.com/llm/inference-optimization/speculative-decoding) — daftar periksa peluncuran produksi.
