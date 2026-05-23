# Resep VLM Berat Terbuka: Yang Sebenarnya Penting

> Literatur VLM open-weight 2024-2026 adalah kumpulan tabel ablasi. MM1 Apple menguji 13 kombinasi encoder gambar, konektor, dan campuran data. Molmo dari Allen AI membuktikan teks manusia yang mendetail mengalahkan distilasi GPT-4V. Cambrian-1 menjalankan 20+ perbandingan encoder. Idefics2 meresmikan ruang desain lima sumbu. VLM prismatik membandingkan 27 resep training pada tolok ukur yang terkontrol. Dari semua kebisingan tersebut, ada serangkaian kecil hasil yang dapat diterapkan di seluruh makalah: pembuat enkode gambar lebih penting daripada arsitektur konektor, campuran data lebih penting daripada keduanya, dan teks manusia yang mendetail mengalahkan data sintetis hasil sulingan. Lesson ini membaca tabel-tabel tersebut sehingga kamu tidak perlu melakukannya.

**Type:** Learn + lab
**Language:** Python (stdlib, pengurai tabel ablasi + pemilih resep)
**Prerequisites:** Fase 12 · 05 (dasar LLaVA)
**Waktu:** ~180 menit

## Tujuan Pembelajaran

- Beri nama ruang desain VLM lima sumbu: encoder gambar, konektor, LLM, campuran data, jadwal resolusi.
- Baca tabel ablasi MM1 / Idefics2 / Cambrian-1 dan prediksi kenop mana yang menggerakkan patokan tertentu.
- Pilih resep (encoder, konektor, data, resolusi) untuk VLM baru dengan mempertimbangkan anggaran komputasi dan campuran tugas.
- Jelaskan mengapa teks manusia yang mendetail mengalahkan distilasi GPT-4V pada jumlah token yang sama.

## Masalah

Ada ratusan VLM berbobot terbuka. Sebagian besar kesenjangan antara "baik" dan "tercanggih" bukanlah arsitektur. Ini adalah data, jadwal resolusi, dan pilihan encoder. Mengetahui kenop mana yang harus diputar terlebih dahulu saat model kamu berkinerja buruk akan menghemat 5 juta kesalahan GPU-jam.

Gelombang 2023 (LLaVA-1.5, InstructBLIP, MiniGPT-4) dijalankan pada pra-training pasangan teks + LLaVA-Instruct-150k. Dasar yang bagus. Menghabiskan sekitar MMMU 35%.

Gelombang tahun 2024 (MM1, Idefics2, Molmo, Cambrian-1, Prismatic VLMs) melakukan ablasi yang menyeluruh. Hasilnya mengejutkan dan praktis.

## Konsep

### Ruang desain lima sumbu

Idefics2 (Laurençon et al., 2024) menamai sumbunya:

1. Pembuat enkode gambar. KLIP ViT-L/14, SigLIP SO400m/14, DINOv2 ViT-g/14, InternViT-6B. Encoder berbeda dalam ukuran patch, resolusi, dan tujuan pra-training.
2. Konektor. MLP (2-4 layer), Q-Former (32 kueri + cross-attn), Perceiver Resampler (64 kueri), C-Abstractor (konvolusional + pengumpulan bilinear).
3. Model bahasa. Llama-3 8B / 70B, Mistral 7B, Phi-3, Gemma-2, Qwen2.5. Ukuran LLM merupakan parameter biaya yang dominan.
4. Training data. Pasangan teks (CC3M, LAION), disisipkan (OBELICS, MMC4), instruksi (LLaVA-Instruct, ShareGPT4V, PixMo, Cauldron).
5. Jadwal resolusi. Memperbaiki 224/336/448, AnyRes, dinamis asli. Meningkat selama training atau konstan.

Setiap VLM produksi membuat pilihan pada setiap sumbu. Sebagian besar varian skor MMMU dijelaskan oleh sumbu 1, 4, dan 5 — bukan oleh konektor mana yang kamu pilih.

### Sumbu 1: encoder > konektor

MM1 Bagian 3.2 menunjukkan: menukar dari CLIP ViT-L/14 ke SigLIP SO400m/14 menambahkan 3+ poin MMMU. Menukar konektor dari MLP ke Perceiver Resampler menambah kurang dari 1 poin. Idefics2 direplikasi: SigLIP > CLIP, Q-Former ≈ MLP ≈ Perceiver pada jumlah token yang sama.

"Cambrian Vision Encoders Match-Up" dari Cambrian-1 (Tong et al., 2024) menjalankan 20+ encoder pada tolok ukur yang berpusat pada visi (CV-Bench). Bagian atas papan peringkat adalah campuran DINOv2 dan SigLIP; CLIP berada di tengah-tengah paket; ImageBind dan ViT-MAE lebih rendah. Kesenjangan dari CLIP ViT-L ke DINOv2 ViT-g/14 adalah ~5-7 poin di CV-Bench.Encoder default tahun 2026 untuk VLM terbuka adalah SigLIP 2 SO400m/14 untuk feature semantik + padat, terkadang digabungkan dengan feature DINOv2 ViT-g/14 ("Agregator Visi Spasial" Cambrian melakukan hal ini).

### Sumbu 2: desain konektor rusak

MM1, Idefics2, Prismatic, dan MM-Interleaved semuanya mencapai kesimpulan yang sama: pada jumlah token visual yang tetap, arsitektur konektor hampir tidak menjadi masalah. MLP 2 lapis pada patch yang dikumpulkan rata-rata berfungsi dalam 1 poin dari 32 kueri Q-Former dengan anggaran token yang sama.

Yang penting adalah jumlah tokennya. Lebih banyak token visual = lebih banyak komputasi LLM = kinerja yang lebih baik hingga titik tertentu, kemudian hasil yang semakin berkurang. 64 token per gambar terlalu sedikit untuk OCR. Token 576-1024 adalah pilihan terbaik bagi sebagian besar VLM terbuka. 2048+ hanya membantu untuk dokumen dan bagan.

Q-Former vs MLP adalah pertanyaan biaya, bukan pertanyaan kualitas: Q-Former membatasi token pada 32-64 terlepas dari resolusi gambar; MLP memancarkan semua token patch. Untuk input resolusi tinggi, Q-Former menyimpan konteks LLM; untuk resolusi rendah, yang membedakan adalah noise.

### Sumbu 3: Ukuran LLM menentukan batas atas

Menggandakan LLM dari 7B ke 13B secara andal menambah 2-4 poin pada MMMU di setiap kertas VLM. Pada 70B kamu memenuhi sebagian besar tolok ukur. Batas atas penalaran multimodal VLM adalah batas atas penalaran teks LLM - pembuat enkode visual hanya dapat memberikan input, bukan memberikan alasan untuk itu.

Inilah sebabnya mengapa Qwen2.5-VL-72B dan Claude Opus 4.7 menghancurkan MMMU-Pro dan ScreenSpot-Pro: otak bahasa sangat besar. VLM 7B tidak dapat menggantikan VLM 70B melalui desain konektor yang cerdas.

### Sumbu 4: data — teks manusia yang mendetail mengalahkan penyulingan

Molmo + PixMo (Deitke et al., 2024) adalah hasil tahun 2024 yang harus dibaca semua orang. Allen AI memiliki anotator manusia yang mendeskripsikan gambar dalam penyampaian ucapan-ke-teks yang padat selama 1-3 menit, sehingga menghasilkan 712 ribu gambar dengan teks padat. Tidak ada distilasi GPT-4V di mana pun dalam training data.

Molmo-72B mengalahkan Llama-3.2-90B-Vision pada 11 dari 11 benchmark. Delta bukanlah arsitektur — ini adalah kualitas teks. Teks manusia yang mendetail berisi informasi 5-10x lebih banyak per gambar dibandingkan teks web pendek dan tetap berdasarkan faktual saat distilasi GPT-4V berhalusinasi.

ShareGPT4V (Chen et al., 2023) dan Cauldron (Idefics2) mengikuti pedoman yang sama dengan teks campuran manusia + GPT-4V. Trennya jelas: untuk tahun 2026, kepadatan teks > kuantitas teks > kenyamanan penyulingan.

### Sumbu 5: resolusi dan jadwalnya

Ablasi Idefics2: 384 -> 448 menambah 1-2 poin. 448 -> 980 dengan pemisahan gambar (AnyRes) menambah 3-5 lagi pada benchmark OCR. Training resolusi datar mencapai tingkat akurasi sedang; resolusi ramping (mulai 224, selesai 448 atau asli) berlatih lebih cepat dan berakhir lebih tinggi.

Cambrian-1 menjalankan trade-off resolusi vs token: pada komputasi tetap, kamu dapat memiliki lebih banyak token pada resolusi lebih rendah atau lebih sedikit token pada resolusi lebih tinggi. Resolusi yang lebih tinggi menang untuk OCR; kemenangan dengan resolusi lebih rendah untuk pemahaman adegan secara umum.

Resep produksi tahun 2026: latih Phase 1 pada 384 tetap, Phase 2 dengan resolusi dinamis hingga 1280 untuk tugas-tugas berat OCR.

### Perbandingan terkontrol Prismatik

VLM Prismatik (Karamcheti et al., 2024) adalah kertas yang mengendalikan semua sumbu. LLM 13B yang sama, data instruksi yang sama, evaluasi yang sama — hanya satu sumbu yang bervariasi pada satu waktu. Hasil:

- Jumlah token visual per gambar menjelaskan ~60% varians.
- Pilihan encoder menjelaskan ~20%.
- Arsitektur konektor menjelaskan ~5%.
- Yang lainnya (data mix, scheduler, LR) sisanya ~15%.Ini adalah penguraian kasar, tetapi ini adalah jawaban terbersih untuk "apa yang harus saya hilangkan terlebih dahulu" dalam literatur.

### Pemilih untuk tahun 2026

Berdasarkan buktinya, resep VLM terbuka default untuk proyek baru pada tahun 2026:

- Encoder: SigLIP 2 SO400m/14 pada resolusi asli dengan NaFlex, digabungkan dengan DINOv2 ViT-g/14 untuk feature padat jika kamu memerlukan segmentasi/grounding.
- Konektor: MLP 2 lapis pada token patch. Lewati Q-Former kecuali kamu dibatasi token.
- LLM: Qwen2.5 / Llama-3.1 / Gemma 2, biaya 7B, kualitas 70B, dipilih berdasarkan latensi target.
- Data: PixMo + ShareGPT4V + Cauldron, diisi dengan data instruksi khusus tugas.
- Resolusi: dinamis (min 256, maks 1280 piksel per sisi panjang).
- Jadwal: Penyelarasan Phase 1 (khusus proyektor), Penyempurnaan penuh Phase 2, Penyempurnaan khusus tugas Phase 3.

Setiap default tersebut dapat ditelusuri kembali ke ablasi terukur dalam makalah yang dikutip di akhir lesson ini.

## Pakai

`code/main.py` adalah pengurai tabel ablasi dan pemetik resep. Ini mengkodekan tabel ablasi MM1 dan Idefics2 (dipadatkan) dan memungkinkan kamu menanyakan:

- "Mengingat anggaran X dan tugas Y, resep apa yang menang?"
- "Jika saya menukar SigLIP dengan CLIP pada Llama 7B, berapakah delta MMMU yang diharapkan?"
- "Sumbu mana yang harus saya ablasi terlebih dahulu untuk mendapatkan jawaban dengan keyakinan 80%?"

Outputnya adalah daftar resep yang diberi peringkat dengan delta patokan yang diharapkan dan rekomendasi "ablate first".

## Kirim

Lesson ini menghasilkan `outputs/skill-vlm-recipe-picker.md`. Dengan adanya campuran tugas target, anggaran komputasi, dan target latensi, sistem ini mengeluarkan resep lengkap (encoder, konektor, LLM, campuran data, jadwal resolusi) dengan kutipan ke ablasi yang membenarkan setiap pilihan. Menghentikan teknisi untuk menemukan kembali tabel ablasi Idefics2 setiap kali proyek VLM baru dimulai.

## Latihan

1. Baca MM1 Bagian 3.2. Untuk LLM 2B tetap dengan anggaran 50 juta gambar, encoder mana yang menang? Apakah jawabannya akan berubah menjadi 13B LLM? Mengapa?

2. Cambrian-1 menemukan bahwa penggabungan DINOv2 + SigLIP memiliki kinerja yang lebih baik jika digabungkan sendiri pada tolok ukur yang berpusat pada visi tetapi tidak menambahkan sinyal pada MMMU. Memprediksi benchmark mana yang menguat dan mana yang tetap datar.

3. Target kamu adalah agen UI seluler di LLM 2B. Pilih encoder, konektor, resolusi, dan campuran data. Justifikasi setiap pilihan dengan tabel ablasi tertentu.

4. Molmo mengirimkan model 4B dan 72B. 4B kompetitif dengan VLM 7B tertutup; 72B mengalahkan Llama-3.2-90B-Vision pada benchmark 11/11. Apa yang dapat kamu ketahui tentang hipotesis dataran tinggi ukuran LLM?

5. Rancang tabel ablasi untuk mengisolasi kualitas campuran data dari kualitas encoder pada VLM 7B. Berapa jumlah minimum training yang dijalankan? Usulkan pengaturan empat sumbu.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Ablasi | "Memutar satu kenop" | Melatih beberapa proses yang berbeda tepat pada satu sumbu ruang desain, menjaga semuanya konstan |
| Konektor | "Jembatan" / "proyektor" | Modul yang dapat dilatih yang memetakan output encoder visi ke dalam ruang token LLM (MLP, Q-Former, Perceiver) |
| Keterangan manusia mendetail | "Teks padat" | Deskripsi multi-kalimat yang ditulis manusia (biasanya 80-300 token) lebih kaya daripada teks alternatif web |
| Distilasi | "Keterangan GPT-4V" | Training data yang dihasilkan oleh VLM berpemilik yang lebih kuat; nyaman tetapi rentan terhadap halusinasi bawaan |
| AnyRes / resolusi dinamis | "Jalur resolusi tinggi" | Strategi untuk memasukkan gambar yang lebih besar dari resolusi asli pembuat enkode melalui ubin atau M-RoPE |
| Jalan resolusi | "Kurikulum" | Jadwal training yang dimulai dengan resolusi rendah dan meningkat, mempercepat pembelajaran penyelarasan |
| Bangku yang berpusat pada visi | "Bangku CV / BLINK" | Evaluasi yang menekankan persepsi visual yang mendetail daripada penalaran yang banyak menggunakan bahasa |
| PixMo | "Data Molmo" | Dataset gambar 712K dengan teks padat milik Allen AI; ucapan manusia ditranskripsikan menjadi teks padat |

## Bacaan Lanjutan

- [McKinzie dkk. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon dkk. — Idefics2 / Apa yang penting dalam membangun VLM (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke dkk. — Molmo dan PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong dkk. — Kambrium-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti dkk. — VLM Prismatik (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
