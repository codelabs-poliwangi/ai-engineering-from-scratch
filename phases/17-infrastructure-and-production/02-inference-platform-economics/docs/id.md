# Ekonomi Platform Inference - Kembang Api, Bersama, Baseten, Modal, Replikasi, Skala Apa Pun

> Pasar inference tahun 2026 bukan lagi persewaan waktu GPU. Ini terbagi menjadi silikon khusus (Groq, Cerebras, SambaNova), platform GPU (Baseten, Together, Fireworks, Modal), dan pasar yang mengutamakan API (Replikasi, DeepInfra). Fireworks menaikkan harga $1/jam per GPU pada 1 Mei 2026, dan penilaian $4 miliar pada 10T+ token/hari menunjukkan bahwa model berbasis volume berfungsi. Baseten menutup Seri E senilai $300 juta dengan harga $5 miliar pada bulan Januari 2026. Aturan penentuan posisi kompetitifnya sederhana: Fireworks mengoptimalkan latensi, Together mengoptimalkan keluasan katalog, Baseten mengoptimalkan penyempurnaan perusahaan, Modal mengoptimalkan DX asli Python, Replikasi mengoptimalkan jangkauan multimodal, Anyscale mengoptimalkan Python terdistribusi. Lesson ini memberi kamu sebuah matrix yang dapat kamu berikan kepada seorang pendiri.

**Type:** Learn
**Language:** Python (stdlib, pembanding ekonomi mainan per panggilan)
**Prerequisites:** Fase 17 · 01 (Platform LLM Terkelola), Fase 17 · 04 (vLLM Melayani Internal)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Sebutkan tiga segmen pasar (silikon khusus, platform GPU, API-first) dan petakan setiap vendor ke suatu segmen.
- Jelaskan mengapa model penetapan harga API "per token" memampatkan kurva biaya mesin penyajian, bukan kurva biaya perangkat keras.
- Hitung biaya efektif per permintaan di setidaknya tiga vendor dan jelaskan kapan per menit (Baseten, Modal) mengalahkan per token.
- Identifikasi platform mana yang merupakan default yang tepat untuk weight kerja tertentu (burst tanpa server, throughput tinggi yang stabil, varian yang disesuaikan, multimodal).

## Masalah

kamu mengevaluasi platform hyperscaler terkelola. kamu memutuskan bahwa kamu memerlukan penyedia yang lebih sempit dan lebih cepat — Fireworks untuk latensi, Together untuk keluasan, Baseten untuk model kustom yang disesuaikan. Sekarang kamu memiliki enam pilihan nyata dan halaman harga tidak sejajar. Kembang api menunjukkan token $/M; Baseten menunjukkan $/menit; Modal menunjukkan $/detik; Replikasi menunjukkan $/prediksi. kamu tidak dapat membandingkannya secara langsung tanpa memodelkan weight kerja.

Lebih buruknya lagi, model bisnis di balik setiap halaman harga berbeda. Fireworks menjalankan mesin kustomnya sendiri (FireAttention) pada GPU bersama; tingkat per token mencerminkan kurva pemanfaatannya. Baseten memberi kamu Truss + GPU khusus; per menit mencerminkan eksklusivitas. Modal benar-benar tanpa server Python — penagihan per detik dengan cold start sub-detik. Output yang sama (respon LLM), tiga fungsi biaya berbeda.

Lesson ini memodelkan keenamnya dan memberi tahu kamu kapan masing-masing menang.

## Konsep

### Tiga segmen

**Silikon khusus** — Groq (LPU), Cerebras (WSE), SambaNova (RDU). Biasanya decode 5-10x lebih cepat dibandingkan cluster berbasis GPU pada model yang sama. Harga per token yang lebih tinggi (Groq adalah ~$0,99/M pada Llama-70B pada akhir tahun 2025) tetapi tidak ada duanya untuk kasus penggunaan yang sensitif terhadap latensi. Groq adalah pilihan produksi untuk agen suara dan terjemahan waktu nyata.

**Platform GPU** — Baseten, Together, Fireworks, Modal, Anyscale. Jalankan di NVIDIA (H100, H200, B200 pada tahun 2026) atau terkadang AMD. Layer ekonomi antara "penyewaan GPU mentah" (RunPod, Lambda) dan "layanan terkelola hyperscaler" (Bedrock).

**Pasar yang mengutamakan API** — Replikasi, DeepInfra, OpenRouter, Fal. Katalog luas, bayar per prediksi atau bayar per detik, menekankan waktu untuk menelepon pertama kali.

### Fireworks — platform GPU dengan latensi yang dioptimalkan- Mesin FireAttention (khusus); dipasarkan sebagai latensi 4x lebih rendah daripada vLLM pada konfigurasi yang setara.
- Tingkat batch pada ~50% dari tarif tanpa server untuk weight kerja non-interaktif.
- Model yang telah disesuaikan disajikan dengan tarif yang sama dengan model dasar — ​​sebuah pembeda nyata dibandingkan penyedia yang mengenakan biaya premium untuk LoRA kamu.
- Pertengahan 2026: menaikkan harga sewa GPU sesuai permintaan sebesar $1/jam efektif 1 Mei 2026. Harga volume dapat dinegosiasikan dalam skala besar.
- Sinyal keuangan: penilaian $4 miliar, 10T+ token/hari ditangani.

### Bersama — dioptimalkan secara luas

- 200+ model termasuk rilis sumber terbuka dalam beberapa hari setelah publikasi upstream.
- 50-70% lebih murah dibandingkan Replikasi pada model LLM yang setara — penentuan posisi "AI Native Cloud" adalah volume dan katalog.
- Inference + penyesuaian + training dalam satu API.

### Baseten — dioptimalkan untuk perusahaan

- Kerangka rangka: pengemasan model dengan dependensi, rahasia, konfigurasi penyajian dalam satu manifes.
- Rentang GPU dari T4 hingga B200. Penagihan per menit dengan mitigasi cold-start yang wajar.
- SOC 2 Tipe II, siap HIPAA. Pilihan fintech dan layanan kesehatan yang umum.
- Penilaian $5 miliar, Seri E Januari 2026 ($300 juta dari CapitalG, IVP, NVIDIA).

### Modal — Dioptimalkan dengan Python-asli

- Infrastruktur-sebagai-code dengan Python murni. Hiasi suatu fungsi dengan `@modal.function(gpu="A100")` dan terapkan dengan satu prompt.
- Penagihan per detik. Dingin dimulai 2-4 detik dengan pemanasan awal; <1s untuk model kecil.
- Seri B senilai $87 juta dengan penilaian $1,1 miliar (2025). Skor pengalaman pengembang terkuat dalam survei independen.

### Replikasi — luasnya multimodal

- Bayar per prediksi. Platform default untuk model gambar, video, dan audio.
- Ekosistem integrasi (plugin Zapier, Vercel, CMS).
- Kurang kompetitif dalam tarif per token LLM tetapi unggul dalam variasi multimodal.

### Skala apa pun — Ray-asli

- Dibangun di atas Ray; RayTurbo adalah mesin inference milik Anyscale (bersaing dengan vLLM).
- Terbaik untuk weight kerja Python terdistribusi yang langkah inferensinya adalah satu simpul dalam grafik yang lebih besar.
- Cluster Ray Terkelola; integrasi yang erat dengan Ray AIR dan Ray Serve.

### Per-token versus per menit — saat masing-masing menang

Per token masuk akal jika weight kerja tidak sensitif terhadap latensi dan bersifat bursty — kamu hanya membayar sesuai penggunaan. Per menit masuk akal ketika pemanfaatannya tinggi dan dapat diprediksi — kamu mengalahkan per token setelah kamu memenuhi GPU.

Aturan kasarnya: untuk weight kerja di atas ~30% pemanfaatan berkelanjutan dari GPU khusus, per menit (Baseten, Modal) mulai mengalahkan per token (Fireworks, Together). Di bawahnya, per token menang karena kamu menghindari pembayaran untuk waktu menganggur.

### Mesin khusus adalah parit sebenarnya

Setiap platform di atas vLLM dan SGLang mengklaim mesin khusus. FireAttention, RayTurbo, tumpukan inference Baseten. Pemasaran bayangan klaim mesin khusus — kerangka jujurnya adalah bahwa vLLM + SGLang mewakili sekitar 80% inference sumber terbuka produksi, dan pembeda pada layer platform adalah DX, atribusi, dan SLA.

### Nomor yang harus kamu ingat

- Penyewaan GPU kembang api: kenaikan $1/jam efektif 1 Mei 2026.
- Klaim kembang api: latensi 4x lebih rendah daripada vLLM pada konfigurasi yang setara.
- Bersama: 50-70% lebih murah daripada Replikasi di LLM.
- Penilaian Baseten: $5 miliar (Seri E, Jan 2026, putaran $300 juta).
- Penilaian modal: $1,1 miliar (Seri B, 2025).
- Denyut per menit per token di atas ~30% pemanfaatan berkelanjutan.

## Pakai

`code/main.py` membandingkan enam vendor pada weight kerja sintetis di seluruh model harga. Melaporkan token $/hari dan $/M yang efektif. Jalankan untuk menemukan titik impas antara per token dan per menit.

## Kirim

Lesson ini menghasilkan `outputs/skill-inference-platform-picker.md`. Berdasarkan profil weight kerja, SLA, dan anggaran, pilih platform inference utama dan beri nama runner-up.

## Latihan

1. Jalankan `code/main.py`. Pada pemanfaatan berkelanjutan berapa Baseten (per menit) mengalahkan Fireworks (per token) untuk model 70B pada satu H100? Turunkan sendiri crossover tersebut dan bandingkan dengan aturan praktis.
2. Produk kamu melayani pembuatan gambar plus obrolan plus ucapan-ke-teks. Pilih platform untuk setiap modalitas dan beri nama pola gateway yang menyatukannya.
3. Kembang api menaikkan harga sebesar $1/jam pada model utama kamu. Buat model dampak biaya campuran jika 40% lalu lintas kamu berpindah ke tingkat batch (diskon 50%).
4. Pelanggan teregulasi memerlukan GPU khusus SOC 2 Tipe II + HIPAA +. Tiga platform mana yang layak dan mana yang unggul di FinOps?
5. Bandingkan biaya per 1.000 prediksi untuk Llama 3.1 70B di Fireworks tanpa server, Together on-demand, Baseten berdedikasi, dan Replikasi API. Mana yang paling murah dengan 10 prediksi/hari? Pada 10.000?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Silikon khusus | "chip non-GPU" | Groq LPU, Cerebras WSE, SambaNova RDU — dioptimalkan untuk dekode |
| ApiPerhatian | "Mesin kembang api" | Kernel attention khusus; dipasarkan dengan latensi 4x lebih rendah dari vLLM |
| rangka | "Format Baseten" | Manifes kemasan model; dependensi + rahasia + konfigurasi penyajian |
| Per-token | "Harga API" | Biaya berdasarkan token yang dikonsumsi; bayar tanpa menganggur |
| Per menit | "harga khusus" | Mengisi daya berdasarkan waktu GPU jam dinding; menang dengan pemanfaatan tinggi |
| Per prediksi | "Replikasi harga" | Biaya per pemanggilan model; umum untuk gambar/video |
| RayTurbo | "Mesin skala apa pun" | Kesimpulan kepemilikan tentang Ray; bersaing dengan vLLM pada cluster Ray |
| Tingkat batch | "Diskon 50%" | Antrian non-interaktif dengan tarif lebih rendah; umum di Fireworks, OpenAI |
| Diselesaikan dengan kecepatan dasar | "Kembang api LoRA" | Mengisi permintaan yang dilayani LoRA dengan tarif model dasar (pembeda) |

## Bacaan Lanjutan

- [Harga Fireworks](https://fireworks.ai/pricing) — tarif per token, tingkat batch, sewa GPU.
- [Harga Baseten](https://www.baseten.co/pricing/) — tarif per menit, kapasitas yang berkomitmen, tingkatan perusahaan.
- [Harga Modal](https://modal.com/pricing) — tarif GPU per detik dan tingkat gratis.
- [Harga AI Bersama](https://www.together.ai/pricing) — katalog model dan tarif per token.
- [Harga Skala Apa Pun](https://www.anyscale.com/pricing) — RayTurbo dan harga Ray terkelola.
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — penilaian komparatif.
- [Infrabase — Penyedia API Inference AI 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — lanskap vendor.
