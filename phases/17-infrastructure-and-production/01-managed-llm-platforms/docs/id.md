# Platform LLM Terkelola — Batuan Dasar, Vertex AI, Azure OpenAI

> Tiga hyperscaler, tiga strategi berbeda. AWS Bedrock adalah model pasar — ​​Claude, Llama, Titan, Stability, Cohere di balik satu API. Azure OpenAI adalah kemitraan OpenAI eksklusif ditambah Provisioned Throughput Units (PTUs) untuk kapasitas khusus. Vertex AI adalah Gemini pertama dengan cerita multimodal dan konteks panjang terbaik. Pada tahun 2026 Analisis Buatan mengukur Azure OpenAI pada median ~50 ms dan Bedrock pada ~75 ms pada setara Llama 3.1 405B — PTU menjelaskan kesenjangan ini karena kapasitas khusus mengalahkan kapasitas bersama sesuai permintaan. Aturan pengambilan keputusan bukanlah "mana yang tercepat" namun "katalog model dan permukaan FinOps mana yang cocok dengan produk saya". Lesson ini mengajarkan kamu untuk memilih dengan tradeoff yang tertulis, bukan getaran.

**Type:** Learn
**Language:** Python (stdlib, pembanding biaya dan latensi mainan)
**Prerequisites:** Fase 11 (Teknik LLM), Fase 13 (Alat & Protokol)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Sebutkan tiga strategi platform (pasar vs eksklusif vs Gemini-pertama) dan cocokkan masing-masing strategi dengan kasus penggunaan produk.
- Jelaskan apa yang diberikan Provisioned Throughput Units (PTU) kepada kamu di Azure OpenAI dan mengapa Bedrock sesuai permintaan biasanya membaca ~25 ms lebih lambat pada skala 405B.
- Diagram permukaan atribusi FinOps untuk setiap platform (Profil Inference Aplikasi Batuan Dasar vs proyek per tim Vertex vs cakupan Azure + reservasi PTU).
- Tuliskan kebijakan "minimum dua penyedia" dan jelaskan mengapa penguncian vendor tunggal merupakan kesalahan yang mahal pada tahun 2026.

## Masalah

kamu memilih Claude 3.7 Soneta untuk produk kamu. Sekarang kamu perlu menyajikannya. kamu dapat memanggil Anthropic API secara langsung, atau kamu dapat memanggilnya melalui AWS Bedrock, atau melalui gateway. API langsung adalah yang paling sederhana; Bedrock menambahkan atribusi BAA, VPC endpoint, IAM, dan CloudWatch. Gateway menambahkan failover, penagihan terpadu, dan batas tarif di seluruh penyedia.

Pertanyaan yang lebih dalam adalah katalog. Jika kamu membutuhkan Claude dan Llama dan Gemini dalam produk yang sama, kamu tidak dapat membeli semuanya dari satu tempat kecuali tempat tersebut adalah Bedrock plus Vertex plus Azure OpenAI secara bersamaan. Hyperscaler tidak dapat dipertukarkan — masing-masing membuat taruhan berbeda mengenai siapa yang memiliki layer model.

Lesson ini memetakan tiga taruhan, kesenjangan latensi, kesenjangan FinOps, dan risiko lock-in.

## Konsep

### Tiga strategi

**AWS Bedrock** — pasar. Claude (Anthropic), Llama (Meta), Titan (AWS pihak pertama), Stabilitas (gambar), Cohere (embeddings), Mistral, plus gambar dan sub-katalog embedding. Satu API, satu permukaan IAM, satu ekspor CloudWatch. Taruhan Bedrock adalah bahwa pelanggan lebih menginginkan opsionalitas daripada menginginkan satu model.

**Azure OpenAI** — kemitraan eksklusif. kamu mendapatkan seri GPT-4 / 4o / 5 / o, DALL·E, Whisper, dan penyempurnaan model OpenAI di pusat data Azure. Tidak ada model non-OpenAI dalam katalog "Azure OpenAI Service" — model tersebut masuk ke Azure AI Foundry (produk terpisah). Taruhan Azure adalah OpenAI tetap menjadi yang terdepan dan pelanggan menginginkan kontrol perusahaan pada hubungan spesifik tersebut.

**Vertex AI** — Gemini yang pertama, yang lainnya yang kedua. Gemini 1.5 / 2.0 / 2.5 Flash dan Pro, plus Model Garden (pihak ketiga). Taruhan Vertex adalah konteks panjang multimodal — konteks Gemini token 1 juta adalah pembedanya.

### Kesenjangan latensi dalam skala besarAnalisis Buatan menjalankan tolok ukur yang berkelanjutan. Pada penerapan Llama 3.1 405B yang setara (bersama sesuai permintaan), latensi token pertama median Azure OpenAI adalah sekitar 50 ms; Batuan dasar sekitar 75 ms. Kesenjangan tersebut bukanlah kegagalan AWS — melainkan perbedaan model kapasitas. Azure menjual PTU (Provisioned Throughput Units), yang mencadangkan kapasitas GPU untuk penyewa kamu. Setara dengan Bedrock (Throughput yang Disediakan) ada tetapi dimulai sekitar $21/jam per unit, dan sebagian besar pelanggan tetap menggunakan layanan bersama sesuai permintaan.

Kapasitas bersama berdasarkan permintaan bersaing dengan lalu lintas pelanggan lainnya. Kapasitas khusus tidak. Jika SLA produk kamu adalah TTFT <100 ms pada P99, kamu dapat membeli PTU di Azure, membeli Throughput yang Disediakan Batuan Dasar, atau menerima varian default.

### Ekonomi Throughput yang Disediakan

Azure PTU: blok komputasi inference yang dicadangkan. Penghematan hingga ~70% vs sesuai permintaan untuk weight kerja yang dapat diprediksi. Biaya tetap per jam, apa pun lalu lintasnya — kamu membayar reservasi bahkan saat tidak ada aktivitas. Titik impas biasanya sekitar 40-60% pemanfaatan berkelanjutan.

Throughput yang Disediakan Batuan Dasar: $21-$50 per jam bergantung pada model dan wilayah. Perhitungan serupa - titik impas adalah sekitar setengah pemanfaatan puncak. Diperlukan komitmen bulanan.

Kapasitas yang disediakan Vertex dijual per SKU Gemini; harga bervariasi menurut model dan wilayah dan kurang diiklankan secara publik.

### Permukaan FinOps — pembeda sesungguhnya

**Profil Inference Aplikasi Batuan Dasar** adalah atribusi terbersih di pasar. Tandai profil dengan `team`, `product`, `feature`; merutekan semua pemanggilan model melaluinya; CloudWatch mengelompokkan biaya per profil tanpa pasca-pemrosesan. Ditambahkan pada tahun 2025, masih merupakan hyperscaler asli yang paling granular.

Atribusi **Vertex** adalah proyek per tim plus label di mana saja. kamu memodelkan setiap tim sebagai proyek GCP, memberi label pada setiap sumber daya, dan menggunakan BigQuery Billing Ekspor + DataStudio untuk rollup. Lebih banyak pekerjaan, tetapi BigQuery memberi kamu SQL sewenang-wenang pada data biaya.

**Azure** mengandalkan cakupan langganan/grup sumber daya plus tag, dengan reservasi PTU sebagai objek biaya kelas satu. Tag diwarisi dari grup sumber daya, bukan permintaan, jadi atribusi per permintaan memerlukan metrik khusus Application Insights atau gateway yang memberi cap pada header.

Polanya: Batuan dasar adalah yang asli paling bersih, Vertex paling fleksibel melalui BigQuery, Azure paling buram kecuali kamu melakukan instrumen.

### Lock-in adalah risiko tahun 2026

Komitmen single-hyperscaler baik-baik saja ketika satu model mendominasi. Pada tahun 2026, perbatasan bergerak setiap bulan — Claude 3,7 pada satu kuartal, Gemini 2,5 pada kuartal berikutnya, GPT-5 pada kuartal berikutnya. Mengunci pada satu platform akan mengunci kamu dari dua pertiga perbatasan.

Pola yang diadopsi tim kerja: minimum dua penyedia untuk setiap panggilan LLM yang penting bagi produk. Bedrock plus Azure OpenAI adalah pasangan umum — Claude dari satu, GPT dari yang lain, failover di antara keduanya, gateway yang sama. Peningkatan biaya dapat diabaikan karena rute gateway optimal; peningkatan ketersediaan selama pemadaman (seperti insiden Azure OpenAI pada Januari 2025, pemadaman AWS us-east-1) sangat menentukan.

### Residensi data, BAA, dan industri yang diatur

Batuan Dasar: BAA di sebagian besar wilayah; Titik akhir VPC; pagar pembatas. Default fintech yang umum.
Azure OpenAI: HIPAA, SOC 2, ISO 27001; Residensi data UE; default yang diatur oleh perusahaan.
Vertex: HIPAA, GDPR, data residensi per wilayah; Tumpukan kepatuhan Google Cloud.Ketiganya memenuhi kotak centang dasar. Perbedaannya terletak pada kebijakan penyimpanan data, cara log ditangani, dan apakah pemantauan penyalahgunaan membaca lalu lintas kamu (keikutsertaan default pada sebagian besar; pilihan untuk tidak ikut serta tersedia untuk perusahaan).

### Nomor yang harus kamu ingat

- TTFT median Azure OpenAI pada setara Llama 3.1 405B: ~50 mdtk (dengan PTU).
- TTFT median batuan dasar sesuai permintaan: ~75 mdtk.
- Throughput yang Disediakan Batuan Dasar: $21-$50/jam per unit.
- Titik impas Azure PTU: ~40-60% pemanfaatan berkelanjutan.
- Penghematan PTU vs sesuai permintaan dengan pemanfaatan tinggi: hingga 70%.

## Pakai

`code/main.py` membandingkan ketiga platform pada weight kerja sintetis — platform ini memodelkan ekonomi sesuai permintaan vs PTU, varians TTFT, dan fidelitas atribusi biaya. Jalankan untuk melihat di mana PTU memberikan keuntungan dan di mana luas model pasar melebihi kesenjangan TTFT.

## Kirim

Lesson ini menghasilkan `outputs/skill-managed-platform-picker.md`. Mengingat profil weight kerja (model yang diperlukan, SLA TTFT, volume harian, persyaratan kepatuhan), laporan ini merekomendasikan platform utama, cadangan, dan rencana instrumentasi FinOps.

## Latihan

1. Jalankan `code/main.py`. Pada tingkat pemanfaatan berkelanjutan apa Azure PTU mengalahkan permintaan model kelas 70B? Hitung titik impas dan bandingkan dengan kisaran 40-60% yang diiklankan.
2. Produk kamu memerlukan Claude 3.7 Sonnet dan GPT-4o. Rancang penerapan dua penyedia — yang mana menuju hyperscaler yang mana, gateway mana yang ada di depan, apa kebijakan failovernya?
3. Pelanggan layanan kesehatan yang teregulasi memerlukan BAA, residensi data AS-Timur, dan TTFT P99 sub-100 md. Pilih platform dan justifikasi dengan tiga feature spesifik.
4. kamu menemukan tagihan Batuan Dasar kamu naik 4x bulan ini tanpa perubahan lalu lintas. Tanpa Profil Inference Aplikasi, bagaimana kamu menemukan pelakunya? Dengan profil, berapa lama waktu yang dibutuhkan?
5. Baca halaman harga Azure OpenAI dan Bedrock. Untuk weight kerja Claude 100 juta token/bulan, mana yang lebih murah — API Antropik langsung, Batuan Dasar sesuai permintaan, atau Throughput yang Disediakan Batuan Dasar?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Batuan Dasar | "Layanan AWS LLM" | Pasar model di Claude, Llama, Titan, Mistral, Cohere |
| Azure OpenAI | "ObrolanGPT Azure" | Model OpenAI eksklusif di pusat data Azure dengan kontrol perusahaan |
| Verteks AI | "LLM Google" | Platform Gemini pertama dengan Model Garden untuk model pihak ketiga |
| PTU | "kapasitas khusus" | Unit Throughput yang Disediakan — GPU inference yang dicadangkan, diberi harga per jam |
| Profil Inference Aplikasi | "Penandaan batuan dasar" | Profil biaya/penggunaan per produk dengan tag, CloudWatch-native |
| Model Taman | "Katalog puncak" | Bagian model pihak ketiga Vertex AI, terpisah dari Gemini |
| Minimal dua penyedia | "Redundansi LLM" | Kebijakan menjalankan setiap jalur LLM penting di ≥2 hyperscaler |
| BAA | "Dokumen HIPAA" | Perjanjian Rekan Bisnis; diperlukan untuk PHI; disediakan oleh ketiga |
| Pemantauan penyalahgunaan | "pengamat log" | Pemindaian keamanan sisi penyedia pada prompt/output; memilih keluar di perusahaan |

## Bacaan Lanjutan- [Harga AWS Bedrock](https://aws.amazon.com/bedrock/pricing/) — kartu tarif resmi dan harga Throughput yang Disediakan.
- [Harga Layanan Azure OpenAI](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — Ekonomi PTU dan kartu tarif.
- [Harga AI Generatif Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Tingkatan Gemini dan biaya tambahan Model Garden.
- [Papan Peringkat LLM Analisis Buatan](https://artificial analysis.ai/) — tolok ukur latensi dan throughput berkelanjutan di seluruh penyedia.
- [Jurnal AI — Panduan CTO AWS Bedrock vs Azure OpenAI 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — kerangka keputusan perusahaan.
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — mekanisme atribusi secara berdampingan.
