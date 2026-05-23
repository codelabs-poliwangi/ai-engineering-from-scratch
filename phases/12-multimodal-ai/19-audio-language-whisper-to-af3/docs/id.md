# Model Bahasa Audio: Busur Bisikan ke Audio Flamingo 3

> Whisper (Radford et al., Desember 2022) menyelesaikan pengenalan ucapan — 680 ribu jam pidato multibahasa dengan pengawasan lemah, Transformer encoder-decoder sederhana, tolok ukur yang membuat setiap rilis ASR berikutnya mengutipnya. Tapi pengakuan bukanlah alasan. Menanyakan "instrumen apa yang ada dalam rekaman ini" atau "emosi apa yang diungkapkan pembicara" atau "apa yang terjadi pada menit ke-3" memerlukan pemahaman audio, bukan transkripsi. Qwen-Audio, SALMONN, LTU, dan NVIDIA's Audio Flamingo 3 (AF3, Juli 2025) secara bertahap membangun tumpukan tersebut: mempertahankan encoder kelas Whisper, memasang Q-former, melatih data instruksi teks audio, menambahkan penalaran rantai pemikiran. Lesson ini berjalan sesuai alurnya.

**Type:** Build
**Language:** Python (stdlib, spektogram log-Mel + kerangka audio Q-mantan)
**Prerequisites:** Fase 6 (Ucapan dan Audio), Fase 12 · 03 (Q-Mantan)
**Waktu:** ~180 menit

## Tujuan Pembelajaran

- Hitung spektogram log-Mel dari bentuk gelombang: windowing, FFT, bank filter, transformasi log.
- Bandingkan opsi encoder: Whisper encoder, BEATs, AF-Whisper hybrid. Ketika masing-masing menang.
- Buat audio Q-former: N kueri yang dapat dipelajari dengan memperhatikan patch spektogram.
- Jelaskan training audio-LLM berjenjang (Whisper-then-LLM) vs end-to-end: mengapa skala end-to-end lebih baik untuk penalaran.

## Masalah

Pengenalan ucapan diselesaikan dengan Whisper. OCR-of-audio adalah komoditas. Tapi "komoditas" berhenti pada transkripsi. Jika model tidak dapat mempertimbangkan apa yang didengarnya — waktu, pembicara, emosi, struktur musik, suara lingkungan — transkripsi saja tidak dapat mendorong feature produk.

Tiga rute yang jelas:

1. Cascade: Transkrip bisikan, alasan LLM atas transkrip tersebut. Berfungsi untuk skenario ucapan murni. Gagal untuk musik, audio lingkungan, multi-speaker tumpang tindih, emosi.

2. Audio-LLM ujung ke ujung: encoder audio memasukkan token audio langsung ke LLM, melewati transkripsi. Mempertahankan informasi akustik (emosi, pembicara, lingkungan). Membutuhkan training data baru.

3. Hibrida: encoder audio + decoder teks yang dapat mentranskripsikan dan memberi alasan. Qwen-Audio dan Audio Flamingo memilih rute ini.

## Konsep

### Spektogram Log-Mel: feature input

Setiap encoder audio dimulai dengan feature yang sama: spektogram log-Mel.

1. Sample ulang ke 16 kHz.
2. Transformasi Fourier waktu singkat dengan jendela 25 md, lompatan 10 md.
3. Ambil besaran hasil FFT.
4. Terapkan kumpulan filter Mel (biasanya 80 filter dengan distance log 0-8000 Hz) untuk melengkung ke frekuensi persepsi.
5. Kompres log (log(1 + x)) untuk rentang dinamis.

Hasil: bentuk array 2D (T, 80) di mana T adalah jumlah kerangka waktu. Untuk klip berdurasi 30 detik pada kecepatan bingkai 100 Hz: (3000, 80).

### Pembuat enkode Whisper

Encoder Whisper adalah Transformer gaya ViT 12 lapis yang memproses spektogram log-Mel sebagai rangkaian kerangka waktu. Output: satu vector keadaan tersembunyi per jangka waktu.

Untuk ASR, dekoder Whisper adalah Transformer attention silang yang menghasilkan token teks yang dikondisikan pada output encoder. Encoder-decoder standar.

Untuk ALM (audio-LLM), kamu menginginkan output encoder sebagai input ke LLM yang berbeda. Polanya: Pembuat enkode bisikan dibekukan, Q-mantan dapat dilatih, LLM dibekukan atau disetel.

### BEAT dan pembuat enkode khusus audio

Whisper dilatih berdasarkan data yang dominan ucapan. Ini lebih lemah untuk musik dan audio lingkungan.

BEATs (Chen et al., 2022) adalah trafo dengan pengawasan mandiri yang dilatih di AudioSet. Menangkap musik dan suara lingkungan lebih baik daripada Whisper pada jumlah parameter yang sama.AF-Whisper (hibrida Audio Flamingo 3): menggabungkan feature Whisper + BEATs sebagai input audio. Bisikan membawa sinyal linguistik, BEAT membawa sinyal akustik.

### Audio Q-mantan

Pola yang sama dengan visual Q-former BLIP-2. Sejumlah kueri yang dapat dipelajari (sering kali 32 atau 64) hadir secara silang pada bingkai output pembuat enkode audio. Kueri menjadi token audio yang digunakan oleh LLM.

Phase penyelarasan training: Q-former saja, loss kontras + teks pada pasangan audio-teks (AudioCaps, Clotho). Phase instruksi: end-to-end, mencairkan LLM, melatih data instruksi.

### Busur — SALMONN, Qwen-Audio, AF3

SALMONN (Tang et al., 2023): Bisikan + BEATs + Q-former + LLaMA. Audio-LLM terbuka pertama dengan kemampuan penalaran yang serius. Tolok ukur pada MMAU menunjukkan ~0,55 komposit.

Qwen-Audio (Chu et al., 2023): arsitektur serupa, dilatih pada dataset yang lebih kaya, disesuaikan untuk dialog multi-putaran. MMAU ~0,60.

LTU — Dengar, Pikirkan, Pahami (Gong et al., 2023): data penalaran eksplisit, fokus pada rangkaian pemikiran melalui klip audio. Lebih kecil tetapi lebih fokus.

Audio Flamingo 3 (Goel dkk., Juli 2025): SOTA terbuka saat ini. Tulang punggung LLM 8B (Qwen2 7B), BEAT concat encoder besar bisikan, Q-former 64 kueri, training pada 1 juta+ pasangan instruksi audio-teks. MMAU 0,72, cocok dengan batas kepemilikan pada beberapa subtugas.

AF3 juga memperkenalkan rangkaian pemikiran sesuai permintaan untuk audio: model secara opsional dapat memancarkan token pemikiran ("biarkan saya mengidentifikasi instrumennya terlebih dahulu: ...") sebelum jawaban akhir. Akurasi pada tugas-tugas penalaran yang kompleks meningkatkan 3-5 poin ketika berpikir diaktifkan.

### Bertingkat vs ujung ke ujung

Pipa bertingkat:

1. Bisikan mentranskripsikan audio → teks.
2. Alasan LLM melalui teks.

Berfungsi sempurna untuk "meringkas podcast ini". Gagal untuk:
- "Bagaimana suasana lagu ini?" — suasana hati ada pada suara, bukan kata-kata.
- "Siapa yang berbicara, Alice atau Bob?" — memerlukan identifikasi pembicara.
- "Pada detik berapa ledakan terjadi?" — landasan temporal yang hilang dalam teks.
- "Apakah ini audio asli atau buatan?" — Deteksi deepfake memerlukan feature akustik.

End-to-end mempertahankan sinyal akustik. Qwen-Audio dan AF3 menangani musik, lingkungan, dan emosi secara asli.

### Resep produksi 2026

Untuk produk pemahaman audio baru:

- Bertingkat jika: transkripsi adalah tujuannya, tidak ada musik, tidak ada kesimpulan emosi.
- AF3 / Qwen-Audio-family jika: musik, emosi, multi-speaker, atau penalaran audio yang kompleks.

Cascaded lebih murah dan sederhana. End-to-end lebih mumpuni.

### MMAU — tolok ukur penalaran audio

MMAU (Massive Multimodal Audio Understanding) adalah tolok ukur penalaran audio 2024-2025:

- 10.000 pasangan QA audio-teks di seluruh ucapan, musik, suara lingkungan.
- Meliputi klasifikasi, penalaran temporal, penalaran kausal, QA terbuka.
- Menguji apa saja yang terlewatkan oleh pipeline pipa bertingkat.

Buka SOTA (AF3) pada 0,72; batas kepemilikan ~0,78 (Gemini 2.5 Pro, Claude Opus 4.7). Kesenjangan yang lebih kecil dibandingkan delta terbuka-vs-tertutup VideoMME, menunjukkan bahwa audio-LLM sudah matang.

## Pakai

`code/main.py`:

- Mengimplementasikan perhitungan spektogram log-Mel di stdlib: windowing, DFT naif, bank filter Mel.
- Kerangka Audio Q-mantan: diberikan bingkai output encoder, menghitung Q, K, V, attention, dan memancarkan N token.
- Perbandingan bertingkat vs ujung ke ujung pada tugas mainan.

## Kirim

Lesson ini menghasilkan `outputs/skill-audio-llm-pipeline-picker.md`. Dengan adanya tugas audio (transkripsi, penandaan musik, inference emosi, diarisasi multi-speaker, klasifikasi lingkungan), ia memilih AF3 bertingkat, ujung ke ujung, atau hibrida.

## Latihan1. Hitung dimension spektogram log-Mel untuk klip 30 detik pada 16kHz, jendela 25ms, hop 10ms, bin 80 Mel. Bagaimana perubahannya pada 48kHz?

2. Mengapa performa Whisper buruk di musik? Feature audio apa yang dimiliki BEAT dan tidak dimiliki Whisper?

3. Audio Q-former dengan 64 kueri vs 32: pada kompleksitas tugas apa 64 membuahkan hasil? 32 menghemat komputasi untuk apa?

4. Baca AF3 Bagian 4 tentang pemikiran on-demand. Usulkan tiga tugas audio yang rantai pemikirannya paling membantu.

5. Menerapkan pipeline diarisasi minimal menggunakan output AF3. Bagaimana kamu memberi sinyal perubahan speaker?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Spektogram Log-Mel | "Feature Mel" | Array 2D (waktu, frekuensi) nilai log-magnitudo setelah bank filter Mel |
| Audio Q-mantan | "Persepsi Audio" | Kemacetan attention silang dari output encoder audio ke kueri berdurasi tetap yang memasukkan LLM |
| Bertingkat | "ASR-lalu-LLM" | Pipeline pipa tempat Whisper mentranskripsikan dan teks alasan LLM; kehilangan informasi akustik |
| ujung ke ujung | "Audio-LLM" | Feature audio masuk ke LLM langsung melalui Q-former; mempertahankan sinyal akustik |
| MENGALAHKAN | "Encoder Audio Set Audio" | Transformer SSL dilatih di AudioSet; kuat pada musik + suara lingkungan |
| MMAU | "Bangku penalaran audio" | 10 ribu pasangan QA di bidang pidato, musik, lingkungan; standar evaluasi 2024 |
| Pemikiran sesuai permintaan | "CoT Audio" | Model secara opsional dapat mengeluarkan token penalaran sebelum jawaban akhir, meningkatkan akurasi 3-5 poin |

## Bacaan Lanjutan

- [Radford dkk. — Bisikan (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu dkk. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel dkk. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang dkk. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong dkk. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
