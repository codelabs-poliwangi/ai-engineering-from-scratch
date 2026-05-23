# Dari CLIP ke BLIP-2 — Q-Mantan sebagai Jembatan Modalitas

> CLIP menyelaraskan gambar dan teks tetapi tidak dapat menghasilkan keterangan, menjawab pertanyaan, atau mengadakan percakapan. BLIP-2 (Salesforce, 2023) menyelesaikannya dengan jembatan kecil yang dapat dilatih: 32 vector kueri yang dapat dipelajari menangani feature ViT yang dibekukan melalui attention silang, lalu dimasukkan langsung ke aliran input LLM yang dibekukan. Parameter jembatan 188M menghubungkan LLM 11B ke ViT-g/14. Setiap VLM berbasis adaptor hingga tahun 2026 — MiniGPT-4, InstructBLIP, sepupu LLaVA — adalah turunannya. Lesson ini membaca arsitektur Q-Former, menjelaskan training dua tahapnya, dan membuat versi mainan yang memasukkan token visual ke dalam decoder teks beku.

**Type:** Build
**Language:** Python (stdlib, attention silang + demo kueri yang dapat dipelajari)
**Prerequisites:** Fase 12 · 02 (KLIP), Fase 7 (Transformer)
**Waktu:** ~180 menit

## Tujuan Pembelajaran

- Jelaskan mengapa hambatan yang dapat dilatih antara pembuat enkode visi beku dan LLM beku mengalahkan penyempurnaan menyeluruh dalam hal biaya dan stabilitas.
- Menerapkan blok attention silang di mana serangkaian kueri tetap yang dapat dipelajari menangani feature gambar eksternal.
- Ikuti prapelatihan dua phase BLIP-2: representasi (ITC + ITM + ITG) lalu generatif (kehilangan LM dengan dekoder beku).
- Bandingkan Q-Former dengan proyektor MLP sederhana yang digunakan di LLaVA dan berdebatlah ketika setiap pilihan menang.

## Masalah

kamu memiliki ViT beku yang menghasilkan 256 token patch redup 1408 per gambar. kamu memiliki LLM 7B beku yang mengharapkan embedding token redup 4096. Jembatan yang jelas — layer linier dari 1408 hingga 4096 — berfungsi, tetapi memasukkan semua 256 token patch ke dalam konteks LLM memerlukan 256 token tambahan per gambar. Lebih dari 32 gambar yang merupakan 8192 token dikonsumsi oleh modalitas visual saja.

Pertanyaan BLIP-2: dapatkah kamu mengompresi representasi gambar 256 token menjadi token yang jauh lebih sedikit (katakanlah 32) sambil menyimpan informasi yang cukup untuk LLM untuk memberi keterangan, menjawab pertanyaan, dan memberi alasan tentang gambar tersebut? Dan bisakah kamu melatih jembatan ini tanpa menyentuh tulang punggung yang membeku, sehingga biaya training tetap sesuai dengan parameter jembatan?

Jawabannya: seorang Q-Mantan. 32 vector "kueri" yang dapat dipelajari yang menangani silang token patch ViT, menghasilkan ringkasan visual 32 token yang digunakan LLM. Total parameter 188 juta. Dilatih dengan tujuan kontrastif, serasi, dan generatif sebelum menyentuh LLM.

## Konsep

### Kueri yang dapat dipelajari

Trik inti Q-Former: alih-alih membiarkan token teks LLM menangani patch gambar, perkenalkan kumpulan 32 vector kueri baru yang dapat dipelajari `Q` dan biarkan *mereka* menangani patch gambar. Kueri adalah parameter model — kueri tersebut dipelajari selama training dan 32 kueri yang sama digunakan untuk setiap gambar.

Setelah attention silang, setiap kueri menyimpan ringkasan terkompresi dari gambar — "deskripsikan objek utama", "deskripsikan latar belakang", "hitung objek", dll. Kueri tidak secara harfiah mengkhususkan pada label semantik; mereka mempelajari pengkodean apa pun yang membuat loss hilir menurun.

### Arsitektur

Q-Former adalah trafo kecil (12 layer, ~100 juta parameter) dengan dua jalur:

1. Jalur kueri: 32 vector kueri mengalir melalui attention mandiri (di antara mereka sendiri), lalu attention silang pada token patch ViT yang dibekukan, lalu FFN.
2. Jalur teks: encoder teks mirip BERT membagikan attention mandiri dan weight FFN dengan jalur kueri. Attention silang dinonaktifkan untuk jalur teks.Pada waktu training kedua jalur berjalan. Kueri dan teks berinteraksi melalui attention mandiri bersama, yang berarti kueri dapat mengkondisikan teks untuk tugas yang memerlukannya (ITM, ITG). Pada waktu inference untuk handoff VLM, hanya kueri yang mengalir, menghasilkan 32 token visual.

### Training dua phase

Latihan awal BLIP-2 dalam dua phase:

Phase 1: pembelajaran representasi (tanpa LLM). Tiga loss:
- ITC (kontrastif gambar-teks): Kontras gaya CLIP antara token kueri gabungan dan token CLS teks.
- ITM (pencocokan gambar-teks): pengklasifikasi biner — apakah pasangan gambar-teks ini cocok? Penambangan negatif yang keras.
- ITG (pembuatan teks berbasis gambar): LM kausal langsung pada teks, dikondisikan pada kueri. Memaksa kueri untuk menyandikan konten yang dapat dihasilkan teks.

Hanya kereta Q-Mantan. ViT dibekukan. Tidak ada LLM yang terlibat.

Phase 2: pembelajaran generatif. Lampirkan LLM beku (OPT-2.7B atau Flan-T5-XL, dll.). Proyeksikan 32 output kueri ke embedding LLM redup melalui layer linier kecil. Tambahkan mereka ke prompt teks. Latih hanya proyeksi linier dan Q-Former pada loss LM pada rangkaian prompt + gambar + keterangan yang digabungkan.

Setelah phase 2, proyeksi Q-Former + menjadi adaptor visual lengkap. Pada inference: gambar → ViT → Q-Mantan → proyek linier → ditambahkan ke teks → LLM beku mengeluarkan output.

### Parameter ekonomi

BLIP-2 dengan ViT-g/14 (1.1B, beku) + OPT-6.7B (6.7B, beku) + Q-Former (188M, dilatih) = total 8B, 188M dilatih. Q-Former sendiri adalah ~2,4% dari parameter tumpukan penuh. Biaya training mencerminkan hal ini: jumlah hari untuk beberapa A100 vs minggu untuk end-to-end.

Kualitas: BLIP-2 menyamai atau mengalahkan Flamingo-80B pada VQA zero-shot sekaligus 50x lebih kecil. Jembatan itu berfungsi.

### InstructBLIP dan Q-Former yang mengetahui instruksi

InstructBLIP (2023) memperluas Q-Former dengan input tambahan: teks instruksi itu sendiri. Pada waktu attention silang, kueri sekarang memiliki akses ke patch gambar dan instruksi. Kueri dapat mengkhususkan per-instruksi ("menghitung mobil", "menggambarkan suasana hati") daripada mempelajari satu ringkasan tetap. Tolok ukur perolehan tugas-tugas yang tertunda.

### MiniGPT-4 dan pendekatan khusus proyektor

MiniGPT-4 mempertahankan Q-Former tetapi hanya melatih proyeksi linier output sambil membekukan yang lainnya. Murah, namun biaya adalah kualitas — pertanyaannya adalah milik BLIP-2, bukan milik kamu. Bagus untuk iterasi cepat, bukan arsitektur terbaik.

### Mengapa LLaVA menjadi lebih sederhana

LLaVA (2023, Lesson 12.05) menggantikan Q-Former dengan MLP 2 lapis biasa yang memproyeksikan setiap token patch ViT ke dalam ruang LLM — 576 token per gambar untuk grid 24x24, semuanya diumpankan ke LLM. Kompresi lebih buruk tetapi membiarkan LLM menangani patch mentah. Pada saat itu hal ini masih kontroversial; pada akhir tahun 2023 MLP menjadi dominan karena data instruksi visual (LLaVA-Instruct-150k) membuktikan bahwa MLP dapat dilatih untuk mempertahankan sinyal yang cukup. Imbalannya: Konteks LLaVA terisi lebih cepat, namun dapat diskalakan secara alami ke multi-gambar dan video.

Pada tahun 2026, pembagian bidang: Q-Former bertahan ketika anggaran token penting (video panjang, banyak gambar); Proyektor MLP mendominasi di mana kualitas mentah per token adalah prioritasnya.

### Attention silang yang terjaga keamanannya: Flamingo, nenek moyang

Flamingo (Lesson 12.04) mendahului BLIP-2 dan menggunakan ide attention silang yang sama tetapi pada setiap layer LLM yang dibekukan, bukan sebagai jembatan tunggal. BLIP-2 menunjukkan kamu dapat mengompres ke layer input saja dan masih berfungsi. Gemini dan Idefics menggabungkan keduanya: token input yang disisipkan ditambah attention silang opsional yang terjaga keamanannya untuk beberapa pengambilan gambar dalam konteks.

### Keturunan 2026- Q-Mantan: BLIP-2, InstructBLIP, MiniGPT-4, dan sebagian besar model bahasa video karena alasan anggaran token.
- Pengambil sample ulang persepsi: varian Flamingo (Lesson 12.04); Keluarga Idefis, Elang, OmniMAE.
- Proyektor MLP: LLaVA, LLaVA-NeXT, LLaVA-OneVision, Cambrian-1.
- Kelompok attention: VILA, PaliGemma.

Keempatnya valid. Pertanyaan penentunya adalah apakah kamu dibatasi pada anggaran token atau kualitas per token.

## Pakai

`code/main.py` membangun attention silang gaya Q-Mantan stdlib:

1. Simulasikan 256 token patch gambar (redup 128).
2. Buat instance 32 kueri yang dapat dipelajari (redup 128).
3. Jalankan attention silang produk titik-skala (Q dari kueri, K/V dari patch).
4. Proyeksikan ke LLM-redup (512) melalui layer linier.
5. Keluarkan 32 token visual siap LLM.

Semua matematika dengan Python murni (loop bersarang di atas vector). Mainan tapi bentuknya benar. Matrix attention-berat dicetak sehingga kamu dapat melihat patch mana yang diambil dari setiap kueri.

## Kirim

Lesson ini menghasilkan `outputs/skill-modality-bridge-picker.md`. Mengingat konfigurasi target VLM (jumlah token encoder visi, anggaran konteks LLM, batasan penerapan, target kualitas), ini merekomendasikan resampler Q-Former vs MLP vs Perceiver dengan justifikasi singkat dan perkiraan jumlah parameter untuk setiap jembatan.

## Latihan

1. Terapkan blok attention silang di PyTorch. Verifikasi bahwa dengan 32 kueri dan 256 kunci/nilai, matrix weight attention adalah 32 x 256 dan setiap baris berjumlah 1 setelah softmax.

2. Pada BLIP-2 phase 1 Q-Former mengalami tiga loss secara bersamaan: ITC, ITM, ITG. Tuliskan tanda tangan penerusan untuk masing-masing dalam code semu. Manakah yang memerlukan jalur encoder teks untuk aktif?

3. Bandingkan jumlah parameter: Q-Former (12 layer, 768 tersembunyi) vs proyektor MLP 2 layer (1408 → 4096, dua layer). Pada skala LLM berapakah biaya 188 juta Q-Former menghasilkan efisiensi training?

4. Baca Bagian 3.2 makalah BLIP-2 (arXiv:2301.12597) tentang bagaimana Q-Former diinisialisasi. Jelaskan mengapa inisialisasi dari basis BERT (bukan acak) mempercepat konvergensi.

5. Untuk video berdurasi 10 menit pada 1 FPS yang diambil sampelnya menjadi 60 bingkai, hitung biaya token per bingkai pada (Q-Former → 32 token/frame) vs (proyektor MLP → 576 token/frame). Mana yang cocok dengan jendela konteks LLM 128 ribu token?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Q-Mantan | "Menanyakan trafo" | Transformer kecil dengan 32 vector kueri yang dapat dipelajari yang menangani feature ViT yang dibekukan |
| Kueri yang dapat dipelajari | "Permintaan lembut untuk penglihatan" | Seperangkat parameter tetap yang berfungsi sebagai sisi kueri attention silang; dipelajari per model, dibagikan ke semua input |
| Attention silang | "Q dari sini, K/V dari sana" | Attention jika kueri, kunci, dan nilai berasal dari sumber berbeda; bagaimana pertanyaan menarik dari patch ViT |
| ITC | "Kontrasif gambar-teks" | Loss gaya CLIP diterapkan pada kueri gabungan Q-Mantan vs teks CLS |
| ITM | "Pencocokan gambar-teks" | Pengklasifikasi biner pada pasangan yang ditambang negatif-keras; memaksa kueri untuk membedakan ketidakcocokan yang sangat halus |
| ITG | "Pembuatan teks berdasarkan gambar" | Kehilangan LM kausal saat teks dihasilkan berdasarkan kueri; memaksa kueri untuk menyandikan konten yang dapat didekodekan teks |
| Pra-training dua phase | "Representasi kemudian generatif" | Phase 1 melatih Q-Former sendiri (ITC/ITM/ITG); Phase 2 melampirkan LLM beku dan hanya melatih proyeksi + Q-Mantan |
| Tulang punggung beku | "Jangan menyempurnakan" | Encoder visi dan weight LLM telah diperbaiki; hanya jembatan kereta api |
| Kepala proyeksi | "Linear ke LLM redup" | Pemetaan layer linier akhir output Q-Mantan ke dimension embedding LLM |
| Pengambil sample ulang persepsi | "Versi Flamingo" | Attention silang kueri yang dapat dipelajari serupa, digunakan oleh Flamingo di setiap layer, bukan sebagai jembatan tunggal |

## Bacaan Lanjutan

- [Li dkk. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — makalah inti.
- [Li dkk. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — pendahulu dari trio ITC/ITM/ITG.
- [Li dkk. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — "align before fuse" — nenek moyang konseptual training phase 1.
- [Dai dkk. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — Q-Former yang sadar akan instruksi.
- [Zhu dkk. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — pendekatan khusus proyektor.
- [Jaegle dkk. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — arsitektur umum untuk attention silang kueri yang dapat dipelajari.
