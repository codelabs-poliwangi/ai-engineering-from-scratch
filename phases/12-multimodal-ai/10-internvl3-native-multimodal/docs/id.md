# InternVL3: Pra-training Multimodal Asli

> Setiap VLM terbuka sebelum InternVL3 mengikuti resep tiga langkah yang sama: ambil LLM teks yang dilatih pada triliunan token teks, pasang encoder visi, lalu sempurnakan jahitannya. Ini berfungsi tetapi memiliki hutang penyelarasan - teks LLM telah menghabiskan seluruh anggaran pra-training-nya pada teks murni dan tidak memahami token visual secara asli. Ketika kamu menambahkan visi post-hoc, LLM harus mempelajari kembali bagaimana menghubungkan input visual dengan penalaran teksnya tanpa melupakan teksnya. InternVL3 (Zhu et al., April 2025) menolak pendekatan post-hoc: satu proses prapelatihan, teks dan multimodal disisipkan dari langkah pertama. Hasilnya cocok dengan Gemini 2.5 Pro di MMMU-Pro pada parameter terbuka 78B. Lesson ini membahas kasus pra-training asli dan perubahan apa yang terjadi saat kamu melakukannya.

**Type:** Learn
**Language:** Python (stdlib, mixer korpus training)
**Prerequisites:** Phase 12 · 05, Phase 12 · 07 (resep)
**Waktu:** ~120 menit

## Tujuan Pembelajaran

- Jelaskan mengapa training VLM post-hoc menumpuk hutang penyelarasan, dengan menyebutkan tiga gejala yang dapat diukur (lupa bencana, penyimpangan jawaban, inkonsistensi teks visual).
- Jelaskan campuran korpus pra-training asli InternVL3 dan mengapa rasio teks : interleaved : keterangan penting.
- Bandingkan V2PE (pengkodean posisi visual variabel) dengan M-RoPE Qwen2-VL.
- Beri nama optimization penerapan Visual Resolusi Router (ViR) dan Decoupled Vision-Language (DvD).

## Masalah

Training VLM post-hoc adalah defaultnya. LLaVA, BLIP-2, Qwen-VL, Idefics — semuanya mengambil LLM yang sudah dilatih sebelumnya (Llama, Vicuna, Qwen, Mistral) dan menambahkan visi. Tahapan training biasanya terlihat seperti:

1. Frozen LLM + frozen vision encoder + proyektor yang dapat dilatih, dilatih tentang pasangan teks untuk menyelaraskan embeddings.
2. Cairkan LLM, latih data instruksi (LLaVA-Instruct, ShareGPT4V).
3. Penyempurnaan khusus tugas opsional.

Tiga gejala penyelarasan utang muncul:

- Lupa yang sangat parah. VLM post-hoc melupakan keterampilan hanya teks. Skor GSM8K turun 5-10 poin. Skor Hellaswag turun. Agen teks murni mengalami kemunduran.
- Jawaban melayang. Ungkapan kecil dari pertanyaan visual yang sama mendapatkan jawaban berbeda. Encoder visi terhubung ke LLM dengan ikatan yang lebih lemah dibandingkan token LLM itu sendiri.
- Inkonsistensi visual-teks. VLM dapat mendeskripsikan gambar dengan benar dan kemudian menjawab pertanyaan yang bertentangan dengan deskripsinya. Token visual tidak berpartisipasi dalam pemeriksaan konsistensi internal LLM seperti halnya teks.

Gejala-gejala ini terdokumentasi dengan baik. MM1.5 Bagian 4 mengkuantifikasinya. Ablasi LLaVA-OneVision mengisyaratkan hal itu. Pra-training asli adalah jawabannya.

## Konsep

### Training awal multimodal asli

InternVL3 berlatih dari awal pada korpus multimodal asli dari langkah pertama. Campurannya adalah:

- 40% data hanya teks (FineWeb, Proof-Pile-2, dll.)
- 35% data gambar-teks yang disisipkan (OBELICS, gaya MMC4)
- 20% data teks gambar berpasangan
- 5% data video-teks

Token visi, token teks, dan interaksi lintas modal semuanya berpartisipasi dalam loss yang sama dari langkah gradient pertama. Tidak ada pra-training penyelarasan, tidak ada phase pembekuan proyektor, tidak ada bencana lupa untuk pulih.

Training adalah satu phase untuk model dasar. Penyetelan instruksi mengikuti, tetapi model dasar sudah memahami token visual sebagai warga kelas satu.

### V2PE (pengkodean posisi visual variabel)

Qwen2-VL menggunakan M-RoPE dengan alokasi sumbu tetap. InternVL3 memperkenalkan V2PE: pengkodean posisi bervariasi per jenis modalitas (teks, gambar, video) dengan penskalaan yang dapat dipelajari. Dalam praktiknya:- Token teks mendapatkan posisi 1D (indeks teks).
- Tambalan gambar mendapatkan posisi 2D (baris, kolom).
- Bingkai video mendapatkan posisi 3D (waktu, baris, kolom).

Ketiganya berbagi basis frekuensi RoPE yang sama, namun alokasi peredupan tersembunyi per pita merupakan parameter yang dipelajari dan bukan pemisahan tetap. Kebebasan untuk menukar resolusi frekuensi temporal dan spasial selama pra-training.

Klaim ablasi V2PE: 1-2 poin pada benchmark video melalui M-RoPE pada komputasi yang sama. Bukan revolusi, tapi bersih.

### Router Resolusi Visual (ViR)

Optimization penerapan. Tidak semua gambar memerlukan pengkodean resolusi penuh. Foto dengan satu objek dengan detail rendah akan membuang-buang token jika dikodekan pada resolusi asli 1280 piksel. ViR adalah pengklasifikasi kecil yang memprediksi resolusi minimum yang diperlukan untuk menjawab pertanyaan, sebelum pengkodean.

Perutean memiliki tiga tingkatan: resolusi rendah (256 token), sedang (576), tinggi (2048+). Untuk 60% kueri dalam lalu lintas produksi, rendah atau sedang sudah cukup. Efek bersih: throughput 2-3x dengan kualitas yang sama.

### Penerapan Vision-Language (DvD) yang dipisahkan

Saat kamu menyajikan VLM besar, pembuat enkode visi berjalan satu kali per gambar tetapi LLM berjalan secara otomatis untuk setiap token output. Kedua komponen memiliki hambatan yang berbeda (visi = bandwidth memori GPU untuk konv + attention; LLM = cache KV). DVD membaginya menjadi GPU terpisah dengan streaming di antaranya.

Untuk model encoder 8B + 400M, DvD kira-kira menggandakan throughput per node vs lokasi yang berdekatan.

### Kualitas satu phase vs multi phase

Klaim benchmark utama InternVL3: pada parameter 78B, cocok dengan MMMU-Pro Gemini 2.5 Pro. Pada 38B, cocokkan GPT-4o. Di 8B, pimpin papan peringkat 8B terbuka. Semuanya dalam satu phase pra-latihan + resep penyetelan instruksi.

Hipotesis penyelarasan-utang dapat diukur: InternVL3-8B kehilangan lebih sedikit poin tolok ukur teks (MMLU, GSM8K) dibandingkan Qwen2.5-VL-7B per unit perolehan tolok ukur visi. Model ini lebih bersifat generalis karena training bersifat satu bagian, bukan dua bagian.

### InternVL3.5 dan InternVL-U

InternVL3.5 (Agustus 2025) menskalakan resepnya. Pendekatan pra-training asli yang sama, lebih banyak data, lebih banyak parameter. Peningkatan MMMU bersifat bertahap.

InternVL-U (2026) menambahkan generasi terpadu — output gambar melalui kepala MMDiT di atas tulang punggung yang sama. Huruf "U" adalah singkatan dari "Pemahaman + generasi", yang merupakan model terpadu gaya Transfusi (Lesson 12.13). Tulang punggung pra-training asli yang sama mendukung pemahaman dan generasi kepala.

### Pertukaran dari prapelatihan asli

Pra-training asli tidak gratis:

- Hitung. Melatih VLM baru dari awal biayanya sama dengan melatih LLM teks — jutaan jam GPU. Adaptasi post-hoc menggunakan kembali weight LLM yang ada, menghemat sebagian besar biaya.
- Data. Korpora gambar-teks yang disisipkan dalam skala besar jarang terjadi. OBELICS adalah 141 juta dokumen; MMC4 adalah 571 juta. Teks saja dikirimkan dengan token 15T. Kelangkaan data pra-training multimodal merupakan kendala yang sulit.
- Penggunaan kembali Base-LLM. Pra-training asli memberikan pilihan untuk mengikuti LLM baru nanti. Post-hoc memungkinkan kamu menukar Llama-3.1 dengan Llama-4 hanya dengan melatih ulang adaptornya.

Taruhan yang dibuat InternVL3: hutang penyelarasan lebih buruk daripada loss penggunaan kembali. Tolok ukur mendukung klaim tersebut. Biaya produksi menghalangi laboratorium di masa depan untuk melakukan replikasi dengan biaya murah. VLM post-hoc akan tetap ada karena harganya tetap lebih murah untuk sebagian besar proyek.

## Pakai

`code/main.py` adalah mixer korpus training dan simulator router ViR. Dia:- Mengambil campuran korpus target (%teks, %interleaved, %caption, %video) dan menghitung langkah yang diharapkan per modalitas.
- Mensimulasikan perutean ViR pada kumpulan kueri (distribusi: 50% detail rendah, 30% sedang, 20% detail tinggi) dan melaporkan jumlah token rata-rata.
- Melaporkan perkiraan throughput DvD yang diberikan encoder vs LLM FLOP.
- Mencetak pra-training post-hoc vs asli secara berdampingan dalam parameter, komputasi, data, dan gejala penyelarasan utang yang diharapkan.

## Kirim

Lesson ini menghasilkan `outputs/skill-native-vs-posthoc-auditor.md`. Dengan adanya usulan rencana training VLM, program ini akan mengaudit apakah akan menggunakan sistem native atau post-hoc, menandai risiko penyelarasan utang, dan merekomendasikan campuran korpus. Gunakan saat kamu mengukur proyek VLM terbuka baru dan perlu memilih strategi training.

## Latihan

1. Perkirakan delta komputasi antara InternVL3-8B (pretrain asli) dan LLaVA-OneVision-7B (post-hoc). Kira-kira rasio jam GPU? Apa yang menjelaskan kesenjangan tersebut?

2. InternVL3 melaporkan 40% teks / 35% sisipan / 20% teks / 5% video. Jika tugas target kamu berisi banyak video, usulkan rasio baru dan jelaskan mengapa model dasar masih memerlukan data teks dan teks yang substansial.

3. Baca MM1.5 Bagian 4 tentang lupa. Sebutkan tolok ukur yang tepat di mana training post-hoc menunjukkan regresi terbesar. Berapa biaya regresinya?

4. ViR merutekan 60% lalu lintas ke pengkodean resolusi rendah. Jenis kueri apa yang salah dirutekan (dikirim ke resolusi rendah saat resolusi tinggi diperlukan)? Usulkan tiga mode kegagalan router.

5. DvD membagi vision dan LLM menjadi GPU terpisah. Dalam pola lalu lintas apa DvD merugikan throughput alih-alih membantu?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Pra-training multimodal asli | "Dari awal bersama" | Token teks + gambar + video ikut serta dalam loss dari langkah 1, bukan dibaut nanti |
| Penyelarasan hutang | "Penalti pasca-hoc" | Regresi terukur dalam keterampilan teks dan konsistensi jawaban yang berasal dari menggabungkan visi ke LLM yang dibekukan |
| V2PE | "Pengkodean pos visual variabel" | Alokasi pengkodean posisi yang dapat dipelajari per-modalitas; Penerus M-RoPE InternVL3 |
| ViR | "Router resolusi" | Pengklasifikasi kecil yang memilih resolusi minimum yang diperlukan per kueri sebelum pengkodean, menyimpan token inference |
| DVD | "Penerapan terpisah" | Encoder visi di satu GPU, LLM di GPU lain, dengan handoff aliran; menggandakan throughput untuk VLM besar |
| MagangVL-U | "Pemahaman + generasi yang terpadu" | Tindak lanjut tahun 2026 yang menambahkan kepala pembuat gambar ke tulang punggung pra-training asli |
| Korpus disisipkan | "OBELIK / MMC4" | Dokumen dengan teks dan gambar dalam urutan bacaan alami; bahan baku prapelatihan native |

## Bacaan Lanjutan

- [Chen dkk. — MagangVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu dkk. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang dkk. — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)
