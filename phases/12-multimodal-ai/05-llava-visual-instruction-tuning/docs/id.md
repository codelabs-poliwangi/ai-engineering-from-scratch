# LLaVA dan Penyetelan Instruksi Visual

> LLaVA (April 2023) adalah arsitektur multimodal yang paling banyak ditiru di planet ini. Ini menggantikan Q-Former BLIP-2 dengan MLP 2 lapis, menggantikan attention silang Flamingo dengan rangkaian token naif, dan dilatih pada 158 ribu putaran instruksi visual yang dihasilkan oleh GPT-4 dari teks hanya teks. Setiap praktisi yang membuat VLM antara tahun 2023 dan 2026 membuat beberapa varian LLaVA. LLaVA-1.5 menambahkan AnyRes. LLaVA-NeXT meningkatkan resolusi. LLaVA-OneVision menyatukan gambar, multi-gambar, dan video dalam satu resep. Lesson ini membaca resep, mengimplementasikan proyektor, dan menjelaskan mengapa "yang lebih sederhana menang".

**Type:** Build
**Language:** Python (stdlib, proyektor + pembuat templat instruksi)
**Prerequisites:** Fase 12 · 02 (CLIP), Fase 11 (Teknik LLM — penyetelan instruksi)
**Waktu:** ~180 menit

## Tujuan Pembelajaran

- Membangun proyektor MLP 2 lapis yang memetakan embedding patch ViT (redup 1024) ke redup embedding LLM (redup 4096).
- Jalani resep dua phase LLaVA: (1) penyelarasan proyektor pada 558 ribu pasang teks, (2) penyetelan instruksi visual pada 158 ribu putaran yang dihasilkan GPT-4.
- Buat prompt format LLaVA dengan placeholder token gambar, prompt sistem, dan giliran pengguna/asisten.
- Jelaskan mengapa komunitas berpindah dari Q-Former ke MLP meskipun Q-Former menang dalam anggaran token.

## Masalah

Q-Former BLIP-2 (Lesson 12.03) mengompresi gambar menjadi 32 token. Bersih, efisien, bagus untuk benchmark. Tapi ada dua masalah.

Pertama, Q-Former dapat dilatih tetapi kekalahannya bukanlah tugas akhir. Phase 1 melatih ITC+ITM+ITG. Phase 2 melatih loss LM. Kueri mempelajari beberapa representasi perantara yang kemudian harus didekodekan oleh LLM. Informasi hilang dalam kemacetan.

Kedua, Q-Former membutuhkan 188 juta parameter, dan pada skala LLaVA tahun 2023 kamu harus mendesainnya bersama dengan target LLM kamu. Ubah LLM, latih kembali Q-Former. Ubah encoder visi, latih kembali. Setiap kombinasi adalah proyek R&D yang terpisah.

Jawaban LLaVA memalukan dalam kesederhanaannya: ambil 576 token patch ViT, teruskan masing-masing melalui MLP 2 lapis (`1024 → 4096 → 4096`), dan buang semua 576 ke dalam urutan input LLM. Tidak ada hambatan. Tidak ada training awal phase 1 dengan tujuan aneh. Latih saja MLP pada loss LM langsung.

Dari mana datanya berasal? Wawasan kedua LLaVA: gunakan GPT-4 (hanya teks) untuk menghasilkan data instruksi. Masukkan GPT-4 keterangan COCO dan data kotak pembatas untuk sebuah gambar, minta gambar tersebut menghasilkan percakapan, deskripsi, dan pertanyaan penalaran yang kompleks. 158k instruksi-respons berubah secara gratis. Tidak ada anotasi manusia.

Hasilnya: VLM yang berjalan pada 8 A100 selama satu hari, mengalahkan Flamingo di MMMU, dan mengirimkan pos pemeriksaan terbuka yang dapat diperluas oleh komunitas. Pada akhir tahun 2023, ia telah menghasilkan 50+ cabang.

## Konsep

### Arsitektur

LLaVA-1.5 pada 13B:
- Pengkode visi: CLIP ViT-L/14 @ 336 (dibekukan selama phase 1, opsional dicairkan phase 2).
- Proyektor: MLP 2 lapis dengan activation GELU, `1024 → 4096 → 4096`.
- LLM: Vicuna-13B (kemudian Llama-3.1-8B).

Meneruskan prompt gambar + teks:

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

Gambar tersebut menempati 576 token konteks LLM. Pada konteks 2048, tersisa 1.472 token untuk teks. Pada konteks 32k, ini adalah kesalahan pembulatan.

### Phase 1: penyelarasan proyektor

Bekukan ViT. Bekukan LLM. Latih hanya MLP 2 lapis. Dataset: 558 ribu pasangan keterangan gambar (LAION-CC-SBU). Loss: pemodelan bahasa pada keterangan, dikondisikan pada token gambar yang diproyeksikan.Dalam satu epoch di batch 128 hal ini dilakukan dalam beberapa jam. Proyektor belajar memetakan ruang ViT ke ruang LLM. Tidak ada pengawasan khusus tugas.

### Phase 2: penyetelan instruksi visual

Cairkan proyektor (masih bisa dilatih). Cairkan LLM (biasanya seluruhnya, terkadang LoRA). Berlatihlah pada 158k putaran instruksi visual.

Data instruksi adalah triknya. Liu dkk. dihasilkan oleh:
1. Ambil gambar COCO.
2. Ekstrak deskripsi teks (5 keterangan manusia + daftar kotak pembatas).
3. Kirim ke GPT-4 dengan tiga templat cepat:
   - Percakapan: "Hasilkan dialog bolak-balik antara pengguna dan asisten tentang gambar ini."
   - Deskripsi mendetail: "Berikan deskripsi gambar yang kaya dan mendetail."
   - Penalaran kompleks: "Ajukan pertanyaan yang memerlukan penalaran tentang gambar, lalu jawablah."
4. Parsing output GPT-4 menjadi pasangan (instruksi, respons).

Semua ini tidak menyentuh gambar secara langsung — hanya deskripsi teks. GPT-4 berhalusinasi konten gambar yang masuk akal. Sedikit berisik, tapi berhasil: 158 ribu putaran sudah cukup untuk membuka dialog.

### Mengapa komunitas menyalin ini

- Tidak ada loss khusus phase 1 yang perlu disesuaikan. Loss LM seluruhnya.
- Proyektor berlatih dalam hitungan jam, bukan hari.
- LLM dapat ditukar (LLaVA-Llama2, LLaVA-Mistral, LLaVA-Llama3) hanya dengan melatih ulang proyektornya.
- Pipeline data instruksi visual menggunakan GPT-4 dan murah untuk dibuat ulang untuk domain baru.

### LLaVA-1.5 dan LLaVA-NeXT

LLaVA-1.5 (Oktober 2023) menambahkan:
- Data tugas akademik (VQA, OKVQA, RefCOCO) dicampur ke dalam penyetelan instruksi.
- System prompt yang lebih baik.
- 2048 → konteks 32k.

LLaVA-NeXT (Januari 2024) menambahkan:
- AnyRes: membagi gambar beresolusi tinggi menjadi kisi 2x2 atau 1x3 dengan potongan 336x336, ditambah satu gambar mini global beresolusi rendah. Setiap hasil panen menjadi 576 token; total sekitar 2880 token visual per gambar. Tugas OCR dan grafik melonjak.
- Campuran data instruksi yang lebih baik dengan ShareGPT4V (teks GPT-4V berkualitas tinggi).
- LLM basis yang lebih kuat (Mistral-7B, Yi-34B).

### LLaVA-OneVision

Lesson 12.08 membahas OneVision secara mendalam. Versi pendek: proyektor yang sama, tetapi dilatih dengan kurikulum yang mencakup gambar tunggal, multi-gambar, dan video dalam satu model dengan anggaran token visual bersama.

### Perbandingan dengan Q-Former

| | Q-Mantan (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| Token visual per gambar | 32 | 576 (basis) atau 2880 (AnyRes) |
| Param yang dapat dilatih | 188M + LM | 40M + LM |
| Loss phase 1 | ITC+ITM+ITG | Hanya LM |
| Kunjungan LLM | Membutuhkan training ulang | Tukar dengan training ulang minimal |
| Multi-gambar | Canggung | Alami (koncat) |
| Video | Canggung | Alami (concat per bingkai) |
| Anggaran token | Kecil | Besar |

MLP unggul dalam kesederhanaan dan fleksibilitas token. Q-Mantan menang dengan anggaran token. Pada akhir tahun 2023, anggaran token tidak lagi menjadi kendala yang mengikat (konteks LLM tumbuh menjadi 32k-128k+) dan kesederhanaan mendominasi.

### Format prompt

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` adalah token pengganti. Sebelum tokenization, diganti dengan 576 token visual (atau 2880 dengan AnyRes). Tokenizer melihat urutan yang sedikit lebih panjang daripada saat dilatih, namun LLM menangani input baru karena phase 1 mengajarkannya.

### Parameter ekonomi

Rincian LLaVA-1.5-7B:
- CLIP ViT-L/14@336: 303M (dibekukan phase 1, sering dicairkan phase 2).
- Proyektor (2x linier): ~22 juta dapat dilatih.
- Llama-7B: 7B.
- Total: 7,3 miliar parameter. Dapat dilatih selama phase 2: proyektor penuh 7B + 22M.

Biaya training untuk phase 2: ~20 jam pada 8xA100. Ini adalah nomor kuncinya — satu hari, satu node, dapat direproduksi. Itu sebabnya LLaVA menyebar.

## Pakai

`code/main.py` mengimplementasikan:1. Proyektor MLP 2 lapis (redupkan 16 → 32 → 32 untuk skala mainan) dengan Python murni.
2. Jalur pembuatan prompt: prompt sistem + `<image>` diganti dengan N token yang diproyeksikan + giliran pengguna + placeholder pembuatan asisten.
3. Visualisator untuk tampilan blok visual 576 token dalam konteks LLM (persentase konteks 2k / 32k / 128k yang dikonsumsi).

## Kirim

Lesson ini menghasilkan `outputs/skill-llava-vibes-eval.md`. Dengan adanya pos pemeriksaan keluarga LLaVA, ia menjalankan rangkaian evaluasi getaran 10 prompt (3 teks, 3 VQA, 2 alasan, 2 penolakan) dan melaporkan kartu skor yang dapat dibaca manusia. Bukan patokan; tes asap untuk memastikan proyektor dan LLM terhubung dengan baik.

## Latihan

1. Hitung jumlah parameter yang dapat dilatih untuk proyektor MLP 2 lapis di `1024 → 4096 → 4096`. Dengan GELU dan bias, berapa fraksi LLaVA-13B yang diwakilinya?

2. Buat prompt LLaVA untuk kasus "penolakan" — gambar tersebut berisi individu pribadi. Tulis tanggapan asisten yang diharapkan. Mengapa LLaVA harus menolak zero-shot ini dan training data apa yang diperlukan untuk memperkuat penolakan tersebut?

3. Baca bagian AnyRes di blog LLaVA-NeXT. Hitung jumlah token visual untuk gambar 1344x672 di AnyRes. Bandingkan dengan token dasar 576 pada 336x336.

4. Proyektor LLaVA phase-1 dilatih dengan kehilangan LM pada teks. Apa yang terjadi jika kamu melewatkan phase 1 dan langsung menuju phase 2 (penyetelan instruksi visual)? Kutip ablasi VLM Prismatik (arXiv:2402.07865) untuk jawabannya.

5. LLaVA-Instruct-150k menggunakan GPT-4 dengan keterangan COCO untuk menghasilkan instruksi. Untuk domain baru (sinar X medis, citra satelit), jelaskan alur data empat langkah untuk menghasilkan instruksi domain. Apa yang salah pada setiap langkah?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Proyektor | "Jembatan MLP" | MLP 2 lapis dengan pemetaan GELU ViT redup ke LLM redup |
| Token gambar | "<gambar> pengganti" | Penanda cepat diganti dengan N token visual yang diproyeksikan sebelum inference |
| Penyetelan instruksi visual | "LLaVA phase 2" | Training kembar tiga yang dihasilkan GPT-4 (gambar, instruksi, respons) |
| Penyelarasan phase 1 | "Training awal proyektor" | Bekukan ViT dan LLM, latih proyektor dengan kehilangan LM pada teks |
| Apapun | "Ubin multi-tanaman" | Pisahkan gambar beresolusi tinggi ke dalam kotak ubin dan gabungkan token visual setiap ubin |
| LLaVA-Instruksikan | "GPT-4 dihasilkan" | 158 ribu pasangan instruksi-respons disintesis dari teks COCO + GPT-4 |
| Pembekuan encoder visi | "Tulang punggung terkunci" | Weight CLIP tidak diperbarui di phase 1, terkadang juga tidak di phase 2 |
| BagikanGPT4V | "Teks yang lebih baik" | 1 juta teks padat yang dihasilkan oleh GPT-4V, digunakan untuk penyelarasan kualitas lebih tinggi |
| VQA | "Menjawab pertanyaan visual" | Tugas menjawab pertanyaan bentuk bebas tentang gambar |
| VLM Prismatik | "Kertas ruang desain" | Ablasi Karamcheti 2024 secara sistematis menguji proyektor dan pilihan data |

## Bacaan Lanjutan- [Liu dkk. — Penyetelan Instruksi Visual (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — makalah LLaVA.
- [Liu dkk. — Peningkatan Baseline dengan Penyetelan Instruksi Visual (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5.
- [Chen dkk. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — dataset teks padat.
- [Karamcheti dkk. — VLM Prismatik (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — ablasi ruang desain.
- [Li dkk. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — gambar tunggal, multi-gambar, video terpadu.
