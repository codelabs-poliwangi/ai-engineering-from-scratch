# MIO dan Model Multimodal Streaming Apa Pun

> GPT-4o mengirimkan produk yang tidak dapat ditiru oleh sebagian besar model terbuka: agen yang mendengar suara, melihat video, dan membalas secara real-time. Jawaban ekosistem terbuka pada akhir tahun 2024 adalah MIO (Wang et al., September 2024). MIO memberi token pada teks, gambar, ucapan, dan musik, melatih satu Transformer sebab akibat pada urutan yang disisipkan, dan menghasilkan modalitas apa pun ke modalitas apa pun. AnyGPT (Zhan et al., Februari 2024) adalah bukti konsep; MIO adalah peningkatan; Unified-IO 2 (Allen AI, Desember 2023) adalah sepupu dengan landasan visi + tindakan. Lesson ini membahas pola apa saja - empat tokenizer, satu Transformer, dekode ramah streaming.

**Type:** Learn
**Language:** Python (stdlib, pengalokasi token empat modalitas + loop dekode streaming)
**Prerequisites:** Fase 12 · 11 (Bunglon), Fase 6 (Ucapan dan Audio)
**Waktu:** ~120 menit

## Tujuan Pembelajaran

- Rancang kosakata bersama yang menampung teks, gambar, ucapan, dan token musik tanpa benturan.
- Bandingkan SEED-Tokenizer (gambar) dan SpeechTokenizer sisa-VQ (ucapan) pada trade-off kompresi + rekonstruksi.
- Jelaskan kurikulum empat phase yang membangun generasi apa pun.
- Sebutkan tiga resep terbuka apa saja dan trade-off utamanya: MIO, AnyGPT, Unified-IO 2.

## Masalah

Model multimoda terpadu mudah untuk diklaim dan sulit dibangun dalam skala besar. Sebagian besar sistem "any-to-any" hingga tahun 2024 sudah disalurkan: model visi → representasi teks → model ucapan → audio. Setiap lompatan kehilangan informasi, menambah latensi, dan mempersulit training. Video demo GPT-4o menunjukkan alternatif model tunggal dengan respons subdetik; sistem terbuka tertinggal beberapa bulan.

Tantangan rekayasa:

- Tokenizer harus ada untuk setiap modalitas, mengompres secara lossless-cukup untuk rekonstruksi, dan menghasilkan token dengan kecepatan yang dapat dikonsumsi oleh Transformer.
- Kosakata tunggal harus mengalokasikan ruang untuk teks (32k+), gambar (16k+), ucapan (4k+), musik (8k+). Minimal empat puluh ribu lebih entri.
- Training data harus mencakup setiap pasangan input-output (teks→gambar, gambar→ucapan, ucapan→gambar, dll.) atau model harus dibuat.
- Inference harus mengalirkan token output dengan cukup cepat untuk latensi percakapan (<500 ms waktu ke byte audio pertama).

## Konsep

### Empat tokenizer untuk empat modalitas

Tumpukan tokenizer MIO:

- Teks: BPE standar, vocab ~32000.
- Gambar: SEED-Tokenizer (2023) — VAE terkuantisasi dengan buku code terpisah, 4096 entri, 32x32 token per gambar.
- Speech: SpeechTokenizer sisa-VQ (2023) — mengkodekan bentuk gelombang 16kHz ke dalam 8 buku code hierarki; tingkat pertama adalah konten kasar, tingkat selanjutnya menambahkan prosodi dan identitas pembicara.
- Musik: sisa-VQ serupa (keluarga MusicGen / Encodec Meta), 4-8 buku code.

Setiap modalitas menghasilkan token bilangan bulat. Token mendapatkan rentang ID yang terpisah dalam kosakata bersama:

```
text:   0..31999
image:  32000..36095  (4096 image tokens)
speech: 36096..40191  (4096 speech base tokens, plus residual layers)
music:  40192..48383  (8192 music tokens)
sep:    48384..48390  (<image>, <speech>, <music>, </...>, etc.)
```

Total: ~48k kosakata. Embedding input dan proyeksi output mencakup semuanya.

### Dekode streaming

Pembuatan ucapan menggunakan sisa-VQ. Transformer memprediksi token ucapan dasar (layer 0); pengukur sisa yang didekodekan secara paralel memprediksi layer berikutnya. Setiap token layer 0 kira-kira memiliki audio 50ms pada 16kHz.

Pola streamingnya:1. Pengguna berbicara melalui mikrofon; tokenizer audio real-time memancarkan token ucapan setiap 50 mdtk.
2. MIO menggunakan token saat tiba (pengisian cepat + penerusan bertahap).
3. Token output mengalir keluar saat dihasilkan; dekoder ucapan paralel mengubahnya menjadi sample audio dengan latensi ~50-150 ms.
4. Waktu hingga byte audio pertama: ~300-500 md pada kertas MIO, mendekati ~250 md GPT-4o.

Mini-Omni (arXiv:2408.16725), GLM-4-Voice (arXiv:2412.02612), dan Moshi (arXiv:2410.00037) merupakan desain LLM ucapan streaming yang saling melengkapi. Moshi khususnya mencapai 160ms bolak-balik pada satu GPU.

### Kurikulum empat phase

Kurikulum training MIO:

1. Phase 1 — penyelarasan. Korpora pasangan modalitas skala besar: teks-gambar, teks-ucapan, teks-musik. Setiap pasangan menggunakan segmen kosakata tokennya sendiri. Melatih kosakata bersama.
2. Phase 2 — disisipkan. Dokumen interleaved multi-modalitas (blog dengan gambar + video, podcast dengan transkrip, dll.). Melatih konteks lintas modalitas.
3. Phase 3 — peningkatan kemampuan bicara. Data audio ekstra untuk meningkatkan kualitas ucapan tanpa kehilangan kemampuan teks.
4. Phase 4 — SFT. Penyetelan instruksi lintas modalitas: VQA, teks, narasi, dialog ucapan-ke-ucapan.

Hilangnya satu phase akan menurunkan kemampuan tertentu: lewati phase 2 dan model kehilangan konteks lintas modalitas; lewati phase 3 dan kemampuan bicaranya buruk.

### Rantai pemikiran visual

MIO memperkenalkan rantai pemikiran visual: model memancarkan token gambar perantara sebagai langkah penalaran. Untuk "apakah kucing itu memanjat pohon?" modelnya:

1. Memancarkan token `<image>` yang merender adegan (dari gambar input atau sketsa).
2. Memancarkan teks menganalisis sketsa.
3. Memancarkan jawaban akhir.

Gambar perantara yang dirender berfungsi sebagai papan gores. Tolok ukur meningkatkan tugas-tugas penalaran spasial. Idenya mencerminkan rantai pemikiran untuk penalaran teks.

### Pesaing dalam bidang apa pun

- AnyGPT (arXiv:2402.12226): 4 modalitas (teks, gambar, ucapan, musik), desain serupa.
- Unified-IO 2 (arXiv:2312.17172): menambahkan output tindakan penglihatan, kedalaman, normal. Lebih banyak keragaman tugas, skala lebih kecil.
- NExT-GPT (arXiv:2309.05519): LLM + dekoder difusi khusus modalitas. Bukan pendekatan model tunggal.
- CoDi (arXiv:2305.11846): difusi yang dapat disusun; apa saja-ke-apa pun melalui laten bersama.

MIO adalah yang paling dekat dengan token murni apa pun. AnyGPT adalah nenek moyang konseptualnya.

### Anggaran latensi

Untuk produk percakapan, latensi setiap komponen penting:

- Mikrofon ke token audio: ~50ms.
- Pra-pengisian (token audio + riwayat): ~100 md pada model 8B.
- Token output pertama: ~50ms.
- Dekoder suara VQ + sisa paralel: ~100-150ms.

Total waktu hingga byte audio pertama: minimum ~300 md. Klaim GPT-4o ~250 md. Moshi mengklaim 160ms. MIO/AnyGPT berada dalam kisaran 400-600ms per benchmark publik.

### Mengapa apa pun tetap sulit

Bahkan pada tahun 2026, model terbuka apa pun mengikuti model tertutup pada dua sumbu:

- Kualitas ucapan. Tokenizer sisa-VQ bersifat lossy; ucapan percakapan terdengar seperti robot dibandingkan dengan suara kelas ElevenLabs.
- Penalaran lintas modalitas. Meminta model "bernyanyi tentang apa yang kamu lihat" masih lebih sering gagal dibandingkan tugas penglihatan murni.

Ini adalah masalah penelitian terbuka. Qwen3-Omni (Lesson 12.20) adalah upaya terbuka paling maju pada tahun 2025.

## Pakai

`code/main.py`:- Mendefinisikan alokasi kosakata empat modalitas dan mencetaknya.
- Merutekan daftar input multimodal (teks, gambar, klip audio, musik) melalui router tokenizer.
- Mensimulasikan dekode streaming untuk respons text-to-speech dengan penghitungan latensi.
- Menghitung waktu yang diharapkan untuk byte audio pertama yang diberikan latensi encoder, prefill, dan decoder.

## Kirim

Lesson ini menghasilkan `outputs/skill-any-to-any-pipeline-auditor.md`. Dengan adanya spesifikasi produk percakapan (modalitas masuk, modalitas keluar, target latensi), ia mengaudit pilihan desain rangkaian MIO dan menghitung anggaran latensi.

## Latihan

1. Produk kamu menerima input ucapan dan mengembalikan output ucapan. Berapa target anggaran latensi end-to-end? Buat daftar komponen yang menghabiskan waktu.

2. Residual SpeechTokenizer-VQ menggunakan 8 buku code. Usulkan mengapa pengodean paralel tingkat residu diperlukan (vs berurutan) dan penghematan latensi apa yang dihasilkannya.

3. Kosakata kamu memiliki 32k teks + 4k gambar + 4k ucapan. Tambahkan musik 8k dan ~10 pemisah. Berapa biaya parameter matrix embedding pada redup tersembunyi 4096?

4. Rantai pemikiran visual memancarkan gambaran perantara. Pertanyaan seperti apa yang bermanfaat? Jenis apa yang dirugikan oleh token tambahan?

5. Baca Moshi (arXiv:2410.00037). Jelaskan teknik "monolog batin" dan bandingkan dengan rangkaian pemikiran visual MIO.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Apa saja-ke-apa saja | "Masuk/keluar multimoda" | Model tunggal yang menerima dan memancarkan teks, gambar, ucapan, dan musik ke segala arah |
| Residu-VQ | "Tumpukan tokenizer ucapan" | Tokenization multi-buku code di mana setiap layer menambahkan informasi; layer dasar adalah konten, layer selanjutnya adalah prosodi |
| SEED-Tokenizer | "Code gambar" | Tokenizer gambar diskrit dengan buku code entri 4096 yang digunakan oleh MIO |
| Rantai pemikiran visual | "Paduan gores visual" | Model menghasilkan gambaran perantara sebagai langkah penalaran sebelum jawaban akhirnya |
| Waktu-ke-byte-audio pertama | "TTFAB" | Latensi dari suara pengguna ke output audio pertama; <500ms untuk nuansa percakapan |
| Kurikulum empat phase | "Resep training" | Penyelarasan -> disisipkan -> peningkatan ucapan -> SFT, dalam urutan itu |

## Bacaan Lanjutan

- [Wang dkk. — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)
- [Zhan dkk. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Lu dkk. — Terpadu-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)
- [Wu dkk. — BERIKUTNYA-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)
- [Tang dkk. — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)
