# Model Omni: Qwen2.5-Omni dan Perpecahan Pemikir-Pembicara

> Demo produk GPT-4o pada bulan Mei 2024 mengganggu bukan karena model dasarnya tetapi karena bentuk produknya — antarmuka suara tempat kamu berbicara, model melihat apa yang dilihat kamera, dan model berbicara kembali dalam waktu kurang dari 250 mdtk. Ekosistem terbuka menghabiskan sisa tahun 2024 dan 2025 berlomba untuk mencapai permukaan produk tersebut. Qwen2.5-Omni (Maret 2025) adalah referensi desain terbuka: Pemikir (Transformer penghasil teks besar) ditambah Pembicara (Transformer penghasil ucapan paralel), dihubungkan dengan token ucapan streaming. Mini-Omni menyederhanakannya, Moshi mencocokkan latensinya, GLM-4-Voice memperluasnya ke bahasa Mandarin. Lesson ini membahas arsitektur Thinker-Talker dan anggaran latensi yang membuat streaming dialog real-time berfungsi.

**Type:** Build
**Language:** Python (stdlib, simulator latensi pipeline streaming + loop VAD)
**Prerequisites:** Fase 12 · 19 (audio-LLM), Fase 12 · 16 (apa saja)
**Waktu:** ~180 menit

## Tujuan Pembelajaran

- Bagi pipeline inference menjadi Thinker (penalaran teks) dan Talker (sintesis ucapan) dan jelaskan mengapa streaming paralel berfungsi.
- Hitung anggaran time-to-first-audio-byte (TTFAB) untuk interaksi percakapan, komponen demi komponen.
- Jelaskan pengkodean posisi selaras waktu TMRoPE di seluruh visi, audio, dan teks dalam Thinker.
- Sebutkan tiga pola percakapan real-time: half-duplex, turn-taking, full-duplex.

## Masalah

Asisten suara real-time harus melakukan banyak hal dengan cepat:

1. Dengarkan pengguna. Tokenization ucapan secara real-time, deteksi aktivitas suara (VAD) untuk mengetahui kapan mereka selesai berbicara.
2. Secara opsional lihat. Input kamera pada 2-4 FPS, dialirkan ke Thinker bersama audio.
3. Pikirkan. Tulis tanggapan yang dikondisikan pada riwayat percakapan.
4. Bicaralah. Sintesis token audio, dekode menjadi bentuk gelombang, streaming ke speaker pengguna.

Setiap langkah menambah latensi. Nuansa percakapan memerlukan total perjalanan pulang pergi <500 md — jika di bawah itu, pengguna tidak lagi menyadari adanya jeda. Klaim GPT-4o ~250 md. Moshi ~160ms. Qwen2.5-Omni ~350-500ms.

Setiap komponen perlu dialirkan. Tidak ada yang bisa "menggabungkan semuanya lalu memecahkan code".

## Konsep

### Pemikir dan Pembicara

Decomposition Qwen2.5-Omni:

- Pemikir: Transformer penghasil teks 7B-80B. Menggunakan teks + gambar + token audio yang disisipkan. Menghasilkan token teks yang mewakili apa yang harus dikatakan.
- Talker: trafo penghasil suara yang lebih kecil (200M-1B). Menggunakan token output teks Thinker ditambah token konteks ucapan terkini. Menghasilkan token ucapan terpisah (indeks sisa-VQ).
- Dekoder ucapan: dekoder bentuk gelombang streaming (SNAC, keluarga MoVQGAN) yang mengambil token ucapan ke sample audio secara real-time.

Perpisahan itu penting. Pemikir harus besar untuk mendapatkan alasan yang baik. Pembicara bisa berukuran kecil karena tugasnya bersifat lokal — mengonversi teks menjadi token ucapan. Pembicara yang Lebih Besar tidak lebih ekspresif; itu lebih lambat.

Menjalankan keduanya secara paralel:

1. Pemikir mengeluarkan token teks t_i.
2. Pembicara menggunakan t_i (melalui streaming) dan mengeluarkan token ucapan s_i, s_{i+1}, ..., s_{i+k}.
3. Dekoder ucapan menggunakan token ucapan yang datang dan mengeluarkan sample audio.
4. Saat Thinker berada pada token teks t_{i+3}, Talker sudah melakukan streaming audio untuk t_0..t_{i+2}.

### TMRoPE — posisi multimoda yang selaras dengan waktu

Pemikir perlu mengintegrasikan bingkai gambar (yang mencapai, katakanlah, 4 FPS), bingkai audio (yang mencapai 50 frame/detik), dan teks dari riwayat percakapan. Urutan urutan yang naif (semua gambar, lalu semua audio, lalu teks) kehilangan keselarasan temporal.TMRoPE assigns absolute timestamps to every token. Token penglihatan pada t=2,3 detik. Token audio pada t=2,32 detik. Token teks dari pengguna "berhenti" pada t=2,35 detik. RoPE memutar attention berdasarkan stempel waktu; model melihatnya sebagai hal yang bersamaan untuk sementara.

Ini adalah infrastruktur agar "dia melambai sambil menyapa" agar berfungsi — model melihat bingkai video dan audio pada momen konseptual yang sama.

### Streaming sintesis ucapan

Token ucapan harus dialirkan. Mini-Omni (Xie & Wu, 2024) memperkenalkan "model bahasa dapat mendengar, berbicara sambil berpikir dalam streaming": Token output Thinker dan token output Talker disisipkan dalam urutan yang sama. Talker fires as soon as Thinker commits the next text token. Tidak ada batasan kelompok.

Moshi (Défossez et al., October 2024) is the fastest open implementation. TTFAB 160ms pada satu A100. Arsitektur: Transformer 7B tunggal yang memancarkan token teks dan ucapan pada posisi bergantian, dengan "monolog batin" yang memisahkan aliran berpikir dari aliran berbicara. This is effectively Thinker + Talker fused into one model with careful training.

### VAD dan pengambilan giliran

Deteksi aktivitas suara berjalan di sisi input. Dua pola:

- Half-duplex: pengguna berbicara, model mendengarkan. Model berbicara, pengguna mendengarkan. Clear handoff via VAD silence detection (~200ms).
- Full-duplex: keduanya dapat berbicara secara bersamaan. Model dapat melakukan backchannel ("uh-huh") atau menyela. Jauh lebih sulit. Moshi mendukung hal ini.

Qwen2.5-Omni supports half-duplex by default, with turn-taking via silence threshold. Dupleks penuh memerlukan penanganan layer aplikasi.

### Qwen3-Omni (November 2025)

Penerusnya. Qwen3-80B Thinker, Talker lebih besar, TMRoPE-v2 yang ditingkatkan. Latensi mendekati 250 ms GPT-4o. Buka weight. Benchmarks on OmniBench competitive with Gemini 2.0 Live.

### Anggaran latensi produksi

Untuk interaksi streaming pada umumnya:

- Mikrofon -> token audio: 40-80ms.
- Prefill (prompt + history): 100-200ms pada 7B, lebih banyak lagi pada 70B.
- Token teks Pemikir Pertama: 40 ms.
- Pembicara memproses token teks pertama: 20 md.
- Token ucapan pertama yang dilakukan: 40 md.
- Dekode sisa-VQ: 30ms.
- Dekode bentuk gelombang ucapan: 50-80ms.

Total TTFAB: 320-510 md pada 7B, 600-900 md pada 70B. Kualitas frontier biasanya berarti 70B+; karenanya kesenjangan latensi perbatasan.

### Matematika tingkat token

At 16kHz speech with 50 Hz base speech tokens, you need 50 speech tokens per second of output. Pembicara harus mengeluarkan ≥50 tok/s untuk mengikutinya. At a typical LLM throughput of 30-80 tok/s on an H100, a small (200-300M) Talker is fast enough; seorang Pembicara 7B akan tertinggal.

This is why small dedicated Talker models exist rather than "just use the main model."

## Pakai

`code/main.py`:

- Simulates a Thinker-Talker pipeline with mock token-emission rates.
- Menghitung TTFAB untuk ukuran model dan laju sample mikrofon yang dapat dikonfigurasi.
- Demonstrates half-duplex turn-taking with VAD silence threshold.

## Kirim

Lesson ini menghasilkan `outputs/skill-omni-streaming-budget.md`. Mengingat target TTFAB dan rangkaian feature produk suara real-time (vision-in, bilingual, full-duplex), pilih Qwen2.5-Omni, Qwen3-Omni, Moshi, atau Mini-Omni dan ukur Thinker/Talker.

## Latihan

1. TTFAB target kamu adalah 300 md. On a 7B Thinker and 300M Talker, write out every component's latency.

2. Qwen2.5-Omni menggunakan TMRoPE. Jelaskan apa yang dilihat model saat pengguna mulai berbicara pada t=1s dan kamera menangkap isyarat pada t=1,2s.

3. Full-duplex support requires the model to emit audio while listening. Propose a training data format that teaches this.4. Baca makalah Moshi Bagian 4. Jelaskan pemisahan "monolog batin" dan mengapa hal itu menghindari perpecahan Pemikir-Pembicara.

5. Hitung anggaran throughput: seberapa cepat Talker harus mengeluarkan token untuk mengimbangi ucapan 16kHz pada 50 token layer dasar/detik?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Pemikir | "Penalaran otak" | Transformer penghasil teks besar yang menghasilkan apa yang harus dikatakan |
| Pembicara | "Mulut yang menghasilkan ucapan" | Transformer kecil yang menghasilkan token ucapan diskrit dari teks Thinker |
| TTFAB | "Anggaran latensi" | Waktu-ke-byte audio pertama: dari akhir ucapan pengguna hingga sample audio pertama keluar |
| TMRoPE | "Tali Selaras Waktu" | Pengkodean posisi menggunakan stempel waktu absolut di seluruh penglihatan, audio, teks |
| Setengah dupleks | "Pengambilan giliran" | Pengguna dan model bergantian; Keheningan VAD mendeteksi tindakan pengguna |
| Dupleks penuh | "Serentak" | Model dapat berbicara dan mendengarkan pada saat yang bersamaan; mampu pipeline belakang |
| Monolog batin | "Pemisahan Moshi" | Desain model tunggal di mana aliran berpikir dan aliran berbicara saling bersilangan |

## Bacaan Lanjutan

- [Xu dkk. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Tim Qwen — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez dkk. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng dkk. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
