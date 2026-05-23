# Attention Jarang Asli (DeepSeek NSA)

> Pada 64 ribu token, attention memakan 70-80% latensi dekode. Setiap laboratorium model terbuka mempunyai rencana untuk memperbaikinya. NSA DeepSeek (kertas terbaik ACL 2025) adalah salah satu yang macet: tiga cabang attention paralel — token berbutir kasar yang dikompresi, token berbutir halus yang dipertahankan secara selektif, dan jendela geser untuk konteks lokal — digabungkan melalui gerbang yang dipelajari. Ini selaras dengan perangkat keras (ramah kernel), dapat dilatih secara asli (berfungsi dalam pra-training, tidak dipasang pada inference), dan pada dekode 64k, ini berjalan lebih cepat daripada FlashAttention sambil mencocokkan atau mengalahkan kualitas attention penuh. Lesson ini membangun tiga cabang secara end-to-end dan menunjukkan mengapa ketersebaran dapat dibedakan secara end-to-end.

**Type:** Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 7 · 12 (cache KV, attention flash), Fase 7 · 15 (varian attention), Fase 10 · 16 (attention berbeda)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Sebutkan tiga cabang attention NSA dan apa yang ditangkap masing-masing cabang tersebut.
- Jelaskan mengapa NSA "dapat dilatih secara alami" sedangkan metode sparse-attention sebelumnya hanya bersifat inferensial.
- Hitung penghematan komputasi attention NSA versus attention penuh pada konteks 64k sebagai fungsi ukuran blok kompresi dan pemilihan top-k.
- Menerapkan kombinasi tiga cabang di stdlib Python pada urutan sintetis pendek dan memverifikasi perilaku weight gating.

## Masalah

Attention penuh pada panjang urutan N membutuhkan `O(N^2)` waktu dan `O(N)` KV cache per layer. Pada 64 ribu token, jumlah bandwidth komputasi dan memori sangat besar. Perkiraan teoretis yang diukur dari makalah NSA: attention menyumbang 70-80% dari total latensi dekode pada 64k. Segala sesuatu di hilir — TTFT, token/detik, biaya per juta token — didominasi oleh biaya attention.

Sedikit attention adalah jawaban yang jelas. Upaya sebelumnya terbagi dalam dua kelompok. Ketersebaran pola tetap (jendela geser, langkah, blok-lokal) membuang informasi dan gagal dalam tugas penarikan kembali jangka panjang. Ketersebaran waktu inference (pemangkasan cache KV, H2O, StreamingLLM) diterapkan pada model yang telah dilatih sebelumnya pada attention padat dan hanya memulihkan sebagian kecil dari potensi percepatan karena model tidak pernah diminta untuk merutekan informasi melalui pola renggang.

Native Sparse Attention (Yuan et al., DeepSeek + PKU + UW, makalah terbaik ACL 2025, arXiv:2502.11089) melakukan keduanya: pola ketersebaran yang dipelajari model selama pra-training, diimplementasikan sebagai algoritme selaras kernel yang benar-benar memberikan penghematan komputasi pada inference. Dua tahun dari sekarang, NSA atau keturunan langsungnya menjadi attention utama pada setiap model konteks panjang perbatasan.

## Konsep

### Tiga cabang paralel

Untuk setiap kueri, NSA menjalankan attention tiga kali, terhadap tiga tampilan berbeda dari cache KV:

1. **Cabang terkompresi.** Token dikelompokkan ke dalam blok berukuran `l` (biasanya 32 atau 64). Setiap blok dikompresi menjadi satu token ringkasan melalui MLP kecil yang dipelajari. Kueri menangani token terkompresi ini, mendapatkan gambaran kasar dari keseluruhan urutan.

2. **Cabang yang dipilih.** Menggunakan skor attention dari cabang terkompresi, k blok teratas yang paling relevan dengan kueri saat ini diidentifikasi. Token terperinci (tidak terkompresi) dari blok tersebut dibaca dan kueri menangani semuanya. Pikirkan attention cabang terkompresi sebagai sinyal perutean untuk pemilihan.3. **Cabang jendela geser.** Kueri menangani token `W` terbaru (biasanya 512) untuk konteks lokal. Cabang ini menangkap pola jangka pendek yang banyak strukturnya (sintaks, inti referensi lokal) yang mungkin terlewatkan oleh dua cabang lainnya.

Tiga output cabang digabungkan melalui gerbang per posisi yang dipelajari:

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` adalah weight gerbang dari MLP kecil pada kueri. Jumlahnya tidak harus 1 — mereka dapat menimbang cabang secara mandiri.

### Mengapa ini "dapat dilatih secara asli"

Langkah pemilihan (blok k atas) bersifat diskrit. Operasi diskrit memutus aliran gradient. Pekerjaan dengan attention jarang sebelumnya melewatkan backprop melalui seleksi (membatasi training) atau menggunakan relaksasi terus menerus yang tidak memberikan ketersebaran nyata dalam inference.

NSA menghindari hal ini: attention cabang terkompresi ADALAH attention kasar yang dapat dibedakan pada keseluruhan rangkaian. Operasi top-k hanya menggunakan kembali skor attention tertinggi dari cabang terkompresi untuk memilih blok halus mana yang akan dimuat. Gradient mengalir melalui skor cabang terkompresi (yang memengaruhi output terkompresi DAN logika pemilihan), dan kontribusi blok yang dipilih terhadap output akhir juga dapat dibedakan. Operasi `top_k` yang tidak dapat dibedakan adalah larangan pada grafik komputasi maju — operasi ini hanya mengontrol blok mana yang dimuat dari memori.

Inilah sebabnya mengapa NSA dapat digunakan dalam pra-training secara menyeluruh. Model tersebut belajar mengarahkan informasi melalui tiga cabang secara bersamaan, menghasilkan pola renggang yang pada inference benar-benar memberikan percepatan yang dijanjikan.

### Kernel yang selaras dengan perangkat keras

Kernel NSA dirancang untuk hierarki memori GPU modern. Kernel memuat kueri berdasarkan grup GQA (loop luar), mengambil blok KV renggang yang sesuai per grup (loop dalam), dan menjalankan attention pada SRAM. Karena setiap grup kueri melihat blok terpilih yang sama (pemilihan dilakukan per grup kueri, bukan per kepala kueri), weight KV diamortisasi ke seluruh grup. Intensitas aritmatika tetap tinggi.

Makalah ini melaporkan kernel Triton berjalan 9x lebih cepat daripada FlashAttention pada 64k dekode, dengan rasio kecepatan bertambah seiring dengan panjang urutan. Kernel maju dan mundur keduanya disediakan.

### Anggaran komputasi

Misalkan `N` adalah panjang urutan, `l` ukuran blok kompresi, `k` jumlah pilihan k teratas, `w` jendela geser, `b` ukuran blok yang dipilih (biasanya sama dengan `l`).

- Cabang terkompresi: `O(N/l)` kunci per kueri, jadi `O(N * N / l)` total.
- Cabang yang dipilih: `O(k * b)` kunci per kueri, jadi `O(N * k * b)`.
- Cabang geser: `O(w)` kunci per kueri, jadi `O(N * w)`.

Jumlah: `O(N * (N/l + k*b + w))`.

Dengan `N = 64k, l = 64, k = 16, b = 64, w = 512`: biaya per kueri adalah `1000 + 1024 + 512 = 2536 keys`. Attention penuh adalah `64000 keys`. Pengurangan komputasi 25x.

Dengan `N = 128k, l = 64, k = 16, b = 64, w = 512`: biaya per kueri adalah `2000 + 1024 + 512 = 3536 keys`. Attention penuh adalah `128000 keys`. Pengurangan 36x. Manfaatnya bertambah seiring dengan panjangnya urutan, yang merupakan inti keseluruhannya.

### Bagaimana perbandingannya| Metode | Dapat dibedakan | Percepatan inference nyata | Penarikan kembali jangka panjang |
|--------|---------------|----------------------|-------------------|
| Hanya jendela geser | ya | ya | gagal |
| Melangkah / jarang blok | ya | ya | sebagian |
| Pemangkasan KV (H2O, StreamingLLM) | T/A (waktu inference) | ya | sebagian |
| MoBA (Tembakan Bulan) | sebagian | ya | bagus |
| NSA | ya (aslinya) | ya (9x pada 64k) | cocok dengan attention penuh |

MoBA (Moonshot, arXiv:2502.13189) diterbitkan secara bersamaan dan menggunakan pendekatan tiga-lebih-lebih-dari-satu yang serupa, dengan menerapkan prinsip MoE pada blok attention. NSA dan MoBA adalah dua arsitektur yang perlu diketahui untuk pra-training konteks panjang tahun 2026.

## Build

`code/main.py` mengimplementasikan tiga cabang pada urutan sintetik pendek dan menunjukkan:

- MLP kompresi (garis dasar mean-pool sederhana digunakan untuk kejelasan pedagogi; NSA sebenarnya menggunakan MLP yang dipelajari).
- Pemilihan blok k teratas didorong oleh skor cabang terkompresi.
- Attention jendela geser pada token `w` terakhir.
- Kombinasi yang terjaga keamanannya.
- Hasil cetakan hitungan komputasi dibandingkan dengan attention penuh.

### Langkah 1: kompres token menjadi blok

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### Langkah 2: attention cabang terkompresi

Jalankan attention softmax dari kueri terhadap kunci terkompresi. Skor cabang terkompresi berfungsi ganda sebagai sinyal untuk pemilihan k teratas.

### Langkah 3: pemilihan blok top-k

Pilih indeks blok terkompresi `k` dengan skor tertinggi. Muat token asli yang tidak terkompresi dari blok tersebut dan perhatikan token tersebut.

### Langkah 4: attention pada jendela geser

Ambil token `w` terakhir dan berikan attention standar terhadap token tersebut.

### Langkah 5: gerbang + kombinasi

MLP kecil pada kueri menghasilkan tiga weight gerbang. Output akhir adalah jumlah tertimbang dari tiga output cabang.

### Langkah 6: menghitung penghitungan

Cetak jumlah kunci yang dihadiri per kueri untuk setiap cabang dan totalnya. Bandingkan dengan `N` (attention penuh). Pada token sintetis 1024 dengan `l = 32, k = 4, w = 128`, NSA melihat kunci `32 + 128 + 128 = 288` per kueri dibandingkan 1024 untuk attention penuh — 3,5x lebih sedikit.

## Pakai

NSA mengirimkannya melalui jalur pra-training DeepSeek yang memiliki konteks panjang. Status integrasi dalam tumpukan inference publik pada April 2026:

- **DeepSeek internal**: weight asli yang dipublikasikan menggunakan NSA atau penerusnya DSA (Deepseek Sparse Attention).
- **vLLM**: dukungan eksperimental NSA dalam pengembangan untuk weight DeepSeek-V3.x.
- **SGLang**: Tolok ukur NSA diterbitkan; jalur produksi mengikuti vLLM.
- **llama.cpp / CPU**: tidak didukung; overhead decomposition kernel tidak sebanding dengan throughput CPU.

Kapan harus menghubungi NSA:

- Pra-training atau training lanjutan dijalankan dengan target konteks 64k-plus dengan anggaran komputasi yang serius.
- Inference pos pemeriksaan konteks panjang DeepSeek sendiri. Bobotnya asli NSA.

Kapan tidak:

- Melayani model terlatih dengan attention padat yang sudah ada. kamu tidak dapat melakukan retrofit NSA tanpa training lanjutan.
- Konteks di bawah 16k. Biaya overhead tiga cabang mendominasi penghematan.
- Obrolan interaktif Batch-1. Manfaat dekode yang sensitif terhadap latensi, tetapi hanya pada konteks yang panjang.

## Kirim

Lesson ini menghasilkan `outputs/skill-nsa-integrator.md`. Mengingat spesifikasi proses pra-training konteks panjang, ini menghasilkan rencana integrasi NSA: ukuran blok kompresi, top-k, jendela geser, lebar gerbang MLP, pilihan kernel, dan evaluasi konteks panjang spesifik yang akan membenarkan perubahan arsitektur.

## Latihan1. Jalankan `code/main.py` pada token sintetis 1024. Sapu `(l, k, w)` di tiga preset dan jumlah komputasi cetak. Identifikasi preset yang mencapai jumlah kunci terendah per kueri sekaligus mempertahankan 95% ingatan dibandingkan attention penuh pada pengujian yang rumit.

2. Ganti kompresor mean-pool dengan MLP kecil yang dipelajari (2 lapis, tersembunyi 32). Latihlah pada tugas sintetis yang sinyalnya adalah rata-rata sebuah blok. Ukur kesenjangan perplexity terhadap garis dasar kumpulan rata-rata pada data yang disimpan.

3. Menerapkan gerbang MLP. Dibutuhkan query sebagai input dan output tiga scalar. Tunjukkan bahwa gerbang berperilaku wajar: weighting yang hampir seragam pada kueri acak, weight yang berat pada cabang yang dipilih ketika kueri mencapai blok jauh di belakang.

4. Hitung anggaran memori cache KV untuk model 70B berkemampuan NSA pada konteks 128k. Head KV ada 8, head redup 128, BF16. Bandingkan dengan attention penuh dan MLA (Fase 10 · 14 menunjukkan angka MLA). Identifikasi panjang urutan di mana cache KV cabang NSA yang terperinci sama dengan attention penuh.

5. Baca Bagian 4 dari makalah NSA (arXiv:2502.11089) dan jelaskan dalam tiga kalimat mengapa skor attention cabang terkompresi digunakan kembali untuk pemilihan top-k daripada menghitung skor routing terpisah. Ikat jawabannya dengan aliran gradient.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Cabang terkompresi | "Tampilan kasar" | Attention terhadap kunci rata-rata blok yang memberikan konteks global dalam kunci O(N/l) per kueri |
| Cabang yang dipilih | "Blok k teratas" | Attention mendalam pada blok `k` dengan skor cabang terkompresi tertinggi |
| Jendela geser | "Konteks lokal" | Attention pada token `W` terakhir untuk pola distance pendek |
| Kemampuan melatih asli | "Pra-latihan dengan ketersebaran" | Pola ketersebaran dipelajari selama pra-training, bukan langsung pada inference |
| Blok kompresi ukuran l | "Ukuran grup untuk tampilan kasar" | Berapa banyak token yang digabungkan menjadi satu ringkasan; 32-64 tipikal |
| Top-k | "Blok untuk disimpan" | Jumlah blok terkompresi yang tokennya tidak terkompresi dapat dibaca; 16 tipikal |
| Jendela geser W | "Radius attention lokal" | Biasanya 512; lebih pendek merusak koherensi lokal, lebih lama membuang-buang komputasi |
| Gerbang Cabang | "Cara mencampur ketiganya" | Output MLP per posisi yang memberi weight pada kontribusi ketiga cabang |
| Penyelarasan perangkat keras | "Ketersebaran ramah kernel" | Pola renggang dipilih sehingga kernel GPU sebenarnya mencapai kecepatan teoretis |
| DSA | "Penerus NSA" | Deepseek Sparse Attention, arsitektur yang mengikuti NSA dalam garis keturunan DeepSeek |

## Bacaan Lanjutan- [Yuan dkk. — Native Sparse Attention: Sparse Attention yang Sesuai dengan Perangkat Keras dan Dapat Dilatih Secara Native (arXiv:2502.11089, Makalah Terbaik ACL 2025)](https://arxiv.org/abs/2502.11089) — makalah
- [Laporan Teknis DeepSeek-V3 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — keluarga arsitektur yang ditargetkan NSA
- [Moonshot AI — MoBA: Campuran Attention Blok untuk LLM Konteks Panjang (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — pekerjaan bersamaan, attention gaya MoE pada blok
- [Beltagy dkk. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — asal-usul jendela geser
- [Xiao dkk. — StreamingLLM: Model Bahasa Streaming yang Efisien dengan Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — garis dasar ketersebaran waktu inference yang ditingkatkan NSA
- [Dao dkk. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — kernel NSA dasar dengan attention penuh mencapai 64k
