# Prediksi Multi-Token (MTP)

> Setiap LLM autoregresif dari GPT-2 hingga Llama 3 berlatih dengan satu loss per posisi: prediksi token berikutnya. DeepSeek-V3 menambahkan loss kedua per posisi: prediksi token setelah itu. Parameter tambahan 14B (pada model 671B) disaring kembali ke model utama melalui aliran gradient, dan kepala MTP yang terlatih digunakan kembali pada inference sebagai perancang penguraian code spekulatif dengan penerimaan 80%+. Throughput generasi 1,8× hadir secara gratis. Lesson ini membangun modul MTP berurutan dari laporan teknis DeepSeek, menghitung loss dan tata letak parameter shared-head, dan menjelaskan mengapa MTP mempertahankan rantai sebab akibat sementara MTP paralel asli Gloeckle dkk memutusnya.

**Type:** Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 10 · 04 (pra-training mini GPT), Fase 10 · 15 (penguraian code spekulatif)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Nyatakan tujuan training MTP dan dapatkan loss gabungan di seluruh kedalaman prediksi.
- Jelaskan perbedaan antara kepala MTP paralel Gloeckle et al. (2024) dan modul MTP sekuensial DeepSeek-V3 dan mengapa desain sekuensial mempertahankan rantai sebab akibat.
- Hitung parameter dan overhead memori dari penambahan modul MTP ke proses pra-training.
- Mengimplementasikan satu modul MTP dari awal: embedding bersama, blok Transformer per kedalaman, proyeksi, dan kepala output bersama.

## Masalah

Prediksi token berikutnya adalah tujuan training LLM standar. Setiap keadaan tersembunyi diawasi untuk memprediksi satu hal: token berikutnya. Ini adalah sinyal yang sangat lemah. Sebagian besar informasi dalam suatu rangkaian melampaui satu token — struktur, koherensi, faktualitas, aliran aritmatika. Model harus mempelajarinya dengan mengumpulkan banyak sinyal satu token dalam triliunan token.

MTP bertanya: bagaimana jika setiap negara bagian tersembunyi diawasi untuk memprediksi beberapa token masa depan sekaligus? Gloeckle dkk. (Meta, 2024) menunjukkan bahwa hal ini membantu. Implementasinya menempatkan beberapa output head independen di atas backbone, masing-masing memprediksi offset yang berbeda. Paralel, sederhana, tetapi kepala-kepala tersebut melihat keadaan tersembunyi yang sama tanpa penyempurnaan hierarki apa pun — dan prediksi-prediksi tersebut tidak berantai secara kausal, sehingga tidak dapat digunakan untuk penguraian code spekulatif.

DeepSeek-V3 (Desember 2024) mendesain ulang MTP sebagai modul berurutan yang menjaga rantai sebab akibat di setiap kedalaman prediksi. Model memprediksi `t+1` dari `h_i^(0)`, lalu memprediksi `t+2` dari status tersembunyi baru `h_i^(1)` yang menggabungkan `h_i^(0)` dengan embedding `E(t+1)`, dan seterusnya. Setiap kedalaman adalah blok trafo kecilnya sendiri. Embedding bersama dan kepala output bersama menjaga overhead parameter tetap sederhana. Pada skala DeepSeek-V3, 14B parameter tambahan di seluruh modul MTP di atas 671B weight model utama. Overhead 2% itu membeli sinyal training yang lebih padat DAN draf penguraian code spekulatif yang siap pakai pada inference.

Lesson ini membangun modul MTP tunggal dan kehilangan kedalaman D dari awal. Perhitungannya rapi. Implementasinya adalah 150 baris.

## Konsep

### Resep MTP berurutan

DeepSeek-V3 menambahkan modul `D` MTP di atas model utama. Setiap modul `k` (untuk `k = 1..D`) memprediksi token pada kedalaman `k` — yaitu, `t_{i+k}` diberi awalan melalui posisi `i`.

Modul `k` terdiri dari:- Blok trafo `T_k` dengan attention-nya sendiri dan MLP.
- Matrix proyeksi `M_k` yang menggabungkan keadaan kedalaman tersembunyi sebelumnya dengan embedding token kebenaran dasar kedalaman berikutnya.
- Embedding bersama `E` (sama seperti model utama).
- Kepala output bersama `Out` (sama seperti model utama).

Saat training, untuk awalan melalui posisi `i`, status tersembunyi per kedalaman adalah:

```
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

Prediksi per kedalaman adalah:

```
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

Loss per kedalaman adalah entropi silang terhadap kebenaran dasar `t_{i+k}`:

```
L_k = CE(logits_{i+k}, t_{i+k})
```

Hilangnya sendi di seluruh kedalaman:

```
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda` adalah faktor weight yang kecil — DeepSeek-V3 menggunakan 0,3 untuk 10% training pertama dan 0,1 setelahnya. Total loss training adalah `L_main + L_MTP`.

### Mengapa berurutan, bukan paralel

MTP paralel asli Gloeckle memiliki kepala output D, masing-masing langsung diterapkan ke `h_i^(0)`. Setiap kepala memprediksi `t_{i+k}` dari keadaan tersembunyi tulang punggung yang sama. Itu berlatih dengan baik, tetapi prediksinya tidak dikondisikan satu sama lain. kamu tidak dapat menggunakan output `head_1` untuk membantu `head_2` — head menyala secara paralel.

Desain sekuensial DeepSeek-V3 dibuat `h_i^(k)` dari `h_i^(k-1)` ditambah embedding token berikutnya `E(t_{i+k})`. Hal ini menjaga rantai sebab akibat: untuk memprediksi `t_{i+k+1}`, modul di kedalaman `k+1` melihat apa yang ada di `t_{i+k}`. Hal ini secara struktural identik dengan bagaimana dekoder autoregresif menggunakan keluarannya sendiri — membuat modul MTP dapat langsung digunakan sebagai perancang penguraian code spekulatif.

Pada inference: masukkan `h_i^(k-1)` dan draf `t_{i+k}` ke dalam modul `k+1`, dapatkan prediksi untuk `t_{i+k+1}`. Mengulang. Itu adalah draf gaya EAGLE, menggunakan modul MTP terlatih sebagai draf jaringan. DeepSeek-V3 melaporkan 80%+ penerimaan pada modul MTP pertama dan kecepatan ~1,8×.

### Akuntansi parameter

Untuk model dengan `h` dan kosakata `V` yang tersembunyi:

- Model utama: miliaran parameter, ditambah satu kepala output berukuran `V * h`.
- Kepala output bersama: gunakan kembali kepala model utama. Tidak ada parameter tambahan.
- Embedding bersama: menggunakan kembali embedding model utama. Tidak ada parameter tambahan.
- Modul per-MTP:
  - Proyeksi `M_k`: `(2h) * h = 2h^2`.
  - Blok Transformer `T_k`: attention (`4h^2` untuk MHA) plus MLP (biasanya `8h^2` untuk SwiGLU dengan rasio 8/3). Tentang `12h^2` per blok.

Total tambahan per modul: `~14h^2`. Untuk `h = 7168` DeepSeek-V3, D = 1 modul: parameter `~14 * 7168^2 = ~720M` di atas kertas. DeepSeek-V3 melaporkan 14B — perbedaannya adalah sebagian besar layer ahli juga menjadi MoE dalam modul MTP.

### Hasil penguraian code spekulatif

Selama pra-training, modul MTP memperlambat training sekitar 10% (lebih banyak komputasi maju, loss ekstra). Imbalannya ada dua:

1. Sinyal latihan yang lebih padat. Setiap negara bagian yang tersembunyi melihat target pengawasan H+1. Efek terukur pada MMLU, GSM8K, MATH, HumanEval: peningkatan beberapa poin persentase yang konsisten dalam ablasi DeepSeek-V3.2. Draf decoding spekulatif gratis pada inference. Modul MTP sudah dilatih untuk memprediksi beberapa token berikutnya. Digunakan kembali sebagai jaringan rancangan, jaringan ini memberikan tingkat penerimaan 80%+. Pada level tersebut, decoding spesifikasi N=3 atau N=5 memberikan throughput 1,8×. Biaya waktu training sebesar 10% terbayar saat pertama kali kamu menjalankan inference.

### Kaitannya dengan EAGLE

EAGLE melatih model draf kecil SECARA TERPISAH setelah pra-training. MTP memasukkan draf tersebut ke dalam pra-training. Kedua pendekatan ini bertemu pada tingkat penerimaan yang serupa tetapi melalui jalur yang berbeda:

| Dimension | EAGLE-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| Saat dilatih | Pasca-pra-training | Selama pra-training |
| Kompatibel dengan weight yang ada | Ya | Tidak (perlu training ulang) |
| Param draf | 1-2 layer trafo | 1 blok trafo + proyeksi |
| Tingkat penerimaan | 0,88-0,92 | 0,80+ pada kedalaman 1 |
| Manfaat lebih dari sekedar percepatan | Hanya decoding spekulatif | Sinyal latihan lebih padat + percepatan |

## Build

`code/main.py` membangun satu modul MTP ujung ke ujung: embedding bersama, proyeksi, blok Transformer, kepala output bersama. Ia kemudian menghitung loss entropi silang per kedalaman pada urutan sintetik pendek dan mencetak jumlah parameter berdasarkan komponen. Kosakata mainan yang terdiri dari 32 token membuat angka-angkanya tetap terbaca.

### Langkah 1: tabel embedding bersama

Satu tabel `vocab_size x hidden` digunakan oleh model utama DAN oleh setiap modul MTP di setiap kedalaman. Bukan salinan kedua — secara harfiah merupakan tensor yang sama.

### Langkah 2: kombinasi per kedalaman

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

Real DeepSeek-V3 menggabungkan dua vector RMSNormed menjadi `[2h]` dan memproyeksikan dengan matrix `h x 2h`. Mainan ini menggunakan penambahan vector untuk singkatnya stdlib.

### Langkah 3: blok trafo pada kedalaman k

Attention diri ditambah MLP. Di dalam mainan, blok attention linier satu lapis dan SwiGLU MLP menjaga struktur tetap terlihat tanpa numpy.

### Langkah 4: kepala output bersama

Gunakan kembali proyeksi output model utama. Logit atas kosakata.

### Langkah 5: kehilangan per kedalaman

Entropi silang softmax(logits) terhadap token kebenaran dasar di offset `k`. Gabungkan seluruh kedalaman dengan faktor penskalaan `lambda / D`.

### Langkah 6: akuntansi parameter

Cetak jumlah total parameter, jumlah bersama (embedding, head), dan jumlah tambahan per modul. Tunjukkan rasio ekstra MTP terhadap ukuran model utama.

## Pakai

MTP diintegrasikan ke dalam DeepSeek-V3 (Desember 2024) dan seri DeepSeek-R1. Pada inference:

- Tumpukan penyajian DeepSeek sendiri menggunakan modul MTP sebagai dekoder spekulatif.
- vLLM dan SGLang memiliki jalur integrasi untuk MTP DeepSeek-V3 mulai April 2026.
- Tutorial ROCm SGLang AMD menunjukkan konfigurasi decoding spekulatif MTP tertentu dengan kecepatan 1,8× terukur pada pos pemeriksaan V3.

Kapan menggunakan MTP dalam proses pra-training baru:

- kamu mengontrol seluruh jalur pra-training dan ingin menyimpan sinyal training yang lebih padat.
- kamu tahu bahwa kamu akan menyajikan model dalam skala besar dan menginginkan decoding spekulatif secara gratis.
- Ukuran tersembunyi kamu setidaknya 4096. Pada skala 1B, overhead lebih merugikan daripada keuntungan yang membantu.

Kapan tidak:

- Menyempurnakan model padat terlatih yang sudah ada. Modul MTP tidak dilatih.
- Teliti model yang ingin kamu bandingkan dengan dasar yang jelas. MTP mengubah arsitektur.

## Kirim

Lesson ini menghasilkan `outputs/skill-mtp-planner.md`. Mengingat spesifikasi proses pra-training (ukuran model, data, komputasi), ia mengembalikan rencana untuk mengintegrasikan MTP: jumlah kedalaman D, jadwal `lambda`, overhead memori, dan pengkabelan penguraian code spekulatif waktu inference.

## Latihan

1. Jalankan `code/main.py`. Tunjukkan loss per kedalaman berkurang secara monoton seiring dengan menguatnya sinyal sintetik. Ubah sintetik untuk menggunakan pola tetap dan pastikan kehilangan kedalaman-1 dan kedalaman-2 menyatu.

2. Hitung parameter overhead untuk model 70B padat (tersembunyi 8192, 80 layer) dengan modul D=1 MTP. Bandingkan dengan overhead 14B yang dilaporkan DeepSeek-V3. Jelaskan mengapa angka DeepSeek lebih tinggi: blok Transformer MTP mewarisi struktur MoE yang sama, sehingga meningkatkan jumlah parameter per modul.

3. Implementasikan D=2 pada mainan: tambahkan modul MTP kedua yang membutuhkan h^(1) dan prediksi `t_{i+2}`. Verifikasi loss gabungan dan akuntansi parameter sesuai dengan persamaan makalah DeepSeek 19-21.

4. Ganti mainan ke MTP paralel (gaya Gloeckle): tambahkan kepala output D di atas keadaan tersembunyi utama, masing-masing memprediksi offset yang berbeda. Ukur bagaimana loss per kedalaman dibandingkan dengan versi sekuensial pada sinyal sintetik yang sama. Versi sekuensial harus menghasilkan loss kedalaman-k yang lebih rendah untuk k > 1 karena mengkondisikan prediksi perantara.

5. Gunakan modul MTP terlatih sebagai draf gaya EAGLE: panggil modul k untuk mengusulkan `t_{i+k}` pada inference. Ukur tingkat penerimaan token draf ini terhadap prediksi model utama pada urutan yang telah ditentukan. Jika kamu mencapai 50%+ pada mainan tersebut, kamu telah mereproduksi properti empiris MTP-as-draft.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Modul MTP | "Blok loss ekstra" | Blok Transformer kecil ditambah proyeksi yang memprediksi posisi token `k` di depan model utama |
| Kedalaman prediksi | "Yang mana offset" | Bilangan bulat `k` sehingga modul `k` memprediksi `t_{i+k}` dari awalan hingga posisi `i` |
| MTP Paralel | "Gaya Gloeckle" | D kepala independen pada keadaan tersembunyi tulang punggung yang sama, tidak ada rantai bersyarat |
| MTP berurutan | "Gaya DeepSeek-V3" | Setiap modul mengkondisikan keadaan tersembunyi kedalaman sebelumnya ditambah embedding token berikutnya; mempertahankan rantai sebab akibat |
| Kepala output bersama | "Gunakan kembali kepala utama" | Modul MTP memanggil kepala LM model utama, bukan proyeksi output terpisah |
| Embedding bersama | "Gunakan kembali tabel utama" | Tabel embedding kosakata yang sama digunakan di mana-mana; tidak ada parameter duplikat |
| Matrix proyeksi M_k | "Gabungkan token tersembunyi + berikutnya" | Layer linier `h x 2h` yang melipat keadaan tersembunyi sebelumnya dan token target yang di-embed ke input kedalaman berikutnya |
| Loss sendi L_MTP | "Rata-rata loss ekstra" | Rata-rata aritmatika loss lintas entropi per kedalaman, diskalakan dengan `lambda` |
| Tingkat penerimaan pada kedalaman 1 | "Seberapa sering rancangan MTP yang benar" | Tingkat prediksi 1 teratas modul D=1 MTP sama dengan prediksi 1 teratas model utama; 80%+ di DeepSeek-V3 |
| Weighting Lambda | "Pentingnya loss ekstra" | Faktor skala per kedalaman; 0,3 pada awal training, 0,1 kemudian pada DeepSeek-V3 |

## Bacaan Lanjutan- [DeepSeek-AI — Laporan Teknis DeepSeek-V3 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — deskripsi MTP sekuensial lengkap (Bagian 2.2), termasuk persamaan loss gabungan dan percepatan 1,8× pada inference
- [Gloeckle dkk. — Large Language Model yang Lebih Baik & Lebih Cepat melalui Prediksi Multi-token (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) — desain DeepSeek garis dasar MTP paralel ditingkatkan
- [Kartu model DeepSeek-V3 di Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — total 685 miliar (671 miliar utama + 14 miliar MTP), catatan penerapan
- [Leviathan dkk. — Inference Cepat dari Transformers melalui Decoding Spekulatif (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — kerangka decoding spekulatif yang cocok dengan MTP
- [Li dkk. — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — Rancangan arsitektur EAGLE tahun 2025, saingannya MTP bersaing dengan
