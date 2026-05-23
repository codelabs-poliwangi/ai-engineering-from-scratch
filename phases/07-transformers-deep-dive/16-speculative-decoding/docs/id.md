# Penguraian Code Spekulatif — Draf, Verifikasi, Ulangi

> Penguraian code autoregresif bersifat serial. Setiap token menunggu token sebelumnya. Penguraian code spekulatif memutus rantai: model murah menyusun N token, model mahal memverifikasi semua N dalam satu forward pass. Jika drafnya tepat, kamu membayar banyak uang untuk N generasi.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 7 · 07 (GPT Causal LM), Fase 7 · 12 (KV Cache & Flash Attention)
**Waktu:** ~60 menit

## Masalah

Pengambilan sample LLM 70 miliar, satu token memerlukan ~30 mdtk pada H100. Model draf 3B membutuhkan waktu ~3 mdtk. Jika kita membiarkan draf 3 miliar 5 token terlebih dahulu, lalu jalankan 70 miliar *sekali* untuk memverifikasi kelima token tersebut, totalnya adalah `5×3 + 30 = 45 ms` untuk maksimal 5 token yang diterima — dibandingkan `5×30 = 150 ms` untuk pembuatan garis lurus. Itulah inti dari decoding spekulatif: tukarkan sejumlah kecil memori GPU tambahan (model draf) untuk latensi dekode 2–4× lebih rendah.

Caranya adalah menjaga distribusinya. Pengambilan sample spekulatif, diperkenalkan oleh Leviathan et al. (2023) dan oleh Chen dkk. secara bersamaan, menjamin bahwa rangkaian output **terdistribusi secara identik** sesuai dengan apa yang akan dihasilkan oleh model besar itu sendiri. Tidak ada tradeoff kualitas. Lebih cepat.

Empat kelompok pasangan pemverifikasi draf mendominasi inference tahun 2026:

1. **Spekulatif vanilla (Leviathan 2023).** Draf model terpisah (misalnya, Llama 3 1B) + verifikator (misalnya, Llama 3 70B).
2. **Medusa (Cai 2024).** Beberapa kepala decoding pada verifikator memprediksi posisi `t+1..t+k` secara paralel. Tidak ada rancangan model terpisah.
3. **Keluarga EAGLE (Li 2024, 2025).** Draf ringan yang menggunakan kembali status tersembunyi verifikator; tingkat penerimaan yang lebih dekat dibandingkan vanilla; 3–4× tipikal.
4. **Decoding Lookahead (Fu 2024).** Iterasi Jacobi; tidak diperlukan rancangan model sama sekali. Spekulasi diri. Niche tapi bebas ketergantungan.

Setiap tumpukan inference produksi pada tahun 2026 mengirimkan decoding spekulatif secara default. vLLM, TensorRT-LLM, SGLang, dan llama.cpp semuanya mendukung setidaknya vanilla + EAGLE-2.

## Konsep

### Algoritma inti

Diberikan verifikator `M_q` dan draf yang lebih murah `M_p`:

1. Biarkan `x_1..x_k` menjadi awalan yang sudah diterjemahkan.
2. **Draf**: gunakan `M_p` untuk mengusulkan secara otomatis `d_{k+1}, d_{k+2}, ..., d_{k+N}` dengan probabilitas draf `p_1..p_N`.
3. **Verifikasi secara paralel**: jalankan `M_q` sekali pada `x_1..x_k, d_{k+1}, ..., d_{k+N}`, dapatkan probabilitas verifikator `q_1..q_{N+1}` untuk posisi `k+1..k+N+1`.
4. **Terima/tolak setiap draf token dari kiri ke kanan**: untuk setiap `i`, terima dengan probabilitas `min(1, q_i(d_i) / p_i(d_i))`.
5. Pada penolakan pertama di posisi `j`: sample `t_j` dari distribusi "sisa" `(q_j - p_j)_+` dinormalisasi. Semua draf setelah `j` akan dibuang.
6. Saat menerima semua `N`: contoh satu token tambahan `t_{N+1}` dari `q_{N+1}` (token bonus gratis).

Trik distribusi sisa adalah wawasan matematis yang menjaga output terdistribusi persis seperti `M_q` mengambil sample dari awal.

### Yang menentukan kecepatan

Misalkan `α` = tingkat penerimaan yang diharapkan per token draf. Misal `c` = rasio biaya draft-to-verifier. Per langkah:

- Generasi naif melakukan 1 panggilan model besar per token.
- Spekulatif melakukan 1 panggilan model besar per `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` token ketika `α` tinggi.Aturan umum di `α = 0.75` dan `N = 5`: panggilan model besar 3× lebih sedikit. Biaya drafnya 5× murah. Total jam dinding turun ~2,5×.

**α bergantung pada:**

- Seberapa baik draf tersebut mendekati verifikator. Keluarga yang sama/training data yang sama meningkatkan α secara signifikan.
- Strategi penguraian code. Draf serakah melawan verifikator serakah: α tinggi. Pengambilan sample suhu: lebih sulit untuk dicocokkan; penerimaan turun.
- Jenis tugas. Code dan output terstruktur menerima lebih banyak (dapat diprediksi); penulisan kreatif bentuk bebas menerima lebih sedikit.

### Medusa — draf tanpa model draf

Medusa menggantikan model rancangan dengan kepala output tambahan pada verifikator. Di posisi `t`:

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

Setiap kepala mengeluarkan lognya sendiri. Pada inference, kamu mengambil sample dari masing-masing kepala untuk mendapatkan urutan kandidat, kemudian memverifikasi dengan satu forward pass menggunakan skema attention pohon yang mempertimbangkan semua kelanjutan kandidat sekaligus.

Kelebihan: tidak ada model kedua. Kekurangan: menambahkan parameter yang bisa dilatih; membutuhkan phase penyesuaian yang diawasi (~1 miliar token); tingkat penerimaannya sedikit lebih rendah daripada spekulatif vanilla dengan draft yang bagus.

### EAGLE — draf yang lebih baik dengan menggunakan kembali status tersembunyi

EAGLE-1/2/3 (Li et al., 2024–2025) menjadikan rancangan model sebagai Transformer kecil (biasanya 1 layer) yang menyerap status tersembunyi layer terakhir pemverifikasi. Karena draf melihat representasi feature verifikator, maka prediksinya berkorelasi kuat dengan distribusi output verifikator. Tingkat penerimaan naik dari ~0,6 (vanila) menjadi 0,85+.

EAGLE-3 (2025) menambahkan pencarian pohon pada kandidat kelanjutan. vLLM dan SGLang mengirimkan EAGLE-2/3 sebagai jalur spesifikasi default untuk Llama 3/4 dan Qwen 3.

### Tarian cache KV

Verifikasi memasukkan `N` draf token ke verifikator dalam satu forward pass. Ini memperluas cache KV pemverifikasi sebanyak `N` entri. Jika beberapa draf ditolak, kamu harus mengembalikan cache ke panjang awalan yang diterima.

Implementasi produksi (vLLM `--speculative-model`, LookaheadDecoder TensorRT-LLM) menangani ini dengan buffer KV awal. Tulis dulu, berkomitmen pada penerimaan. Ini tidak sulit secara konseptual, tetapi rumit.

## Build

Lihat `code/main.py`. Kami menerapkan algoritma pengambilan sample spekulatif inti (langkah penolakan + distribusi sisa) dengan:

- Sebuah "model besar" yang merupakan deterministik-softmax atas distribusi code tangan (sehingga kita dapat memverifikasi matematika penerimaan secara analitis).
- Sebuah "model rancangan" yang merupakan gangguan dari model besar.
- Lingkaran penerimaan/penolakan yang menghasilkan distribusi marjinal yang sama dengan pengambilan sample langsung.

### Langkah 1: langkah penolakan

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` adalah bilangan acak seragam. `q_prob` adalah probabilitas pemverifikasi untuk token yang dirancang. `p_prob` adalah probabilitas rancangan model. Teorema Leviathan menyatakan bahwa keputusan Bernoulli ini, diikuti dengan pengambilan sample dari sisa penolakan, mempertahankan distribusi pemverifikasi dengan tepat.

### Langkah 2: distribusi sisa

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

Kurangi `p` dari `q` berdasarkan elemen, jepit nilai negatif ke nol, normalkan ulang. Contoh dari penolakan apa pun.

### Langkah 3: satu langkah spekulatif

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

Lima diterima → satu bonus → enam token diproduksi dalam satu tiket verifikasi.

### Langkah 4: mengukur tingkat penerimaanJalankan 10.000 langkah spekulatif pada berbagai tingkat kualitas draf. Tingkat penerimaan plot vs. perbedaan KL antara distribusi draf dan verifikator. kamu akan melihat hubungan yang monoton dan bersih.

### Langkah 5: verifikasi kesetaraan distribusi

Secara empiris: histogram token yang dihasilkan oleh loop spekulatif harus sesuai dengan histogram yang dihasilkan dengan pengambilan sample langsung dari verifikator. Ini adalah teorema Leviathan dalam praktiknya. Uji chi-kuadrat mengkonfirmasi kesalahan pengambilan sample.

## Pakai

Produksi:

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

TensorRT-LLM memiliki jalur Medusa tercepat pada pertengahan tahun 2026. `faster-whisper` membungkus decoding spekulatif untuk Whisper-large dengan draf kecil.

**Memilih draf:**

| Strategi | Kapan harus memilih | Mempercepat |
|----------|--------------|---------|
| Draf vanilla (keluarga Llama 1B/3B) | Prototipe cepat, tanpa training | 1,8–2,3× |
| Kepala Medusa | kamu dapat menyempurnakan pemverifikasi | 2–3× |
| EAGLE-2/3 | Produksi, kecepatan maks | 3–4× |
| Melihat ke Depan | Tanpa draf, tanpa training, tanpa parameter tambahan | 1,3–1,6× |

**Kapan TIDAK melakukan dekode spesifikasi:**

- Pembuatan 1-5 token dalam urutan tunggal. Overhead mendominasi.
- Pengambilan sample yang sangat kreatif / suhu tinggi (α turun).
- Penerapan dengan keterbatasan memori (model rancangan menambahkan VRAM).

## Kirim

Lihat `outputs/skill-spec-decode-picker.md`. Keterampilan memilih strategi decoding spekulatif (vanilla / Medusa / EAGLE / lookahead) dan menyetel parameter (N, suhu draf) untuk weight kerja inference baru.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Konfirmasikan bahwa distribusi token spekulatif cocok dengan distribusi sample langsung verifikator pada 50.000 token dalam chi-kuadrat p > 0,05.
2. **Sedang.** Percepatan plot (token per penerusan model besar) sebagai fungsi dari `N` untuk `α = 0.5, 0.7, 0.85`. Identifikasi `N` yang optimal untuk setiap α. (Petunjuk: token yang diharapkan per panggilan verifikasi = `(1 - α^{N+1}) / (1 - α)`.)
3. **Sulit.** Implementasikan Medusa kecil: ambil GPT puncak dari Lesson 14, tambahkan 3 kepala LM tambahan yang memprediksi posisi t+2, t+3, t+4. Berlatihlah dengan tinyshakespeare dengan kehilangan banyak kepala sendi. Bandingkan tingkat penerimaan vs rancangan vanilla yang dibuat dengan memotong model yang sama.
4. **Sulit.** Implementasikan rollback: mulai dengan cache KV awalan 10 token, masukkan 5 token draf, simulasikan penolakan di posisi 3. Verifikasikan pembacaan cache kamu dengan benar sesuai dengan "awalan + 2 draf pertama yang diterima" pada iterasi berikutnya.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Model rancangan | "Yang murah" | Model yang lebih kecil yang mengusulkan kandidat token; biasanya 10–50× lebih murah dibandingkan verifikator. |
| Verifikator | "Yang besar" | Model target yang distribusinya kami pertahankan; berjalan sekali per langkah spekulatif. |
| Tingkat penerimaan (α) | "Seberapa sering drafnya benar" | Probabilitas per token bahwa verifikator menerima draf tersebut. 0,7–0,9 tipikal. |
| Distribusi sisa | "Pengembalian penolakan" | `(q - p)_+` dinormalisasi; pengambilan sample dari penolakan ini mempertahankan distribusi verifikator. |
| Token bonus | "Yang gratis" | Ketika seluruh N draf diterima, ambil sample satu lagi dari distribusi langkah berikutnya dari verifikator. |
| Medusa | "Spekulatif tanpa draf" | Beberapa kepala LM pada verifikator memprediksi posisi t+1..t+k secara paralel. |
| ELANG | "Draf negara bagian tersembunyi" | Draf trafo kecil dikondisikan pada status tersembunyi layer terakhir verifikator. |
| Penguraian code ke depan | "Iterasi Jacobi" | Spekulasi diri menggunakan iterasi titik tetap; tidak ada model rancangan. |
| Attention pohon | "Verifikasi banyak kandidat sekaligus" | Verifikasi percabangan yang mempertimbangkan beberapa kelanjutan draf secara bersamaan. |
| Kembalikan KV | "Batalkan draf yang ditolak" | Gores penyangga KV; berkomitmen pada penerimaan, membuang pada penolakan. |

## Bacaan Lanjutan

- [Leviathan, Kalman, Matias (2023). Inference Cepat dari Transformers melalui Decoding Spekulatif](https://arxiv.org/abs/2211.17192) — algoritma inti dan teorema kesetaraan.
- [Chen dkk. (2023). Mempercepat Penguraian Code Large Language Model dengan Pengambilan Sample Spekulatif](https://arxiv.org/abs/2302.01318) — pengenalan bersamaan; bukti penolakan Bernoulli yang bersih.
- [Cai dkk. (2024). Medusa: Kerangka Akselerasi Inference LLM Sederhana dengan Beberapa Kepala Decoding](https://arxiv.org/abs/2401.10774) — Makalah Medusa; verifikasi attention pohon.
- [Li dkk. (2024). EAGLE: Pengambilan Sample Spekulatif Memerlukan Pemikiran Ulang Ketidakpastian Feature](https://arxiv.org/abs/2401.15077) — EAGLE-1; draf yang dikondisikan negara tersembunyi.
- [Li dkk. (2024). EAGLE-2: Inference Model Bahasa yang Lebih Cepat dengan Pohon Draf Dinamis](https://arxiv.org/abs/2406.16858) — EAGLE-2; kedalaman pohon yang dinamis.
- [Li dkk. (2025). EAGLE-3: Meningkatkan Akselerasi Inference Large Language Model melalui Tes Waktu Training](https://arxiv.org/abs/2503.01840) — EAGLE-3.
- [Fu dkk. (2024). Hancurkan Ketergantungan Sekuensial Inference LLM Menggunakan Decoding Lookahead](https://arxiv.org/abs/2402.02057) — pendekatan lookahead, tanpa draf.
- [dokumen vLLM — Penguraian Code Spekulatif](https://docs.vllm.ai/en/latest/features/spec_decode.html) — referensi produksi kanonik dengan keempat strategi yang terhubung.
- [Implementasi referensi SafeAILab / EAGLE](https://github.com/SafeAILab/EAGLE) — code referensi untuk EAGLE-1/2/3.
