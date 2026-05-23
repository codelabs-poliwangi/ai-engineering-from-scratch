# KV Cache, Attention Flash & Optimization Inference

> Training bersifat paralel dan terikat FLOP. Inference bersifat serial dan terikat memori. Kemacetan berbeda, trik berbeda.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 7 · 02 (Attention Diri), Fase 7 · 05 (Trafo Penuh), Fase 7 · 07 (GPT)
**Waktu:** ~75 menit

## Masalah

Dekoder autoregresif yang naif melakukan `O(N²)` berfungsi menghasilkan token `N`: pada setiap langkah ia menghitung ulang attention pada awalan penuh. Untuk respons token 4K yang merupakan operasi attention 16 juta, sebagian besar operasi tersebut mubazir. Setiap keadaan tersembunyi dari token awalan bersifat deterministik setelah dihitung — kamu hanya perlu menjalankan kueri token baru terhadap kunci dan nilai cache dari semuanya sebelumnya.

Selain itu, attention itu sendiri memindahkan banyak data. Attention standar mewujudkan matrix skor N×N, output softmax N×d, output akhir N×d — terlalu banyak membaca dan menulis ke HBM. Untuk N≥2K, attention menjadi terikat pada memori sebelum menjadi terikat pada FLOP. Kernel attention klasik kurang menggunakan GPU modern sebesar 4–10×.

Dua optimization, keduanya dari Dao dkk., mendorong inference perbatasan dari "lambat" menjadi "cepat":

1. **Cache KV.** Menyimpan vector K dan V dari setiap token awalan. Setiap attention token baru adalah satu kueri terhadap kunci cache. Inference berkurang dari `O(N²)` menjadi `O(N)` per langkah pembuatan.
2. **Flash Attention.** Susun komputasi attention sehingga matrix N×N penuh tidak pernah mencapai HBM. Semua softmax + matmul terjadi di SRAM. 2–4× percepatan jam dinding di A100; 5–10× pada H100 dengan FP8.

Pada tahun 2026 keduanya bersifat universal. Setiap tumpukan inference produksi (vLLM, TensorRT-LLM, SGLang, llama.cpp) mengasumsikannya. Setiap model perbatasan dikirimkan dengan Flash Attention yang diaktifkan.

## Konsep

![Pertumbuhan cache KV dan ubin Attention Flash](../assets/kv-cache-flash-attn.svg)

### Matematika cache KV

Per layer dekoder, per token, per kepala:

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

Untuk model 7B dengan 32 layer, 32 kepala, d_head=128, fp16:

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

Untuk Llama 3 70B (80 layer, d_head=128, GQA dengan kepala 8 KV):

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

10 GB itulah sebabnya Llama 3 70B pada konteks 128K memerlukan sebagian besar 40 GB A100 hanya untuk cache KV pada ukuran batch 1.

**GQA adalah kemenangan cache KV.** MHA dengan 64 head akan berukuran 32 GB. MLA mengompres lebih jauh lagi.

### Flash Attention — trik pemasangan ubin

Attention standar:

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

Tiga perjalanan pulang pergi HBM. Pada H100, bandwidth HBM adalah 3 TB/dtk; SRAM adalah 30 TB/dtk. Setiap perjalanan HBM merupakan faktor 10 perlambatan vs menjaga semuanya tetap berjalan.

Attention Kilat:

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

Satu perjalanan HBM per ubin. Total jejak memori berkurang dari `O(N²)` menjadi `O(N)`. Backward pass menghitung ulang beberapa nilai dari forward pass alih-alih menyimpannya — memori lain menang.

**Trik numerik.** Menjalankan softmax mempertahankan `(max, sum)` di seluruh ubin sehingga normalisasi akhir tepat. Bukan perkiraan — Flash Attention menghitung output yang sedikit identik dengan attention standar (modulo fp16 non-associativity).

**Evolusi versi:**| Versi | Tahun | Perubahan kunci | Mempercepat perangkat keras referensi |
|---------|------|-----------|-------------------------------|
| Kilatan 1 | 2022 | Kernel SRAM ubin | 2× pada A100 |
| Kilatan 2 | 2023 | Paralelisme yang lebih baik, urutan kausal-pertama | 3× pada A100 |
| Kilatan 3 | 2024 | Asinkroni hopper, FP8 | 1,5–2× pada H100 (~740 TFLOP FP16) |
| Kilatan 4 | 2026 | Pipa Blackwell 5 phase, perangkat lunak exp2 | Inference-pertama (hanya maju pada awalnya) |

Flash 4 hanya dapat diteruskan saat diluncurkan. Training masih menggunakan Flash 3. Dukungan GQA dan varlen untuk Flash 4 tertunda (pertengahan 2026).

### Penguraian code spekulatif — latensi lainnya menang

Model murah mengusulkan N token. Model besar memverifikasi semua N secara paralel. Jika verifikasi menerima k token, kamu membayar 1 forward pass model besar untuk k generasi. Khas k=3–5 pada code dan prosa.

Default tahun 2026:
- **EAGLE 2 / Medusa.** Kepala draf terintegrasi yang berbagi status tersembunyi pemverifikasi. Kecepatan 2–3× tanpa kehilangan kualitas.
- **Decoding spekulatif dengan model draf.** Peningkatan kecepatan 2–4× pada hardware konsumen.
- **Decoding Lookahead.** Iterasi Jacobi; tidak diperlukan rancangan model. Ceruk tapi gratis.

### Pengelompokan berkelanjutan

Inference batch klasik: tunggu hingga urutan paling lambat selesai, lalu mulai batch baru. Membuang GPU ketika respons singkat selesai lebih awal.

Pengelompokan berkelanjutan (pertama kali dikirimkan dalam Orca, sekarang dalam vLLM, TensorRT-LLM, SGLang): menukar permintaan baru ke dalam batch segera setelah permintaan lama selesai. Peningkatan throughput 5–10× untuk weight kerja obrolan biasa.

### PagedAttention — Cache KV sebagai memori virtual

feature utama vLLM. Cache KV dialokasikan dalam blok 16 token; tabel halaman memetakan posisi logis ke blok fisik. Memungkinkan kamu berbagi KV di seluruh sample paralel (pencarian berkas, pengambilan sample paralel), prefiks hot-swap untuk caching cepat, dan defragment memori. Peningkatan throughput 4× dibandingkan alokasi berdekatan yang naif.

## Build

Lihat `code/main.py`. Kami menerapkan:

1. Dekoder tambahan `O(N²)` yang naif.
2. Dekoder `O(N)` KV yang di-cache.
3. Softmax ubin yang mensimulasikan algoritma running-max Flash Attention.

### Langkah 1: cache KV

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

Sederhana: terus kembangkan vector K, V per token dalam daftar per layer, per kepala.

### Langkah 2: ubin softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

Output yang sedikit identik ke `softmax(qK) V` dalam satu kali pengambilan, tetapi setiap saat set kerjanya adalah blok `tile × d_head`, bukan `N × d_head` penuh.

### Langkah 3: bandingkan decoding naif vs yang di-cache pada pembuatan 100 token

Hitung operasi attention. Naif: `O(N²)` = 5050. Dalam cache: `O(N)` = 100. Code mencetak keduanya.

## Pakai

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

produksi vLLM:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

Caching awalan di seluruh permintaan adalah keuntungan besar di tahun 2026 — system prompt yang sama, beberapa contoh, atau dokumen konteks panjang menggunakan kembali KV di seluruh panggilan. Untuk weight kerja agen dengan prompt alat berulang, cache awalan secara rutin memperoleh perolehan throughput 5×.

## Kirim

Lihat `outputs/skill-inference-optimizer.md`. Keterampilan memilih implementasi attention, strategi cache KV, kuantisasi, dan decoding spekulatif untuk penerapan inference baru.

## Latihan1. **Mudah.** Jalankan `code/main.py`. Konfirmasikan bahwa decoder naif dan cache menghasilkan output yang sama; perhatikan perbedaan op-count.
2. **Medium.** Menerapkan cache awalan: jika diberi prompt P dan beberapa penyelesaian, jalankan satu penerusan ke P untuk mengisi cache KV, lalu cabang per penyelesaian. Ukur percepatan vs pengkodean ulang P untuk masing-masingnya.
3. **Sulit.** Mengimplementasikan mainan PagedPerhatian: Cache KV di blok 16 token tetap dengan daftar gratis. Saat suatu urutan selesai, kembalikan bloknya ke kumpulan. Simulasikan 1.000 penyelesaian obrolan dengan durasi yang bervariasi. Bandingkan fragmentasi memori vs alokasi yang berdekatan.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Cache KV | "Trik yang membuat decoding menjadi cepat" | Disimpan K dan V dari setiap token awalan; pertanyaan baru ditujukan kepada mereka alih-alih menghitung ulang. |
| HBM | "Memori utama GPU" | Memori Bandwidth Tinggi; 80 GB pada H100, 192 GB pada B200. ~bandwidth 3 TB/dtk. |
| SRAM | "Memori dalam chip" | Memori cepat per-SM, ~256 KB per SM pada H100. ~bandwidth 30 TB/dtk. |
| Attention Kilat | "Kernel attention ubin" | Menghitung attention tanpa mewujudkan N×N dalam HBM. |
| Pengelompokan berkelanjutan | "Pengelompokan tanpa menunggu" | Tukar urutan yang sudah jadi, yang baru, tanpa menguras batch. |
| PagedPerhatian | "judul vLLM" | Cache KV dialokasikan dalam blok tetap dengan tabel halaman; menghilangkan fragmentasi. |
| Cache awalan | "Gunakan kembali prompt yang panjang" | Cache KV untuk awalan bersama di seluruh permintaan; pemotongan biaya besar bagi agen. |
| Penguraian code spekulatif | "Draf + verifikasi" | Model rancangan murah mengusulkan token; model besar memverifikasi k dalam satu kali jalan. |

## Bacaan Lanjutan

- [Dao dkk. (2022). FlashAttention: Attention Akurat yang Cepat dan Hemat Memori dengan IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1.
- [Dao (2023). FlashAttention-2: Attention Lebih Cepat dengan Paralelisme dan Partisi Kerja yang Lebih Baik](https://arxiv.org/abs/2307.08691) — Flash 2.
- [Shah dkk. (2024). FlashAttention-3: Attention Cepat dan Akurat dengan Asinkroni dan Presisi Rendah](https://arxiv.org/abs/2407.08608) — Flash 3.
- [Catatan rilis FlashAttention-4 (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Pipeline 5 phase Blackwell dan trik software-exp2; baca repo README untuk peringatan peluncuran khusus maju yang disebutkan dalam lesson ini.
- [Kwon dkk. (2023). Manajemen Memori yang Efisien untuk Penyajian Large Language Model dengan PagedAttention](https://arxiv.org/abs/2309.06180) — makalah vLLM.
- [Leviathan dkk. (2023). Inference Cepat dari Transformers melalui Decoding Spekulatif](https://arxiv.org/abs/2211.17192) — decoding spesifikasi.
- [Li dkk. (2024). EAGLE: Pengambilan Sample Spekulatif Memerlukan Pemikiran Ulang Ketidakpastian Feature](https://arxiv.org/abs/2401.15077) — Makalah EAGLE-1/2 untuk pendekatan rancangan terpadu yang dikutip dalam lesson.
- [Cai dkk. (2024). Medusa: Kerangka Akselerasi Inference LLM Sederhana dengan Beberapa Kepala Decoding](https://arxiv.org/abs/2401.10774) — pendekatan Medusa yang dirujuk bersama EAGLE.
- [dokumen vLLM — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — penjelasan mendalam kanonik tentang blok 16 token dan desain tabel halaman.
