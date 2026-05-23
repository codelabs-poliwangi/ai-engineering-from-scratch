# Pemeriksaan Gradient dan Perhitungan Ulang Activation

> Backprop menyimpan setiap activation perantara. Pada parameter 70B dan konteks 128K, itu berarti 3 TB activation per peringkat. Pos pemeriksaan memperdagangkan FLOP untuk memori: menghitung ulang alih-alih menyimpan. Pertanyaannya adalah segmen mana yang harus dihilangkan, dan jawabannya bukanlah “semuanya”.

**Type:** Build
**Language:** Python (dengan numpy, obor opsional)
**Prerequisites:** Phase 10 Lesson 04 (Pra-Training Mini-GPT), Phase 10 Lesson 05 (Penskalaan & Distribusi)
**Waktu:** ~70 menit

## Masalah

Training Transformer menyimpan, untuk setiap layer, input untuk setiap operasi yang dibedakan secara terbalik: input attention, proyeksi Q/K/V, output softmax, input FFN, output norm, dan aliran sisa. Untuk layer dengan ukuran tersembunyi `d`, panjang urutan `L`, batch `B`, ini berada pada urutan `12 * B * L * d` float per layer.

Untuk `d=8192, L=8192, B=1`, itu berarti 800 MB/layer di BF16. Model 64 layer memerlukan activation sebesar 51 GB — dan itu sebelum kamu mengalikannya dengan ukuran microbatch, sebelum kamu menambahkan perantara attention-softmax (`L^2` per kepala), dan sebelum kamu memfaktorkan salinan parsial tensor-paralel.

Tagihan dua sisi: weight BF16 ditambah status optimizer mungkin muat dalam 80 GB, namun activation membuat kamu melewatinya. Pos pemeriksaan gradient (alias penghitungan ulang activation) adalah perbaikan standar. Hilangkan sebagian besar activation; ulangi maju selama mundur untuk mendapatkannya kembali. Biaya: FLOP tambahan. Manfaat: memori turun sebesar rasio segmen pos pemeriksaan terhadap total layer.

Jika dilakukan secara naif, pos pemeriksaan membutuhkan biaya FLOP forward-pass sekitar 33% lebih banyak per langkah. Dilakukan dengan baik - pos pemeriksaan selektif sesuai dengan "pilihan cerdas" Korthikanti dkk. — kamu menghemat 5x memori dengan overhead FLOP di bawah 5%. Dan dengan matmuls FP8, offload FSDP, dan MoE paralel ahli, hal ini sangat penting: kamu tidak mampu membeli memori atau komputasi yang terbuang.

## Konsep

### Apa yang Sebenarnya Dibutuhkan Mundur

`output = layer(input)`. Keinginan mundur `grad_input` dan `grad_params`. Untuk menghitungnya diperlukan:

- `input` (untuk menghitung `grad_params = input.T @ grad_output` untuk layer linier)
- beberapa zat antara turunan activation (turunan ReLU/GELU/softmax bergantung pada nilai activation)

Forward pass menyimpannya secara otomatis di grafik autograd. Setiap `tensor.retain_grad()` dan setiap operasi yang memerlukan masukannya menyimpan referensi.

### Pos Pemeriksaan Penuh yang Naif

Bagi jaringan menjadi segmen `N`. Selama penerusan, simpan hanya *input* ke setiap segmen. Ketika ke belakang membutuhkan zat perantara, jalankan kembali umpan ke depan segmen tersebut untuk mewujudkannya, lalu bedakan.

Contoh: Trafo 32 lapis dipecah menjadi 32 segmen yang masing-masing 1 lapis.

- Memori: 32 input layer (kecil) vs 32 * (volume activation per layer) (besar).
- Komputasi ekstra: 1 maju ekstra per segmen, yaitu, ~33% lebih banyak total FLOP maju (karena mundur adalah 2x maju, langkah penuh menjadi 1 + 1 + 2 = 4 unit, bukan 1 + 2 = 3).

Ini adalah Chen dkk yang asli. Resep 2016: satu pos pemeriksaan setiap layer `sqrt(L)` untuk menyeimbangkan memori dan komputasi. Untuk L=64, itu berarti 8 pos pemeriksaan.

### Pos Pemeriksaan Selektif (Korthikanti 2022)

Tidak semua activation dikenakan biaya yang sama. Output softmax attention adalah `B*L*L*heads` dan tumbuh *secara kuadrat* dengan panjang urutan. Activation tersembunyi FFN adalah `B*L*4d` dan berkembang secara linier. Untuk urutan yang panjang softmax mendominasi.Pos pemeriksaan selektif menyimpan activation yang murah (proyeksi linier, residu) dan hanya menghitung ulang activation yang mahal (attention). kamu membayar FLOP minimal untuk menghitung ulang tetapi menghemat memori O(L^2).

Megatron-Core mengimplementasikan ini sebagai penghitungan ulang activation "selektif". Digunakan di sebagian besar latihan perbatasan tahun 2024+.

### Pembongkaran

Alternatif untuk menghitung ulang: mengirimkan activation ke RAM CPU antara maju dan mundur. Membutuhkan bandwidth PCIe; bermanfaat ketika bandwidth menganggur melebihi biaya rematerialisasi. Strategi campuran adalah hal yang umum: memeriksa beberapa layer, membongkar yang lain.

FSDP2 mengirimkan offload sebagai opsi kelas satu. Offload bersinar ketika GPU mengalami hambatan pada memori tetapi transfer CPU-GPU memiliki ruang kepala.

### Menghitung Ulang Model Biaya

FLOP per langkah dengan pos pemeriksaan naif setiap `k` layer dari `L`:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

Dengan pos pemeriksaan selektif kamu hanya menghitung ulang kernel attention, bukan seluruh layer:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Model Penghematan Memori

Volume activation per layer: `A`. Untuk layer `L`, total memori activation: `L * A`.

Pos pemeriksaan penuh (ukuran segmen 1): simpan hanya `L * input_volume` (~`L * 1/10 A` untuk trafo standar). Menyimpan ~`9 * L * A * 1/10`.

Periksa setiap layer `k`: simpan layer `L/k * A` plus `k-1` dalam segmen aktif.

Di `k = sqrt(L)`, biaya memori dan komputasi ulang berskala dengan `sqrt(L)` — tradeoff optimal untuk layer biaya seragam.

### Kapan Tidak ke Pos Pemeriksaan

- Layer terdalam dari phase pipa sudah dalam penerbangan. Bagaimanapun mereka harus menyelesaikannya.
- Layer pertama dan terakhir jika mendominasi komputasi tahapan (jarang terjadi pada Transformer).
- Kernel attention sudah menggunakan FlashAttention — Flash sudah menghitung ulang softmax dengan cepat, jadi pos pemeriksaan tingkat layer tambahan menambahkan sedikit tambahan.

### Pola Penerapan

1. **Pembungkus fungsi:** membungkus segmen di `torch.utils.checkpoint.checkpoint(fn, input)`. PyTorch hanya menyimpan `input`, menghitung ulang semuanya secara terbalik.

2. **Berbasis dekorator:** memberi label pada layer sebagai titik pemeriksaan; pelatih memutuskan pada waktu konfigurasi segmen mana yang akan dibungkus.

3. **Penghitungan ulang eksplisit secara manual:** tulis sendiri pass belakang, panggil `recompute_forward` khusus yang menduplikasi penerusan dengan input yang disimpan.

Ketiganya memberikan hasil fungsional yang sama. Pembungkus adalah idiom standar.

### Interaksi dengan TP/PP/FP8

- **Tensor paralel:** input pos pemeriksaan harus dikumpulkan atau disebarkan ulang saat dihitung ulang; menangani biaya komunikasi.
- **Pipeline parallel:** pola umumnya adalah memeriksa maju setiap phase pipeline sehingga microbatch urutan terbalik dapat menggunakan kembali memori activation.
- **Penghitungan ulang FP8:** Riwayat amax yang diperbarui selama penghitungan ulang harus cocok dengan penerus asli, atau skala FP8 akan menyimpang. Sebagian besar framework menggambarkan skalanya.

## Build

### Langkah 1: Model Mainan Dengan Segmen

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Langkah 2: Naif Mundur Membutuhkan Semua Activation

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Langkah 3: Memori Pos Pemeriksaan-Setiap-k

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Langkah 4: Model Biaya

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Langkah 5: Penaksir Memori

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Langkah 6: Ukuran Segmen Optimal

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Langkah 7: Keputusan Pos Pemeriksaan Selektif

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## Pakai- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — pembungkus kanonik di PyTorch. Membungkus suatu fungsi; hanya menyimpan input, menghitung ulang secara mundur.
- **Penghitungan ulang activation Megatron-Core**: mendukung mode `selective`, `full`, dan `block`. Standar pada training perbatasan tahun 2024+.
- **Pembongkaran FSDP2**: `module.to_empty(device="cpu")` dengan `offload_policy` dalam activation pecahan FSDP2 ke CPU alih-alih menghitung ulang.
- **DeepSpeed ​​ZeRO-Offload**: Pembongkaran CPU untuk status optimizer dan activation, melengkapi pos pemeriksaan.

## Kirim

Lesson ini menghasilkan `outputs/prompt-activation-recompute-policy.md` — sebuah prompt yang mengambil konfigurasi model kamu (layer, tersembunyi, seq, batch) dan memori GPU yang tersedia dan mengeluarkan kebijakan penghitungan ulang per layer (tidak ada / selektif / penuh / pembongkaran).

## Latihan

1. Verifikasi kebenarannya. Jalankan `model_forward` + `model_backward` (activation penuh) vs `model_forward_checkpointed` + `model_backward_checkpointed` (segmen). Gradient parameter harus identik dengan presisi mesin.

2. Sapu ukuran segmen `k` dari 1 menjadi `L`. Plot FLOP overhead dan memori. Temukan lutut kurva.

3. Menerapkan pos pemeriksaan selektif: simpan input modul attention tetapi bukan perantaranya. Ukur overhead FLOP vs pos pemeriksaan layer penuh untuk model 32 layer pada seq=8192.

4. Tambahkan pembongkaran. Simpan input segmen ke "buffer CPU" yang disimulasikan (daftar terpisah). Ukur "bandwidth PCIe" sebagai byte/waktu dan temukan titik impas antara pembongkaran dan penghitungan ulang.

5. Tolok ukur trafo PyTorch asli dengan dan tanpa `torch.utils.checkpoint`. Ukur memori (melalui `torch.cuda.max_memory_allocated`) dan waktu langkah.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|----------------------|
| Pos pemeriksaan gradient | "Hemat memori dengan mengulang ke depan" | Hanya input segmen toko; menghitung ulang perantara selama mundur untuk mendapatkan tensor dukungan gradient |
| Perhitungan ulang activation | "Sama seperti pos pemeriksaan" | Nama rasa HPC untuk teknik yang sama |
| Ukuran segmen (k) | "Berapa banyak layer per pos pemeriksaan" | Jumlah layer yang zat antaranya dijatuhkan dan dirematerialisasikan bersama |
| Pos pemeriksaan selektif | "Trik Korthikanti" | Hitung ulang hanya activation yang mahal untuk disimpan (attention softmax); simpan yang murah |
| Pos pemeriksaan penuh | "Versi naif" | Hitung ulang perantara setiap layer di setiap segmen |
| Blokir pos pemeriksaan | "Berbutir kasar" | Pos pemeriksaan seluruh blok trafo; granularitas terbesar |
| FLOP atas | "Pajak komputasi" | FLOP ekstra per langkah = (menghitung ulang FLOP) / (fwd + bwd FLOPs); 33% naif, 5% selektif |
| Pembongkaran activation | "Kirim ke CPU" | Pindahkan activation ke RAM CPU maju->mundur; alternatif untuk menghitung ulang |
| aturan sqrt-L | "Optimum klasik" | Untuk layer berbiaya seragam, distance titik pemeriksaan optimal adalah layer sqrt(L) |
| Volume attention-softmax | "Masalah O(L^2)" | L^2 * kepala * pelampung batch; mendominasi memori activation pada konteks panjang |

## Bacaan Lanjutan- [Chen et al., 2016 -- "Melatih Jaring Dalam dengan Biaya Memori Sublinear"](https://arxiv.org/abs/1604.06174) -- makalah asli yang memformalkan pos pemeriksaan gradient
- [Korthikanti et al., 2022 -- "Mengurangi Perhitungan Ulang Activation pada Model Transformer Besar"](https://arxiv.org/abs/2205.05198) -- perhitungan ulang activation selektif dan analisis biaya formal
- [Pudipeddi et al., 2020 -- "Melatih Jaringan Neural Besar dengan Memori Konstan menggunakan Algoritma Eksekusi Baru"](https://arxiv.org/abs/2002.05645) -- pendekatan memori konstan alternatif melalui rematerialisasi mode terbalik
- [Ren dkk., 2021 -- "ZeRO-Offload: Demokratisasi Training Model Berskala Miliar"](https://arxiv.org/abs/2101.06840) -- activation offload dalam skala besar
- [dokumen PyTorch torch.utils.checkpoint](https://pytorch.org/docs/stable/checkpoint.html) -- API standar
- [Dokumentasi penghitungan ulang activation Megatron-Core](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- mode selektif, penuh, dan blok
