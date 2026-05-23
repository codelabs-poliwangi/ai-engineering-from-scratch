# Campuran Pakar (MoE)

> Transformer 70B yang padat mengaktifkan setiap parameter untuk setiap token. MoE 671B hanya mengaktifkan 37B per token dan mengalahkannya di setiap benchmark. Ketersebaran adalah gagasan penskalaan yang paling penting pada dekade ini.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 7 · 05 (Trafo Penuh), Fase 7 · 07 (GPT)
**Waktu:** ~45 menit

## Masalah

FLOP Transformer padat pada inference sama dengan jumlah parameternya (dikalikan 2 untuk lintasan maju). Tingkatkan model yang padat dan setiap token membayar tagihan penuh. Pada tahun 2024, batas komputasi sudah terbentur: untuk menjadi lebih cerdas, kamu memerlukan lebih banyak FLOP per token secara eksponensial.

Campuran Para Ahli memutus tautan ini. Ganti setiap FFN dengan `E` pakar independen + router yang memilih `k` pakar per token. Parameter total = `E × FFN_size`. Parameter aktif per token = `k × FFN_size`. Konfigurasi umum tahun 2026: `E=256`, `k=8`. Skala penyimpanan dengan `E`, komputasi skala dengan `k`.

Perbatasan tahun 2026 hampir seluruhnya adalah MoE: DeepSeek-V3 (total 671B / 37B aktif), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2, gpt-oss. Di papan peringkat independen Analisis Buatan, 10 model sumber terbuka teratas semuanya adalah MoE.

## Konsep

![Layer MoE: router memilih k dari E pakar per token](../assets/moe.svg)

### Pertukaran FFN

Blok trafo padat:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

Blok MoE:

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

Setiap pakar adalah FFN independen (biasanya SwiGLU). Router adalah layer linier tunggal. Setiap token memilih ahli `k` miliknya sendiri dan mendapatkan campuran output mereka yang terjaga keamanannya.

### Masalah penyeimbangan weight

Jika router menempatkan 90% token melalui pakar 3, pakar lainnya akan kelaparan. Tiga perbaikan telah dicoba:

1. **Loss penyeimbangan weight tambahan** (Switch Transformer, Mixtral). Tambahkan penalti yang sebanding dengan varians dalam penggunaan ahli. Berfungsi, tetapi menambahkan hyperparameter dan sinyal gradient kedua.
2. **Kapasitas ahli + penurunan token** (Peralihan awal). Setiap pakar memproses paling banyak `C × N/E` token; token overflow lewati layer. Kualitas buruk.
3. **Penyeimbangan bebas loss tambahan** (DeepSeek-V3). Tambahkan bias per pakar yang dipelajari yang menggeser pilihan teratas router. Bias diperbarui di luar loss training. Tidak ada penalti pada tujuan utama. Pembukaan besar tahun 2024.

Pendekatan DeepSeek-V3: setelah setiap langkah training, untuk setiap pakar, periksa apakah penggunaannya di atas atau di bawah target. Singkirkan bias tersebut dengan `±γ`. Seleksi menggunakan `scores + bias`. Probabilitas ahli yang digunakan untuk gating adalah `scores` mentah yang tidak berubah. Memisahkan perutean dari ekspresi.

### Pakar bersama

DeepSeek-V2/V3 juga membagi pakar menjadi *dibagi* dan *dirutekan*. Setiap token melewati semua pakar bersama. Pakar yang diarahkan dipilih melalui top-k. Pakar bersama menangkap pengetahuan umum; pakar yang diarahkan berspesialisasi. V3 menjalankan 1 pakar bersama ditambah 8 teratas dari 256 yang dirutekan.

### Pakar yang sangat teliti

MoE Klasik (GShard, Switch): setiap pakar selebar FFN penuh. `E` kecil (8–64), `k` kecil (1–2).

MoE berbutir halus modern (DeepSeek-V3, Qwen-MoE): setiap pakar lebih sempit (ukuran 1/8 FFN). `E` besar (256+), `k` lebih besar (8+). Parameter totalnya sama, tetapi skala kombinasinya jauh lebih cepat. `C(256, 8) = 400 trillion` kemungkinan "ahli" per token. Kualitas meningkat, latensi tetap datar.### Profil biaya

Per token, per layer:

| Konfigurasi | Param/token aktif | Jumlah parameter |
|--------|-----------------------|--------------|
| Campuran 8×22B | ~39B | 141B |
| Llama 3 70B (padat) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 mengalahkan Llama 3 70B (padat) di hampir semua benchmark sambil melakukan **lebih sedikit FLOP aktif per token**. Lebih banyak parameter = lebih banyak pengetahuan. FLOP yang lebih aktif = lebih banyak komputasi per token. MoE memisahkan mereka.

### Hasil tangkapan: memori

Semua pakar hidup dengan GPU, apa pun yang diaktifkan. Model 671B memerlukan ~1,3 TB VRAM untuk weight fp16. Penerapan Frontier MoE memerlukan paralelisme pakar — pakar shard di seluruh GPU, merutekan token di seluruh jaringan. Latensi didominasi oleh komunikasi semua ke semua, bukan matmul.

## Build

Lihat `code/main.py`. Layer MoE kompak di stdlib murni dengan:

- `n_experts=8` Pakar SwiGLU (masing-masing satu linier, sebagai ilustrasi)
- perutean top-k=2
- anak timbangan gating yang dinormalisasi softmax
- penyeimbangan tambahan tanpa loss melalui bias per pakar

### Langkah 1: router

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Bias mempengaruhi seleksi, bukan weight gerbang. Itulah trik DeepSeek-V3 — bias mengoreksi ketidakseimbangan weight tanpa mengarahkan prediksi model.

### Langkah 2: jalankan 100 token melalui router

Lacak pakar mana yang memecat seberapa sering. Tanpa bias, penggunaan akan menjadi tidak seimbang. Dengan loop pembaruan bias (`-γ` untuk pakar yang terlalu banyak digunakan, `+γ` untuk yang kurang digunakan), penggunaan menyatu ke distribusi seragam dalam beberapa iterasi.

### Langkah 3: perbandingan jumlah param

Cetak "setara padat" dari konfigurasi MoE. Berbentuk DeepSeek-V3: 256 dirutekan + 1 dibagikan, 8 aktif, d_model=7168. Jumlah parameter totalnya sangat menakjubkan. Hitungan aktifnya adalah sepertujuh dari Llama 3 70B yang padat.

## Pakai

Memeluk Pemuatan wajah:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

Kesimpulan produksi 2026: vLLM mendukung perutean MoE secara asli. SGLang memiliki jalur pakar-paralel tercepat. Keduanya secara otomatis menangani pemilihan top-k dan paralelisme ahli.

**Kapan memilih MoE:**
- kamu menginginkan kualitas terdepan dengan biaya inference per token yang lebih rendah.
- kamu memiliki infrastruktur VRAM / pakar-paralel.
- Weight kerja kamu bersifat token-heavy (obrolan, code) bukan konteks-berat (dokumen panjang).

**Kapan TIDAK memilih MoE:**
- Penyebaran Edge — kamu membayar penyimpanan penuh untuk FLOP aktif apa pun.
- Pelayanan pengguna tunggal yang kritis terhadap latensi — perutean ahli menambah overhead.
- Model kecil (<7B) — Keunggulan kualitas MoE hanya muncul di atas ambang batas komputasi (~6B parameter aktif).

## Kirim

Lihat `outputs/skill-moe-configurator.md`. Keterampilan memilih tata letak E, k, dan pakar bersama untuk MoE baru berdasarkan anggaran parameter, token training, dan target penerapan.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Perhatikan bagaimana pembaruan bias bebas loss tambahan meratakan penggunaan ahli dalam lebih dari 50 iterasi.
2. **Sedang.** Ganti router yang dipelajari dengan router berbasis hash (deterministik, tanpa pembelajaran). Bandingkan kualitas dan keseimbangan. Mengapa router yang dipelajari lebih baik?
3. **Sulit.** Menerapkan "perutean yang cocok dengan peluncuran" gaya GRPO (trik DeepSeek-V3.2): log yang diaktifkan oleh pakar selama inference, memaksakan perutean yang sama selama komputasi gradient. Ukur dampaknya pada pengaturan gradient kebijakan mainan.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Ahli | "Satu FFN di antara banyak" | Jaringan feed-forward yang independen; parameter yang didedikasikan untuk sebagian kecil komputasi FFN. |
| Perute | "Gerbang" | Layer linier kecil yang memberi skor pada setiap token terhadap setiap pakar; pilihan top-k. |
| Perutean k teratas | "k pakar aktif per token" | Perhitungan FFN setiap token melewati tepat k pakar, ditimbang berdasarkan gerbang. |
| Loss tambahan | "Penalti keseimbangan weight" | Istilah loss ekstra yang menghukum penggunaan ahli yang tidak tepat. |
| Bebas loss tambahan | "Trik DeepSeek-V3" | Saldo melalui bias per-ahli pada pilihan router saja; tidak ada gradient tambahan. |
| Pakar bersama | "Selalu aktif" | Pakar ekstra yang dilalui setiap token; menangkap pengetahuan umum. |
| Paralelisme ahli | "Pecahan oleh ahli" | Mendistribusikan pakar yang berbeda ke GPU yang berbeda; token rute di seluruh jaringan. |
| Ketersebaran | "Param aktif <total param" | Rasio `k × expert_size / (E × expert_size)`; 37/671 ≈ 5,5% untuk DeepSeek-V3. |

## Bacaan Lanjutan

- [Shazeer dkk. (2017). Jaringan Neural yang Sangat Besar: Layer Campuran Pakar dengan Gerbang Jarang](https://arxiv.org/abs/1701.06538) — idenya.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Menskalakan ke Triliun Model Parameter dengan Ketersebaran yang Sederhana dan Efisien](https://arxiv.org/abs/2101.03961) — Switch, MoE klasik.
- [Jiang dkk. (2024). Campuran Pakar](https://arxiv.org/abs/2401.04088) — Campuran 8×7B.
- [DeepSeek-AI (2024). Laporan Teknis DeepSeek-V3](https://arxiv.org/abs/2412.19437) — MLA + MoE + MTP bebas loss tambahan.
- [Wang dkk. (2024). Strategi Penyeimbangan Weight Bebas Loss Tambahan untuk Pakar Campuran](https://arxiv.org/abs/2408.15664) — makalah penyeimbangan berbasis bias.
- [Dai dkk. (2024). DeepSeekMoE: Menuju Spesialisasi Pakar Utama dalam Model Bahasa Campuran Pakar](https://arxiv.org/abs/2401.06066) — pembagian pakar yang terperinci + bersama yang digunakan router lesson ini.
- [Kim dkk. (2022). DeepSpeed-MoE: Memajukan Inference dan Training Campuran Pakar](https://arxiv.org/abs/2201.05596) — makalah asli bersama pakar.
