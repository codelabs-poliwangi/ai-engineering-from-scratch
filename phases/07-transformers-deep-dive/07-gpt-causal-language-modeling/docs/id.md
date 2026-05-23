# GPT — Pemodelan Bahasa Kausal

> BERT melihat kedua sisi. GPT hanya melihat masa lalu. Topeng segitiga adalah satu baris code paling penting dalam AI modern.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 7 · 02 (Attention Diri), Fase 7 · 05 (Trafo Penuh), Fase 7 · 06 (BERT)
**Waktu:** ~75 menit

## Masalah

Model bahasa menjawab satu pertanyaan: jika diberi token `t-1` pertama, berapakah distribusi probabilitas pada token `t`? Latih sinyal tersebut — prediksi token berikutnya — dan kamu akan mendapatkan model yang dapat menghasilkan teks arbitrer satu token dalam satu waktu.

Untuk melatihnya secara end-to-end pada seluruh rangkaian secara paralel, kamu memerlukan prediksi setiap posisi untuk bergantung hanya pada posisi sebelumnya. Kalau tidak, model itu akan menipu dengan melihat jawabannya.

Topeng kausal melakukan hal ini. Ini adalah matrix segitiga atas tunggal dengan nilai `-inf` yang ditambahkan ke skor attention sebelum softmax. Setelah softmax, posisi tersebut menjadi 0. Setiap posisi hanya dapat melayani dirinya sendiri dan posisi sebelumnya. Dan karena kamu menerapkannya satu kali ke seluruh rangkaian, kamu mendapatkan N prediksi token berikutnya yang paralel dalam satu lintasan ke depan.

GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), GPT-4 (2023), GPT-5 (2024), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi — semuanya merupakan Transformer kausal khusus dekoder dengan loop inti yang sama. Hanya lebih besar, data lebih baik, dan RLHF lebih baik.

## Konsep

![Topeng kausal menciptakan matrix attention berbentuk segitiga](../assets/causal-attention.svg)

### Topeng

Diberikan barisan dengan panjang `N`, buatlah matrix `N × N`:

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

Tambahkan `M` ke skor attention mentah sebelum softmax. `exp(-inf) = 0`, jadi posisi bertopeng tidak memberikan weight apa pun. Setiap baris matrix attention merupakan distribusi probabilitas pada posisi sebelumnya saja.

Biaya implementasi: satu panggilan `torch.tril()`. Waktu untuk menghitung: nanodetik. Dampak di lapangan: segalanya.

### Training paralel, inference serial

Training: meneruskan seluruh urutan `(N, d_model)` satu kali, hitung N loss lintas entropi (satu per posisi), jumlah, backprop. Paralel sepanjang urutan. Inilah sebabnya mengapa training GPT berskala — kamu memproses 1 juta token dalam satu batch dalam satu pass GPU.

Inference: kamu menghasilkan token demi token. Umpan `[t1, t2, t3]`, dapatkan `t4`. Umpan `[t1, t2, t3, t4]`, dapatkan `t5`. Umpan `[t1, t2, t3, t4, t5]`, dapatkan `t6`. Cache KV (Lesson 12) menyimpan status tersembunyi `t1…tn` sehingga kamu tidak menghitung ulang setiap langkah. Tapi kedalaman serial pada inference = panjang output. Itulah pajak autoregresif dan mengapa decoding merupakan hambatan latensi di setiap LLM.

### Loss — shift per satu

Token yang diberikan `[t1, t2, t3, t4]`:

- Input: `[t1, t2, t3]`
- Target: `[t2, t3, t4]`

Untuk setiap posisi `i`, hitung `-log P(target_i | inputs[:i+1])`. Jumlah. Ini adalah entropi silang untuk keseluruhan rangkaian.

Setiap trafo LM yang pernah kamu dengar tentang kereta api mengalami loss ini. Pra-training, penyesuaian, SFT — loss yang sama, data berbeda.

### Strategi penguraian code

Setelah training, pilihan pengambilan sample menjadi lebih penting daripada yang dipikirkan orang.| Metode | Apa fungsinya | Kapan menggunakan |
|--------|--------------|-------------|
| Serakah | Argmax setiap langkah | Tugas deterministik, penyelesaian code |
| Suhu | Bagilah logit dengan T, contoh | Tugas kreatif, T lebih tinggi = lebih banyak keragaman |
| Top-k | Sample hanya dari token top-k | Membunuh ekor dengan probabilitas rendah |
| Top-p (inti) | Sample dari himpunan terkecil dengan prob kumulatif ≥ p | bawaan 2020+; beradaptasi dengan bentuk distribusi |
| Min-p | Simpan token dengan `p > min_p * max_p` | 2024+; lebih baik dalam menolak ekor panjang daripada top-p |
| Penguraian code spekulatif | Model draf mengusulkan N token, model besar memverifikasi | Pengurangan latensi 2–3× dengan kualitas yang sama |

Pada tahun 2026, suhu min-p + 0,7 adalah standar yang wajar untuk model weight terbuka. Penguraian code spekulatif adalah taruhan tabel untuk setiap tumpukan inference produksi.

### Apa yang membuat "resep GPT" berhasil

1. **Khusus decoder.** Tidak ada overhead encoder. Satu attention + FFN per layer.
2. **Penskalaan.** 124 juta → 1,5 miliar → 175 miliar → triliun. Hukum penskalaan Chinchilla (Lesson 13) memberi tahu kamu cara melakukan komputasi.
3. **Pembelajaran dalam konteks.** Muncul sekitar tahun 6B–13B. Model dapat mengikuti beberapa contoh tanpa melakukan penyesuaian.
4. **RLHF.** Pasca training tentang preferensi manusia mengubah teks mentah yang telah dilatih sebelumnya menjadi asisten obrolan.
5. **Pra-norm + RoPE + SwiGLU.** Training stabil dalam skala besar.

Arsitektur inti tidak banyak berubah sejak GPT-2. Segala sesuatu yang menarik telah terjadi dalam data, skala, dan pasca training.

## Build

### Langkah 1: topeng sebab akibat

Lihat `code/main.py`. Satu kalimat:

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Tambahkan ke skor attention sebelum softmax. Itulah keseluruhan mekanismenya.

### Langkah 2: model GPT 2 lapis

Tumpuk dua blok dekoder (attention mandiri bertopeng + FFN, tanpa attention silang). Tambahkan embedding token, pengkodean posisi, dan pelepasan embedding (terkait dengan matrix embedding token — trik standar sejak GPT-2).

### Langkah 3: prediksi token berikutnya, end-to-end

Pada vocab mainan 20 token, hasilkan logit di setiap posisi. Hitung loss lintas entropi terhadap target shift-per-satu. Tidak ada gradient — ini adalah pemeriksaan kewarasan forward-pass.

### Langkah 4: pengambilan sample

Terapkan serakah, suhu, top-k, top-p, min-p. Jalankan masing-masing pada prompt tetap dan bandingkan hasilnya. Fungsi pengambilan sample adalah 10 baris.

## Pakai

PyTorch, ungkapan 2026:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

Di bawah tenda, `generate()` menjalankan forward pass, menarik logit posisi akhir, mengambil sample token berikutnya, menambahkannya, dan mengulanginya. Setiap tumpukan inference LLM produksi (vLLM, TensorRT-LLM, llama.cpp, Ollama, MLX) mengimplementasikan loop yang sama dengan optimization yang berat — pra-pengisian batch, batching berkelanjutan, paging cache KV, decoding spekulatif.

**GPT vs BERT, masing-masing satu baris:** GPT memprediksi `P(x_t | x_{<t})`. BERT memprediksi `P(x_masked | x_unmasked)`. Loss menentukan apakah model dapat dihasilkan.

## Kirim

Lihat `outputs/skill-sampling-tuner.md`. Keterampilan memilih parameter pengambilan sample untuk tugas generasi baru dan menandai kapan decoding deterministik diperlukan.

## Latihan1. **Mudah.** Jalankan `code/main.py` dan verifikasi matrix attention kausal berbentuk segitiga bawah setelah softmax. Pemeriksaan langsung: baris 3 harus memiliki weight hanya di kolom 0–3.
2. **Sedang.** Menerapkan penelusuran berkas untuk lebar 4. Bandingkan perplexity berkas-4 vs serakah pada 10 prompt singkat. Apakah beam selalu menang? (Petunjuk: biasanya untuk terjemahan, bukan untuk obrolan terbuka.)
3. **Sulit.** Menerapkan decoding spekulatif: gunakan model 2 lapis kecil sebagai draf dan model 6 lapis sebagai pemverifikasi. Ukur percepatan jam dinding pada 100 penyelesaian dengan panjang 64. Konfirmasikan output sesuai dengan keinginan verifikator.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Topeng kausal | "Segitiga" | Matrix segitiga atas `-inf` ditambahkan ke skor attention sehingga posisi `i` hanya melihat posisi `≤ i`. |
| Prediksi token berikutnya | "Loss" | Entropi silang distribusi model terhadap token berikutnya yang sebenarnya di setiap posisi. |
| Autoregresif | "Hasilkan satu per satu" | Umpan output kembali sebagai input; paralelisme hanya selama training, bukan selama pembangkitan. |
| Logit | "Skor pra-softmax" | Output mentah dari kepala LM sebelum softmax; pengambilan sample terjadi pada ini. |
| Suhu | "Tombol kreativitas" | Bagilah logit dengan T; T→0 = serakah, T→∞ = seragam. |
| Atas-p | "Pengambilan sample inti" | Pangkas distribusi ke himpunan terkecil dengan jumlah ≥p; sample dari apa yang tersisa. |
| Min-p | "Lebih baik dari top-p" | Simpan token di tempat `p ≥ min_p × max_p`; menyesuaikan cutoff dengan ketajaman distribusi. |
| Penguraian code spekulatif | "Draf + verifikasi" | Model murah mengusulkan N token; model besar memverifikasi secara paralel. |
| Guru memaksa | "Trik training" | Selama training, berikan token sebelumnya yang sebenarnya, bukan prediksi model. Standar untuk setiap seq2seq LM. |

## Bacaan Lanjutan

- [Radford dkk. (2018). Meningkatkan Pemahaman Bahasa dengan Pra-Training Generatif](https://cdn.openai.com/research-covers/bahasa-unsupervised/bahasa_understanding_paper.pdf) — GPT-1.
- [Radford dkk. (2019). Model Bahasa adalah Pembelajar Multitask Tanpa Pengawasan](https://cdn.openai.com/better-bahasa-models/bahasa_models_are_unsupervised_multitask_learners.pdf) — GPT-2.
- [Brown dkk. (2020). Model Bahasa adalah Pembelajar yang Sedikit Pembelajar](https://arxiv.org/abs/2005.14165) — GPT-3 dan pembelajaran dalam konteks.
- [Leviathan, Kalman, Matias (2023). Inference Cepat dari Transformers melalui Decoding Spekulatif](https://arxiv.org/abs/2211.17192) — kertas decoding spesifikasi.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — code referensi LM kausal kanonik.
