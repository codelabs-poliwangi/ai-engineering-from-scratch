# Decoding Spekulatif dan EAGLE

> LLM perbatasan yang menghasilkan satu token memerlukan penerusan penuh pada miliaran parameter. Ketentuan forward pass tersebut terlalu banyak: sering kali model yang jauh lebih kecil dapat menebak 3-5 token berikutnya dengan benar, dan model besar hanya perlu *memverifikasi* tebakannya. Jika tebakannya benar, kamu mendapat 5 token dengan harga satu. Penguraian code spekulatif (Leviathan dkk. 2023) membuat hal ini menjadi tepat, dan EAGLE-3 (2025) mendorong tingkat penerimaan menjadi ~4,5 token per verifikasi — peningkatan 4-5x pada distribusi output yang sesuai.

**Type:** Build
**Language:** Python (dengan numpy)
**Prerequisites:** Fase 10 Lesson 12 (Optimization Inference), Fase 10 Lesson 04 (Pra-training Mini-GPT)
**Waktu:** ~75 menit

## Masalah

Throughput dekode untuk model kelas 70B pada H100 biasanya 40-80 token/detik. Setiap token memerlukan forward pass penuh untuk membaca semua weight model dari HBM. kamu tidak dapat memperkecil model tanpa mengubah keluarannya. kamu tidak dapat menambah ukuran batch melebihi memori. kamu terjebak - kecuali kamu dapat membiarkan model mengeluarkan lebih dari satu token per penerusan.

Generasi autoregresif pada dasarnya terlihat serial: `x_{t+1} = sample(p(· | x_{1:t}))`. Namun ada peluang konkurensi. Jika kamu memiliki prediktor murah yang mengatakan "4 token berikutnya mungkin [a, b, c, d]" kamu dapat memverifikasi semua 5 posisi dalam **satu forward pass dari model besar** dan menerima awalan pencocokan terpanjang.

Leviathan, Kalai, Matias (2023, "Inference Cepat dari Transformers melalui Decoding Spekulatif") melakukan hal ini dengan tepat melalui aturan terima/tolak cerdas yang mempertahankan distribusi pengambilan sample model target. Distribusi output yang sama, 2-4× lebih cepat.

## Konsep

### Pengaturan Dua Model

- **Model target** `M_p`: model besar, lambat, dan berkualitas tinggi yang sebenarnya kamu inginkan sampelnya. Distribusi: `p(x)`.
- **Model draf** `M_q`: model kecil, cepat, dan berkualitas lebih rendah. Distribusi: `q(x)`. 5-30× lebih kecil.

Per langkah:

1. Draf model mengusulkan token `K` secara otomatis: `x_1, x_2, ..., x_K ~ q`.
2. Model target menjalankan SATU penerusan ke depan pada semua posisi `K+1` secara paralel, menghasilkan `p(x_k)` untuk setiap token yang diusulkan.
3. Terima/tolak setiap token dari kiri ke kanan melalui aturan pengambilan sample penolakan yang dimodifikasi di bawah. Terima awalan yang paling lama cocok.
4. Jika ada token yang ditolak, ambil contoh penggantinya dari distribusi yang telah diperbaiki dan hentikan. Jika tidak, cicipi satu token bonus dari `p(· | x_1...x_K)`.

Jika drafnya cocok dengan target dengan sempurna, kamu mendapatkan token K+1 per target maju. Jika draft salah pada posisi 1 maka anda hanya mendapat 1 token saja.

### Aturan Ketepatan

Penguraian code spekulatif **terbukti setara dalam distribusi dengan pengambilan sample dari p**. Aturan penolakan:

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

dimana `(p - q)+` menunjukkan bagian positif dari perbedaan poin. Ketika draf dan target setuju (`p ≈ q`) penerimaannya hampir 1. Ketika tidak setuju, distribusi sisa dikonstruksikan sehingga keseluruhan sample masih persis `p`.

**Kasus serakah.** Untuk pengambilan sample suhu=0, cukup periksa `argmax(p) == x_t`. Jika ya, terima; jika tidak, keluarkan `argmax(p)` dan hentikan.

### Kecepatan yang Diharapkan

Jika tingkat penerimaan tingkat token model draf adalah `α`, token yang diharapkan dihasilkan per penerusan target adalah:

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```Di `α = 0.8, K = 4`: `(1 - 0.8^5)/(1 - 0.8) = 3.36` token per penerusan. Biaya satu target maju kira-kira `cost_q * K + cost_p` (K langkah draf ditambah satu verifikasi target). Jika `cost_p >> cost_q * K` rasio percepatannya adalah `3.36× / 1 = 3.36×` pada throughput.

Satu-satunya parameter sebenarnya adalah `α`, yang sepenuhnya bergantung pada penyelarasan target draf. Draf yang bagus adalah segalanya.

### Melatih Draf: Distilasi

Model kecil yang acak menghasilkan draf yang buruk. Resep standarnya adalah menyaring dari target:

1. Pilih arsitektur kecil (~1 miliar untuk target 70 miliar, ~500 juta untuk target 7 miliar).
2. Jalankan model target pada korpus teks besar; menyimpan distribusi token berikutnya.
3. Latih rancangan tersebut dengan perbedaan KL terhadap distribusi target (bukan terhadap bukti kebenaran dasar).

Hasilnya: `α` biasanya 0,6-0,8 pada pengkodean, 0,7-0,85 pada obrolan bahasa alami. Mempercepat 2-3× dalam produksi.

### EAGLE: Penyusunan Pohon + Penggunaan Kembali Feature

Li, Wei, Zhang, Zhang (2024, "EAGLE: Speculative Sampling Needs Rethinking Feature Uncertainty") mengamati dua inefisiensi dalam penguraian code spekulatif standar:

1. Draf melakukan K langkah serial, masing-masing tumpukan penuh. Namun draf tersebut dapat menggunakan kembali feature target (status tersembunyi) dari verifikasi terbaru — target telah menghitung representasi kaya yang diperoleh ulang dari awal oleh draf tersebut.
2. Draf menghasilkan rantai linier. Jika draf dapat menghasilkan *pohon* kandidat (setiap node memiliki beberapa tebakan), satu forward pass target dapat memverifikasi beberapa jalur kandidat secara paralel melalui attention mask pohon, dan memilih cabang terpanjang yang diterima.

Perubahan EAGLE-1:
- Draf input = status tersembunyi akhir target pada posisi t, bukan token mentah.
- Rancangan arsitektur = 1 layer dekoder Transformer (bukan model kecil yang terpisah).
- Output = pohon K = 4-8 kandidat per kedalaman, kedalaman 4-6.

EAGLE-2 (2024) menambahkan topologi pohon dinamis: pohon tumbuh lebih luas di tempat yang rancangannya tidak pasti dan tetap sempit di tempat yang yakin. Meningkatkan `α_effective` tanpa menambah biaya verifikasi.

EAGLE-3 (Li dkk. 2025, "EAGLE-3: Meningkatkan Akselerasi Inference Large Language Model melalui Tes Waktu Training") menghilangkan ketergantungan feature layer atas yang tetap dan melatih draf dengan loss "simulasi waktu pengujian" yang baru — draf dilatih berdasarkan output yang sesuai dengan distribusi waktu pengujian target, bukan distribusi training yang dipaksakan oleh guru. Tingkat penerimaan meningkat dari 0,75 (EAGLE-2) menjadi 0,82 (EAGLE-3), dan rata-rata token/verifikasi dari 3,0 menjadi 4,5.

### Verifikasi Attention Pohon

Saat draf menghasilkan output berupa pohon, model target akan memverifikasinya dalam satu penerusan menggunakan **topeng attention pohon** — masker sebab-akibat yang mengkodekan topologi pohon, bukan garis murni. Setiap token hanya melayani nenek moyangnya di pohon. Lulus verifikasi masih satu maju, satu matmul; topeng topologi hanya memerlukan beberapa entri KV tambahan.

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

Jika `a, b` bersaing dengan kandidat token pertama dan `c, d, e, f` adalah kandidat token kedua, keenam posisi diverifikasi dalam satu forward pass. Outputnya adalah awalan terpanjang di sepanjang jalur yang diterima.

### Saat Menang, Saat Tidak

**Menang:**
- Obrolan / penyelesaian dengan teks yang dapat diprediksi (code, bahasa Inggris umum, output terstruktur). `α` tinggi.
- Pengaturan dengan komputasi GPU yang tidak digunakan selama decode (fase terikat memori). Penyusunan pohon menggunakan FLOP yang tersedia.**Kalah / tidak menang:**
- Output sangat stokastik (penulisan kreatif pada suhu tinggi). `α` turun menuju `1/|vocab|`.
- Penyajian batch dengan konkurensi sangat tinggi — batching sudah memenuhi FLOP, sedikit ruang untuk verifikasi pohon.
- Model target yang sangat kecil dimana drafnya tidak jauh lebih kecil.

Toko produksi biasanya melaporkan kecepatan waktu 2-3× pada obrolan, 3-5× pada pembuatan code, dan mendekati nol pada penulisan kreatif.

## Build

`code/main.py`:

- Referensi `speculative_decode(target, draft, prompt, K, temperature)` yang menerapkan aturan penolakan yang tepat dan memverifikasi bahwa aturan tersebut mempertahankan distribusi target (KL empiris <0,01 vs pengambilan sample target biasa).
- Perancang pohon bergaya EAGLE yang membuat pohon depth-K dengan percabangan top-p.
- Pembuat topeng attention pohon yang menghasilkan pola sebab akibat yang tepat untuk verifikator.
- Harness tingkat penerimaan yang menjalankan keduanya pada LM kecil (menyaring satu GPT-2-kecil dari target medium GPT-2).

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## Pakai

- **vLLM** dan **SGLang** mengirimkan decoding spekulatif kelas satu. Bendera: `--speculative_model`, `--num_speculative_tokens`. Dukungan EAGLE-2/3 melalui bendera `--spec_decoding_algorithm eagle`.
- **NVIDIA TensorRT-LLM** mendukung pohon Medusa dan EAGLE secara asli.
- **Draf model referensi**: `Qwen/Qwen3-0.6B-spec` (draf untuk Qwen3-32B), `meta-llama/Llama-3.2-1B-Instruct-spec` (draf untuk 70B).
- **Medusa head** (Cai dkk. 2024, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"): alih-alih membuat draf model, tambahkan K head prediksi paralel ke target itu sendiri. Lebih mudah diterapkan, penerimaannya sedikit lebih rendah dibandingkan EAGLE.

## Kirim

Lesson ini menghasilkan `outputs/skill-speculative-tuning.md` — keterampilan yang memprofilkan weight kerja model target dan memilih: model draf, K (panjang draf), lebar pohon, suhu, dan kapan harus kembali ke dekode biasa.

## Latihan

1. Menerapkan aturan penolakan yang tepat dan memverifikasinya secara empiris. Jalankan 10 ribu sample melalui `speculative_decode` dan melalui pengambilan sample target biasa; menghitung distance TV antara dua distribusi output. Seharusnya <0,01.

2. Hitung rumus percepatan. Mengingat `α` dan `K` yang tetap, plot token yang diharapkan per target-forward. Temukan K optimal untuk α ∈ {0.5, 0.7, 0.9}.

3. Latih draf kecil. Ambil target 124 juta GPT-2 dan saring draf 30 juta GPT-2 pada 100 juta token dengan kehilangan KL. Ukur `α` pada teks yang didiamkan. Diharapkan: 0,6-0,7.

4. Terapkan penyusunan pohon gaya EAGLE. Daripada membuat rantai, buatlah rancangan output 3 cabang teratas di setiap kedalaman. Build topeng attention pohon. Verifikasikan target menerima cabang terpanjang yang benar.

5. Ukur mode kegagalan. Jalankan dekode spekulatif pada suhu=1,5 (stokastisitas tinggi). Tunjukkan α runtuh dan algoritme lebih lambat dibandingkan dekode biasa karena overhead draft.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Model sasaran | "Model besar" | Model lambat dan berkualitas tinggi yang kamu inginkan sampelnya (distribusi p) |
| Model rancangan | "Spekulator" | Prediktor kecil dan cepat (distribusi q); 5-30x lebih kecil |
| K/panjang draf | "Lihat ke depan" | Jumlah token yang berspekulasi per verifikasi pass |
| α / tingkat penerimaan | "Tingkat klik" | Probabilitas per token bahwa proposal draf diterima |
| Aturan penolakan yang tepat | "Tes penerimaan" | r < p/q bandingkan yang mempertahankan distribusi target |
| Distribusi sisa | "Pq yang dikoreksi" | (p - q)+ / ||(p - q)+||_1, distribusi ke sample dari penolakan |
| Penyusunan pohon | "Spekulasi bercabang" | Draf menghasilkan pohon kandidat, diverifikasi dalam sekali jalan dengan attention mask terstruktur pohon |
| Topeng attention pohon | "Topeng Topologi" | Masker kausal mengkodekan topologi pohon sehingga setiap node hanya memperhatikan leluhurnya |
| Kepala Medusa | "Kepala paralel" | K prediksi ekstra mengarah pada target itu sendiri; tidak ada rancangan model terpisah |
| Penggunaan kembali feature EAGLE | "Draf negara bagian tersembunyi" | Input draf adalah status tersembunyi terakhir target, bukan token mentah, sehingga menyusutkan draf |
| Loss simulasi waktu pengujian | "Training EAGLE-3" | Latih draf tentang output yang sesuai dengan distribusi waktu ujian target, bukan pemaksaan guru |

## Bacaan Lanjutan

- [Leviathan, Kalai, Matias, 2023 — "Inference Cepat dari Transformers melalui Penguraian Spekulatif"](https://arxiv.org/abs/2211.17192) — aturan penolakan yang tepat dan analisis percepatan teoritis
- [Chen, Borgeaud, Irving dkk., 2023 — "Mempercepat Dekode Large Language Model dengan Pengambilan Sample Spekulatif"](https://arxiv.org/abs/2302.01318) — makalah pengambilan sample spekulatif secara bersamaan di DeepMind
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Kerangka Akselerasi Inference LLM Sederhana dengan Beberapa Kepala Decoding"](https://arxiv.org/abs/2401.10774) — alternatif kepala paralel untuk model draf
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Pengambilan Sample Spekulatif Memerlukan Pemikiran Ulang Ketidakpastian Feature"](https://arxiv.org/abs/2401.15077) — penggunaan kembali feature dan penyusunan pohon
- [Li et al., 2024 — "EAGLE-2: Inference Model Bahasa yang Lebih Cepat dengan Pohon Draf Dinamis"](https://arxiv.org/abs/2406.16858) — topologi pohon dinamis
- [Li dkk., 2025 — "EAGLE-3: Meningkatkan Akselerasi Inference Large Language Model melalui Tes Waktu Training"](https://arxiv.org/abs/2503.01840) — pencocokan waktu pengujian waktu training
- [Fu, Haotian, Peng dkk., 2024 — "Memecahkan Ketergantungan Sekuensial Inference LLM Menggunakan Penguraian Lookahead"](https://arxiv.org/abs/2402.02057) — Penguraian Jacobi/lookahead, alternatif bebas spekulan
