# Optimization Kebijakan Proksimal (PPO)

> A2C membuang setiap peluncuran setelah satu pembaruan. PPO membungkus gradient kebijakan dalam rasio kepentingan yang terpotong sehingga kamu dapat melakukan 10+ periode pada data yang sama tanpa kebijakan meledak. Schulman dkk. (2017). Masih menjadi algoritme gradient kebijakan default pada tahun 2026.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 9 · 06 (PERKUAT), Fase 9 · 07 (Aktor-Kritikus)
**Waktu:** ~75 menit

## Masalah

A2C (Lesson 07) sesuai kebijakan: gradient `E_{π_θ}[A · ∇ log π_θ]` memerlukan sample data dari *saat ini* `π_θ`. Ambil satu pembaruan, dan `π_θ` berubah; data yang kamu gunakan sekarang di luar kebijakan. Gunakan kembali dan gradient kamu menjadi bias.

Peluncuran itu mahal. Di Atari, satu peluncuran di 8 envs × 128 langkah = 1024 transisi dan waktu lingkungan belasan detik. Membuangnya setelah satu langkah gradient adalah sia-sia.

Optimization Kebijakan Wilayah Kepercayaan (TRPO, Schulman 2015) adalah perbaikan pertama: membatasi setiap pembaruan sehingga perbedaan KL antara kebijakan lama dan baru tetap di bawah `δ`. Secara teoritis bersih, tetapi memerlukan penyelesaian gradient konjugasi setiap pembaruan. Tidak ada yang menjalankan TRPO pada tahun 2026.

PPO (Schulman dkk. 2017) menggantikan batasan wilayah kepercayaan dengan tujuan sederhana yang terpotong. Satu baris code tambahan. Sepuluh zaman per peluncuran. Tidak ada gradient konjugasi. Jaminan teoretis yang cukup baik. Sembilan tahun kemudian, algoritma ini masih menjadi algoritma gradient kebijakan default untuk segala hal mulai dari MuJoCo hingga RLHF.

## Konsep

![Tujuan pengganti PPO terpotong: pemotongan rasio pada 1 ± ε](../assets/ppo.svg)

**Rasio kepentingan.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

Ini adalah rasio kemungkinan kebijakan baru vs kebijakan yang mengumpulkan data. `r_t = 1` berarti tidak ada perubahan. `r_t = 2` berarti kebijakan baru ini memiliki kemungkinan dua kali lebih besar untuk menerima `a_t` dibandingkan kebijakan lama.

**Pengganti yang terpotong.**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

Dua istilah:

- Jika keunggulan `A_t > 0` dan rasio mencoba melampaui `1 + ε`, klip akan meratakan gradient — jangan memaksakan tindakan yang baik lebih jauh dari `+ε` di atas probabilitas lama.
- Jika keunggulan `A_t < 0` dan rasio mencoba melampaui `1 - ε` (artinya kita akan membuat tindakan buruk lebih mungkin terjadi dibandingkan dengan pengurangan yang terpotong), klip akan membatasi gradient — jangan mendorong tindakan buruk di bawah `-ε`.

`min` menangani arah lain: jika rasio telah bergerak ke arah *menguntungkan*, kamu masih mendapatkan gradient (tidak ada kliping di sisi yang akan merugikan kamu).

Khas `ε = 0.2`. Plot tujuan sebagai fungsi dari `r_t`: fungsi linier sepotong-sepotong dengan atap datar di "sisi baik" dan lantai datar di "sisi buruk".

**Loss PPO penuh.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

Struktur aktor-kritikus yang sama seperti A2C. Tiga koefisien, biasanya `c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`.

**Lingkaran training.**1. Kumpulkan `N × T` transisi di seluruh `N` lingkungan paralel untuk `T` langkah masing-masing.
2. Hitung keunggulan (GAE), bekukan sebagai konstanta.
3. Bekukan `π_{θ_old}` sebagai cuplikan `π_θ` saat ini.
4. Untuk `K` epoch, untuk setiap minibatch `(s, a, A, V_target, log π_old(a|s))`:
   - Hitung `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`.
   - Terapkan `L^{CLIP}` + kehilangan nilai + entropi.
   - Langkah gradient.
5. Buang peluncurannya. Kembali ke langkah 1.

`K = 10` dan minibatch 64 adalah kumpulan hyperparameter standar. PPO kuat: angka pastinya jarang menjadi masalah dalam rentang ±50%.

**Varian penalti KL.** Makalah asli mengusulkan alternatif menggunakan penalti KL adaptif: `L = L^{PG} - β · KL(π_θ || π_old)` dengan `β` disesuaikan berdasarkan KL yang diamati. Versi kliping menjadi dominan; varian KL bertahan di RLHF (di mana kebijakan referensi KL adalah batasan terpisah yang selalu kamu inginkan).

## Build

### Langkah 1: rekam `log π_old(a | s)` pada waktu peluncuran

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

Snapshot diambil satu kali, pada waktu peluncuran. Itu tidak berubah selama masa pembaruan.

### Langkah 2: menghitung keunggulan GAE (Lesson 07)

Sama seperti A2C. Normalisasikan seluruh batch.

### Langkah 3: pembaruan pengganti terpotong

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

Pola "terpotong → gradient nol" adalah inti dari PPO. Jika kebijakan baru telah menyimpang terlalu jauh ke arah yang menguntungkan, pembaruan akan berhenti.

### Langkah 4: nilai dan entropi

Tambahkan MSE standar ke target kritik dan bonus entropi pada aktor, sama seperti A2C.

### Langkah 5: diagnostik

Tiga hal yang harus diperhatikan setiap pembaruan:

- **Berarti KL** `E[log π_old - log π_θ]`. Harus tetap di `[0, 0.02]`. Jika melebihi `0.1`, kurangi `K_EPOCHS` atau `LR`.
- **Fraksi klip** — pecahan sample yang rasionya berada di luar `[1-ε, 1+ε]`. Seharusnya `~0.1-0.3`. Jika `~0`, klip tidak pernah terpicu → angkat `LR` atau `K_EPOCHS`. Jika `~0.5+`, kamu terlalu menyesuaikan peluncuran → turunkan.
- **Penjelasan varians** `1 - Var(V_target - V_pred) / Var(V_target)`. Metrik kualitas kritis. Harus naik menuju 1 saat kritikus mempelajarinya.

## Jebakan

- **Koefisien klip salah disetel.** `ε = 0.2` adalah standar de-facto. Membuka `0.1` membuat pembaruan menjadi terlalu lambat; `0.3+` mengundang ketidakstabilan.
- **Terlalu banyak periode.** `K > 20` sering kali menyebabkan ketidakstabilan karena kebijakan tersebut menyimpang jauh dari `π_old`. Batasi zaman, terutama untuk jaringan besar.
- **Tidak ada normalisasi reward.** Skala reward yang besar memakan rentang klip. Normalisasikan imbalan (menjalankan std) sebelum menghitung keuntungan.
- **Melupakan normalisasi keuntungan.** Normalisasi rata-rata nol/unit-std per batch merupakan standar. Melewatkannya akan merusak PPO pada sebagian besar tolok ukur.
- **Learning rate tidak menurun.** PPO mendapat manfaat dari peluruhan LR linier menjadi nol. LR konstan seringkali lebih buruk.
- **Kesalahan matematika rasio kepentingan.** Selalu `exp(log_new - log_old)` untuk stabilitas numerik, bukan `new / old`.
- **Tanda gradient salah.** Maksimalkan pengganti = *minimalkan* `-L^{CLIP}`. Tanda terbalik adalah bug PPO yang paling umum.

## Pakai

PPO adalah algoritme RL default tahun 2026 di sejumlah domain yang mengejutkan:| Kasus penggunaan | Varian PPO |
|----------|-------------|
| MuJoCo / kontrol robotika | PPO dengan kebijakan Gaussian, GAE(0.95) |
| Atari / permainan diskrit | PPO dengan kebijakan kategoris, meluncurkan 128 langkah |
| RLHF untuk LLM | PPO dengan penalti KL untuk model referensi, hadiah dari RM di akhir respons |
| Agen Permainan Skala Besar | IMPALA + PPO (AlphaStar, OpenAI Lima) |
| Penalaran LLM | GRPO (Lesson 12) — Varian PPO tanpa kritik |
| Data hanya preferensi | DPO — keruntuhan PPO+KL dalam bentuk tertutup, tidak ada pengambilan sample online |

*Bentuk loss* PPO — pengganti + nilai + entropi yang terpotong — adalah perancah untuk DPO, GRPO, dan hampir setiap pipeline pipa RLHF.

## Kirim

Simpan sebagai `outputs/skill-ppo-trainer.md`:

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## Latihan

1. **Mudah.** Jalankan PPO di GridWorld 4×4 dengan `ε=0.2, K=4`. Bandingkan efisiensi sample dengan A2C (satu epoch per peluncuran) pada langkah env yang cocok.
2. **Sedang.** Sapu `K ∈ {1, 4, 10, 30}`. Plot langkah pengembalian vs env dan lacak rata-rata KL per pembaruan. Pada `K` apa KL meledak dalam tugas ini?
3. **Sulit.** Ganti pengganti yang terpotong dengan penalti KL adaptif (`β` digandakan jika `KL > 2·target`, dibelah dua jika `KL < target/2`). Bandingkan pengembalian akhir, stabilitas, dan bebas klip.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Rasio pentingnya | "r_t(θ)" | `π_θ(a|s) / π_old(a|s)`; penyimpangan dari kebijakan pengumpulan data. |
| Pengganti terpotong | "Trik utama PPO" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; gradient datar melewati klip di sisi yang menguntungkan. |
| Wilayah kepercayaan | "Niat TRPO/PPO" | Batasi KL setiap pembaruan untuk menjamin peningkatan yang monoton. |
| Penalti KL | "Wilayah kepercayaan lunak" | PPO Alternatif: `L - β · KL(π_θ || π_old)`. Adaptif `β`. |
| Fraksi klip | "Seberapa sering kliping memicu" | Diagnostik — harus 0,1-0,3; luar berarti salah penyetelan. |
| Training multi-zaman | "Penggunaan kembali data" | K epoch pada setiap peluncuran; biaya varians diperdagangkan untuk efisiensi sample. |
| Sesuai kebijakan | "Sebagian besar berdasarkan kebijakan" | PPO secara nominal sesuai kebijakan tetapi periode K>1 menggunakan data yang sedikit di luar kebijakan dengan aman. |
| PPO-KL | "PPO lainnya" | varian penalti KL; digunakan di RLHF di mana referensi KL sudah menjadi kendala. |

## Bacaan Lanjutan

- [Schulman dkk. (2017). Algoritma Optimization Kebijakan Proksimal](https://arxiv.org/abs/1707.06347) — makalah.
- [Schulman dkk. (2015). Optimization Kebijakan Wilayah Kepercayaan](https://arxiv.org/abs/1502.05477) — TRPO, pendahulu PPO.
- [Andrychowicz dkk. (2021). Apa yang Penting dalam RL Sesuai Kebijakan? Studi Empiris Skala Besar](https://arxiv.org/abs/2006.05990) — setiap hyperparameter PPO dihapuskan.
- [Ouyang dkk. (2022). Melatih model bahasa untuk mengikuti instruksi dengan input manusia](https://arxiv.org/abs/2203.02155) — InstructGPT; resep PPO-in-RLHF.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — eksposisi modern yang bersih dengan PyTorch.
- [Implementasi PPO CleanRL](https://github.com/vwxyzjn/cleanrl) — referensi PPO file tunggal yang digunakan oleh banyak makalah.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — resep produksi PPO pada model bahasa; baca di samping Lesson 09 (RLHF).
- [Engstrom dkk. (2020). Pentingnya Implementasi dalam Gradient Kebijakan Mendalam](https://arxiv.org/abs/2005.12729) — makalah "37 optimization tingkat code"; trik PPO mana yang menahan weight dan mana yang merupakan cerita rakyat.
