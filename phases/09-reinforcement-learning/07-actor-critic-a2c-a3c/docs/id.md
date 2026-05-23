# Aktor-Kritikus — A2C dan A3C

> REINFORCE berisik. Tambahkan kritik yang mempelajari `V̂(s)`, kurangi dari keuntungannya, dan kamu mendapatkan keuntungan yang memiliki ekspektasi yang sama tetapi variansinya jauh lebih rendah. Itu adalah aktor-kritikus. A2C menjalankannya secara serempak; A3C menjalankannya di seluruh thread. Keduanya adalah model mental untuk setiap metode deep-RL modern.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 9 · 04 (Pembelajaran TD), Fase 9 · 06 (PERKUAT)
**Waktu:** ~75 menit

## Masalah

Vanilla REINFORCE berfungsi, tetapi variansnya sangat buruk. Pengembalian Monte Carlo `G_t` dapat mencapai faktor 10 antar episode. Mengalikan gangguan tersebut dengan `∇ log π` dan membuat rata-rata menghasilkan penaksir gradient yang membutuhkan ribuan episode untuk memindahkan kebijakan pada distance yang sama dengan yang kamu dapat memindahkannya dengan pembaruan DQN yang jauh lebih sedikit.

Variansnya berasal dari penggunaan pengembalian mentah. Jika kamu mengurangi garis dasar `b(s_t)` — fungsi status apa pun, termasuk nilai yang dipelajari — ekspektasinya tidak berubah dan variansnya turun. Garis dasar terbaik yang dapat ditelusuri adalah `V̂(s_t)`. Sekarang kuantitas yang dikalikan `∇ log π` adalah *keuntungan*:

`A(s, a) = G - V̂(s)`

Suatu tindakan dianggap baik jika menghasilkan keuntungan di atas rata-rata; buruk jika di bawah. MEMPERKUAT dengan kritikus terpelajar adalah *aktor-kritikus*. Kritikus memberikan aktor tersebut guru dengan variansi rendah. Ini adalah setiap metode kebijakan mendalam setelah tahun 2015 (A2C, A3C, PPO, SAC, IMPALA).

## Konsep

![Aktor-kritikus: kebijakan bersih ditambah nilai bersih, sisa TD sebagai keuntungan](../assets/actor-critic.svg)

**Dua jaringan, satu loss bersama:**

- **Aktor** `π_θ(a | s)`: kebijakan. Diambil sampelnya untuk bertindak. Dilatih dengan gradient kebijakan.
- **Kritik** `V_φ(s)`: memperkirakan pengembalian yang diharapkan dari negara bagian. Dilatih untuk meminimalkan `(V_φ(s) - target)²`.

**Keuntungannya.** Dua bentuk standar:

- *Keuntungan MC:* `A_t = G_t - V_φ(s_t)`. Tidak bias, varians lebih tinggi.
- *Keuntungan TD:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`. Bias (menggunakan `V_φ`), variansnya jauh lebih rendah. Disebut juga *sisa TD* `δ_t`.

**keuntungan n-langkah.** Interpolasi di antara keduanya:

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` adalah TD murni. `n = ∞` adalah MC. Sebagian besar implementasi menggunakan `n = 5` untuk Atari, `n = 2048` untuk PPO di MuJoCo.

**Estimasi Keuntungan Umum (GAE).** Schulman dkk. (2016) mengusulkan rata-rata tertimbang secara eksponensial untuk semua keuntungan n-langkah:

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

dengan `λ ∈ [0, 1]`. `λ = 0` adalah TD (varians rendah, bias tinggi). `λ = 1` adalah MC (varians tinggi, tidak bias). `λ = 0.95` adalah default tahun 2026 — sesuaikan hingga putaran bias/varians berada di tempat yang kamu inginkan.

**A2C: aktor-kritikus keunggulan sinkron.** Kumpulkan `T` langkah-langkah di lingkungan paralel `N`. Hitung keuntungan untuk setiap langkah. Perbarui aktor dan kritikus pada kumpulan gabungan. Mengulang. Saudara A3C yang lebih sederhana dan lebih skalabel.

**A3C: aktor-kritikus keunggulan asinkron.** Mnih dkk. (2016). Memunculkan `N` thread pekerja, masing-masing menjalankan env. Setiap pekerja menghitung gradient secara lokal pada peluncurannya sendiri, lalu menerapkannya secara asinkron ke server parameter bersama. Tidak diperlukan buffer pemutaran ulang — pekerja melakukan dekorasi dengan menjalankan lintasan yang berbeda. A3C membuktikan bahwa kamu dapat berlatih menggunakan CPU dalam skala besar. Pada tahun 2026, A2C (batched parallel envs) berbasis GPU mendominasi karena GPU menginginkan batch yang besar.

**Loss gabungan.**`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

Tiga istilah: loss gradient kebijakan, regresi nilai, bonus entropi. `c_v ~ 0.5`, `c_e ~ 0.01` adalah titik awal kanonik.

## Build

### Langkah 1: seorang kritikus

Kritikus linier `V_φ(s) = w · features(s)` diperbarui dengan MSE:

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

Dalam tabel, kritikus berkumpul dalam beberapa ratus episode. Di Atari, ganti kritik linier dengan trunk CNN + value head bersama.

### Langkah 2: keuntungan n-langkah

Mengingat peluncuran durasi `T` dan final bootstrap `V(s_T)`:

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` adalah sasaran kritik. `advantages` itulah yang mengalikan `∇ log π`.

### Langkah 3: pembaruan gabungan

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

Sesuai kebijakan, satu peluncuran per pembaruan, tingkat pembelajaran terpisah untuk aktor dan kritikus.

### Langkah 4: paralelisasi (A3C vs A2C)

- **A3C:** memutar `N` thread. Masing-masing menjalankan lingkungannya sendiri dan umpan depannya sendiri. Dorong pembaruan gradient secara berkala ke master bersama. Tidak ada kunci pada master — balapan baik-baik saja, hanya menambah kebisingan.
- **A2C:** jalankan instance `N` env dalam satu proses, susun pengamatan ke dalam batch `[N, obs_dim]`, forward pass batch, backward pass batch. Pemanfaatan GPU lebih tinggi, deterministik, lebih mudah untuk dipikirkan. Defaultnya pada tahun 2026.

Code mainan kami berulir tunggal untuk kejelasan; menulis ulang ke batch A2C adalah tiga baris numpy.

## Jebakan

- **Bias kritik sebelum gradient aktor.** Jika kritiknya acak, garis dasarnya tidak informatif dan kamu berlatih dengan noise murni. Lakukan pemanasan selama beberapa ratus langkah sebelum menerapkan gradient kebijakan, atau gunakan learning rate aktor yang lambat.
- **Normalisasi keunggulan.** Normalisasi keunggulan ke rata-rata nol/std unit per batch. Menstabilkan training secara besar-besaran dengan biaya hampir nol.
- **Batang bersama.** Gunakan ekstraktor feature bersama untuk aktor dan kritikus pada input gambar. Pisahkan kepala. Feature bersama tumpangan gratis di kedua loss.
- **Kontrak sesuai kebijakan.** A2C menggunakan kembali data untuk satu pembaruan. Semakin banyak gradient kamu menjadi bias (koreksi pengambilan sample penting adalah apa yang ditambahkan PPO).
- **Keruntuhan entropi.** Tanpa `c_e > 0`, kebijakan menjadi hampir deterministik dalam beberapa ratus pembaruan dan berhenti dieksplorasi.
- **Skala imbalan.** Besaran keuntungan bergantung pada skala imbalan. Normalisasikan imbalan (misalnya, pembagian running-std) untuk besaran gradient yang konsisten di seluruh tugas.

## Pakai

A2C/A3C jarang menjadi pilihan akhir pada tahun 2026 tetapi mereka adalah arsitektur yang kemudian disempurnakan:

| Metode | Kaitannya dengan A2C |
|--------|----------------|
| PPO | Rasio kepentingan A2C + terpotong untuk pembaruan multi-zaman |
| IMPALA | Koreksi di luar kebijakan A3C + V-trace |
| SAC (Fase 9 · 07) | A2C di luar kebijakan dengan kritik yang bersifat lunak (lesson berikutnya) |
| GRPO (Phase 9 · 12) | A2C tanpa kritik — keunggulan relatif kelompok |
| DPO | A2C jatuh ke dalam loss peringkat preferensi, tidak ada pengambilan sample |
| AlphaStar / OpenAI Lima | A2C dengan training liga + pra-training imitasi |

Jika kamu melihat "keuntungan" dalam makalah tahun 2026, pikirkanlah aktor-kritikus.

## Kirim

Simpan sebagai `outputs/skill-actor-critic-trainer.md`:

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## Latihan1. **Mudah.** Latih aktor-kritikus dengan keunggulan MC (`G_t - V(s_t)`) di GridWorld 4×4. Bandingkan efisiensi sample dengan REINFORCE-with-running-mean-baseline dari Lesson 06.
2. **Sedang.** Beralih ke keunggulan sisa TD (`r + γ V(s') - V(s)`). Ukur varians dari kumpulan keuntungan. Berapa penurunannya?
3. **Sulit.** Menerapkan GAE(λ). Sapu `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`. Plot pengembalian akhir vs efisiensi sample. Di manakah titik manis bias/varians untuk tugas ini?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Aktor | "Jaring kebijakan" | `π_θ(a|s)`, diperbarui berdasarkan gradient kebijakan. |
| Kritikus | "Nilai bersih" | `V_φ(s)`, diperbarui oleh regresi UMK ke target pengembalian / TD. |
| Keuntungan | "Jauh lebih baik dari rata-rata" | `A(s, a) = Q(s, a) - V(s)` atau estimatornya. Pengganda untuk `∇ log π`. |
| sisa TD | "δ" | `δ_t = r + γ V(s') - V(s)`; perkiraan keuntungan satu langkah. |
| GAE | "Tombol interpolasi" | Jumlah keunggulan n-langkah yang ditimbang secara eksponensial, diparameterisasi dengan `λ`. |
| A2C | "Aktor-kritikus sinkron" | Dikumpulkan di seluruh envs; satu langkah gradient per peluncuran. |
| A3C | "Aktor-kritikus async" | Utas pekerja mendorong gradient ke server param bersama. kertas asli; kurang umum pada tahun 2026. |
| tali sepatu | "Gunakan V di cakrawala" | Pangkas peluncurannya, tambahkan `γ^n V(s_{t+n})` untuk menutup jumlahnya. |

## Bacaan Lanjutan

- [Mnih dkk. (2016). Metode Asinkron untuk Pembelajaran Penguatan Mendalam](https://arxiv.org/abs/1602.01783) — A3C, makalah aktor-kritikus asinkron yang asli.
- [Schulman dkk. (2016). Kontrol Berkelanjutan Dimension Tinggi Menggunakan Estimasi Keuntungan Umum](https://arxiv.org/abs/1506.02438) — GAE.
- [Sutton & Barto (2018). Bab. 13 — Metode Aktor-Kritikus](http://incompleteideas.net/book/RLbook2020.pdf) — fondasi; pasangkan ini dengan Ch. 9 tentang perkiraan fungsi ketika kritiknya adalah neural network.
- [Espeholt dkk. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — aktor-kritikus terdistribusi yang skalabel dengan koreksi di luar kebijakan V-trace.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — implementasi produksi A2C/PPO layak dibaca.
- [Konda & Tsitsiklis (2000). Algoritma Aktor-Kritikus](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — hasil konvergensi dasar untuk decomposition aktor-kritikus dua skala waktu.
