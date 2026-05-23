# Gradient Kebijakan — MEMPERKUAT dari Awal

> Berhenti memperkirakan nilai. Parameterisasikan kebijakan secara langsung, hitung gradient keuntungan yang diharapkan, selangkah lebih maju. Williams (1992) menuliskannya dalam satu teorema. Itu sebabnya PPO, GRPO, dan setiap loop LLM RL ada.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 3 · 03 (Propagasi Balik), Fase 9 · 03 (Monte Carlo), Fase 9 · 04 (Pembelajaran TD)
**Waktu:** ~75 menit

## Masalah

Q-learning dan DQN membuat parameter fungsi *nilai*. kamu memilih tindakan berdasarkan `argmax Q`. Itu bagus untuk tindakan terpisah dan status terpisah. Itu rusak ketika tindakan terus menerus (yang `argmax` lebih dari torsi 10 dimension?) atau ketika kamu menginginkan kebijakan stokastik (`argmax` bersifat deterministik berdasarkan konstruksi).

Gradient kebijakan malah membuat parameter *kebijakan*. `π_θ(a | s)` adalah neural network yang menghasilkan distribusi tindakan. Contoh dari itu untuk bertindak. Hitung gradient pengembalian yang diharapkan sehubungan dengan `θ`. Melangkah menanjak. Tidak `argmax`. Tidak ada rekursi Bellman. Hanya pendakian gradient di `J(θ) = E_{π_θ}[G]`.

Teorema REINFORCE (Williams 1992) memberi tahu kamu bahwa gradient ini dapat dihitung: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. Jalankan sebuah episode. Hitung pengembaliannya. Kalikan dengan `∇ log π_θ(a | s)` di setiap langkah. Rata-rata. Pendakian gradient. Selesai.

Setiap algoritma LLM-RL pada tahun 2026 — PPO, DPO, GRPO — merupakan penyempurnaan dari REINFORCE. Memahaminya dengan jari kamu adalah prasyarat untuk sisa fase ini, dan untuk Fase 10 · 07 (implementasi RLHF) dan Fase 10 · 08 (DPO).

## Konsep

![Gradient kebijakan: kebijakan softmax, gradient log-π, pembaruan berbobot pengembalian](../assets/policy-gradient.svg)

**Teorema gradient kebijakan.** Untuk kebijakan apa pun `π_θ` yang diparameterisasi oleh `θ`:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

dimana `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` adalah potongan pengembalian dari langkah `t`. Harapannya melebihi lintasan penuh `τ` yang diambil sampelnya dari `π_θ`.

**Buktinya singkat.** Bedakan `J(θ) = Σ_τ P(τ; θ) G(τ)` dengan ekspektasi. Gunakan `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (trik turunan log). Faktorkan `log P(τ; θ) = Σ log π_θ(a_t | s_t) + environment terms that do not depend on θ`. Istilah lingkungan lenyap. Dua baris aljabar memberi kamu teorema.

**Trik pengurangan varians.** Vanilla REINFORCE memiliki varian yang mematikan — pengembaliannya berisik, `∇ log π` berisik, produk mereka sangat berisik. Dua perbaikan standar:

1. **Pengurangan garis dasar.** Ganti `G_t` dengan `G_t - b(s_t)` untuk garis dasar apa pun `b(s_t)` yang tidak bergantung pada `a_t`. Tidak memihak karena `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`. Pilihan umum: `b(s_t) = V̂(s_t)` dipelajari oleh seorang kritikus → aktor-kritikus (Lesson 07).
2. **Reward-to-go-go.** Ganti `Σ_t G_t · ∇ log π_θ(a_t | s_t)` dengan `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`. Hanya keuntungan di masa depan yang penting untuk tindakan tertentu — imbalan di masa lalu tidak memberikan kontribusi yang berarti.

Jika digabungkan, kamu mendapatkan:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

yang merupakan PERKUAT dengan garis dasar — nenek moyang langsung dari A2C (Lesson 07) dan PPO (Lesson 08).

**Parameterisasi kebijakan Softmax.** Untuk tindakan terpisah, pilihan standarnya:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

di mana `f_θ` adalah neural network apa pun yang menghasilkan skor per tindakan. Gradient memiliki bentuk yang bersih:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

yaitu skor tindakan yang diambil dikurangi nilai yang diharapkan berdasarkan kebijakan.**Kebijakan Gaussian untuk tindakan berkelanjutan.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)` memiliki formulir tertutup. Itu saja yang dibutuhkan SAC Fase 9 · 07.

## Build

### Langkah 1: jaringan kebijakan softmax

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

Gunakan kebijakan linier (satu vector weight per tindakan) untuk env tabel. Untuk Atari, tukar CNN dan pertahankan softmax head.

### Langkah 2: pengambilan sample dan probabilitas log

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Langkah 3: peluncuran dengan masalah log ditangkap

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Langkah 4: MEMPERKUAT pembaruan

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

Gradient `∇ log π(a|s) = e_a - π(·|s)` (satu titik dari `a` dikurangi probabilitas) adalah inti dari gradient kebijakan softmax. Membakarnya ke dalam memori otot.

### Langkah 5: garis dasar

Rata-rata `G` selama episode terbaru adalah pengurangan varians yang cukup untuk menjalankan GridWorld 4×4; dibutuhkan ~500 episode untuk berkumpul. Tingkatkan dasar ke `V̂(s)` yang terpelajar dan kamu akan mendapatkan aktor-kritikus.

## Jebakan

- **Meledaknya gradient.** Keuntungannya bisa sangat besar. Selalu normalkan `G` menjadi `~N(0, 1)` di seluruh batch sebelum mengalikannya dengan `∇ log π`.
- **Keruntuhan entropi.** Kebijakan ini terlalu dini menyatu dengan tindakan yang hampir deterministik, berhenti melakukan eksplorasi, dan terhenti. Cara mengatasinya: tambahkan bonus entropi `β · H(π(·|s))` ke tujuan.
- **Varian tinggi.** Vanilla REINFORCE membutuhkan ribuan episode. Garis dasar kritik (Lesson 07) atau wilayah kepercayaan TRPO/PPO (Lesson 08) adalah perbaikan standarnya.
- **Contoh inefisiensi.** Sesuai kebijakan berarti kamu membuang setiap transisi setelah satu pembaruan. Koreksi di luar kebijakan melalui pengambilan sample kepentingan mengembalikan data, dengan mengorbankan varians (rasio PPO adalah weight IS yang terpotong).
- **Gradient non-stasioner.** Gradient yang sama dari 100 episode lalu menggunakan `π` yang lama. Metode sesuai kebijakan diperbarui setiap beberapa peluncuran karena alasan ini.
- **Penugasan kredit.** Tanpa reward-to-go, reward di masa lalu akan menimbulkan kebisingan. Selalu gunakan reward-to-go.

## Pakai

Pada tahun 2026, REINFORCE jarang dijalankan secara langsung tetapi rumus gradiennya ada di mana-mana:

| Kasus penggunaan | Metode turunan |
|----------|---------------|
| Kontrol berkelanjutan | PPO/SAC dengan kebijakan Gaussian |
| LLM RLHF | PPO dengan penalti KL, berjalan berdasarkan kebijakan tingkat token |
| Penalaran LLM (DeepSeek) | GRPO — MEMPERKUAT dengan dasar kelompok-relatif, tidak ada kritik |
| Multi-agen | REINFORCE kritik terpusat (MADDPG, COMA) |
| Robotika aksi diskrit | A2C, A3C, PPO |
| Pengaturan hanya preferensi | DPO — REINFORCE ditulis ulang sebagai kehilangan kemungkinan preferensi, tanpa pengambilan sample |

Saat kamu membaca `loss = -advantage * log_prob` dalam skrip training tahun 2026, itu adalah PERKUAT dengan garis dasar. Seluruh makalah (DPO, GRPO, RLOO) adalah trik pengurangan varians di atas satu baris ini.

## Kirim

Simpan sebagai `outputs/skill-policy-gradient-trainer.md`:

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## Latihan

1. **Mudah.** Menerapkan REINFORCE di GridWorld 4×4 dengan kebijakan softmax linier. Berlatih untuk 1.000 episode tanpa garis dasar. Plot kurva pembelajaran; mengukur varians (std pengembalian).
2. **Sedang.** Tambahkan garis dasar rata-rata berjalan. Berlatih lagi. Bandingkan efisiensi dan varians sample dengan proses vanilla. Seberapa besar penurunan garis dasar (baseline) terhadap langkah-langkah konvergensi?
3. **Sulit.** Tambahkan bonus entropi `β · H(π)`. Sapu `β ∈ {0, 0.01, 0.1, 1.0}`. Plot keuntungan akhir dan entropi kebijakan. Di manakah titik terbaik dari tugas ini?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Gradient kebijakan | "Latih kebijakan secara langsung" | `∇J(θ) = E[G · ∇ log π_θ(a|s)]`; berasal dari trik turunan log. |
| MEMPERKUAT | "Algoritma PG asli" | Williams (1992); Pengembalian Monte Carlo dikalikan dengan gradient kebijakan log. |
| Trik turunan log | "Penaksir fungsi skor" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; membuat gradient ekspektasi menjadi mudah diatur. |
| Dasar | "Pengurangan varians" | Setiap `b(s)` yang dikurangi dari `G`; tidak memihak karena `E[b · ∇ log π] = 0`. |
| Hadiah untuk dibawa pulang | "Hanya pengembalian di masa depan yang dihitung" | `G_t^{from t}` bukan `G_0` lengkap; benar dan variansinya lebih rendah. |
| Bonus entropi | "Mendorong eksplorasi" | `+β · H(π(·|s))` jangka waktu menjaga kebijakan agar tidak runtuh. |
| Sesuai kebijakan | "Latihlah apa yang baru saja kamu lihat" | Ekspektasi gradient adalah hal yang buruk. kebijakan saat ini — tidak dapat menggunakan kembali data lama secara langsung. |
| Keuntungan | "Jauh lebih baik dari rata-rata" | `A(s, a) = G(s, a) - V(s)`; kuantitas yang ditandatangani REINFORCE-with-baseline berlipat ganda. |

## Bacaan Lanjutan

- [Williams (1992). Algoritma Mengikuti Gradient Statistik Sederhana untuk Pembelajaran Penguatan Koneksionis](https://link.springer.com/article/10.1007/BF00992696) - makalah REINFORCE asli.
- [Sutton dkk. (2000). Metode Gradient Kebijakan untuk Pembelajaran Penguatan dengan Pendekatan Fungsi](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — teorema gradient kebijakan modern dengan perkiraan fungsi.
- [Sutton & Barto (2018). Bab. 13 — Metode Gradient Kebijakan](http://incompleteideas.net/book/RLbook2020.pdf) — presentasi buku teks.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — eksposisi pedagogis yang jelas dengan code PyTorch.
- [Peters & Schaal (2008). Pembelajaran Penguatan Keterampilan Motorik dengan Gradient Kebijakan](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — pengurangan varians dan pandangan gradient alami yang menghubungkan REINFORCE dengan keluarga wilayah kepercayaan (TRPO, PPO).
