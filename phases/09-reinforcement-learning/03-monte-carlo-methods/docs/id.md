# Metode Monte Carlo — Belajar dari Episode Lengkap

> Pemrograman dinamis membutuhkan model. Monte Carlo tidak membutuhkan apa pun selain episode. Jalankan kebijakan, perhatikan keuntungannya, buat rata-ratanya. Ide paling sederhana di RL — dan ide yang membuka segala hal di hilir.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 9 · 01 (MDP), Fase 9 · 02 (Pemrograman Dinamis)
**Waktu:** ~75 menit

## Masalah

Pemrograman dinamis memang elegan, tetapi mengasumsikan kamu dapat menanyakan `P(s' | s, a)` untuk setiap status dan tindakan. Hampir tidak ada hal seperti itu di dunia nyata. Robot tidak dapat secara analitis menghitung distribusi piksel kamera setelah torsi gabungan. Algoritme penetapan harga tidak dapat mengintegrasikan setiap kemungkinan reaksi pelanggan. LLM tidak dapat menghitung semua kemungkinan kelanjutan setelah token.

kamu memerlukan metode yang hanya membutuhkan kemampuan *mengambil sample* dari lingkungan. Jalankan kebijakannya. Dapatkan lintasan `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`. Gunakan untuk memperkirakan nilai. Itu adalah Monte Carlo.

Peralihan dari DP ke MC secara filosofis penting: kita beralih dari *model yang diketahui + cadangan yang tepat* ke *peluncuran sample + pengembalian rata-rata*. Variansnya melonjak, tetapi penerapannya meledak. Setiap algoritme RL setelah lesson ini - TD, Q-learning, REINFORCE, PPO, GRPO - pada intinya adalah estimator Monte Carlo, terkadang dengan bootstrapping berlapis di atasnya.

## Konsep

![Monte Carlo: peluncuran, penghitungan keuntungan, rata-rata; kunjungan pertama vs setiap kunjungan](../assets/monte-carlo.svg)

**Ide inti, dalam satu baris:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` di mana `G^{(i)}(s)` diamati kembali setelah kunjungan ke `s` berdasarkan kebijakan `π`.

**MC kunjungan pertama vs setiap kunjungan.** Mengingat episode yang mengunjungi negara bagian `s` beberapa kali, MC kunjungan pertama hanya menghitung pengembalian dari kunjungan pertama; setiap kunjungan MC menghitung semua kunjungan. Keduanya tidak memihak dalam batasannya. Kunjungan pertama lebih mudah untuk dianalisis (sample iid). Setiap kunjungan menggunakan lebih banyak data per episode dan biasanya menyatu lebih cepat dalam praktiknya.

**Rata-rata tambahan.** Daripada menyimpan semua pengembalian, perbarui rata-rata berjalan:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

Atur ulang: `V_new = V_old + α · (target - V_old)` dengan `α = 1/n`. Tukar `1/n` dengan ukuran langkah konstan `α ∈ (0, 1)` dan kamu mendapatkan estimator MC non-stasioner yang melacak perubahan di `π`. Perpindahan tersebut merupakan lompatan keseluruhan dari MC ke TD ke setiap algoritma RL modern.

**Eksplorasi kini menjadi masalah.** DP menyentuh setiap negara bagian dengan pencacahan. MC hanya melihat kunjungan kebijakan negara bagian. Jika `π` bersifat deterministik, seluruh wilayah negara bagian tidak akan pernah diambil sampelnya, dan perkiraan nilainya tetap nol selamanya. Tiga perbaikan, dalam urutan historis:

1. **Penjelajahan dimulai.** Mulai setiap episode dari pasangan acak (s, a). Menjamin cakupan; tidak realistis dalam praktiknya (kamu tidak dapat "mengatur ulang" robot ke keadaan sewenang-wenang).
2. **ε-serakah.** Bertindak serakah w.r.t. Q saat ini, tetapi dengan kemungkinan `ε` memilih tindakan acak. Semua pasangan tindakan negara diambil sampelnya secara asimtotik.
3. **MC di luar kebijakan.** Kumpulkan data berdasarkan kebijakan perilaku `μ`, pelajari tentang kebijakan target `π` melalui pengambilan sample kepentingan. Variansnya tinggi, tetapi ini merupakan jembatan ke metode replay-buffer seperti DQN.

**Kontrol Monte Carlo.** Evaluasi → tingkatkan → evaluasi, sama seperti iterasi kebijakan, namun evaluasi didasarkan pada pengambilan sample:1. Jalankan `π`, dapatkan sebuah episode.
2. Perbarui `Q(s, a)` dari pengembalian yang diamati.
3. Jadikan `π` ε-greedy w.r.t. `Q`.
4. Ulangi.

Menyatu ke `Q*` dan `π*` dengan probabilitas 1 dalam kondisi ringan (setiap pasangan sering dikunjungi, `α` memuaskan Robbins-Monro).

## Build

### Langkah 1: peluncuran → daftar (s, a, r)

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

Tidak ada model, hanya `env.reset()` dan `env.step(s, a)`. Antarmuka yang sama seperti lingkungan gym tetapi dipreteli.

### Langkah 2: menghitung pengembalian (sapuan terbalik)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

Satu tiket, `O(T)`. Pengulangan mundur `G_t = r_{t+1} + γ G_{t+1}` menghindari penjumlahan ulang.

### Langkah 3: evaluasi MC kunjungan pertama

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

Tiga baris berfungsi: tandai status seperti yang terlihat pada kunjungan pertama, jumlah kenaikan, perbarui rata-rata yang berjalan.

### Langkah 4: Kontrol MC yang serakah (sesuai kebijakan)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Langkah 5: bandingkan dengan DP gold standar

Perkiraan MC kamu `V^π` harus sesuai dengan hasil DP dari Lesson 02 sebagai episode → ∞. Dalam praktiknya: 50.000 episode di 4×4 GridWorld membuat kamu berada dalam `~0.1` dari jawaban DP.

## Jebakan

- **Episode tak terbatas.** MC memerlukan episode untuk *berakhir*. Jika kebijakan kamu dapat diulang selamanya, batasi `max_steps` dan perlakukan batas tersebut sebagai kegagalan implisit. GridWorld dengan kebijakan acak secara rutin kehabisan waktu — itu normal, pastikan kamu menghitungnya dengan benar.
- **Varians.** MC menggunakan pengembalian penuh. Pada episode yang panjang, perbedaannya sangat besar — ​​satu hadiah sial di akhir akan menggeser `V(s_0)` dengan jumlah yang sama. Metode TD (Lesson 04) memotongnya dengan bootstrapping.
- **Liputan negara.** MC serakah di Q baru dengan ikatan hanya akan mencoba satu tindakan. kamu *harus* menjelajah (ε-serakah, penjelajahan dimulai, UCB).
- **Kebijakan non-stasioner.** Jika `π` berubah (seperti pada kontrol MC), pengembalian lama berasal dari kebijakan yang berbeda. Constant-α MC menangani ini; MC rata-rata sample tidak.
- **Pengambilan sample kepentingan di luar kebijakan.** Weight `π(a|s)/μ(a|s)` dikalikan pada seluruh lintasan. Varians meledak dengan cakrawala. Batasi dengan IS tertimbang per keputusan atau alihkan ke TD.

## Pakai

Peran metode Monte Carlo tahun 2026:

| Kasus penggunaan | Mengapa MC |
|----------|--------|
| Permainan cakrawala pendek (blackjack, poker) | Episode berakhir secara alami; pengembaliannya bersih. |
| Evaluasi offline terhadap kebijakan yang dicatat | Pengembalian diskon rata-rata atas lintasan yang disimpan. |
| Pencarian Pohon Monte Carlo (AlphaZero) | Peluncuran MC dari pemilihan panduan daun pohon. |
| Evaluasi LLM RL | Hitung imbalan rata-rata atas penyelesaian sample untuk kebijakan tertentu. |
| Estimasi dasar dalam PPO | Target keunggulan `A_t = G_t - V(s_t)` menggunakan MC `G_t`. |
| Mengajar RL | Algoritme paling sederhana yang benar-benar berfungsi — hapus bootstrapping untuk melihat intinya. |

Algoritme deep-RL modern (PPO, SAC) melakukan interpolasi antara MC murni (pengembalian penuh) dan TD murni (bootstrap satu langkah) melalui `n`-pengembalian langkah atau GAE. Kedua titik akhir adalah turunan dari estimator yang sama.

## Kirim

Simpan sebagai `outputs/skill-mc-evaluator.md`:

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## Latihan1. **Mudah.** Menerapkan evaluasi MC kunjungan pertama terhadap kebijakan seragam-acak di GridWorld 4×4. Jalankan 10.000 episode. Plot `V(0,0)` sebagai fungsi jumlah episode terhadap jawaban DP.
2. **Sedang.** Terapkan kontrol MC ε-serakah dengan `ε ∈ {0.01, 0.1, 0.3}`. Bandingkan pengembalian rata-rata setelah 20.000 episode. Seperti apa kurvanya? Di manakah letak trade-off bias-varians?
3. **Sulit.** Menerapkan MC *di luar kebijakan* dengan pengambilan sample penting: mengumpulkan data berdasarkan kebijakan acak seragam `μ`, memperkirakan `V^π` untuk kebijakan optimal deterministik `π`. Bandingkan IS biasa vs IS per keputusan vs IS tertimbang. Manakah yang memiliki varian terendah?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Monte Carlo | "Pengambilan sample acak" | Perkirakan ekspektasi dengan merata-ratakan sample iid dari distribusi. |
| Kembalikan `G_t` | "Hadiah masa depan" | Jumlah hadiah yang didiskon dari langkah `t` hingga akhir episode: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| MC kunjungan pertama | "Hitung setiap negara bagian satu kali" | Hanya kunjungan pertama dalam sebuah episode yang berkontribusi terhadap perkiraan nilai. |
| MC setiap kunjungan | "Gunakan semua kunjungan" | Setiap kunjungan berkontribusi; sedikit bias tetapi lebih efisien dalam pengambilan sample. |
| ε-serakah | "Kebisingan eksplorasi" | Pilih tindakan serakah dengan masalah `1-ε`; tindakan acak dengan masalah `ε`. |
| Pentingnya pengambilan sample | "Mengoreksi pengambilan sample dari distribusi yang salah" | Menimbang ulang pengembalian produk `π(a|s)/μ(a|s)` untuk memperkirakan `V^π` dari data `μ`. |
| Sesuai kebijakan | "Belajar dari data saya sendiri" | Kebijakan sasaran = kebijakan perilaku. Vanila MC, PPO, SARSA. |
| Di luar kebijakan | "Belajar dari data orang lain" | Kebijakan sasaran ≠ kebijakan perilaku. MC sample penting, Q-learning, DQN. |

## Bacaan Lanjutan

- [Sutton & Barto (2018). Bab. 5 — Metode Monte Carlo](http://incompleteideas.net/book/RLbook2020.pdf) — perlakuan kanonik.
- [Singh & Sutton (1996). Pembelajaran Penguatan dengan Mengganti Jejak Kelayakan](https://link.springer.com/article/10.1007/BF00114726) — analisis kunjungan pertama vs setiap kunjungan.
- [Precup, Sutton, Singh (2000). Jejak Kelayakan untuk Evaluasi Kebijakan di Luar Kebijakan](http://incompleteideas.net/papers/PSS-00.pdf) — MC di luar kebijakan dan pengendalian varians.
- [Mahmood dkk. (2014). Pengambilan Sample Kepentingan Tertimbang untuk Pembelajaran di Luar Kebijakan](https://arxiv.org/abs/1404.6362) — penaksir IS varian rendah modern.
- [Tesauro (1995). TD-Gammon, Program Backgammon Mengajar Mandiri](https://dl.acm.org/doi/10.1145/203330.203343) — demonstrasi empiris skala besar pertama dari permainan mandiri MC/TD yang menyatu dengan permainan manusia super; pendahulu konseptual untuk setiap lesson di paruh kedua fase ini.
