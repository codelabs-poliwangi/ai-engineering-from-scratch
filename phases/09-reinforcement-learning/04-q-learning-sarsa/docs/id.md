# Perbedaan Temporal — Q-Learning & SARSA

> Monte Carlo menunggu hingga episode berakhir. TD diperbarui setelah setiap langkah dengan melakukan bootstrap pada estimasi nilai berikutnya. Q-learning di luar kebijakan dan optimis; SARSA sesuai kebijakan dan berhati-hati. Keduanya adalah satu baris code. Keduanya mendukung setiap metode deep-RL dalam fase ini.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 9 · 01 (MDP), Fase 9 · 02 (Pemrograman Dinamis), Fase 9 · 03 (Monte Carlo)
**Waktu:** ~75 menit

## Masalah

Monte Carlo berfungsi tetapi memiliki dua permintaan yang mahal. Dibutuhkan episode yang berakhir, dan hanya diperbarui setelah pengembalian terakhir masuk. Jika episode kamu terdiri dari 1.000 langkah, MC menunggu 1.000 langkah untuk memperbarui apa pun. Ini memiliki variansi tinggi, bias rendah, dan praktiknya lambat.

Pemrograman dinamis memiliki profil yang berlawanan — pencadangan bootstrap dengan varian nol — tetapi memerlukan model yang diketahui.

Pembelajaran perbedaan temporal (TD) membagi perbedaan. Dari transisi tunggal `(s, a, r, s')`, bentuklah target satu langkah `r + γ V(s')` dan dorong `V(s)` ke arah transisi tersebut. Tidak ada modelnya. Tidak ada episode lengkap. Bias dari penggunaan perkiraan `V` di RHS, namun variansnya jauh lebih rendah dibandingkan MC dan pembaruan online dari langkah pertama.

Ini adalah poros di mana semua RL modern — DQN, A2C, PPO, SAC — berputar. Phase 9 selanjutnya adalah layer perkiraan fungsi dan trik yang dibangun di atas pembaruan TD satu langkah yang akan kamu tulis dalam lesson ini.

## Konsep

![Q-learning vs SARSA: maks di luar kebijakan vs Q(s', a') di luar kebijakan](../assets/td.svg)

**Pembaruan TD(0) untuk V:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

Kuantitas dalam tanda kurung adalah kesalahan TD `δ = r + γ V(s') - V(s)`. Ini adalah analog online `G_t - V(s_t)` di MC. Konvergensi membutuhkan `α` memuaskan Robbins-Monro (`Σ α = ∞`, `Σ α² < ∞`) dan semua negara bagian sering dikunjungi.

**Q-learning.** Metode TD di luar kebijakan untuk pengendalian:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` mengasumsikan kebijakan *serakah* akan diikuti mulai dari `s'` dan seterusnya, terlepas dari tindakan apa yang sebenarnya diambil oleh agen. Pemisahan tersebut membuat Q-learning belajar `Q*` sementara agen menjelajah melalui ε-greedy. Mnih dkk. (2015) mengubahnya menjadi pembelajaran Q mendalam di Atari (Lesson 05).

**SARSA.** Metode TD sesuai kebijakan:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

Namanya tupel `(s, a, r, s', a')`. SARSA menggunakan tindakan `a'` yang *sebenarnya* diambil agen selanjutnya, bukan `argmax` yang serakah. Menyatu ke `Q^π` untuk apa pun ε-greedy `π` yang sedang berjalan, yang dalam batas `ε → 0` menjadi `Q*`.

**Perbedaan berjalan di tebing.** Pada tugas berjalan di tebing klasik (fall-off-cliff = reward -100), Q-learning mempelajari jalur optimal di sepanjang tepi tebing, namun terkadang menerima penalti selama eksplorasi. SARSA mempelajari jalur yang lebih aman dengan selangkah menjauh dari tebing karena ia memperhitungkan kebisingan eksplorasi ke dalam nilai Q-nya. Dengan training, keduanya mencapai hasil optimal di `ε → 0`. Dalam praktiknya, hal ini penting: ketika eksplorasi benar-benar dilakukan pada saat penempatan, perilaku SARSA lebih konservatif.

**SARSA yang diharapkan.** Ganti `Q(s', a')` dengan nilai yang diharapkan pada `π`:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

Varians lebih rendah dari SARSA (tidak ada sample `a'`), target kebijakan yang sama. Seringkali menjadi default dalam buku teks modern.**n-langkah TD dan TD(λ).** Interpolasi antara TD(0) dan MC dengan menunggu `n` langkah sebelum melakukan bootstrap. `n=1` adalah TD, `n=∞` adalah MC. Rata-rata TD(λ) di seluruh `n` dengan weight geometris `(1-λ)λ^{n-1}`. Kebanyakan deep-RL menggunakan `n` antara 3 dan 20.

## Build

### Langkah 1: SARSA pada kebijakan ε-serakah

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

Delapan baris. Perbedaan *satu-satunya* dari Q-learning adalah garis targetnya.

### Langkah 2: Pembelajaran Q

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` memisahkan target dari perilaku. Simbol yang satu ini adalah perbedaan antara on-policy dan off-policy.

### Langkah 3: kurva pembelajaran

Lacak pengembalian rata-rata per 100 episode. Q-learning menyatu lebih cepat pada GridWorld deterministik sederhana; SARSA lebih konservatif dalam melakukan cliff-walking. Pada GridWorld 4×4 di `code/main.py`, keduanya hampir optimal setelah ~2.000 episode dengan `α=0.1, ε=0.1`.

### Langkah 4: bandingkan dengan DP yang sebenarnya

Jalankan iterasi nilai (Lesson 02) untuk mendapatkan `Q*`. Periksa `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`. Agen TD tabular yang sehat mendarat di `~0.5` di GridWorld 4×4 setelah 10.000 episode.

## Jebakan

- **Nilai Q awal penting.** Inisiatif yang optimis (`Q = 0` untuk tugas dengan imbalan negatif) mendorong eksplorasi. Sikap pesimistis dapat menjebak kebijakan yang serakah selamanya.
- **jadwal α.** Konstanta `α` dapat digunakan untuk masalah non-stasioner. Peluruhan `α_n = 1/n` memberikan konvergensi dalam teori namun terlalu lambat dalam praktiknya — sematkan `α` di `[0.05, 0.3]` dan pantau kurva pembelajarannya.
- **ε jadwal.** Mulai tinggi (`ε=1.0`), menurun menjadi `ε=0.05`. “GLIE” (serakah pada batas dengan eksplorasi tak terbatas) adalah kondisi konvergensi.
- **Bias maksimal dalam Q-learning.** Operator `max` dibiaskan ke atas saat `Q` berisik. Menyebabkan perkiraan yang berlebihan — Pembelajaran Q Ganda Hasselt (digunakan oleh DDQN dalam Lesson 05) memperbaikinya dengan dua tabel Q.
- **Episode yang tidak berakhir.** TD dapat belajar tanpa terminal, namun kamu perlu membatasi langkah atau menangani bootstrap dengan benar pada bagian penutup. Standar: perlakukan cap sebagai non-terminal, pertahankan bootstrap.
- **Hashing status.** Jika status berupa tupel/tensor, gunakan kunci hashable (tupel, bukan daftar; tupel float berbentuk bulat, bukan mentah).

## Pakai

Pemandangan TD 2026:

| Tugas | Metode | Alasan |
|------|--------|--------|
| Lingkungan tabel kecil | Q-pembelajaran | Mempelajari kebijakan optimal secara langsung. |
| Penting bagi keselamatan dalam kebijakan | SARSA / SARSA yang Diharapkan | Konservatif selama eksplorasi. |
| Keadaan high-dimensional | DQN (Fase 9 · 05) | Fungsi Q neural network dengan pemutaran ulang dan jaring target. |
| Tindakan berkelanjutan | SAC / TD3 (Fase 9 · 07) | Pembaruan TD pada jaringan Q; jaring kebijakan mengeluarkan tindakan. |
| LLM RL (berbasis model penghargaan) | PPO/GRPO (Phase 9 · 08, 12) | Aktor-kritikus dengan keunggulan gaya TD melalui GAE. |
| RL Luar Talian | CQL / IQL (Fase 9 · 08) | Q-learning dengan regularisasi konservatif. |

Sembilan puluh persen dari "RL" yang kamu baca di makalah tahun 2026 adalah penjabaran dari Q-learning atau SARSA. Pahami pembaruan tabel di jari kamu sebelum membaca lebih dalam.

## Kirim

Simpan sebagai `outputs/skill-td-agent.md`:

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## Latihan1. **Mudah.** Menerapkan Q-learning dan SARSA di GridWorld 4×4. Kurva pembelajaran plot (pengembalian rata-rata per 100 episode) untuk 2.000 episode. Siapa yang menyatu lebih cepat?
2. **Sedang.** Build lingkungan berjalan di tebing (4×12, baris terakhir adalah tebing dengan hadiah -100 dan reset untuk memulai). Bandingkan kebijakan akhir Q-learning dan SARSA. Tangkapan layar jalur yang diambil masing-masing. Mana yang lebih dekat ke tebing?
3. **Sulit.** Menerapkan Pembelajaran Q Ganda. Pada GridWorld dengan imbalan berisik (kebisingan Gaussian σ=5 ditambahkan ke imbalan per langkah), tunjukkan Q-learning melebih-lebihkan `V*(0,0)` dengan jumlah yang berarti sedangkan Double Q-learning tidak.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| kesalahan TD | "Sinyal pembaruan" | `δ = r + γ V(s') - V(s)`, sisa bootstrap. |
| TD(0) | "TD satu langkah" | Perbarui setelah setiap transisi hanya menggunakan perkiraan negara bagian berikutnya. |
| Q-pembelajaran | "RL 101 di luar kebijakan" | Pembaruan TD dengan `max` mengenai tindakan selanjutnya; belajar `Q*` terlepas dari kebijakan perilaku. |
| SARS | "Pembelajaran Q sesuai kebijakan" | Pembaruan TD menggunakan tindakan aktual selanjutnya; pelajari `Q^π` untuk ε-serakah π saat ini. |
| SARSA yang diharapkan | "SARSA varian rendah" | Ganti sample `a'` dengan ekspektasinya di bawah π. |
| GLI | "Jadwal eksplorasi yang benar" | Serakah dalam Batas dengan Eksplorasi Tanpa Batas; diperlukan untuk konvergensi Q-learning. |
| Bootstrap | "Menggunakan perkiraan saat ini dalam target" | Yang membedakan TD dengan MC. Sumber bias tetapi pengurangan varians secara besar-besaran. |
| Bias maksimalisasi | "Q-learning melebih-lebihkan" | `max` perkiraan yang terlalu berisik bersifat bias ke atas; diperbaiki dengan Double Q-learning. |

## Bacaan Lanjutan

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — makalah asli dan bukti konvergensi.
- [Sutton & Barto (2018). Bab. 6 — Pembelajaran Perbedaan Temporal](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0), SARSA, Q-learning, SARSA yang diharapkan.
- [Hasselt (2010). Pembelajaran Q ganda](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — perbaikan untuk bias maksimalisasi.
- [Seijen, Hasselt, Whiteson, Wiering (2009). Analisis Teoritis dan Empiris terhadap SARSA yang Diharapkan](https://ieeexplore.ieee.org/document/4927542) — motivasi SARSA yang diharapkan.
- [Rummery & Niranjan (1994). Pembelajaran Q online menggunakan sistem koneksionis](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — makalah yang menciptakan SARSA (kemudian disebut "Q-learning koneksionis yang dimodifikasi").
- [Sutton & Barto (2018). Bab. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — menggeneralisasi TD(0) ke TD(n), jalur dari Q-learning ke jejak kelayakan dan, kemudian, GAE di PPO.
