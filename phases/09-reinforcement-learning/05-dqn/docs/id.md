# Jaringan Q Dalam (DQN)

> 2013: Mnih melatih satu jaringan Q-learning pada piksel mentah, mengalahkan setiap agen RL klasik di tujuh game Atari. 2015: diperluas menjadi 49 game, diterbitkan di Nature, memicu era deep-RL. DQN adalah Q-learning ditambah tiga trik yang membuat perkiraan fungsi stabil.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 3 · 03 (Backpropagation), Fase 9 · 04 (Q-learning, SARSA)
**Waktu:** ~75 menit

## Masalah

Pembelajaran Q tabular memerlukan nilai Q terpisah untuk setiap pasangan (status, tindakan). Papan catur memiliki ~10⁴³ status. Bingkai Atari berukuran 210×160×3 = 100.800 feature. Tabular RL mati di ribuan negara bagian, apalagi miliaran negara bagian.

Perbaikannya sudah jelas jika dipikir-pikir: ganti Q-table dengan neural network, `Q(s, a; θ)`. Namun, jika dipikir-pikir, butuh waktu puluhan tahun. Perkiraan fungsi naif dengan Q-learning berbeda dalam "triad mematikan" - perkiraan fungsi + bootstrapping + pembelajaran di luar kebijakan. Mnih dkk. (2013, 2015) mengidentifikasi tiga trik rekayasa yang menstabilkan pembelajaran:

1. **Pengalaman pemutaran ulang** menghiasi transisi.
2. **Jaringan target** membekukan target bootstrap.
3. **Kliping hadiah** menormalkan besaran gradient.

DQN di Atari adalah pertama kalinya arsitektur tunggal dengan satu set hyperparameter memecahkan lusinan masalah kontrol dari piksel mentah. Segala sesuatu yang "deep-RL" dibangun sejak itu — DDQN, Rainbow, Dueling, Distributional, R2D2, Agent57 — ditumpuk di atas basis tiga trik ini.

## Konsep

![Loop training DQN: env, buffer pemutaran ulang, jaringan online, jaringan target, loss Bellman TD](../assets/dqn.svg)

**Tujuannya.** DQN meminimalkan kehilangan TD satu langkah pada fungsi Q neural:

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = jaringan online, diperbarui setiap langkah dengan gradient descent. `θ^-` = jaringan target, disalin secara berkala dari `θ` (setiap ~10.000 langkah). `D` = buffer pemutaran ulang transisi sebelumnya.

**Tiga trik, berdasarkan urutan kepentingannya:**

**Nikmati pemutaran ulang.** Ring buffer `~10⁶` transisi. Setiap langkah training mengambil sample minibatch secara seragam dan acak. Hal ini memutus korelasi temporal (frame yang berurutan hampir identik), memungkinkan jaringan belajar dari transisi yang jarang dan bermanfaat berkali-kali, dan mendekorelasi pembaruan gradient yang berurutan. Tanpanya, TD sesuai kebijakan dengan neural network akan berbeda di Atari.

**Jaringan target.** Menggunakan jaringan yang sama `Q(·; θ)` di kedua sisi persamaan Bellman membuat target bergerak setiap pembaruan — "mengejar ekor kamu sendiri". Cara mengatasinya: pertahankan jaringan kedua `Q(·; θ^-)` dengan weight beku. Setiap `C` langkah, salin `θ → θ^-`. Ini menstabilkan target regresi untuk ribuan langkah gradient sekaligus. Pembaruan lunak `θ^- ← τ θ + (1-τ) θ^-` (digunakan di DDPG, SAC) adalah varian yang lebih lancar.

**Kliping hadiah.** Besaran hadiah Atari bervariasi dari 1 hingga 1000+. Memotong ke `{-1, 0, +1}` menghentikan game apa pun untuk mendominasi gradient. Salah ketika besarnya imbalan penting; baik untuk Atari di mana hanya tanda yang penting.

**Double DQN.** Hasselt (2016) memperbaiki bias maksimalisasi: gunakan jaringan online untuk *memilih* tindakan, jaringan target untuk *mengevaluasi* tindakan tersebut.

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

Penggantian drop-in, secara konsisten lebih baik. Gunakan secara default.**Peningkatan lainnya (Rainbow, 2017):** pemutaran ulang yang diprioritaskan (contoh transisi kesalahan TD tinggi lebih banyak), arsitektur duel (`V(s)` dan kepala keuntungan terpisah), jaringan berisik (eksplorasi yang dipelajari), pengembalian n-langkah, distribusi Q (C51/QR-DQN), bootstrapping multi-langkah. Masing-masing menambahkan beberapa persen; keuntungannya kira-kira bersifat aditif.

## Build

Code di sini hanya stdlib bebas numpy - kami menggunakan MLP layer tunggal tersembunyi yang dilinting dengan tangan pada GridWorld kecil yang berkelanjutan, sehingga setiap langkah training berjalan dalam mikrodetik. Algoritmenya identik dengan Atari DQN dalam skala besar.

### Langkah 1: memutar ulang buffer

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

~kapasitas 50.000 untuk Atari; 5.000 cukup untuk mainan kita.

### Langkah 2: jaringan Q kecil (MLP manual)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

Lintasan ke depan: linier → ReLU → linier. Itu adalah keseluruhan jaringnya.

### Langkah 3: pembaruan DQN

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

Bentuknya adalah Q-learning dari Lesson 04 dengan dua perbedaan: (a) kita melakukan backprop melalui `Q(·; θ)` yang dapat dibedakan alih-alih mengindeks tabel, (b) target menggunakan `Q(·; θ^-)`.

### Langkah 4: loop luar

Untuk setiap episode, bertindak ε-serakah di `Q(·; θ)`, dorong transisi ke buffer, cicipi minibatch, ambil langkah gradient, sinkronkan secara berkala `θ^- ← θ`. Polanya:

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

Di GridWorld kecil kami dengan status one-hot 16-redup, agen mempelajari kebijakan yang hampir optimal dalam ~500 episode. Di Atari, skalakan ini ke 200 juta frame dan tambahkan ekstraktor feature CNN.

## Jebakan

- **Triad mematikan.** Perkiraan fungsi + di luar kebijakan + bootstrap bisa berbeda. DQN melakukan mitigasi dengan target net + replay; jangan hapus juga.
- **Eksplorasi.** ε harus berkurang, biasanya dari 1,0 menjadi 0,01 selama ~10% training pertama. Tanpa eksplorasi awal yang cukup, Q-net akan menyatu dengan cekungan lokal.
- **Estimasi berlebihan.** `max` Q yang terlalu berisik bersifat bias ke atas. Selalu gunakan Double DQN dalam produksi.
- **Skala hadiah.** Memotong atau menormalkan hadiah; besaran gradient sebanding dengan besarnya imbalan.
- **Putar ulang buffer coldstart.** Jangan berlatih hingga buffer memiliki beberapa ribu transisi. Gradient awal pada ~20 sample pakaian berlebih.
- **Frekuensi sinkronisasi target.** Terlalu sering ≈ tidak ada target bersih; terlalu jarang ≈ target basi. Atari DQN menggunakan 10.000 langkah env. Aturan praktisnya: sinkronkan setiap ~1/100 cakrawala training.
- **Preprocessing observasi.** Atari DQN menumpuk 4 frame untuk membuat status Markov. Setiap env dengan info kecepatan memerlukan penumpukan bingkai atau status berulang.

## Pakai

Pada tahun 2026, DQN jarang merupakan algoritma yang canggih tetapi tetap menjadi referensi algoritma di luar kebijakan:

| Tugas | Metode pilihan | Kenapa bukan DQN? |
|------|------------------|--------------|
| Tindakan diskrit seperti Atari | Pelangi DQN atau Muesli | Kerangka kerja yang sama, lebih banyak trik. |
| Kontrol berkelanjutan | SAC / TD3 (Fase 9 · 07) | DQN tidak memiliki jaringan kebijakan. |
| Sesuai kebijakan / throughput tinggi | PPO (Phase 9 · 08) | Tidak ada buffer pemutaran ulang; lebih mudah untuk diukur. |
| RL Luar Talian | CQL / IQL / Transformer Keputusan | Target Q konservatif, tidak ada ledakan bootstrap. |
| Ruang tindakan terpisah yang besar (pemberi rekomendasi) | DQN dengan embedding tindakan, atau IMPALA | Bagus; dekorasi penting. |
| LLM RL | PPO/GRPO | Tingkat urutan, bukan tingkat langkah; loss yang berbeda. |

Pelajarannya masih berjalan. Putar ulang dan jaringan target muncul di SAC, TD3, DDPG, SAC-X, buffer putar mandiri AlphaZero, dan setiap metode RL offline. Pemotongan hadiah tetap ada sebagai normalisasi keuntungan di PPO. Arsitektur adalah cetak birunya.

## KirimkanSimpan sebagai `outputs/skill-dqn-trainer.md`:

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Plot kurva pengembalian per episode. Berapa episode hingga rata-rata berjalan melebihi -10?
2. **Sedang.** Nonaktifkan jaringan target (gunakan jaringan online untuk kedua sisi target Bellman). Ukur ketidakstabilan training - apakah return berfluktuasi atau menyimpang?
3. **Sulit.** Tambahkan Double DQN: gunakan jaringan online untuk memilih `argmax a'`, target jaringan untuk mengevaluasi. Bandingkan bias `Q(s_0, best_a)` vs `V*(s_0)` yang sebenarnya setelah 1.000 episode dengan vs tanpa Double DQN di GridWorld yang memberikan banyak hadiah.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| DQN | "Pembelajaran Q Mendalam" | Q-learning dengan fungsi neural Q, buffer replay, dan jaringan target. |
| Pengalaman memutar ulang | "Transisi acak" | Buffer cincin mengambil sample secara seragam pada setiap langkah gradient; mendekorasi data. |
| Jaringan sasaran | "Bootstrap beku" | Salinan berkala Q yang digunakan dalam target Bellman; menstabilkan training. |
| Triad mematikan | "Mengapa RL menyimpang" | Perkiraan fungsi + bootstrapping + di luar kebijakan = tidak ada jaminan konvergensi. |
| DQN Ganda | "Perbaiki untuk bias maksimalisasi" | Jaringan online memilih tindakan, jaringan target mengevaluasinya. |
| Duel DQN | "Kepala V dan A" | Mengurai Q = V + A - mean(A); output yang sama, aliran gradient yang lebih baik. |
| Pelangi | "Semua triknya" | DDQN + PER + duel + n-langkah + berisik + distribusi dalam satu. |
| PER | "Pemutaran Ulang yang Diprioritaskan" | Transisi sample sebanding dengan besarnya kesalahan TD. |

## Bacaan Lanjutan

- [Mnih dkk. (2013). Memainkan Atari dengan Pembelajaran Penguatan Mendalam](https://arxiv.org/abs/1312.5602) — makalah lokakarya NeurIPS tahun 2013 yang memulai RL mendalam.
- [Mnih dkk. (2015). Kontrol tingkat manusia melalui pembelajaran penguatan mendalam](https://www.nature.com/articles/nature14236) — makalah Nature, 49 game DQN.
- [Hasselt, Guez, Perak (2016). Pembelajaran Penguatan Mendalam dengan Pembelajaran Q Ganda](https://arxiv.org/abs/1509.06461) — DDQN.
- [Wang dkk. (2016). Arsitektur Jaringan Duel](https://arxiv.org/abs/1511.06581) — duel DQN.
- [Hessel dkk. (2018). Rainbow: Menggabungkan Peningkatan pada Deep RL](https://arxiv.org/abs/1710.02298) — makalah trik bertumpuk.
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — eksposisi modern yang jelas.
- [Sutton & Barto (2018). Bab. 9 — Prediksi Sesuai Kebijakan dengan Perkiraan](http://incompleteideas.net/book/RLbook2020.pdf) — perlakuan buku teks tentang "triad mematikan" (perkiraan fungsi + bootstrapping + di luar kebijakan) yang dirancang untuk dijinakkan oleh jaringan target dan buffer pemutaran ulang DQN.
- [Implementasi DQN CleanRL](https://docs.cleanrl.dev/rl-algorithms/dqn/) — referensi DQN file tunggal yang digunakan dalam studi ablasi; bagus untuk dibaca bersamaan dengan versi awal lesson ini.
