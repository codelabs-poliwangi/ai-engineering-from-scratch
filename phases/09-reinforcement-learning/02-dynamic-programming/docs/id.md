# Pemrograman Dinamis — Iterasi Kebijakan & Iterasi Nilai

> Pemrograman dinamis adalah RL dengan kecurangan. kamu sudah mengetahui fungsi transisi dan penghargaan; kamu cukup mengulangi persamaan Bellman hingga `V` atau `π` berhenti bergerak. Ini adalah tolok ukur yang coba didekati oleh setiap metode berbasis pengambilan sample.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 9 · 01 (MDP)
**Waktu:** ~75 menit

## Masalah

kamu memiliki MDP dengan model yang dikenal: kamu dapat menanyakan `P(s' | s, a)` dan `R(s, a, s')` untuk pasangan tindakan negara mana pun. Seorang manajer inventaris mengetahui distribusi permintaan. Permainan papan memiliki transisi deterministik. Gridworld adalah empat baris Python. kamu memiliki *model*.

RL tanpa model (Q-learning, PPO, REINFORCE) diciptakan untuk kasus ketika kamu tidak memiliki model — kamu hanya dapat mengambil sample dari lingkungan. Namun jika kamu memilikinya, ada metode yang lebih cepat dan lebih baik: pemrograman dinamis. Bellman merancangnya pada tahun 1957. Mereka masih mendefinisikan kebenaran: ketika orang mengatakan "kebijakan optimal untuk MDP ini", yang mereka maksud adalah kebijakan DP akan kembali.

kamu membutuhkannya pada tahun 2026 karena tiga alasan. Pertama, setiap lingkungan tabel dalam penelitian RL (GridWorld, FrozenLake, CliffWalking) diselesaikan dengan DP untuk menghasilkan kebijakan standar emas. Kedua, nilai eksak memungkinkan kamu *men-debug* metode pengambilan sample: jika perkiraan Q-learning untuk `V*(s_0)` tidak setuju dengan jawaban DP sebesar 30%, Q-learning kamu memiliki bug. Ketiga, RL offline modern dan metode perencanaan (MCTS, pencarian AlphaZero, RL berbasis model di Fase 9 · 10) semuanya mengulangi cadangan Bellman melalui model yang dipelajari atau diberikan.

## Konsep

![Iterasi kebijakan dan iterasi nilai, berdampingan](../assets/dp.svg)

**Dua algoritma, keduanya merupakan iterasi titik tetap di Bellman.**

**Perulangan kebijakan.** Mengganti dua langkah hingga kebijakan berhenti berubah.

1. *Evaluasi:* kebijakan yang diberikan `π`, hitung `V^π` dengan berulang kali menerapkan `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` hingga konvergen.
2. *Peningkatan:* diberikan `V^π`, membuat `π` serakah w.r.t. `V^π`: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

Konvergensi dijamin karena (a) setiap langkah perbaikan akan menjaga `π` tetap sama atau secara ketat meningkatkan `V^π` untuk beberapa negara bagian, (b) ruang kebijakan deterministik terbatas. Biasanya menyatu dalam ~5–20 iterasi terluar bahkan untuk state space yang besar.

**Perulangan nilai.** Meruntuhkan evaluasi dan perbaikan dalam satu sapuan. Terapkan persamaan *optimalitas* Bellman:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

Ulangi sampai `max_s |V_{new}(s) - V(s)| < ε`. Ekstrak kebijakan di akhir dengan mengambil tindakan serakah. Per iterasinya jauh lebih cepat — tidak ada loop evaluasi internal — namun biasanya memerlukan lebih banyak iterasi agar dapat menyatu.

**Perulangan kebijakan umum (GPI).** Kerangka pemersatu. Fungsi nilai dan kebijakan terkunci dalam lingkaran perbaikan dua arah; metode apa pun yang mendorong keduanya menuju konsistensi timbal balik (iterasi nilai async, iterasi kebijakan yang dimodifikasi, Q-learning, aktor-kritikus, PPO) adalah contoh dari GPI.

**Mengapa `γ < 1` penting.** Operator Bellman adalah `γ` -kontraksi dalam sup-norm: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. Kontraksi menyiratkan titik tetap yang unik dan konvergensi geometri. Jatuhkan `γ < 1` dan kamu kehilangan jaminan — kamu memerlukan cakrawala yang terbatas atau keadaan terminal yang menyerap.

## Build

### Langkah 1: buat model MDP GridWorldGunakan GridWorld 4×4 yang sama dari Lesson 01. Kami menambahkan varian stokastik: dengan probabilitas `0.1` agen tergelincir ke arah tegak lurus acak.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` mengembalikan daftar `(s', r, p)`. Ini adalah keseluruhan modelnya.

### Langkah 2: evaluasi kebijakan

Dengan adanya kebijakan `π(s) = {action: prob}`, ulangi persamaan Bellman hingga `V` berhenti bergerak:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Langkah 3: perbaikan kebijakan

Ganti `π` dengan kebijakan serakah w.r.t. `V`. Jika `π` tidak berubah, kembalilah — kita berada pada kondisi optimal.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Langkah 4: jahit menjadi satu

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

Konvergensi tipikal pada 4×4: 4–6 iterasi luar. Output `V*(0,0) ≈ -6` dan kebijakan yang secara tegas mengurangi jumlah langkah.

### Langkah 5: iterasi nilai (versi satu putaran)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

Titik tetap yang sama, baris code lebih sedikit.

## Jebakan

- **Lupa menangani terminal.** Jika kamu menerapkan Bellman ke kondisi menyerap, ia tetap mengambil "tindakan terbaik" yang tidak mengubah apa pun. Jaga dengan `if s == terminal: V[s] = 0`.
- **Sup-norm vs konvergensi L2.** Gunakan `max |V_new - V|`, bukan rata-rata. Jaminan teoritis berada pada sup-norm.
- **Pembaruan di tempat vs pembaruan sinkron.** Memperbarui `V[s]` di tempat (Gauss-Seidel) menyatu lebih cepat daripada dict `V_new` (Jacobi) yang terpisah. Code produksi digunakan di tempat.
- **Ikatan kebijakan.** Jika dua tindakan memiliki nilai Q yang sama, `argmax` dapat memutus ikatan secara berbeda pada setiap iterasi, sehingga menyebabkan pemeriksaan "stabil kebijakan" berosilasi. Gunakan tie-break yang stabil (tindakan pertama dalam urutan tetap).
- **Ledakan ruang angkasa.** DP adalah `O(|S| · |A|)` per sapuan. Bekerja hingga ~10⁷ negara bagian. Selain itu, kamu memerlukan perkiraan fungsi (Fase 9 · 05 dan seterusnya).

## Pakai

Pada tahun 2026, DP adalah garis dasar kebenaran dan lingkaran dalam perencana:

| Kasus penggunaan | Metode |
|----------|--------|
| Selesaikan MDP tabel kecil dengan tepat | Iterasi nilai (lebih sederhana) atau iterasi kebijakan (langkah luar lebih sedikit) |
| Verifikasi implementasi Q-learning / PPO | Bandingkan dengan DP-optimal V* pada lingkungan mainan |
| RL berbasis model (Fase 9 · 10) | Cadangan Bellman pada model transisi yang dipelajari |
| Perencanaan di AlphaZero / MuZero | Pencarian Pohon Monte Carlo = cadangan Bellman async |
| RL Luar Talian (CQL, IQL) | Q-iterasi Konservatif — DP dengan penalti atas tindakan OOD |

Setiap kali seseorang mengatakan "fungsi nilai optimal", yang mereka maksud adalah "titik tetap DP". Saat kamu melihat `V*` atau `Q*` di kertas, bayangkan lingkaran ini.

## Kirim

Simpan sebagai `outputs/skill-dp-solver.md`:

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## Latihan

1. **Mudah.** Jalankan iterasi nilai pada GridWorld 4×4 dengan `γ ∈ {0.9, 0.99}`. Berapa kali penyisiran hingga `max |ΔV| < 1e-6`? Cetak `V*` sebagai kisi 4×4.
2. **Sedang.** Bandingkan iterasi kebijakan vs iterasi nilai pada *stochastic* GridWorld (probabilitas slip `0.1`). Hitungan: sapuan, waktu jam dinding, final `V*(0,0)`. Mana yang konvergen lebih cepat dalam iterasi? Di jam dinding?
3. **Sulit.** Buat iterasi kebijakan yang dimodifikasi: pada langkah evaluasi, jalankan hanya sapuan `k` dan bukan ke konvergensi. Plot kesalahan `V*(0,0)` vs `k` untuk `k ∈ {1, 2, 5, 10, 50}`. Apa yang ditunjukkan oleh kurva tersebut mengenai trade-off evaluasi/perbaikan?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Iterasi kebijakan | "Algoritma DP" | Evaluasi bergantian (`V^π`) dan perbaikan (serakah `π` w.r.t. `V^π`) hingga kebijakan berhenti berubah. |
| Nilai iterasi | "DP Lebih Cepat" | Pencadangan optimalitas Bellman diterapkan dalam satu sapuan; menyatu ke `V*` secara geometris. |
| Operator pelayan | "Rekursi" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; `γ` -kontraksi dalam batas normal. |
| Kontraksi | "Mengapa DP menyatu" | Setiap operator `T` dengan `||T x - T y|| ≤ γ ||x - y||` memiliki titik tetap yang unik. |
| GPI | "Semuanya DP" | Iterasi Kebijakan Umum: metode apa pun yang mendorong `V` dan `π` menuju konsistensi bersama. |
| Pembaruan sinkron | "Gaya Jacobi" | Gunakan `V` lama selama penyisiran; dapat dianalisis dengan rapi tetapi lebih lambat. |
| Pembaruan di tempat | "Gaya Gauss-Seidel" | Gunakan `V` saat sedang diperbarui; menyatu lebih cepat dalam praktiknya. |

## Bacaan Lanjutan

- [Sutton & Barto (2018). Bab. 4 — Pemrograman Dinamis](http://incompleteideas.net/book/RLbook2020.pdf) — presentasi kanonik tentang iterasi kebijakan dan iterasi nilai.
- [Bertsekas (2019). Pembelajaran Penguatan dan Kontrol Optimal](http://www.athenasc.com/rlbook.html) — perlakuan ketat terhadap argumen pemetaan kontraksi.
- [Puterman (2005). Proses Keputusan Markov](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — modifikasi iterasi kebijakan dan analisis konvergensinya.
- [Howard (1960). Pemrograman Dinamis dan Proses Markov](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — makalah iterasi kebijakan asli.
- [Bertsekas & Tsitsiklis (1996). Pemrograman Neuro-Dinamis](http://www.athenasc.com/ndpbook.html) — jembatan dari DP ke perkiraan-DP / RL dalam yang digunakan oleh setiap lesson berikutnya.
