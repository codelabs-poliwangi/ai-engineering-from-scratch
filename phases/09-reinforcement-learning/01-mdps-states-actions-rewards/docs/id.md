# MDP, Status, Tindakan & Hadiah

> Proses Keputusan Markov terdiri dari lima hal: status, tindakan, transisi, penghargaan, diskon. Segala sesuatu di RL — Q-learning, PPO, DPO, GRPO — dioptimalkan dalam bentuk ini. Learn sekali, baca sisa pembelajaran penguatan secara gratis.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 1 · 06 (Probability & Distributions), Fase 2 · 01 (Taksonomi ML)
**Waktu:** ~45 menit

## Masalah

kamu sedang menulis bot catur. Atau perencana inventaris. Atau agen tradeoff. Atau loop PPO yang melatih model penalaran. Empat domain berbeda, satu fakta mengejutkan: keempat domain tersebut runtuh pada objek matematika yang sama.

Pembelajaran yang diawasi memberi kamu `(x, y)` pasangan dan meminta kamu menyesuaikan suatu fungsi. Pembelajaran penguatan tidak memberi kamu label — hanya aliran status, tindakan yang kamu ambil, dan imbalan scalar. Apakah langkah tersebut memenangkan permainan? Apakah keputusan restock menghemat uang? Apakah tradeoff tersebut menghasilkan keuntungan? Apakah token yang baru saja dihasilkan LLM menghasilkan hadiah yang lebih tinggi dari juri?

kamu tidak dapat belajar dari aliran ini sampai kamu meresmikannya. "Apa yang saya lihat", "apa yang saya lakukan", "apa yang terjadi selanjutnya", "betapa bagusnya hal itu" - masing-masing harus menjadi objek yang dapat kamu pikirkan. Formalisasi itu adalah Proses Keputusan Markov. Setiap algoritma RL dalam fase ini, termasuk loop RLHF dan GRPO di akhir, mengoptimalkan bentuk ini.

## Konsep

![Proses pengambilan keputusan Markov: status, tindakan, transisi, penghargaan, diskon](../assets/mdp.svg)

**Lima objek.**

- **Negara Bagian** `S`. Segala sesuatu yang perlu diputuskan oleh agen. Di GridWorld, sel. Dalam catur, papan. Dalam LLM, jendela konteks ditambah memori apa pun.
- **Tindakan** `A`. Pilihannya. Bergerak ke atas/bawah/kiri/kanan. Mainkan gerakan. Keluarkan token.
- **Transisi** `P(s' | s, a)`. Mengingat keadaan `s` dan tindakan `a`, distribusi ke negara bagian berikutnya. deterministik dalam catur, stokastik dalam inventaris, hampir deterministik dalam decoding LLM.
- **Hadiah** `R(s, a, s')`. Sinyal scalar. Menang = +1, kalah = -1. Pendapatan dikurangi biaya. Istilah rasio kemungkinan log dalam GRPO.
- **Diskon** `γ ∈ [0, 1)`. Berapa besarnya imbalan di masa depan vs saat ini. `γ = 0.99` membeli cakrawala ~100 langkah; `γ = 0.9` membeli ~10.

**Properti Markov** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`. Masa depan hanya bergantung pada keadaan saat ini. Jika tidak, representasi negara tidak lengkap – bukan kegagalan metode, melainkan kegagalan negara.

**Kebijakan dan pengembalian.** Kebijakan `π(a | s)` memetakan distribusi status ke tindakan. Pengembalian `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` adalah jumlah diskon dari hadiah di masa mendatang. Nilai `V^π(s) = E[G_t | s_t = s]` adalah pengembalian yang diharapkan mulai dari `s` berdasarkan kebijakan `π`. Nilai-Q `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` adalah pengembalian yang diharapkan dimulai dengan tindakan tertentu. Setiap algoritme RL memperkirakan salah satu dari keduanya, lalu menyempurnakan `π`.

**Persamaan Bellman.** Persamaan titik tetap yang digunakan semua hal dalam fase ini:

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

Hasil yang diharapkan ini dibagi menjadi "hadiah langkah ini" ditambah "nilai diskon tempat kamu mendarat". Rekursif. Setiap algoritme di Fase 9 mengulangi persamaan ini ke konvergensi (pemrograman dinamis), mengambil sample darinya (Monte Carlo), atau mem-bootstrapnya satu langkah (perbedaan temporal).

## Build

### Langkah 1: MDP deterministik kecilDunia Grid 4×4. Agen dimulai dari kiri atas, terminal di kanan bawah, hadiah -1 per langkah, tindakan `{up, down, left, right}`. Lihat `code/main.py`.

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

Lima baris. Itu adalah keseluruhan lingkungan. Transisi deterministik, penalti langkah konstan, keadaan terminal yang menyerap.

### Langkah 2: meluncurkan kebijakan

Kebijakan adalah fungsi dari distribusi negara ke tindakan. Yang paling sederhana: seragam acak.

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

Jalankan kebijakan acak 1000 kali. Pengembalian rata-rata adalah sekitar -60 hingga -80 untuk papan 4×4 ini. Pengembalian optimalnya adalah -6 (jalur lurus bawah-kanan). Menutup kesenjangan tersebut adalah segalanya di Fase 9.

### Langkah 3: hitung `V^π` persis melalui persamaan Bellman

Untuk MDP kecil persamaan Bellman adalah sistem linier. Hitung status, terapkan ekspektasi, ulangi hingga nilainya berhenti berubah.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

Ini adalah evaluasi kebijakan yang berulang. Ini adalah algoritma pertama dalam Sutton & Barto dan landasan teoritis dari setiap metode RL berikutnya.

### Langkah 4: `γ` adalah hyperparameter dengan makna fisik

Cakrawala efektif kira-kira `1 / (1 - γ)`. `γ = 0.9` → 10 langkah. `γ = 0.99` → 100 langkah. `γ = 0.999` → 1000 langkah.

Terlalu rendah dan agen bertindak secara rabun. Terlalu tinggi dan pemberian kredit menjadi berisik, karena banyak langkah awal yang berbagi tanggung jawab untuk mendapatkan imbalan di masa depan. LLM RLHF biasanya menggunakan `γ = 1` karena episodenya pendek dan terbatas. Kontrol tugas menggunakan `0.95–0.99`. Game strategi jangka panjang menggunakan `0.999`.

## Jebakan

- **Keadaan Non-Markovian.** Jika kamu memerlukan tiga pengamatan terakhir untuk memutuskan, "keadaan" bukan hanya pengamatan saat ini. Perbaiki: tumpukan bingkai (DQN pada tumpukan Atari 4) atau gunakan status berulang (LSTM/GRU atas observasi).
- **Hadiah yang jarang.** Hadiah yang hanya dapat dimenangkan membuat pembelajaran hampir tidak mungkin dilakukan di ruang negara bagian yang besar. Bentuk imbalan (sinyal perantara) atau bootstrap dengan imitasi (Fase 9 · 09).
- **Peretasan hadiah.** Mengoptimalkan hadiah proxy sering kali menghasilkan perilaku patologis. Agen balap perahu OpenAI berputar-putar mengumpulkan powerup selamanya alih-alih menyelesaikan balapan. Selalu tentukan imbalan berdasarkan hasil yang ditargetkan, bukan proksinya.
- **Diskon salah spesifikasi.** `γ = 1` pada tugas cakrawala tak terbatas membuat setiap nilai menjadi tak terbatas. Selalu batasi dengan cakrawala terbatas atau `γ < 1`.
- **Skala imbalan.** Penghargaan sebesar {+100, -100} vs {+1, -1} memberikan kebijakan optimal yang sama namun besaran gradiennya sangat berbeda. Normalisasikan ke `[-1, 1]`-ish sebelum menyambungkan ke PPO/DQN.

## Pakai

Tumpukan 2026 mengurangi setiap pipeline RL menjadi MDP sebelum menyentuh code:

| Situasi | Negara | Aksi | Hadiah | |
|-----------|-------|--------|--------|---|
| Kontrol (penggerak, manipulasi) | Sudut sambungan + kecepatan | Torsi terus menerus | Berbentuk khusus tugas | 0,99 |
| Permainan (catur, Go, poker) | Papan + sejarah | Langkah hukum | Menang=+1 / kalah=-1 | 1.0 (terbatas) |
| Persediaan / harga | Stok + permintaan | Jumlah pesanan | Pendapatan - biaya | 0,95 |
| RLHF untuk LLM | Token konteks | Token berikutnya | Skor model penghargaan di akhir | 1.0 (episode ~200 token) |
| GRPO untuk penalaran | Respon cepat + sebagian | Token berikutnya | Verifikator 0/1 di akhir | 1.0 |

Tulis lima tupel sebelum menulis loop training apa pun. Sebagian besar laporan bug "RL tidak berfungsi" menelusuri kembali ke formulasi MDP yang rusak di atas kertas.

## Kirim

Simpan sebagai `outputs/skill-mdp-modeler.md`:```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## Latihan

1. **Mudah.** Terapkan GridWorld 4×4 dan peluncuran kebijakan acak di `code/main.py`. Jalankan 10.000 episode. Laporkan rata-rata dan std pengembalian. Bandingkan dengan return optimal (-6).
2. **Sedang.** Jalankan `policy_evaluation` dengan `γ ∈ {0.5, 0.9, 0.99}` untuk kebijakan seragam-acak. Cetak `V` sebagai kotak 4×4 untuk masing-masingnya. Jelaskan mengapa nilai negara bagian di dekat terminal tumbuh lebih cepat dengan `γ` yang lebih besar.
3. **Sulit.** Putar stokastik GridWorld: setiap tindakan tergelincir ke arah yang berdekatan dengan probabilitas `p = 0.1`. Evaluasi kembali kebijakan seragam. Apakah `V[start]` menjadi lebih baik atau lebih buruk? Mengapa?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| MDP | "Pengaturan pembelajaran penguatan" | Tuple `(S, A, P, R, γ)` memuaskan properti Markov. |
| Negara | "Apa yang dilihat agen" | Statistik yang memadai untuk dinamika masa depan berdasarkan jenis kebijakan yang dipilih. |
| Kebijakan | "Perilaku Agen" | Distribusi bersyarat `π(a | s)` atau peta deterministik `s → a`. |
| Kembali | "Total hadiah" | Jumlah diskon `Σ γ^t r_t` dari langkah saat ini. |
| Nilai | "Betapa bagusnya suatu negara bagian" | Pengembalian yang diharapkan di bawah `π` mulai dari `s`. |
| Nilai-Q | "Betapa bagusnya suatu tindakan" | Pengembalian yang diharapkan di bawah `π` dimulai dari `s` dengan tindakan pertama `a`. |
| Persamaan Bellman | "Rekursi pemrograman dinamis" | Decomposition nilai titik tetap / Q menjadi hadiah satu langkah ditambah nilai penerus yang didiskon. |
| Diskon `γ` | "Masa depan vs sekarang" | Weight geometris pada imbalan di masa depan; cakrawala efektif `~1/(1-γ)`. |

## Bacaan Lanjutan

- [Sutton & Barto (2018). Pembelajaran Penguatan: Sebuah Pengantar, edisi ke-2] (http://incompleteideas.net/book/RLbook2020.pdf) — buku teks. Bab. 3 mencakup persamaan MDP dan Bellman; Bab. 1 memotivasi hipotesis penghargaan yang mendasari setiap lesson berikutnya.
- [Belman (1957). Pemrograman Dinamis](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — asal mula persamaan Bellman.
- [OpenAI Spinning Up — Bagian 1: Konsep Utama](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — primer MDP ringkas dari sudut RL yang dalam.
- [Puterman (2005). Proses Keputusan Markov](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — referensi penelitian operasi tentang MDP dan metode solusi tepat.
- [Littman (1996). Algoritma untuk Pengambilan Keputusan Berurutan (tesis PhD)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — turunan paling bersih dari MDP sebagai spesialisasi pemrograman dinamis.
