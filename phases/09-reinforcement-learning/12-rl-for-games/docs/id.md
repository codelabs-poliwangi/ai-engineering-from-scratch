# RL untuk Game — AlphaZero, MuZero, dan Era Penalaran LLM

> 1992: TD-Gammon mengalahkan juara manusia di backgammon dengan TD murni. 2016: AlphaGo mengalahkan Lee Sedol. 2017: AlphaZero mendominasi catur, shogi, dan Go dari awal. 2024: DeepSeek-R1 membuktikan resep yang sama, dengan GRPO menggantikan PPO, bekerja berdasarkan penalaran. Game menjadi tolok ukur yang mendorong setiap terobosan di fase ini.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 9 · 05 (DQN), Fase 9 · 08 (PPO), Fase 9 · 09 (RLHF), Fase 9 · 10 (MARL)
**Waktu:** ~120 menit

## Masalah

Game memiliki semua yang diinginkan RL. Hadiah bersih (menang/kalah). Episode tak terbatas (reset putar mandiri). Simulasi sempurna (permainan *adalah* simulator). Ruang tindakan kontinu yang diskrit atau kecil. Struktur multi-agen yang memaksakan ketahanan permusuhan.

Dan permainan adalah cara setiap terobosan besar RL diuji. TD-Gammon (backgammon, 1992). Atari-DQN (2013). AlfaGo (2016). AlfaZero (2017). OpenAI Lima (Dota 2, 2019). Bintang Alpha (StarCraft II, 2019). MuZero (model yang dipelajari, 2019). AlphaTensor (perkalian matrix, 2022). AlphaDev (algoritma pengurutan, 2023). DeepSeek-R1 (penalaran matematika, 2025) — demonstrasi terbaru bahwa teknik game-RL bekerja pada teks.

Batu penjuru ini mensurvei tiga arsitektur penting — AlphaZero, MuZero, dan GRPO — melalui satu lensa pemersatu: **permainan mandiri + penelusuran + peningkatan kebijakan**. Masing-masing menggeneralisasikan yang sebelumnya; GRPO khususnya adalah resep AlphaZero yang diterapkan pada penalaran LLM, dengan token sebagai tindakan dan verifikasi matematis sebagai sinyal kemenangan.

## Konsep

![AlphaZero ↔ MuZero ↔ GRPO: loop yang sama, lingkungan berbeda](../assets/rl-games.svg)

**Lingkaran pemersatu.**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017).** Perak dkk. Diberikan permainan (catur, shogi, Go) dengan aturan yang diketahui:

- Jaringan nilai kebijakan: satu menara `f_θ(s) → (p, v)`. `p` adalah langkah hukum sebelumnya. `v` adalah hasil pertandingan yang diharapkan.
- Pencarian Pohon Monte Carlo (MCTS): di setiap gerakan, perluas pohon kemungkinan kelanjutan. Gunakan `(p, v)` sebagai + bootstrap sebelumnya. Pilih node berdasarkan UCB (PUCT): `a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`.
- Bermain mandiri: bermain game agen-vs-agen. Pada perpindahan `t`, distribusi kunjungan MCTS `π_t` menjadi sasaran training kebijakan.
- Kalah: `L = (v - z)² - π · log p + c · ||θ||²`. `z` adalah hasil pertandingan (+1 / 0 / -1).

Nol pengetahuan manusia. Nol heuristik buatan tangan. Sebuah resep tunggal yang menguasai catur, shogi, dan Go setelah masing-masing beberapa puluh juta permainan yang dimainkan sendiri.

**MuZero (2019).** Schrittwieser dkk. Menghapus persyaratan bahwa aturan diketahui.

- Daripada menggunakan lingkungan yang tetap, pelajari *model dinamika laten* `(h, g, f)`:
  - `h(s)`: menyandikan pengamatan ke keadaan laten.
  - `g(s_latent, a)`: memprediksi keadaan laten berikutnya + hadiah.
  - `f(s_latent)`: memprediksi kebijakan sebelum + nilai.
- MCTS berjalan di *ruang laten yang dipelajari*. Pencarian yang sama, putaran training yang sama.
- Bekerja pada Go, catur, shogi *dan* Atari — satu algoritma, tanpa pengetahuan aturan.

**Stochastic MuZero (2022).** Menambahkan dinamika stokastik dan node peluang; meluas ke permainan kelas backgammon.

**Muesli, Gumbel MuZero (2022-2024).** Peningkatan efisiensi sample dan penelusuran deterministik.

**GRPO (2024-2025).** Resep DeepSeek-R1. Loop berbentuk AlphaZero yang sama, diterapkan pada penalaran model bahasa:- "Game": menjawab soal matematika / coding / penalaran. "Menang" = pemverifikasi (kasus uji lolos, jawaban numerik cocok) menghasilkan 1.
- Kebijakan: LLM. Tindakan: token. Status: prompt + respons sejauh ini.
- Tidak ada kritik (V_φ gaya PPO). Sebagai gantinya, untuk setiap prompt, contoh penyelesaian `G` dari kebijakan. Hitung imbalan untuk masing-masing. Gunakan **keuntungan relatif grup** `A_i = (r_i - mean_r) / std_r` sebagai sinyal untuk pembaruan bergaya REINFORCE.
- Hukuman KL sebagai acuan kebijakan untuk mencegah penyimpangan (seperti RLHF).
- Loss penuh:

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

Tidak ada model penghargaan, tidak ada kritik, tidak ada MCTS. Garis dasar relatif kelompok menggantikan ketiganya. Mencocokkan atau melampaui kualitas PPO-RLHF pada tolok ukur penalaran di sebagian kecil dari komputasi.

**Resep R1 selengkapnya.** DeepSeek-R1 (DeepSeek 2025) adalah dua model dalam satu kertas:

- **R1-Zero.** Mulai dari model dasar DeepSeek-V3. Tidak ada SFT. Terapkan GRPO secara langsung dengan dua komponen penghargaan: *hadiah akurasi* (berbasis aturan — apakah jawaban akhir diurai ke nomor yang benar / apakah code lulus pengujian unit) dan *format hadiah* (apakah penyelesaian membungkus rantai pemikirannya dalam tag `<think>…</think>`). Selama ribuan langkah, panjang respons rata-rata bertambah dari ~100 menjadi ~10.000 token dan skor benchmark matematika naik hingga mendekati level pratinjau o1. Model belajar bernalar dari awal. Kelemahannya: rantai pemikirannya sering kali tidak dapat dibaca, bahasanya bercampur, dan gayanya kurang sempurna.
- **R1.** Memperbaiki masalah keterbacaan R1-Zero dengan pipeline empat phase:
  1. **SFT cold-start.** Kumpulkan beberapa ribu demonstrasi CoT panjang dengan format bersih. Dibimbing-menyempurnakan model dasar pada mereka. Ini memberikan titik awal yang dapat dibaca.
  2. **GRPO yang berorientasi pada penalaran.** Terapkan GRPO dengan imbalan format+akurasi ditambah imbalan *konsistensi bahasa* untuk mencegah alih code.
  3. **Pengambilan sample penolakan + SFT putaran 2.** Contoh ~600 ribu lintasan penalaran dari pos pemeriksaan RL, simpan hanya lintasan yang memiliki jawaban akhir yang benar dan CoT yang dapat dibaca, dan gabungkan dengan ~200 ribu contoh SFT non-penalaran (penulisan, QA, kognisi mandiri). Sempurnakan kembali basisnya.
  4. **GRPO spektrum penuh.** Satu putaran RL lagi yang mencakup penalaran (imbalan berbasis aturan) dan penyelarasan umum (imbalan berbasis preferensi bermanfaat/tidak berbahaya).

Hasilnya cocok dengan o1 pada AIME dan MATH-500 pada weight terbuka, dan cukup kecil untuk disuling. Makalah yang sama juga merilis enam model padat sulingan (Qwen-1.5B hingga Llama-70B) dengan SFT pada jejak penalaran R1 — tidak ada RL pada siswa. Penyulingan guru RL yang kuat secara konsisten mengalahkan RL dari awal pada skala siswa.

**Mengapa GRPO bukan PPO untuk alasan.** Tiga alasan dalam makalah DeepSeekMath (Februari 2024): (1) tidak ada jaringan nilai untuk dilatih, mengurangi separuh memori; (2) garis dasar kelompok secara alami menangani imbalan akhir lintasan yang jarang yang dihasilkan oleh tugas-tugas penalaran; (3) normalisasi yang dilakukan secara cepat membuat keuntungan dapat dibandingkan antar permasalahan dengan tingkat kesulitan yang sangat berbeda, hal yang tidak dapat dilakukan oleh satu kritikus PPO.

**Bebas penelusuran vs berbasis penelusuran.** Permainan telah bercabang:

- *Permainan informasi sempurna dengan cakrawala panjang* (Go, catur): masih berbasis pencarian. AlphaZero / MuZero mendominasi.
- *Penalaran LLM*: belum ada MCTS yang diproduksi; GRPO dalam peluncuran penuh, best-of-N untuk komputasi inference. Model penghargaan proses (PRM) mengisyaratkan pencarian tingkat langkah ditambahkan kembali.

## BangunKode di `code/main.py` mengimplementasikan **GRPO dalam bentuk mini** — bandit dengan banyak kelompok sample. Algoritmenya sama dengan LLM; hanya kebijakan dan lingkungannya yang lebih sederhana. Ini mengajarkan *loss* dan *keuntungan relatif kelompok*, yang merupakan inovasi tahun 2025.

### Langkah 1: lingkungan pemverifikasi kecil

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

Di GRPO nyata, pemverifikasi menjalankan pengujian unit atau memeriksa kesetaraan matematika.

### Langkah 2: kebijakan: softmax atas K token jawaban per prompt

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

Setara dengan output layer akhir dari LLM yang dikondisikan pada prompt.

### Langkah 3: pengambilan sample kelompok dan keuntungan relatif kelompok

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

Keuntungan relatif grup adalah trik DeepSeek 2024. Tidak diperlukan kritik. "baseline" adalah mean grup, dan normalisasi menggunakan std grup.

### Langkah 4: bandingkan dengan baseline REINFORCE (bebas nilai)

Penyiapan yang sama, komputasi yang sama, REINFORCE biasa. GRPO menyatu lebih cepat dan lebih stabil.

### Langkah 5: amati entropi dan KL

Diagnostik yang sama seperti RLHF: berarti KL sebagai referensi, entropi kebijakan, imbalan seiring waktu. Setelah stabil, training selesai.

## Jebakan

- **Hadiah peretasan melalui permainan verifikator.** GRPO mewarisi risiko RLHF: jika verifikator salah atau dapat dieksploitasi, LLM akan menemukan eksploitasi tersebut. Verifikator yang kuat (beberapa kasus uji, bukti formal) penting.
- **Ukuran grup terlalu kecil.** Varians dasar grup seperti `1/√G`. Di bawah `G = 4`, sinyal keunggulannya berisik; pilihan standarnya adalah `G = 8` hingga `64`.
- **Bias panjang.** Penyelesaian LLM dengan panjang yang berbeda memiliki probabilitas log yang berbeda. Normalisasikan berdasarkan jumlah token, atau gunakan log-prob tingkat urutan, atau potong hingga panjang maksimal.
- **Siklus permainan mandiri murni.** Latihan gaya AlphaZero bisa terjebak dalam putaran dominasi pada permainan jumlah umum. Dimitigasi oleh kelompok lawan yang beragam (permainan liga, Lesson 10).
- **Kebijakan penelusuran tidak cocok.** AlphaZero melatih kebijakan untuk meniru output penelusuran. Jika jaring kebijakan terlalu kecil untuk mewakili distribusi pencarian, training akan terhenti.
- **Compute floor.** MuZero / AlphaZero memerlukan komputasi besar-besaran. Satu ablasi seringkali memakan waktu ratusan jam GPU. Ada demo miniatur (misalnya AlphaZero di Connect Four) untuk pembelajaran.
- **Cakupan verifikasi.** Pengujian unit yang lolos untuk solusi kereta memperkuat bug tersebut. Rancang pemverifikasi yang menangkap kasus-kasus ekstrem.

## Pakai

Lanskap game-RL 2026, berdasarkan domain:

| Domain | Metode dominan |
|--------|-----------------|
| Permainan papan zero-sum dua pemain (Go, catur, shogi) | AlphaZero / MuZero / KataGo |
| Permainan kartu info tidak sempurna (poker) | CFR + pembelajaran mendalam (DeepStack, Libratus, Pluribus) |
| Game Atari / Piksel | Muesli / MuZero / IMPALA-PPO |
| Strategi multipemain besar (Dota, StarCraft) | PPO + permainan mandiri + liga (OpenAI Five, AlphaStar) |
| Penalaran matematika/code LLM | GRPO (DeepSeek-R1, Qwen-RL, replikasi terbuka) |
| Penyelarasan LLM | DPO / RLHF-PPO (bukan GRPO; verifikator adalah preferensi tidak dapat diverifikasi) |
| Robotika | PPO + DR (bukan game-RL, tetapi menggunakan alat gradient kebijakan yang sama) |
| Masalah kombinatorial | Varian AlphaZero (AlphaTensor, AlphaDev) |

*Resep* — permainan mandiri, peningkatan penelusuran, penyulingan kebijakan — mencakup teks, piksel, dan kontrol fisik. GRPO adalah contoh termuda; lebih banyak lagi yang akan datang.

## Kirim

Simpan sebagai `outputs/skill-game-rl-designer.md`:

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## Latihan1. **Mudah.** Implementasikan bandit GRPO di `code/main.py`. Latihlah dengan 2 prompt × 4 token jawaban masing-masing. Bergabunglah dalam <1.000 pembaruan dengan `G=8`.
2. **Medium.** Colokkan PPO (terpotong) dan vanilla REINFORCE. Bandingkan efisiensi sample dan varian imbalan dengan GRPO pada bandit yang sama.
3. **Sulit.** Perluas ke "rantai penalaran" sepanjang 2: agen mengeluarkan dua token dan pemverifikasi memberi penghargaan kepada pasangan tersebut. Ukur bagaimana GRPO menangani penugasan kredit dalam rangkaian dua langkah. (Petunjuk: hitung keuntungan grup per *urutan penuh*, sebarkan ke kedua posisi token.)

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| MCTS | "Pencarian pohon dengan jaring yang dipelajari" | Pencarian Pohon Monte Carlo; Seleksi UCB1/PUCT dengan `(p, v)` sebelumnya. |
| AlfaZero | "Bermain mandiri + MCTS" | Jaring nilai kebijakan dilatih untuk mencocokkan kunjungan MCTS dan hasil pertandingan. |
| MuZero | "Model terpelajar AlphaZero" | Lingkaran yang sama tetapi dalam ruang laten melalui dinamika yang dipelajari. |
| GRPO | "PPO Bebas Kritik" | Optimalisasi Kebijakan Relatif Grup; PERKUAT dengan baseline rata-rata grup + KL. |
| PUCT | "UCB AlphaZero" | `Q + c · p · √N / (1 + N_a)` — menyeimbangkan perkiraan nilai dengan sebelumnya. |
| Bermain sendiri | "Agen vs diri masa lalu" | Standar untuk zero-sum; sinyal training simetris. |
| Pertandingan liga | "Permainan mandiri berbasis populasi" | Pengeksploitasi di masa lalu + saat ini + dijadikan sample sebagai lawan. |
| Hadiah verifikasi | "RL yang Dapat Diverifikasi" | Hadiah berasal dari pemeriksa deterministik (lulus tes, jawaban cocok). |
| Hadiah proses | "PRM" | Nilai setiap langkah penalaran, bukan hanya jawaban akhir. |

## Bacaan Lanjutan

- [Perak dkk. (2017). Menguasai permainan Go tanpa sepengetahuan manusia (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Perak dkk. (2018). Algoritme pembelajaran penguatan umum yang menguasai catur, shogi, dan Go melalui permainan mandiri (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser dkk. (2020). Menguasai Atari, Go, catur dan shogi dengan perencanaan menggunakan model yang dipelajari (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals dkk. (2019). Level Grandmaster di StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Mendorong Batasan Penalaran Matematis dalam Model Bahasa Terbuka (GRPO)](https://arxiv.org/abs/2402.03300) — makalah yang memperkenalkan GRPO dan garis dasar relatif kelompok.
- [DeepSeek-AI (2025). DeepSeek-R1: Memberi Insentif terhadap Kemampuan Penalaran di LLM melalui Pembelajaran Penguatan](https://arxiv.org/abs/2501.12948) — resep R1 empat phase lengkap ditambah ablasi R1-Zero.
- [Brown dkk. (2019). AI manusia super untuk poker multipemain (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — CFR + pembelajaran mendalam dalam skala besar.
- [Tesauro (1995). Pembelajaran Perbedaan Temporal dan TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — makalah yang memulai semuanya.
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — referensi produksi untuk menerapkan GRPO dengan fungsi hadiah khusus.
- [Tim Qwen (2024). Qwen2.5-Math — replikasi GRPO](https://github.com/QwenLM/Qwen2.5-Math) — membuka replikasi resep R1 pada berbagai skala.
- [Sutton & Barto (2018). Bab. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — kerangka buku teks untuk permainan mandiri, pencarian, dan "hadiah yang dirancang" yang dibuat oleh R1 pada skala LLM.
