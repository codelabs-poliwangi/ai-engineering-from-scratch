# Multi-Agen RL

> RL agen tunggal mengasumsikan lingkungan tidak bergerak. Letakkan dua agen pembelajaran di dunia yang sama dan asumsi tersebut terpatahkan: masing-masing agen adalah bagian dari lingkungan yang lain, dan keduanya berubah. RL multi-agen adalah serangkaian trik untuk membuat pembelajaran menyatu ketika asumsi Markov tidak lagi berlaku.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 9 · 04 (Pembelajaran Q), Fase 9 · 06 (PERKUAT), Fase 9 · 07 (Aktor-Kritikus)
**Waktu:** ~45 menit

## Masalah

Robot yang belajar menavigasi ruangan adalah masalah RL agen tunggal. Tim sepak bola tidak. Lawan AlphaStar vs StarCraft tidak. Pasar agen penawaran tidak. Dua mobil menegosiasikan perhentian empat arah tidak. Banyak dari banyak masalah di dunia nyata yang tidak demikian.

Dalam setiap pengaturan multi-agen, dari sudut pandang salah satu agen, agen lainnya *adalah* bagian dari lingkungan. Ketika mereka belajar dan mengubah perilakunya, lingkungan menjadi tidak stasioner. Properti Markov — "negara bagian berikutnya hanya bergantung pada negara bagian saat ini dan tindakan saya" — dilanggar karena negara bagian berikutnya juga bergantung pada apa yang dipilih oleh agen *lainnya*, dan kebijakan mereka adalah target yang berpindah-pindah.

Hal ini mematahkan bukti konvergensi tabular (jaminan Q-learning mengasumsikan lingkungan stasioner). Hal ini juga merusak RL mendalam yang naif: agen saling mengejar satu sama lain, tidak pernah menyatu pada kebijakan yang stabil. kamu memerlukan teknik khusus multi-agen: training terpusat / eksekusi terdesentralisasi, garis dasar kontrafaktual, permainan liga, permainan mandiri.

Aplikasi tahun 2026: kawanan robot, perutean lalu lintas, armada kendaraan otonom, simulator pasar, sistem LLM multi-agen (Fase 16), dan permainan apa pun dengan lebih dari satu pemain cerdas.

## Konsep

![Empat rezim MARL: indep, kritik terpusat, permainan mandiri, liga](../assets/marl.svg)

**Formalisme: Permainan Markov.** Generalisasi MDP: status `S`, aksi bersama `a = (a_1, …, a_n)`, transisi `P(s' | s, a)`, dan hadiah per agen `R_i(s, a, s')`. Setiap agen `i` memaksimalkan keuntungannya sendiri berdasarkan kebijakannya sendiri `π_i`. Jika imbalannya sama, maka **sepenuhnya kooperatif**. Jika zero-sum, itu adalah **adversarial**. Jika dicampur, hasilnya adalah **jumlah umum**.

**Tantangan inti:**

- **Non-stasioneritas.** `P(s' | s, a_i)` dari pandangan agen `i` bergantung pada `π_{-i}`, yang sedang berubah.
- **Penugasan kredit.** Dengan hadiah bersama, agen mana yang menyebabkannya?
- **Koordinasi eksplorasi.** Agen harus mengeksplorasi strategi yang saling melengkapi, bukan mengeksplorasi keadaan yang sama secara berlebihan.
- **Skalabilitas.** Ruang aksi bersama tumbuh secara eksponensial di `n`.
- **Kemampuan observasi parsial.** Setiap agen hanya melihat observasinya sendiri; negara global tersembunyi.

**Empat rezim dominan:**

**1. Pembelajaran Q independen / PPO independen (IQL, IPPO).** Setiap agen mempelajari Q atau kebijakannya sendiri, memperlakukan orang lain sebagai bagian dari lingkungan. Sederhana, terkadang berhasil (terutama dengan pemutaran ulang pengalaman yang bertindak sebagai trik pemodelan agen yang menghaluskan). Konvergensi teoretis: tidak ada. Dalam praktiknya: baik untuk tugas-tugas yang berpasangan longgar, buruk untuk tugas-tugas yang berpasangan erat.**2. Training terpusat, pelaksanaan terdesentralisasi (CTDE).** Paradigma modern yang paling umum. Setiap agen memiliki *kebijakan* sendiri `π_i` yang mengkondisikan pengamatan lokal `o_i` — standar eksekusi terdesentralisasi pada penerapan. Selama *training*, kritik terpusat `Q(s, a_1, …, a_n)` mengkondisikan keadaan global penuh dan aksi bersama. Contoh:
- **MADDPG** (Lowe et al. 2017): DDPG dengan kritik terpusat per agen.
- **COMA** (Foerster dkk. 2017): dasar kontrafaktual — tanyakan "apa imbalannya jika saya mengambil tindakan `a'`?" — mengisolasi kontribusi saya.
- **MAPPO** / **IPPO** dengan kritik bersama (Yu et al. 2022): PPO dengan fungsi nilai terpusat. Dominan pada tahun 2026 untuk koperasi MARL.
- **QMIX** (Rashid dkk. 2018): decomposition nilai — `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` dengan pencampuran monotonik.

**3. Putar mandiri.** Dua salinan dari agen yang sama diputar satu sama lain. Kebijakan lawan *adalah* kebijakan saya dari cuplikan masa lalu. AlphaGo / AlphaZero / MuZero. OpenAI Lima. Berfungsi paling baik untuk permainan zero-sum; sinyal training-nya simetris.

**4. Pertandingan liga.** Perpanjangan permainan mandiri ke lingkungan jumlah umum / permusuhan: pertahankan populasi kebijakan masa lalu dan saat ini, ambil sample lawan dari liga, latih melawan mereka. Menambahkan pengeksploitasi (berspesialisasi dalam mengalahkan yang terbaik saat ini) dan pengeksploitasi utama (berspesialisasi dalam mengalahkan pengeksploitasi). Bintang Alpha (StarCraft II). Dibutuhkan ketika permainan mengakui siklus strategi "batu-kertas-gunting".

**Komunikasi.** Izinkan agen untuk mengirim pesan yang dipelajari `m_i` satu sama lain. Bekerja dalam pengaturan kooperatif. Foerster dkk. (2016) menunjukkan bahwa komunikasi antar agen yang dapat dibedakan dapat dilatih secara end-to-end. Sistem multi-agen berbasis LLM saat ini (Phase 16) pada dasarnya berkomunikasi dalam bahasa alami.

## Build

Lesson ini menggunakan GridWorld 6×6 dengan dua agen kooperatif. Mereka memulai dari sudut yang berlawanan dan harus mencapai tujuan bersama. Hadiah bersama: `-1` per langkah saat salah satu agen masih bergerak, `+10` saat keduanya tiba. Lihat `code/main.py`.

### Langkah 1: lingkungan multi-agen

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

Ruang tindakan *bersama* adalah `|A|² = 16`. Keadaan dunia berada pada dua posisi.

### Langkah 2: pembelajaran Q mandiri

Setiap agen menjalankan tabel Q-nya sendiri yang dikunci pada status gabungan. Di setiap langkah: keduanya memilih tindakan ε-serakah, mengumpulkan transisi bersama, masing-masing memperbarui Q-nya sendiri dengan hadiah bersama.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

Kerjakan tugas ini karena imbalannya padat dan selaras. Gagal pada tugas-tugas yang sangat terkait (misalnya, ketika satu agen harus *menunggu* agen lainnya).

### Langkah 3: Q terpusat dengan pembaruan nilai terurai

Gunakan satu Q atas tindakan bersama `Q(s, a_1, a_2)`. Pembaruan dari hadiah bersama. Desentralisasi pada pelaksanaan dengan meminggirkan: `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. Memperdagangkan ruang aksi bersama yang eksponensial untuk pandangan global yang *benar*.

### Langkah 4: permainan mandiri sederhana (agen 2 musuh)

Agen yang sama, dua peran. Latih agen A melawan agen B; setelah episode `K`, salin weight A ke B. Training simetris, kemajuan konsisten. Resep AlphaZero dalam bentuk mini.

## Jebakan- **Permainan ulang non-stasioner.** Pengalaman pemutaran ulang dengan agen independen lebih buruk daripada agen tunggal karena transisi lama dihasilkan oleh lawan yang sudah usang. Cara mengatasinya: memberi label ulang atau menimbang berdasarkan kekinian.
- **Ambiguitas penetapan kredit.** Hadiah yang dibagikan setelah episode yang panjang; tidak ada cara yang jelas untuk mengatakan agen mana yang berkontribusi. Cara mengatasinya: garis dasar kontrafaktual (COMA), atau pembentukan imbalan per agen.
- **Penyimpangan / kejar-kejaran kebijakan.** Respon terbaik tiap agen berubah seiring update masing-masing. Cara mengatasinya: kritik terpusat, learning rate lambat, atau dibekukan satu per satu.
- **Hadiah peretasan melalui koordinasi.** Agen menemukan eksploitasi terkoordinasi yang tidak diantisipasi oleh perancang. Agen lelang berkumpul untuk menawar nol. Cara mengatasinya: desain hadiah yang hati-hati, batasan perilaku.
- **Redundansi eksplorasi.** Kedua agen mengeksplorasi pasangan tindakan-status yang sama. Cara mengatasinya: bonus entropi per agen, atau pengondisian peran.
- **Siklus liga.** Permainan mandiri murni bisa terjebak dalam siklus dominasi. Cara mengatasinya: bermain liga dengan lawan yang beragam.
- **Contoh ledakan.** `n` agen × ruang negara × aksi bersama. Perkiraan dengan perkiraan fungsi; ruang tindakan terfaktor (satu kepala output kebijakan per agen).

## Pakai

Peta penerapan MARL 2026:

| Domain | Metode | Catatan |
|--------|--------|-------|
| Navigasi / manipulasi kooperatif | MAPPO / QMIX | CTDE; kritik bersama + aktor yang terdesentralisasi. |
| Permainan dua pemain (catur, Go, poker) | Bermain sendiri dengan MCTS (AlphaZero) | Jumlah nol; training simetris. |
| Multiplayer kompleks (Dota, StarCraft) | Pertandingan liga + latihan awal imitasi | OpenAI Lima, AlphaStar. |
| Armada kendaraan otonom | CTDE MAPPO / PPO dengan attention | Obs parsial; ukuran tim yang bervariasi. |
| Pasar lelang | Keseimbangan teori permainan + RL | RL bidang rata-rata ketika `n` → ∞. |
| Sistem multi-agen LLM (Phase 16) | Komunikasi bahasa alami + pengondisian peran | Perulangan RL pada layer perencanaan agen. |

Pada tahun 2026, area pertumbuhan terbesar MARL berbasis LLM: sekumpulan agen model bahasa yang bernegosiasi, berdebat, membangun perangkat lunak. RL muncul sebagai optimization preferensi pada output *tingkat lintasan*, bukan tingkat token (Fase 16 · 03).

## Kirim

Simpan sebagai `outputs/skill-marl-architect.md`:

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## Latihan

1. **Mudah.** Latih Q-learning mandiri di koperasi 2 agen GridWorld. Berapa episode hingga mean return > 0? Plot kurva pembelajaran bersama.
2. **Sedang.** Tambahkan tugas "koordinasi": sasaran tercapai hanya jika kedua agen melangkah ke sana pada giliran yang sama. Apakah Q independen masih konvergen? Apa yang rusak?
3. **Sulit.** Menerapkan kritik terpusat untuk training gaya MAPPO dan membandingkan kecepatan konvergensi dengan PPO independen pada tugas koordinasi.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Permainan Markov | "MDP multi-agen" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; setiap agen memiliki imbalannya masing-masing. |
| CTDE | "Training terpusat, pelaksanaan terdesentralisasi" | Kritikus bersama pada saat training; kebijakan masing-masing agen hanya menggunakan obs lokal. |
| IPPO | "PPO Mandiri" | Setiap agen menjalankan PPO secara terpisah. Garis dasar sederhana; sering diremehkan. |
| MAPPO | "PPO multi-agen" | PPO dengan fungsi nilai terpusat yang dikondisikan pada keadaan global. |
| QMIX | "Decomposition nilai monotonik" | `Q_tot = f_monotone(Q_1, …, Q_n)` memungkinkan argmax terdesentralisasi. |
| KOMA | "Multi-agen kontrafaktual" | Keuntungan = Q saya dikurangi Q yang diharapkan meminggirkan tindakan saya. |
| Bermain sendiri | "Agen vs diri masa lalu" | Agen tunggal, dua peran; standar untuk permainan zero-sum. |
| Pertandingan liga | "Training penduduk" | Tembolok kebijakan masa lalu, ambil sample lawan dari kumpulan; menangani siklus strategi. |

## Bacaan Lanjutan

- [Lowe dkk. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — CTDE dengan kritik terpusat.
- [Foerster dkk. (2017). Gradient Kebijakan Multi-Agen Kontrafaktual (COMA)](https://arxiv.org/abs/1705.08926) — garis dasar kontrafaktual untuk penugasan kredit.
- [Rashid dkk. (2018). QMIX: Faktorisasi Fungsi Nilai Monotonik](https://arxiv.org/abs/1803.11485) — decomposition nilai dengan monotonisitas.
- [Yu dkk. (2022). Efektivitas PPO yang Mengejutkan dalam Permainan Multi-Agen Kooperatif (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO ternyata sangat kuat bagi MARL.
- [Vinyals dkk. (2019). Level Grandmaster di StarCraft II menggunakan pembelajaran penguatan multi-agen (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — permainan liga dalam skala besar.
- [Perak dkk. (2017). Menguasai permainan Go tanpa sepengetahuan manusia (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — permainan mandiri murni dalam permainan zero-sum.
- [Sutton & Barto (2018). Bab. 15 - Ilmu Saraf & Bab. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — mencakup pembahasan singkat buku teks tentang pengaturan multi-agen dan masalah non-stasioneritas yang dirancang untuk dipecahkan oleh CTDE.
- [Zhang, Yang & Başar (2021). Pembelajaran Penguatan Multi-Agen: Tinjauan Selektif](https://arxiv.org/abs/1911.10635) — survei yang mencakup MARL kooperatif, kompetitif, dan campuran dengan hasil konvergensi.
