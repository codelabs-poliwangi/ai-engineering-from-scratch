# Pemodelan Hadiah & RLHF

> Manusia tidak dapat menulis fungsi penghargaan untuk "respon asisten yang baik", namun mereka dapat membandingkan dua respons dan memilih yang lebih baik. Sesuaikan model penghargaan dengan perbandingan tersebut, lalu gunakan model bahasa yang sesuai dengan model tersebut. Christiano 2017. InstructGPT 2022. Resep yang mengubah GPT-3 menjadi ChatGPT. Pada tahun 2026 sebagian besar digantikan oleh DPO – tetapi model mentalnya tetap ada.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 5 · 05 (Sentimen), Fase 9 · 08 (PPO)
**Waktu:** ~45 menit

## Masalah

kamu melatih model bahasa tentang tujuan prediksi token berikutnya. Itu menulis tata bahasa Inggris. Ia juga berbohong, mengoceh, dan tidak mau menolak. kamu tidak dapat memperbaikinya dengan lebih banyak training awal — teks web adalah masalahnya, bukan obatnya.

kamu menginginkan *hadiah scalar* yang menyatakan "respons A lebih baik daripada respons B untuk instruksi X." Menulis fungsi penghargaan itu dengan tangan adalah hal yang mustahil. "Kebermanfaatan" bukanlah ekspresi bentuk tertutup atas token. Namun manusia dapat membandingkan dua output dan menandai preferensinya. Itu murah untuk dikumpulkan dalam skala besar.

RLHF (Christiano et al. 2017; Ouyang et al. 2022) mengubah preferensi menjadi model penghargaan, kemudian mengoptimalkan LM melalui PPO terhadap penghargaan tersebut. Dalam tiga langkah: SFT → RM → PPO. Ini adalah resep yang mengirimkan ChatGPT, Claude, Gemini, dan semua LLM selaras lainnya pada tahun 2023–2025.

Pada tahun 2026 langkah PPO sebagian besar digantikan oleh DPO (Phase 10 · 08) karena lebih murah dan hampir sama baiknya untuk penyesuaian penyelarasan. Namun bagian *model penghargaan* masih mendasari setiap sample Best-of-N, setiap pipeline RL-dari-penghargaan yang dapat diverifikasi, dan setiap model penalaran yang menggunakan model imbalan proses. Pahami RLHF dan kamu memahami keseluruhan tumpukan penyelarasan.

## Konsep

![RLHF tiga phase: SFT, training RM pada preferensi berpasangan, PPO dengan penalti KL](../assets/rlhf.svg)

**Phase 1: Supervised Fine-Tuning (SFT).** Mulai dari model dasar yang telah dilatih sebelumnya. Sempurnakan demonstrasi perilaku target yang ditulis oleh manusia (respon mengikuti instruksi, balasan bermanfaat, dll.). Hasilnya: model `π_SFT` yang *bias terhadap perilaku baik* namun tetap memiliki ruang tindakan yang tidak terbatas.

**Phase 2: Training Model Penghargaan.**

- Kumpulkan pasangan tanggapan `(y_+, y_-)` ke prompt `x`, yang diberi label oleh manusia sebagai "y_+ lebih disukai daripada y_-."
- Latih model hadiah `R_φ(x, y)` untuk memberikan skor yang lebih tinggi kepada `y_+`.
- Kalah: **logistik berpasangan Bradley-Terry**:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ adalah sigmoidnya. Perbedaan imbalan menyiratkan log-odds preferensi. BT telah menjadi standar sejak tahun 1952 (Bradley-Terry) dan merupakan pilihan dominan di RLHF modern.

- `R_φ` biasanya diinisialisasi dari model SFT dengan kepala scalar di atas. Tulang punggung Transformer yang sama; satu layer linier mengeluarkan hadiahnya.

**Phase 3: PPO melawan RM dengan penalti KL.**

- Inisialisasi kebijakan yang dapat dilatih `π_θ` dari `π_SFT`. Simpan *referensi* yang dibekukan `π_ref = π_SFT`.
- Hadiah di akhir respons `y`:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  Penalti KL mencegah `π_θ` berpindah secara sewenang-wenang dari `π_SFT` — ini adalah *pengatur*, bukan wilayah yang sulit dipercaya. `β` biasanya `0.01`-`0.05`.
- Jalankan PPO (Lesson 08) dengan hadiah ini. Keuntungan dihitung pada lintasan tingkat token, tetapi RM hanya memberi skor pada respons penuh.**Mengapa KL?** Tanpanya, PPO akan dengan senang hati menemukan strategi peretasan hadiah — RM hanya dilatih mengenai penyelesaian dalam distribusi. Respons di luar distribusi mungkin mendapat skor lebih tinggi daripada respons yang ditulis oleh manusia. KL menempatkan `π_θ` di dekat manifold tempat RM dilatih. Ini adalah satu-satunya tombol terpenting dalam RLHF.

**2026 status:**

- **DPO** (Rafailov 2023): aljabar bentuk tertutup meruntuhkan Phase 2+3 menjadi satu loss yang diawasi atas data preferensi. No RM, no PPO. Kualitas yang sama pada tolok ukur penyelarasan untuk sebagian kecil komputasi. Tercakup dalam Fase 10 · 08.
- **GRPO** (DeepSeek 2024–2025): PPO dengan garis dasar relatif grup, bukan kritik, imbalan dari *verifikator* (pengoperasian code / kecocokan jawaban matematika) alih-alih RM yang dilatih oleh manusia. Dominan untuk model penalaran. Tercakup dalam Fase 9 · 12.
- **Model imbalan proses (PRM):** memberi skor pada solusi parsial (setiap langkah penalaran), yang digunakan dalam varian RLHF dan GRPO untuk penalaran.
- **AI Konstitusional / RLAIF:** menggunakan LLM yang selaras untuk menghasilkan preferensi, bukan manusia. Menskalakan anggaran preferensi.

## Build

Lesson ini menggunakan "prompt" dan "respon" sintetik kecil yang direpresentasikan sebagai string. RM adalah pencetak skor linier atas representasi sekantong token. Tidak ada LLM yang sebenarnya — *bentuk* pipeline pipa yang penting, bukan skalanya. Lihat `code/main.py`.

### Langkah 1: data preferensi sintetis

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

Dalam RLHF sebenarnya hal ini digantikan oleh pelabel manusia. Bentuknya — `(prompt, preferred_response, rejected_response)` — identik.

### Langkah 2: Model penghargaan Bradley-Terry

Skor linier: `R(x, y) = w · bag(y)`. Latih untuk meminimalkan kehilangan log berpasangan BT:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

Setelah beberapa ratus pembaruan, `w` memberikan weight positif pada token yang bersifat baik dan negatif pada token yang buruk.

### Langkah 3: Kebijakan seperti PPO di atas RM

Kebijakan mainan kami menghasilkan satu token dari kosakata. Kami menilai token di bawah RM, menghitung `log π_θ(token | prompt)`, menambahkan penalti KL-ke-referensi, dan menerapkan pengganti PPO yang terpotong.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### Langkah 4: pantau KL

Lacak rata-rata `KL(π_θ || π_ref)` setiap pembaruan. Jika kebijakan tersebut melampaui `~5-10` maka kebijakan tersebut telah menyimpang jauh dari `π_SFT` — `β` yang lebih rendah akan meningkat atau peretasan hadiah akan dimulai. Ini adalah diagnostik teratas dalam RLHF nyata.

### Langkah 5: resep produksi dengan TRL

Setelah kamu memahami alur mainan, berikut adalah loop yang sama seperti yang ditulis oleh pengguna perpustakaan sebenarnya. [TRL] Hugging Face (https://huggingface.co/docs/trl) adalah implementasi referensi — `RewardTrainer` untuk Phase 2 dan `PPOTrainer` (dengan referensi KL bawaan) untuk Phase 3.

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

Tiga hal yang dilakukan perpustakaan untuk kamu. `adap_kl_ctrl=True` menerapkan jadwal adaptif-β: jika KL yang diamati melebihi `target_kl`, β menjadi dua kali lipat; jika di bawah setengah, β menjadi setengah. Model referensi dibekukan berdasarkan konvensi — kamu tidak boleh berbagi parameter secara tidak sengaja dengan `policy`. Dan value head berada pada tulang punggung yang sama dengan kebijakan (`AutoModelForCausalLMWithValueHead` melampirkan kepala MLP scalar), itulah sebabnya TRL melaporkan `policy/kl` dan `value/loss` secara terpisah.

## Jebakan- **Optimization berlebihan / peretasan hadiah.** RM tidak sempurna; `π_θ` menemukan penyelesaian permusuhan yang mendapat skor tinggi tetapi buruk. Gejala: imbalan meningkat tanpa batas waktu sementara skor penilaian manusia tidak berubah atau menurun. Cara mengatasinya: berhenti lebih awal, naikkan `β`, perluas training data RM.
- **Peretasan durasi.** RM yang dilatih untuk memberikan respons yang membantu sering kali secara implisit memberi imbalan pada durasi. Kebijakan ini belajar untuk memberikan tanggapan. Remediasi: imbalan yang dinormalisasi panjang, atau RLAIF dengan RM yang sadar akan panjangnya.
- **RM terlalu kecil.** RM harus setidaknya sebesar polis. RM yang kecil tidak bisa menilai output kebijakan dengan tepat.
- **Penyetelan KL.** β terlalu rendah → drift dan peretasan hadiah. Terlalu tinggi β → kebijakan hampir tidak berubah. Trik standarnya adalah *adaptif* β yang menargetkan KL tetap per langkah.
- **Kebisingan data preferensi.** ~30% label manusia menimbulkan gangguan atau ambigu. Kalibrasi dengan melatih RM pada data yang difilter sesuai kesepakatan atau gunakan suhu di BT.
- **Masalah di luar kebijakan.** Data PPO sedikit di luar kebijakan setelah periode pertama. Pantau pecahan klip seperti pada Lesson 08.

## Pakai

RLHF pada tahun 2026 berlapis:

| Layer | Sasaran | Metode |
|-------|--------|--------|
| Mengikuti instruksi, kegunaan, tidak membahayakan | Penyelarasan | DPO (Phase 10 · 08) lebih disukai daripada RLHF-PPO. |
| Kebenaran penalaran (matematika, code) | Kemampuan | GRPO dengan imbalan verifikator (Phase 9 · 12). |
| Tugas multi-langkah cakrawala panjang | Agen | PPO / GRPO dengan model imbalan proses melalui langkah-langkah. |
| Perilaku keselamatan/penolakan | Keamanan | RLHF-PPO dengan RM keamanan terpisah, atau AI Konstitusional. |
| Best-of-N pada inference | Penyelarasan cepat | Gunakan RM pada waktu dekode; tidak diperlukan training kebijakan. |
| Penyulingan hadiah | Komputasi inference | Latih "hadiah kepala" kecil di atas LM beku. |

RLHF adalah *metode* pada tahun 2022–2024. Pada tahun 2026, jalur pipa penyelarasan produksi adalah yang mengutamakan DPO, dan hanya PPO untuk langkah-langkah intensif RM atau kritis terhadap keselamatan.

## Kirim

Simpan sebagai `outputs/skill-rlhf-architect.md`:

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## Latihan

1. **Mudah.** Latih model hadiah Bradley-Terry di `code/main.py` pada 500 pasangan preferensi sintetis. Ukur akurasi berpasangan pada 100 pasang yang dibagikan. Harus melebihi 90%.
2. **Medium.** Jalankan loop mainan PPO-RLHF dengan `β ∈ {0.0, 0.1, 1.0}`. Untuk masing-masing, plot skor RM vs KL untuk referensi atas pembaruan. Yang mana yang menjalankan reward-hack?
3. **Sulit.** Menerapkan DPO (loss kemungkinan preferensi bentuk tertutup) pada data preferensi yang sama dan membandingkannya dengan pipeline RLHF-PPO dalam komputasi yang digunakan dan skor RM akhir yang dicapai.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| RLHF | "Penyelarasan RL" | Pipa SFT + RM + PPO tiga phase (Christiano 2017, Ouyang 2022). |
| Model Hadiah (RM) | "Jaring pencetak gol" | Fungsi scalar yang dipelajari sesuai dengan preferensi berpasangan melalui Bradley-Terry. |
| Bradley-Terry | "Loss logistik berpasangan" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; tujuan RM standar. |
| Penalti KL | "Tetap di dekat referensi" | `β · KL(π_θ || π_ref)` sebagai hadiah; pengatur anti-peretasan hadiah. |
| Hadiah peretasan | "Hukum Goodhart" | Kebijakan mengeksploitasi kelemahan RM; gejala : reward up, human eval flat. |
| RLAIF | "Preferensi berlabel AI" | RLHF dimana label berasal dari LM lain, bukan manusia. |
| PRM | "Model Penghargaan Proses" | Mencetak sebagian langkah penalaran; digunakan dalam pipeline penalaran. |
| AI Konstitusional | "Metode Antropis" | Preferensi yang dihasilkan AI dipandu oleh aturan eksplisit. |

## Bacaan Lanjutan- [Christiano dkk. (2017). Pembelajaran Penguatan Mendalam dari Preferensi Manusia](https://arxiv.org/abs/1706.03741) — makalah yang memulai RLHF.
- [Ouyang dkk. (2022). InstructGPT — Melatih model bahasa untuk mengikuti instruksi dengan input manusia](https://arxiv.org/abs/2203.02155) — resep di balik ChatGPT.
- [Stiennon dkk. (2020). Belajar meringkas dengan input manusia](https://arxiv.org/abs/2009.01325) — RLHF sebelumnya untuk meringkas.
- [Rafailov dkk. (2023). Optimization Preferensi Langsung](https://arxiv.org/abs/2305.18290) — DPO; default pasca-RLHF pada tahun 2026.
- [Bai dkk. (2022). AI Konstitusional: Tidak Berbahaya dari Umpan Balik AI](https://arxiv.org/abs/2212.08073) — RLAIF dan putaran kritik diri.
- [Makalah RLHF antropis (Bai dkk. 2022). Melatih Asisten yang Bermanfaat dan Tidak Berbahaya](https://arxiv.org/abs/2204.05862) — makalah HH.
- [Perpustakaan Hugging Face TRL](https://huggingface.co/docs/trl) — produksi `RewardTrainer` dan `PPOTrainer`. Baca sumber pelatih untuk mengetahui detail KL adaptif dan kepala nilai.
- [Hugging Face — Mengilustrasikan Pembelajaran Penguatan dari Umpan Balik Manusia](https://huggingface.co/blog/rlhf) oleh Lambert, Castricato, von Werra, Havrilla — panduan kanonik dari pipeline tiga phase dengan diagram.
- [von Werra dkk. (2020). TRL: Pembelajaran Penguatan Transformer](https://github.com/huggingface/trl) — perpustakaan; `examples/` memiliki skrip RLHF ujung ke ujung untuk Llama, Mistral, dan Qwen.
- [Sutton & Barto (2018). Bab. 17.4 — Merancang Sinyal Hadiah](http://incompleteideas.net/book/RLbook2020.pdf) — pandangan hipotesis hadiah; prasyarat penting untuk memikirkan tentang peretasan hadiah.
