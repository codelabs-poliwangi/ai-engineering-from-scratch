# Instruksi-Mengikuti sebagai Sinyal Penyelarasan

> Setiap kritik selanjutnya terhadap RLHF menentang pipeline pipa ini. Sebelum kamu mempelajari bagaimana tekanan optimization mendistorsi proxy, kamu harus melihat proxy tersebut. InstructGPT (Ouyang et al., 2022) mendefinisikan arsitektur referensi: penyesuaian yang diawasi pada pasangan instruksi-respons, model penghargaan yang dilatih pada peringkat preferensi berpasangan, dan PPO terhadap model penghargaan dengan penalti KL terhadap kebijakan SFT. InstructGPT 1.3B lebih disukai daripada GPT-3 175B. Hasil tunggal tersebut adalah alasan mengapa setiap laboratorium perbatasan pada tahun 2026 masih mengirimkan pipa pasca-training berbentuk RLHF.

**Type:** Learn
**Language:** Python (stdlib, pipeline mainan tiga phase)
**Prerequisites:** Phase 10 · 06 (SFT), Phase 10 · 07 (RLHF), Phase 10 · 08 (DPO)
**Waktu:** ~45 menit

## Tujuan Pembelajaran

- Sebutkan tiga tahapan pipeline InstructGPT dan loss yang digunakan di masing-masing tahapan.
- Jelaskan mengapa model 1,3B yang disesuaikan dengan instruksi mengalahkan 175B GPT-3 mentah dalam evaluasi preferensi manusia.
- Nyatakan apa yang dilindungi oleh hukuman KL pada phase 3 dan mengapa penghapusan hukuman tersebut akan mengakibatkan perilaku pencarian modus.
- Menjelaskan penyelarasan pajak dan mitigasi PPO-ptx Ouyang dkk. digunakan untuk melawannya.

## Masalah

Model bahasa terlatih melengkapi teks. Mereka tidak menjawab pertanyaan. Tanyakan GPT-3 "tulis fungsi Python yang membalikkan daftar" dan kamu sering kali mendapatkan kembali prompt lain, karena sebagian besar distribusi training adalah teks web yang dilanjutkan dengan lebih banyak teks web. Model melakukan tugasnya - pekerjaannya salah.

Proksi yang digunakan setiap laboratorium serius untuk memperbaikinya adalah preferensi manusia. Dua penyelesaian diberikan kepada penilai; penilai memilih yang lebih baik; model penghargaan mempelajari penilai. Kemudian putaran RL menggeser kebijakan ke arah output yang mendapat skor tinggi dari model penghargaan. Itulah tesis InstructGPT lengkap dalam tiga kalimat. Sisa makalahnya adalah teknik.

## Konsep

### Phase 1: penyempurnaan yang diawasi (SFT)

Kumpulkan pasangan tanggapan cepat yang tanggapannya adalah apa yang akan ditulis oleh manusia yang bermaksud baik. Ouyang dkk. menggunakan 13 ribu prompt dari pelabel dan OpenAI API. Sempurnakan model dasar pada data ini dengan loss lintas entropi standar.

Apa yang SFT berikan kepada kamu: model sekarang menjawab pertanyaan alih-alih melanjutkannya. Apa yang tidak diberikannya kepada kamu: sinyal apa pun tentang jawaban mana yang lebih disukai penilai ketika beberapa jawaban masuk akal.

### Phase 2: model hadiah (RM)

Untuk setiap prompt, contoh penyelesaian K dari model SFT. Seorang pelabel memberi peringkat pada mereka. Latih model penghargaan yang memberi skor pada pasangan respons cepat sehingga, untuk pasangan yang `y_w` lebih disukai daripada `y_l`:

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

Ini adalah loss preferensi berpasangan Bradley-Terry. RM biasanya diinisialisasi dari model SFT dengan kepala LM diganti dengan kepala scalar.

Model imbalannya kecil: 6B sudah cukup untuk InstructGPT 175B. Mereka juga rapuh – bagian 5 dari makalah ini sebagian besar membahas tentang perilaku peretasan hadiah yang muncul dalam skala kecil.

### Phase 3: PPO dengan penalti KL

Tentukan tujuannya:

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

Maksimalkan dengan PPO. Istilah KL menjaga `pi` agar tidak menyimpang jauh dari kebijakan SFT. Tanpanya, optimizer akan menemukan contoh yang berlawanan — string yang mendapat skor tinggi di bawah RM karena RM tidak pernah melihatnya, bukan karena manusia sebenarnya lebih menyukainya.

Koefisien KL `beta` adalah satu-satunya hyperparameter RLHF yang paling penting. Terlalu rendah: hadiah peretasan. Terlalu tinggi: tidak ada peningkatan dibandingkan SFT.

### Pajak penyelarasanSetelah RLHF, model ini lebih disukai oleh manusia tetapi mengalami kemunduran pada benchmark standar (SQuAD, HellaSwag, DROP). Ouyang dkk. sebut saja ini pajak penyelarasan dan perbaiki dengan PPO-ptx: gabungkan gradient pra-training ke dalam tujuan RL sehingga model tidak lupa cara melakukan tugas-tugas hilir yang tidak pernah dihargai.

```
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx menjadi standar. Anthropic, DeepMind, dan Meta semuanya menggunakan beberapa varian.

### Hasilnya

InstructGPT 1.3B (SFT + RM + PPO-ptx) lebih disukai oleh pemberi label dibandingkan GPT-3 dasar 175B sekitar 70%. Kesenjangan ini melebar karena permintaan pengujian tersembunyi dari lalu lintas produksi. Dua hal yang perlu dibaca dari nomor ini:

1. Keselarasan adalah poros yang berbeda dari kemampuan. Model 175B memiliki kemampuan lebih; model 1.3B memiliki lebih banyak keselarasan; pelabel lebih menyukai yang selaras.
2. Tingkat kemampuan ditentukan oleh model dasar. kamu tidak dapat menggunakan model dasar untuk mengetahui fakta yang tidak pernah dilihatnya.

### Mengapa ini menjadi titik referensi untuk Fase 18

Setiap kritik dalam lesson selanjutnya — peretasan hadiah (Lesson 2), DPO (Lesson 3), penjilatan (Lesson 4), CAI (Lesson 5), agen tidur (Lesson 7), pemalsuan penyelarasan (Lesson 9) — menentang beberapa bagian dari jalur pipa ini. Serangan peretasan hadiah phase 2. DPO runtuh phase 2 dan 3. CAI menggantikan pemberi label manusia. Sycophancy menunjukkan pemberi label adalah sinyal yang bias. Pemalsuan penyelarasan menunjukkan bahwa kebijakan tersebut dapat sepenuhnya melewati phase 3. kamu tidak dapat mengikuti kritik-kritik ini tanpa memikirkan terlebih dahulu.

## Pakai

`code/main.py` menyimulasikan tiga tahapan pada data preferensi mainan. Dasar "kebijakan" adalah koin bias atas tindakan {A, B, C}. SFT Phase 1 meniru tindakan pelabel pada 200 prompt. Phase 2 cocok dengan model hadiah Bradley-Terry dari 500 peringkat berpasangan. Phase 3 menjalankan pembaruan PPO yang disederhanakan dengan penalti KL terhadap kebijakan SFT. kamu dapat menyaksikan peningkatan reward, pertumbuhan divergensi KL, dan penyimpangan kebijakan — dan kamu dapat menonaktifkan istilah KL untuk melihat peretasan reward muncul dalam 50 langkah pembaruan.

Apa yang harus dilihat:

- Hadiahi lintasan dengan `beta = 0.1` vs `beta = 0.0`.
- KL(pi || pi_SFT) atas langkah-langkah training.
- Distribusi tindakan akhir dibandingkan dengan preferensi pemberi label.

## Kirim

Lesson ini menghasilkan `outputs/skill-instructgpt-explainer.md`. Dengan adanya deskripsi jalur pipa RLHF atau abstrak makalah, hal ini mengidentifikasi phase mana yang sedang dimodifikasi, loss apa yang digunakan pada setiap phase, dan apakah terdapat penalti KL atau pengatur yang setara.

## Latihan

1. Jalankan `code/main.py`. Tetapkan `beta = 0.0` dan laporkan distribusi tindakan setelah 200 langkah PPO. Jelaskan perilaku pencarian modus dalam satu paragraf.

2. Ubah model imbalan agar memiliki bias +0,5 untuk tindakan B (bug imbalan yang disimulasikan). Jalankan PPO dengan `beta = 0.1`. Apakah hukuman KL mencegah kebijakan mengeksploitasi bias tersebut? Pada `beta` manakah eksploitasi terlihat?

3. Baca Ouyang dkk. (arXiv:2203.02155) Gambar 1. Reproduksi kurva preferensi pemberi label dengan menjalankan PPO selama 1, 5, 20, 100 langkah dan mengukur preferensi terhadap model SFT.

4. Bagian 4.3 makalah ini melaporkan InstructGPT 1.3B mengalahkan 175B GPT-3 sekitar 70% dari keseluruhan waktu. Mengapa rasionya lebih tinggi pada petunjuk produksi yang tersembunyi dibandingkan dengan petunjuk yang dibuat oleh pemberi label sendiri?

5. Ganti loss PPO dengan DPO (Phase 10 · 08) pada data preferensi yang sama. Bandingkan penyimpangan kebijakan akhir (KL ke SFT) dan imbalan akhir. Metode mana yang lebih maju dalam hadiah yang sesuai?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| SFT | "penyetelan instruksi" | Phase 1: penyempurnaan lintas-entropi pada pasangan respons cepat |
| Model penghargaan | "RM" | Regresor scalar atas (prompt, respon) dilatih dengan Bradley-Terry pada label berpasangan |
| Bradley-Terry | "kehilangan preferensi berpasangan" | -log sigmoid(r_w - r_l); mengurangi peringkat berpasangan menjadi klasifikasi biner |
| Penalti KL | "pengatur" | `beta * KL(pi || pi_SFT)` — menjaga kebijakan RL tetap dekat dengan jangkar SFT |
| PPO-ptx | "PPO dengan campuran pra-training" | Menambahkan sebagian kecil kemungkinan log pra-training ke tujuan PPO untuk mengimbangi pajak penyelarasan |
| Pajak penyelarasan | "regresi RLHF" | Penurunan pasca-RLHF pada tolok ukur standar yang tidak ditargetkan oleh RLHF |
| Preferensi pelabel | "kebenaran dasar" | Contoh pemeringkatan manusia; RM adalah proksi statistik untuk hal ini, bukan untuk "nilai kemanusiaan" |

## Bacaan Lanjutan

- [Ouyang dkk. — Melatih model bahasa untuk mengikuti instruksi dengan input manusia (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — makalah InstructGPT, landasan untuk setiap pipeline RLHF setelahnya
- [Stiennon dkk. — Belajar meringkas dari input manusia (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) — pendahulu RLHF-for-summarization
- [Christiano dkk. — Pembelajaran penguatan mendalam dari preferensi manusia (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) — formulasi RL berbasis preferensi asli
- [Bai dkk. — Melatih Asisten yang Bermanfaat dan Tidak Berbahaya dengan RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) — Perpanjangan HH Anthropic dari pipeline InstructGPT
