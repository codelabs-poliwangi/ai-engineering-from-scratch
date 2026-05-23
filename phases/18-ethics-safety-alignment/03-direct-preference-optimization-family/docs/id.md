# Keluarga Optimization Preferensi Langsung

> Rafailov dkk. (2023) menunjukkan optimal RLHF memiliki bentuk tertutup dalam hal data preferensi, sehingga kamu dapat melewati model imbalan eksplisit dan mengoptimalkan kebijakan secara langsung. Wawasan tersebut melahirkan sebuah keluarga — IPO, KTO, SimPO, ORPO, BPO — masing-masing memperbaiki modus kegagalan DPO. Pada tahun 2026, algoritme penyelarasan langsung mengirimkan lebih banyak proses pasca-training terdepan dibandingkan PPO. Namun kurva optimization berlebihan dari Lesson 2 masih berlaku: DAA tidak lolos dari Goodhart, mereka hanya berpindah ke tempat yang digigitnya.

**Type:** Learn
**Language:** Python (stdlib, pembanding loss preferensi enam varian)
**Prerequisites:** Fase 18 · 01 (InstructGPT), Fase 18 · 02 (Reward hacking), Fase 10 · 08 (dasar-dasar DPO)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Turunkan bentuk tertutup DPO dari optimal RLHF-dengan-KL.
- Sebutkan modus kegagalan masing-masing perbaikan IPO, KTO, SimPO, ORPO, BPO di DPO.
- Membedakan "kesenjangan imbalan implisit" dari "kekuatan preferensi" dan menjelaskan mengapa pemetaan identitas IPO penting.
- Jelaskan mengapa Rafailov dkk. (NeurIPS 2024) membuktikan DAA melakukan optimization berlebihan meskipun tidak memiliki RM eksplisit.

## Masalah

Tujuan RLHF (Lesson 1):

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

mempunyai nilai optimum yang diketahui:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

Jadi imbalannya secara implisit ditentukan oleh rasio kebijakan optimal terhadap referensi:

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

Gantikan ini ke dalam kemungkinan preferensi Bradley-Terry dan fungsi partisi `Z(x)` dibatalkan karena hanya bergantung pada `x`. Yang tersisa hanyalah hilangnya parameter kebijakan saja – tidak diperlukan model imbalan. Itu adalah DPO.

Kerutannya: derivasi mengasumsikan hasil optimal dapat dicapai, data preferensi terdistribusi, dan kebijakan referensi adalah jangkar mode sebenarnya. Tak satu pun dari hal ini yang benar. Setiap anggota keluarga memperbaiki asumsi yang dilanggar secara berbeda.

## Konsep

### DPO (Rafailov dkk., 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

Apa yang salah:

- Kesenjangan imbalan tersirat `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` tidak terbatas. Preferensi yang kecil dapat menghasilkan kesenjangan yang sangat besar.
- Loss mendorong masalah log yang dipilih dan ditolak ke arah yang berlawanan. Ini dapat mendorong masalah log absolut yang dipilih ke bawah selama masalah log yang ditolak turun lebih cepat. Inilah fenomena Degraded Chosen Response (Respon Terpilih yang Terdegradasi).
- Preferensi di luar distribusi (pasangan langka langka vs pasangan langka langka) menghasilkan imbalan implisit yang sewenang-wenang.

### IPO (Azar dkk., 2024)

Optimization Preferensi Identitas menggantikan log-sigmoid dengan pemetaan identitas pada probabilitas preferensi. Loss tersebut menjadi kesalahan kuadrat pada target yang dibatasi:

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

Margin dibatasi oleh `1/(2 beta)`. Kekuatan preferensi dan kesenjangan imbalan implisit adalah proporsional. Tidak ada ledakan.

### KTO (Ethayarajh dkk., 2024)

Optimization Kahneman-Tversky menghilangkan struktur berpasangan seluruhnya. Dengan adanya output berlabel tunggal dan sinyal biner "diinginkan" atau "tidak diinginkan", maka dipetakan ke utilitas teori prospek:

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

dengan weight keuntungan dan loss yang berbeda (keengganan rugi). Manfaat: kamu dapat menggunakan data yang tidak berpasangan, yang jauh lebih banyak.

### SimPO (Meng dkk., 2024)

Optimization Preferensi Sederhana menyelaraskan sinyal training dengan pembangkitan. Hapus seluruh kebijakan referensi dan normalkan kemungkinan log berdasarkan panjangnya:

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

dengan margin `gamma` untuk menstabilkan. Normalisasi panjang menghilangkan insentif untuk mengeksploitasi mode kegagalan bias panjang DPO (`y_w` yang lebih panjang menghasilkan kesenjangan log-prob yang lebih besar berdasarkan konstruksi).### ORPO (Hong dkk., 2024)

Optimization Preferensi Rasio Odds menambahkan istilah preferensi ke kemungkinan log negatif SFT standar:

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

Tidak ada kebijakan referensi — istilah SFT adalah pengaturnya. Berlatih dalam satu phase dari model dasar hingga model selaras. Tidak ada pos pemeriksaan SFT terpisah.

### BPO (pengajuan ICLR 2026, OpenReview id=b97EwMUWu7)

Mengidentifikasi masalah Respon Terpilih yang Terdegradasi: DPO mempertahankan peringkat `y_w > y_l` namun log-prob absolut dari `y_w` bisa turun. BPO menambahkan koreksi satu baris yang memberikan penalti terhadap pergerakan ke bawah pada respons yang dipilih. Melaporkan akurasi +10,1% pada Llama-3.1-8B-Instruksikan penalaran matematika melalui DPO.

### Hasil universal: DAA masih terlalu optimal

Rafailov dkk. "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms" (NeurIPS 2024) melatih kebijakan dengan DPO, IPO, SLiC pada beberapa dataset di seluruh anggaran KL. Kurva imbalan emas vs KL mempunyai kesamaan dengan Gao dkk. bentuk puncak dan keruntuhan. Hadiah implisit menanyakan sample di luar distribusi selama training; Regularisasi KL tidak menstabilkan hal ini.

DAA tidak luput dari Goodhart. Mereka mengubah tampilan dari "model imbalan yang terlalu dioptimalkan" menjadi "rasio kebijakan referensi yang terlalu dioptimalkan". Perbaikan universal — data yang lebih baik, ansambel, penghentian lebih awal — berlaku untuk keduanya.

### Memilih di antara mereka (2026)

- Jika kamu memiliki data preferensi berpasangan yang besar: DPO dengan beta konservatif, SimPO jika bias panjangnya terlihat jelas.
- Jika kamu memiliki umpan balik biner yang tidak berpasangan: KTO.
- Jika kamu menginginkan pipeline pipa satu phase dari model dasar: ORPO.
- Jika kamu melihat masalah log terpilih yang terdegradasi di log DPO: BPO.
- Jika kekuatan preferensi sangat bervariasi dan DPO jenuh: IPO.

Setiap lab menjalankan kelimanya dengan baterai dan memilih pemenang per tugas. Tidak ada alasan optimalnya sama untuk penalaran dan keamanan matematika.

## Pakai

`code/main.py` membandingkan enam loss (DPO, IPO, KTO, SimPO, ORPO, BPO) pada dataset preferensi mainan dengan kekuatan preferensi sebenarnya bervariasi berdasarkan pasangan. Setiap loss dioptimalkan terhadap sample 500 pasang yang sama dengan kebijakan softmax kecil. Plot tingkat kemenangan akhir, penyimpangan log-prob yang dipilih, dan penyebaran imbalan implisit per metode.

## Kirim

Lesson ini menghasilkan `outputs/skill-preference-loss-selector.md`. Mengingat statistik dataset (berpasangan vs tidak berpasangan, kekuatan preferensi variabel vs seragam, distribusi panjang) dan target (satu phase atau SFT-lalu-preferensi), rekomendasikan kehilangan preferensi dan laporkan mode kegagalan yang dilindunginya.

## Latihan

1. Jalankan `code/main.py`. Laporkan penurunan masalah log terakhir yang dipilih untuk DPO dan BPO. BPO harus mempertahankan probabilitas absolut pilihan yang lebih tinggi — verifikasi ini.

2. Modifikasi data preferensi sehingga semua pasangan mempunyai kekuatan yang sama. Manakah dari enam metode yang paling kuat? Yang mana yang terdegradasi? Jelaskan keuntungan IPO di sini.

3. Buatlah tanggapan yang ditolak rata-rata 2x lebih lama dari yang dipilih. Tanpa mengubah apa pun, tunjukkan eksploitasi panjang DPO secara numerik dan perbaikan SimPO.

4. Rafailov dkk. (NeurIPS 2024) mengklaim DAA terlalu optimal. Reproduksi versi satu titik: plot divergensi KL yang dipilih-dikurangi-ditolak dan amati optimization berlebihan dalam DPO pada beta besar.

5. Membaca abstrak makalah BPO (OpenReview b97EwMUWu7). Tuliskan koreksi satu baris yang ditambahkan BPO ke DPO. Konfirmasi penerapan di `code/main.py`.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| DPO | "RLHF tanpa model penghargaan" | Loss yang diperoleh dari RLHF optimum bentuk tertutup; hanya parameter kebijakan |
| Hadiah implisit | "rasio log" | `beta * log(pi(y|x) / pi_ref(y|x))` — hadiah tersirat DPO |
| penawaran umum perdana | "DPO terbatas" | Mengganti log-sigmoid dengan identitas; kesenjangan imbalan implisit dibatasi oleh `1/(2 beta)` |
| KTO | "DPO tidak berpasangan" | Utilitas teori prospek pada label tunggal dengan keengganan terhadap loss |
| SimPO | "DPO bebas referensi" | Kemungkinan log + margin yang dinormalisasi panjang; tidak ada kebijakan referensi |
| ORPO | "DPO satu phase" | NLL + istilah preferensi rasio peluang; kereta dari model dasar dalam satu lintasan |
| BPO | "DPO Pelestarian Terpilih" | DPO ditambah penalti untuk mengurangi log-prob absolut dari respons yang dipilih |
| Terdegradasi Terpilih | "yang dipilih turun" | DPO mengurangi log-prob yang dipilih selama ditolak turun lebih cepat |
| DAA | "algoritma penyelarasan langsung" | Metode kehilangan preferensi apa pun yang melewatkan RM | eksplisit

## Bacaan Lanjutan

- [Rafailov dkk. — Optimization Preferensi Langsung (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar dkk. — Paradigma Teori Umum untuk Memahami Pembelajaran dari Preferensi Manusia (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh dkk. — KTO: Penyelarasan Model sebagai Optimization Teori Prospek (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Optimization Pelestarian Perilaku (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov dkk. — Hukum Penskalaan untuk Optimization Berlebihan RM di DAA (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
