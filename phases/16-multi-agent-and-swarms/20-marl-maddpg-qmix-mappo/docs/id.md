# MARL — MADDPG, QMIX, MAPPO

> Warisan pembelajaran penguatan koordinasi multi-agen, yang masih menginformasikan sistem agen LLM pada tahun 2026. **MADDPG** (Lowe et al., NeurIPS 2017, arXiv:1706.02275) memperkenalkan Training Terpusat, Eksekusi Terdesentralisasi (CTDE): setiap kritikus melihat status dan tindakan semua agen selama training; pada waktu ujian hanya aktor lokal yang berlari. Bekerja untuk pengaturan kooperatif, kompetitif, dan campuran. **QMIX** (Rashid et al., ICML 2018, arXiv:1803.11485) adalah decomposition nilai dengan jaringan pencampuran monoton; Q per-agen digabungkan menjadi Q gabungan sehingga `argmax` terdistribusi dengan rapi — dominan di StarCraft Multi-Agent Challenge (SMAC). **MAPPO** (Yu et al., NeurIPS 2022, arXiv:2103.01955) adalah PPO dengan fungsi nilai terpusat; "sangat efektif" di dunia partikel, SMAC, Google Research Football, Hanabi dengan penyetelan minimal. Hal ini mendasari kebijakan training bagi tim agen yang harus bertindak secara desentralisasi. MAPPO adalah **dasar default 2026 koperasi-MARL**. Lesson ini dibangun dari mainan dunia grid kecil dan memasukkan tiga ide ke dalam memori otot sebelum menyentuh training agen LLM.

**Type:** Learn
**Language:** Python (stdlib, implementasi kecil bebas NumPy)
**Prerequisites:** Fase 09 (Pembelajaran Penguatan), Fase 16 · 09 (Jaringan Kawanan Paralel)
**Waktu:** ~90 menit

## Masalah

Sistem agen LLM semakin melatih kebijakan untuk koordinasi antar agen: kapan harus menunda, kapan harus bertindak, rekan mana yang harus dihubungi. Literatur yang memberi tahu kamu cara melatih kebijakan tersebut adalah Multi-Agent Reinforcement Learning (MARL), yang mendahului gelombang LLM dan memiliki sejumlah kecil algoritma yang dominan.

Membaca makalah MARL tanpa pola kosakata itu menyakitkan. Training terpusat dengan eksekusi terdesentralisasi (CTDE), decomposition nilai, dan kritik terpusat bukanlah kata kunci — ini adalah jawaban spesifik untuk masalah spesifik:

- RL independen (setiap agen belajar sendiri) tidak stasioner dari sudut pandang masing-masing agen. Buruk.
- RL terpusat (satu agen mengontrol semua) tidak berskala dan melanggar batasan eksekusi.
- CTDE mendapatkan yang terbaik dari keduanya: berlatih dengan informasi global, menerapkan kebijakan lokal.

## Konsep

### Tiga lingkungan yang digunakan surat kabar

- **Dunia Partikel (lingkungan partikel multi-agen).** Fisika 2D sederhana dengan tugas kooperatif/kompetitif. Testbed asli MADDPG.
- **StarCraft Multi-Agent Challenge (SMAC).** Manajemen mikro kooperatif, observasi parsial. Tempat pengujian QMIX. Tindakan diskrit, keadaan berkelanjutan.
- **Google Research Football, Hanabi, MPE.** Garis dasar MAPPO.

Env yang berbeda memiliki tipe tindakan/pengamatan yang berbeda. Algoritme memilih yang sesuai.

### MADDPG (2017) — pola CTDE

Setiap agen `i` memiliki aktor `mu_i(o_i)` yang memetakan pengamatannya menjadi tindakan. Setiap agen juga memiliki kritikus `Q_i(x, a_1, ..., a_n)` yang melihat semua observasi dan semua tindakan selama training. Aktor diperbarui berdasarkan gradient kebijakan terhadap evaluasi kritikus.

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

Mengapa CTDE: pada waktu training, kami mengetahui tindakan semua orang; kami menggunakannya untuk mengurangi varians di setiap kritik. Pada waktu penerapan, setiap agen hanya melihat `o_i` dan menelepon `mu_i(o_i)`.

Mode kegagalan: kritik bertambah dengan N agen (input mencakup semua tindakan). Tidak melampaui ~10 agen tanpa perkiraan.

### QMIX (2018) — decomposition nilai

Koperasi saja. Imbalan global adalah jumlah fungsi monoton dari nilai Q per agen:

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```Jaminan monotonisitas `argmax_a Q_tot` dapat dihitung oleh masing-masing agen yang memilih `argmax_{a_i} Q_i` secara mandiri. Itulah **properti eksekusi terdesentralisasi** yang kamu perlukan. Pada waktu training, jaringan pencampuran menghasilkan `Q_tot` dari Qs per agen.

Mengapa QMIX menang di SMAC: manajemen mikro kooperatif StarCraft memiliki agen yang homogen, obs lokal, penghargaan global — sangat cocok untuk decomposition nilai.

Mode kegagalan: batasan monotonisitas bersifat membatasi; beberapa tugas memiliki struktur penghargaan yang tidak dapat diurai secara monoton (satu agen berkorban untuk tim). Ekstensi (QTRAN, QPLEX) rilekskan ini.

### MAPPO (2022) — default yang diabaikan

PPO Multi-Agen: PPO dengan fungsi nilai terpusat. Setiap agen memiliki kebijakannya sendiri; semua agen berbagi (atau memiliki fungsi nilai per agen) yang melihat status penuh. Yu dkk. 2022 membandingkan MAPPO dengan MADDPG, QMIX, dan ekstensinya pada lima tolok ukur dan menemukan:

- MAPPO cocok atau mengalahkan metode MARL di luar kebijakan di dunia partikel, SMAC, Google Research Football, Hanabi, MPE.
- Diperlukan penyetelan hyperparameter minimal.
- Training yang stabil; dapat direproduksi melalui benih.

Hingga tulisan ini dibuat, masyarakat meremehkan kebijakan MARL. Pada tahun 2026, MAPPO menjadi baseline default untuk koperasi MARL; metode baru apa pun harus mengalahkannya.

### Mengapa insinyur agen LLM harus peduli

Tiga kegunaan langsung:

1. **Training router.** Agen meta memilih sub-agen mana yang menangani tugas. Ini adalah masalah MARL dengan N sub-agen terdesentralisasi dan satu router terpusat. MAPPO cocok.
2. **Munculnya peran.** Dalam simulasi agen generatif, melatih agen untuk mengadopsi peran yang saling melengkapi dari waktu ke waktu merupakan masalah MARL yang terselubung. Decomposition nilai gaya QMIX memaksa saling melengkapi melalui konstruksi.
3. **Penggunaan alat multi-agen.** Saat agen berbagi alat dan bersaing untuk mendapatkan anggaran, melatih mereka melalui CTDE akan menghasilkan kebijakan lokal yang dapat diterapkan dan memperhatikan keterbatasan sumber daya.

Peringatan praktis: pada tahun 2026, sebagian besar sistem agen LLM produksi mendorong kebijakannya sendiri, bukan melatihnya. MARL berguna ketika kamu memiliki (a) banyak data interaksi, (b) sinyal imbalan yang jelas, dan (c) kemauan untuk berinvestasi dalam infrastruktur training.

### CTDE sebagai pola desain di luar RL

Bahkan tanpa training, CTDE adalah pola arsitektur yang berguna:

- Selama *desain*, asumsikan visibilitas tim penuh.
- Pada *runtime*, terapkan eksekusi terdesentralisasi: setiap agen hanya melihat `o_i`.

Pola ini memaksa kamu untuk menjaga status per agen tetap eksplisit dan memikirkan tentang observasi parsial terlebih dahulu. Banyak sistem multi-agen produksi secara diam-diam mengasumsikan status bersama di mana saja — disiplin CTDE mencegah hal tersebut.

### Masalah non-stasioneritas

Ketika beberapa agen belajar secara bersamaan, lingkungan masing-masing agen (yang mencakup kebijakan agen lainnya) tidak stasioner. Bukti RL agen tunggal klasik rusak. Algoritme MARL dalam lesson ini semuanya membahas hal ini:

- MADDPG: kritikus global melihat semua tindakan, sehingga perkiraan nilainya tidak bergerak.
- QMIX: decomposition nilai memindahkan pembelajaran ke ruang Q gabungan di mana optimalitas terdefinisi dengan baik.
- MAPPO: fungsi nilai terpusat mengurangi varians dari perubahan kebijakan pihak lain.

Dalam sistem agen LLM, non-stasioneritas bermanifestasi sebagai "agen saya bekerja bulan lalu, sekarang agen lain di hulu berubah, agen saya berperilaku buruk." Melatih MARL dengan CTDE adalah perbaikan prinsip; perbaikan tingkat cepat lebih cepat tetapi kurang tahan lama.

### Apa yang TIDAK dibahas dalam lesson iniMelatih jaringan sebenarnya adalah topik Fase 09. Lesson ini membangun versi kebijakan skrip yang menunjukkan CTDE, decomposition nilai, dan pola nilai terpusat tanpa pembaruan gradient. Tujuannya adalah untuk menginternalisasi pola sebelum kamu mengambil pustaka MARL lengkap (multi-agen PyMARL, MARLlib, RLlib).

## Build

`code/main.py` menerapkan tiga demonstrasi pola, semuanya di dunia jaringan kooperatif 2 agen kecil:

- Lingkungan: 2 agen di grid 4x4, satu pelet hadiah. Hadiah = 1 jika ada agen yang mencapai pelet; tugas selesai.
- `IndependentAgents` — setiap agen memperlakukan agen lainnya sebagai lingkungan. Dasar.
- `MADDPGStyle` — kritikus terpusat menghitung nilai gabungan; pembaruan kebijakan aktor darinya. Perbaikan kebijakan yang tertulis.
- `QMIXStyle` — decomposition nilai dengan mixer monoton.
- `MAPPOStyle` — fungsi nilai terpusat; pembaruan kebijakan terhadap baseline bersama.

Keempatnya menjalankan episode yang sama dan melaporkan rata-rata langkah menuju tujuan. Varian CTDE berkumpul pada jalur yang lebih pendek dibandingkan garis dasar independen.

Jalankan:

```
python3 code/main.py
```

Hasil yang diharapkan: agen independen rata-rata mengambil ~6 langkah; Varian CTDE berkumpul menuju ~3,5 langkah (optimal untuk grid 4x4 adalah 3). Perbedaan pola muncul meskipun ada kebijakan yang tertulis.

## Pakai

`outputs/skill-marl-picker.md` adalah keterampilan yang memilih algoritme MARL untuk tugas multi-agen tertentu: kooperatif vs kompetitif, homogen vs heterogen, jenis ruang tindakan, skala, sinyal hadiah.

## Kirim

MARL dalam produksi jarang terjadi. Saat kamu menggunakannya:

- **Mulai dengan MAPPO.** Makalah tahun 2022 menetapkan hal ini sebagai dasar; mereproduksinya terlebih dahulu akan menghemat waktu berminggu-minggu untuk mengejar metode yang lebih menarik.
- **Mencatat aliran observasi dan tindakan setiap agen.** Men-debug MARL tanpa jejak per agen tidak ada gunanya.
- **Pisahkan code training dari code eksekusi.** CTDE adalah suatu disiplin ilmu; biarkan jalur eksekusinya benar-benar hanya melihat `o_i`.
- **Peringatan pembentukan hadiah.** MARL sangat sensitif terhadap desain hadiah. Salah satu bug koordinasi sedang terbentuk dan agen belajar untuk mengeksploitasinya. Jalankan tes permusuhan.
- **Untuk agen LLM**, pertimbangkan kebijakan tingkat cepat terlebih dahulu. Hanya berinvestasi dalam training MARL ketika semua data interaksi + sinyal penghargaan + infrastruktur tersedia.

## Latihan

1. Jalankan `code/main.py`. Ukur kesenjangan langkah menuju tujuan antara agen independen dan agen gaya MAPPO. Apakah kesenjangan bertambah atau menyusut pada grid 6x6?
2. Menerapkan varian kompetitif: dua agen, satu pelet, hanya yang pertama mencapai yang mendapat hadiah. Pola manakah yang menangani persaingan dengan baik? MADDPG secara historis.
3. Baca MADDPG (arXiv:1706.02275) Bagian 3. Terapkan aturan pembaruan kritik yang tepat secara simbolis dalam kodesemu dengan kata-kata kamu sendiri.
4. Baca MAPPO (arXiv:2103.01955). Mengapa penulis berpendapat bahwa nilai terpusat + PPO mengalahkan MARL di luar kebijakan dalam tolok ukur mereka? Sebutkan tiga klaim terkuat.
5. Terapkan CTDE sebagai pola desain pada sistem agen LLM hipotetis (misalnya, agen penelitian + ringkasan + pembuat code). Informasi gabungan apa yang tersedia pada waktu desain yang tidak tersedia saat runtime?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| MARL | "RL Multi-Agen" | Pembelajaran penguatan untuk sistem multi-agen. |
| CTDE | "Training Terpusat, Eksekusi Terdesentralisasi" | Berlatih dengan informasi global; diterapkan dengan kebijakan lokal. |
| MADDPG | "DDPG Multi-Agen" | CTDE dengan kritikus per agen melihat semua pengamatan + tindakan. |
| QMIX | "Decomposition nilai" | Pencampuran monotonik per agen Qs. Koperasi. |
| MAPPO | "PPO Multi-Agen" | PPO dengan fungsi nilai terpusat. Garis dasar default tahun 2026. |
| Decomposition nilai | "Jumlah Q individu" | Gabungan Q direpresentasikan sebagai fungsi monoton dari per agen Qs. |
| Non-stasioneritas | "Memindahkan target" | Lingkungan masing-masing agen berubah seiring pembelajaran yang lain. Masalah inti MARL. |
| Sesuai kebijakan / di luar kebijakan | "Belajar dari saat ini / putar ulang" | PPO sesuai kebijakan (MAPPO); DDPG dan Q-learning berada di luar kebijakan. |
| SMAC | "Tantangan Multi-Agen StarCraft" | Tolok ukur manajemen mikro koperasi; Tempat asal QMIX. |

## Bacaan Lanjutan

- [Lowe dkk. — Kritikus Aktor Multi-Agen untuk Lingkungan Campuran Koperasi-Kompetitif](https://arxiv.org/abs/1706.02275) — MADDPG; NeuroIPS 2017
- [Rashid dkk. — QMIX: Faktorisasi Fungsi Nilai Monotonik untuk Pembelajaran Penguatan Multi-Agen yang Mendalam](https://arxiv.org/abs/1803.11485) — QMIX; ICML 2018
- [Yu dkk. — Efektivitas PPO yang Mengejutkan dalam Permainan Multi-Agen Kooperatif](https://arxiv.org/abs/2103.01955) — MAPPO; NeuroIPS 2022
- [postingan blog BAIR di MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — framing hasil MAPPO yang dapat dibaca
- [Repositori SMAC](https://github.com/oxwhirl/smac) — Tantangan Multi-Agen StarCraft
