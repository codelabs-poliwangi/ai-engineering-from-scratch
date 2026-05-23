# Optimization Swarm untuk LLM (PSO, ACO)

> Optimization yang terinspirasi oleh bio membuat LLM kembali lagi. **LMPSO** (arXiv:2504.09247) menggunakan PSO yang kecepatan setiap partikelnya cepat dan LLM menghasilkan kandidat berikutnya; bekerja dengan baik pada output urutan terstruktur (ekspresi matematika, program). **Model Swarms** (arXiv:2410.11163) memperlakukan setiap pakar LLM sebagai partikel PSO pada manifold weight model dan melaporkan **perolehan rata-rata 13,3%** pada 12 baseline pada 9 set data hanya dengan 200 instance. **SwarmPrompt** (ICAART 2025) menggabungkan PSO + Gray Wolf untuk optimization cepat. **AMRO-S** (arXiv:2603.12933) adalah spesialis feromon yang terinspirasi ACO untuk perutean LLM multi-agen — **percepatan 4,7x**, bukti perutean yang dapat diinterpretasikan, pembaruan asinkron dengan gerbang kualitas yang memisahkan inference dari pembelajaran. Lesson ini mengimplementasikan PSO pada ruang parameter prompt dan ACO pada perutean agen, mengukur mengapa algoritma klasik ini sesuai dengan era LLM, dan kapan tidak.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 16 · 09 (Jaringan Swarm Paralel), Fase 16 · 14 (Konsensus dan BFT)
**Waktu:** ~75 menit

## Masalah

kamu memiliki prompt yang mendapat skor 62% pada evaluasi tugas kamu. kamu ingin memperbaikinya. Langkah naifnya adalah penyesuaian manual bebas gradient, yang skalanya buruk. Pembelajaran penguatan memerlukan sinyal penghargaan dan peluncuran yang cukup untuk dilatih. Backprop melalui prompt sebenarnya tidak mungkin dilakukan - prompt adalah string terpisah, bukan parameter yang dapat dibedakan.

Optimization klasik yang terinspirasi oleh bio — PSO untuk ruang pencarian berkelanjutan, ACO untuk pemilihan jalur — dirancang tepat untuk rezim ini: bebas gradient, berbasis populasi, murah per evaluasi. Pasangkan dengan LLM untuk langkah pencarian bebas gradient, dan kamu mendapatkan optimizer yang sangat praktis.

Pola yang sama berlaku untuk *perutean* agen dalam sistem multi-agen. Jejak feromon gaya ACO mencatat agen mana yang bekerja paling baik pada jenis tugas apa, memungkinkan router mengeksploitasi jejak tersebut, dan meluruhkan feromon sehingga rute dapat ditemukan kembali.

## Konsep

### Penyegaran PSO (Kennedy & Eberhart 1995)

Optimization Kawanan Partikel: populasi partikel dalam ruang pencarian berkelanjutan. Setiap partikel memiliki posisi `x_i` dan kecepatan `v_i`. Setiap iterasi:

```
v_i <- w * v_i + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)
x_i <- x_i + v_i
evaluate fitness(x_i)
update p_best_i if improved
update g_best if global best
```

Dimana `p_best` adalah yang terbaik dari partikel, `g_best` adalah yang terbaik dari gerombolan, `w, c1, c2` adalah inersia + weight kognitif + sosial, `r1, r2` adalah faktor acak.

### PSO pada output LLM — LMPSO

arXiv:2504.09247 mengadaptasi PSO untuk output terstruktur yang dihasilkan LLM (ekspresi matematika, program). Setiap partikel merupakan kandidat output. Velocity adalah *prompt* yang menjelaskan cara memodifikasi output saat ini ke arah yang terbaik bagi pribadi/global. LLM menghasilkan output baru dari prompt kecepatan. "Inersia" kecepatan adalah prompt seperti "buat perubahan kecil secara bertahap".

Ini bekerja dengan baik ketika:
- Outputnya terstruktur (dapat diurai, dapat dievaluasi).
- Kebugaran otomatis (uji coba, evaluasi aritmatika).
- Populasinya kecil (~10-30 partikel) sehingga total panggilan LLM tetap dapat dikelola.

Ini tidak berfungsi dengan baik ketika kebugaran memerlukan tinjauan manusia — biaya per-iterasi menjadi mahal.

### Model KawananarXiv:2410.11163 mengambil PSO dari layer output dan masuk ke layer *model*. Setiap "partikel" adalah LLM ahli (parameter). Kawanan tersebut menggerakkan parameter ke arah yang terbaik secara kolektif melalui pembaruan bebas gradient. Dilaporkan: rata-rata perolehan sebesar 13,3% pada 12 garis dasar pada 9 dataset, dengan hanya 200 instans per iterasi.

Wawasan utamanya adalah bahwa model pakar LLM sudah berada di dekatnya dalam manifold parameter bersama (weight adaptor, delta LoRA). PSO pada subruang berdimensi rendah ini murah dan efektif.

### Penyegaran ACO (Dorigo 1992)

Optimization Koloni Semut: semut melintasi grafik; setiap jalur memiliki jejak feromon. Probabilitas perpindahan semut berdasarkan kekuatan feromon. Semut yang menyelesaikan tugasnya menyimpan feromon sebanding dengan kualitas larutan. Feromon membusuk seiring waktu.

### AMRO-S — ACO untuk perutean agen

arXiv:2603.12933 menggunakan ACO untuk perutean multi-agen. Setiap jenis tugas adalah "tujuan"; setiap agen adalah rute yang mungkin. Feromon memperkuat rute yang menghasilkan output yang baik. Kontribusi utama:

- **Bukti perutean yang dapat ditafsirkan.** Kekuatan feromon adalah sinyal yang dapat dibaca manusia.
- **Pembaruan asinkron dengan gerbang kualitas.** Feromon diperbarui hanya setelah pemeriksaan kualitas berhasil, memisahkan inference dari pembelajaran.
- **percepatan 4,7x** pada tolok ukur perutean multi-agen.

Gerbang kualitas itu penting: tanpanya, agen yang cepat namun salah akan menghasilkan feromon, dan sistem akan mengunci rute yang buruk.

### Kapan menggunakan PSO / ACO untuk LLM

**Gunakan PSO ketika:**
- Ruang pencarian bersifat kontinu atau dipetakan ke parameter kontinu (embedding cepat, weight LoRA, parameter pembuatan numerik).
- Fitness murah dan otomatis.
- Populasi bisa kecil (10-30).

**Gunakan ACO ketika:**
- kamu memiliki masalah perutean atau pemilihan jalur.
- Keputusan diperkuat seiring berjalannya waktu (jenis tugas yang sama muncul kembali).
- kamu memerlukan bukti yang dapat ditafsirkan untuk keputusan perutean.

**Jangan gunakan keduanya ketika:**
- Kebugaran memerlukan tinjauan manusia (terlalu mahal per iterasi).
- Ruang pencarian bersifat diskrit dan kombinatorial sehingga PSO tidak tercakup (sebagai gantinya gunakan algoritma genetika).
- Keputusan real-time memerlukan latensi yang ketat (PSO/ACO menyatu secara perlahan dibandingkan dengan heuristik single-pass).

### Mengapa bio-terinspirasi masih menang

Metode berbasis gradient memerlukan sinyal yang dapat dibedakan. Output LLM dan keputusan perutean tidak dapat dibedakan dengan mudah. Metode gradient semu (router yang dipelajari dengan penguatan, tuner cepat gaya DPO) berfungsi tetapi memerlukan training yang mahal.

PSO dan ACO hanya memerlukan fungsi *evaluator*. Jika kamu dapat menilai output kandidat atau keputusan perutean, kamu dapat mengoptimalkan ruang tersebut. Hal ini membuat standar penerapannya jauh lebih rendah.

### Batasan praktis

- **Anggaran populasi.** N partikel × T iterasi × biaya per-eval. Untuk evaluasi LLM pada ~$0,02 / panggilan, PSO 20 partikel yang menjalankan 50 iterasi berharga ~$20. Rencanakan dengan tepat.
- **Eksplorasi vs eksploitasi.** Laju peluruhan feromon dan trade off inersia PSO; pembusukan terlalu cepat → lupakan solusi; terlalu lambat → terjebak pada local optima awal.
- **Penyimpangan bencana.** Kedua algoritme dapat menyatu dan kemudian menyimpang jika lanskap kebugaran berubah (distribusi data baru). Pantau stabilitas kebugaran terbaik.

## Build

`code/main.py` mengimplementasikan:- `LMPSO` — PSO melalui parameter prompt numerik (suhu, weight top_k). "Generasi LLM" setiap partikel disimulasikan sebagai fungsi kebugaran tertulis. Menjalankan algoritma untuk 30 iterasi dan menunjukkan konvergensi g_best.
- `AMRO_S` — Perutean bergaya ACO. 3 agen, 4 jenis tugas, matrix feromon, 100 tugas yang diarahkan. Mencetak distribusi (task_type → pilihan agen) dari waktu ke waktu untuk menunjukkan pembentukan jejak.
- Perbandingan: perutean acak vs perutean ACO pada aliran tugas yang sama. Mengukur kualitas dan latensi.

Jalankan:

```
python3 code/main.py
```

Hasil yang diharapkan:
- LMPSO: kebugaran g_best meningkat dari acak menjadi mendekati optimal selama 30 iterasi.
- AMRO-S: tabel feromon stabil pada agen yang tepat per jenis tugas; Perutean ACO mengalahkan kualitas acak sebesar ~30-40% dan juga mengurangi latensi (percobaan ulang lebih sedikit).

## Pakai

`outputs/skill-swarm-optimizer.md` membantu memilih antara PSO, ACO, algoritme genetika, dan optimizer berbasis gradient untuk masalah optimization LLM/agen.

## Kirim

- **Mulai dari yang kecil.** 10-20 partikel, 20-50 iterasi. Tingkatkan skala hanya jika kurva konvergensi menunjukkan peningkatan yang jelas.
- **Mencatat feromon atau g_best per iterasi.** Men-debug optimizer gerombolan tanpa jejak sangatlah sulit.
- **Pembaruan gerbang kualitas.** Khusus untuk perutean ACO: agen yang cepat dan salah tidak boleh menghasilkan feromon.
- **Reset peluruhan pada shift distribusi.** Saat distribusi eval kamu berubah, feromon lama akan menjadi basi; mengatur ulang atau menggandakan tingkat peluruhan untuk sementara.
- **Batasi biaya per iterasi.** Keluarkan metrik biaya per iterasi. PSO dengan biaya $500/iterasi dan keuntungan 0,5% tidak dapat dikirimkan.

## Latihan

1. Jalankan `code/main.py`. Amati konvergensi LMPSO. Variasikan ukuran populasi 5, 10, 20, 50. Pada ukuran berapa waktu konvergensi menjadi jenuh?
2. Terapkan eksperimen "catastrophic drift": setelah iterasi 30, ubah fungsi kebugaran. Seberapa cepat PSO beradaptasi? Apakah menyetel ulang `p_best` membantu?
3. Tambahkan gerbang kualitas ke AMRO-S: deposit feromon hanya pada proses dengan skor eval > 0,7. Bagaimana hal ini mengubah konvergensi vs versi tanpa gerbang?
4. Baca LMPSO (arXiv:2504.09247). Petakan "kecepatan sebagai prompt" kertas kembali ke kecepatan numerik kamu. Apa yang hilang dalam simulasi dan apa yang dipertahankan?
5. Baca AMRO-S (arXiv:2603.12933). Terapkan "jalur cepat inference" yang dipisahkan dengan pembaruan feromon asinkron. Bagaimana hal ini mengubah latensi sistem di bawah weight berkelanjutan?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| PSO | "Optimization Kawanan Partikel" | Kennedy-Eberhart 1995. Optimizer bebas gradient berbasis populasi. |
| ACO | "Optimization Koloni Semut" | Dorigo 1992. Optimization jalur/rute melalui jalur feromon. |
| LMPSO | "PSO dengan generasi LLM" | arXiv:2504.09247. Kecepatan adalah sebuah prompt; LLM menghasilkan kandidat. |
| Kawanan Model | "PSO pada weight ahli" | arXiv:2410.11163. Pembaruan bebas gradient pada subruang parameter model. |
| AMRO-S | "ACO untuk perutean agen" | arXiv:2603.12933. Matrix feromon pada agen × tipe tugas. |
| p_terbaik / g_terbaik | "Pribadi/terbaik global" | Solusi terbaik per partikel dan segerombolan telah ditemukan sejauh ini. |
| Feromon | "Perutean memori" | Kekuatan di ujung tanduk; membusuk seiring waktu; simpanan pada kualitas. |
| Pembaruan dengan gerbang kualitas | "Hanya belajar dari hasil yang baik" | Deposit feromon dikondisikan pada pemeriksaan kualitas. |
| Pergeseran bencana | "Pergeseran distribusi" | Perubahan lanskap kebugaran; p_best lama dan feromon menjadi basi. |

## Bacaan Lanjutan- [Kennedy & Eberhart — Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968) — makalah PSO tahun 1995
- [Dorigo — Optimization Koloni Semut](https://www.aco-metaheuristic.org/about.html) — yayasan ACO tahun 1992
- [LMPSO — Optimization Kawanan Partikel Model Bahasa](https://arxiv.org/abs/2504.09247) — PSO untuk output LLM terstruktur
- [Model Swarms — optimization pakar LLM bebas gradient](https://arxiv.org/abs/2410.11163) — PSO pada subruang weight model
- [AMRO-S — perutean multi-agen ant-koloni](https://arxiv.org/abs/2603.12933) — perutean berbasis feromon dengan gerbang kualitas
