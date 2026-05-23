# Perencanaan dengan HTN dan Pencarian Evolusioner

> Perencanaan simbolis menangani kasus-kasus dimana rencana tersebut terbukti benar. Pencarian code evolusioner menangani kasus-kasus di mana fungsi kebugaran dapat diperiksa oleh mesin. ChatHTN (2025) dan AlphaEvolve (2025) menunjukkan apa yang bisa dibuka masing-masing saat dipasangkan dengan LLM.

**Type:** Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 02 (ReWOO dan Plan-and-Execute)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Jelaskan Jaringan Tugas Hierarki: tugas, metode, operator, prasyarat, efek.
- Jelaskan loop hybrid ChatHTN — pencarian simbolik dengan decomposition fallback LLM.
- Jelaskan lingkaran evolusi AlphaEvolve dan mengapa ia hanya berfungsi dengan evaluator terprogram.
- Menerapkan perencana mainan HTN plus pencarian evolusi mainan di stdlib.

## Masalah

ReWOO (Lesson 02), Plan-and-Execute, dan ReAct mencakup sebagian besar perencanaan agen. Dua kasus yang tidak mereka liput dengan baik:

1. **Rencana dengan kebenaran yang dapat dibuktikan.** Penjadwalan, jalur penerbangan, alur kerja kepatuhan — rencana tersebut harus sesuai dengan konstruksinya. Rencana LLM yang lancar yang terkadang berhalusinasi suatu langkah tidak dapat diterima.
2. **Optimization dengan fungsi kebugaran yang dapat diperiksa mesin.** Perkalian matrix, heuristik penjadwalan, lintasan compiler — tujuannya bukanlah "rencana yang benar" namun "rencana terbaik".

Perencanaan HTN dan AlphaEvolve memecahkan dua masalah berbeda. Keduanya menggunakan LLM sebagai amplifier, bukan pengganti.

## Konsep

### Jaringan Tugas Hierarki

HTN adalah:

- **Tugas** — gabungan (untuk didekomposisi) dan primitif (dapat dieksekusi langsung).
- **Metode** — cara menguraikan tugas gabungan menjadi subtugas, dengan prasyarat.
- **Operator** — tindakan primitif dengan prasyarat dan efek.
- **Negara** — serangkaian fakta.

Perencanaan: diberikan tugas tujuan dan keadaan awal, temukan decomposition menjadi operator primitif yang prasyaratnya dipenuhi secara berurutan.

HTN lebih tua dari LLM dan masih menjadi referensi untuk rencana yang terbukti benar.

### ObrolanHTN (Gopalakrishnan dkk., 2025)

ChatHTN (arXiv:2505.11814) menyisipkan HTN simbolis dengan kueri LLM:

1. Cobalah untuk menguraikan tugas gabungan saat ini dengan metode yang ada.
2. Jika tidak ada metode yang berlaku, tanyakan kepada LLM: "bagaimana kamu menguraikan `task` dalam keadaan `s`?"
3. Terjemahkan tanggapan LLM ke dalam subtugas kandidat.
4. Validasi terhadap skema operator; menolak decomposition yang tidak valid.
5. Berulang.

Klaim utama makalah ini: setiap rencana yang dihasilkan terbukti masuk akal karena saran LLM hanya masuk sebagai penguraian kandidat, tidak pernah sebagai pengeditan rencana langsung. Layer simbolis memiliki kebenaran; LLM memperluas perpustakaan metode.

Pembelajaran metode online (OpenReview `gwYEDY9j2x`, tindak lanjut 2025) menambahkan pembelajar yang menggeneralisasi decomposition yang dihasilkan LLM melalui regresi — memotong frekuensi kueri LLM hingga 75%.

### AlphaEvolve (Novikov dkk., 2025)

AlphaEvolve (arXiv:2506.13131, DeepMind, Juni 2025) adalah binatang yang berbeda: pencarian code evolusioner yang diatur oleh ansambel Gemini 2.0 Flash/Pro.

Lingkaran:

1. Mulailah dengan program awal + evaluator terprogram (mengembalikan skor kebugaran).
2. Ensemble LLM mengusulkan mutasi.
3. Jalankan mutasi melalui evaluator.
4. Pertahankan yang terbaik; bermutasi lagi.

Kemenangan yang dipublikasikan:

- Perbaikan pertama pada Strassen untuk perkalian matrix kompleks 4x4 dalam 56 tahun (48 perkalian scalar).
- 0,7% memulihkan komputasi Google melalui heuristik penjadwalan Borg.
- 32% percepatan FlashAttention pada weight kerja frontier.Kendala beratnya: fungsi kebugaran harus dapat diperiksa oleh mesin. Pencarian evolusioner atas jawaban-jawaban prosa tidak menemui titik temu.

### Kapan menggunakan yang mana

| Kelas masalah | Gunakan | Mengapa |
|---------------|-----|-----|
| Penjadwalan dengan batasan keras | HTN + ObrolanHTN | Kesehatan yang dapat dibuktikan |
| Optimization kompiler | Evolusi Alpha | Kebugaran yang dapat diperiksa mesin |
| Eksekusi tugas multi-langkah | Bereaksi / ReWOO | LLM dalam lingkaran, tidak ada jaminan formal |
| Peningkatan code dengan tes | Evolusi Alpha | Tes adalah evaluator |
| Otomatisasi terikat kebijakan | HTN | Prasyarat menyandikan kebijakan |

### Dimana letak kesalahan pola ini

- **HTN tanpa operator.** Tanpa skema prasyarat/efek, klaim kesehatan akan gagal. "LLM menyarankan decomposition" ChatHTN memerlukan skema untuk menolak gerakan yang tidak valid.
- **AlphaEvolve tanpa evaluator sungguhan.** "Tanyakan pada LLM apakah kodenya lebih baik" bukan fungsi kebugaran. Evaluator harus deterministik dan cepat.
- **Rekayasa berlebihan.** Sebagian besar tugas agen juga tidak memerlukannya. Raih ReAct atau ReWOO terlebih dahulu.

## Build

`code/main.py` mengimplementasikan dua mainan:

- Perencana HTN stdlib dengan operator, metode, prasyarat, efek, dan `LLMFallback` yang berfungsi ketika tidak ada metode yang cocok dengan tugas gabungan. "LLM" adalah dekomposer bernaskah sehingga perencana berjalan offline.
- Pencarian evolusioner stdlib pada program aritmatika: menumbuhkan ekspresi yang outputnya meminimalkan `|f(x) - target|` pada set pengujian. Evaluator bersifat deterministik.

Jalankan:

```
python3 code/main.py
```

Jejak tersebut menunjukkan perencana HTN menguraikan tugas gabungan (dengan fallback LLM rencana tengah) dan lingkaran evolusi yang menyatu pada ekspresi target.

## Pakai

- **Perencana HTN** — `pyhop`, `SHOP3`, atau buat sendiri untuk penegakan kebijakan khusus domain.
- **ChatHTN** — code penelitian; pola (simbolis + LLM fallback) port dengan rapi ke perencana HTN mana pun.
- **AlphaEvolve** — makalah DeepMind; polanya (ensemble + evaluator) dapat direproduksi. OpenEvolve dan fork sumber terbuka serupa sedang bermunculan.
- **Kerangka agen** — belum ada yang mengirimkan HTN atau AlphaEvolve kelas satu. Bangunlah sebagai subagen atau pekerja latar belakang.

## Kirim

`outputs/skill-hybrid-planner.md` menghasilkan perancah perencana hibrid (HTN atau evolusioner) dengan cakupan peran LLM secara eksplisit.

## Latihan

1. Perluas perencana HTN dengan penelusuran mundur: ketika kondisi pasca operator gagal saat runtime, putar kembali dan coba metode berikutnya.
2. Tambahkan cache metode LLM ke ChatHTN: ketika LLM menguraikan tugas `T` dalam pola status `P`, simpan hasilnya. Periksa kembali perpustakaan metode terlebih dahulu pada panggilan berikutnya.
3. Tukar evaluator penelusuran evolusioner ke rangkaian pengujian nyata. Kembangkan fungsi pengurutan yang lolos 20 kasus uji; generasi laporan ke konvergensi.
4. Baca catatan desain evaluator AlphaEvolve. Rancang evaluator untuk domain yang kamu minati (optimization kueri SQL, minimalisasi rangkaian pengujian, penerapan YAML).
5. Gabungkan: gunakan HTN untuk menguraikan tugas gabungan menjadi subtugas, lalu gunakan penelusuran evolusioner pada operator primitif setiap subtugas. Di mana ia bersinar, di mana ia melakukan rekayasa berlebihan?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| HTN | "Perencana hierarki" | Decomposition tugas dengan operator, prasyarat, efek |
| Metode | "Aturan decomposition" | Cara memecah tugas gabungan menjadi subtugas |
| Operator | "Tindakan primitif" | Langkah konkrit dengan prasyarat dan efek |
| ObrolanHTN | "LLM + HTN" | Perencana simbolis menanyakan LLM ketika tidak ada metode yang cocok |
| Evolusi Alpha | "Pencarian code evolusi" | Ensemble code mutasi LLM; evaluator deterministik memilih |
| Fungsi kebugaran | "Penilai" | Skor deterministik dan dapat diperiksa mesin atas output |
| Pembelajaran metode online | "Decomposition LLM dalam cache" | Simpan + menggeneralisasi rencana LLM untuk memotong biaya permintaan |

## Bacaan Lanjutan

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — perencana hybrid simbolis + LLM
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — pencarian code evolusi dengan mutasi LLM
- [Antropik, Agen Bangunan yang Efektif](https://www.anthropic.com/research/building- Effective-agents) — kapan harus menggunakan perencana vs putaran sederhana
