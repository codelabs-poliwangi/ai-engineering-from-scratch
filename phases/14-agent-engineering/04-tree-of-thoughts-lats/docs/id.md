# Pohon Pikiran dan LATS: Pencarian yang Disengaja

> Satu lintasan rantai pemikiran tidak mempunyai ruang untuk mundur. ToT (Yao et al., 2023) mengubah penalaran menjadi pohon dengan evaluasi diri pada setiap node. LATS (Zhou et al., 2024) menyatukan ToT dengan ReAct dan Reflexion dalam Monte Carlo Tree Search. Game of 24 berubah dari 4% (CoT) menjadi 74% (ToT); LATS mencapai 92,7% pass@1 di HumanEval.

**Type:** Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 01 (Agent Loop), Fase 14 · 03 (Refleksi)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Penalaran bingkai sebagai pencarian: node adalah "pikiran", tepian adalah "ekspansi", nilai adalah "seberapa menjanjikan".
- Menerapkan pencarian pohon BFS gaya ToT stdlib dengan penilaian evaluasi mandiri.
- Perluas ke loop mainan LATS MCTS dengan pilih / perluas / simulasikan / backpropagation.
- Putuskan kapan pencarian bernilai pengganda token (Game 24, pembuatan code) dan kapan satu lintasan sudah cukup (Tanya Jawab sederhana).

## Masalah

Rantai pemikiran adalah perjalanan linear. Jika langkah pertama salah, setiap langkah berikutnya akan berdampak buruk. Pada Game of 24 (gunakan empat digit dengan + − × ör untuk menghasilkan 24), GPT-4 CoT mencapai akurasi 4%. Model ini mengambil subekspresi yang salah sejak awal dan tidak dapat memulihkannya.

Yang dibutuhkan oleh penalaran adalah kemampuan untuk mengusulkan banyak kandidat, mengevaluasi mereka, memilih kandidat yang menjanjikan, dan melakukan kemunduran ketika jalan buntu muncul. Itu adalah pencarian. Pohon Pikiran dan LATS adalah dua formulasi kanonik.

## Konsep

### Pohon Pikiran (Yao et al., NeurIPS 2023)

Setiap simpul adalah langkah perantara yang koheren (“sebuah pemikiran”). Setiap node dapat memperluas hingga K pemikiran anak. LLM mengevaluasi sendiri setiap node dengan prompt penilaian. Pencarian menjelajahi pohon — BFS, DFS, atau beam.

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

Evaluasi diri adalah bagian yang menahan weight. Makalah ini menunjukkan tiga varian: klasifikasi `sure / likely / impossible`, skor numerik `1..10`, dan suara antar kandidat. Ketiganya mengalahkan CoT secara substansial di Game 24 (4% -> 74% dengan GPT-4).

### LATS (Zhou dkk., ICML 2024)

LATS menyatukan ToT, ReAct, dan Reflexion di bawah MCTS. LLM memainkan tiga peran:

- **Kebijakan**: mengusulkan tindakan kandidat selanjutnya (gaya React).
- **Fungsi nilai**: mencetak sebagian lintasan (evaluasi mandiri gaya ToT).
- **Refleksi diri**: jika gagal, tulis refleksi dalam bahasa alami (Gaya Refleksi) dan gunakan untuk memulai kembali peluncuran di masa mendatang.

Umpan balik lingkungan (pengamatan) digabungkan ke dalam fungsi nilai sehingga pencarian diinformasikan oleh hasil alat yang nyata, bukan hanya opini model. Hasil pada waktu kertas: HumanEval pass@1 92,7% dengan GPT-4 (SOTA), rata-rata WebShop 75,9 dengan GPT-3.5 (mendekati penyesuaian berbasis gradient).

### MCTS, minimal

Empat fase per iterasi:

1. **Pilih** — berjalan dari akar ke daun menggunakan UCT (batas kepercayaan atas untuk pohon).
2. **Perluas** — menghasilkan K anak melalui kebijakan.
3. **Simulasi** — peluncuran dari turunan yang menggunakan kebijakan, skor daun dengan fungsi nilai (atau imbalan lingkungan).
4. **Backpropagate** — memperbarui jumlah kunjungan dan perkiraan nilai di sepanjang jalur.

Rumus UCT: `Q(s, a) + c * sqrt(ln N(s) / N(s, a))`. Istilah pertama adalah eksploitasi; kedua adalah eksplorasi. Sesuaikan `c` per tugas.

### Kenyataan biaya

Pencarian meledakkan token. ToT di Game of 24 menggunakan 100–1000x token CoT. LATS serupa. Ini tidak gratis; pencarian cadangan untuk:- Tugas di mana satu lintasan terbukti tidak mencukupi (Game 24, code kompleks).
- Tugas di mana jam dinding kurang penting dibandingkan kebenaran.
- Tugas dengan fungsi nilai yang murah dan andal (tes unit untuk code, target eksplisit untuk matematika).

Jika tugas kamu memiliki satu jawaban yang benar dan evaluator yang berisik, pencarian sering kali memperburuk keadaan - pencarian tersebut menemukan jawaban yang salah dengan "skor bagus".

### Pemosisian 2026

Kebanyakan agen produksi tidak menjalankan LATS. Mereka menjalankan ReAct dengan verifikasi berbasis alat (CRITIC, Lesson 05). Pencarian muncul di ceruk khusus:

- Agen pengkodean yang menjalankan tes sebagai fungsi nilai (gaya HumanEval).
- Agen penelitian mendalam yang menjelajahi berbagai jalur kueri.
- Alur kerja yang penuh perencanaan di dalam subgraf LangGraph.

AlphaEvolve (Lesson 11) adalah ekstrem tahun 2025: penelusuran evolusioner atas code, kebugaran yang dapat diperiksa mesin, perolehan batas (peningkatan matmul 4x4 pertama dalam 56 tahun).

## Build

`code/main.py` mengimplementasikan:

- BFS ToT kecil pada tugas "pilih operasi aritmatika" yang bergaya.
- Mainan loop LATS MCTS pada tugas yang sama (Pilih / Perluas / Simulasikan / Propagasi Balik) dengan pilihan UCT.
- Fungsi nilai yang menyusun skor simbolis ditambah skor evaluasi diri.

Jalankan:

```
python3 code/main.py
```

Jejaknya menunjukkan ToT memperluas tiga kandidat per node dengan BFS, dibandingkan dengan LATS yang melakukan konvergensi pada peluncuran terbaik melalui MCTS. Jumlah token dicetak untuk keduanya.

## Pakai

LangGraph mengirimkan eksplorasi gaya ToT sebagai pola subgraf; blog tim LangChain di LATS (Mei 2024) adalah tutorial referensi. LlamaIndex mengirimkan agen `TreeOfThoughts`. Bagi sebagian besar agen produksi tahun 2026, pola ini berada di balik gerbang `if task_complexity > threshold: use_search()` — lihat pola evaluator-optimizer di Lesson 05.

## Kirim

`outputs/skill-search-policy.md` memilih antara ReAct linier, ToT, LATS, dan penelusuran evolusioner berdasarkan bentuk tugas, anggaran, dan ketelitian evaluator.

## Latihan

1. Jalankan mainan LATS dengan UCT c=0.1 vs c=2.0. Perubahan apa saja yang terjadi pada jejaknya?
2. Tukar fungsi nilai dengan pencetak skor yang lebih ribut (tambahkan jitter acak). Apakah MCTS masih menemukan daun terbaik? Berapa sinyal terhadap kebisingan minimum yang dapat ditoleransi?
3. Menerapkan ToT beam-search (pertahankan top-k di setiap level) dan bandingkan dengan BFS. Mana yang lebih baik dengan anggaran terbatas?
4. Baca LATS Bagian 5.1. Mereproduksi jumlah lintasan HumanEval: berapa banyak peluncuran yang diperlukan untuk mencapai pass@1 yang dilaporkan?
5. Baca diskusi makalah LATS tentang "ketika LATS kurang membantu". Tulis bentuk tugas pemetaan aturan keputusan satu paragraf ke strategi pencarian.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Pohon Pikiran | "Percabangan CoT" | Yao dkk. — pohon simpul pemikiran dengan evaluasi diri |
| LATS | "MCTS untuk LLM" | Zhou dkk. — menyatukan ToT + ReAct + Reflexion di bawah MCTS |
| UCT | "Keyakinan atas terikat" | Pilih rumus penyeimbangan eksploitasi (Q) dan eksplorasi (ln N/n) |
| Fungsi nilai | "Betapa bagusnya negara bagian ini" | Skor LLM atau penghargaan lingkungan yang diminta; umpan backprop |
| Kebijakan | "Pengusul tindakan" | Generator bergaya React; memancarkan pemikiran/tindakan calon selanjutnya |
| Peluncuran | "Simulasi lintasan" | Berjalan dari node ke daun menggunakan kebijakan, skor dengan nilai |
| Backpropagation | "Perbarui leluhur" | Dorong hadiah daun ke atas, perbarui jumlah kunjungan dan Q |
| Biaya pencarian | "Ledakan token" | 100-1000x CoT pada Game ke-24; anggaran sebelum kamu mengadopsi |

## Bacaan Lanjutan- [Yao dkk., Pohon Pikiran (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — makalah kanonik
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) — MCTS dengan umpan balik Refleksi
- [Ikhtisar LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — pola subgraf untuk penelusuran
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — penelusuran evolusioner dengan evaluator terprogram
