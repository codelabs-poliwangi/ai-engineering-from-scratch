# Debat dan Kolaborasi Multi-Agen

> Du dkk. (ICML 2024, "Society of Minds") menjalankan N contoh model yang mengajukan jawaban secara independen, lalu saling mengkritik secara berulang selama putaran R agar menyatu. Meningkatkan faktualitas, mengikuti aturan, penalaran. Topologi renggang mengalahkan biaya token secara penuh.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 12 (Pola Alur Kerja), Fase 14 · 05 (Perbaikan Mandiri dan KRITIS)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Jelaskan protokol debat: N pengusul, R putaran, berkumpul pada jawaban bersama.
- Jelaskan mengapa debat meningkatkan faktualitas, kepatuhan terhadap aturan, dan penalaran.
- Jelaskan topologi sparse: tidak setiap pendebat perlu bertemu satu sama lain.
- Menerapkan debat stdlib atas LLM bernaskah dengan varian full-mesh dan sparse; mengukur biaya token vs akurasi.

## Masalah

Penyempurnaan Diri (Lesson 05) adalah salah satu model yang mengkritik dirinya sendiri - berisiko terhadap pemikiran kelompok. CRITIC (Lesson 05) mendasarkan kritik pada alat eksternal — tidak selalu tersedia. Debat memperkenalkan cara ketiga: banyak contoh, kritik silang, konvergensi demi ketidaksepakatan.

## Konsep

### Masyarakat Pikiran (Du dkk., ICML 2024)

- N contoh model secara independen mengajukan jawaban atas pertanyaan yang sama.
- Selama putaran R, masing-masing model membaca proposal model lain dan mengkritiknya.
- Model memperbarui jawaban mereka berdasarkan kritik.
- Setelah putaran R, kembalikan jawaban konvergen.

Eksperimen asli menggunakan N=3, R=2 karena biaya. Akurasi meningkat dengan lebih banyak agen dan lebih banyak putaran pada masalah sulit (MMLU, GSM8K, Chess Move Validity, pembuatan biografi).

Kombinasi lintas model mengalahkan perdebatan model tunggal: ChatGPT + Bard bersama-sama > baik sendiri-sendiri.

### Topologi renggang

“Meningkatkan Debat Multi-Agen dengan Topologi Komunikasi Jarang” (arXiv:2406.11776, 2024-2025) menunjukkan debat full-mesh tidak selalu optimal. Topologi renggang (star, ring, hub-and-spoke) dapat mencocokkan akurasi dengan biaya token yang lebih rendah. Setiap pendebat hanya melihat sebagian dari rekannya.

Implikasi:

- Jaring penuh N=5, R=3 = 5 × 3 = 15 proposal, masing-masing membaca 4 rekan = 60 operasi kritik.
- Bintang N=5, R=3 (satu hub + 4 jari-jari) = 15 proposal, jari-jari hanya membaca hub = 12 ops kritik.

### Saat debat membantu

- **Faktualitas.** N proposal independen, pemeriksaan silang mengurangi halusinasi.
- **Mengikuti aturan.** Validitas gerakan catur — satu model melewatkan aturan, yang lain menangkapnya.
- **Penalaran terbuka.** Berbagai kerangka mempersempit jawaban yang benar.

### Saat perdebatan itu menyakitkan

- **UX yang sensitif terhadap latensi.** Putaran serial N × R adalah latensi yang mungkin tidak kamu miliki.
- **Skala yang sensitif terhadap biaya.** N × R token per pertanyaan.
- **Pencarian faktual sederhana.** Satu pencarian lebih murah daripada lima debat.

### 2026 contoh praktis

- **Pekerja orkestra antropik** (Lesson 12) — salah satu varian perdebatan dengan langkah sintesis.
- **LangGraph supervisor** (Lesson 13) — router pusat + agen spesialis dapat mengimplementasikan debat sebagai sebuah node.
- **OpenAI Agents SDK** (Lesson 16) — agen melakukan penyerahan bolak-balik untuk kritik berulang.
- **Eval multi-agen** — debat berpasangan + optimizer-evaluator untuk sinyal eval.

### Dimana letak kesalahan pola ini

- **Konvergensi runtuh.** Semua agen berkumpul pada jawaban pertama yang salah. Mitigasi dengan putaran perselisihan yang diperlukan.
- **Kegagalan hub.** Dalam topologi bintang, hub yang buruk akan merusak semua orang. Putar atau gunakan beberapa hub.
- **homogenisasi cepat.** Semua agen menggunakan prompt yang sama; mereka menghasilkan jawaban yang sama. Gunakan beragam petunjuk dan/atau model.

## Build`code/main.py` mengimplementasikan debat stdlib:

- Kelas `Debater` (dengan skrip LLM dengan penyimpangan opini per debat).
- Pelari `FullMeshDebate` dan `SparseDebate`.
- Tiga pertanyaan: satu faktual, satu berdasarkan aturan, satu alasan.
- Metrik: jawaban konvergen, putaran ke konvergensi, operasi kritik total.

Jalankan:

```
python3 code/main.py
```

Output: akurasi dan biaya per protokol; pertandingan jarang mesh penuh pada 2/3 pertanyaan dengan biaya lebih rendah.

## Pakai

- **Pekerja orkestra antropis** untuk debat sederhana 2-3 pekerja.
- **LangGraph** untuk debat multi-putaran stateful dengan pos pemeriksaan.
- **Khusus** untuk penelitian atau jaminan kebenaran khusus.

## Kirim

`outputs/skill-debate.md` merancah perdebatan multi-agen dengan topologi yang dapat dikonfigurasi, N, R, dan aturan konvergensi.

## Latihan

1. Menerapkan aturan “ketidaksepakatan paksa”: pada putaran 1, setiap debat harus menghasilkan proposal yang berbeda. Ukur pengaruhnya terhadap kecepatan konvergensi.
2. Tambahkan agregasi berdasarkan keyakinan: pendebat kembali (jawaban, keyakinan); weight agregator berdasarkan keyakinan. Apakah itu membantu?
3. Tukar satu "agen" dengan LLM bernaskah berbeda dengan opini berbeda. Apakah heterogenitas meningkatkan akurasi?
4. Ukur biaya token untuk mesh penuh vs jarang pada 3 pertanyaan kamu. Biaya plot vs akurasi.
5. Bacalah makalah Society of Minds. Pindahkan mainan kamu ke N=5, R=3. Apa yang rusak? Apa yang menjadi lebih baik?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Debat | "Kritik multi-agen" | N pengusul, R putaran kritik silang, berkumpul |
| Jaring penuh | "Semua orang membaca semua orang" | Setiap pendebat membaca setiap rekan setiap putaran |
| Topologi renggang | "Tampilan sejawat terbatas" | Pendebat hanya membaca sebagian dari rekan |
| Hub-dan-bicara | "Topologi bintang" | Salah satu perdebatan utama, jari-jari N-1 hanya membaca hub |
| Konvergensi | "Perjanjian" | Para pendebat berkumpul pada jawaban bersama |
| Masyarakat Pikiran | "Makalah debat Du dkk" | Metode debat multi-agen ICML 2024 |

## Bacaan Lanjutan

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — debat multi-agen kanonik
- [Topologi Komunikasi Jarang (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — hasil topologi jarang
- [Antropik, Membangun Agen yang Efektif](https://www.anthropic.com/research/building- Effective-agents) — pekerja orkestra sebagai varian debat
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — mitra kritik diri model tunggal
