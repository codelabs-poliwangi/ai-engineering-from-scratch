# Penyempurnaan Diri dan KRITIS: Peningkatan Output Berulang

> Self-Refine (Madaan et al., 2023) menggunakan satu LLM dalam tiga peran — menghasilkan, memberi umpan balik, menyempurnakan — dalam satu lingkaran. Keuntungan rata-rata: +20 mutlak pada 7 tugas. CRITIC (Gou et al., 2023) memperkuat langkah umpan balik dengan mengarahkan verifikasi melalui alat eksternal. Pada tahun 2026, pola ini dikirimkan dalam setiap framework sebagai "evaluator-optimizer" (Anthropic) atau loop pagar pembatas (OpenAI Agents SDK).

**Type:** Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 01 (Agent Loop), Fase 14 · 03 (Refleksi)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Nyatakan tiga prompt Perbaiki Diri (hasilkan, berikan umpan balik, perbaiki) dan jelaskan mengapa sejarah penting untuk prompt perbaiki.
- Jelaskan wawasan kritis CRITIC: LLM tidak dapat diandalkan dalam verifikasi mandiri tanpa landasan eksternal.
- Menerapkan loop Self-Refine stdlib dengan riwayat dan pemverifikasi eksternal opsional.
- Petakan pola ini ke alur kerja "evaluator-optimizer" Anthropic dan pagar pembatas output OpenAI Agents SDK.

## Masalah

Seorang agen menghasilkan jawaban yang hampir benar. Mungkin sebaris code memiliki kesalahan sintaksis. Mungkin ringkasannya terlalu panjang. Mungkin sebuah rencana melewatkan kasus yang sulit. Yang kamu inginkan adalah: agen mengkritik keluarannya sendiri, lalu memperbaikinya.

Self-Refine menunjukkan ini berfungsi dengan model tunggal, tanpa training data, tanpa RL. Namun ada kendalanya: LLM buruk dalam verifikasi diri berdasarkan fakta nyata. CRITIC memberi nama perbaikannya — mengarahkan langkah verifikasi melalui alat eksternal (pencarian, penerjemah code, kalkulator, pelari pengujian).

Bersama-sama, kedua makalah ini menentukan standar tahun 2026 untuk perbaikan berulang: menghasilkan, memverifikasi (secara eksternal jika memungkinkan), menyempurnakan, menghentikan ketika pemverifikasi lolos.

## Konsep

### Penyempurnaan Diri (Madaan dkk., NeurIPS 2023)

Satu LLM, tiga peran:

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

Detail penting: `refine` melihat riwayat lengkap — semua output dan kritik sebelumnya — sehingga tidak mengulangi kesalahan. Makalah ini menghapuskan hal ini: sejarah penurunan dan kualitas menurun tajam.

Judul: +20 peningkatan absolut rata-rata pada 7 tugas (matematika, code, akronim, dialog) termasuk GPT-4. Tanpa training, tanpa alat eksternal, model tunggal.

### KRITIK (Gou dkk., arXiv:2305.11738, v4 Feb 2024)

Kelemahan Self-Refine: langkah feedback adalah penilaian LLM itu sendiri. Untuk klaim faktual, hal ini tidak dapat diandalkan (halusinasi sering kali terlihat meyakinkan bagi model yang menghasilkannya). CRITIC menggantikan `feedback(task, output)` dengan `verify(task, output, tools)` dimana `tools` mencakup:

- Mesin pencari untuk klaim faktual.
- Penerjemah code untuk kebenaran code.
- Kalkulator untuk aritmatika.
- Pemverifikasi khusus domain (pengujian unit, pemeriksa tipe, linter).

Verifikator menghasilkan kritik terstruktur yang didasarkan pada hasil alat. Pemurni kemudian mengkondisikan kritik ini.

Judul: CRITIC mengungguli Self-Refine pada tugas faktual karena kritiknya beralasan. Pada tugas tanpa pemverifikasi eksternal (penulisan kreatif, pemformatan), CRITIC direduksi menjadi Self-Refine.

### Kondisi berhenti

Dua bentuk umum:

1. **Pemverifikasi lolos.** Pengujian eksternal berhasil. Lebih disukai jika tersedia (pengujian unit, pemeriksa tipe, pernyataan pagar pembatas).
2. **Tidak ada input yang dikeluarkan.** Model mengatakan "outputnya baik-baik saja". Lebih murah tapi tidak bisa diandalkan; pasangkan dengan batas iterasi maksimal.

Default 2026: gabungkan keduanya. "Hentikan jika pemverifikasi lolos ATAU model mengatakan baik DAN iterasi >= 2 ATAU iterasi >= max_iterations."

### Evaluator-Optimizer (Antropik, 2024)Postingan Anthropic pada Desember 2024 menyebut ini sebagai salah satu dari lima pola alur kerja. Dua peran:

- Evaluator: menilai output dan menghasilkan kritik.
- Optimizer: merevisi output yang diberi kritik.

Ulangi sampai evaluator lolos. Inilah Self-Refine/CRITIC dalam framing Anthropic. Detail teknik penting yang ditambahkan Anthropic: prompt evaluator dan optimizer harus sangat berbeda sehingga model tidak hanya sekedar stempel.

### Pagar pembatas output OpenAI Agents SDK

OpenAI Agents SDK mengirimkan pola ini sebagai "pagar pembatas output". Pagar pembatas adalah validator yang berjalan pada output akhir agen. Jika pagar pembatas tersandung (menaikkan `OutputGuardrailTripwireTriggered`), outputnya ditolak dan agen dapat mencoba lagi. Pagar pembatas dapat memanggil alat (gaya CRITIC) atau menjadi fungsi murni (gaya Self-Refine).

### 2026 jebakan

- **Lingkaran stempel karet.** Model yang sama dalam melakukan pembangkitan dan kritik dengan gaya cepat yang sama menyatu pada "kelihatannya bagus untuk saya". Gunakan prompt yang berbeda secara struktural, atau model kritik yang lebih murah dan lebih kecil.
- **Penyempurnaan berlebihan.** Setiap proses penyempurnaan menambahkan latensi dan token. Anggaran 1-3 lolos; setelah itu, tingkatkan ke peninjauan manusia.
- **CRITIC pada tugas-tugas sepele.** Jika tidak ada pemverifikasi eksternal, CRITIC merosot menjadi Self-Refine; jangan membayar latensi untuk pemverifikasi rintisan.

## Build

`code/main.py` mengimplementasikan Self-Refine dan CRITIC pada tugas mainan: membuat daftar singkat berdasarkan topik. Pemverifikasi memeriksa format (3 poin, masing-masing kurang dari 60 karakter). CRITIC menambahkan "pemverifikasi fakta" eksternal yang menghukum halusinasi yang diketahui.

Komponen:

- `generate` — produser naskah.
- `feedback` — Kritik diri ala LLM.
- `verify_external` — Verifikator ground bergaya CRITIC.
- `refine` — menulis ulang output berdasarkan riwayat.
- Kondisi berhenti — verifikasi lolos atau maksimal 4 iterasi.

Jalankan:

```
python3 code/main.py
```

Bandingkan proses Self-Refine vs CRITIC. CRITIC menangkap kesalahan faktual. Self-Refine tidak terjawab karena pemverifikasi eksternal memiliki ground sedangkan self-critic tidak.

## Pakai

Optimizer-evaluator Anthropic adalah pola ini dalam bahasa yang ramah Claude. Pagar pembatas output OpenAI Agents SDK berbentuk CRITIC (pagar pembatas dapat memanggil alat). LangGraph mengirimkan node refleksi yang berbunyi seperti Self-Refine. Penggunaan Komputer Gemini 2.5 Google menambahkan evaluator keamanan per langkah yang merupakan varian CRITIC: setiap tindakan diverifikasi sebelum dilakukan.

## Kirim

`outputs/skill-refine-loop.md` mengonfigurasi loop evaluator-optimizer berdasarkan bentuk tugas, ketersediaan verifier, dan anggaran iterasi. Memancarkan prompt untuk generator, evaluator/verifikator, dan optimizer, ditambah kebijakan penghentian.

## Latihan

1. Jalankan mainan dengan max_iterations=1. Apakah CRITIC masih membantu?
2. Ganti verifikator eksternal dengan yang berisik (acak 30% positif palsu). Apa fungsi loop? Ini adalah realitas sebagian besar tumpukan pagar pembatas pada tahun 2026.
3. Menerapkan varian "generator-kritikus pada model berbeda": pembuatan model besar, kritik model kecil. Apakah ini mengalahkan model yang sama?
4. Baca KRITIK Bagian 3 (arXiv:2305.11738 v4). Sebutkan tiga kategori alat verifikasi dan berikan contohnya masing-masing.
5. Memetakan `output_guardrails` OpenAI Agents SDK ke peran verifikator CRITIC. Apa yang salah dari SDK, dan apa yang benar?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Memperbaiki Diri | "LLM yang memperbaiki dirinya sendiri" | Hasilkan -> umpan balik -> perbaiki loop dalam satu model, dengan riwayat |
| KRITIK | "Verifikasi berbasis alat" | Ganti umpan balik dengan pemverifikasi eksternal (pencarian, code, perhitungan, pengujian) |
| Evaluator-Optimizer | "Pola alur kerja antropis" | Dua peran — penilaian evaluator, revisi optimizer — dilingkarkan ke konvergensi |
| Pagar pembatas output | "Pemeriksaan pasca-hoc" | Validator OpenAI Agents SDK yang berjalan setelah agen menghasilkan output |
| Langkah verifikasi | "Fase Kritik" | Keputusan menahan weight: membumi atau menilai sendiri |
| Sempurnakan riwayat | "Model apa yang sudah dicoba" | Output sebelumnya + kritik ditambahkan untuk menyempurnakan prompt; penurunan dan kualitas runtuh |
| Lingkaran stempel karet | "Kegagalan kesepakatan diri" | Kritik yang sama langsung menghasilkan "kelihatannya bagus"; perbaiki dengan prompt yang berbeda secara struktural |
| Kondisi berhenti | "Uji konvergensi" | Verifikator lolos ATAU tidak ada umpan balik DAN batas iterasi; tidak pernah bersyarat tunggal |

## Bacaan Lanjutan

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — makalah kanonik
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — verifikasi berbasis alat
- [Antropik, Agen Bangunan yang Efektif](https://www.anthropic.com/research/building- Effective-agents) — pola alur kerja evaluator-optimizer
- [Dokumen SDK Agen OpenAI](https://openai.github.io/openai-agents-python/) — pagar pembatas output sebagai pemverifikasi berbentuk CRITIC
