# Tolok ukur: WebArena dan OSWorld

> WebArena menguji kemampuan agen web di empat aplikasi yang dihosting sendiri. OSWorld menguji kemampuan agen desktop di Ubuntu, Windows, macOS. Saat dirilis (2023–2024) keduanya menunjukkan kesenjangan besar antara agen terbaik di kelasnya dan manusia. Kesenjangannya semakin menyempit; mode kegagalan tidak berubah.

**Type:** Learn
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 19 (SWE-bench, GAIA)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Jelaskan empat aplikasi yang dihosting sendiri di WebArena dan mengapa evaluasi berbasis eksekusi penting.
- Jelaskan mengapa OSWorld menggunakan tangkapan layar OS sebenarnya dan bukan API aksesibilitas.
- Sebutkan dua mode kegagalan utama OSWorld: landasan GUI dan pengetahuan operasional.
- Ringkas apa yang ditambahkan OSWorld-G dan OSWorld-Human di atas tolok ukur dasar.

## Masalah

Agen generalis dapat memanggil alat. Bisakah mereka mengarahkan browser dalam 20 klik untuk menyelesaikan pembayaran belanja? Bisakah mereka mengkonfigurasi kotak Linux hanya dengan menggunakan keyboard dan mouse? Ini adalah pertanyaan yang dijawab WebArena dan OSWorld.

## Konsep

### WebArena (Zhou dkk., ICLR 2024)

- 812 tugas jangka panjang di empat aplikasi web yang dihosting sendiri: situs belanja, forum, alat pengembangan mirip GitLab, CMS bisnis.
- Ditambah utilitas: peta, kalkulator, papan gores.
- Evaluasi didasarkan pada eksekusi melalui API gym — apakah pesanan telah dilakukan, apakah masalah telah diselesaikan, apakah halaman CMS diperbarui?
- Saat dirilis: agen GPT-4 terbaik mencapai keberhasilan 14,41% vs manusia 78,24%.

Pembingkaian yang dihosting sendiri penting — tolok ukurnya tidak berubah-ubah karena aplikasi target dipasangi pin dan dapat direproduksi.

### Ekstensi

- **VisualWebArena** — tugas berbasis visual yang keberhasilannya bergantung pada interpretasi gambar (tangkapan layar sebagai observasi kelas satu).
- **TheAgentCompany** (Des 2024) — menambahkan terminal + pengkodean; lebih seperti lingkungan kerja distance jauh yang nyata.

### OSWorld (Xie dkk., NeurIPS 2024)

- 369 tugas komputer nyata di Ubuntu, Windows, macOS.
- Kontrol keyboard dan mouse bentuk bebas untuk aplikasi nyata.
- Tangkapan layar 1920×1080 sebagai observasi.
- Saat dirilis: model terbaik 12,24% vs manusia 72,36%.

### Mode kegagalan primer

1. **Pembumian GUI.** Piksel → pemetaan elemen. Model kesulitan melokalisasi elemen UI dengan andal pada resolusi 1920×1080.
2. **Pengetahuan operasional.** Menu mana yang memiliki pengaturan, pintasan keyboard mana, panel preferensi mana. Ekor pengetahuan yang dibangun manusia selama bertahun-tahun.

### Tindak lanjut

- **OSWorld-G** — 564 sample grounding suite + set training Jedi. Mengurai landasan dari perencanaan sehingga kamu dapat mengukurnya secara terpisah.
- **OSWorld-Human** — lintasan aksi emas yang dikurasi secara manual. Menunjukkan agen teratas menggunakan langkah 1,4-2,7x lebih banyak dari yang diperlukan (kesenjangan efisiensi lintasan).

### Mengapa ini penting

Penggunaan komputer Claude, OpenAI CUA, Penggunaan Komputer Gemini 2.5 (Lesson 21) semuanya melatih weight kerja yang dibentuk oleh WebArena dan OSWorld. Tolok ukurnya adalah sasarannya; model produksi adalah jawaban yang dikirimkan.

### Ketika benchmarking salah

- **Evaluasi hanya tangkapan layar.** OSWorld digerakkan oleh tangkapan layar; mengevaluasi agen yang menggunakan DOM atau API aksesibilitas di OSWorld tidak memenuhi tantangan mendasar.
- **Mengabaikan panjang lintasan.** Hanya mencetak tingkat keberhasilan yang meleset dari permukaan OSWorld-Human dengan inefisiensi langkah 1,4-2,7x.
- **Aplikasi yang dihosting sendiri sudah basi.** Aplikasi WebArena embed versi tertentu; pembaruan tanpa kurasi ulang merusak perbandingan.

## Build

`code/main.py` mengimplementasikan harness agen web mainan:- Mesin status "aplikasi belanja" minimal: list_items, add_to_cart, checkout.
- Lintasan emas untuk 3 tugas.
- Agen bernaskah yang mencoba setiap tugas.
- Evaluator berbasis eksekusi (pemeriksaan status) dan metrik efisiensi lintasan (langkah vs emas).

Jalankan:

```
python3 code/main.py
```

Output: tingkat keberhasilan per tugas dan efisiensi lintasan, yang mencerminkan metodologi OSWorld-Human.

## Pakai

- **WebArena Terverifikasi** dihosting sendiri di cluster internal untuk evaluasi berkelanjutan.
- **OSWorld** dalam armada VM untuk agen desktop.
- **Agen penggunaan komputer** (Lesson 21) — Claude, OpenAI CUA, Gemini — semuanya terlatih dalam weight kerja seperti ini.
- **Aliran produk kamu sendiri** — menangkap lintasan emas untuk 20 tugas teratas kamu; menjalankan agen melawan mereka setiap minggu.

## Kirim

`outputs/skill-web-desktop-harness.md` membangun pemanfaatan agen web/desktop dengan evaluasi berbasis eksekusi dan metrik efisiensi lintasan.

## Latihan

1. Perpanjang tali mainan dengan aplikasi kedua (forum). Tulis 3 tugas ditambah lintasan emas.
2. Tambahkan pelaporan efisiensi lintasan per tugas. Pada mainan kamu, apakah agennya 1x, 2x, atau 3x lebih dari emas?
3. Menerapkan alat "pengalih attention" — alat yang tidak pernah digunakan oleh lintasan emas. Apakah agen yang diberi skrip tergoda?
4. Baca OSWorld-G. Bagaimana kamu memisahkan kegagalan landasan dari kegagalan perencanaan dalam evaluasi kamu sendiri?
5. Baca aplikasi WebArena README. Apa yang rusak saat kamu mengupgrade salah satu versi aplikasi yang di-embed?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Arena Web | "Patokan agen web" | 812 tugas di 4 aplikasi yang dihosting sendiri; evaluasi gaya gym |
| VisualWebArena | "Arena Web Visual" | WebArena yang berbasis visual; tangkapan layar adalah pengamatan |
| OSDunia | "Patokan agen desktop" | 369 tugas di Ubuntu/Windows/macOS |
| landasan GUI | "Pemetaan piksel-ke-elemen" | Model melokalkan elemen UI pada 1920x1080 |
| Pengetahuan operasional | "Pengetahuan OS" | Menu yang mana, pintasan yang mana, panel preferensi yang mana |
| OSWorld-G | "Ruang pembumian" | 564 sample khusus grounding + set training |
| OSDunia-Manusia | "Lintasan Emas" | Urutan tindakan ahli manual untuk mengukur efisiensi |
| Efisiensi lintasan | "Melangkah di atas emas" | Jumlah langkah agen dibagi minimum manusia |

## Bacaan Lanjutan

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — tolok ukur web empat aplikasi
- [Xie dkk., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — tolok ukur desktop lintas OS
- [Anthropic, Memperkenalkan penggunaan komputer](https://www.anthropic.com/news/3-5-models-and-computer-use) — Kemampuan Claude yang berbentuk patokan
- [OpenAI, Agen Pengguna Komputer](https://openai.com/index/computer-using-agent/) — nomor OSWorld dan WebArena
