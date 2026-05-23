# AI Scientist v2 — Penelitian Otonom Tingkat Lokakarya

> Ilmuwan AI Sakana v2 (Yamada dkk., arXiv:2504.08066) menjalankan putaran penelitian lengkap: hipotesis, code, eksperimen, angka, penulisan, penyerahan. Ini adalah sistem pertama yang menghasilkan makalah yang lulus tinjauan sejawat pada lokakarya ICLR 2025. Evaluasi independen (Beel et al.) menemukan 42% eksperimen gagal karena kesalahan pengkodean dan tinjauan literatur sering kali salah memberi label pada konsep yang sudah ada sebagai novel. Dokumen Sakana sendiri memperingatkan bahwa basis code mengeksekusi code yang ditulis LLM dan merekomendasikan isolasi Docker. Kedua bagian dari gambaran itu adalah intinya.

**Type:** Learn
**Language:** Python (stdlib, mainan mesin negara loop penelitian)
**Prerequisites:** Fase 15 · 03 (AlphaEvolve), Fase 15 · 04 (DGM)
**Waktu:** ~60 menit

## Masalah

Penelitian adalah tugas terbuka. Berbeda dengan penelusuran algoritmik AlphaEvolve atau modifikasi mandiri yang dibatasi patokan DGM, hasil penelitian tidak memiliki kriteria kebenaran yang dapat diperiksa oleh mesin. Sebuah makalah dinilai oleh reviewer, bukan tes unit. Hal ini membuat lingkaran ini semakin sulit untuk ditutup – dan akan lebih berharga jika ditutup, karena penelitian adalah tempat dimana kemajuan terus bertambah.

AI Scientist v1 (Sakana, 2024) menutup loop dengan memulai dari template yang dibuat oleh manusia. LLM mengisi eksperimen dalam perancah tetap. AI Scientist v2 (Yamada et al., 2025) menghilangkan persyaratan template dengan menggunakan penelusuran pohon agen dengan loop kritik model bahasa visi. Sistem ini menghasilkan ide, menerapkan eksperimen, menghasilkan angka, menulis makalah, dan mengulangi umpan balik pengulas.

Putusan tinjauan sejawat: satu makalah yang dihasilkan v2 diterima di lokakarya ICLR 2025 (dengan pengungkapan). Keputusan evaluasi independen: sistem ini jauh dari dapat diandalkan. Keduanya benar.

## Konsep

### Arsitektur

1. **Pembuatan ide.** LLM mengusulkan ide penelitian berdasarkan topik dan literatur sebelumnya. v1 templat bekas; v2 menggunakan penelusuran agen pada sejumlah hipotesis.
2. **Pemeriksaan kebaruan.** Langkah pengambilan literatur memeriksa apakah ide telah dipublikasikan. Pada phase inilah evaluasi Beel dkk menemukan adanya kesalahan pelabelan – metode yang sudah ada sering kali diklasifikasikan sebagai metode baru.
3. **Rencana eksperimen.** Agen menyusun protokol eksperimental dan menulis code.
4. **Eksekusi.** Code dijalankan di sandbox. Kegagalan dimasukkan kembali ke dalam loop percobaan ulang. Dalam pengukuran Beel et al., 42% eksperimen gagal karena kesalahan pengkodean pada phase ini.
5. **Pembuatan gambar.** Model bahasa visi membaca gambar yang dihasilkan dan menulis ulang gambar tersebut agar lebih jelas. Ini adalah tambahan teknis utama v2.
6. **Penulisan.** LLM menyusun makalah, mengulanginya dengan peninjau internal.
7. **Opsional: penyerahan.** Makalah diserahkan ke suatu tempat.

### Arti dari hasil penerimaan lokakarya

Satu makalah yang dihasilkan v2 lolos tinjauan sejawat pada lokakarya ICLR 2025. Penulis mengungkapkan asal usul makalah tersebut kepada komite program. Penerimaan adalah titik data; ini bukanlah izin untuk mengklaim sistem "melakukan penelitian".

Konteks penting: makalah lokakarya memiliki standar yang lebih rendah dibandingkan makalah konferensi utama. Tinjauan sejawat berisik; sebagian kecil kiriman diterima pada hari tertentu. Keberhasilan adalah bukti konsep, bukan klaim keandalan. Makalah Nature 2026 mendokumentasikan siklus end-to-end dan ditulis bersama oleh peneliti manusia; ini bukan "sistem menulis makalah Nature."

### Apa yang ditemukan oleh evaluasi independen

Beel dkk. (arXiv:2502.14297) menjalankan evaluasi eksternal. Temuan utama:- **Kegagalan eksperimen.** 42% eksperimen gagal karena kesalahan pengkodean (impor buruk, ketidakcocokan bentuk, variabel tidak ditentukan). Perulangan percobaan ulang menangkap beberapa, tidak semua.
- **Pelabelan yang salah terhadap hal-hal baru.** Langkah pengambilan literatur sering kali menandai konsep-konsep yang sudah ada sebagai sesuatu yang baru. Ini adalah penelitian yang setara dengan halusinasi.
- **Kesenjangan kualitas presentasi.** Kritik figur bahasa visi menghasilkan visual tingkat publikasi, yang menutupi kelemahan eksperimental yang mendasarinya.

Temuan terakhir adalah yang penting untuk fase ini. Sebuah sistem yang menghasilkan output yang meyakinkan tanpa melakukan penelitian yang meyakinkan adalah lebih berbahaya, bukan lebih aman, dibandingkan sistem yang jelas-jelas gagal. Evaluasi harus mencapai klaim yang mendasarinya, tidak berhenti pada angkanya saja.

### Masalah pelarian dari kotak pasir

Repositori Sakana sendiri README memperingatkan:

> Karena sifat perangkat lunak ini, yang mengeksekusi code yang dihasilkan LLM, kami tidak dapat menjamin keamanannya. Terdapat risiko paket berbahaya, akses web yang tidak terkontrol, dan munculnya proses yang tidak diinginkan. Gunakan risiko kamu sendiri dan pertimbangkan isolasi Docker.

Ini adalah bentuk operasional otonomi dalam domain yang belum terverifikasi. LLM menulis code; code berjalan; code dapat melakukan apa saja yang boleh dilakukan oleh proses tersebut. Tanpa sandbox yang membatasi tindakan sistem file, jaringan, dan proses, agen penelitian mandiri mana pun dapat mengambil data, membakar komputasi, atau menulis ulang dirinya sendiri.

Cerita sandbox AlphaEvolve lebih mudah karena evaluatornya ketat. Loop AI Scientist v2 menjalankan code terbuka dengan tujuan terbuka. Itulah mengapa diperlukan isolasi yang lebih kuat (minimum Docker; lebih disukai seccomp / gVisor) dan peninjauan manual terhadap setiap pengiriman sebelum meninggalkan sistem.

### Tempat v2 berada di tumpukan perbatasan

| Sistem | Sasaran | Jenis output | Penilai | Kegagalan yang diketahui |
|---|---|---|---|---|
| Evolusi Alpha | algoritma | code | satuan + patokan | dibatasi oleh ketelitian evaluator |
| DGM | perancah agen | code | Bangku SWE | peretasan hadiah |
| Ilmuwan AI v2 | makalah penelitian | teks + code + gambar | tinjauan sejawat (lemah) | kegagalan percobaan, kesalahan pelabelan, kelemahan penyembunyian polesan |

v2 memiliki evaluator otomatis terlemah dari ketiganya, permukaan output terluas, dan jalur terpendek ke artefak publik. Pengendalian operasional (sandbox, review, pengungkapan) melakukan sebagian besar pekerjaan keselamatan.

## Pakai

`code/main.py` mensimulasikan loop v2 sebagai mesin negara: ide → pemeriksaan kebaruan → eksperimen → gambar → penulisan → ulasan → terima atau ulangi. Setiap negara bagian memiliki probabilitas kegagalan yang dapat dikonfigurasi yang diambil dari Beel et al. temuan. Jalankan simulator untuk N loop dan hitung:

- Berapa banyak ide yang mencapai pengajuan.
- Berapa banyak kiriman yang memiliki cacat eksperimental kritis yang disembunyikan oleh kertas yang dipoles.
- Bagaimana anggaran percobaan ulang mengorbankan kualitas vs hasil.

## Kirim

`outputs/skill-ai-scientist-sandbox-review.md` adalah daftar periksa tinjauan dua gerbang untuk apa pun yang dihasilkan oleh agen loop penelitian sebelum meninggalkan kotak pasir.

## Latihan

1. Jalankan `code/main.py` dengan parameter default. Berapa fraksi putaran yang menghasilkan kertas "bersih"? Fraksi manakah yang menghasilkan makalah dengan cacat eksperimen-kegagalan yang telah disempurnakan oleh kritik gambar?

2. Defaultnya sudah menggunakan Beel dkk 42% / 25%. Jalankan kembali dengan `--experiment-failure 0.20 --novelty-mislabel 0.10` dan kemudian dengan `--experiment-failure 0.60 --novelty-mislabel 0.40`. Bagaimana pembagian yang sempurna namun cacat di antara kedua proses tersebut?

3. Baca README repo AI Scientist v2 Sakana tentang persyaratan sandbox. Sebutkan dua batasan tambahan (selain Docker) yang akan kamu terapkan untuk pengoperasian otonom beberapa hari.4. Baca Beel dkk. Bagian 4 tentang kesenjangan kualitas presentasi. Rancang satu evaluator tambahan yang dapat menangkap makalah yang tampak bagus namun cacat secara eksperimental.

5. Usulkan protokol peninjauan manusia untuk output agen penelitian yang skalanya lebih baik daripada "seorang PhD membaca setiap makalah." Identifikasi hambatan dan desain di sekitarnya.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| Ilmuwan AI v1 | "Agen penelitian templat Sakana" | Eksperimen yang diisi ke dalam perancah tetap |
| Ilmuwan AI v2 | "Agen penelitian bebas template" | Pencarian pohon agen dengan kritik gambar VLM |
| Pencarian pohon agen | "Agen Penelitian Percabangan" | Memperluas beberapa rencana eksperimen secara paralel; plum oleh kritikus internal |
| Kritik bahasa visi | "VLM memoles gambar" | Model multimodal membaca angka dan menulis ulang untuk kejelasan |
| Pengambilan literatur | "Pemeriksaan baru" | Menelusuri karya sebelumnya untuk mengonfirmasi kebaruan ide — didokumentasikan dengan label yang salah |
| Penyamaran Polandia | "Kertas cantik, penelitian rusak" | Kualitas presentasi melebihi kualitas eksperimental; menyembunyikan kelemahan |
| Pelarian kotak pasir | "Code LLM pecah" | Code yang dieksekusi agen melakukan hal-hal yang tidak diinginkan oleh perancang loop |

## Bacaan Lanjutan

- [Yamada dkk. (2025). AI Scientist-v2](https://arxiv.org/abs/2504.08066) — makalah.
- [Blog Sakana tentang publikasi Nature 2026](https://sakana.ai/ai-scientist-nature/) — ringkasan vendor dengan konteks tinjauan sejawat.
- [Beel dkk. (2025). Evaluasi independen terhadap The AI ​​Scientist](https://arxiv.org/abs/2502.14297) — nomor evaluasi eksternal.
- [Makalah Sakana AI Scientist v1](https://arxiv.org/abs/2408.06292) — template pendahulunya.
- [Anthropic — Mengukur otonomi agen AI](https://www.anthropic.com/research/measuring-agent-autonomy) — kerangka agen penelitian terbuka yang lebih luas.
