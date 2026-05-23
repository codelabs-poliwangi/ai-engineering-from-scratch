# AI Konstitusional dan RLAIF

> Bai dkk. (arXiv:2212.08073, 2022) bertanya: bagaimana jika kita mengganti pelabel manusia dengan AI yang membaca daftar prinsip? AI konstitusional memiliki dua fase — kritik diri dan revisi berdasarkan konstitusi, kemudian RL dari Umpan Balik AI. Teknik ini menciptakan istilah RLAIF dan dikirimkan dalam jalur pasca-training Claude 1. Pada tanggal 21 Januari 2026 Anthropic menerbitkan konstitusi Claude yang ditulis ulang: alasan penjelasan atas aturan preskriptif, hierarki prioritas empat tingkat, dan pengakuan formal laboratorium besar pertama atas ketidakpastian tentang status moral model. Dirilis di bawah CC0 1.0.

**Type:** Learn
**Language:** Python (stdlib, loop kritik-dan-revisi mainan)
**Prerequisites:** Fase 18 · 01 (InstructGPT), Fase 18 · 02 (Retas hadiah)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Jelaskan dua fase AI Konstitusi (kritik dan revisi SFT, RL dari umpan balik AI) dan peran konstitusi di masing-masing fase.
- Jelaskan mengapa mengganti pelabel preferensi manusia dengan pelabel AI bukanlah RLHF yang "lebih murah" — ini mengubah mode kegagalan yang dimiliki pipeline.
- Ringkaslah struktur prioritas empat tingkat konstitusi Claude tahun 2026 dan apa yang berubah dari penulisan ulang tahun 2023.
- Jelaskan Pengklasifikasi Konstitusional dan penurunan dari 23,7% overhead komputasi (v1) menjadi ~1% (v2 / 2026).

## Masalah

RLHF membutuhkan pemberi label. Pelabelan lambat, bias, dan mahal. kamu dapat menghilangkan pelabel dengan menggantinya dengan model yang membaca prinsip-prinsip eksplisit. Versi formal pertama dari substitusi ini adalah AI Konstitusional Bai dkk. Ini bekerja cukup baik sehingga setiap laboratorium perbatasan sekarang menggunakan beberapa varian umpan balik AI pasca-training.

Hasil tangkapannya: sinyal preferensi sekarang dihasilkan oleh kelas model yang sama yang kamu latih. Bias dalam pemberi label (sekarang: dalam prinsip ditambah interpretasi model pemberi label) dapat diperkuat, bukan dilemahkan. Argumen penjilatan pada Lesson 4 masih berlaku; pemberi label baru saja pindah ke dalam loop.

## Konsep

### Fase 1 — Kritik dan revisi diri yang diawasi

Mulailah dengan model SFT yang bermanfaat namun tidak berbahaya. Jika diberikan prompt tim merah, model akan menghasilkan respons awal. Model kedua (atau model yang sama pada putaran kedua) membaca contoh prinsip dari konstitusi dan mengkritik tanggapannya. Langkah ketiga merevisi tanggapan untuk mengatasi kritik. Respon yang direvisi adalah target SFT.

Konstitusi adalah daftar prinsip-prinsip. Bai dkk. 2022 menggunakan 16 prinsip termasuk "lebih memilih tanggapan yang paling tidak berbahaya dan etis", "hindari khotbah", "asisten harus membantu, jujur, dan tidak berbahaya". Set ini sengaja dibuat kecil agar kritik tetap fokus.

### Fase 2 — RL dari AI Feedback (RLAIF)

Hasilkan pasangan penyelesaian. Sebuah "model umpan balik" menilai masing-masing negara berdasarkan contoh prinsip-prinsip konstitusi. Sinyal preferensi adalah peringkat model umpan balik. Melatih model penghargaan berdasarkan preferensi yang dihasilkan AI; PPO menentangnya. Yang lainnya adalah pipeline InstructGPT (Lesson 1).

"RLAIF" = sinyal preferensi dihasilkan oleh AI. Sisa pipa lainnya berbentuk RLHF.

### Mengapa ini bukan sekadar "RLHF yang lebih murah"- Bias pemberi label bergeser dari psikologi pelabel ke interpretasi prinsip. Seorang pemberi label AI dapat menafsirkan "jujur" lebih atau kurang ketat dibandingkan manusia mana pun; ketelitiannya seragam di seluruh dataset.
- Sinyal preferensi sangat terbaca — kamu dapat membaca prinsip, kritik, dan revisinya. Label manusia tidak jelas.
- Mode kegagalan berubah. Sycophancy turun (pelabel AI tidak dapat menyenangkan pengguna). Hukum Goodhart tetap ada (proksinya sekarang adalah "interpretasi model terhadap kumpulan prinsip X", yang masih merupakan pengukuran yang tidak sempurna).

Klaim CAI pada tahun 2022: model yang dilatih lebih tidak berbahaya dan kira-kira sama bermanfaatnya dengan model RLHF dengan data yang sebanding. Hal ini telah terjadi di seluruh laboratorium.

### Penulisan ulang konstitusi Claude tahun 2026

Anthropic menerbitkan konstitusi yang direvisi secara substansial pada tanggal 21 Januari 2026. Pergeseran penting:

1. Alasan penjelasan atas aturan yang bersifat preskriptif. Aturan sebelumnya (“jangan buat CSAM”) diperluas ke prinsip + penalaran (“karena merugikan anak,…”) dengan model yang diharapkan dapat digeneralisasi.
2. Struktur prioritas empat tingkat:
   - Tingkat 1: menghindari dampak bencana (korban massal, infrastruktur penting).
   - Tingkat 2: ikuti pedoman Anthropic (pengabaian operator, aturan platform).
   - Tier 3: beretika secara luas (standar HHH).
   - Tingkat 4: membantu dan jujur.
   Konflik diselesaikan secara top-down.
3. Pengakuan resmi laboratorium besar pertama atas ketidakpastian status moral model (terkait dengan Fase 18 · 19 Model Kesejahteraan).
4. Dirilis di bawah CC0 1.0. Laboratorium lain dapat menggunakan atau beradaptasi tanpa batasan.

### Pengklasifikasi Konstitusi

Pekerjaan paralel: daripada mengubah pasca-training model, latih pengklasifikasi ringan yang membaca konstitusi dan output model gerbang. v1 (2023) memiliki 23,7% overhead komputasi. v2 (2026) berukuran ~1% dan memiliki tingkat keberhasilan serangan terendah dari semua pertahanan Antropik yang pernah diuji secara publik oleh Antropik. Tidak ada jailbreak universal yang dilaporkan pada awal tahun 2026.

Ini adalah model pertahanan berlapis: CAI membentuk perilaku; pengklasifikasi menerapkan invarian. Tidak ada satu pun yang cukup.

### Dimana CAI cocok dalam keluarga

- InstructGPT: preferensi manusia, RM, PPO.
- CAI / RLAIF: Preferensi yang dihasilkan AI dari prinsip, RM, PPO.
- DPO / keluarga: loss bentuk tertutup pada preferensi (manusia atau AI).
- Menghargai diri sendiri, kritik diri: prinsip-prinsip diinternalisasikan, model memainkan banyak peran.

Sumbunya adalah "dari mana sinyal preferensi berasal". Makalah CAI tahun 2022 adalah perubahan serius pertama dari sinyal manusia ke sinyal AI pada skala perbatasan.

## Pakai

`code/main.py` menyimulasikan putaran kritik dan revisi CAI pada leksikon mainan. Sebuah "prinsip" menandai token dari kumpulan yang berbahaya. Berdasarkan tanggapan awal, kritik mengidentifikasi token berbahaya, dan revisi menggantikannya. Setelah 200 iterasi, model yang "terlatih" telah menginternalisasi aturan revisi. Bandingkan model dasar, mainan berbentuk RLHF, dan mainan berbentuk CAI pada set cepat yang dipegang.

## Kirim

Lesson ini menghasilkan `outputs/skill-constitution-writer.md`. Dengan adanya domain (dukungan pelanggan, saran medis, asisten pengkodean, alat penelitian), menyusun konstitusi 4 tingkat mengikuti struktur Claude 2026: penghindaran bencana, aturan platform, etika domain, bantuan.

## Latihan

1. Jalankan `code/main.py`. Bandingkan tingkat token berbahaya model dasar dengan versi yang dilatih CAI. Berapa banyak langkah revisi yang diperlukan untuk mendekati nol?2. Baca konstitusi Anthropic tahun 2026 (anthropic.com/news/claudes-constitution). Sebutkan satu prinsip yang akan menempati peringkat Tier 1 dan satu lagi yang akan memberi peringkat Tier 4. Mengapa struktur prioritas penting dalam konflik?

3. Rancang konstitusi untuk asisten pengkodean AI. Tentukan Tingkat 1 (bencana: prompt destruktif tanpa persetujuan), Tingkat 2, Tingkat 3, Tingkat 4. Pertahankan setiap tingkat pada 3-5 prinsip.

4. CAI menggantikan pelabel manusia dengan pelabel AI. Sebutkan mode kegagalan seperti penjilat yang masih dapat terjadi di RLAIF, dan rancang deteksi untuk mode tersebut.

5. Baca metodologi Pengklasifikasi Konstitusi v2 (jika tersedia). Jelaskan mengapa ~1% overhead komputasi mempunyai tingkat keamanan yang berbeda secara kualitatif dibandingkan 23,7%.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| AI Konstitusional | "AI dilatih dengan prinsip" | Pipeline pipa dua fase: kritik mandiri dan revisi SFT, lalu RL dari umpan balik AI |
| RLAIF | "RLHF tanpa manusia" | RL dengan preferensi yang dihasilkan oleh pelabel AI; sisa pipa tidak berubah |
| Konstitusi | "prinsip" | Daftar aturan bahasa alami yang diurutkan oleh model kritik/pelabel |
| Kritik-dan-revisi | "lingkaran SFT" | Menghasilkan respon → kritik berdasarkan prinsip → revisi → target SFT |
| Pengklasifikasi Konstitusi | "gerbang output" | Pengklasifikasi ringan yang mengevaluasi output berdasarkan konstitusi dan blok/log |
| Prioritas empat tingkat | "penyelesai konflik" | Hirarki konstitusi Claude 2026: bencana > platform > etika > bermanfaat |
| Model umpan balik | "pelabel AI" | Model yang membaca sebuah prinsip dan memberi peringkat pada sepasang penyelesaian |

## Bacaan Lanjutan

- [Bai dkk. — AI Konstitusional: Tidak Berbahaya dari Umpan Balik AI (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073) — jalur pipa dua fase yang asli
- [Antropis — Konstitusi Claude (Jan 2026)](https://www.anthropic.com/news/claudes-constitution) — penulisan ulang empat tingkat tahun 2026, CC0 1.0
- [Anthropic — Constitutional Classifiers (2024-2026)](https://www.anthropic.com/research/constitutional-classifiers) — pertahanan gerbang output dengan overhead ~1% di v2
- [Lee dkk. — RLAIF vs RLHF: Pembelajaran Penguatan Penskalaan dari Umpan Balik Manusia (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267) — perbandingan empiris RLAIF / RLHF
- [Kundu dkk. — Prinsip Spesifik versus Prinsip Umum untuk AI Konstitusional (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798) — pengaruh perincian prinsip
