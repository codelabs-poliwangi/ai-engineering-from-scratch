# Privasi Diferensial untuk LLM

> DP-SGD tetap menjadi standar — pembaruan gradient yang disuntikkan noise memberikan jaminan formal (epsilon, delta). Overhead dalam komputasi, memori, dan utilitas sangat besar; penyempurnaan DP yang efisien parameter (LoRA + DP-SGD) adalah konfigurasi umum tahun 2025 (ACM 2025). Dua kumpulan bukti yang mengalami ketegangan: inference keanggotaan berbasis kenari (Duan et al., 2024) melaporkan keberhasilan yang terbatas terhadap model bahasa; ekstraksi training data (Carlini et al., 2021; Nasr et al., 2025) memulihkan hafalan kata demi kata yang substansial. Resolusi (arXiv:2503.06808, Maret 2025): kesenjangannya terletak pada apa yang diukur — canary yang disisipkan vs data yang "paling dapat diekstraksi". Desain canary baru memungkinkan MIA berbasis loss tanpa model bayangan dan menghasilkan audit DP nontrivial pertama dari LLM yang dilatih berdasarkan data nyata dengan jaminan DP yang realistis. Alternatif: PMixED (arXiv:2403.15638) — prediksi pribadi pada waktu inference melalui campuran pakar pada distribusi token berikutnya; Pembuatan data sintetis DP (Google Research 2024). Serangan yang muncul: Pembalikan Privasi Diferensial melalui Umpan Balik LLM — kebocoran skor kepercayaan.

**Type:** Build
**Language:** Python (stdlib, injeksi kebisingan DP-SGD dan demonstrasi akuntan ε-δ)
**Prerequisites:** Fase 01 · 09 (teori informasi), Fase 10 · 01 (training model besar)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Tentukan privasi diferensial (epsilon, delta) dan nyatakan resep DP-SGD.
- Jelaskan ketegangan tahun 2024-2025: canary MIA vs ekstraksi training data memberikan gambaran yang berbeda.
- Jelaskan PMixED dan mengapa prediksi privat waktu inference merupakan alternatif dari training DP.
- Jelaskan Pembalikan Privasi Diferensial melalui serangan Umpan Balik LLM.

## Masalah

LLM menghafal. Carlini dkk. Tahun 2021 menunjukkan model bahasa produksi mereproduksi teks training kata demi kata sesuai permintaan. DP adalah pertahanan formal: latih sehingga keluarannya terbukti tidak sensitif terhadap contoh training apa pun. Bukti tahun 2024-2025 menunjukkan DP-SGD diperlukan namun nilai ε yang diterapkan mungkin tidak cocok dengan model ancaman.

## Konsep

### (ε, δ)-privasi diferensial

Algoritme acak M adalah (ε, δ)-DP jika untuk dua dataset yang berbeda dalam satu contoh dan kejadian apa pun S:
P(M(D') di S) <= e^ε * P(M(D') di S) + δ.

Interpretasi: distribusi output cukup dekat (diparametrikan dengan ε) sehingga kontribusi individu mana pun tidak dapat disimpulkan secara andal, kecuali dengan probabilitas δ.

### DP-SGD

Abadi dkk. 2016. Resep standar:
1. Cicipi batch mini.
2. Hitung gradient per contoh.
3. Klip setiap gradient per contoh ke ambang batas C.
4. Jumlahkan gradient yang terpotong dan tambahkan noise Gaussian dengan std σ * C.
5. Gunakan jumlah berisik untuk memperbarui parameter.

Biaya privasi dilacak oleh seorang akuntan (Akuntan Momen, akuntan Rényi DP). Nilai ε yang dilaporkan dalam literatur LLM sangat bervariasi berdasarkan model ancaman, sensitivitas data, dan target utilitas; tidak ada ε default yang "aman" secara universal. Contoh yang dipublikasikan berkisar sekitar ε ≈ 1–10 di beberapa pengaturan training LLM, namun ini hanya ilustratif — bukan default yang direkomendasikan. ε yang lebih rendah umumnya memerlukan lebih banyak kebisingan dan dapat meningkatkan hilangnya utilitas.

### LoRA + DP-SGD

DP-SGD penuh model frontier sangatlah mahal. LoRA (Hu dkk. 2022) membatasi pembaruan gradient pada adaptor kecil, sehingga mengurangi penyimpanan gradient per contoh. LoRA + DP-SGD adalah konfigurasi umum tahun 2025. Jaminan DP berlaku untuk adaptor; model dasar dianggap tetap.

### Ketegangan 2024-2025

Dua bukti:- **Canary MIA (Duan et al. 2024).** Masukkan canary unik ke dalam training data, ukur apakah penyerang inference keanggotaan dapat mengidentifikasinya. Melaporkan keberhasilan yang terbatas pada model bahasa. Menyarankan MIA itu sulit.
- **Ekstraksi training data (Carlini 2021, Nasr dkk. 2025).** Meminta model dengan awalan; mengukur apakah ia memulihkan teks kata demi kata dari training. Melaporkan hafalan yang substansial. Menyarankan MIA itu mudah dalam arti yang relevan.

Resolusi Maret 2025 (arXiv:2503.06808): keduanya mengukur hal yang berbeda. MIA bertanya "apakah contoh e ada di D?" pada burung kenari insert. Ekstraksi menanyakan "apa yang dapat saya pulihkan dari D?" Contoh yang "paling mudah diekstraksi" adalah yang penting bagi privasi; kenari tidak melaporkan hal ini karena tidak dioptimalkan untuk dapat diekstraksi.

Desain kenari baru. MIA berbasis loss tanpa model bayangan. Audit DP nontrivial pertama dari LLM berdasarkan data nyata dengan jaminan DP yang realistis.

### Alternatif untuk training DP

- **PMixED (arXiv:2403.15638).** Prediksi pribadi pada waktu inference. Campuran para ahli dalam distribusi token berikutnya; setiap pakar melihat sebagian training data; agregasi menambah kebisingan untuk DP. Menghindari training DP sepenuhnya.
- **Pembuatan data sintetis DP (Google Research 2024).** Penyempurnaan LoRA dengan DP-SGD, mengambil sample data sintetis, melatih pengklasifikasi downstream pada data sintetis.

Keduanya menghindari biaya utilitas training DP penuh dengan mengorbankan model ancaman yang berbeda.

### Pembalikan Privasi Diferensial melalui Umpan Balik LLM

Serangan yang muncul pada tahun 2025. Gunakan skor keyakinan model yang dilatih DP sebagai ramalan untuk mengidentifikasi ulang individu. Bahkan ketika output tidak bocor, distribusi kepercayaan bisa terjadi.

Pertahanannya: jangan mengekspos rahasia, atau memotong/mengukurnya sebelum diekspos. Ini merupakan persyaratan tambahan di luar training (ε, δ)-DP.

### Cocok untuk Fase 18

Lesson 20-21 adalah bias/keadilan. Lesson 22 adalah privasi. Lesson 23 adalah asal melalui watermarking. Lesson 27 mencakup layer asal data peraturan.

## Pakai

`code/main.py` menyimulasikan DP-SGD pada dataset klasifikasi biner mainan. kamu dapat menyapu pengganda kebisingan σ dan norm kliping C serta melacak anggaran (ε, δ) dan biaya akurasi. Sebuah "serangan kenari" menyisipkan contoh training unik dan mengukur apakah pengujian kehilangan log dapat mendeteksinya sebelum dan sesudah DP.

## Kirim

Lesson ini menghasilkan `outputs/skill-dp-audit.md`. Mengingat klaim DP pada penerapan model bahasa, DP akan mengaudit: nilai (ε, δ), akuntan yang digunakan, protokol evaluasi MIA, dan apakah vector paparan kepercayaan telah dinilai.

## Latihan

1. Jalankan `code/main.py`. Sapu σ dalam {0.5, 1.0, 2.0} dan laporkan trade-off akurasi (ε, δ). Identifikasi titik di mana utilitas runtuh.

2. Menerapkan penyisipan canary dan pengujian log-loss. Ukur tingkat deteksi sebelum dan sesudah DP-SGD pada σ = 1.0.

3. Baca Nasr dkk. 2025 tentang ekstraksi training data. Mengapa keberhasilan ekstraksi tidak menurun pada ε sedang? Apa implikasinya terhadap MIA sebagai evaluasi?

4. Rancang penerapan menggunakan PMixED (arXiv:2403.15638) yang beroperasi sepenuhnya pada waktu inference. Model ancaman apa yang dapat diatasi oleh PMixED namun tidak dapat diatasi oleh DP-SGD?

5. Sketsa Pembalikan DP melalui serangan Umpan Balik LLM. Rancang tindakan penanggulangan yang membatasi kebocoran skor kepercayaan dan memperkirakan biaya penerapannya.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| DP | "(ε, δ)-privasi diferensial" | Privasi formal: distribusi output ditutup pada perubahan dataset tetangga |
| DP-SGD | "SGD yang disuntikkan kebisingan" | Kliping gradient + penambahan noise Gaussian; training DP standar |
| LoRA + DP-SGD | "penyempurnaan pribadi yang efisien" | DP-SGD pada adaptor tingkat rendah; konfigurasi standar 2025 |
| MIA | "inference keanggotaan" | Serangan yang menentukan apakah suatu contoh ada dalam training data |
| kenari | "contoh tanda air yang dimasukkan" | Contoh training unik yang digunakan untuk mengukur kebocoran DP |
| Pcampur | "campuran inference pribadi" | DP waktu inference melalui campuran pakar pada distribusi token berikutnya |
| Pembalikan DP | "serangan kebocoran kepercayaan" | Serangan yang menggunakan kepercayaan model sebagai ramalan untuk identifikasi ulang |

## Bacaan Lanjutan

- [Abadi dkk. — DP-SGD (arXiv:1607.00133)](https://arxiv.org/abs/1607.00133) — algoritma training DP standar
- [Carlini dkk. — Mengekstraksi Data Training (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805) — makalah ekstraksi kanonik
- [Duan dkk. — Canary MIA di LLM (arXiv:2402.07841, 2024)](https://arxiv.org/abs/2402.07841) — MIA dengan kesuksesan terbatas
- [Kowalczyk dkk. — Audit DP untuk LLM (arXiv:2503.06808, Maret 2025)](https://arxiv.org/abs/2503.06808) — resolusi ketegangan
- [PMixED (arXiv:2403.15638)](https://arxiv.org/abs/2403.15638) — prediksi pribadi waktu inference
