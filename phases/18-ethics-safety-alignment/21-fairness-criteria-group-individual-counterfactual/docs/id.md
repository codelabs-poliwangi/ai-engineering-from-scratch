# Kriteria Kewajaran — Kelompok, Individu, Kontrafaktual

> Tiga kelompok menyusun literatur keadilan. Keadilan kelompok: paritas demografis, peluang yang disamakan, kesetaraan akurasi penggunaan bersyarat — rata-rata tingkat kesetaraan di seluruh kelompok yang dilindungi. Keadilan individu (Dwork et al. 2012): individu yang serupa menerima keputusan yang sama; Kondisi Lipschitz pada peta keputusan. Keadilan kontrafaktual (Kusner et al. 2017): suatu keputusan dianggap adil bagi individu jika keputusan tersebut tidak berubah ketika atribut sensitif diubah secara kontrafaktual. Hasil teoretis tahun 2024 (NeurIPS 2024): terdapat trade-off CF-vs-akurasi yang melekat; metode model-agnostik mengubah prediktor yang optimal tetapi tidak adil menjadi prediktor CF dengan kehilangan akurasi terbatas. Mundur dari kontrafaktual (arXiv:2401.13935, Januari 2024): paradigma baru yang menghindari perlunya intervensi terhadap atribut yang dilindungi secara hukum. Rekonsiliasi filosofis (ICLR Blogposts 2024): dengan grafik sebab akibat, pemenuhan ukuran keadilan kelompok tertentu memerlukan keadilan kontrafaktual.

**Type:** Learn
**Language:** Python (stdlib, perbandingan tiga kriteria)
**Prerequisites:** Fase 18 · 20 (bias), Fase 02 (ML klasik)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Nyatakan tiga kriteria keadilan kelompok (kesetaraan demografis, peluang yang disamakan, kesetaraan akurasi penggunaan bersyarat) dan satu hasil ketidakmungkinan.
- Jelaskan keadilan individu melalui Dwork et al. Formulasi Lipschitz 2012.
- Jelaskan keadilan kontrafaktual dan ketergantungan grafik sebab akibat.
- Menjelaskan kemunduran kontrafaktual dan mengapa mereka menghindari masalah intervensi pada atribut yang dilindungi.

## Masalah

Lesson 20 adalah tentang mengukur bias. Lesson 21 adalah tentang menentukan standar keadilan yang harus dipenuhi oleh pengukuran. Ketiga kelompok tersebut memberikan standar yang berbeda secara struktural – suatu model dapat berupa keadilan kelompok dan tidak adil individu, adil kontrafaktual, dan tidak adil kelompok. Memilih standar adalah keputusan kebijakan; tidak ada standar yang optimal secara universal.

## Konsep

### Keadilan kelompok

- **Paritas demografis.** P(Y=1 | A=a) = P(Y=1 | A=a') untuk semua grup. Tingkat penerimaan yang sama.
- **Peluang yang disamakan.** P(Y=1 | Y*=y, A=a) = P(Y=1 | Y*=y, A=a'). TPR dan FPR yang sama antar grup.
- **Persamaan akurasi penggunaan bersyarat.** P(Y*=y | Y=y, A=a) = P(Y*=y | Y=y, A=a'). Nilai prediksi yang sama antar kelompok.

Ketidakmungkinan (Chouldechova, Kleinberg-Mullainathan-Raghavan 2017): ketiga hal ini tidak dapat dipenuhi secara bersamaan dalam tarif dasar yang tidak setara.

### Keadilan individu

Dwork dkk. 2012. Peta keputusan f adil secara individual sehubungan dengan metrik kesamaan tugas spesifik d jika |f(x) - f(x')| <= L * d(x, x') untuk beberapa konstanta Lipschitz L. Individu serupa mendapatkan keputusan serupa.

Membutuhkan pendefinisian d. Pertanyaan kebijakan, bukan statistik.

### Keadilan kontrafaktual

Kusner dkk. 2017. Suatu keputusan dianggap adil secara kontrafaktual bagi individu i jika, berdasarkan model populasi kausal, keputusan tersebut tidak berubah ketika atribut sensitif i diubah secara kontrafaktual.

Membutuhkan DAG kausal. DAG adalah pilihan pemodelan. Keadilan kontrafaktual sama dibenarkannya dengan DAG.

### Pertukaran CF vs akurasi

Teoritis NeurIPS 2024: terdapat trade-off yang melekat antara keadilan kontrafaktual dan akurasi prediksi. Metode model-agnostik dapat mengubah prediktor yang optimal namun tidak adil menjadi prediktor CF, dengan biaya akurasi yang terbatas. Biaya akurasi bergantung pada besarnya koefisien atribut sensitif dalam prediktor tidak adil yang optimal.### Mundur dari kontrafaktual

arXiv:2401.13935 (Januari 2024). Kontrafaktual tradisional memerlukan intervensi pada atribut sensitif – “apakah keputusan akan berubah jika orang tersebut berjenis kelamin berbeda.” Secara hukum, hal ini bermasalah: atribut yang dilindungi tidak dapat diintervensi dalam undang-undang klasifikasi.

Melacak kembali kontrafaktual membalikkan arah: alih-alih mengintervensi atribut tersebut, tanyakan kombinasi feature aktual individu yang mana yang akan menghasilkan hasil kontrafaktual. Hal ini menghindari keberatan hukum.

### Rekonsiliasi filosofis

Postingan Blog ICLR 2024. Dengan adanya grafik sebab akibat, pemenuhan ukuran keadilan kelompok tertentu memerlukan keadilan kontrafaktual. Ketiga keluarga tersebut tidak ortogonal; mereka adalah aspek berbeda dari struktur sebab akibat yang sama.

Hal ini tidak menyelesaikan teorema ketidakmungkinan (tarif dasar yang tidak setara masih menghalangi keadilan kelompok secara simultan). Namun hal ini menunjukkan pertentangan antara “kelompok” dan “individu/kontrafaktual” sebagian merupakan artefak karena tidak eksplisitnya model sebab akibat.

### Cocok untuk Fase 18

Lesson 20 adalah pengukuran bias. Lesson 21 adalah definisi keadilan. Lesson 22 adalah privasi (privasi diferensial). Lesson 23 adalah memberi tanda air. Ini adalah lesson yang berhubungan dengan alokasi yang melengkapi Lesson 7-11 yang berhubungan dengan penipuan.

## Pakai

`code/main.py` membuat dataset klasifikasi biner mainan dengan atribut sensitif dan tarif dasar yang tidak setara. Hitung paritas demografis, peluang yang disamakan, dan kesetaraan akurasi penggunaan bersyarat pada pengklasifikasi sederhana. Amati tiga metrik yang tidak setuju. Terapkan reweighting untuk paritas demografis dan amati biayanya pada dua paritas demografis lainnya.

## Kirim

Lesson ini menghasilkan `outputs/skill-fairness-criterion.md`. Berdasarkan klaim atau kebijakan keadilan, identifikasi kriteria mana yang diklaim, apakah model tersebut dapat memenuhi kriteria lainnya berdasarkan tarif dasar tidak setara yang diklaim, dan DAG penyebab apa yang menjadi dasar klaim tersebut.

## Latihan

1. Jalankan `code/main.py`. Laporkan tiga metrik grup pada data default. Terapkan reweighting dan pelaporan ulang yang ditargetkan pada paritas demografis.

2. Menerapkan Dwork dkk. Metrik keadilan individu tahun 2012 menggunakan L2 pada feature yang tidak sensitif. Laporkan berapa banyak pasangan yang melanggar Lipschitz dengan konstanta L=1.

3. Baca Kusner dkk. 2017. Buatlah DAG kausal dua feature sederhana untuk penilaian resume dan identifikasi kondisi keadilan kontrafaktual yang disiratkannya.

4. Makalah backtracking-counterfactuals tahun 2024 menghindari intervensi terhadap atribut yang dilindungi. Jelaskan skenario di mana hal ini penting bagi kepatuhan hukum.

5. Rekonsiliasi ICLR tahun 2024 berpendapat bahwa keadilan kelompok dan keadilan kontrafaktual merupakan dua aspek dari struktur yang sama. Pilih dua dari tiga kriteria di `code/main.py` dan nyatakan asumsi sebab akibat yang membuat kriteria tersebut setara.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Paritas demografi | "tarif yang sama" | P(Y=1 | A=a) sama antar grup |
| Peluang yang disamakan | "sama dengan TPR/FPR" | Tingkat positif benar dan positif palsu yang sama antar kelompok |
| Akurasi penggunaan bersyarat | "sama dengan PPV/NPV" | Nilai prediksi yang sama antar kelompok |
| Keadilan individu | "Kondisi Lipschitz" | Individu serupa mendapatkan keputusan serupa |
| Keadilan kontrafaktual | "invariansi perubahan kausal" | Keputusan tidak berubah karena perubahan atribut kontrafaktual |
| Mundur kontrafaktual | "jelaskan melalui aktual" | Kontrafaktual beralasan mundur dari hasil, bukan maju dari atribut |
| Teorema ketidakmungkinan | "tiga konflik" | Chouldechova / KMR 2017: kriteria kelompok saling eksklusif di bawah tarif dasar yang tidak setara |

## Bacaan Lanjutan

- [Pekerjaan dkk. — Keadilan melalui Kesadaran (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913) — keadilan individu
- [Kusner, Loftus, Russell, Silva — Keadilan Kontrafaktual (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856) — keadilan kontrafaktual
- [Chouldechova — Prediksi adil dengan dampak berbeda (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056) — ketidakmungkinan
- [Backtracking Counterfactuals (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935) — paradigma baru untuk intervensi atribut yang dilindungi
