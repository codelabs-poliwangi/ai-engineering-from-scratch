# Pengawasan yang Dapat Diskalakan dan Generalisasi Lemah hingga Kuat

> Luka bakar dkk. (OpenAI Superalignment, "Weak-to-Strong Generalization", 2023) mengusulkan proksi untuk masalah superalignment: menyempurnakan model yang kuat menggunakan label yang dihasilkan oleh model yang lebih lemah. Jika model yang kuat melakukan generalisasi dengan benar dari pengawasan lemah yang tidak sempurna, metode penyelarasan skala manusia saat ini dapat meluas ke sistem manusia super. Pengawasan yang terukur dan W2SG saling melengkapi. Pengawasan yang terukur (debat, pemodelan penghargaan rekursif, decomposition tugas) meningkatkan kemampuan efektif pengawas sehingga dapat mengikuti model yang diawasi. W2SG memastikan model yang kuat dapat melakukan generalisasi dengan benar dari pengawasan tidak sempurna apa pun yang diberikan oleh pengawas. Debat Membantu W2SG (arXiv:2501.13124, Januari 2025) menggabungkannya.

**Type:** Learn
**Language:** Python (stdlib, simulator celah W2SG)
**Prerequisites:** Fase 18 · 01 (mengikuti instruksi), Fase 18 · 10 (Kontrol AI), Fase 09 (fondasi RL)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Definisikan pengawasan yang terukur dan generalisasi dari lemah ke kuat dan jelaskan bagaimana keduanya saling melengkapi.
- Jelaskan Burns dkk. Penyiapan eksperimental tahun 2023: menyempurnakan GPT-4 menggunakan label dari GPT-2.
- Jelaskan metrik pemulihan kesenjangan kinerja (PGR) dan apa yang diukurnya.
- Sebutkan tiga mekanisme pengawasan skalabel utama (debat, pemodelan penghargaan rekursif, decomposition tugas) dan masing-masing satu kekuatannya.

## Masalah

Setiap teknik penyelarasan sejauh ini di Fase 18 mengasumsikan pengawas dapat mengevaluasi perilaku model. Jika modelnya adalah manusia super, maka pengawas adalah mata rantai yang lemah. Pertanyaan superalignment: dapatkah supervisor yang lebih lemah menghasilkan model yang lebih kuat dan selaras?

Luka bakar dkk. kurangi hal ini menjadi pengaturan empiris yang dioperasionalkan: awasi yang kuat dengan yang lemah, ukur seberapa besar kemampuan model yang kuat bertahan dalam pengawasan yang lemah. Ini bukanlah solusi untuk superalignment – ​​ini adalah sebuah proksi yang memungkinkan kita mengukur kemajuan.

## Konsep

### W2SG: Burns dkk. pengaturan

- Model lemah: kelas GPT-2.
- Model kuat: kelas GPT-4.
- Target: batas atas GPT-4 yang kuat pada tugas tersebut.

Prosedur:
1. Dapatkan prediksi zero-shot model lemah pada suatu tugas.
2. Menyempurnakan model kuat pada data berlabel lemah.
3. Ukur akurasi model yang kuat dan telah disetel dengan baik.

Garis dasar: akurasi model yang lemah, batasan pengawasan model yang kuat pada label emas.

Metrik kesenjangan: Pemulihan Kesenjangan Kinerja (PGR) = (diselaraskan - lemah) / (batas atas - lemah). PGR sebesar 1,0 berarti pengawasan yang lemah menutup kesenjangan sepenuhnya; PGR 0 berarti lemahnya pengawasan tidak membantu.

### Luka bakar dkk. temuan empiris

Di seluruh tugas NLP, teka-teki catur, dan pemodelan hadiah, PGR secara konsisten positif: model yang kuat digeneralisasikan melampaui kesalahan supervisor yang lemah. PGR spesifik tugas bervariasi dari ~20% hingga ~80%. Model yang kuat "memahami" tugas yang dimaksudkan dari label yang tidak sempurna, mengisinya dengan prior yang sudah terlatih.

Keterbatasan Burns dkk. catatan:
- Kesenjangan lemah-ke-kuat di sini adalah kemampuan, bukan keselarasan. Model yang lemah telah selaras; model yang benar-benar manusia super mungkin bisa disejajarkan dengan manusia tetapi dengan kesenjangan kemampuan yang jauh lebih besar.
- Generalisasi model yang kuat mungkin lebih berkaitan dengan tugas yang diselesaikan sebelumnya dibandingkan dengan memulihkan maksud sebenarnya.
- Memunculkan pengetahuan terpendam dari model yang kuat adalah masalah sebenarnya; PGR mengukur operasionalisasi tertentu.

### Pengawasan terukur: tiga mekanisme- **Debat (Irving dkk. 2018).** Dua contoh U berargumentasi pada pihak yang berlawanan; hakim yang lemah memutuskan. Asumsinya: menemukan jawaban benar yang meyakinkan lebih mudah daripada menemukan jawaban salah yang meyakinkan. Penelitian terbaru pada tahun 2024-2025 (Khan dkk., Michael dkk.) menunjukkan perdebatan terkadang membantu dan terkadang tidak, dan apakah debat membantu bergantung pada struktur tugas.
- **Pemodelan Penghargaan Rekursif (Leike dkk. 2018).** U membantu manusia melatih model penghargaan untuk U+1. Kemampuan efektif pengawas tumbuh seiring dengan kemampuan model.
- **Decomposition Tugas (Christiano, Shlegeris, Amodei 2018).** Uraikan tugas sulit menjadi sub-tugas yang dapat diperiksa manusia, secara rekursif. Mengasumsikan decomposition.

Setiap mekanisme mengasumsikan sesuatu tentang struktur tugas atau penyelarasan komponen perantara.

### Mengapa pengawasan yang terukur dan W2SG saling melengkapi

Pengawasan yang terukur meningkatkan kualitas sinyal efektif pengawas.
W2SG menutup kesenjangan dari sinyal tidak sempurna apa pun yang dapat diberikan oleh pengawas.

Lang dkk. — Debat Membantu Generalisasi Lemah-ke-Kuat (arXiv:2501.13124) menggabungkan kedua hal tersebut: protokol debat memberikan label lemah yang lebih baik, dan model yang kuat dilatih berdasarkan label tersebut. Melaporkan keuntungan PGR pada tugas NLP.

### Drama organisasi

Tim Superalignment OpenAI dibubarkan pada Mei 2024 setelah kepergian Jan Leike ke Anthropic. Agenda (pengawasan terukur, W2SG, penelitian penyelarasan otomatis) dilanjutkan di laboratorium Antropik dan akademik — MATS (Lesson 28), Redwood (Lesson 10), Apollo (Lesson 8), METR (Lesson 28). Struktur organisasi berubah; pertanyaan penelitian tidak.

### Cocok untuk Fase 18

Lesson 6-10 menggambarkan ancaman dan paradigma defensif dengan asumsi U tidak dapat dipercaya. Lesson 11 adalah paradigma ofensif: buatlah pengawas cukup kuat untuk memverifikasi keselarasan U. Lesson 12-16 kemudian beralih ke perangkat praktis evaluasi permusuhan.

## Pakai

`code/main.py` menyimulasikan penyempurnaan W2SG pada tugas sintetis. Pelabelan yang lemah memiliki akurasi 70% dengan kesalahan terstruktur; model yang kuat memiliki batas atas 95% pada label emas. kamu menyempurnakan model kuat pada label lemah, mengukur ZPT, dan membandingkan model kuat pada emas dan lemah saja.

## Kirim

Lesson ini menghasilkan `outputs/skill-w2sg-pgr.md`. Dengan adanya deskripsi pengaturan pengawasan, hal ini mengidentifikasi pengawas yang lemah, model yang kuat, kualitas pengawasan, dan menghitung (atau meminta) PGR. Hal ini menandai apakah klaim tersebut adalah "lemah bisa mengawasi yang kuat" atau "lemah + mekanisme pengawasan bisa mengawasi yang kuat."

## Latihan

1. Jalankan `code/main.py`. Laporkan PGR untuk akurasi_lemah = 0,60, 0,70, 0,80. Jelaskan bentuk kurva PGR.

2. Ubah pelabel lemah menjadi kesalahan terstruktur (misalnya, selalu salah pada kelas input tertentu). Apakah PGR bertambah, berkurang, atau tetap sama? Menjelaskan.

3. Baca Burns dkk. 2023 Bagian 4.3 (tugas NLP). Mereproduksi intuisi "loss tambahan kepercayaan": ketika model yang kuat lebih percaya diri daripada label yang lemah, siapa yang menang?

4. Rancang protokol pengawasan terukur yang menggabungkan perdebatan dan decomposition tugas untuk tugas rekayasa perangkat lunak. Sebutkan salah satu modus kegagalan setiap komponen dan jelaskan bagaimana kombinasi tersebut mengatasi atau gagal mengatasi masing-masing komponen.

5. Mengartikulasikan apa yang akan memalsukan klaim “generalisasi lemah ke kuat adalah jalan yang layak menuju superalignment”. Bersikaplah spesifik tentang tanda empiris yang perlu kamu lihat.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Pengawasan terukur | "membuat pengawas lebih kuat" | Mekanisme yang meningkatkan kemampuan pengawas untuk mengevaluasi model yang lebih berkemampuan |
| W2SG | "lemah mengawasi yang kuat" | Menyempurnakan model yang kuat pada label yang lemah dan mengukur kemampuan yang dipulihkan |
| PGR | "kesenjangan kinerja pulih" | (disetel - lemah) / (langit-langit - lemah); 1.0 = tertutup penuh, 0 = tidak ada bantuan |
| Debat | "dua contoh U berdebat" | Mekanisme pengawasan yang terukur di mana hakim yang lemah memilih antara dua pembela U |
| RRM | "pemodelan imbalan rekursif" | U membantu melatih model penghargaan untuk U+1; kemampuan pengawas melacak U |
| Decomposition tugas | "sub-tugas yang diperiksa manusia" | Bagi tugas sulit menjadi sub-tugas yang dapat diverifikasi manusia, secara rekursif |
| Penyelarasan Super | "menyelaraskan AI manusia super" | Agenda penelitian yang berkaitan dengan penyelarasan model yang tidak dapat dievaluasi secara langsung oleh manusia |

## Bacaan Lanjutan

- [Burns dkk. — Generalisasi Lemah-ke-Kuat (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — makalah W2SG
- [Irving, Christiano, Amodei — Keamanan AI melalui debat (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — mekanisme debat
- [Leike dkk. — Penyelarasan agen yang dapat diskalakan melalui pemodelan imbalan (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — pemodelan imbalan rekursif
- [Khan dkk. — Berdebat dengan LLM yang Lebih Persuasif Menghasilkan Jawaban yang Lebih Jujur (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) — Studi empiris tahun 2024 tentang debat dengan pendebat yang lebih kuat
- [Lang dkk. — Debat Membantu Generalisasi Lemah-ke-Kuat (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) — kombinasi debat tahun 2025 + W2SG
