# Voting, Konsistensi Diri, dan Topologi Debat

> Agregasi termurah: sample N agen independen, suara terbanyak. Wang dkk. Konsistensi mandiri tahun 2022 melakukan hal ini dengan satu model yang dijadikan sample sebanyak N kali. Multi-agen memperluasnya dengan agen **heterogen** untuk menghindari monokultur — model berbeda, prompt berbeda, suhu berbeda, konteks berbeda. Selain suara mayoritas, topologi perdebatan juga penting: MultiAgentBench (arXiv:2503.01935, ACL 2025) mengevaluasi koordinasi bintang/rantai/pohon/grafik dan menemukan **grafik terbaik untuk penelitian**, dengan "pajak koordinasi" melewati ~4 agen. AgentVerse (ICLR 2024) mendokumentasikan dua pola yang muncul – perilaku sukarela dan perilaku konformitas – dan konformitas merupakan feature (menemukan konsensus) dan risiko (groupthink, Lesson 24). Lesson ini memetakan ruang topologi, membangun setiap varian, dan mengukur pajak koordinasi.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 16 · 07 (Masyarakat Pikiran dan Debat), Fase 16 · 14 (Konsensus dan BFT)
**Waktu:** ~75 menit

## Masalah

Debat dapat meningkatkan akurasi (Du et al., arXiv:2305.14325). Hal ini juga dapat menurunkannya. Apakah perdebatan dapat membantu tergantung pada empat pilihan struktural:

1. Siapa berbicara kepada siapa (topologi).
2. Berapa putaran (Du 2023: putaran dan agen penting secara independen).
3. Apakah agen bersifat heterogen (model dasar yang berbeda mematahkan monokultur).
4. Apakah terdapat suara permusuhan (pengawas baja vs. pengawak jerami).

Tim yang menggabungkan "5 agen dan memilih" pada suatu tugas sering kali mengalami kemunduran dibandingkan dengan satu agen. Kegagalan tersebut tidak terjadi secara acak. Mereka melacak topologi dan heterogenitas. Lesson ini adalah peta topologi.

## Konsep

### Konsistensi diri, garis dasar model tunggal

Wang dkk. 2022 ("Konsistensi Diri Meningkatkan Penalaran Rantai Pemikiran") mengambil sample model yang sama sebanyak N kali pada suhu > 0 dan memberikan suara terbanyak pada jawaban jalur penalaran. Hasil pada GSM8K: peningkatan substansial dengan N=40 sample melalui satu dekode serakah. Konsistensi diri adalah pendahulu dari pemungutan suara multi-agen.

Batasan: konsistensi diri menggunakan satu model dasar. Kesalahan dikorelasikan dengan konstruksi. Jika model memiliki bias sistematis, semua N sample akan mengalami bias yang sama.

### Pemungutan suara multi-agen, perluasan yang heterogen

Ganti N sample dengan N agen *berbeda*. Model dasar berbeda (Claude, GPT, Llama), prompt berbeda, akses alat berbeda. Manfaatnya: kesalahan yang tidak berkorelasi. Biaya: agen yang berbeda memerlukan biaya yang berbeda pula; mengoordinasikannya menambah biaya tambahan.

Nama kanonik tahun 2026 untuk debat heterogen adalah **A-HMAD** — Debat Multi-Agen Heterogen Adversarial. Tidak diadopsi secara universal, namun makalah menggunakan istilah tersebut untuk "perdebatan model yang berbeda, yang mengurangi kesalahan yang berkorelasi dari keruntuhan monokultur."

### Keempat topologi

```
star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (fully connected)
```

Bintang: satu hub, yang lainnya hanya berbicara ke hub. Setara dengan supervisor-pekerja tanpa pipeline belakang.
Rantai: linier, setiap agen melihat output sebelumnya. Seperti pipa.
Pohon: hierarki, digunakan oleh sistem agen hierarki (Lesson 06).
Grafik: apa pun ke apa pun. Termasuk klik yang terhubung sepenuhnya dan DAG sewenang-wenang.

### Pajak koordinasi (MultiAgentBench)

MultiAgentBench (MARBLE, ACL 2025, arXiv:2503.01935) membuat tolok ukur bintang, rantai, pohon, grafik pada rangkaian tugas termasuk penelitian, pengkodean, dan perencanaan. Hasil utama yang diukur:- **Grafik** topologi menang dalam tugas penelitian. Informasi mengalir dari mana saja ke mana saja; agen dapat saling mengkritik.
- **Bintang** menang dalam tugas faktual yang dijawab cepat. Filter dan konsolidasi hub.
- **Rantai** menang pada pipeline pipa bertahap (penyempurnaan bertahap).
- **Pajak koordinasi** muncul melewati ~4 agen dalam topologi grafik. Biaya jam dinding dan token tumbuh lebih cepat daripada kualitas.

Batas atas 4 agen bersifat empiris, bukan fundamental. Hal ini mencerminkan kapasitas konteks LLM 2026: konteks masing-masing agen terisi dengan output rekan-rekannya, dan nilai marjinal dari penambahan agen N+1 turun setelah semua orang dapat melihat semua orang.

### Strategi Debat Multi-Agen ("Haruskah kita menjadi MAD?")

arXiv:2311.17371 adalah survei strategi MAD tahun 2023. Temuan utama yang ditiru oleh orang lain: Varian MAD yang *secara struktural mirip* dengan konsistensi mandiri (pengambilan sample + agregasi independen) sering kali memiliki kinerja di bawah konsistensi mandiri ketika menggunakan anggaran yang sama. MAD paling membantu ketika para agen benar-benar heterogen dan perdebatan mempunyai struktur yang saling bermusuhan (satu agen menentangnya).

### Pola yang muncul AgentVerse

AgentVerse (ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) mendokumentasikan dua perilaku yang muncul dari perdebatan multi-agen bahkan tanpa desain eksplisit:

- **Relawan.** Seorang agen menawarkan bantuan ("Saya dapat mengambil langkah berikutnya") tanpa diminta. Berguna: ini mengalokasikan pekerjaan ke agen yang paling mampu untuk sebuah subtugas.
- **Kesesuaian.** Agen menyesuaikan pendiriannya agar sesuai dengan kritik, meskipun kritikus tersebut salah. Ini setara dengan perdebatan tentang penjilatan (Lesson 14).

Kesesuaian adalah alasan mengapa perdebatan sampai kesepakatan memberi penghargaan kepada para penindas. Putaran terbatas dengan juri terpisah dimitigasi.

### Heterogenitas: kenop sebenarnya yang menggerakkan akurasi

Pola 2024-2026 dalam literatur praktis: menukar salah satu agen N kamu dengan model dasar yang berbeda memberikan peningkatan akurasi yang lebih besar daripada meningkatkan N sebanyak 1. Intuisinya adalah monokultur — setiap sumber kesalahan independen baru bernilai lebih dari sample tambahan yang berkorelasi.

Pada batasnya, heterogenitas mengalahkan numerik. Tiga model berbeda mengalahkan lima salinan dari satu model pada sebagian besar tugas yang memiliki kebenaran dasar.

### Metode juri

Kerangka kerja Sibyl (dikutip dalam literatur Minsky-LLM) memformalkan "juri" - sekelompok kecil agen khusus yang menyaring jawaban melalui pemungutan suara di setiap phase. Tidak seperti suara mayoritas biasa, juri mempunyai peran: satu agen melakukan pemeriksaan silang, satu lagi memberikan konteks, satu lagi menilai masuk akal. Metode juri merupakan titik tengah antara pemungutan suara biasa (murah, rawan monokultur) dan MAD penuh (mahal, rawan konformitas).

### Ketika pemungutan suara dengan perdebatan mendominasi

- Pertanyaannya memiliki kebenaran dasar (fakta, matematika, perilaku code). Konvergensi suara mempunyai arti.
- Agen dapat mengakses sumber atau alat yang berbeda (tersedia heterogenitas).
- Putaran dibatasi (2-3 tipikal) dan ada juri atau verifikator terpisah.
- Anggaran memungkinkan 3-5 agen. Di luar 5-7 pada topologi grafik, pajak koordinasi mendominasi.

### Saat pemungutan suara dengan perdebatan menyakitkan

- Pertanyaannya berbentuk opini. Agen berkumpul pada jawaban mana pun yang terlihat paling percaya diri, bukan yang paling benar.
- Semua agen berbagi model dasar. Monokultur membuat konsensus menjadi tidak berarti.
- Putaran tidak terbatas. Kesesuaian selalu menang.
- Tugasnya sederhana. Agen tunggal dengan konsistensi diri pada N=5 lebih murah dan akurat.

## Build

`code/main.py` mengimplementasikan:- `run_star(agents, hub, question)` — hub mensurvei setiap pekerja, mengumpulkannya.
- `run_chain(agents, question)` — penyempurnaan berurutan.
- `run_tree(root, children, question)` — hierarki dengan agregasi kedalaman-2.
- `run_graph(agents, question, rounds)` — debat menyeluruh, putaran terbatas.
- Tombol heterogenitas bernaskah: setiap agen memiliki `error_bias` yang menunjukkan kesalahan sistematisnya.
- Harness pengukuran yang menjalankan setiap topologi pada N=3, 5, 7 dan melaporkan (akurasi, total_tokens, simulasi_jam dinding).

Jalankan:

```
python3 code/main.py
```

Output yang diharapkan: tabel topologi × N → (akurasi, token, latensi). Grafik menang di N=3-5 pada tugas bergaya penelitian; bintang menang dalam tugas-tugas faktual cepat; grafik di N=7 menunjukkan pajak koordinasi (latensi meningkat lebih cepat daripada akurasi).

## Pakai

`outputs/skill-topology-picker.md` adalah keterampilan yang membaca deskripsi tugas dan merekomendasikan topologi (bintang/rantai/pohon/grafik), N (jumlah agen), profil heterogenitas (model dasar yang akan digunakan), dan batasan bulat.

## Kirim

Untuk ansambel apa pun:

- Mulailah dengan **konsistensi diri pada N=5** menggunakan satu model dasar yang kuat. Ini adalah dasar yang murah.
- Tingkatkan ke **pemungutan suara heterogen pada N=3** jika akurasi penting. Ukur deltanya.
- Hanya tingkatkan ke **topologi debat** jika tugas memiliki struktur (penelitian, multi-langkah) dan putaran terbatas dapat dilakukan.
- Selalu mencatat cluster minoritas. Ketika minoritas terus-menerus benar, kamu mendapat sinyal keberagaman.
- Patokan jam dinding dan token beserta akurasinya. "Akurasi yang lebih baik dengan biaya 10x" adalah keputusan bisnis.

## Latihan

1. Jalankan `code/main.py`. Plot kurva pajak koordinasi untuk topologi grafik: akurasi vs N, token vs N. Pada N berapakah kurva tersebut belok?
2. Menerapkan A-HMAD: tiga agen dengan bias yang berbeda. Bagaimana perbandingan baseline all-same-bias dengan A-HMAD mengenai serangan monokultur dari Lesson 14?
3. Tambahkan peran "hakim" ke topologi grafik yang tidak memilih, hanya mencetak konsensus akhir. Apakah ini mengubah perilaku konformitas yang muncul?
4. Baca makalah AgentVerse (ICLR 2024). Identifikasi perilaku muncul mana yang paling kuat ditunjukkan oleh penerapan kamu. Bisakah kamu mendapatkan perilaku sebaliknya melalui perubahan yang cepat?
5. Baca MultiAgentBench (arXiv:2503.01935) Bagian 4 (eksperimen topologi). Reproduksi hasil "penelitian-kemenangan-grafik" pada satu tugas dari kertas menggunakan tali pengaman kamu.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Konsistensi diri | "Contoh N kali, pilih" | Wang 2022. Model tunggal, N suhu>0 sample, suara mayoritas pada jalur penalaran. |
| Heterogenitas | "Model berbeda" | Kumpulan model dasar atau keluarga prompt yang berbeda. Menghancurkan monokultur. |
| GILA | "Debat multi-agen" | Istilah umum untuk agen yang bertukar kritik dalam satu putaran. Lihat Du 2023. |
| A-HMAD | "Gila Heterogen yang Memusuhi" | Varian MAD menekankan model yang berbeda + struktur permusuhan. |
| Topologi | "Siapa berbicara dengan siapa" | Bintang, rantai, pohon, grafik. Menentukan aliran informasi. |
| Pajak koordinasi | "Pengembalian yang semakin berkurang" | Di atas ~4 agen pada grafik, biaya tumbuh lebih cepat daripada kualitas. |
| Perilaku relawan | "Bantuan tak terduga" | Pola yang muncul AgentVerse: seorang agen menawarkan untuk mengambil langkah. |
| Perilaku konformitas | "Perjanjian di bawah tekanan" | Pola yang muncul AgentVerse: seorang agen sejajar dengan seorang kritikus. |
| juri | "Panel khusus kecil" | Ansambel gaya Sibyl dengan peran (penguji, konteks, pencetak gol). |

## Bacaan Lanjutan- [Wang dkk. — Konsistensi Diri Meningkatkan Penalaran Rantai Pemikiran](https://arxiv.org/abs/2203.11171) — garis dasar model tunggal
- [Du dkk. — Meningkatkan Faktualitas dan Penalaran melalui Debat Multiagen](https://arxiv.org/abs/2305.14325) — baik agen DAN putaran penting secara independen
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — tolok ukur topologi yang menampilkan grafik terbaik untuk penelitian, rantai untuk pipeline pipa
- [Haruskah kita menjadi MAD?](https://arxiv.org/abs/2311.17371) — survei strategi MAD; menemukan MAD sering kali kalah karena konsistensi diri dengan anggaran yang sama
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) — pola sukarelawan dan konformitas yang muncul
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) — implementasi benchmark referensi
