# Teori Pikiran dan Koordinasi Muncul

> Li dkk. (arXiv:2310.10701) menunjukkan bahwa agen LLM dalam permainan teks kooperatif menunjukkan **Teori Pikiran tingkat tinggi yang muncul** (ToM) — memikirkan apa yang diyakini agen lain tentang keyakinan agen ketiga — tetapi gagal dalam perencanaan jangka panjang karena manajemen konteks dan halusinasi. Riedl (arXiv:2510.05174) mengukur sinergi tingkat tinggi di seluruh populasi dan menemukan bahwa **hanya** kondisi cepat ToM menghasilkan diferensiasi terkait identitas dan saling melengkapi yang diarahkan pada tujuan; LLM berkapasitas lebih rendah hanya menunjukkan kemunculan palsu. Artinya, kemunculan koordinasi bersifat cepat-bersyarat dan bergantung pada model, tidak bebas. Lesson ini mengimplementasikan agen minimal yang sadar ToM, menjalankan tugas kooperatif dengan dan tanpa prompt ToM, dan mengukur delta koordinasi terhadap protokol Riedl 2025.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 16 · 07 (Masyarakat Pikiran dan Debat), Fase 16 · 17 (Agen Generatif)
**Waktu:** ~75 menit

## Masalah

Koordinasi multi-agen seringkali terlihat ajaib: agen membagi pekerjaan, saling mengantisipasi, dan menghindari redundansi. Biasanya "kemunculan" ini merupakan artefak rekayasa cepat - seseorang menyuruh agen untuk "berkoordinasi". Hapus prompt, hapus koordinasi.

Temuan Riedl pada tahun 2025 lebih ketat: dalam kondisi terkendali, koordinasi hanya muncul ketika agen diminta untuk mempertimbangkan **pikiran agen lain** (ToM). Tanpa prompt ToM, bahkan model yang kuat pun akan menunjukkan pola koordinasi yang tidak dapat bertahan dalam kendali statistik. Hal ini penting untuk produksi: tim mengirimkan feature "koordinasi multi-agen" yang bergantung pada kecepatan dan rapuh.

Lesson ini memperlakukan ToM sebagai kemampuan spesifik (bernalar tentang keyakinan tentang keyakinan), membangun agen minimal yang sadar akan ToM, dan mengukur seperti apa koordinasi sebenarnya vs.

## Konsep

### Apa yang dimaksud dengan ToM

Psikologi perkembangan: anak usia 3 tahun menganggap dunia batin seseorang cocok dengan dunia batinnya. Seorang anak berusia 5 tahun memahami bahwa orang lain memiliki keyakinan yang berbeda. Seorang anak berusia 7 tahun beralasan tentang keyakinan tentang keyakinan ("dia berpikir bahwa saya pikir bolanya ada di bawah cangkir"). Ini adalah ToM orde nol, pertama, dan kedua.

Untuk agen LLM, pesanan ToM dipetakan ke:

- **Zeroth-order:** tidak ada model lain. Agen hanya bertindak berdasarkan pengamatannya sendiri.
- **Urutan pertama:** agen memiliki model keyakinan masing-masing agen. "Alice percaya X."
- **Urutan kedua:** agen memodelkan keyakinan rekursif. "Alice percaya bahwa Bob percaya X."

Li dkk. 2023 menemukan bahwa ToM tingkat pertama dan kedua muncul di agen LLM dalam permainan kooperatif tetapi menurun dengan jangka waktu yang panjang dan komunikasi yang tidak dapat diandalkan.

### Tes Sally-Anne, secara singkat

Tes kepercayaan salah tahun 1985: Sally menaruh kelereng di keranjang A, lalu pergi. Anne memindahkannya ke keranjang B. Di mana Sally akan mencari ketika dia kembali? Seorang anak dengan ToM orde pertama mengatakan keranjang A (Keyakinan Sally berbeda dengan kenyataan). Seorang anak tanpa berkata keranjang B.

LLM era GPT-4 lulus tes gaya Sally-Anne jika diajukan dengan jelas. Mereka gagal jika narasinya panjang, adegannya berubah beberapa kali, atau pertanyaannya diutarakan secara tidak langsung. Itulah keadaan praktis ToM tahun 2026 dalam LLM produksi.

### Pengukuran koordinasi Riedl

Riedl (arXiv:2510.05174) membuat tes skala populasi: N agen, tujuan kooperatif, kondisi cepat variabel. Ukuran:1. **Diferensiasi terkait identitas.** Apakah agen mengembangkan perbedaan peran yang stabil dari waktu ke waktu?
2. **Komplementaritas yang diarahkan pada tujuan.** Apakah tindakan agen saling melengkapi (subtugas yang berbeda) dan bukannya duplikat?
3. **Sinergi tingkat tinggi.** Ukuran statistik apakah kelompok mencapai apa yang tidak dapat dicapai oleh subkelompok mana pun.

Hasil: hanya pada kondisi prompt ToM ketiga metrik menghasilkan sinyal di atas garis dasar. Tanpa dorongan ToM, metrik hampir mendekati peluang untuk model berkapasitas sedang. Model besar menunjukkan beberapa koordinasi tanpa dorongan ToM yang eksplisit namun efeknya lebih kecil dibandingkan dengan dorongan eksplisit.

### Ilusi koordinasi

Tanpa kendali statistik, “koordinasi yang muncul” dalam demo sering kali mencerminkan:

- Rekayasa cepat yang menghasilkan koordinasi (system prompt yang mengatakan "bekerja sama").
- Bias pengamat (kita melihat pola yang kita harapkan).
- Pemilihan proses sukses pasca-hoc.

Sistem produksi yang memasarkan “koordinasi yang muncul” tanpa sinyal yang terukur harus diperlakukan sebagai pemasaran. Ukur sebelum mengklaim.

### Agen minimal yang sadar ToM

Struktur:

```
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]

observation update:
  - update own_beliefs from direct observation
  - update other_models[agent_id] from their action + prior beliefs

action selection:
  - enumerate candidate actions
  - for each, predict what each other agent will do next given their modeled beliefs
  - pick action that maximizes joint outcome under those predictions
```

Atribut `other_models` adalah status ToM. ToM orde pertama hanya mempertahankan satu level. Pesanan kedua menambahkan `other_models[i][other_models_of_j]` — apa yang menurut saya agen saya yakini oleh agen j.

### Mengapa cakrawala panjang menyakitkan

Li dkk. dokumen: batasan konteks menyebabkan agen lupa keyakinan mana yang menjadi milik siapa. Halusinasi menambah keyakinan salah pada model agen lain. Keduanya menghasilkan kesalahan "Saya pikir dia pikir X" yang bertambah seiring waktu.

Mitigasi yang didokumentasikan dalam makalah dan tindak lanjutnya pada tahun 2024-2026:

- **Status ToM eksplisit di prompt.** Format terstruktur: `{agent_id: belief_list}`. Memaksa pengambilan untuk menjaga pengikatan identitas-kepercayaan.
- **Rantai penalaran yang lebih pendek.** Pembaruan ToM yang lebih sedikit per giliran mengurangi halusinasi yang bertambah.
- **Penyimpanan ToM eksternal.** Mempertahankan model di luar konteks LLM; hanya menyuntikkan bagian yang relevan per putaran.

### Saat ToM gagal dalam produksi

- **Pengaturan permusuhan.** Agen dengan ToM yang baik lebih mudah untuk dimanipulasi (kamu dapat memodelkan apa yang mereka modelkan terhadap kamu, lalu mengeksploitasinya).
- **Tim yang heterogen.** Jika modelnya berbeda, model ToM yang berfungsi untuk satu lawan tidak akan digeneralisasikan.
- **Tugas yang bergantung pada kebenaran dasar.** ToM adalah tentang keyakinan; jika kebenaran bergantung pada fakta, ToM bisa menjadi pengalih attention.

### Koordinasi sebenarnya dapat kamu ukur

Tiga sinyal praktis bahwa koordinasi tim adalah nyata dan tidak dilakukan secara langsung:

1. **Komplementaritas dari waktu ke waktu.** Pada tugas multi-giliran, apakah tindakan agen mencakup subtugas yang terpisah?
2. **Antisipasi.** Apakah tindakan agen A di tikungan T+1 bergantung pada prediksi tentang tindakan B di T+2 yang ternyata benar?
3. **Koreksi.** Saat A salah membaca keyakinan B di giliran T, apakah A mengoreksi di giliran T+2?

Ini dapat diukur dalam sistem multi-agen yang dicatat. Ini adalah versi substantif dari narasi “koordinasi”.

## Build

`code/main.py` mengimplementasikan:

- `ToMAgent` — melacak keyakinan sendiri dan model keyakinan per agen lain.
- Tugas kooperatif: tiga agen harus mengumpulkan tiga token dari tiga kotak; setiap kotak dapat menampung satu token. Agen tidak dapat berkomunikasi; mereka menyimpulkan niat dari tindakan masing-masing.
- Dua konfigurasi: `zeroth_order` (tanpa ToM) dan `first_order` (ToM dengan model keyakinan satu tingkat).
- Pengukuran lebih dari 200 uji coba acak: tingkat penyelesaian, tingkat duplikasi (dua agen menargetkan kotak yang sama), rata-rata penyelesaian.

Jalankan:

```
python3 code/main.py
```Hasil yang diharapkan: agen tingkat nol menduplikasi upaya pada tingkat ~35% dan menyelesaikan ~60% uji coba dalam 10 putaran. Agen ToM tingkat pertama menggandakan ~5% dan menyelesaikan ~95%. Delta adalah efek koordinasi yang terukur.

## Pakai

`outputs/skill-tom-auditor.md` adalah keterampilan yang mengaudit klaim sistem multi-agen tentang "koordinasi darurat". Memeriksa cara berpakaian yang cepat, signifikansi statistik terhadap kontrol, dan saling melengkapi yang terukur.

## Kirim

Daftar periksa klaim koordinasi:

- **Kondisi kontrol.** Versi sistem kamu tanpa prompt koordinasi. Ukur keduanya.
- **Uji statistik.** Apakah perbedaan antara sistem dan kontrol signifikan di `p < 0.05` pada metrik kamu?
- **Ukuran saling melengkapi.** Keterputusan tindakan seiring berjalannya waktu, bukan hanya kesuksesan akhir.
- **Log kasus kegagalan.** Jika agen salah berkoordinasi, seperti apa status ToM?
- **Pengungkapan kapasitas model.** Jika efeknya hilang pada model yang lebih kecil, katakan demikian.

## Latihan

1. Jalankan `code/main.py`. Konfirmasikan ToM pesanan pertama mengurangi tingkat duplikasi sebesar ~7x. Apakah kesenjangan tetap ada saat kamu menskalakan ke 5 agen dan 5 kotak?
2. Menerapkan ToM orde kedua (agen A memodelkan apa yang dipikirkan B tentang C). Apakah ini lebih baik dari urutan pertama? Pada tugas apa?
3. Masukkan **halusinasi** ke dalam kondisi ToM: balikkan satu keyakinan secara acak per giliran. Seberapa besar hal ini menurunkan kinerja tingkat pertama?
4. Baca Li dkk. (arXiv:2310.10701). Reproduksi temuan "degradasi jangka panjang": seiring bertambahnya jumlah belokan dari 10 menjadi 30, bagaimana performa ToM orde pertama kamu berubah?
5. Baca Riedl 2025 (arXiv:2510.05174). Terapkan statistik sinergi tingkat tinggi pada log simulasi kamu. Apakah efeknya muncul tanpa kondisi prompt ToM?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Teori Pikiran | "Memahami pikiran orang lain" | Kapasitas untuk mencontoh keyakinan agen lain. Dinilai berdasarkan pesanan (0, 1, 2+). |
| Tes Sally-Anne | "Tes kepercayaan yang salah" | psikologi perkembangan 1985; LLM lulus versi biasa, gagal dalam versi kompleks. |
| ToM orde pertama | "A percaya X" | Mencontohkan keyakinan orang lain tentang fakta. |
| ToM orde kedua | "A percaya B percaya X" | Pemodelan rekursif satu tingkat lebih dalam. |
| Diferensiasi terkait identitas | "Peran stabil dari waktu ke waktu" | Metrik Riedl: peran tetap ada, tidak acak. |
| Saling melengkapi yang diarahkan pada tujuan | "Tindakan terputus-putus" | Agen menargetkan subtugas yang berbeda, bukan subtugas yang sama. |
| Sinergi tingkat tinggi | "Grup melebihi subset mana pun" | Ukuran statistik Riedl untuk koordinasi nyata. |
| Ilusi koordinasi | "Kelihatannya terkoordinasi" | Penampilan koordinasi yang cepat tanpa sinyal yang terukur. |

## Bacaan Lanjutan- [Li dkk. — Teori Pikiran untuk Kolaborasi Multi-Agen melalui Large Language Model](https://arxiv.org/abs/2310.10701) — ToM yang muncul dalam permainan kooperatif; mode kegagalan cakrawala panjang
- [Riedl — Koordinasi yang Muncul dalam Model Bahasa Multi-Agen](https://arxiv.org/abs/2510.05174) — pengukuran skala populasi; Prompt ToM adalah kondisi menahan weight
- [Premack & Woodruff — Apakah simpanse mempunyai teori pikiran?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) — asal mula tahun 1978 konsep ToM
- [Baron-Cohen, Leslie, Frith — Apakah anak autis memiliki teori pikiran?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/) — makalah Sally-Anne (1985)
