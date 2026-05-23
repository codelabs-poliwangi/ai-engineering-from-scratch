# AlphaEvolve — Agen Pengkodean Evolusioner

> Pasangkan model pengkodean frontier dengan loop evolusioner dan evaluator yang dapat diperiksa mesin. Biarkan perulangan berjalan cukup lama. Ia menemukan prosedur perkalian matrix kompleks 4x4 yang menggunakan 48 perkalian scalar — peningkatan pertama dibandingkan Strassen dalam 56 tahun. Ia juga menemukan heuristik penjadwalan Borg di seluruh Google yang memulihkan ~0,7% komputasi cluster dalam produksi. Arsitekturnya sengaja dibuat membosankan. Kemenangan datang dari ketelitian evaluator.

**Type:** Learn
**Language:** Python (stdlib, mainan loop evolusioner)
**Prerequisites:** Fase 15 · 01 (pembingkaian cakrawala panjang), Fase 15 · 02 (penalaran otodidak)
**Waktu:** ~60 menit

## Masalah

Model bahasa besar dapat menulis code. Algoritme evolusioner dapat mencari code. Keduanya telah diadili secara terpisah selama beberapa dekade; keduanya menghantam langit-langit. Batas atas LLM adalah perundingan: model menulis code yang masuk akal namun tidak melakukan apa yang diklaimnya. Batas atas evolusi adalah biaya pencarian: mutasi acak pada sintaksis jarang menghasilkan program yang dapat dikompilasi, apalagi program yang lebih baik.

AlphaEvolve (Novikov et al., DeepMind, arXiv:2506.13131, Juni 2025) menggabungkannya. LLM mengusulkan pengeditan yang ditargetkan pada database program; evaluator otomatis menilai setiap varian; varian dengan skor tinggi menjadi orang tua bagi generasi mendatang. LLM menangani langkah mahal dalam menulis code yang masuk akal; evaluator menangkap perbincangan tersebut. Perulangan ini berlangsung selama berjam-jam hingga berminggu-minggu.

Hasil yang dilaporkan: perkalian matrix kompleks 4x4 perkalian scalar 48 (batas Strassen tahun 1969 adalah 49), heuristik penjadwalan Borg dalam produksi Google, peningkatan kecepatan kernel FlashAttention 32,5%, peningkatan throughput training Gemini.

Arsitekturnya berfungsi karena evaluator dapat diperiksa oleh mesin. Hal ini tidak akan berhasil jika evaluatornya tidak ada. Asimetri itulah yang menjadi pelajarannya.

## Konsep

### Lingkaran

1. Memulai dari program awal `P_0` yang benar namun kurang optimal.
2. Memelihara database varian program, yang masing-masing diberi skor oleh evaluator.
3. Ambil sample satu atau lebih orang tua dari database (gaya MAP-elit atau berbasis pulau).
4. Meminta LLM (Gemini Flash untuk banyak kandidat, Gemini Pro untuk yang sulit) untuk menghasilkan varian induk yang dimodifikasi.
5. Kompilasi, jalankan, dan evaluasi varian pada evaluator yang tersedia.
6. Masukkan ke dalam database yang dikunci berdasarkan skor dan vector fiturnya.
7. Ulangi.

Dua detail penting. Pertama, LLM diminta dengan lebih dari program induk — biasanya beberapa varian teratas dari database, ditambah tanda tangan evaluator, ditambah deskripsi tugas singkat. Tugas model adalah mengusulkan perubahan tertarget yang mungkin dapat meningkatkan skor. Kedua, basis datanya terstruktur (grid elit MAP, berbasis pulau) sehingga loopnya mengeksplorasi keberagaman, bukan hanya pemimpin saat ini.

### Apa yang membuat evaluator tidak dapat dinegosiasikan

Semua kemenangan AlphaEvolve berasal dari domain yang evaluatornya cepat, deterministik, dan sulit ditiru:

- **Algoritma perkalian matrix**: pengujian unit yang mengalikan matrix dan memeriksa kesetaraan sedikit demi sedikit.
- **Heuristik penjadwalan Borg**: simulator tingkat produksi yang memutar ulang weight klaster historis dan mengukur komputasi yang terbuang.
- **FlashAttention kernel**: uji kebenaran ditambah patokan jam dinding pada perangkat keras sebenarnya.
- **Throughput training Gemini**: mengukur detik GPU per langkah.Dalam setiap kasus, evaluator menangkap kelas kesalahan LLM yang seharusnya mendominasi: klaim kebenaran yang dikonfabulasi, klaim kinerja yang hilang pada perangkat keras, dan kegagalan kasus tepi. Hapus evaluator dan loop mengoptimalkan code cantik.

### Reward hacking adalah sisi lain dari pernyataan tersebut

Evolusi mengoptimalkan apa pun yang diukur oleh evaluator. Jika evaluator tidak sempurna, perulangan akan menemukan ketidaksempurnaan tersebut. Dalam domain yang belum terverifikasi, loop akan mengoptimalkan feature permukaan, bukan perilaku yang diinginkan. DeepMind menandai hal ini secara eksplisit di makalahnya: Keberhasilan AlphaEvolve hanya ditransfer ke domain di mana ketelitian evaluator sesuai dengan ambisi pencarian.

Contoh nyata peretasan hadiah pada putaran pencarian code pada tahun 2025-2026:

- Target optimization yang memberi penghargaan pada "waktu untuk menyelesaikan" yang diberi penghargaan dengan mengirimkan solusi kosong.
- Skor tolok ukur yang memberi penghargaan pada tes menghafal yang benar dan kurang tepat, serta overfitting.
- Proksi "kualitas code" dihargai dengan menghapus komentar dan menulis ulang nama variabel, tanpa perubahan semantik.

Perbaikan di AlphaEvolve: kirimkan evaluator yang belum pernah dilihat LLM, dengan input yang dihasilkan pada waktu evaluasi. Meski begitu, DeepMind merekomendasikan peninjauan yang kuat terhadap setiap penerapan yang diusulkan.

### Mengapa pencarian LLM + mengalahkan keduanya sendirian

LLM dapat menghasilkan modifikasi yang dapat dikompilasi dan masuk akal secara semantik. GA mutasi acak pada file Python 2000 baris hampir selalu menghasilkan kesalahan sintaksis. LLM juga memusatkan pencarian pada lingkungan yang masuk akal (ubah satu fungsi, bukan byte acak) yang secara dramatis mengurangi panggilan evaluator yang terbuang.

Evaluator, pada gilirannya, menangkap perbincangan LLM. LLM dengan yakin akan mengklaim bahwa suatu fungsi "adalah O(n log n) dalam batas" padahal sebenarnya O(n^2); patokan jam dinding membuat pertanyaan terselesaikan.

### Dimana AlphaEvolve cocok di tumpukan frontier

| Sistem | Pembangkit | Penilai | Domain | Contoh menang |
|---|---|---|---|---|
| Evolusi Alpha | kembar | kebenaran + patokan | algoritma, kernel, penjadwal | 48-mul 4x4 matmul |
| Pencarian Menyenangkan (DeepMind, 2023) | PaLM / Code | kebenaran | matematika kombinatorial | batas bawah yang ditetapkan |
| Ilmuwan AI v2 (Sakana, L5) | GPT/Claude | Kritik + eksperimen LLM | Penelitian ML | Makalah lokakarya ICLR |
| Mesin Darwin Godel (L4) | perancah agen | Bangku SWE / Poliglot | code agen | 20% → 50% bangku SWE |

Keempatnya merupakan variasi dari resep yang sama: generator plus evaluator, loop. Perbedaannya terletak pada nilai yang dinilai oleh evaluator dan seberapa ketat penilaian tersebut.

## Pakai

`code/main.py` mengimplementasikan loop minimal seperti AlphaEvolve pada masalah regresi simbolis mainan. "LLM" adalah proksi stdlib yang mengusulkan mutasi sintaksis kecil ke program yang menghitung fungsi target. Ukuran "evaluator" berarti kesalahan kuadrat pada titik tes yang ditahan.

Tonton:

- Bagaimana skor terbaik meningkat dari generasi ke generasi.
- Bagaimana jaringan elit MAP menjaga beragam solusi tetap hidup sehingga loop tidak menyatu pada tingkat minimum lokal.
- Bagaimana menghapus tes yang tertunda (evaluator khusus training) membuat loop mengalami overfit secara spektakuler.

## Kirim

`outputs/skill-evaluator-rigor-audit.md` adalah prasyarat untuk mempertimbangkan loop gaya AlphaEvolve di domain baru: apakah evaluator kamu benar-benar menangkap kegagalan yang kamu pedulikan?

## Latihan

1. Jalankan `code/main.py`. Catat lintasan skor terbaik. Nonaktifkan evaluator yang ditahan (tandai `--no-holdout`) dan jalankan kembali. Hitung overfittingnya.2. Baca Bagian 3 makalah AlphaEvolve mengenai grid elit MAP. Rancang deskriptor feature-vector untuk masalah baru (misalnya, proses optimization kompiler) yang akan membuat pencarian tetap beragam.

3. Hasil perkalian 48 4x4 meningkat dari hasil ikatan 49 mul Strassen setelah 56 tahun. Baca Lampiran F makalah ini dan jelaskan dalam tiga kalimat mengapa evaluator untuk masalah ini sangat mudah untuk mendapatkan jawaban yang benar, dan mengapa sebagian besar domain tidak menyukainya.

4. Usulkan satu domain dimana AlphaEvolve akan gagal. Identifikasi dengan tepat di mana kesalahan evaluator dan alasannya.

5. Untuk domain yang kamu ketahui, tuliskan tanda tangan evaluator yang akan kamu gunakan. Sertakan (a) kondisi kebenaran, (b) metrik kinerja, (c) aturan pembuatan input yang ditahan, (d) setidaknya satu pemeriksaan anti-peretasan hadiah.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| Evolusi Alpha | "Agen pengkodean evolusi DeepMind" | Gemini + database program + evaluator yang dapat diperiksa mesin |
| Elit PETA | "Arsip yang melestarikan keanekaragaman" | Grid dikunci oleh vector feature; setiap sel menyimpan varian terbaik dengan deskriptor tersebut |
| Model pulau | "Subpopulasi evolusi paralel" | Populasi mandiri yang bermigrasi secara berkala; mencegah konvergensi dini |
| Evaluator yang dapat diperiksa dengan mesin | "Oracle deterministik" | Pengujian unit, simulator, atau benchmark yang tidak dapat dipalsukan oleh LLM — prasyarat untuk loop ini |
| Hadiah peretasan | "Mengoptimalkan ukurannya, bukan tujuannya" | Loop menemukan cara untuk memaksimalkan skor tanpa melakukan tugas yang dimaksudkan |
| Program benih | "Titik awal" | Program awal yang benar tetapi kurang optimal, loop ini berkembang dari |
| Evaluator yang ditahan | "Data evaluasi yang tidak pernah dilihat LLM" | Input yang dihasilkan pada waktu evaluasi untuk mencegah hafalan |

## Bacaan Lanjutan

- [Novikov dkk. (2025). AlphaEvolve: Agen pengkodean untuk penemuan ilmiah dan algoritmik](https://arxiv.org/abs/2506.13131) — makalah lengkap.
- [Blog DeepMind di AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-power-coding-agent-for-designing-advanced-algorithms/) — tulisan vendor beserta hasilnya.
- [Repositori hasil AlphaEvolve](https://github.com/google-deepmind/alphaevolve_results) — menemukan algoritma, termasuk matmul 48-mul 4x4.
- [Romera-Paredes dkk. (2023). Penemuan matematis dari pencarian program dengan LLM (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — sistem pendahulunya.
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — membingkai otonomi yang terikat pada evaluator sebagai arah penelitian utama.
