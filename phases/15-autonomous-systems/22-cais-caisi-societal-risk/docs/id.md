# CAIS, CAISI, dan Risiko Skala Sosial

> Center for AI Safety (CAIS, San Francisco, didirikan pada tahun 2022 oleh Hendrycks dan Zhang) menerbitkan framework empat risiko — penggunaan berbahaya, ras AI, risiko organisasi, AI nakal — dan pernyataan Mei 2023 tentang risiko kepunahan yang ditandatangani oleh ratusan profesor dan pemimpin perusahaan. Rilis tahun 2026 dari CAIS: Dasbor AI untuk evaluasi model perbatasan, Indeks Tenaga Kerja Distance Jauh (dengan AI Skala), Makalah Strategi Superintelligence, buletin AI Frontiers. Entitas yang berbeda: NIST Center for AI Standards and Innovation (CAISI) — perjanjian sukarela yang dihadapi pemerintah AS dan evaluasi kemampuan yang tidak terklasifikasi yang berfokus pada risiko siber, bio, dan senjata kimia. CAIS menandai risiko organisasi sebagai salah satu dari empat risiko tingkat atas: budaya keselamatan, audit yang ketat, pertahanan berlapis, dan keamanan informasi merupakan hal mendasar namun sering kali diimbangi dengan kecepatan penerapan. California SB-53, jika ditandatangani, akan menjadi peraturan risiko bencana tingkat negara bagian AS yang pertama.

**Type:** Learn
**Language:** Python (stdlib, inventaris empat risiko, dan pencocokan mitigasi)
**Prerequisites:** Phase 15 · 19 (RSP), Phase 15 · 20 (PF + FSF)
**Waktu:** ~45 menit

## Masalah

Lesson 19 dan 20 membahas kebijakan penskalaan internal laboratorium. Lesson 21 mencakup evaluasi kemampuan mandiri. Lesson ini mencakup perspektif ketiga: organisasi masyarakat sipil dan pemerintah yang membentuk diskusi publik dan dasar peraturan untuk risiko bencana AI.

Dua entitas berbeda penting. CAIS adalah organisasi penelitian nirlaba yang menerbitkan kerangka pemikiran tentang risiko AI dan mengoordinasikan pernyataan publik. CAISI adalah pusat pemerintahan AS di NIST yang menjalankan perjanjian sukarela dengan laboratorium dan evaluasi kemampuan yang tidak rahasia. Nama-nama itu berima; misinya tidak tumpang tindih. Seorang praktisi harus mengetahui keduanya.

Isi praktisnya: Kerangka kerja empat risiko CAIS adalah taksonomi risiko skala sosial yang paling banyak dikutip dalam literatur. Budaya keselamatan dan risiko organisasi adalah salah satu dari empat hal tersebut, dan merupakan hal yang paling langsung berada di bawah kendali praktisi. SB-53 (California) akan menjadi peraturan risiko bencana tingkat negara bagian AS yang pertama jika ditandatangani; penyusunan undang-undang tersebut penting karena peraturan tingkat negara bagian secara historis mengarahkan tindakan federal dalam kebijakan teknologi AS.

## Konsep

### CAIS — Pusat Keamanan AI

- Didirikan: tahun 2022 di San Francisco, oleh Dan Hendrycks dan rekannya (nama "Zhang" mengacu pada kolaborator awal, bukan salah satu pendiri saat ini; lihat situs web CAIS untuk mengetahui kepemimpinan saat ini).
- Status: 501(c)(3) nirlaba.
- Hasil penting tahun 2023: pernyataan tentang risiko kepunahan, ditandatangani bersama oleh ratusan peneliti dan CEO. Menyatakan: "Memitigasi risiko kepunahan akibat AI harus menjadi prioritas global bersama dengan risiko skala sosial lainnya seperti pandemi dan perang nuklir."
- Output tahun 2026: Dasbor AI untuk evaluasi model frontier, Indeks Tenaga Kerja Distance Jauh (bersama dengan Scale AI), Makalah Strategi Superintelligence, buletin AI Frontiers.

### Kerangka kerja empat risiko

Kerangka kerja CAIS mengelompokkan risiko bencana AI ke dalam empat kategori tingkat atas:1. **Penggunaan berbahaya**: pelaku kejahatan menggunakan AI untuk menimbulkan loss (sintesis senjata biologis, disinformasi, serangan siber).
2. **Perlombaan AI**: tekanan persaingan antar laboratorium, perusahaan, atau negara mendorong penerapan AI melampaui titik aman.
3. **Risiko organisasi**: dinamika laboratorium internal (kegagalan budaya keselamatan, audit yang tidak memadai, keamanan yang kekurangan sumber daya) menghasilkan penerapan yang buruk.
4. **Rogue AI**: AI yang cukup mampu mengejar tujuan yang bertentangan dengan kesejahteraan manusia.

Ini bukan satu-satunya taksonomi; itu yang paling banyak dikutip. Kategori-kategori tersebut tidak eksklusif – AI jahat yang diproduksi oleh organisasi yang memperdagangkan audit untuk kecepatan dalam perlombaan adalah keempat kategori tersebut.

### Di mana risiko organisasi berada

Dari empat kategori, risiko organisasi adalah yang paling dapat ditindaklanjuti oleh para praktisi. Budaya keselamatan laboratorium, ketelitian audit, layer pertahanan, dan keamanan informasi menentukan apakah model mereka dikirimkan dengan kontrol dari Lesson 10–18 benar-benar diterapkan, atau apakah kontrol tersebut merupakan item daftar periksa yang tidak diverifikasi oleh siapa pun.

Pengungkit risiko organisasi yang konkrit:

- **Budaya keselamatan**: apakah anggota tim merasa mampu menyampaikan kekhawatiran tanpa mengorbankan karier? Survei CAIS menemukan bahwa hal ini merupakan prediktor kuat terhadap pengaruh lainnya.
- **Audit yang ketat**: eksternal dan internal. Audit internal saja menghasilkan laporan yang optimis.
- **Pertahanan berlapis**: tidak ada satu layer pun yang cukup (tema yang sedang berjalan pada Fase 15).
- **Keamanan informasi**: kebocoran weight model, kebocoran data evaluasi, kebocoran teknik bypass monitor. RAND SL-4 dalam Lesson 19 adalah standar khusus.

### CAISI — Pusat Standar dan Inovasi AI

- Beroperasi dalam NIST.
- Menjalankan perjanjian sukarela dengan laboratorium perbatasan.
- Menerbitkan evaluasi kemampuan yang tidak terklasifikasi yang berfokus pada risiko siber, bio, dan senjata kimia.
- Berbeda dari CAIS; akronimnya bertabrakan; periksa URL (nist.gov) untuk mengonfirmasi mana yang sedang kamu baca.

Peran CAISI adalah sebagai mitra publik dan pemerintah dalam keterlibatan laboratorium swasta METR (Lesson 21). Laporan CAISI tidak diklasifikasikan; Laporan METR seringkali menggunakan NDA-gated. Seorang praktisi yang membaca keduanya mendapatkan gambaran yang lebih lengkap.

### Kalifornia SB-53

RUU Senat Kalifornia (sesi 2025–2026) membahas risiko bencana dari model perbatasan. Ketentuan utama sebagaimana dirancang:

- Ambang batas kemampuan spesifik yang memicu kewajiban tingkat negara bagian.
- Perlindungan pelapor bagi karyawan lab AI.
- Persyaratan pelaporan insiden untuk kegagalan bencana.

Jika ditandatangani, peraturan ini akan menjadi peraturan berisiko bencana tingkat negara bagian AS yang pertama. Terlepas dari status penandatanganannya, kerangka rancangan undang-undang tersebut menentukan cara badan legislatif negara bagian lainnya menyikapi masalah ini. Praktisi di California harus melacak status RUU tersebut; praktisi di tempat lain harus membacanya untuk memahami seperti apa peraturan di tingkat negara bagian AS nantinya.

### Risiko skala sosial bukanlah masalah yang hanya terjadi satu lapis

Tema Fase 15 – pertahanan mendalam – juga berlaku pada layer masyarakat. Tidak ada satu organisasi, peraturan, atau framework yang mampu menutup risiko bencana. Ekosistem hanya berfungsi jika:

- Kebijakan penskalaan kapal laboratorium (Lesson 19, 20).
- Evaluator eksternal melakukan pengukuran (Lesson 21).
- Masyarakat sipil melacak dan mempublikasikan (CAIS).
- Pemerintah menjalankan program sukarela dan peraturan dasar (CAISI, SB-53).
- Praktisi membangun kontrol berlapis-lapis (Lesson 10–18).

Ini adalah sintesis terakhir untuk fase ini: setiap lesson sebelumnya adalah satu layer dalam tumpukan yang kelengkapannya lebih penting daripada kekuatan satu layer mana pun.## Pakai

`code/main.py` mengimplementasikan alat inventaris risiko kecil. Mengingat penerapan yang diusulkan, ini menandai penerapan tersebut berdasarkan empat kategori risiko dan mengembalikan daftar periksa mitigasi. Ini adalah alat bantu membaca framework, bukan pengganti penilaian manusia.

## Kirim

`outputs/skill-societal-risk-review.md` meninjau penerapan postur risiko skala sosial: yang mana dari empat kategori yang dicakupnya, mitigasi apa yang ada, apa saja paparan risiko organisasi.

## Latihan

1. Jalankan `code/main.py`. Umpan dalam tiga penerapan sintetis pada skala berbeda. Konfirmasikan empat tag risiko sesuai dengan yang kamu harapkan; mengidentifikasi satu kasus ketika alat diberi tag di bawah atau di atas.

2. Baca makalah empat risiko CAIS secara lengkap. Pilih satu kategori risiko dan tulis dua paragraf tentang apa yang kamu yakini sebagai perkembangan paling penting pada tahun 2026 dalam kategori tersebut.

3. Bacalah draf California SB-53 terkini. Identifikasi satu ketentuan yang kamu yakini memperkuat postur risiko bencana dan satu ketentuan yang kamu yakini melemahkannya. Benarkan keduanya.

4. Pilih penerapan AI produksi yang kamu ketahui (milik kamu atau yang sudah dipublikasikan). Nilailah berdasarkan sub-pengungkit risiko organisasi: budaya keselamatan, ketelitian audit, pertahanan berlapis, keamanan informasi. Manakah yang paling lemah? Berapa biaya untuk menjadikannya setara?

5. Buat sketsa framework empat risiko versi 2028 yang mencerminkan satu tahun kemampuan tambahan dan satu tahun pengalaman penerapan tambahan. Apa yang akan kamu tambahkan, hapus, atau kelompokkan kembali?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| CAIS | "Pusat Keamanan AI" | Nirlaba; kerangka empat risiko; Pernyataan kepunahan tahun 2023 |
| CAISI | "Keamanan AI pemerintah AS" | Pusat NIST; perjanjian sukarela; evaluasi tidak terklasifikasi |
| Kerangka kerja empat risiko | "Taksonomi CAIS" | penggunaan jahat, ras AI, risiko organisasi, AI jahat |
| Penggunaan berbahaya | "Aktor jahat menggunakan AI" | Senjata biologis, disinformasi, serangan siber |
| balapan AI | "Tekanan kompetitif" | Lab/perusahaan/negara mendorong penerapan melampaui keselamatan |
| Risiko organisasi | "Kegagalan internal laboratorium" | Budaya keselamatan, audit, pertahanan, infosec |
| AI nakal | "Agen yang tidak selaras" | AI yang mampu mengejar tujuan yang bertentangan dengan kesejahteraan manusia |
| Kalifornia SB-53 | "Peraturan tingkat negara bagian" | RUU 2025–2026; peraturan risiko bencana negara bagian AS yang pertama jika ditandatangani |

## Bacaan Lanjutan

- [Center for AI Safety](https://safe.ai/) — institusi yang menjadi pusat framework empat risiko.
- [CAIS — Risiko AI yang Dapat Menyebabkan Bencana](https://safe.ai/ai-risk) — makalah empat risiko.
- [CAIS — pernyataan Mei 2023 tentang risiko kepunahan](https://safe.ai/statement-on-ai-risk) — pernyataan singkat bersama.
- [NIST CAISI](https://www.nist.gov/caisi) — pusat inovasi dan standar AI yang dikelola pemerintah.
- [Anthropic — Mengukur otonomi agen dalam praktiknya](https://www.anthropic.com/research/measuring-agent-autonomy) — menghubungkan komitmen tingkat laboratorium dengan framing skala masyarakat.
