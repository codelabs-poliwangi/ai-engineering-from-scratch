# Asal Data dan Training-Tata Kelola Data

> EU AI Act mewajibkan standar opt-out yang dapat dibaca mesin untuk GPAI pada bulan Agustus 2025 (melalui pengecualian TDM Petunjuk Hak Cipta UE). California AB 2013 (ditandatangani pada tahun 2024) — Transparansi training data AI generatif mengharuskan pengembang untuk mempublikasikan ringkasan dataset dengan 12 bidang yang diamanatkan. Penyelarasan DPA 2025 berdasarkan kepentingan yang sah: DPC Irlandia (21 Mei 2025) menerima training LLM Meta tentang konten dewasa UE/EEA publik pihak pertama dengan perlindungan setelah opini EDPB; Pengadilan Tinggi Regional Cologne (23 Mei 2025) menolak prompt tersebut; DPA Hamburg menghilangkan urgensinya; ICO Inggris (23 September 2025) mengeluarkan tanggapan peraturan yang positif terhadap perlindungan training AI LinkedIn (transparansi, penyertaan yang disederhanakan, jangka waktu keberatan yang diperpanjang) dan terus melakukan pemantauan — bukan izin formal. ANPD Brasil (2 Juli 2024) menangguhkan pemrosesan Meta karena kurangnya transparansi informasi; tindakan pencegahan tersebut dicabut pada 30 Agustus 2024 setelah Meta mengajukan rencana kepatuhan. Masalah utama yang tidak dapat diubah: kerangka persetujuan cookie dirancang untuk pelacakan waktu nyata dan dapat dibalik; setelah data berada dalam weight model, penghapusan secara bedah tidak mungkin dilakukan — tidak ada hak praktis untuk menghapus GDPR untuk neural network terlatih. Jendela kepatuhan ada pada waktu pengumpulan. Data Provenance Initiative (dataprovenance.org, Longpre, Mahari, Lee dkk., "Consent in Crisis", Juli 2024): audit skala besar menunjukkan penurunan pesat pada data AI bersama ketika penerbit menambahkan pembatasan robots.txt.

**Type:** Learn
**Language:** Python (stdlib, generator perancah California AB 2013 12-bidang)
**Prerequisites:** Phase 18 · 24 (peraturan), Phase 18 · 26 (kartu)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Jelaskan 12 bidang wajib California AB 2013 untuk transparansi training data AI Generatif.
- Nyatakan posisi DPA 2025 tentang training LLM dengan kepentingan sah (DPC Irlandia, ICO Inggris, Hamburg, Cologne).
- Jelaskan masalah yang tidak dapat diubah: mengapa hak penghapusan GDPR tidak memiliki persamaan praktis untuk neural network terlatih.
- Nyatakan temuan "Persetujuan dalam Krisis" dari Data Provenance Initiative.

## Masalah

Tata kelola training data merupakan bagian hulu dari setiap kartu model (Lesson 26) dan kewajiban regulasi (Lesson 24). Pada tahun 2024-2025, lanskap peraturan dikonsolidasikan berdasarkan tiga prinsip: infrastruktur opt-out, pengungkapan per dataset, dan akomodasi kepentingan sah untuk data yang tersedia untuk publik. Penyedia yang tidak mematuhi pada waktu pengumpulan tidak dapat melakukan remediasi di hilir.

## Konsep

### Kalifornia AB 2013Ditandatangani pada tahun 2024. Dokumentasi harus diposting pada atau sebelum tanggal 1 Januari 2026 untuk sistem yang dirilis pada atau setelah tanggal 1 Januari 2022. Pasal 3111(a) mewajibkan pengembang untuk mempublikasikan ringkasan tingkat tinggi dari dataset yang digunakan dalam training dengan 12 item undang-undang:
1. Sumber atau pemilik dataset.
2. Deskripsi tentang bagaimana dataset dapat mencapai tujuan yang diharapkan dari sistem AI.
3. Jumlah titik data dalam dataset (rentang umum dapat diterima; perkiraan untuk dataset dinamis).
4. Deskripsi tipe titik data (tipe label untuk dataset berlabel; karakteristik umum untuk dataset tidak berlabel).
5. Apakah dataset mencakup data apa pun yang dilindungi oleh hak cipta, merek dagang, atau paten, atau seluruhnya berada dalam domain publik.
6. Apakah dataset tersebut dibeli atau dilisensikan.
7. Apakah dataset menyertakan informasi pribadi (sesuai Cal. Civ. Code §1798.140(v)).
8. Apakah dataset menyertakan informasi konsumen agregat (sesuai Cal. Civ. Code §1798.140(b)).
9. Pembersihan, pengolahan, atau modifikasi lainnya oleh pengembang, dengan tujuan yang dimaksudkan.
10. Jangka waktu pengumpulan data, dengan pemberitahuan jika pengumpulan sedang berlangsung.
11. Tanggal dataset pertama kali digunakan selama pengembangan.
12. Apakah sistem menggunakan atau terus-menerus menggunakan pembuatan data sintetis.

Item 12 (data sintetis) relatif baru bagi Gebru dkk. lembar data 2018. Butir 7 (informasi pribadi) memicu kewajiban Privacy Rights Act (CPRA). Undang-undang tersebut mengecualikan sistem keamanan/integritas, pengoperasian pesawat terbang, dan sistem keamanan nasional khusus federal (Pasal 3111(b)).

### EU AI Act (Lesson 24) dan penolakan TDM

Pengecualian penambangan teks dan data Petunjuk Hak Cipta UE mengizinkan training tentang konten yang tersedia untuk umum kecuali pemegang hak memilih untuk tidak ikut serta. UU AI UE Code Praktik GPAI Bab hak cipta mewajibkan penyedia GPAI untuk menghormati sinyal penyisihan yang dapat dibaca mesin (robots.txt, klaim C2PA "Tanpa Training AI", dll.).

### Konvergensi DPA 2025 atas kepentingan yang sah

DPC Irlandia (21 Mei 2025): Rencana Meta untuk melatih konten pengguna dewasa UE/EEA publik pihak pertama yang diterima dengan perlindungan setelah opini EDPB. Pengadilan Tinggi Regional Cologne (23 Mei 2025) menolak prompt terhadap Meta: tidak ikut serta saja sudah cukup. DPA Hamburg membatalkan prosedur urgensi demi konsistensi di seluruh UE. ICO Inggris (23 September 2025) mengeluarkan tanggapan peraturan yang positif — bukan izin formal — terhadap dimulainya kembali training AI di LinkedIn dengan perlindungan serupa dan pemantauan berkelanjutan.

Prinsip konvergen: kepentingan yang sah dapat membenarkan training tentang konten pihak pertama yang tersedia untuk umum dengan pilihan untuk tidak ikut serta. Persetujuan tidak diperlukan.

### ANPD Brasil (Juni 2024)

Menangguhkan pemrosesan data pengguna Brasil oleh Meta untuk training AI karena transparansi informasi yang tidak memadai. Hasil yang berbeda dibandingkan DPA UE – ANPD memprioritaskan transparansi dibandingkan penerimaan kepentingan yang sah.

### Masalah yang tidak dapat diubah

Persetujuan cookie dirancang untuk pelacakan real-time dan dapat dibalik. Training data berbeda: setelah data memasuki weight model, penghapusan secara bedah tidak dapat dilakukan. Training ulang dari awal adalah satu-satunya perbaikan yang menyeluruh, dan biayanya sangat mahal.

Remediasi parsial:
- **Tidak dipelajari.** Perkiraan penghapusan; diukur dengan MIA (Lesson 22).
- **Mempengaruhi lokalisasi berbasis fungsi.** Identifikasi weight yang paling dipengaruhi oleh data; memperbarui secara selektif.
- **Sempurnakan penekanannya.** Latih model untuk menolak output yang berasal dari data.

Tidak ada yang bisa menyelesaikan masalah sepenuhnya. Jendela kepatuhan berada pada waktu pengumpulan.

### Inisiatif Asal Datadataprovenance.org. Longpre, Mahari, Lee dkk. "Persetujuan dalam Krisis" (Juli 2024): audit skala besar terhadap training data AI bersama. Temuan: penerbit menambahkan pembatasan robots.txt dengan kecepatan yang semakin cepat. Kelompok masyarakat yang dapat dilatih secara terbuka sedang berkontraksi dengan cepat. Pada tahun 2023 -> 2024 terdapat sekitar 25% sumber training teratas yang menambahkan beberapa batasan. Implikasi: ketersediaan training data di masa depan bergantung pada paradigma akuisisi baru (perizinan, pembuatan sintetis, partisipasi berinsentif).

### Cocok untuk Fase 18

Lesson 26 adalah dokumentasi tingkat model. Lesson 27 adalah tata kelola tingkat dataset. Bersama-sama mereka mendefinisikan layer transparansi. Lesson 28 memetakan ekosistem penelitian yang menangani pertanyaan-pertanyaan ini.

## Pakai

`code/main.py` menghasilkan perancah ringkasan dataset 12 bidang yang sesuai dengan California AB 2013 untuk dataset mainan. kamu dapat mengisi kolom dan mengamati mana yang memicu kewajiban tindak lanjut privasi atau hak cipta.

## Kirim

Lesson ini menghasilkan `outputs/skill-provenance-check.md`. Mengingat dataset yang digunakan dalam training, training ini memeriksa cakupan 12 bidang AB 2013, kepatuhan infrastruktur opt-out, penyelarasan DPA, dan penilaian risiko yang tidak dapat diubah.

## Latihan

1. Jalankan `code/main.py`. Menghasilkan ringkasan 12 bidang untuk dataset mainan dan mengidentifikasi bidang mana yang kurang ditentukan.

2. Penyisihan TDM Petunjuk Hak Cipta Uni Eropa dapat dibaca oleh mesin. Usulkan format standar untuk sinyal opt-out dan bandingkan dengan robots.txt dan C2PA "Tanpa Training AI".

3. Baca "Persetujuan dalam Krisis" dari Data Provenance Initiative (Juli 2024). Jelaskan tiga kategori konten yang paling cepat dibatasi dan jelaskan salah satu konsekuensi ekonominya.

4. Penyelarasan DPA 2025 menerima kepentingan sah untuk training konten publik. Buatlah skenario di mana kepentingan yang sah saja tidak cukup dan identifikasikan dasar hukum yang dibutuhkan oleh penyedia layanan.

5. Buat sketsa manifes asal training data yang disusun dengan bidang AB 2013 dan rantai asal yang ditandatangani C2PA untuk setiap dataset. Identifikasi satu hambatan teknis dan satu hambatan hukum.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| AB 2013 | "hukum California" | Transparansi training data AI generatif; 12 bidang yang diamanatkan |
| Pengecualian TDM | "penambangan teks dan data" | Pengecualian training data Petunjuk Hak Cipta UE dengan pilihan untuk tidak ikut serta |
| Kepentingan yang sah | "basis UE" | Dasar Pasal 6 GDPR yang mungkin membenarkan training tentang konten publik |
| Sinyal penyisihan | "tanpa kereta yang dapat dibaca mesin" | robots.txt, C2PA "Tanpa Training AI," TDM.Reservasi |
| ireversibilitas | "tidak dapat membatalkan training" | Data dalam model weight tidak dapat dilepas melalui pembedahan |
| Berhenti belajar | "perkiraan penghapusan" | Intervensi pasca training untuk mengurangi ketergantungan model pada data spesifik |
| Persetujuan dalam Krisis | "audit DPI" | Temuan Juli 2024 tentang percepatan pembatasan robots.txt |

## Bacaan Lanjutan- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013) — Undang-undang transparansi training data AI generatif
- [UU AI UE + Code Praktik GPAI (Lesson 24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — Bab hak cipta
- [Longpre, Mahari, Lee dkk. — Persetujuan dalam Krisis (dataprovenance.org, Juli 2024)](https://www.dataprovenance.org/consent-in-crisis-paper) — Audit DPI
- [IAPP — Amandemen GDPR Omnibus Digital UE (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) — konteks peraturan
