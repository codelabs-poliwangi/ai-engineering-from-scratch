# Agen Generatif dan Simulasi Muncul

> Taman dkk. 2023 (UIST '23, arXiv:2304.03442) mengisi **Smallville**, kotak pasir berisi 25 agen, dengan arsitektur tiga bagian: **aliran memori** (log bahasa alami), **refleksi** (sintesis tingkat lebih tinggi yang dihasilkan agen tentang alirannya sendiri), dan **rencana** (perilaku tingkat hari, lalu subrencana). Hasil yang penting adalah munculnya pesta Hari Valentine: satu agen yang diunggulkan dengan "ingin mengadakan pesta Hari Valentine," tanpa naskah lebih lanjut, membuat undangan yang disebarkan ke seluruh masyarakat, tanggal terkoordinasi, dan pesta pun diadakan — dari 24 agen yang memulai tanpa sepengetahuannya. Ablasi menunjukkan ketiga komponen tersebut diperlukan agar dapat dipercaya. Kegagalan yang terdokumentasi adalah kesalahan norm spasial (memasuki toko yang tutup, berbagi kamar mandi untuk satu orang). Ini adalah arsitektur referensi untuk simulasi agen dan evaluasi sosial multi-agen pada tahun 2026.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 16 · 04 (Model Primitif), Fase 16 · 13 (Memori Bersama)
**Waktu:** ~75 menit

## Masalah

Sebagian besar sistem multi-agen merupakan tim yang memiliki skrip yang ketat: rencana perencana, code pembuat code, ulasan peninjau. Itu berfungsi untuk tugas-tugas yang terdefinisi dengan baik. Hal ini tidak menangkap perilaku yang muncul dan tidak tertulis yang muncul ketika agen memiliki ingatan, prioritas, dan dunia terbuka. Penelitian, simulasi masyarakat, dan semakin banyak game AI membutuhkan jenis kedua ini.

Arsitektur Smallville adalah tolak ukurnya. Hingga Park 2023, simulasi agen terbaik adalah pengikut skrip yang dangkal; setelah itu, polanya menjadi default untuk agen generatif di dunia terbuka. Jika kamu membuat simulasi agen pada tahun 2026, kamu menggunakan tiga komponen Smallville atau secara eksplisit membenarkan alasannya.

## Konsep

### Ketiga komponen

**Aliran memori.** Log observasi, tindakan, refleksi, dan rencana yang hanya dapat ditambahkan. Setiap entri memiliki stempel waktu, jenis, deskripsi (bahasa alami), dan metadata turunan: **kekinian**, **kepentingan** (dinilai sendiri 1-10 oleh agen), dan **relevansi** (kemiripan kosinus dengan kueri saat ini).

```
[2026-02-14 09:12:03] observation: Isabella Rodriguez asked me if I like jazz
[2026-02-14 09:14:22] reflection:   I enjoy long conversations about music
[2026-02-14 10:05:00] plan:         Attend Isabella's Valentine's Day party tonight
```

Pengambilan memori menggabungkan tiga skor: `score = w_recency * e^(-decay * age) + w_importance * importance + w_relevance * cos_sim`. Entri top-k masuk ke prompt saat ini.

**Refleksi.** Secara berkala (setiap N memori atau peristiwa penting), agen menghasilkan sintesis tingkat tinggi dari memori terkini. Entri refleksi kembali ke aliran dan dapat diambil seperti memori lainnya. Beginilah cara agen membangun "pemahaman" - yang setara dengan keyakinan jangka panjang dalam arsitektur.

**Rencana.** Decomposition dari atas ke bawah. Pertama, rencana harian secara garis besar ("pergi bekerja, makan malam bersama Klaus"). Kemudian rencana tingkat jam. Kemudian rencana tingkat tindakan. Rencana dapat direvisi: ketika suatu observasi bertentangan dengan rencana, agen akan merencanakan ulang segmen yang terkena dampak.

### Mengapa ketiganya penting (ablasi)

Taman dkk. menjalankan ablasi dengan menjatuhkan setiap observasi, refleksi, dan rencana. Setiap ablasi merusak kepercayaan:

- Tanpa **observasi**, agen akan kehilangan konteks dan bertindak berdasarkan keyakinan yang sudah basi.
- Tanpa **refleksi** agen tidak dapat membentuk keyakinan tingkat tinggi; interaksi tetap dangkal.
- Tanpa **rencana**, perilaku menjadi gangguan reaktif; tujuan menghilang.

Skor kepercayaan dari penilai manusia adalah yang tertinggi pada ketiganya; menjatuhkan siapa pun menghasilkan regresi yang terukur.

### Kemunculan Hari ValentineSalah satu agen, Isabella Rodriguez, diunggulkan dengan tujuan "ingin mengadakan pesta Hari Valentine di Hobbs Cafe pada 14 Februari pukul 5 sore". 24 agen lainnya tidak menerima benih tersebut. Selama hari simulasi:

1. Rencana Isabella termasuk mengundang orang.
2. Setiap ajakan menjadi pengamatan dalam aliran ingatan tetangga.
3. Refleksi tetangga itu menimbulkan keyakinan: "Isabella sedang mengadakan pesta."
4. Rencana tetangganya mencakup "menghadiri pesta pada tanggal 14 Februari".
5. Tetangga bercerita kepada tetangga lainnya. Ajakan tersebut menyebar tanpa koordinasi pusat.
6. Pada jam 5 sore tanggal 14 Februari, beberapa agen berkumpul di Hobbs Cafe.

Ini adalah kemunculan dalam arti teknis: perilaku tingkat sistem (sebuah partai) muncul dari interaksi lokal (undangan bilateral + perencanaan individu) tanpa adanya orkestra pusat.

### Mode kegagalan yang terdokumentasi

Taman dkk. secara eksplisit mendokumentasikan:

- **Kesalahan norm spasial.** Agen masuk ke toko yang tutup. Agen mencoba menggunakan kamar mandi untuk satu orang yang sama. Agen makan di ruangan yang tidak diperuntukkan untuk makan. Model tersebut tidak menyimpulkan norm-norm sosial-fisik dari lingkungan saja.
- **Memori meluap.** Simulasi mendalam yang dijalankan menyebabkan biaya pengambilan memori meningkat. Solusi praktis: pemadatan memori berkala (meringkas-dan-memangkas) dan memecah entri-entri yang tidak terlalu penting.
- **Halusinasi refleksi.** Refleksi dapat menciptakan hubungan yang tidak ada dalam aliran memori. Mitigasi: sertakan id memori sumber dalam prompt refleksi dan verifikasi pada waktu pengambilan.

Ini adalah mode kegagalan yang relevan dengan produksi: setiap simulasi agen tahun 2026 mewarisinya.

### Aturan implementasi tiga komponen

1. **Memori hanya dapat ditambahkan.** Jangan pernah mengubah entri memori. Koreksi adalah entri baru.
2. **Skor kepentingannya murah.** Hubungi LLM untuk memberi peringkat kepentingan 1-10 pada waktu penulisan. Simpan skornya dalam cache.
3. **Pengambilan berdasarkan peringkat, bukan difilter.** Top-k berdasarkan skor gabungan; jangan gunakan filter keras (yang kehilangan konteks).
4. **Refleksi berjalan secara berkala.** Dipicu ketika jumlah pentingnya memori yang belum diproses melebihi ambang batas (misalnya, 150).
5. **Rencana dapat direvisi.** Jika observasi baru bertentangan dengan rencana, buat ulang segmen yang terkena dampak saja, bukan keseluruhan rencana.

### Agen generatif di luar Smallville

Literatur lanjutan tahun 2024-2026 memperluas arsitekturnya:

- **Simulasi sosial multi-agen untuk riset kebijakan/pasar.** Populasi mirip Smallville menyimulasikan perilaku pengguna sebagai respons terhadap feature. Lebih cepat dari pengujian A/B; keakuratannya diperdebatkan.
- **NPC AI untuk game.** RPG dengan agen Smallville menghasilkan alur cerita yang muncul, bukan misi yang dituliskan.
- **Tolok ukur evaluasi agen generatif.** Daripada akurasi tugas, metriknya menjadi kepercayaan + koherensi perilaku dalam jangka panjang.

Arsitektur adalah acuannya. Ekstensi menukar komponen (penyimpanan vector untuk memori, refleksi yang ditambah pengambilan, rencana neurosimbolis) tetapi tetap mempertahankan struktur tiga bagian.

### Mengapa hal ini penting untuk rekayasa multi-agen

Smallville adalah bukti konsep bahwa kemunculan multi-agen itu murah bila komponennya tepat. Arsitekturnya kini telah direplikasi pada model sumber terbuka (LLM yang lebih kecil kehilangan kredibilitasnya dengan baik, bukan secara tajam). Sistem produksi apa pun yang memerlukan **perilaku sosial yang muncul** menggunakan bentuk ini. Sistem apa pun yang memerlukan **eksekusi tugas yang ketat** menggunakan pola supervisor/peran/primitif dari awal fase ini.

## Build`code/main.py` mengimplementasikan tiga komponen di stdlib Python dengan kebijakan agen tertulis (bukan LLM nyata). Demo ini mereproduksi kemunculan pesta Valentine dalam bentuk mini:

- `MemoryStream` — log khusus tambahan dengan pengambilan terkini/pentingnya/relevansi.
- `reflect(stream)` — refleksi tertulis atas kenangan yang sangat penting baru-baru ini.
- `plan(agent_state)` — rencana tingkat hari dan tingkat jam berdasarkan keyakinan saat ini.
- Skenario: 5 agen. Agen 1 memulai dengan "melempar pesta pada jam 5 sore." Melalui simulasi, undangan menyebar dan agen berkumpul.

Jalankan:

```
python3 code/main.py
```

Output yang diharapkan: jejak tick-by-tick. Pada centang terakhir, setidaknya 3 dari 5 agen menunjukkan party tersebut dalam rencana mereka, dan mereka berkumpul di lokasi party. Unggulan tunggal menghasilkan kedatangan yang terkoordinasi tanpa orkestrasi apa pun.

## Pakai

`outputs/skill-simulation-designer.md` merancang simulasi agen generatif: jumlah agen, skema memori, irama refleksi, cakrawala rencana, dan metrik evaluasi.

## Kirim

Aturan untuk simulasi produksi:

- **Memori adalah databasenya.** Pilih penyimpanan sebenarnya (DB vector, Postgres) dalam skala besar. Stdlib dalam memori adalah untuk prototipe.
- **Catat jejak pengambilan.** Untuk setiap tindakan, catat kenangan teratas yang mendorongnya. Ini adalah kemampuan debug kamu.
- **Anggaran token per agen.** Pengambilan + refleksi + paket setiap agen per tick adalah O(k) panggilan LLM. N agen × T tick × panggilan per tick dapat memperkecil anggaran kamu.
- **Memadatkan memori secara berkala.** Ringkas dan pangkas entri yang tidak terlalu penting. Kebijakan retensi adalah keputusan desain, bukan detail.
- **Mendeteksi pelanggaran norm spasial/sosial** secara eksplisit. Arsitektur tidak mempelajarinya.

## Latihan

1. Jalankan `code/main.py`. Konfirmasikan 3+ agen berkumpul di pesta. Tingkatkan agen menjadi 10 — apakah kemunculannya masih terjadi?
2. Hapus langkah refleksi. Seperti apa perilakunya? Peta temuan ablasi di Taman 2023.
3. Perkenalkan tujuan unggulan yang bersaing ("Klaus ingin memberikan ceramah penelitian pada jam 5 sore"). Apakah agen terpecah, atau apakah satu tujuan mendominasi? Apa yang menentukannya?
4. Tambahkan batasan spasial: Hobbs Cafe menampung paling banyak 4 agen. Apakah pegangan simulasi meluap dengan baik, atau apakah ia mencapai pola kegagalan "kamar mandi satu orang"?
5. Baca Park dkk. (arXiv:2304.03442) Bagian 6 (eksperimen perilaku yang muncul). Identifikasi satu perilaku yang tidak dapat direproduksi dalam miniatur kamu. Komponen arsitektur apa yang perlu kamu tingkatkan?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Aliran memori | "Buku harian agen" | Tambahkan log observasi, tindakan, refleksi, rencana saja. |
| Kekinian | "Betapa baru ingatannya" | Skor peluruhan eksponensial berdasarkan usia. |
| Pentingnya | "Seberapa peduli agen" | Nilai mandiri 1-10 pada waktu penulisan. Di-cache. |
| Relevansi | "Seberapa terkait dengan kueri saat ini" | Kesamaan kosinus (berbasis embedding). |
| Refleksi | "Keyakinan tingkat tinggi" | Sintesis yang dihasilkan dari ingatan terkini, dicerna kembali sebagai ingatan baru. |
| Rencana | "Decomposition hari/jam/tindakan" | Pohon rencana top-down. Dapat direvisi ketika pengamatannya bertentangan. |
| Kota Kecil | "Kotak pasir Park 2023" | Simulasi 25 agen yang menghasilkan kemunculan Hari Valentine. |
| Kepercayaan | "Metrik kualitas" | Skor penilai manusia untuk mengetahui apakah perilaku tampak seperti agen yang masuk akal. |

## Bacaan Lanjutan- [Taman dkk. — Agen Generatif: Simulacra Interaktif Perilaku Manusia](https://arxiv.org/abs/2304.03442) — arsitektur referensi
- [Halaman kertas UIST '23](https://dl.acm.org/doi/10.1145/3586183.3606763) — tempat publikasi
- [Rilis code Smallville](https://github.com/joonspk-research/generative_agents) — referensi implementasi Python
- [Hayes-Roth 1985 — Arsitektur Papan Tulis untuk Kontrol](https://www.sciencedirect.com/science/article/abs/pii/0004370285900639) — penemuan sebelumnya untuk agen memori terstruktur
