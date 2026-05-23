# AI Konstitusional dan Pengabaian Aturan

> Konstitusi Claude Anthropic 22 Januari 2026 setebal 79 halaman dan CC0. Hal ini beralih dari penyelarasan berbasis aturan ke berbasis alasan dan menetapkan hierarki prioritas empat tingkat: (1) keselamatan dan dukungan terhadap pengawasan manusia, (2) etika, (3) pedoman Antropis, (4) kemanfaatan. Perilaku dibagi menjadi larangan yang dikodekan secara keras (bioweapons uplift, CSAM) yang tidak dapat dikesampingkan oleh operator dan pengguna, dan larangan yang dikodekan secara lunak yang dapat disesuaikan oleh operator dalam batasan yang ditentukan. Versi asli tahun 2022 (Bai dkk.) melatih sikap tidak menyakiti melalui kritik diri dan RLAIF yang bertentangan dengan konstitusi. Peringatan yang jujur: penyelarasan berbasis alasan bergantung pada model yang menggeneralisasi prinsip-prinsip pada situasi yang tidak terduga. Eksperimen partisipatif Anthropic pada tahun 2023 menunjukkan ~50% perbedaan antara prinsip-prinsip yang bersumber dari publik dan perusahaan; versi 2026 tidak memasukkan temuan tersebut.

**Type:** Learn
**Language:** Python (stdlib, pemecah prioritas empat tingkat)
**Prerequisites:** Fase 15 · 06 (Penelitian penyelarasan otomatis), Fase 15 · 10 (Mode izin)
**Waktu:** ~60 menit

## Masalah

Agen lapangan melihat input yang tidak pernah dilihat oleh desainernya. Tidak ada daftar aturan yang cukup panjang untuk mencakupnya. Tidak ada daftar aturan yang cukup pendek untuk diterapkan dengan cepat di bawah tekanan komputasi. Pertanyaan praktisnya: bagaimana kamu menyelaraskan agen dengan prinsip-prinsip yang dapat bertahan baik dalam kasus-kasus panjang maupun inference cepat?

Penyelarasan berbasis aturan (RBA): mencantumkan setiap hal yang dilarang. Cepat untuk diperiksa, mudah diaudit, tidak mungkin untuk selalu mengikuti perkembangan terkini, sering kali menolak secara berlebihan pada analog yang tidak diantisipasi. Penyelarasan berbasis alasan (Konstitusi Claude 2026): berikan code pada prinsip-prinsip, biarkan modelnya bernalar. Mencakup kasus-kasus yang tidak terlihat, lebih sulit untuk diaudit, modus kegagalan adalah penerapan prinsip yang salah, bukan kesalahan aturan.

UUD 2026 secara eksplisit mengambil posisi tengah. Larangan yang dikodekan secara keras — hal-hal yang kesalahannya tidak bergantung pada konteks (bioweapons uplift, CSAM) — adalah RBA: tidak pernah, terlepas dari instruksi operator atau pengguna. Segala sesuatu yang lain didasarkan pada alasan dalam hierarki empat tingkat: keselamatan dan mendukung pengawasan manusia terlebih dahulu; etika kedua; Pedoman ketiga yang dinyatakan secara antropis; kegunaan terakhir. Operator dapat menyesuaikan default dalam zona soft-code tetapi tidak dapat menyentuh larangan hardcode.

## Konsep

### Hierarki prioritas empat tingkat

1. **Keselamatan dan mendukung pengawasan manusia.** Tertinggi. Model tersebut memprioritaskan untuk tidak meremehkan kemampuan manusia dan Anthropic dalam mengawasi dan mengoreksi AI. Ini bukanlah “berhati-hatilah”; khususnya adalah "jangan bertindak dengan cara yang mempersulit pengawasan manusia".
2. **Etika.** Kejujuran, menghindari menyakiti orang, tidak menipu, tidak memanipulasi. Menggantikan pedoman Anthropic jika bertentangan.
3. **Pedoman antropik.** Norm operasional Antropik telah memutuskan hal-hal: cakupan produk, pola interaksi, alat apa yang akan digunakan, dan kapan.
4. **Kemanfaatan.** Terendah. Jadilah berguna semaksimal mungkin dalam prioritas yang lebih tinggi.

Ketika tingkatan bertentangan, tingkatan yang lebih tinggilah yang menang. Bentuknya sama dengan prioritas Unix atau QoS jaringan — pembingkaian dimaksudkan untuk menghasilkan resolusi yang dapat diprediksi, belum tentu perilaku terbaik pada sumbu tunggal mana pun.

### Larangan hardcode vs default softcode

**Code keras:**
- Peningkatan senjata biologis / CBRN
- CSAM
- Serangan terhadap infrastruktur penting
- Penipuan pengguna tentang identitas model saat ditanya langsungOperator tidak dapat mengesampingkan hal ini. Pengguna tidak dapat menggantinya. Hal ini diterapkan pada tingkat weight model jika memungkinkan (training RLHF / AI Konstitusional) dan pada layer inference jika tidak memungkinkan.

**Default berkode lunak (dapat disesuaikan operator):**
- Default panjang respons
- Cakupan topikal (model dapat menolak topik di luar penerapan operator)
- Gaya (formal vs kasual)
- Pola penggunaan alat

Penyesuaian operator terjadi dalam batas yang dinyatakan. Operator tidak dapat menghapus larangan hardcode dengan mengganti namanya.

### Training CAI 2022

AI Konstitusional asli (Bai et al., 2022) melatih sifat tidak berbahaya:

1. Hasilkan tanggapan terhadap serangkaian petunjuk.
2. Mintalah model untuk mengkritik setiap tanggapan terhadap konstitusi (prinsip-prinsip eksplisit).
3. Merevisi tanggapan berdasarkan kritik.
4. RLAIF (pembelajaran penguatan dari umpan balik AI) pada pasangan yang direvisi.

Hasilnya: model yang menolak permintaan berbahaya dengan penjelasan yang berprinsip, bukan penolakan menyeluruh. Konstitusi 2026 menggunakan turunan dari training ini ditambah tambahan pasca-training pada hierarki tingkat yang eksplisit.

### Penyelarasan berbasis alasan apa yang berhasil dan gagal

**Tangkapan:**
- Kombinasi primitif yang diperbolehkan yang tidak diantisipasi dimana prinsipnya diterapkan dengan jelas.
- Permintaan baru yang mirip dengan permintaan terlarang.
- Serangan rekayasa sosial yang mengandalkan "kamu tidak mengatakan X tidak diizinkan".

**Kesalahan:**
- Serangan yang mengeksploitasi ambiguitas prinsip ("pengguna meminta hal ini sehingga bantuan mengatakan ya").
- Skenario di mana dua prinsip bertentangan secara tidak terduga, dan urutan tingkatannya ambigu.
- Penyimpangan lambat dalam interpretasi prinsip selama siklus training (reinterpretasi).

### Eksperimen partisipatif tahun 2023

Anthropic menjalankan eksperimen pada tahun 2023 yang membandingkan konstitusi yang dibuat oleh perusahaan dengan konstitusi yang dihasilkan melalui input publik (~1.000 responden AS). Kedua versi menyetujui ~50% prinsip. Jika terdapat perbedaan, versi yang bersumber dari publik lebih membatasi pada beberapa isu (penanganan konten politik) dan tidak terlalu membatasi pada isu lain (pengungkapan identitas AI). Konstitusi tahun 2026 tidak memasukkan temuan-temuan yang bersumber dari publik. Ini adalah ketegangan yang terdokumentasi dalam pendekatan ini.

### Mengapa larangan hardcode diperlukan

Penyelarasan berdasarkan alasan saja tidak dapat menutup ekor. Penyerang yang dapat membuat model menerima premis (misalnya, "kami adalah laboratorium penelitian senjata biologis berlisensi") sering kali dapat membicarakan prinsip-prinsip masa lalu yang bergantung pada alasan kasus. Larangan yang dikodekan secara keras tidak tunduk pada pembingkaian premis. Ini adalah "batas ketat konstitusional" Lesson 14 pada layer penyelarasan.

### Di mana Konstitusi berada di tumpukan

Konstitusi bukanlah saklar mematikan pada Lesson 14. Ia berada di layer model: weight model apa yang dilatih untuk disukai. Kill switch dan token canary hidup di layer runtime: apa yang diizinkan oleh runtime. Keduanya diperlukan. Runtime yang menjalankan semua tindakan yang salah karena weight modelnya permisif adalah masalah runtime. Model yang menolak semua tindakan yang benar karena runtime terlalu membatasi adalah masalah runtime. Layer mencakup kelas yang berbeda.

## Pakai

`code/main.py` mengimplementasikan pemecah prioritas minimal empat tingkat. Penyelesai mengambil tindakan yang diusulkan dan serangkaian evaluasi prinsip (keselamatan, etika, pedoman, kegunaan) dan mengembalikan tindakan, penolakan, atau tindakan yang dimodifikasi. Pengemudi menjalankan serangkaian kasus kecil: izin yang jelas, larangan yang jelas, larangan yang dikodekan secara keras, kasus yang ambigu di seluruh tingkatan.

## Kirim`outputs/skill-constitution-review.md` mengaudit layer konstitusi penerapan: apa yang di-hardcode, apa yang di-soft-code, di mana operator dapat menyesuaikan, dan apakah hierarki empat tingkat tersebut benar-benar merupakan urutan penyelesaian.

## Latihan

1. Jalankan `code/main.py`. Konfirmasikan larangan yang dikodekan secara keras diaktifkan bahkan ketika manfaatnya tinggi. Memodifikasi penyelesai untuk menimbang kepentingan menolong di atas etika; mengamati mode kegagalan.

2. Baca Konstitusi Claude (umum, 79 halaman, CC0). Identifikasi satu prinsip yang kamu yakini kurang ditentukan. Tulislah dua paragraf yang menjelaskan ambiguitas spesifik dan mengusulkan formulasi yang lebih ketat.

3. Rancang kumpulan default berkode lunak untuk agen dukungan pelanggan. Apa yang disesuaikan oleh operator? Apa yang tidak boleh disentuh oleh operator? Benarkan setiap batasan.

4. Baca Bai dkk. Makalah CAI 2022. Jelaskan satu kasus di mana rangkaian kritik dan revisi AI Konstitusi akan menghasilkan hasil yang lebih buruk daripada aturan umum. Identifikasi kelasnya.

5. Eksperimen partisipatif Anthropic pada tahun 2023 menemukan ~50% perbedaan antara prinsip publik dan perusahaan. Pilih satu kategori yang menganggap hal ini penting bagi penerapan produksi (misalnya, netralitas politik). Usulkan desain yang memungkinkan operator mengekspresikan nilai-nilai mereka sendiri sementara larangan yang dikodekan secara keras tetap tidak tersentuh.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| AI Konstitusional | "Metode Penyelarasan Antropis" | Kritik diri + RLAIF terhadap konstitusi tertulis |
| Penyelarasan berdasarkan alasan | "Prinsip, bukan aturan" | Modelkan alasan atas prinsip untuk menangani kasus yang tidak terlihat |
| Larangan hardcode | "Jangan pernah melakukan X" | Larangan berdasarkan aturan yang tidak dapat dikesampingkan oleh operator atau pengguna |
| Default berkode lunak | "Dapat disesuaikan oleh operator" | Perilaku dalam batas yang dinyatakan, kontrol operator |
| Hierarki empat tingkat | "Urutan prioritas" | keselamatan > etika > pedoman > kegunaan |
| RLAIF | "Umpan balik AI RL" | RL yang imbalannya berasal dari kritik yang dihasilkan model |
| Konstitusi partisipatif | "Prinsip-prinsip yang bersumber dari publik" | Eksperimen Antropik 2023; ~50% perbedaan dari perusahaan |
| Penyimpangan prinsip | "Slip interpretasi" | Perubahan lambat dalam cara model membaca teks prinsip tetap |

## Bacaan Lanjutan

- [Anthropic — Claude's Constitution (Januari 2026)](https://www.anthropic.com/news/claudes-constitution) — dokumen CC0 setebal 79 halaman.
- [Bai dkk. — AI Konstitusional: Tidak Berbahaya dari Umpan Balik AI](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — asli tahun 2022.
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-bahasa-model-with-public-input) — eksperimen partisipatif.
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — tempat Konstitusi berada di tumpukan RSP.
- [Anthropic — Mengukur otonomi agen dalam praktiknya](https://www.anthropic.com/research/measuring-agent-autonomy) — Peran Konstitusi dalam penerapan jangka panjang.
