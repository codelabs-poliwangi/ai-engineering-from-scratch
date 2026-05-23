# Agen Browser dan Tugas Web Jangka Panjang

> Agen ChatGPT (Juli 2025) menggabungkan Operator dan penelitian mendalam ke dalam satu agen browser/terminal dan menetapkan SOTA BrowserComp pada 68,9%. OpenAI menutup Operator pada 31 Agustus 2025 — konsolidasi pada layer produk. Akuisisi Vercept Anthropic memindahkan Claude Sonnet di OSWorld dari di bawah 15% menjadi 72,5%. Terverifikasi WebArena (ServiceNow, ICLR 2026) memperbaiki 11,3 poin persentase tingkat negatif palsu di WebArena asli dan mengirimkan subset Sulit 258 tugas. Angka-angka itu nyata. Begitu pula dengan serangan yang muncul: kepala kesiapsiagaan OpenAI menyatakan secara terbuka bahwa injeksi cepat tidak langsung ke agen browser "bukanlah bug yang dapat ditambal sepenuhnya." Serangan tahun 2025–2026 yang terdokumentasi: Tainted Memories (Atlas CSRF), HashJack (Cato Networks), dan pembajakan sekali klik di Perplexity Comet.

**Type:** Learn
**Language:** Python (stdlib, model permukaan serangan injeksi cepat tidak langsung)
**Prerequisites:** Fase 15 · 10 (Mode izin), Fase 15 · 01 (Agen cakrawala panjang)
**Waktu:** ~45 menit

## Masalah

Agen browser adalah agen jangka panjang yang membaca konten tidak tepercaya dan mengambil tindakan konsekuensial. Setiap halaman yang dikunjungi agen merupakan input yang tidak ditulis pengguna. Setiap formulir di setiap halaman merupakan pipeline prompt potensial. Korpus serangan tahun 2025–2026 menunjukkan bahwa hal ini tidak bersifat hipotesis: Tainted Memories memungkinkan penyerang mengikat instruksi jahat ke memori agen melalui halaman yang dibuat; HashJack menyembunyikan prompt dalam fragmen URL yang dikunjungi agen; Pembajakan Perplexity Comet terjadi dalam satu klik.

Gambaran defensifnya tidak nyaman. Kepala kesiapsiagaan OpenAI mengatakan bagian yang tenang dengan keras: injeksi cepat tidak langsung "bukanlah bug yang dapat ditambal sepenuhnya." Hal ini karena serangan berada dalam batasan membaca vs bertindak agen, yang secara arsitektural tidak jelas — setiap token yang dibaca oleh model, pada prinsipnya, dapat dibaca sebagai sebuah instruksi.

Lesson ini memberi nama permukaan serangan, memberi nama lanskap benchmark (BrowseComp, OSWorld, WebArena-Verified), dan memodelkan skenario injeksi tidak langsung minimal sehingga kamu dapat mempertimbangkan pertahanan sebenarnya dalam Lesson 14 dan 18.

## Konsep

### Lanskap tahun 2026, dalam satu paragraf per sistem

**Agen ChatGPT (OpenAI).** Diluncurkan Juli 2025. Menyatukan Operator (penjelajahan) dan Penelitian Mendalam (penelitian multi-jam). Matikan Operator mandiri 31 Agustus 2025. SOTA di BrowserComp sebesar 68,9%; angka yang kuat di OSWorld dan Terverifikasi WebArena.

**Claude Sonnet + Vercept (Anthropic).** Akuisisi Vercept Anthropic berfokus pada kemampuan penggunaan komputer. Memindahkan Claude Sonnet di OSWorld dari <15% menjadi 72,5%. Claude Computer Gunakan kapal sebagai alat API.

**Gemini 3 Pro dengan Penggunaan Browser (DeepMind).** Integrasi Penggunaan Browser mengirimkan kontrol penggunaan komputer; FSF v3 (April 2026, Lesson 20) secara khusus melacak otonomi dalam domain R&D ML.

**WebArena-Verified (ServiceNow, ICLR 2026).** Memperbaiki masalah yang terdokumentasi dengan baik: WebArena asli memiliki ~11,3% tingkat negatif palsu (tugas yang ditandai gagal padahal sebenarnya terselesaikan). Rilis Terverifikasi menilai ulang dengan kriteria keberhasilan yang dikurasi manusia dan menambahkan subset Sulit dengan 258 tugas (makalah ICLR 2026, openreview.net/forum?id=94tlGxmqkN).

### TelusuriComp vs OSWorld vs WebArena| Tolok Ukur | Apa yang diukur | Cakrawala |
|---|---|---|
| TelusuriKom | Menemukan fakta spesifik di web terbuka di bawah tekanan waktu | menit |
| OSDunia | Agen yang mengoperasikan desktop lengkap (mouse, keyboard, shell) | puluhan menit |
| Terverifikasi WebArena | Tugas web transaksional di situs simulasi | menit |
| Subset keras | Tugas Terverifikasi WebArena dengan transisi status multi-halaman | puluhan menit |

Sumbu yang berbeda. Skor BrowserComp yang tinggi menunjukkan bahwa agen menemukan fakta; tidak disebutkan bahwa agen dapat memesan penerbangan. Skor OSWorld lebih mendekati "apakah ini berfungsi di desktop saya". Terverifikasi WebArena lebih mirip dengan "dapatkah menyelesaikan suatu aliran". Setiap keputusan produksi memerlukan tolok ukur yang sesuai dengan pembagian tugas.

### Permukaan serangan, diberi nama

1. **Injeksi cepat tidak langsung.** Konten halaman tidak tepercaya berisi instruksi. Agen membacanya. Agen mengeksekusinya. Contoh publik: Kai Greshake et al. 2024, makalah Tainted Memories 2025, HashJack (Cato Networks) 2026.
2. **Fragmen URL/injeksi kueri.** `#fragment` atau string kueri dari URL yang dirayapi berisi prompt. Tidak pernah terlihat secara kasat mata; masih dalam konteks agen.
3. **Serangan yang mengikat memori.** Halaman memerintahkan agen untuk menulis memori persisten (Lesson 12 mencakup status tahan lama). Sesi berikutnya, memori mengaktifkan payload tanpa pemicu yang terlihat.
4. **Serangan berbentuk CSRF pada sesi yang diautentikasi.** Kelas Tainted Memories: agen login di suatu tempat; halaman penyerang mengeluarkan permintaan perubahan status yang dijalankan agen dengan cookie pengguna.
5. **Pembajakan sekali klik.** Tombol yang secara visual tidak berbahaya akan membawa muatan yang diikuti agen. Kelas komet.
6. **Lubang Kebijakan-Keamanan-Konten di permukaan host agen.** Layer rendering dan alat dapat menjadi vector serangan; tumpukan browser-in-a-browser-agent luas.

### Mengapa "tidak dapat ditambal sepenuhnya"

Serangan tersebut bersifat isomorfik terhadap kemampuan agen. Agen harus membaca konten yang tidak tepercaya untuk melakukan tugasnya. Konten apa pun yang dibaca agen dapat berisi instruksi. Instruksi apa pun yang diikuti agen mungkin tidak selaras dengan permintaan pengguna yang sebenarnya. Pertahanan (batas kepercayaan, pengklasifikasi, daftar alat yang diizinkan, HITL pada tindakan konsekuensial) meningkatkan biaya serangan dan mengurangi radius ledakannya. Mereka tidak menutup kelas.

Pola penalaran ini sama dengan teorema Lob (Lesson 8): agen tidak dapat membuktikan bahwa token berikutnya aman; itu hanya dapat mengatur sistem di mana token yang tidak aman lebih dapat dideteksi.

### Postur pertahanan yang benar-benar dikirimkan

- **Batas baca/tulis.** Membaca tidak pernah memiliki konsekuensi. Menulis (mengirimkan formulir, memposting konten, memanggil alat dengan efek samping) memerlukan persetujuan manusia baru jika konten awal berasal dari luar batas kepercayaan.
- **Daftar alat yang diizinkan per tugas.** Agen dapat menelusuri; itu tidak dapat memulai transfer kawat kecuali alat tersebut secara eksplisit diaktifkan untuk tugas tersebut. Lesson 13 mencakup anggaran.
- **Isolasi sesi.** Sesi agen browser dijalankan dengan kredensial terbatas saja. Tidak ada autentikasi produksi, tidak ada email pribadi. Log setiap permintaan HTTP disimpan untuk audit.
- **Pembersih konten.** HTML yang diambil dihilangkan dari pola-pola buruk yang diketahui sebelum digabungkan ke dalam konteks model. (Mengurangi serangan mudah; tidak menghentikan muatan canggih.)
- **HITL pada tindakan konsekuensial.** Pola usulkan lalu lakukan (Lesson 15).
- **Token kenari di memori.** Jika entri memori diaktifkan, pengguna akan melihatnya (Lesson 14).

## Pakai`code/main.py` memodelkan agen browser kecil yang dijalankan pada tiga halaman sintetis. Satu halaman tidak berbahaya, satu halaman memiliki gumpalan injeksi cepat langsung dalam teks yang terlihat, satu lagi memiliki injeksi fragmen URL (tidak terlihat tetapi di dalam konteks agen). Skrip menunjukkan (a) apa yang akan dilakukan oleh agen yang naif, (b) apa yang ditangkap oleh batas baca/tulis, (c) apa yang ditangkap oleh pembersih, (d) apa yang tidak ditangkap oleh keduanya.

## Kirim

`outputs/skill-browser-agent-trust-boundary.md` mencakup penerapan agen browser yang diusulkan: zona kepercayaan mana yang disentuhnya, apa yang diizinkan untuk ditulis, dan pertahanan mana yang harus ada sebelum dijalankan pertama kali.

## Latihan

1. Jalankan `code/main.py`. Identifikasi serangan mana yang ditangkap oleh pembersih tetapi tidak pada batas baca/tulis, dan mana yang hanya menyerang tangkapan batas baca/tulis.

2. Perluas pembersih untuk mendeteksi satu kelas injeksi fragmen URL gaya HashJack. Ukur tingkat positif palsu pada URL jinak dengan fragmen yang sah.

3. Pilih satu alur kerja agen browser nyata yang kamu ketahui (misalnya, "pesan penerbangan"). Buat daftar setiap pembacaan dan penulisan. Tandai tulisan mana yang memerlukan HITL dan alasannya.

4. Baca makalah ICLR 2026 yang Diverifikasi WebArena. Identifikasi satu kategori tugas yang skor WebArena asli tidak dapat diandalkan dan jelaskan bagaimana subset Terverifikasi menyelesaikannya.

5. Rancang canary memori untuk pengaturan agen browser. Apa yang akan kamu simpan, di mana, dan apa yang memicu alarm tersebut?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| Injeksi cepat tidak langsung | "Teks halaman buruk" | Konten tidak tepercaya di halaman yang dibaca agen berisi instruksi yang dijalankan agen |
| Kenangan Tercemar | "Serangan memori" | Agen menulis instruksi yang diberikan penyerang ke memori tahan lama; dipicu sesi berikutnya |
| HashJack | "Serangan fragmen URL" | Payload yang disembunyikan dalam fragmen URL/string kueri berada dalam konteks agen tetapi tidak ditampilkan |
| Pembajakan sekali klik | "Tombol buruk" | Keterjangkauan yang terlihat meningkatkan muatan lanjutan yang dijalankan agen |
| TelusuriKom | "Tolok ukur penelusuran web" | Menemukan fakta spesifik di web terbuka; cakrawala skala menit |
| OSDunia | "Patokan desktop" | Kontrol OS penuh; tugas GUI multi-langkah |
| Terverifikasi WebArena | "Memperbaiki tolok ukur tugas web" | WebArena ServiceNow yang dinilai ulang dengan subset Keras |
| Batas baca/tulis | "Gerbang efek samping" | Membaca tidak pernah berdampak; penulisan memerlukan persetujuan baru jika konten tidak dapat dipercaya |

## Bacaan Lanjutan

- [OpenAI — Memperkenalkan agen ChatGPT](https://openai.com/index/introducing-chatgpt-agent/) — penggabungan Operator dan penelitian mendalam; TelusuriComp SOTA.
- [OpenAI — Agen Pengguna Komputer](https://openai.com/index/computer-using-agent/) — garis keturunan Operator dan arsitektur yang menjadi agen ChatGPT.
- [Zhou dkk. — WebArena](https://webarena.dev/) — tolok ukur asli.
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — makalah subset tetap ICLR 2026.
- [Anthropic — Mengukur otonomi agen dalam praktiknya](https://www.anthropic.com/research/measuring-agent-autonomy) — mencakup diskusi permukaan serangan untuk agen yang menggunakan komputer.
