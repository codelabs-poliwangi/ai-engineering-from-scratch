# Membangun Klien MCP — Penemuan, Doa, Manajemen Sesi

> Sebagian besar konten MCP mengirimkan tutorial server dan melambaikan tangan ke klien. Code klien adalah tempat hidup orkestrasi yang sulit: pemijahan proses, negosiasi kemampuan, penggabungan daftar alat di beberapa server, pengambilan sample callback, koneksi ulang, dan resolusi tabrakan namespace. Lesson ini membangun klien multi-server yang mengangkat tiga server MCP berbeda ke dalam satu namespace alat datar untuk model tersebut.

**Type:** Build
**Language:** Python (stdlib, klien MCP multi-server)
**Prerequisites:** Phase 13 · 07 (membangun server MCP)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Memunculkan server MCP sebagai proses anak, selesaikan `initialize`, dan kirim `notifications/initialized`.
- Pertahankan status sesi per server (kemampuan, daftar alat, id notifikasi yang terakhir dilihat).
- Gabungkan daftar alat di beberapa server menjadi satu namespace dengan penanganan tabrakan.
- Rutekan panggilan alat ke server pemiliknya dan susun kembali responsnya.

## Masalah

Host agen nyata (Claude Desktop, Cursor, Goose, Gemini CLI) memuat beberapa server MCP sekaligus. Seorang pengguna mungkin memiliki server sistem file, server Postgres, dan server GitHub yang berjalan secara bersamaan. Pekerjaan klien:

1. Spawn tiap server.
2. Jabat tangan masing-masing secara mandiri.
3. Hubungi `tools/list` pada masing-masing dan ratakan hasilnya.
4. Saat model mengeluarkan `notes_search`, cari model tersebut di namespace gabungan dan rutekan ke server yang tepat.
5. Tangani notifikasi dari server mana pun (`tools/list_changed`) tanpa memblokir.
6. Sambungkan kembali jika terjadi kegagalan pengangkutan.

Menggulung semua itu dengan tangan itulah yang membedakan "mainan" dari "bisa diservis". SDK resmi membungkus hal ini, tetapi model mentalnya harus menjadi milik kamu.

## Konsep

### Proses pemijahan anak

`subprocess.Popen` dengan `stdin=PIPE, stdout=PIPE, stderr=PIPE`. Setel `bufsize=1` dan gunakan mode teks untuk membaca baris demi baris. Setiap server adalah satu proses; klien memegang satu pegangan `Popen` per server.

### Status sesi per server

Objek `Session` per server menampung:

- `process` — pegangan Popen.
- `capabilities` — apa yang dinyatakan server di `initialize`.
- `tools` — hasil `tools/list` terakhir.
- `pending` — peta id permintaan ke janji/masa depan menunggu tanggapan.

Permintaan pada dasarnya tidak sinkron; a `tools/call` yang dikirim ke server A saat server B sedang dalam panggilan tidak boleh diblokir. Gunakan utas dengan antrian atau asyncio.

### Ruang nama digabungkan

Saat klien melihat daftar alat agregat, nama dapat bertabrakan. Dua server mungkin mengekspos `search`. Klien memiliki tiga opsi:

1. **Awalan berdasarkan nama server.** `notes/search`, `files/search`. Jelas tapi jelek.
2. **Diam, siapa yang datang lebih dulu.** `search` di server selanjutnya akan menggantikan server sebelumnya. Berisiko; menyembunyikan tabrakan.
3. **Penolakan tabrakan.** Menolak memuat server kedua; memberitahukan pengguna. Paling aman untuk host yang sensitif terhadap keamanan.

Claude Desktop menggunakan awalan demi server. Kursor menggunakan penolakan tabrakan dengan kesalahan yang jelas. VS Code MCP juga mengadopsi awalan demi server.

### Perutean

Setelah penggabungan, tabel pengiriman memetakan `tool_name -> session`. Model mengeluarkan panggilan berdasarkan nama; klien menemukan sesi tersebut dan menulis pesan `tools/call` ke stdin server tersebut, lalu menunggu responsnya.

### Pengambilan sample panggilan balikJika server mendeklarasikan kemampuan `sampling` di `initialize`, server dapat mengirimkan `sampling/createMessage` meminta klien untuk menjalankan LLM-nya. Klien harus:

1. Blokir permintaan lebih lanjut ke server tersebut hingga sample terselesaikan, atau alurkan jika implementasinya mendukung konkurensi.
2. Hubungi penyedia LLM-nya.
3. Kirim respon kembali ke server.

Lesson 11 mencakup pengambilan sample secara menyeluruh. Lesson ini menghentikannya untuk kelengkapan.

### Penanganan notifikasi

`notifications/tools/list_changed` berarti panggilan ulang `tools/list`. `notifications/resources/updated` berarti membaca ulang sumber daya jika sedang digunakan. Pemberitahuan tidak boleh menghasilkan tanggapan — jangan mencoba untuk menerimanya.

Bug klien yang umum: memblokir loop baca di `tools/call` saat notifikasi berada di aliran. Gunakan thread pembaca latar belakang yang memasukkan setiap pesan ke dalam antrian; thread utama dequeue dan pengiriman.

### Sambungan ulang

Transportasi bisa gagal: server mogok, OS menghentikan proses, pipa stdio rusak. Klien mendeteksi EOF di stdout dan menganggap sesi tersebut mati. Pilihan:

- Restart server secara diam-diam dan jabat tangan kembali. OK untuk server read-only murni.
- Tunjukkan kegagalan kepada pengguna. OK untuk server stateful dengan sesi yang terlihat pengguna.

Phase 13 · 09 meliputi semantik koneksi ulang HTTP yang dapat di-streaming; stdio lebih sederhana.

### Keepalive dan id sesi

HTTP yang dapat dialirkan menggunakan header `Mcp-Session-Id`. Stdio tidak memiliki id sesi - identitas proses ADALAH sesinya. Ping yang tetap hidup adalah opsional; pipa stdio tidak pecah jika tidak ada aktivitas.

## Pakai

`code/main.py` memunculkan tiga server MCP yang disimulasikan sebagai subproses, masing-masing melakukan jabat tangan, menggabungkan daftar alatnya, dan merutekan panggilan alat ke server yang benar. "Server" sebenarnya adalah proses Python lain yang menjalankan penanggap mainan (bukan LLM nyata). Jalankan untuk melihat:

- Tiga inisialisasi, masing-masing dengan rangkaian kemampuannya sendiri.
- Tiga hasil `tools/list` digabungkan menjadi namespace 7 alat.
- Keputusan perutean berdasarkan nama alat.
- Tabrakan dicegah dengan awalan namespace.

Apa yang harus dilihat:

- Kelas data `Session` menjaga status per server dengan bersih.
- Utas pembaca latar belakang menghapus antrean setiap baris di stdout tanpa memblokir utas utama.
- Tabel pengirimannya sederhana `dict[str, Session]`.
- Penanganan tabrakan bersifat eksplisit: ketika dua server mendeklarasikan nama yang sama, server selanjutnya akan diganti namanya dengan awalan.

## Kirim

Lesson ini menghasilkan `outputs/skill-mcp-client-harness.md`. Mengingat daftar deklaratif server MCP (nama, prompt, argumen), keterampilan menghasilkan harness yang memunculkannya, menggabungkan daftar alat, dan mengirimkan fungsi perutean dengan resolusi tabrakan.

## Latihan

1. Jalankan `code/main.py` dan lihat log spawn server. Matikan salah satu proses server yang disimulasikan dengan SIGTERM dan amati bagaimana klien mendeteksi EOF dan menandai sesi tersebut sebagai mati.

2. Menerapkan awalan namespace. Ketika dua server mengekspos `search`, ganti nama server kedua menjadi `<server>/search`. Perbarui tabel pengiriman dan verifikasi rute panggilan alat dengan benar.

3. Tambahkan backoff gaya kumpulan koneksi untuk memulai ulang server: backoff eksponensial pada kegagalan berturut-turut, dibatasi hingga 30 detik, mengirimkan pemberitahuan kepada pengguna setelah tiga kegagalan.

4. Buat sketsa klien yang mendukung 100 server MCP secara bersamaan. Struktur data apa yang menggantikan dikt pengiriman sederhana? (Petunjuk: coba untuk penspasian nama awalan, ditambah metrik untuk jumlah alat per server.)5. Port klien ke MCP Python SDK resmi. SDK membungkus `stdio_client` dan `ClientSession`. Code harus menyusut dari ~200 baris menjadi ~40 baris sambil mempertahankan perutean multi-server.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Klien MCP | "Agen tuan rumah" | Proses yang memunculkan server dan mengatur panggilan alat |
| Sesi | "Status per server" | Kemampuan, daftar alat, dan pembukuan permintaan yang tertunda |
| Ruang nama yang digabungkan | "Satu daftar alat" | Kumpulan nama alat yang datar di semua server aktif |
| Tabrakan ruang nama | "Alat dua server yang sama" | Klien harus mengawali, menolak, atau mendahului duplikat |
| Perutean | "Siapa yang menerima telepon ini?" | Pengiriman dari nama alat ke server pemilik |
| Pembaca latar belakang | "Stdout tanpa pemblokiran" | Thread atau tugas yang menguras server stdout ke dalam antrian |
| Panggilan balik pengambilan sample | "LLM-sebagai-layanan" | Pengendali klien untuk `sampling/createMessage` dari server |
| `notifications/*_changed` | "Mutasi primitif" | Sinyal klien harus menemukan kembali atau membaca ulang |
| Kebijakan penyambungan kembali | "Saat server mati" | Mulai ulang semantik ketika transportasi gagal |
| Sesi Stdio | "Proses = sesi" | Tidak ada ID sesi; proses anak seumur hidup adalah sesi |

## Bacaan Lanjutan

- [Protokol Konteks Model — Spesifikasi klien](https://modelcontextprotocol.io/spesifikasi/2025-11-25/client) — perilaku klien kanonik
- [MCP — Panduan klien Quickstart](https://modelcontextprotocol.io/quickstart/client) — tutorial klien hello-world dengan Python SDK
- [MCP Python SDK — modul klien](https://github.com/modelcontextprotocol/python-sdk) — referensi `ClientSession` dan `stdio_client`
- [MCP TypeScript SDK — Klien](https://github.com/modelcontextprotocol/typescript-sdk) — TS paralel
- [VS Code — MCP dalam ekstensi](https://code.visualstudio.com/api/extension-guides/ai/mcp) — cara VS Code menggandakan beberapa server MCP dalam satu host editor
