# Transportasi MCP — stdio vs HTTP yang Dapat Dialirkan vs Migrasi SSE

> stdio bekerja secara lokal dan tidak di tempat lain. HTTP yang dapat dialirkan (26-03-2025) adalah standar distance jauh. Transportasi HTTP+SSE yang lama tidak digunakan lagi dan dihapus pada pertengahan tahun 2026. Memilih transportasi yang salah memerlukan migrasi; memilih yang tepat akan membeli server MCP yang dapat dihosting distance jauh dengan kontinuitas sesi dan perlindungan pengikatan ulang DNS.

**Type:** Learn
**Language:** Python (stdlib, kerangka titik akhir HTTP yang dapat dialirkan)
**Prerequisites:** Fase 13 · 07, 08 (server dan klien MCP)
**Waktu:** ~45 menit

## Tujuan Pembelajaran

- Pilih antara stdio dan HTTP Streamable berdasarkan bentuk penerapan (lokal vs distance jauh, proses tunggal vs armada).
- Menerapkan pola titik akhir tunggal HTTP Streamable: POST untuk permintaan, GET untuk aliran sesi.
- Menerapkan validasi `Origin` dan semantik id sesi untuk mengalahkan pengikatan ulang DNS.
- Migrasikan server HTTP+SSE lama ke HTTP Streamable sebelum tenggat waktu penghapusan pada pertengahan tahun 2026.

## Masalah

Transportasi distance jauh MCP pertama (2024-11) adalah HTTP+SSE: dua titik akhir, satu untuk POST klien dan satu pipeline Server-Sent-Events untuk aliran server-ke-klien. Itu berhasil. Itu juga canggung: dua titik akhir per sesi, cache yang rusak di depan beberapa CDN, dan ketergantungan yang kuat pada koneksi SSE yang berumur panjang yang dihentikan secara agresif oleh beberapa WAF.

Spesifikasi 26-03-2025 menggantikannya dengan HTTP yang Dapat Dialirkan: satu titik akhir, POST untuk permintaan klien, GET untuk membuat aliran sesi, keduanya berbagi header `Mcp-Session-Id`. Setiap server yang dibangun atau dimigrasi sejak saat itu menggunakan HTTP yang Dapat Dialiri. Mode SSE lama tidak digunakan lagi — Atlassian Rovo menghapusnya pada 30 Juni 2026; Keboola 1 April 2026; sebagian besar server perusahaan yang tersisa pada akhir tahun 2026.

Dan stdio masih penting untuk server lokal. Claude Desktop, VS Code, dan setiap klien berbentuk IDE menelurkan server melalui stdio. Model mental yang tepat: stdio untuk "mesin ini", HTTP yang dapat dialirkan untuk "melalui jaringan". Tidak ada penyeberangan.

## Konsep

### stdio

- Transportasi proses anak. Klien memunculkan server, berkomunikasi melalui stdin/stdout.
- Satu objek JSON per baris. Dibatasi baris baru.
- Tidak ada id sesi; identitas proses adalah sesi.
- Tidak diperlukan autentikasi (anak mewarisi batas kepercayaan orang tua).
- Jangan pernah gunakan untuk server distance jauh — kamu memerlukan SSH atau socat untuk melakukan terowongan, lalu gunakan HTTP yang Dapat Dialiri.

### HTTP yang dapat dialirkan

Titik akhir tunggal `/mcp` (atau jalur apa pun). Mendukung tiga metode HTTP:

- **POST /mcp.** Klien mengirimkan pesan JSON-RPC. Server membalas dengan respons JSON tunggal, atau aliran SSE yang berisi satu atau lebih respons (berguna untuk respons batch dan pemberitahuan terkait permintaan tersebut).
- **GET /mcp.** Klien membuka pipeline SSE yang berumur panjang. Server menggunakannya untuk permintaan server-ke-klien (sampling, notifikasi, elisitasi).
- **HAPUS /mcp.** Klien secara eksplisit mengakhiri sesi.

Sesi diidentifikasi oleh header `Mcp-Session-Id` yang disetel server pada respons pertama dan klien menggemakan setiap permintaan berikutnya. ID sesi HARUS acak secara kriptografis (128+ bit); ID yang dipilih klien ditolak demi keamanan.

### Titik akhir tunggal vs dua

Mode dua titik akhir dari spesifikasi lama masih dapat dipanggil pada tahun 2026 — spesifikasi menyatakannya "kompatibel dengan versi lama". Namun semua server baru harus berupa titik akhir tunggal. SDK resmi mengeluarkan titik akhir tunggal; gunakan mode lawas hanya saat berbicara dengan remote yang belum dimigrasi.

### `Origin` validasi dan pengikatan ulang DNSBrowser bukan klien MCP (saat ini), namun penyerang dapat membuat halaman web yang meyakinkan browser untuk POST ke `localhost:1234/mcp` — tempat server MCP lokal pengguna mendengarkan. Jika server tidak memeriksa `Origin`, kebijakan asal yang sama pada browser tidak akan menyimpannya karena `Origin: http://evil.com` valid lintas asal.

Spesifikasi 25-11-2025 mengharuskan server menolak permintaan yang `Origin` tidak ada dalam daftar yang diizinkan. Daftar yang diizinkan biasanya berisi host klien MCP (`https://claude.ai`, `vscode-webview://*`) dan varian localhost untuk UI lokal.

### Siklus hidup id sesi

1. Klien mengirimkan permintaan pertama tanpa `Mcp-Session-Id`.
2. Server memberikan id acak, menetapkan `Mcp-Session-Id` pada header respons.
3. Klien menggemakan header tersebut pada semua permintaan berikutnya dan pada `GET /mcp` untuk streaming.
4. Sesi dapat dicabut oleh server; klien melihat 404 pada permintaan berikutnya dan harus menginisialisasi ulang.
5. Klien dapat secara eksplisit MENGHAPUS sesi untuk mematikan bersih.

### Tetap hidup dan terhubung kembali

Koneksi SSE terputus. Klien membangun kembali dengan MENDAPATKAN kembali dengan `Mcp-Session-Id` yang sama. Server HARUS mengantri peristiwa yang terlewat selama pemadaman (hingga jangka waktu yang wajar) dan diputar ulang melalui header `last-event-id` yang digaungkan klien.

Fase 13 · 13 mencakup Tugas, yang memungkinkan pekerjaan yang berjalan lama tetap bertahan bahkan setelah sesi penuh tersambung kembali.

### Pemeriksaan kompatibilitas mundur

Klien yang ingin mendukung server lama dan baru:

1. POSTING ke `/mcp`.
2. Jika respons `200 OK` dengan JSON atau SSE, ini adalah HTTP yang Dapat Dialiri.
3. Jika responsnya adalah `200 OK` dengan `Content-Type: text/event-stream` DAN header `Location` yang menunjuk ke titik akhir sekunder, ini adalah HTTP+SSE lama; ikuti `Location`.

### Cloudflare, ngrok, dan hosting

Server MCP distance jauh produksi pada tahun 2026 berjalan di Cloudflare Workers (dengan SDK Agen MCP), Vercel Functions, atau Node/Python dalam container. Kuncinya: hosting kamu harus mendukung koneksi HTTP yang tahan lama untuk SSE GET. Tingkat gratis Vercel dibatasi pada 10 detik dan tidak cocok. Cloudflare Workers mendukung streaming tanpa batas.

### Komposisi gerbang

Saat kamu menjalankan beberapa server MCP dengan gateway (Fase 13 · 17), gateway tersebut adalah titik akhir HTTP Streamable tunggal yang menulis ulang id sesi dan melakukan multipleks di bagian hulu. Alat digabungkan pada layer gateway; klien melihat satu server logis.

### Mode kegagalan transportasi

- **stdio SIGPIPE.** Kematian proses anak di tengah penulisan memunculkan SIGPIPE; server harus keluar dengan bersih. Klien harus mendeteksi EOF dan menandai sesi mati.
- **HTTP 502/504.** Cloudflare, nginx, dan proxy lainnya memunculkan hal ini pada kegagalan upstream. Klien HTTP yang dapat dialirkan harus mencoba lagi satu kali setelah kemunduran singkat.
- **Koneksi SSE terputus.** TCP RST, waktu tunggu proxy habis, atau perubahan jaringan klien akan menutup aliran. Klien terhubung kembali dengan `Mcp-Session-Id` dan opsional `last-event-id` untuk melanjutkan.
- **Pencabutan sesi.** Server membatalkan id sesi; klien melihat 404 pada permintaan berikutnya. Klien harus berjabat tangan kembali.
- **Jam miring.** Perhitungan Resource-TTL pada klien berbeda dari server. Klien harus memperlakukan stempel waktu server sebagai sesuatu yang otoritatif.

### Kapan harus melewati HTTP yang Dapat DialiriBeberapa perusahaan menerapkan server MCP di belakang gRPC atau transportasi antrian pesan di dalam jaringan mereka sendiri. Ini tidak standar — spesifikasi MCP tidak mendefinisikannya secara formal. Gateway dapat mengekspos platform HTTP Streamable ke klien MCP saat menggunakan gRPC secara internal. Jaga agar permukaan luar tetap sesuai spesifikasi; gateway memiliki terjemahannya.

## Pakai

`code/main.py` mengimplementasikan titik akhir HTTP Streamable minimal menggunakan `http.server` (stdlib). Ini menangani POST, GET, dan DELETE pada `/mcp`, menetapkan `Mcp-Session-Id` pada respons pertama, memvalidasi `Origin`, dan menolak permintaan dari asal yang tidak diizinkan. Penangan menggunakan kembali logika pengiriman server catatan Lesson 07.

Apa yang harus dilihat:

- Penangan POST membaca isi JSON-RPC, mengirimkan, dan menulis respons JSON (varian respons tunggal; varian SSE secara struktural serupa).
- Cek `Origin` menolak pemeriksaan default `http://evil.example` tetapi menerima `http://localhost`.
- Id sesi adalah string hex 128-bit acak; server menyimpan status per sesi di memori.

## Kirim

Lesson ini menghasilkan `outputs/skill-mcp-transport-migrator.md`. Dengan adanya server MCP HTTP+SSE (lama), keterampilan ini menghasilkan rencana migrasi ke HTTP yang Dapat Dialirkan dengan kontinuitas id sesi, pemeriksaan Asal, dan dukungan pemeriksaan yang kompatibel dengan versi sebelumnya.

## Latihan

1. Jalankan `code/main.py`. POSTING `initialize` dari `curl` dan amati header respons `Mcp-Session-Id`. POST permintaan kedua yang menggemakan header dan memverifikasi kontinuitas sesi.

2. Tambahkan pengendali GET yang membuka aliran SSE. Kirim satu acara `notifications/progress` setiap lima detik. Hubungkan kembali dengan MENDAPATKAN kembali dengan id sesi yang sama dan konfirmasikan server menerimanya.

3. Menerapkan logika pemutaran ulang `last-event-id`. Saat menyambung kembali, putar ulang peristiwa apa pun yang dihasilkan sejak id tersebut.

4. Perluas validasi `Origin` untuk mendukung pola wildcard (`https://*.example.com`) dan konfirmasikan bahwa validasi menerima `https://app.example.com` tetapi menolak `https://evil.example.com.attacker.net`.

5. Ambil server HTTP+SSE lama dari registri resmi (ada beberapa) dan buat sketsa migrasinya: perubahan apa yang terjadi dalam penanganan titik akhir, pembuatan id sesi, dan semantik header.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| transportasi stdio | "Proses anak lokal" | JSON-RPC melalui stdin/stdout, dibatasi baris baru |
| HTTP yang dapat dialirkan | "Transportasi distance jauh" | POST titik akhir tunggal + GET + SSE opsional, spesifikasi 26-03-2025 |
| HTTP+SSE | "Warisan" | Model dua titik akhir dihapus pada pertengahan tahun 2026 |
| `Mcp-Session-Id` | "Judul sesi" | Id acak yang ditetapkan server bergema pada setiap permintaan berikutnya |
| `Origin` daftar yang diizinkan | "Pertahanan pengikatan ulang DNS" | Tolak permintaan yang Asalnya tidak disetujui |
| Titik akhir tunggal | "Satu URL" | `/mcp` menangani POST / GET / DELETE untuk semua operasi sesi |
| `last-event-id` | "Pemutaran ulang SSE" | Header digunakan untuk melanjutkan streaming yang terputus tanpa melewatkan acara |
| Probe kompatibel mundur | "Deteksi lama vs baru" | Pemeriksaan bentuk respons klien yang memilih transportasi |
| HTTP berumur panjang | "Streaming SSE" | Server mendorong peristiwa selama beberapa menit atau jam pada satu koneksi TCP |
| Pencabutan sesi | "Paksa memulai kembali" | Server membatalkan id sesi; klien harus berjabat tangan lagi |

## Bacaan Lanjutan- [MCP — Spesifikasi transportasi dasar 25-11-2025](https://modelcontextprotocol.io/spesification/2025-11-25/basic/transports) — referensi kanonik untuk stdio dan HTTP yang Dapat Dialiri
- [MCP — Spesifikasi transport dasar 26-03-2025](https://modelcontextprotocol.io/spesification/2025-03-26/basic/transports) — revisi yang memperkenalkan Streamable HTTP
- [Cloudflare — Transportasi MCP](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Pola HTTP Streamable yang dihosting oleh pekerja
- [AWS — mekanisme transportasi MCP](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — perbandingan berbagai bentuk penerapan
- [Atlassian — pemberitahuan penghentian HTTP+SSE](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — contoh tenggat waktu migrasi yang konkret
