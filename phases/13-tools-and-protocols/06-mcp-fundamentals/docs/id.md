# Dasar-Dasar MCP — Primitif, Siklus Hidup, Basis JSON-RPC

> Setiap integrasi sebelum MCP hanya dilakukan sekali saja. Model Context Protocol, pertama kali dikirimkan oleh Anthropic pada November 2024 dan sekarang dikelola oleh Agentic AI Foundation dari Linux Foundation, menstandardisasi penemuan dan pemanggilan sehingga klien mana pun dapat berbicara ke server mana pun. Spesifikasi 25-11-2025 menyebutkan enam primitif (tiga server, tiga klien), siklus hidup tiga fase, dan format kabel JSON-RPC 2.0. Learn itu dan sisa bab MCP dari fase ini menjadi bacaan.

**Type:** Learn
**Language:** Python (stdlib, pengurai JSON-RPC)
**Prerequisites:** Fase 13 · 01 hingga 05 (antarmuka alat dan pemanggilan fungsi)
**Waktu:** ~45 menit

## Tujuan Pembelajaran

- Sebutkan keenam primitif MCP (alat, sumber daya, petunjuk di server; akar, pengambilan sample, elisitasi pada klien) dan berikan masing-masing satu kasus penggunaan.
- Telusuri siklus hidup tiga fase (inisialisasi, pengoperasian, matikan) dan nyatakan siapa yang mengirim pesan mana pada setiap fase.
- Mengurai dan memancarkan permintaan, respons, dan amplop pemberitahuan JSON-RPC 2.0.
- Jelaskan apa itu negosiasi kapabilitas di `initialize` dan apa yang akan gagal tanpanya.

## Masalah

Sebelum MCP, setiap agen yang menggunakan alat memiliki protokolnya sendiri. Kursor memiliki sistem alat berbentuk MCP tetapi tidak kompatibel. Claude Desktop dikirimkan dengan yang berbeda. Ekstensi Copilot VS Code memiliki yang ketiga. Sebuah tim yang membuat alat "kueri Postgres" menulis alat yang sama tiga kali, masing-masing ke API host yang berbeda. Menggunakannya kembali memerlukan penyalinan code.

Hasilnya adalah ledakan integrasi satu kali pada masa Kambrium dan batas kecepatan ekosistem.

MCP memperbaikinya dengan menstandarkan format kabel. Satu server MCP berfungsi di setiap klien MCP: Claude Desktop, ChatGPT, Cursor, VS Code, Gemini, Goose, Zed, Windsurf, 300+ klien pada April 2026. 110 juta unduhan SDK bulanan. 10.000+ server publik. Linux Foundation mulai dikelola pada bulan Desember 2025 di bawah Agentic AI Foundation yang baru.

Revisi spesifikasi yang digunakan pada fase ini adalah **25-11-2025**. Ia menambahkan Tugas async (SEP-1686), elisitasi mode URL (SEP-1036), pengambilan sample dengan alat (SEP-1577), persetujuan cakupan tambahan (SEP-835), dan semantik indikator sumber daya OAuth 2.1. Fase 13 · 09 hingga 16 mencakup perluasan tersebut. Lesson ini berhenti di pangkalan.

## Konsep

### Tiga server primitif

1. **Alat.** Tindakan yang dapat dipanggil. Putaran empat langkah yang sama dari Fase 13 · 01.
2. **Sumber Daya.** Data yang terpapar. Konten hanya-baca yang dapat ditangani oleh URI: `file:///path`, `db://query/...`, skema khusus.
3. **Permintaan.** Templat yang dapat digunakan kembali. Prompt garis miring di UI host; server menyediakan template, klien mengisi argumen.

### Tiga klien primitif

4. **Roots.** Kumpulan URI yang boleh disentuh oleh server. Klien mendeklarasikannya; server menghormati mereka.
5. **Sampling.** Server meminta model klien untuk melakukan penyelesaian. Mengaktifkan loop agen yang dihosting server tanpa kunci API sisi server.
6. **Elicitation.** Server meminta input terstruktur dari pengguna klien di tengah penerbangan. Formulir atau URL (SEP-1036).

Setiap kemampuan di MCP dimiliki oleh salah satu dari enam kemampuan ini. Fase 13 · 10 hingga 14 mencakup masing-masing secara mendalam.

### Format kabel: JSON-RPC 2.0

Setiap pesan adalah objek JSON dengan bidang berikut:

- Permintaan: `{jsonrpc: "2.0", id, method, params}`.
- Tanggapan: `{jsonrpc: "2.0", id, result | error}`.
- Pemberitahuan: `{jsonrpc: "2.0", method, params}` — tidak `id`, tidak ada respons yang diharapkan.

Spesifikasi dasar memiliki ~15 metode, dikelompokkan berdasarkan primitif. Yang penting:- `initialize` / `initialized` (jabat tangan)
- `tools/list`, `tools/call`
- `resources/list`, `resources/read`, `resources/subscribe`
- `prompts/list`, `prompts/get`
- `sampling/createMessage` (server-ke-klien)
- `notifications/tools/list_changed`, `notifications/resources/updated`, `notifications/progress`

### Siklus hidup tiga fase

**Phase 1: inisialisasi.**

Klien mengirimkan `initialize` dengan `capabilities` dan `clientInfo`. Server merespons dengan `capabilities`, `serverInfo` miliknya sendiri, dan versi spesifikasi yang digunakannya. Klien mengirimkan `notifications/initialized` ketika sudah mencerna respons. Mulai saat ini, kedua belah pihak dapat mengirimkan permintaan sesuai kemampuan yang dinegosiasikan.

**Phase 2: operasi.**

Dua arah. Klien menelepon `tools/list` untuk menemukan, lalu `tools/call` untuk memanggil. Server dapat mengirimkan `sampling/createMessage` jika menyatakan kemampuan tersebut. Server dapat mengirim `notifications/tools/list_changed` ketika set alatnya bermutasi. Klien dapat mengirim `notifications/roots/list_changed` ketika pengguna mengubah cakupan root.

**Fase 3: penutupan.**

Kedua belah pihak menutup transportasi. Tidak ada metode pematian terstruktur di MCP; transport (stdio atau Streamable HTTP, Fase 13 · 09) membawa sinyal akhir koneksi.

### Negosiasi kemampuan

`capabilities` dalam jabat tangan `initialize` adalah kontraknya. Contoh dari server:

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

Server menyatakan dapat mengeluarkan notifikasi `tools/list_changed` dan mendukung `resources/subscribe`. Klien setuju dengan menyatakan miliknya sendiri:

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

Jika klien tidak mendeklarasikan `sampling`, server tidak boleh memanggil `sampling/createMessage`. Simetris: jika server tidak mendeklarasikan `resources.subscribe`, klien tidak boleh mencoba berlangganan.

Hal inilah yang mencegah terjadinya pergeseran ekosistem. Klien yang tidak mendukung pengambilan sample masih merupakan klien MCP yang valid; server yang tidak memanggil `sampling` masih merupakan server MCP yang valid. Mereka hanya tidak menggunakan feature itu bersama-sama.

### Konten terstruktur dan bentuk kesalahan

`tools/call` mengembalikan `content` array blok yang diketik: `text`, `image`, `resource`. Fase 13 · 14 menambahkan Aplikasi MCP (`ui://` UI interaktif) ke daftar itu.

Kesalahan menggunakan code kesalahan JSON-RPC. Penambahan yang ditentukan spesifikasi: `-32002` "Sumber daya tidak ditemukan", `-32603` "Kesalahan internal", ditambah data kesalahan khusus MCP sebagai `error.data`.

### Kemampuan klien vs detail panggilan alat

Perplexity umum: `capabilities.tools` adalah apakah klien mendukung pemberitahuan perubahan daftar alat. Apakah klien AKAN memanggil alat tertentu merupakan pilihan waktu proses yang ditentukan oleh modelnya, bukan tanda kemampuan. Bendera kemampuan adalah kontrak tingkat spesifikasi. Pilihan modelnya ortogonal.

### Mengapa JSON-RPC dan bukan REST?

JSON-RPC 2.0 (2010) adalah protokol dua arah yang ringan. REST diprakarsai oleh klien. MCP memerlukan pesan yang dimulai oleh server (pengambilan sample, notifikasi), sehingga JSON-RPC dengan bentuk permintaan/respons simetris merupakan pilihan yang tepat. JSON-RPC juga menulis dengan rapi melalui stdio dan WebSocket/Streamable HTTP tanpa menciptakan kembali bentuk permintaan HTTP.

## Pakai`code/main.py` mengirimkan parser dan emitor JSON-RPC 2.0 minimal, lalu menjalankan urutan `initialize` → `tools/list` → `tools/call` → `shutdown` dengan tangan, mencetak setiap pesan. Tidak ada transportasi nyata; hanya bentuk pesannya. Bandingkan dengan spesifikasi yang ditautkan dalam Bacaan Lanjutan untuk memverifikasi setiap amplop.

Apa yang harus dilihat:

- `initialize` mendeklarasikan kemampuan dua arah; tanggapannya memiliki `serverInfo` dan `protocolVersion: "2025-11-25"`.
- `tools/list` mengembalikan array `tools`; setiap entri memiliki `name`, `description`, `inputSchema`.
- `tools/call` menggunakan `params.name` dan `params.arguments`.
- Respons `content` adalah array blok `{type, text}`.

## Kirim

Lesson ini menghasilkan `outputs/skill-mcp-handshake-tracer.md`. Dengan adanya transkrip gaya pcap dari interaksi klien-server MCP, keterampilan tersebut memberi anotasi pada setiap pesan dengan primitif mana, fase siklus hidup mana, dan kemampuan apa yang bergantung pada pesan tersebut.

## Latihan

1. Jalankan `code/main.py`. Identifikasi jalur di mana negosiasi kemampuan terjadi dan jelaskan apa yang akan berubah jika server tidak mendeklarasikan `tools.listChanged`.

2. Perluas parser untuk menangani `notifications/progress`. Bentuk pesan: `{method: "notifications/progress", params: {progressToken, progress, total}}`. Keluarkan saat `tools/call` yang sudah berjalan lama sedang berlangsung dan konfirmasikan bahwa pengendali klien akan menampilkan bilah kemajuan.

3. Baca spesifikasi MCP 25-11-2025 dari atas ke bawah — seluruh dokumen terdiri dari sekitar 80 halaman. Identifikasi satu tanda kemampuan yang TIDAK dibutuhkan sebagian besar server. Petunjuk: ini berkaitan dengan langganan sumber daya.

4. Buat sketsa di atas kertas primitif yang akan menjadi milik feature "cron job" hipotetis. (Petunjuk: server ingin klien memanggilnya pada waktu yang dijadwalkan. Tak satu pun dari enam primitif yang cocok saat ini.) Peta jalan MCP tahun 2026 memiliki rancangan SEP untuk ini.

5. Parsing satu log sesi dari server MCP terbuka di GitHub. Hitung permintaan vs tanggapan vs pesan pemberitahuan. Hitung berapa bagian lalu lintas yang merupakan siklus hidup vs operasi.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| MCP | "Protokol Konteks Model" | Protokol terbuka untuk penemuan dan pemanggilan model-ke-alat |
| Server primitif | "Apa yang diekspos server" | alat (tindakan), sumber daya (data), petunjuk (templat) |
| Klien primitif | "Apa yang klien izinkan digunakan oleh server" | root (ruang lingkup), pengambilan sample (callback LLM), elisitasi (input pengguna) |
| JSON-RPC 2.0 | "Format kawat" | Amplop permintaan/tanggapan/pemberitahuan simetris |
| `initialize` jabat tangan | "Negosiasi kemampuan" | Pasangan pesan pertama; server dan klien mendeklarasikan feature yang mereka dukung |
| `tools/list` | "Penemuan" | Klien meminta server untuk kumpulan alatnya saat ini |
| `tools/call` | "Doa" | Klien meminta server untuk mengeksekusi alat dengan argumen |
| `notifications/*_changed` | "Peristiwa mutasi" | Server memberi tahu klien bahwa daftar primitifnya telah berubah |
| Blok konten | "Hasil yang diketik" | `{type: "text" | "image" | "resource" | "ui_resource"}` dalam hasil alat |
| September | "Proposal Evolusi Spesifikasi" | Draf proposal yang diberi nama (misalnya SEP-1686 untuk Tugas asinkron) |

## Bacaan Lanjutan- [Protokol Konteks Model — Spesifikasi 25-11-2025](https://modelcontextprotocol.io/spesifikasi/2025-11-25) — dokumen spesifikasi kanonik
- [Protokol Konteks Model — Konsep arsitektur](https://modelcontextprotocol.io/docs/concepts/architecture) — model mental enam primitif
- [Anthropic — Memperkenalkan Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — postingan peluncuran November 2024
- [Blog MCP — Ulang tahun MCP pertama](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — retrospektif satu tahun dan perubahan spesifikasi 25-11-2025
- [WorkOS — pembaruan spesifikasi MCP 2025-11-25](https://workos.com/blog/mcp-2025-11-25-spec-update) — ringkasan SEP-1686, 1036, 1577, 835, dan 1724
