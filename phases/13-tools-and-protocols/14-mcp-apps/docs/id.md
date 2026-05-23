# Aplikasi MCP — Sumber Daya UI Interaktif melalui `ui://`

> Output alat hanya teks membatasi apa yang dapat ditampilkan agen. Aplikasi MCP (SEP-1724, resmi 26 Januari 2026) memungkinkan alat mengembalikan HTML interaktif sandbox yang dirender sebaris di Claude Desktop, ChatGPT, Cursor, Goose, dan VS Code. Dasbor, formulir, peta, pemandangan 3D, semuanya melalui satu ekstensi. Lesson ini menjelaskan skema sumber daya `ui://`, `text/html;profile=mcp-app` MIME, protokol postMessage iframe-sandbox, dan permukaan keamanan yang memungkinkan server merender HTML.

**Type:** Build
**Language:** Python (stdlib, pemancar sumber daya UI), HTML (aplikasi contoh)
**Prerequisites:** Fase 13 · 07 (server MCP), Fase 13 · 10 (sumber daya)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Kembalikan sumber daya `ui://` dari panggilan alat dan atur MIME dan metadata yang benar.
- Deklarasikan UI terkait alat dengan `_meta.ui.resourceUri`, `_meta.ui.csp`, dan `_meta.ui.permissions`.
- Mengimplementasikan iframe sandbox postMessage JSON-RPC untuk komunikasi UI-ke-host.
- Terapkan CSP dan default kebijakan izin yang melindungi dari serangan yang berasal dari UI.

## Masalah

Alat `visualize_timeline` era 2025 dapat mengembalikan "Berikut adalah 14 catatan yang disusun secara kronologis: ...". Itu adalah sebuah paragraf. Pengguna sebenarnya menginginkan timeline interaktif. Sebelum Aplikasi MCP, pilihannya adalah: API widget khusus klien (artefak Claude, OpenAI Custom GPT HTML), atau tanpa UI sama sekali.

Aplikasi MCP (SEP-1724, dikirimkan 26 Januari 2026) menstandarkan kontrak. Hasil alat berisi `resource` dengan URI `ui://...` dan MIME-nya `text/html;profile=mcp-app`. Host merendernya dalam iframe sandbox dengan CSP terbatas dan tidak ada akses jaringan kecuali diberikan secara eksplisit. UI di dalam iframe memposting pesan ke host melalui dialek JSON-RPC postMessage kecil.

Setiap klien yang kompatibel (Claude Desktop, ChatGPT, Goose, VS Code) merender sumber daya `ui://` yang sama dengan cara yang sama. Satu server, satu bundel HTML, UI universal.

## Konsep

### Skema sumber daya `ui://`

Sebuah alat kembali:

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

Tuan rumah kemudian memanggil `resources/read` di URI `ui://notes/timeline` dan membalas:

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Kotak pasir iframe

Host merender HTML di dalam kotak pasir `<iframe>` dengan:

- `sandbox="allow-scripts allow-same-origin"` (atau lebih ketat per deklarasi server)
- CSP yang dideklarasikan server diterapkan melalui header respons.
- Tanpa cookie, tanpa penyimpanan lokal dari asal host.
- Akses jaringan terbatas pada `connectSrc` di CSP.

### protokol pascaPesan

Iframe berkomunikasi dengan host melalui `window.postMessage`. Dialek kecil JSON-RPC 2.0:

Selalu sematkan `targetOrigin` ke asal rekan yang sebenarnya, dan di pihak penerima, validasi `event.origin` terhadap daftar yang diizinkan sebelum memproses muatan apa pun. Jangan pernah menggunakan `"*"` untuk kedua sisi pipeline ini — isi pipeline membawa panggilan alat dan pembacaan sumber daya.

```js
// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

Metode sisi host yang tersedia yang dapat dipanggil oleh UI:

- `host.callTool(name, arguments)` — memanggil alat server.
- `host.readResource(uri)` — membaca sumber daya MCP.
- `host.getPrompt(name, arguments)` — mengambil template prompt.
- `host.close()` — menutup UI.

Setiap panggilan tetap melalui protokol MCP dan mewarisi izin server.

### Izin

Daftar `_meta.ui.permissions` meminta kemampuan tambahan:- `camera` — mengakses kamera pengguna (digunakan untuk UI pemindaian dokumen).
- `microphone` — input suara.
- `geolocation` — lokasi.
- `network:*` — akses jaringan yang lebih luas dari yang dimungkinkan oleh `connectSrc` saja.

Setiap izin adalah prompt yang dilihat pengguna sebelum UI dirender.

### Risiko keamanan

HTML dalam iframe tetaplah HTML. Permukaan serangan baru:

- **Injeksi cepat melalui UI.** UI server berbahaya dapat menampilkan teks yang tampak seperti pesan sistem dan menipu pengguna. Rendering host harus membedakan UI server dari UI host.
- **Eksfiltrasi melalui `connectSrc`.** Jika CSP mengizinkan `connect-src: *`, UI dapat mengirim data ke mana saja. Standarnya harus ketat.
- **Clickjacking.** UI melapisi host chrome. Host harus mencegah manipulasi indeks-z dan menerapkan aturan opasitas.
- **Mencuri fokus.** UI mengambil fokus keyboard dan menangkap pesan berikutnya. Tuan rumah harus mencegat.

Fase 13 · 15 mencakup hal ini secara mendalam sebagai bagian dari keamanan MCP; lesson ini memperkenalkan mereka.

### `ui/initialize` jabat tangan

Setelah iframe dimuat, ia mengirimkan `ui/initialize` melalui postMessage:

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

Host merespons dengan kemampuan dan token sesi. UI menggunakan token sesi pada setiap panggilan host berikutnya.

### Primitif AppRenderer / AppFrame SDK

SDK ext-apps memperlihatkan dua primitif kenyamanan:

- `AppRenderer` (sisi server) — membungkus komponen React / Vue / Solid dan memancarkan sumber daya `ui://` dengan MIME dan metadata yang tepat.
- `AppFrame` (sisi klien) — menerima sumber daya, memasang iframe, dan memediasi postMessage.

kamu dapat menggunakan ini atau memutar HTML dan JSON-RPC secara manual.

### Status ekosistem

Aplikasi MCP dikirimkan pada 26 Januari 2026. Dukungan klien per April 2026:

- **Claude Desktop.** Dukungan penuh sejak Januari 2026.
- **ChatGPT.** Dukungan penuh melalui SDK Aplikasi (protokol Aplikasi MCP yang sama).
- **Kursor.** Beta; aktifkan melalui pengaturan.
- **Code VS.** Khusus pembuatan orang dalam.
- **Angsa.** Dukungan penuh.
- **Zed, Selancar Angin.** Peta Jalan.

Server dalam produksi: dasbor, visualisasi peta, tabel data, pembuat bagan, pratinjau IDE kotak pasir.

## Pakai

`code/main.py` memperluas server catatan dengan alat `visualize_timeline` yang mengembalikan sumber daya `ui://notes/timeline`, ditambah pengendali untuk `resources/read` pada URI tersebut yang mengembalikan bundel HTML kecil namun lengkap dengan garis waktu SVG. HTML bertemplat stdlib — tidak ada sistem build. postMessage dibuat sketsa di komentar JS karena stdlib tidak dapat menggerakkan browser.

Apa yang harus dilihat:

- `_meta.ui` pada respons alat membawa resourceUri, CSP, izin.
- HTML dirender tanpa akses jaringan; semua data sebaris.
- JS memanggil `host.callTool` melalui `window.parent.postMessage` (didokumentasikan tetapi tidak aktif dalam demo stdlib ini).

## Kirim

Lesson ini menghasilkan `outputs/skill-mcp-apps-spec.md`. Dengan adanya alat yang akan memanfaatkan UI interaktif, keterampilan tersebut menghasilkan kontrak Aplikasi MCP lengkap: `ui://` URI, CSP, izin, titik masuk postMessage, dan daftar periksa keamanan.

## Latihan

1. Jalankan `code/main.py` dan periksa HTML yang dipancarkan. Buka HTML langsung di browser; verifikasi render SVG. Kemudian buat sketsa kontrak postMessage yang akan digunakan UI untuk memanggil `host.callTool("notes_update", ...)`.

2. Perketat CSP: hapus `'unsafe-inline'` dan gunakan kebijakan skrip berbasis nonce. Apa saja perubahan pada code pembuatan HTML?3. Tambahkan sumber daya UI kedua `ui://notes/editor` dengan formulir untuk mengedit catatan. Saat pengguna mengirimkan, iframe memanggil `host.callTool("notes_update", ...)`.

4. Audit permukaan serangan UI. Di mana server jahat dapat menyuntikkan konten? Apa yang dilindungi oleh kotak pasir iframe dan apa yang tidak?

5. Baca spesifikasi SEP-1724 dan identifikasi satu kemampuan di MCP Apps SDK yang tidak digunakan oleh implementasi mainan ini. (Petunjuk: sinkronisasi status tingkat komponen.)

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Aplikasi MCP | "Sumber daya UI interaktif" | Perpanjangan SEP-1724 dikirimkan 26-01-2026 |
| `ui://` | "Skema URI Aplikasi" | Skema sumber daya untuk paket UI |
| `text/html;profile=mcp-app` | "MIME" | Tipe konten untuk HTML Aplikasi MCP |
| Kotak pasir iframe | "Render wadah" | Sandboxing browser pada UI dengan CSP dan izin |
| postMessage JSON-RPC | "Kabel UI-ke-host" | Dialek JSON-RPC-over-postMessage kecil untuk panggilan host |
| `_meta.ui` | "Pengikatan Alat-UI" | Metadata yang menghubungkan hasil alat ke sumber daya UI |
| CSP | "Kebijakan-Keamanan-Konten" | Mendeklarasikan sumber yang diizinkan untuk skrip, jaringan, gaya |
| Perender Aplikasi | "Server SDK primitif" | Mengonversi komponen framework menjadi sumber daya `ui://` |
| Bingkai Aplikasi | "Klien SDK primitif" | Pembantu pemasangan iframe yang memediasi postMessage |
| `ui/initialize` | "jabat tangan" | Posting pertamaPesan dari UI ke host |

## Bacaan Lanjutan

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — implementasi referensi dan SDK
- [Spesifikasi Aplikasi MCP 26-01-2026](https://github.com/modelcontextprotocol/ext-apps/blob/main/spesifikasi/2026-01-26/apps.mdx) — dokumen spesifikasi formal
- [MCP — Ikhtisar ekstensi aplikasi](https://modelcontextprotocol.io/extensions/apps/overview) — dokumentasi tingkat tinggi
- [Blog MCP — Peluncuran Aplikasi MCP](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — postingan peluncuran Januari 2026
- [Referensi API Aplikasi MCP](https://apps.extensions.modelcontextprotocol.io/api/) — Referensi SDK bergaya JSDoc
