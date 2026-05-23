# Membangun Server MCP — Python + TypeScript SDK

> Kebanyakan tutorial MCP hanya menampilkan stdio hello-worlds. Server sebenarnya memaparkan alat plus sumber daya plus prompt, menangani negosiasi kemampuan, mengeluarkan kesalahan terstruktur, dan bekerja dengan cara yang sama di seluruh SDK. Lesson ini membangun server catatan end-to-end: stdlib stdio transport, pengiriman JSON-RPC, tiga server primitif, dan gaya fungsi murni yang dimasukkan ke dalam FastMCP Python SDK atau TypeScript SDK ketika kamu lulus.

**Type:** Build
**Language:** Python (stdlib, server stdio MCP)
**Prerequisites:** Fase 13 · 06 (dasar-dasar MCP)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Menerapkan metode `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, dan `prompts/get`.
- Tulis loop pengiriman yang membaca pesan JSON-RPC dari stdin dan menulis tanggapan ke stdout.
- Memancarkan respons kesalahan terstruktur sesuai spesifikasi JSON-RPC 2.0 dan code tambahan MCP.
- Lulus implementasi stdlib ke FastMCP (Python SDK) atau TypeScript SDK tanpa menulis ulang logika alat.

## Masalah

Sebelum kamu dapat menggunakan transport distance jauh (Fase 13 · 09) atau layer auth (Fase 13 · 16), kamu memerlukan server lokal yang bersih. Lokal berarti stdio: server dihasilkan oleh klien sebagai proses anak, pesan mengalir melalui stdin/stdout yang dibatasi baris baru.

Spesifikasi 25-11-2025 menetapkan bahwa pesan stdio dikodekan sebagai objek JSON dengan pemisah `\n` yang eksplisit. Tidak ada SSE di sini; SSE adalah mode distance jauh yang lama dan akan dihapus pada pertengahan tahun 2026 (server Rovo MCP Atlassian tidak lagi menggunakannya pada tanggal 30 Juni 2026; Keboola pada tanggal 1 April 2026). Untuk stdio, satu objek JSON per baris adalah format seluruh kabel.

Server catatan adalah bentuk yang bagus karena menjalankan ketiga server primitif. Alat melakukan mutasi (`notes_create`). Sumber daya memaparkan data (`notes://{id}`). Templat pengiriman permintaan (`review_note`). Bentuk lesson ini dapat digeneralisasikan ke bidang apa pun.

## Konsep

### Lingkaran pengiriman

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

Tiga aturan:

- Jangan mencetak apa pun ke stdout yang bukan amplop JSON-RPC. Log debug menuju ke stderr.
- Setiap permintaan HARUS dicocokkan dengan respons yang membawa `id` yang sama.
- Notifikasi TIDAK BOLEH ditanggapi.

### Menerapkan `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

Nyatakan hanya apa yang kamu dukung. Klien mengandalkan kemampuan yang diatur ke feature gerbang.

### Menerapkan `tools/list` dan `tools/call`

`tools/list` mengembalikan `{tools: [...]}` dengan setiap entri memiliki `name`, `description`, `inputSchema`. `tools/call` mengambil `{name, arguments}` dan mengembalikan `{content: [blocks], isError: bool}`.

Blok konten diketik. Yang paling umum:

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

Kesalahan alat terjadi dalam dua bentuk. Kesalahan tingkat protokol (metode tidak diketahui, parameter buruk) adalah kesalahan JSON-RPC. Kesalahan tingkat alat (panggilan valid tetapi alat gagal) dikembalikan sebagai `{content: [...], isError: true}`. Hal ini memungkinkan model melihat kegagalan dalam konteksnya.

### Menerapkan sumber daya

Sumber daya dirancang hanya untuk dibaca. `resources/list` mengembalikan manifes; `resources/read` mengembalikan konten. URI dapat berupa `file://...`, `http://...`, atau skema khusus seperti `notes://`.

Saat kamu mengekspos data sebagai sumber daya dan bukan sebagai alat:- Model tidak "menyebutnya"; klien dapat memasukkannya ke dalam konteks berdasarkan permintaan pengguna.
- Langganan membiarkan server mendorong pembaruan ketika sumber daya berubah (Fase 13 · 10).
- Fase 13 · 14 memperluasnya dengan `ui://` untuk sumber daya interaktif.

### Menerapkan prompt

Prompt adalah templat dengan argumen bernama. Tuan rumah menampilkannya sebagai prompt garis miring. Prompt `review_note` mungkin menggunakan argumen `note_id` dan menghasilkan template prompt multi-pesan yang diumpankan klien ke modelnya.

### Kehalusan transportasi Stdio

- JSON yang dibatasi baris baru. Tidak ada pembingkaian dengan awalan panjang.
- Jangan penyangga. `sys.stdout.flush()` setelah setiap penulisan.
- Klien mengontrol seumur hidup. Saat stdin ditutup (EOF), keluar dengan bersih.
- Jangan menangani SIGPIPE secara diam-diam; masuk dan keluar.

### Anotasi

Setiap alat dapat membawa `annotations` yang menjelaskan properti keselamatan:

- `readOnlyHint: true` — murni dibaca, aman untuk dicoba lagi.
- `destructiveHint: true` — efek samping yang tidak dapat diubah; klien harus mengkonfirmasi.
- `idempotentHint: true` — input yang sama menghasilkan output yang sama.
- `openWorldHint: true` — berinteraksi dengan sistem eksternal.

Klien menggunakan ini untuk memutuskan UX (dialog konfirmasi, indikator status) dan perutean (Fase 13 · 17).

### Jalur kelulusan

Server stdlib di `code/main.py` terdiri dari sekitar 180 baris. FastMCP (Python) menggabungkan logika yang sama ke gaya dekorator:

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK memiliki bentuk yang setara. Jalur kelulusan akan dibuka jika kamu sudah siap; konsepnya (kemampuan, pengiriman, blok konten) adalah sama.

## Pakai

`code/main.py` adalah server MCP catatan lengkap melalui stdio, hanya stdlib. Ini menangani `initialize`, `tools/list`, `tools/call` untuk tiga alat (`notes_list`, `notes_search`, `notes_create`), `resources/list` dan `resources/read` untuk setiap not, dan prompt `review_note`. kamu dapat mengendarainya dengan menyalurkan pesan JSON-RPC:

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

Apa yang harus dilihat:

- Petugas operator adalah `dict[str, Callable]` yang dikunci berdasarkan nama metode.
- Setiap pelaksana alat mengembalikan daftar blok konten, bukan string kosong.
- `isError: true` diatur ketika eksekutor melakukan raise.

## Kirim

Lesson ini menghasilkan `outputs/skill-mcp-server-scaffolder.md`. Dengan adanya domain (catatan, tiket, file, database), keterampilan ini merancang server MCP dengan pemisahan alat/sumber daya/prompt yang tepat dan jalur kelulusan SDK.

## Latihan

1. Jalankan `code/main.py` dan jalankan dengan pesan JSON-RPC buatan tangan. Latihan `notes_create`, lalu `resources/read` untuk mengambil catatan baru.

2. Tambahkan alat `notes_delete` dengan `annotations: {destructiveHint: true}`. Pastikan klien akan memunculkan dialog konfirmasi (ini memerlukan host asli; Claude Desktop berfungsi).

3. Terapkan `resources/subscribe` sehingga server mendorong `notifications/resources/updated` setiap kali catatan diubah. Tambahkan tugas keepalive.

4. Port server ke FastMCP. File Python harus menyusut hingga di bawah 80 baris. Perilaku kawat harus sama; verifikasi dengan rangkaian pengujian JSON-RPC yang sama.

5. Baca bagian `server/tools` spesifikasi dan identifikasi satu bidang definisi alat yang tidak diterapkan di server lesson ini. (Petunjuk: ada beberapa; pilih satu dan tambahkan.)

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Server MCP | "Hal yang memperlihatkan alat" | Proses yang menggunakan MCP JSON-RPC melalui stdio atau HTTP |
| transportasi stdio | "Model proses anak" | Server dihasilkan oleh klien; berkomunikasi melalui stdin/stdout |
| pengirim | "Metode router" | Peta nama metode JSON-RPC ke fungsi handler |
| Blok konten | "Potongan hasil alat" | Elemen yang diketik dalam larik `content` dari respons alat |
| `isError` | "Kegagalan tingkat alat" | Sinyal alat tersebut gagal; membedakan dari kesalahan JSON-RPC |
| Anotasi | "Petunjuk keselamatan" | bendera readOnly/destruktif/idempoten/openWorld |
| MCP Cepat | "SDK Python" | Kerangka kerja tingkat tinggi berbasis dekorator di atas protokol MCP |
| URI Sumber Daya | "Data yang dapat dialamatkan" | `file://`, `db://`, atau skema khusus yang mengidentifikasi sumber daya |
| Templat cepat | "Ringkasan prompt garis miring" | Templat yang disediakan server dengan slot argumen untuk UI host |
| Deklarasi kemampuan | "Feature beralih" | Bendera per-primitif dideklarasikan di `initialize` |

## Bacaan Lanjutan

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — referensi implementasi Python
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — implementasi TS paralel
- [FastMCP — kerangka server](https://gofastmcp.com/) — API Python bergaya dekorator untuk server MCP
- [MCP — Panduan memulai cepat server](https://modelcontextprotocol.io/quickstart/server) — tutorial menyeluruh menggunakan SDK
- [MCP — Spesifikasi alat server](https://modelcontextprotocol.io/spesifikasi/2025-11-25/server/tools) — referensi lengkap untuk pesan alat/*
