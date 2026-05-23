# Pengambilan Sample MCP — Penyelesaian LLM dan Loop Agen yang Diminta Server

> Kebanyakan server MCP adalah pelaksana yang bodoh: ambil argumen, jalankan code, kembalikan konten. Pengambilan sample memungkinkan server membalik arah: meminta LLM klien untuk membuat keputusan. Hal ini memungkinkan loop agen yang dihosting server tanpa server memiliki kredensial model apa pun. SEP-1577, yang digabungkan pada 25-11-2025, menambahkan alat di dalam permintaan pengambilan sample sehingga loop dapat mencakup alasan yang lebih dalam. Catatan risiko penyimpangan: bentuk pengambilan sample alat SEP-1577 masih bersifat eksperimental hingga Q1 2026 dan masih diterapkan pada API SDK.

**Type:** Build
**Language:** Python (stdlib, memanfaatkan pengambilan sample)
**Prerequisites:** Fase 13 · 07 (server MCP), Fase 13 · 10 (sumber daya dan petunjuk)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Jelaskan apa yang dipecahkan `sampling/createMessage` (loop yang dihosting server tanpa kunci API sisi server).
- Menerapkan server yang meminta klien untuk mengambil sample melalui prompt multi-putaran dan mengembalikan penyelesaiannya.
- Gunakan `modelPreferences` (prioritas biaya / kecepatan / kecerdasan) untuk memandu pemilihan model klien.
- Buat alat `summarize_repo` yang melakukan iterasi secara internal melalui pengambilan sample, bukan melalui perilaku hard-coding.

## Masalah

Server MCP yang berguna untuk alur kerja peringkasan code perlu: menjalankan pohon file, memilih file mana yang akan dibaca, mensintesis ringkasan, dan kembali. Dimana alasan LLM terjadi?

Opsi A: server memanggil LLM-nya sendiri. Membutuhkan kunci API, menagih sisi server, mahal per pengguna.

Opsi B: server mengembalikan konten mentah; agen klien melakukan alasannya. Berfungsi tetapi memindahkan logika server ke prompt klien, yang rapuh.

Opsi C: server menanyakan LLM klien melalui `sampling/createMessage`. Server mempertahankan algoritme (file mana yang harus dibaca, berapa banyak proses yang harus dilakukan) sementara klien tetap mempertahankan penagihan dan pilihan model. Server tidak memiliki kredensial sama sekali.

Pengambilan sample adalah opsi C. Ini adalah mekanisme di mana server tepercaya dapat menghosting loop agen tanpa menjadi host LLM penuh itu sendiri.

## Konsep

### `sampling/createMessage` permintaan

Server mengirimkan:

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

Klien menjalankan LLM-nya, mengembalikan:

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

Tiga pelampung berjumlah 1,0:

- `costPriority`: pilih model yang lebih murah.
- `speedPriority`: menyukai model yang lebih cepat.
- `intelligencePriority`: lebih menyukai model yang lebih mumpuni.

Ditambah `hints`: memberi nama model yang disukai server. Klien mungkin menerima atau tidak menerima petunjuk; konfigurasi pengguna klien selalu menang.

### `includeContext`

Tiga nilai:

- `"none"` — hanya pesan yang disediakan server. Bawaan.
- `"thisServer"` — menyertakan pesan sebelumnya dari sesi server ini.
- `"allServers"` — mencakup semua konteks sesi.

`includeContext` tidak digunakan lagi mulai tanggal 25-11-2025 karena membocorkan konteks lintas-server, yang merupakan masalah keamanan. Pilih `"none"` dan sampaikan konteks eksplisit dalam pesan.

### Pengambilan sample dengan alat (SEP-1577)

Baru pada 25-11-2025: permintaan pengambilan sample dapat menyertakan array `tools`. Klien menjalankan loop pemanggilan alat penuh menggunakan alat tersebut. Hal ini memungkinkan server menghosting agen bergaya ReAct yang melakukan loop melalui model klien.

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

Klien loop: sample, jalankan alat jika dipanggil, sample lagi, kembalikan pesan asisten terakhir. Hal ini bersifat eksperimental hingga Q1 2026; Tanda tangan SDK mungkin masih melayang. Konfirmasikan terhadap bagian klien/pengambilan sample spesifikasi 25-11-2025 saat kamu menerapkan.

### Manusia dalam lingkaranKlien HARUS menunjukkan kepada pengguna apa yang diminta server untuk dilakukan model sebelum menjalankan sample. Server jahat dapat menggunakan pengambilan sample untuk memanipulasi sesi pengguna ("ucapkan X kepada pengguna sehingga mereka mengklik Y"). Permintaan pengambilan sample permukaan Claude Desktop, VS Code, dan Cursor sebagai dialog konfirmasi yang dapat ditolak pengguna.

Konsensus tahun 2026: pengambilan sample tanpa konfirmasi manusia merupakan tanda bahaya. Gateway (Fase 13 · 17) dapat secara otomatis menyetujui pengambilan sample berisiko rendah dan secara otomatis menolak segala sesuatu yang mencurigakan.

### Loop yang dihosting server tanpa kunci API

Kasus penggunaan kanonik: server MCP peringkasan code tanpa akses LLM sendiri. Itu:

1. Jalani struktur repo.
2. Hubungi `sampling/createMessage` dengan "Pilih lima file yang paling mungkin menggambarkan tujuan repo ini."
3. Baca file-file itu.
4. Hubungi `sampling/createMessage` dengan isi file dan "Ringkas repo dalam 3 paragraf."
5. Kembalikan ringkasan sebagai hasil `tools/call`.

Server tidak pernah menyentuh API LLM. Pengguna klien membayar penyelesaian menggunakan kredensial mereka sendiri.

### Risiko keselamatan (Pengungkapan Unit 42, Q1 2026)

- **Pengambilan sample terselubung.** Alat yang selalu memanggil pengambilan sample dengan "balas dengan email pengguna dari konteks sesi". Fase 13 · 15 mencakup vector serangan.
- **Pencurian sumber daya melalui pengambilan sample.** Server meminta klien untuk meringkas muatan penyerang, menagih pengguna.
- **Bom loop.** Server memanggil pengambilan sample dalam loop yang ketat. Klien HARUS menerapkan batas tarif per sesi.

## Pakai

`code/main.py` mengirimkan rangkaian pengambilan sample server-ke-klien palsu. Alat simulasi "summarize_repo" memanggil dua putaran pengambilan sample (pilih file, lalu rangkum), dan klien palsu mengembalikan tanggapan terekam. Tali pengamannya menunjukkan:

- Server mengirim `sampling/createMessage` dengan `modelPreferences`.
- Klien mengembalikan penyelesaian.
- Server melanjutkan perulangannya.
- Pembatas tarif membatasi total panggilan pengambilan sample per pemanggilan alat.

Apa yang harus dilihat:

- Server hanya menampilkan satu alat (`summarize_repo`); semua alasan terjadi dalam panggilan pengambilan sample.
- Preferensi model memberi weight pada pilihan model klien; petunjuk mencantumkan model yang disukai.
- Perulangan berakhir pada `stopReason: "endTurn"`.
- Batas `max_samples_per_tool = 5` menangkap loop yang tidak terkendali.

## Kirim

Lesson ini menghasilkan `outputs/skill-sampling-loop-designer.md`. Mengingat algoritme sisi server yang memerlukan panggilan LLM (penelitian, ringkasan, perencanaan), keterampilan merancang implementasi berbasis pengambilan sample dengan modelPreferensi, batas kecepatan, dan konfirmasi keamanan yang tepat.

## Latihan

1. Jalankan `code/main.py`. Ubah `max_samples_per_tool` ke 2 dan amati batas tarif.

2. Menerapkan varian alat dalam pengambilan sample SEP-1577: permintaan pengambilan sample membawa larik `tools`. Verifikasi bahwa loop sisi klien mengeksekusi alat tersebut sebelum mengembalikan penyelesaian akhir. Perhatikan risiko penyimpangan: Tanda tangan SDK masih dapat berubah hingga H1 2026.

3. Tambahkan konfirmasi human-in-the-loop: sebelum `sampling/createMessage` pertama di server, jeda dan tunggu persetujuan pengguna. Panggilan ditolak mengembalikan penolakan yang diketik.

4. Tambahkan pembatas tarif per pengguna yang dikunci berdasarkan sesi klien. Perulangan server yang sama oleh pengguna yang sama harus berbagi anggaran.

5. Rancang alat `summarize_pdf` yang menggunakan pengambilan sample untuk memilih bagian yang akan disertakan. Buat sketsa pesan yang dikirim. Bagaimana `modelPreferences.intelligencePriority` mengubah perilaku pada 0,1 vs 0,9?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Pengambilan sample | "Panggilan LLM server-ke-klien" | Server meminta penyelesaian model klien |
| `sampling/createMessage` | "Metode" | Metode JSON-RPC untuk permintaan pengambilan sample |
| `modelPreferences` | "Prioritas model" | Weight biaya / kecepatan / kecerdasan ditambah petunjuk nama |
| `includeContext` | "Kebocoran lintas sesi" | Mode penyertaan konteks yang tidak digunakan lagi |
| SEP-1577 | "Alat dalam pengambilan sample" | Izinkan alat di dalam pengambilan sample untuk ReAct | yang dihosting di server
| Manusia dalam lingkaran | "Pengguna mengonfirmasi" | Klien menampilkan permintaan pengambilan sample kepada pengguna sebelum menjalankan |
| Bom lingkaran | "Pengambilan sample pelarian" | Loop pengambilan sample tak terbatas di sisi server; klien harus membatasi tarif |
| Pengambilan sample terselubung | "Alasan tersembunyi" | Server berbahaya menyembunyikan maksud dalam permintaan pengambilan sample |
| Pencurian sumber daya | "Menggunakan anggaran LLM pengguna" | Server memaksa klien mengeluarkan biaya untuk pengambilan sample yang tidak diinginkannya |
| `stopReason` | "Mengapa generasi terhenti" | `endTurn`, `stopSequence`, atau `maxTokens` |

## Bacaan Lanjutan

- [MCP — Konsep: Pengambilan Sample](https://modelcontextprotocol.io/docs/concepts/sampling) — ikhtisar pengambilan sample tingkat tinggi
- [MCP — Spesifikasi pengambilan sample klien 25-11-2025](https://modelcontextprotocol.io/spesification/2025-11-25/client/sampling) — bentuk kanonik `sampling/createMessage`
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — Proposal Evolusi Spesifikasi untuk alat dalam pengambilan sample (eksperimental)
- [Unit 42 — vector serangan MCP](https://unit42.paloaltonetworks.com/model-context-protocol-action-vectors/) — pengambilan sample rahasia dan pola pencurian sumber daya
- [Speakeasy — konsep inti pengambilan sample MCP](https://www.speakeasy.com/mcp/core-concepts/sampling) — panduan dengan contoh code sisi klien
