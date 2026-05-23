# Roots dan Elisitasi — Input Pengguna Pelingkupan dan Di Tengah Penerbangan

> Jalur yang dikodekan secara keras akan rusak saat pengguna membuka proyek lain. Argumen alat yang telah diisi sebelumnya rusak ketika pengguna kurang menentukan. Roots mencakup server ke kumpulan URI yang dikontrol pengguna; elisitasi menjeda panggilan di tengah-tengah panggilan alat untuk meminta input terstruktur dari pengguna melalui formulir atau URL. Dua primitif klien, dua perbaikan untuk mode kegagalan MCP umum. SEP-1036 (elisitasi mode URL, 25-11-2025) masih bersifat eksperimental hingga H1 2026 — periksa versi SDK sebelum bergantung padanya.

**Type:** Build
**Language:** Python (stdlib, root + demo elisitasi)
**Prerequisites:** Fase 13 · 07 (server MCP)
**Waktu:** ~45 menit

## Tujuan Pembelajaran

- Deklarasikan `roots` dan tanggapi `notifications/roots/list_changed`.
- Batasi operasi file server ke URI di dalam set root yang dideklarasikan.
- Gunakan `elicitation/create` untuk meminta konfirmasi atau input terstruktur dari pengguna di tengah panggilan alat.
- Pilih antara mode formulir dan mode URL (yang terakhir bersifat eksperimental; risiko penyimpangan dicatat).

## Masalah

Dua kegagalan nyata yang terjadi pada server MCP dalam produksi.

**Asumsi jalur rusak.** Server ditulis dengan `~/notes`. Seorang pengguna di mesin lain dengan catatan di `~/Documents/Notes` mendapat panggilan alat yang gagal secara diam-diam (tidak ada file yang ditemukan) atau lebih buruk lagi, menulis ke tempat yang salah.

**Argumen yang mungkin diketahui pengguna tidak ada.** Pengguna meminta "hapus catatan laporan TPS lama". Model memanggil `notes_delete(title: "TPS report")` tetapi ada tiga catatan yang cocok dari tahun 2023, 2024, dan 2025. Alat ini tidak dapat menebak. Gagal dengan "ambigu" itu menjengkelkan; menjalankan ketiganya adalah bencana.

Roots memperbaiki yang pertama: klien mendeklarasikan di `initialize` kumpulan URI yang mungkin disentuh server. Elisitasi memperbaiki yang kedua: server menjeda panggilan alat dan mengirimkan `elicitation/create` untuk meminta pengguna memilih yang mana.

## Konsep

### Akar

Klien mendeklarasikan daftar root di `initialize`:

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

Server kemudian dapat menghubungi `roots/list`:

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

Server HARUS memperlakukan akar sebagai batas: file apa pun yang dibaca atau ditulis di luar kumpulan akar akan ditolak. Hal ini tidak diterapkan oleh klien (server masih berupa code yang dipercaya pengguna), namun server yang sesuai spesifikasi menghormatinya.

Saat pengguna menambah atau menghapus root, klien mengirimkan `notifications/roots/list_changed`. Server memanggil ulang `roots/list` dan memperbarui batasnya.

### Mengapa root merupakan klien primitif

Root dideklarasikan oleh klien karena mewakili model persetujuan pengguna. Pengguna mengatakan kepada Claude Desktop "berikan server catatan ini akses ke dua direktori ini". Server tidak dapat memperluas cakupan tersebut.

### Elisitasi: default mode formulir

`elicitation/create` mengambil skema formulir ditambah prompt bahasa alami:

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

Klien merender formulir, mengumpulkan jawaban pengguna, mengembalikan:

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

Tiga tindakan yang mungkin dilakukan: `accept` (pengguna mengisinya), `decline` (pengguna menutupnya), `cancel` (pengguna membatalkan seluruh panggilan alat).

Skema formulir datar — objek bersarang tidak didukung di v1. SDK biasanya menolak sesuatu yang lebih kompleks daripada satu layer.

### Elisitasi: mode URL (SEP-1036, eksperimental)

Baru pada 25-11-2025. Alih-alih skema, server mengirimkan URL:

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```Klien membuka URL di browser, menunggu selesai, kembali ketika pengguna kembali. Berguna untuk alur OAuth, otorisasi pembayaran, dan penandatanganan dokumen jika formulir tidak mencukupi.

Catatan risiko penyimpangan: bentuk respons SEP-1036 masih belum berubah; beberapa SDK mengembalikan URL panggilan balik, yang lain mengembalikan token penyelesaian. Baca catatan rilis SDK kamu sebelum menggunakan mode URL dalam produksi.

### Saat elisitasi adalah alat yang tepat

- Konfirmasi pengguna sebelum tindakan destruktif (petunjuk destruktif + elisitasi).
- Disambiguasi (pilih salah satu dari N kecocokan).
- Penyiapan yang dijalankan pertama kali (kunci API, direktori, preferensi).
- Aliran gaya OAuth (mode URL).

### Ketika elisitasi salah

- Mengisi argumen yang diperlukan alat yang bisa diminta oleh model dalam bentuk prosa. Gunakan prompt ulang yang normal, bukan dialog elisitasi.
- Panggilan frekuensi tinggi. Elisitasi menyela pembicaraan; jangan menembakkannya di dalam satu lingkaran.
- Apa pun yang dapat divalidasi oleh server setelah kejadian tersebut. Validasi, kembalikan kesalahan, biarkan model bertanya kepada pengguna melalui teks.

### Jembatan manusia dalam lingkaran

Elisitasi ditambah pengambilan sample bersama-sama memungkinkan model "human-in-the-loop" MCP. Loop agen server dapat dijeda untuk input pengguna (elisitasi) atau penalaran model (sampling). Fase 13 · 11 pengambilan sample tercakup; lesson ini mencakup elisitasi. Satukan keduanya untuk kontrol putaran tengah penuh.

## Pakai

`code/main.py` memperluas server catatan dengan:

- `roots/list` respons yang ditanyakan ulang oleh server setelah pemberitahuan perubahan daftar root.
- Alat `notes_delete` yang menggunakan `elicitation/create` untuk membedakan saat beberapa nada cocok.
- Alat `notes_setup` yang menggunakan elisitasi mode URL untuk membuka halaman konfigurasi yang pertama kali dijalankan (disimulasikan).
- Pemeriksaan batas yang menolak operasi pada URI di luar akar yang dinyatakan.

Demo ini menjalankan tiga skenario: jalur bahagia (satu pertandingan), disambiguasi (tiga pertandingan, kebakaran elisitasi), penulisan di luar akar (ditolak).

## Kirim

Lesson ini menghasilkan `outputs/skill-elicitation-form-designer.md`. Mengingat alat yang mungkin memerlukan konfirmasi atau disambiguasi pengguna, keterampilan merancang skema formulir elisitasi dan templat pesan.

## Latihan

1. Jalankan `code/main.py`. Memicu jalur disambiguasi; konfirmasikan jawaban pengguna yang disimulasikan dialihkan kembali ke alat.

2. Tambahkan alat baru `notes_archive` yang memerlukan konfirmasi elisitasi setiap saat (petunjuk merusak). Periksa UX: bagaimana perbandingannya dengan model yang menanyakan ulang dalam teks?

3. Menerapkan perolehan mode URL untuk alur OAuth yang dijalankan pertama kali. Catat risiko penyimpangan dan tambahkan pelindung versi SDK.

4. Perluas penanganan `roots/list`: ketika pemberitahuan tiba, server harus membaca ulang secara atom dan memindai ulang pegangan file terbuka yang mungkin sekarang berada di luar cakupan.

5. Baca thread diskusi terbitan SEP-1036 di GitHub. Identifikasi satu pertanyaan terbuka yang memengaruhi cara server menangani callback mode URL.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Akar | "Batas persetujuan" | URI klien telah mengizinkan server untuk menyentuh |
| `roots/list` | "Server meminta cakupan" | Klien mengembalikan set root saat ini |
| `notifications/roots/list_changed` | "Pengguna mengubah cakupan" | Klien memberi sinyal bahwa set root telah bermutasi |
| Elisitasi | "Tanya pengguna di tengah panggilan" | Permintaan yang dimulai oleh server untuk input pengguna terstruktur |
| `elicitation/create` | "Metode" | Metode JSON-RPC untuk permintaan elisitasi |
| Modus formulir | "Bentuk berbasis skema" | Skema JSON Datar dirender sebagai formulir di UI klien |
| Modus URL | "Pengalihan peramban" | SEP-1036 eksperimental; membuka URL dan menunggu |
| `accept` / `decline` / `cancel` | "Hasil tanggapan pengguna" | Tiga cabang yang ditangani server |
| Disambiguasi | "Pilih satu" | Kasus penggunaan elisitasi umum ketika suatu alat memiliki N kandidat |
| Bentuk datar | "Hanya properti tingkat atas" | Skema elisitasi tidak dapat disarangkan |

## Bacaan Lanjutan

- [MCP — Spesifikasi akar klien](https://modelcontextprotocol.io/spesifikasi/draft/client/roots) — referensi akar kanonik
- [MCP — Spesifikasi elisitasi klien](https://modelcontextprotocol.io/spesification/draft/client/elicite) — referensi elisitasi kanonik
- [Cisco — Apa yang baru dalam elisitasi MCP, konten terstruktur, penyempurnaan OAuth](https://blogs.cisco.com/developer/whats-new-in-mcp-elicite-structured-content-and-oauth-enhancements) — panduan penambahan 25-11-2025
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — Proposal perolehan mode URL (eksperimental, risiko penyimpangan)
- [The New Stack — Bagaimana elisitasi menghadirkan human-in-the-loop ke alat AI](https://thenewstack.io/how-eliitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — panduan UX
