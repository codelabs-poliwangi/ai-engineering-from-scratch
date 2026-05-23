# A2A — Protokol Agen-ke-Agen

> MCP adalah agen-ke-alat. A2A (Agent2Agent) adalah agen-ke-agen — sebuah protokol terbuka yang memungkinkan agen buram yang dibangun pada framework berbeda berkolaborasi. Dirilis oleh Google pada bulan April 2025, didonasikan ke Linux Foundation pada bulan Juni 2025, mencapai v1.0 pada bulan April 2026 dengan 150+ pendukung termasuk AWS, Cisco, Microsoft, Salesforce, SAP, dan ServiceNow. Ini menyerap ACP IBM dan menambahkan ekstensi pembayaran AP2. Lesson ini membahas Kartu Agen, siklus hidup Tugas, dan dua pengikatan transportasi.

**Type:** Build
**Language:** Python (stdlib, Kartu Agen + Task harness)
**Prerequisites:** Fase 13 · 06 (dasar-dasar MCP), Fase 13 · 08 (klien MCP)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Membedakan kasus penggunaan agen-ke-alat (MCP) dan agen-ke-agen (A2A).
- Publikasikan Kartu Agen di `/.well-known/agent.json` dengan keterampilan dan metadata titik akhir.
- Jalani siklus hidup Tugas (dikirim → berfungsi → diperlukan input → selesai / gagal / dibatalkan / ditolak).
- Gunakan Pesan dengan Bagian (teks, file, data) dan Artefak sebagai output.

## Masalah

Agen layanan pelanggan perlu mendelegasikan penulisan laporan kepada agen penulis khusus. Opsi pra-A2A:

- API REST Kustom. Berfungsi tetapi setiap pemasangan hanya dilakukan sekali saja.
- Basis code bersama. Mengharuskan kedua agen untuk menjalankan framework yang sama.
- PKS. Tidak cocok: MCP ditujukan untuk alat panggilan, bukan untuk dua agen yang berkolaborasi sambil mempertahankan alasan internal masing-masing agen yang tidak jelas.

A2A mengisi kekosongan tersebut. Ini memodelkan interaksi ketika satu agen mengirimkan Tugas ke agen lain, dengan siklus hidup, pesan, dan artefak. Status internal agen yang dipanggil tetap buram — pemanggil hanya melihat transisi status tugas dan output akhirnya.

A2A adalah protokol "biarkan agen di seluruh framework berbicara satu sama lain". Itu tidak menggantikan MCP; keduanya saling melengkapi.

## Konsep

### Kartu Agen

Setiap agen yang mematuhi A2A menerbitkan kartu di `/.well-known/agent.json`:

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

Penemuan berbasis URL: ambil kartu, pelajari URL titik akhir A2A, sebutkan keterampilan.

### Kartu Agen yang Ditandatangani (AP2)

Ekstensi AP2 (September 2025) menambahkan tanda tangan kriptografi ke Kartu Agen. Penerbit menandatangani kartunya sendiri dengan JWT; konsumen memverifikasi. Mencegah peniruan identitas.

### Siklus hidup tugas

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

Klien memulai dengan `tasks/send`. Agen yang dipanggil melakukan transisi melalui negara bagian; klien berlangganan pembaruan negara melalui SSE atau jajak pendapat.

### Pesan dan Bagian

Sebuah pesan membawa satu atau lebih Bagian:

- `text` — konten biasa.
- `file` — gumpalan base64 dengan mimeType.
- `data` — mengetik payload JSON (input terstruktur untuk agen yang dipanggil).

Contoh:

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artefak

Outputnya adalah Artefak, bukan string mentah. Artefak adalah output yang diberi nama dan diketik:

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artefak dapat dialirkan sebagai potongan. Penelepon terakumulasi.

### Dua ikatan transportasi

1. **JSON-RPC melalui HTTP.** titik akhir `/a2a`, POST untuk permintaan, SSE opsional untuk streaming. Pengikatan bawaan.
2. **gRPC.** Untuk lingkungan perusahaan yang menggunakan gRPC asli.

Kedua pengikatan membawa bentuk pesan logis yang sama.

### Pelestarian opasitas

Prinsip desain utama: keadaan internal agen yang dipanggil tidak jelas. Penelepon melihat status tugas dan artefak. Rantai pemikiran agen yang dipanggil, panggilan alatnya, delegasi sub-agennya — semuanya tidak terlihat. Ini berbeda dengan MCP yang pemanggilannya bersifat transparan.Dasar Pemikiran: A2A memungkinkan pesaing untuk berkolaborasi tanpa mengungkapkan internalnya. A2A dapat "menelepon agen layanan pelanggan ini" tanpa penelepon mengetahui bagaimana agen tersebut mengimplementasikan layanan tersebut.

### Garis Waktu

- **09-04-2025.** Google mengumumkan A2A.
- **23-06-2025.** Disumbangkan ke Linux Foundation.
- **2025-08.** Menyerap ACP IBM.
- **2025-09.** Perpanjangan AP2 (Pembayaran Agen) dikirimkan.
- **2026-04.** v1.0 dirilis dengan 150+ organisasi pendukung.

### Hubungan dengan MCP

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Kasus penggunaan | Agen-ke-alat | Agen-ke-agen |
| Opasitas | Panggilan alat transparan | Penalaran batin yang buram |
| Penelepon biasa | Waktu proses agen | Agen lain |
| Negara | Hasil panggilan alat | Tugas dengan siklus hidup |
| Otorisasi | OAuth 2.1 (Fase 13 · 16) | Kartu Agen bertanda tangan JWT (AP2) |
| Transportasi | Stdio / HTTP yang Dapat Dialirkan | JSON-RPC melalui HTTP/gRPC |

Gunakan MCP saat kamu ingin memanggil alat tertentu. Gunakan A2A ketika kamu ingin mendelegasikan seluruh tugas ke agen lain. Banyak sistem produksi menggunakan keduanya: agen menggunakan MCP untuk layer alatnya dan A2A untuk layer kolaborasinya.

## Pakai

`code/main.py` menerapkan pemanfaatan A2A minimal: agen penelitian menerbitkan kartunya, agen penulis menerima `tasks/send` dengan bagian-bagian termasuk PDF dan instruksi teks, transisi melalui pengerjaan → input_wajib → berfungsi → selesai, dan mengembalikan artefak teks. Semua stdlib; menggunakan transportasi dalam memori untuk fokus pada bentuk pesan.

Apa yang harus dilihat:

- Bentuk JSON Kartu Agen.
- Penugasan id tugas dan transisi status.
- Pesan dengan bagian tipe campuran.
- Cabang yang membutuhkan input di tengah tugas.
- Pengembalian artefak setelah selesai.

## Kirim

Lesson ini menghasilkan `outputs/skill-a2a-agent-spec.md`. Mengingat agen baru yang harus dapat dipanggil oleh agen lain, keterampilan menghasilkan JSON Kartu Agen, skema keterampilan, dan cetak biru titik akhir.

## Latihan

1. Jalankan `code/main.py`. Telusuri seluruh siklus hidup Tugas, termasuk jeda yang memerlukan input saat agen yang dipanggil meminta klarifikasi.

2. Tambahkan Kartu Agen yang ditandatangani. Masuk dengan HMAC melalui JSON kanonik kartu. Tulis pemverifikasi dan konfirmasikan kegagalannya pada kartu yang bermutasi.

3. Menerapkan streaming tugas: agen penulis memancarkan tiga potongan artefak tambahan melalui SSE dan pemanggil mengumpulkannya.

4. Rancang agen A2A yang membungkus server MCP. Petakan setiap alat MCP ke keterampilan A2A. Perhatikan trade-offnya — opacity apa yang hilang?

5. Baca pengumuman A2A v1.0 dan identifikasi satu feature yang belum diterapkan oleh framework apa pun pada April 2026. (Petunjuk: ini berkaitan dengan delegasi tugas multi-hop.)

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| A2A | "Protokol Agen-ke-Agen" | Protokol terbuka untuk kolaborasi agen buram |
| Kartu Agen | "`.well-known/agent.json`" | Metadata yang dipublikasikan yang menjelaskan keahlian dan titik akhir agen |
| Keterampilan | "Unit yang dapat dipanggil" | Operasi bernama yang didukung agen (analog dengan alat MCP) |
| Tugas | "Unit delegasi" | Item kerja dengan siklus hidup dan artefak akhir |
| Pesan | "Input tugas" | Membawa Bagian (teks, file, data) |
| Bagian | "Potongan yang diketik" | `text` / `file` / `data` elemen pesan |
| Artefak | "Output tugas" | Dinamakan, output yang diketik dikembalikan setelah selesai |
| AP2 | "Protokol Pembayaran Agen" | Perpanjangan Kartu Agen yang ditandatangani untuk kepercayaan dan pembayaran |
| Opasitas | "Kolaborasi kotak hitam" | Internal agen yang dipanggil disembunyikan dari penelepon |
| Diperlukan input | "Jeda tugas" | Status siklus hidup ketika agen memerlukan info lebih lanjut |

## Bacaan Lanjutan

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — spesifikasi A2A kanonik
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — implementasi referensi dan SDK
- [Linux Foundation — siaran pers peluncuran A2A](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — transfer tata kelola pada bulan Juni 2025
- [Google Cloud — peningkatan protokol A2A](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — peta jalan dan momentum partner
- [Google Dev — pencapaian A2A 1.0](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — catatan rilis v1.0 dan panduan kompatibilitas mundur
