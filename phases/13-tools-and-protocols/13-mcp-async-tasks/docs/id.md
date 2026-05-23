# Tugas Async (SEP-1686) — Telepon Sekarang, Ambil Nanti untuk Pekerjaan Jangka Panjang

> Pekerjaan agen nyata membutuhkan waktu beberapa menit hingga beberapa jam: pengoperasian CI, sintesis penelitian mendalam, ekspor batch. Alat sinkron memanggil koneksi terputus, waktu habis, atau memblokir UI. SEP-1686, digabungkan pada 25-11-2025, menambahkan tugas primitif: permintaan apa pun dapat ditambah menjadi tugas, dan hasilnya dapat diambil nanti atau dialirkan melalui pemberitahuan negara. Catatan risiko penyimpangan: Tugas bersifat eksperimental hingga H1 2026; Permukaan SDK masih dirancang berdasarkan spesifikasi.

**Type:** Build
**Language:** Python (stdlib, mesin status tugas asinkron)
**Prerequisites:** Fase 13 · 07 (server MCP), Fase 13 · 09 (transportasi)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Identifikasi kapan harus mempromosikan alat dari sinkron ke penambahan tugas (>30 detik kerja sisi server).
- Jalani siklus hidup tugas: `working` → `input_required` → `completed` / `failed` / `cancelled`.
- Pertahankan status tugas sehingga crash tidak kehilangan pekerjaan dalam penerbangan.
- Jajak pendapat `tasks/status` dan ambil `tasks/result` dengan benar.

## Masalah

Alat `generate_report` menjalankan pipeline ekstraksi multi-menit. Opsi pada model sinkron:

1. Biarkan sambungan tetap terbuka selama tiga menit. Transportasi distance jauh menjatuhkannya; waktu istirahat klien; UI membeku.
2. Segera kembalikan dengan placeholder; mengharuskan klien untuk melakukan polling titik akhir kustom. Melanggar keseragaman MCP.
3. Api-dan-lupakan; tidak ada hasil.

Tidak ada yang bagus. SEP-1686 menambahkan yang keempat: penambahan tugas. Permintaan apa pun (biasanya `tools/call`) dapat ditandai sebagai tugas. Server segera mengembalikan id tugas. Klien melakukan polling `tasks/status` dan mengambil `tasks/result` setelah selesai. Status sisi server bertahan saat dimulai ulang.

## Konsep

### Penambahan tugas

Permintaan menjadi tugas dengan menyetel `params._meta.task.required: true` (atau `optional: true`, server memutuskan). Server segera merespons dengan:

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` adalah janji server untuk mempertahankan status; setelah ttl hasil tugas dibuang.

### Keikutsertaan per alat

Anotasi alat dapat mendeklarasikan dukungan tugas:

- `taskSupport: "forbidden"` — alat ini selalu berjalan secara sinkron. Aman untuk alat yang cepat.
- `taskSupport: "optional"` — klien dapat meminta penambahan tugas.
- `taskSupport: "required"` — klien HARUS menggunakan augmentasi tugas.

Alat `generate_report` akan menjadi `required`. Alat `notes_search` akan menjadi `forbidden`.

### Negara bagian

```
working  -> input_required -> working  (loop via elicitation)
working  -> completed
working  -> failed
working  -> cancelled
```

Mesin status hanya dapat ditambahkan: sekali `completed`, `failed`, atau `cancelled`, tugasnya bersifat terminal.

### Metode

- `tasks/status {taskId}` — mengembalikan status saat ini dan petunjuk kemajuan.
- `tasks/result {taskId}` — memblokir atau mengembalikan 404 jika belum selesai.
- `tasks/cancel {taskId}` — idempoten; status terminal abaikan.
- `tasks/list` — opsional; menghitung tugas aktif dan baru selesai.

### Status streaming berubah

Jika server mendukungnya, klien dapat berlangganan pemberitahuan status:

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

Klien yang melakukan streaming daripada polling mendapatkan UX yang lebih baik. Polling selalu didukung sebagai permukaan minimal.

### Keadaan tahan lama

Spesifikasi ini memerlukan server yang mendeklarasikan dukungan tugas untuk mempertahankan status. Kecelakaan tidak boleh kehilangan hasil yang telah diselesaikan dalam waktu ttl. Penyimpanan berkisar dari SQLite hingga Redis hingga sistem file. Harness Lesson 13 menggunakan sistem file.

### Pembatalan semantik`tasks/cancel` adalah idempoten. Jika tugas sedang dijalankan, server akan mencoba berhenti (periksa pembatalan kooperatif pelaksana). Jika sudah terminal, permintaannya tidak boleh dilakukan.

### Pemulihan kerusakan

Saat proses server dimulai ulang:

1. Muat semua status tugas yang masih ada.
2. Tandai setiap tugas `working` yang prosesnya mati sebagai `failed` dengan kesalahan `CRASH_RECOVERY`.
3. Pertahankan `completed` / `failed` / `cancelled` untuk ttl mereka.

### Tugas asinkron ditambah pengambilan sample

Sebuah tugas dapat memanggil `sampling/createMessage` sendiri. Beginilah cara kerja tugas penelitian yang berjalan lama: thread tugas server mengambil sample model klien sesuai kebutuhan, sementara UI klien menampilkan tugas sebagai `working` dengan pembaruan kemajuan berkala.

### Mengapa ini bersifat eksperimental

SEP-1686 dikirimkan pada 25-11-2025 tetapi peta jalan yang lebih luas menyebutkan tiga masalah terbuka: primitif langganan yang tahan lama, subtugas (hubungan tugas orang tua-anak), dan standardisasi hasil-TTL. Spesifikasi ini diperkirakan akan berkembang hingga tahun 2026. Code produksi harus memperlakukan Tasks sebagai tugas yang stabil hanya untuk kasus umum dan melindungi terhadap perubahan SDK di masa mendatang untuk subtugas.

## Pakai

`code/main.py` mengimplementasikan penyimpanan tugas yang tahan lama (didukung sistem file) dan alat `generate_report` yang berjalan di thread latar belakang. Klien memanggil alat tersebut, segera mendapatkan id tugas, melakukan polling `tasks/status` saat pekerja memperbarui kemajuan, dan mengambil `tasks/result` setelah selesai. Pembatalan berfungsi; pemulihan kerusakan disimulasikan dengan mematikan thread pekerja dan memuat ulang status.

Apa yang harus dilihat:

- Status tugas JSON dipertahankan ke `/tmp/lesson-13-tasks/<id>.json`.
- Pembaruan thread pekerja bidang `progress`; jajak pendapat menunjukkan kemajuannya.
- Pembatalan dari sisi klien menetapkan suatu acara; pekerja memeriksa dan keluar lebih awal.
- Status isi ulang saat "crash" menandai tugas dalam penerbangan sebagai `failed` dengan `CRASH_RECOVERY`.

## Kirim

Lesson ini menghasilkan `outputs/skill-task-store-designer.md`. Mengingat alat yang sudah berjalan lama (meneliti, membangun, mengekspor), keterampilan merancang penyimpanan tugas (bentuk status, ttl, daya tahan), memilih tanda TaskSupport yang tepat, dan membuat sketsa pemberitahuan kemajuan.

## Latihan

1. Jalankan `code/main.py`. Mulailah tugas `generate_report`, status polling, lalu ambil hasilnya.

2. Tambahkan panggilan `tasks/cancel` di tengah proses. Verifikasikan bahwa pekerja menghormatinya dan negara bagian menjadi `cancelled`.

3. Simulasikan pemulihan kerusakan: matikan thread pekerja, mulai ulang pemuat, dan amati mode kegagalan `CRASH_RECOVERY`.

4. Perluas penyimpanan ke SQLite. Kemenangan dalam ketahanan juga sama; opsi kueri terbuka (daftar semua tugas dari sesi X).

5. Baca postingan peta jalan MCP untuk tahun 2026. Identifikasi masalah terbuka terkait Tugas yang paling mungkin memengaruhi desain SDK API di tahun depan.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Tugas | "Panggilan alat yang sudah berjalan lama" | Permintaan ditambah dengan `_meta.task` untuk eksekusi asinkron |
| SEP-1686 | "Spesifikasi tugas" | Proposal Evolusi Spesifikasi yang menambahkan Tugas pada 25-11-2025 |
| `_meta.task` | "Amplop tugas" | Metadata per permintaan yang berisi id, negara bagian, ttl |
| tugasDukungan | "Bendera alat" | `forbidden` / `optional` / `required` per alat |
| `tasks/status` | "Metode jajak pendapat" | Ambil status saat ini dan petunjuk kemajuan opsional |
| `tasks/result` | "Ambil hasil" | Mengembalikan payload yang telah selesai atau 404 jika belum selesai |
| `tasks/cancel` | "Hentikan" | Permintaan pembatalan idempoten |
| ttl | "Anggaran retensi" | Milidetik server berjanji untuk mempertahankan status tugas |
| `notifications/tasks/updated` | "Dorongan negara" | Peristiwa perubahan status yang diprakarsai server |
| Toko tahan lama | "Keadaan aman dari kecelakaan" | Layer persistensi sistem file / SQLite / Redis |

## Bacaan Lanjutan

- [MCP — GitHub edisi SEP-1686](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — proposal asal dan diskusi lengkap
- [WorkOS — tugas asinkron MCP untuk alur kerja agen AI](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — panduan desain dengan dasar pemikiran
- [DeepWiki — sistem tugas MCP dan operasi asinkron](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — mekanik dan mesin status
- [FastMCP — Tugas](https://gofastmcp.com/servers/tasks) — Pola penerapan tugas tingkat SDK
- [Blog MCP — peta jalan 2026](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — isu terbuka dan prioritas 2026 termasuk subtugas
