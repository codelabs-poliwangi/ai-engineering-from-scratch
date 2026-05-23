# Sumber Daya dan Anjuran MCP — Eksposur Konteks Melampaui Alat

> Alat mendapatkan 90 persen attention MCP. Dua server primitif lainnya memecahkan masalah yang berbeda. Sumber daya memaparkan data untuk dibaca; prompt mengekspos templat yang dapat digunakan kembali sebagai prompt garis miring. Banyak server harus menggunakan sumber daya alih-alih membungkus pembacaan dalam alat, dan prompt alih-alih alur kerja hard-coding dalam prompt klien. Lesson ini memberi nama aturan keputusan dan menjalankan pesan `resources/*` dan `prompts/*`.

**Type:** Build
**Language:** Python (stdlib, sumber daya + pengendali prompt)
**Prerequisites:** Fase 13 · 07 (server MCP)
**Waktu:** ~45 menit

## Tujuan Pembelajaran

- Putuskan antara mengekspos kemampuan sebagai alat, sumber daya, atau prompt untuk domain tertentu.
- Terapkan `resources/list`, `resources/read`, `resources/subscribe` dan tangani `notifications/resources/updated`.
- Implementasikan `prompts/list` dan `prompts/get` dengan templat argumen.
- Kenali kapan host memunculkan prompt sebagai prompt garis miring vs konteks yang disuntikkan otomatis.

## Masalah

Server MCP yang naif untuk aplikasi catatan menampilkan semuanya sebagai alat: `notes_read`, `notes_list`, `notes_search`. Ini menggabungkan setiap akses data dalam panggilan alat berdasarkan model. Konsekuensi:

- Model harus memutuskan apakah akan memanggil `notes_read` untuk setiap kueri yang mungkin mendapat manfaat dari konteks.
- Konten hanya-baca tidak dapat berlangganan atau dialirkan ke panel samping host.
- UI Klien (panel lampiran sumber daya Claude Desktop, pemilih "Sertakan file" Kursor) tidak dapat menampilkan data.

Pemisahan yang tepat: mengekspos data sebagai sumber daya, mengekspos tindakan yang bermutasi atau dihitung sebagai alat, mengekspos alur kerja multi-langkah yang dapat digunakan kembali sebagai petunjuk. Setiap primitif memiliki keterjangkauan UX dan pola aksesnya sendiri.

## Konsep

### Alat vs sumber daya vs prompt — aturan pengambilan keputusan

| Kemampuan | Primitif |
|------------|-----------|
| Pengguna ingin mencari, memfilter, atau mengubah data | alat |
| Pengguna ingin host memasukkan data ini sebagai konteks | sumber daya |
| Pengguna menginginkan alur kerja bertemplat yang dapat mereka jalankan kembali | cepat |

Pedoman: jika model mendapat manfaat dari memanggilnya pada setiap kueri terkait, itu adalah sebuah alat. Jika pengguna mendapat manfaat dari melampirkannya ke percakapan, itu adalah sumber daya. Jika seluruh alur kerja multi-langkah adalah unit yang ingin digunakan kembali oleh pengguna, ini adalah prompt.

### Sumber Daya

`resources/list` mengembalikan `{resources: [{uri, name, mimeType, description?}]}`. `resources/read` mengambil `{uri}` dan mengembalikan `{contents: [{uri, mimeType, text | blob}]}`.

URI dapat berupa apa saja yang dapat dialamatkan:

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14` (skema khusus)
- `memory://session-2026-04-22/recent` (khusus server)

`contents[]` mendukung teks dan biner. Biner menggunakan `blob` sebagai string berkode base64 ditambah `mimeType`.

### Langganan sumber daya

Deklarasikan `{resources: {subscribe: true}}` dalam kemampuan. Klien menelepon `resources/subscribe {uri}`. Server mengirimkan `notifications/resources/updated {uri}` ketika sumber daya berubah. Klien membaca ulang.

Kasus penggunaan: server catatan yang sumber dayanya berupa file di disk; pengamat file memicu pemberitahuan pembaruan; Claude Desktop menarik kembali file ke dalam konteks ketika diedit di luar host.

### Templat sumber daya (tambahan 25-11-2025)

`resourceTemplates` memungkinkan kamu mengekspos pola URI berparameter: `notes://{id}` dengan `id` sebagai target penyelesaian. Klien dapat melengkapi id secara otomatis di pemilih sumber daya.

### Anjuran`prompts/list` mengembalikan `{prompts: [{name, description, arguments?}]}`. `prompts/get` mengambil `{name, arguments}` dan mengembalikan `{description, messages: [{role, content}]}`.

Prompt adalah templat yang mengisi daftar pesan yang diumpankan host ke modelnya. Misalnya, prompt `code_review` mengambil argumen `file_path` dan mengembalikan urutan tiga pesan: pesan sistem, pesan pengguna dengan isi file, dan kickoff asisten dengan templat penalaran.

### Host dan petunjuknya

Claude Desktop, VS Code, dan Cursor menampilkan prompt sebagai prompt garis miring di UI obrolan. Pengguna mengetik `/code_review` dan mengambil argumen dari formulir. Prompt server adalah kontrak antara "pintasan pengguna" dan "prompt lengkap dikirim ke model".

Belum semua klien mendukung prompt — periksa negosiasi kemampuan. Server dengan kemampuan prompt dideklarasikan tetapi klien tanpa dukungan prompt tidak akan melihat prompt garis miring.

### Notifikasi "daftar berubah".

Sumber daya dan prompt memancarkan `notifications/list_changed` saat set bermutasi. Server catatan yang baru saja mengimpor 20 catatan baru mengeluarkan `notifications/resources/list_changed`; klien menelepon kembali `resources/list` untuk mengambil tambahan.

### Konvensi tipe konten

Untuk teks: `mimeType: "text/plain"`, `text/markdown`, `application/json`.
Untuk biner: `image/png`, `application/pdf`, ditambah bidang `blob`.
Untuk Aplikasi MCP (Lesson 14): `text/html;profile=mcp-app` dalam `ui://` URI.

### Sumber daya dinamis

URI sumber daya tidak harus sesuai dengan file statis. `notes://recent` dapat mengembalikan lima catatan terbaru pada setiap pembacaan. `db://query/users/active` dapat menjalankan kueri berparameter. Server bebas menghitung konten secara dinamis.

Aturan: jika klien dapat melakukan cache berdasarkan URI, URI harus stabil. Jika komputasi bersifat one-shot, URI harus menyertakan stempel waktu atau nonce sehingga cache klien tidak kehabisan waktu.

### Langganan vs polling

Klien berkemampuan berlangganan mendapatkan dorongan server melalui `notifications/resources/updated`. Klien atau host pra-langganan yang tidak mendukungnya melakukan polling dengan membaca ulang. Keduanya memenuhi spesifikasi. Deklarasi kemampuan server memberi tahu klien mana yang didukungnya.

Biaya berlangganan: status per sesi di server (siapa yang berlangganan apa). Jaga agar himpunan berlangganan tetap dibatasi; klien yang terputus harus time out.

### Prompt vs system prompt

Prompt di MCP bukanlah system prompt. System prompt host (instruksi pengoperasiannya sendiri) dan prompt MCP (template yang disediakan server yang dipanggil oleh pengguna) hidup berdampingan. Klien yang berperilaku baik tidak pernah membiarkan prompt server menimpa prompt sistemnya sendiri; itu melapisi mereka.

## Pakai

`code/main.py` memperluas server catatan dari Lesson 07 dengan:

- Sumber daya per catatan (`notes://note-1`, dll.) dengan dukungan `resources/subscribe`.
- Prompt `review_note` yang dirender menjadi template tiga pesan.
- Simulasi pengamat file yang memancarkan `notifications/resources/updated` saat catatan diubah.
- Sumber daya dinamis `notes://recent` yang selalu mengembalikan lima nada terbaru.

Jalankan demo untuk melihat alur selengkapnya.

## Kirim

Lesson ini menghasilkan `outputs/skill-primitive-splitter.md`. Mengingat server MCP yang diusulkan, keterampilan mengkategorikan setiap kemampuan sebagai alat/sumber daya/prompt dengan alasannya.

## Latihan

1. Jalankan `code/main.py`. Amati daftar sumber daya awal, lalu picu pengeditan catatan dan verifikasi peristiwa `notifications/resources/updated` diaktifkan.2. Tambahkan pemancar `resources/list_changed`: ketika catatan baru dibuat, kirimkan pemberitahuan sehingga klien menemukan kembali.

3. Rancang tiga prompt untuk server GitHub MCP: `summarize_pr`, `triage_issue`, `release_notes`. Masing-masing dengan skema argumen. Isi prompt harus dapat dijalankan tanpa pengeditan lebih lanjut.

4. Ambil alat yang ada di server Lesson 07 dan klasifikasikan apakah alat tersebut harus tetap menjadi alat atau dipecah menjadi pasangan sumber daya plus alat. Benarkan dalam satu kalimat.

5. Baca bagian `server/resources` dan `server/prompts` spesifikasi. Identifikasi satu bidang di `resources/read` yang jarang diisi tetapi didukung spesifikasi. Petunjuk: lihat `_meta` pada konten sumber daya.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Sumber Daya | "Data terbuka" | Konten beralamat URI yang dapat dibaca oleh host |
| URI Sumber Daya | "Penunjuk ke data" | Pengidentifikasi dengan awalan skema (`file://`, `notes://`, dll.) |
| `resources/subscribe` | "Perhatikan perubahan" | Pembaruan push server keikutsertaan klien untuk URI tertentu |
| `notifications/resources/updated` | "Sumber daya berubah" | Memberi sinyal kepada klien bahwa sumber daya berlangganan memiliki konten baru |
| Templat sumber daya | "URI yang diparameterisasi" | Pola URI dengan petunjuk penyelesaian untuk pemilih host |
| Prompt | "Templat prompt garis miring" | Templat multi-pesan bernama dengan slot argumen |
| Argumen cepat | "Input templat" | Parameter yang diketik yang dikumpulkan host sebelum merender |
| `prompts/get` | "Render templat" | Server mengembalikan daftar pesan yang terisi |
| Blok konten | "Potongan yang diketik" | `{type: text | image | resource | ui_resource}` |
| UX prompt garis miring | "Pintasan pengguna" | Host menampilkan prompt sebagai prompt yang dimulai dengan `/` |

## Bacaan Lanjutan

- [MCP — Konsep: Sumber Daya](https://modelcontextprotocol.io/docs/concepts/resources) — URI sumber daya, langganan, dan templat
- [MCP — Konsep: Anjuran](https://modelcontextprotocol.io/docs/concepts/prompts) — templat prompt dan integrasi prompt garis miring
- [MCP — Spesifikasi sumber daya server 25-11-2025](https://modelcontextprotocol.io/spesification/2025-11-25/server/resources) — referensi pesan lengkap `resources/*`
- [MCP — Spesifikasi prompt server 25-11-2025](https://modelcontextprotocol.io/spesification/2025-11-25/server/prompts) — referensi pesan lengkap `prompts/*`
- [MCP — Situs info protokol: sumber daya](https://modelcontextprotocol.info/docs/concepts/resources/) — panduan komunitas yang diperluas pada dokumen resmi
