# Gerbang dan Registri MCP — Bidang Kontrol Perusahaan

> Perusahaan tidak boleh membiarkan setiap pengembang menginstal server MCP acak. Gerbang memusatkan autentikasi, RBAC, audit, pembatasan laju, caching, dan deteksi keracunan alat, lalu memaparkan permukaan alat yang digabungkan sebagai satu titik akhir MCP. Registri MCP Resmi (Anthropic + GitHub + PulseMCP + Microsoft, terverifikasi namespace) adalah upstream kanonik. Lesson ini menyebutkan lokasi gateway yang cocok, menjalankan implementasi minimal, dan mensurvei lanskap vendor pada tahun 2026.

**Type:** Learn
**Language:** Python (stdlib, gateway minimal)
**Prerequisites:** Fase 13 · 15 (keracunan alat), Fase 13 · 16 (OAuth 2.1)
**Waktu:** ~45 menit

## Tujuan Pembelajaran

- Jelaskan lokasi gateway MCP (antara klien MCP dan beberapa server MCP backend).
- Menerapkan lima tanggung jawab gateway: auth, RBAC, audit, rate limit, kebijakan.
- Menerapkan manifes hash alat yang di-embed di layer gateway.
- Membedakan Registri MCP Resmi dari metaregistrasi (Glama, MCPMarket, MCP.so, Smithery, LobeHub).

## Masalah

Fortune 500 memiliki 30 server MCP yang disetujui, 5000 pengembang, persyaratan kepatuhan dan audit, serta tim keamanan yang menginginkan kebijakan terpusat. Membiarkan setiap pengembang menginstal server sewenang-wenang di IDE mereka bukanlah hal yang mudah.

Pola gerbang:

1. Gateway berjalan sebagai satu titik akhir HTTP Streamable yang terhubung dengan pengembang.
2. Gateway menyimpan kredensial untuk setiap server MCP backend.
3. Setiap permintaan pengembang diautentikasi dan dicakup melalui OAuth gateway itu sendiri.
4. Gateway merutekan panggilan ke server backend, menerapkan kebijakan.
5. Semua panggilan dicatat untuk audit.

Portal Cloudflare MCP, Kong AI Gateway, IBM ContextForge, MintMCP, TrueFoundry, Envoy AI Gateway — semua feature gateway atau gateway dikirimkan pada tahun 2025-2026.

Sementara itu, Registri MCP Resmi diluncurkan sebagai hulu kanonik: server yang dikuratori, terverifikasi namespace, dan diberi nama DNS terbalik yang dapat digunakan oleh gateway. Metaregistries (Glama, MCPMarket, MCP.so, Smithery, LobeHub) mengumpulkan server di berbagai sumber.

## Konsep

### Lima tanggung jawab gerbang

1. **Auth.** OAuth 2.1 untuk mengidentifikasi pengembang; memetakan ke peran pengguna.
2. **RBAC.** Kebijakan per pengguna: server mana, alat apa, cakupan apa.
3. **Audit.** Setiap panggilan dicatat dengan siapa, apa, kapan, hasilnya.
4. **Batas kapasitas.** Batasan per pengguna / per alat / per server untuk mencegah penyalahgunaan.
5. **Kebijakan.** Tolak deskripsi beracun, tegakkan Aturan Dua, edit PII.

### Gateway sebagai titik akhir tunggal

Bagi pengembang, gateway tampak seperti satu server MCP. Secara internal itu merutekan ke N backend. Id sesi (Fase 13 · 09) ditulis ulang di batas.

### Penyimpanan kredensial

Pengembang tidak pernah melihat token backend. Gateway menampungnya (atau proksi ke penyedia identitas yang menampungnya). Pengembang dengan `notes:read` di gateway dapat mengakses server MCP catatan secara transitif dengan kredensial backend gateway itu sendiri — namun hanya berdasarkan kebijakan yang mengikat akses transitif.

### Alat-hash embed di gateway

Gateway menyimpan manifes deskripsi alat yang disetujui (hash SHA256). Pada waktu penemuan, ia mengambil `tools/list` masing-masing backend, membandingkan hash dengan manifes, dan menghapus alat apa pun yang deskripsinya telah bermutasi. Ini adalah pertahanan tarik-menarik dari Fase 13 · 15 yang diterapkan secara terpusat.

### Kebijakan sebagai kodeGerbang tingkat lanjut menyatakan kebijakan di OPA/Rego, Kyverno, atau Styra. Aturan seperti "pengguna `alice` dapat menghubungi `github.open_pr` hanya pada repo di organisasi `acme`" dikodekan secara deklaratif. Gerbang sederhana menggunakan Python dengan code tangan. Kedua bentuk tersebut valid.

### Perutean yang sadar sesi

Jika sesi pengguna menyertakan campuran server, gateway akan dimultipleks: sesi MCP tunggal pengembang menampung N sesi backend, satu sesi per server. Pemberitahuan dari rute backend mana pun melalui gateway ke sesi pengembang.

### Penggabungan ruang nama

Gateway menggabungkan ruang nama alat dari semua backend, biasanya dengan awalan saat bertabrakan. `github.open_pr`, `notes.search`. Hal ini membuat perutean menjadi tidak ambigu.

### Registri

- **Registrasi MCP Resmi (`registry.modelcontextprotocol.io`).** Diluncurkan di bawah pengawasan Anthropic, GitHub, PulseMCP, dan Microsoft. Namespace terverifikasi (DNS terbalik: `io.github.user/server`). Difilter sebelumnya untuk kualitas dasar.
- **Glama.** Metaregistrasi yang berpusat pada penelusuran yang menggabungkan banyak sumber.
- **MCPMarket.** Direktori yang cenderung komersial dengan daftar vendor.
- **MCP.so.** Direktori komunitas; kiriman terbuka.
- **Smithery.** Alur instalasi bergaya manajer paket.
- **LobeHub.** Registri yang terintegrasi dengan UI di aplikasi LobeChat mereka.

Gerbang perusahaan mengambil dari Registri Resmi secara default, mengizinkan penambahan yang dikurasi admin dari metaregistrasi, dan menolak apa pun yang dilepas pinnya.

### Penamaan DNS terbalik

Registri Resmi mewajibkan nama DNS terbalik untuk server publik: `io.github.alice/notes`. Namespace mencegah jongkok dan membuat delegasi kepercayaan lebih jelas.

### Survei vendor, April 2026

| Penjual | Kekuatan |
|--------|----------|
| Portal Cloudflare MCP | Dihosting oleh tepi; OAuth terintegrasi; tingkat gratis |
| Gerbang Kong AI | K8s-asli; kebijakan yang terperinci; log ke OpenTelemetry |
| IBM ContextForge | IAM Perusahaan; kepatuhan; ekspor audit |
| Pengecoran Sejati | condong ke DevOps; metrik-pertama |
| MintMCP | Berorientasi platform pengembang |
| Gerbang AI Utusan | Sumber terbuka; filter yang dapat disesuaikan |

Fase 17 (infrastruktur produksi) mendalami operasi gateway.

## Pakai

`code/main.py` mengirimkan gateway minimal dalam ~150 baris: mengautentikasi pengguna dengan token Pembawa palsu, menerapkan kebijakan RBAC per pengguna, merutekan permintaan ke dua server MCP backend, menulis setiap panggilan ke log audit, menerapkan batas kapasitas, dan menolak alat backend apa pun yang hash deskripsinya tidak cocok dengan manifes yang di-embed.

Apa yang harus dilihat:

- `RBAC` dict yang diketik oleh `user_id` dengan entri `server_tool` yang diperbolehkan.
- `AUDIT_LOG` adalah daftar acara yang hanya dapat ditambahkan.
- Batas tarif menggunakan keranjang token per pengguna.
- Manifes yang di-embed adalah dikte `server::tool -> hash`.

## Kirim

Lesson ini menghasilkan `outputs/skill-gateway-bootstrap.md`. Mengingat rencana MCP perusahaan (pengguna, backend, kepatuhan), keterampilan menghasilkan spesifikasi konfigurasi gateway.

## Latihan

1. Jalankan `code/main.py`. Melakukan panggilan sebagai pengguna yang diizinkan; kemudian sebagai pengguna yang tidak diizinkan; kemudian ledakan yang overshoot kecepatan. Verifikasi ketiga aliran.

2. Tambahkan kebijakan yang menyunting PII dari hasil sebelum dikembalikan ke klien. Gunakan pass regex sederhana untuk string berbentuk SSN; catat celahnya (email, nomor telepon).

3. Perluas log audit untuk mengeluarkan rentang OpenTelemetry GenAI. Fase 13 · 20 mencakup atribut yang tepat.

4. Rancang kebijakan RBAC untuk tim pengembang yang terdiri dari 50 orang dengan lima backend (notes, github, postgres, jira, slack). Siapa yang mendapat status read-only pada masing-masingnya? Siapa yang mendapat tulisan?5. Baca postingan MCP perusahaan Cloudflare dari atas ke bawah. Identifikasi satu feature yang dikirimkan Cloudflare yang tidak dimiliki gateway stdlib ini.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Gerbang | "Proksi MCP" | Sentralisasi server antara klien dan backend |
| Penyimpanan kredensial | "Token backend tetap berada di sisi server" | Pengembang tidak pernah melihat token hulu |
| Perutean sadar sesi | "Sesi multi-backend" | Multipleks gateway N sesi backend per sesi pengembang |
| Embedding hash alat | "Manifes yang disetujui" | SHA256 dari setiap deskripsi alat yang disetujui; blok tarikan permadani secara terpusat |
| RBAC | "Kebijakan per pengguna" | Kontrol akses berbasis peran untuk alat dan server |
| Kebijakan-sebagai-code | "Aturan deklaratif" | Kebijakan OPA/Rego, Kyverno, Styra diberlakukan di gateway |
| Catatan audit | "Siapa, apa, kapan" | Tambahkan log peristiwa khusus untuk kepatuhan |
| Batas tarif | "Ember token per pengguna" | Batasan per menit untuk mencegah penyalahgunaan |
| Registri MCP Resmi | "Hulu kanonik" | `registry.modelcontextprotocol.io`, terverifikasi namespace |
| Penamaan DNS terbalik | "Ruang nama registri" | `io.github.user/server` konvensi |

## Bacaan Lanjutan

- [Registri MCP Resmi](https://registry.modelcontextprotocol.io/) — upstream kanonik, terverifikasi namespace
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) — pola gateway dengan OAuth dan kebijakan
- [agent-community — Registri gerbang MCP](https://github.com/agentic-community/mcp-gateway-registry) — gerbang referensi sumber terbuka
- [TrueFoundry — Apa itu gateway MCP?](https://www.truefoundry.com/blog/what-is-mcp-gateway) — artikel perbandingan feature
- [IBM — Penempaan konteks MCP](https://github.com/IBM/mcp-context-forge) — gerbang perusahaan dari IBM
