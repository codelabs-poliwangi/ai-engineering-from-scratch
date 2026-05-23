# Capstone 13 — Server MCP dengan Registri dan Tata Kelola

> Model Context Protocol tidak lagi menjadi masa depan dan menjadi spesifikasi penggunaan alat default pada tahun 2026. Anthropic, OpenAI, Google, dan setiap IDE utama mengirimkan klien MCP. Pinterest menerbitkan ekosistem internal server MCP-nya. Registri AAIF memformalkan metadata kemampuan di `.well-known`. AWS ECS menerbitkan referensi penerapan tanpa kewarganegaraan. Agen angsa Block menempatkan protokol yang sama di dalam asisten yang dihosting. Bentuk produksi tahun 2026 adalah: Transportasi StreamableHTTP, cakupan OAuth 2.1, gerbang kebijakan OPA, dan registri yang memungkinkan tim platform menemukan, memvalidasi, dan mengaktifkan server. Build ujung ke ujung itu.

**Type:** Batu penjuru
**Language:** Python (server, melalui FastMCP) atau TypeScript (@modelcontextprotocol/sdk), Go (layanan registri)
**Prerequisites:** Fase 11 (rekayasa LLM), Fase 13 (peralatan dan MCP), Fase 14 (agen), Fase 17 (infrastruktur), Fase 18 (keselamatan)
**Fase yang dilakukan:** P11 · P13 · P14 · P17 · P18
**Waktu:** 25 jam

## Masalah

MCP menjadi lingua franca penggunaan alat. Claude Code, Cursor 3, Amp, OpenCode, Gemini CLI, dan setiap agen yang dikelola kini menggunakan server MCP. Tantangan produksi bukanlah pembuatan server (FastMCP membuatnya mudah) namun menerapkannya dalam skala besar dengan persyaratan perusahaan: cakupan OAuth per penyewa, kebijakan OPA pada alat destruktif, penskalaan stateless StreamableHTTP, registri untuk penemuan, log audit per panggilan alat. Ekosistem MCP internal Pinterest dan spesifikasi Registri AAIF menetapkan standar tahun 2026.

kamu akan membangun server MCP yang memperlihatkan 10 alat internal (hanya baca Postgres, daftar S3, Jira, Linear, Datadog, dll.), UI registri untuk penemuan platform, dan gerbang persetujuan manusia untuk alat perusak. Uji weight menunjukkan penskalaan horizontal StreamableHTTP. Jejak audit memenuhi tinjauan keamanan perusahaan.

## Konsep

Revisi MCP 2026 mengamanatkan StreamableHTTP sebagai transportasi default. Berbeda dengan bentuk stdio-dan-SSE sebelumnya, StreamableHTTP tidak memiliki kewarganegaraan secara default: satu titik akhir HTTP menerima permintaan JSON-RPC, mengalirkan respons, dan mendukung koneksi jangka panjang untuk notifikasi. Stateless berarti dapat diskalakan secara horizontal di belakang penyeimbang weight.

Otorisasi adalah OAuth 2.1 dengan cakupan per alat. Token membawa cakupan seperti `jira:read`, `s3:list`, `postgres:query:readonly`. Server MCP memeriksa cakupan pada waktu pemanggilan alat, bukan hanya saat sesi dimulai. Untuk alat berisiko tinggi, server menolak panggilan apa pun yang cakupannya tidak ditingkatkan ke `approved:by:human` dalam N menit terakhir — peningkatan tersebut berasal dari kartu ulasan Slack.

Registri adalah layanan terpisah. Setiap server MCP memperlihatkan dokumen `.well-known/mcp-capabilities` dengan manifes alatnya, URL transport, persyaratan autentikasinya. Jajak pendapat registri, validasi, dan indeks. Tim platform menggunakan UI registri untuk melihat alat apa saja yang tersedia, cakupan apa yang mereka perlukan, dan tim mana yang memilikinya.

## Arsitektur

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## Tumpukan- Kerangka server: FastMCP (Python) atau `@modelcontextprotocol/sdk` (TypeScript)
- Transportasi: StreamableHTTP melalui HTTPS (tanpa kewarganegaraan)
- Auth: OAuth 2.1 dengan identitas weight kerja melalui SPIFFE / SPIRE
- Kebijakan: Aturan OPA / Rego per alat; layanan keputusan kebijakan per permintaan
- Registri: dihosting sendiri, menggunakan manifes `.well-known/mcp-capabilities`
- Persetujuan manusia: Pesan interaktif kendur untuk alat penghancur
- Penerapan: AWS ECS Fargate atau Fly.io, satu server per penyewa atau dibagikan dengan pelingkupan penyewa
- Audit: keranjang per penyewa JSONL terstruktur dengan silsilah per panggilan

## Build

1. **Permukaan alat.** Tampilkan 10 alat internal: Kueri hanya-baca Postgres, objek daftar S3, penelusuran/pengambilan Jira, Penelusuran/pengambilan linier, kueri metrik Datadog, pencarian on-call PagerDuty, hanya-baca GitHub, Penelusuran Notion, Penelusuran Slack, Bacaan Salesforce. Setiap alat memiliki skema yang diketik dan label cakupan.

2. **Server FastMCP.** Pasang alat. Konfigurasikan transportasi StreamableHTTP. Tambahkan middleware untuk introspeksi token OAuth dan penerapan cakupan.

3. **Kebijakan OPA.** Kebijakan rego per alat: cakupan apa yang mengizinkan pemanggilan, redaksi PII apa yang berlaku, batasan ukuran payload apa yang berlaku. Layanan pengambilan keputusan dipanggil pada setiap panggilan alat.

4. **Layanan registri.** Pisahkan layanan Go atau TS yang melakukan polling `.well-known/mcp-capabilities` dari server terdaftar, memvalidasi dengan Skema JSON, dan menampilkan daftar/penelusuran/validasi/aktifkan-nonaktifkan UI.

5. **Manifes kemampuan.** Setiap server mengekspos `.well-known/mcp-capabilities` dengan: daftar alat, persyaratan autentikasi, URL transportasi, tim pemilik, SLO.

6. **Pemisahan alat destruktif.** Alat yang statusnya bermutasi (Pembuatan Jira, Pembuatan Linear, Penulisan Postgres) aktif di server MCP kedua dengan alur autentikasi yang lebih ketat: token harus memiliki cakupan `approved:by:human` yang ditingkatkan melalui kartu Slack dalam waktu 15 menit.

7. **Log audit.** JSONL khusus tambahan per penyewa: `{timestamp, user, tool, args_redacted, response_redacted, outcome}`. Redaksi PII melalui Presidio sebelum menulis.

8. **Uji weight.** 100 klien serentak di StreamableHTTP. Peragakan penskalaan horizontal dengan menambahkan replika kedua; menunjukkan penyeimbang weight mendistribusikan ulang tanpa kekakuan sesi.

9. **Uji kesesuaian.** Jalankan rangkaian kesesuaian MCP resmi pada kedua server. Lulus semua bagian wajib.

## Pakai

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## Kirim

`outputs/skill-mcp-server.md` menjelaskan penyampaiannya. Server MCP tingkat produksi + registri + layer audit untuk alat internal dengan cakupan OAuth 2.1 dan gerbang OPA.

| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | Kesesuaian spesifikasi | Manifes kemampuan StreamableHTTP + lulus uji kesesuaian MCP |
| 20 | Keamanan | Penegakan cakupan, cakupan OPA di setiap alat, kebersihan rahasia |
| 20 | Observabilitas | Log audit per panggilan alat dengan redaksi PII |
| 20 | Skala | Demonstrasi skala horizontal uji weight 100 klien |
| 15 | UX Registri | Temukan / validasi / aktifkan-nonaktifkan alur kerja |
| **100** | | |

## Latihan

1. Tambahkan alat baru (pencarian Confluence). Kirim melalui alur validasi registri tanpa menyentuh server inti.

2. Tulis kebijakan OPA yang menyunting hasil kueri Postgres yang berisi kolom bernama `email`, `ssn`, atau `phone`. Latihan dengan kueri penyelidikan.

3. Tolok ukur StreamableHTTP vs stdio pada latensi lokal. Laporkan per panggilan p50/p95.

4. Menerapkan kuota per penyewa: maksimum N panggilan per menit per alat per penyewa. Terapkan melalui aturan OPA kedua.5. Jalankan rangkaian kesesuaian MCP dari [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) dan perbaiki setiap kegagalan.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| StreamableHTTP | "Transportasi MCP 2026" | Streaming HTTP+ tanpa kewarganegaraan; menggantikan SSE + stdio untuk server jaringan |
| Manifes kemampuan | "Dokter terkenal" | `.well-known/mcp-capabilities` dengan daftar alat, autentikasi, URL pengangkutan |
| OPA/Rego | "Mesin kebijakan" | Agen Kebijakan Terbuka untuk mengotorisasi panggilan alat terhadap aturan eksternal |
| Ketinggian ruang lingkup | "Disetujui oleh manusia" | Cakupan berumur pendek diberikan melalui persetujuan Slack, diperlukan untuk alat destruktif |
| Registri | "Penemuan alat" | Layanan yang mengindeks server MCP dari manifestasi kemampuannya |
| Identitas weight kerja | "SPIFFE / SPIRE" | Identitas layanan kriptografi untuk penerbitan token OAuth |
| Rangkaian kesesuaian | "Tes spesifikasi" | Baterai uji MCP resmi untuk kebenaran nyata alat StreamableHTTP + |

## Bacaan Lanjutan

- [Peta Jalan Model Context Protocol 2026](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, metadata kemampuan, registri
- [Spesifikasi Registri MCP AAAF](https://github.com/modelcontextprotocol/registry) — spesifikasi registri tahun 2026
- [penerapan referensi AWS ECS](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — referensi penerapan produksi
- [ekosistem MCP internal Pinterest](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — referensi penerapan internal
- [Blokir `goose` penggunaan MCP](https://block.github.io/goose/) — pola konsumsi agen referensi
- [FastMCP](https://github.com/jlowin/fastmcp) — Kerangka server Python
- [Agen Kebijakan Terbuka](https://www.openpolicyagent.org/) — referensi mesin kebijakan
- [SPIFFE / SPIRE](https://spiffe.io) — referensi identitas weight kerja
