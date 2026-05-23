# Capstone — Membangun Ekosistem Alat yang Lengkap

> Fase 13 mengajarkan setiap bagian. Batu penjuru ini menghubungkan mereka ke dalam satu sistem berbentuk produksi: server MCP dengan alat + sumber daya + prompt + tugas + UI, OAuth 2.1 di edge, gateway RBAC, klien multi-server, panggilan sub-agen A2A, penelusuran OTel ke dalam kolektor, deteksi keracunan alat di CI, dan bundel AGENTS.md + SKILL.md. Pada akhirnya kamu dapat mempertahankan setiap pilihan arsitektur.

**Type:** Build
**Language:** Python (stdlib, pemanfaatan ekosistem ujung ke ujung)
**Prerequisites:** Fase 13 · 01 hingga 21
**Waktu:** ~120 menit

## Tujuan Pembelajaran

- Buat server MCP yang mengekspos alat, sumber daya, petunjuk, dan tugas dengan aplikasi `ui://`.
- Depan server dengan gateway OAuth 2.1 yang menerapkan RBAC dan hash yang di-embed.
- Tulis klien multi-server yang menelusuri dengan atribut OTel GenAI ujung ke ujung.
- Mendelegasikan sebagian weight kerja ke sub-agen A2A; verifikasi opacity dipertahankan.
- Kemas seluruh tumpukan dengan AGENTS.md + SKILL.md sehingga agen lain dapat mengendarainya.

## Masalah

Kirim sistem "penelitian dan laporan":

- Pengguna bertanya: "rangkum tiga makalah arXiv tahun 2026 yang paling banyak dikutip tentang protokol agen."
- Sistem: cari arXiv melalui MCP; mendelegasikan ringkasan makalah ke agen penulis khusus melalui A2A; hasil agregat; merender laporan interaktif sebagai sumber daya Aplikasi MCP `ui://`; mencatat setiap langkah ke Otel.

Semua primitif dari Fase 13 muncul. Ini bukan mainan — sistem asisten penelitian produksi yang dikirimkan pada tahun 2026 oleh Anthropic (produk Claude Research), OpenAI (GPT dengan Apps SDK), dan pihak ketiga memiliki bentuk yang persis seperti ini.

## Konsep

### Arsitektur

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### Lacak hierarki

```
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

Satu jejak id. Setiap rentang memiliki atribut `gen_ai.*` yang tepat.

### Postur keamanan

- OAuth 2.1 + PKCE dengan indikator sumber daya yang embed audiens ke gateway.
- Gateway memegang kredensial hulu; pengguna tidak pernah melihatnya.
- RBAC: `alice` memiliki `research:read`, `research:write`, dapat memanggil semua alat. `bob` memiliki `research:read`, tidak dapat menelepon `generate_report`.
- Manifes deskripsi yang di-embed: menghapus server mana pun yang hash alatnya berubah.
- Audit Aturan Dua: tidak ada alat yang menggabungkan input yang tidak tepercaya, data sensitif, dan tindakan konsekuensial.

### Merender

Tugas terakhir `generate_report` mengembalikan blok konten ditambah sumber daya `ui://report/current`. Host klien (Claude Desktop, dll.) merender dasbor interaktif dalam iframe kotak pasir. Dasbor berisi daftar makalah yang diurutkan, jumlah kutipan, dan tombol yang memanggil `host.callTool('summarize_paper', {arxiv_id})` untuk makalah apa pun yang diklik pengguna.

### Pengemasan

Semuanya dikirimkan sebagai:

```
research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # the top-level workflow
  servers/
    research-mcp/               # the MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # the A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

Pengguna menyebarkan dengan `docker compose up`. Pengguna Claude Code, Cursor, Codex, dan opencode dapat menggerakkan sistem dengan menggunakan skill `run-research`.

### Kontribusi setiap lesson Fase 13

| Lesson | Apa yang digunakan batu penjuru |
|--------|------------------------|
| 01-05 | Antarmuka alat, portabilitas penyedia, panggilan paralel, skema, linting |
| 06-10 | Primitif MCP, server, klien, transportasi, sumber daya + petunjuk |
| 11-14 | Pengambilan sample, root + elisitasi, tugas asinkron, aplikasi `ui://` |
| 15-17 | Keracunan alat, OAuth 2.1, gateway + registri |
| 18 | Delegasi sub-agen A2A |
| 19 | penelusuran Otel GenAI |
| 20 | Perutean gateway untuk layer LLM |
| 21 | SKILL.md + AGEN.md kemasan |

## Pakai`code/main.py` menggabungkan pola lesson sebelumnya menjadi satu demo yang dapat dijalankan. Semua stdlib, semua dalam proses sehingga kamu dapat membacanya dari ujung ke ujung. Ini menjalankan alur penuh untuk skenario penelitian dan laporan: jabat tangan dengan gateway, simulasi OAuth 2.1, penggabungan alat/daftar, generate_report sebagai tugas, panggilan A2A ke penulis, sumber daya ui:// dikembalikan, rentang OTel dipancarkan.

Apa yang harus dilihat:

- Satu jejak id di setiap hop.
- Kebijakan gateway memblokir pengguna kedua untuk menulis.
- Siklus hidup tugas berjalan → selesai dan mengembalikan konten teks dan ui://.
- Keadaan dalam panggilan A2A tidak jelas bagi orkestrator.
- AGENTS.md dan SKILL.md adalah satu-satunya file yang dibutuhkan agen lain untuk mereproduksi alur kerja.

## Kirim

Lesson ini menghasilkan `outputs/skill-ecosystem-blueprint.md`. Mengingat kebutuhan produk (penelitian, ringkasan, otomatisasi), keterampilan tersebut menghasilkan arsitektur lengkap: primitif MCP mana, kontrol gateway mana, panggilan A2A mana, telemetri mana, pengemasan mana.

## Latihan

1. Jalankan `code/main.py`. Catat id jejak tunggal dan rentang sarangnya. Hitung berapa banyak primitif dari Fase 13 yang disentuh demo.

2. Perluas demo: tambahkan server MCP backend kedua (misalnya `bibliography`) dan konfirmasikan gateway menggabungkan alatnya ke dalam namespace yang sama.

3. Ganti agen penulis A2A palsu dengan agen asli yang menjalankan subproses. Gunakan tali pengaman Lesson 19.

4. Tambahkan langkah redaksi PII di gateway perutean antara orkestrator dan LLM. Konfirmasikan email dalam permintaan pengguna dihapus.

5. Tulis AGENTS.md untuk rekan satu tim yang akan memelihara sistem ini. Diperlukan waktu kurang dari lima menit untuk membaca dan memberi mereka semua yang mereka perlukan untuk menggerakkan batu penjuru di Cursor atau Codex.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Batu penjuru | "Demo integrasi fase-13" | Sistem end-to-end menggunakan setiap |.primitif
| Penelitian dan laporan | "Skenario" | Cari, rangkum, render pola |
| Ekosistem | "Semua bagian menjadi satu" | Server + klien + gateway + sub-agen + telemetri + paket |
| Hirarki penelusuran | "ID jejak tunggal" | Rentang setiap hop berbagi jejak; orang tua-anak melalui id rentang |
| Token yang dikeluarkan gateway | "Otentikasi transitif" | Klien hanya melihat token gateway; gateway memegang kredibilitas hulu |
| Ruang nama yang digabungkan | "Semua alat dalam satu daftar datar" | Penggabungan multi-server di gateway, awalan-saat-tabrakan |
| Batas opasitas | "Panggilan A2A menyembunyikan internal" | Alasan sub-agen tidak terlihat oleh orkestra |
| Tumpukan tiga lapis | "AGEN.md + KETERAMPILAN.md + MCP" | Konteks proyek + alur kerja + alat |
| Pertahanan Mendalam | "Beberapa layer keamanan" | Hash yang di-embed, OAuth, RBAC, Aturan Dua, log audit |
| Matrix kepatuhan spesifikasi | "Apa yang kami kirimkan sesuai spesifikasi" | Hasil pemetaan daftar periksa untuk persyaratan 25-11-2025 |

## Bacaan Lanjutan

- [MCP — Spesifikasi 25-11-2025](https://modelcontextprotocol.io/spesification/2025-11-25) — referensi gabungan
- [Blog MCP — peta jalan 2026](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — tujuan protokol
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — Referensi A2A v1.0
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — konvensi penelusuran kanonik
- [Antropik — Ikhtisar SDK Agen Claude](https://code.claude.com/docs/en/agent-sdk/overview) — pola runtime agen produksi
