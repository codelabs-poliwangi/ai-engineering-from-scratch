# SDK Keterampilan dan Agen — Keterampilan Antropik, AGENTS.md, SDK Aplikasi OpenAI

> MCP mengatakan "alat apa yang ada". Keterampilan mengatakan "bagaimana melakukan suatu tugas". Tumpukan 2026 melapisi keduanya. Keterampilan Agen Anthropic (standar terbuka, Desember 2025) dikirimkan sebagai SKILL.md dengan pengungkapan progresif. SDK Aplikasi OpenAI adalah MCP plus metadata widget. AGENTS.md (sekarang dalam 60.000+ repo) berada di root repo sebagai konteks agen tingkat proyek. Lesson ini menyebutkan apa yang dicakup oleh masing-masing agen dan membuat bundel SKILL.md + AGENTS.md minimal yang dikirimkan ke seluruh agen.

**Type:** Learn
**Language:** Python (stdlib, parser dan pemuat SKILL.md)
**Prerequisites:** Fase 13 · 07 (server MCP)
**Waktu:** ~45 menit

## Tujuan Pembelajaran

- Bedakan tiga layer: AGENTS.md (konteks proyek), SKILL.md (pengetahuan yang dapat digunakan kembali), MCP (alat).
- Tulis SKILL.md dengan frontmatter YAML dan pengungkapan progresif.
- Memuat gaya sistem file keterampilan ke dalam runtime agen.
- Susun keterampilan dengan server MCP dan AGENTS.md sehingga satu paket berfungsi di Claude Code, Cursor, dan Codex.

## Masalah

Seorang insinyur menyaring alur kerja penulisan catatan rilis menjadi prompt multi-langkah: "Baca PR gabungan terbaru. Kelompokkan berdasarkan area. Ringkas masing-masing. Tulis entri log perubahan mengikuti gaya tim. Posting ke draf Slack." Mereka memasukkannya ke dalam dokumen Notion untuk tim mereka.

Sekarang mereka ingin menggunakan alur kerja ini dari Claude Code, Cursor, dan Codex CLI. Setiap agen memiliki cara berbeda untuk memuat instruksi: prompt garis miring Code Claude, aturan Kursor, Kodex `.codex.md`. Insinyur menyalin alur kerja tiga kali dan memelihara tiga salinan.

AGENTS.md dan SKILL.md bersama-sama memperbaikinya:

- **AGENTS.md** berada di root repo. Setiap agen yang kompatibel membacanya pada sesi dimulai. "Bagaimana cara kerja proyek ini? Apa saja konvensinya? Prompt mana yang menjalankan pengujian?"
- **SKILL.md** adalah paket portabel: materi depan YAML (nama, deskripsi) + isi penurunan harga + sumber daya opsional. Agen yang mendukung keterampilan memuatnya berdasarkan nama sesuai permintaan.
- **MCP** (Fase 13 · 06-14) menangani alat yang dibutuhkan skill untuk dipanggil.

Tiga layer, satu artefak portabel.

## Konsep

### AGEN.md (agen.md)

Diluncurkan akhir tahun 2025, diadopsi oleh 60.000+ repo pada April 2026. Satu file di root repo. Format:

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

Agen membaca ini saat sesi dimulai dan menggunakannya untuk mengkalibrasi perilaku mereka untuk proyek tersebut. Setiap agen pengkodean pada tahun 2026 mendukung AGENTS.md: Claude Code, Cursor, Codex, Copilot Workspace, opencode, Windsurf, Zed.

### KETERAMPILAN.md format

Keterampilan Agen Anthropic (dirilis sebagai standar terbuka Desember 2025):

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

Frontmatter mendeklarasikan identitas skill. Isinya adalah prompt yang ditampilkan kepada model saat keterampilan dimuat.

### Pengungkapan progresif

Keterampilan dapat mereferensikan sub-sumber daya yang diambil agen hanya ketika diperlukan. Contoh:

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md mengatakan "lihat style-guide.md untuk aturan gaya." Agen menarik style-guide.md hanya ketika skill aktif berjalan. Hal ini untuk menghindari prompt yang membengkak dengan detail yang mungkin tidak diperlukan oleh model.

### Penemuan sistem file

Waktu proses agen memindai direktori yang dikenal untuk file SKILL.md:

- `~/.anthropic/skills/*/SKILL.md`
- Proyek `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

Pemuatan dilakukan berdasarkan nama folder dan materi depan `name`. Claude Code, Anthropic Claude Agent SDK, dan SkillKit (lintas agen) semuanya mengikuti pola ini.

### SDK Agen Claude Antropik`@anthropic-ai/claude-agent-sdk` (TypeScript) dan `claude-agent-sdk` (Python) memuat keterampilan di awal sesi, memaparkannya sebagai "agen" yang dapat dipanggil di dalam runtime. Perulangan agen dikirimkan ke suatu keterampilan ketika pengguna memanggilnya.

### SDK Aplikasi OpenAI

Diluncurkan pada Oktober 2025; dibangun langsung di MCP. Menyatukan Konektor OpenAI sebelumnya dan Tindakan GPT Khusus dalam satu platform pengembang. Aplikasi Apps SDK adalah:

- Server MCP (alat, sumber daya, petunjuk).
- Ditambah metadata widget untuk UI ChatGPT.
- Ditambah sumber daya Aplikasi MCP `ui://` opsional untuk permukaan interaktif.

Protokol yang sama, UX yang lebih kaya.

### Portabilitas lintas agen melalui SkillKit

Alat seperti SkillKit dan layer distribusi lintas agen serupa menerjemahkan satu SKILL.md ke dalam format asli masing-masing 32+ agen AI (Claude Code, Cursor, Codex, Gemini CLI, OpenCode, dll.). Satu sumber kebenaran; banyak konsumen.

### Tumpukan tiga lapis

| Layer | Berkas | Dimuat ketika | Tujuan |
|-------|------|-------------|---------|
| AGEN.md | akar repo | sesi dimulai | konvensi tingkat proyek |
| KETERAMPILAN.md | direktori keterampilan | keterampilan dipanggil | alur kerja yang dapat digunakan kembali |
| Server MCP | proses eksternal | alat yang dibutuhkan | tindakan yang dapat dipanggil |

Ketiganya tersusun: agen membaca AGENTS.md saat sesi dimulai, pengguna memanggil keterampilan, instruksi keterampilan mencakup panggilan alat MCP, agen mengirimkan melalui klien MCP.

## Pakai

`code/main.py` mengirimkan parser dan pemuat stdlib SKILL.md. Ia menemukan keterampilan di bawah `./skills/`, mengurai materi depan YAML ditambah isi penurunan harga, dan menghasilkan dict yang dikunci berdasarkan nama keterampilan. Kemudian mensimulasikan loop agen yang memanggil `release-notes-writer` berdasarkan nama.

Apa yang harus dilihat:

- Frontmatter YAML diurai dengan parser stdlib minimal (tidak ada ketergantungan `pyyaml`).
- Tubuh keterampilan disimpan kata demi kata; agen menambahkannya ke prompt sistem saat pemanggilan.
- Pengungkapan progresif didemonstrasikan melalui fungsi `read_subresource` yang mengambil file referensi sesuai permintaan.

## Kirim

Lesson ini menghasilkan `outputs/skill-agent-bundle.md`. Mengingat alur kerja, keterampilan menghasilkan gabungan bundel SKILL.md + AGENTS.md + MCP-server-blueprint, portabel di seluruh agen.

## Latihan

1. Jalankan `code/main.py`. Tambahkan keterampilan kedua di bawah `skills/` dan konfirmasikan bahwa pemuat mengambilnya.

2. Tulis AGENTS.md untuk repo kursus ini. Sertakan prompt pengujian, konvensi gaya, dan model mental Fase 13.

3. Pindahkan alur kerja multi-langkah dari dokumen internal tim kamu ke SKILL.md. Pastikan itu dimuat dalam Code Claude.

4. Terjemahkan keterampilan ke dalam format aturan asli Cursor dan Codex dengan tangan. Hitung perbedaan antar format — ini adalah permukaan terjemahan yang diotomatisasi oleh SkillKit.

5. Baca postingan blog Keterampilan Agen Antropik. Identifikasi satu feature di Claude Agent SDK yang tidak tercakup dalam pemuat lesson ini. (Petunjuk: sub-doa agen.)

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| KETERAMPILAN.md | "File keterampilan" | Frontmatter YAML ditambah isi penurunan harga, dimuat oleh runtime agen |
| AGEN.md | "Konteks agen repo-root" | File konvensi tingkat proyek dibaca saat sesi dimulai |
| Pengungkapan progresif | "Sub-sumber daya yang dimuat dengan lambat" | File referensi badan keterampilan ditarik hanya saat diperlukan |
| Materi Depan | "Blok YAML di atas" | Metadata (nama, deskripsi) di `---` pembatas |
| SDK Agen Claude | "Waktu berjalan keterampilan Antropik" | `@anthropic-ai/claude-agent-sdk`, memuat keterampilan dan rute |
| SDK Aplikasi OpenAI | "MCP + meta widget" | Permukaan pengembangan OpenAI dibangun di atas MCP plus kait UI ChatGPT |
| Penemuan keterampilan | "Pemindaian sistem file" | Telusuri direktori yang dikenal untuk SKILL.md, kunci berdasarkan nama |
| Portabilitas lintas agen | "Satu skill banyak agen" | Terjemahkan satu SKILL.md ke 32+ agen melalui alat bergaya SkillKit |
| Keterampilan Agen | "Pengetahuan portabel" | Templat tugas yang dapat digunakan kembali di luar konsep alat MCP |
| SDK Aplikasi | "MCP ditambah ChatGPT UI" | Konektor dan GPT Khusus disatukan di MCP |

## Bacaan Lanjutan

- [Anthropic — Pengumuman Keahlian Agen](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — peluncuran Desember 2025
- [Antropik — dokumen Keahlian Agen](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — referensi format SKILL.md
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — Platform pengembang berbasis MCP untuk ChatGPT
- [agents.md](https://agents.md/) — format AGENTS.md dan daftar adopsi
- [Anthropics — anthropics/skills GitHub](https://github.com/anthropics/skills) — contoh keterampilan resmi
