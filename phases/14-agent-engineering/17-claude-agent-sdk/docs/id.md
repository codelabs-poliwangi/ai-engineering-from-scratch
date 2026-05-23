# Claude Agent SDK: Subagen dan Penyimpanan Sesi

> Claude Agent SDK adalah bentuk perpustakaan dari memanfaatkan Claude Code. Alat bawaan, subagen untuk isolasi konteks, kait, propagasi jejak W3C, paritas penyimpanan sesi. Agen Terkelola Claude adalah alternatif yang dihosting untuk pekerjaan asinkron yang sudah berjalan lama.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 01 (Agent Loop), Fase 14 · 10 (Perpustakaan Keterampilan)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Jelaskan perbedaan antara Anthropic Client SDK (API mentah) dan Claude Agent SDK (bentuk harness).
- Jelaskan subagen — paralelisasi dan isolasi konteks — dan kapan harus mencapainya.
- Beri nama permukaan penyimpanan sesi SDK Python (`append`, `load`, `list_sessions`, `delete`, `list_subkeys`) dan peran `--session-mirror`.
- Mengimplementasikan stdlib harness dengan alat bawaan, pemijahan subagen dengan konteks terisolasi, kait siklus hidup, dan penyimpanan sesi.

## Masalah

API LLM mentah memberi kamu satu kali perjalanan pulang pergi. Agen produksi memerlukan eksekusi alat, server MCP, kait siklus hidup, pemijahan subagen, persistensi sesi, dan propagasi jejak. Claude Agent SDK mengirimkan bentuk ini sebagai perpustakaan — memanfaatkan code yang sama yang digunakan Claude Code, diekspos untuk agen khusus.

## Konsep

### SDK Klien vs SDK Agen

- **SDK Klien (`anthropic`).** API Pesan Mentah. kamu memiliki loopnya, alatnya, negara bagiannya.
- **Agen SDK (`claude-agent-sdk`).** Eksekusi alat bawaan, koneksi MCP, hook, pemijahan subagen, penyimpanan sesi. Lingkaran Code Claude sebagai perpustakaan.

### Alat bawaan

SDK mengirimkan 10+ alat yang siap digunakan: baca/tulis file, shell, grep, glob, pengambilan web, dan banyak lagi. Alat khusus mendaftar melalui antarmuka skema alat standar.

### Subagen

Dua tujuan yang didokumentasikan oleh Anthropic:

1. **Paralelisasi.** Jalankan pekerjaan mandiri secara bersamaan. "Temukan file pengujian untuk masing-masing dari 20 modul ini" adalah 20 tugas subagen paralel.
2. **Isolasi konteks.** Subagen menggunakan jendela konteksnya sendiri; hanya hasil yang dikembalikan ke orkestrator. Anggaran orkestra dipertahankan.

Tambahan terbaru Python SDK: `list_subagents()`, `get_subagent_messages()` untuk membaca transkrip subagen.

### Penyimpanan sesi

Paritas protokol dengan TypeScript:

- `append(session_id, message)` — tambahkan belokan.
- `load(session_id)` — memulihkan percakapan.
- `list_sessions()` — menghitung.
- `delete(session_id)` — dengan sesi kaskade ke subagen.
- `list_subkeys(session_id)` — mencantumkan kunci subagen.

`--session-mirror` (tanda CLI) mencerminkan transkrip ke file eksternal saat dialirkan, untuk proses debug.

### Kait

Kait siklus hidup yang dapat kamu daftarkan:

- `PreToolUse`, `PostToolUse` — panggilan gerbang atau alat audit.
- `SessionStart`, `SessionEnd` — pasang dan bongkar.
- `UserPromptSubmit` — bertindak berdasarkan input pengguna sebelum model melihatnya.
- `PreCompact` — dijalankan sebelum pemadatan konteks.
- `Stop` — pembersihan saat keluar agen.
- `Notification` — peringatan pipeline samping.

Hooks adalah bagaimana pro-alur kerja (referensi kurikulum Fase 14) dan sistem serupa menambahkan perilaku lintas sektoral.

### Konteks penelusuran W3C

Rentang OTel yang aktif pada pemanggil disebarkan ke dalam subproses CLI melalui header konteks jejak W3C. Seluruh jejak multi-proses muncul sebagai satu jejak di backend kamu.

### Agen yang Dikelola ClaudeAlternatif yang dihosting (header beta `managed-agents-2026-04-01`). Pekerjaan asinkron yang berjalan lama, cache cepat bawaan, pemadatan bawaan. Kontrol tradeoff untuk infrastruktur yang dikelola.

### Dimana letak kesalahan pola ini

- **Subagent over-spawn.** Memunculkan 100 subagen untuk 100 tugas kecil. Overhead mendominasi. Sebagai gantinya.
- **Hook creep.** Setiap tim menambahkan hook; balon waktu startup. Tinjau hook setiap tiga bulan.
- **Sesi membengkak.** Sesi terakumulasi; ukurannya bertambah. Gunakan `list_sessions` + kebijakan kedaluwarsa.

## Build

`code/main.py` mengimplementasikan bentuk SDK di stdlib:

- `Tool`, `ToolRegistry` dengan `read_file` bawaan, `write_file`, `list_dir`.
- `Subagent` — konteks pribadi, proses terisolasi, hasil dikembalikan.
- `SessionStore` — menambahkan, memuat, membuat daftar, menghapus, list_subkeys.
- `Hooks` — `pre_tool_use`, `post_tool_use`, `session_start`, `session_end`.
- Demo: agen utama memunculkan 3 subagen secara paralel (masing-masing terisolasi), mengumpulkan hasil, melanjutkan sesi.

Jalankan:

```
python3 code/main.py
```

Pelacakan menunjukkan isolasi konteks subagen (ukuran konteks pengelola tetap dibatasi), eksekusi hook, dan persistensi sesi.

## Pakai

- **Claude Agent SDK** untuk produk Claude-first yang menginginkan bentuk harness Claude Code.
- **Agen Terkelola Claude** untuk pekerjaan asinkron yang dihosting dan berjalan lama.
- **OpenAI Agents SDK** (Lesson 16) untuk rekanan yang mengutamakan OpenAI.
- **LangGraph + alat khusus** jika kamu menginginkan mesin status berbentuk grafik.

## Kirim

`outputs/skill-claude-agent-scaffold.md` merancah aplikasi Claude Agent SDK dengan subagen, hook, penyimpanan sesi, lampiran server MCP, dan propagasi jejak W3C.

## Latihan

1. Tambahkan pemijahan subagen yang mengelompokkan 20 tugas ke dalam kelompok yang terdiri dari 5 subagen paralel. Ukur ukuran konteks orkestrator vs satu tugas per tugas.
2. Terapkan hook `PreToolUse` yang membatasi kecepatan panggilan `write_file` (5 per menit per sesi). Lacak perilakunya.
3. Hubungkan `list_subkeys` untuk merender pohon subagen. Seperti apa sarang yang dalam?
4. Pindahkan mainan ke paket `claude-agent-sdk` Python yang asli. Perubahan apa saja pada registrasi alat?
5. Baca dokumen Agen Terkelola Claude. Kapan kamu akan beralih dari yang dihosting sendiri ke dikelola?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Agen SDK | "Claude Code sebagai perpustakaan" | Bentuk harness: perkakas, MCP, kait, subagen, penyimpanan sesi |
| Subagen | "Agen anak" | Konteks terpisah, anggaran sendiri; hasil menggelembung |
| Penyimpanan sesi | "DB Percakapan" | Bertahan, memuat, membuat daftar, menghapus putaran dengan kaskade subagen |
| Kait | "Panggilan balik siklus hidup" | Alat pra/pasca, sesi, pengiriman cepat, padatkan, hentikan |
| Konteks jejak W3C | "Jejak lintas proses" | Rentang induk menyebar ke subproses CLI |
| Agen Terkelola | "Harness yang dihosting" | Pekerjaan asinkron yang berjalan lama dan dihosting secara antropik |
| `--session-mirror` | "Cermin transkrip" | Sesi penulisan beralih ke file eksternal saat streaming |
| Server MCP | "Permukaan alat" | Alat/sumber daya eksternal yang melekat pada agen |

## Bacaan Lanjutan- [Ikhtisar Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — bentuk perpustakaan Claude Code
- [Antropik, Agen bangunan dengan Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — pola produksi
- [Ikhtisar Agen Terkelola Claude](https://platform.claude.com/docs/en/managed-agents/overview) — alternatif yang dihosting
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — rekanan
