# AutoGen v0.4: Model Aktor dan Kerangka Agen

> AutoGen v0.4 (Microsoft Research, Jan 2025) mendesain ulang orkestrasi agen seputar model aktor. Pertukaran pesan asinkron, agen berbasis peristiwa, isolasi kesalahan, konkurensi alami. Kerangka kerja ini sekarang berada dalam mode pemeliharaan sementara Microsoft Agent Framework (pratinjau publik Oktober 2025) menjadi penerusnya.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 01 (Agent Loop), Fase 14 · 12 (Pola Alur Kerja)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Jelaskan model aktor: agen sebagai aktor, pesan sebagai satu-satunya IPC, isolasi kegagalan per aktor.
- Beri nama tiga layer API AutoGen v0.4 — Core, AgentChat, Extensions — dan kegunaan masing-masingnya.
- Jelaskan mengapa pemisahan pengiriman pesan dari penanganan memberikan isolasi kesalahan dan konkurensi alami.
- Mengimplementasikan runtime aktor stdlib dengan Python dan mem-porting aliran peninjauan code dua agen ke dalamnya.

## Masalah

Sebagian besar framework agen bersifat sinkron: satu agen memproduksi, satu agen menggunakan, dalam tumpukan panggilan. Kegagalan merusak tumpukan. Konkurensi diaktifkan. Distribusi memerlukan penulisan ulang.

Jawaban AutoGen v0.4: model aktor. Setiap agen adalah aktor dengan kotak masuk pribadi. Pesan adalah satu-satunya interaksi. Waktu proses memisahkan pengiriman dari penanganan. Kegagalan terisolasi pada satu aktor. Konkurensi adalah sesuatu yang asli. Distribusi hanyalah transportasi yang berbeda.

## Konsep

### Aktor

Seorang aktor mempunyai:

- Swasta negara (tidak pernah disentuh langsung dari luar).
- Kotak masuk (antrian pesan).
- Penangan: `receive(message) -> effects` yang efeknya dapat berupa "balasan", "kirim ke aktor lain", "munculkan aktor baru", "perbarui status", "hentikan diri sendiri".

Dua aktor tidak dapat berbagi memori. Mereka hanya bisa mengirim pesan.

### Tiga layer API di AutoGen v0.4

1. **Inti.** Kerangka aktor tingkat rendah. `AgentRuntime`, `Agent`, `Message`, `Topic`. Pertukaran pesan asinkron, didorong oleh peristiwa.
2. **AgentChat.** API tingkat tinggi berbasis tugas (pengganti ConversableAgent v0.2). `AssistantAgent`, `UserProxyAgent`, `RoundRobinGroupChat`, `SelectorGroupChat`.
3. **Ekstensi.** Integrasi — OpenAI, Anthropic, Azure, alat, memori.

### Mengapa pemisahan itu penting

Dalam model v0.2, panggilan `agent_a.chat(agent_b)` secara sinkron memblokir agent_a hingga agent_b kembali. Di v0.4, `send(agent_b, msg)` memasukkan pesan ke kotak masuk agen_b dan mengembalikannya. Waktu proses dikirimkan nanti. Tiga konsekuensi:

- **Isolasi kesalahan.** Agen B yang mogok tidak membuat Agen A mogok — runtime menangkap kegagalan pada pengendali B dan memutuskan apa yang harus dilakukan (catat, coba lagi, surat mati).
- **Konkurensi alami.** Banyak pesan dalam penerbangan sekaligus; aktor memproses kotak masuk mereka secara bersamaan.
- **Siap distribusi.** Kotak masuk + transportasi adalah abstraksi yang sama baik aktor sedang dalam proses atau berada di host lain.

### Topologi

- **RoundRobinGroupChat.** Agen bergiliran dalam rotasi tetap.
- **SelectorGroupChat.** Agen pemilih memilih siapa yang berikutnya berdasarkan konteks percakapan.
- **Magentic-One.** Referensi tim multi-agen untuk penjelajahan web, eksekusi code, penanganan file. Dibangun di AgentChat.

### Observabilitas

Dukungan OpenTelemetry sudah ada di dalamnya. Setiap pesan memancarkan rentang; panggilan alat membawa atribut `gen_ai.*` sesuai konvensi semantik OTel GenAI 2026 (Lesson 23).

### Status: mode pemeliharaanAwal 2026: AutoGen v0.7.x stabil untuk penelitian dan pembuatan prototipe. Microsoft telah mengalihkan pengembangan aktif ke Microsoft Agent Framework (pratinjau publik 1 Oktober 2025; 1.0 GA ditargetkan pada akhir Q1 2026). Pola AutoGen berjalan dengan rapi - model aktor adalah ide yang tahan lama.

## Build

`code/main.py` mengimplementasikan runtime aktor stdlib:

- `Message` — mengetik muatan dengan `sender`, `recipient`, `topic`, `body`.
- `Actor` — abstrak dengan `receive(message, runtime)`.
- `Runtime` — perulangan peristiwa dengan antrean bersama, pengiriman, isolasi kegagalan.
- Demo dua aktor: `ReviewerAgent` code ulasan, `ChecklistAgent` menjalankan daftar periksa; mereka bertukar pesan sampai konsensus.

Jalankan:

```
python3 code/main.py
```

Jejak tersebut menunjukkan pengiriman pesan, simulasi kegagalan pada satu aktor yang tidak menyebabkan crash pada aktor lainnya, dan konvergensi pada keputusan bersama.

## Pakai

- **AutoGen v0.4/v0.7** (pemeliharaan) — stabil untuk penelitian, pembuatan prototipe, pola multi-agen.
- **Microsoft Agent Framework** (pratinjau publik) — jalur maju; ide model aktor yang sama dalam API yang diperbarui.
- **Topologi gerombolan LangGraph** (Lesson 13) — pola serupa melalui handoff alat bersama.
- **Waktu proses aktor khusus** — saat kamu memerlukan transportasi khusus (NATS, RabbitMQ, gRPC).

## Kirim

`outputs/skill-actor-runtime.md` menghasilkan runtime aktor minimal ditambah template tim (RoundRobin atau Selector) untuk tugas multi-agen tertentu.

## Latihan

1. Tambahkan antrian surat mati: ketika pawang mengangkat, parkirkan pesan yang gagal untuk diperiksa manusia. Seberapa sering DLQ tertembak di mainan kamu?
2. Implementasikan `SelectorGroupChat`: aktor pemilih memilih siapa yang memproses pesan berikutnya berdasarkan status percakapan.
3. Tambahkan transportasi terdistribusi: tukar antrian dalam proses dengan server JSON-over-HTTP sehingga aktor dapat berjalan dalam proses terpisah.
4. Hubungkan rentang OTel per pesan (atau stand-in tanpa operasi). Keluarkan `gen_ai.agent.name`, `gen_ai.operation.name` per Lesson 23.
5. Baca postingan arsitektur AutoGen v0.4. Pindahkan mainan kamu ke `autogen_core` API yang asli. Apa yang kamu lewatkan yang penting dalam produksi?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Aktor | "Agen" | Status pribadi + kotak masuk + pengendali; tidak ada memori bersama |
| Pesan | "Acara" | Muatan yang diketik; satu-satunya cara aktor berinteraksi |
| Kotak masuk | "Kotak Surat" | Antrean pesan tertunda per aktor |
| Waktu proses | "Agen tuan rumah" | Perulangan peristiwa yang merutekan pesan dan mengisolasi kegagalan |
| Topik | "Pipeline" | Dinamakan rute terbitkan-langganan antar aktor |
| Isolasi kesalahan | "Biarkan saja crash" | Kegagalan satu aktor tidak akan merugikan aktor lainnya |
| Obrolan Grup RoundRobin | "Tim dengan rotasi tetap" | Agen bergiliran memesan |
| SelectorGroupChat | "Tim dengan rute konteks" | Pilihan pemilih siapa yang berikutnya |
| Magentik-Satu | "Tim referensi" | Pasukan multi-agen untuk web + code + file |

## Bacaan Lanjutan

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — postingan desain ulang
- [Ikhtisar LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — alternatif berbentuk grafik
- [Konvensi semantik OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — mencakup emisi AutoGen secara default
