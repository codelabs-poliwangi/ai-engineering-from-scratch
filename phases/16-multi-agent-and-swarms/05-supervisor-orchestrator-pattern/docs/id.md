# Pola Supervisor / Orchestrator-Pekerja

> Satu agen utama merencanakan dan mendelegasikan; pekerja khusus mengeksekusi dalam konteks paralel dan melaporkan kembali. Ini adalah pola di balik sistem Penelitian Anthropic (Claude Opus 4 sebagai pemimpin, Soneta 4 sebagai subagen), diukur sebesar +90,2% dibandingkan agen tunggal Opus 4 pada evaluasi penelitian internal. Pos teknik Anthropic melaporkan bahwa 80% varian di BrowserComp dijelaskan oleh penggunaan token saja — multi-agen menang terutama karena setiap subagen mendapat jendela konteks baru. Lesson ini membangun pola supervisor dari primitif dan mencakup lesson teknik tahun 2026 dari penerapan produksi.

**Type:** Learn + Build
**Language:** Python (stdlib, `threading`)
**Prerequisites:** Fase 16 · 04 (Model Primitif)
**Waktu:** ~75 menit

## Masalah

Penelitian adalah tugas prototipe yang gagal dilakukan oleh sistem agen tunggal. kamu bertanya "apa yang berubah dalam sistem multi-agen antara tahun 2023 dan 2026?" Seorang agen membaca lima makalah secara berurutan, mengisi separuh konteksnya dengan teksnya, dan kemudian harus mempertimbangkan semuanya secara bersamaan. Ia melupakan kertas pertama pada saat mencapai kertas kelima. Itu tidak bisa diparalelkan.

Pola supervisor memperbaiki hal ini: satu agen utama merencanakan pencarian, mendelegasikan setiap sub-pertanyaan kepada seorang pekerja, dan melakukan sintesis. Setiap pekerja mendapat jendela 200 ribu tokennya sendiri untuk pertanyaan sempit. Pemimpin tidak pernah melihat dokumen mentah — hanya ringkasan pekerja.

Sistem Penelitian produksi Anthropic melaporkan +90,2% pada evaluasi penelitian internal vs satu Opus 4. Postingan yang sama mencatat bahwa 80% varian BrowserComp dijelaskan oleh *penggunaan token saja*. Konteks baru per subagen adalah mekanisme utamanya.

## Konsep

### Polanya

```
                 ┌──────────────┐
                 │   Lead       │  plans, decomposes,
                 │  (Opus 4)    │  synthesizes
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

Pemimpin tidak pernah membaca bahan mentahnya. Para pekerja tidak pernah melihat pekerjaan satu sama lain sampai timbalnya disintesis. Setiap panah adalah penyerahan dengan artefak sempit.

### Mengapa ia menang

Tiga mekanisme:

1. **Konteks baru per subagen.** Seorang pekerja yang menjelajahi "warisan FIPA-ACL" tidak membawa 40 ribu token yang dihabiskan untuk perencanaan. Ia mendapat jendela 200k untuk satu pertanyaan.
2. **Spesialisasi melalui prompt.** Prompt pemimpin adalah "decomposition dan sintesis", bukan "penelitian". Prompt setiap pekerja sempit: "temukan apa yang berubah di X." Prompt yang terfokus menghasilkan output yang terfokus.
3. **Paralelisme.** Pekerja berjalan secara bersamaan. Waktu jam dinding kira-kira `max(worker_times) + plan + synthesis`, bukan `sum(worker_times)`.

### Lesson teknik (Anthropic 2025)

Postingan Anthropic mencantumkan beberapa lesson produksi yang masih relevan di tahun 2026:

- **Meningkatkan upaya untuk mengatasi kompleksitas kueri.** Kueri sederhana: satu agen, 3-10 panggilan alat. Kueri kompleks: 10+ agen. Pemimpin harus memperkirakan hal ini, bukan penelepon.
- **Luas, lalu sempit.** Uraikan menjadi sub-pertanyaan yang luas terlebih dahulu, lalu masukkan lebih banyak pekerja per sub-pertanyaan jika jawabannya memerlukan kedalaman.
- **Penerapan pelangi.** Agen sudah berjalan lama dan memiliki status. Biru-hijau tradisional tidak berfungsi. Anthropic menggunakan pelangi: peluncuran versi baru secara bertahap sementara versi lama habis.
- **Penggunaan token mendominasi.** Multi-agen adalah ~15× token agen tunggal. Hanya jalankan ketika nilai tugas sesuai dengan biayanya.

### Giliran LangGraphLangGraph awalnya mengirimkan perpustakaan `langgraph-supervisor` dengan helper `create_supervisor` tingkat tinggi. Pada tahun 2025 LangChain memindahkan rekomendasi untuk menerapkan pola supervisor melalui pemanggilan alat secara langsung, karena pemanggilan alat memberikan kontrol lebih besar atas *apa yang dilihat supervisor* (rekayasa konteks). Perpustakaan masih berfungsi; dokumen sekarang merekomendasikan formulir pemanggilan alat.

### Mode kegagalan

- **Pemimpin berhalusinasi mengenai rencana tersebut.** Jika pemimpin menghasilkan sub-pertanyaan yang tidak menguraikan pertanyaan sebenarnya, pekerja akan melakukan penelitian tepat pada sasaran yang salah.
- **Pekerja melakukan eksplorasi berlebihan.** Tanpa batasan cakupan yang jelas, pekerja melampaui sub-pertanyaan yang ditugaskan kepada mereka dan mencemari langkah sintesis.
- **Konflik sintesis.** Dua pekerja mengembalikan fakta yang bertentangan. Pemimpin harus bertanya kembali (menambahkan putaran) atau mencatat ketidaksepakatan secara eksplisit. Memilih satu sisi secara diam-diam adalah kegagalan terburuk: pengguna tidak pernah tahu terjadi perselisihan.

### Ketika supervisor salah

- **Tugas berurutan.** Jika langkah 2 benar-benar membutuhkan output langkah 1, paralelisme tidak menghasilkan apa-apa. Gunakan pipeline pipa (CrewAI Sequential, grafik linier LangGraph).
- **Pertanyaan sederhana.** Agen tunggal menanganinya lebih cepat dan lebih murah. Gunakan pemeriksaan "skala upaya" pemimpin sebelum memunculkan pekerja.
- **Determinisme yang ketat.** Supervisor menggunakan delegasi pilihan LLM. Grafik statis lebih baik ketika audit/pemutaran ulang lebih penting daripada kemampuan beradaptasi.

## Build

`code/main.py` mengimplementasikan supervisor dari tiga pekerja paralel menggunakan `threading`. Pemimpin menguraikan kueri menjadi sub-pertanyaan, pekerja menjalankan setiap sub-pertanyaan secara bersamaan, dan pemimpin melakukan sintesis. Tidak ada LLM nyata — pekerja dibuat untuk mensimulasikan pengambilan dan ringkasan.

Struktur kunci:

- `Lead.plan(query)` membagi kueri menjadi 3 sub-pertanyaan.
- `Worker.run(sub_q)` mengembalikan ringkasan palsu (bisa berupa agen yang menggunakan alat apa pun dalam produksi).
- `Lead.run(query)` memulai pekerja dalam rangkaian, penggabungan, dan sintesis.

Jalankan:

```
python3 code/main.py
```

Output menunjukkan rencana, pelacakan pekerja paralel dengan stempel waktu mulai/berakhir, dan sintesis akhir. kamu dapat melihat kemenangan jam dinding: tiga pekerja berdurasi 0,3 detik berjalan dalam ~0,35 detik, bukan 0,9.

## Pakai

`outputs/skill-supervisor-designer.md` mengambil kueri pengguna dan menghasilkan desain pola supervisor: system prompt prospek, peran pekerja, aturan decomposition sub-pertanyaan, dan templat sintesis. Gunakan ini sebelum membangun sistem agen bergaya penelitian baru.

## Kirim

Daftar periksa sebelum menerapkan pola supervisor:

- **Pemasangan model.** Memimpin pada model tingkat penalaran (kelas Opus, kelas `o3`). Pekerja dengan model yang lebih cepat dan lebih murah (Sonnet, `o4-mini`).
- **Waktu pekerja habis.** Pekerja mana pun yang melebihi 2× waktu proses median akan terbunuh; pemimpinnya akan muncul kembali dengan cakupan yang lebih sempit atau melanjutkan tanpanya.
- **Batas token per pekerja.** Batas tegas (misalnya 10× input sintesis yang diharapkan) mencegah pekerja yang melarikan diri menghabiskan anggaran.
- **Kemampuan untuk diamati.** Telusuri rencana pemimpin, panggilan alat setiap pekerja, dan sintesisnya. Ini adalah dasar untuk segala proses debug post-hoc.
- **Peluncuran pelangi.** Agen stateful yang sudah berjalan lama memerlukan transisi versi bertahap, bukan hot swap.

## Latihan1. Jalankan `code/main.py`, lalu modifikasi lead untuk menghasilkan 5 pekerja, bukan 3. Amati efek jam dinding. Berapa jumlah pekerja yang menghasilkan overhead overhead melebihi penghematan paralel dalam demo ini?
2. Menerapkan batas waktu pekerja: bunuh pekerja mana pun yang berjalan lebih dari 0,5 detik dan minta pimpinan mensintesis hasil yang tersisa. Observabilitas apa yang perlu kamu ketahui bahwa seorang pekerja diberhentikan?
3. Tambahkan langkah deteksi konflik pada sintesis pemimpin: jika dua pekerja memberikan jawaban yang bertentangan, pemimpin akan mencatat ketidaksepakatan tersebut daripada memilih satu. Bagaimana kamu mendeteksi kontradiksi tanpa menelepon LLM?
4. Baca postingan Rekayasa Sistem Penelitian Anthropic. Sebutkan tiga praktik yang perlu diterapkan oleh demo mainan ini agar dapat dijalankan dalam produksi.
5. Bandingkan `create_supervisor` (warisan) LangGraph vs rekomendasi pemanggilan alat yang baru. Mana yang memberi kamu kendali lebih baik atas apa yang dilihat supervisor? Mengapa Anthropic secara eksplisit hanya meneruskan sub-jawaban dan bukan konteks pekerja mentah ke dalam sintesis?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Pengawas | "Agen utama" | Agen orkestra yang merencanakan, mendelegasikan, dan mensintesis. Tidak melakukan pekerjaan itu sendiri. |
| Pekerja | "Subagen" | Agen terfokus yang dipanggil oleh supervisor dengan cakupan sempit dan jendela konteksnya sendiri. |
| Pekerja orkestra | "Pola pengawas" | Hal yang sama, nama yang berbeda. Literatur tahun 2026 menggunakan keduanya. |
| Konteks segar | "Bersihkan jendela" | Konteks pekerja dimulai dari system prompt dan pertanyaan yang ditetapkan, bukan riwayat pemimpin. |
| Penyebaran pelangi | "Peluncuran bertahap" | Agen stateful yang sudah berjalan lama memerlukan versi drain-and-replace, bukan biru-hijau. |
| Dominasi token | "Konteks adalah variabelnya" | 80% varians evaluasi penelitian berasal dari total token yang digunakan, bukan pilihan model, menurut Anthropic. |
| Skala upaya | "Agen pencocokan memperhitungkan kompleksitas" | Prospek memperkirakan kesulitan kueri, menghasilkan 1 vs 10+ pekerja yang sesuai. |
| Konflik sintesis | "Pekerja tidak setuju" | Dua pekerja mengembalikan fakta yang bertentangan; pemimpin harus memunculkan ketidaksepakatan, bukan diam-diam memilih salah satu. |

## Bacaan Lanjutan

- [Rekayasa antropik — Cara kami membangun sistem penelitian multi-agen](https://www.anthropic.com/engineering/multi-agent-research-system) — referensi produksi untuk pola supervisor
- [Alur kerja dan agen LangGraph](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — supervisor pemanggil alat kini menjadi formulir yang direkomendasikan
- [Referensi supervisor LangGraph](https://reference.langchain.com/python/langgraph-supervisor) — helper lama, masih digunakan pada produksi tahun 2026
- [Buku masak OpenAI — Agen Pengorganisasian: Rutinitas dan Handoff](https://developers.openai.com/cookbook/examples/orchestrating_agents) — varian supervisor berbasis handoff
