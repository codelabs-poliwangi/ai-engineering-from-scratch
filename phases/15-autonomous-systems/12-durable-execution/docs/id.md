# Agen Latar Belakang yang Berjalan Lama: Eksekusi yang Tahan Lama

> Agen produksi jangka panjang tidak berjalan di `while True`. Setiap panggilan LLM menjadi aktivitas dengan pos pemeriksaan, percobaan ulang, dan pemutaran ulang. Integrasi OpenAI Agents SDK Temporal dimulai pada GA pada Maret 2026. Rutinitas Code Claude (Antropik) menjalankan pemanggilan Code Claude terjadwal tanpa proses lokal yang persisten. Sesi dijeda berdasarkan input manusia, bertahan dalam penerapan, dan dilanjutkan dari pos pemeriksaan terbaru yang dikunci oleh `thread_id`. Di balik ergonomi baru terdapat pola lama — orkestrasi alur kerja — dengan satu input baru: panggilan LLM sebagai aktivitas non-deterministik yang harus diputar ulang secara deterministik dalam pemulihan.

**Type:** Learn
**Language:** Python (stdlib, mesin status eksekusi tahan lama minimal)
**Prerequisites:** Fase 15 · 10 (Mode izin), Fase 15 · 01 (Agen cakrawala panjang)
**Waktu:** ~60 menit

## Masalah

Pertimbangkan agen yang beroperasi selama empat jam. Ia memanggil tiga alat, meminta pengguna dua kali, dan membuat empat puluh panggilan LLM. Di tengah jalan, host yang dijalankan saat reboot. Apa yang terjadi?

- Dalam loop `while True` yang naif: semuanya hilang. Prosesnya dimulai ulang dari awal. Ketiga panggilan alat (dengan efek samping nyata) dijalankan lagi. Pengguna diminta lagi untuk hal-hal yang telah mereka setujui. Empat puluh panggilan LLM ditagih ulang.
- Dengan eksekusi yang tahan lama: proses dilanjutkan dari pos pemeriksaan terbaru. Kegiatan yang sudah selesai tidak dilaksanakan kembali; hasilnya diputar ulang dari log tahan lama. Pengguna tidak menyetujui kembali hal-hal yang telah mereka setujui. Panggilan LLM yang sudah dilakukan tidak ditagih ulang.

Ini adalah pola alur kerja yang sama yang telah dikirimkan mesin selama satu dekade (Temporal, Cadence, Uber's Cherami). Apa yang baru adalah bahwa panggilan LLM sekarang merupakan suatu jenis aktivitas — non-deterministik, mahal, dengan efek samping — dan sangat cocok dengan pola ini.

Tema lesson yang sedang berjalan: keandalan jangka panjang menurun (METR mengamati "degradasi 35 menit" — tingkat keberhasilan turun kira-kira secara kuadratik dengan cakrawala). Eksekusi yang tahan lama memungkinkan pengoperasian yang lebih lama dari dukungan profil keandalan, yang merupakan cara baru untuk gagal dengan aman jika desainnya benar dan tidak aman jika desainnya salah.

## Konsep

### Aktivitas, alur kerja, dan pemutaran ulang

- **Alur Kerja**: code orkestrasi deterministik. Mendefinisikan urutan aktivitas, cabang, penantian. Harus bersifat deterministik sehingga dapat diputar ulang dari log peristiwa tanpa perbedaan yang mengejutkan.
- **Aktivitas**: unit kerja non-deterministik yang berpotensi gagal. Panggilan LLM, panggilan alat, penulisan file, permintaan HTTP. Tiap aktivitas dicatat beserta masukannya dan (setelah selesai) keluarannya.
- **Log peristiwa**: penyimpanan pendukung yang tahan lama. Setiap aktivitas dimulai, selesai, gagal, coba lagi, dan setiap keputusan alur kerja dicatat.
- **Putar Ulang**: saat pemulihan, code alur kerja dijalankan kembali dari awal; setiap aktivitas yang sudah selesai mengembalikan hasil lognya tanpa mengeksekusi ulang. Hanya kegiatan yang belum selesai saja yang benar-benar dijalankan.

Bentuknya sama dengan React yang merender ulang DOM virtual, atau Git yang membangun kembali pohon kerja dari commit. Determinisme dalam orkestrator inilah yang membuat daya tahan menjadi murah.

### Mengapa panggilan LLM sesuai dengan polanya

Panggilan LLM adalah:
- Non-deterministik (suhu > 0; bahkan suhu 0 melayang di seluruh versi model).
- Mahal (uang dan latensi).
- Berpotensi gagal (batas kecepatan, batas waktu).
- Berefek samping (jika mereka memanggil alat).Ini persisnya profil aktivitasnya. Menggabungkan setiap panggilan LLM sebagai aktivitas memberi kamu percobaan ulang dengan backoff eksponensial, pemeriksaan saat memulai ulang, dan jejak yang dapat diputar ulang untuk proses debug.

### Pos pemeriksaan dikunci oleh `thread_id`

LangGraph, Microsoft Agent Framework, Cloudflare Durable Objects, dan Claude Code Routines semuanya digabungkan pada bentuk API yang sama: `thread_id` (atau yang setara) mengidentifikasi sesi; setiap transisi status tetap ada di backend (default PostgreSQL, SQLite untuk dev, Redis untuk cache); melanjutkan membaca pos pemeriksaan terbaru.

Pilihan backend penting:

- **PostgreSQL**: tahan lama, dapat dikueri, dan bertahan dalam penerapan. Default untuk LangGraph.
- **SQLite**: khusus pengembang lokal; kehilangan data di seluruh host.
- **Redis**: cepat namun singkat kecuali AOF/snapshot dikonfigurasi.
- **Objek Tahan Lama Cloudflare**: didistribusikan secara transparan; dicakup oleh kunci unik; bertahan selama berjam-jam hingga berminggu-minggu.

### Input manusia sebagai negara bagian kelas satu

Usulkan lalu lakukan (Lesson 15) membutuhkan kondisi "menunggu manusia" yang tahan lama. Alur kerja dijeda, antrean eksternal menahan permintaan yang tertunda, dan persetujuan dilanjutkan tepat pada titik tersebut. Tanpa daya tahan, ini adalah upaya terbaik; dengan itu, persetujuan semalam tiba dan alur kerja dimulai di pagi hari.

### Degradasi 35 menit

METR mengamati bahwa setiap kelas agen yang diukur menunjukkan penurunan keandalan melebihi ~35 menit operasi berkelanjutan. Menggandakan durasi tugas kira-kira akan melipatgandakan tingkat kegagalan. Eksekusi yang tahan lama tidak memperbaiki hal ini; ini memungkinkan kamu berjalan lebih lama dari yang didukung profil keandalan. Pola amannya adalah menggabungkan daya tahan dengan pos pemeriksaan yang memerlukan HITL baru saat masuk kembali, dan dengan tombol pemutus anggaran (Lesson 13) yang membatasi total komputasi berapa pun waktu jam dinding.

### Ketika eksekusi yang tahan lama adalah jawaban yang salah

- Berjalan lebih pendek dari beberapa menit tanpa input manusia. Biaya overhead > manfaat.
- Pengambilan informasi yang benar-benar hanya baca.
- Tugas di mana kebenaran memerlukan end-to-end dalam satu jendela konteks (beberapa tugas penalaran; beberapa generasi one-shot).

## Pakai

`code/main.py` mengimplementasikan mesin eksekusi tahan lama minimal di stdlib Python. Ini mendukung:

- dekorator `@activity` yang mencatat input dan output ke log peristiwa JSON.
- Fungsi alur kerja yang mengurutkan aktivitas.
- Fungsi `run_or_replay(workflow, event_log)` yang memutar ulang aktivitas yang telah selesai tanpa menjalankannya kembali.

Pengemudi menyimulasikan alur kerja tiga aktivitas, mengalami error di tengah jalan, dan menunjukkan (a) percobaan ulang yang naif yang mengeksekusi ulang semuanya versus (b) pemutaran ulang yang hanya menjalankan aktivitas yang hilang.

## Kirim

`outputs/skill-durable-execution-review.md` meninjau usulan penerapan agen jangka panjang untuk bentuk eksekusi tahan lama yang benar: aktivitas, determinisme, backend pos pemeriksaan, status input manusia, dan kebijakan HITL-on-resume.

## Latihan

1. Jalankan `code/main.py`. Amati perbedaan jumlah eksekusi aktivitas antara percobaan ulang naif dan pemutaran ulang. Ubah titik kerusakan dan tunjukkan perubahan jumlah pemutaran ulang yang sesuai.

2. Ubah mesin mainan untuk menggunakan `thread_id` secara eksplisit. Simulasikan dua sesi bersamaan yang berbagi mesin dan pastikan log peristiwanya tidak bertabrakan.

3. Lakukan satu aktivitas di mesin mainan. Perkenalkan non-determinisme (stempel waktu jam dinding di dalam keputusan alur kerja). Tunjukkan perbedaan pada tayangan ulang. Jelaskan bagaimana mesin nyata menangani hal ini (pendaftaran efek samping, `Workflow.now()` API).4. Baca postingan LangChain "Runtime di belakang agen dalam produksi". Cantumkan setiap status di mana runtime tetap ada dan sebutkan mode kegagalan yang dicakup masing-masing status.

5. Rancang kebijakan pos pemeriksaan untuk tugas pengkodean otonom 6 jam. Di mana kamu pos pemeriksaan? Seperti apa resume-on-crash itu? Apa yang memerlukan HITL baru?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| Alur Kerja | "Skrip Agen" | Code orkestrasi deterministik; dapat diputar ulang dari log peristiwa |
| Aktivitas | "Sebuah langkah" | Unit non-deterministik (Panggilan LLM, Panggilan Alat); dicatat sebelum dan sesudah |
| Catatan peristiwa | "Toko pendukung" | Rekor tahan lama dari setiap transisi negara |
| Putar ulang | "Lanjutkan" | Jalankan kembali alur kerja; aktivitas yang selesai mengembalikan hasil yang dicatat tanpa eksekusi ulang |
| Pos pemeriksaan | "Simpan poin" | Status bertahan yang dikunci oleh thread_id; kemenangan terbaru di resume |
| thread_id | "Kunci sesi" | Pengidentifikasi yang mencakup status tahan lama |
| Degradasi 35 menit | "Pembusukan keandalan" | METR: tingkat keberhasilan turun ~secara kuadratik dengan horizon |
| Non-determinisme | "Melayang saat diputar ulang" | Jam dinding, acak, output LLM; harus didaftarkan sebagai efek samping |

## Bacaan Lanjutan

- [Antropis — SDK Agen Code Claude: loop agen](https://code.claude.com/docs/en/agent-sdk/agent-loop) — semantik anggaran, putaran, dan resume.
- [Microsoft — Agent Framework: human-in-the-loop dan checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — bentuk RequestInfoEvent.
- [LangChain — Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — persyaratan runtime yang konkrit.
- [SDK Agen OpenAI + Integrasi temporal (pengumuman Trigger.dev)](https://trigger.dev) — bentuk aktivitas untuk panggilan LLM.
- [Anthropic — Mengukur otonomi agen dalam praktiknya](https://www.anthropic.com/research/measuring-agent-autonomy) — referensi degradasi berdurasi 35 menit.
