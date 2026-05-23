# Capstone 17 — Personal AI Tutor (Adaptif, Multimodal, dengan Memori)

> Khanmigo (Khan Academy), Duolingo Max, Google LearnLM / Gemini for Education, Quizlet Q-Chat, dan Synthesis Tutor semuanya memberikan bimbingan belajar multimodal adaptif dalam skala besar pada tahun 2026. Bentuk umumnya adalah kebijakan Socrates (jangan pernah membuang jawabannya begitu saja), model pembelajar yang diperbarui setelah setiap interaksi (gaya penelusuran pengetahuan Bayesian), input suara + teks + foto-matematika, pengambilan grafik kurikulum, penjadwalan pengulangan dengan distance tertentu, dan filter keamanan keras untuk konten sesuai usia. Puncaknya adalah mengirimkan tutor khusus mata lesson (aljabar K-12 atau intro Python), menjalankan studi kemanjuran selama dua minggu dengan 10 pelajar, dan lulus audit keamanan konten.

**Type:** Batu penjuru
**Language:** Python (backend, model pelajar), TypeScript (aplikasi web), SQL (grafik kurikulum melalui Postgres + Neo4j)
**Prerequisites:** Fase 5 (NLP), Fase 6 (pidato), Fase 11 (rekayasa LLM), Fase 12 (multimoda), Fase 14 (agen), Fase 17 (infrastruktur), Fase 18 (keselamatan)
**Fase yang dilakukan:** P5 · P6 · P11 · P12 · P14 · P17 · P18
**Waktu:** 30 jam

## Masalah

Bimbingan belajar adaptif dulunya merupakan bidang penelitian teknologi pendidikan. Pada tahun 2026 ini menjadi produk konsumen. Khanmigo dikerahkan di sebagian besar distrik sekolah AS. Duolingo Max mencapai puluhan juta MAU. LearnLM / Gemini for Education dari Google mendukung bimbingan belajar di Google Classroom. Quizlet Q-Chat berada di samping kartu flash. Synthesis Tutor menjadi viral dengan tutor untuk anak-anak yang penasaran. Elemen umum: input multimodal (mengetik, berbicara, memotret persamaan), pedagogi Socrates (tanyakan dulu, jelaskan nanti), model pembelajar yang diperbarui setelah setiap interaksi, dan keamanan ketat sesuai usia.

kamu akan membuat salah satunya untuk kelompok tertentu. Bilah pengukurannya adalah studi kemanjuran aktual: skor pra-tes dan pasca-tes selama dua minggu dengan 10 peserta didik. Loop suara harus terasa alami (sub-tumpukan capstone 03). Memori harus menghormati privasi. Filter keamanan harus melewati tim merah yang sadar COPPA untuk K-12.

## Konsep

Empat komponen. **Kebijakan pengajar** adalah lingkaran Socrates: ketika pelajar meminta jawabannya, kebijakan tersebut menanyakan pertanyaan utama; ketika mereka melakukannya dengan benar, ia berpindah ke konsep berikutnya; ketika mereka terjebak, ia menawarkan petunjuk perancah. **Model pembelajar** adalah penelusuran pengetahuan Bayesian (atau varian sederhana) yang memperbarui probabilitas penguasaan per node kurikulum setelah setiap interaksi. **Grafik Kurikulum** adalah konsep Neo4j dengan tepi prasyarat; kebijakan tersebut memandu grafik untuk memilih konsep berikutnya. **Memori** adalah penyimpanan episodik + semantik (gaya agenmemori) yang menyimpan interaksi, kesalahan, dan preferensi di masa lalu.

UX bersifat multimodal. Input teks untuk jawaban yang diketik. Input suara melalui LiveKit + Whisper (gunakan kembali batu penjuru 03). Input foto untuk soal matematika melalui dots.ocr atau PaliGemma 2. Output suara melalui Cartesia Sonic-2. Keamanan menggunakan Llama Guard 4 ditambah filter sesuai usia (memblokir konten dewasa, kekerasan, menyakiti diri sendiri) dan kebijakan penyimpanan memori yang sadar akan COPPA.

Studi kemanjuran adalah hasil yang dapat dicapai. 10 peserta didik, pre-test dan post-test, dua minggu. Laporkan perolehan pembelajaran delta dan interval kepercayaan. Bandingkan dengan baseline non-adaptif (konten yang sama disampaikan secara linear tanpa kebijakan tutor).

## Arsitektur

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## Tumpukan- Pilihan mata lesson: aljabar K-12 atau intro Python (pilih salah satu untuk kedalamannya)
- Kebijakan tutor: LangGraph melalui Claude Sonnet 4.7 (dengan cache cepat)
- Model pembelajar: Penelusuran pengetahuan Bayesian (klasik) atau FSRS untuk penspasian
- Grafik Kurikulum: Konsep Neo4j + tepi prasyarat + konten OER
- Memori: vector persisten gaya agentmemory + penyimpanan episodik + semantik
- Suara: Agen LiveKit 1.0 + Cartesia Sonic-2 (menggunakan kembali sub-tumpukan batu penjuru 03)
- Foto matematika: dot.ocr atau PaliGemma 2 untuk pengenalan persamaan
- Keamanan: Llama Guard 4 + filter khusus sesuai usia
- Eval: Pembuatan pertanyaan tingkat mekar, rangkaian tes sebelum/sesudah, alat studi kemanjuran

## Build

1. **Grafik Kurikulum.** Buat Neo4j yang terdiri dari 50-150 simpul konsep (misalnya, aljabar K-12 dari "garis bilangan" ke "rumus kuadrat") dengan tepi yang disyaratkan. Lampirkan konten OER per node (Buka Teks, OpenStax).

2. **Model pembelajar.** Inisialisasi penelusuran pengetahuan Bayesian dengan prior: tebak, selipkan, tingkatkan pembelajaran. Perbarui penguasaan per konsep setelah setiap interaksi. Bertahan per pelajar.

3. **Kebijakan tutor.** LangGraph dengan node: `read_signal` (apakah jawaban pelajar benar / sebagian / macet?), `select_concept` (grafik kurikulum berjalan dengan memilih konsep prioritas tertinggi), `scaffold` (prompt Socrates), `update_mastery`.

4. **Memori.** Setiap interaksi menulis ke penyimpanan episodik. Kesalahan dan preferensi dipromosikan ke memori semantik. Kebijakan penyimpanan sadar COPPA: hapus otomatis setelah 1 tahun, dapat diakses oleh orang tua.

5. **Jalur suara.** Pekerja Agen LiveKit melekat pada kebijakan tutor. ASR melalui Whisper-v3-turbo. TTS melalui Cartesia Sonic-2. Didukung tongkang (gunakan kembali mekanik batu penjuru 03).

6. **Jalur matematika-foto.** Unggah atau ambil gambar; jalankan dots.ocr atau PaliGemma 2 untuk mengenali persamaannya; umpan ke tutor sebagai input terstruktur.

7. **Keamanan.** Setiap output model lolos Llama Guard 4 + filter sesuai usia (memblokir tindakan menyakiti diri sendiri, konten dewasa, kekerasan). Akses memori dicakup oleh ID pelajar; permukaan akses orang tua untuk dihapus.

8. **Studi kemanjuran.** 10 peserta didik, pra-tes (standar dasar 30 pertanyaan), interaksi tutor selama dua minggu (3 sesi/minggu), pasca-tes. Bandingkan dengan kelompok dasar non-adaptif yang terdiri dari 10 pelajar pada konten yang sama.

9. **Laporan kemajuan mingguan.** Per pelajar, buat ringkasan PDF secara otomatis tentang topik yang dieksplorasi, lintasan penguasaan, dan rekomendasi langkah selanjutnya.

## Pakai

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## Kirim

`outputs/skill-ai-tutor.md` adalah hasil yang dapat dicapai. Tutor adaptif khusus mata lesson dengan input multimodal, model pembelajar, memori, keamanan, dan kemanjuran terukur.

| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | Pembelajaran memperoleh delta | Delta pra/pasca tes dalam studi dua minggu yang melibatkan 10 peserta didik |
| 20 | Kesetiaan Sokrates | Skor rubrik pada sample transkrip |
| 20 | UX multimodal | Koherensi suara + foto + teks ujung ke ujung |
| 20 | Postur keselamatan + privasi | Tingkat kelulusan Llama Guard 4 + retensi sadar COPPA |
| 15 | Luasnya kurikulum dan kualitas grafik | Cakupan konsep + konsistensi grafik prasyarat |
| **100** | | |

## Latihan

1. Jalankan pembelajaran efikasi dengan dan tanpa model pembelajar adaptif (urutan konsep acak). Laporkan delta tersebut. Harapkan adaptif untuk menang, tetapi ukurannya adalah angka yang menarik.

2. Tambahkan penyelidikan multimodal: pertanyaan konsep yang sama disampaikan seperti teks, suara, dan foto. Ukur apakah pembelajar menyatu lebih cepat dengan modalitas yang mereka sukai.3. Build dasbor induk: topik yang dipraktikkan, lintasan penguasaan, konsep yang akan datang, peristiwa keselamatan (apa pun pagar pembatas yang terkena dampak). selaras dengan COPPA.

4. Tambahkan mode peralihan bahasa: tutor menerima input bahasa Spanyol dan mengajar dalam bahasa Spanyol. Ukur cakupan X-Guard.

5. Tekankan privasi memori: verifikasi bahwa pelajar A tidak dapat melihat data pelajar B bahkan melalui serangan penyerapan ulang klip suara. Catat upaya akses dan peringatan.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Kebijakan Sokrates | "Tanya, jangan dibuang" | Tutor mengajukan pertanyaan yang mengarahkan daripada memberikan jawaban |
| Penelusuran pengetahuan Bayesian | "BKT" | Persamaan model pembelajar klasik untuk probabilitas penguasaan per konsep |
| FSRS | "Penjadwal Pengulangan Spasi Gratis" | Penjadwal pengulangan spasi 2024, lebih baik dari SM-2 |
| Grafik Kurikulum | "Konsep DAG" | Neo4j konsep dengan tepi prasyarat |
| Memori episodik | "Log per interaksi" | Setiap interaksi disimpan untuk diambil nanti |
| Memori semantik | "Penyimpanan pola yang dipelajari" | Kesalahan dan preferensi yang dipadatkan dipromosikan dari episodik |
| KOPA | "Hukum privasi anak" | Undang-undang AS membatasi pengumpulan data dari anak-anak di bawah 13 tahun |

## Bacaan Lanjutan

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — referensi konsumen tutor K-12
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — referensi tutor pembelajaran bahasa
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — model referensi yang dihosting
- [Quizlet Q-Chat](https://quizlet.com) — referensi alternatif
- [Tutor Sintesis](https://www.synthesis.com) — referensi permulaan
- [Algoritma FSRS](https://github.com/open-spaced-repetition/fsrs4anki) — penjadwal pengulangan spasi
- [Pelacakan Pengetahuan Bayesian](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — model pembelajar klasik
- [Agen LiveKit](https://github.com/livekit/agents) — tumpukan suara
