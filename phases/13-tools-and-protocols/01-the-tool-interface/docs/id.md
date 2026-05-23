# Antarmuka Alat — Mengapa Agen Membutuhkan I/O Terstruktur

> Model bahasa menghasilkan token. Sebuah program mengambil tindakan. Kesenjangan antara keduanya adalah antarmuka alat: kontrak yang memungkinkan model meminta suatu tindakan dan host mengeksekusinya. Setiap tumpukan tahun 2026 — fungsi yang memanggil OpenAI, Anthropic, dan Gemini; `tools/call` MCP; Bagian tugas A2A — adalah pengkodean berbeda dari loop empat langkah yang sama. Lesson ini memberi nama loop dan menunjukkan mesin minimum untuk menjalankannya.

**Type:** Learn
**Language:** Python (stdlib, tanpa LLM)
**Prerequisites:** Fase 11 (API penyelesaian LLM)
**Waktu:** ~45 menit

## Tujuan Pembelajaran

- Jelaskan mengapa LLM yang hanya dapat menghasilkan teks tidak dapat dengan sendirinya mengambil tindakan terhadap dunia nyata.
- Gambarkan loop pemanggilan alat empat langkah (jelaskan → putuskan → jalankan → amati) dan sebutkan siapa yang memiliki setiap langkah.
- Tulis deskripsi alat sebagai tiga bagian: nama, input Skema JSON, dan fungsi eksekutor deterministik.
- Membedakan alat yang murni dan yang menimbulkan efek samping serta menyatakan mengapa pemisahan itu penting demi keselamatan.

## Masalah

LLM memancarkan distribusi probabilitas pada token berikutnya. Itu adalah seluruh permukaan output. Jika kamu bertanya kepada model obrolan "bagaimana cuaca di Bengaluru saat ini", ia dapat menulis kalimat yang masuk akal, namun tidak dapat menghubungi API cuaca. Kalimat tersebut mungkin benar secara kebetulan atau sudah basi selama tiga hari.

Menutup celah tersebut adalah tujuan dari antarmuka alat. Program host — runtime agen kamu, Claude Desktop, ChatGPT, Cursor, atau skrip khusus — mengiklankan daftar alat yang dapat dipanggil ke model. Model, ketika memutuskan suatu tindakan diperlukan, mengeluarkan payload terstruktur yang memberi nama alat dan argumennya. Tuan rumah mem-parsing payload tersebut, menjalankan alat secara nyata, dan mengembalikan hasilnya. Perulangan berlanjut hingga model memutuskan tidak diperlukan panggilan lagi.

Versi pertama kontrak ini dikirimkan pada Juni 2023 sebagai parameter "fungsi" OpenAI. Anthropic diikuti dengan blok `tool_use` di Claude 2.1. Gemini menambahkan `functionDeclarations` beberapa bulan kemudian. Setiap penyedia kini menampilkan bentuk yang sama: daftar alat yang diketik Skema JSON, pemanggilan alat muatan JSON. Protokol Konteks Model (November 2024) menggeneralisasi kontrak sehingga satu alat registri melayani setiap model. A2A (April 2026, v1.0) melapisi primitif yang sama untuk delegasi agen-ke-agen.

Perulangan empat langkah adalah invarian di bawah semua ini. Segala sesuatu yang lain di Fase 13 adalah sebuah elaborasi.

## Konsep

### Langkah pertama: jelaskan

Tuan rumah mendeklarasikan setiap alat dengan tiga bidang.

- **Nama.** Pengenal yang stabil dan dapat dibaca mesin. `get_weather`, bukan "masalah cuaca".
- **Deskripsi.** Ringkasan satu paragraf dalam bahasa alami. "Gunakan ketika pengguna bertanya tentang kondisi terkini untuk kota tertentu. Jangan gunakan untuk data historis."
- **Skema input.** Objek Skema JSON (draf 2020-12) yang menjelaskan argumen alat.

Model menerima daftarnya. Penyedia modern membuat serial deklarasi ini ke dalam prompt sistem menggunakan templat khusus penyedia, sehingga kamu sebagai pemanggil hanya menangani formulir terstruktur.

### Langkah kedua: putuskan

Berdasarkan pesan pengguna dan alat yang tersedia, model memilih salah satu dari tiga perilaku.1. **Jawab secara langsung** dalam teks. Tidak ada panggilan alat.
2. **Panggil satu atau beberapa alat.** Memancarkan objek panggilan terstruktur. Di bawah `parallel_tool_calls: true` (default pada OpenAI dan Gemini, ikut serta dalam Anthropic), model dapat melakukan beberapa panggilan dalam satu putaran.
3. **Refuse.** Output terstruktur mode ketat dapat menghasilkan blok `refusal` yang diketik, bukan panggilan.

Muatan panggilan alat memiliki tiga bidang stabil: panggilan `id`, alat `name`, dan objek JSON `arguments`. Id ada sehingga host dapat menghubungkan hasil selanjutnya dengan panggilan tertentu, yang penting ketika panggilan paralel kembali rusak.

### Langkah ketiga: jalankan

Host menerima panggilan, memvalidasi argumen terhadap skema yang dideklarasikan, dan menjalankan eksekutor. Argumen yang tidak valid berarti model berhalusinasi terhadap suatu bidang atau menggunakan tipe yang salah — mode kegagalan yang sangat umum pada model yang lemah. Host produksi melakukan salah satu dari tiga hal pada argumen yang tidak valid: gagal dengan cepat dan menampilkan kesalahan pada model, memperbaiki JSON dengan parser yang dibatasi, atau mencoba kembali model dengan kesalahan validasi yang disertakan dalam prompt.

Pelaksananya sendiri adalah code biasa. Python, TypeScript, prompt shell, kueri basis data. Ini menghasilkan hasil, yang biasanya berupa string tetapi dapat berupa nilai JSON apa pun atau blok konten terstruktur (teks, gambar, atau referensi sumber daya di MCP). Hasilnya harus dapat diserialkan.

### Langkah keempat: amati

Tuan rumah menambahkan hasil alat ke percakapan (sebagai pesan peran `tool` dengan pencocokan `id`) dan memanggil kembali model tersebut. Model tersebut sekarang memiliki output alat dalam konteks dan dapat menghasilkan jawaban akhir atau meminta lebih banyak panggilan. Hal ini berlanjut hingga model berhenti mengeluarkan panggilan atau host mencapai batas keamanan pada jumlah iterasi.

### Kepercayaan terpecah

Alat hadir dalam dua rasa yang penting untuk keamanan.

- **Murni.** Hanya baca, deterministik, tanpa efek samping. `get_weather`, `search_docs`, `get_current_time`. Aman untuk dihubungi secara spekulatif.
- **Konsekuensial.** Bermutasi status, membelanjakan uang, menyentuh data pengguna. `send_email`, `delete_file`, `execute_trade`. Harus diberi gerbang.

"Aturan Dua" Meta tahun 2026 untuk keamanan agen mengatakan satu giliran dapat menggabungkan paling banyak dua hal: input yang tidak tepercaya, data sensitif, tindakan konsekuensial. Antarmuka alat adalah tempat kamu menerapkan aturan tersebut — dengan menolak panggilan, memerlukan konfirmasi pengguna, atau meningkatkan cakupan. Lihat Fase 13 · 15 untuk bab keamanan lengkap dan Fase 14 · 09 untuk kebijakan izin tingkat agen.

### Tempat perulangan berada

| Konteks | Siapa yang menjelaskan | Siapa yang memutuskan | Siapa yang mengeksekusi |
|---------|---------------|-------------|--------------|
| Panggilan fungsi satu putaran (OpenAI/Anthropic/Gemini) | Pengembang aplikasi | LLM | Pengembang aplikasi |
| MCP | Server MCP | LLM melalui klien MCP | Server MCP |
| A2A | Penerbit Kartu Agen | Agen panggilan | Disebut agen |
| Browser web (agen pemanggil fungsi) | Ekstensi peramban / WebMCP | LLM | Waktu proses peramban |

Dimana-mana, empat langkah yang sama. Nama kolom berubah; strukturnya tidak.

### Mengapa tidak meminta model untuk memancarkan JSON saja?

"Minta model untuk membalas dalam JSON" adalah pola pemanggilan pra-fungsi. Kegagalan tersebut terjadi sekitar 5 hingga 15 persen pada model frontier dan lebih banyak lagi pada model yang lebih kecil. Mode kegagalan termasuk kurung kurawal yang hilang, koma di belakang, bidang halusinasi, dan tipe yang salah. kamu kemudian memerlukan izin perbaikan JSON, percobaan ulang, atau dekoder terbatas.Pemanggilan fungsi asli lebih baik karena tiga alasan. Pertama, penyedia melatih model secara end-to-end pada bentuk panggilan yang tepat, sehingga tingkat JSON yang valid naik menjadi 98 hingga 99 persen pada mode ketat. Kedua, muatan panggilan berada di slot protokolnya sendiri, bukan di dalam teks bebas — sehingga panggilan alat tidak pernah bocor ke dalam balasan yang terlihat oleh pengguna. Ketiga, penyedia menegakkan kepatuhan skema dengan decoding yang dibatasi (mode ketat OpenAI, `tool_use` Anthropic, `responseSchema` Gemini). Outputnya dijamin tervalidasi.

Fase 13 · 02 menjalankan ketiga API penyedia secara berdampingan. Fase 13 · 04 mendalami output terstruktur.

### Pemutus arus

Perulangan berakhir ketika model berhenti mengeluarkan panggilan atau host mencapai jumlah giliran maksimum. Host produksi mengaturnya antara 5 dan 20 putaran. Selain itu, kamu hampir pasti berada dalam lingkaran yang modelnya tidak dapat keluar. Code Claude defaultnya adalah 20; Asisten OpenAI ke 10; Mode agen kursor ke 25.

Alternatifnya — loop tak terbatas — muncul setiap enam bulan saat "agen menghabiskan $400 untuk panggilan API dalam semalam" post-mortem. Jangan mengirim tanpa batas.

Fase 14 · 12 mencakup pemulihan kesalahan dan penyembuhan diri secara mendalam; Fase 17 mencakup batasan laju produksi.

### Kemana perginya Fase 13 dari sini

- Lesson 02 hingga 05 memoles permukaan panggilan alat tingkat penyedia.
- Lesson 06 hingga 14 menggeneralisasi loop menjadi MCP.
- Lesson 15 hingga 18 mempertahankan loop terhadap server yang bermusuhan, pengguna yang bermusuhan, dan permukaan autentikasi distance jauh yang tidak diautentikasi.
- Lesson 19 hingga 22 memperluas pola tersebut ke kolaborasi agen-ke-agen, kemampuan observasi, perutean, dan pengemasan.
- Lesson 23 menyampaikan ekosistem lengkap menggunakan setiap primitif.

Setiap lesson yang tersisa merupakan penjabaran dari putaran empat langkah ini. Ingatlah hal ini sebagai invarian.

## Pakai

`code/main.py` menjalankan loop empat langkah tanpa LLM. Fungsi "pengambil keputusan" palsu mensimulasikan model dengan mencocokkan pola pada pesan pengguna; pelaksana, validator skema, dan harness langkah observasi adalah nyata. Jalankan untuk melihat koreografi permintaan/respons lengkap dengan status perantara yang dapat dicetak, lalu ganti penentu palsu dengan penyedia asli mana pun di lesson berikutnya.

Apa yang harus dilihat:

- Registri alat menampung tiga bidang per alat: nama, deskripsi, skema, dan referensi pelaksana.
- Validator adalah subset Skema JSON minimal (tipe, wajib, enum, min/maks) yang ditulis dalam stdlib saja. Fase 13 · 04 mengirimkan yang lebih lengkap.
- Loop membatasi jumlah iterasi pada lima. Agen produksi membutuhkan pemutus arus seperti ini.

## Kirim

Lesson ini menghasilkan `outputs/skill-tool-interface-reviewer.md`. Mengingat definisi draf alat (nama + deskripsi + skema + garis besar pelaksana), keterampilan mengauditnya untuk kebugaran loop: apakah namanya stabil terhadap mesin, apakah deskripsi merupakan ringkasan penggunaan yang lengkap, apakah skema menggunakan Skema JSON 2020-12 dengan benar, dan apakah klasifikasi murni vs konsekuensial eksplisit.

## Latihan

1. Tambahkan alat keempat ke `code/main.py` yang disebut `get_stock_price(ticker)`. Tulis deskripsinya sebagai "Gunakan ketika pengguna menanyakan harga saham saat ini dengan ticker. Jangan gunakan untuk harga historis atau ringkasan pasar." Jalankan harness dan konfirmasikan pertanyaan rute penentu palsu yang menyebutkan ticker ke alat baru.

2. Hancurkan validator skema. Lewati panggilan yang objek `arguments` tidak memiliki bidang wajib, dan konfirmasikan bahwa host menolaknya sebelum dieksekusi. Kemudian lewati panggilan dengan bidang tambahan yang tidak diketahui. Putuskan: haruskah tuan rumah menolak atau mengabaikan? Benarkan pilihan kamu dengan argumen keamanan.3. Klasifikasikan setiap alat dalam harness sebagai murni atau konsekuensial. Tambahkan tanda `consequential: true` ke entri registri yang membutuhkannya, dan ubah loop untuk mencetak baris "akan mengonfirmasi dengan pengguna" setiap kali alat konsekuensial dipilih. Ini adalah bentuk gerbang konfirmasi yang dibutuhkan setiap host produksi.

4. Gambarlah loop empat langkah di atas kertas dengan tabel kolom penyedia di atas terisi untuk klien favorit kamu (Claude Desktop, Cursor, ChatGPT, atau tumpukan khusus). Referensi silang dengan varian khusus MCP di Fase 13 · 06.

5. Baca panduan pemanggilan fungsi OpenAI dari atas ke bawah. Identifikasi satu bidang yang ada dalam permintaan tetapi tidak dalam putaran empat langkah seperti yang disajikan di sini. Jelaskan apa yang ditambahkan dan mengapa hal ini nyaman dan bukan penting.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Alat | "Sesuatu yang dapat disebut oleh model" | Tiga nama + input tipe JSON-Skema + fungsi eksekutor |
| Pemanggilan fungsi | "Penggunaan alat asli" | Dukungan API tingkat penyedia untuk mengeluarkan panggilan alat terstruktur, bukan prosa |
| Panggilan alat | "Permintaan model untuk bertindak" | Payload JSON dengan `id`, `name`, `arguments` yang dipancarkan oleh model |
| Hasil alat | "Alat apa yang dikembalikan" | Output eksekutor, dibungkus dalam pesan peran `tool` dengan id yang cocok |
| Panggilan alat paralel | "Banyak panggilan sekaligus" | Beberapa objek panggilan dalam satu putaran model, independen dan dapat diurutkan berdasarkan id |
| Modus ketat | "JSON Terjamin" | Penguraian code terbatas yang memaksa output model untuk memvalidasi terhadap skema yang dideklarasikan |
| Alat murni | "Alat hanya-baca" | Tidak ada efek samping; aman untuk dijalankan kembali |
| Alat konsekuensial | "Alat Aksi" | Bermutasi keadaan eksternal; memerlukan gerbang, audit, atau konfirmasi pengguna |
| Lingkaran empat langkah | "Siklus panggilan alat" | jelaskan → putuskan → jalankan → amati |
| Tuan rumah | "Waktu berjalan agen" | Program yang menyimpan registri alat, memanggil model, dan menjalankan eksekutor |

## Bacaan Lanjutan

- [OpenAI — Panduan pemanggilan fungsi](https://platform.openai.com/docs/guides/function-calling) — referensi kanonik untuk deklarasi alat dan bentuk panggilan bergaya OpenAI
- [Antropik — Ikhtisar penggunaan alat](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — format blok `tool_use` / `tool_result` milik Claude
- [Google — Panggilan fungsi Gemini](https://ai.google.dev/gemini-api/docs/function-calling) — `functionDeclarations` dan semantik panggilan paralel di Gemini
- [Protokol Konteks Model — Spesifikasi 25-11-2025](https://modelcontextprotocol.io/spesification/2025-11-25) — generalisasi penyedia-agnostik dari antarmuka alat
- [Skema JSON — catatan rilis 2020-12](https://json-schema.org/draft/2020-12/release-notes) — dialek skema yang digunakan oleh setiap alat modern API
