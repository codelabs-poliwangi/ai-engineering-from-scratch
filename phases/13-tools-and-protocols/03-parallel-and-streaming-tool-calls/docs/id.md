# Panggilan Alat Paralel dan Streaming dengan Alat

> Tiga pencarian cuaca independen yang diserialkan adalah tiga perjalanan pulang pergi. Jalankan secara paralel dan total waktu dikurangi menjadi satu panggilan paling lambat. Setiap penyedia perbatasan kini mengeluarkan beberapa panggilan alat dalam satu putaran. Imbalannya nyata; pipanya halus. Lesson ini membahas kedua bagian: penyebaran paralel dan penyusunan kembali argumen streaming, dengan penekanan pada jebakan korelasi id.

**Type:** Build
**Language:** Python (stdlib, kumpulan thread + harness streaming)
**Prerequisites:** Fase 13 · 02 (pemanggilan fungsi menyelam lebih dalam)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Jelaskan mengapa `parallel_tool_calls: true` ada dan kapan harus menonaktifkannya.
- Hubungkan potongan argumen yang dialirkan ke id panggilan alat yang tepat selama fan-out paralel.
- Pasang kembali sebagian string `arguments` menjadi JSON lengkap tanpa penguraian lebih awal.
- Jalankan tolok ukur cuaca tiga kota yang menunjukkan latensi berurutan vs paralel.

## Masalah

Tanpa panggilan paralel, agen yang menjawab "bagaimana cuaca di Bengaluru, Tokyo, dan Zurich" melakukan ini:

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

Tiga perjalanan bolak-balik LLM, yang masing-masing juga membayar latensi pelaksana. Kira-kira 4x waktu jam dinding ideal.

Dengan panggilan paralel:

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

Satu perjalanan pulang pergi LLM. Waktu pelaksana adalah maksimal dari ketiganya, bukan penjumlahannya. Tolok ukur produksi pada OpenAI, Anthropic, dan Gemini menunjukkan pengurangan jam kerja sebesar 60 hingga 70 persen pada weight kerja fan-out.

Harganya adalah kompleksitas korelasi. Jika ketiga panggilan selesai secara tidak berurutan, hasil kamu harus membawa `tool_call_id` yang cocok sehingga model dapat menyejajarkannya. Saat hasil dialirkan, kamu harus merakit sebagian fragmen argumen menjadi JSON lengkap sebelum mengeksekusi. Gemini 3 menambahkan id unik untuk memecahkan masalah dunia nyata di mana dua panggilan paralel ke alat yang sama tidak dapat dibedakan.

## Konsep

### Mengaktifkan paralel

- **OpenAI.** `parallel_tool_calls: true` aktif secara default. Setel `false` untuk memaksa serial.
- **Antropis.** Paralel melalui `disable_parallel_tool_use: false` (default pada Claude 3.5 dan lebih tinggi). Setel `true` untuk serial.
- **Gemini.** Selalu berkemampuan paralel; `tool_config.function_calling_config.mode = "AUTO"` biarkan model memutuskan.

Nonaktifkan paralel ketika alat memiliki ketergantungan pengurutan (`create_file` lalu `write_file`), ketika output satu panggilan menginformasikan input panggilan lain, atau ketika pembatas kecepatan tidak dapat menangani fan-out.

### Korelasi identitas

Setiap panggilan yang dilakukan model memiliki `id`. Setiap hasil yang dikembalikan host harus menyertakan id yang sama. Tanpa hal ini, hasilnya akan ambigu.

- **OpenAI.** `tool_call_id` pada setiap pesan peran alat.
- **Antropis.** `tool_use_id` di setiap blok `tool_result`.
- **Gemini.** `id` di setiap `functionResponse` (Gemini 3 dan lebih tinggi; Gemini 2 cocok dengan nama yang terputus untuk panggilan paralel dengan nama yang sama).

### Menjalankan panggilan secara bersamaan

Host menjalankan setiap eksekutor panggilan pada thread, coroutine, atau pekerja distance jauhnya sendiri. Harness paling sederhana menggunakan kumpulan benang; produksi menggunakan asyncio dengan `asyncio.gather` atau konkurensi terstruktur. Urutan penyelesaian tidak dapat diprediksi — id adalah pengidentifikasi.

Satu bug umum: membalas dengan hasil dalam urutan daftar panggilan, bukan urutan penyelesaian. Hal ini biasanya berhasil karena model hanya peduli pada `tool_call_id`, namun jika hasilnya hilang atau terduplikasi, pengiriman yang tidak berurutan membuat proses debug menjadi lebih sulit. Lebih suka membalas dalam urutan penyelesaian dengan id eksplisit.### Panggilan alat streaming

Saat model streaming, `arguments` tiba dalam potongan. Tiga aliran potongan terpisah untuk tiga panggilan paralel disisipkan pada kabel. kamu memerlukan satu akumulator per id.

Bentuk menurut penyedia:

- **OpenAI.** Setiap potongan adalah `choices[0].delta.tool_calls[i].function.arguments` (string parsial). Potongan tersebut membawa `index` (posisi dalam daftar panggilan). kamu mengumpulkan per indeks, membaca `id` saat pertama kali muncul, dan mengurai JSON saat `finish_reason = "tool_calls"`.
- **Antropis.** Acara streaming adalah `message_start`, lalu satu `content_block_start` per blok dengan tipe `tool_use` (berisi id, nama, input kosong). `content_block_delta` acara membawa `input_json_delta` potongan. `content_block_stop` menutup setiap blok.
- **Gemini.** `streamFunctionCallArguments` (Gemini 3 dan lebih tinggi) mengeluarkan potongan dengan `functionCallId` sehingga panggilan disisipkan dengan rapi. Sebelum Gemini 3, streaming mengembalikan satu panggilan lengkap dalam satu waktu.

### JSON parsial dan jebakan parse-awal

kamu tidak dapat mengurai `arguments` sampai selesai. JSON parsial seperti `{"city": "Beng` tidak valid dan akan dinaikkan. Gerbang yang benar adalah sinyal akhir panggilan penyedia: `finish_reason = "tool_calls"` OpenAI, `content_block_stop` Anthropic, atau acara akhir streaming Gemini. Baru kemudian coba `json.loads`. Pendekatan yang lebih kuat menggunakan parser JSON inkremental yang menghasilkan peristiwa saat struktur selesai; Panduan streaming OpenAI merekomendasikan hal ini untuk UX yang menampilkan indikator "berpikir" secara langsung. Penghitungan tanda kurung kurawal tidak dapat diandalkan sebagai uji kelengkapan (tanda kurung kurawal di dalam string yang dikutip atau konten yang di-escape menyebabkan positif palsu) dan hanya boleh digunakan sebagai heuristik debug informal.

### Penyelesaian di luar pesanan

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

Balasan tuan rumah masih harus mengutip id:

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

Urutan balasan tidak menjadi masalah kebenaran di OpenAI atau Anthropic. Gemini menerima pesanan apa pun selama ID cocok.

### Tolok ukur: berurutan vs paralel

Harness di `code/main.py` menyimulasikan tiga eksekutor dengan latensi 400, 600, dan 800 ms. Sequential menjalankannya dalam total 1800 ms. Menjalankannya secara paralel dalam maks(400, 600, 800) = 800 ms. Perbedaannya konstan, tidak proporsional, sehingga penghematan bertambah seiring bertambahnya jumlah alat.

Peringatan di dunia nyata: panggilan paralel menekankan API hilir. Penyebaran 10 arah ke layanan dengan tarif terbatas akan gagal. Fase 13 · 17 mencakup tekanan balik tingkat gateway; coba lagi semantik direncanakan untuk fase mendatang.

### Streaming jam dinding fan-out

Jika model itu sendiri melakukan streaming, kamu dapat mulai mengeksekusi segera setelah argumen satu panggilan selesai, daripada menunggu semua panggilan selesai. Ini adalah optimization dokumen OpenAI tetapi tidak semua SDK diekspos. Harness dalam lesson ini melakukannya: segera setelah aliran yang disimulasikan menghasilkan objek argumen yang lengkap, host memulai panggilan itu.

## Pakai

`code/main.py` memiliki dua bagian. Yang pertama menjalankan tiga simulasi panggilan cuaca secara berurutan dan paralel menggunakan `concurrent.futures.ThreadPoolExecutor` dan mencetak waktu jam dinding. Babak kedua memutar ulang respons streaming palsu — potongan `arguments` untuk tiga panggilan paralel yang disisipkan dalam satu streaming — dan menyusunnya kembali per-id dengan `StreamAccumulator`. Tidak ada LLM, tidak ada jaringan, hanya logika perakitan ulang.

Apa yang harus dilihat:- Pengatur waktu berurutan mencapai 1,8 detik. Pengatur waktu paralel mencapai 0,8 detik pada latensi palsu yang sama.
- Akumulator menangani potongan yang datang tidak berurutan dengan melakukan buffering per-id dan menguraikan hanya ketika JSON setiap panggilan selesai.
- Pelaksana memulai segera setelah argumen id selesai, bukan setelah semua streaming berakhir.

## Kirim

Lesson ini menghasilkan `outputs/skill-parallel-call-safety-check.md`. Dengan adanya registri alat, keterampilan akan mengaudit alat mana yang aman untuk diparalelkan, alat mana yang memiliki ketergantungan pengurutan, dan alat mana yang akan overshoot kecepatan hilir — mengembalikan registri yang telah direvisi dengan tanda `parallel_safe` per alat.

## Latihan

1. Jalankan `code/main.py` dan variasikan latensi yang disimulasikan. Konfirmasikan bahwa rasio paralel-sekuensial kira-kira `max/sum` (pengoperasian sebenarnya sedikit menyimpang dari ideal karena penjadwalan thread, serialisasi, dan overhead harness). Pada distribusi latensi manakah paralel berhenti menjadi penting?

2. Perluas akumulator untuk menangani kasus "panggilan dibatalkan di tengah-tengah streaming" dengan menghilangkan buffernya dan memunculkan peristiwa `cancelled`. Penyedia mana yang mendokumentasikan kasus ini secara eksplisit? Periksa semantik `content_block_stop` Anthropic dan perilaku `finish_reason: "length"` OpenAI.

3. Ganti kumpulan thread dengan `asyncio.gather`. Tolok ukur keduanya. kamu akan melihat keuntungan kecil pada async karena biaya peralihan konteks yang lebih rendah, tetapi hanya jika pelaksana melakukan I/O yang sebenarnya.

4. Pilih dua alat yang TIDAK boleh diparalelkan (misalnya `create_file` lalu `write_file`). Tambahkan grafik `ordering_dependency` ke registri dan gerbangkan penyebaran paralel pada grafik tersebut. Ini adalah mesin minimum untuk penjadwalan sadar ketergantungan, yang diformalkan oleh fase rekayasa agen di masa depan.

5. Baca bagian pemanggilan fungsi paralel OpenAI dan dokumen `disable_parallel_tool_use` Anthropic. Identifikasi satu jenis alat dunia nyata di mana Anthropic merekomendasikan untuk menonaktifkan paralelisme. (Petunjuk: mutasi konsekuensial pada sumber daya yang sama.)

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Panggilan alat paralel | "Penyebaran dalam satu putaran" | Model mengeluarkan beberapa panggilan alat dalam satu pesan asisten |
| `parallel_tool_calls` | "Bendera OpenAI" | Mengaktifkan atau menonaktifkan emisi multi-panggilan |
| `disable_parallel_tool_use` | "Kebalikan antropik" | Bendera penyisihan; defaultnya adalah paralel diaktifkan |
| ID panggilan alat | "Pegangan korelasi" | Pengidentifikasi per panggilan, pesan hasil harus digaungkan |
| Akumulator | "Penyangga aliran" | Buffer string per-id untuk sebagian `arguments` bongkahan |
| Penyelesaian di luar pesanan | "Tercepat dulu" | Panggilan paralel berakhir dalam urutan yang tidak dapat diprediksi; id adalah lem |
| Grafik ketergantungan | "Kendala pemesanan" | Alat yang keluarannya dimasukkan ke dalam input alat lain; tidak dapat memparalelkan |
| Parse-perangkap awal | "JSON.parse meledak" | Mencoba mengurai string `arguments` | yang tidak lengkap
| `streamFunctionCallArguments` | "Feature Gemini 3" | Potongan argumen yang dialirkan dengan id unik per panggilan |
| Balasan penyelesaian pesanan | "Jangan menunggu semuanya" | Balasan dengan hasil yang tiba, dikunci dengan id |

## Bacaan Lanjutan- [OpenAI — Panggilan fungsi paralel](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — perilaku default dan tanda penyisihan
- [Antropik — Penggunaan alat: penerapan penggunaan alat](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` dan pengelompokan hasil
- [Google — bagian paralel panggilan fungsi Gemini](https://ai.google.dev/gemini-api/docs/function-calling) — panggilan paralel berkorelasi id dari Gemini 3
- [OpenAI — Respons streaming dengan alat](https://platform.openai.com/docs/api-reference/responses-streaming) — penyusunan ulang argumen terpotong untuk aliran OpenAI
- [Antropik — Pesan streaming](https://docs.anthropic.com/en/api/messages-streaming) — `content_block_delta` dengan `input_json_delta`
