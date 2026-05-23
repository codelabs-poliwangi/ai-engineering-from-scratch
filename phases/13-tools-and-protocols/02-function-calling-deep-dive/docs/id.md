# Fungsi Memanggil Penyelaman Mendalam — OpenAI, Anthropic, Gemini

> Ketiga penyedia terdepan berkumpul pada putaran panggilan alat yang sama pada tahun 2024 dan kemudian menyimpang dalam hal lainnya. OpenAI menggunakan `tools` dan `tool_calls`. Antropik menggunakan blok `tool_use` dan `tool_result`. Gemini menggunakan `functionDeclarations` dan korelasi id unik. Lesson ini membedakan ketiganya secara berdampingan sehingga code yang dikirimkan pada satu penyedia tidak rusak saat kamu mem-portingnya.

**Type:** Build
**Language:** Python (stdlib, penerjemah skema)
**Prerequisites:** Phase 13 · 01 (antarmuka alat)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Sebutkan tiga perbedaan bentuk antara payload pemanggil fungsi OpenAI, Anthropic, dan Gemini (deklarasi, panggilan, hasil).
- Terjemahkan satu deklarasi alat di ketiga format penyedia dan prediksi perbedaan batasan mode ketat.
- Gunakan `tool_choice` di setiap penyedia untuk memaksa, melarang, atau memilih panggilan alat secara otomatis.
- Ketahui batas keras per penyedia (jumlah alat, kedalaman skema, panjang argumen) dan tanda kesalahan yang dikeluarkan masing-masing penyedia ketika batas dilanggar.

## Masalah

Bentuk permintaan pemanggilan fungsi berbeda-beda menurut penyedia. Tiga contoh nyata dari tumpukan produksi tahun 2026:

**API Penyelesaian / Respons Obrolan OpenAI.** kamu lulus `tools: [{type: "function", function: {name, description, parameters, strict}}]`. Respons model berisi `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]` dengan `arguments` adalah string JSON yang harus kamu urai. Mode ketat (`strict: true`) menerapkan kepatuhan skema melalui decoding terbatas.

**Anthropic Messages API.** kamu lulus `tools: [{name, description, input_schema}]`. Tanggapannya muncul kembali sebagai `content: [{type: "text"}, {type: "tool_use", id, name, input}]`. `input` sudah diurai (objek, bukan string). kamu membalas dengan pesan `user` baru yang berisi blok `{type: "tool_result", tool_use_id, content}`.

**Google Gemini API.** kamu melewati `tools: [{functionDeclarations: [{name, description, parameters}]}]` (berada di bawah `functionDeclarations`). Responsnya muncul sebagai `candidates[0].content.parts: [{functionCall: {name, args, id}}]` di mana `id` unik di Gemini 3 dan untuk korelasi panggilan paralel. kamu membalas dengan `{functionResponse: {name, id, response}}`.

Lingkaran yang sama. Nama bidang berbeda, sarang berbeda, konvensi string-vs-objek berbeda, mekanisme korelasi berbeda. Sebuah tim yang menulis agen cuaca di OpenAI membayar biaya pelabuhan dua hari ke Anthropic dan satu hari lagi ke Gemini hanya untuk pipa ledeng.

Lesson ini membangun penerjemah yang menyatukan tiga format menjadi satu deklarasi alat kanonik dan rute di tepinya. Fase 13 · 17 menggeneralisasi pola yang sama ke dalam gateway LLM.

## Konsep

### Struktur umum

Setiap penyedia membutuhkan lima hal:

1. **Daftar alat.** Nama per alat, deskripsi, dan skema input.
2. **Pilihan alat.** Paksa alat tertentu, larang alat, atau biarkan model yang memutuskan.
3. **Emisi panggilan.** Output terstruktur yang memberi nama alat dan argumen.
4. **ID panggilan.** Menghubungkan respons dengan panggilan yang benar (penting untuk paralel).
5. **Injeksi hasil.** Pesan atau blok yang menghubungkan hasil kembali ke panggilan.

### Bentuk berbeda, bidang demi bidang| Aspek | OpenAI | Antropik | kembar |
|--------|--------|-----------|--------|
| Amplop deklarasi | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Bidang skema | `parameters` | `input_schema` | `parameters` |
| Wadah tanggapan | `tool_calls[]` pada pesan asisten | `content[]` jenis `tool_use` | `parts[]` jenis `functionCall` |
| Tipe argumen | JSON yang dirangkai | objek yang diurai | objek yang diurai |
| Format identitas | `call_...` (OpenAI menghasilkan) | `toolu_...` (Antropik) | UUID (Gemini 3+) |
| Blok hasil | peran `tool`, `tool_call_id` | `user` dengan `tool_result`, `tool_use_id` | `functionResponse` dengan pencocokan `id` |
| Paksa-alat | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| Melarang alat | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| Skema ketat | `strict: true` | skema-is-skema (selalu diberlakukan) | `responseSchema` pada tingkat permintaan |

### Batasan yang benar-benar akan kamu capai

- **OpenAI.** 128 alat per permintaan. Kedalaman skema 5. String argumen <= 8192 byte. Mode ketat tidak memerlukan `$ref`, tidak ada `oneOf`/`anyOf`/`allOf` dengan tumpang tindih, setiap properti terdaftar di `required`.
- **Antropik.** 64 alat per permintaan. Kedalaman skema secara efektif tidak terbatas tetapi batas praktisnya 10. Tidak ada tanda mode ketat; skema adalah kontrak dan model cenderung patuh.
- **Gemini.** 64 fungsi per permintaan. Jenis skema adalah subset OpenAPI 3.0 (sedikit perbedaan dari Skema JSON 2020-12). Panggilan paralel id unik sejak Gemini 3.

### `tool_choice` perilaku

Tiga mode yang didukung semua orang, diberi nama berbeda.

- **Otomatis.** Alat pengambilan model atau teks. Bawaan.
- **Wajib / Apa saja.** Model harus memanggil setidaknya satu alat.
- **Tidak ada.** Model tidak boleh memanggil alat.

Ditambah satu mode unik untuk setiap penyedia:

- **OpenAI.** Paksa alat tertentu berdasarkan nama.
- **Antropis.** Paksa alat tertentu berdasarkan nama; `disable_parallel_tool_use` bendera memisahkan tunggal vs multi.
- **Gemini.** `mode: "VALIDATED"` merutekan setiap respons melalui validator skema, apa pun maksud modelnya.

### Panggilan paralel

`parallel_tool_calls: true` OpenAI (default) mengirimkan beberapa panggilan dalam satu pesan asisten. kamu menjalankan semuanya dan membalas dengan pesan peran alat batch yang berisi satu entri per `tool_call_id`. Antropis secara historis melakukan panggilan tunggal; `disable_parallel_tool_use: false` (default pada Claude 3.5) mengaktifkan multi. Gemini 2 mengizinkan panggilan paralel tetapi tidak memberikan id stabil; Gemini 3 menambahkan UUID sehingga tanggapan yang tidak sesuai berkorelasi dengan baik.

### Streaming

Ketiganya mendukung panggilan alat streaming. Format kawat berbeda:

- **OpenAI.** Potongan Delta `tool_calls[i].function.arguments` tiba secara bertahap. kamu terakumulasi hingga `finish_reason: "tool_calls"`.
- **Antropis.** Acara permulaan blok / blok-delta / penghentian blok. `input_json_delta` potongan membawa sebagian argumen.
- **Gemini.** `streamFunctionCallArguments` (baru di Gemini 3) memancarkan potongan dengan `functionCallId` sehingga beberapa panggilan paralel dapat disisipkan.

Fase 13 · 03 mendalami perakitan ulang paralel + streaming. Lesson ini berfokus pada deklarasi dan bentuk panggilan tunggal.

### Kesalahan dan perbaikan

Kesalahan argumen yang tidak valid juga terlihat berbeda.- **OpenAI (tidak ketat).** Model mengembalikan `arguments: "{bad json}"`, penguraian JSON kamu gagal, kamu memasukkan pesan kesalahan dan menelepon kembali.
- **OpenAI (ketat).** Validasi terjadi selama decoding; JSON yang tidak valid tidak mungkin dilakukan tetapi `refusal` dapat muncul.
- **Antropik.** `input` mungkin berisi kolom yang tidak diharapkan; skema bersifat penasehat. Validasi sisi server.
- **Gemini.** Keunikan OpenAPI 3.0: `enum` pada bidang objek diabaikan secara diam-diam; validasi diri kamu sendiri.

### Pola penerjemah

Deklarasi alat kanonik dalam code kamu terlihat seperti ini (kamu memilih bentuknya):

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

Tiga fungsi kecil menerjemahkannya ke dalam tiga bentuk penyedia. Harness di `code/main.py` melakukan hal ini, lalu meneruskan panggilan alat palsu melalui bentuk respons masing-masing penyedia. Tidak diperlukan jaringan — lesson ini mengajarkan bentuk, bukan HTTP.

Tim produksi membungkus penerjemah ini dalam `AbstractToolset` (Pydantic AI), `UniversalToolNode` (LangGraph), atau `BaseTool` (LlamaIndex). Fase 13 · 17 mengirimkan gerbang yang memperlihatkan API berbentuk OpenAI di depan salah satu dari ketiganya.

## Pakai

`code/main.py` mendefinisikan satu kelas data `Tool` kanonik dan tiga penerjemah yang memancarkan deklarasi JSON OpenAI, Anthropic, dan Gemini. Ini kemudian menguraikan respons penyedia buatan tangan dari setiap bentuk ke dalam objek panggilan kanonik yang sama, menunjukkan bahwa semantiknya identik. Jalankan dan bedakan ketiga deklarasi secara berdampingan.

Apa yang harus dilihat:

- Ketiga blok deklarasi hanya berbeda pada nama amplop dan field.
- Tiga blok respons berbeda berdasarkan lokasi panggilan (blok `tool_calls` tingkat atas, blok `content[]`, entri `parts[]`).
- Satu fungsi `canonical_call()` mengekstrak `{id, name, args}` dari ketiga bentuk respons.

## Kirim

Lesson ini menghasilkan `outputs/skill-provider-portability-audit.md`. Mengingat integrasi pemanggilan fungsi terhadap satu penyedia, keterampilan menghasilkan audit portabilitas: penyedia mana yang membatasi batasannya, bidang mana yang perlu diganti namanya, dan apa yang rusak saat di-porting ke penyedia lainnya.

## Latihan

1. Jalankan `code/main.py` dan verifikasi bahwa ketiga JSON deklarasi penyedia semuanya membuat serial objek `Tool` yang sama. Ubah alat kanonik untuk menambahkan parameter enum dan konfirmasikan hanya penerjemah Gemini yang perlu menangani kekhasan OpenAPI.

2. Tambahkan parser `ListToolsResponse` untuk setiap penyedia yang mengekstrak daftar alat yang dikembalikan model setelah `list_tools` atau panggilan penemuan. OpenAI tidak memilikinya secara asli; perhatikan asimetri ini.

3. Terapkan konversi `tool_choice`: petakan `ToolChoice(mode="force", tool_name="x")` kanonik ke dalam ketiga bentuk penyedia. Kemudian petakan `mode="any"` dan `mode="none"`. Periksa tabel perbedaan lesson.

4. Pilih salah satu dari tiga penyedia dan baca panduan pemanggilan fungsinya secara menyeluruh. Temukan satu bidang dalam spesifikasi skemanya yang tidak didukung oleh dua bidang lainnya. Kandidat: OpenAI `strict`, Antropis `disable_parallel_tool_use`, Gemini `function_calling_config.allowed_function_names`.

5. Tulis vector uji: pemanggilan alat yang argumennya melanggar skema yang dideklarasikan. Jalankan melalui validator masing-masing penyedia (stdlib di Lesson 01 akan berfungsi sebagai proxy) dan catat kesalahan mana yang terjadi. Dokumentasikan penyedia mana yang akan kamu gunakan dalam produksi untuk ketelitian.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Pemanggilan fungsi | "Penggunaan alat" | API tingkat penyedia untuk emisi panggilan alat terstruktur |
| Deklarasi alat | "Spesifikasi alat" | Nama + deskripsi + payload input Skema JSON |
| `tool_choice` | "Paksa / larang" | Mode otomatis / wajib / tidak ada / nama tertentu |
| Modus ketat | "Penegakan skema" | Bendera OpenAI yang membatasi decoding agar sesuai dengan skema |
| `tool_use` blok | "Bentuk panggilan antropik" | Blok konten sebaris dengan id, nama, input |
| `functionCall` bagian | "Bentuk panggilan Gemini" | Entri `parts[]` yang berisi nama, argumen, dan id |
| Argumen-sebagai-string | "JSON yang dirangkai" | OpenAI mengembalikan argumen sebagai string JSON, bukan objek |
| Panggilan alat paralel | "Penyebaran dalam satu putaran" | Beberapa panggilan alat dalam satu pesan asisten |
| Penolakan | "Model menurun" | Blok penolakan hanya mode ketat alih-alih panggilan |
| Subset OpenAPI 3.0 | "Keunikan skema Gemini" | Gemini menggunakan dialek mirip Skema JSON dengan sedikit perbedaan |

## Bacaan Lanjutan

- [OpenAI — Panduan pemanggilan fungsi](https://platform.openai.com/docs/guides/function-calling) — referensi kanonik termasuk mode ketat dan panggilan paralel
- [Antropik — Ikhtisar penggunaan alat](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` dan `tool_result` blok semantik
- [Google — panggilan fungsi Gemini](https://ai.google.dev/gemini-api/docs/function-calling) — panggilan paralel, id unik, dan subset OpenAPI
- [Vertex AI — Referensi pemanggilan fungsi](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — platform perusahaan Gemini
- [OpenAI — Output terstruktur](https://platform.openai.com/docs/guides/structured-outputs) — detail penerapan skema mode ketat
