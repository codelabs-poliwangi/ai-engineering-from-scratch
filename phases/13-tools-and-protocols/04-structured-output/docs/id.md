# Output Terstruktur — Skema JSON, Pydantic, Zod, Decoding Terbatas

> "Minta model dengan baik untuk mengembalikan JSON" gagal 5 hingga 15 persen, bahkan pada model frontier. Output terstruktur menutup kesenjangan tersebut dengan decoding yang dibatasi: model secara harfiah dicegah untuk mengeluarkan token yang dapat melanggar skema. Mode ketat OpenAI, penggunaan alat tipe skema Anthropic, `responseSchema` dari Gemini, `output_type` dari Pydantic AI, dan `.parse` dari Zod adalah lima bentuk permukaan dari ide yang sama. Lesson ini membangun validator skema dan pelajar kontrak mode ketat yang akan digunakan untuk setiap jalur ekstraksi produksi.

**Type:** Build
**Language:** Python (stdlib, subset Skema JSON 2020-12)
**Prerequisites:** Fase 13 · 02 (pemanggilan fungsi menyelam lebih dalam)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Tulis Skema JSON 2020-12 untuk target ekstraksi menggunakan batasan yang tepat (enum, min/maks, wajib, pola).
- Jelaskan mengapa mode ketat dan decoding terbatas memberikan jaminan berbeda dari "validasi setelah pembuatan".
- Bedakan tiga mode kegagalan: kesalahan parse, pelanggaran skema, penolakan model.
- Kirim pipa ekstraksi dengan perbaikan yang diketik dan penanganan penolakan yang diketik.

## Masalah

Agen yang membaca email pesanan pembelian perlu mengubah teks bebas menjadi `{customer, line_items, total_usd}`. Tiga pendekatan.

**Pendekatan pertama: prompt untuk JSON.** "Balas dalam JSON dengan bidang pelanggan, item_baris, total_usd." Bekerja 85 hingga 95 persen waktunya pada model frontier. Gagal dalam enam cara: kurung kurawal hilang, koma di belakang, tipe salah, bidang berhalusinasi, terpotong pada batas token, bocoran prosa seperti "Ini JSON kamu:".

**Pendekatan kedua: validasi setelah pembuatan.** Hasilkan secara bebas, parsing, validasi berdasarkan skema, coba lagi jika gagal. Dapat diandalkan namun mahal — kamu membayar untuk setiap percobaan ulang, dan bug pemotongan membutuhkan satu giliran tambahan per kejadian.

**Pendekatan ketiga: decoding terbatas.** Penyedia menerapkan skema pada waktu decode. Token yang tidak valid akan disembunyikan dari distribusi pengambilan sample. Outputnya dijamin terurai dan dijamin tervalidasi. Kegagalan diciutkan ke satu mode: penolakan (model memutuskan input tidak sesuai dengan skema).

Setiap penyedia perbatasan pada tahun 2026 mengirimkan beberapa bentuk pendekatan ketiga.

- **OpenAI.** `response_format: {type: "json_schema", strict: true}` plus `refusal` sebagai respons jika model ditolak.
- **Antropik.** Penegakan skema pada input `tool_use`; `stop_reason: "refusal"` bukan apa-apa, tapi `end_turn` tanpa panggilan alat adalah sinyalnya.
- **Gemini.** `responseSchema` pada tingkat permintaan; pada tahun 2026 Gemini mengirimkan batasan tata bahasa tingkat token untuk jenis tertentu.
- **Pydantic AI.** `output_type=InvoiceModel` memancarkan `RunResult` yang diketik ke `InvoiceModel`.
- **Zod (TypeScript).** Parser runtime yang memvalidasi output penyedia terhadap skema Zod; berpasangan dengan `beta.chat.completions.parse` OpenAI.

Benang merahnya: deklarasikan skema satu kali, terapkan secara menyeluruh.

## Konsep

### Skema JSON 2020-12 — lingua franca

Setiap penyedia menerima Skema JSON 2020-12. Konstruksi yang paling sering kamu gunakan:- `type`: salah satu dari `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`.
- `properties`: peta nama bidang ke subskema.
- `required` : daftar nama field yang harus dimunculkan.
- `enum`: kumpulan nilai yang diizinkan secara tertutup.
- `minimum` / `maximum` (angka), `minLength` / `maxLength` / `pattern` (string).
- `items`: subskema diterapkan ke setiap elemen array.
- `additionalProperties`: `false` melarang kolom tambahan (default bervariasi berdasarkan mode).

Mode ketat OpenAI menambahkan tiga persyaratan: setiap properti harus terdaftar di `required`, `additionalProperties: false` di mana saja, dan tidak ada `$ref` yang belum terselesaikan. Jika kamu melanggarnya, API akan mengembalikan 400 pada waktu permintaan.

### Pydantic, pengikatan Python

Pydantic v2 menghasilkan Skema JSON dari model berbentuk kelas data melalui `model_json_schema()`. Pydantic AI membungkus ini sehingga kamu menulis:

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

dan framework agen menerjemahkan skema ke mode ketat OpenAI, Anthropic `input_schema`, atau Gemini `responseSchema` di edge. Output model muncul kembali sebagai instance `Invoice` yang diketik. Kesalahan validasi memunculkan `ValidationError` dengan jalur kesalahan yang diketik.

### Zod, pengikatan TypeScript

Zod (`z.object({customer: z.string(), ...})`) setara dengan TS. Node SDK OpenAI mengekspos `zodResponseFormat(Invoice)` yang diterjemahkan menjadi payload Skema JSON API.

### Penolakan

Mode ketat tidak dapat memaksa model untuk menjawab. Jika input tidak sesuai dengan skema ("email adalah puisi, bukan faktur"), model akan mengeluarkan bidang `refusal` yang berisi alasannya. Code kamu harus menangani ini sebagai hasil kelas satu, bukan kegagalan. Penolakan ini juga berguna sebagai sinyal keamanan: model yang diminta untuk mengekstrak nomor kartu kredit dari email dengan konten yang dilindungi akan mengembalikan penolakan dengan disertai alasan keamanan.

### Dekode terbatas di tempat terbuka

Implementasi weight terbuka menggunakan tiga teknik.

1. **Decoding berbasis tata bahasa** (`outlines`, `guidance`, `lm-format-enforcer`): membuat robot terbatas deterministik dari skema; di setiap langkah, tutupi logit token yang akan melanggar FSM.
2. **Logit masking dengan parser JSON**: menjalankan parser JSON streaming yang sejalan dengan model; di setiap langkah, hitung kumpulan token berikutnya yang valid.
3. **Decoding spekulatif dengan verifikator**: model draf murah mengusulkan token, verifikator menerapkan skema.

Penyedia komersial memilih salah satu dari ini di belakang layar. Teknologi canggih pada tahun 2026 lebih cepat daripada pembangkitan biasa untuk output terstruktur pendek dan kecepatannya kira-kira sama untuk output panjang.

### Tiga mode kegagalan

1. **Kesalahan penguraian.** Outputnya bukan JSON yang valid. Tidak dapat terjadi dalam mode ketat. Masih bisa terjadi pada penyedia yang tidak ketat.
2. **Pelanggaran skema.** Outputnya diurai tetapi melanggar skema. Tidak dapat terjadi dalam mode ketat. Umum di luar itu.
3. **Penolakan.** Model ditolak. Harus ditangani sebagai hasil yang diketik.

### Coba lagi strategi

Saat kamu berada di luar mode ketat (penggunaan alat Antropik, OpenAI non-ketat, Gemini lama), pola pemulihannya adalah:

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```Satu kali percobaan biasanya sudah cukup. Tiga percobaan ulang berhasil menangkap kesalahan model yang lemah. Di atas tiga adalah tanda skema yang buruk: model tidak dapat memenuhinya untuk beberapa input, dan prompt atau skema perlu diperbaiki.

### Dukungan model kecil

Penguraian code terbatas berfungsi pada model kecil. Model terbuka berparameter 3B dengan penerapan tata bahasa mengungguli model berparameter 70B dengan prompt mentah pada tugas terstruktur. Inilah alasan utama mengapa output terstruktur penting bagi produksi: hal ini memisahkan keandalan dari ukuran model.

## Pakai

`code/main.py` mengirimkan validator JSON Schema 2020-12 minimal di stdlib (tipe, wajib, enum, min/maks, pola, item, properti tambahan). Ini membungkus skema `Invoice` dan menjalankan output LLM palsu melalui validator, menunjukkan kesalahan penguraian, pelanggaran skema, dan jalur penolakan. Tukar output palsu dengan respons nyata penyedia mana pun dalam produksi.

Apa yang harus dilihat:

- Validator mengembalikan daftar `[ValidationError]` yang diketik dengan jalur dan pesan. Itu adalah bentuk yang kamu inginkan untuk muncul pada prompt coba lagi.
- Cabang penolakan TIDAK mencoba lagi. Ini mencatat dan mengembalikan penolakan yang diketik. Fase 14 · 09 menggunakan penolakan sebagai sinyal keselamatan.
- Pemeriksaan `additionalProperties: false` diaktifkan pada input pengujian adversarial, menunjukkan mengapa mode ketat menutup pintu pada bidang halusinasi.

## Kirim

Lesson ini menghasilkan `outputs/skill-structured-output-designer.md`. Dengan adanya target ekstraksi teks bebas (faktur, tiket dukungan, resume, dll.), keterampilan tersebut menghasilkan Skema JSON 2020-12 yang kompatibel dengan mode ketat dan model Pydantic yang mencerminkannya, dengan penolakan yang diketik dan penanganan percobaan ulang dihentikan.

## Latihan

1. Jalankan `code/main.py`. Tambahkan kasus uji keempat yang `total_usd` adalah angka negatif. Konfirmasikan validator menolaknya dengan jalur batasan `minimum`.

2. Perluas validator untuk mendukung `oneOf` dengan diskriminator. Kasus umum: `line_item` adalah produk atau layanan, ditandai dengan `kind`. Mode ketat memiliki aturan halus di sini; periksa panduan output terstruktur OpenAI.

3. Tulis skema Faktur yang sama dengan Pydantic BaseModel dan bandingkan output `model_json_schema()` dengan skema buatan tangan kamu. Identifikasi satu bidang yang ditetapkan Pydantic secara default yang dihilangkan oleh versi lintingan tangan.

4. Mengukur tingkat penolakan. Buat sepuluh input yang tidak boleh diekstraksi (lirik lagu, bukti matematika, email kosong) dan jalankan melalui penyedia nyata dengan mode ketat. Hitung penolakan vs output halusinasi. Ini adalah kebenaran dasar kamu untuk percobaan ulang yang sadar akan penolakan.

5. Baca panduan output terstruktur OpenAI dari atas ke bawah. Identifikasi konstruksi yang dilarang secara eksplisit dalam mode ketat yang diizinkan oleh Skema JSON biasa. Kemudian rancang skema yang menggunakan konstruksi terlarang secara tidak penting dan lakukan refaktorisasi agar benar-benar kompatibel.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Skema JSON 2020-12 | "Spesifikasi skema" | Dialek skema rancangan IETF yang digunakan setiap penyedia modern |
| Modus ketat | "Skema terjamin" | Bendera OpenAI yang menerapkan skema melalui decoding terbatas |
| Penguraian code terbatas | "Penyembunyian logit" | Penegakan waktu dekode yang menutupi token berikutnya yang tidak valid |
| Penolakan | "Model menurun" | Hasil yang diketik ketika input tidak sesuai dengan skema |
| Kesalahan penguraian | "JSON tidak valid" | Output tidak diurai sebagai JSON; tidak mungkin di bawah ketat |
| Pelanggaran skema | "Bentuk salah" | Diurai tetapi dilanggar tipe/wajib/enum/rentang |
| `additionalProperties: false` | "Tidak ada tambahan yang diperbolehkan" | Melarang bidang yang tidak diketahui; diperlukan dalam OpenAI ketat |
| Model Dasar Pydantic | "Output yang diketik" | Kelas Python yang memancarkan dan memvalidasi Skema JSON |
| Skema Zod | "Jenis output TypeScript" | Skema runtime TS untuk validasi output penyedia |
| Penegakan tata bahasa | "Dekode terbatas dengan weight terbuka" | Logit masking berbasis FSM, seperti pada garis besar/panduan |

## Bacaan Lanjutan

- [OpenAI — Output terstruktur](https://platform.openai.com/docs/guides/structured-outputs) — mode ketat, penolakan, dan persyaratan skema
- [OpenAI — Memperkenalkan output terstruktur](https://openai.com/index/introducing-structured-outputs-in-the-api/) — postingan peluncuran Agustus 2024 yang menjelaskan jaminan decoding
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — mengetikkan pengikatan output_type yang bersambung ke setiap penyedia
- [Skema JSON — catatan rilis 2020-12](https://json-schema.org/draft/2020-12/release-notes) — spesifikasi kanonik
- [Microsoft — Output terstruktur di Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — catatan penerapan perusahaan dan peringatan mode ketat
