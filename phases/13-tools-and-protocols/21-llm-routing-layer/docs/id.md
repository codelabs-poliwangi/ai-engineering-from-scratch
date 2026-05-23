# Layer Perutean LLM — LiteLLM, OpenRouter, Portkey

> Penguncian penyedia itu mahal. Weight kerja pemanggilan alat yang berbeda sesuai dengan model yang berbeda. Gateway perutean memberikan satu permukaan API, percobaan ulang, failover, pelacakan biaya, dan pagar pembatas. Tiga arketipe mendominasi pada tahun 2026: LiteLLM (open-source self-hosted), OpenRouter (managed SaaS), Portkey (tingkat produksi, open-source pada Maret 2026). Lesson ini menyebutkan kriteria keputusan dan menjalankan gateway perutean stdlib.

**Type:** Learn
**Language:** Python (stdlib, perutean + failover + pelacak biaya)
**Prerequisites:** Fase 13 · 02 (pemanggilan fungsi), Fase 13 · 17 (gateway)
**Waktu:** ~45 menit

## Tujuan Pembelajaran

- Membedakan opsi perutean yang dihosting sendiri, dikelola, dan tingkat produksi.
- Menerapkan rantai fallback yang mencoba ulang kegagalan penyedia dalam urutan prioritas yang ditentukan.
- Lacak biaya per permintaan dan penggunaan token di seluruh penyedia.
- Putuskan antara LiteLLM, OpenRouter, dan Portkey untuk batasan produksi tertentu.

## Masalah

Skenario di mana perutean penyedia penting:

1. **Biaya.** Harga Claude Sonnet 3x lipat dari harga Haiku. Untuk tugas triase, Haiku sudah cukup; untuk tugas sintesis, Soneta sangat berharga. Rute per permintaan.

2. **Failover.** OpenAI mengalami saat yang buruk. Setiap permintaan gagal. kamu ingin fallback otomatis ke Anthropic tanpa menerapkan ulang.

3. **Latensi.** UI live chat memerlukan waktu pembuatan token pertama yang cepat. Peringkasan batch tidak. Rute berdasarkan SLA latensi.

4. **Kepatuhan.** Pengguna UE harus tinggal di wilayah UE. Rute berdasarkan wilayah.

5. **Eksperimen.** A/B dua model pada weight kerja yang sama. Rute berdasarkan ember uji.

Pengkodean manual semua ini per integrasi bersifat berulang. Gateway perutean memberikan satu API yang kompatibel dengan OpenAI dan menangani sisanya.

## Konsep

### Bentuk proksi yang kompatibel dengan OpenAI

Semua orang berbicara dalam bentuk OpenAI. Gerbang perutean mengekspos `/v1/chat/completions`, menerima skema OpenAI, dan secara internal melakukan proksi ke Anthropic / Gemini / Cohere / Ollama / apa pun. Klien tidak peduli.

### Model alias

Alih-alih `claude-3-5-sonnet-20251022`, code kamu mengatakan `our_smart_model`. Gateway memetakan alias ke model nyata. Saat Anthropic mengirimkan Claude 4, kamu mengubah alias sisi server; code kamu tidak menyentuh apa pun.

### Rantai cadangan

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

Gateway mendefinisikan ini dalam konfigurasi. Percobaan ulang memperhitungkan anggaran sehingga kaskade fallback tidak menambah biaya.

### Caching semantik

Prompt yang identik atau hampir identik mengenai cache, bukan penyedia. Penghematan pada loop agen berulang bisa mencapai 30 hingga 60 persen. Kunci berbasis embedding; prompt yang hampir identik berbagi slot cache.

### Pagar Pembatas

Tingkat gerbang:

- **Redaksi PII.** Pass berbasis Regex atau ML sebelum mengirimkan prompt.
- **Pelanggaran kebijakan.** Menolak prompt dengan konten terlarang.
- **Filter output.** Penyelesaian pembersihan untuk mencari kebocoran.

Portkey dan Kong sama-sama mengirimkan pagar pembatas yang berpendirian keras. LiteLLM membiarkannya opsional.

### Batas tarif per kunci

Satu kunci API = satu tim. Anggaran per kunci mencegah satu tim menggunakan kuota bersama. Kebanyakan gateway mendukung hal ini.

### Pertukaran yang dihosting sendiri vs terkelola| Faktor | LiteLLM (dihosting sendiri) | OpenRouter (dikelola) | Portkey (produksi) |
|--------|----------------------|----------------------|----------------------|
| Code | Sumber terbuka, Python | SaaS Terkelola | Sumber terbuka (Mar 2026) + terkelola |
| Pengaturan | Menyebarkan proxy | Daftar | Baik |
| Penyedia | 100+ | 300+ | 100+ |
| Penagihan | Kunci kamu sendiri | Kredit OpenRouter | Kunci kamu sendiri |
| Observabilitas | OpenTelemetri | Dasbor | Redaksi lengkap Otel + PII |
| Terbaik untuk | Tim yang menginginkan kendali penuh | Pembuatan prototipe cepat | Produksi dengan kepatuhan |

LiteLLM menang jika kamu memiliki tim SRE dan menginginkan kedaulatan data. OpenRouter menang bila kamu menginginkan satu langganan dan tanpa infra. Portkey menang ketika kamu membutuhkan pagar pembatas dan kepatuhan yang siap pakai.

### Pelacakan biaya

Setiap permintaan membawa `provider`, `model`, `input_tokens`, `output_tokens`. Kalikan dengan harga per model per token (diambil dari lembar harga yang dikelola gateway). Agregasi per pengguna / per tim / per proyek.

### MCP ditambah perutean

Gateway dapat merutekan panggilan LLM DAN permintaan pengambilan sample MCP. Ketika modelPreferences permintaan pengambilan sample lebih memilih model tertentu, gateway diterjemahkan ke backend kanan. Di sinilah Fase 13 · 17 (gerbang MCP) dan gerbang perutean lesson ini terkadang digabungkan menjadi satu layanan.

### Strategi perutean

- **Prioritas statis.** Pertama dalam daftar; kembali karena kesalahan.
- **Penyeimbangan weight.** Round-robin atau berbobot.
- **Sadar biaya.** Pilih model termurah yang memenuhi latensi/kualitas.
- **Sadar latensi.** Pilih model tercepat dalam N menit terakhir.
- **Sadar tugas.** Pengklasifikasi cepat mengarahkan pengkodean ke satu model, dan meringkas ke model lainnya.

## Pakai

`code/main.py` mengimplementasikan gateway perutean dalam ~150 baris: menerima permintaan berbentuk OpenAI, menerjemahkan ke stub per penyedia, menjalankan rantai fallback prioritas, melacak biaya per permintaan, dan menerapkan penerusan redaksi PII pada input. Jalankan dengan tiga skenario: permintaan normal, pemadaman penyedia utama yang memicu fallback, kebocoran PII yang tertangkap oleh redaksi.

Apa yang harus dilihat:

- `ROUTES` dict: alias -> daftar penyedia konkrit yang diurutkan berdasarkan prioritas.
- Percobaan ulang loop fallback pada 5xx.
- Pelacak biaya mengalikan penggunaan token dengan tarif per model.
- Redaktor PII menghapus pola berbentuk SSN sebelum meneruskan.

## Kirim

Lesson ini menghasilkan `outputs/skill-routing-config-designer.md`. Mengingat profil weight kerja (latensi, biaya, kepatuhan), keterampilan memilih LiteLLM/OpenRouter/Portkey dan menghasilkan konfigurasi perutean.

## Latihan

1. Jalankan `code/main.py`. Memicu skenario pemadaman; konfirmasikan fallback mendarat pada penyedia kedua dan biaya diatribusikan dengan benar.

2. Tambahkan cache semantik: SHA256 dari prompt adalah kunci pencarian; cache hits kembali secara instan. Ukur penghematan biaya pada panggilan berulang.

3. Tambahkan pengklasifikasi prompt yang merutekan prompt "code..." ke alias yang mengutamakan kecerdasan dan prompt "meringkas..." ke alias yang mengutamakan kecepatan.

4. Rancang anggaran per tim: setiap tim memiliki batas pembelanjaan bulanan; gateway menolak permintaan setelah batas tercapai. Pilih perincian penerapan (per permintaan atau berjendela).

5. Baca dokumen LiteLLM, OpenRouter, dan Portkey secara berdampingan. Sebutkan satu feature yang dikirimkan masing-masing feature yang tidak dimiliki oleh dua feature lainnya.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Gerbang perutean | "Proksi LLM" | Layer permukaan satu-API di depan banyak penyedia |
| Kompatibel dengan OpenAI | "Berbicara skema OpenAI" | Menerima bentuk `/v1/chat/completions`, diterjemahkan ke backend mana pun |
| Alias ​​model | "model_pintar_kami" | Beri nama dalam code kamu yang dipetakan oleh gateway ke model konkret |
| Rantai mundur | "Coba lagi daftar" | Daftar penyedia yang diurutkan dicoba namun gagal |
| Caching semantik | "Cache embedding cepat" | Kuncinya adalah embed prompt; hampir duplikat berbagi cache hit |
| Pagar Pembatas | "Filter input/output" | Sunting PII, tolak pelanggaran kebijakan |
| Batas tarif per kunci | "Anggaran tim" | Kuota terbatas pada kunci API |
| Pelacakan biaya | "Pembelanjaan per permintaan" | Penggunaan token agregat x harga per model |
| LiteLLM | "Proksi terbuka" | Gerbang perutean OSS yang dapat dihosting sendiri |
| BukaRouter | "SaaS yang dikelola" | Gateway yang dihosting dengan penagihan berbasis kredit |
| Kunci Port | "Opsi produksi" | Sumber terbuka + dikelola dengan pagar pembatas bawaan |

## Bacaan Lanjutan

- [LiteLLM — dokumen](https://docs.litellm.ai/) — gerbang perutean yang dihosting sendiri
- [OpenRouter — mulai cepat](https://openrouter.ai/docs/quickstart) — SaaS perutean terkelola
- [Portkey — dokumen](https://portkey.ai/docs) — perutean produksi dengan pagar pembatas
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — panduan keputusan
- [Relayplane — perbandingan gateway LLM 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — survei vendor
