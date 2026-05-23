# Ekonomi Caching Cepat dan Caching Semantik

> **Cuplikan harga tertanggal 2026-04.** Klaim numerik di bawah mencerminkan kartu tarif vendor yang diambil dalam publikasi lesson ini; verifikasi terhadap dokumen tertaut sebelum mengutipnya di bagian hilir.

> Caching terjadi pada dua layer. Cache cepat/awalan L2 (tingkat penyedia) menggunakan kembali attention KV untuk awalan berulang — Dokumen cache cepat Anthropic mengiklankan pengurangan biaya hingga 90% dan pengurangan latensi 85% pada prompt panjang; untuk pembacaan cache Claude 3.5 Sonnet adalah $0,30/M vs $3,00/M segar dengan TTL 5 menit dan premium tulis 2x untuk opsi TTL 1 jam (docs.anthropic.com, 2026-04). Caching prompt OpenAI berlaku secara otomatis untuk prompt ≥1024 token dan harga input cache dengan diskon sekitar 90% vs yang baru (platform.openai.com, 2026-04); tarif cache per model yang tepat bergantung pada kartu tarif langsung. Caching semantik L1 (tingkat aplikasi) melewatkan LLM sepenuhnya saat embed hit kesamaan. Vendor "akurasi 95%" mengacu pada ketepatan kecocokan, bukan tingkat keberhasilan — tingkat keberhasilan produksi yang dilaporkan berkisar dari 10% (obrolan terbuka) hingga 70% (FAQ terstruktur); tidak ada penyedia yang menerbitkan data dasar resmi, jadi perlakukan ini sebagai telemetri komunitas, bukan jaminan. Jebakan produksi: paralelisasi mematikan caching (N permintaan paralel yang dikeluarkan sebelum penulisan cache pertama dapat membengkak menghabiskan beberapa kali lipat), dan konten dinamis di dalam awalan mencegah cache mencapai seluruhnya. ProjectDiscovery melaporkan perpindahan rasio hit dari 7% menjadi 74% (2025-11) dengan memindahkan teks dinamis dari awalan yang dapat di-cache.

**Type:** Learn
**Language:** Python (stdlib, mainan simulator cache dua lapis)
**Prerequisites:** Fase 17 · 04 (vLLM Melayani Internal), Fase 17 · 06 (SGLang RadixAttention)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Bedakan caching prompt/awalan L2 (penggunaan kembali KV di penyedia) dari caching semantik L1 (bypass LLM pada prompt serupa).
- Jelaskan penandaan eksplisit `cache_control` Anthropic dan dua opsi TTL (5 menit vs 1 jam) beserta pengganda harganya.
- Hitung penghematan bulanan yang diharapkan berdasarkan tingkat keberhasilan, campuran prompt/respons, dan harga token.
- Sebutkan anti-pola paralelisasi yang meningkatkan tagihan sebesar 5-10x dan anti-pola konten dinamis yang mengurangi tingkat hit.

## Masalah

kamu menambahkan cache cepat ke layanan RAG kamu. Tagihannya tetap datar. kamu mengukur tingkat keberhasilan; itu adalah 7%. Prompt kamu terlihat statis tetapi sebenarnya tidak — system prompt menyertakan tanggal saat ini yang diformat menjadi menit, ID permintaan, dan contoh penyusunan ulang secara acak untuk keberagaman. Setiap permintaan menulis entri cache baru, terbaca nol.

Secara terpisah, agen kamu menjalankan sepuluh panggilan alat paralel per pertanyaan pengguna. Kesepuluhnya tiba di penyedia sebelum penulisan cache pertama selesai. Sepuluh menulis, nol membaca. Tagihan kamu 5-10x lipat dari biaya "dengan caching".

Caching adalah sebuah protokol, bukan sebuah bendera. Dua layer, dua mode kegagalan berbeda.

## Konsep

### L2 — prompt penyedia/cache awalan

Penyedia menyimpan attention KV untuk awalan yang dapat di-cache dan menggunakannya kembali pada permintaan berikutnya yang cocok dengan awalan tersebut. kamu membayar biaya menulis sekali, membaca hampir gratis.**Antropik (seri Claude 3.5 / 3.7 / 4)**: penanda `cache_control` eksplisit dalam permintaan. kamu menandai blok mana yang dapat di-cache. TTL: 5 menit (biaya tulis 1,25x basis) atau 1 jam (biaya tulis 2x basis). Cache berbunyi: $0,30/M di Claude 3.5 Sonnet vs $3,00/M baru — 10x lebih murah (docs.anthropic.com, mulai 2026-04). Tarif berbeda untuk setiap model (Opus/Haiku diterbitkan terpisah); selalu periksa ulang halaman harga langsung.

**OpenAI**: cache otomatis untuk prompt ≥1024 token (platform.openai.com, 2026-04). Tidak ada bendera eksplisit. Input dalam cache kira-kira 10x lebih murah dibandingkan input baru pada kartu tarif gpt-4o/gpt-5 saat ini. Baik dokumen maupun catatan rilis tidak mempublikasikan garis dasar tingkat keberhasilan resmi; laporan masyarakat mengelompok sekitar 30–60% dengan desain cepat yang cermat. Pantau `usage.cached_tokens` untuk mengukur sendiri.

**Google (Gemini)**: cache konteks melalui API eksplisit; Konteks 1 juta token berarti caching membayar lebih banyak lagi.

**Dihosting sendiri (vLLM, SGLang)**: Fase 17 · 06 mencakup RadixAttention — pola yang sama di komputasi kamu sendiri.

### L1 — cache semantik tingkat aplikasi

Sebelum memanggil LLM sama sekali, lakukan hash pada prompt, sematkan, dan cari permintaan cache serupa (kesamaan kosinus di atas ambang batas, biasanya 0,95+). Saat dipukul, kembalikan respons yang di-cache. Jika tidak terjawab, hubungi LLM dan simpan hasilnya.

Sumber terbuka: Kesamaan Vector Redis, GPTCache, Qdrant. Komersial: Portkey Cache, Helicone Cache.

Klaim akurasi vendor mengacu pada seberapa sering respons cache yang dikembalikan sesuai secara semantik — bukan seberapa sering kamu menekan. Tingkat keberhasilan produksi:

- Obrolan terbuka: 10-15%.
- FAQ / dukungan terstruktur: 40-70%.
- Pertanyaan code: 20-30% (varian kecil membunuh hit).
- Agen suara mengulangi prompt: 50-80% (normalisasi suara tetap).

### Anti-pola paralelisasi

Agen kamu melakukan 10 panggilan alat secara paralel. Kesepuluhnya memiliki prompt sistem token 4K yang sama. Penulisan cache antropik dilakukan berdasarkan permintaan; penulisan cache pertama selesai sekitar 300 ms setelah penyedia melihat prompt tersebut. Permintaan 2-10 tiba di jendela milidetik yang sama dan masing-masing melihat cache hilang. kamu membayar 10 premi tulis, 0 diskon baca.

Perbaiki: batch dengan berurutan terlebih dahulu — buat permintaan 1 saja, lalu aktifkan 2-10 setelah cache 1 terisi. Menambahkan 300 ms ke panggilan alat pertama; menghemat 5-10x tagihan.

### Anti-pola konten dinamis

Prompt sistem kamu terlihat seperti:

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

Setiap permintaan itu unik. Setiap permintaan menulis. Nol hit.

Perbaiki: pindahkan semuanya yang benar-benar statis ke awalan yang dapat di-cache; tambahkan konten dinamis setelah batas cache:

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery memindahkan tingkat cache hit dari 7% menjadi 74% dengan cara ini dan menerbitkan anatominya.

### Tumpukan batch + cache untuk weight kerja semalaman

API Batch (Fase 17 · 15) memberikan diskon 50% dengan penyelesaian 24 jam. Input yang di-cache di atas membuat kamu ~10x lebih dari itu. Weight kerja klasifikasi, pelabelan, dan pembuatan laporan dalam semalam dapat turun hingga ~10% dari biaya yang tidak di-cache secara sinkron dengan menumpuk.

### Nomor yang harus kamu ingat

Poin harga diambil pada tahun 2026-04 dari dokumen vendor tertaut dan terjadi setiap beberapa bulan — periksa ulang sebelum mengandalkannya.- Pembacaan cache antropik: $0,30/M pada Claude 3.5 Soneta, kira-kira 10x lebih murah daripada input baru (docs.anthropic.com).
- Premium penulisan cache antropik: 1,25x (TTL 5 menit) atau 2x (TTL 1 jam).
- Cache otomatis OpenAI: berlaku untuk permintaan ≥1024 token; input yang di-cache dengan harga sekitar 10% dari input baru pada kartu tarif saat ini (platform.openai.com).
- Tingkat cache hit semantik (dilaporkan komunitas): ~10% obrolan terbuka; hingga ~70% FAQ terstruktur. Bukan data dasar yang didokumentasikan oleh vendor.
- ProjectDiscovery: 7% → 74% hit rate dengan memindahkan dinamis dari awalan (blog proyek, 2025-11).
- Anti-pola paralelisasi: laporan umum tentang inflasi tagihan 5–10x ketika N permintaan paralel melewatkan penulisan cache pertama.

## Pakai

`code/main.py` menyimulasikan cache L1 + L2 pada weight kerja campuran. Laporan mengenai tarif, tagihan, dan menunjukkan penalti paralelisasi.

## Kirim

Lesson ini menghasilkan `outputs/skill-cache-auditor.md`. Mengingat template dan lalu lintas yang cepat, mengaudit kemampuan cache dan merekomendasikan restrukturisasi.

## Latihan

1. Jalankan `code/main.py`. Alihkan tanda paralelisasi. Berapa perubahan tagihannya?
2. Prompt sistem kamu memiliki tanggal. Keluarkan. Tampilkan matematika tingkat hit sebelum/sesudah.
3. Hitung titik impas untuk TTL 1 jam (2x tulis) vs TTL 5 menit (1,25x tulis) berdasarkan tingkat kedatangan permintaan kamu.
4. Cache semantik pada ambang batas 0,95 mencapai 20%. Pada 0,85, ini mencapai 50% tetapi kamu melihat respons cache yang salah. Pilih ambang batas yang tepat dan berikan justifikasi.
5. kamu mengelompokkan 10 subkueri paralel per pertanyaan pengguna. Tulis ulang agar ramah cache tanpa menambahkan latensi end-to-end.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Cache cepat L2 | "cache awalan" | Penyedia menyimpan KV untuk awalan berulang |
| `cache_control` | "Penanda cache antropik" | Atribut eksplisit menandai blok yang dapat di-cache |
| Tulis cache premium | "tulis pajak" | Biaya tambahan untuk kesalahan cache pertama (1,25x atau 2x) |
| Cache semantik L1 | "embed cache" | Hash-dan-sematkan tingkat aplikasi sebelum memanggil LLM |
| GPTCache | "Lib cache LLM" | Pustaka cache OSS L1 yang populer |
| Tingkat hit cache | "hit / total" | Sebagian kecil permintaan dilayani dari cache |
| Anti-pola paralelisasi | "perangkap N-tulis" | N permintaan paralel kehilangan cache N kali |
| Perangkap konten dinamis | "perangkap waktu segera" | Byte dinamis dalam awalan kill hit rate |
| RadixPerhatian | "cache intra-replika" | Implementasi cache awalan SGLang |

## Bacaan Lanjutan

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — semantik dan TTL resmi `cache_control`.
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — perilaku dan kelayakan caching otomatis.
- [TianPan — Caching Semantik untuk Produksi LLM](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Memotong Biaya LLM 59% Dengan Caching Cepat](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Caching Cepat](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
