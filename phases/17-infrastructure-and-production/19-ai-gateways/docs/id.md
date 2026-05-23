# Gerbang AI — LiteLLM, Portkey, Gerbang AI Kong, Bifrost

> Gerbang berada di antara aplikasi dan penyedia model kamu. Feature inti adalah perutean penyedia, fallback, percobaan ulang, pembatasan kecepatan, referensi rahasia, kemampuan observasi, pagar pembatas. Pembagian pasar pada tahun 2026: **LiteLLM** adalah MIT OSS dengan 100+ penyedia, kompatibel dengan OpenAI, tetapi rusak sekitar ~2000 RPS (memori 8 GB, kegagalan berjenjang dalam tolok ukur yang dipublikasikan); terbaik untuk Python, <500 RPS, dev/prototyping. **Portkey** diposisikan pada bidang kontrol (pagar pembatas, redaksi PII, deteksi jailbreak, jalur audit), menggunakan sumber terbuka Apache 2.0 Maret 2026, overhead latensi 20-40 ms, tingkat produksi $49/bln. **Kong AI Gateway** dibuat di Kong Gateway — benchmark milik Kong pada 12 CPU yang sama: 228% lebih cepat dibandingkan Portkey, 859% lebih cepat dibandingkan LiteLLM; Harga $100/model/bulan (maks 5 pada tingkat Plus); cocok untuk perusahaan jika kamu sudah menggunakan Kong. **Bifrost** (Maxim AI) — percobaan ulang otomatis dengan backoff yang dapat dikonfigurasi, kembali ke Anthropic di OpenAI 429. **Cloudflare / Vercel AI Gateways** — percobaan ulang dasar yang terkelola, tanpa operasi. Residensi data mendorong keputusan self-host; Portkey dan Kong berada di tengah dengan OSS + opsional yang dikelola.

**Type:** Learn
**Language:** Python (stdlib, simulator perutean gerbang mainan)
**Prerequisites:** Fase 17 · 01 (Platform LLM Terkelola), Fase 17 · 16 (Perutean Model)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Sebutkan enam feature gateway inti (routing, fallback, retries, rate limit, secret, observability, guardrails).
- Memetakan empat gateway 2026 (LiteLLM, Portkey, Kong AI, Bifrost) untuk menskalakan batas atas dan kasus penggunaan.
- Kutip benchmark Kong (228% vs Portkey, 859% vs LiteLLM) dan jelaskan mengapa hal ini penting untuk >500 RPS.
- Pilih yang dihosting sendiri vs dikelola berdasarkan residensi data dan anggaran operasi.

## Masalah

Produk kamu memanggil OpenAI, Anthropic, dan Llama yang dihosting sendiri. Setiap penyedia memiliki SDK, model kesalahan, batas tarif, dan skema autentikasi yang berbeda. kamu menginginkan failover (jika OpenAI 429s, coba Anthropic), penyimpanan kredensial tunggal, observabilitas terpadu, dan batas tarif per penyewa.

Menemukan kembali hal ini pada layer aplikasi memasangkan setiap layanan ke setiap penyedia. Layer gateway menggabungkannya menjadi satu proses dengan satu API (biasanya kompatibel dengan OpenAI) yang menyebar ke penyedia.

## Konsep

### Enam feature inti

1. **Perutean penyedia** — OpenAI, Anthropic, Gemini, yang dihosting sendiri, dll. di balik satu API.
2. **Fallback** — pada 429, 5xx, atau kualitas gagal, coba lagi di tempat lain.
3. **Percobaan ulang** — kemunduran eksponensial, upaya terbatas.
4. **Batas tarif** — per penyewa, per kunci, per model.
5. **Referensi rahasia** — menarik kredensial dari vault saat runtime (tidak pernah ada di aplikasi).
6. **Observability** — Atribut OTel + GenAI (Fase 17 · 13) + atribusi biaya.
7. **Pagar Pembatas** — Redaksi PII, deteksi jailbreak, filter topik yang diizinkan.

### LiteLLM — MIT OSS, Python

- 100+ penyedia, kompatibel dengan OpenAI, konfigurasi router, fallback, kemampuan observasi dasar.
- Memecah sekitar 2000 RPS dalam patokan Kong; Jejak memori 8 GB, kegagalan berjenjang pada weight berkelanjutan.
- Paling cocok: Aplikasi Python, <500 RPS, gateway dev/staging, perutean eksperimental.
- Biaya: $0 untuk OSS; tingkat bebas cloud ada.

### Portkey — mengontrol posisi bidang

- Apache 2.0 OSS per Maret 2026. Pagar pembatas, redaksi PII, deteksi jailbreak, jalur audit.
- Overhead latensi 20-40 ms per permintaan.
- $49/bln untuk tingkat produksi dengan retensi + SLA.
- Paling cocok: industri teregulasi yang memerlukan pagar pembatas + kemampuan observasi yang digabungkan.

### Kong AI Gateway — permainan skala- Dibangun di Kong Gateway (produk gateway API matang, lua+OpenResty).
- Tolok ukur Kong sendiri pada setara 12-CPU: 228% lebih cepat dari Portkey, 859% lebih cepat dari LiteLLM.
- Harga: $100/model/bulan, maks 5 pada tingkat Plus.
- Paling cocok: sudah ada di Kong; >1000RPS; bersedia memberi izin.

### Bifrost (Maxim AI)

- Percobaan ulang otomatis dengan backoff yang dapat dikonfigurasi.
- Penggantian ke Anthropic di OpenAI 429 adalah resep kanonik.
- Peserta baru; komersial.

### Gerbang Cloudflare AI / Gerbang Vercel AI

- Terkelola, operasi nol. Percobaan ulang dasar dan kemampuan observasi.
- Paling cocok: Aplikasi JavaScript yang menyajikan edge di Cloudflare/Vercel.
- Terbatas dibandingkan dengan Kong/Portkey pada pagar pembatas dan batasan tarif.

### Dihosting sendiri vs dikelola

Residensi data adalah fungsi pemaksaan. Host mandiri layanan kesehatan dan keuangan (LiteLLM atau Portkey OSS atau Kong). Produk konsumen terkelola secara default (Cloudflare AI Gateway) atau tingkat menengah (dikelola Portkey). Hibrid: dihosting sendiri untuk penyewa teregulasi, dikelola untuk penyewa lain.

### Anggaran latensi

- LiteLLM: tipikal overhead 5-15 ms.
- Portkey: 20-40 ms di atas kepala.
- Kong: 3-8 ms di atas kepala.
- Cloudflare/Vercel: overhead 1-3 ms (keunggulan tepi).

Latensi gateway langsung ditambahkan ke TTFT. Untuk TTFT P99 <100 ms SLA, Kong, atau Cloudflare. Untuk P99 <500 ms, apa saja.

### Semantik batas nilai penting

Token-bucket sederhana berfungsi hingga skala sedang. Multi-penyewa memerlukan jendela geser + tunjangan burst + tingkatan per penyewa. LiteLLM mengirimkan token-bucket; Jendela geser kapal Kong; Kapal Portkey berjenjang.

### Gerbang + kemampuan observasi + penulisan perutean

Fase 17 · 13 (observabilitas) + 16 (perutean model) + 19 (gateway) adalah layer yang sama dalam produksi. Pilih satu alat yang mencakup ketiganya atau sambungkan dengan hati-hati: sebagian besar penerapan pada tahun 2026 menggabungkan Helicone (observabilitas) atau Portkey (pagar pembatas) dengan Kong (skala) untuk peran terpisah.

### Nomor yang harus kamu ingat

- LiteLLM: rusak pada ~2000 RPS, memori 8 GB.
- Portkey: 20-40 ms di atas kepala; Apache 2.0 sejak Maret 2026.
- Kong: 228% lebih cepat dari Portkey, 859% lebih cepat dari LiteLLM.
- Harga Kong: $100/model/bulan, maksimal 5 pada tingkat Plus.
- Cloudflare/Vercel: 1-3 ms di atas kepala di bagian tepi.

## Pakai

`code/main.py` menyimulasikan perutean gateway dengan fallback di 3 penyedia dengan injeksi 429/5xx. Melaporkan latensi, tingkat percobaan ulang, dan tingkat keberhasilan fallback.

## Kirim

Lesson ini menghasilkan `outputs/skill-gateway-picker.md`. Mengingat skala, postur operasi, kepatuhan, anggaran latensi, pilihan gerbang.

## Latihan

1. Jalankan `code/main.py`. Konfigurasikan fallback dari OpenAI→Anthropic→self-hosted. Berapa tingkat keberhasilan yang diharapkan pada tingkat kesalahan penyedia 5%?
2. SLA kamu adalah TTFT P99 <200 ms pada garis dasar 300 ms. Gerbang mana yang sesuai anggaran?
3. Pelanggan layanan kesehatan memerlukan hosting mandiri + redaksi PII + audit. Pilih Portkey OSS atau Kong.
4. Bandingkan LiteLLM vs Kong: pada batas RPS berapa tim harus bermigrasi?
5. Rancang kebijakan batas tarif untuk SaaS multi-penyewa: tingkat gratis, tingkat uji coba, tingkat berbayar. Token-ember atau jendela geser?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Gerbang | "broker API" | Proses duduk antara aplikasi dan penyedia |
| LiteLLM | "yang MIT" | Python OSS, 100+ penyedia, mencapai 2K RPS |
| Kunci Port | "gerbang pagar pembatas" | Bidang kendali + kemampuan observasi, Apache 2.0 |
| Gerbang Kong AI | "yang berskala" | Dibangun di Kong Gateway, pemimpin benchmark |
| es beku | "Gerbang Maxim" | Percobaan ulang + Resep fallback antropis |
| Gerbang AI Cloudflare | "tepi dikelola" | Gerbang terkelola yang diterapkan di tepi, operasi nol |
| Redaksi PII | "penghapusan data" | Regex + NER mask sebelum dikirim ke model |
| Deteksi jailbreak | "pelindung injeksi cepat" | Pengklasifikasi berdasarkan input pengguna |
| Jejak audit | "log yang diatur" | Catatan yang tidak dapat diubah dari setiap panggilan LLM |
| Token-ember | "batas tarif sederhana" | Pembatas tarif berbasis isi ulang |
| Jendela geser | "batas tarif yang tepat" | Pembatas laju jangka waktu; keadilan yang lebih baik |

## Bacaan Lanjutan

- [Tolok Ukur Gerbang Kong AI](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — Perbandingan AI Gateways 2026](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Alat Gerbang LLM Teratas 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Dokumen Kong AI Gateway](https://docs.konghq.com/gateway/latest/ai-gateway/)
