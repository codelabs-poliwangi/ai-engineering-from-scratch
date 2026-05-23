# FinOps untuk LLM — Ekonomi Unit dan Atribusi Multi-Penyewa

> FinOps tradisional membatasi pembelanjaan LLM. Biaya adalah transaksi token, bukan waktu aktif sumber daya. Tag tidak dipetakan — panggilan API adalah transaksi, bukan aset. Keputusan teknik (desain cepat, jendela konteks, panjang output) adalah keputusan finansial. Pedoman tahun 2026 memiliki tiga dimension atribusi untuk diinstrumentasikan pada hari pertama: per pengguna (`user_id`) untuk penetapan harga dan perluasan kursi, per tugas (`task_id` + `route`) untuk biaya permukaan produk dan prioritas, per penyewa (`tenant_id`) untuk ekonomi unit dan pembaruan. Empat layer token — prompt, alat, memori, respons — satu keranjang menyembunyikan pembelanjaan. Tangga penerapan untuk produk multi-penyewa: batas tarif per penyewa (2-3x puncak yang diharapkan, selesaikan 429 + coba lagi setelahnya); batas pembelanjaan harian (1,5-3x batas atas kontrak; memicu pengetatan suku bunga + peringatan); matikan tombol saat pembelanjaan z-score > 4 (jeda otomatis + halaman saat panggilan). Pola atribusi: tag-dan-agregat, penggabung telemetri (ID pelacakan → penagihan; akurasi tertinggi), pengambilan sample dan ekstrapolasi, alokasi berbasis model, bersumber peristiwa, streaming waktu nyata. Metrik unit: biaya per kueri yang terselesaikan, biaya per artefak yang dihasilkan — bukan $/M token. Pemberian tag retroaktif selalu meleset; instrumen saat pembuatan permintaan.

**Type:** Learn
**Language:** Python (stdlib, simulator atribusi biaya mainan dengan tombol pemutus)
**Prerequisites:** Fase 17 · 13 (Kemampuan Observabilitas), Fase 17 · 14 (Caching)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Jelaskan mengapa FinOps tradisional (tag + tingkatan) tidak sesuai dengan pembelanjaan LLM dan sebutkan tiga dimension atribusi baru.
- Hitung empat layer token (prompt, alat, memori, respons) dan mengapa penagihan satu keranjang menyembunyikan biaya.
- Rancang tangga penegakan hukum (tarif → batas pembelanjaan → tombol pemutus) untuk produk multi-penyewa.
- Pilih metrik unit (biaya per kueri/artefak yang terselesaikan) alih-alih token $/M.

## Masalah

Tagihan kamu tertulis $40.000. kamu tidak tahu:
- Penyewa mana yang menghabiskannya.
- Feature produk apa yang mendorongnya.
- Apakah ada pengguna yang melakukan pelecehan.
- Apakah penyebabnya adalah pembengkakan yang cepat, panggilan alat, atau penguatan memori.

Tag-dan-agregat di sisi penyedia berfungsi untuk sumber daya cloud (EC2, S3) tempat tag disebarkan ke item baris. Panggilan API LLM tidak diberi tag otomatis — kamu harus memberi stempel pengguna/tugas/penyewa di situs panggilan dan melanjutkan. Atribusi retroaktif selalu melewatkan kasus edge.

## Konsep

### Tiga dimension atribusi

**Per-pengguna** (`user_id`): siapa yang mengeluarkan biaya berapa. Mendorong penetapan harga kursi, percakapan perluasan, mengidentifikasi pengguna yang mahir.

**Per tugas** (`task_id` + `route`): berapa biaya permukaan produk. Mendorong prioritas feature, keputusan mematikan feature mahal.

**Per-penyewa** (`tenant_id`): pelanggan mana yang diuntungkan. Mendorong keekonomian unit, penetapan harga pembaruan, ambang batas tingkat.

Instrumen ketiganya di lokasi panggilan pada hari pertama. Retroaktif selalu lebih buruk.

### Empat layer token

| Layer | Contoh | % tipikal dari total |
|-------|---------|---------------------|
| Prompt | sistem + input pengguna | 40-60% |
| Alat | umpan balik hasil panggilan alat | 20-40% (weight kerja agen) |
| Memori | percakapan sebelumnya / dokumen yang diambil | 10-30% |
| Tanggapan | output model | 10-30% |

Menggabungkan keempatnya bersama-sama membuat optimization menjadi buta. Pisahkan semuanya dalam skema atribusi kamu.

### Tangga penegakan1. **Batas tarif** per penyewa. 2-3x puncak yang diharapkan. Kembalikan 429 dengan `Retry-After`. Penyewa melihat gesekan; tidak ada tagihan kejutan.

2. **Batas pembelanjaan harian** per penyewa. Plafon kontrak 1,5-3x. Pemicu: memperketat batas tarif + memperingatkan keberhasilan pelanggan.

3. **Matikan tombol** pada skor-z pembelanjaan > 4 relatif terhadap garis dasar penyewa. Penyewa jeda otomatis; halaman panggilan; tingkatkan ke ops + CS.

### Pola atribusi

- **Tag-dan-agregat**: stempel header metadata; agregat nanti. Sederhana; kasar.
- **Penggabung telemetri**: menggabungkan pelacakan ke penagihan melalui ID pelacakan. Akurasi tertinggi. Apa yang dilakukan tim yang matang.
- **Pengambilan sample + ekstrapolasi**: sample 5-10%, kalikan. Hemat biaya untuk pembelanjaan kasar; merindukan ekor.
- **Alokasi berbasis model**: regresi untuk menyimpulkan pemicu biaya. Untuk data lama tanpa tag.
- **Bersumber peristiwa**: biaya sebagai peristiwa dalam aliran (Kafka / Kinesis). Waktu nyata.
- **Streaming waktu nyata**: pembaruan dasbor dalam hitungan detik.

### Biaya per X adalah metrik unit

Token $/M adalah pembicaraan vendor. Metrik produk:

- Biaya per tiket dukungan terselesaikan.
- Biaya per artikel yang dihasilkan.
- Biaya per tugas agen yang berhasil.
- Biaya per menit sesi pengguna.

Kaitkan biaya dengan hasil produk. Jika tidak, optimization tidak akan tertahan.

### Bentuk jejak atribusi biaya

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

Memancarkan pada setiap panggilan. Simpan di danau data. Agregat per dimension. Fase 17 · 13 tumpukan observabilitas adalah tempatnya.

### Tumpukan tabungan gabungan

Tumpukan: cache + batch + rute + gateway. Dengan keempatnya:
- Cache L2 (Fase 17 · 14): ~10x input lebih murah.
- Batch (Fase 17 · 15): diskon 50%.
- Rute ke model murah (Phase 17 · 16): pengurangan biaya sebesar 60%.
- Efisiensi gateway (Fase 17 · 19): redundansi + percobaan ulang.

Tumpukan kasus terbaik: ~5-10% dari garis dasar naif. Sebagian besar tim memiliki 2-3 tuas yang aktif; beberapa tumpukan keempatnya.

### Nomor yang harus kamu ingat

- Dimension atribusi: per pengguna, per tugas, per penyewa.
- Empat layer token: prompt, alat, memori, respons.
- Tombol pemutus: gunakan skor-z > 4.
- Metrik unit: biaya per kueri yang terselesaikan, bukan $/M token.
- Optimization bertumpuk: ~5-10% dari kemungkinan dasar.

## Pakai

`code/main.py` menyimulasikan layanan LLM multi-penyewa dengan tangga penegakan tiga tingkat. Menyuntikkan penyewa yang kasar dan mendemonstrasikan pengaktifan tombol pemutus.

## Kirim

Lesson ini menghasilkan `outputs/skill-finops-plan.md`. Berdasarkan produk dan skala, rancang skema atribusi dan jenjang penegakan hukum.

## Latihan

1. Jalankan `code/main.py`. Pada skor z berapa tombol pemutus menyala? Bagaimana kamu memilih ambang batas?
2. Rancang dasbor biaya per penyewa dan per tugas. Apa 5 tampilan yang kamu buat pertama kali?
3. Penyewa terbesar kamu adalah unit-ekonomi-negatif. Usulkan tiga intervensi yang diurutkan berdasarkan dampak pelanggan.
4. Hitung biaya per tiket yang terselesaikan untuk produk dukungan: token/tiket 3 juta, ~800 tiket/hari, tarif cache GPT-5.
5. Perdebatkan apakah pemberian tag retroaktif dapat berhasil. Kapan itu bisa diterima?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Atribusi per pengguna | "biaya tingkat pengguna" | `user_id` dicap pada setiap panggilan |
| Atribusi per tugas | "biaya feature" | `task_id` + `route` mengidentifikasi permukaan produk |
| Atribusi per penyewa | "biaya pelanggan" | `tenant_id`; mendorong ekonomi unit |
| Empat layer token | "layer biaya" | prompt + alat + memori + respons |
| Batas tarif | "429 penjaga" | Plafon per penyewa diberlakukan di gateway |
| Batas pembelanjaan harian | "langit-langit harian" | Anggaran cakupan penyewa dengan peringatan |
| Tombol pemutus | "jeda otomatis" | Belanjakan skor-z > 4 pemicu penangguhan otomatis |
| Biaya per penyelesaian | "metrik unit produk" | Biaya terkait dengan hasil produk, bukan token |
| Penggabung telemetri | "pelacakan ke penagihan" | Pola atribusi dengan akurasi tertinggi |
| Optimization bertumpuk | "cache+batch+rute+gerbang" | Menambah penghematan menjadi ~5-10% baseline |

## Bacaan Lanjutan

- [FinOps Foundation — Tinjauan FinOps untuk AI](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Panduan Biaya per Unit 2026](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Diterapkan - Atribusi Biaya Agen LLM 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — LLM terkelola di Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
