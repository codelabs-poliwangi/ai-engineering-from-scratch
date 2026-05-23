# Rekayasa Kekacauan untuk Produksi LLM

> Rekayasa kekacauan untuk LLM adalah disiplinnya sendiri pada tahun 2026. Prasyarat sebelum menjalankan eksperimen dalam produksi: SLI/SLO yang ditentukan, kemampuan observasi pelacakan+metrik+log, rollback otomatis, runbook, on-call. Arsitektur memiliki empat bidang: kontrol (penjadwal eksperimen), target (layanan, infra, penyimpanan data), keamanan (penjaga + batalkan + filter lalu lintas), kemampuan observasi (metrik + jejak + log), umpan balik (ke dalam penyesuaian SLO). Pagar pembatas bersifat wajib: peringatan laju pembakaran menjeda eksperimen jika anggaran kesalahan harian melebihi perkiraan 2x; jendela penindasan + korelasi jejak-ID menghilangkan kebisingan peringatan. Irama: kenari kecil mingguan + ulasan SLO; hari pertandingan bulanan + postmortem; audit ketahanan lintas tim + pemetaan ketergantungan setiap triwulan. Eksperimen khusus LLM: kelebihan memori, kegagalan jaringan, pemadaman penyedia, format prompt yang salah, badai penggusuran cache KV. Perkakas: Harness Chaos Engineering (rekomendasi yang diturunkan dari LLM, penurunan skala radius ledakan, integrasi alat MCP); LitmusChaos (CNCF); Chaos Mesh (asli CNCF Kubernetes).

**Type:** Learn
**Language:** Python (stdlib, pelari eksperimen kekacauan mainan)
**Prerequisites:** Fase 17 · 23 (SRE untuk AI), Fase 17 · 13 (Observabilitas)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Sebutkan lima prasyarat rekayasa chaos (SLI/SLO, observabilitas, rollback, runbook, on-call) dan jelaskan mengapa melewatkan hal apa pun dapat merusak praktik.
- Diagram empat bidang (kontrol, target, keamanan, observabilitas) dan putaran umpan balik ke dalam SLO.
- Sebutkan lima eksperimen khusus LLM (kelebihan memori, kegagalan jaringan, pemadaman penyedia, format prompt salah, badai penggusuran KV).
- Pilih alat — Harness, LitmusChaos, Chaos Mesh — tumpukan yang diberikan.

## Masalah

Pengujian kekacauan di tumpukan tradisional telah dilakukan. Tumpukan LLM menambahkan mode kegagalan baru. Prompt token 4K dengan karakter racun menghentikan tokenizer selama 12 detik. Penyedia hulu 429s; gateway kamu mencoba ulang; layanan kamu OOM pada coba lagi konkurensi yang diperkuat. Badai penggusuran cache KV dalam weight burst menyebabkan kaskade pengisian ulang yang memenuhi komputasi.

Tak satu pun dari ini muncul di pengujian unit. Rekayasa kekacauan adalah cara kamu menemukannya sebelum pengguna menemukannya.

## Konsep

### Prasyarat

Jangan membuat kekacauan dalam produksi tanpa:

1. **SLI/SLO** — indikator dan tujuan tingkat layanan yang ditentukan.
2. **Kemampuan untuk diamati** — jejak, metrik, log, yang dihubungkan ke dasbor.
3. **Kembalikan otomatis** — Fase 17 · 20 pengembalian bendera kebijakan.
4. **Runbook** — terstruktur, Fase 17 · 23.
5. **On-call** — seseorang yang akan merespons.

Kehilangan segala cara, kekacauan menjadi kejadian nyata.

### Empat pesawat + umpan balik

**bidang kendali** — penjadwal eksperimen (alur kerja lakmus, jadwal Chaos Mesh, UI Harness).

**Bidang target** — layanan, pod, node, penyeimbang weight, penyimpanan data.

**Pesawat pengaman** — tombol pemutus, jendela penekan, batas radius ledakan, gerbang anggaran kesalahan.

**Bidang observasi** — metrik normal + korelasi ID jejak untuk membedakan kegagalan yang disebabkan oleh kekacauan dan kegagalan alami.

**Feedback loop** — temuan dimasukkan kembali ke dalam penyesuaian SLO, pembaruan runbook, perbaikan code.

### Pagar pembatas adalah wajib

- **Peringatan tingkat pembakaran**: jeda eksperimen jika pembakaran anggaran kesalahan harian melebihi 2x yang diharapkan.
- **Jendela penindasan**: menonaktifkan peringatan non-eksperimen dalam radius ledakan selama eksperimen.
- **Korelasi Pelacakan-ID**: semua kesalahan yang disebabkan oleh eksperimen membawa tag sehingga panggilan dapat menghilangkan penipuan.

### Lima eksperimen khusus LLM1. **Memori kelebihan weight** — memaksa badai preemption cache KV dengan mengirimkan permintaan konteks panjang dengan konkurensi tinggi. Amati: apakah layanan tersebut hilang atau mogok dengan baik?

2. **Kegagalan jaringan** — memutus konektivitas antara gateway inference dan penyedia. Perhatikan: apakah fallback berlaku dalam SLA? (Fase 17 · 19)

3. **Simulasi pemadaman penyedia** — 100% 429 dari OpenAI. Amati: apakah perutean gagal ke Anthropic? (Fase 17 · 16, 19)

4. **Permintaan salah format** — memasukkan payload yang menghambat tokenizer (misalnya, unicode yang sangat bertumpuk, titik code UTF-8 yang sangat besar). Amati: apakah satu permintaan mengunci seorang pekerja?

5. **Badai penggusuran KV** — memaksa penggusuran dengan memenuhi anggaran blok vLLM. Amati: apakah LMCache pulih atau layanan menurun?

### Irama

- **Mingguan** — percobaan kenari kecil dalam pementasan, mungkin 5% hasil.
- **Bulanan** — hari pertandingan terjadwal berdasarkan skenario tertentu; kehadiran lintas tim; postmortem.
- **Triwulanan** — audit ketahanan lintas tim; pembaruan peta ketergantungan.

### Perkakas

- **Harness Chaos Engineering** — komersial; Rekomendasi eksperimen yang diturunkan dari AI; penurunan skala radius ledakan; Integrasi alat MCP.
- **LitmusChaos** — CNCF lulus; Berbasis alur kerja Kubernetes.
- **Chaos Mesh** — kotak pasir CNCF; Gaya CRD asli Kubernetes.
- **GREMLIN** — komersial; dukungan luas.
- **AWS FIS** / **Azure Chaos Studio** — penawaran cloud terkelola.

### Mulai dari yang kecil

Eksperimen pertama: pod-kill satu replika dekode dalam lalu lintas stabil. Amati perubahan rute dan pemulihan. Jika ini berhasil dan terlihat aman, lanjutkan ke kekacauan jaringan.

Eksperimen khusus LLM pertama: menyuntikkan satu penyedia 429 selama 5 menit. Amati kemunduran. Sebagian besar tim menyadari bahwa cadangan mereka belum sepenuhnya teruji.

### Nomor yang harus kamu ingat

- Empat bidang: kontrol, target, keamanan, observasi.
- Jeda laju pembakaran: 2x perkiraan pembakaran anggaran harian.
- Irama: kenari mingguan, hari pertandingan bulanan, audit triwulanan.
- Lima percobaan LLM: memori, jaringan, penyedia, prompt salah, badai KV.

## Pakai

`code/main.py` menyimulasikan tiga eksperimen kekacauan dengan gerbang bidang keselamatan. Melaporkan eksperimen mana yang akan menyebabkan pembatalan laju pembakaran.

## Kirim

Lesson ini menghasilkan `outputs/skill-chaos-plan.md`. Mengingat tumpukan dan kematangan, pilih tiga eksperimen pertama dan perkakasnya.

## Latihan

1. Jalankan `code/main.py`. Eksperimen manakah yang menghambat laju pembakaran dan mengapa?
2. Rancang lima eksperimen chaos pertama untuk layanan RAG berbasis vLLM. Sertakan kriteria keberhasilan.
3. Peringatan laju pembakaran kamu menjeda eksperimen. Bagaimana kamu menentukan akar permasalahan — kekacauan atau alami?
4. Berdebat apakah kekacauan harus dilakukan dalam produksi atau hanya dalam pementasan. Kapan produksi merupakan jawaban yang tepat?
5. Sebutkan tiga mode kegagalan khusus LLM yang tidak dapat direproduksi oleh kekacauan jaringan umum.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| SLI / SLO | "target layanan" | Indikator + tujuan; prasyarat yang diperlukan |
| Radius ledakan | "ruang lingkup" | Kumpulan layanan/pengguna yang terpengaruh oleh eksperimen |
| Peringatan tingkat pembakaran | "gerbang anggaran" | Diaktifkan ketika tingkat pembakaran anggaran kesalahan > 2x yang diharapkan |
| Hari pertandingan | "latihan bulanan" | Latihan kekacauan lintas tim terjadwal |
| LitmusKekacauan | "Alur kerja CNCF" | Alat kekacauan CNCF Kubernetes yang lulus |
| Jaring Kekacauan | "CRD CNCF" | Kotak pasir CNCF Kekacauan asli Kubernetes |
| Memanfaatkan CE | "komersial dengan bantuan AI" | Memanfaatkan kekacauan dengan rekomendasi AI |
| Prompt salah | "bom tokenizer" | Input yang menghentikan tokenization |
| Badai penggusuran KV | "kaskade pencegahan" | Penggusuran massal memicu pengisian ulang |

## Bacaan Lanjutan

- [Sekolah DevSecOps — Panduan Chaos Engineering 2026](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observabilitas untuk LLM (buku)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Teknik Harness Chaos](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
