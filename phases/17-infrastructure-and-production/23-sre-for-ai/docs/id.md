# SRE untuk AI — Respons Insiden Multi-Agen, Runbook, Deteksi Prediktif

> AI SRE menggunakan LLM yang didasarkan pada data infrastruktur (log, runbook, topologi layanan) melalui RAG untuk mengotomatiskan fase investigasi, dokumentasi, dan koordinasi. Pola arsitektur tahun 2026 adalah orkestrasi multi-agen — agen khusus (log, metrik, runbook) yang dikoordinasikan oleh supervisor; AI mengajukan hipotesis dan pertanyaan, manusia menyetujui keputusan penilaian. Datadog Bits AI dan Azure SRE Agent mengirimkan ini sebagai produk terkelola. Runbook terus berkembang: NeuBird Hawkeye menggunakan evaluasi permusuhan (dua model menganalisis insiden yang sama; persetujuan = keyakinan, ketidaksepakatan = ketidakpastian); memori operasional tetap ada di seluruh perubahan tim. Remediasi otomatis tetap dilakukan dengan hati-hati: AI menyarankan, manusia menyetujuinya. Tindakan yang sepenuhnya otonom bersifat sempit (restart pod, rollback penerapan khusus) dengan batasan yang ketat — siapa pun yang menjual "atur dan lupakan" adalah penjualan yang berlebihan. Emerging frontier: prediksi pra-insiden. Penelitian MIT melaporkan LLM yang dilatih tentang log historis + suhu GPU + pola kesalahan API memperkirakan 89% pemadaman 10-15 menit lebih awal. Proyeksi: 95% LLM perusahaan mengalami failover otomatis pada akhir tahun 2026.

**Type:** Learn
**Language:** Python (stdlib, mainan simulator triase insiden multi-agen)
**Prerequisites:** Fase 17 · 13 (Observabilitas), Fase 17 · 24 (Rekayasa Kekacauan)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Diagram arsitektur AI SRE multi-agen: supervisor + agen khusus (log, metrik, runbook) + gerbang persetujuan manusia.
- Jelaskan mengapa remediasi otomatis bersifat sempit (mulai ulang pod, kembalikan penerapan) dibandingkan luas (layanan arsitek ulang).
- Sebutkan pola evaluasi permusuhan (NeuBird Hawkeye): dua model setuju = percaya diri; tidak setuju = meningkat.
- Mengutip hasil deteksi dini MIT 89% dan kendala operasionalnya: prediksi tanpa aktuasi hanyalah dasbor.

## Masalah

Seorang teknisi yang bertugas menerima panggilan pada jam 3 pagi. "Tingkat kesalahan tinggi saat checkout." Mereka memeriksa Datadog, Loki, tiga runbook, log penerapan. 30 menit kemudian mereka menyadari bahwa akar masalahnya adalah vLLM OOM dari lonjakan cache KV. Mereka memulai ulang podnya; kesalahan teratasi.

Pada tahun 2026, 20 menit pertama penyelidikan tersebut dapat dilakukan secara otomatis. Mengelompokkan log berdasarkan layanan, menghubungkan dengan penerapan terkini, mencocokkan dengan runbook — semuanya menggunakan alat RAG +. Agen yang diawasi dapat melakukan triase first-pass dan menyajikan hipotesis sebelum manusia membuka Datadog.

Remediasi yang sepenuhnya otonom adalah masalah yang berbeda. Mulai ulang pod: aman. Menskalakan kumpulan GPU: aman jika kebijakan mengizinkan. Merancang ulang layanan: sama sekali tidak. Disiplin menarik garis sempit.

## Konsep

### Arsitektur multi-agen

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

Supervisor memecah insiden menjadi sub-kueri. Agen khusus memiliki akses alat (pencarian log, PromQL, pengambilan dokumen). Supervisor mensintesis, menyajikan hipotesis + bukti kepada manusia. Manusia menyetujui atau mengalihkan.

### Cakupan remediasi otomatis

**Aman (sempit)**: memulai ulang pod, mengembalikan penerapan tertentu, menskalakan kumpulan dalam batas yang telah disetujui sebelumnya, mengaktifkan tanda feature yang telah disetujui sebelumnya.

**Tidak aman (luas)**: mengubah topologi layanan, mengubah batas sumber daya, menerapkan code baru, mengubah IAM, mengubah database.

Siapa pun yang menjual "atur dan lupakan" adalah penjualan yang berlebihan. Perangkat aman berkembang seiring dengan semakin matangnya AI SRE, namun batasannya tetap nyata.

### Evaluasi permusuhan (NeuBird Hawkeye)Dua model secara independen menganalisis kejadian yang sama. Jika mereka sepakat mengenai akar permasalahannya, maka kepercayaan diri akan tinggi. Jika mereka tidak setuju, tingkatkan ke manusia dengan kedua hipotesis terlihat. Pola sederhana, filter efektif terhadap akar penyebab halusinasi.

### Memori operasional

Pergantian tim adalah pembunuhan diam-diam dari SRE tradisional - pengetahuan suku hilang. AI SRE menyimpan runbook + post-mortem dalam DB vector; agen mengambil setiap insiden baru. Ketika insinyur baru bergabung, AI memiliki sejarah yang lengkap.

### Prediksi sebelum insiden

Penelitian MIT tahun 2025: LLM yang dilatih tentang log historis, suhu GPU, pola kesalahan API memperkirakan 89% pemadaman 10-15 menit sebelum terjadi pada set pengujian.

Pemeriksaan realitas: prediksi tanpa aktuasi hanyalah dasbor. Pertanyaan operasionalnya adalah “ketika kita memperkirakan, apa yang kita lakukan?” Pengurasan preventif? Pager? Skala otomatis? Jawabannya tergantung pada kebijakan.

### Produk pada tahun 2026

- **Datadog Bits AI** — mengelola kopilot SRE di dalam Datadog.
- **Agen SRE Azure** — Azure-asli.
- **NeuBird Hawkeye** — evaluasi permusuhan + memori operasional.
- **PagerDuty AIOps** — triase + deduplikasi.
- **Incident.io Autopilot** — komandan insiden + koordinasi.

### Runbook sebagai code

Runbook berevolusi dari halaman Confluence ke penurunan harga berversi dengan bagian terstruktur (gejala, hipotesis, verifikasi, tindakan). Runbook terstruktur memberikan pengambilan RAG yang lebih baik. Mulai peluncuran AI-SRE dengan mengubah runbook tidak terstruktur menjadi terstruktur.

### Nomor yang harus kamu ingat

- Deteksi dini MIT: 89% pemadaman, waktu tunggu 10-15 menit.
- Triase multi-agen: supervisor + (log, metrik, runbook) + manusia.
- Kumpulan remediasi otomatis yang aman: memulai ulang pod, mengembalikan penerapan, menskalakan dalam batas.
- Evaluasi permusuhan: dua model independen; persetujuan = keyakinan.

## Pakai

`code/main.py` menyimulasikan triase multi-agen: agen log menemukan kesalahan, agen metrik menemukan lonjakan CPU, agen runbook cocok dengan masalah umum. Supervisor membuat peringkat hipotesis.

## Kirim

Lesson ini menghasilkan `outputs/skill-ai-sre-plan.md`. Mengingat panggilan saat ini, volume insiden, kematangan tim, merancang peluncuran AI SRE.

## Latihan

1. Jalankan `code/main.py`. Bagaimana jika agen log dan metrik tidak setuju? Bagaimana cara supervisor menyelesaikannya?
2. Tentukan tiga tindakan remediasi otomatis yang "aman" untuk layanan kamu. Benarkan masing-masing.
3. Tulis templat runbook terstruktur: bagian, bidang wajib, prompt verifikasi.
4. Deteksi prediktif terjadi pada lead 12 menit. Apa kebijakan kamu — pager, pra-pengurasan, atau keduanya?
5. Berdebat apakah tim beranggotakan 3 orang harus mengadopsi AI SRE pada tahun 2026 atau menunggu. Pertimbangkan kematangan, volume, risiko.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| AI SRE | "agen untuk panggilan" | Investigasi insiden + koordinasi yang didukung LLM |
| Agen penyelia | "sang orkestra" | Agen tingkat atas memecah insiden menjadi sub-kueri |
| Agen khusus | "agen domain" | Sub-agen dengan akses alat (log, metrik, runbook) |
| Remediasi otomatis | "AI memperbaikinya" | Tindakan sempit yang telah disetujui sebelumnya; BUKAN arsitektur ulang yang luas |
| Memori operasional | "runbook vector" | Post-mortem + runbook dalam vector DB untuk RAG |
| Evaluasi permusuhan | "pemeriksaan dua model" | Analisis independen; persetujuan = keyakinan |
| NeuBird Hawkeye | "yang bermusuhan" | Produk dengan pola adversarial-eval + memori |
| Sedikit AI | "Agen SRE Datadog" | AI SRE yang dikelola Datadog |
| Prediksi pra-insiden | "deteksi dini" | 10-15 menit waktu tunggu untuk prediksi pemadaman |

## Bacaan Lanjutan- [incident.io — Panduan Lengkap AI SRE 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — AI yang Berpusat pada Manusia untuk SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI di SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)
