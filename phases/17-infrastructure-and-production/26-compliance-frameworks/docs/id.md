# Kepatuhan — SOC 2, HIPAA, GDPR, PCI-DSS, EU AI Act, ISO 42001

> Cakupan multi-kerangka adalah taruhan utama untuk kesepakatan perusahaan pada tahun 2026. **UU AI UE**: berlaku sejak 1 Agustus 2024. Sebagian besar persyaratan berisiko tinggi diberlakukan mulai 2 Agustus 2026. Denda hingga €15 juta atau 3% omzet tahunan global untuk kewajiban sistem berisiko tinggi (Pasal 99(4)); hingga €35 juta atau 7% untuk praktik AI yang dilarang (Pasal 99(3)). Berlaku secara global jika melayani pengguna UE. **Colorado AI Act**: berlaku efektif 30 Juni 2026 (ditunda mulai Februari 2026 karena SB25B-004) — penilaian dampak untuk sistem berisiko tinggi, hak untuk mengajukan banding atas keputusan AI. Virginia serupa untuk kredit/pekerjaan/perumahan/pendidikan. **SOC 2 Tipe II**: persyaratan AI B2B de facto (Tipe II, bukan Tipe I, untuk fintech). **GDPR**: denda khusus AI terbesar yang terdokumentasi adalah €30,5 juta terhadap Clearview AI (DPA Belanda, September 2024); Garante Italia mengeluarkan €15 juta untuk melawan OpenAI pada Desember 2024 (kemudian dibatalkan saat naik banding pada Maret 2026). Redaksi PII real-time pada inference adalah standar yang dapat dipertahankan; pembersihan pasca-pemrosesan saja tidak cukup. **HIPAA**: terikat layanan kesehatan — tidak dapat mengirim PHI ke layanan AI eksternal tanpa BAA. **PCI-DSS**: Cakupan layer interaksi AI memerlukan konfigurasi + perjanjian kontrak, tidak otomatis. **ISO 42001**: standar tata kelola AI yang baru, persyaratan pengadaan yang semakin meningkat seiring dengan ISO 27001. Profil referensi: OpenAI mempertahankan SOC 2 Tipe 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, PCI-DSS untuk komponen pembayaran ChatGPT. Pemetaan lintas kerangka mengurangi kelelahan audit: kontrol akses dipetakan di seluruh ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a).

**Type:** Learn
**Language:** (Python opsional — kepatuhan adalah kebijakan + proses, bukan code)
**Prerequisites:** Fase 17 · 25 (Keamanan), Fase 17 · 13 (Kemampuan Observabilitas)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Sebutkan tujuh framework tahun 2026 yang relevan dengan produk LLM dan cocokkan masing-masing dengan segmen pelanggan.
- Sebutkan garis waktu penegakan UU AI UE (berlaku pada Agustus 2024; penegakan risiko tinggi pada Agustus 2026) dan batas denda dua tingkat (€15 juta/3% untuk kewajiban berisiko tinggi, €35 juta/7% untuk praktik terlarang).
- Jelaskan mengapa pembersihan PII pascapemrosesan tidak cukup untuk GDPR dan sebutkan redaksi layer inference real-time sebagai standar yang dapat dipertahankan.
- Menjelaskan pemetaan kontrol lintas kerangka (misalnya, peta kontrol akses ke ISO 27001 A.5.15-5.18 + GDPR Art. 32 + HIPAA §164.312(a)).

## Masalah

Pengadaan pelanggan perusahaan meminta SOC 2 Tipe II, GDPR, HIPAA BAA, ISO 27001, dan "Pernyataan kepatuhan EU AI Act". Tim kamu memiliki SOC 2 Tipe I. kamu enam bulan dari Tipe II dan belum memulai pencatatan Pasal 30 GDPR.

Cakupan multi-framework bukanlah masalah LLM — ini adalah masalah SaaS perusahaan, dengan overlay khusus LLM. Tim pengadaan pada tahun 2026 menginginkan matrix dengan baris per kerangka dan kolom per kontrol, bukan PDF.

## Konsep

### Tujuh framework| Kerangka | Ruang Lingkup | Persyaratan khusus LLM |
|-----------|-------|-----------|
| SOC 2 Tipe II | Dasar SaaS B2B | Kontrol proses diaudit selama 6-12 bulan |
| HIPAA | Kesehatan AS | BAA diperlukan; PHI tidak bisa meninggalkan infrastruktur tanpa perjanjian yang ditandatangani |
| GDPR | Pengguna UE | Redaksi PII waktu nyata; hak subjek data; Pasal 30 catatan |
| PCI-DSS | Data pembayaran | Konfigurasi + kontrak untuk pembayaran yang menyentuh AI |
| Undang-Undang AI UE | Melayani pengguna UE | Klasifikasi tingkat risiko; sistem berisiko tinggi: penilaian kesesuaian, dokumentasi, pencatatan |
| Undang-Undang AI Colorado | Melayani warga CO | Penilaian dampak; hak untuk mengajukan banding |
| ISO 42001 | Tata Kelola AI | Muncul; berpasangan dengan ISO 27001 |

### Garis waktu UU AI UE

- 1 Agustus 2024 : berlaku.
- 2 Februari 2025: praktik AI yang dilarang diberlakukan.
- 2 Agustus 2026: sistem berisiko tinggi diberlakukan (penilaian kesesuaian, dokumentasi, logging).
- Agustus 2027: sistem berisiko tinggi pada produk berdasarkan undang-undang yang diselaraskan.

Tingkat risiko: Tidak dapat diterima (dilarang), Risiko tinggi (kesesuaian + pencatatan), Risiko terbatas (transparansi), Risiko minimal (tidak ada batasan). Sebagian besar SaaS LLM B2B memiliki risiko terbatas; risiko tinggi terjadi pada lapangan kerja, kredit, pendidikan, penegakan hukum, migrasi, dan layanan penting.

Denda (Pasal 99): hingga €15 juta atau 3% omzet tahunan global karena pelanggaran kewajiban sistem berisiko tinggi (Pasal 99(4)); hingga €35 juta atau 7% untuk praktik AI yang dilarang (Pasal 99(3)); mana yang lebih tinggi yang berlaku.

### GDPR — redaksi real-time adalah standarnya

Pembersihan pasca-pemrosesan (redaksi PII setelah LLM melihatnya) bukanlah sikap yang dapat dipertahankan — model sudah melihat datanya. Redaksi layer inference waktu nyata adalah standar tahun 2026:

- Pengenalan entitas sebelum panggilan LLM.
- Tokenization yang konsisten (pendekatan Mesh) menjaga semantik.
- Simpan hanya prompt yang telah disunting + keikutsertaan mentah yang disetujui.

Penegakan terbaru: €30,5 juta terhadap Clearview AI (DPA Belanda, September 2024) merupakan denda GDPR khusus AI terbesar yang terdokumentasi hingga saat ini; €15 juta terhadap OpenAI (Garante Italia, Des 2024) adalah denda terbesar khusus LLM, meskipun denda tersebut dibatalkan saat naik banding pada Maret 2026 dan keputusan tersebut masih dalam peninjauan lebih lanjut. Klaim pasca-pemrosesan gagal diaudit.

### HIPAA — BAA bukan opsional

kamu tidak dapat mengirim PHI ke layanan AI eksternal tanpa Perjanjian Rekan Bisnis yang ditandatangani. Ketiga platform LLM hyperscaler (Bedrock, Azure OpenAI, Vertex) menawarkan BAA. API langsung OpenAI menawarkan BAA. API langsung antropik menawarkan BAA. Konfirmasi sebelum mengirim PHI.

### SOC 2 Tipe II

Tipe I: kontrol dirancang dan didokumentasikan.
Tipe II: pengendalian beroperasi secara efektif selama 6-12 bulan.

Pengadaan B2B pada tahun 2026 defaultnya adalah Tipe II. Tipe I adalah starter; Tipe II adalah gerbang.

Penggerak audit yang umum: log akses (siapa yang melihat apa), manajemen perubahan (bagaimana penerapannya), penilaian risiko (setiap triwulan), respons insiden (diuji?). Log audit dari Fase 17 · 25 dapat digunakan kembali secara langsung.

### Pemetaan lintas kerangka

Satu kebijakan kontrol akses memenuhi beberapa kontrol framework:

| Kontrol | Kerangka |
|---------|-----------|
| Akses pencatatan | ISO 27001 A.5.15-5.18, GDPR Seni. 32, HIPAA §164.312(a) |
| Manajemen perubahan | ISO 27001 A.8.32, Persyaratan PCI DSS. 6, cakupan pemberitahuan pelanggaran HIPAA |
| Enkripsi dalam perjalanan | ISO 27001 A.8.24, GDPR Seni. 32, HIPAA §164.312(e) |
| Manajemen rahasia | ISO 27001 A.8.19, Persyaratan PCI DSS. 8, SOC 2 CC6.1 |

Alat kepatuhan (Drata, Vanta, Secureframe) mengotomatiskan pemetaan ini. Sepadan dengan biaya dalam skala besar.

### ISO 42001 — baru munculDiterbitkan akhir tahun 2023. Persyaratan pengadaan yang semakin meningkat seiring dengan ISO 27001. Kerangka tata kelola AI termasuk manajemen risiko, kualitas data, transparansi, dan pengawasan manusia.

### Profil referensi OpenAI

OpenAI mempertahankan SOC 2 Tipe 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, PCI-DSS untuk komponen pembayaran ChatGPT. Itu kira-kira adalah taruhan meja perusahaan pada tahun 2026.

### Nomor yang harus kamu ingat

- Denda EU AI Act: hingga €15 juta / 3% (kewajiban berisiko tinggi, Pasal 99(4)); hingga €35 juta / 7% (praktik yang dilarang, Pasal 99(3)).
- Penegakan UU AI UE yang berisiko tinggi: 2 Agustus 2026.
- Denda GDPR khusus AI terbesar yang terdokumentasi: €30,5 juta, Clearview AI (DPA Belanda, September 2024).
- Denda GDPR khusus LLM terbesar: €15 juta, OpenAI (Garante Italia, Des 2024; dibatalkan pada banding Maret 2026).
- Jendela SOC 2 Tipe II: 6-12 bulan kontrol yang dioperasikan.
- Tanggal efektif Colorado AI Act: 30 Juni 2026 (ditunda dari Februari 2026 oleh SB25B-004).

## Pakai

`code/main.py` adalah spreadsheet pemetaan kepatuhan dengan Python — jika diberi kontrol, mencantumkan framework yang dipenuhinya.

## Kirim

Lesson ini menghasilkan `outputs/skill-compliance-matrix.md`. Mengingat segmen pelanggan dan geografi, tentukan framework dan kontrol yang diperlukan.

## Latihan

1. Pelanggan perusahaan pertama kamu memerlukan pernyataan SOC 2 Tipe II, HIPAA BAA, EU AI Act. Apa postur kepatuhan minimum yang layak untuk memenangkan kesepakatan?
2. Klasifikasikan tiga produk LLM hipotetis berdasarkan tingkat risiko EU AI Act. Perubahan apa saja yang berisiko tinggi?
3. kamu tidak sengaja mengirimkan PHI ke penyedia tanpa BAA. Telusuri respons insiden.
4. Berdebat apakah ISO 42001 "diperlukan pada tahun 2026" untuk vendor AI pasar menengah.
5. Petakan bidang log audit LLM kamu (Fase 17 · 25) ke setidaknya tiga kontrol framework.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| SOC 2 Tipe II | "kontrol yang diaudit" | Kontrol beroperasi selama 6-12 bulan, dibuktikan secara independen |
| HIPAA BAA | "kontrak layanan kesehatan" | Perjanjian Rekan Bisnis; diperlukan untuk PHI |
| GDPR | "Privasi UE" | Redaksi PII waktu nyata adalah standar tahun 2026 yang dapat dipertahankan |
| Undang-Undang AI UE | "Aturan AI UE" | Penegakan hukum berisiko tinggi Agustus 2026; €15 juta / 3% (kewajiban berisiko tinggi) — €35 juta / 7% (praktik terlarang) |
| Undang-Undang AI Colorado | "Hukum negara bagian AI AS" | 30 Juni 2026 berlaku efektif (tertunda SB25B-004); penilaian dampak |
| ISO 42001 | "tata kelola AI" | Kerangka kerja yang muncul untuk risiko AI + transparansi |
| ISO 27001 | "ISMS keamanan" | Dasar Sistem Manajemen Keamanan Informasi |
| Penilaian kesesuaian | "Paket dokumen AI UE" | Persyaratan berisiko tinggi: dokumen, pengujian, logging |
| Pemetaan lintas kerangka | "satu kontrol, banyak frame" | Kebijakan tunggal memenuhi beberapa kerangka kontrol |

## Bacaan Lanjutan

- [Keamanan dan Privasi OpenAI](https://openai.com/security-and-privacy/) — referensi profil kepatuhan.
- [GuardionAI — Kepatuhan LLM 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — Panduan Audit SOC 2 Tipe 2 2026: 10 Kontrol AI](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-power-controls-every-saas-team-needs)
- [Teks resmi EU AI Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — sumber utama.
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — sumber utama.
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — Standar sistem manajemen AI.
