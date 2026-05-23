# CrewAI: Kru dan Alur Berbasis Peran

> CrewAI adalah framework multi-agen berbasis peran tahun 2026 — Agen, Tugas, Kru, Proses sebagai empat primitif. Panduan produksi dari dokumen: "untuk aplikasi siap produksi apa pun, mulailah dengan Aliran."

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 12 (Pola Alur Kerja), Fase 14 · 14 (Model Aktor)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Sebutkan empat primitif CrewAI — Agen, Tugas, Kru, Proses — dan peran masing-masing.
- Bedakan Crews (kolaborasi berbasis peran otonom) dari Flows (alur kerja deterministik berbasis peristiwa).
- Jelaskan mengapa dokumen merekomendasikan memulai dengan Aliran untuk produksi dan Kru untuk eksplorasi.
- Menerapkan pelari Kru stdlib ditambah pelari Aliran stdlib; tunjukkan saat masing-masing bersinar.

## Masalah

Tim yang mengadopsi framework multi-agen mengalami hal yang sama: "kolaborasi otonom" terdengar bagus, tetapi ketika pelanggan melaporkan bug, kamu memerlukan pengulangan deterministik. CrewAI membaginya secara eksplisit — Kru untuk kolaborasi kreatif, Alur untuk alur kerja berbentuk produksi yang digerakkan oleh peristiwa, dapat diaudit.

## Konsep

### Empat primitif

- **Agen.** Peran + sasaran + latar belakang + alat. Latar belakangnya memuat banyak hal - ia membentuk nada dan penilaian.
- **Tugas.** Deskripsi + output_yang diharapkan + agen yang ditugaskan. Unit kerja yang dapat digunakan kembali.
- **Kru.** Kontainer yang mengurutkan Agen dan Tugas. Memiliki Proses eksekusi.
- **Proses.** Berurutan atau Hierarki (dengan Agen manajer) atau Konsensual.

### Kru vs Arus

- **Kru.** Otonom, digerakkan oleh LLM. Cocok untuk tugas terbuka: penelitian, brainstorming, draf pertama. Kerangka kerja mengambil bentuknya saat runtime.
- **Aliran.** Grafik milik code yang digerakkan oleh peristiwa. Setiap langkah memicu pemicu (dekorator fungsi, pencocokan acara). Baik untuk produksi: dapat diamati, dapat diuji, deterministik.

Dokumen CrewAI 2026 mengatakan: memulai aplikasi produksi dengan Flows; memasukkan Kru sebagai sub-langkah ketika otonomi menghasilkan dampaknya.

### Sistem memori

CrewAI mengirimkan empat jenis memori secara langsung: jangka pendek (dalam proses), jangka panjang (di seluruh proses), entitas (fakta per entitas), kontekstual (perakitan waktu pengambilan). Integrasi dengan toko vector adalah pihak pertama.

### Integrasi AWS Batuan Dasar

CrewAI telah mendokumentasikan integrasi AWS Bedrock dengan kait observasi CloudWatch, AgentOps, dan Langfuse. Dokumen AWS menyebutkan kecepatan 5,76x vs LangGraph pada tugas QA dalam tolok ukurnya — menganggap angka spesifik framework sebagai angka terarah, bukan absolut.

### Bentuk ketergantungan

Independen dari LangChain. Python 3.10–3.13. Menggunakan `uv` untuk manajemen ketergantungan. 30rb+ bintang GitHub pada awal tahun 2026.

### Dimana letak kesalahan pola ini

- **Crew-as-prod.** Menggunakan Crew bentuk bebas dalam prod tanpa pembungkus Flow. Variabilitas output tinggi; debugging itu menyakitkan.
- **Backstory mengasapi.** Backstories sepanjang 2.000 kata mendorong anggaran kontekstual. Jaga agar tetap rapat.
- **Proses perplexity.** Proses hierarki menambahkan Agen manajer yang merutekan; gunakan hanya jika kamu memiliki 4+ spesialis.

## Build

`code/main.py` mengimplementasikan versi stdlib dari keduanya:

- `Agent`, `Task`, `Crew`, `SequentialCrew` (satu tugas dalam satu waktu), `HierarchicalCrew` (rute manajer).
- `Flow` dengan `@start()` dan dekorator `@listen()` (pengganti fungsi biasa) yang aktif pada acara yang disebutkan.
- Tugas tiga langkah yang sama (penelitian, kerangka, draf) diterapkan dua arah.

Jalankan:

```
python3 code/main.py
```Jejak kru berubah-ubah dan bervariasi; Jejak aliran tetap dan dapat diamati. Itu adalah pilihannya.

## Pakai

- **Aliran CrewAI** untuk produksi.
- **Kru CrewAI** untuk eksplorasi, pemasangan, draf pertama.
- **LangGraph** (Lesson 13) jika kamu menginginkan mesin status yang lebih eksplisit.
- **AutoGen v0.4** (Lesson 14) jika kamu menginginkan konkurensi model aktor.

## Kirim

`outputs/skill-crew-or-flow.md` memilih Crew vs Flow untuk suatu tugas dan merancang implementasi minimal.

## Latihan

1. Ubah demo berbasis Kru menjadi Aliran. Hitung titik kontak di mana variabilitas menurun.
2. Tambahkan memori entitas ke Kru: fakta tentang pelanggan tetap ada di seluruh tugas.
3. Menerapkan proses Hierarki: Agen manajer memilih spesialis mana yang dijalankan berikutnya berdasarkan output sebelumnya.
4. Baca intro dokumen CrewAI. Pindahkan mainan kamu ke `crewai` API yang asli. Perubahan apa pada kemampuan pengujian?
5. Hubungkan AgentOps atau Langfuse ke salah satu proses kamu. Jejak mana yang kamu lewatkan di versi stdlib?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Agen | "Persona" | Peran + tujuan + latar belakang + alat |
| Tugas | "Satuan Kerja" | Deskripsi + output yang diharapkan + penerima tugas |
| kru | "Tim Agen" | Wadah untuk Agen + Tugas + Proses |
| Proses | "Strategi Eksekusi" | Berurutan / Hirarkis / Konsensual |
| Aliran | "Alur kerja deterministik" | Didorong oleh peristiwa, dimiliki code, dapat diuji |
| Latar Belakang | "Permintaan persona" | Pembentuk nada dan penilaian untuk Agen |
| Memori entitas | "Fakta per-entitas" | Memori dicakup ke pelanggan/akun/masalah |

## Bacaan Lanjutan

- [Pengenalan dokumen CrewAI](https://docs.crewai.com/en/introduction) — konsep dan jalur produksi yang direkomendasikan
- [Antropik, Membangun Agen yang Efektif](https://www.anthropic.com/research/building- Effective-agents) — ketika multi-agen membantu dan ketika tidak
- [Ikhtisar LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — alternatif mesin negara
