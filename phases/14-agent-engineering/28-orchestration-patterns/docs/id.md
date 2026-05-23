# Pola Orkestrasi: Supervisor, Swarm, Hierarchical

> Empat pola orkestrasi yang muncul pada framework tahun 2026: supervisor-pekerja, gerombolan/peer-to-peer, hierarki, debat. Panduan Anthropic: "Ini tentang membangun sistem yang tepat untuk kebutuhan kamu." Mulailah dengan sederhana; tambahkan topologi hanya ketika satu agen ditambah lima pola alur kerja tidak mencukupi.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 12 (Pola Alur Kerja), Fase 14 · 25 (Debat Multi-Agen)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Sebutkan empat pola orkestrasi berulang dan kapan masing-masing pola tersebut cocok.
- Jelaskan rekomendasi LangChain 2026: pengawasan berbasis panggilan alat vs perpustakaan pengawas.
- Jelaskan aturan "bangun sistem yang tepat" Anthropic dan cara menentukan pilihan topologi.
- Menerapkan keempatnya di stdlib terhadap LLM skrip umum.

## Masalah

Tim meraih "multi-agen" sebelum mereka membutuhkannya. Empat pola berulang di seluruh framework; setelah kamu dapat menamainya, kamu dapat memilih yang tepat — atau melewatkan topologi sepenuhnya.

## Konsep

### Supervisor-pekerja

- Pengiriman LLM perutean pusat ke agen spesialis.
- Memutuskan: mengulang kembali ke diri sendiri, menyerahkan ke spesialis, mengakhiri.
- Spesialis tidak berbicara satu sama lain; semua perutean melewati supervisor.

Framework: LangGraph `create_supervisor`, Pekerja orkestra antropik, Proses Hierarki CrewAI.

**Rekomendasi LangChain 2026:** lakukan pengawasan melalui panggilan alat langsung daripada `create_supervisor`. Memberikan kontrol rekayasa konteks yang lebih baik — kamu memutuskan dengan tepat apa yang dilihat setiap spesialis.

### Kawanan / rekan ke rekan

- Agen menyerahkan secara langsung melalui permukaan alat bersama.
- Tidak ada router pusat.
- Latensi lebih rendah dari supervisor (hop lebih sedikit).
- Lebih sulit untuk dipikirkan (tidak ada satu pun titik kendali).

Kerangka kerja: topologi gerombolan LangGraph, penyerahan SDK Agen OpenAI (ketika semua agen dapat menyerahkan ke agen lainnya).

### Hierarki

- Supervisor yang mengelola sub-supervisor yang mengelola pekerja.
- Diimplementasikan sebagai subgraf bersarang di LangGraph; kru bersarang di CrewAI.
- Menskalakan populasi agen yang besar dengan mengorbankan kompleksitas operasional.

Saat kamu membutuhkannya: ketika anggaran konteks pengawas tunggal tidak dapat menampung deskripsi semua spesialis.

### Debat

- Pengusul paralel + kritik silang berulang (Lesson 25).
- Bukan orkestrasi — lebih banyak verifikasi — tetapi muncul sebagai pilihan topologi dalam framework.

### Kru AI Kru vs Aliran

CrewAI meresmikan dua mode penerapan:

- **Aliran** untuk otomatisasi berbasis peristiwa deterministik (titik awal yang direkomendasikan untuk produksi).
- **Kru** untuk kolaborasi berbasis peran yang otonom.

Hal ini ortogonal terhadap empat pola di atas namun sesuai dengan topologi: Aliran biasanya bersifat supervisor atau hierarkis; Kru biasanya adalah supervisor dengan router LLM.

### Panduan Antropis

"Sukses di bidang LLM bukan tentang membangun sistem yang paling canggih. Ini tentang membangun sistem yang tepat untuk kebutuhan kamu."

Urutan keputusan:

1. Agen tunggal + pola alur kerja (Lesson 12) — mulai dari sini.
2. Supervisor-pekerja — bila kamu memiliki 2-4 spesialis.
3. Swarm - ketika latensi lebih penting daripada kejelasan alasan.
4. Hirarkis — hanya jika anggaran konteks pengawas gagal.
5. Perdebatan — ketika akurasi lebih penting daripada biaya.

### Dimana letak kesalahan pola ini- **Pemikiran yang mengutamakan topologi.** "Kita memerlukan multi-agen" sebelum mengidentifikasi masalah apa yang dapat dipecahkan oleh multi-agen.
- **Memantulkan handoff dalam gerombolan.** A -> B -> A -> B. Gunakan penghitung hop.
- **Hierarki palsu.** Tiga layer karena "perusahaan"; dua tim sebenarnya. Runtuh.

## Build

`code/main.py` mengimplementasikan keempat pola di stdlib terhadap LLM yang ditulis:

- `Supervisor` — router pusat.
- `Swarm` — peer-to-peer dengan serah terima langsung.
- `Hierarchical` — pengawas dari pengawas.
- `Debate` — pengusul + kritik paralel.

Setiap pola menangani tugas tiga maksud yang sama (pengembalian dana/bug/penjualan). Bentuk jejak berbeda-beda.

Jalankan:

```
python3 code/main.py
```

Output: pelacakan per pola + jumlah operasi. Supervisor paling bersih; gerombolan adalah yang terpendek; hierarkis adalah yang terdalam; perdebatan adalah hal yang paling mahal.

## Pakai

- **LangGraph** untuk supervisor dan hierarki (subgraf bersarang).
- **OpenAI Agents SDK** untuk handoff sebagai alat (berbentuk supervisor).
- **CrewAI Flow** untuk deterministik produksi.
- **Khusus** untuk debat atau saat kamu menginginkan kontrol yang tepat.

## Kirim

`outputs/skill-orchestration-picker.md` memilih topologi dan mengimplementasikannya.

## Latihan

1. Ubah supervisor-pekerja menjadi segerombolan dengan menghapus router. Apa yang rusak? Apa yang membaik?
2. Tambahkan penghitung hop ke gerombolan: tolak setelah 3 handoff. Apakah ia menangkap A->B->A yang memantul?
3. Membangun sistem hierarki dua tingkat untuk domain 12 spesialis. Di manakah anggaran konteks gagal tanpa disarangkan?
4. Profil keempat pola tersebut pada weight kerja berbentuk produksi. Manakah yang menang pada metrik mana (latensi, biaya, akurasi, kemampuan debug)?
5. Baca postingan Anthropic "Membangun Agen yang Efektif". Petakan setiap alur produksi kamu ke salah satu dari empat alur tersebut. Adakah yang tidak dipetakan dengan rapi?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Supervisor-pekerja | "Router + spesialis" | Pengiriman LLM pusat ke spesialis; mereka tidak berbicara satu sama lain |
| Kawanan | "Rekan ke rekan" | Penyerahan langsung melalui alat bersama; tidak ada router pusat |
| Hierarki | "Pengawas dari pengawas" | Subgraf bersarang untuk populasi besar |
| Debat | "Pengusul + kritik" | Pengusul paralel, kritik silang (Lesson 25) |
| Pengawasan berbasis panggilan alat | "Pengawas tanpa perpustakaan" | Menerapkan supervisor sebagai alat langsung memerlukan kontrol konteks |
| kru | "Tim otonom" | Mode kolaborasi berbasis peran CrewAI |
| Aliran | "Alur kerja deterministik" | Mode produksi berbasis peristiwa CrewAI |

## Bacaan Lanjutan

- [Antropik, Membangun Agen yang Efektif](https://www.anthropic.com/research/building- Effective-agents) — lima pola + agen vs alur kerja
- [Ikhtisar LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor, gerombolan, hierarki
- [Dokumen CrewAI](https://docs.crewai.com/en/introduction) — Kru vs Aliran
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — pola debat
