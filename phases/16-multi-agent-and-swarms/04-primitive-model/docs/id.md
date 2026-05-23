# Model Primitif Multi-Agen

> Setiap framework multi-agen yang dikirimkan pada tahun 2026 — AutoGen, LangGraph, CrewAI, OpenAI Agents SDK, Microsoft Agent Framework — merupakan sebuah titik dalam ruang desain empat dimension. Empat primitif, tidak lebih: agen, serah terima, negara bersama, orkestrator. Lesson ini membangunnya dari nol, menjalankan sistem mainan pada keempatnya, lalu memetakan setiap kerangka utama ke sumbu yang sama sehingga kamu dapat membaca rilis baru dalam satu paragraf.

**Type:** Learn
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 (Rekayasa Agen), Fase 16 · 01 (Mengapa Multi-Agen)
**Waktu:** ~60 menit

## Masalah

Setiap enam bulan, framework multi-agen baru dikirimkan. AutoGen pada tahun 2023. CrewAI pada tahun 2024. LangGraph dan OpenAI Swarm pada tahun 2024. Google ADK pada bulan April 2025. Microsoft Agent Framework RC pada bulan Februari 2026. Setiap siaran pers mengklaim sebagai "abstraksi yang tepat".

Jika kamu mencoba mempelajarinya satu per satu, kamu akan kehabisan tenaga. API terlihat berbeda. Dokumen tidak setuju tentang apa itu "agen". Satu kerangka menyebut memori bersama sebagai "papan tulis", kerangka lain menyebutnya "kumpulan pesan", kerangka ketiga menyebutnya "StateGraph". kamu mulai curiga bahwa lapangan sedang berputar-putar.

Tidak. Di bawah pemasaran, keempat primitif itu stabil. Learn sekali, baca setiap kerangka baru dalam satu paragraf.

## Konsep

### Empat primitif

1. **Agen** — system prompt dan daftar alat. Tanpa kewarganegaraan; setiap proses dimulai dari prompt sistemnya dan riwayat pesan saat ini.
2. **Handoff** — transfer kendali terstruktur dari satu agen ke agen lainnya. Secara mekanis, pemanggilan alat yang mengembalikan agen baru atau tepi grafik yang mengikuti suatu kondisi.
3. **Keadaan bersama** — struktur data apa pun yang dapat dibaca oleh lebih dari satu agen (terkadang ditulis). Kumpulan pesan, papan tulis, penyimpanan nilai kunci, memori vector.
4. **Orchestrator** — siapa pun yang memutuskan siapa yang akan berbicara selanjutnya. Pilihan: grafik eksplisit (deterministik), pemilih pembicara LLM (lunak), panggilan handoff pembicara terakhir (OpenAI Swarm), atau penjadwal melalui antrian (arsitektur gerombolan).

Itulah keseluruhan ruang desain. Setiap framework memilih default untuk setiap sumbu; sisanya adalah sintaksis permukaan.

### Bagaimana setiap framework tahun 2026 memetakannya

| Kerangka | Agen | Penyerahan | Status bersama | Orkestrator |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agen SDK | `Agent(instructions, tools)` | alat mengembalikan Agen | masalah penelepon | panggilan serah terima LLM berikutnya |
| AutoGen v0.4 / AG2 | `ConversableAgent` | pemilih pembicara di GroupChat | kumpulan pesan | fungsi pemilih (LLM atau round-robin) |
| kruAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Output tugas dirantai | manajer LLM atau pesanan statis |
| Grafik Lang | fungsi simpul | tepi grafik + kondisi | `StateGraph` peredam | grafik, deterministik |
| Kerangka Agen Microsoft | agen + pola orkestrasi | pola khusus | utas / konteks | pola khusus |
| Google ADK | agen + kartu A2A | Tugas A2A | Artefak A2A | tuan rumah memutuskan |

Perbedaan permukaan terlihat sangat besar. Di bawahnya: empat kenop yang sama.

### Mengapa ini penting

Setelah kamu melihat primitifnya, perbandingan framework menjadi daftar periksa singkat:

- Apakah orkestrator memercayai LLM untuk merutekan (Swarm) atau apakah ia embed perutean dalam code (LangGraph)?
- Apakah riwayat lengkap status bersama (GroupChat) atau diproyeksikan (peredam StateGraph)?
- Bisakah agen saling memodifikasi permintaan (manajer CrewAI) atau hanya menyerahkan (Swarm)?Ketiga pertanyaan tersebut menjawab 80% framework mana yang sesuai dengan masalah tertentu. kamu berhenti berbelanja untuk "framework multi-agen terbaik" dan mulai merancang sumbu yang benar-benar kamu pedulikan.

### Wawasan tanpa kewarganegaraan

Setiap negara primitif, kecuali negara bagian bersama, tidak memiliki kewarganegaraan. Agen adalah fungsi dari (prompt, alat). Handoff adalah pemanggilan fungsi. Orchestrator adalah penjadwal. **Satu-satunya hal yang bersifat stateful dalam sistem adalah status bersama.** Di sinilah letak semua bug yang menarik: keracunan memori (Lesson 15), pengurutan pesan, pembuatan versi, pertikaian tulis.

Kerangka kerja yang menyembunyikan status bersama (Swarm) mendorong masalah ke pemanggil. Kerangka kerja yang memusatkannya (pos pemeriksaan LangGraph, kumpulan AutoGen) membuatnya dapat diperiksa tetapi mengalihkan biaya koordinasi ke implementasi negara bersama.

### Anatomi primitif tunggal

#### Agen

```
Agent = (system_prompt, tools, model, optional_name)
```

Tidak ada memori. Tidak ada negara bagian. Dua agen dengan prompt sistem dan alat yang sama dapat dipertukarkan. Segala sesuatu yang tampak seperti status per agen sebenarnya berada dalam status bersama atau protokol handoff.

#### Penyerahan

```
Handoff = (from_agent, to_agent, reason, payload)
```

Tiga implementasi mendominasi:

- **Pengembalian fungsi** — alat mengembalikan agen berikutnya. Ini adalah pola OpenAI Swarm. Agen membawa perutean dalam skema alat mereka.
- **Tepi grafik** — LangGraph. Tepinya bersifat deklaratif. LLM menghasilkan nilai; suatu kondisi memilih node berikutnya.
- **Pemilihan pembicara** — Obrolan Grup AutoGen. Fungsi pemilih (terkadang merupakan panggilan LLM) membaca kumpulan dan memilih siapa yang berbicara selanjutnya.

#### Status bersama

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

Minimal, daftar pesan. Seringkali lebih banyak: artefak terstruktur (output Tugas CrewAI), konteks yang diketik (pengurang LangGraph), memori eksternal (MCP, DB vector).

Dua topologi: **kumpulan penuh** (setiap agen melihat setiap pesan) dan **diproyeksikan** (agen melihat tampilan cakupan peran). Kolam penuh sederhana dan skalanya buruk. Kumpulan yang diproyeksikan berskala tetapi memerlukan desain skema dimuka.

#### Orkestrator

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

Empat rasa:

- **Statis** — grafik ditetapkan pada waktu pembuatan (deterministik LangGraph, Urutan CrewAI).
- **LLM-selected** — LLM membaca kumpulan dan memilih pembicara berikutnya (AutoGen, CrewAI Hierarchical).
- **Handoff-driven** — agen saat ini memutuskan dengan memanggil alat handoff (Swarm).
- **Antrian** — pekerja keluar dari antrean bersama; tidak ada pembicara berikutnya yang eksplisit (arsitektur gerombolan, Matrix).

### Apa yang berubah antar framework

Setelah primitif diperbaiki, keputusan desain yang tersisa adalah:

- **Strategi memori** — pos pemeriksaan singkat vs tahan lama (pos pemeriksaan LangGraph).
- **Batas keamanan** — siapa yang dapat menyetujui serah terima (human-in-the-loop).
- **Akuntansi biaya** — anggaran token per agen.
- **Observability** — menelusuri handoff, status bertahan untuk diputar ulang.

Semua dapat diimplementasikan di atas yang primitif. Tak satu pun dari mereka yang merupakan primitif baru.

## Build

`code/main.py` mengimplementasikan empat primitif di ~150 baris stdlib Python. Tidak ada LLM yang sebenarnya — setiap agen memiliki kebijakan yang tertulis sehingga fokusnya tetap pada struktur koordinasi.

File mengekspor:

- `Agent` — kelas data nama, system prompt, alat, fungsi kebijakan.
- `Handoff` — fungsi yang mengembalikan agen baru.
- `SharedState` — kumpulan pesan yang aman untuk thread.
- `Orchestrator` — tiga varian: `StaticOrchestrator`, `HandoffOrchestrator`, `LLMSelectorOrchestrator` (simulasi).Demo ini menjalankan alur tiga agen yang sama (penelitian → tulis → tinjauan) melalui ketiga jenis orkestrator dan mencetak kumpulan pesan di bagian akhir. kamu dapat melihat bahwa keluarannya hanya berbeda pada *siapa yang memilih berikutnya*; agen dan status bersama identik di seluruh proses.

Jalankan:

```
python3 code/main.py
```

Hasil yang diharapkan: tiga orkestrator berjalan, satu per pola. Masing-masing mencetak kumpulan pesan terakhir. Proses yang digerakkan oleh handoff menjangkau lebih sedikit agen jika peneliti memutuskan hal itu dilakukan lebih awal - itulah miniatur tradeoff perutean LLM.

## Pakai

`outputs/skill-primitive-mapper.md` adalah keterampilan yang membaca basis code multi-agen atau dokumen framework dan mengembalikan pemetaan empat primitif. Jalankan pada rilis framework baru untuk mendapatkan pemahaman satu paragraf sebelum membaca dokumen secara mendalam.

## Kirim

Sebelum mengadopsi framework baru, tuliskan pemetaan primitifnya. Jika kamu tidak bisa, dokumennya tidak lengkap atau framework menciptakan primitif kelima (jarang terjadi — periksa varian negara bagian yang belum pernah kamu lihat).

Sematkan pemetaan di dokumen arsitektur kamu. Saat anggota tim baru bergabung, kirimkan pemetaannya sebelum dokumen API. Ketika versi framework berubah, bedakan pemetaannya, bukan log perubahannya.

## Latihan

1. Jalankan `code/main.py` tiga kali dengan kebijakan agen yang berbeda. Amati bagaimana pilihan orkestrator mengubah agen mana yang dijalankan.
2. Menerapkan tipe orkestrator keempat: tipe orkestrator berbasis antrean di mana agen melakukan jajak pendapat mengenai status pekerjaan yang dibagikan. Kebuntuan apa yang bisa terjadi, dan bagaimana cara mendeteksinya?
3. Ambil panduan mulai cepat LangGraph (https://docs.langchain.com/oss/python/langgraph/workflows-agents) dan tulis ulang sebagai empat primitif. Abstraksi LangGraph manakah yang dipetakan 1:1 dan manakah yang merupakan pembungkus praktis?
4. Baca buku masak OpenAI Swarm (https://developers.openai.com/cookbook/examples/orchestrating_agents). Identifikasi yang mana dari empat primitif Swarm yang paling ergonomis, dan mana yang diberikan kepada penelepon.
5. Temukan satu framework dalam tabel ini yang menyembunyikan status bersama seluruhnya. Jelaskan apa yang salah ketika agen perlu berkoordinasi antar serah terima tanpa membaca ulang riwayat.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Agen | "LLM dengan alat" | Tiga `(system_prompt, tools, model)`. Tanpa kewarganegaraan. |
| Penyerahan | "Pengalihan kendali" | Panggilan terstruktur yang memberi nama agen berikutnya dan muatan opsional. Tiga implementasi: pengembalian fungsi, tepi grafik, pemilihan speaker. |
| Status bersama | "Memori" / "konteks" | Satu-satunya bagian stateful dari sistem multi-agen. Kumpulan pesan atau papan tulis. |
| Orkestrator | "Koordinator" | Siapapun yang memutuskan siapa yang mencalonkan diri selanjutnya. Grafik statis, pemilih LLM, digerakkan oleh handoff, atau digerakkan oleh antrian. |
| Primitif | "Abstraksi" | Salah satu dari empat sumbu yang diparameterisasi oleh setiap framework. Bukan feature framework. |
| Kumpulan pesan | "Riwayat obrolan bersama" | Status bersama dengan riwayat lengkap. Mudah untuk dipikirkan, skalanya buruk. |
| Keadaan yang diproyeksikan | "Tampilan tercakup" | Tampilan khusus peran ke dalam status bersama. Berskala, memerlukan desain skema. |
| Pemilihan pembicara | "Siapa yang berbicara selanjutnya" | Pola orkestrator di mana suatu fungsi (seringkali LLM) memilih agen berikutnya dari grup. |

## Bacaan Lanjutan- [Buku masak OpenAI: Agen Orkestrating — Rutinitas dan Handoff](https://developers.openai.com/cookbook/examples/orchestrating_agents) — artikulasi paling jelas dari orkestrasi berbasis handoff
- [Dokumen stabil AutoGen](https://microsoft.github.io/autogen/stable/) — Pilihan pembicara GroupChat + adalah referensi untuk orkestrasi pilihan LLM
- [Alur kerja dan agen LangGraph](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — orkestrasi tepi grafik dan status bersama berbasis peredam
- [Pengenalan CrewAI](https://docs.crewai.com/en/introduction) — agen cerita latar belakang tujuan peran, proses berurutan/hierarki
- [AG2 (kelanjutan komunitas AutoGen)](https://github.com/ag2ai/ag2) — jalur AutoGen v0.2 langsung setelah Microsoft memindahkan v0.4 ke dalam pemeliharaan
