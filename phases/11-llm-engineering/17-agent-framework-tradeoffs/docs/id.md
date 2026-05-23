# Pengorbanan Kerangka Agen — LangGraph vs CrewAI vs AutoGen vs Agno

> Setiap framework menjual demo yang sama (agen peneliti membuat laporan) dan menyembunyikan bug yang sama (skema negara berkelahi dengan layer orkestrasi). Pilih framework yang abstraksinya sesuai dengan bentuk masalah kamu; yang lainnya adalah lem yang kamu tulis dua kali.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 11 · 09 (Pemanggilan Fungsi), Fase 11 · 16 (LangGraph)
**Waktu:** ~45 menit

## Masalah

kamu memiliki tugas yang memerlukan lebih dari satu panggilan LLM. Mungkin alur kerja penelitian (rencanakan, cari, rangkum, kutip). Mungkin itu adalah pipa peninjauan code (parse diff, kritik, patch, validasi). Mungkin asisten multi-turn yang memesan penerbangan, menulis email, dan mengarsipkan laporan pengeluaran. kamu memilih framework.

Tiga hari kemudian, kamu menemukan kebocoran abstraksi framework. CrewAI memberi kamu peran tetapi melawan kamu ketika "peneliti" perlu menyerahkan rencana terstruktur kepada "penulis". AutoGen memberi kamu obrolan antar agen tetapi tidak memiliki status kelas satu sehingga pos pemeriksaan kamu hanyalah kumpulan log percakapan. LangGraph memberi kamu grafik keadaan tetapi memaksa kamu memberi nama setiap transisi sebelum kamu mengetahui apa yang akan dilakukan agen. Agno memberi kamu agen tunggal primitif yang berteriak ketika kamu mencoba menyebar ke tiga pekerja secara bersamaan.

Cara mengatasinya bukanlah "memilih framework terbaik". Hal ini untuk mencocokkan abstraksi inti framework dengan bentuk masalah kamu. Lesson ini menggambarkan peta itu.

## Konsep

![Matrix kerangka agen: abstraksi inti vs bentuk masalah](../assets/framework-matrix.svg)

Empat framework mendominasi lanskap tahun 2026. Abstraksi intinya tidak sama.

| Kerangka | Abstraksi inti | Paling cocok | Paling cocok |
|-----------|------------------|----------|-----------|
| **Grafik Lang** | `StateGraph` — status yang diketik, node, tepi bersyarat, checkpointer. | Alur kerja dengan status eksplisit dan interupsi human-in-the-loop; agen produksi memerlukan debugging perjalanan waktu. | Brainstorming longgar dan berbasis peran yang topologinya tidak diketahui. |
| **KruAI** | `Crew` — peran (tujuan, latar belakang), tugas, proses (berurutan atau hierarki). | Alur kerja permainan peran atau berbasis persona dengan rencana linier/hierarki pendek. | Apa pun yang bersifat stateful di luar riwayat giliran kru; percabangan yang rumit. |
| **Gen Otomatis** | `ConversableAgent` pair — dua atau lebih agen yang berbicara secara bergantian hingga kondisi keluar. | *Dialog* multi-agen (guru-siswa, pengusul-kritikus, aktor-resensi) di mana pemikiran muncul dari obrolan. | Alur kerja deterministik dengan DAG yang diketahui; apa pun yang memerlukan status tahan lama saat dimulai ulang. |
| **Agno** | `Agent` — satu LLM + alat + memori, dapat disusun menjadi beberapa tim. | Agen tunggal yang cepat dibangun dan tim yang ringan; multi-modalitas yang kuat dan driver penyimpanan internal. | Grafik yang dalam dan bercabang secara eksplisit dengan reduksi khusus. |

### Apa sebenarnya arti "abstraksi".

Abstraksi inti framework adalah hal yang kamu gambar di papan tulis saat kamu menampilkan arsitektur.- **LangGraph** → kamu menggambar grafik. Node adalah langkah, tepian adalah transisi, dan objek keadaan di setiap titik diketik. Model mental adalah mesin negara.
- **CrewAI** → kamu menggambar bagan organisasi. Setiap peran memiliki deskripsi pekerjaan dan manajer mengarahkan tugas. Model mental adalah sebuah tim kecil yang terdiri dari para spesialis.
- **AutoGen** → kamu menggambar Slack DM. Dua agen saling mengirim pesan; sepertiga bergabung jika kamu membutuhkan moderator. Model mentalnya adalah obrolan.
- **Agno** → kamu menggambar sebuah kotak dengan peralatan yang tergantung di situ. Letakkan kotak di samping satu sama lain untuk sebuah tim. Model mentalnya adalah "agen yang dilengkapi baterai".

### Pertanyaan negara

Negara bagian adalah tempat sebagian besar pilihan framework gagal dalam produksi.

- **LangGraph.** Status yang diketik (`TypedDict` atau model Pydantic), reduksi per bidang, checkpointer kelas satu (SQLite/Postgres/Redis). Melanjutkan, menyela, dan melakukan perjalanan waktu gratis. *(Lihat Fase 11 · 16.)*
- **CrewAI.** Status mengalir sebagai string antar tugas melalui bidang `context`, atau disusun melalui `output_pydantic`. Tidak ada penyimpanan per kru yang tahan lama; kamu lari sendiri jika kru harus selamat dari restart.
- **AutoGen.** Status adalah riwayat obrolan dan `context` yang ditentukan pengguna. Transkrip percakapan tetap ada; keadaan alur kerja sewenang-wenang tidak terjadi kecuali kamu menulis adaptor.
- **Agno.** Driver penyimpanan internal (SQLite, Postgres, Mongo, Redis, DynamoDB) yang terpasang pada `Agent` melalui `storage=` — sesi percakapan dan memori pengguna tetap ada secara otomatis. Bukan checkpointer grafik lengkap; toko sesi.

### Pertanyaan bercabang

Setiap cabang agen non-sepele. Siapa yang memutuskan cabang itu penting.

- **LangGraph** — kamu yang memutuskan, melalui tepi bersyarat. Perutean adalah fungsi Python dengan cabang bernama. Cabang-cabangnya adalah kelas satu dalam grafik yang dikompilasi; pos pemeriksaan mencatat cabang mana yang diambil.
- **CrewAI** — manajer memutuskan dalam mode hierarki; dalam mode berurutan kamu memutuskan pada waktu pembuatan. Perutean tersirat dalam daftar tugas; tidak ada kata "jika" kelas satu di luar prompt manajer.
- **AutoGen** — agen memutuskan melalui obrolan. Percabangan muncul dari siapa yang berbicara selanjutnya. `GroupChatManager` memilih pembicara berikutnya; kamu dapat menulis sendiri `speaker_selection_method` tetapi defaultnya adalah berbasis LLM.
- **Agno** — agen memutuskan alat mana yang akan dihubungi selanjutnya. Tim memiliki mode koordinator/router/kolaborator; bercabang lebih dari itu adalah tanggung jawab pengembang.

### Pertanyaan observabilitas

- **LangGraph** — OpenTelemetry melalui LangSmith atau eksportir OTel mana pun. Setiap transisi node adalah rentang jejak; pos pemeriksaan berfungsi ganda sebagai jejak yang dapat diputar ulang. LangSmith adalah opsi pihak pertama; Langfuse/Phoenix juga memiliki adaptor.
- **CrewAI** — OpenTelemetry kelas satu sejak akhir tahun 2025; integrasi dengan Langfuse, Phoenix, Opik, AgentOps.
- **AutoGen** — Integrasi OpenTelemetry melalui `autogen-core`; AgentOps dan Opik memiliki konektor. Perincian pelacakan dilakukan per pesan agen, bukan per node.
- **Agno** — bendera `monitoring=True` bawaan ditambah eksportir OpenTelemetry; integrasi yang erat dengan Langfuse untuk pelacakan sesi.

### Biaya dan latensiKeempat framework menambahkan overhead per panggilan (logika framework, validasi, serialisasi). Urutan kasar peningkatan overhead: Agno ≈ LangGraph < CrewAI ≈ AutoGen. Perbedaannya didominasi oleh seberapa banyak tambahan perutean LLM yang dilakukan framework tersebut. Manajer hierarki CrewAI menggunakan token untuk memutuskan siapa yang berikutnya; `GroupChatManager` AutoGen juga. LangGraph hanya membelanjakan token saat kamu menulis `llm.invoke`. Jalur agen tunggal Agno tipis.

Jika biaya per proses penting, pilih perutean eksplisit (LangGraph edge, AutoGen `speaker_selection_method`) dibandingkan perutean yang dipilih LLM.

### Interoperabilitas

- **LangGraph** ↔ **LangChain** alat, retriever, LLM. Adaptor MCP kelas satu (alat yang diimpor sebagai server MCP).
- **CrewAI** ↔ alat yang diwarisi dari `BaseTool`; Alat LangChain, alat LlamaIndex, dan alat MCP semuanya beradaptasi. Delegasi kru-ke-kru melalui `allow_delegation=True`.
- **AutoGen** → `FunctionTool` membungkus semua callable Python; Adaptor MCP tersedia. Penggabungan erat dengan ekosistem AG2 untuk pola agen-ke-agen.
- **Agno** → `@tool` dekorator atau subkelas BaseTool; adaptor MCP; alat dapat dibagikan ke seluruh agen dan tim.

## Keterampilan

> kamu dapat menjelaskan, dalam satu kalimat, mengapa framework tertentu tepat untuk masalah agen tertentu.

Daftar periksa pra-pembuatan:

1. **Gambarlah bentuknya.** Apakah ini grafik (keadaan yang diketik, transisi yang diberi nama)? Sebuah permainan peran (para spesialis menyerahkan pekerjaan)? Obrolan (agen berbicara sampai selesai)? Agen tunggal dengan alat?
2. **Tentukan siapa yang mencabang.** Percabangan yang ditentukan pengembang → LangGraph. Diputuskan oleh manajer-agen → Hirarki CrewAI. Muncul obrolan → AutoGen. Alat-panggilan-diputuskan → Agno.
3. **Periksa APBN.** Apakah kamu memerlukan resume dari pos pemeriksaan? Perjalanan waktu? Gangguan manusia di tengah jalan? Jika ya, LangGraph adalah defaultnya; Sesi Agno mencakup keadaan cakupan percakapan.
4. **Periksa anggaran biaya.** Perutean yang dipilih LLM memerlukan token tambahan per giliran. Jika agen berjalan ribuan kali sehari, pilihlah perutean eksplisit.
5. **Anggarkan biaya overhead framework.** Setiap framework merupakan ketergantungan yang lain. Jika tugasnya adalah dua panggilan LLM dan sebuah alat, tulis 30 baris Python biasa; tidak ada framework lebih murah daripada tidak ada framework.

Menolak untuk menggunakan framework sebelum kamu dapat menggambar grafik, bagan organisasi, obrolan, atau kotak agen. Tolak untuk memilih salah satu yang memaksa kamu melawan model negaranya demi hal yang sebenarnya kamu butuhkan.

## Matrix Keputusan

| Bentuk Masalah | Kerangka kerja pilihan | Mengapa |
|---------------|---------------------|-----|
| Alur kerja DAG dengan status yang diketik, persetujuan manusia, berjalan lama | Grafik Lang | Status kelas satu, pos pemeriksaan, interupsi, perjalanan waktu. |
| Pipeline penelitian/penulisan dengan peran berbeda | Subgraf CrewAI (berurutan) atau LangGraph | Peran per tugas mudah untuk diungkapkan di CrewAI; tingkatkan dengan LangGraph ketika percabangan menjadi rumit. |
| Dialog pengusul-kritikus atau guru-siswa | Generasi Otomatis | Obrolan dua agen adalah bentuk aslinya. |
| Agen tunggal dengan alat, sesi, memori | Agno | Pengaturan tertipis, penyimpanan dan memori internal. |
| Ribuan fanout paralel dengan reduksi | LangGraph + `Send` | Satu-satunya yang memiliki primitif pengiriman paralel kelas satu. |
| Prototipe cepat, tanpa komitmen framework | SDK penyedia Python + biasa | Tidak ada framework yang merupakan framework tercepat. |

## Latihan1. **Mudah.** Ambil tugas yang sama — "meneliti kantor pusat Anthropic, menulis ringkasan 200 kata, mengutip sumber" — dan menerapkannya di LangGraph (empat node: merencanakan, mencari, menulis, mengutip) dan di CrewAI (tiga peran: peneliti, penulis, editor). Laporkan biaya token per proses dan baris code.
2. **Medium.** Buat tugas yang sama di AutoGen (obrolan peneliti ↔ penulis, editor bergabung melalui `GroupChat`) dan Agno (agen tunggal dengan `search_tools` dan `write_tools`, ditambah toko sesi). Beri peringkat keempat implementasi berdasarkan (a) biaya per pengoperasian, (b) kemampuan untuk melanjutkan setelah crash, (c) kemampuan untuk memasukkan persetujuan manusia sebelum langkah penulisan.
3. **Sulit.** Buat skrip pohon keputusan `pick_framework.py` yang mengambil deskripsi masalah singkat (JSON: `{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`) dan menampilkan rekomendasi dengan justifikasi satu kalimat. Verifikasikan pada enam kasus yang kamu desain sendiri.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Orkestrasi | "Bagaimana agen berkoordinasi" | Layer yang memutuskan node/peran/agen mana yang dijalankan selanjutnya. |
| Keadaan tahan lama | "Lanjutkan setelah restart" | Negara yang bertahan dari kematian proses, melekat pada pos pemeriksaan atau penyimpanan sesi. |
| Perutean yang dipilih LLM | "Biarkan model yang memutuskan" | LLM perencana memilih langkah berikutnya setiap giliran; fleksibel tetapi membayar token pada setiap keputusan. |
| Perutean eksplisit | "Pengembang memutuskan" | Fungsi Python atau tepi statis memilih langkah berikutnya; murah dan dapat diaudit. |
| kru | "Tim CrewAI" | Peran + tugas + proses (berurutan atau hierarki) terikat menjadi satu runnable. |
| Obrolan Grup | "Obrolan multi-agen AutoGen" | Percakapan terkelola antara N agen dengan pemilih pembicara. |
| Tim (Agno) | "Agno multi-agen" | Mode rute/koordinasi/kolaborasi melalui sekumpulan agen. |
| Grafik Negara | "Grafik LangGraph" | Status yang diketik, simpul, tepi bersyarat, primitif checkpointer. |

## Bacaan Lanjutan

- [Dokumentasi LangGraph](https://langchain-ai.github.io/langgraph/) — StateGraph, checkpointer, interupsi, perjalanan waktu.
- [Dokumentasi CrewAI](https://docs.crewai.com/) — Kru, Alur, Agen, Tugas, Proses.
- [Dokumentasi AutoGen](https://microsoft.github.io/autogen/) — ConversableAgent, GroupChat, tim, alat.
- [Dokumentasi Agno](https://docs.agno.com/) — Agen, Tim, Alur Kerja, penyimpanan, memori.
- [Anthropic — Membangun agen yang efektif (Des 2024)](https://www.anthropic.com/research/building- Effective-agents) — pustaka pola (rangkaian cepat, perutean, paralelisasi, pekerja orkestra, optimizer-evaluator) framework-agnostik.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — hal primitif yang didandani oleh setiap framework.
- [Wu dkk., "AutoGen: Mengaktifkan Aplikasi LLM Generasi Berikutnya melalui Percakapan Multi-Agen" (2023)](https://arxiv.org/abs/2308.08155) — Makalah desain AutoGen.
- [Park dkk., "Agen Generatif: Simulacra Interaktif Perilaku Manusia" (UIST 2023)](https://arxiv.org/abs/2304.03442) — fondasi permainan peran yang menjadi dasar tumpukan persona gaya CrewAI.
- Fase 11 · 16 (LangGraph) — framework yang menjadi tolok ukur lesson ini.
- Fase 11 · 19 (Refleksi) — pola yang terpetakan dengan rapi ke LangGraph tetapi tidak cocok untuk CrewAI.
- Fase 11 · 22 (Kemampuan observasi produksi) — cara melengkapi framework mana pun yang kamu pilih.
