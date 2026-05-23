# Warisan FIPA-ACL dan Kisah Pidato

> Sebelum MCP, sebelum A2A, ada FIPA-ACL. Pada tahun 2000, IEEE Foundation for Intelligent Physical Agents meratifikasi bahasa komunikasi agen dengan dua puluh performatif, dua bahasa konten, dan serangkaian protokol interaksi — jaringan kontrak, berlangganan/beritahu, permintaan-kapan. Hal ini memudar dari industri karena overhead ontologi terlalu berat untuk web, namun kebangkitan sistem multi-agen LLM secara diam-diam menerapkan kembali gagasan yang sama tanpa semantik formal: kontrak JSON mewakili performatif, bahasa alami mewakili ontologi. Lesson ini membaca FIPA-ACL dengan serius sehingga kamu dapat melihat keputusan protokol tahun 2026 mana yang merupakan penemuan kembali, mana yang merupakan hal baru, dan di mana gelombang saat ini akan menemukan kembali masalah-masalah yang telah diselesaikan pada tahun 2000an.

**Type:** Learn
**Language:** Python (stdlib)
**Prerequisites:** Fase 16 · 01 (Mengapa Multi-Agen)
**Waktu:** ~60 menit

## Masalah

Lanskap protokol agen pada tahun 2026 sibuk: MCP untuk alat, A2A untuk agen, ACP untuk audit perusahaan, ANP untuk kepercayaan terdesentralisasi, NLIP untuk konten bahasa alami, ditambah CA-MCP dan dua lusin proposal penelitian. Setiap spesifikasi menyatakan dirinya sebagai dasar.

Pembacaan jujurnya adalah bahwa kebanyakan dari mereka menemukan kembali pohon keputusan berumur dua puluh tahun yang sangat spesifik. Teori tindak tutur dari Austin (1962) dan Searle (1969) memberi kita “ucapan adalah tindakan”. KQML (1993) mengubahnya menjadi protokol kabel. FIPA-ACL (diratifikasi tahun 2000) menghasilkan standarisasi referensi: dua puluh performatif, bahasa konten SL0/SL1, protokol interaksi untuk contract-net dan subscribe-notify. JADE dan JACK adalah platform referensi Java. Upaya tersebut memudar sekitar tahun 2010 karena overhead ontologi terlalu berat dan web menjadi pemenang.

Saat kamu melihat `tools/call` MCP, siklus hidup tugas A2A, atau penyimpanan konteks bersama CA-MCP, kamu sedang melihat pengulangan keputusan FIPA yang lebih lembut dan asli JSON. Mengetahui warisan tersebut memberi tahu kamu dua hal: "inovasi" baru mana yang sebenarnya merupakan penemuan kembali, dan mode kegagalan lama mana yang akan ditemukan kembali oleh spesifikasi baru.

## Konsep

### Tindak tutur, dalam satu paragraf

Austin memperhatikan bahwa beberapa kalimat tidak menggambarkan dunia - mereka mengubahnya. "Saya berjanji." "Saya meminta." "Saya nyatakan." Ia menyebutnya sebagai ujaran performatif. Searle memformalkan lima kategori: asertif, direktif, komisif, ekspresif, deklaratif. KQML (Finin et al., 1993) menjadikan hal ini operasional untuk agen perangkat lunak: pesan adalah performatif (tindakan) ditambah konten (tentang apa tindakan tersebut). FIPA-ACL membersihkan kesenjangan KQML dan menstandarkan sekitar dua puluh performatif.

### Dua puluh pertunjukan FIPA (daftar sebagian)

| Performatif | Maksud |
|---|---|
| `inform` | "Sudah kubilang P itu benar" |
| `request` | "Saya meminta kamu melakukan X" |
| `query-if` | "Apakah P benar?" |
| `query-ref` | “Berapa nilai Xnya?” |
| `propose` | "Saya usulkan kita melakukan X" |
| `accept-proposal` | "Saya menerima lamaran" |
| `reject-proposal` | "Saya menolak lamaran" |
| `agree` | "Saya setuju untuk melakukan X" |
| `refuse` | "Saya menolak melakukan X" |
| `confirm` | "Saya konfirmasi P benar" |
| `disconfirm` | "Saya menolak P" |
| `not-understood` | "Pesan kamu tidak diuraikan" |
| `cfp` | "Panggilan proposal di X" |
| `subscribe` | "Beri tahu saya jika X berubah" |
| `cancel` | "Batalkan X yang sedang berlangsung" |
| `failure` | "Saya mencoba X dan gagal" |Daftar lengkapnya ada di `fipa00037.pdf` (Struktur Pesan FIPA ACL). Intinya adalah jangan menghafalnya — intinya adalah bahwa setiap hal ini sesuai dengan protokol LLM primitif yang akhirnya ditambahkan kembali.

### Pesan FIPA-ACL kanonik

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

Tujuh bidang membawa amplop protokol; satu bidang (`content`) membawa muatan. Bidang lainnya persis seperti yang kamu temukan kembali setiap kali kamu melakukan percobaan ulang, threading, dan ontologi ke protokol JSON.

### Dua platform lama

**JADE** (kerangka Pengembangan Agen Java, 1999–2020an) adalah runtime yang sesuai dengan FIPA yang paling banyak digunakan. Agen memperluas kelas dasar, bertukar pesan ACL, menjalankan di dalam container, dan berkoordinasi menggunakan "perilaku". Pustaka protokol interaksi dikirimkan dengan contract-net, subscribe-notify, request-when, dan proposal-accept.

**JACK** (Perangkat Lunak Berorientasi Agen, komersial) menekankan alasan BDI (Keyakinan-Keinginan-Niat) di atas pesan FIPA. Lebih formal, kurang diadopsi.

Keduanya menolak setelah tumpukan web memakan kasus penggunaan multi-agen. MCP dan A2A adalah "wadah" runtime tahun 2026.

### Mengapa FIPA memudar

- **Overhead ontologi.** FIPA memerlukan ontologi bersama untuk mengurai `content`. Menyetujui ontologi adalah proses standar yang memakan waktu bertahun-tahun. Web hanya menggunakan HTTP + JSON.
- **Semantik formal tidak digunakan oleh siapa pun.** SL (Bahasa Semantik) memberikan kondisi kebenaran yang ketat, namun sebagian besar sistem produksi menggunakan konten bentuk bebas dan mengabaikan formalisme.
- **Penguncian perkakas.** JADE hanya untuk Java; JACK bersifat komersial. Tim Polyglot berhasil mengatasi keduanya.
- **Internet memenangkan tumpukan.** REST, lalu JSON-RPC, lalu gRPC menggantikan transport ACL.

### Kebangkitan LLM adalah FIPA-lite

Bandingkan FIPA `request` dengan MCP `tools/call`:

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

Amplop yang sama, sintaksis berbeda. Keduanya membawa: siapa, siapa, niat, muatan, id korelasi. Tidak ada revolusi satu sama lain – keduanya merupakan trade-off yang berbeda pada desain yang sama.

Survei tahun 2025 oleh Liu dkk. ("Survei Protokol Interoperabilitas Agen: MCP, ACP, A2A, ANP", arXiv:2505.02279) menjadikan silsilah ini eksplisit: MCP berhubungan dengan tindak tutur penggunaan alat, A2A dengan tindak tutur rekan-agen, ACP dengan tindak tutur jejak audit, ANP dengan ekstensi identitas terdesentralisasi. Spesifikasi baru adalah turunan ACL dengan sintaksis JSON dan semantik yang lebih longgar.

### Pertukarannya, dinyatakan dengan jelas

**Apa yang FIPA berikan kepada kamu dan penurunan spesifikasi modern:**

- Semantik formal — kamu dapat membuktikan `inform` menyiratkan bahwa pengirim memercayai konten tersebut.
- Katalog performatif kanonik — kamu tidak perlu berdebat ulang "haruskah kita memiliki `cancel`?".
- Pola protokol interaksi selama puluhan tahun — jaringan kontrak, berlangganan-beritahu, usulkan-terima — dengan properti kebenaran yang diketahui.

**Spesifikasi modern apa yang diberikan kepada kamu dan tidak diberikan FIPA:**

- Payload asli JSON kompatibel dengan setiap alat modern.
- Konten dalam bahasa alami yang dapat ditafsirkan oleh LLM tanpa ontologi code tangan.
- Transportasi tumpukan web (HTTP, SSE, WebSocket).
- Penemuan kemampuan melalui dokumen yang menggambarkan diri sendiri (MCP `listTools`, Kartu Agen A2A).

Semantik maksud yang lebih longgar untuk implementasi yang lebih mudah. Itulah tradeoff yang sebenarnya.

### Protokol interaksi layak untuk porting

FIPA mengirimkan ~15 protokol interaksi. Ada tiga hal yang layak untuk diterapkan ke dalam sistem multi-agen LLM:1. **Contract Net Protocol (CNP).** Manajer menerbitkan `cfp` (panggilan untuk proposal); penawar merespons dengan `propose`; manajer menerima/menolak. Ini adalah pola pasar tugas yang kanonik (Fase 16 · 16 Negosiasi).
2. **Berlangganan/Beritahu.** Pelanggan mengirim `subscribe`; penerbit mengirimkan `inform` setiap kali topik berubah. Ini adalah setiap bus acara pada tahun 2026.
3. **Permintaan-Kapan.** "Lakukan X ketika kondisi Y terpenuhi." Tindakan tertunda dengan prasyarat. Analog tahun 2026 adalah tugas yang ditangguhkan dalam mesin alur kerja yang tahan lama (Fase 16 · 22 Penskalaan Produksi).

Masing-masing dipetakan dengan rapi ke antrean pesan modern, polling HTTP +, atau streaming SSE.

### Apa yang rusak saat kamu melepaskan ontologi

Tanpa ontologi bersama, agen menyimpulkan makna dari konten bahasa alami. Mode kegagalan tahun 2026 yang terdokumentasi adalah **penyimpangan semantik**: dua agen menggunakan kata yang sama (`"customer"`) untuk konsep yang sedikit berbeda, agen penerima bertindak berdasarkan interpretasi yang salah, tidak ada validator skema yang menangkapnya. Persyaratan ontologi FIPA akan menolak pesan tersebut pada waktu parse.

Mitigasi tanpa ontologi penuh:

- Skema JSON di `content` — menolak kesalahan struktural pada kabel.
- Artefak yang diketik (A2A) — menolak modalitas yang salah.
- Performatif eksplisit di dalam amplop — membuat maksud menjadi jelas meskipun kontennya menggunakan bahasa alami.

### Spesifikasi tahun 2026, dipetakan berdasarkan warisan tindak tutur

| Spesifikasi modern | Analog FIPA | Apa yang disimpannya | Apa yang dijatuhkannya |
|---|---|---|---|
| MCP `tools/call` | `request` | maksud eksplisit, id korelasi | semantik formal, ontologi |
| MCP `resources/read` | `query-ref` | maksud eksplisit, id korelasi | semantik formal |
| Siklus hidup Tugas A2A | kontrak-net + permintaan-kapan | siklus hidup async, transisi status | jaminan kelengkapan formal |
| Acara streaming A2A | berlangganan/beri tahu | dorongan asinkron | berlangganan predikat yang diketik |
| Konteks bersama CA-MCP | papan tulis (Hayes-Roth 1985) | memori bersama multi-penulis | model konsistensi logis |
| NLIP | konten berbahasa alami | LLM-asli | skema |

Membaca tabel dari atas ke bawah, polanya adalah: pertahankan struktur primitif, hilangkan formalisme, biarkan LLM mengatasi ambiguitas.

## Build

`code/main.py` mengimplementasikan penerjemah FIPA-ACL stdlib murni. Ini mengkodekan dan mendekode amplop ACL kanonik dan menunjukkan bagaimana setiap bentuk pesan MCP/A2A direduksi menjadi tujuh bidang yang sama. Demonya:

- Mengkodekan lima pesan gaya MCP dan gaya A2A sebagai FIPA-ACL.
- Mendekode FIPA-ACL kembali ke versi modernnya.
- Menjalankan negosiasi Jaringan Kontrak mainan antara satu manajer dan tiga penawar menggunakan `cfp`, `propose`, `accept-proposal`, `reject-proposal`.

Jalankan:

```
python3 code/main.py
```

Outputnya adalah jejak berdampingan yang menunjukkan setiap pesan modern dalam bentuk JSON 2026 dan formulir FIPA-ACL, lalu bolak-balik tawaran bersih kontrak. Protokol primitif yang sama bertahan dalam perjalanan bolak-balik; hanya sintaksisnya saja yang berbeda.

## Pakai

`outputs/skill-fipa-mapper.md` adalah keterampilan yang membaca spesifikasi protokol agen apa pun dan menghasilkan pemetaan FIPA-ACL. Gunakan sebelum mengadopsi protokol baru untuk menjawab: "Apakah ini benar-benar baru, atau `inform` dengan sintaksis JSON?"

## Kirim

Jangan kembalikan FIPA-ACL. Kembalikan daftar periksanya:- Apa maksud primitif (performatif) dari setiap pesan?
- Apakah ada id korelasi untuk permintaan-respons dan pembatalan?
- Apakah ada bahasa konten eksplisit (JSON-RPC, teks biasa, artefak yang diketik terstruktur)?
- Apakah protokol interaksi kelas satu, atau apakah kamu menerapkan kembali contract-net dari awal?
- Apa yang terjadi jika dua agen tidak sepakat mengenai makna konten (penyimpangan semantik)?

Dokumentasikan lima pertanyaan ini untuk setiap protokol baru sebelum kamu mengirimkannya ke produksi.

## Latihan

1. Jalankan `code/main.py`. Amati pengkodean pulang-pergi. Identifikasi performatif FIPA mana yang sesuai dengan `tools/call`, `resources/read`, dan pembuatan tugas A2A.
2. Perpanjang demo kontrak bersih dengan performatif `cancel` yang memungkinkan manajer menarik tugas di tengah tawaran. Kasus kegagalan apa yang dapat diselesaikan oleh `cancel` yang tidak dapat diselesaikan dengan percobaan ulang saja?
3. Baca Struktur Pesan FIPA ACL (http://www.fipa.org/specs/fipa00037/) bagian 4.1–4.3. Pilih satu performatif yang tidak tercakup dalam lesson ini dan jelaskan analog JSON-RPC modernnya.
4. Baca Liu dkk., arXiv:2505.02279. Untuk masing-masing MCP, A2A, ACP, ANP, buat daftar keluarga performatif FIPA yang dipertahankan dan dihilangkan.
5. Rancang Skema JSON minimal untuk bidang `content` dari performatif `request` di sistem kamu sendiri. Apa yang skema tersebut berikan kepada kamu yang tidak diberikan oleh bahasa alami murni, dan berapa biayanya?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Tindak tutur | "Ucapan yang melakukan sesuatu" | Austin/Searle: ucapan sebagai tindakan. Induk teoretis dari ACL. |
| FIPA | "Hal XML lama itu" | Yayasan IEEE untuk Agen Fisik Cerdas. ACL terstandarisasi pada tahun 2000. |
| ACL | "Bahasa Komunikasi Agen" | Format amplop FIPA: performatif + konten + metadata. |
| Performatif | "Kata kerja" | Kelas maksud pesan: `inform`, `request`, `propose`, `cfp`, dll. |
| KQML | "Pendahulu FIPA" | Pengetahuan Query dan Bahasa Manipulasi (1993). Lebih sederhana, lebih sempit. |
| Ontologi | "Kosakata bersama" | Definisi formal dari konsep yang dibicarakan oleh bahasa konten. |
| SL0 / SL1 | "Bahasa konten FIPA" | Bahasa Semantik tingkat 0 dan 1 - rumpun bahasa konten formal. |
| Kontrak Bersih | "Pasar tugas" | Manajer mengeluarkan cfp; penawar mengusulkan; manajer menerima. Protokol interaksi kanonik. |
| Protokol interaksi | "Pola pesan" | Urutan performatif dengan kebenaran yang diketahui: permintaan-kapan, berlangganan-beritahu, dll. |

## Bacaan Lanjutan

- [Liu dkk. — Survei Protokol Interoperabilitas Agen: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — survei kanonik tahun 2025 yang menghubungkan spesifikasi modern dengan warisan FIPA
- [Spesifikasi Struktur Pesan ACL FIPA (fipa00037)](http://www.fipa.org/specs/fipa00037/) — format amplop tahun 2000 yang diratifikasi
- [Spesifikasi Perpustakaan Undang-Undang Komunikatif FIPA (fipa00037)](http://www.fipa.org/specs/fipa00037/) — katalog performatif lengkap
- [Spesifikasi MCP 25-11-2025](https://modelcontextprotocol.io/spesification/2025-11-25) — penggunaan alat modern yang setara dengan `request`/`query-ref`
- [Spesifikasi A2A](https://a2a-protocol.org/latest/spesifikasi/) — agen modern yang setara dengan contract-net dan subscribe-notify
