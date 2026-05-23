# Negosiasi dan Tawar-menawar

> Agen menegosiasikan sumber daya, harga, alokasi tugas, dan persyaratan. Kumpulan tolok ukur tahun 2026 jelas: NegotiationArena (arXiv:2402.05863) menunjukkan LLM dapat meningkatkan hasil ~20% melalui manipulasi persona ("keputusasaan"); "Mengukur Kemampuan Tawar-menawar" (arXiv:2402.15813) menunjukkan pembeli lebih sulit daripada penjual dan skala tidak membantu — **OG-Narrator** mereka (penghasil penawaran deterministik + narator LLM) mendorong tingkat kesepakatan dari 26,67% menjadi 88,88%; Kompetisi Negosiasi Otonomi Skala Besar (arXiv:2503.06416) menjalankan ~180 ribu negosiasi dan menemukan bahwa agen **rantai pemikiran yang menyembunyikan** menang dengan menyembunyikan alasan dari pihak lawan; Bhattacharya dkk. Tahun 2025 berdasarkan metrik Proyek Negosiasi Harvard memberi peringkat Llama-3 paling efektif, Claude-3 agresif, GPT-4 paling adil. Lesson ini mengimplementasikan Protokol Jaringan Kontrak (nenek moyang FIPA, Lesson 02), menghubungkan pembeli/penjual gaya LLM, menjalankan decomposition gaya OG-Narrator, dan mengukur bagaimana tingkat kesepakatan berubah dengan setiap pilihan struktural.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 16 · 02 (Warisan FIPA-ACL), Fase 16 · 09 (Jaringan Kawanan Paralel)
**Waktu:** ~75 menit

## Masalah

Dua agen perlu menyepakati harga. Jika dibiarkan sendiri dengan petunjuk bahasa yang murni, LLM 2024-2026 menutup transaksi dengan harga yang sangat rendah (~27% pada penawaran dengan parameter ketat di arXiv:2402.15813). Skala tidak menentukannya: GPT-4 secara struktural tidak lebih baik dalam melakukan tawar-menawar dibandingkan GPT-3.5; lebih baik dalam *bahasa* tawar-menawar.

Akar permasalahannya adalah LLM menggabungkan dua pekerjaan - memutuskan tawaran dan menceritakan tawaran tersebut. OG-Narator memisahkan ini: generator penawaran deterministik menghitung pergerakan numerik; LLM hanya menceritakan. Tingkat kesepakatan melonjak menjadi ~89%.

Hal ini mencerminkan temuan klasik multi-agen: pemisahan mekanisme dari layer komunikasi adalah pemenangnya. Contract Net Protocol (FIPA, 1996; Smith, 1980) adalah mekanisme pasar tugas yang menjadi acuan. Pasang LLM ke dalam slot narasi dan kamu mendapatkan pasar tugas modern yang didukung LLM.

## Konsep

### Kontrak Bersih, dalam satu paragraf

Protokol Jaringan Kontrak Smith tahun 1980: seorang **manajer** menyiarkan **panggilan untuk proposal (cfp)**; **penawar** merespons dengan pesan **usulan** yang berisi penawaran mereka; manajer memilih pemenang dan mengirimkan **proposal-terima** kepada pemenang dan **proposal-tolak** kepada yang kalah. Pemenangnya melakukan pekerjaannya. Pesan opsional: **menolak** (penawar menolak mengusulkan). FIPA mengkodifikasikan ini sebagai protokol interaksi `fipa-contract-net`.

### Mengapa OG-Narator menang

"Mengukur Kemampuan Tawar-menawar Model Bahasa" (arXiv:2402.15813) mengamati bahwa:

- LLM sering kali melanggar aturan tawar-menawar (menawarkan dengan harga yang tidak masuk akal, mengabaikan ZOPA pihak lain).
- Mereka berlabuh dengan buruk (menerima tawaran pertama yang buruk; menawarkan balasan dengan jumlah yang simbolis dan bukan dalam jumlah yang strategis).
- Skala saja tidak memperbaiki masalah ini. Model yang lebih besar membuat bahasa lebih masuk akal dengan kesalahan strategis serupa.

Decomposition OG-Narator:

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

Pembuat penawaran adalah strategi negosiasi klasik: model tawar-menawar Rubinstein, strategi Zeuthen, atau saling balas harga yang sederhana. LLM menceritakan. Pesan tersebut berisi harga deterministik dan kerangka bahasa alami.

Tingkat kesepakatan melonjak karena:
- Harga tetap berada di zona tawar.
- Jangkar bersifat strategis, bukan emosional.
- LLM melakukan keahliannya: menulis.

### Temuan NegosiasiArena

arXiv:2402.05863 menyediakan tolok ukur kanonik. Temuan utama:- LLM dapat meningkatkan keuntungan ~20% dengan mengadopsi persona ("Saya sangat ingin menjual ini pada hari Jumat") — manipulasi persona adalah taktik yang nyata.
- Agen yang adil/kooperatif dieksploitasi oleh pihak yang bermusuhan; pertahanan memerlukan sikap balasan yang eksplisit.
- Pasangan simetris menghasilkan hasil yang tidak adil pada sekitar 40% skenario benchmark.

Ini bukan "LLM adalah negosiator yang buruk." Ini adalah "LLM bernegosiasi terlalu banyak seperti manusia, termasuk bagian yang dapat dieksploitasi."

### Penyembunyian rantai pemikiran

Kompetisi Negosiasi Otonomi Skala Besar (arXiv:2503.06416) menjalankan ~180 ribu negosiasi di banyak strategi LLM. Pemenang menyembunyikan alasan mereka dari rekan-rekan mereka:

- Jika agen mencetak "Saya hanya akan pergi ke $75; harga reservasi saya adalah $70" pada kertas gores yang terlihat oleh publik, lawan akan membacanya.
- Pemenang menghitung strategi secara pribadi; pipeline output hanya berisi penawaran dan narasi minimum yang diperlukan.

Ini adalah gema teori permainan klasik tahun 2026 (Aumann 1976 tentang rasionalitas dan informasi): mengungkapkan hasil dari biaya penilaian pribadi kamu. LLM tidak mengetahui hal ini dan dengan senang hati mengetikkan keberatan mereka dalam jejak penalaran yang dapat dilihat oleh rekannya.

Kesimpulan teknis: pisahkan konteks pesan pribadi dari konteks pesan publik. Bukan opsional.

### Bhattacharya dkk. 2025 — peringkat model

Mengenai metrik Proyek Negosiasi Harvard (negosiasi berprinsip, rasa hormat BATNA, timbal balik kepentingan):

- **Llama-3** paling efektif dalam melakukan tawar-menawar (tarif kesepakatan + imbalan).
- **Claude-3** adalah negosiator paling agresif (jangkar tinggi, konsesi terlambat).
- **GPT-4** adalah yang paling adil (variasi terkecil dalam hasil di seluruh pasangan).

Ini adalah cuplikan tahun 2025. Intinya bukanlah model mana yang menang pada bulan April 2026 — namun model dasar yang berbeda memiliki gaya negosiasi yang gigih. Ansambel heterogen (Lesson 15) memasukkan ini sebagai sumber keberagaman.

### Alokasi tugas melalui Contract Net + LLM

Penggunaan kembali Net Kontrak secara modern untuk multi-agen LLM:

1. Agen manajer menguraikan tugas menjadi beberapa unit.
2. Menyiarkan `cfp` dengan deskripsi tugas kepada agen pekerja.
3. Setiap pekerja mengembalikan penawaran: `(price, eta, confidence)` yang harganya bisa berupa token, unit komputasi, atau dolar.
4. Manajer memilih pemenang (tunggal atau ganda, tergantung tugas) dan penghargaan.
5. Pekerja yang ditolak bebas mengajukan tawaran untuk tugas lain.

Jumlah ini melampaui 100 pekerja karena koordinasi dilakukan melalui siaran dan respons, bukan obrolan yang disinkronkan. Digunakan dalam produksi: pola orkestrasi Microsoft Agent Framework, beberapa implementasi LangGraph.

### LLM-Negosiasi Interaktif Pemangku Kepentingan

NeurIPS 2024 (https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) memperkenalkan permainan skor multi-partai dengan **skor rahasia** dan **ambang batas penerimaan minimum**. Setiap pemangku kepentingan mempunyai utilitas swasta; LLM harus menyimpulkannya dari pesan. Ini adalah generalisasi dari tawar-menawar dua partai terhadap pembentukan koalisi partai-N. Relevan untuk pasar tugas produksi dengan kemampuan pekerja yang heterogen.

### Aturan narasi-vs-mekanisme

Di seluruh tolok ukur negosiasi tahun 2024-2026, aturan teknis yang konsisten adalah:

> Biarkan LLM menceritakan. Jangan biarkan LLM menghitung tawaran tersebut.Jika penawaran harus berupa angka (harga, ETA, kuantitas), buat penawaran tersebut secara deterministik dari status negosiasi dan minta LLM membuat kerangkanya. Jika penawaran perlu berupa struktur proposal (decomposition tugas, penetapan peran), biarkan LLM menyusunnya, tetapi validasi berdasarkan skema dan pemeriksaan batasan sebelum dikirim.

## Build

`code/main.py` mengimplementasikan:

- `ContractNetManager`, `ContractNetTask`, `Bid` — manajer + penawar, menyiarkan cfp, mengumpulkan proposal, memberikan penghargaan.
- `og_narrator_bargain(state, rng)` — Pembeli OG-Narator: konsesi gaya Zeuthen yang deterministik menuju titik tengah.
- `seller_response(state, rng)` — kebijakan penawaran balik penjual deterministik (kebenaran dasar struktural untuk kedua gaya).
- `naive_llm_bargain(state, rng)` — menyimulasikan tawar-menawar yang semuanya LLM: memilih harga dengan varian tinggi, sering kali di luar ZOPA.
- Pengukuran: tingkat kesepakatan lebih dari 1000 uji coba dengan harga reservasi baru yang diambil sampelnya per uji coba.

Jalankan:

```
python3 code/main.py
```

Hasil yang diharapkan: tingkat kesepakatan LLM yang naif ~65-75%; Tingkat kesepakatan OG-Narator ~85-95%; kesenjangan poin 15-25 adalah keuntungan struktural dari penguraian generasi penawaran dari narasi. Ditambah contoh alokasi pasar tugas Jaringan Kontrak dengan tiga penawar dan satu tugas.

## Pakai

`outputs/skill-bargainer-designer.md` merancang protokol tawar-menawar: siapa yang menghasilkan penawaran (deterministik atau LLM), siapa yang menceritakan, bagaimana papan gores pribadi terpisah dari pesan publik, dan bagaimana tingkat kesepakatan dipantau.

## Kirim

Daftar periksa tawar-menawar produksi:

- **Goresan terpisah.** Status privat tidak pernah mencapai konteks mitranya. Hal ini tidak dapat dinegosiasikan.
- **Pembuatan penawaran deterministik.** Harga, jumlah, ETA: hitung, jangan minta.
- **Validasi semua penawaran masuk** berdasarkan skema. Tolak tawaran di luar ZOPA di batas protokol.
- **Putaran terikat.** Maksimum 3-5 putaran; meningkat menjadi mediator jika terjadi kebuntuan.
- **Ukur tingkat kesepakatan dan variansi pembayaran** secara terus menerus. Penurunan tingkat transaksi merupakan salah satu gejalanya — sering kali merupakan penyimpangan yang terjadi secara tiba-tiba atau serangan dari pihak lawan.
- **Catat semua proposal yang ditolak** dengan alasan deterministik. Bagi manajer Contract Net, penawar yang kalah perlu memahami alasannya.

## Latihan

1. Jalankan `code/main.py`. Konfirmasikan bahwa OG-Narator mengalahkan LLM yang naif dalam hal tingkat kesepakatan. Berapa banyak?
2. Menerapkan **peningkatan pembayaran berbasis persona** (arXiv:2402.05863) — pembeli mengadopsi persona "sangat ingin membeli minggu ini" hanya dalam narasi, generator penawaran tidak berubah. Apakah tingkat kesepakatan atau imbalannya berubah?
3. Menerapkan **penyembunyian** rantai pemikiran: pertahankan string scratchpad pribadi yang tidak diteruskan ke rekanan. Apa yang terjadi jika kamu tidak sengaja membocorkannya (simulasikan dengan menukar pipeline)?
4. Memperluas Kontrak Bersih ke lelang N-penawar dengan harga cadangan. Ketika semua tawaran melebihi cadangan, bagaimana manajer memutuskan antara harga terendah dan kualitas tertinggi? Aturan penghargaan mana yang kamu pilih dan mengapa?
5. Baca Bhattacharya dkk. 2025 tentang metrik Proyek Negosiasi Harvard. Terapkan dua tawar-menawar dengan gaya berbeda (agresif vs adil). Ukur varians hasil pada pasangan simetris dan asimetris.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Kontrak Bersih | "Pasar tugas" | Smith 1980, FIPA 1996. cfp + mengusulkan + menerima/menolak. Pasar tugas kanonik. |
| ZOPA | "Zona kemungkinan kesepakatan" | Tumpang tindih antara jumlah maksimum pembeli dan jumlah minimum penjual. Penawaran di luar itu tidak dapat ditutup. |
| BATNA | "Alternatif terbaik untuk perjanjian yang dinegosiasikan" | Pengganti kamu jika kesepakatan ini gagal. Tetapkan harga reservasi kamu. |
| OG-Narator | "Penawaran generator + narator" | Decomposition: penawaran deterministik, narasi LLM. |
| Strategi Zeuthen | "Konsesi yang meminimalkan risiko" | Generator penawaran klasik yang memberikan persetujuan berdasarkan batas risiko. |
| Tawar-menawar Rubinstein | "Keseimbangan penawaran-bergantian" | Model teori permainan untuk tawar-menawar cakrawala tak terbatas dengan diskon. |
| penyembunyian CoT | "Sembunyikan alasanmu" | Pemenang di arXiv:2503.06416 menyimpan buku gores pribadi; penawaran acara pipeline publik saja. |
| Manipulasi persona | "Postur emosional" | arXiv:2402.05863: ~20% keuntungan dari persona keputusasaan/urgensi. |

## Bacaan Lanjutan

- [NegotiationArena](https://arxiv.org/abs/2402.05863) — patokan; temuan manipulasi dan eksploitasi persona
- [Mengukur Kemampuan Tawar-menawar Model Bahasa](https://arxiv.org/abs/2402.15813) — OG-Narrator dan hasil pembeli-lebih keras-dari-penjual
- [Kompetisi Negosiasi Otonomi Skala Besar](https://arxiv.org/abs/2503.06416) — ~180 ribu negosiasi; penyembunyian rantai pemikiran menang
- [Negosiasi Interaktif Pemangku Kepentingan LLM (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) — permainan skor multi-pihak dengan utilitas rahasia
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — mekanisme klasik, IEEE Transactions on Computers
