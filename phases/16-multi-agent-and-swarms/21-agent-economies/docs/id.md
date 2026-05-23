# Ekonomi Agen, Insentif Token, Reputasi

> Agen otonom jangka panjang (kurva kerja METR 1 hingga 8 jam) memerlukan agen ekonomi. **Tumpukan 5 lapis** yang muncul adalah: **DePIN** (komputasi fisik) → **Identitas** (DID W3C + modal reputasi) → **Kognisi** (RAG + MCP) → **Penyelesaian** (abstraksi akun) → **Tata Kelola** (DAO Agentik). Jaringan insentif agen produksi mencakup **Bittensor** (subnet TAO memberi penghargaan pada model khusus tugas), **Fetch.ai / ASI Alliance** (ASI-1 Mini LLM + token FET), dan **Gonka** (PoW berbasis Transformer yang mengalokasikan ulang komputasi ke tugas-tugas AI yang produktif). Karya akademis: LaMAS terdesentralisasi AAMAS 2025 menggunakan **Atribusi kredit nilai Shapley** untuk memberikan penghargaan yang adil kepada agen yang berkontribusi; Riset Google "Desain mekanisme untuk large language model" mengusulkan **lelang token** dengan pembayaran harga kedua berdasarkan agregasi monoton. Lesson ini membangun pasar agen minimal, menerapkan atribusi kredit nilai Shapley ke pipeline multi-agen, dan menjalankan lelang token harga kedua sehingga mesin teori permainan dapat diterapkan secara konkret.

**Type:** Learn
**Language:** Python (stdlib)
**Prerequisites:** Fase 16 · 16 (Negosiasi dan Perundingan), Fase 16 · 09 (Jaringan Kawanan Paralel)
**Waktu:** ~75 menit

## Masalah

Sistem multi-agen menjadi rumit ketika agen menghasilkan nilai secara bersama-sama namun perlu diberi penghargaan secara individu. Mekanisme klasik - pembagian yang setara, kontributor terakhir mengambil semuanya - tidak adil atau dapat dimainkan. Penghargaan berbasis koalisi melalui nilai-nilai Shapley adil berdasarkan konstruksi tetapi mahal untuk dihitung. Literatur tahun 2025-2026 mendorong perkiraan yang berguna: pengambilan sample Shapley, lelang agregasi monoton, dan reputasi on-chain yang diperoleh dari kontribusi yang dikonfirmasi.

Di luar atribusi kredit, bidang ini telah beralih ke agen ekonomi yang sebenarnya: Bittensor TAO memberi penghargaan pada komputasi penambangan untuk menyempurnakan model spesifik subnet, Fetch.ai/ASI memberi penghargaan pada penggunaan ASI-1 Mini LLM dengan token FET, Gonka mengalokasikan kembali bukti kerja Transformer ke arah tugas-tugas AI yang produktif. Agen yang bertransaksi secara mandiri ada saat ini; pertanyaannya adalah bagaimana menyelaraskan insentif.

Lesson ini memperlakukan ekonomi agen sebagai sebuah kelompok permasalahan yang spesifik – atribusi kredit, desain mekanisme, dan reputasi – dan membangun masing-masing masalah tersebut dengan perhitungan minimal sehingga ide-idenya melekat.

## Konsep

### Tumpukan ekonomi agen 5 lapis

1. **DePIN (komputasi fisik).** Infrastruktur terdesentralisasi yang menyewakan GPU, penyimpanan, bandwidth. Subnet Bittensor, Jaringan Render, Akash. Tidak khusus untuk agen; agen menggunakannya.
2. **Identitas.** Pengidentifikasi Terdesentralisasi (DID) W3C memberi setiap agen ID tahan lama yang tidak bergantung pada platform apa pun. Reputasi bertambah pada DID. Agent Network Protocol (ANP) menggunakan DID sebagai layer penemuan.
3. **Kognisi.** Lingkaran penalaran agen: LLM + RAG + MCP. Inilah yang dibangun oleh fase-fase lainnya.
4. **Penyelesaian.** Abstraksi akun (ERC-4337) memungkinkan agen membayar bahan bakar dari saldo mereka sendiri tanpa memegang ETH. Agen dapat membayar layanan, satu sama lain, atau menghitung.
5. **Tata Kelola.** DAO Agen: struktur tata kelola tempat manusia *dan* agen memberikan suara pada perubahan protokol, dengan hak suara terkait dengan reputasi.

Tidak semua sistem produksi menggunakan kelima hal tersebut. Bittensor menggunakan 1, 2, sebagian 3, sebagian 4, tidak satu pun dari 5. Agen OpenAI tidak menggunakan apa pun kecuali 3. Tumpukan adalah peta referensi, bukan persyaratan.

### Bittensor, Fetch.ai, Gonka — apa yang dijalankan**Bittensor (TAO).** Subnet adalah tugas khusus (pemodelan bahasa, pembuatan gambar, perkiraan). Penambang mengirimkan output model. Validator memberi peringkat pada mereka; penilaian tertimbang taruhan mendistribusikan hadiah TAO. Setiap subnet memiliki evaluasinya sendiri. Lesson ekonominya: membayar untuk kualitas output tugas tertentu, bukan menghitung yang digunakan.

**Fetch.ai / ASI Alliance.** ASI-1 Mini LLM berjalan di jaringan Fetch.ai; pengguna membayar token FET untuk inference. Narasi agen sebagai rekan lebih kuat di sini: agen di Fetch dapat memanggil agen lain untuk suatu tugas dan membayar dalam FET.

**Gonka.** Bukti kerja Transformer: "pekerjaan" adalah lintasan maju dari sebuah Transformer. Penambang memperoleh penghasilan dengan menjalankan tugas inference yang telah mengetahui output yang benar (dari training data). PoW yang produktif sumber daya, bukan PoW berbasis hash.

Ketiganya berada pada tingkat produksi per April 2026. Distribusi pembayarannya berbeda. Bittensor menghargai kualitas dibandingkan dengan validator subnet; Ambil utilitas hadiah yang diukur berdasarkan pengguna yang membayar; Gonka menghargai pekerjaan inference yang dapat diverifikasi.

### Atribusi kredit nilai Shapley

Tiga agen berkolaborasi dalam sebuah tugas. Outputnya mendapat skor 0,8. Siapa yang menyumbang apa?

Nilai Shapley: alokasi kredit unik yang memenuhi empat aksioma (efisiensi, simetri, linearitas, nol). Untuk agen `i`:

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

dimana `S_i_O` adalah himpunan agen sebelum `i` dalam pemesanan `O`. Dalam praktiknya: hitung semua permutasi, catat kontribusi marjinal masing-masing agen dalam setiap permutasi, rata-rata.

Untuk N=3 agen, terdapat 6 permutasi. Untuk N=10, 3,6M — jadi dalam praktiknya kamu mengambil sample pesanan daripada menghitung.

### Lelang harga kedua untuk agregasi

Google Research ("Desain mekanisme untuk large language model") mengusulkan lelang token harga kedua untuk menggabungkan output LLM. Penyiapan: N agen masing-masing mengusulkan penyelesaian; masing-masing memiliki nilai pribadi untuk dipilih. Juru lelang memilih proposal dengan nilai tertinggi dan membayar nilai *tertinggi kedua*. Dalam agregasi monoton (nilai bergantung pada proposal mana yang dipilih, bukan berapa banyak yang mengajukan penawaran), hal ini memang benar — agen menawar nilai sebenarnya.

Mengapa hal ini penting bagi sistem LLM: kamu dapat melakukan outsourcing tugas penyelesaian ke beberapa agen dengan harga berbeda; lelang memilih yang terbaik + membayar secara adil, dan agen tidak mempunyai insentif untuk salah melaporkan.

### Modal reputasi

Skor reputasi yang terikat DID terakumulasi dari kontribusi yang dikonfirmasi. Aturan pembaruan sederhana:

```
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

Dengan faktor peluruhan `alpha` mendekati 1. Reputasi:

- Murah dibaca untuk keputusan perutean ("mengirim tugas sulit ke agen bereputasi tinggi").
- Mahal untuk dipalsukan (terakumulasi seiring waktu, terikat pada DID).
- Dapat dipotong: kontribusi yang gagal verifikasi dikurangi.

### AAMAS 2025 LaMAS terdesentralisasi

Proposal LaMAS (AAMAS 2025) menggabungkan: identitas DID, atribusi kredit nilai Shapley, dan mekanisme lelang sederhana. Klaim utamanya: desentralisasi langkah atribusi kredit membuat sistem dapat diaudit dan kebal terhadap manipulasi satu poin.

### Dimana perekonomian berantakan- **Manipulasi harga oracle.** Jika fungsi kredit dapat dipermainkan, agen akan mempermainkannya. Setiap mekanisme membutuhkan ujian permusuhan.
- **Serangan Sybil.** Salah satu operator membuat N agen palsu untuk meningkatkan kontribusi mereka sendiri. DID lambat tapi jangan hentikan ini; biaya untuk menempa reputasi adalah mitigasinya.
- **Biaya verifikasi.** Atribusi kredit hanya berlaku adil bagi verifikator. Kalau verifikasinya murah (LLM kecil), bisa dimainkan; jika mahal (panel manusia), sistem tidak akan berskala.
- **Peraturan yang berlebihan.** Ekonomi agen bersinggungan dengan peraturan keuangan. Bittensor, Fetch, dan Gonka semuanya beroperasi di wilayah abu-abu hukum di beberapa yurisdiksi pada tahun 2026.

### Ketika ekonomi agen masuk akal

- **Jaringan terbuka dengan operator heterogen.** Tidak ada satu tim pun yang mengontrol semua agen.
- **Output yang dapat diverifikasi.** Tanpa verifikasi, atribusi kredit hanyalah tebakan.
- **Alur kerja jangka panjang.** Tugas sekali pakai tidak mendapatkan keuntungan dari akumulasi reputasi.
- **Pembayaran yang diberi token layak secara hukum** di wilayah hukum kamu.

Dalam sistem perusahaan tertutup, ilmu ekonomi memberi jalan pada alokasi yang lebih sederhana (manajer menugaskan pekerjaan, metrik bersifat internal). Literatur ekonomi sebagian besar berlaku untuk jaringan terbuka.

## Build

`code/main.py` mengimplementasikan:

- `shapley(value_fn, agents)` — perhitungan Shapley yang tepat dengan enumerasi untuk N kecil.
- `second_price_auction(bids)` — mekanisme yang jujur; pemenang membayar tertinggi kedua.
- `Reputation` — Reputasi yang terikat dengan DID dengan penurunan dan penurunan eksponensial.
- Demo 1: tiga agen berkolaborasi, kredit atribut Shapley yang tepat.
- Demo 2: lima agen menawar slot tugas; pemenang pilihan lelang harga kedua + pembayaran.
- Demo 3: 100 putaran penugasan tugas kepada agen dengan perwakilan heterogen; perutean tertimbang rep mengalahkan secara acak.

Jalankan:

```
python3 code/main.py
```

Output yang diharapkan: Nilai Shapley untuk setiap agen; hasil lelang menunjukkan keseimbangan penawaran yang sebenarnya; perutean tertimbang rep menunjukkan peningkatan kualitas 10-20% dibandingkan perutean acak setelah pemanasan.

## Pakai

`outputs/skill-economy-designer.md` merancang ekonomi agen minimal: pilihan layer identitas, mekanisme atribusi kredit, mekanisme pembayaran, aturan reputasi.

## Kirim

Menjalankan ekonomi agen pada tahun 2026:

- **Mulailah dengan reputasi, bukan token.** Reputasi itu murah untuk diterapkan dan hanya bernilai; token menambah kompleksitas hukum dan ekonomi.
- **Verifikasi sebelum kamu memberi hadiah.** Jangan pernah mendistribusikan kredit tanpa langkah verifikasi independen. Kualitas yang dilaporkan sendiri menghasilkan permainan sybil.
- **Sample Shapley, bukan persis Shapley.** Contoh pemesanan 100-1000; pencacahan yang tepat tidak berskala.
- **Batas faktor pembusukan dan reputasi dasar.** Pembusukan tanpa batas menghapus kontributor yang sah; pembusukan yang terlalu lambat akan memberikan imbalan bagi agen dengan reputasi tinggi yang sudah basi.
- **Audit mekanisme secara berlawanan.** Jalankan skenario tim merah sebelum membuka jaringan. Setiap mekanisme memiliki teori permainan; kamu ingin menemukan lubangnya, bukan penyerangnya.

## Latihan1. Jalankan `code/main.py`. Konfirmasikan jumlah nilai Shapley ke nilai total (aksioma efisiensi). Ubah fungsi nilai; apakah alokasi Shapley berubah ke arah yang diharapkan?
2. Menerapkan *sampling* Shapley (pemesanan Monte Carlo atas K). Bagaimana K mempengaruhi akurasi perkiraan? Bandingkan dengan eksak untuk N=4.
3. Menerapkan langkah pembentukan koalisi sebelum lelang: agen dapat bergabung menjadi beberapa tim dan mengajukan penawaran sebagai satu unit. Koalisi apa yang terbentuk? Apakah hasil Pareto lebih baik dibandingkan penawaran individual?
4. Baca postingan desain mekanisme Riset Google. Identifikasi satu asumsi yang, jika dilanggar, akan merusak kebenaran. Seperti apa mode kegagalan dalam pengaturan LLM?
5. Baca makalah LaMAS terdesentralisasi AAMAS 2025. Menerapkan langkah Shapley mereka pada 10 agen pada tugas sintetis. Berapa lama waktu yang dibutuhkan untuk menghitung secara pasti? Seberapa dekat pengambilan sample dengan 100 kali seri?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| DePIN | "Infrastruktur fisik terdesentralisasi" | Komputasi/penyimpanan/bandwidth dengan insentif token. Bittensor, Akash, Render. |
| MELAKUKAN | "Pengidentifikasi terdesentralisasi" | Spesifikasi W3C untuk ID portabel. Reputasi agen terikat pada DID, bukan pada platform. |
| ERC-4337 | "Abstraksi akun" | Kontrak akun yang dapat mensponsori gas, memungkinkan pembayaran agen. |
| Nilai Shapley | "Atribusi kredit yang adil" | Alokasi unik yang memenuhi efisiensi, simetri, linearitas, nol. |
| Lelang harga kedua | "Lelang Vickrey" | Mekanisme yang jujur: pemenang membayar tawaran tertinggi kedua. Kompatibel dengan agregasi monoton. |
| Modal reputasi | "Akumulasi skor kualitas" | Skor terikat DID dari kontribusi yang dikonfirmasi; membusuk seiring berjalannya waktu. |
| DAO Agen | "Agen + manusia memerintah" | DAO dengan pemilih agen sebagai kelas satu, hak suara terkait dengan reputasi. |
| Kredit TAO / FET / GPU | "Denominasi token" | Bittensor TAO, Fetch.ai FET, berbagai token DePIN. |

## Bacaan Lanjutan

- [Ekonomi Agen](https://arxiv.org/abs/2602.14219) — survei tumpukan ekonomi agen 5 lapis tahun 2026
- [Google Research — Desain mekanisme untuk large language model](https://research.google/blog/mechanism-design-for-large-lingual-models/) — lelang token dengan agregasi monoton
- [AAMAS 2025 — LaMAS terdesentralisasi](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — Atribusi kredit nilai Shapley
- [Dokumentasi Bittensor TAO](https://docs.bittensor.com/) — struktur subnet dan distribusi hadiah
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini LLM dan token FET
- [Spesifikasi Pengidentifikasi Terdesentralisasi (DID) W3C](https://www.w3.org/TR/did-core/) — landasan identitas
