# Arsitektur Paralel / Kawanan / Jaringan

> Berbeda dengan supervisor: tidak ada penentu pusat. Agen membaca bus acara bersama, mengambil pekerjaan secara asinkron, dan menulis kembali hasilnya. LangGraph secara eksplisit mendukung "Swarm Architecture" untuk lingkungan yang terdesentralisasi dan dinamis. Matrix (arXiv:2511.21686) mewakili kontrol dan aliran data sebagai pesan berseri yang melewati antrean terdistribusi untuk menghilangkan hambatan orkestrator. Pengorbanannya jelas: determinisme dan ketertelusuran untuk skalabilitas. Swarm menyesuaikan tugas dengan banyak sub-masalah independen; itu tidak sesuai dengan tugas-tugas yang memerlukan satu rencana yang koheren.

**Type:** Learn + Build
**Language:** Python (stdlib, `threading`, `queue`)
**Prerequisites:** Phase 16 · 05 (Pola Pengawas), Phase 16 · 04 (Model Primitif)
**Waktu:** ~75 menit

## Masalah

Supervisor menskalakan ke beberapa pekerja. Bagaimana dengan ratusan? Supervisor itu sendiri yang menjadi penghambat: setiap keputusan tentang siapa melakukan apa disalurkan melalui satu agen. Satu langkah rencana yang lambat akan menghentikan keseluruhan sistem.

Arsitektur gerombolan membalikkan desain. Alih-alih perencana pusat mengirimkan pekerjaan, pekerja memilih pekerjaan dari antrean bersama. "Koordinasi" dimasukkan ke dalam semantik bus acara. Tidak ada orkestra; sistem menskalakan hingga antrian mencapainya.

## Konsep

### Bentuknya

```
                ┌──── shared queue ────┐
                │                      │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Worker  Worker  Worker   Worker
      A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            results pool
```

Tidak ada orkestra. Setiap pekerja mengulangi: menarik tugas, memproses, menulis hasil (dan secara opsional mengantri tindak lanjut).

### Saat gerombolan cocok

- **Banyak tugas mandiri.** Mengikis, mengubah, mengklasifikasikan. Tugas tidak bergantung satu sama lain.
- **Pekerjaan berdurasi variabel.** Jika beberapa tugas membutuhkan waktu 100 md dan tugas lainnya membutuhkan waktu 10 detik, segerombolan orang akan menyeimbangkan weight secara otomatis — pekerja cepat akan melakukan pekerjaan berikutnya. Seorang supervisor harus mengantisipasi durasinya.
- **Throughput dibandingkan determinisme.** kamu lebih mementingkan waktu penyelesaian total, bukan pemesanan yang ketat.

### Ketika gerombolan gagal

- **Alur kerja yang dipesan.** Jika langkah 3 memerlukan output langkah 2, segerombolan orang berisiko menembaki langkah 3 sebelum langkah 2 selesai.
- **Tugas rencana global.** Pertanyaan penelitian yang kompleks mendapat manfaat dari perencana. Sekelompok peneliti menghasilkan fakta independen, bukan laporan yang koheren.
- **Debugging.** Tanpa log pusat dan pekerjaan asinkron, mereproduksi bug memerlukan biaya yang mahal.

### Matrix (arXiv:2511.21686)

Matrix adalah makalah tahun 2025 yang membawa gerombolan ke kesimpulan alaminya: aliran kontrol dan aliran data adalah pesan berseri pada antrian terdistribusi. Tidak ada koordinator pusat. Toleransi kesalahan berasal dari ketahanan pesan. Skalabilitas adalah masalah perantara pesan, bukan masalah sistem.

Kontribusi: model pemrograman di mana koordinasi multi-agen adalah "topik pesan apa yang diikuti oleh agen ini?" daripada "agen mana yang selanjutnya dipilih supervisor?" Hal ini membuat sistem terlihat seperti mesh acara pub/sub.

### Arsitektur Kawanan LangGraph

Dokumen LangGraph 2025 secara eksplisit mendeskripsikan "Arsitektur Swarm" sebagai salah satu pola multi-agen: agen adalah node, tetapi tepinya membentuk grafik terarah dengan siklus dan node mana pun dapat diaktifkan dari kumpulan. Seorang pekerja memilih pekerjaan yang tersedia berdasarkan kondisi, bukan berdasarkan penugasan supervisor.

### Mode kegagalan: kelaparan dan hot-spotting

Jika semua pekerja melakukan tugas tercepat yang tersedia, tugas yang sudah berjalan lama tidak akan diambil sampai hanya tugas tersebut yang tersisa. Kelaparan antrian klasik.

Mitigasi:
- Antrian prioritas dengan penuaan eksplisit (meningkatkan prioritas dengan waktu tunggu).
- Spesialisasi pekerja: beberapa pekerja hanya mengambil tugas yang "panjang".
- Tekanan balik: membatasi berapa banyak tugas cepat yang masuk antrian.### Tautan perutean berbasis konten

Swarm berpasangan secara alami dengan perutean berbasis konten (Lesson 22). Daripada menggunakan antrean umum, gunakan satu antrean per jenis pesan. Pekerja spesialis hanya berlangganan tipenya saja. Ini adalah dasar untuk arsitektur bus pesan yang dapat menjangkau ribuan agen.

## Build

`code/main.py` mengimplementasikan sekumpulan 4 thread pekerja yang diambil dari `queue.Queue` bersama. Tugas memiliki durasi yang bervariasi (ada yang cepat, ada yang lambat). Demo ini kontras:

- **Dasar berurutan:** satu pekerja memproses semua tugas secara serial.
- **Penugasan tetap:** setiap tugas telah ditetapkan sebelumnya kepada pekerja tertentu (gaya supervisor).
- **Swarm:** pekerja menarik dari antrean bersama.

Saldo gerombolan dimuat secara otomatis; penugasan tetap membuat pekerja cepat menganggur ketika tugas yang diberikan lambat.

Jalankan:

```
python3 code/main.py
```

Output menunjukkan jumlah tugas per pekerja (swarm mendistribusikan secara tidak merata namun optimal) dan waktu jam dinding.

## Pakai

`outputs/skill-swarm-fit.md` mengevaluasi apakah suatu tugas harus menggunakan gerombolan vs penyelia. Input: independensi tugas, varians durasi, persyaratan pemesanan, kebutuhan kemampuan debug.

## Kirim

Daftar periksa:

- **Antrian prioritas seiring bertambahnya usia.** Mencegah kelaparan tugas yang panjang.
- **Impotensi pekerja.** Sebuah tugas dapat ditarik lebih dari satu kali jika seorang pekerja mengalami error di tengah proses. Pekerja harus idempoten.
- **Antrian tahan lama.** Gunakan Kafka, Redis Streams, atau antrean yang didukung database untuk produksi. `queue.Queue` hanya ada di memori.
- **Kemampuan observasi per tugas.** Setiap tugas memiliki ID jejak; setiap pekerja mencatat awal/akhirnya.
- **Tekanan balik.** Jika antrean bertambah lebih cepat daripada yang dikuras oleh pekerja, maka akan memperlambat produsen.

## Latihan

1. Jalankan `code/main.py`. Seberapa cepat gerombolan daripada sekuensial pada weight kerja durasi variabel? Seberapa cepat dibandingkan penugasan tetap?
2. Tambahkan varian antrian prioritas (gunakan `queue.PriorityQueue`). Tetapkan prioritas berdasarkan bidang "pentingnya" tugas. Amati apakah tugas-tugas berprioritas rendah pernah mengalami kelaparan di bawah weight yang terus-menerus.
3. Menerapkan pendeteksi hot-spot: mencatat ketika ada pekerja yang memproses tugas 3× lebih banyak dibandingkan pekerja paling lambat. Apa yang dimaksud dengan distribusi durasi tugas?
4. Baca abstrak makalah Matrix (arXiv:2511.21686) dan Bagian 3. Identifikasi satu tradeoff spesifik yang diterima Matrix (perolehan skalabilitas) dan yang ditinggalkan (kemampuan penelusuran, determinisme).
5. Konversi demo gerombolan untuk menggunakan tupel `queue.Queue` (task_type, payload), dengan pekerja hanya berlangganan tipe tertentu. Aturan perutean apa yang masuk akal ketika tugas bersifat heterogen?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Arsitektur kawanan | "Agen terdesentralisasi" | Pekerja menarik dari antrian bersama; tidak ada orkestra pusat. |
| Bus acara | "Agen berlangganan topik" | Broker pesan yang merutekan tugas ke pekerja berdasarkan jenis atau konten. |
| Kelaparan | "Tugas tidak pernah berjalan" | Tugas dengan prioritas rendah tidak pernah dipilih karena pekerjaan dengan prioritas lebih tinggi datang terus menerus. |
| Titik panas | "Seorang pekerja tenggelam" | Ketidakseimbangan weight di mana satu pekerja mendapat sebagian besar tugas. |
| Tekanan balik | "Perlambat produser" | Mekanisme yang memberi sinyal ke hulu untuk berhenti berproduksi ketika antrian terisi. |
| Pekerja idempoten | "Aman untuk dijalankan kembali" | Tugas yang diproses dua kali menghasilkan hasil yang sama. Diperlukan karena pekerja mungkin mengalami crash di tengah proses. |
| Antrian tahan lama | "Bertahan dari kerusakan" | Antrean didukung oleh disk atau penyimpanan yang direplikasi; tugas tidak hilang saat pekerja mogok. |
| Kerangka matrix | "Kawanan penyampaian pesan penuh" | Aliran data dan kontrol merupakan pesan berseri pada antrian terdistribusi. |

## Bacaan Lanjutan

- [Alur kerja dan agen LangGraph — Arsitektur Swarm](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — dukungan gerombolan eksplisit
- [Matrix — Kerangka Terdesentralisasi untuk Sistem Multi-Agen](https://arxiv.org/abs/2511.21686) — gerombolan penyampaian pesan lengkap
- [Rekayasa antropik — mengapa supervisor tidak mengerumuni Penelitian](https://www.anthropic.com/engineering/multi-agent-research-system) — mengapa sistem produksi tertentu secara eksplisit memilih supervisor daripada kawanan
- [Dokumen model aktor AutoGen v0.4](https://microsoft.github.io/autogen/stable/) — penulisan ulang aktor yang digerakkan oleh peristiwa, lebih mirip gerombolan daripada GroupChat v0.2
