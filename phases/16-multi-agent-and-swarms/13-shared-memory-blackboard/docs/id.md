# Memori Bersama dan Pola Papan Tulis

> Ada dua pendekatan yang diterapkan secara berdampingan di sistem multi-agen pada tahun 2026: **kumpulan pesan** (semua orang melihat pesan semua orang, seperti di AutoGen GroupChat atau MetaGPT) dan **papan tulis dengan langganan** (agen berlangganan peristiwa yang relevan, seperti di MCP Sadar Konteks atau kerangka Matrix). Keduanya adalah satu-satunya bagian stateful dari sistem multi-agen — yang berarti keduanya adalah tempat tinggalnya bug yang menarik. Mode kegagalan referensi adalah **keracunan memori**: satu agen berhalusinasi tentang sebuah "fakta", agen lain memperlakukannya sebagai terverifikasi, dan akurasi menurun secara bertahap sehingga lebih sulit untuk di-debug daripada crash langsung. Lesson ini membangun kedua struktur dari stdlib, memasukkan serangan keracunan, dan menunjukkan tiga mitigasi yang benar-benar berfungsi dalam produksi.

**Type:** Learn + Build
**Language:** Python (stdlib, `threading`)
**Prerequisites:** Fase 16 · 04 (Model Primitif), Fase 16 · 09 (Jaringan Kawanan Paralel)
**Waktu:** ~75 menit

## Masalah

Sistem multi-agen memerlukan tempat bagi agen untuk berbagi fakta. Opsi literalnya adalah "meneruskan semuanya dalam pesan" — namun hal ini menciptakan kembali status bersama dengan penyalinan tambahan. Cara lainnya adalah dengan "memberi setiap orang log global" — namun log global tumbuh tanpa batas dan mudah meracuni. Yang ketiga adalah "memproyeksikan tampilan per agen" — dapat diskalakan tetapi memiliki banyak skema.

Ketika salah satu agen berhalusinasi dan menuliskan halusinasi tersebut ke keadaan bersama, setiap agen hilir yang membaca keadaan tersebut mengadopsi halusinasi tersebut sebagai fakta. Pada saat manusia menyadarinya, rantai penalaran sudah mencapai lima langkah dan akar permasalahannya adalah pesan ketiga yang pernah ditulis. Men-debug kerusakan akurasi multi-agen lebih sulit daripada men-debug kerusakan.

Ini adalah keracunan ingatan. Ini adalah kelompok kegagalan kedua yang paling banyak didokumentasikan dalam taksonomi MAST (Cemri et al., arXiv:2503.13657) dan bersifat struktural: setiap desain memori bersama tanpa sumber dan pemverifikasi yang tidak dapat ditulis pada akhirnya akan menunjukkannya.

## Konsep

### Dua topologi utama

**Kumpulan pesan lengkap.** Setiap agen membaca setiap pesan. AutoGen GroupChat dan MetaGPT menggunakan ini. Sederhana, transparan, dapat diperiksa, namun tidak melebihi ~10 agen karena konteks masing-masing agen dipenuhi dengan pekerjaan agen lainnya.

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**Papan tulis dengan langganan.** Agen menyatakan minat pada topik; media hanya merutekan pesan yang relevan. CA-MCP (arXiv:2601.11595) dan kerangka desentralisasi Matrix (arXiv:2511.21686) menggunakan ini. Berskala lebih jauh, namun memerlukan desain skema awal agar langganan menjadi bermakna.

```
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### Saat masing-masing menang

- **Kelompok penuh** menang jika jumlah agen sedikit (<10), heterogen, dan percakapan bersifat jangka pendek. Bernalar tentang siapa yang mengatakan apa adalah hal yang sepele ketika semua orang melihat segalanya.
- **Blackboard** menang jika jumlah agennya banyak, perannya homogen, tetapi jumlah instancenya banyak (swarm), dan percakapannya berlangsung lama. Perutean menghemat biaya token dan polusi konteks.

Sistem produksi sering kali bercampur: kumpulan kecil penuh di bagian atas (layer perencanaan), papan tulis di bawah (layer pekerja).

### Keracunan memori, dalam satu skenario

Tiga agen sedang mengerjakan tugas penelitian. Agen A adalah agen pengambilan. Agen B adalah peringkas. Agen C adalah seorang analis.1. A mengambil halaman dan menulis pesan ke status bersama: "Studi ini melaporkan peningkatan akurasi sebesar 42%."
2. Halaman yang diambil sebenarnya bertuliskan "peningkatan 4,2%". A berhalusinasi desimal.
3. B, membaca status bersama, menulis: "Peningkatan akurasi sebesar 42% dilaporkan (sumber: A)."
4. C, membaca status bersama, menulis: "Rekomendasikan adopsi — peningkatan 42% bersifat transformatif."
5. Laporan akhir menyebutkan angka 42% yang tidak pernah ada.

Tidak ada agen yang jatuh. Tidak ada tes yang gagal. Sistem "berhasil". Halusinasi tersebut berpindah dari konteks satu agen ke dalam alasan setiap agen hilir melalui keadaan bersama.

### Mengapa ini bersifat struktural

Tanpa keadaan bersama, halusinasi agen A tetap berada dalam konteks A. Agen hilir akan mengambil ulang atau mengambil ulang dan mungkin menangkap kesalahan tersebut. Dengan keadaan bersama yang naif, konteks A menjadi konteks semua orang, dan halusinasinya dicuci menjadi fakta.

Permasalahannya bukanlah keadaan yang terbagi-bagi — melainkan keadaan yang dibagi **tanpa sumber dan tanpa pemverifikasi independen**. Tiga mitigasi mengatasi hal ini:

1. **Atribusikan asal pada setiap penulisan.** Setiap entri dalam catatan negara bagian bersama siapa yang menulisnya, kapan, berdasarkan prompt apa, dan (jika berlaku) sumber apa yang dikutip oleh agen. Agen hilir membaca dengan skeptis terhadap asal muasalnya.
2. **Versi menulis; perlakukan entri tersebut sebagai tambahan saja.** Koreksi adalah entri baru yang menggantikan entri lama, bukan pembaruan yang sudah ada. Jejak audit dipertahankan.
3. **Simpan setidaknya satu agen yang tidak dapat menulis ke status bersama.** Agen pemverifikasi read-only mengambil sample entri, mengambil ulang sumber, dan menandai ketidakkonsistenan. Karena tidak bisa menulis ke kolam, tidak bisa diracuni oleh kolam.

### Preseden papan tulis (Hayes-Roth, 1985)

Pola papan tulis mendahului agen LLM selama empat dekade. Hayes-Roth (1985, "A Blackboard Architecture for Control") menggambarkan Sumber Pengetahuan spesialis yang mengamati papan tulis global, menyumbangkan solusi parsial, dan memicu sumber-sumber lain. Papan tulis 2026 (CA-MCP, Matrix) memiliki pola yang sama dengan agen LLM sebagai Sumber Pengetahuan dan gumpalan JSON sebagai solusi parsial. Literatur lama telah mendokumentasikan solusi untuk mengatasi perselisihan, kontrol oportunistik, dan konsistensi yang ditemukan kembali oleh sistem modern.

### Proyeksi vs tampilan penuh

Papan tulis murni memberikan proyeksi yang sama kepada setiap pelanggan (cakupan topik). Desain yang lebih agresif adalah **proyeksi per agen**: setiap agen mendapatkan tampilan yang disesuaikan dengan perannya. Pereduksi status LangGraph adalah implementasi kanonis tahun 2026 — fungsi peredam melipatgandakan status global menjadi potongan peran tertentu.

Proyeksi per agen berskala lebih jauh tetapi memerlukan skema. Tanpanya, kamu membangun kembali proyeksi ad-hoc sesuai permintaan setiap agen.

### Tulis pola pertentangan

Beberapa agen menulis secara bersamaan adalah masalah konkurensi, bukan hanya masalah LLM. Tiga pola berfungsi:

- **Penulis berurutan (produser tunggal).** Semua penulisan dilakukan melalui satu agen koordinator yang membuat serial. Sederhana, namun menjadi hambatan.
- **Konkurensi optimis dengan pembuatan versi.** Setiap entri memiliki versi; penulis gagal karena ketidakcocokan versi dan coba lagi. Teknik basis data klasik.
- **Pembagian topik.** Agen yang berbeda memiliki topik yang berbeda. Tidak ada perselisihan lintas topik. Membutuhkan batas partisi yang dirancang.

Sebagian besar framework tahun 2026 menggunakan penulis sekuensial secara default karena panggilan LLM cukup lambat sehingga jarang terjadi perselisihan dan kemacetan tidak merugikan.

### Pemverifikasi yang tidak dapat ditulisi

Mitigasi yang paling menahan weight adalah pemverifikasi hanya-baca. Aturan pelaksanaan:- Verifikator berbagi status dengan tim (membaca papan tulis atau pool).
- Pemverifikasi tidak memiliki pegangan tulis pada status bersama — hanya pada pipeline verifikasi terpisah.
- Verifikator secara independen mengambil sumber yang dikutip dalam penulisan. Menandai ketidaksepakatan.
- Output Verifikator disalurkan ke manusia atau agen pengambil keputusan terpisah, dan tidak pernah dimasukkan kembali ke dalam kumpulan.

Tanpa pemisahan ini, output pemverifikasi akan menjadi entri baru dalam pool, yang berarti pool yang terracun akan meracuni verifikator, sehingga meracuni verifikasinya.

## Build

`code/main.py` mengimplementasikan kedua topologi di stdlib Python ditambah serangan keracunan mainan dan tiga mitigasi.

- `MessagePool` — log tambahan thread-safe dengan pembacaan penuh.
- `Blackboard` — pub/sub topik dengan langganan per agen.
- `ProvenanceEntry` — setiap catatan penulisan (penulis, stempel waktu, prompt_hash, source_uri).
- `PoisoningScenario` — menjalankan tugas penelitian tiga agen di mana agen A berhalusinasi desimal. Mencetak laporan akhir.
- `Verifier` — agen hanya-baca yang mengambil ulang sumber dan menandai ketidakkonsistenan. Menjalankan skenario yang sama dengan kehadiran verifikator.

Jalankan:

```
python3 code/main.py
```

Hasil yang diharapkan:
- Jalankan 1 (tanpa verifikator): 42% yang berhalusinasi menyebar ke laporan akhir.
- Jalankan 2 (dengan verifikator): verifikator menandai ketidakkonsistenan, kumpulan diberi label "ditandai", laporan akhir menyertakan pencabutan.

## Pakai

`outputs/skill-memory-auditor.md` adalah keterampilan yang mengaudit desain memori bersama sistem multi-agen untuk mengetahui asal, pembuatan versi, dan pemisahan pemverifikasi. Jalankan pada arsitektur multi-agen baru sebelum produksi.

## Kirim

Untuk desain memori bersama apa pun:

- Catat asal pada setiap penulisan: `(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`.
- Jadikan log hanya tambahan. Koreksi adalah entri baru yang merujuk pada entri yang digantikan.
- Menyebarkan setidaknya satu agen pemverifikasi read-only dengan akses sumber independen.
- Merutekan output pemverifikasi ke pipeline terpisah, bukan kembali ke kumpulan bersama.
- Catat rasio penulisan yang merupakan supersesi — rasio yang meningkat adalah bukti awal pola halusinasi.

## Latihan

1. Jalankan `code/main.py`. Konfirmasikan proses 1 menyebarkan halusinasi dan proses 2 menangkapnya.
2. Tambahkan halusinasi kedua: agen B menemukan ukuran dataset. Verifikator harus menangkap keduanya tanpa harus mengatur keduanya.
3. Alihkan kumpulan penuh ke papan tulis dengan partisi topik (`prices`, `summaries`, `analyses`). Skenario peracunan mana yang membuat partisi topik menjadi lebih sulit untuk dilakukan, dan skenario mana yang tidak membantu?
4. Baca Hayes-Roth (1985, "Arsitektur Papan Tulis untuk Kontrol"). Identifikasi dua pola kontrol dari makalah yang tidak dibahas dalam lesson ini yang dapat dimanfaatkan oleh sistem tahun 2026.
5. Baca CA-MCP (arXiv:2601.11595). Petakan Penyimpanan Konteks Bersamanya ke kelas MessagePool atau Blackboard di `code/main.py`. Primitif manakah yang ditambahkan CA-MCP di atasnya?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Kumpulan pesan | "Riwayat obrolan bersama" | Log tambahan saja yang dibaca oleh setiap agen. Transparansi penuh, penskalaan buruk. |
| Papan Tulis | "Ruang kerja bersama" | Pub/sub dengan topik tertentu. Agen berlangganan topik yang relevan. Berskala lebih jauh. |
| Asal | "Siapa yang menulis apa" | Metadata pada setiap penulisan: penulis, stempel waktu, prompt, sumber. |
| Keracunan memori | "Halusinasi menyebar" | Kesalahan satu agen memasuki keadaan bersama, agen hilir mengadopsinya sebagai fakta. |
| Hanya tambahkan | "Tidak ada pembaruan di tempat" | Koreksi adalah entri baru yang menggantikan. Mempertahankan jejak audit. |
| Pemverifikasi yang tidak dapat ditulis | "Auditor independen" | Agen hanya baca yang mengambil ulang sumber dan menandai ketidakkonsistenan. |
| Proyeksi | "Tampilan tercakup" | Tampilan per agen dihitung dari status global. Pereduksi LangGraph adalah kasus kanonik. |
| Sumber Pengetahuan | "Agen Spesialis" | Istilah Hayes-Roth tahun 1985 untuk peserta papan tulis. |

## Bacaan Lanjutan

- [Cemri dkk. — Mengapa Sistem LLM Multi-Agen Gagal?](https://arxiv.org/abs/2503.13657) — Taksonomi MAST; keracunan memori adalah sub-keluarga kegagalan koordinasi
- [CA-MCP — MCP Multi-Server Kontekstual](https://arxiv.org/abs/2601.11595) — Penyimpanan Konteks Bersama untuk server MCP terkoordinasi
- [Matrix — kerangka multi-agen terdesentralisasi](https://arxiv.org/abs/2511.21686) — papan tulis berbasis antrean pesan tanpa orkestrator pusat
- [Status dan reduksi LangGraph](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — pola proyeksi per agen dalam produksi
- [Anthropic — Bagaimana kami membangun sistem penelitian multi-agen kami](https://www.anthropic.com/engineering/multi-agent-research-system) — catatan asal dan verifikasi dari penerapan produksi
