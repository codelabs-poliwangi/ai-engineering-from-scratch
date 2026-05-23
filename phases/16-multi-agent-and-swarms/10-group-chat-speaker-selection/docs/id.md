# Obrolan Grup dan Pemilihan Pembicara

> AutoGen GroupChat dan AG2 GroupChat berbagi satu percakapan di N agen; fungsi pemilih (LLM, round-robin, atau custom) memilih siapa yang berbicara selanjutnya. Ini adalah pola dasar percakapan multi-agen yang muncul — agen tidak mengetahui peran mereka dalam grafik statis, mereka hanya bereaksi terhadap kumpulan bersama. Semantik GroupChat AutoGen v0.2 dipertahankan di fork AG2; AutoGen v0.4 menulis ulangnya sebagai model aktor yang digerakkan oleh peristiwa. Microsoft memasukkan AutoGen ke mode pemeliharaan pada bulan Februari 2026 dan menggabungkannya dengan Kernel Semantik ke dalam Microsoft Agent Framework (RC Februari 2026). Primitif GroupChat bertahan di AG2 dan Microsoft Agent Framework — pelajari sekali, gunakan di mana saja.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 16 · 04 (Model Primitif)
**Waktu:** ~60 menit

## Masalah

Grafik statis (LangGraph) sangat bagus jika alur kerjanya diketahui. Percakapan nyata tidaklah statis: terkadang pembuat code bertanya kepada pengulas, terkadang peneliti, terkadang penulis. Melakukan hardcoding pada setiap handoff yang mungkin menghasilkan ledakan tepi. kamu ingin *agen bereaksi terhadap kumpulan bersama*, dengan beberapa fungsi memutuskan siapa yang berbicara selanjutnya.

Itulah tepatnya yang dilakukan AutoGen GroupChat.

## Konsep

### Bentuknya

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

Setiap agen melihat setiap pesan. Fungsi pemilih dipanggil di setiap giliran untuk memilih siapa yang berbicara selanjutnya.

### Tiga pemilih rasa

**Round-robin.** Siklus tetap. deterministik. Menskalakan secara linear dalam N tetapi mengabaikan konteks — pembuat code mendapat giliran bahkan ketika topiknya adalah tinjauan hukum.

**LLM dipilih.** Panggilan ke LLM yang membaca kumpulan terbaru dan menampilkan pembicara terbaik berikutnya. Sadar konteks tetapi lambat: setiap belokan menambahkan panggilan LLM. Default AutoGen.

**Kustom.** Fungsi Python dengan logika apa pun yang kamu inginkan. Khas: Dipilih LLM dengan aturan fallback (misalnya, "selalu berikan giliran kepada pemverifikasi setelah pembuat code").

### API Agen Percakapan

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` memegang pemilih. Ketika agen menyelesaikan giliran, manajer memanggil pemilih, yang mengembalikan agen berikutnya. Perulangan berlanjut hingga kondisi terminasi.

### Penghentian

Tiga pola umum:

- **Putaran maksimal.** Batas keras pada putaran total.
- **Token "TERMINATE".** Agen dapat mengirimkan pesan sentinel; manajer berhenti ketika muncul.
- **Pemeriksaan tercapainya tujuan.** Pemverifikasi ringan menjalankan setiap giliran dan menghentikan obrolan setelah selesai.

### Pemisahan AutoGen → AG2 dan penggabungan Microsoft Agent Framework

Pada awal tahun 2025, Microsoft memulai penulisan ulang besar-besaran AutoGen (v0.4) seputar model aktor berbasis peristiwa. Komunitas membagi semantik GroupChat AutoGen v0.2 menjadi AG2, mempertahankan API yang telah diintegrasikan oleh pengguna awal.

Pada bulan Februari 2026, Microsoft mengumumkan AutoGen akan masuk ke mode pemeliharaan, dengan model aktor berbasis peristiwa digabungkan ke dalam **Microsoft Agent Framework** (RC Februari 2026, kini digabungkan dengan Kernel Semantik). Konsep GroupChat bertahan di kedua jalur; detail implementasinya berbeda. AG2 adalah upstream pilihan untuk code yang kompatibel dengan v0.2.

### Saat GroupChat cocok

- **Percakapan darurat.** kamu tidak ingin melakukan pra-kabel pada setiap kemungkinan pembicara berikutnya.
- **Tugas pencampuran peran.** Pembuat code bertanya kepada peneliti, peneliti bertanya kepada petugas arsip, petugas arsip meminta balik kepada pembuat code. Aliran bukanlah DAG.
- **Pemecahan masalah yang bersifat eksplorasi.** Pikirkan "pertemuan curah pendapat", bukan "jalur pertemuan".

### Ketika gagal- **Determinisme yang ketat.** Pemilih LLM bisa jadi tidak konsisten. Prompt yang sama, proses yang berbeda, pembicara berikutnya yang berbeda.
- **Sycophancy Cascades.** Agen tunduk pada siapa pun yang berbicara paling percaya diri. Kontra-prompt secara eksplisit.
- **Konteks mengasapi.** Setiap agen membaca setiap pesan; setelah 10 putaran, konteksnya sangat besar. Gunakan proyeksi (Lesson 15) untuk mencakup tampilan.
- **Pembicara hebat.** Salah satu agen mendominasi percakapan karena pemilih menyukai spesialisasinya. Perkenalkan keseimbangan speaker sebagai feature pemilih.

### Obrolan grup vs penyelia

Primitif yang sama, default berbeda:

- Supervisor: satu agen merencanakan dan yang lainnya melaksanakan. Selector adalah "bertanya kepada perencana apa yang harus dilakukan."
- Obrolan grup: semua agen adalah rekan; pemilih adalah fungsi pada kumpulan bersama.

Keduanya menggunakan empat primitif dari Lesson 04. Obrolan grup defaultnya adalah orkestrasi yang dipilih LLM dan status bersama kumpulan penuh.

## Build

`code/main.py` mengimplementasikan GroupChat dari awal di stdlib. Tiga agen (coder, reviewer, manajer), varian round-robin dan LLM yang dipilih, dan penghentian pada token `TERMINATE`.

Demo tersebut mencetak transkrip percakapan ditambah jejak keputusan pemilih untuk kedua varian.

Jalankan:

```
python3 code/main.py
```

## Pakai

`outputs/skill-groupchat-selector.md` mengonfigurasi pemilih GroupChat untuk tugas tertentu — sistem round-robin vs pilihan LLM vs kustom, dan input pemilih apa (pesan terkini, spesialisasi agen, jumlah giliran) yang akan digunakan.

## Kirim

Daftar periksa:

- **Batas putaran maksimal.** Selalu. 10-20 untuk tugas-tugas biasa.
- **Metrik keseimbangan speaker.** Lacak putaran per agen; waspada ketika ketidakseimbangan melebihi ambang batas.
- **Token penghentian.** `TERMINATE` atau agen verifikator khusus.
- **Proyeksi atau memori terbatas.** Setelah ~10 pesan, pertimbangkan untuk memberikan setiap agen hanya tampilan terbatas untuk mencegah konteks membengkak.
- **Pencatatan log pemilih.** Untuk varian yang dipilih LLM, catat input pemilih dan pilihannya. Kalau tidak, debugging tidak mungkin dilakukan.

## Latihan

1. Jalankan `code/main.py`. Bandingkan percakapan dalam sistem round-robin vs yang dipilih LLM. Agen mana yang mendominasi masing-masing agen?
2. Tambahkan aturan "maks-bicara-per-agen" di pemilih. Bagaimana pengaruhnya terhadap transkrip?
3. Menerapkan penghentian yang telah mencapai tujuan: berhenti ketika pengulas mengembalikan "disetujui". Seberapa sering hal ini terpicu sebelum batas bulat?
4. Baca dokumen stabil AutoGen di GroupChat (https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html). Identifikasi pemilih default yang digunakan oleh `GroupChatManager`.
5. Baca repo AG2 (https://github.com/ag2ai/ag2) dan bandingkan GroupChat v0.2 dengan versi berbasis peristiwa v0.4. Properti konkret apa (throughput, toleransi kesalahan, komposisi) yang ditambahkan v0.4?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Obrolan Grup | "Agen dalam satu ruang obrolan" | Kumpulan pesan bersama + fungsi pemilih. AutoGen / AG2 primitif. |
| Pemilihan pembicara | "Siapa yang berbicara selanjutnya" | Fungsi yang memilih agen berikutnya. Round-robin, dipilih LLM, atau khusus. |
| Manajer Obrolan Grup | "Tuan rumah pertemuan" | Komponen AutoGen yang memiliki pemilih dan berputar secara bergantian. |
| Agen Percakapan | "Agen pangkalan" | kelas dasar AutoGen; agen yang dapat mengirim dan menerima pesan. |
| Token penghentian | "Kata 'berhenti'" | String penjaga (biasanya `TERMINATE`) yang mengakhiri obrolan. |
| Pembicara panas | "Satu agen mendominasi" | Mode kegagalan di mana pemilih terus memilih agen yang sama. |
| Konteks mengasapi | "Kolam tumbuh tanpa batas" | Setiap agen membaca setiap pesan sebelumnya; konteks tumbuh seiring dengan perubahan. |
| Proyeksi | "Tampilan tercakup" | Tampilan khusus peran ke dalam kumpulan bersama untuk mencegah pengasapan konteks. |

## Bacaan Lanjutan

- [dokumen obrolan grup AutoGen](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — implementasi referensi
- [Repo AG2](https://github.com/ag2ai/ag2) — kelanjutan komunitas AutoGen v0.2
- [Microsoft Agent Framework docs](https://microsoft.github.io/agent-framework/) — penerus gabungan, RC Februari 2026
- [Catatan rilis AutoGen v0.4](https://microsoft.github.io/autogen/stable/) — detail penulisan ulang model aktor yang digerakkan oleh peristiwa
