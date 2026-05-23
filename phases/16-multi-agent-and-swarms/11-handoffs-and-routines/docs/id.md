# Handoff dan Rutinitas — Orkestrasi Tanpa Kewarganegaraan

> Swarm OpenAI (Oktober 2024) menyaring orkestrasi multi-agen menjadi dua primitif: **rutinitas** (instruksi + alat sebagai system prompt) dan **handoff** (alat yang mengembalikan Agen lain). Tidak ada mesin negara, tidak ada DSL percabangan — LLM merutekan dengan memanggil alat handoff kanan. OpenAI Agents SDK (Maret 2025) adalah penerus produksi. Swarm sendiri tetap menjadi referensi konseptual yang paling bersih — seluruh sumbernya muat dalam beberapa ratus baris. Pola ini viral karena permukaan API secara kasar adalah "agent = prompt + alat; handoff = agen yang mengembalikan fungsi". Batasan: tanpa kewarganegaraan, jadi memori adalah masalah penelepon.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 16 · 04 (Model Primitif)
**Waktu:** ~60 menit

## Masalah

Setiap framework multi-agen ingin kamu mempelajari DSL-nya: node dan edge LangGraph, kru dan tugas CrewAI, AutoGen GroupChat dan manajer. DSL adalah abstraksi nyata, namun membuat segalanya terasa lebih berat dari yang seharusnya.

Swarm mendorong ke arah yang berlawanan: gunakan kemampuan pemanggilan alat yang sudah dimiliki model. Handoff menjadi panggilan alat. Orkestra adalah agen mana pun yang sedang mengadakan percakapan. Mesin negara tersirat dalam system prompt agen.

## Konsep

### Dua primitif

**Rutin.** System prompt yang menentukan peran agen dan alat yang tersedia. Anggap saja seperti serangkaian instruksi terbatas: "kamu adalah agen triase; jika pengguna bertanya tentang pengembalian dana, serahkan ke agen pengembalian dana."

**Handoff.** Alat yang dapat dipanggil agen yang mengembalikan objek Agen baru. Runtime Swarm mendeteksi nilai pengembalian Agen dan mengalihkan agen aktif untuk giliran berikutnya.

Itulah keseluruhan abstraksinya.

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

System prompt agen triase membuatnya memilih handoff yang tepat berdasarkan pesan pengguna. Pemanggilan alat LLM melakukan perutean.

### Kenapa bisa viral

- **API Kecil.** Dua konsep untuk dipelajari.
- **Menggunakan apa yang sudah dilakukan model.** Pemanggilan alat sudah berada pada tingkat produksi di seluruh penyedia.
- **Tidak ada weight mesin negara.** kamu tidak mendeskripsikan grafiknya; prompt para agen menggambarkan kepada siapa mereka menyerahkannya.

### Perdagangan tanpa kewarganegaraan

Swarm secara eksplisit tidak memiliki kewarganegaraan di antara proses yang dijalankan. Kerangka kerja ini menyimpan riwayat pesan selama dijalankan, namun tidak menyimpan apa pun. Memori, kontinuitas, tugas-tugas yang berjalan lama — semua masalah penelepon.

Dalam produksi (OpenAI Agents SDK, Maret 2025) ini adalah salah satu hal utama yang berubah: SDK menambahkan manajemen sesi bawaan, pagar pembatas, dan penelusuran sambil menjaga handoff tetap primitif.

### Saat Swarm/handoff cocok

- **Pola triase.** Agen garis depan mengarahkan pengguna ke spesialis.
- **Serah terima berbasis keterampilan.** "Jika tugas memerlukan code, hubungi pembuat code; jika memerlukan penelitian, hubungi peneliti."
- **Percakapan singkat dan terbatas.** Dukungan pelanggan, FAQ hingga tiket, alur kerja sederhana.

### Saat Swarm kesulitan

- **Sesi panjang dengan memori bersama.** Handoff menyetel ulang status percakapan ke prompt plus riwayat agen baru. Tidak ada status persisten di seluruh agen tanpa memori yang dikelola pemanggil.
- **Eksekusi paralel.** Handoff dilakukan satu per satu — agen aktif berpindah. Paralelisme mengharuskan penelepon mengatur beberapa proses Swarm.
- **Audit dan putar ulang.** Proses tanpa status sulit untuk diputar ulang dengan tepat; pilihan handoff LLM tidak bersifat deterministik.

### OpenAI Agents SDK (Maret 2025)

Penerus produksi menambahkan:- **Keadaan sesi.** Thread persisten di seluruh proses.
- **Pagar Pembatas.** Kait validasi input/output.
- **Pelacakan.** Setiap panggilan alat dan handoff dicatat.
- **Filter handoff.** Mengontrol konteks yang ditransfer saat handoff.

Primitif handoff bertahan; ergonomi produksi ditambahkan di sekitarnya.

### Kawanan vs Obrolan Grup

Keduanya menggunakan perutean berbasis LLM, namun berbeda dalam **siapa yang memilih berikutnya**:

- GroupChat: pemilih (fungsi atau LLM) memilih pembicara berikutnya dari luar.
- Swarm: agen saat ini memilih penggantinya dengan memanggil alat handoff.

Swarm adalah "agen memutuskan apa yang selanjutnya"; GroupChat adalah "manajer memutuskan apa yang akan dilakukan selanjutnya". Keputusan Swarm bergantung pada panggilan alat agen aktif; Kehidupan GroupChat di `GroupChatManager`.

## Build

`code/main.py` mengimplementasikan Swarm dari awal: kelas data Agen, mekanisme handoff (alat mengembalikan Agen), dan run loop yang mendeteksi peralihan agen.

Demo: agen triase mengarahkan ke spesialis pengembalian dana, penjualan, atau dukungan. Setiap spesialis memiliki alatnya sendiri. Run loop mencetak setiap handoff.

Jalankan:

```
python3 code/main.py
```

## Pakai

`outputs/skill-handoff-designer.md` merancang topologi handoff untuk tugas tertentu: agen mana yang ada, handoff mana yang dapat mereka panggil, konteks apa yang ditransfer.

## Kirim

Daftar periksa:

- **Pencatatan log handoff.** Setiap handoff menulis peristiwa pelacakan dengan cuplikan konteks dari-agen, ke-agen.
- **Aturan transfer konteks.** Putuskan apa yang terjadi saat handoff: riwayat lengkap (mahal), N pesan terakhir, atau ringkasan.
- **Pagar pembatas saat handoff.** Handoff ke spesialis dengan izin alat yang berbeda harus diautentikasi — jika tidak, injeksi yang cepat dapat memaksa handoff yang tidak diinginkan.
- **Deteksi loop.** Dua agen yang saling menyerahkan adalah kegagalan umum; deteksi dengan pemeriksaan cincin K terakhir yang sederhana.
- **Agen cadangan.** Jika target handoff tidak ada, kembali ke default yang aman.

## Latihan

1. Jalankan `code/main.py`, lakukan triase ke agen pengembalian dana. Konfirmasikan pengembalian dana pada agen aktif putaran kedua.
2. Tambahkan aturan deteksi loop: jika dua agen yang sama telah menyerahkan 3 kali berturut-turut, paksakan keluar. Rancang cadangannya.
3. Baca dokumen OpenAI Agents SDK tentang filter handoff. Terapkan versi "ringkas saat serah terima": agen keluar memampatkan konteks menjadi ringkasan poin sebelum agen masuk mengambil alih.
4. Bandingkan handoff Swarm dengan pemilih GroupChatManager. Pola manakah yang membuat injeksi cepat menjadi lebih buruk, dan mengapa?
5. Baca buku masak Swarm (https://developers.openai.com/cookbook/examples/orchestating_agents). Identifikasi satu keputusan desain eksplisit yang dibuat Swarm agar OpenAI Agents SDK diubah atau dipertahankan.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Rutin | "Agen meminta" | System prompt + daftar alat. Mendefinisikan peran dan handoff yang tersedia. |
| Penyerahan | "Transfer ke agen lain" | Alat yang dapat dipanggil oleh agen aktif yang mengembalikan Agen baru. Runtime mengalihkan agen aktif. |
| Tanpa kewarganegaraan | "Tidak ada memori di antara proses" | Kawanan tidak menyimpan apa pun; memori adalah tanggung jawab penelepon. |
| Agen aktif | "Siapa yang berbicara sekarang" | Agen sedang mengadakan percakapan. Handoff mengubah ini. |
| Transfer konteks | "Apa yang bergerak saat serah terima" | Kebijakan mengenai riwayat yang dilihat agen masuk: penuh, N terakhir, atau ringkasan. |
| Lingkaran penyerahan | "Agen ping-pong" | Mode kegagalan di mana dua agen terus saling menyerahkan. |
| SDK Agen OpenAI | "Kawanan Produksi" | penerus Maret 2025; menambahkan sesi, pagar pembatas, penelusuran di atas primitif handoff. |
| Filter penyerahan | "Gerbang transfer" | Feature SDK untuk memeriksa dan mengubah konteks pada batas handoff. |

## Bacaan Lanjutan

- [Buku masak OpenAI — Agen Pengorganisasian: Rutinitas dan Serah Terima](https://developers.openai.com/cookbook/examples/orchestrating_agents) — artikulasi referensi
- [Repo OpenAI Swarm](https://github.com/openai/swarm) — implementasi asli, disimpan sebagai referensi konseptual
- [Dokumen SDK Agen OpenAI](https://openai.github.io/openai-agents-python/) — penerus produksi dengan sesi dan penelusuran
- [Catatan Anthropic handoff-in-Claude](https://docs.anthropic.com/en/docs/claude-code) — bagaimana subagen Claude Code menggunakan pola seperti handoff melalui `Task`
