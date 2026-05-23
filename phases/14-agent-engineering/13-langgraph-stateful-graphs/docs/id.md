# LangGraph: Grafik Stateful dan Eksekusi Tahan Lama

> LangGraph adalah referensi tahun 2026 untuk orkestrasi stateful tingkat rendah. Agen adalah mesin negara; node adalah fungsi; ujung-ujungnya adalah transisi; keadaan tidak dapat diubah dan diperiksa setelah setiap langkah. Lanjutkan dari kegagalan apa pun tepat di titik terakhirnya.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 01 (Agent Loop), Fase 14 · 12 (Pola Alur Kerja)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Jelaskan model inti LangGraph: mesin status dengan status yang tidak dapat diubah, node fungsi, tepi bersyarat, dan pos pemeriksaan pasca-langkah.
- Sebutkan empat kemampuan yang disoroti dokumen: eksekusi yang tahan lama, streaming, human-in-the-loop, memori komprehensif.
- Jelaskan tiga topologi orkestrasi yang didukung LangGraph: supervisor, peer-to-peer (swarm), hierarki (subgraf bersarang).
- Mengimplementasikan grafik status stdlib dengan status yang tidak dapat diubah, tepi bersyarat, dan siklus pos pemeriksaan/lanjutkan.

## Masalah

Agen dan alur kerja memiliki masalah yang sama: ketika proses 40 langkah gagal pada langkah 38, kamu ingin melanjutkan dari langkah 38, bukan memulai dari awal. Model negara bagian kelas dua membiarkan operator meretas percobaan ulang di sekitar perpustakaan yang mengasumsikan proses baru.

Jawaban desain LangGraph: status adalah objek yang diketik kelas satu, mutasi bersifat eksplisit, dan pos pemeriksaan tetap ada setelah setiap node. Resume adalah panggilan `load_state(session_id)`.

## Konsep

### Grafik

Sebuah grafik ditentukan oleh:

- **Jenis status.** Dict yang diketik (atau model Pydantic) yang dibaca dan dimutasi oleh setiap node.
- **Node.** Fungsi murni `(state) -> state_update`. Pembaruan digabungkan ke dalam status setelah pengembalian.
- **Edges.** Transisi bersyarat atau langsung antar node.
- **Masuk dan keluar.** `START` dan `END` node sentinel menandai batasnya.

Contoh: agen dengan node `classify`, `refund`, `bug`, `sales`, `done` — alur kerja perutean sebagai grafik.

### Eksekusi yang tahan lama

Setelah setiap node kembali, runtime membuat serialisasi status dan menulisnya ke checkpointer (SQLite, Postgres, Redis, custom). Jika gagal pada langkah N, runtime dapat `resume(session_id)` dan melanjutkan dari langkah N+1 dengan status yang tepat.

Dokumen LangGraph secara eksplisit menyoroti pengguna produksi yang penting: Klarna, Uber, J.P. Morgan. Klaimnya bukanlah bentuk grafik; hanya saja bentuk grafik ditambah pos pemeriksaan membuat pemulihan menjadi murah.

### Streaming

Setiap node dapat menghasilkan output parsial. Grafik mengalirkan peristiwa per-node-delta ke pemanggil sehingga UI diperbarui saat grafik berjalan.

### Manusia dalam lingkaran

Periksa dan ubah status antar node. Implementasi: jeda sebelum node kritis, keadaan permukaan menjadi manusia, terima modifikasi, lanjutkan. Checkpointer memudahkan hal ini karena status sudah diserialkan.

### Memori

Jangka pendek (dalam jangka waktu — riwayat percakapan dalam status) dan jangka panjang (di seluruh proses — tetap melalui checkpointer ditambah penyimpanan jangka panjang yang terpisah). LangGraph terintegrasi dengan sistem memori eksternal (Mem0, custom) melalui alat.

### Tiga topologi

1. **Supervisor.** Pengiriman LLM router pusat ke subagen spesialis. `create_supervisor()` di `langgraph-supervisor` (meskipun tim LangChain pada tahun 2026 merekomendasikan melakukan ini melalui panggilan alat secara langsung untuk kontrol konteks yang lebih banyak).
2. **Swarm / peer-to-peer.** Agen menyerahkan secara langsung melalui permukaan alat bersama. Tidak ada router pusat.
3. **Hierarki.** Supervisor yang mengelola sub-supervisor, diterapkan sebagai subgraf bertingkat.### Dimana letak kesalahan pola ini

- **Pos pemeriksaan terlalu kecil.** Hanya percakapan di pos pemeriksaan yang membuat status alat dan penulisan memori tidak dapat dipulihkan. Status penuh harus dibuat bersambung.
- **Node non-deterministik.** Resume mengasumsikan input node menghasilkan pembaruan status yang sama. Benih acak, jam dinding, API eksternal harus ditangkap.
- **Penggunaan sisi bersyarat secara berlebihan.** Graf dengan setiap sisi bersyarat adalah mesin keadaan yang tidak dapat dipikirkan. Lebih suka rantai linier dengan cabang sesekali.

## Build

`code/main.py` mengimplementasikan grafik stateful stdlib:

- `State` — dikte yang diketik dengan `messages`, `step`, `route`, `output`, `human_approval`.
- `Node` — dapat dipanggil mengambil status dan mengembalikan dikt pembaruan.
- `StateGraph` — node + edge + edge bersyarat + run + resume.
- `SQLiteCheckpointer` (palsu dalam memori) — membuat status serial setelah setiap node; `load(session_id)` memulihkan.
- Grafik demo: klasifikasikan -> cabang (pengembalian dana / bug / penjualan) -> gerbang manusia -> kirim.

Jalankan:

```
python3 code/main.py
```

Jejaknya menunjukkan kegagalan proses pertama di gerbang manusia, ketekunan, lalu melanjutkan menghasilkan hasil akhir.

## Pakai

- **LangGraph** — referensi, siap produksi. Gunakan `create_react_agent`, `create_supervisor`, atau buat grafik kamu sendiri.
- **AutoGen v0.4** (Lesson 14) — alternatif model aktor untuk skenario konkurensi tinggi.
- **Claude Agent SDK** (Lesson 17) — harness terkelola dengan penyimpanan sesi bawaan.
- **Kustom** — saat kamu memerlukan kontrol yang tepat atas bentuk status atau backend checkpointer.

## Kirim

`outputs/skill-state-graph.md` menghasilkan grafik status berbentuk LangGraph di setiap target runtime dengan pos pemeriksaan dan resume yang tersambung.

## Latihan

1. Tambahkan tepi bersyarat dari `classify` ke `end` ketika keyakinan klasifikasi berada di bawah ambang batas. Lanjutkan proses setelah set manusia `route` secara manual.
2. Tukar yang palsu seperti SQLite dengan checkpointer SQLite yang asli. Ukur overhead serialisasi per langkah.
3. Menerapkan tepi paralel: dua node berjalan secara bersamaan, digabungkan dengan peredam khusus. Apa yang dibeli oleh negara yang tidak dapat diubah di sini?
4. Baca referensi `langgraph-supervisor`. Kirim mainan tersebut ke `create_supervisor`. Bandingkan bentuk jejaknya.
5. Tambahkan streaming: setiap node menghasilkan sebagian status saat berjalan. Cetak delta saat tiba.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Grafik keadaan | "Agen sebagai mesin negara" | Status yang diketik + node + tepi + reduksi |
| Pos pemeriksaan | "Backend kegigihan" | Membuat serial status setelah setiap node; mengaktifkan melanjutkan |
| Peredam | "Penggabungan negara" | Fungsi yang menggabungkan status saat ini dengan pembaruan node |
| Tepi bersyarat | "Cabang" | Tepian dipilih berdasarkan fungsi keadaan |
| Subgraf | "Grafik Bersarang" | Graf yang digunakan sebagai simpul di dalam graf lain |
| Eksekusi tahan lama | "Melanjutkan dari kegagalan" | Mulai ulang pada node terakhir yang berhasil dengan status |
| Pengawas | "Router LLM" | Dispatcher pusat untuk subagen spesialis |
| Kawanan | "Agen P2P" | Agen menyerahkan melalui alat bersama; tidak ada router pusat |

## Bacaan Lanjutan- [Ikhtisar LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — dokumen referensi
- [referensi langgraph-supervisor](https://reference.langchain.com/python/langgraph/supervisor/) — API pola supervisor
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — alternatif model aktor
- [Ikhtisar SDK Agen Claude](https://platform.claude.com/docs/en/agent-sdk/overview) — penyimpanan sesi dan subagen
