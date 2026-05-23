# Pola Alur Kerja Antropis: Sederhana Di Atas Kompleks

> Schluntz dan Zhang (Anthropic, Des 2024) membedakan alur kerja (jalur yang telah ditentukan sebelumnya) dari agen (penggunaan alat dinamis). Lima pola alur kerja mencakup sebagian besar kasus. Mulailah dengan panggilan API langsung. Tambahkan agen hanya ketika langkah-langkahnya tidak dapat diprediksi.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 01 (Agent Loop)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Sebutkan lima pola alur kerja Anthropic: rangkaian cepat, perutean, paralelisasi, pekerja orkestra, optimizer-evaluator.
- Jelaskan perbedaan agen-vs-alur kerja dan biaya teknis masing-masing.
- Identifikasi kapan harus memilih alur kerja daripada agen (dan sebaliknya).
- Menerapkan kelima pola di stdlib terhadap LLM skrip.

## Masalah

Tim menggunakan framework multi-agen untuk masalah yang memerlukan pemanggilan fungsi tunggal. Kerugiannya nyata: framework menambahkan layer yang mengaburkan prompt, menyembunyikan aliran kontrol, dan mengundang kompleksitas prematur. Postingan Schluntz dan Zhang pada bulan Desember 2024 adalah penolakan industri yang paling banyak dikutip: mulai dari yang sederhana, tambahkan kerumitan hanya jika hal tersebut menghasilkan loss.

## Konsep

### Alur kerja vs agen

- **Alur Kerja.** LLM dan alat diatur melalui jalur code yang telah ditentukan sebelumnya. Insinyur memiliki grafiknya.
- **Agen.** LLM secara dinamis mengarahkan alatnya sendiri dan mengambil langkahnya sendiri. Model memiliki grafik.

Keduanya mempunyai tempatnya masing-masing. Alur kerja lebih murah, lebih cepat, dan lebih mudah untuk di-debug. Agen membuka masalah terbuka namun membuat mode kegagalan lebih sulit untuk dipikirkan.

### LLM yang ditambah

Landasan untuk kelima pola: satu LLM dengan tiga kemampuan yang terhubung — pencarian (pengambilan), alat (tindakan), memori (kegigihan). Panggilan API apa pun dapat menggunakan ini.

### Lima pola

1. **Rantaian cepat.** Output panggilan 1 adalah input ke panggilan 2. Gunakan ketika tugas memiliki decomposition linier yang bersih. Gerbang terprogram opsional antar langkah.

2. **Perutean.** LLM pengklasifikasi memilih LLM hilir atau alat mana yang akan dipanggil. Gunakan ketika input yang berbeda secara kategori memerlukan penanganan yang berbeda (dukungan tingkat 1 vs pengembalian dana vs bug vs penjualan).

3. **Paralelisasi.** Jalankan N panggilan LLM secara bersamaan, hasil agregat. Dua bentuk: pembagian (bagian berbeda) dan pemungutan suara (prompt yang sama, N proses, mayoritas/sintesis).

4. **Pekerja orkestrator.** LLM orkestrator secara dinamis memutuskan pekerja mana (juga LLM) yang akan dijalankan dan mensintesis hasilnya. Mirip dengan perulangan agen tetapi orkestrator tidak melakukan perulangan tanpa batas.

5. **Evaluator-optimizer.** Satu LLM mengusulkan jawaban, LLM lain mengevaluasinya. Ulangi sampai evaluator lolos. Ini adalah Penyempurnaan Diri (Lesson 05) yang digeneralisasikan.

### Dimana alur kerja mengalahkan agen

- **Tugas yang dapat diprediksi.** Jika kamu dapat menyebutkan langkah-langkahnya, kamu harus melakukannya.
- **Tugas yang terikat biaya.** Alur kerja memiliki jumlah langkah yang dibatasi; agen bisa berputar.
- **Tugas yang terikat dengan kepatuhan.** Auditor ingin membaca grafik, bukan menyimpulkannya dari lintasan.

### Saat agen mengalahkan alur kerja

- **Penelitian terbuka.** Kapan langkah berikutnya bergantung pada hasil langkah terakhir.
- **Tugas dengan durasi yang bervariasi.** Menit hingga jam kerja yang jumlah langkahnya tidak diketahui.
- **Domain baru.** Jika kamu belum mengetahui alur kerja yang tepat — eksplorasi dulu, kodifikasi nanti.

### Pendamping rekayasa konteks"Rekayasa konteks yang efektif untuk agen AI" (Anthropic 2025) memformalkan disiplin yang berdekatan: jendela 200 ribu adalah anggaran, bukan wadah. Apa yang harus disertakan, kapan harus dipadatkan, kapan harus membiarkan konteksnya berkembang. Dicakup secara rinci dalam lesson Fase 14 tentang kompresi konteks (Fase 14 lesson sebelumnya 06 dalam kurikulum ini sebelum penomoran ulang).

## Build

`code/main.py` mengimplementasikan kelima pola alur kerja terhadap `ScriptedLLM`:

- `prompt_chain(input, steps)` — berurutan.
- `route(input, classifier, handlers)` — klasifikasi + pengiriman.
- `parallel_vote(prompt, n, aggregator)` — N berjalan, agregat.
- `orchestrator_workers(task, workers)` — orkestra memilih pekerja.
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — loop sampai lulus.

Jalankan:

```
python3 code/main.py
```

Setiap pola mencetak jejaknya sendiri. Total baris code per pola adalah ~10-15; biaya framework diukur dalam ribuan.

## Pakai

- Panggilan API langsung untuk sebagian besar tugas.
- Framework hanya jika polanya benar-benar membutuhkan status tahan lama (LangGraph), konkurensi model aktor (AutoGen v0.4), atau templat peran (CrewAI).
- Gunakan SDK Agen Claude jika kamu menginginkan bentuk harness Code Claude tanpa membangunnya kembali.

## Kirim

`outputs/skill-workflow-picker.md` memilih pola yang tepat untuk deskripsi tugas tertentu, termasuk alasan keputusan dan jalur pemfaktoran ulang ke agen jika alur kerja gagal.

## Latihan

1. Menerapkan perutean dengan ambang batas keyakinan. Di bawah ambang batas -> eskalasi ke manusia. Di manakah titik ambang batas untuk kasus penggunaan dukungan tingkat-1?
2. Tambahkan batas waktu ke `parallel_vote`. Apa yang terjadi jika satu panggilan terputus? Bagaimana cara kamu mengumpulkan suara yang hilang?
3. Ubah `evaluator_optimizer` menjadi bandit: pertahankan 2 output teratas di seluruh iterasi sehingga hasil baik yang terlambat tidak ditimpa oleh hasil buruk yang terlambat.
4. Gabungkan rangkaian cepat dengan perutean: router memilih salah satu dari tiga rangkaian. Ukur biaya token vs satu alternatif cepat.
5. Pilih salah satu feature produksi kamu. Gambarlah grafik alur kerja. Hitung langkah. Akankah agen lebih baik di sini?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Alur Kerja | "Aliran yang telah ditentukan sebelumnya" | Grafik LLM dan panggilan alat milik insinyur |
| Agen | "AI Otonom" | Grafik milik model; arah alat dinamis |
| LLM Ditambah | "LLM dengan alat" | LLM + pencarian + alat + memori; satuan atom |
| Rangkaian cepat | "Panggilan berurutan" | Output dari panggilan N adalah input untuk memanggil N+1 |
| Perutean | "Pengiriman pengklasifikasi" | Pilih rantai/model mana yang menangani input |
| Paralelisasi | "Menyebar" | N panggilan bersamaan; agregat dengan membagi atau memilih |
| Pekerja orkestra | "Agen pengirim" | Orchestrator LLM memilih LLM spesialis secara dinamis |
| Optimizer-evaluator | "Pengusul + hakim" | Ulangi hingga evaluator lulus; Penyempurnaan Diri yang digeneralisasikan |

## Bacaan Lanjutan

- [Anthropic, Building Effective Agents (Des 2024)](https://www.anthropic.com/research/building- Effective-agents) — lima pola alur kerja
- [Antropik, Rekayasa konteks yang efektif untuk agen AI](https://www.anthropic.com/engineering/ Effective-context-engineering-for-ai-agents) — disiplin pendamping
- [Ikhtisar LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — ketika grafik berstatus menghasilkan biayanya
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — pola pekerja orkestra, diproduksi
