# ReWOO dan Plan-and-Execute: Perencanaan Terpisah

> ReAct menyisipkan pemikiran dan tindakan dalam satu aliran. ReWOO memisahkannya: satu rencana besar di awal, lalu jalankan. Token 5x lebih sedikit, akurasi +4% di HotpotQA, dan kamu dapat menyaring perencana menjadi model 7B. Plan-and-Execute menggeneralisasikannya; Plan-and-Act menskalakannya ke navigasi web.

**Type:** Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 14 · 01 (Agent Loop)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Jelaskan mengapa pemisahan Planner/Worker/Solver ReWOO menghemat token dan meningkatkan ketahanan pada loop interleaved ReAct.
- Mengimplementasikan rencana DAG, eksekutor yang diurutkan ketergantungan, dan pemecah yang menyusun output pekerja — semuanya stdlib.
- Putuskan kapan tugas harus dijalankan sebagai ReAct plan-then-execute vs interleaved, menggunakan framing "lima pola alur kerja" tahun 2026 (Antropis).
- Kenali kapan data rencana sintetik Plan-and-Act diperlukan untuk tugas web atau seluler jangka panjang.

## Masalah

Loop observasi-tindakan-pikiran yang disisipkan di ReAct sederhana dan fleksibel, namun setiap pemanggilan alat harus membawa konteks penuh sebelumnya — termasuk setiap pemikiran sebelumnya. Penggunaan token tumbuh secara kuadratik seiring dengan kedalaman. Lebih buruk lagi: ketika suatu alat gagal di tengah-tengah loop, model harus mendapatkan kembali seluruh rencana dari pengamatan kesalahan.

ReWOO (Xu dkk., arXiv:2305.18323, Mei 2023) memperhatikan hal ini dan bertaruh: rencanakan semuanya terlebih dahulu, ambil bukti secara paralel, susun jawabannya di akhir. Satu panggilan LLM untuk merencanakan, N alat memerlukan bukti (bisa paralel), satu panggilan LLM untuk menyelesaikan. Perdagangannya kurang fleksibel (rencananya statis) untuk efisiensi token yang jauh lebih baik dan mode kegagalan yang lebih jelas.

## Konsep

### Tiga peran

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Perencana menghasilkan DAG. Setiap node memberi nama alat, argumennya, dan node mana yang sebelumnya bergantung pada alat tersebut (referensi seperti `#E1`, `#E2`). Pekerja mengeksekusi node dalam urutan topologi. Solver menyatukan semuanya.

### Mengapa token 5x lebih sedikit

ReAct menambah panjang prompt secara linear dengan jumlah langkah. Pada langkah 10, prompt berisi pemikiran 1 ditambah tindakan 1 ditambah observasi 1 ditambah pemikiran 2 ditambah tindakan 2 ditambah observasi 2, dan seterusnya. Setiap langkah perantara juga menyertakan prompt asli secara berlebihan.

ReWOO membayar satu prompt perencana (besar), N prompt pekerja kecil (masing-masing hanya panggilan alat, tanpa rantai), dan satu prompt pemecah. Di HotpotQA, makalah ini mengukur ~5x lebih sedikit token sambil mencetak +4 akurasi absolut.

### Mengapa lebih kuat

Jika pekerja 3 gagal di ReAct, loop harus keluar dari kesalahan di tengah-tengah aliran. Di ReWOO, pekerja 3 mengembalikan string kesalahan; pemecah melihatnya dalam konteks dengan rencana awal dan dapat menurunkannya dengan baik. Lokalisasi kegagalan terjadi per node, bukan per langkah.

### Distilasi perencana

Hasil kedua dari makalah ini: karena perencana tidak melihat observasi, kamu dapat menyempurnakan model 7B pada output perencana dari guru 175B. Model kecil menangani perencanaan; model besar tidak diperlukan dalam inference. Ini sekarang menjadi standar — banyak agen produksi tahun 2026 menggunakan perencana kecil dan pelaksana besar atau sebaliknya.

### Rencanakan dan Jalankan (LangChain, 2023)

Postingan tim LangChain pada Agustus 2023 menggeneralisasi ReWOO menjadi nama pola: Plan-and-Execute. Perencana di muka mengeluarkan daftar langkah, pelaksana menjalankan setiap langkah, perencana ulang opsional dapat merevisi setelah mengamati hasilnya. Ini lebih mirip dengan ReAct daripada ReWOO (perencana ulang mengembalikan pengamatan ke dalam perencanaan) tetapi tetap mempertahankan penghematan token.### Rencana dan Tindakan (Erdogan et al., arXiv:2503.09572, ICML 2025)

Plan-and-Act menskalakan pola tersebut ke agen web dan seluler jangka panjang. Kontribusi utamanya adalah data rencana sintetik: generator lintasan berlabel menghasilkan training data yang rencananya eksplisit. Digunakan untuk menyempurnakan model perencana yang terus bekerja melewati 30–50 langkah pada tugas seperti WebArena di mana satu lintasan ReAct kehilangan koherensi.

### Kapan harus memilih yang mana

| Pola | Kapan |
|---------|------|
| Bereaksi | Tugas singkat, lingkungan tidak diketahui, memerlukan penanganan pengecualian reaktif |
| UlangWOO | Tugas terstruktur dengan alat yang dikenal, peka terhadap token, bukti yang dapat diparalelkan |
| Rencanakan dan Jalankan | Seperti ReWOO tetapi dengan perencanaan ulang setelah eksekusi sebagian |
| Rencana dan Tindakan | Cakrawala panjang (>30 langkah), penggunaan web/seluler/komputer |
| Pohon Pikiran | Pencarian layak dibayar (Lesson 04) |

Panduan Anthropic Desember 2024: mulailah dengan yang paling sederhana. Jika tugasnya adalah satu panggilan alat ditambah ringkasan, jangan buat ReWOO. Jika tugasnya adalah tugas penelitian 40 langkah, jangan lakukan ReAct sendirian.

## Build

`code/main.py` mengimplementasikan mainan ReWOO:

- `Planner` — kebijakan tertulis yang mengeluarkan rencana DAG dari prompt.
- `Worker` — mengirimkan panggilan alat setiap node melalui registri.
- `Solver` — komposisi tertulis yang membaca bukti dan menghasilkan jawaban akhir.
- Resolusi ketergantungan — referensi seperti `#E1` diganti dengan output pekerja sebelumnya.

Demo tersebut menjawab "Berapa jumlah penduduk ibu kota Perancis, jika dibulatkan menjadi jutaan?" menggunakan rencana dua langkah: (1) mencari modal, (2) mencari populasi, lalu menyelesaikannya.

Jalankan:

```
python3 code/main.py
```

Jejak menunjukkan rencana lengkap terlebih dahulu, lalu hasil pekerja, lalu komposisi pemecah. Bandingkan jumlah token (kami mencetak jumlah karakter kasar) dengan proses interleaved gaya ReAct - ReWOO menang dalam tugas terstruktur semacam ini.

## Pakai

LangGraph mengirimkan Plan-and-Execute sebagai resep (`create_react_agent` untuk ReAct, grafik khusus untuk plan-execute). Aliran CrewAI mengkodekan pola secara langsung: kamu menentukan tugas di awal dan Aliran DAG mengeksekusinya. Pendekatan data sintetik Plan-and-Act sebagian besar masih berupa penelitian; pola runtime (rencana eksplisit DAG) dikirimkan dalam produksi melalui LangGraph dan CrewAI Flows.

## Kirim

`outputs/skill-rewoo-planner.md` menghasilkan DAG paket ReWOO dari permintaan pengguna, berdasarkan katalog alat. Ini memvalidasi rencana (asiklik, setiap referensi diselesaikan, setiap alat ada) sebelum diserahkan ke pelaksana.

## Latihan

1. Memparalelkan eksekusi pekerja untuk node rencana independen. Apa manfaatnya bagi kamu pada DAG 6-node dengan 2 grup paralel?
2. Tambahkan node perencana ulang yang aktif jika ada pekerja yang mengembalikan kesalahan. Apa perubahan terkecil pada ReWOO yang menjadikannya Plan-and-Execute?
3. Ganti `Planner` dengan model kecil (kelas 7B) dan pertahankan `Solver` pada model frontier. Bandingkan kualitas end-to-end — di manakah kegagalan pemisahan?
4. Baca Bagian 4 makalah ReWOO tentang distilasi perencana. Reproduksi hasil 175B -> 7B secara konseptual: training data apa yang kamu perlukan, dan bagaimana kamu menilai kualitas rencana?
5. Pindahkan mainan ke bentuk lintasan Rencana-dan-Bertindak: rencana adalah urutan, bukan DAG. Pengorbanan apa yang berubah?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| UlangWOO | "Penalaran tanpa observasi" | Rencanakan, lalu ambil bukti secara paralel, lalu selesaikan — tidak ada observasi dalam prompt perencanaan |
| Rencanakan dan Jalankan | "Pola rencana-eksekusi LangChain" | ReWOO dengan node replanner opsional setelah eksekusi |
| Rencana dan Tindakan | "Rencana-eksekusi berskala" | Pemisahan perencana/pelaksana secara eksplisit dengan training data rencana sintetik untuk tugas jangka panjang |
| Referensi bukti | "#E1, #E2,..." | Placeholder simpul rencana diganti dengan output pekerja sebelumnya pada waktu pengiriman |
| Distilasi perencana | "Perencana kecil, pelaksana besar" | Sempurnakan model kecil pada jejak perencana dari guru besar |
| Efisiensi token | "Lebih sedikit perjalanan pulang pergi" | Token 5x lebih sedikit di HotpotQA vs ReAct di koran |
| Pelaksana DAG | "Pengirim topologi" | Menjalankan node rencana dalam urutan ketergantungan; paralel di setiap level |

## Bacaan Lanjutan

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — makalah kanonik
- [Erdogan dkk., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — perencana-pelaksana berskala dengan rencana sintetik
- [Tutorial LangGraph Plan-and-Execute](https://docs.langchain.com/oss/python/langgraph/overview) — resep framework
- [Antropik, Agen Bangunan yang Efektif](https://www.anthropic.com/research/building- Effective-agents) — pilih pola paling sederhana yang berhasil
