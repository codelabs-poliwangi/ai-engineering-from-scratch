# Agen Multimodal dan Penggunaan Komputer (Capstone)

> Produk frontier 2026 adalah agen multimodal yang membaca tangkapan layar, mengklik tombol, menavigasi UI web, mengisi formulir, dan menyelesaikan alur kerja secara menyeluruh. SeeClick dan CogAgent (2024) membuktikan primitif berbasis GUI. Ferret-UI menambahkan ponsel. ChartAgent memperkenalkan penggunaan alat visual untuk grafik. VisualWebArena dan AgentVista (2026) adalah tolok ukur yang harus dikejar — dan bahkan Gemini 3 Pro dan Claude Opus 4.7 mendapat skor ~30% pada tugas berat AgentVista. Batu penjuru ini menyatukan setiap alur Fase 12: persepsi (VLM resolusi tinggi), penalaran (LLM dengan penggunaan alat), landasan (output koordinat), memori cakrawala panjang, dan evaluasi.

**Type:** Batu penjuru
**Language:** Python (stdlib, skema tindakan + kerangka loop agen)
**Prerequisites:** Fase 12 · 05 (LLaVA), Fase 12 · 09 (Qwen-VL JSON), Fase 14 (Rekayasa Agen)
**Waktu:** ~240 menit

## Tujuan Pembelajaran

- Rancang lingkaran agen multimodal: pahami → alasan → bertindak → amati → ulangi.
- Buat skema output landasan GUI (klik koordinat, ketik teks, gulir, seret) yang dapat dipancarkan VLM sebagai JSON.
- Bandingkan agen khusus tangkapan layar vs agen pohon aksesibilitas vs agen hibrid.
- Siapkan evaluasi tolok ukur agen multimodal pada potongan kecil VisualWebArena.

## Masalah

Alur kerja situs pemesanan: "temukan saya penerbangan ke Tokyo untuk tanggal 15 April, kursi di lorong di bawah $800, pesanlah."

Agen multimoda perlu:

1. Ambil tangkapan layar browser.
2. Parsing tangkapan layar + URL + sasaran menjadi sebuah rencana.
3. Keluarkan tindakan terstruktur: klik (di x,y), ketik "Tokyo" (di elemen E), gulir ke bawah, pilih (tombol radio).
4. Terapkan tindakan tersebut ke browser.
5. Amati keadaan baru (screenshot berikutnya).
6. Ulangi sampai tugas selesai.

Setiap langkah adalah panggilan VLM multimodal. Output VLM harus berupa JSON yang dapat diurai. Kesalahan bertambah di seluruh langkah, jadi pemulihan itu penting.

## Konsep

### Pembumian GUI — yang primitif

Landasan GUI adalah: diberi tangkapan layar dan instruksi bahasa alami, output koordinat (x, y) untuk diklik (atau tindakan lainnya).

SeeClick (arXiv:2401.10935) adalah hasil terbuka pertama dalam skala besar: menyempurnakan VLM pada data GUI sintetik + nyata, koordinat output sebagai token teks biasa. Bekerja.

CogAgent (arXiv:2312.08914) menambahkan pengkodean resolusi tinggi 1120x1120 untuk UI padat. Skor: ~84% pada navigasi web.

Ferret-UI (arXiv:2404.05719) berfokus pada UI seluler, terintegrasi dengan data aksesibilitas iOS.

Format output biasanya JSON:

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` membantu pemulihan: jika koordinat melayang di antara tangkapan layar, petunjuk semantik memungkinkan sistem melakukan grounding kembali.

### Skema tindakan

Skema tindakan tipikal memiliki 6-10 jenis tindakan:

- `click`: (x, y)
- `type`: (teks, x?, y?)
- `scroll`: (arah, jumlah)
- `drag`: (x0, y0, x1, y1)
- `select`: (indeks_pilihan)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (md)
- `done`: (sukses, penjelasan)

Agen mengeluarkan satu tindakan per langkah. Pembungkus browser mengeksekusi dan mengembalikan keadaan baru.

### Hanya tangkapan layar vs pohon aksesibilitas

Dua mode input:

- Hanya tangkapan layar: gambar penuh, tanpa info struktural. Paling umum; berfungsi di aplikasi apa pun.
- Pohon aksesibilitas: info aksesibilitas DOM / iOS terstruktur. Jauh lebih andal untuk grounding; bekerja di mana pohon tersedia.
- Hibrida: keduanya, dengan pohon sebagai landasan yang andal untuk tindakan atom dan tangkapan layar untuk konteks semantik.Agen produksi menggunakan hibrida jika memungkinkan. Browser automation (Selenium + accessibility) always has the tree; aplikasi desktop terkadang demikian.

### Memori cakrawala panjang

Alur kerja 20 langkah menghasilkan 20 tangkapan layar. Konteks VLM terisi dengan cepat. Tiga strategi kompresi:

- Summary-chain: after every 5 steps, summarize what has happened, drop old screenshots.
- Skip-frame: keep the first, last, and every 3rd screenshot.
- Tool-recorded log: execute actions, keep a text log of what was done; jangan melihat kembali tangkapan layar lama.

API penggunaan komputer Claude menggunakan pola log. Lebih sederhana, lebih dapat diandalkan.

### Penggunaan alat visual

ChartAgent (arXiv:2510.04514) introduces visual tool use for chart understanding: crop, zoom, OCR, call external detection. The agent can output "crop to region (100, 200, 300, 400) then call OCR" as a tool call. Alat ini mengembalikan teks; VLM melanjutkan alasannya.

Pola ini menggeneralisasi: prompt set-of-mark, anotasi wilayah, dan alat deteksi eksternal semuanya sesuai dengan skema "keluarkan panggilan alat, terima respons terstruktur" yang sama.

### Tolok ukur tahun 2026

- ScreenSpot-Pro. GUI didasarkan pada ~1k tangkapan layar web. Buka SOTA Qwen2.5-VL-72B ~85%. Perbatasan ~90%.
- VisualWebArena. Tugas web ujung ke ujung (toko, forum, iklan baris). Buka SOTA ~20%. Gemini 3 Pro ~27%.
- AgenVista (arXiv:2602.23166). Tolok ukur tersulit tahun 2026. Alur kerja yang realistis di 12 domain. Model Frontier mendapat skor 27-40%; model terbuka 10-20%.
- WebArena / Toko Web. Tolok ukur yang lebih lama; jenuh oleh perbatasan.

### Kenapa masih sulit

Hambatan kinerja agen:

1. Landasan visual pada skala halus. "Click the small X" fails often at mobile resolution.
2. Perencanaan jangka panjang. After 10 actions, the agent drifts from the goal.
3. Pemulihan kesalahan. When a click fails (wrong button), detecting + recovering is rarely trained data.
4. Konteks lintas halaman. Melompat antar tab atau formulir panjang akan kehilangan status.

Arahan penelitian: arsitektur memori, perencanaan ulang eksplisit, verifikasi multimodal (pencocokan tangkapan layar untuk keberhasilan tindakan).

### Pembuatan batu penjuru

The capstone task: build a computer-use agent that:

1. Reads the HTML + screenshot of a booking-site mock page.
2. Plans a multi-step sequence: search → select → fill form → submit.
3. Emits JSON actions matching the action schema.
4. Mengevaluasi bagian 10 tugas yang tetap.

The lesson provides scaffold code that is easy to extend into a real browser.

## Pakai

`code/main.py` adalah perancah batu penjuru:

- Definisi JSON skema tindakan (10 tindakan).
- Status browser tiruan sebagai dict.
- Kerangka loop agen: menerima status, memancarkan tindakan, menerapkan, memutar.
- 10-task mini-benchmark (synthetic pages) to measure end-to-end success rate.
- Kait pemulihan kesalahan ketika suatu tindakan gagal.

## Kirim

Lesson ini menghasilkan `outputs/skill-multimodal-agent-designer.md`. Mengingat produk penggunaan komputer (domain, rangkaian tindakan, target evaluasi), merancang loop agen lengkap, strategi memori, mode landasan, dan skor tolok ukur yang diharapkan.

## Latihan

1. Extend the action schema with a `screenshot_region` tool (crop + zoom). Tugas apa yang bermanfaat?

2. Baca AgentVista (arXiv:2602.23166). Jelaskan kategori tugas tersulit dan mengapa model frontier masih gagal.

3. Kompresi memori cakrawala panjang: rancang rantai ringkasan dengan ≤4 tangkapan layar tetap aktif, nomor apa pun dicatat.

4. Build an error-recovery hook: on action failure (button not found), what does the agent do next?5. Bandingkan Claude 4.7 khusus tangkapan layar dengan tangkapan layar hibrid + pohon aksesibilitas Qwen2.5-VL pada 10 tugas web. Tugas mana yang menang?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| landasan GUI | "Klik koordinat" | Output model (x,y) untuk target instruksi pada tangkapan layar |
| Skema tindakan | "Definisi alat" | Deskripsi JSON tentang tindakan yang valid (klik, ketik, gulir, seret) |
| Pohon aksesibilitas | "DOM Terstruktur" | Hierarki UI yang dapat dibaca mesin dari browser/iOS API |
| Agen hibrida | "Tangkapan Layar + Pohon" | Menggunakan gambar dan informasi terstruktur; lebih dapat diandalkan daripada sendirian |
| Penggunaan alat visual | "Perbesar/pangkas/deteksi" | Agen menyebut alat penglihatan eksternal (OCR, deteksi) rencana tengah |
| Rantai ringkasan | "Kompresi memori" | Ringkasan teks berkala menggantikan riwayat tangkapan layar yang panjang |
| VisualWebArena | "Bangku web E2E" | Tolok ukur 2024 untuk tugas web ujung ke ujung |
| AgenVista | "Bangku keras 2026" | Alur kerja realistis 12 domain; bahkan skor Gemini 3 Pro ~30% |

## Bacaan Lanjutan

- [Cheng dkk. — LihatKlik (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong dkk. — Agen Cog (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [Kamu dkk. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [Agen Bagan (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh dkk. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
