# A2A — Protokol Agen-ke-Agen

> Google mengumumkan A2A pada bulan April 2025; pada bulan April 2026 spesifikasinya ada di https://a2a-protocol.org/latest/spesifikasi/ dan 150+ organisasi mendukungnya. A2A adalah pelengkap horizontal untuk MCP (Lesson 13): dimana MCP adalah vertikal (agen ↔ alat), A2A adalah peer-to-peer (agen ↔ agen). Ini mendefinisikan Kartu Agen (penemuan), tugas dengan artefak (teks, data terstruktur, video), siklus hidup tugas buram, dan autentikasi. Sistem produksi semakin banyak memasangkan MCP dengan A2A. Google Cloud meluncurkan dukungan A2A ke Vertex AI Agent Builder selama tahun 2025-2026.

**Type:** Learn + Build
**Language:** Python (stdlib, `http.server`, `json`)
**Prerequisites:** Fase 16 · 04 (Model Primitif)
**Waktu:** ~75 menit

## Masalah

Agen kamu perlu menghubungi agen lain di sistem lain. Bagaimana? kamu dapat mengekspos titik akhir HTTP, menentukan skema JSON yang dipesan lebih dahulu, dan berharap pihak lain yang menyampaikannya. Setiap pasangan agen menjadi integrasi khusus.

A2A adalah protokol kabel universal untuk panggilan itu. Penemuan standar, model tugas standar, transportasi standar, artefak standar. Seperti HTTP+REST tetapi untuk agen sebagai warga kelas satu.

## Konsep

### Empat elemen

**Kartu Agen.** Dokumen JSON di `/.well-known/agent.json` yang menjelaskan agen: nama, keterampilan, titik akhir, modalitas yang didukung, persyaratan autentikasi. Penemuan terjadi dengan membaca kartu.

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Tugas.** Unit kerja. Objek async dan stateful dengan siklus hidup: `submitted → working → completed / failed / canceled`. Klien mengirimkan tugas, jajak pendapat, atau berlangganan pembaruan.

**Artefak.** Jenis hasil yang dihasilkan oleh tugas. Teks, JSON terstruktur, gambar, video, audio. Artefak diketik sehingga modalitas yang berbeda menjadi kelas satu.

**Siklus hidup buram.** A2A tidak menentukan *bagaimana* agen distance jauh menyelesaikan tugas. Klien melihat transisi keadaan dan artefak; implementasinya bebas menggunakan framework apapun.

### Perpecahan MCP/A2A

- **MCP** (Lesson 13): agen ↔ alat. Agen membaca/menulis melalui JSON-RPC ke server alat. Tanpa kewarganegaraan secara default.
- **A2A**: agen ↔ agen. Protokol rekan; kedua belah pihak adalah agen dengan alasan mereka sendiri.

Sistem multi-agen produksi menggunakan keduanya. Rekan A2A memanggil alat MCP di sisinya. Perpecahan ini menjaga kedua kekhawatiran tetap bersih.

### Alur penemuan

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

Atau dengan streaming: Berlangganan SSE ke `/tasks/{id}/events` untuk pembaruan push.

### Otentikasi

A2A mendukung tiga pola umum:

- **Token pembawa** — OAuth2 atau buram.
- **mTLS** — TLS bersama; organisasi membuktikan identitas satu sama lain.
- **Permintaan yang ditandatangani** — HMAC melalui payload.

Auth dinyatakan dalam Kartu Agen; klien menemukan dan mematuhinya.

### 150+ organisasi pada April 2026

Adopsi perusahaan mendorong skala A2A. Judulnya: A2A menjadi cara sistem agen perusahaan melintasi batas kepercayaan. Google Cloud mengirimkan dukungan Vertex AI Agent Builder A2A; Microsoft Agent Framework mendukungnya; sebagian besar framework utama (LangGraph, CrewAI, AutoGen) mengirimkan adaptor A2A.

### Dimana A2A menang

- **Panggilan lintas organisasi.** Agen di perusahaan A menelepon agen di perusahaan B. Tanpa A2A, setiap pasangan adalah kontrak yang dipesan lebih dahulu.
- **Kerangka kerja heterogen.** Agen LangGraph memanggil agen CrewAI memanggil agen Python khusus. A2A menjadi normal.
- **Artefak yang diketik.** Hasil video, JSON terstruktur, audio — semuanya kelas satu.
- **Tugas yang berjalan lama.** Siklus proses + polling yang tidak jelas membuat tugas yang memakan waktu berjam-jam menjadi mudah.

### Dimana A2A kesulitan- **Panggilan mikro yang sensitif terhadap latensi.** Siklus hidup A2A tidak sinkron. Agen-ke-agen sub-milidetik tidak cocok; menggunakan RPC langsung.
- **Agen dalam proses yang digabungkan secara ketat.** Jika kedua agen berjalan dalam proses Python yang sama, perjalanan bolak-balik HTTP A2A berlebihan.
- **Tim kecil.** Overhead spesifikasi nyata; agen internal saja mungkin tidak memerlukan formalitas.

### A2A vs ACP, ANP, NLIP

Beberapa spesifikasi terkait muncul pada tahun 2024-2026:

- **ACP** (IBM/Linux Foundation) — pendahulu A2A, cakupannya lebih sempit.
- **ANP** (Agent Network Protocol) — yang mengutamakan penemuan rekan sejawat, mengutamakan desentralisasi.
- **NLIP** (Protokol Interaksi Bahasa Alami Ecma, distandarisasi Desember 2025) — jenis konten bahasa alami.

A2A adalah protokol rekan yang paling banyak diadopsi pada April 2026. Lihat arXiv:2505.02279 (Liu dkk., "Survei Protokol Interoperabilitas Agen") untuk perbandingan.

## Build

`code/main.py` mengimplementasikan server dan klien minimal A2A menggunakan `http.server` dan JSON. Servernya:

- mengekspos `/.well-known/agent.json`,
- menerima `POST /tasks`,
- mengelola status tugas,
- mengembalikan artefak di `GET /tasks/{id}`.

Klien:

- mengambil Kartu Agen,
- menyerahkan tugas,
- jajak pendapat sampai selesai,
- membaca artefak.

Jalankan:

```
python3 code/main.py
```

Skrip memulai server di thread latar belakang, lalu menjalankan klien di dalamnya. kamu melihat alur lengkapnya: penemuan, pengiriman, jajak pendapat, artefak.

## Pakai

`outputs/skill-a2a-integrator.md` merancang integrasi A2A: konten Kartu Agen, skema tugas, pilihan autentikasi, streaming vs polling.

## Kirim

Daftar periksa:

- **Sematkan versi spesifikasi.** A2A masih terus berkembang; Kartu Agen harus mendeklarasikan versi protokol.
- **Pembuatan tugas yang idempoten.** Pengiriman duplikat (percobaan ulang jaringan) akan menghasilkan satu tugas.
- **Skema artefak.** Menyatakan bentuk yang dikembalikan agen; konsumen harus memvalidasi.
- **Batas tarif + autentikasi.** A2A dapat dilihat oleh publik; menerapkan keamanan web standar.
- **Surat mati untuk tugas yang gagal.** Periksa pola dari waktu ke waktu untuk mengetahui jenis kegagalan yang berulang.

## Latihan

1. Jalankan `code/main.py`. Konfirmasikan klien menemukan server dan menerima artefak yang benar.
2. Tambahkan keterampilan kedua ke server (misalnya, "meringkas"). Perbarui Kartu Agen. Tulis klien yang memilih keterampilan berdasarkan jenis tugas.
3. Menerapkan titik akhir streaming SSE: `/tasks/{id}/events` yang mengeluarkan perubahan status. Apa yang perlu dilakukan klien secara berbeda?
4. Baca spesifikasi A2A (https://a2a-protocol.org/latest/spesifikasi/). Identifikasi tiga hal yang diamanatkan oleh spesifikasi yang tidak diterapkan oleh demo ini.
5. Bandingkan A2A (Penemuan Kartu Agen) dengan MCP (daftar kemampuan sisi server melalui `listTools`). Apa trade-off antara agen yang menggambarkan diri sendiri dan penyelidikan kemampuan?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| A2A | "Agen-ke-agen" | Protokol rekan bagi agen untuk memanggil agen lain di seluruh sistem. Google 2025.|
| Kartu Agen | "Kartu nama agen" | JSON di `/.well-known/agent.json` menjelaskan keterampilan, titik akhir, autentikasi. |
| Tugas | "Satuan Kerja" | Objek stateful async dengan siklus hidup; artefak yang dihasilkan setelah selesai. |
| Artefak | "Hasilnya" | Output yang diketik: teks, JSON terstruktur, gambar, video, audio. Media kelas satu. |
| Siklus hidup buram | "Cara penyelesaiannya urusan agen" | Klien melihat transisi keadaan; server bebas memilih kerangka/alat. |
| Penemuan | "Menemukan agen" | `GET /.well-known/agent.json` mengembalikan kartu. |
| MCP vs A2A | "Alat vs rekan" | MCP: alat agen vertikal ↔. A2A: agen horizontal ↔ agen. |
| ACP / ANP / NLIP | "Protokol saudara" | Spesifikasi yang berdekatan; A2A adalah yang paling banyak diadopsi tahun 2026. |

## Bacaan Lanjutan

- [Spesifikasi A2A](https://a2a-protocol.org/latest/spesifikasi/) — spesifikasi kanonik
- [Blog Google Developers — Pengumuman A2A](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — postingan peluncuran April 2025
- [Repo GitHub A2A](https://github.com/a2aproject/A2A) — implementasi referensi dan SDK
- [Liu dkk. — Survei Protokol Interoperabilitas Agen](https://arxiv.org/html/2505.02279v1) — Perbandingan MCP, ACP, A2A, ANP
