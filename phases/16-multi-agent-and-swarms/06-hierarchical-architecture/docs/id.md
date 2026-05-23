# Arsitektur Hierarki dan Mode Kegagalannya

> Hierarki adalah supervisor yang disarangkan. Agen manajer, sub-manajer, dan pekerja. CrewAI `Process.hierarchical` adalah versi buku teks: `manager_llm` mendelegasikan tugas secara dinamis dan memvalidasi output. Setara dengan LangGraph adalah `create_supervisor(create_supervisor(...))`. Ini adalah pola alami jika tugasnya adalah bagan organisasi nyata. Hal ini juga merupakan pola yang paling mungkin terjerumus ke dalam perulangan manajerial – agen manajer menugaskan pekerjaan dengan buruk, salah menafsirkan sub-output, atau gagal mencapai konsensus. Sequential sering kali mengalahkannya.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 16 · 05 (Pola Supervisor)
**Waktu:** ~60 menit

## Masalah

Setelah pola penyelia berhasil, langkah alami berikutnya adalah "bagaimana jika para pekerja itu sendiri adalah penyelia?" Tim memiliki sub-tim; perusahaan memiliki departemen dari departemen. Arsitektur hierarki mencerminkan hal itu.

Masalah: Manajer LLM tidak sama dengan manajer manusia. Seorang manajer manusia mempunyai prioritas yang stabil mengenai apa yang diketahui oleh laporannya. Seorang manajer LLM mempertimbangkan kembali organisasinya dari apa pun yang ada dalam konteksnya. Penyimpangan kecil dalam konteks itu, dan seluruh pohon salah mengalokasikan pekerjaan.

## Konsep

### Bentuknya

```
                 Manager
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       Sub-Mgr A         Sub-Mgr B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5
```

Setiap node internal merencanakan, mendelegasikan, dan mensintesis. Hanya daun yang berfungsi.

### Tempat yang bersinar

- **Hapus pemetaan organisasi.** Jika tugas sebenarnya bersifat departemen ("tinjauan hukum dokumen, tinjauan keuangan dokumen, tinjauan teknik dokumen, lalu rangkum untuk eksekutif"), hierarkinya jelas.
- **Ringkasan lokal.** Setiap sub-manajer menyatukan output timnya sebelum manajer puncak melihatnya. Manajer puncak melihat tiga ringkasan sub-manajer, bukan lima belas output pekerja.

### Dimana rusaknya

Tiga mode kegagalan yang terus ditemukan pada pemeriksaan post-mortem pada tahun 2026:

1. **Kesalahan penetapan tugas.** Manajer membaca sasaran, berhalusinasi tentang decomposition, dan mendelegasikan ke sub-manajer yang salah. Karena sub-manajer dengan patuh mengerjakan apa yang diberikan, kesalahan hanya muncul pada sintesis teratas — satu tingkat yang dihilangkan dari tempat manusia dapat menangkapnya.
2. **Kesalahan penafsiran output.** Sub-manajer mengembalikan "tidak dapat memverifikasi klaim X". Manajer puncak merangkumnya sebagai "klaim X belum dikonfirmasi". Makna melayang di setiap tingkatan.
3. **Lingkaran konsensus.** Dua sub-manajer tidak setuju; manajer puncak meminta mereka untuk berdamai; mereka mendelegasikan kembali ke bawah; pekerja dijalankan kembali; sub-manajer memberikan jawaban yang sedikit berbeda; lingkaran. `Process.hierarchical` CrewAI menjaga hal ini dengan batas langkah, namun batas itu sendiri sekarang menjadi hyperparameter.

### Pertanyaan penentu

Berurutan (pipa linier) vs hierarki: apakah tugas kamu benar-benar memiliki sub-tim independen, atau apakah itu satu aliran linier yang berpura-pura menjadi pohon? Jika yang terakhir, gunakan berurutan. Jika yang pertama, gunakan aturan rekonsiliasi yang hierarkis namun anggarannya eksplisit.

### Implementasi CrewAI

`Process.hierarchical` menghubungkan manajer LLM ke kru spesialis. Manajer:

- menerima tugas tingkat atas,
- menugaskan subtugas ke kru,
- mengevaluasi output kru,
- memutuskan apakah akan menerima, mendelegasikan ulang, atau mengulangi.

Dokumentasi: https://docs.crewai.com/en/introduction (cari "Proses Hierarki" di bawah Konsep Inti).

### Implementasi LangGraphLangGraph menggunakan panggilan `create_supervisor` bersarang. Pengawas internal mempunyai grafiknya sendiri; pengawas luar memperlakukan grafik dalam sebagai simpul buram. Ini lebih bersih daripada CrewAI untuk debugging (kamu dapat menelusuri setiap grafik secara terpisah) tetapi lebih sulit untuk mengekspresikan pembentukan ulang pohon secara dinamis.

Referensi: https://reference.langchain.com/python/langgraph-supervisor.

## Build

`code/main.py` menjalankan hierarki 3 tingkat:

- manajer puncak: membagi tugas menjadi cabang "rekayasa" dan "hukum",
- sub-manajer teknik: dibagi menjadi pekerja "frontend" dan "backend",
- sub-manajer resmi: satu pekerja.

Demo membandingkan jalur yang menyenangkan (semua orang setuju) dengan **jalur yang terganggu** di mana decomposition manajer puncak salah memberi label "legal" sebagai "keuangan" dan melihat rangkaian kesalahan — sub-manajer dengan patuh melakukan pekerjaan keuangan, penyintesis teratas melaporkan temuan keuangan, pertanyaan hukum awal tidak terjawab.

Jalankan:

```
python3 code/main.py
```

Output menunjukkan kedua jalur dengan perbedaan yang jelas antara "apa yang diminta" vs "apa yang disampaikan".

## Pakai

`outputs/skill-hierarchy-fitness.md` mengevaluasi apakah tugas tertentu harus menggunakan supervisor hierarkis, berurutan, atau datar. Input: uraian tugas, struktur organisasi, anggaran rekonsiliasi. Output: rekomendasi pola dengan mode kegagalan spesifik yang harus diwaspadai.

## Kirim

Jika kamu mengirimkan hierarki:

- **Batasi kedalaman pohon pada 2.** Tiga level sudah menyembunyikan sebagian besar error dari kemampuan observasi.
- **Anggaran rekonsiliasi eksplisit.** Tetapkan putaran maksimal sebelum manajer puncak harus berkomitmen. Biasanya 2.
- **Asal pada setiap sintesis.** Ringkasan setiap node harus menyebutkan output daun mana yang menghasilkannya.
- **Peringatan terhadap penyimpangan decomposition.** Catat decomposition pengelola per langkah; berbeda dengan permintaan pengguna. Jika decomposition tidak lagi mencakup kueri, aktifkan peringatan.

## Latihan

1. Jalankan `code/main.py` dan bandingkan senang vs gelisah. Berapa tingkat penyerahan manajer yang diperlukan sebelum output teratas sepenuhnya menyimpang dari pertanyaan pengguna?
2. Tambahkan level ketiga (atas → sub → sub-sub → pekerja). Ukur seberapa sering jalur yang terganggu terkoreksi vs menyimpang sepenuhnya seiring bertambahnya kedalaman.
3. Menerapkan pekerja "canary" di setiap sub-manajer yang selalu menanyakan pertanyaan pengguna asli tanpa perubahan. Gunakan jawaban canary untuk mendeteksi penyimpangan decomposition. Bagaimana seharusnya reaksi manajer ketika kenari tidak setuju dengan jawaban yang disintesis?
4. Baca dokumen `Process.hierarchical` CrewAI. Identifikasi satu pagar pembatas beton yang diterapkan CrewAI (batas langkah, batasan manager_llm) dan jelaskan mode kegagalan yang ditargetkan.
5. Bandingkan supervisor LangGraph yang disarangkan dengan hierarki CrewAI. Mana yang membuat putaran rekonsiliasi lebih murah untuk dideteksi?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Hierarki | "Pola bagan organisasi" | Pengawas atas pengawas; hanya daun yang berfungsi. |
| Manajer LLM | "Bos" | LLM yang menguraikan, menetapkan, dan memvalidasi pada node internal. |
| Penyimpangan decomposition | "Bos kehilangan alur cerita" | Perpecahan manajer puncak tidak lagi mencakup pertanyaan awal. |
| Lingkaran rekonsiliasi | "Pertemuan tanpa akhir" | Sub-manajer tidak setuju; delegasi ulang teratas; pekerja dijalankan kembali; loop sampai anggaran habis. |
| Langit-langit kedalaman-2 | "Jangan masuk lebih dalam dari 2 level" | Pagar pembatas empiris: 3+ level meruntuhkan kemampuan observasi. |
| Pertanyaan kenari | "Kebenaran dasar di setiap tingkatan" | Seorang pekerja yang selalu menanyakan kueri asli tidak berubah, untuk mendeteksi penyimpangan. |
| Rantai asal | "Siapa bilang apa" | Telusuri setiap sintesis kembali ke output daun yang memproduksinya. |

## Bacaan Lanjutan

- [Pengantar CrewAI — Process.hierarchical](https://docs.crewai.com/en/introduction) — hierarki buku teks dengan manajer LLM
- [Referensi supervisor LangGraph](https://reference.langchain.com/python/langgraph-supervisor) — supervisor bertingkat melalui `create_supervisor`
- [Rekayasa antropik — Sistem penelitian](https://www.anthropic.com/engineering/multi-agent-research-system) — mengapa Anthropic sengaja memilih supervisor datar daripada hierarki
- [Cemri dkk. — Mengapa Sistem LLM Multi-Agen Gagal?](https://arxiv.org/abs/2503.13657) — Taksonomi MAST; bagian tentang kegagalan koordinasi mendokumentasikan penyimpangan decomposition
