# Lalu Lintas Bayangan, Peluncuran Canary, dan Penerapan Progresif untuk LLM

> Peluncuran LLM menggabungkan bagian tersulit dari penerapan perangkat lunak: tidak ada pengujian unit, mode kegagalan yang menyebar, sinyal yang tertunda. Urutannya adalah (1) mode bayangan — duplikat permintaan produk ke model kandidat, catat, bandingkan tanpa dampak pengguna; menangkap masalah distribusi yang jelas namun bukan merupakan jaminan kualitas; (2) peluncuran canary — perpindahan lalu lintas progresif 10% → 25% → 50% → 75% → 100% dengan gerbang di setiap langkah; melacak persentil latensi, biaya/permintaan, tingkat kesalahan/penolakan, distribusi panjang output, tingkat umpan balik pengguna; (3) Pengujian A/B untuk alternatif berbeda setelah stabilitas dikonfirmasi. Non-determinisme tidak dapat direduksi — variasi akurasi hingga 15% di seluruh proses dengan input yang identik karena non-asosiasi FP GPU ditambah varian ukuran batch. Biaya bersifat variabel, tidak konstan — model yang 20% ​​lebih baik bisa menjadi 3x lebih mahal per panggilan. Kecepatan rollback sangat menentukan: jika rollback memerlukan penerapan ulang, kamu terlalu lambat. Kebijakan ada di config/flags; model berada di registri dengan intisari yang di-embed; rollback = kebijakan balik + kembalikan ambang batas + sematkan model lama dalam hitungan detik.

**Type:** Learn
**Language:** Python (stdlib, simulator perkembangan kenari mainan)
**Prerequisites:** Phase 17 · 13 (Observabilitas), Phase 17 · 21 (Pengujian A/B)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Membedakan mode bayangan (perbandingan tanpa dampak), canary (lalu lintas langsung progresif), dan A/B (perbandingan terkonfirmasi stabilitas).
- Menghitung lima metrik kenari khusus LLM (latensi, biaya/permintaan, kesalahan/penolakan, distribusi panjang output, umpan balik pengguna).
- Jelaskan mengapa non-determinisme LLM (hingga 15%) mengubah arti "stabil" dalam peluncuran.
- Rancang jalur rollback yang membutuhkan waktu beberapa detik (pembalikan kebijakan) bukan jam (penyebaran ulang).

## Masalah

kamu mengirimkan model baru. Evaluasi offline menunjukkan peningkatan akurasi 3%. kamu menyalakannya dalam produksi. Dalam 24 jam, biaya naik 40%, jempol ke bawah pengguna naik 8%, tiga tiket pelanggan melaporkan "jawaban aneh". kamu memutar kembali. Penempatan ulang memerlukan waktu 3 jam. Akhir pekanmu hancur.

Setiap bagian dari hal itu bisa dihindari. Mode bayangan akan mengalami lonjakan biaya sebesar 40% sebelum pengguna mana pun melihatnya. Canary akan berhenti di 10% ketika jempol ke bawah digerakkan. Pengembalian tanda kebijakan akan memakan waktu 30 detik. Disiplin inilah yang mengisi kesenjangan antara "evaluasi offline terlihat bagus" dan "pengguna sebenarnya senang".

## Konsep

### Modus bayangan

Kandidat menerima permintaan yang sama dengan produksi; output dicatat, tidak dikembalikan ke pengguna. Tidak ada dampak terhadap pengguna. Catatan:

- Konten output (berbeda dengan produksi).
- Jumlah token (biaya delta).
- Latensi.
- Penolakan dan kesalahan.

Hasil tangkapan: kenaikan biaya, regresi panjang, perubahan penolakan yang jelas, kesalahan besar. TIDAK menangkap: pengguna delta berkualitas akan melihatnya. Bayangan adalah uji asap, bukan uji kualitas.

### Peluncuran kenari

Pergeseran lalu lintas progresif dengan gerbang. Perkembangan umum: 1% → 10% → 25% → 50% → 75% → 100%. Gerbang pada 5 metrik di setiap langkah:

1. **Persentil latensi** — P50, P95, P99. Pelanggaran: kenari memiliki P99 > 1,5x baseline.
2. **Biaya per permintaan** — gabungan $. Pelanggaran: >20% di atas garis dasar.
3. **Tingkat kesalahan / penolakan** — 5xx ditambah penolakan eksplisit. Pelanggaran: 2x garis dasar.
4. **Distribusi panjang output** — rata-rata + P99. Pelanggaran: pergeseran distribusi.
5. **Tingkat umpan balik pengguna** — tidak suka / pengajuan tiket. Pelanggaran: 1,5x garis dasar.

### Non-determinisme adalah varian baru

Input yang identik menghasilkan output yang tidak identik. Alasan:- GPU FP non-associativity (urutan pengurangan floating-point bervariasi berdasarkan batch).
- Varians ukuran batch (prompt yang sama dalam batch 128 vs batch 16).
- Pengambilan sample (suhu > 0).

Diukur: variasi akurasi run-to-run hingga 15% pada set eval yang identik. "Stabil" dalam peluncuran berarti metrik berada dalam varians yang diharapkan, tidak identik dengan dasar. Atur gerbang di atas lantai kebisingan.

### Biaya adalah variabel

Model yang 20% lebih baik bisa 3x lebih mahal per panggilan. Biaya/permintaan adalah salah satu dari lima gerbang. Mengirimkan model yang "lebih baik" yang merusak keekonomian unit adalah sebuah kemunduran.

### Rollback adalah senjatanya

- Bendera kebijakan (sistem bendera feature): persentase flip dalam konfigurasi; membutuhkan waktu beberapa detik.
- Embedding model (intisari registri): model yang di-embed tidak ditingkatkan secara otomatis.
- Rollback = mengembalikan flag + menyetel intisari yang di-embed ke sebelumnya. Detik, bukan jam.

Jika tumpukan kamu memerlukan penerapan ulang untuk melakukan rollback, perbaiki sebelum bergulir.

### Perkakas

**Argo Rollouts** / **Flagger** — Pengontrol pengiriman progresif Kubernetes. Integrasikan dengan perutean berbobot Istio/Linkerd.

**Perutean berbobot Istio** — pembagian lalu lintas tingkat mesh layanan.

**KServe / Seldon Core** — penyajian model dengan canary bawaan.

**Bendera feature** — LaunchDarkly, Flagsmith, Unleash. Perubahan pada tingkat kebijakan, tidak ada penerapan ulang.

### Irama metrik

Pemeriksaan gerbang Canary setiap 5-15 menit tergantung volume lalu lintas. Lalu lintas 1% dengan 10 permintaan/mnt memberikan 50-150 titik data per jendela — cukup untuk latensi tetapi mengganggu input pengguna. 10% memberi ~10x lebih banyak. Kemajuan harus berhenti cukup lama untuk mengumpulkan sample yang cukup di setiap langkah.

### Langkah A/B bersifat opsional

Jika model baru sangat berbeda (perilaku berbeda, kurva biaya berbeda, corak berbeda), uji A/B pada 50% setelah canary lolos. Jika ini hanya versi perbaikan, lewati ke 100% saat canary gates lewat.

### Nomor yang harus kamu ingat

- Perkembangan burung kenari: 1% → 10% → 25% → 50% → 75% → 100%.
- Batasan non-determinisme: varians run-to-run hingga 15% pada input yang identik.
- Lima metrik kenari: latensi, biaya, kesalahan/penolakan, panjang output, umpan balik pengguna.
- Gerbang biaya: >20% di atas garis dasar merupakan pelanggaran.
- Kembalikan: detik, bukan jam.

## Pakai

`code/main.py` menyimulasikan peluncuran canary dengan regresi yang disuntikkan. Melaporkan pada phase mana peluncuran dihentikan dan gerbang mana yang dipicu.

## Kirim

Lesson ini menghasilkan `outputs/skill-rollout-runbook.md`. Dengan mempertimbangkan model kandidat, garis dasar, dan toleransi risiko, rancang rencana shadow→canary→100%.

## Latihan

1. Jalankan `code/main.py`. Suntikkan regresi biaya 25%. Pada phase manakah burung kenari berhenti?
2. Model baru kamu memiliki peningkatan akurasi 3% secara offline tetapi biaya/permintaan +18%. Apakah itu sebuah kapal? Tergantung pada kebijakannya — tulis kedua jalur.
3. Rancang rollback yang membutuhkan waktu kurang dari 60 detik secara menyeluruh. Buat daftar infrastruktur yang diperlukan.
4. Non-determinisme menunjukkan ±7% pada evaluasi kamu. Atur gerbang kenari agar kamu tidak menimbulkan alarm palsu. Pengganda apa yang kamu gunakan?
5. Mode bayangan mendapat lonjakan biaya 40% sebelum kenari. Tulis aturan peringatan yang menyala dalam bayangan.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Modus bayangan | "duplikat ke yang baru" | Pengiriman ke kandidat tanpa dampak untuk logging |
| kenari | "lalu lintas progresif" | Peluncuran bertahap yang dapat dilihat pengguna dengan gerbang |
| Gerbang | "pemeriksaan peluncuran" | Ambang batas metrik yang menghalangi perkembangan |
| Non-determinisme | "Varians LLM" | Perbedaan run-to-run yang tidak dapat direduksi |
| Bendera kebijakan | "bendera balikkan kembali" | Rollback tingkat konfigurasi, detik bukan jam |
| Pin model | "intisari registri" | Referensi yang tidak dapat diubah ke versi model |
| Peluncuran Argo | "K8 progresif" | Pengontrol canary/rollback asli Kubernetes |
| Melayani | "inference K8" | Model penyajian dengan kenari primitif |
| Istio berbobot | "jala terbelah" | Pemisah lalu lintas jaringan layanan |

## Bacaan Lanjutan

- [TianPan — Merilis Feature AI Tanpa Mengganggu Produksi](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Menerapkan Model ML dengan Aman](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four- controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Pola Penerapan LLM Tingkat Lanjut](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Dokumen Peluncuran Argo](https://argo-rollouts.readthedocs.io/)
- [Dokumen pelapor](https://docs.flagger.app/)
