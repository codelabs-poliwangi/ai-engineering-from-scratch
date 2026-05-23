# Capstone 12 — Pipeline Pemahaman Video (Adegan, QA, Pencarian)

> Dua Belas Labs memproduksi Marengo + Pegasus. VideoDB mengirimkan API CRUD-untuk-video. Molmo 2 AI2 menerbitkan pos pemeriksaan VLM terbuka. Konteks panjang Gemini menangani video berjam-jam secara asli. TimeLens-100K mendefinisikan landasan temporal dalam skala besar. Alur tahun 2026 telah diselesaikan: segmentasi adegan, teks per adegan + embedding, penyelarasan transkrip, indeks multi-vector, dan kueri yang menjawab dengan stempel waktu (mulai, akhir) ditambah pratinjau bingkai. Batu penjurunya adalah menghabiskan 100 jam, mencapai standar publik, dan mengukur halusinasi dalam pertanyaan penghitungan dan tindakan.

**Type:** Batu penjuru
**Language:** Python (pipa), TypeScript (UI)
**Prerequisites:** Fase 4 (CV), Fase 6 (pidato), Fase 7 (Transformer), Fase 11 (rekayasa LLM), Fase 12 (multimodal), Fase 17 (infrastruktur)
**Fase yang dilakukan:** P4 · P6 · P7 · P11 · P12 · P17
**Waktu:** 30 jam

## Masalah

QA video berdurasi panjang adalah masalah multimodal yang paling membutuhkan bandwidth pada skala tahun 2026. Gemini 2.5 Pro dapat membaca video berdurasi 2 jam secara asli, tetapi memasukkan video berdurasi 100 jam ke dalam korpus yang dapat dikueri masih memerlukan indeks tingkat adegan. Bentuk produksi menggabungkan segmentasi adegan (TransNetV2 atau PySceneDetect), pemberian teks per adegan dengan VLM (Gemini 2.5, Qwen3-VL-Max, atau Molmo 2), penyelarasan transkrip (Whisper-v3-turbo dengan stempel waktu kata), dan indeks multi-vector yang menyimpan teks, embedding bingkai, dan transkrip secara berdampingan. Pipeline kueri menjawab dengan stempel waktu (mulai, akhir) ditambah pratinjau bingkai.

Tolok ukur bersifat publik (ActivityNet-QA, NeXT-GQA) ditambah kumpulan kustom 100 kueri kamu sendiri. Halusinasi pada pertanyaan jenis penghitungan dan tindakan adalah kelas kegagalan yang terkenal; batu penjuru secara eksplisit mengukurnya.

## Konsep

Tiga pipeline pipa berjalan secara paralel saat penyerapan. **Segmentasi adegan** memotong video menjadi beberapa adegan. **VLM captioning** menghasilkan teks per adegan dan bingkai yang di-embed dari bingkai utama. **Penyelarasan ASR** menghasilkan stempel waktu tingkat kata. Ketiga aliran tersebut digabungkan berdasarkan (scene_id, rentang waktu). Setiap adegan mendapat tiga tipe vector dalam indeks multi-vector (Qdrant): embedding teks, embedding bingkai utama, embedding transkrip.

Pada saat kueri, pertanyaan bahasa alami ditujukan terhadap ketiga vector; hasil digabungkan dengan RRF; adaptor pembumian temporal (gaya TimeLens) menyempurnakan jendela (mulai, akhir) dalam adegan teratas. Synthesizer VLM (Gemini 2.5 Pro atau Qwen3-VL-Max) mengambil kueri + adegan teratas + bingkai yang dipotong dan jawaban dengan stempel waktu yang dikutip dan pratinjau bingkai.

Pengukuran halusinasi itu penting. Pertanyaan berhitung (“berapa banyak orang yang memasuki ruangan?”) dan tipe tindakan (“apakah koki menuangkan sebelum mengaduk?”) terkenal tidak dapat diandalkan. Laporkan keakuratan secara terpisah dari pertanyaan deskriptif.

## Arsitektur

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## Tumpukan

- Segmentasi pemandangan: TransNetV2 (tercanggih 2024-26) atau PySceneDetect
- ASR: Whisper-v3-turbo melalui fast-whisper dengan stempel waktu kata
- Keterangan + penjawab VLM: Gemini 2.5 Pro atau Qwen3-VL-Max atau Molmo 2
- Pembumian sementara: Adaptor terlatih TimeLens-100K atau VideoITG
- Indeks: Qdrant dengan dukungan multi-vector (caption / frame / transkrip)
- UI: Next.js 15 dengan pemutar video HTML5 dan thumbnail adegan
- Eval: ActivityNet-QA, NeXT-GQA, kumpulan label tangan 100 pertanyaan khusus
- Tolok ukur halusinasi: subset penghitungan dan tipe tindakan dengan label tangan

## Bangun1. **Ingest walker.** Terima URL YouTube atau MP4 lokal. Turunkan skala ke 720p jika diperlukan. Bertahan `{video_id, file_path}`.

2. **Segmentasi adegan.** Jalankan TransNetV2 atau PySceneDetect untuk menghasilkan `[{scene_id, start_ms, end_ms, keyframe_path}]`. Target 100 jam: ~6k-8k adegan.

3. **ASR pass.** Jalankan Whisper-v3-turbo pada audio; ekspor stempel waktu tingkat kata; dibagi menjadi potongan transkrip per adegan.

4. **Teks ​​VLM.** Per adegan, panggil Gemini 2.5 Pro (atau Qwen3-VL-Max) dengan bingkai utama dan templat teks singkat. Menghasilkan teks + embedding bingkai.

5. **Indeks multi-vector.** Koleksi Qdrant dengan tiga vector bernama. Muatan: `{video_id, scene_id, start_ms, end_ms, keyframe_url}`.

6. **Kueri.** Pertanyaan bahasa alami memicu tiga kueri padat; bergabung dengan fusi peringkat timbal balik; top-k=5 adegan.

7. **Pembumian sementara.** Jalankan adaptor gaya TimeLens di adegan atas untuk menyempurnakan jendela (awal, akhir) dalam adegan.

8. **VLM synth.** Hubungi Gemini 2.5 Pro dengan kueri + 3 klip adegan teratas (sebagai gambar atau klip pendek) + transkrip. Memerlukan `(video_id, start_ms, end_ms)` kutipan.

9. **Eval.** Jalankan ActivityNet-QA dan NeXT-GQA. Buat kumpulan kustom 100 kueri. Laporkan akurasi keseluruhan + perincian per kelas (penghitungan, tindakan, deskriptif).

## Pakai

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## Kirim

`outputs/skill-video-qa.md` adalah hasil yang dapat dicapai. Dengan adanya URL YouTube atau video yang diunggah, pipeline tersebut mengindeks adegan dan menjawab pertanyaan dengan kutipan yang diberi stempel waktu.

| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | IoU landasan sementara | Persimpangan-over-union pada set grounding yang ditahan |
| 20 | Akurasi QA | NeXT-GQA dan 100 kueri khusus |
| 20 | Serap throughput | Jam video per dolar yang dibelanjakan |
| 20 | UI dan kutipan UX | Tautan stempel waktu, strip thumbnail, lompat ke bingkai |
| 15 | Tingkat halusinasi | Akurasi penghitungan dan tipe tindakan secara terpisah |
| **100** | | |

## Latihan

1. Tukar Gemini 2.5 Pro dengan Qwen3-VL-Max pada captioning pass. Laporkan delta kualitas teks pada sample 50 adegan yang diberi rating manusia.

2. Kurangi embedding bingkai per adegan menjadi satu vector gabungan, bukan multi-vector. Ukur regresi pengambilan.

3. Buat mode "penghitungan ketat": synthesizer mengekstrak setiap instance yang dihitung dengan stempel waktu dan pengguna mengklik untuk memverifikasi. Ukur apakah verifikasi pengguna mengurangi halusinasi.

4. Biaya penyerapan tolok ukur: jam video per dolar di tiga pilihan VLM. Pilihlah titik yang tepat.

5. Tambahkan transkrip yang didiari pembicara: jalankan diarisasi pembicara pyannote pada audio dan sematkan transkrip per pembicara. Tunjukkan "apa yang Alice katakan tentang X?" pertanyaan.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Segmentasi adegan | "Deteksi tembakan" | Memotong video menjadi beberapa adegan pada batas pengambilan gambar |
| Indeks multi-vector | "Keterangan + bingkai + transkrip" | Koleksi Qdrant dengan vector bernama per representasi |
| Landasan sementara | "Kapan tepatnya itu terjadi" | Menyempurnakan jendela (mulai, akhir) untuk jawaban kueri |
| Embedding bingkai | "Representasi visual" | Embedding vector pada bingkai utama; digunakan untuk kesamaan adegan-visual |
| Fusi RRF | "Fusi peringkat timbal balik" | Gabungkan strategi di beberapa daftar peringkat; trik pengambilan hibrida klasik |
| Menghitung halusinasi | "Salah hitung" | Mode kegagalan VLM yang diketahui pada pertanyaan "berapa X" |
| ActivityNet-QA | "Tolok ukur Video-QA" | Tolok ukur akurasi QA video berdurasi panjang |

## Bacaan Lanjutan- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — buka pos pemeriksaan VLM
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — landasan temporal dalam skala besar
- [Konteks panjang Video Gemini](https://deepmind.google/technologies/gemini) — referensi yang dihosting
- [VideoDB](https://videodb.io) — Referensi API CRUD untuk video
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — referensi komersial
- [TransNetV2](https://github.com/soCzech/TransNetV2) — model segmentasi adegan
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — alternatif terbuka klasik
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — tolok ukur evaluasi referensi
