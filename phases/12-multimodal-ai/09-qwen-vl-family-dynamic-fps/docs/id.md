# Keluarga Qwen-VL dan Video FPS Dinamis

> Keluarga Qwen-VL — Qwen-VL (2023), Qwen2-VL (2024), Qwen2.5-VL (2025), Qwen3-VL (2025) — merupakan silsilah model bahasa visi terbuka yang paling berpengaruh pada tahun 2026. Setiap generasi membuat satu taruhan arsitektural yang menentukan yang akan ditiru oleh ekosistem terbuka lainnya dalam waktu dua belas bulan: resolusi dinamis asli melalui M-RoPE, pengambilan sample FPS dinamis dengan penyelarasan waktu absolut, jendela attention dalam ViT, dan format output agen terstruktur. Dengan Qwen3-VL, resepnya telah stabil: encoder 2D-RoPE-ViT dengan input rasio aspek asli, proyektor MLP ke dalam basis bahasa Qwen3 yang besar, dan tahapan training yang menekankan OCR, grounding, dan perilaku agen sebagai target kelas satu. Lesson ini membacakan keluarga secara kronologis sehingga kamu memahami mengapa setiap tombol berada di tempatnya.

**Type:** Learn
**Language:** Python (stdlib, encoder M-RoPE + sampler FPS dinamis)
**Prerequisites:** Fase 12 · 06 (patch-n'-pack)
**Waktu:** ~120 menit

## Tujuan Pembelajaran

- Hitung rotasi tiga sumbu M-RoPE (temporal, tinggi, lebar) dan jelaskan mengapa ketiganya diperlukan.
- Pilih strategi pengambilan sample FPS dinamis untuk video dan pertimbangkan akurasi token per detik vs deteksi peristiwa.
- Sebutkan empat pemutakhiran generasi Qwen-VL secara berurutan dan apa saja yang diaktifkan.
- Hubungkan format output agen JSON gaya Qwen2.5-VL dan parsing panggilan alat terstruktur dari respons VLM.

## Masalah

Qwen-VL dikirimkan pada Agustus 2023 sebagai respons langsung terhadap LLaVA-1.5 dan BLIP-2. Kesenjangan yang ditargetkan oleh tim Qwen ada tiga: resolusi, video, dan output terstruktur.

Resolusi: LLaVA-1.5 berjalan pada 336x336. Cocok untuk foto, tidak berguna untuk faktur berbahasa Mandarin atau tangkapan layar spreadsheet yang padat. Inovasi pertama Qwen-VL adalah 448x448 dan output kotak pembatas yang membumi, sehingga model dapat menunjuk pada berbagai hal.

Video: Video-LLaMA menumpuk encoder per frame dan memasukkannya ke LLM. Ini berfungsi untuk klip pendek, bukan untuk video multi-menit yang sinyalnya adalah sumbu temporal. Tim Qwen menginginkan satu pembuat enkode yang memahami waktu.

Output terstruktur: LLaVA mengeluarkan teks bentuk bebas. Agen membutuhkan JSON. Qwen-VL dilatih tentang format output JSON eksplisit termasuk koordinat kotak pembatas sebagai teks.

Setiap generasi Qwen-VL memperluas salah satu dari tiga sumbu ini.

## Konsep

### Qwen-VL (Agustus 2023)

Generasi pertama: OpenCLIP ViT-bigG/14 sebagai encoder (params 2,5B), Q-Former yang kompatibel dengan LLama (1 langkah dengan 256 kueri), basis Qwen-7B. Kontribusi:

- Resolusi 448x448 (kemudian SOTA untuk VLM terbuka).
- Grounding: dilatih pada pasangan gambar-teks dengan output token koordinat eksplisit. "Kucing itu ada di <box>(112, 204), (280, 344)</box>".
- Training multibahasa Cina + Inggris sejak awal.

Tolok ukur pada saat itu: bersaing dengan GPT-4V dalam bahasa Inggris, dominan dalam bahasa Mandarin. Pengawasan landasan adalah berita utama sebenarnya.

### Qwen2-VL (September 2024) — M-RoPE dan resolusi asli

Qwen2-VL menggantikan tumpukan resolusi tetap + Q-Former dengan encoder ViT resolusi dinamis asli. Perubahan penting:- Resolusi dinamis asli. ViT menerima HxW apa pun yang habis dibagi 28 (patch 14 dengan penggabungan spasial 2x). Gambar pada 1120x672 (tambalan gabungan 40x24) menghasilkan 960 token visual. Tidak ada pengubahan ukuran, tidak ada ubin, tidak ada thumbnail.
- M-RoPE (Tali Multimodal). Setiap token membawa posisi 3D (t, h, w) bukan 1D. Untuk gambar t=0, untuk video t = frame_index. RoPE memutar vector kueri/kunci dengan frekuensi per sumbu. Tidak ada tabel embedding posisi.
- Proyektor MLP. Jatuhkan Q-Mantan; gunakan MLP 2 lapis pada token patch yang digabungkan.
- Video dengan FPS dinamis. Video diambil sampelnya pada 1-2 FPS secara default, namun model menerima jumlah frame yang berubah-ubah.

Hasil: Qwen2-VL-7B cocok dengan GPT-4o pada beberapa tolok ukur multimodal dan mengalahkannya di DocVQA (94,5 vs 88,4). Perubahan arsitektur adalah langkah yang menentukan.

### Qwen2.5-VL (Februari 2025) — FPS dinamis + waktu absolut

Pergeseran besar Qwen2.5-VL adalah video. FPS Dinamis bukan sekadar "mengambil sample lebih banyak bingkai bila diperlukan". Makalah ini diformalkan:

- Token waktu absolut. Alih-alih indeks posisi (bingkai 0, 1, 2...), gunakan stempel waktu sebenarnya. "Pada 0:04, kucing itu melompat." Model ini melihat token `<time>0.04</time>` disisipkan dengan token bingkai.
- FPS Dinamis. Ambil sample pada 1 FPS untuk rekaman lambat, 4+ FPS untuk aksi. Pengguna atau pelatih memilih; M-RoPE beradaptasi.
- Jendela attention di ViT. Attention spasial diberi jendela (lokal dalam blok) untuk throughput; attention global setiap beberapa layer.
- Format output JSON eksplisit. Dilatih tentang data panggilan alat: "{\"tool\": \"click\", \"coords\": [380, 220]}". Agen siap di luar kotak.
- Penskalaan MRoPE-v2. Skala posisi dengan ukuran input maksimal sehingga video berdurasi 10 menit tidak kehabisan rentang frekuensi.

Tolok ukur: Qwen2.5-VL-72B mengalahkan GPT-4o di sebagian besar tolok ukur video, cocok dengan Gemini 2.0 pada dokumen, dan menetapkan SOTA model terbuka untuk landasan GUI (ScreenSpot: akurasi 84% vs 38% untuk GPT-4o).

### Qwen3-VL (November 2025)

Qwen3-VL adalah peningkatan bertahap yang mengkonsolidasikan daripada menciptakan kembali: tulang punggung LLM yang lebih besar (Qwen3-72B), training data yang diperluas, peningkatan OCR, penalaran yang lebih kuat melalui "mode berpikir" Qwen3. ViT dan M-RoPE tetap tinggal. Makalah ini berfokus pada peningkatan data dan training dibandingkan arsitektur.

Kesimpulannya: pada tahun 2025 arsitektur Qwen-VL telah stabil. Generasi tambahan menskalakan komputasi dan data, bukan primitif.

### M-RoPE secara matematis

RoPE Klasik memutar kueri `q` dimension `d` berdasarkan posisi `m` menggunakan koordinat berpasangan:

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE membagi redup yang tersembunyi menjadi tiga pita. Ucapkan `d = 96`. Tetapkan 32 redup untuk temporal, 32 untuk tinggi, 32 untuk lebar. Setiap pita berputar berdasarkan posisi sumbunya masing-masing. Patch pada (t=5, h=10, w=20) mendapat rotasi `R_t(5)`, `R_h(10)`, `R_w(20)` yang diterapkan pada tiga pitanya.

Token teks menggunakan `t = text_index, h = 0, w = 0` (atau pilihan yang dinormalisasi), menjaga kompatibilitas. Bingkai video menggunakan `t = frame_time, h = row, w = col`. Gambar tunggal menggunakan `t = 0`.

Keuntungannya: pengkodean satu posisi menangani teks, gambar, dan video tanpa code percabangan atau tabel posisi yang berbeda.

### Logika pengambilan sample FPS dinamis

Mengingat video berdurasi `T` detik dan anggaran token target `B`:1. Hitung FPS maksimum yang kamu mampu: `fps_max = B / (T * tokens_per_frame)`.
2. Pilih target FPS dari `{1, 2, 4, 8}` yang memenuhi `fps <= fps_max`.
3. Jika gerakannya tinggi (heuristik aliran optik atau permintaan pengguna eksplisit), pilih FPS yang lebih tinggi. Jika gerakannya rendah, pilih lebih rendah.
4. Sample secara seragam pada FPS yang dipilih; masukkan token `<time>t</time>` di antara frame.

Qwen2.5-VL melatih logika ini secara implisit; pada inference, pengguna mengontrol melalui parameter `fps`. Urutan tindakan 60 detik pada 4 FPS dengan 81 token per frame = 19440 token, dapat dikelola dalam konteks 32k.

### Output agen terstruktur

Training agen Qwen2.5-VL secara eksplisit menargetkan panggilan alat terstruktur:

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

Parsing bersifat deterministik: JSON.parse pada output model. Bandingkan dengan "klik di (1024, 512)" bentuk bebas yang memerlukan penanganan regex dan ambiguitas. Pergeseran inilah yang menyebabkan skor ScreenSpot Qwen2.5-VL melonjak dari 55% Qwen2-VL menjadi 84%.

## Pakai

`code/main.py` mengimplementasikan:

- Perhitungan posisi M-RoPE untuk urutan pencampuran teks, patch gambar, dan bingkai video.
- Sampler FPS Dinamis: diberikan (durasi, anggaran, level_gerakan), pilih FPS dan keluarkan stempel waktu bingkai.
- Parser output JSON mainan Qwen2.5-VL yang menangani respons panggilan alat dengan bidang koordinat.

Jalankan, lalu rasakan perbedaannya saat kamu menukar FPS tetap dengan FPS dinamis pada video berdurasi 5 menit.

## Kirim

Lesson ini menghasilkan `outputs/skill-qwen-vl-pipeline-designer.md`. Dengan adanya tugas video (pemantauan, agen, pengenalan tindakan, aksesibilitas), tugas tersebut memancarkan konfigurasi Qwen2.5-VL (anggaran bingkai, strategi FPS, tanda jendela attention, mode output agen) dan perkiraan latensi. Gunakan ini setiap kali kamu menerapkan model keluarga Qwen-VL untuk produk video.

## Latihan

1. Hitung rotasi M-RoPE untuk patch pada (t=3, h=5, w=7) dengan 48 tersembunyi (16 per band, basis theta 10.000). Tunjukkan sudut rotasi untuk tiga pasang pertama di setiap pita.

2. Rekaman kamera keamanan berdurasi 10 menit pada 1 FPS menghasilkan berapa frame? Pada resolusi 384 dengan 3x pool, berapa total tokennya? Apakah konteks 32k default Qwen2.5-VL menanganinya?

3. Pilih FPS untuk reli tenis 30 detik vs demo resep 30 detik vs rekaman agen UI 30 detik. Benarkan masing-masing dengan logika FPS dinamis.

4. Qwen2.5-VL menjatuhkan Q-Former seluruhnya. Mengapa MLP sederhana berhasil pada tahun 2025 tetapi tidak pada tahun 2023? (Petunjuk: skala data dan kualitas encoder.)

5. Parsing tiga output panggilan alat Qwen2.5-VL JSON ke dalam dicts Python. Apa yang gagal untuk format JSON yang salah dan strategi pemulihan apa yang direkomendasikan oleh buku masak Qwen?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| M-Tali | "Tali Multimodal" | Embedding posisi putar 3D dengan pita temporal, tinggi, dan lebar di redup tersembunyi |
| FPS Dinamis | "Pengambilan sample cerdas" | Rasio pengambilan sample bingkai dipilih per video berdasarkan gerakan, durasi, dan anggaran token |
| Token waktu absolut | "Token stempel waktu" | `<time>t</time>` disisipkan dalam urutan sehingga model melihat detik sebenarnya bukan indeks bingkai |
| Attention jendela | "Attention lokal" | Attention spasial terbatas pada jendela kecil untuk kecepatan; attention global ditambahkan secara berkala |
| Output agen terstruktur | "Modus JSON" | Pengawasan training data mengajarkan VLM untuk memancarkan JSON yang dapat diurai dengan koordinat dan nama alat |
| min_piksel / maks_piksel | "Batas resolusi" | Qwen2.5-VL per permintaan mengontrol jumlah piksel total yang membatasi dan karenanya jumlah token |
| Pembumian | "Tunjukkan itu" | Menghasilkan koordinat kotak pembatas sebagai token teks; digunakan sejak Qwen-VL v1 |

## Bacaan Lanjutan

- [Bai dkk. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang dkk. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Tim Qwen — Laporan Teknis Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Tim Qwen — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu dkk. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
