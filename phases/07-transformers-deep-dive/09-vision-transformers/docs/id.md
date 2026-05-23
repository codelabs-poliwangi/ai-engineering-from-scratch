# Transformer Visi (ViT)

> Gambar adalah kisi-kisi tambalan. Sebuah kalimat adalah sekumpulan token. Transformer yang sama memakan keduanya.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 7 · 05 (Trafo Penuh), Fase 4 · 03 (CNN), Fase 4 · 14 (Intro Vision Transformers)
**Waktu:** ~45 menit

## Masalah

Sebelum tahun 2020, visi komputer berarti konvolusi. Setiap SOTA di ImageNet, COCO, dan benchmark deteksi menggunakan tulang punggung CNN. Transformers untuk bahasa.

Dosovitskiy dkk. (2020) - "Sebuah Gambar Bernilai 16x16 Kata" - menunjukkan bahwa kamu dapat menghilangkan konvolusi sepenuhnya. Iris gambar menjadi patch berukuran tetap, proyeksikan setiap patch secara linear ke dalam embedding, masukkan urutannya ke encoder Transformer vanilla. Pada skala yang memadai (pra-training ImageNet-21k atau lebih besar), ViT cocok atau mengalahkan model berbasis ResNet.

ViT adalah awal dari pola yang lebih luas pada tahun 2026: satu arsitektur, banyak modalitas. Bisikan menandai audio. ViT memberi token pada gambar. Token tindakan untuk robotika. Token piksel untuk video. Trafo tidak peduli - berikan secara berurutan dan ia akan belajar.

Pada tahun 2026, ViT dan turunannya (DeiT, Swin, DINOv2, ViT-22B, SAM 3) memiliki sebagian besar vision. CNN masih unggul dalam perangkat edge dan tugas-tugas yang sensitif terhadap latensi. Segala sesuatu yang lain memiliki ViT di suatu tempat di tumpukan.

## Konsep

![Gambar → tambalan → token → Transformer](../assets/vit.svg)

### Langkah 1 — melakukan patchify

Pisahkan gambar `H × W × C` menjadi `N × (P·P·C)` rangkaian patch datar. Penyiapan umum: `224 × 224` image, `16 × 16` patch → 196 patch yang masing-masing berisi 768 nilai.

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Ukuran tambalan adalah tuasnya. Tambalan yang lebih kecil = lebih banyak token, resolusi lebih baik, biaya attention kuadrat. Tambalan yang lebih besar = lebih kasar, lebih murah.

### Langkah 2 — embedding linier

Sebuah matrix yang dipelajari memproyeksikan setiap patch datar ke `d_model`. Setara dengan konvolusi ukuran kernel `P` dan langkah `P`. Di PyTorch ini secara harfiah `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` — implementasi 2 baris.

### Langkah 3 — tambahkan token `[CLS]`, tambahkan embedding posisi

- Tambahkan token `[CLS]` yang dapat dipelajari. Keadaan tersembunyi terakhirnya adalah representasi gambar yang digunakan untuk klasifikasi.
- Tambahkan embeddings posisi yang dapat dipelajari (ViT-asli) atau 2D sinusoidal (varian selanjutnya).
- Pada tahun 2024+, RoPE diperluas ke 2D untuk posisinya, terkadang tanpa embedding yang jelas.

### Langkah 4 — encoder trafo standar

Tumpuk blok L `LayerNorm → Self-Attention → + → LayerNorm → MLP → +`. Identik dengan BERT. Tidak ada layer khusus penglihatan. Inilah inti pedagogi dari makalah ini.

### Langkah 5 — kepala

Untuk klasifikasi: ambil `[CLS]` keadaan tersembunyi → linier → softmax. Untuk DINOv2 atau SAM, buang `[CLS]`, gunakan embedding patch secara langsung.

### Varian yang penting

| Model | Tahun | Ubah |
|-------|------|--------|
| ViT | 2020 | Yang asli. Memperbaiki ukuran patch, attention global penuh. |
| Dewa | 2021 | Distilasi; hanya dapat dilatih di ImageNet-1k. |
| Babi | 2021 | Hierarki dengan jendela yang bergeser. Biaya sub-kuadrat tetap. |
| DINOv2 | 2023 | Diawasi sendiri (tanpa label). Feature penglihatan umum terbaik. |
| ViT-22B | 2023 | parameter 22B; hukum penskalaan berlaku. |
| SigLIP | 2023 | Pasangan bahasa ViT+, kehilangan kontrastif sigmoid. |
| SAM 3 | 2025 | Segmentasikan apa saja; ViT-Large + dekoder topeng yang dapat diminta. |

### Mengapa butuh waktu cukup lamaViT memerlukan *banyak* data untuk mencocokkan CNN karena tidak memiliki bias induktif CNN (invariansi terjemahan, lokalitas). Tanpa lebih dari 100 juta gambar berlabel atau pra-training yang diawasi sendiri dan kuat, CNN tetap unggul dalam komputasi yang sesuai. DeiT memperbaikinya pada tahun 2021 dengan trik distilasi; DINOv2 memperbaikinya secara permanen pada tahun 2023 dengan pengawasan mandiri.

## Build

Lihat `code/main.py`. Patchify stdlib murni + embedding linier + pemeriksaan kewarasan. Tanpa training — ViT pada skala realistis apa pun memerlukan PyTorch dan waktu GPU berjam-jam.

### Langkah 1: gambar palsu

Gambar RGB 24 × 24 sebagai daftar baris tupel `(R, G, B)`. Kami menggunakan patch 6×6 → 16 patch, masing-masing 108-d embed vector.

### Langkah 2: melakukan patchify

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Urutan raster: baris-mayor melintasi grid. Setiap ViT menggunakan pemesanan ini.

### Langkah 3: embedding linier

Lipat gandakan setiap patch datar dengan matrix acak `(patch_flat_size, d_model)`. Pastikan bentuk output adalah `(N_patches + 1, d_model)` setelah menambahkan `[CLS]`.

### Langkah 4: hitung parameter untuk ViT yang realistis

Cetak jumlah parameter untuk ViT-Base: 12 layer, 12 kepala, d=768, patch=16. Bandingkan dengan ResNet-50 (~25 juta). ViT-Base mendarat di ~86 juta. ViT-Besar ~307M. ViT-Besar ~632M.

## Pakai

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**Sematan DINOv2 adalah default tahun 2026 untuk feature gambar.** Bekukan tulang punggung, latih kepala mungil. Berfungsi untuk klasifikasi, pengambilan, deteksi, pemberian teks. Pos pemeriksaan DINOv2 Meta mengungguli CLIP pada setiap tugas visi non-teks.

**Pengambilan ukuran patch.** Model kecil menggunakan 16×16 (ViT-B/16). Prediksi padat (segmentasi) menggunakan 8×8 atau 14×14 (SAM, DINOv2). Model yang sangat besar menggunakan 14×14.

## Kirim

Lihat `outputs/skill-vit-configurator.md`. Keahlian ini memilih varian ViT dan ukuran patch untuk tugas vision baru berdasarkan ukuran dataset, resolusi, dan anggaran komputasi.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Verifikasi jumlah patch sama dengan `(H/P) * (W/P)` dan dimension patch datar sama dengan `P*P*C`.
2. **Sedang.** Menerapkan embedding posisi sinusoidal 2D — dua code sinusoidal independen untuk `row` dan `col` dari setiap patch, digabungkan. Masukkan mereka ke dalam PyTorch ViT kecil dan bandingkan akurasi vs embedding posisi yang dapat dipelajari di CIFAR-10.
3. **Hard.** Buat ViT 3 lapis (PyTorch), latih 1.000 gambar MNIST dengan patch 4×4. Ukur akurasi tes. Sekarang tambahkan pra-training DINOv2 pada 1.000 gambar yang sama (sederhana: cukup latih encoder untuk memprediksi embedding patch dari patch yang disamarkan). Apakah akurasi meningkat?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Tambalan | "Token pengubah visi" | Vector datar nilai piksel untuk wilayah `P × P × C` pada gambar. |
| Tambalan | "Potong + ratakan" | Iris gambar menjadi bagian yang tidak tumpang tindih, ratakan masing-masing menjadi vector. |
| `[CLS]` token | "Ringkasan gambar" | Token yang dapat dipelajari sebelumnya; embedding terakhirnya adalah representasi gambar. |
| Bias induktif | "Apa yang diasumsikan oleh model" | ViT memiliki lebih sedikit prior dibandingkan CNN; membutuhkan lebih banyak data untuk menutupi kesenjangan tersebut. |
| DINOv2 | "ViT yang diawasi sendiri" | Dilatih tanpa label menggunakan augmentasi gambar + guru momentum. Feature gambar umum terbaik pada tahun 2026. |
| SigLIP | "Penerus CLIP" | Encoder teks ViT + dilatih dengan loss kontrastif sigmoid; lebih baik daripada CLIP pada komputasi yang cocok. |
| Babi | "ViT Berjendela" | ViT hierarkis dengan attention lokal + jendela yang bergeser; sub-kuadrat. |
| Daftarkan token | "Trik 2023" | Beberapa token tambahan yang dapat dipelajari yang menyerap attention; meningkatkan feature DINOv2. |

## Bacaan Lanjutan

- [Dosovitskiy dkk. (2020). Sebuah Gambar Bernilai 16x16 Kata: Transformer untuk Pengenalan Gambar dalam Skala Besar](https://arxiv.org/abs/2010.11929) — makalah ViT.
- [Touvron dkk. (2021). Melatih Transformer & distilasi gambar yang hemat data melalui attention](https://arxiv.org/abs/2012.12877) — DeiT.
- [Liu dkk. (2021). Swin Transformer: Transformer Visi Hierarki menggunakan Jendela Bergeser](https://arxiv.org/abs/2103.14030) — Swin.
- [Oquab dkk. (2023). DINOv2: Mempelajari Feature Visual yang Kuat tanpa Pengawasan](https://arxiv.org/abs/2304.07193) — DINOv2.
- [Darcet dkk. (2023). Vision Transformers Perlu Register](https://arxiv.org/abs/2309.16588) — perbaikan token register untuk DINOv2.
