# Pengaturan GPU & Awan

> Training CPU baik untuk pembelajaran. Training sebenarnya membutuhkan GPU.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 0, Lesson 01
**Waktu:** ~45 menit

## Tujuan Pembelajaran

- Verifikasi ketersediaan GPU lokal menggunakan `nvidia-smi` dan CUDA API PyTorch
- Konfigurasikan Google Colab dengan GPU T4 untuk eksperimen berbasis cloud gratis
- Perkalian matrix benchmark pada CPU vs GPU dan mengukur kecepatannya
- Perkirakan model terbesar yang sesuai dengan VRAM kamu menggunakan aturan praktis fp16

## Masalah

Sebagian besar lesson di fase 1-3 berjalan dengan baik di CPU. Namun begitu kamu mulai melatih CNN, Transformer, atau LLM (fase 4+), kamu memerlukan akselerasi GPU. Latihan yang memakan waktu 8 jam pada CPU membutuhkan waktu 10 menit pada GPU.

kamu memiliki tiga opsi: GPU lokal, GPU cloud, atau Google Colab (gratis).

## Konsep

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

## Build

### Opsi 1: GPU NVIDIA lokal

Periksa apakah kamu memilikinya:

```bash
nvidia-smi
```

Instal PyTorch dengan CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Opsi 2: Google Colab

1. Buka [colab.research.google.com](https://colab.research.google.com)
2. Waktu Proses > Ubah jenis waktu proses > GPU T4
3. Jalankan `!nvidia-smi` untuk memverifikasi

Unggah buku catatan dari kursus ini langsung ke Colab.

### Opsi 3: GPU Awan

Untuk Lambda Labs, RunPod, atau Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Tidak ada GPUnya? Tidak masalah.

Sebagian besar lesson bekerja pada CPU. Yang membutuhkan GPU akan mengatakannya dan menyertakan tautan Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Build: Patokan GPU vs CPU

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Latihan

1. Jalankan benchmark di atas dan bandingkan waktu CPU vs GPU
2. Jika kamu tidak memiliki GPU, jalankan di Google Colab dan bandingkan
3. Periksa berapa banyak memori GPU yang kamu miliki dan perkirakan model terbesar yang dapat kamu muat (aturan praktisnya: 2 byte per parameter untuk fp16)

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|----------------------|
| CUDA | "Pemrograman GPU" | Platform komputasi paralel NVIDIA yang memungkinkan kamu menjalankan code pada GPU |
| VRAM | "Memori GPU" | RAM video pada GPU, terpisah dari RAM sistem. Membatasi ukuran model. |
| fp16 | "Setengah presisi" | Floating point 16-bit, menggunakan setengah memori fp32 dengan kehilangan akurasi minimal |
| Inti Tensor | "Perangkat keras matrix cepat" | Core GPU khusus untuk perkalian matrix, 4-8x lebih cepat dari core biasa |
