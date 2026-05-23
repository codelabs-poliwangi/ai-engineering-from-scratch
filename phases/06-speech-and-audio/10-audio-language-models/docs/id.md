# Model Bahasa Audio — Qwen2.5-Omni, Audio Flamingo, Audio GPT-4o

> 2026 model bahasa audio alasan atas ucapan + suara lingkungan + musik. Qwen2.5-Omni-7B cocok dengan Audio GPT-4o di MMAU-Pro. Audio Flamingo Next mengalahkan Gemini 2.5 Pro di LongAudioBench. Kesenjangan antara terbuka dan tertutup pada dasarnya tertutup — kecuali pada tugas multi-audio, di mana setiap orang hampir dilakukan secara acak.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 6 · 04 (ASR), Fase 12 · 03 (Model Bahasa Visi), Fase 7 · 10 (Transformer Audio)
**Waktu:** ~45 menit

## Masalah

kamu memiliki audio berdurasi 5 detik: anjing menggonggong, seseorang berteriak "berhenti!", lalu hening. Pertanyaan berguna mencakup banyak sumbu:

- **Transkripsi.** "Apa yang dikatakan?" — wilayah ASR.
- **Penalaran semantik.** "Apakah orang tersebut dalam bahaya?" — membutuhkan pemahaman bersama tentang gonggongan + teriakan + keheningan.
- **Penalaran musik.** "Instrumen apa yang memainkan melodi?"
- **Pengambilan audio panjang.** "Di manakah dalam kuliah 90 menit ini instruktur menjelaskan gradient descent?"

Model tunggal yang menjawab semua ini dengan satu prompt adalah **model bahasa audio** (LALM / ALM). Terpisah dari ASR murni: LALM menghasilkan jawaban dalam bahasa alami dalam bentuk bebas, bukan hanya transkrip.

## Konsep

![Model bahasa audio: encoder audio + proyektor + decoder LLM](../assets/alm-architecture.svg)

### Templat tiga komponen

Setiap LALM 2026 memiliki kerangka yang sama:

1. **Encoder audio.** Whisper encoder · BEATs · CLAP · WavLM · atau encoder khusus per model.
2. **Proyektor.** Linear atau MLP yang menjembatani feature encoder audio ke dalam ruang embedding token LLM.
3. **LLM.** Dekoder berbasis Llama / Qwen / Gemma. Mengambil teks yang disisipkan + token audio; menghasilkan teks.

Training:

- **Phase 1.** Bekukan encoder + LLM; proyektor kereta hanya pada data ASR / captioning.
- **Phase 2.** Penyempurnaan penuh / LoRA pada tugas audio yang mengikuti instruksi (QA, penalaran, pemahaman musik).
- **Phase 3 (opsional).** Suara masuk/keluar menambahkan dekoder ucapan. Qwen2.5-Omni dan AF3-Chat melakukan ini.

### Peta model 2026

| Model | Tulang punggung | Pembuat enkode audio | Modalitas output | Akses |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Kustom + Bisikan | teks + ucapan | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Kustom | teks + ucapan | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | teks | NVIDIA non-komersial |
| Audio Flamingo Berikutnya | Qwen2 | AF-CLAP v2 | teks | NVIDIA non-komersial |
| SALMON | Vicuna | Bisikan + BEAT | teks | Apache-2.0 |
| LTU / LTU-AS | Lama | CAV-MAE | teks | Apache-2.0 |
| GAMA | Lama | AST + Q-Mantan | teks | Apache-2.0 |
| Gemini 2.5 Flash/Pro (tertutup) | kembar | kepemilikan | teks + ucapan | API |
| Audio GPT-4o (tertutup) | GPT-4o | kepemilikan | teks + ucapan | API |

### Pemeriksaan realitas tolok ukur (2026)

**MMAU-Pro.** 1800 pasang QA yang mencakup ucapan/suara/musik/campuran. Subset multi-audio disertakan.

| Model | Secara keseluruhan | Pidato | Suara | Musik | Multi-audio |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73,4% | 51,9% | 64,9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73,4% | 50,5% | 64,9% | 21,2% |
| Audio GPT-4o | 52,5% | — | — | — | 26,5% |
| Qwen2.5-Omni-7B | 52,2% | 57,4% | 47,6% | 61,5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Berikutnya | SOTA di LongAudioBench | — | — | — | — |**Kolom multi-audio memberatkan semua orang.** Peluang acak pada 4 pilihan pilihan ganda = 25%; sebagian besar model mendapat skor sekitar sana. LALM masih kesulitan membandingkan dua klip.

### Dimana LALM berguna pada tahun 2026

- **Audit kepatuhan terhadap rekaman pusat panggilan.** "Apakah agen menyebutkan pengungkapan yang diwajibkan?"
- **Aksesibilitas.** Menjelaskan peristiwa suara kepada pengguna tunarungu (bukan hanya transkripsi).
- **Moderasi konten.** Mendeteksi bahasa kekerasan + nada mengancam + konteks latar belakang.
- **Podcast / bab pertemuan.** Ringkasan semantik, bukan hanya giliran pembicara.
- **Analisis katalog musik.** "Temukan semua lagu dengan perubahan kunci bagian B."

### Jika TIDAK (belum) berguna

- Teori musik yang terperinci (di bawah level akord).
- Penalaran yang dikaitkan dengan pembicara dalam percakapan panjang (menurun menjadi 10 menit).
- Perbandingan multi-audio (22-26% hampir di atas acak).
- Alasan streaming waktu nyata (sebagian besar adalah inference batch offline).

## Build

### Langkah 1: kueri Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Langkah 2: pola proyektor

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

Itu saja. Proyektor biasanya terdiri dari 1-3 layer linier. Melatihnya pada pasangan ASR (audio → transkrip) adalah tugas dalih Phase-1.

### Langkah 3: melakukan benchmarking MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

Laporkan per kategori (ucapan / suara / musik / multi-audio) secara terpisah. Nomor agregat menyembunyikan tempat model gagal.

## Pakai

| Tugas | pilihan 2026 |
|------|-----------|
| QA audio bentuk bebas (terbuka) | Qwen2.5-Omni-7B |
| Terbuka terbaik pada audio panjang | Audio Flamingo Berikutnya |
| Paling baik ditutup | Gemini 2.5 Pro |
| Agen pengisi suara / pengisi suara | Qwen2.5-Audio Omni atau GPT-4o |
| Alasan musik | Audio Flamingo 3 atau 2 (AF-CLAP khusus musik) |
| Audit pusat panggilan | Gemini 2.5 Pro melalui API, dengan RAG di atas dokumen kebijakan kamu |

## Jebakan

- **Kepercayaan berlebihan pada multi-audio.** Jika tugas kamu memerlukan "klip mana yang memiliki X", performa tingkat peluang acak adalah nyata.
- **Degradasi audio panjang.** 10 menit terakhir, sebagian besar atribusi speaker model rusak. Buatlah catatan harian terlebih dahulu (Lesson 6), lalu rangkum.
- **Halusinasi saat diam.** Masalah gaya Bisikan yang sama diwarisi oleh LALM yang menggunakan encoder Whisper. Gerbang VAD.
- **Pemilahan patokan.** Entri blog vendor menyoroti kategori kasus terbaik. Jalankan sendiri subset multi-audio MMAU-Pro.

## Kirim

Simpan sebagai `outputs/skill-alm-picker.md`. Pilih LALM + subset benchmark + modalitas output (teks vs ucapan) untuk tugas pemahaman audio tertentu.

## Latihan

1. **Mudah.** Jalankan `code/main.py` untuk melihat pola proyektor mainan + perutean LALM palsu (embedding audio, token teks) → token output.
2. **Sedang.** Skor Qwen2.5-Omni-7B pada 100 item pidato MMAU-Pro. Bandingkan dengan jumlah yang dilaporkan surat kabar.
3. **Sulit.** Buat dasar teks audio minimal: encoder BEATs + proyektor 2 lapis + Llama-3.2-1B beku. Sempurnakan hanya proyektor di AudioCaps. Bandingkan dengan SALMONN di Clotho-AQA.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| LALM | Obrolan AudioGPT | Enkoder audio + proyektor + dekoder LLM. |
| Proyektor | Adaptor | Feature audio pemetaan MLP kecil ke dalam ruang embedding LLM. |
| MMAU | Tolok ukur | 10 ribu pasangan audio-QA di seluruh ucapan, suara, musik. |
| MMAU-Pro | MMAU yang lebih sulit | 1800 pertanyaan multi-audio / penalaran berat. |
| Bangku Audio Panjang | Evaluasi bentuk panjang | Klip multi-menit dengan pertanyaan semantik. |
| Suara masuk / keluar suara | Ucapan asli | Model menyerap ucapan dan mengeluarkan ucapan tanpa jalan memutar teks. |

## Bacaan Lanjutan- [Chu dkk. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — arsitektur referensi.
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — pidato-in-speech-out.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — pemimpin audio panjang terbuka.
- [NVIDIA (2026). Audio Flamingo Berikutnya](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA.
- [Tang dkk. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — pelopor encoder ganda.
- [Papan peringkat MMAU-Pro](https://mmaubenchmark.github.io/) — peringkat langsung tahun 2026.
