# Pengenalan Ucapan (ASR) — CTC, RNN-T, Attention

> Pengenalan ucapan adalah klasifikasi audio pada setiap langkah waktu, direkatkan oleh model urutan yang mengetahui bahasa Inggris dan keheningan. CTC, RNN-T, dan attention adalah tiga cara untuk melakukannya. Pilih satu dan pahami alasannya.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 6 · 02 (Spektogram & Mel), Fase 5 · 08 (CNN & RNN untuk Teks), Fase 5 · 10 (Attention)
**Waktu:** ~45 menit

## Masalah

kamu memiliki klip 16 kHz berdurasi 10 detik. kamu ingin string: "nyalakan lampu dapur". Tantangannya bersifat struktural: bingkai audio tidak sejajar dengan karakter. Kata "oke" mungkin membutuhkan waktu 200 ms atau 1200 ms. Keheningan menandai ucapannya. Beberapa fonem lebih panjang dibandingkan fonem lainnya. Jumlah token output tidak diketahui sebelumnya.

Tiga formulasi memecahkan masalah ini:

1. **CTC (Connectionist Temporal Classification).** Memancarkan probabilitas token per frame termasuk *kosong* khusus. Ciutkan berulang dan kosong pada waktu dekode. Non-autoregresif, cepat. Digunakan oleh wav2vec 2.0, MMS.
2. **RNN-T (Transduser Jaringan Neural Berulang).** Jaringan gabungan memprediksi token berikutnya berdasarkan bingkai encoder dan token sebelumnya. Dapat dialirkan. Digunakan oleh ASR pada perangkat Google, NVIDIA Parkit.
3. **Encoder-decoder attention.** Encoder mengompresi audio ke status tersembunyi, decoder melakukan tindakan silang untuk menghasilkan token secara otomatis. Digunakan oleh Whisper, SeamlessM4T.

Pada tahun 2026, SOTA WER pada LibriSpeech test-clean adalah 1,4% (Parakeet-TDT-1.1B, NVIDIA) dan 1,58% (Whisper-Large-v3-turbo). Perbedaannya kecil; perbedaan penerapannya sangat besar.

## Konsep

![Tiga formulasi ASR: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**Intuisi CTC.** Biarkan encoder mengeluarkan distribusi tingkat bingkai `T` melalui token `V+1` (karakter V + kosong). Untuk string target `y` dengan panjang `U < T`, penyelarasan bingkai apa pun yang diciutkan menjadi `y` akan dihitung. Loss CTC mencakup seluruh penyelarasan tersebut. Inference: argmax per-frame, ciutkan berulang, hapus bagian yang kosong.

Keuntungan: non-autoregresif, dapat dialirkan, tidak melihat ke depan. Kekurangan: *asumsi independensi bersyarat* — setiap prediksi frame tidak bergantung pada frame lainnya, sehingga tidak ada model bahasa internal. Perbaiki dengan LM eksternal melalui pencarian sinar atau fusi dangkal.

**Intuisi RNN-T.** Menambahkan jaringan *prediktor* yang embed riwayat token dan *penggabung* yang menggabungkan status prediktor dengan bingkai encoder ke dalam distribusi gabungan melalui `V+1` (`+1` adalah null / no-emit). Secara eksplisit memodelkan ketergantungan bersyarat yang diabaikan CTC. Dapat dialirkan karena setiap langkah hanya mengkondisikan frame sebelumnya dan token sebelumnya.

Keuntungan: dapat dialirkan + LM internal. Kekurangan: training lebih kompleks dan haus memori (kisi kehilangan 3D); Kernel loss RNN-T adalah keseluruhan kategori perpustakaan tersendiri.

**Attention encoder-decoder.** Encoder (6-32 layer Transformer) pada bingkai log-mel. Dekoder (6-32 layer Transformer) menangani output encoder secara silang untuk menghasilkan token secara otomatis. Tidak ada batasan penyelarasan — attention dapat melihat ke mana saja di audio. Tidak dapat dialirkan kecuali kamu membatasi attention (chunked Whisper-Streaming, 2024).

Keuntungan: kualitas tertinggi pada ASR offline, mudah dilatih dengan perkakas seq2seq standar. Kekurangan: latensi autoregresif sebanding dengan panjang output; tidak dapat streaming tanpa rekayasa.

### WER: nomor satu**Tingkat Kesalahan Kata** = `(S + D + I) / N`, dengan S=penggantian, D=penghapusan, I=penyisipan, N=jumlah kata referensi. Cocok dengan distance edit Levenshtein pada tingkat kata. Lebih rendah lebih baik. WER di atas 20% umumnya tidak dapat digunakan; di bawah 5% adalah paritas manusia untuk ucapan yang dibaca. Angka tahun 2026 pada tolok ukur standar:

| Model | Tes LibriSpeech-bersih | Tes LibriSpeech-lainnya | Ukuran |
|-------|------------------------|------------------------|------|
| Parkit-TDT-1.1B | 1,40% | 2,78% | 1.1B parameter |
| Bisikan-Besar-v3-turbo | 1,58% | 3,03% | 809M |
| Kilat Kenari-1B | 1,48% | 2,87% | 1B |
| M4T v2 yang mulus | 1,7% | 3,5% | 2.3B |

Semua ini berbasis encoder-decoder atau RNN-T. Sistem CTC murni (wav2vec 2.0) berada pada kisaran 1,8–2,1% pada pengujian bersih.

## Build

### Langkah 1: dekode CTC serakah

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

Dua aturan: ciutkan pengulangan berturut-turut, kosongkan. Contoh: `a a _ _ a b b _ c` → `a a b c`.

### Langkah 2: CTC pencarian sinar

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

Produksi menggunakan pencarian balok pohon awalan dengan fusi LM; inilah kerangka konseptualnya.

### Langkah 3: APA

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### Langkah 4: inference terhadap Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

One-liner untuk ASR umum terkuat pada tahun 2026. Berjalan pada GPU 24 GB pada ~20× realtime.

### Langkah 5: streaming dengan Parkit atau wav2vec 2.0

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

Streaming ASR memerlukan attention encoder yang terpotong dan status carryover; gunakan perpustakaan yang mendukungnya (NeMo untuk Parkit, pipa `transformers` dengan `chunk_length_s`).

## Pakai

Tumpukan tahun 2026:

| Situasi | Pilih |
|-----------|------|
| Bahasa Inggris, offline, kualitas maksimal | Bisikan-besar-v3-turbo |
| Multibahasa, tangguh | MulusM4T v2 |
| Streaming, latensi rendah | Parkit-TDT-1.1B atau Riva |
| Edge, seluler, latensi <500 ms | Bisikan-Tiny terkuantisasi atau Moonshine (2024) |
| Bentuk panjang | Whisper dengan chunking berbasis VAD (WhisperX) |
| Khusus domain (medis, hukum) | Sempurnakan penggabungan wav2vec 2.0 + domain LM |

## Kesalahan yang masih dikirimkan pada tahun 2026

- **Tanpa VAD.** Menjalankan Bisikan dalam diam menghasilkan halusinasi ("Terima kasih sudah menonton!"). Selalu gerbang dengan VAD.
- **Karakter vs kata vs subkata WER.** Laporkan WER tingkat kata *setelah* normalisasi (huruf kecil, tanda baca dihilangkan).
- **Penyimpangan ID Bahasa.** LID otomatis Whisper salah merutekan klip berisik ke bahasa Jepang atau Welsh; paksa `language="en"` bila kamu mengetahuinya.
- **Klip panjang tanpa potongan.** Whisper memiliki jendela berdurasi 30 detik. Gunakan `chunk_length_s=30, stride=5` untuk waktu yang lebih lama.

## Kirim

Simpan sebagai `outputs/skill-asr-picker.md`. Pilih model, strategi decoding, chunking, dan fusi LM untuk target penerapan tertentu.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Ia dengan rakus menerjemahkan output CTC buatan tangan dan menghitung WER berdasarkan referensi.
2. **Medium.** Terapkan penelusuran berkas pohon prefiks pada Langkah 2 dengan benar (perhitungkan aturan penggabungan kosong). Bandingkan dengan serakah pada 10 contoh dataset sintetis.
3. **Sulit.** Gunakan `whisper-large-v3-turbo` di [LibriSpeech test-clean](https://www.openslr.org/12). Hitung WER pada 100 ucapan pertama. Bandingkan dengan angka yang dipublikasikan.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| CTC | Loss token kosong | Marjinal atas semua penyelarasan frame-to-token; non-AR. |
| RNN-T | Kehilangan streaming | CTC + prediktor token berikutnya; menangani urutan kata. |
| Attention enc-des | Gaya berbisik | Encoder + decoder lintas kehadiran; kualitas offline terbaik. |
| ADA | Nomor yang kamu laporkan | `(S+D+I)/N` pada tingkat kata. |
| Kosong | Kekosongan | Token khusus di CTC menandakan "tidak ada emisi bingkai ini". |
| Fusi LM | Model bahasa eksternal | Tambahkan masalah log LM berbobot selama pencarian berkas. |
| VAD | Gerbang Keheningan | Detektor aktivitas suara; memangkas non-ucapan. |

## Bacaan Lanjutan

- [Kuburan dkk. (2006). Klasifikasi Temporal Koneksionis](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — makalah CTC.
- [Kuburan (2012). Transduksi Urutan dengan RNN](https://arxiv.org/abs/1211.3711) — makalah RNN-T.
- [Radford dkk. / OpenAI (2022). Whisper: Pengenalan Ucapan yang Kuat melalui Pengawasan Lemah Berskala Besar](https://arxiv.org/abs/2212.04356) — makalah kanonik tahun 2022; ekstensi v3-turbo pada tahun 2024.
- [NVIDIA NeMo — kartu Parkit-TDT](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — Pemimpin Papan Peringkat ASR Terbuka 2026.
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — benchmark langsung untuk 25+ model.
