# Anti-Spoofing Suara & Tanda Air Audio — ASVspoof 5, AudioSeal, WaveVerify

> Kloning suara dikirimkan lebih cepat daripada pertahanan. Sistem suara produksi tahun 2026 memerlukan dua hal: detektor (AASIST, RawNet2) yang mengklasifikasikan ucapan nyata vs palsu, dan tanda air (AudioSeal) yang tahan terhadap kompresi dan pengeditan. Kirim keduanya atau jangan kirimkan kloning suara.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 6 · 06 (Pengenalan Speaker), Fase 6 · 08 (Kloning Suara)
**Waktu:** ~75 menit

## Masalah

Tiga pertahanan terkait:

1. **Deteksi anti-spoofing / deepfake.** Diberikan klip audio, apakah itu sintetis atau asli? Tolok ukur ASVspoof (ASVspoof 2019 → 2021 → 5) adalah standar emas.
2. **Tanda air audio.** Sematkan sinyal tak kasat mata dalam audio yang dihasilkan yang nantinya dapat diekstraksi oleh detektor. AudioSeal (Meta) dan WavMark adalah opsi terbuka.
3. **Asal terotentikasi.** Penandatanganan kriptografi file audio + metadata. C2PA / Inisiatif Keaslian Konten.

Deteksi menangani musuh yang tidak mau bekerja sama. Watermarking menangani kepatuhan — audio yang dihasilkan AI harus dapat diidentifikasi. Keduanya diwajibkan pada tahun 2026.

## Konsep

![Anti-spoofing vs watermarking vs asal — tiga layer pertahanan](../assets/spoofing-watermark.svg)

### ASVspoof 5 — tolok ukur 2024-2025

Perubahan terbesar dari edisi sebelumnya:

- **Data crowdsourced** (bukan studio bersih) — kondisi realistis.
- **~2000 speaker** (vs ~100 sebelumnya).
- **32 algoritma serangan.** TTS + konversi suara + gangguan permusuhan.
- **Dua trek.** Deteksi mandiri Penanggulangan (CM); ASV (SASV) yang mampu melakukan spoofing untuk sistem biometrik.

Tercanggih di ASVspoof 5: ~7,23% EER. Pada ASVspoof 2019 LA yang lebih lama: 0,42% EER. Penerapan di dunia nyata: harapkan 5-10% EER pada klip di alam liar.

### AASIST dan RawNet2 — kelompok model deteksi

**AASIST** (2021, diperbarui hingga 2026). Grafik-attention pada feature spektral. SOTA saat ini pada tugas penanggulangan ASVspoof 5.

**RawNet2.** Front-end konvolusional melalui bentuk gelombang mentah + tulang punggung TDNN. Garis dasar yang lebih sederhana; masih kompetitif dengan fine tuning.

**Feature NeXt-TDNN + SSL.** Varian 2025: Gaya ECAPA + feature WavLM + kehilangan fokus. Mencapai EER 0,42% di ASVspoof 2019 LA.

### AudioSeal — tanda air default tahun 2024

**AudioSeal** Meta (Jan 2024, v0.2 Des 2024). Desain kunci:

- **Dilokalisasi.** Mendeteksi tanda air per bingkai pada resolusi sample 16 kHz (1/16000 dtk).
- **Generator + detektor dilatih bersama.** Generator belajar embed sinyal yang tidak terdengar; detektor belajar menemukannya melalui augmentasi.
- **Kuat.** Bertahan dalam kompresi MP3 / AAC, EQ, perpindahan kecepatan ±10%, campuran kebisingan +10 dB SNR.
- **Cepat.** Detektor berjalan pada 485× realtime; 1000× lebih cepat dari WavMark.
- **Kapasitas.** Payload 16-bit (dapat mengkodekan ID model, stempel waktu pembuatan, ID pengguna) yang dapat di-embed di setiap ucapan.

### Tanda Wav

Garis dasar terbuka pra-AudioSeal. Jaringan saraf yang dapat dibalik, 32 bit/detik. Masalah:

- Sinkronisasi brute force lambat.
- Dapat dihilangkan dengan noise Gaussian atau kompresi MP3.
- Tidak ramah waktu nyata.

### Verifikasi Gelombang (Juli 2025)

Mengatasi kelemahan AudioSeal — khususnya manipulasi temporal (pembalikan, kecepatan). Menggunakan generator berbasis FiLM + detektor Mixture-of-Experts. Kompetitif dengan AudioSeal pada serangan standar; menangani pengeditan sementara.

### Kesenjangan yang dieksploitasi oleh musuhDari AudioMarkBench: "di bawah pergeseran nada, semua tanda air menunjukkan Akurasi Pemulihan Bit di bawah 0,6, yang menunjukkan penghapusan hampir selesai." **Pergeseran nada adalah serangan universal.** Tanda air No 2026 sepenuhnya tahan terhadap modifikasi nada yang agresif. Inilah sebabnya mengapa kamu memerlukan deteksi (AASIST) bersamaan dengan watermarking.

### C2PA / Inisiatif Keaslian Konten

Bukan teknik ML — format manifes. File audio membawa metadata yang ditandatangani secara kriptografis tentang alat pembuatan, penulis, tanggal. Audobox / Mulus menggunakannya. Baik untuk asal usulnya; tidak melakukan apa pun jika aktor jahat mengkodekan ulang dan menghapus metadata.

## Build

### Langkah 1: detektor feature spektral sederhana (mainan)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

Pidato sintetik sering kali memiliki energi frekuensi tinggi yang sangat datar. Detektor produksi menggunakan AASIST, bukan ini. Tapi intuisi tetap berlaku.

### Langkah 2: Embedding + deteksi AudioSeal

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### Langkah 3: evaluasi — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Langkah 4: integrasi produksi

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

Setiap generasi mengirimkan: (1) tanda air, (2) manifes yang ditandatangani, (3) log audit yang sesuai dengan kebijakan penyimpanan.

## Pakai

| Kasus penggunaan | Pertahanan |
|----------|---------|
| Pengiriman TTS / Kloning Suara | Embedding AudioSeal pada setiap output (tidak dapat dinegosiasikan) |
| Buka kunci suara biometrik | ansambel AASIST + ECAPA; tantangan keaktifan |
| Deteksi penipuan pusat panggilan | AASIST pada 20% sample panggilan masuk |
| Keaslian podcast | Penandatanganan C2PA saat unggahan, AudioSeal jika dibuat oleh AI |
| Detektor penelitian / training | ASVspoof 5 set kereta/pengembangan/eval |

## Jebakan

- **Tanda air tanpa detektor yang pernah berjalan.** Tidak ada gunanya. Kirim detektor ke CI kamu.
- **Deteksi tanpa kalibrasi.** AASIST dilatih tentang pakaian ASVspoof LA; akurasi dunia nyata menurun. Kalibrasi pada domain kamu.
- **Kesenjangan pergeseran nada.** Pergeseran nada yang agresif menghilangkan sebagian besar tanda air. Memiliki cadangan deteksi.
- **Metadata strip-and-rehost.** C2PA dapat dilewati dengan enkode ulang. Selalu tambahkan pertahanan kriptografi + persepsi (tanda air) secara bersamaan.
- **Kehidupan sebagai deteksi.** Minta pengguna mengucapkan frasa acak. Mencegah serangan replay tetapi tidak mengkloning secara real-time.

## Kirim

Simpan sebagai `outputs/skill-spoof-defender.md`. Pilih model deteksi, tanda air, manifes asal, dan pedoman operasional untuk penerapan gen suara.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Detektor mainan + embedding/deteksi tanda air mainan pada audio sintetis.
2. **Medium.** Instal `audioseal`, sematkan payload 16-bit dalam output TTS, dekode ulang. Rusak audio dengan noise dan ukur Akurasi Pemulihan Bit.
3. **Hard.** Sempurnakan RawNet2 atau AASIST di ASVspoof 2019 LA. Ukur EER. Uji pada kumpulan klip yang dihasilkan F5-TTS — lihat bagaimana penurunan deteksi OOD.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| ASV spoof | Tolok ukur | Tantangan dua tahunan; 2024 = ASV spoof 5. |
| CM (penanggulangan) | Detektor | Pengklasifikasi: ucapan nyata vs sintetik/konversi. |
| SASV | Verifikasi pembicara + CM | Deteksi biometrik + spoof terintegrasi. |
| Segel Audio | Tanda air meta | Muatan 16-bit yang dilokalkan, 485× lebih cepat dari WavMark. |
| Akurasi Pemulihan Bit | Kelangsungan hidup tanda air | Sebagian kecil bit muatan dipulihkan setelah serangan. |
| C2PA | Manifes asal | Metadata kriptografi tentang pembuatan/pengarangan. |
| AASIST | Keluarga detektor | SOTA anti-spoofing berbasis attention grafik. |

## Bacaan Lanjutan- [Todisco dkk. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — tolok ukur saat ini.
- [Defossez dkk. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — tanda air default.
- [Chen dkk. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — Detektor MoE untuk serangan temporal.
- [Jung dkk. (2022). AASIST](https://arxiv.org/abs/2110.01200) — tulang punggung deteksi SOTA.
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — evaluasi ketahanan.
- [Spesifikasi C2PA](https://c2pa.org/spesifikasi/spesifikasi/) — format manifes asal.
