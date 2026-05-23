# Capstone 07 — Pipeline Penyempurnaan Ujung-ke-Ujung (Data ke SFT ke DPO untuk Dilayani)

> Model 8B yang dilatih berdasarkan data kamu sendiri, diselaraskan DPO dengan preferensi kamu sendiri, dikuantisasi, diterjemahkan secara spekulatif, dan disajikan dengan token $/1 juta yang terukur. Tumpukan terbuka 2026 adalah Axolotl v0.8, TRL 0.15, Unsloth untuk iterasi, GPTQ/AWQ/GGUF untuk kuantisasi, vLLM 0.7 dengan EAGLE-3 untuk penyajian. Puncaknya adalah menjalankan seluruh pipeline secara dapat direproduksi — YAML masuk, titik akhir dilayani keluar — dan menerbitkan kartu model berdasarkan Model Openness Framework 2026.

**Type:** Batu penjuru
**Language:** Python (pipa), YAML (konfigurasi), Bash (skrip)
**Prerequisites:** Fase 2 (ML), Fase 3 (DL), Fase 7 (Transformer), Fase 10 (LLM dari awal), Fase 11 (rekayasa LLM), Fase 17 (infrastruktur), Fase 18 (keselamatan)
**Fase yang dilakukan:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**Waktu:** 35 jam

## Masalah

Setiap tim AI yang serius pada tahun 2026 selalu menyiapkan jalur penyempurnaan. Bukan karena mereka mengirimkan model dasar terdepan, namun karena adaptasi hilir — domain SFT, DPO terhadap preferensi berlabel, draf yang disaring untuk decoding spekulatif, disajikan dengan EAGLE-3 — adalah tempat terjadinya kemenangan terukur. Axolotl v0.8 menangani konfigurasi SFT multi-GPU. TRL 0.15 menangani DPO dan GRPO. Unsloth memberi kamu iterasi GPU tunggal yang cepat. vLLM 0.7 dengan EAGLE-3 mendorong throughput decode 2-3x tanpa kehilangan kualitas. Perkakasnya berfungsi; keahliannya ada di YAML, kebersihan data, dan disiplin evaluasi.

kamu akan menjalankan basis 8B (Llama 3.3, Qwen3, atau Gemma 3) melalui SFT lalu DPO pada data khusus tugas, mengukur untuk penyajian, dan mengukur keuntungan terhadap lm-evaluation-harness, RewardBench-2, MT-Bench-v2, dan MMLU-Pro. kamu akan menghasilkan kartu model berdasarkan Kerangka Keterbukaan Model 2026. Intinya adalah reproduktifitas - satu prompt menjalankan kembali seluruh pipeline dari ujung ke ujung.

## Konsep

Pipa tersebut memiliki lima phase. **Data**: dedup (MinHash / Datatrove), filter kualitas (pengklasifikasi gaya Nemotron-CC), scrub PII, pemeriksaan kebersihan terpisah terhadap kontaminasi tolok ukur publik. **SFT**: Axolotl YAML, ZeRO-3 pada 8xH100, jadwal kosinus, urutan dikemas, 2-3 epoch. **DPO atau GRPO**: Konfigurasi TRL, 1 periode, pasangan preferensi, baik yang diberi label manusia atau yang dinilai model, penyetelan beta. **Kuantisasi**: GPTQ + AWQ + GGUF untuk fleksibilitas penerapan. **Melayani**: vLLM 0.7 dengan kepala spekulatif EAGLE-3 (atau SGLang dengan SpecForge), penerapan K8, HPA saat menunggu antrian.

Ablasi adalah hasil yang dapat dicapai: SFT saja vs SFT+DPO vs SFT+GRPO pada tiga tolok ukur tugas spesifik. Metrik penyajian: token pada batch 1/8/32, tingkat penerimaan EAGLE-3, $/1 juta token. Evaluasi keamanan: Tingkat kelulusan Llama Guard 4. Kartu model: evaluasi bias, benih reproduktifitas, perizinan data.

## Arsitektur

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## Tumpukan

- Data: Datatrove untuk dedup, pengklasifikasi Nemotron-CC untuk kualitas, Presidio untuk PII
- Basis: Llama 3.3 8B, Qwen3 14B, atau Gemma 3 12B
- SFT: Axolotl v0.8 dengan ZeRO-3, Flash Attention 3, urutan yang dikemas
- Penyetelan preferensi: TRL 0,15 untuk DPO atau GRPO; Unsloth untuk iterasi GPU tunggal
- Kuantisasi: GPTQ (Marlin), AWQ, GGUF melalui llama.cpp
- Melayani: vLLM 0.7 dengan decoding spekulatif EAGLE-3 (atau SGLang 0.4 + SpecForge)
- Evaluasi: lm-evaluation-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro
- Evaluasi keamanan: Llama Guard 4, ShieldGemma-2
- Infrastruktur: Kubernetes + plugin perangkat NVIDIA, HPA pada metrik antrian-tunggu
- Observabilitas: W&B untuk training, Langfuse untuk inference

## Bangun1. **Pipa data.** Jalankan dedup Datatrove pada korpus mentah. Terapkan pengklasifikasi kualitas gaya Nemotron-CC. Presidio menghapus PII. Tulis pemisahan train/val dengan seed eksplisit.

2. **Pemeriksaan kontaminasi.** Untuk setiap pemisahan validasi, hitung MinHash terhadap set pengujian MMLU-Pro, MT-Bench-v2, RewardBench-2. Tolak segala tumpang tindih.

3. **Axolotl SFT.** YAML dengan ZeRO-3, FA3, pengepakan urutan. 2-3 zaman pada 8xH100. Masuk ke W&B.

4. **TRL DPO / GRPO.** Ambil pos pemeriksaan SFT, jalankan satu epoch DPO pada pasangan preferensi (atau GRPO dengan hadiah yang dapat diverifikasi dalam matematika/code). Sapu beta.

5. **Kuantisasi.** Menghasilkan tiga kuantitas: GPTQ-INT4-Marlin, AWQ-INT4, GGUF-Q4_K_M untuk llama.cpp. Catat ukuran dan throughput nominal.

6. **Sajikan dengan decoding spekulatif.** Konfigurasi vLLM 0.7 dengan draft head EAGLE-3 yang dilatih melalui Red Hat Speculators. Ukur tingkat penerimaan dan latensi ekor pada batch 1/8/32. Laporkan token $/1 juta vs Anthropic/OpenAI pada evaluasi yang sama.

7. **Matrix eval.** Jalankan lm-eval-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro di basis, khusus SFT, SFT+DPO, SFT+GRPO. Menghasilkan sebuah tabel.

8. **Eval keamanan.** Tingkat kelulusan Llama Guard 4 pada set pengembangan. Filter output ShieldGemma-2.

9. **Kartu model.** Templat Kementerian Keuangan 2026: bagian data, training, evaluasi, keselamatan, lisensi, reproduktifitas dengan YAML dan komitmen SHA.

## Pakai

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## Kirim

`outputs/skill-finetuning-pipeline.md` menjelaskan penyampaiannya. Sebuah prompt tunggal menjalankan data melalui SFT melalui DPO melalui quant melalui serve melalui eval, dan mengeluarkan kartu model + titik akhir yang dilayani.

| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | Evaluasi delta vs pangkalan | Keuntungan terukur pada tugas target (MMLU-Pro, MT-Bench-v2, khusus tugas) |
| 20 | Reproduksibilitas pipeline pipa | Satu prompt dijalankan ulang dari ujung ke ujung dengan benih yang identik |
| 20 | Kebersihan data | Tingkat Dedup, cakupan scrub PII, pemeriksaan kontaminasi hijau |
| 20 | Efisiensi penyajian | token/s di bs=1/8/32, tingkat penerimaan EAGLE-3, $/1 juta token |
| 15 | Kartu model + evaluasi keselamatan | Kelengkapan Kementerian Keuangan 2026 + Tingkat kelulusan Llama Guard 4 |
| **100** | | |

## Latihan

1. Jalankan SFT saja vs SFT+DPO vs SFT+GRPO pada tolok ukur khusus tugas yang sama. Laporkan metode preferensi mana yang menang dan seberapa besar.

2. Tukar Llama 3.3 8B dengan Qwen3 14B. Ukur token $/1 juta dengan kualitas yang sesuai.

3. Ukur tingkat penerimaan EAGLE-3 pada data domain vs ShareGPT generik. Laporkan delta dan pengaruhnya terhadap anggaran latensi.

4. Menyuntikkan 1% kontaminasi (membocorkan jawaban MMLU-Pro ke dalam training data) dan menjalankan kembali evaluasi. Tonton akurasi MMLU-Pro melonjak tidak realistis. Build gerbang CI pemeriksaan kontaminasi yang dapat mendeteksi hal ini.

5. Tambahkan LoRA SFT sebagai alternatif penyempurnaan penuh. Ukur kesenjangan kualitas pada memori 10x lebih rendah.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Axolotl | "Pelatih SFT" | Pelatih terpadu berbasis YAML untuk SFT, DPO, dan distilasi |
| TRL | "Penyetel preferensi" | Pustaka Memeluk Wajah untuk DPO, GRPO, PPO di LLM |
| GRPO | "Optimization kebijakan relatif kelompok" | Resep RL DeepSeek R1 dengan hadiah yang dapat diverifikasi |
| EAGLE-3 | "Draf penguraian code spekulatif" | Draf kepala yang memprediksi N token ke depan; vLLM memverifikasi dengan model target |
| Kementerian Keuangan | "Kerangka Keterbukaan Model" | Standar 2026 untuk rilis model penilaian pada data, code, lisensi |
| Pemeriksaan kontaminasi | "Kebersihan terpisah" | Deteksi kebocoran set pengujian ke dalam training berbasis MinHash |
| Tingkat penerimaan | "Metrik EAGLE / MTP" | Sebagian kecil token yang dirancang yang diterima model target |## Bacaan Lanjutan

- [Dokumentasi Axolotl](https://axolotl-ai-cloud.github.io/axolotl/) — referensi pelatih SFT / DPO
- [Dokumentasi TRL](https://huggingface.co/docs/trl) — Implementasi referensi DPO dan GRPO
- [Unsloth](https://github.com/unslothai/unsloth) — referensi iterasi GPU tunggal
- [Makalah DeepSeek R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — Metodologi GRPO
- [dokumentasi vLLM + EAGLE-3](https://docs.vllm.ai) — tumpukan penyajian referensi
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — pelatih decoding spekulatif alternatif
- [Model Openness Framework 2026](https://isocpp.org/) — standar penilaian rilis terbuka
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — pelari eval kanonik
