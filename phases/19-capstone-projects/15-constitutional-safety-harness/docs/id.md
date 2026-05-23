# Capstone 15 — Tali Pengaman Konstitusional + Jangkauan Tim Merah

> Pengklasifikasi Konstitusional Anthropic, Llama Guard 4 dari Meta, ShieldGemma-2 dari Google, Keamanan Konten Nemotron 3 dari NVIDIA, dan X-Guard untuk cakupan multibahasa menentukan tumpukan pengklasifikasi keamanan tahun 2026. garak, PyRIT, NVIDIA Aegis, dan promptfoo menjadi alat evaluasi permusuhan standar. NeMo Guardrails v0.12 mengikatnya ke dalam jalur produksi. Batu penjuru ini menyatukan semuanya: tali pengaman berlapis di sekitar aplikasi target, agen tim merah otonom yang menjalankan 6+ kelompok serangan, dan proses kritik diri konstitusional yang menghasilkan delta tidak berbahaya yang terukur.

**Type:** Batu penjuru
**Language:** Python (pipeline keselamatan, tim merah), YAML (konfigurasi kebijakan)
**Prerequisites:** Fase 10 (LLM dari awal), Fase 11 (rekayasa LLM), Fase 13 (peralatan), Fase 14 (agen), Fase 18 (etika, keselamatan, penyelarasan)
**Fase yang dilakukan:** P10 · P11 · P13 · P14 · P18
**Waktu:** 25 jam

## Masalah

Batasan keselamatan LLM pada tahun 2026 bukanlah apakah pengklasifikasi berfungsi (secara kasar, memang berfungsi) tetapi bagaimana menyusunnya dengan benar di sekitar aplikasi produksi tanpa menolak secara berlebihan atau meninggalkan lubang yang jelas. Llama Guard 4 menangani pelanggaran kebijakan Inggris. X-Guard (132 bahasa) menangani jailbreak multibahasa. ShieldGemma-2 menangkap injeksi cepat berbasis gambar. Keamanan Konten NVIDIA Nemotron 3 mencakup kategori perusahaan. Pengklasifikasi Konstitusional Anthropic adalah pendekatan terpisah yang digunakan selama training dan bukan saat bertugas.

Evolusi serangan juga penting. PAIR dan TAP mengotomatiskan penemuan jailbreak. GCG menjalankan serangan sufiks berbasis gradient. Serangan multi-turn dan alih code mengeksploitasi memori agen. Setiap LLM yang diterapkan memerlukan tim merah — garak dan PyRIT adalah pendorong kanoniknya — ditambah mitigasi yang terdokumentasi dan temuan yang diberi skor CVSS.

kamu akan memperkuat aplikasi target (baik model yang disesuaikan dengan instruksi 8B atau salah satu chatbot RAG dari batu penjuru lainnya), menjalankan 6+ kelompok serangan terhadapnya, dan menghasilkan pengukuran tidak berbahaya sebelum/sesudah.

## Konsep

Pipa pengaman terdiri dari lima layer. **Input sanitasi**: menghapus karakter dengan lebar nol, mendekode base64/rot13, menormalkan Unicode. **Layer kebijakan**: Rel NeMo Guardrails v0.12 (di luar domain, toksisitas, ekstraksi PII). **Gerbang pengklasifikasi**: Llama Guard 4 pada input, X-Guard pada non-Inggris, ShieldGemma-2 pada input gambar. **Model**: target LLM. **Filter output**: Llama Guard 4 pada output, scrub Presidio PII, penegakan kutipan jika berlaku. **Tingkat HITL**: output yang ditandai berisiko tinggi masuk ke antrean Slack.

Rentang tim merah berjalan pada penjadwal. PAIR dan TAP secara mandiri menemukan jailbreak. GCG menjalankan serangan sufiks berbasis gradient. Serangan pengkodean ASCII / base64 / rot13. Serangan multi-turn (adopsi persona, eksploitasi memori). Serangan alih code (campuran bahasa Inggris dengan bahasa Swahili atau Thailand). Setiap proses menghasilkan file temuan terstruktur dengan penilaian CVSS dan garis waktu pengungkapan.

Latihan kritik diri konstitusional adalah intervensi pada saat training. Ambil 1.000 prompt upaya yang merugikan, mintalah model tersebut menyusun tanggapannya, kritiklah terhadap konstitusi tertulis (aturan jangan merugikan), dan latih kembali lingkaran kritik tersebut. Ukur delta tidak berbahaya sebelum/sesudah pada evaluasi yang tertunda.

## Arsitektur

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## Tumpukan- Pengklasifikasi keamanan: Llama Guard 4, ShieldGemma-2, Keamanan Konten NVIDIA Nemotron 3, X-Guard
- Kerangka pagar pembatas: NeMo Guardrails v0.12 + OPA
- Pengemudi tim merah: garak (NVIDIA), PyRIT (Microsoft Azure), NVIDIA Aegis, promptfoo
- Agen jailbreak: PAIR (Chao et al., 2023), Tree-of-Attacks (TAP), akhiran GCG
- Training konstitusional: Lingkaran kritik diri gaya antropik + SFT tentang kritik
- Lulur PII: Presidio
- Target: model yang disesuaikan dengan instruksi 8B atau salah satu chatbot RAG batu penjuru lainnya

## Build

1. **Penyiapan target.** Buat model yang disesuaikan dengan instruksi 8B di vLLM (atau gunakan kembali chatbot RAG dari batu penjuru lain). Ini adalah aplikasi yang sedang diuji.

2. **Pembungkus pipa pengaman.** Pasang pipa lima lapis di sekeliling target. Pastikan setiap layer dapat diamati secara individual (rentang per layer di Langfuse).

3. **Cakupan pengklasifikasi.** Muat Llama Guard 4, X-Guard (multibahasa), ShieldGemma-2 (gambar). Jalankan masing-masing pada set kecil berlabel untuk menetapkan garis dasar.

4. **Penjadwal tim merah.** Jadwal garak, PyRIT, agen PAIR, agen TAP, pelari GCG, penyerang multi-turn, dan penyerang alih code. Masing-masing berjalan pada antrian terpisah.

5. **Rangkaian serangan.** Enam jenis serangan: (1) Jailbreak otomatis PAIR, (2) Pohon serangan TAP, (3) akhiran gradient GCG, (4) pengkodean ASCII / base64 / rot13, (5) persona multi-putaran, (6) peralihan code multibahasa. Laporkan tingkat keberhasilan per keluarga.

6. **Kritik diri konstitusional.** Kurasi 1.000 petunjuk upaya berbahaya. Untuk masing-masing, target menyusun tanggapan. Seorang kritikus LLM memberikan skor yang bertentangan dengan konstitusi tertulis ("jangan merugikan", "mengutip bukti", "menolak permintaan ilegal"). Anjuran di mana objek kritik ditulis ulang; target penyesuaian pada pasangan yang mendapat peningkatan kritik. Ukur sebelum/sesudah tidak berbahaya pada evaluasi yang tertunda.

7. **Pengukuran penolakan yang berlebihan.** Melacak tingkat positif palsu pada rangkaian prompt yang tidak berbahaya (misalnya, XSTest). Targetnya harus tetap membantu dalam pertanyaan-pertanyaan yang tidak berbahaya.

8. **Skor CVSS.** Untuk setiap jailbreak yang berhasil, skor pada CVSS 4.0 (vector serangan, kompleksitas, dampak). Menghasilkan garis waktu pengungkapan dan rencana mitigasi.

9. **Otomasi rentang.** Segala sesuatu di atas berjalan pada cron; temuan menulis ke antrian; peringatan regresi penolakan yang berlebihan akan menyerang Slack.

## Pakai

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## Kirim

`outputs/skill-safety-harness.md` adalah hasil yang dapat dicapai. Pipa keselamatan berlapis tingkat produksi ditambah rangkaian tim merah yang dapat direproduksi dengan delta tidak berbahaya sebelum/sesudah.

| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | Cakupan permukaan serangan | 6+ serangan keluarga dilakukan, 2+ bahasa |
| 20 | Pertukaran positif benar / positif palsu | Tingkat blok serangan vs tingkat kelulusan XSTest yang baik |
| 20 | Delta kritik diri | Sebelum/sesudah tidak berbahaya pada evaluasi yang diadakan |
| 20 | Dokumentasi dan pengungkapan | Temuan dengan skor CVSS dengan garis waktu |
| 15 | Otomatisasi dan pengulangan | Semuanya berjalan di cron dengan peringatan |
| **100** | | |

## Latihan

1. Jalankan plugin garak untuk injeksi cepat pada chatbot RAG dan bandingkan tingkat keberhasilan serangan dengan dan tanpa layer filter output.

2. Tambahkan kelompok serangan ketujuh: injeksi cepat tidak langsung melalui dokumen yang diambil. Ukur pertahanan ekstra yang diperlukan.

3. Menerapkan mode "tolak dengan bantuan": ketika pagar pembatas menghalangi, target menawarkan jawaban terkait yang lebih aman daripada penolakan datar. Ukur delta XSTest.

4. Kesenjangan cakupan multibahasa: temukan bahasa yang kinerjanya buruk bagi X-Guard. Usulkan penyempurnaan dataset yang menargetkannya.5. Jalankan kritik diri konstitusional pada model 30B dan ukur apakah delta tersebut berskala.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Keamanan berlapis | "Pertahanan mendalam" | Beberapa pagar pembatas di input, gerbang, output, HITL |
| Penjaga Llama 4 | "Pengklasifikasi keamanan Meta" | Pengklasifikasi konten input/output referensi tahun 2026 |
| PASANGAN | "Agen pembobol penjara" | Makalah (Chao dkk.) tentang penemuan jailbreak berbasis LLM |
| KETUK | "Pohon Serangan" | Varian pencarian pohon dari PAIR |
| GCG | "Gradient koordinat serakah" | Serangan sufiks permusuhan berbasis gradient |
| Kritik diri konstitusional | "Training gaya antropik" | Draf target -> skor kritik -> tulis ulang -> latih ulang |
| Tes XSST | "Perangkat pemeriksaan jinak" | Tolok ukur regresi penolakan berlebihan |
| CVSS 4.0 | "Skor tingkat keparahan" | Penilaian kerentanan standar untuk temuan keselamatan |

## Bacaan Lanjutan

- [Pengklasifikasi Konstitusi Antropik](https://www.anthropic.com/research/constitutional-classifiers) — referensi waktu training
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — pengklasifikasi input/output tahun 2026
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — gambar + keamanan multimoda
- [Keamanan Konten NVIDIA Nemotron 3](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — referensi perusahaan
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — Keamanan multibahasa 132 bahasa
- [garak](https://github.com/NVIDIA/garak) — Toolkit tim merah NVIDIA
- [PyRIT](https://github.com/Azure/PyRIT) — Kerangka kerja tim merah Microsoft
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — kerangka rel
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — makalah agen jailbreak
