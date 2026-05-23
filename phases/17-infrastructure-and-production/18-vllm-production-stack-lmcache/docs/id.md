# Tumpukan Produksi vLLM dengan Pembongkaran LMCache KV

> Tumpukan produksi vLLM adalah referensi penerapan Kubernetes — router, mesin, dan observabilitas yang dihubungkan bersama. LMCache adalah layer pembongkaran KV yang mengekstrak cache KV dari memori GPU dan menggunakannya kembali di seluruh kueri dan mesin (CPU DRAM, lalu disk/Ceph). Konektor Offloading vLLM 0.11.0 KV (Januari 2026) menjadikannya asinkron dan dapat dicolokkan melalui Connector API (v0.9.0+). Latensi pembongkaran tidak dihadapi pengguna. LMCache berharga bahkan tanpa awalan bersama — ketika GPU kehabisan slot KV, permintaan yang didahulukan dapat dipulihkan dari CPU alih-alih menghitung ulang pra-pengisian. Tolok ukur yang dipublikasikan pada 16x H100 (80GB HBM) di 4 a3-highgpu-4g: ketika cache KV melebihi HBM, baik CPU offload asli maupun LMCache secara substansial meningkatkan throughput; pada tapak KV rendah, semua konfigurasi sesuai dengan garis dasar dengan overhead kecil.

**Type:** Learn
**Language:** Python (stdlib, mainan simulator tumpahan KV)
**Prerequisites:** Fase 17 · 04 (vLLM Melayani Internal), Fase 17 · 06 (SGLang/RadixAttention)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Diagram layer tumpukan produksi vLLM: router, mesin, pembongkaran KV, kemampuan observasi.
- Jelaskan API Konektor Pembongkaran KV (v0.9.0+) dan bagaimana jalur asinkron 0.11.0 menyembunyikan latensi pembongkaran.
- Hitung kapan LMCache CPU-DRAM membantu (KV > HBM) vs menambahkan overhead (KV cukup kecil untuk memuat HBM).
- Pilih antara offload CPU vLLM asli dan konektor LMCache mengingat batasan penerapan.

## Masalah

Penyajian vLLM kamu menampilkan GPU pada 100% HBM dengan peristiwa preemption setiap kali konkurensi meningkat. Permintaan akan dikeluarkan, dimasukkan kembali ke dalam antrean, dan kamu mengisi ulang prompt 2K token yang sama sebanyak empat kali dalam satu menit. Komputasi GPU dihabiskan untuk pengisian awal yang berlebihan; goodput jauh di bawah throughput mentah.

Menambahkan lebih banyak GPU memerlukan biaya secara linier. Menambahkan lebih banyak HBM tidak dimungkinkan. Namun CPU DRAM murah — satu soket memiliki 512 GB+ dengan latensi yang lebih buruk daripada HBM tetapi bagus untuk cache KV "hangat sementara".

LMCache mengekstrak cache KV ke DRAM CPU sehingga permintaan yang didahului pulih dengan cepat, dan prefiks berulang di seluruh mesin berbagi cache tanpa setiap mesin melakukan pra-pengisian ulang.

## Konsep

### tumpukan produksi vLLM

`github.com/vllm-project/production-stack` adalah referensi penerapan Kubernetes:

- **Router** — sadar cache (Fase 17 · 11). Mengkonsumsi acara KV.
- **Mesin** — pekerja vLLM. Satu per GPU atau per grup TP/PP.
- **Pembongkaran cache KV** — Penerapan LMCache atau konektor asli.
- **Observabilitas** — Goresan Prometheus, dasbor Grafana, jejak OTel.
- **bidang kendali** — penemuan layanan, konfigurasi, pembaruan berkelanjutan.

Dikirim sebagai grafik Helm + operator.

### API Konektor Pembongkaran KV (v0.9.0+)

vLLM 0.9.0 memperkenalkan API Konektor untuk backend cache KV yang dapat dicolokkan. Mesin kamu memindahkan blok ke konektor; konektor menyimpannya (RAM, disk, penyimpanan objek, LMCache). Permintaan memerlukan blok, konektor memuatnya kembali.

vLLM 0.11.0 (Januari 2026) menambahkan jalur offload asinkron — offload dapat terjadi di latar belakang sehingga mesin tidak memblokirnya dalam kasus umum. Latensi dan throughput end-to-end masih bergantung pada bentuk weight kerja, tingkat pencapaian cache KV, dan tekanan sistem; Catatan vLLM sendiri menyatakan bahwa pembongkaran kernel khusus dapat menurunkan throughput pada tingkat hit yang rendah dan bahwa penjadwalan async telah diketahui memiliki masalah interaksi dengan decoding spekulatif.

### Pembongkaran CPU asli vs LMCache

**Pembongkaran CPU vLLM asli**: mesin-lokal. Menyimpan blok KV di RAM host. Cepat diterapkan, tanpa lompatan jaringan. Tidak melintasi mesin.**Konektor LMCache**: skala cluster. Menyimpan blok di server LMCache bersama (tingkat CPU DRAM + Ceph/S3). Blok dapat diakses oleh mesin apa pun. 16x benchmark H100 dipublikasikan.

Pilih yang asli ketika satu mesin memiliki tekanan HBM. Pilih LMCache ketika beberapa mesin berbagi awalan (RAG dengan system prompt umum, multi-penyewa dengan templat bersama).

### Perilaku tolok ukur

16x H100 (80 GB HBM) tersebar di 4 pengujian a3-highgpu-4g:

- Jejak KV rendah (prompt singkat, konkurensi rendah): semua konfigurasi sesuai dengan garis dasar, LMCache menambahkan ~3-5% overhead.
- Jejak sedang: LMCache mulai membantu penggunaan kembali awalan di seluruh mesin.
- KV melebihi HBM: offload CPU asli dan LMCache keduanya meningkatkan throughput secara substansial; Keuntungan LMCache lebih besar karena cross-engine sharing.

### Saat LMCache sangat menentukan

- Pelayanan multi-penyewa di mana system prompt dibagikan ke seluruh penyewa.
- RAG tempat potongan dokumen berulang di seluruh kueri.
- Varian yang disempurnakan (LoRA) pada basis yang sama di mana penggunaan kembali model dasar KV memotong pekerjaan yang mubazir.
- Weight kerja preemption-berat: memulihkan dari CPU lebih murah daripada mengisi ulang.

### Kapan TIDAK diaktifkan

- Tekanan HBM kecil — kamu membayar overhead tanpa manfaat.
- Konteks pendek (<1 ribu token) — waktu transfer > isi ulang.
- Weight kerja permintaan tunggal penyewa tunggal — tidak perlu digunakan kembali untuk diambil.

### Integrasi dengan penyajian terpilah

Fase 17 · 17 penyajian terpilah + senyawa LMCache: Transfer KV dari kumpulan prefill ke lahan kumpulan dekode di LMCache jika tidak digunakan; pertanyaan berikutnya diambil dari LMCache. Fase 17 · 11 router yang sadar cache dapat merutekan ke mesin yang cache lokalnya ATAU LMCache-berbagi cocok.

### Nomor yang harus kamu ingat

- vLLM 0.9.0: API Konektor dikirimkan.
- vLLM 0.11.0 (Jan 2026): jalur offload asinkron; dampak latensi end-to-end bergantung pada weight kerja, tingkat keberhasilan KV, dan tekanan sistem (bukan jaminan mutlak).
- Benchmark 16x H100: LMCache membantu ketika jejak KV melebihi HBM.
- Tekanan HBM kecil: overhead 3-5% tanpa manfaat.

## Pakai

`code/main.py` menyimulasikan weight kerja preemption yang berat dengan dan tanpa LMCache. Pengisian ulang laporan dihindari, peningkatan throughput, dan pemanfaatan HBM titik impas.

## Kirim

Lesson ini menghasilkan `outputs/skill-vllm-stack-decider.md`. Mengingat bentuk weight kerja dan penerapan vLLM, putuskan yang asli vs LMCache vs tidak keduanya.

## Latihan

1. Jalankan `code/main.py`. Pada penggunaan HBM manakah LMCache mulai membayar?
2. Penyewa membagikan system prompt 6 ribu token dalam 200 kueri/jam. Hitung penghematan LMCache yang diharapkan per penyewa.
3. Server LMCache adalah satu titik kegagalan. Rancang strategi HA (replika, fallback ke strategi asli).
4. LMCache disimpan ke Ceph pada disk yang berputar. Untuk KV token 4K pada 70B FP8 (500 MB), berapa waktu baca vs isi ulang?
5. Berdebat apakah jalur asinkron vLLM 0.11.0 "gratis" — di manakah overhead disembunyikan?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Tumpukan produksi | "penerapan referensi" | Bagan + operator Kubernetes Helm vLLM |
| API Konektor | "Antarmuka ujung belakang KV" | vLLM 0.9.0+ antarmuka toko KV yang dapat dicolokkan |
| Pembongkaran CPU asli | "tumpahan mesin-lokal" | Simpan KV di RAM host dari mesin yang sama |
| Cache LMC | "cluster cache KV" | Server cache KV lintas mesin pada DRAM CPU + disk |
| 0.11.0 asinkron | "pembongkaran non-pemblokiran" | Offload tersembunyi di balik aliran mesin |
| Pencegahan | "mengusir untuk memberi ruang" | Cache KV diacak saat HBM penuh |
| Penggunaan kembali awalan | "prompt sistem yang sama" | Beberapa pertanyaan berbagi permulaan; cache terkena |
| Tingkat Ceph | "tingkat disk" | Penyimpanan tahan lama di bawah DRAM dalam hierarki cache |

## Bacaan Lanjutan

- [Blog vLLM — Konektor Pembongkaran KV (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Bagan helm + operator.
- [LMCache untuk Inference LLM Skala Perusahaan (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Implementasi konektor.
- [catatan rilis vLLM 0.11.0](https://github.com/vllm-project/vllm/releases) — detail jalur asinkron.
