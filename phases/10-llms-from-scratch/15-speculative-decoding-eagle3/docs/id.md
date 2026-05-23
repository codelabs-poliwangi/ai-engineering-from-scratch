# Decoding Spekulatif dan EAGLE-3

> Fase 7 · Lesson 16 membuktikan matematikanya: aturan penolakan Leviathan mempertahankan distribusi verifikator dengan tepat. Lesson ini adalah tampilan tumpukan training dari decoding spekulatif produksi tahun 2026. EAGLE-3 mengubah rancangan model dari perkiraan murah menjadi jaringan kecil yang dibuat khusus dan dilatih pada status tersembunyi milik pemverifikasi, lalu menambahkan loop pengujian waktu training yang menyelaraskan distribusi rangkaian dan inferensinya. Hasil: percepatan end-to-end 3× hingga 6,5×, tarif per token yang diterima di atas 0,9 pada chat, tidak ada tradeoff distribusi. Setiap tumpukan inference produksi pada tahun 2026 mengirimkannya secara default.

**Type:** Build
**Language:** Python (stdlib)
**Prerequisites:** Fase 7 · 16 (matematika decoding spekulatif), Fase 10 · 12 (optimization inference)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Nyatakan teorema Leviathan dalam satu kalimat dan buktikan bahwa loop spekulatif menghasilkan sample yang didistribusikan secara identik ke verifikator.
- Telusuri perkembangan dua tahun dari decoding spesifikasi vanilla (Leviathan 2023) hingga EAGLE, EAGLE-2, dan EAGLE-3 dan sebutkan batasan pasti setiap langkah yang dihapus.
- Hitung percepatan yang diharapkan dari tingkat penerimaan `α` dan rasio biaya draft-to-verifier `c`, dan pilih panjang draft yang optimal `N` untuk setiap rezim.
- Terapkan loop spekulatif penuh dari awal: draf, verifikasi, tolak sample dari sisa, putar kembali cache KV saat ditolak, keluarkan token bonus saat diterima penuh.

## Masalah

Penguraian code autoregresif pada model 70B mungkin berjalan pada 35 token per detik pada H100. GPU-nya masih belum jenuh. Bandwidth memori adalah batas tertinggi: setiap token memuat 70B weight dari HBM, melakukan satu langkah aritmatika, dan menghasilkan satu float. Unit komputasi sebagian besar tidak digunakan.

Penguraian code spekulatif mengubahnya menjadi masalah throughput yang sebenarnya dapat kamu pecahkan. Draf murah mengusulkan `N` token di `N` small forward pass. Pemverifikasi berjalan satu kali pada awalan ditambah semua draf `N`. Apabila pendistribusian verifikator pada posisi `i` sesuai dengan draf (secara statistik akan kami perinci), kami menerima; jika tidak, kami menolak dan mengambil sample koreksi dari distribusi sisa. Satu penerus model besar menghasilkan hingga `N+1` token yang diterima, bukan hanya satu.

Teorema yang penting adalah Leviathan, Kalman, Matias (ICML 2023): distribusi output identik dengan apa yang dihasilkan pengambilan sample dari verifikator secara langsung. Tidak kira-kira. Secara identik. Inilah alasan mengapa decoding spekulatif dapat diterima dalam produksi — ini adalah optimization latensi murni tanpa tradeoff kualitas.

Apa yang diberikan Fase 7 · Lesson 16 kepada kamu adalah matematika. Apa yang diberikan lesson ini kepada kamu adalah tumpukan training. Draf yang bagus bernilai kecepatan 2x lebih banyak daripada draf murah. EAGLE, EAGLE-2, dan EAGLE-3 (Li et al., 2024–2025) mengubah "draft = versi lebih kecil dari model yang sama" menjadi disiplin teknik presisi. Server inference produksi 2026 defaultnya adalah EAGLE-3.

## Konsep

### Invarian: Pengambilan sample penolakan LeviathanBiarkan `p(t)` menjadi draf distribusi untuk token berikutnya dengan beberapa awalan, dan `q(t)` menjadi verifikator. Contoh draf token `d ~ p`. Terima dengan kemungkinan `min(1, q(d) / p(d))`. Saat ditolak, sample dari distribusi sisa `(q - p)_+ / ||(q - p)_+||_1`. Sample yang dihasilkan didistribusikan menurut `q`. Hal ini benar, betapapun buruknya `p` — semakin buruk, semakin sering kamu menolaknya, namun hasilnya tetap sama persis.

Tumpuk `N` panggilan ini berturut-turut menggunakan satu penerusan verifikasi pada `prefix + d_1 + ... + d_N`. Verifikator mengembalikan `q_1, q_2, ..., q_{N+1}` secara bersamaan. Berjalan dari kiri ke kanan. Pada penolakan pertama di posisi `j`, sample dari `residual(q_j, p_j)` dan berhenti. Jika diterima penuh, cicipi satu token bonus dari `q_{N+1}`.

### Yang menentukan kecepatan

Biarkan `α` menjadi tingkat penerimaan yang diharapkan per token yang dirancang. Misalkan `c = cost(draft) / cost(verifier)` menjadi rasio biaya. Jumlah token yang diterima per verifikator yang diharapkan adalah:

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

Total waktu tunggu yang diharapkan per token yang diterima adalah `(N * c + 1) / E[accepted]`. Minimalkan hal itu sehubungan dengan `N` dan kamu akan mendapatkan manfaatnya. Untuk `α = 0.8, c = 0.05`: optimal `N` adalah sekitar 5–7, kecepatannya 3,2×. Untuk `α = 0.95, c = 0.02`: optimal `N` adalah sekitar 8–10, percepatan meningkat 5×.

Pengungkit terbesar adalah `α`. Beralih dari `α = 0.6` (vanilla draft) ke `α = 0.9` (EAGLE-3) pada `N = 5` tetap akan membawa kamu dari 2,2 token diterima yang diharapkan per verifikator maju ke 4.1. Throughput hampir 2× lebih banyak dari pemverifikasi yang sama.

### Perkembangan dua tahun

**Vanilla spekulatif (Leviathan, 2023).** Model draft adalah LLM kecil yang dilatih secara mandiri dari keluarga yang sama. Mudah untuk dihubungkan, `α ≈ 0.6`, kecepatan terbaiknya sekitar 2×.

**EAGLE-1 (Li et al., 2024).** Draf adalah Transformer kecil — biasanya satu atau dua layer — yang mengambil status tersembunyi layer terakhir pemverifikasi sebagai input dan memprediksi token berikutnya secara langsung. Karena draf melihat representasi feature verifikator, maka distribusinya lebih dekat dengan verifikator. `α` naik ke 0,7–0,8.

**EAGLE-2 (Li dkk., 2024).** Menambahkan pohon draf dinamis: alih-alih mengusulkan satu rangkaian token `N`, usulkan pohon kecil kandidat, beri skor pada masing-masing dengan pemverifikasi dalam satu langkah ke depan (attention pohon), dan telusuri jalur dengan probabilitas tertinggi. Panjang draf menjadi adaptif per langkah. `α` per token jalur yang diterima naik di atas 0,85.

**EAGLE-3 (Li et al., 2025, NeurIPS).** Dua perubahan lagi. Pertama, hilangkan seluruh kehilangan prediksi feature — EAGLE-1/2 melatih draf agar sesuai dengan status tersembunyi pemverifikasi, yang membatasi jumlah bantuan data. EAGLE-3 berlatih langsung pada prediksi token. Kedua, tes waktu training (TTT): selama training draf, masukkan kembali prediksi draf sebelumnya sebagai input melalui beberapa langkah, dengan cara yang sama saat beroperasi pada inference. Hal ini menyelaraskan distribusi training dan pengujian serta menghentikan akumulasi kesalahan. Kecepatan terukur: hingga 6,5× pada obrolan, peningkatan throughput sebesar 38% pada batch 64 di SGLang pada H100.

### Pengembalian cache KVVerifikasi memperluas cache KV pemverifikasi sebanyak `N` entri dalam sekali jalan. Jika penolakan terjadi pada posisi `j`, konten cache yang melewati posisi `j-1` kini salah. Dua implementasi umum: tulis ke buffer awal dan lakukan penerimaan (vLLM, TensorRT-LLM), atau simpan cache KV fisik ditambah panjang logis dan potong saat ditolak. Apa pun yang terjadi, biaya rollback adalah byte per layer per head, yang dapat diabaikan jika dibandingkan dengan biaya forward-pass.

Untuk penelusuran pohon EAGLE-2, pemverifikasi menjalankan attention dengan topeng non-kausal yang mengikuti topologi pohon. Rekayasanya rumit tetapi perhitungannya adalah panggilan attention kilat standar dengan topeng khusus.

### Draf arsitektur pada tahun 2026

| Strategi | Jenis draf | `α` | Mempercepat | Biaya training |
|----------|-----------|-----|---------|---------------|
| vanila | Pisahkan LLM kecil | 0,55-0,70 | 1,8-2,3× | Tidak ada (gunakan kembali model kecil yang ada) |
| Medusa | Ekstra LM menuju verifikator | 0,65-0,75 | 2-3× | ~1 miliar token SFT |
| EAGLE-1 | Trafo 1 lapis pada keadaan tersembunyi | 0,70-0,80 | 2,5-3× | ~60 miliar token |
| EAGLE-2 | EAGLE-1 + pohon rancangan dinamis | 0,80-0,88 | 3-4× | ~60 miliar token |
| EAGLE-3 | Penggabungan feature multi-layer + TTT | 0,88-0,92 | 3,5-6,5× | ~60-200 miliar token |
| Melihat ke Depan | Tidak ada draf (iterasi Jacobi) | T/A | 1,3-1,6× | Tidak ada |

Pada produksi tahun 2026: vLLM dan SGLang defaultnya adalah EAGLE-3 bila tersedia, sedangkan EAGLE-2 sebaliknya. TensorRT-LLM memiliki jalur Medusa tercepat untuk model publik Meta dan NVIDIA. llama.cpp mengirimkan draft vanilla untuk penerapan CPU.

## Build

Lihat `code/main.py`. Ini adalah loop spekulatif Leviathan lengkap dengan semua bagiannya: draft-of-N, pass paralel verifikator, penolakan per posisi, pengambilan sample sisa, token bonus, pengembalian KV, dan verifikasi empiris bahwa distribusi output cocok dengan pengambilan sample langsung dari `q`.

### Langkah 1: aturan penolakan

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### Langkah 2: distribusi sisa

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### Langkah 3: langkah spekulatif penuh

Fungsi `spec_step` menyusun token `N` dari `p`, lalu memverifikasi semuanya dalam satu evaluasi `q` paralel. Untuk setiap token yang dirancang, ia menerapkan aturan penolakan, dan pada penolakan pertama, ia mengambil sample koreksi dari sisa. Jika semuanya diterima, token bonus akan dikeluarkan dari `q_{N+1}`.

### Langkah 4: Pembukuan rollback KV

Simulator melacak logika `kv_length` per pekerja. Tentang penerimaan draf `k`, `kv_length += k`. Pada penolakan di posisi `j`, cache sudah ditulis melewati `j`, namun panjang logikanya diatur ke `prefix_length + j + 1` — satu melewati token koreksi. Bacaan selanjutnya terpotong ke panjang logis.

### Langkah 5: pemeriksaan Leviathan

Jalankan 50.000 langkah spekulatif. Hitung distribusi empiris dari token yang diterima. Bandingkan dengan 50.000 sample langsung dari `q`. Statistik chi-kuadrat harus berada di bawah nilai kritis. Teorema ini lolos dalam praktik.

### Langkah 6: percepatan vs. αSapu kualitas draf dengan mengalihkan `p` dari `q` pada amplitudo yang berbeda. Ukur `α`, lalu plot token yang diharapkan per panggilan verifikator sebagai fungsi dari `α` dan `N`. Code tersebut mencetak tabel yang menunjukkan bagaimana kualitas draf kelas EAGLE-3 (`α ≈ 0.9`) membuka 4–5 token per panggilan verifikator.

## Pakai

Tingkat produksi `vllm serve` dengan EAGLE-3:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

SGLang dengan EAGLE-3 pada batch 64 pada H100: throughput kira-kira 1,38× lebih banyak dibandingkan decoding vanilla batch-64, menurut kertas EAGLE-3.

Kapan harus melakukan decoding spekulatif:

- Weight kerja obrolan interaktif apa pun yang latensi p50 lebih penting daripada throughput puncak.
- Pembuatan code dan output terstruktur (JSON, SQL). `α` berada di atas 0,9 karena target sebarannya sangat mudah diprediksi.
- Generasi jangka panjang (ribuan token). Percepatan yang diamortisasi terus membuahkan hasil.

Kapan tidak:

- Model sangat kecil (< 3B). Drafnya tidak jauh lebih murah dibandingkan verifikator.
- Penerapan CPU batch-1 kecil. Overhead memori pada model draf mungkin tidak sepadan.
- Pengambilan sample materi iklan bersuhu sangat tinggi yang menyebabkan `α` runtuh.

## Kirim

Lesson ini menghasilkan `outputs/skill-eagle3-tuner.md`. Mengingat weight kerja inference (model, ukuran batch, latensi target, profil tugas), ini merekomendasikan strategi decoding spekulatif dan parameter penyetelan (kelompok draf, `N`, kedalaman pohon, peralihan sadar suhu).

## Latihan

1. Jalankan `code/main.py`. Konfirmasikan statistik chi-kuadrat pada pemeriksaan distribusi Leviathan tetap di bawah nilai kritis 95% pada 50.000 sample.

2. Sapu `N` dari 1 hingga 10 dengan `α` diadakan pada 0,9 dan `c` diadakan pada 0,04. Plot token yang diharapkan per panggilan verifikator dan waktu dinding sebenarnya per token. Temukan `N` yang meminimalkan waktu dinding. Jelaskan bentuk kurva tersebut!

3. Modifikasi code untuk menyimulasikan pencarian pohon EAGLE-2: pada setiap langkah, draf mengusulkan pohon berbentuk `[2, 2, 2]` (delapan jalur kandidat). Pemverifikasi berjalan satu kali, dan jalur yang diterima dengan probabilitas tertinggi menang. Hitung `α` per daun dan total token per panggilan verifikator. Bandingkan dengan decoding spesifikasi rantai linier pada komputasi yang setara.

4. Menerapkan simulator rollback KV batch untuk dua urutan bersamaan. Urutan A telah menerima semua draf; urutan B ditolak pada posisi 2. Tunjukkan bahwa `kv_length` yang benar diperbarui per urutan dan tidak ada usaha yang terbuang.

5. Bacalah Bagian 4 makalah EAGLE-3 (Tes Waktu Training). Jelaskan dalam dua kalimat mengapa draf training yang naif tanpa TTT mengalami bias paparan, dan mengapa memasukkan draf prediksinya sendiri selama training dapat memperbaikinya. Hubungkan ini dengan literatur pengambilan sample terjadwal di seq2seq.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Aturan Leviathan | "min(1, q di atas p)" | Bernoulli menerima/menolak dengan probabilitas `min(1, q(d)/p(d))`, mempertahankan distribusi pemverifikasi tepat ketika kamu mengambil sample dari sisa penolakan |
| Distribusi sisa | "(q dikurangi p) plus, dinormalisasi" | `(q - p)_+` dijepit pada nol dan dinormalisasi ulang — distribusi yang benar untuk dijadikan sample sejak penolakan |
| Tingkat penerimaan α | "seberapa sering drafnya benar" | Kemungkinan keberhasilan Bernoulli per token yang diharapkan berdasarkan aturan penolakan; mengatur semua matematika speedup |
| EAGLE-1 | "draf negara bagian tersembunyi" | Draf trafo kecil dikondisikan pada keadaan tersembunyi layer terakhir verifikator (Li et al., 2024) |
| EAGLE-2 | "pohon rancangan dinamis" | EAGLE-1 ditambah pohon kandidat kelanjutan yang diberi skor dengan attention pohon dalam satu kelulusan verifikator |
| EAGLE-3 | "tes waktu training" | Menghilangkan loss prediksi feature, melatih prediksi token langsung dengan draf memasukkan outputnya sendiri selama training |
| Tes waktu training (TTT) | "perbaikan bias eksposur" | Jalankan draf secara otomatis selama training sehingga distribusi input training dan pengujian cocok — analog langsung dari pengambilan sample terjadwal |
| Kembalikan KV | "batalkan draf yang ditolak" | Pembukuan yang menyetel ulang cache KV pemverifikasi ke panjang awalan yang diterima setelah penolakan |
| Token bonus | "yang gratis" | Ketika semua draf `N` diterima, cicipi satu draf tambahan dari `q_{N+1}` tanpa biaya verifikasi tambahan |
| Attention pohon | "verifikasi banyak kandidat sekaligus" | Attention dengan topeng non-kausal yang menghormati topologi pohon rancangan; menghitung `q_i` untuk setiap node di pohon dalam satu forward pass |

## Bacaan Lanjutan

- [Leviathan, Kalman, Matias — Inference Cepat dari Transformers melalui Decoding Spekulatif (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — makalah dasar dan teorema kesetaraan
- [Chen dkk. — Mempercepat Penguraian Code Large Language Model dengan Pengambilan Sample Spekulatif (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) — pengenalan independen secara bersamaan dengan bukti bersih
- [Li dkk. — EAGLE: Pengambilan Sample Spekulatif Memerlukan Pemikiran Ulang Ketidakpastian Feature (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — EAGLE-1, draf kondisi tersembunyi
- [Li dkk. — EAGLE-2: Inference Model Bahasa yang Lebih Cepat dengan Pohon Draf Dinamis (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — pencarian pohon dinamis
- [Li dkk. — EAGLE-3: Meningkatkan Akselerasi Inference melalui Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) — default produksi tahun 2026
- [Cai dkk. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — pendekatan alternatif bebas draf
- [dokumentasi Penguraian Code Spekulatif vLLM](https://docs.vllm.ai/en/latest/features/spec_decode.html) — referensi produksi kanonik dengan semua strategi yang terhubung
