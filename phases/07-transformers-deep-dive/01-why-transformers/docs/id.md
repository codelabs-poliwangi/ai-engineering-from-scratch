# Mengapa Transformers — Masalah dengan RNN

> RNN memproses token satu per satu. Transformers memproses semua token sekaligus. Taruhan arsitektur tunggal tersebut mengubah setiap kurva penskalaan dalam pembelajaran mendalam setelah tahun 2017.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 3 (Inti Pembelajaran Mendalam), Fase 5 · 09 (Urutan-ke-Urutan), Fase 5 · 10 (Mekanisme Attention)
**Waktu:** ~45 menit

## Masalah

Sebelum tahun 2017, setiap model rangkaian tercanggih di planet ini — bahasa, terjemahan, ucapan — adalah neural network berulang. LSTM dan GRU memenangkan tolok ukur terjemahan setara ImageNet selama setengah dekade. Itu adalah satu-satunya alat yang dimiliki seseorang.

Mereka memiliki tiga kelemahan fatal. Perhitungan berurutan berarti kamu tidak dapat memparalelkan sepanjang sumbu waktu: token `t+1` memerlukan status tersembunyi dari token `t`. Urutan 1.024 token berarti 1.024 langkah serial pada GPU yang dapat melakukan 1.000.000 operasi floating-point per siklus. Waktu jam dinding training diskalakan secara linear dengan panjang urutan pada perangkat keras yang dirancang untuk paralelisme.

Vanishing gradient berarti informasi 50 token kembali sudah dikompresi melalui 50 non-linearitas. Unit berulang yang terjaga keamanannya (LSTM, GRU) melunakkan himpitan tetapi tidak pernah menghilangkannya. Ketergantungan jangka panjang — "buku yang saya baca musim panas lalu di pesawat ke Kyoto adalah..." — selalu gagal.

Status tersembunyi dengan lebar tetap berarti pembuat enkode memasukkan seluruh urutan sumber ke dalam satu vector sebelum dekoder melihat apa pun. Tidak masalah apakah sumbernya 5 token atau 500; kemacetannya bentuknya sama.

Makalah tahun 2017 berjudul "Attention Is All You Need" mengusulkan sesuatu yang radikal: hilangkan kekambuhan sepenuhnya. Biarkan setiap posisi mengikuti setiap posisi lainnya secara paralel. Berlatihlah dalam satu perkalian matrix besar, bukan 1.024 perkalian berurutan.

Hasilnya mendominasi setiap modalitas pada tahun 2026. Bahasa (GPT-5, Claude 4, Llama 4), penglihatan (ViT, DINOv2, SAM 3), audio (Whisper), biologi (AlphaFold 3), robotika (RT-2). Blok yang sama, input berbeda.

## Konsep

![Komputasi sekuensial RNN vs attention paralel Transformer](../assets/rnn-vs-transformer.svg)

**Pengulangan sebagai hambatan.** RNN menghitung `h_t = f(h_{t-1}, x_t)`. Setiap langkah bergantung pada langkah sebelumnya. kamu tidak dapat menghitung `h_5` sebelum `h_4`. Pada GPU modern dengan 10.000+ inti paralel, hal ini menghabiskan 99% silikon dalam rangkaian panjang.

**Attention sebagai siaran.** Attention mandiri menghitung `output_i = sum_j(a_ij * v_j)` untuk setiap pasangan `(i, j)` secara bersamaan. Seluruh matrix attention N×N terisi dalam satu matmul batch. Tidak ada langkah yang bergantung pada langkah lain. GPU menyukainya.

**Percepatannya tidak konstan.** Ini adalah perbedaan antara `O(N)` kedalaman serial dan `O(1)` kedalaman serial. Dalam praktiknya, Transformer berlatih 5–10× lebih cepat per masa pada perangkat keras yang cocok pada N=512, dan kesenjangan melebar seiring dengan panjang urutan hingga kamu mencapai dinding memori attention `O(N²)` (yang kemudian diperbaiki oleh Flash Attention — lihat Lesson 12).

**Berapa harga trafo.** Memori attention berskala `O(N²)`. Untuk konteks 2K, oke. Untuk konteks 128K, kamu memerlukan jendela geser, ekstrapolasi RoPE, ubin Flash Attention, atau varian attention linier. Pengulangan terjadi `O(N)` baik dalam waktu maupun memori; Transformer menukar waktu dengan memori dan kemudian memenangkan waktu kembali melalui paralelisme.**Pergeseran bias induktif.** RNN mengasumsikan lokalitas dan kekinian. Transformers tidak berasumsi apa pun - setiap pasangan adalah kandidat yang patut mendapat attention. Itulah sebabnya Transformer memerlukan lebih banyak data agar dapat dilatih dengan baik, namun dapat diperluas skalanya lebih jauh lagi setelah data tersebut dimiliki. Chinchilla (2022) memformalkan ini: dengan token yang cukup, Transformer selalu mengalahkan RNN dengan jumlah parameter yang sama.

## Build

Tidak ada jaringan neural di sini — kami menyimulasikan hambatan inti secara numerik sehingga kamu dapat merasakan kesenjangan pada laptop kamu.

### Langkah 1: ukur kedalaman serial

Lihat `code/main.py`. Kami membangun dua fungsi. Seseorang mengkodekan urutan sebagai rantai tambahan (serial, seperti RNN). Seseorang mengkodekannya sebagai pengurangan paralel (siaran, seperti attention). Matematika yang sama, grafik ketergantungan yang berbeda.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

Kami mengatur waktu keduanya secara berurutan hingga 100.000 elemen. Versi RNN adalah O(N) dan satu pipeline CPU. Bahkan dalam Python murni, pengurangan gaya attention mengalahkannya dengan panjang ≥ 1.000 karena `sum()` Python diimplementasikan dalam C dan melakukan iterasi tanpa overhead juru bahasa per langkah.

### Langkah 2: menghitung operasi teoritis

Kedua algoritma melakukan penambahan N. Perbedaannya adalah *kedalaman ketergantungan*: berapa banyak operasi yang harus dilakukan secara berurutan sebelum operasi berikutnya dapat dimulai. Kedalaman RNN = N. Kedalaman attention = log(N) dengan pengurangan pohon, atau 1 dengan pemindaian paralel. Kedalaman, bukan jumlah operasi, menentukan waktu GPU.

### Langkah 3: penskalaan empiris pada rangkaian panjang

Kami mencetak tabel waktu yang membuat celah O(N) terlihat. Pada laptop Mac 2026, urutan di bawah 1.000 elemen terlalu cepat untuk diukur. Urutan 100.000 menunjukkan pemindaian linier yang bersih. Skalakan itu ke Transformer 16.384 token dengan setara LSTM 12-layer dan kamu akan melihat mengapa jam dinding training menjadi pemblokir pada tahun 2016.

## Pakai

Kapan masih harus memilih RNN pada tahun 2026:

| Situasi | Pilih |
|-----------|------|
| Inference streaming, satu token pada satu waktu, memori konstan | RNN atau model ruang negara (Mamba, RWKV) |
| Urutan yang sangat panjang (>1 juta token) di mana memori attention meledak | Attention linier, Mamba 2, Hyena |
| Perangkat tepi tanpa akselerator matmul | RNN yang dapat dipisahkan secara mendalam masih menang pada FLOPs/watt |
| Yang lainnya (training, inference batch, konteks hingga 128K) | Transformer |

Model ruang negara (SSM) seperti Mamba pada dasarnya adalah RNN dengan parameterisasi terstruktur yang memberikan yang terbaik dari keduanya: memori pemindaian `O(N)`, training paralel melalui pemindaian selektif. Mereka memulihkan 90% kualitas trafo dengan penskalaan konteks panjang yang lebih baik. Pada tahun 2026, sebagian besar laboratorium perbatasan melatih model Transformer SSM+ hibrida (misalnya Jamba, Samba) — perulangan tidak mati, melainkan sebuah komponen.

## Kirim

Lihat `outputs/skill-architecture-picker.md`. Keterampilan memilih arsitektur untuk masalah urutan baru dengan mempertimbangkan batasan panjang, throughput, dan anggaran training. Ia harus selalu menolak untuk merekomendasikan RNN murni untuk training yang berjalan di atas 1 miliar token tanpa menyatakan trade-offnya.

## Latihan

1. **Mudah.** Ambil `rnn_style` dari `code/main.py` dan ganti status tersembunyi scalar dengan vector status tersembunyi dengan panjang 64. Ukur ulang. Berapa peningkatan overhead serial dengan dimension keadaan tersembunyi?
2. **Medium.** Menerapkan jumlah awalan paralel (pemindaian Hillis-Steele) dengan Python murni. Pastikan itu menghasilkan output numerik yang sama dengan pemindaian serial pada panjang 1024. Hitung kedalamannya.
3. **Hard.** Pindahkan pengurangan gaya attention ke PyTorch di GPU. Atur waktu keduanya saat kamu menyapu panjang urutan dari 64 menjadi 65.536. Gambarkan dan jelaskan bentuk kurvanya.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Kekambuhan | "RNN berurutan" | Perhitungan di mana langkah `t` bergantung pada langkah `t-1`, memaksa eksekusi serial sepanjang sumbu waktu. |
| Kedalaman serial | "Seberapa dalam grafiknya" | Rantai operasi dependen yang terpanjang; membatasi jam dinding bahkan pada perangkat keras yang tak terbatas. |
| Attention | "Biarkan token saling memandang" | Jumlah tertimbang `sum_j a_ij v_j` dimana `a_ij` berasal dari skor kemiripan antara posisi i dan j. |
| Jendela konteks | "Berapa banyak yang dilihat model" | Jumlah posisi yang dapat diambil oleh layer attention sebagai input; skala biaya memori kuadrat di sini. |
| Bias induktif | "Asumsi dimasukkan ke dalam arsitektur" | Sebelumnya tentang seperti apa datanya; CNN mengasumsikan invariansi terjemahan, RNN mengasumsikan keterkinian. |
| Model ruang negara | "RNN dengan aljabar di belakangnya" | Pengulangan diparameterisasi untuk training paralel melalui matrix ruang keadaan terstruktur. |
| Kemacetan kuadrat | "Mengapa konteks membutuhkan biaya yang sangat besar" | Memori attention = `O(N²)` dalam panjang urutan; Flash Attention menyembunyikan konstanta, bukan penskalaannya. |

## Bacaan Lanjutan

- [Vaswani dkk. (2017). Hanya Attention yang kamu Butuhkan](https://arxiv.org/abs/1706.03762) — makalah yang menghentikan kekambuhan di NLP arus utama.
- [Bahdanau, Cho, Bengio (2014). Neural MT dengan Belajar Bersama untuk Menyelaraskan dan Menerjemahkan](https://arxiv.org/abs/1409.0473) — tempat munculnya attention, diarahkan ke RNN.
- [Hochreiter, Schmidhuber (1997). Memori Jangka Pendek Panjang](https://www.bioinf.jku.at/publications/older/2604.pdf) — makalah LSTM asli, sebagai catatan.
- [Gu, Dao (2023). Mamba: Pemodelan Urutan Waktu Linier dengan Ruang Keadaan Selektif](https://arxiv.org/abs/2312.00752) — jawaban berulang modern untuk Transformer.
