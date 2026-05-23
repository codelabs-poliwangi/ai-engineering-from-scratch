# Norm dan Distance

> Fungsi distance kamu mendefinisikan arti "serupa". Pilih yang salah dan semuanya di hilir rusak.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lesson 01 (Intuisi Linear Algebra), 02 (Vector, Matrix & Operasi)
**Waktu:** ~90 menit

## Tujuan Pembelajaran

- Mengimplementasikan fungsi L1, L2, cosinus, Mahalanobis, Jaccard, dan mengedit distance dari awal
- Pilih metrik distance yang sesuai untuk tugas ML tertentu dan jelaskan mengapa alternatif gagal
- Hubungkan norm L1 dan L2 ke regularisasi LASSO dan Ridge serta wilayah batasan geometrisnya
- Tunjukkan bagaimana dataset yang sama menghasilkan nearest neighbor yang berbeda berdasarkan metrik yang berbeda

## Masalah

kamu memiliki dua vector. Mungkin itu adalah embedding kata. Mungkin itu adalah profil pengguna. Mungkin itu adalah array piksel. Perlu kamu ketahui: seberapa dekat mereka?

Jawabannya bergantung sepenuhnya pada fungsi distance mana yang kamu pilih. Dua titik data bisa menjadi nearest neighbor di bawah satu metrik dan berjauhan di bawah metrik lainnya. Pengklasifikasi KNN kamu, mesin rekomendasi kamu, database vector kamu, algoritma pengelompokan kamu, loss function kamu -- semuanya bergantung pada pilihan ini. Lakukan kesalahan dan model kamu akan mengoptimalkan hal yang salah.

Tidak ada distance terbaik yang universal. L2 berfungsi untuk data spasial. Kesamaan kosinus mendominasi NLP. Jaccard menangani set. Edit distance menangani string. Mahalanobis menjelaskan korelasi. Wasserstein memindahkan massa probabilitas. Masing-masing memberikan asumsi yang berbeda tentang arti "serupa".

Lesson ini membangun setiap fungsi distance utama dari awal, menunjukkan kepada kamu kapan masing-masing fungsi tersebut merupakan alat yang tepat, dan menunjukkan bagaimana data yang sama menghasilkan nearest neighbor yang sangat berbeda bergantung pada metrik yang kamu gunakan.

## Konsep

### Norm: mengukur besaran vector

Norm mengukur "ukuran" suatu vector. Setiap fungsi distance antara dua vector dapat dituliskan sebagai norm selisihnya: d(a, b) = ||a - b||. Jadi memahami norm berarti memahami distance.

### Norm L1 (distance Manhattan)

Norm L1 menjumlahkan nilai absolut semua komponen.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

Disebut distance Manhattan karena mengukur seberapa jauh kamu berjalan di jaringan kota di mana kamu hanya dapat bergerak sepanjang sumbu. Tidak ada diagonal.

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

Kapan menggunakan L1:
- Data renggang berdimensi tinggi (feature teks, pengkodean one-hot)
- Bila kamu menginginkan ketahanan terhadap outlier (satu perbedaan besar tidak mendominasi)
- Masalah pemilihan feature (regularisasi L1 mendorong ketersebaran)

Koneksi ke regularisasi L1 (Lasso): menambahkan ||w||_1 ke loss function kamu akan menghukum jumlah nilai weight absolut. Hal ini mendorong weight kecil ke nol, melakukan pemilihan feature otomatis. Penalti L1 menciptakan daerah kendala berbentuk wajik dalam ruang weight, dan sudut wajik terletak pada sumbu yang beberapa bobotnya nol.

Koneksi ke loss function: Mean Absolute Error (MAE) adalah distance rata-rata L1 antara prediksi dan target. Ini menghukum semua kesalahan secara linier, membuatnya kuat terhadap outlier dibandingkan dengan MSE.

### Norm L2 (distance Euclidean)

Norm L2 adalah distance garis lurus. Akar kuadrat dari jumlah komponen kuadrat.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

Ini adalah distance yang kamu pelajari di kelas geometri. Pythagoras dalam n dimension.

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```Kapan menggunakan L2:
- Data kontinu berdimensi rendah hingga menengah
- Ketika skala feature sebanding
- Distance fisik (data spasial, pembacaan sensor)
- Kemiripan gambar pada tingkat piksel

Koneksi ke regularisasi L2 (Ridge): menambahkan ||w||_2^2 ke loss function kamu akan memberikan penalti pada weight yang besar. Berbeda dengan L1, ini tidak mendorong weight ke nol. Ini mengecilkan semua weight menuju nol secara proporsional. Penalti L2 menciptakan daerah kendala melingkar, sehingga tidak ada sudut pada sumbu. Weight menjadi kecil tetapi jarang sekali yang tepat nol.

Koneksi ke loss function: Mean Squared Error (MSE) adalah rata-rata kuadrat distance L2. Mengkuadratkan memberikan hukuman yang lebih berat pada kesalahan besar dibandingkan kesalahan kecil.

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Norm Lp : keluarga umum

L1 dan L2 adalah kasus khusus dari norm Lp:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

Nilai p yang berbeda menghasilkan bentuk "bola satuan" yang berbeda (kumpulan semua titik pada distance 1 dari titik asal):

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity Norm (distance Chebyshev)

Saat p mendekati tak terhingga, norm Lp menyatu ke komponen absolut maksimum.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

Distance antara dua titik ditentukan oleh dimension tunggal dimana perbedaannya paling besar. Semua dimension lainnya diabaikan.

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

Kapan menggunakan L-infinity:
- Ketika penyimpangan terburuk dalam satu dimension penting
- Papan permainan (raja dalam catur bergerak dalam L-infinity: satu langkah ke segala arah berharga 1)
- Toleransi manufaktur (setiap dimension harus sesuai spesifikasi)

### Kesamaan Kosinus dan Distance Kosinus

Kesamaan kosinus mengukur sudut antara dua vector, mengabaikan besarnya.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

Berkisar dari -1 (berlawanan arah) hingga +1 (arah sama). Vector tegak lurus memiliki kesamaan kosinus 0.

Distance kosinus mengubahnya menjadi distance: cosine_distance = 1 - cosine_similarity. Ini berkisar dari 0 (arah yang sama) hingga 2 (arah berlawanan).

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

Mengapa cosinus mendominasi NLP dan embeddings: dalam teks, panjang dokumen tidak akan mempengaruhi kesamaan. Dokumen tentang kucing yang panjangnya dua kali lipat dari dokumen lain tentang kucing harus tetap "serupa". Kesamaan kosinus mengabaikan besaran (panjang) dan hanya memperhatikan arah. Dua dokumen dengan sebaran kata yang sama tetapi panjangnya berbeda menunjuk ke arah yang sama dan mendapatkan kesamaan cosinus 1,0.

Kapan menggunakan kesamaan kosinus:
- Kemiripan teks (vector TF-IDF, embedding kata, embedding kalimat)
- Domain apa pun yang besarnya adalah kebisingan dan arahnya adalah sinyal
- Sistem rekomendasi (vector preferensi pengguna)
- Embedding pencarian (database vector hampir selalu menggunakan produk kosinus atau titik)

### Kesamaan Perkalian Titik vs Kesamaan Kosinus

Hasil kali titik dua buah vector adalah:

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

Kesamaan kosinus adalah perkalian titik yang dinormalisasi dengan kedua besaran. Jika kedua vector sudah dinormalisasi satuan (magnitudo = 1), perkalian titik dan kesamaan kosinus adalah identik.

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

Jika berbeda: perkalian titik mencakup informasi besaran. Vector dengan magnitudo lebih besar mendapat skor perkalian titik yang lebih tinggi. Hal ini penting dalam beberapa sistem pengambilan di mana kamu ingin item "populer" mendapat peringkat lebih tinggi. Besarnya bertindak sebagai sinyal kualitas atau kepentingan implisit.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```Dalam praktiknya:
- Gunakan kesamaan kosinus jika kamu menginginkan kesamaan arah yang murni
- Gunakan perkalian titik ketika besaran membawa informasi yang berarti
- Banyak database vector (Pinecone, Weaviate, Qdrant) memungkinkan kamu memilih di antara keduanya
- Jika embeddings kamu dinormalisasi L2, pilihannya tidak menjadi masalah

### Distance Mahalanobis

Distance Euclidean memperlakukan semua dimension secara setara. Namun jika feature kamu berkorelasi atau memiliki skala berbeda, L2 memberikan hasil yang menyesatkan.

Distance Mahalanobis menjelaskan struktur kovarians data.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

di mana S adalah covariance matrix data.

Secara intuitif: Distance Mahalanobis pertama-tama mendekorelasi dan menormalkan data (memutihkan), kemudian menghitung distance L2 dalam ruang yang diubah tersebut. Jika S adalah matrix identitas (tidak berkorelasi, feature unit varians), distance Mahalanobis tereduksi menjadi distance Euclidean.

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

Kapan menggunakan distance Mahalanobis:
- Deteksi outlier (titik dengan distance Mahalanobis yang besar dari mean adalah outlier)
- Klasifikasi ketika feature memiliki skala dan korelasi yang berbeda
- Ketika kamu memiliki cukup data untuk memperkirakan covariance matrix yang andal
- Kontrol kualitas di bidang manufaktur (pemantauan proses multivariat)

### Kemiripan Jaccard (untuk set)

Ukuran kesamaan Jaccard tumpang tindih antara dua set.

```
J(A, B) = |A intersect B| / |A union B|
```

Berkisar dari 0 (tidak ada tumpang tindih) hingga 1 (kumpulan identik). Distance Jaccard = 1 - Kemiripan Jaccard.

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

Kapan menggunakan Jaccard:
- Membandingkan kumpulan tag, kategori, atau feature
- Kesamaan dokumen berdasarkan keberadaan kata (bukan frekuensi)
- Deteksi hampir duplikat (perkiraan MinHash dari Jaccard)
- Membandingkan vector feature biner (data ada/tidaknya)
- Mengevaluasi model segmentasi (Intersection over Union = Jaccard)

### Edit Distance (Distance Levenshtein)

Edit distance menghitung jumlah minimum operasi karakter tunggal yang diperlukan untuk mengubah satu string menjadi string lainnya. Operasinya adalah: menyisipkan, menghapus, atau mengganti.

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

Dihitung menggunakan pemrograman dinamis. Isi matrix dengan entri (i, j) adalah distance edit antara i karakter pertama string A dan j karakter pertama string B.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

Kapan menggunakan edit distance:
- Pemeriksaan dan koreksi ejaan
- Penyelarasan urutan DNA (dengan operasi berbobot)
- Pencocokan string fuzzy
- Deduplikasi data teks yang berantakan

### KL Divergence (bukan distance, tapi digunakan seperti satu)

Divergensi KL mengukur perbedaan satu distribusi probabilitas dengan distribusi probabilitas lainnya. Dicakup dalam Lesson 09, namun hal ini termasuk dalam diskusi ini karena orang menggunakannya sebagai "distance" meskipun bukan distance.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

Properti penting: Divergensi KL TIDAK simetris.

```
D_KL(P || Q) != D_KL(Q || P)
```

Ini berarti gagal memenuhi persyaratan dasar metrik distance. Itu juga tidak memenuhi pertidaksamaan segitiga. Ini adalah perbedaan, bukan distance.

Forward KL (D_KL(P || Q)) adalah "pencarian jahat": Q mencoba mencakup semua mode P.
Reverse KL (D_KL(Q || P)) adalah "pencarian mode": Q berfokus pada mode tunggal P.

Saat kamu melihat divergensi KL:
- VAE (istilah KL dalam ELBO mendorong distribusi laten ke arah prior)
- Penyulingan pengetahuan (siswa mencoba mencocokkan distribusi guru)
- RLHF (penalti KL membuat model yang disempurnakan tetap dekat dengan model dasar)
- Metode gradient kebijakan (membatasi pembaruan kebijakan)

### Distance Wasserstein (Distance Penggerak Bumi)Distance Wasserstein mengukur "usaha" minimum yang diperlukan untuk mengubah satu distribusi probabilitas menjadi distribusi probabilitas lainnya. Anggap saja: jika satu distribusi berupa tumpukan tanah dan distribusi lainnya berupa lubang, berapa banyak kotoran yang harus kamu pindahkan dan seberapa jauh?

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

Untuk distribusi 1D, disederhanakan menjadi integral selisih mutlak fungsi distribusi kumulatif:

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

Mengapa Wasserstein penting:
- Ini adalah metrik yang benar (simetris, memenuhi pertidaksamaan segitiga)
- Ini memberikan gradient bahkan ketika distribusi tidak tumpang tindih (divergensi KL mencapai tak terbatas)
- Properti ini menjadikannya pusat bagi Wasserstein GAN (WGAN), yang memecahkan ketidakstabilan training GAN asli

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

Kapan menggunakan Wasserstein:
- Training GAN (WGAN, WGAN-GP)
- Membandingkan distribusi yang mungkin tidak tumpang tindih
- Masalah transportasi yang optimal
- Pengambilan gambar (membandingkan histogram warna)

### Mengapa Tugas Berbeda Membutuhkan Distance Berbeda

| Tugas | Distance terbaik | Mengapa |
|------|--------------|-----|
| Kesamaan teks | Kosinus | Besaran adalah kebisingan, arah adalah makna |
| Perbandingan piksel gambar | L2 | Hubungan spasial penting, feature-fiturnya sebanding dengan skala |
| Feature redup tinggi yang jarang | L1 | Kuat, tidak memperkuat perbedaan besar yang jarang terjadi |
| Setel tumpang tindih (tag, kategori) | Jaccard | Data secara alami bernilai set, bukan vector |
| Pencocokan string | Sunting distance | Peta operasi ke intuisi pengeditan manusia |
| Deteksi outlier | Mahalanobi | Memperhitungkan korelasi dan skala feature |
| Membandingkan distribusi | Divergensi KL | Mengukur informasi yang hilang dengan menggunakan Q, bukan P |
| training GAN | Wasserstein | Memberikan gradient meskipun distribusi tidak tumpang tindih |
| Embedding (DB vector) | Perkalian kosinus atau titik | Embedding dilatih untuk menyandikan makna ke arah |
| Rekomendasi | Produk titik | Besaran dapat menyandikan popularitas atau kepercayaan |
| Urutan DNA | Distance edit tertimbang | Biaya substitusi bervariasi menurut pasangan nukleotida |
| QC Manufaktur | L-tak terhingga | Penyimpangan kasus terburuk dalam dimension apa pun penting |

### Koneksi ke Fungsi Rugi

Loss function adalah fungsi distance yang diterapkan pada prediksi vs target.

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### Koneksi ke Regularisasi

Regularisasi menambahkan penalti norm pada weight pada loss function.

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

Mengapa L1 menghasilkan ketersebaran tetapi L2 tidak: bayangkan wilayah kendala dalam ruang weight 2D. L1 berbentuk berlian, L2 berbentuk lingkaran. Kontur loss function (elips) kemungkinan besar akan menyentuh wajik di suatu sudut, yang salah satu bobotnya nol. Mereka menyentuh lingkaran pada titik halus, di mana kedua bobotnya bukan nol.

### Pencarian Tetangga Terdekat

Setiap fungsi distance menyiratkan masalah pencarian nearest neighbor: dengan adanya titik kueri, temukan titik terdekat dalam dataset.

Pencarian nearest neighbor yang tepat adalah O(n * d) per kueri dalam dataset n titik dengan dimension d. Untuk dataset besar, ini terlalu lambat.

Algoritma Approximate Nearest Neighbor (ANN) menukar sejumlah kecil akurasi untuk peningkatan kecepatan yang besar:

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW (Hierarchical Navigable Small World) adalah algoritma dominan dalam database vector modern. Ini membangun grafik multi-layer di mana setiap node terhubung ke perkiraan tetangga terdekatnya. Pencarian dimulai dari layer atas (jarang, lompat jauh) dan turun ke layer bawah (padat, lompat pendek).

## Build

### Langkah 1: Semua fungsi norm dan jarakLihat `code/distances.py` untuk implementasi selengkapnya. Setiap fungsi dibangun dari awal hanya menggunakan matematika dasar Python.

### Langkah 2: Data sama, distance berbeda, tetangga berbeda

Demo di `distances.py` membuat dataset, memilih titik kueri, dan menunjukkan bagaimana nearest neighbor berubah bergantung pada metrik distance. Titik yang "terdekat" di bawah L1 belum tentu paling dekat di bawah L2 atau kosinus.

### Langkah 3: Embed penelusuran kesamaan

Code tersebut menyertakan penelusuran kesamaan embedding tiruan yang menemukan "dokumen" paling mirip dengan kueri menggunakan kesamaan kosinus vs distance L2, yang menunjukkan bahwa peringkatnya dapat berbeda.

## Pakai

Penggunaan praktis yang paling umum: menemukan item serupa dalam database vector.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

Saat kamu memanggil `model.encode(text)` dan kemudian mencari database vector, inilah yang terjadi. Model embedding memetakan teks ke vector. Basis data vector menghitung kesamaan kosinus (atau perkalian titik) antara vector kueri kamu dan setiap vector yang disimpan, menggunakan algoritme ANN untuk menghindari pemeriksaan semuanya.

## Latihan

1. Hitung distance tak terhingga L1, L2, dan L antara (1, 2, 3) dan (4, 0, 6). Verifikasi bahwa L-inf <= L2 <= L1 selalu berlaku untuk pasangan titik mana pun. Buktikan mengapa pemesanan ini dijamin.

2. Buatlah dua buah vector yang kemiripan kosinusnya tinggi (> 0,9) tetapi distance L2nya besar (> 10). Jelaskan secara geometris apa yang terjadi. Kemudian buat dua vector yang kemiripan kosinusnya rendah (<0,3) tetapi distance L2 kecil (<0,5).

3. Mengimplementasikan fungsi yang mengambil dataset dan titik kueri serta mengembalikan nearest neighbor pada distance L1, L2, kosinus, dan Mahalanobis. Temukan dataset yang keempatnya tidak sepakat mengenai titik mana yang terdekat.

4. Hitung distance Wasserstein antara [0.5, 0.5, 0, 0] dan [0, 0, 0.5, 0.5] dengan tangan menggunakan metode CDF. Kemudian hitung antara [0,25, 0,25, 0,25, 0,25] dan [0, 0, 0,5, 0,5]. Mana yang lebih besar dan mengapa?

5. Terapkan MinHash untuk perkiraan kesamaan Jaccard. Hasilkan 100 set acak, hitung Jaccard yang tepat untuk semua pasangan, dan bandingkan dengan perkiraan MinHash menggunakan fungsi hash 50, 100, dan 200. Plot kesalahan perkiraan.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|----------------------|
| Norm | "Ukuran vector" | Fungsi yang memetakan suatu vector ke scalar non-negatif, memenuhi pertidaksamaan segitiga, homogenitas mutlak, dan nol hanya untuk vector nol |
| Norm L1 | "Distance Manhattan" | Jumlah nilai komponen absolut. Menghasilkan ketersebaran dalam optimization. Kuat terhadap outlier |
| Norm L2 | "Distance Euclidean" | Akar kuadrat dari jumlah komponen kuadrat. Distance garis lurus dalam ruang Euclidean |
| Lp norm | "Norm umum" | Akar ke-p dari jumlah pangkat ke-p komponen absolut. L1 dan L2 adalah kasus khusus |
| Norm L-tak terhingga | "Norm maks" atau "Distance Chebyshev" | Nilai komponen absolut maksimum. Batas Lp ketika p mendekati tak terhingga |
| Kesamaan kosinus | "Sudut antar vector" | Perkalian titik dinormalisasi dengan kedua besaran. Berkisar dari -1 hingga +1. Mengabaikan panjang vector |
| Distance kosinus | "1 dikurangi kesamaan cosinus" | Mengonversi kesamaan kosinus menjadi distance. Berkisar dari 0 hingga 2 |
| Produk titik | "Kosinus tidak normal" | Jumlah produk berdasarkan komponen. Sama dengan kesamaan kosinus dikalikan kedua besaran |
| Distance Mahalanobis | "Distance sadar korelasi" | Distance L2 pada suatu ruang yang telah diputihkan (dikorelasikan dan dinormalisasi) menggunakan covariance matrix data |
| Kemiripan Jaccard | "Atur tumpang tindih" | Ukuran persimpangan dibagi dengan ukuran kesatuan. Untuk himpunan, bukan vector |
| Sunting distance | "Distance Levenshtein" | Penyisipan, penghapusan, dan substitusi minimum untuk mengubah satu string menjadi string lainnya |
| Divergensi KL | "Distance antar distribusi" | Bukan distance sebenarnya (tidak simetris). Mengukur bit tambahan dari penggunaan Q untuk menyandikan P |
| Distance Wasserstein | "Distance penggerak bumi" | Pekerjaan minimum untuk mengangkut massa dari satu distribusi ke distribusi lainnya. Metrik yang sebenarnya |
| Perkiraan nearest neighbor | "pencarian JST" | Algoritma (HNSW, LSH, IVF) yang menemukan kira-kira titik terdekat jauh lebih cepat daripada pencarian tepat |
| HNSW | "Algoritma DB vector" | Grafik Dunia Kecil yang Dapat Dinavigasi secara Hierarki. Grafik multi-lapis untuk perkiraan pencarian nearest neighbor dengan cepat |
| Regularisasi L1 | "Laso" | Menambahkan norm weight L1 ke loss. Mendorong weight ke nol (sparsitas) |
| Regularisasi L2 | "Punggungan" atau "peluruhan berat" | Menambahkan norm weight L2 kuadrat ke loss. Menyusut weight menuju nol tanpa ketersebaran |
| Jaring Elastis | "L1 + L2" | Menggabungkan regularisasi L1 dan L2. Menangani grup feature yang berkorelasi lebih baik daripada hanya satu saja |

## Bacaan Lanjutan

- [FAISS: Perpustakaan untuk Pencarian Kesamaan yang Efisien](https://github.com/facebookresearch/faiss) - Perpustakaan Meta untuk pencarian ANN skala miliar
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - makalah yang memperkenalkan distance Penggerak Bumi ke GAN
- [Hashing Sensitif Lokalitas (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - algoritma ANN dasar
- [Estimasi Representasi Kata yang Efisien (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec, dengan kesamaan kosinus menjadi default untuk embedding
- [dokumentasi sklearn.neighbors](https://scikit-learn.org/stable/modules/neighbors.html) - panduan praktis untuk metrik distance dan algoritma tetangga di scikit-learn
