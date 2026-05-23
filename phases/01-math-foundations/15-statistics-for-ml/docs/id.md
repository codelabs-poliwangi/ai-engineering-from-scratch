# Statistik untuk Machine Learning

> Statistik adalah cara kamu mengetahui apakah model kamu benar-benar berfungsi atau hanya beruntung.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 1, Lesson 06 (Probabilitas dan Distribusi), 07 (Teorema Bayes)
**Waktu:** ~120 menit

## Tujuan Pembelajaran

- Hitung statistik deskriptif, korelasi Pearson/Spearman, dan covariance matrix dari awal
- Lakukan uji hipotesis (uji-t, chi-kuadrat) dan tafsirkan nilai p dan interval kepercayaan dengan benar
- Gunakan pengambilan sample ulang bootstrap untuk membangun interval kepercayaan untuk metrik apa pun tanpa asumsi distribusi
- Membedakan signifikansi statistik dari signifikansi praktis dengan menggunakan ukuran ukuran efek

## Masalah

kamu melatih dua model. Model A mendapat skor 0,87 pada set pengujian kamu. Model B mendapat skor 0,89. kamu menerapkan Model B. Tiga minggu kemudian, metrik produksi menjadi lebih buruk dari sebelumnya. Apa yang telah terjadi?

Model B sebenarnya tidak mengungguli Model A. Perbedaan 0,02 adalah kebisingan. Kumpulan pengujian kamu terlalu kecil, atau variansnya terlalu tinggi, atau keduanya. kamu mengirimkan keacakan dengan berpakaian sebagai perbaikan.

Hal ini terjadi terus-menerus. Perombakan papan peringkat Kaggle. Makalah yang gagal direproduksi. Pengujian A/B yang menyatakan pemenang berdasarkan beberapa ratus sample. Akar permasalahannya selalu sama: seseorang melewatkan statistik.

Statistik memberi kamu alat untuk membedakan sinyal dari kebisingan. Ini memberi tahu kamu kapan perbedaan itu nyata, seberapa yakin kamu seharusnya, dan berapa banyak data yang kamu perlukan sebelum kamu dapat memercayai suatu hasil. Setiap pipeline ML, setiap perbandingan model, setiap eksperimen memerlukan statistik. Tanpa itu, kamu hanya menebak-nebak.

## Konsep

### Statistik Deskriptif: Meringkas Data kamu

Sebelum membuat model apa pun, kamu perlu mengetahui seperti apa data kamu. Statistik deskriptif memampatkan dataset menjadi beberapa angka yang menangkap bentuknya.

**Ukuran tendensi sentral** jawab "dimana titik tengahnya?"

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

Mean adalah titik keseimbangan. Median adalah tanda setengah jalan. Ketika mereka menyimpang, distribusi kamu menjadi tidak seimbang. Distribusi pendapatan memiliki mean >> median (condong ke kanan dari miliarder). Distribusi loss selama training sering kali memiliki mean << median (condong ke kiri dari sample mudah).

**Ukuran penyebaran** jawab "seberapa sebaran datanya?"

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**Persentil** membagi data yang diurutkan menjadi 100 bagian yang sama. Persentil ke-25 (Q1) berarti 25% nilai berada di bawah titik ini. Persentil ke-50 adalah median. Persentil ke-75 adalah Q3.

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

Di ML, kamu memperhatikan persentil untuk latensi inference, distribusi keyakinan prediksi, dan memahami distribusi kesalahan. Model dengan kesalahan rata-rata rendah tetapi kesalahan P99 yang buruk mungkin tidak berguna untuk aplikasi yang kritis terhadap keselamatan.

**Statistik sample vs populasi.** Saat menghitung varians dari suatu sample, bagilah dengan (n-1), bukan n. Ini adalah koreksi Bessel. Ini mengkompensasi fakta bahwa rata-rata sample kamu bukanlah rata-rata populasi sebenarnya. Dengan n sebagai penyebut, kamu secara sistematis meremehkan varians sebenarnya. Dengan (n-1), estimasi tersebut tidak bias.

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

Dalam praktiknya: jika n besar (ribuan sample), perbedaannya dapat diabaikan. Jika n kecil (puluhan sample), itu penting.

### Korelasi: Bagaimana Variabel Bergerak Bersama

Korelasi mengukur kekuatan dan arah hubungan linier antara dua variabel.

**Koefisien korelasi Pearson** mengukur hubungan linier:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```Pearson mengasumsikan hubungannya linier dan kedua variabel berdistribusi normal. Ini sensitif terhadap outlier. Sebuah titik ekstrim dapat menyeret r dari 0,1 ke 0,9.

**Korelasi peringkat Spearman** mengukur asosiasi monoton:

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**Kapan menggunakan masing-masing:**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**Aturan utama:** korelasi tidak berarti sebab akibat. Penjualan es krim dan kematian akibat tenggelam berkorelasi karena keduanya meningkat di musim panas. Akurasi model kamu dan jumlah parameter berkorelasi, namun menambahkan parameter tidak secara otomatis meningkatkan akurasi (lihat: overfitting).

### Matrix Kovariansi

Kovarian antara dua variabel mengukur seberapa bervariasinya keduanya:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

Untuk feature d, covariance matrix C adalah matrix dxd dengan C[i][j] = Cov(feature_i, feature_j). Entri diagonal C[i][i] adalah varian dari setiap feature.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**Koneksi ke PCA.** PCA secara eigen mendekomposisi covariance matrix. Eigenvector merupakan principal component (arah varian maksimum). Eigenvalue memberi tahu kamu berapa banyak varian yang ditangkap setiap komponen. Hal inilah yang dibahas pada Lesson 10, namun sekarang kamu mengerti mengapa covariance matrix adalah hal yang tepat untuk didekomposisi: matrix ini mengkodekan semua hubungan linier berpasangan dalam data kamu.

**Hubungan dengan korelasi.** Matrix korelasi adalah covariance matrix dari variabel terstandar (masing-masing dibagi dengan deviasi standarnya). Korelasi menormalkan kovarians sehingga semua nilai berada pada [-1, 1].

### Pengujian Hipotesis

Pengujian hipotesis adalah kerangka untuk mengambil keputusan dalam kondisi ketidakpastian. kamu memulai dengan klaim, mengumpulkan data, dan menentukan apakah data tersebut konsisten dengan klaim.

**Pengaturan:**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**Nilai p** adalah probabilitas melihat data ekstrem seperti yang kamu amati, dengan asumsi H0 benar. BUKAN kemungkinan H0 benar. Ini adalah kesalahpahaman paling umum dalam statistik.

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**Interval kepercayaan** memberikan rentang nilai yang masuk akal untuk suatu parameter:

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

Lebar interval kepercayaan memberi tahu kamu tentang presisi. Interval yang lebar berarti ketidakpastian yang tinggi. Interval yang sempit berarti perkiraan kamu tepat (tetapi belum tentu akurat jika data kamu bias).

### Uji-t

Uji-t membandingkan rata-rata. Ada beberapa rasa.

**Uji-t satu sample:** apakah rata-rata populasi berbeda dari nilai yang dihipotesiskan?

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**Uji-t dua sample (independen):** apakah rata-rata dua kelompok berbeda?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**Uji-t berpasangan:** ketika pengukuran dilakukan berpasangan (model yang sama dievaluasi pada pemisahan data yang sama):

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

Di ML, uji-t berpasangan adalah hal yang umum: kamu menjalankan kedua model pada 10 lipatan validasi silang yang sama dan membandingkan skornya secara berpasangan.

### Uji Chi-kuadrat

Uji chi-kuadrat memeriksa apakah frekuensi yang diamati sesuai dengan frekuensi yang diharapkan. Berguna untuk data kategorikal.

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### Pengujian A/B untuk Model ML

Pengujian A/B di ML tidak sama dengan pengujian A/B web. Perbandingan model memiliki tantangan khusus:

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**Prosedurnya:**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### Signifikansi Statistik vs Signifikansi Praktis

Suatu hasil mungkin signifikan secara statistik namun secara praktis tidak berarti. Dengan data yang cukup, perbedaan kecil sekalipun menjadi signifikan secara statistik.

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**Ukuran efek** mengukur seberapa besar perbedaannya, terlepas dari ukuran sample:

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```Selalu laporkan nilai p dan ukuran efeknya. Nilai p memberi tahu kamu apakah perbedaannya nyata. Ukuran efek memberi tahu kamu apakah itu penting.

### Soal Perbandingan Berganda

Saat kamu menguji banyak hipotesis, beberapa di antaranya akan menjadi "signifikan" secara kebetulan. Jika kamu menguji 20 hal pada alpha = 0,05, kamu mengharapkan 1 positif palsu meskipun tidak ada yang nyata.

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Koreksi Bonferroni:** bagi alpha dengan jumlah pengujian.

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

Di ML, hal ini penting saat kamu membandingkan model di beberapa metrik, menguji banyak konfigurasi hyperparameter, atau mengevaluasi beberapa dataset.

### Metode Bootstrap

Bootstrapping memperkirakan distribusi pengambilan sample suatu statistik dengan mengambil sample ulang data kamu dengan penggantinya. Tidak diperlukan asumsi tentang distribusi yang mendasarinya.

**Algoritma:**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Interval kepercayaan bootstrap (metode persentil):**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**Mengapa bootstrap penting untuk ML:**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**Bootstrap untuk perbandingan model:**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

Uji ini lebih kuat dibandingkan uji t berpasangan karena tidak membuat asumsi distribusi.

### Tes Parametrik vs Non-parametrik

**Pengujian parametrik** mengasumsikan distribusi tertentu (biasanya normal):

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**Pengujian non-parametrik** tidak membuat asumsi distribusi:

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**Kapan menggunakan non-parametrik:**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**Kapan menggunakan parametrik:**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

Dalam eksperimen ML, kamu biasanya memiliki n yang kecil (5 atau 10 lipatan validasi silang), sehingga pengujian non-parametrik seperti peringkat bertanda Wilcoxon seringkali lebih tepat daripada pengujian t.

### Teorema Batas Pusat: Implikasi Praktis

CLT mengatakan distribusi rata-rata sample mendekati distribusi normal ketika n bertambah, terlepas dari distribusi populasi yang mendasarinya.

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**Mengapa ini penting bagi ML:**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**Apa yang TIDAK dilakukan CLT:**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### Kesalahan Statistik Umum di ML Papers

1. **Pengujian pada set training.** Menjamin overfitting. Selalu simpan data yang tidak pernah dilihat model selama training.

2. **Tidak ada interval kepercayaan.** Melaporkan angka akurasi tunggal tanpa ketidakpastian membuat hasil tidak dapat direproduksi dan diverifikasi.

3. **Mengabaikan beberapa perbandingan.** Menguji 50 konfigurasi dan melaporkan konfigurasi terbaik tanpa koreksi akan meningkatkan tingkat positif palsu.

4. **Membingungkan signifikansi statistik dan praktis.** Nilai p sebesar 0,001 pada peningkatan akurasi 0,01% tidak berarti.

5. **Menggunakan akurasi pada data yang tidak seimbang.** Akurasi 99% pada dataset dengan kelas negatif 99% berarti model tidak mempelajari apa pun. Gunakan presisi, recall, F1, atau AUC.

6. **Metrik pemilihan ceri.** Hanya melaporkan metrik yang membuat model kamu menang. Evaluasi yang jujur ​​melaporkan semua metrik yang relevan.

7. **Membocorkan informasi di seluruh pemisahan training/pengujian.** Normalisasi sebelum pemisahan, atau menggunakan data masa depan untuk memprediksi masa lalu.

8. **Set pengujian kecil tanpa estimasi varians.** Mengevaluasi 100 sample dan mengklaim peningkatan 2% adalah noise, bukan sinyal.

9. **Dengan asumsi independensi ketika data tidak independen.** Gambar medis dari pasien yang sama, beberapa kalimat dari dokumen yang sama. Pengamatan dalam suatu kelompok berkorelasi.

10. **P-hacking.** Mencoba pengujian, subset, atau kriteria pengecualian yang berbeda hingga kamu mendapatkan p < 0,05. Hasilnya adalah artefak pencarian.

## Build

kamu akan menerapkan:1. **Statistik deskriptif dari awal** (rata-rata, median, modus, deviasi standar, persentil, IQR)
2. **Fungsi korelasi** (Pearson dan Spearman, dengan covariance matrix)
3. **Uji hipotesis** (uji t satu sample, uji t dua sample, uji chi-kuadrat)
4. **Interval kepercayaan bootstrap** (untuk statistik apa pun, tidak diperlukan asumsi)
5. **Simulator pengujian A/B** (menghasilkan data, menguji, memeriksa kesalahan Tipe I dan Tipe II)
6. **Demo signifikansi statistik vs praktis** (menunjukkan bahwa n yang besar membuat segalanya menjadi "signifikan")

Semuanya dari awal, hanya menggunakan `math` dan `random`. Tidak ada numpy, tidak ada scipy.

## Istilah Kunci

| Istilah | Definisi |
|---|---|
| Berarti | Jumlah nilai dibagi dengan hitungan. Sensitif terhadap outlier. |
| median | Nilai tengah dari data yang diurutkan. Kuat terhadap outlier. |
| Deviasi standar | Akar kuadrat dari varians. Ukuran tersebar dalam satuan aslinya. |
| Persentil | Nilai yang berada di bawah persentase data tertentu. |
| IQR | Rentang interkuartil. Q3 dikurangi Q1. Penyebarannya di kalangan 50% menengah. |
| Korelasi Pearson | Mengukur hubungan linier antara dua variabel. Rentang [-1, 1]. |
| Korelasi Spearman | Mengukur asosiasi monotonik menggunakan peringkat. |
| Covariance matrix | Matrix kovarian berpasangan antara semua feature. |
| Hipotesis nol | Asumsi default tidak ada pengaruh atau tidak ada perbedaan. |
| nilai-p | Probabilitas data yang ekstrim ini mengingat hipotesis nolnya benar. |
| Interval kepercayaan | Rentang nilai yang masuk akal untuk suatu parameter pada tingkat kepercayaan tertentu. |
| uji-t | Menguji apakah rata-rata berbeda secara signifikan. Menggunakan distribusi-t. |
| Uji chi-kuadrat | Menguji apakah frekuensi yang diamati berbeda dari frekuensi yang diharapkan. |
| Ukuran efek | Besarnya perbedaan, tidak bergantung pada ukuran sample. D Cohen adalah hal yang umum. |
| Koreksi Bonferroni | Membagi ambang batas signifikansi dengan jumlah pengujian untuk mengontrol positif palsu. |
| tali sepatu | Pengambilan sample ulang dengan penggantian untuk memperkirakan distribusi pengambilan sample. |
| Kesalahan tipe I | Positif palsu. Menolak H0 padahal benar. |
| Kesalahan tipe II | Negatif palsu. Gagal menolak H0 padahal H0 salah. |
| Kekuatan statistik | Kemungkinan menolak H0 palsu dengan benar. Daya = 1 dikurangi tingkat kesalahan Tipe II. |
| Teorema limit pusat | Sample berarti menyatu ke distribusi normal seiring bertambahnya ukuran sample. |
| Uji parametrik | Mengasumsikan distribusi tertentu untuk data (biasanya normal). |
| Uji non parametrik | Tidak membuat asumsi distribusi. Bekerja pada peringkat atau tanda. |
