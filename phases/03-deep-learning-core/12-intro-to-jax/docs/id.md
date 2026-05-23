# Pengantar JAX

> PyTorch mengubah tensor. TensorFlow membuat grafik. JAX mengkompilasi fungsi murni. Yang terakhir mengubah cara kamu berpikir tentang pembelajaran mendalam.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 03 Lesson 01-10, NumPy dasar
**Waktu:** ~90 menit

## Tujuan Pembelajaran

- Tulis code neural network fungsi murni menggunakan API fungsional JAX (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Jelaskan perbedaan desain utama antara mutasi bersemangat PyTorch dan model kompilasi fungsional JAX
- Terapkan kompilasi jit dan vektorisasi vmap untuk mempercepat loop training dibandingkan dengan Python naif
- Latih jaringan sederhana di JAX dan kontraskan manajemen status eksplisit dengan pendekatan berorientasi objek PyTorch

## Masalah

kamu tahu cara membangun neural network di PyTorch. kamu menentukan `nn.Module`, panggil `.backward()`, langkahkan optimizer. Ini berhasil. Jutaan orang menggunakannya.

Namun PyTorch memiliki batasan yang tertanam dalam DNA-nya: ia menelusuri operasi dengan penuh semangat, satu per satu, dengan Python. Setiap `tensor + tensor` adalah peluncuran kernel terpisah. Setiap langkah training menafsirkan ulang code Python yang sama. Ini berfungsi dengan baik sampai kamu perlu melatih model 540 miliar parameter di 2.048 TPU. Kemudian biaya overhead membunuh kamu.

Google DeepMind melatih Gemini di JAX. Anthropic melatih Claude di JAX. Ini bukanlah operasi kecil -- ini adalah training neural network terbesar yang dijalankan di Bumi. Mereka memilih JAX karena JAX memperlakukan loop training kamu sebagai program yang dapat dikompilasi, bukan rangkaian panggilan Python.

JAX adalah NumPy dengan tiga kekuatan super: diferensiasi otomatis, kompilasi JIT ke XLA, dan vektorisasi otomatis. kamu menulis fungsi yang memproses satu contoh. JAX memberi kamu fungsi yang memproses batch, menghitung gradient, mengkompilasi ke code mesin, dan berjalan di beberapa perangkat. Semua tanpa mengubah fungsi aslinya.

## Konsep

### Filosofi JAX

JAX adalah kerangka fungsional. Tidak ada kelas, tidak ada status yang dapat diubah, tidak ada metode `.backward()`. Sebaliknya:

| PyTorch | JAX |
|---------|-----|
| `nn.Module` kelas dengan status | Fungsi murni: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eksekusi yang penuh semangat | Kompilasi JIT melalui XLA |
| `for x in batch:` putaran manual | `jax.vmap(f)` vektorisasi otomatis |
| `DataParallel` / `FSDP` | `jax.pmap(f)` paralelisme otomatis |
| Dapat diubah `model.parameters()` | Pytree array yang tidak dapat diubah |

Ini bukan preferensi gaya. Ini adalah batasan kompiler. Kompilasi JIT memerlukan fungsi murni -- input yang sama selalu menghasilkan output yang sama, tanpa efek samping. Pembatasan itulah yang memungkinkan percepatan 100x.

### jax.numpy: Permukaan yang Dikenali

JAX mengimplementasikan kembali NumPy API pada akselerator:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

Nama fungsi yang sama. Aturan penyiaran yang sama. Semantik pemotongan yang sama. Namun arraynya menggunakan GPU/TPU, dan setiap operasi dapat dilacak oleh kompiler.

Satu perbedaan penting: array JAX tidak dapat diubah. Tidak `a[0] = 5`. Sebagai gantinya: `a = a.at[0].set(5)`. Hal ini terasa canggung selama seminggu, lalu diklik -- kekekalan inilah yang membuat transformasi seperti `grad`, `jit`, dan `vmap` dapat disusun.

### jax.grad: Autodiff Fungsional

PyTorch melampirkan gradient ke tensor (`.grad`). JAX melampirkan gradient ke fungsi.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
````jax.grad` mengambil fungsi dan mengembalikan fungsi baru yang menghitung gradient. Tidak ada panggilan `.backward()`. Tidak ada grafik komputasi yang disimpan di tensor. Gradient hanyalah fungsi lain yang dapat kamu panggil, buat, atau kompilasi JIT.

Ini disusun secara sewenang-wenang:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Turunan kedua. Derivatif ketiga. Jacobian. goni. Semua dengan menulis `grad`. PyTorch juga dapat melakukan ini (`torch.autograd.functional.hessian`), tetapi sudah terpasang. Di JAX, ini adalah fondasinya.

Kendalanya: `grad` hanya berfungsi pada fungsi murni. Tidak ada pernyataan cetak di dalamnya (dijalankan selama penelusuran, bukan eksekusi). Tidak ada mutasi keadaan eksternal. Tidak ada pembuatan nomor acak tanpa manajemen kunci eksplisit.

### jit: Kompilasi ke XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

Pada panggilan pertama, JAX menelusuri fungsinya -- ia mencatat operasi mana yang terjadi, tanpa menjalankannya. Kemudian jejak tersebut diserahkan ke XLA (Accelerated Linear Algebra), kompiler Google untuk TPU dan GPU. XLA menggabungkan operasi, menghilangkan salinan memori yang berlebihan, dan menghasilkan code mesin yang dioptimalkan.

Panggilan selanjutnya mengabaikan Python sepenuhnya. Code yang dikompilasi berjalan pada akselerator dengan kecepatan C++.

Ketika JIT membantu:
- Langkah-langkah training (perhitungan yang sama diulang ribuan kali)
- Inference (model sama, input berbeda)
- Fungsi apa pun yang dipanggil lebih dari sekali dengan input berbentuk serupa

Saat JIT sakit:
- Fungsi dengan aliran kontrol Python yang bergantung pada nilai (`if x > 0` di mana x adalah array yang dilacak)
- Komputasi sekali pakai (overhead kompilasi melebihi runtime)
- Debugging (pelacakan menyembunyikan eksekusi sebenarnya)

Pembatasan aliran kendali itu nyata. `jax.lax.cond` menggantikan `if/else`. `jax.lax.scan` menggantikan `for` loop. Ini bukan opsional -- ini adalah harga kompilasi.

### vmap: Vektorisasi Otomatis

kamu menulis fungsi yang memproses satu contoh:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` mengangkatnya untuk memproses batch:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` berarti: jangan melakukan batch pada `params` (bersama), batch pada sumbu 0 dari `x`. Tidak ada loop manual `for`. Tidak ada pembentukan kembali. Tidak ada threading dimension batch. JAX mengetahui dimension batch dan membuat vektorisasi seluruh komputasi.

Ini bukan gula sintaksis. `vmap` menghasilkan code vector gabungan yang berjalan 10-100x lebih cepat daripada loop Python. Dan itu terdiri dari `jit` dan `grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Gradient per contoh. Satu baris. Ini hampir mustahil di PyTorch tanpa peretasan.

### pmap: Paralelisme Data di Seluruh Perangkat

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` mereplikasi fungsi di semua perangkat yang tersedia (GPU/TPU) dan membagi batch. Di dalam fungsinya, `jax.lax.pmean` dan `jax.lax.psum` menyinkronkan gradient di seluruh perangkat.

Google melatih Gemini di ribuan chip TPU v5e menggunakan `pmap` (dan penggantinya `shard_map`). Model pemrograman: tulis versi perangkat tunggal, bungkus dengan `pmap`, selesai.

### Pytrees: Struktur Data Universal

JAX beroperasi pada "pytrees" -- kombinasi daftar, tupel, dict, dan array yang bersarang. Parameter model kamu adalah pytree:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```Setiap transformasi JAX -- `grad`, `jit`, `vmap` -- mengetahui cara melintasi pytrees. `jax.tree.map(f, tree)` berlaku `f` pada setiap daun. Beginilah cara optimizer memperbarui semua parameter sekaligus:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

Tidak ada metode `.parameters()`. Tidak ada registrasi parameter. Struktur pohon adalah modelnya.

### Fungsional vs Berorientasi Objek

PyTorch menyimpan status di dalam objek:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX menggunakan fungsi murni dengan status eksplisit:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Params diteruskan. Tidak ada yang disimpan. Tidak ada yang bermutasi. Hal ini membuat setiap fungsi dapat diuji, disusun, dan dikompilasi. Ini juga berarti kamu mengelola sendiri parameternya -- atau menggunakan perpustakaan seperti Flax atau Equinox.

### Ekosistem JAX

JAX memberi kamu primitif. Perpustakaan memberi kamu ergonomi:

| Perpustakaan | Peran | Gaya |
|---------|------|-------|
| **rami** (Google) | Layer neural network | `nn.Module` dengan status eksplisit |
| **Ekuinoks** (Patrick Kidger) | Layer neural network | Berbasis Pytree, Pythonic |
| **Optax** (DeepMind) | Optimizer + jadwal LR | Transformasi gradient yang dapat dikomposisi |
| **Orbax** (Google) | Pos pemeriksaan | Simpan/pulihkan pytrees |
| **CLU** (Google) | Metrik + pencatatan | Utilitas loop training |

Optax adalah perpustakaan optimizer standar. Ini memisahkan transformasi gradient (Adam, SGD, kliping) dari pembaruan parameter, sehingga mudah untuk dibuat:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### Kapan Menggunakan JAX vs PyTorch

| Faktor | JAX | PyTorch |
|--------|-----|---------|
| Dukungan TPU | Kelas satu (Google membuat keduanya) | Dikelola komunitas (torch_xla) |
| Dukungan GPU | Bagus (CUDA via XLA) | Terbaik di kelasnya (CUDA asli) |
| Men-debug | Keras (tracing + kompilasi) | Mudah (bersemangat, baris demi baris) |
| Ekosistem | Berfokus pada penelitian (Flax, Equinox) | Besar-besaran (HuggingFace, torchvision, dll.) |
| Mempekerjakan | Niche (Google/DeepMind/Antropik) | Arus Utama (di mana saja) |
| Training skala besar | Unggul (XLA, pmap, mesh) | Bagus (FSDP, DeepSpeed) |
| Kecepatan pembuatan prototipe | Lebih lambat (overhead fungsional) | Lebih cepat (bermutasi dan pergi) |
| Kesimpulan produksi | Penyajian TensorFlow, Vertex AI | TorchServe, Triton, ONNX |
| Siapa yang menggunakannya | DeepMind (Gemini), Antropis (Claude) | Meta (Llama), OpenAI (GPT), Stabilitas AI |

Jawaban jujurnya: gunakan PyTorch kecuali kamu memiliki alasan khusus untuk menggunakan JAX. Alasannya adalah -- akses TPU, kebutuhan akan gradient per contoh, training multi-perangkat dalam skala besar, atau bekerja di Google/DeepMind/Anthropic.

### Angka Acak di JAX

JAX tidak memiliki keadaan acak global. Setiap operasi acak memerlukan kunci PRNG eksplisit:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

Ini menjengkelkan pada awalnya. Namun ini menjamin reproduktifitas di seluruh perangkat dan kompilasi -- sebuah properti yang `torch.manual_seed` PyTorch tidak dapat menjamin dalam pengaturan multi-GPU.

## Build

### Langkah 1: Penyiapan dan Data

Kami akan melatih MLP 3 lapis di MNIST menggunakan JAX dan Optax. 784 input, dua layer tersembunyi 256 dan 128 neuron, 10 kelas output.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Langkah 2: Inisialisasi Parameter

Tidak ada kelas. Hanya fungsi yang mengembalikan pytree:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

Inisialisasinya, dilakukan secara manual. Tiga kunci PRNG dipisahkan dari satu benih. Setiap weight adalah larik yang tidak dapat diubah dalam dikt bersarang.

### Langkah 3: Maju Teruskan

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```Fungsi murni. Param masuk, prediksi keluar. Tidak ada `self`, tidak ada status tersimpan. `loss_fn` menghitung entropi silang dari awal -- softmax, log, mean negatif.

### Langkah 4: Langkah Training yang Dikompilasi JIT

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` mengembalikan nilai loss dan gradient dalam satu lintasan. Dekorator `@jax.jit` mengkompilasi kedua fungsi ke XLA. Setelah panggilan pertama, setiap langkah training berjalan tanpa menyentuh Python.

### Langkah 5: Lingkaran Latihan

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 zaman. ~97% akurasi pengujian. Epoch pertama lambat (kompilasi JIT). Epoch 2-10 cepat.

Perhatikan apa yang hilang: tidak ada `.zero_grad()`, tidak ada `.backward()`, tidak ada `.step()`. Seluruh pembaruan adalah satu panggilan fungsi yang dibuat. Gradient dihitung, diubah oleh Adam, dan diterapkan ke parameter -- semuanya di dalam `train_step`.

## Pakai

### Rami: Standar Google

Flax adalah perpustakaan neural network JAX yang paling umum. Ia menambahkan `nn.Module` kembali, tetapi dengan pengelolaan status eksplisit:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

Strukturnya sama dengan PyTorch, tetapi `params` terpisah dari modelnya. `model.init()` membuat parameter. `model.apply(params, x)` menjalankan operan ke depan. Objek model tidak memiliki status.

### Ekuinoks: Alternatif Pythonic

Equinox (oleh Patrick Kidger) merepresentasikan model sebagai pytrees:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

Modelnya sendiri adalah pytree. Tidak diperlukan `.apply()`. Parameter hanyalah daun model. Ini lebih dekat dengan cara berpikir JAX.

### Optax: Optimizer yang Dapat Dikomposisi

Optax memisahkan transformasi gradient dari pembaruan:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Pemotongan gradient, pemanasan learning rate, penurunan berat badan -- semuanya disusun sebagai rantai transformasi. Setiap transformasi melihat gradient, memodifikasinya, dan meneruskannya ke transformasi berikutnya. Tidak ada kelas optimizer monolitik.

## Kirim

**Instalasi:**

```bash
pip install jax jaxlib optax flax
```

Untuk dukungan GPU:

```bash
pip install jax[cuda12]
```

Untuk TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Pendapatan performa:**

- Panggilan JIT pertama lambat (kompilasi). Lakukan pemanasan sebelum melakukan benchmarking.
- Hindari loop Python pada array JAX di dalam JIT. Gunakan `jax.lax.scan` atau `jax.lax.fori_loop`.
- `jax.debug.print()` bekerja di dalam JIT. `print()` biasa tidak.
- Profil dengan `jax.profiler` atau TensorBoard. Kompilasi XLA dapat menyembunyikan kemacetan.
- JAX mengalokasikan 75% memori GPU secara default. Setel `XLA_PYTHON_CLIENT_PREALLOCATE=false` untuk menonaktifkan.

**Titik pemeriksaan:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**Lesson ini menghasilkan:**
- `outputs/prompt-jax-optimizer.md` -- prompt untuk memilih konfigurasi optimizer JAX yang tepat
- `outputs/skill-jax-patterns.md` -- keterampilan yang mencakup pola fungsional di JAX

## Latihan

1. Tambahkan putus sekolah ke MLP. Di JAX, dropout memerlukan kunci PRNG -- memasukkan kunci melalui forward pass dan membaginya untuk setiap layer dropout. Bandingkan akurasi tes dengan dan tanpa.

2. Gunakan `jax.vmap` untuk menghitung gradient per contoh untuk kumpulan 32 gambar MNIST. Hitung norm gradient untuk setiap contoh. Contoh manakah yang memiliki gradient terbesar, dan mengapa?

3. Ganti fungsi penerusan manual dengan `mlp_forward(params, x)` generik yang berfungsi untuk sejumlah layer. Gunakan `jax.tree.leaves` untuk menentukan kedalaman secara otomatis.

4. Tolok ukur langkah training dengan dan tanpa `@jax.jit`. Hitung waktu masing-masing 100 langkah. Seberapa besar speedup pada hardware anda? Berapa biaya kompilasi pada panggilan pertama?5. Terapkan kliping gradient dengan menulis `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. Berlatih dengan dan tanpa kliping. Plot norm gradient pada training untuk melihat efeknya.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|----------------------|
| XLA | "Hal yang membuat JAX cepat" | Aljabar Linear yang Dipercepat -- kompiler yang menggabungkan operasi dan menghasilkan kernel GPU/TPU yang dioptimalkan dari grafik komputasi |
| JIT | "Kompilasi tepat waktu" | JAX menelusuri fungsi pada panggilan pertama, mengkompilasi ke XLA, kemudian menjalankan versi kompilasi pada panggilan berikutnya |
| Fungsi murni | "Tidak ada efek samping" | Sebuah fungsi yang keluarannya hanya bergantung pada input -- tanpa status global, tanpa mutasi, tanpa keacakan tanpa kunci eksplisit |
| vpeta | "Pengelompokan otomatis" | Mengubah fungsi yang memproses satu contoh menjadi fungsi yang memproses batch, tanpa menulis ulang |
| peta | "Paralelisme otomatis" | Mereplikasi fungsi di beberapa perangkat dan membagi kumpulan input |
| pohon cemara | "Dict array bersarang" | Struktur bersarang apa pun dari daftar, tupel, dikt, dan larik yang dapat dilintasi dan diubah oleh JAX |
| Menelusuri | "Merekam perhitungan" | JAX menjalankan fungsi dengan nilai abstrak untuk membuat grafik komputasi, tanpa menghitung hasil nyata |
| Autodiff fungsional | "lulusan suatu fungsi" | Menghitung turunan dengan mentransformasikan fungsi, bukan dengan melampirkan penyimpanan gradient ke tensor |
| pajak | "Perpustakaan optimizer JAX" | Pustaka transformasi gradient yang dapat disusun -- Adam, SGD, kliping, penjadwalan -- yang berantai bersama |
| Rami | "Modul nn JAX" | Pustaka neural network Google untuk JAX, menambahkan abstraksi layer sambil menjaga status tetap eksplisit |

## Bacaan Lanjutan

- Dokumentasi JAX: https://jax.readthedocs.io/ -- dokumen resmi, dengan tutorial luar biasa tentang grad, jit, dan vmap
- "JAX: transformasi komposisi program Python+NumPy" (Bradbury et al., 2018) -- makalah asli yang menjelaskan filosofi desain
- Dokumentasi rami: https://flax.readthedocs.io/ -- Pustaka neural network Google untuk JAX
- Patrick Kidger, "Equinox: neural network di JAX melalui PyTrees yang dapat dipanggil dan transformasi yang difilter" (2021) -- alternatif Pythonic untuk Flax
- DeepMind, "Optax: transformasi dan optimization gradient yang dapat dikomposisi" -- pustaka optimizer standar
- "You Don't Know JAX" (Colin Raffel, 2020) -- panduan praktis tentang gotcha dan pola JAX, dari salah satu penulis T5
