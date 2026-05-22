# Vectors, Matrices & Operations

> Setiap neural network pada dasarnya adalah matrix multiplication dengan beberapa langkah tambahan.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## Learning Objectives

- Membangun `Matrix` class dengan element-wise operations, matrix multiplication, transpose, determinant, dan inverse
- Membedakan element-wise multiplication dari matrix multiplication, lalu menjelaskan kapan masing-masing digunakan
- Mengimplementasikan satu dense neural network layer (`relu(W @ x + b)`) hanya dengan `Matrix` class yang dibuat from scratch
- Menjelaskan broadcasting rules dan bagaimana bias addition bekerja di neural network frameworks

## The Problem

Kamu ingin membangun neural network. Kamu membaca code dan melihat baris ini:

```
output = activation(weights @ input + bias)
```

Operator `@` adalah matrix multiplication. `weights` adalah matrix. `input` adalah vector. Kalau kamu belum paham operasi ini, baris tersebut terasa seperti magic. Kalau sudah paham, baris itu adalah seluruh forward pass dari sebuah layer dalam tiga operasi.

Setiap image yang diproses model adalah matrix berisi pixel values. Setiap word embedding adalah vector. Setiap layer dalam neural network adalah matrix transformation. Kamu tidak bisa membangun AI systems dengan nyaman tanpa lancar membaca matrix operations, sama seperti kamu tidak bisa menulis code tanpa paham variables.

Lesson ini membangun fluency tersebut from scratch.

## The Concept

### Vectors: ordered lists of numbers

Vector adalah list angka yang memiliki direction dan magnitude. Dalam AI, vectors merepresentasikan data points, features, atau parameters.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

Vector 2D `[3, 4]` menunjuk ke coordinates (3, 4) pada plane. Panjangnya atau magnitude-nya adalah 5, mengikuti segitiga 3-4-5.

### Matrices: grids of numbers

Matrix adalah grid 2D: rows dan columns. Matrix berukuran m x n memiliki m rows dan n columns.

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

Dalam neural networks, weight matrices mentransform input vectors menjadi output vectors. Layer dengan 784 inputs dan 128 outputs memakai weight matrix berukuran 128x784.

### Why shapes matter

Matrix multiplication punya aturan ketat: `(m x n) @ (n x p) = (m x p)`. Inner dimensions harus match.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

Kalau kamu mendapat shape mismatch error di PyTorch, biasanya inilah penyebabnya.

### The operations map

| Operation | What it does | Neural network use |
|-----------|-------------|-------------------|
| Addition | Element-wise combine | Adding bias to output |
| Scalar multiply | Scale every element | Learning rate * gradients |
| Matrix multiply | Transform vectors | Layer forward pass |
| Transpose | Flip rows and columns | Backpropagation |
| Determinant | Single number summary | Checking invertibility |
| Inverse | Undo a transformation | Solving linear systems |
| Identity | Do-nothing matrix | Initialization, residual connections |

### Element-wise vs matrix multiplication

Perbedaan ini sering membuat beginner tersandung.

Element-wise berarti mengalikan posisi yang sama. Kedua matrices harus memiliki shape yang sama.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

Matrix multiplication berarti dot products antara rows dan columns. Inner dimensions harus match.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Operasinya berbeda, hasilnya berbeda, dan rules-nya juga berbeda.

### Broadcasting

Saat kamu menambahkan bias vector ke matrix of outputs, shape-nya tidak selalu sama. Broadcasting memperluas array yang lebih kecil agar cocok.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Setiap modern framework melakukan ini otomatis. Memahami broadcasting mencegah kebingungan ketika shapes tampak berbeda, tetapi code tetap berjalan.

## Build It

### Step 1: Vector class

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Step 2: Matrix class with core operations

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Step 3: See it work

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Step 4: Connect to neural networks

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

Ini adalah satu dense layer: `output = relu(W @ x + b)`. Setiap dense layer di setiap neural network melakukan pola ini.

## Use It

NumPy melakukan semua operasi di atas dengan lebih sedikit baris dan jauh lebih cepat.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Operator `@` di Python memanggil `__matmul__`. NumPy mengimplementasikannya dengan optimized BLAS routines yang ditulis dalam C dan Fortran. Math-nya sama, tetapi bisa 100x lebih cepat.

Broadcasting di NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy otomatis melakukan broadcast bias 1D ke kedua rows. Inilah cara bias addition bekerja di setiap neural network framework.

## Ship It

Lesson ini menghasilkan prompt untuk mengajar matrix operations melalui geometric intuition. Lihat `outputs/prompt-matrix-operations.md`.

`Matrix` class yang dibangun di sini menjadi foundation untuk mini neural network framework yang akan dibuat pada Phase 3, Lesson 10.

## Exercises

1. **Verify the inverse.** Kalikan `A @ A.inverse_2x2()` dan pastikan hasilnya adalah identity matrix. Coba dengan tiga matrix 2x2 yang berbeda. Apa yang terjadi ketika determinant bernilai zero?

2. **Implement 3x3 inverse.** Perluas `Matrix` class agar bisa menghitung inverse untuk matrix 3x3 menggunakan adjugate method. Test terhadap `np.linalg.inv` milik NumPy.

3. **Build a two-layer network.** Hanya dengan `Matrix` class milikmu, buat two-layer neural network: input (3) -> hidden (4) -> output (2). Initialize random weights, jalankan forward pass, dan verifikasi semua shapes benar.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Vector | "An arrow" | Ordered list of numbers. Dalam AI: point di high-dimensional space. |
| Matrix | "A table of numbers" | Linear transformation. Matrix memetakan vectors dari satu space ke space lain. |
| Matrix multiply | "Just multiply the numbers" | Dot products antara setiap row dari matrix pertama dan setiap column dari matrix kedua. Order matters. |
| Transpose | "Flip it" | Menukar rows dan columns. Mengubah matrix m x n menjadi n x m. Penting dalam backpropagation. |
| Determinant | "Some number from the matrix" | Mengukur seberapa besar matrix menskalakan area (2D) atau volume (3D). Zero berarti transformation menghancurkan satu dimension. |
| Inverse | "Undo the matrix" | Matrix yang membalik transformation. Hanya ada jika determinant tidak zero. |
| Identity matrix | "The boring matrix" | Matrix equivalent dari multiplying by 1. Dipakai dalam residual connections (ResNets). |
| Broadcasting | "Magic shape fixing" | Memperluas array kecil agar match dengan array besar dengan repeating pada missing dimensions. |
| Element-wise | "Regular multiplication" | Mengalikan posisi yang sama. Kedua arrays harus memiliki shape yang sama atau broadcastable. |

## Further Reading

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - visual intuition untuk setiap operation di lesson ini
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - aturan persis yang diikuti NumPy
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - reference singkat untuk linear algebra yang sering dipakai di ML
