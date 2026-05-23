# Membangun Tokenizer dari Awal

> Lesson 01 memberimu mainan. Lesson ini memberi kamu senjata.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 10, Lesson 01 (Tokenizer: BPE, WordPiece, SentencePiece)
**Waktu:** ~90 menit

## Tujuan Pembelajaran

- Membangun tokenizer BPE tingkat produksi yang menangani Unicode, normalisasi spasi putih, dan token khusus
- Menerapkan fallback tingkat byte sehingga tokenizer dapat menyandikan input apa pun (termasuk emoji, CJK, dan code) tanpa token yang tidak diketahui
- Tambahkan pola regex pra-tokenization yang membagi teks pada batas kata sebelum menerapkan penggabungan BPE
- Latih tokenizer khusus pada korpus dan evaluasi rasio kompresinya terhadap tiktoken pada teks multibahasa

## Masalah

Tokenizer BPE kamu dari Lesson 01 berfungsi pada teks bahasa Inggris. Sekarang lemparkan orang Jepang ke sana. Atau emoji. Atau code Python dengan tab dan spasi campuran.

Itu rusak.

Bukan karena BPE salah -- karena implementasinya belum tuntas. Tokenizer produksi menangani byte mentah dalam pengkodean apa pun, menormalkan Unicode sebelum pemisahan, mengelola token khusus yang tidak pernah digabungkan, menghubungkan pra-tokenization dengan pemisahan subkata, dan melakukan semua ini dengan cukup cepat untuk tidak menghambat jalur pipa training yang memproses 15 triliun token.

Tokenizer GPT-2 memiliki 50.257 token. Llama 3 memiliki 128.256. GPT-4 memiliki sekitar 100.000. Ini bukan angka mainan. Tabel gabungan di belakang kosakata tersebut dilatih pada ratusan gigabyte teks, dan mesin di sekitarnya -- normalisasi, pra-tokenization, injeksi token khusus, pemformatan templat obrolan -- adalah yang memisahkan tokenizer yang menangani "hello world" dari tokenizer yang menangani seluruh internet.

kamu akan membangun mesin itu.

## Konsep

### Pipeline Pipa Penuh

Tokenizer produksi bukanlah satu algoritma. Ini adalah rangkaian lima phase, masing-masing memecahkan masalah yang berbeda.

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

Setiap phase memiliki pekerjaan tertentu:

| Phase | Apa Fungsinya | Mengapa Itu Penting |
|-------|-------------|----------------|
| Normalisasikan | NFKC Unicode, huruf kecil opsional, aksen strip opsional | Ligatur "fi" (U+FB01) menjadi "fi" (dua karakter). Tanpa ini, kata yang sama mendapat tanda yang berbeda. |
| Pra-Tokenization | Pisahkan teks menjadi beberapa bagian sebelum BPE | Mencegah BPE bergabung melintasi batas kata. "kucing" tidak boleh menghasilkan token "ec". |
| Penggabungan BPE | Terapkan aturan penggabungan yang dipelajari ke urutan byte | Kompresi inti. Mengubah byte mentah menjadi token subkata. |
| Token Khusus | Suntikkan [BOS], [EOS], [PAD], penanda templat obrolan | Token ini memiliki ID tetap. Mereka tidak pernah ikut serta dalam penggabungan BPE. Model membutuhkannya untuk struktur. |
| Pemetaan ID | Konversikan string token menjadi ID bilangan bulat | Model melihat bilangan bulat, bukan string. |

### BPE Tingkat Byte

Tokenizer lesson 01 beroperasi pada byte UTF-8. Itu adalah keputusan yang tepat. Namun kami melewatkan sesuatu yang penting: apa yang terjadi jika byte tersebut tidak valid UTF-8?

BPE tingkat byte menyelesaikan masalah ini dengan memperlakukan setiap nilai byte yang mungkin (0-255) sebagai token yang valid. Kosakata dasar kamu tepat 256 entri. File apa pun -- teks, biner, rusak -- dapat diberi token tanpa menghasilkan token yang tidak diketahui.

GPT-2 menambahkan trik: petakan setiap byte ke karakter Unicode yang dapat dicetak sehingga kosakata tetap dapat dibaca manusia. Byte 0x20 (spasi) menjadi karakter "G" dalam pemetaannya. Ini murni kosmetik. Algoritme tidak peduli.Kekuatan sebenarnya: BPE tingkat byte menangani setiap bahasa di dunia. Karakter Cina masing-masing berukuran 3 UTF-8 byte. Bahasa Jepang bisa 3-4 byte. Arab, Dewanagari, emoji -- semuanya hanya urutan byte. Algoritme BPE menemukan pola dalam urutan byte ini dengan cara yang persis sama seperti menemukan pola dalam byte ASCII bahasa Inggris.

### Pra-Tokenization

Sebelum BPE menyentuh teks kamu, kamu perlu membaginya menjadi beberapa bagian. Hal ini mencegah algoritme penggabungan membuat token yang menjangkau batas kata.

GPT-2 menggunakan pola regex untuk membagi teks:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Pola ini terbagi menjadi kontraksi ("jangan" menjadi "jangan" + "'t"), kata-kata dengan spasi, angka, tanda baca, dan spasi di awal opsional. Spasi terdepan tetap melekat pada kata -- sehingga "kucing" menjadi [" si", " kucing"], bukan ["si", " ", "kucing"].

Llama menggunakan SentencePiece, yang melewatkan regex seluruhnya. Ini memperlakukan aliran byte mentah sebagai satu urutan panjang dan memungkinkan algoritma BPE mengetahui batasannya. Ini lebih sederhana tetapi memberi BPE lebih banyak kebebasan untuk membuat token lintas kata.

Pilihan itu penting. Regex GPT-2 mencegah tokenizer mempelajari bahwa "the" di akhir satu kata dan "the" di awal kata berikutnya harus digabungkan. SentencePiece mengizinkannya, yang terkadang menghasilkan kompresi yang lebih efisien tetapi token yang kurang dapat diinterpretasikan.

### Token Khusus

Setiap tokenizer produksi mencadangkan ID token untuk penanda struktural:

| Tanda | Tujuan | Digunakan Oleh |
|-------|---------|---------|
| `[BOS]` / `<s>` | Awal urutan | Lama 3, GPT |
| `[EOS]` / `</s>` | Akhir urutan | Semua model |
| `[PAD]` | Padding untuk penyelarasan batch | BERT, T5 |
| `[UNK]` | Token tidak dikenal (BPE tingkat byte menghilangkan ini) | BERT, Bagian Kata |
| `<\|im_start\|>` | Batas pesan obrolan dimulai | ObrolanGPT, Qwen |
| `<\|im_end\|>` | Batas pesan obrolan berakhir | ObrolanGPT, Qwen |
| `<\|user\|>` | Penanda giliran pengguna | Lama 3 |
| `<\|assistant\|>` | Asisten penanda belok | Lama 3 |

Token khusus tidak pernah dibagi berdasarkan BPE. Mereka dicocokkan tepat sebelum algoritme penggabungan dijalankan, diganti dengan ID tetapnya, dan teks di sekitarnya diberi token secara normal.

### Templat Obrolan

Di sinilah kebanyakan orang menjadi bingung dan sebagian besar implementasinya gagal.

Saat kamu mengirim pesan ke model obrolan, API menerima daftar pesan:

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

Model tidak melihat JSON. Ia melihat urutan token datar. Templat obrolan mengubah pesan menjadi urutan datar menggunakan token khusus. Setiap model melakukan hal ini secara berbeda:

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

Jika templatnya salah, modelnya akan menghasilkan sampah. Itu dilatih dalam satu format yang tepat. Setiap penyimpangan -- baris baru yang hilang, token yang ditukar, spasi tambahan -- menempatkan input di luar distribusi training.

### Kecepatan

Python terlalu lambat untuk tokenization produksi.

tiktoken (OpenAI) ditulis dalam Rust dengan binding Python. Tokenizer HuggingFace juga merupakan Rust. Kalimat Kalimat adalah C++. Ini mencapai kecepatan 10-100x dibandingkan Python murni.

Sebagai gambaran: melakukan tokenization 15 triliun token untuk pra-training Llama 3 dengan 1 juta token per detik (Python cepat) akan memakan waktu 174 hari. Dengan 100 juta token per detik (Rust), dibutuhkan 1,7 hari.

kamu sedang membangun dengan Python untuk memahami algoritme. Dalam produksi, kamu akan menggunakan implementasi yang dikompilasi dan hanya menyentuh pembungkus Python.

## Build

### Langkah 1: Pengkodean Tingkat ByteYayasan. Ubah string apa pun menjadi rangkaian byte, petakan setiap byte menjadi karakter yang dapat dicetak untuk ditampilkan, dan balikkan prosesnya.

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

Uji teks multibahasa untuk melihat jumlah byte:

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"halo" adalah 5 byte. "你好" berukuran 6 byte (3 per karakter). Emoji api berukuran 4 byte. Tokenizer tingkat byte tidak peduli bahasa apa itu. Byte adalah byte.

### Langkah 2: Pra-Tokenizer dengan Regex

Pisahkan teks menjadi beberapa bagian menggunakan pola regex GPT-2. Setiap potongan diberi token secara independen oleh BPE.

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

Modul `regex` mendukung pelolosan properti Unicode (`\p{L}` untuk huruf, `\p{N}` untuk angka). Modul perpustakaan standar `re` tidak, jadi kita kembali ke kelas karakter ASCII. Untuk tokenizer multibahasa produksi, instal `regex`.

Cobalah:

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

Spasi terdepan tetap melekat pada kata tersebut. Kontraksi terbelah pada tanda apostrof. Tanda baca menjadi bagian tersendiri. BPE tidak akan pernah menggabungkan token melintasi batas-batas ini.

### Langkah 3: BPE pada Urutan Byte

Algoritme inti dari Lesson 01, tetapi sekarang beroperasi pada potongan yang telah diberi token sebelumnya secara independen.

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### Langkah 4: Penanganan Token Khusus

Token khusus memerlukan pencocokan tepat dan ID tetap. Mereka mengabaikan BPE sepenuhnya.

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### Langkah 5: Kelas Tokenizer Lengkap

Gabungkan semuanya: normalisasi, pisahkan pada token khusus, pra-tokenization, penggabungan BPE, petakan ke ID.

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### Langkah 6: Tes Multibahasa

Ujian sesungguhnya. Lemparkan bahasa Inggris, Cina, emoji, dan code ke dalamnya.

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

Karakter Cina masing-masing menghasilkan 3 byte. Emoji menghasilkan 4 byte. Tidak satu pun dari ini yang merusak tokenizer. Tidak ada yang menghasilkan token yang tidak diketahui. Itulah kekuatan BPE tingkat byte.

## Pakai

### Membandingkan Tokenizer Asli

Muat tokenizer sebenarnya dari Llama 3, GPT-4, dan Mistral. Lihat bagaimana masing-masing menangani paragraf multibahasa yang sama.

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

kamu akan melihat jumlah token berbeda untuk teks yang sama. Llama 3 dengan kosakata 128K lebih agresif dalam menggabungkan pola-pola umum. GPT-4 dengan 100K berada di tengah. Mistral dengan 32K menghasilkan lebih banyak token tetapi memiliki layer embedding yang lebih kecil.

Pengorbanannya selalu sama: kosakata yang lebih besar berarti urutan yang lebih pendek tetapi lebih banyak parameter.

## Kirim

Lesson ini menghasilkan prompt untuk membuat dan men-debug tokenizer produksi. Lihat `outputs/prompt-tokenizer-builder.md`.

## Latihan

1. **Mudah:** Tambahkan metode `get_token_bytes(id)` yang menampilkan byte mentah untuk ID token apa pun. Gunakan ini untuk memeriksa apa yang sebenarnya diwakili oleh token gabungan kamu yang paling umum.
2. **Medium:** Menerapkan pra-tokenizer gaya Llama yang membagi spasi dan angka, namun tetap mempertahankan spasi di depan. Bandingkan kosakatanya dengan pendekatan regex GPT-2 pada korpus yang sama.
3. **Sulit:** Tambahkan metode templat obrolan yang mengambil daftar pesan `{"role": ..., "content": ...}` dan menghasilkan urutan token yang benar untuk format obrolan Llama 3. Uji terhadap implementasi HuggingFace.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|----------------------|
| BPE tingkat byte | "Tokenizer yang berfungsi pada byte" | BPE dengan kosakata dasar nilai 256 byte -- menangani input apa pun tanpa token yang tidak diketahui |
| Pra-tokenization | "Pemisahan sebelum BPE" | Regex atau pemisahan berbasis aturan yang mencegah penggabungan BPE melintasi batas kata |
| Normalisasi NFKC | "Pembersihan Unicode" | Decomposition kanonik diikuti dengan komposisi kompatibilitas -- pengikat "fi" menjadi "fi", lebar penuh "A" menjadi "A" |
| Templat obrolan | "Bagaimana pesan menjadi token" | Format yang tepat untuk mengonversi daftar pesan peran/konten menjadi urutan token datar -- khusus model dan harus sesuai dengan format training |
| Token khusus | "Token kontrol" | ID token cadangan yang melewati BPE -- [BOS], [EOS], [PAD], penanda obrolan -- sama persis sebelum digabungkan |
| Kesuburan | "Token per kata" | Rasio token output terhadap kata input -- 1,3 untuk bahasa Inggris di GPT-4, 2-3 untuk bahasa Korea, lebih tinggi berarti konteks terbuang |
| tiktok | "Tokenizer OpenAI" | Implementasi Rust BPE dengan binding Python -- 10-100x lebih cepat dibandingkan Python |
| Gabungkan tabel | "Kosakata" | Daftar urutan gabungan pasangan byte yang dipelajari selama training -- ini ADALAH pengetahuan yang dipelajari tokenizer |

## Bacaan Lanjutan

- [Sumber tiktoken OpenAI](https://github.com/openai/tiktoken) -- Implementasi Rust BPE yang digunakan oleh GPT-3.5/4
- [Tokenizer HuggingFace](https://github.com/huggingface/tokenizers) -- Pustaka tokenizer Rust yang mendukung BPE, WordPiece, Unigram
- [Makalah Llama 3 (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- detail tentang 128 ribu kosakata dan training tokenizer
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- tokenization tanpa bahasa
- [Sumber tokenizer GPT-2](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- pemetaan byte-ke-Unicode asli
