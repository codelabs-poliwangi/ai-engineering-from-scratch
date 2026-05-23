# Caching Cepat dan Caching Konteks

> System prompt kamu adalah 4.000 token. Konteks RAG kamu adalah 20.000 token. kamu mengirim keduanya dengan setiap permintaan. kamu juga membayar keduanya — setiap saat. Caching yang cepat memungkinkan penyedia menjaga awalan tersebut tetap hangat dan menagih kamu 10% dari tarif normal untuk penggunaan kembali. Jika digunakan dengan benar, ini akan mengurangi biaya inference sebesar 50–90% dan latensi token pertama sebesar 40–85%.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 11 · 01 (Rekayasa Cepat), Fase 11 · 05 (Rekayasa Konteks), Fase 11 · 11 (Caching dan Biaya)
**Waktu:** ~60 menit

## Masalah

Agen pengkodean mengirimkan system prompt 15.000 token yang sama kepada Claude di setiap percakapan. Dua puluh putaran dengan token input $3/M berarti biaya input saja sebesar $0,90 — sebelum pesan aktual apa pun dari pengguna. Kalikan dengan 10.000 percakapan harian dan tagihannya mencapai $9.000/hari untuk SMS yang tidak pernah berubah.

kamu tidak dapat mengecilkan prompt tanpa merusak kualitas. kamu tidak dapat menghindari pengirimannya — model membutuhkannya di setiap kesempatan. Satu-satunya langkah adalah berhenti membayar harga penuh untuk awalan yang telah dilihat oleh penyedia.

Langkah itu adalah cache yang cepat. Anthropic mengirimkannya pada bulan Agustus 2024 (dengan varian TTL yang diperpanjang 1 jam pada tahun 2025), OpenAI mengotomatiskannya pada akhir tahun itu, Google mengirimkan cache konteks eksplisit bersama Gemini 1.5, dan ketiganya kini menawarkannya sebagai feature kelas satu pada model terdepan mereka.

## Konsep

![Caching cepat: tulis sekali, baca murah](../assets/prompt-caching.svg)

**Mekaniknya.** Jika awalan permintaan cocok dengan awalan dari permintaan terbaru, penyedia akan menyajikan cache KV dari proses sebelumnya alih-alih mengkodekan ulang token. kamu membayar sedikit premi tulis untuk pertama kalinya dan diskon baca yang besar setiap kali berikutnya.

**Tiga pilihan penyedia pada tahun 2026.**

| Penyedia | Gaya API | Dapatkan diskon | Tulis premium | TTL bawaan | Minimal dapat di-cache |
|---------|-----------|--------------|---------------|-------------|---------------|
| Antropik | Penanda eksplisit `cache_control` pada blok konten | Diskon 90% input | biaya tambahan 25% | 5 menit (dapat diperpanjang hingga 1 jam) | 1.024 token (Sonnet/Opus), 2.048 (Haiku) |
| OpenAI | Deteksi awalan otomatis | Diskon 50% input | tidak ada | Hingga 1 jam (usaha terbaik) | 1.024 token |
| Google (Gemini) | Eksplisit `CachedContent` API | Biaya penyimpanan; baca pada ~25% dari normal | Biaya penyimpanan per token·jam | Ditetapkan pengguna (default 1 jam) | 4.096 token (Flash), 32.768 (Pro) |

**Invarian.** Ketiga awalan cache saja. Jika ada token yang berbeda di antara permintaan, semua yang terjadi setelah token yang berbeda pertama adalah sebuah kesalahan. Letakkan bagian *stabil* di atas, bagian *variabel* di bawah.

### Tata letak yang ramah cache

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

Melanggar prompt — letakkan pesan pengguna di atas system prompt, sisipkan pengambilan dinamis di antara beberapa pengambilan — dan cache tidak akan pernah muncul.

### Perhitungan titik impas

Premi tulis 25% Anthropic berarti blok yang di-cache harus dibaca setidaknya dua kali untuk menghemat uang. 1 tulis + 1 baca rata-rata 0,675x biaya per permintaan (menghemat 32%); 1 tulis + 10 baca rata-rata 0,205x (menghemat 80%). Aturan praktisnya: cache apa pun yang kamu harapkan untuk digunakan kembali setidaknya 3 kali dalam TTL.

## Build

### Langkah 1: Caching prompt antropik dengan penanda eksplisit

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

Penanda `cache_control` memberitahu Anthropic untuk menyimpan blok selama 5 menit. Penggunaan kembali dalam jendela itu berhasil; digunakan kembali setelah habis masa berlakunya dan menulis lagi.

**Bidang penggunaan respons:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```Periksa kedua kolom di CI — jika `cache_read_input_tokens` tetap nol di seluruh permintaan, kunci cache kamu akan hilang.

### Langkah 2: TTL diperpanjang satu jam

Untuk pekerjaan batch yang berjalan lama, masa berlaku default 5 menit akan berakhir di antara pekerjaan. Setel `ttl`:

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

TTL 1 jam berharga 2x lipat premi tulis (50% dibandingkan harga dasar, bukan 25%) tetapi memberikan hasil yang cepat pada batch mana pun yang menggunakan kembali awalan lebih dari 5 kali.

### Langkah 3: Caching otomatis OpenAI

OpenAI tidak memberi kamu apa pun untuk dikonfigurasi. Awalan apa pun di atas 1.024 token yang cocok dengan permintaan terbaru mendapat diskon 50% secara otomatis.

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

Aturan tata letak ramah cache yang sama juga berlaku. Dua hal mematikan cache OpenAI yang tidak mematikan cache Anthropic: mengubah bidang `user` (digunakan sebagai komponen kunci cache) dan menyusun ulang alat.

### Langkah 4: Caching konteks eksplisit Gemini

Gemini memperlakukan cache sebagai objek kelas satu yang kamu buat dan beri nama:

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini membebankan biaya penyimpanan per token·jam selama cache masih ada, dan membaca pada ~25% dari kecepatan input normal. Ini adalah bentuk yang tepat ketika kamu menggunakan kembali prompt raksasa yang sama di banyak sesi selama berhari-hari.

### Langkah 5: mengukur tingkat keberhasilan dalam produksi

Lihat `code/main.py` untuk mengetahui simulasi akuntan tiga penyedia yang melacak jumlah tulis/baca/kehilangan dan menghitung biaya gabungan per 1.000 permintaan. Gerbang diterapkan pada tingkat pencapaian target — sebagian besar penyiapan Antropik produksi akan melihat >80% fraksi baca setelah pemanasan.

## Kesalahan yang masih dikirimkan pada tahun 2026

- **Stempel waktu dinamis di bagian atas.** `"Current time: 2026-04-22 15:30:02"` di bagian atas system prompt. Setiap permintaan meleset. Pindahkan stempel waktu di bawah titik henti cache.
- **Penyusunan ulang alat.** Membuat serial alat dalam urutan yang stabil — perombakan dict antar penerapan akan menghentikan setiap hit.
- **Teks ​​bebas yang hampir duplikat.** "kamu sangat membantu." vs "Kamu adalah asisten yang sangat membantu." — perbedaan satu byte = kesalahan total.
- **Blok terlalu kecil.** Anthropic menerapkan lantai 1.024 token (2.048 untuk Haiku). Blok yang lebih kecil secara diam-diam tidak melakukan cache.
- **Dasbor biaya buta.** Pisahkan "token input" menjadi cache dan tidak cache. Kalau tidak, penurunan lalu lintas tampak seperti kemenangan cache.

## Pakai

Tumpukan cache tahun 2026:

| Situasi | Pilih |
|-----------|------|
| Agen dengan prompt sistem 10k+ yang stabil, banyak putaran | Antropis `cache_control` dengan TTL 5 menit |
| Pekerjaan batch menggunakan kembali awalan selama 30+ menit | Antropis dengan `ttl: "1h"` |
| Endpoint tanpa server di GPT-5, tanpa infra | OpenAI otomatis (buat awalan kamu stabil dan panjang) |
| Penggunaan kembali korpus code/dokumen raksasa selama beberapa hari | Gemini eksplisit `CachedContent` |
| Penggantian lintas penyedia | Pertahankan tata letak awalan yang dapat di-cache tetap sama di seluruh penyedia sehingga setiap klik berfungsi |

Gabungkan dengan caching semantik (Fase 11 · 11) untuk layer pesan pengguna: caching cepat menangani penggunaan kembali *identik-token*, caching semantik menangani penggunaan kembali *identik dengan makna.

## Kirim

Simpan `outputs/skill-prompt-caching-planner.md`:

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## Latihan1. **Mudah.** Lakukan percakapan 10 putaran dengan system prompt 5.000 token melawan Claude. Jalankan tanpa `cache_control` lalu dengan. Laporkan tagihan token input untuk masing-masing.
2. **Medium.** Tulis test harness yang, dengan templat cepat dan log permintaan, menghitung tingkat keberhasilan yang diharapkan dan penghematan dolar per penyedia (Anthropic 5m, Anthropic 1h, OpenAI otomatis, Gemini eksplisit).
3. **Sulit.** Buat optimizer tata letak: dengan prompt dan daftar kolom bertanda `stable=True/False`, tulis ulang prompt untuk menempatkan satu titik henti sementara cache pada posisi ramah cache maksimum tanpa kehilangan informasi. Verifikasi pada titik akhir Antropik yang sebenarnya.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Caching cepat | "Membuat prompt panjang menjadi murah" | Menggunakan kembali cache KV sisi penyedia untuk awalan yang cocok; Diskon 50-90% untuk token input berulang. |
| `cache_control` | "Penanda Antropik" | Atribut blok konten yang menyatakan "semua yang ada di sini dapat disimpan dalam cache"; `{"type": "ephemeral"}`. |
| Tulis cache | "Membayar premi" | Permintaan pertama yang mengisi cache; ditagih dengan tingkat input ~1,25x di Anthropic, gratis di OpenAI. |
| Baca cache | "Diskon" | Permintaan selanjutnya yang cocok dengan awalan; ditagih sebesar 10% (Antropis), 50% (OpenAI), ~25% (Gemini). |
| TTL | "Berapa lama umurnya" | Beberapa detik cache tetap hangat; Anthropic 5m default (dapat diperpanjang 1 jam), upaya terbaik OpenAI hingga 1 jam, set pengguna Gemini. |
| TTL Diperpanjang | "Cache Antropik 1 jam" | `{"type": "ephemeral", "ttl": "1h"}`; 2x menulis premium tetapi layak untuk digunakan kembali secara batch. |
| Pertandingan awalan | "Mengapa cache saya hilang" | Cache hanya ditemukan ketika setiap token dari awal hingga breakpoint identik dengan byte. |
| Caching konteks (Gemini) | "Yang eksplisit" | Objek cache yang diberi nama dan ditagih penyimpanan milik Google; terbaik untuk penggunaan kembali corpora besar selama beberapa hari. |

## Bacaan Lanjutan

- [Antropik — Caching cepat](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`, TTL 1 jam, tabel titik impas.
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — pencocokan awalan otomatis.
- [Google — Cache konteks](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API dan harga penyimpanan.
- [Rekayasa antropik — Caching cepat untuk weight kerja konteks panjang](https://www.anthropic.com/news/prompt-caching) — postingan peluncuran asli dengan nomor latensi.
- Fase 11 · 05 (Rekayasa Konteks) — di mana memotong prompt agar cache dapat mendarat.
- Fase 11 · 11 (Caching dan Biaya) — memasangkan cache cepat dengan cache semantik pada pesan pengguna.
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — model memori KV-cache yang mengekspos cache ke pengguna; menjelaskan mengapa awalan yang di-cache ~10× lebih murah untuk dibaca ulang daripada menghitung ulang.
- [Agrawal et al., "SARATHI: Inference LLM yang Efisien dengan Membonceng Dekode dengan Prefill yang Dipotong" (2023)](https://arxiv.org/abs/2308.16369) — prefill adalah pintasan cache prompt fase; tulisan ini menjelaskan mengapa TTFT turun drastis pada cache hit sementara TPOT tidak terpengaruh.
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — cache cepat berada di samping decoding spekulatif, Flash Attention, dan MQA/GQA sebagai pengungkit yang membengkokkan kurva biaya inference; baca ini untuk tiga lainnya.
