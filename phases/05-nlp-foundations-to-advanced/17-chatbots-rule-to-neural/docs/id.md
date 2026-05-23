# Chatbots — Berbasis Aturan hingga Agen Neural hingga LLM

> ELIZA membalas dengan kecocokan pola. DialogFlow memetakan maksud. GPT menjawab dari weight. Claude menjalankan alat dan memverifikasi. Setiap era memecahkan kegagalan terburuk sebelumnya.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 5 · 13 (Menjawab Pertanyaan), Phase 5 · 14 (Pengambilan Informasi)
**Waktu:** ~75 menit

## Masalah

Seorang pengguna mengatakan "Saya ingin mengubah penerbangan saya." Sistem harus mengetahui apa yang mereka inginkan, informasi apa yang hilang, bagaimana mendapatkannya, dan bagaimana menyelesaikan tindakan. Kemudian pengguna berkata "tunggu, bagaimana jika saya membatalkannya?" dan sistem harus mengingat konteksnya, beralih tugas, dan mempertahankan status.

Percakapan sulit dilakukan pada sistem ML. Masukannya bersifat terbuka. Outputnya harus koheren dalam banyak putaran. Sistem mungkin perlu bertindak terhadap dunia (mengubah penerbangan, menagih kartu). Setiap langkah yang salah terlihat oleh pengguna.

Arsitektur Chatbot telah melalui empat paradigma, masing-masing diperkenalkan karena paradigma sebelumnya terlalu terlihat gagal. Lesson ini menuntun mereka secara berurutan. Lanskap produksi tahun 2026 merupakan gabungan dari dua lanskap produksi terakhir.

## Konsep

![Evolusi Chatbot: berbasis aturan → pengambilan → saraf → agen](../assets/chatbot.svg)

**Berbasis aturan (ELIZA, AIML, DialogFlow).** Pola buatan tangan mencocokkan input pengguna dan menghasilkan respons. Pengklasifikasi maksud merutekan ke alur yang telah ditentukan sebelumnya. Mesin negara yang mengisi slot mengumpulkan informasi yang diperlukan. Bekerja dengan cemerlang dalam lingkup sempit yang dirancang untuknya. Gagal segera di luarnya. Masih dikirimkan dalam domain yang sangat penting bagi keselamatan (otentikasi perbankan, pemesanan maskapai penerbangan) di mana halusinasi tidak dapat ditoleransi.

**Berbasis pengambilan.** Sistem bergaya FAQ. Menyandikan setiap pasangan (ucapan, tanggapan). Saat runtime, enkode pesan pengguna dan ambil respons tersimpan terdekat. Pikirkan feature "artikel serupa" klasik Zendesk. Menangani parafrase lebih baik daripada aturan. Tidak ada generasi, jadi tidak ada halusinasi.

**Neural (seq2seq).** Encoder-decoder dilatih tentang log percakapan. Menghasilkan respons dari awal. Lancar tetapi rentan terhadap output umum ("Saya tidak tahu") dan penyimpangan faktual. Tidak pernah bisa diandalkan pada topik. Alasan Google, Facebook, dan Microsoft memiliki chatbot yang mengecewakan pada 2016-2019.

**Agen LLM.** Model bahasa yang dikemas dalam loop yang merencanakan, memanggil alat, dan memverifikasi hasil. Bukan chatbot dengan prompt yang panjang. Lingkaran agen: rencana → alat panggilan → amati hasil → putuskan langkah selanjutnya. Retrieval-first grounding (RAG) mencegahnya berhalusinasi. Panggilan alat memungkinkannya melakukan sesuatu. Ini adalah arsitektur tahun 2026.

Keempat paradigma tersebut bukanlah pengganti yang berurutan. Chatbot produksi tahun 2026 merutekan keempatnya: berbasis aturan untuk autentikasi dan tindakan destruktif, pengambilan FAQ, pembuatan saraf untuk frasa alami, agen LLM untuk kueri terbuka yang ambigu.

## Build

### Langkah 1: pencocokan pola berbasis aturan

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

ELIZA dalam 20 baris. Trik refleksi (“Saya merasa sedih” → “Mengapa kamu merasa sedih”) adalah demo psikoterapis kanonik dari Weizenbaum 1966. Masih instruktif.

### Langkah 2: berbasis pengambilan (FAQ)

Cuplikan ilustratif ini memerlukan `pip install sentence-transformers` (yang menarik obor). `code/main.py` yang dapat dijalankan untuk lesson ini menggunakan kesamaan stdlib Jaccard, sehingga lesson berjalan tanpa ketergantungan eksternal.

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

Penolakan berbasis ambang batas adalah pilihan desain utama. Jika kecocokan terbaik tidak cukup, kembalikan `None` dan biarkan sistem meningkat.### Langkah 3: pembuatan saraf (dasar)

Gunakan encoder-decoder kecil yang disesuaikan dengan instruksi (FLAN-T5) atau model percakapan yang disesuaikan. Produksi tidak dapat digunakan sendiri pada tahun 2026 (kontradiksi, penyimpangan di luar topik, omong kosong faktual), tetapi dikirimkan dalam sistem hibrida untuk ungkapan alami. Model khusus dekoder gaya DialoGPT memerlukan pemisah belokan eksplisit dan penanganan EOS untuk menghasilkan balasan yang koheren; pipeline text2text FLAN-T5 berfungsi dengan baik untuk contoh pengajaran.

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### Langkah 4: Lingkaran agen LLM

Bentuk produksi tahun 2026:

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

Tiga hal untuk disebutkan. Alat adalah fungsi yang dapat dipanggil yang dapat dipanggil oleh LLM. Perulangan berakhir ketika LLM mengembalikan jawaban akhir dan bukan pemanggilan alat. Anggaran langkah mencegah putaran tak terbatas pada tugas-tugas yang ambigu.

Produksi nyata menambahkan: landasan pengambilan terlebih dahulu (memasukkan dokumen yang relevan sebelum setiap panggilan LLM), pagar pembatas (menolak tindakan destruktif tanpa konfirmasi), kemampuan observasi (mencatat setiap langkah), dan evaluasi (pemeriksaan otomatis bahwa perilaku agen tetap sesuai spesifikasi).

### Langkah 5: perutean hibrid

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

Polanya: aturan deterministik untuk segala hal yang merusak, pengambilan FAQ terekam, agen LLM untuk hal lainnya. Inilah yang dikirimkan pada sistem dukungan pelanggan pada tahun 2026.

## Pakai

Tumpukan tahun 2026:

| Kasus penggunaan | Arsitektur |
|---------|---------------|
| Pemesanan, pembayaran, otentikasi | Mesin negara berbasis aturan + pengisian slot |
| FAQ dukungan pelanggan | Pengambilan jawaban yang dikurasi |
| Obrolan bantuan terbuka | Agen LLM dengan panggilan alat RAG + |
| Alat internal / asisten IDE | Agen LLM dengan alat panggilan (cari, baca, tulis) |
| Chatbot pendamping/karakter | Menyetel LLM dengan prompt sistem persona, pengambilan pengetahuan |

Selalu gunakan perutean hibrid dalam produksi. Tidak ada satu arsitektur pun yang dapat menangani setiap permintaan dengan baik. Layer perutean itu sendiri biasanya merupakan pengklasifikasi maksud kecil.

## Mode kegagalan yang masih dikirimkan

- **Fabrikasi yang percaya diri.** Agen LLM mengklaim telah menyelesaikan tindakan yang tidak dilakukannya. Mitigasi: verifikasi hasil, catat panggilan alat, jangan biarkan LLM mengklaim telah melakukan sesuatu tanpa pengembalian alat yang berhasil.
- **Injeksi cepat.** Pengguna menyisipkan teks yang menggantikan system prompt. Peringkat LLM01 dalam 10 Teratas OWASP untuk Aplikasi LLM 2025. Dua jenis: injeksi langsung (ditempelkan ke dalam obrolan) dan injeksi tidak langsung (tersembunyi dalam dokumen, email, atau output alat yang dibaca agen).

  Tingkat serangan bervariasi berdasarkan skenario. Tingkat keberhasilan yang diukur berkisar ~0,5-8,5% di seluruh model frontier dalam tolok ukur penggunaan alat dan pengkodean secara umum. Pengaturan spesifik yang berisiko tinggi (serangan adaptif terhadap agen pengkodean AI, orkestrasi yang rentan) telah mencapai ~84%. CVE produksi mencakup EchoLeak (CVE-2025-32711, CVSS 9.3) — kelemahan eksfiltrasi data zero-click di Microsoft 365 Copilot yang dipicu oleh email yang dikendalikan penyerang.

  Mitigasi: perlakukan input pengguna sebagai tidak tepercaya sepanjang loop; melakukan sanitasi sebelum alat dipanggil; mengisolasi output alat dari prompt utama; menggunakan pola Plan-Verify-Execute (PVE) di mana agen merencanakan terlebih dahulu, kemudian memverifikasi setiap tindakan terhadap rencana tersebut sebelum mengeksekusi (ini menghentikan hasil alat dari memasukkan tindakan baru yang tidak direncanakan); memerlukan konfirmasi pengguna untuk tindakan destruktif; menerapkan hak istimewa paling rendah pada cakupan alat.Tidak ada rekayasa cepat yang bisa sepenuhnya menghilangkan risiko ini. Layer pertahanan waktu proses eksternal (LLM Guard, validasi daftar yang diizinkan, deteksi anomali semantik) diperlukan.
- **Scope creep.** Agen keluar dari tugasnya karena panggilan alat mengembalikan informasi terkait secara tangensial. Mitigasi: kontrak alat yang sempit; menjaga agar sistem tetap fokus; tambahkan evaluasi untuk tingkat di luar tugas.
- **Perulangan tak terbatas.** Agen terus memanggil alat yang sama. Mitigasi: langkah anggaran, deduplikasi panggilan alat, penilaian LLM tentang "apakah kita membuat kemajuan."
- **Kelelahan jendela konteks.** Percakapan panjang membuat percakapan paling awal keluar dari konteks. Mitigasi: merangkum kejadian-kejadian lama, mengambil kejadian-kejadian yang relevan di masa lalu berdasarkan kesamaan, atau menggunakan model konteks panjang.

## Kirim

Simpan sebagai `outputs/skill-chatbot-architect.md`:

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## Latihan

1. **Mudah.** Terapkan respons berbasis aturan di atas dengan 10 pola untuk bot pemesanan kedai kopi. Kasus uji tepi: pesanan ganda, modifikasi, pembatalan, niat tidak jelas.
2. **Sedang.** Buat FAQ hybrid + penggantian LLM. 50 entri FAQ terekam untuk produk SaaS, penggantian LLM dengan pengambilan melalui situs dokumen. Ukur tingkat penolakan dan akurasi pada 100 pertanyaan dukungan nyata.
3. **Sulit.** Implementasikan loop agen di atas dengan tiga alat (penelusuran, baca data pengguna, kirim email). Jalankan evaluasi dengan 50 skenario pengujian termasuk upaya injeksi cepat. Laporkan tingkat tugas di luar tugas, tingkat tugas yang gagal, dan keberhasilan penyuntikan.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Maksud | Apa yang diinginkan pengguna | Label kategorikal (book_flight, reset_password). Diarahkan ke pawang. |
| celah | Sepotong info | Parameter yang dibutuhkan bot (tanggal, tujuan). Pengisian slot adalah urutan pertanyaan. |
| RAG | Pengambilan plus generasi | Ambil dokumen yang relevan, lalu dasarkan respons LLM. |
| Panggilan alat | Pemanggilan fungsi | LLM mengeluarkan panggilan terstruktur dengan nama + argumen. Runtime dijalankan, mengembalikan hasil. |
| Lingkaran agen | Rencanakan, bertindak, verifikasi | Pengontrol yang menjalankan panggilan LLM disisipkan dengan panggilan alat hingga tugas selesai. |
| Injeksi segera | Permintaan serangan pengguna | Input berbahaya yang mencoba mengesampingkan system prompt. |

## Bacaan Lanjutan- [Weizenbaum (1966). ELIZA — Program Komputer Untuk Studi Komunikasi Bahasa Alami](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — makalah chatbot berbasis aturan asli.
- [Thoppilan dkk. (2022). LaMDA: Model Bahasa untuk Aplikasi Dialog](https://arxiv.org/abs/2201.08239) — Makalah neural-chatbot Google yang terakhir, tepat sebelum agen LLM mengambil alih.
- [Yao dkk. (2022). ReAct: Mensinergikan Penalaran dan Tindakan dalam Model Bahasa](https://arxiv.org/abs/2210.03629) — makalah yang memberi nama pola loop agen.
- [Panduan Anthropic dalam membangun agen yang efektif](https://www.anthropic.com/research/building- Effective-agents) — Panduan produksi tahun 2024 yang masih berlaku pada tahun 2026.
- [Greshake dkk. (2023). Bukan tujuan kamu mendaftar: Mengkompromikan Aplikasi Terintegrasi LLM Dunia Nyata dengan Injeksi Cepat Tidak Langsung](https://arxiv.org/abs/2302.12173) — makalah injeksi cepat.
- [10 Teratas OWASP untuk Aplikasi LLM 2025 — Injeksi Cepat LLM01](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — peringkat yang menjadikan injeksi cepat sebagai masalah keamanan teratas.
- [AWS — Mengamankan Agen Amazon Bedrock terhadap Suntikan Prompt Tidak Langsung](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — pertahanan layer orkestrasi praktis termasuk alur Plan-Verify-Execute dan konfirmasi pengguna.
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — CVE eksfiltrasi data zero-click kanonik dari injeksi cepat tidak langsung. Kasus referensi mengapa agen akses tulis memerlukan pertahanan waktu proses.
