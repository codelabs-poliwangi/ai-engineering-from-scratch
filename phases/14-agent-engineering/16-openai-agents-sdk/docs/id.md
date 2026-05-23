# OpenAI Agents SDK: Handoff, Pagar Pembatas, Penelusuran

> OpenAI Agents SDK adalah framework multi-agen ringan yang dibangun di atas Responses API. Lima primitif: Agen, Handoff, Pagar Pembatas, Sesi, Penelusuran. Handoff adalah alat bernama `transfer_to_<agent>`. Pagar pembatas tersandung pada input atau output. Pelacakan diaktifkan secara default.

**Type:** Learn + Build
**Language:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Penggunaan Alat)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Sebutkan lima primitif OpenAI Agents SDK.
- Jelaskan handoff: mengapa handoff dimodelkan sebagai alat, nama apa yang dilihat model, dan bagaimana konteksnya ditransfer.
- Membedakan pagar pembatas input, pagar pembatas output, dan pagar pembatas perkakas; jelaskan `run_in_parallel` vs mode pemblokiran.
- Menerapkan runtime stdlib dengan handoff + pagar pembatas + penelusuran gaya rentang.

## Masalah

Agen yang tidak dapat mendelegasikan dengan rapi akan memasukkan semuanya ke dalam satu prompt. Agen tanpa pagar pembatas mengirimkan PII, output yang melanggar kebijakan, atau loop selamanya. SDK OpenAI mengkodifikasikan tiga primitif yang membuat pekerjaan multi-agen mudah dilakukan.

## Konsep

### Lima primitif

1. **Agen.** LLM + instruksi + alat + serah terima.
2. **Handoff.** Delegasi ke agen lain. Direpresentasikan ke model sebagai alat bernama `transfer_to_<agent_name>`.
3. **Pagar Pembatas.** Validasi pada input (hanya agen pertama), output (hanya agen terakhir), atau pemanggilan alat (per alat fungsi).
4. **Sesi.** Riwayat percakapan otomatis lintas giliran.
5. **Pelacakan.** Rentang bawaan untuk generasi LLM, pemanggilan alat, serah terima, pagar pembatas.

### Handoff sebagai alat

Model melihat `transfer_to_billing_agent` dalam daftar alatnya. Menyebutnya memberi sinyal pada runtime untuk:

1. Salin konteks percakapan (atau ciutkan melalui `nest_handoff_history` beta).
2. Inisialisasi agen target dengan instruksinya.
3. Lanjutkan proses dengan agen target.

Inilah pola supervisor (Lesson 13 / Lesson 28) yang dihasilkan.

### Pagar Pembatas

Tiga rasa:

- **Pagar pembatas input.** Jalankan pada input agen pertama. Tolak permintaan yang tidak aman atau di luar cakupan sebelum panggilan LLM apa pun.
- **Pagar pembatas output.** Jalankan pada output agen terakhir. Tangkap kebocoran PII, pelanggaran kebijakan, respons yang salah.
- **Pagar pembatas alat.** Jalankan alat per fungsi. Validasi argumen, periksa izin, audit eksekusi.

Modus:

- **Paralel** (default). LLM Pagar Pembatas berjalan di samping LLM utama. Latensi ekor lebih rendah. Jika tersandung, pekerjaan utama LLM akan dibuang (token waste).
- **Pemblokiran** (`run_in_parallel=False`). Pagar Pembatas LLM berjalan lebih dulu. Jika tersandung, tidak ada token yang terbuang pada panggilan utama.

Tripwire naikkan `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`.

### Menelusuri

Aktif secara default. Setiap generasi LLM, pemanggilan alat, handoff, dan pagar pembatas mengeluarkan rentang. `OPENAI_AGENTS_DISABLE_TRACING=1` memilih untuk tidak ikut serta. `add_trace_processor(processor)` penggemar mencakup backend kamu sendiri bersama dengan OpenAI.

### Sesi

`Session` menyimpan riwayat percakapan di backend (SQLite, Redis, custom). `Runner.run(agent, input, session=session)` memuat dan menambahkan secara otomatis.

### Dimana letak kesalahan pola ini

- **Handoff drift.** Agen A menyerahkan ke Agen B yang kemudian menyerahkan kembali ke Agen A. Tambahkan penghitung hop.
- **Jalan pintas pagar pembatas.** Pagar pembatas alat hanya menyala pada alat yang berfungsi; alat bawaan (pembaca file, pengambilan web) memerlukan kebijakan terpisah.
- **Pelacakan berlebihan.** Konten sensitif dalam rentang tertentu. Pasangkan dengan aturan pengambilan konten OTel GenAI (Lesson 23) — simpan secara eksternal, referensi berdasarkan ID.

## Build

`code/main.py` mengimplementasikan bentuk SDK di stdlib:- `Agent`, `FunctionTool`, `Handoff` (sebagai alat fungsi dengan semantik transfer).
- `Runner` dengan pagar pembatas input/output/alat, pengiriman handoff, dan penghitung hop.
- Pemancar rentang sederhana untuk menunjukkan bentuk jejak.
- Agen triase yang menyerahkan penagihan atau dukungan berdasarkan permintaan pengguna; perjalanan pagar pembatas pada satu input.

Jalankan:

```
python3 code/main.py
```

Pelacakan menunjukkan dua handoff yang berhasil, satu trip pagar pembatas input, dan pohon rentang yang mencerminkan apa yang dikeluarkan oleh SDK sebenarnya.

## Pakai

- **OpenAI Agents SDK** untuk produk yang mengutamakan OpenAI.
- **Claude Agent SDK** (Lesson 17) untuk produk Claude-first.
- **LangGraph** (Lesson 13) bila kamu menginginkan status eksplisit dan resume yang tahan lama.
- **Kustom** saat kamu memerlukan kontrol yang tepat (suara, multi-penyedia, penerapan gabungan).

## Kirim

`outputs/skill-agents-sdk-scaffold.md` merancah aplikasi Agen SDK dengan agen triase, handoff, pagar pembatas input/output/alat, penyimpanan sesi, dan pemroses penelusuran.

## Latihan

1. Tambahkan penghitung hop handoff: tolak setelah N transfer. Lacak perilakunya.
2. Terapkan `nest_handoff_history` sebagai opsi — ciutkan pesan sebelumnya menjadi satu ringkasan sebelum ditransfer.
3. Tulis pagar pembatas output pemblokiran. Bandingkan latensi pada prompt yang akan membuat tersandung vs yang lolos.
4. Hubungkan `add_trace_processor` ke logger JSON. Bentuk apa yang dipancarkannya per rentang?
5. Baca dokumen SDK. Pindahkan mainan stdlib kamu ke `openai-agents-python`. Apa kesalahan model kamu?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Agen | "LLM + instruksi" | Agen mengetikkan SDK; memiliki peralatan dan serah terima |
| Penyerahan | "Transfer" | Alat yang dipanggil model untuk didelegasikan ke agen lain |
| Pagar Pembatas | "Pemeriksaan kebijakan" | Validasi pada pemanggilan input/output/alat |
| Kawat Tripel | "Perjalanan pagar pembatas" | Pengecualian muncul ketika pagar pembatas menolak |
| Sesi | "Toko Sejarah" | Memori percakapan tetap ada di antara proses |
| Menelusuri | "Rentang" | Observabilitas bawaan melalui LLM + alat + handoff + pagar pembatas |
| Memblokir pagar pembatas | "Pemeriksaan berurutan" | Pagar pembatas berjalan lebih dulu; tidak ada token yang terbuang dalam perjalanan |
| Pagar Pembatas Paralel | "Pemeriksaan serentak" | Pagar pembatas berjalan di sampingnya; latensi lebih rendah, membuang token saat perjalanan |

## Bacaan Lanjutan

- [Dokumen OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — primitif, handoff, pagar pembatas, penelusuran
- [Ikhtisar SDK Agen Claude](https://platform.claude.com/docs/en/agent-sdk/overview) — Mitra rasa Claude
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building- Effective-agents) — kapan harus melakukan serah terima
- [Konvensi semantik OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Agen SDK standar mencakup peta hingga
