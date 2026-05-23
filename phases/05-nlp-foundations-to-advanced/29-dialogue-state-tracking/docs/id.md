# Pelacakan Status Dialog

> "Saya ingin restoran murah di utara... sebenarnya membuatnya moderat... dan menambahkan Italia." Tiga putaran, tiga pembaruan negara bagian. DST menjaga dict nilai slot tetap sinkron sehingga pemesanan berfungsi.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 5 · 17 (Chatbots), Fase 5 · 20 (Output Terstruktur)
**Waktu:** ~75 menit

## Masalah

Dalam sistem dialog berorientasi tugas, tujuan pengguna dikodekan sebagai sekumpulan pasangan nilai slot: `{cuisine: italian, area: north, price: moderate}`. Setiap giliran pengguna dapat menambah, mengubah, atau menghapus slot. Sistem harus membaca seluruh percakapan dan menampilkan status saat ini dengan benar.

Jika ada satu slot yang salah maka sistem akan memesan restoran yang salah, menjadwalkan penerbangan yang salah, atau membebankan biaya pada kartu yang salah. DST adalah penghubung antara apa yang dikatakan pengguna dan apa yang dijalankan backend.

Mengapa hal ini masih penting di tahun 2026 meskipun ada LLM:

- Domain yang sensitif terhadap kepatuhan (perbankan, layanan kesehatan, pemesanan maskapai penerbangan) memerlukan nilai slot deterministik, bukan pembuatan bentuk bebas.
- Agen penggunaan alat masih memerlukan resolusi slot sebelum memanggil API.
- Koreksi multi-putaran lebih sulit daripada yang terlihat: "sebenarnya tidak, lakukan pada hari Kamis."

Pipeline pipa modern: konsep DST klasik + ekstraktor LLM + pagar pembatas output terstruktur.

## Konsep

![DST: riwayat dialog → status nilai slot](../assets/dst.svg)

**Struktur tugas.** Skema mendefinisikan domain (restoran, hotel, taksi) dan slotnya (masakan, area, harga, orang). Setiap slot dapat kosong, diisi dengan nilai dari himpunan tertutup (harga: {murah, sedang, mahal}), atau nilai bentuk bebas (nama: "Ketel Tembaga").

**Dua formulasi DST.**

- **Klasifikasi.** Untuk setiap pasangan (slot, nilai_kandidat), prediksi ya/tidak. Berfungsi untuk slot kosakata tertutup. Standar sebelum tahun 2020.
- **Generation.** Berdasarkan dialog, buat nilai slot sebagai teks bebas. Berfungsi untuk slot kosakata terbuka. Standar modern.

**Metrik.** Joint Goal Accuracy (JGA) — pecahan belokan yang *setiap* slotnya benar. Semua atau tidak sama sekali. Papan peringkat MultiWOZ 2.4 menduduki puncak sekitar 83% pada tahun 2026.

**Arsitektur.**

1. **Berbasis aturan (slot regex + kata kunci).** Dasar yang kuat untuk domain sempit. Dapat di-debug.
2. **TripPy / BERT-DST.** Pembuatan berbasis salinan dengan pengkodean BERT. Standar pra-LLM.
3. **LDST (LLaMA + LoRA).** LLM yang disesuaikan dengan instruksi dengan prompt slot domain. Mencapai kualitas tingkat ChatGPT di MultiWOZ 2.4.
4. **Bebas Ontologi (2024–26).** Lewati skema; menghasilkan nama dan nilai slot secara langsung. Menangani domain terbuka.
5. **Output cepat + terstruktur (2024–26).** LLM dengan skema Pydantic + decoding terbatas. 5 baris code, siap produksi.

### Mode kegagalan klasik

- **Referensi bersama lintas belokan.** "Mari kita tetap pada opsi pertama." Perlu memutuskan opsi mana.
- **Tulis ulang vs tambahkan.** Pengguna mengatakan "tambahkan bahasa Italia". Apakah kamu mengganti masakan atau menambahkan?
- **Konfirmasi tersirat.** "OK keren" — apakah itu menerima pemesanan yang ditawarkan?
- **Koreksi.** "Sebenarnya, buatlah jam 7 malam." Harus memperbarui waktu tanpa mengosongkan slot lainnya.
- **Referensi ke ucapan sistem sebelumnya.** "Ya, yang itu." Yang "itu"?

## Build

### Langkah 1: ekstraktor slot berbasis aturan

Lihat `code/main.py`. Kamus regex + sinonim mencakup 70% ucapan kanonik dalam domain sempit:

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

Rapuh di luar kosakata kanonik. Berfungsi untuk konfirmasi slot deterministik.

### Langkah 2: putaran pembaruan status

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

Tiga invarian:- Jangan pernah menyetel ulang slot yang tidak disentuh pengguna.
- Negasi eksplisit ("apalagi masakannya") harus jelas.
- Koreksi pengguna ("sebenarnya...") harus menimpa, bukan menambahkan.

### Langkah 3: DST berbasis LLM dengan output terstruktur

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instruktur + Pydantic menjamin objek status yang valid. Tidak ada regex, tidak ada ketidakcocokan skema, tidak ada slot berhalusinasi.

### Langkah 4: Evaluasi JGA

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

Kalibrasi: berapa fraksi putaran yang dilakukan sistem untuk mendapatkan SEMUA slot dengan benar? Untuk MultiWOZ 2.4, sistem teratas tahun 2026: 80-83%. Sistem dalam domain kamu harus melebihi kosakata kamu yang sempit atau dasar LLM akan mengalahkan kamu.

### Langkah 5: menangani koreksi

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

Jika koreksi terdeteksi, timpa slot yang terakhir diperbarui, bukan menambahkan. Sulit untuk menjadi benar tanpa bantuan LLM. Pola modern: selalu biarkan LLM meregenerasi seluruh negara bagian dari sejarah daripada memperbarui secara bertahap — hal ini secara alami menangani koreksi.

## Jebakan

- **Biaya regenerasi riwayat penuh.** Membiarkan status regenerasi LLM setiap giliran memerlukan total token sebesar O(n²). Batasi sejarah atau rangkum putaran lama.
- **Penyimpangan skema.** Menambahkan slot baru post-hoc akan merusak training data lama. Versi skema kamu.
- **Sensitivitas huruf besar-kecil.** "Italia" vs "Italia" vs "ITALIAN" — dinormalisasi di semua tempat.
- **Warisan implisit.** Jika pengguna sebelumnya telah menentukan "untuk 4 orang", permintaan baru untuk waktu yang berbeda tidak akan menghapus orang. Selalu sampaikan riwayat lengkapnya.
- **Bentuk bebas vs himpunan tertutup.** Nama, waktu, dan alamat memerlukan slot bentuk bebas; dapur dan area ditutup. Campurkan keduanya dalam skema.

## Pakai

Tumpukan tahun 2026:

| Situasi | Pendekatan |
|-----------|----------|
| Domain sempit (satu atau dua maksud) | Berbasis aturan + regex |
| Domain luas, data berlabel tersedia | LDST (LLaMA + LoRA pada data gaya MultiWOZ) |
| Domain luas, tanpa label, siap produksi | LLM + Instruktur + Skema Pydantic |
| Diucapkan / suara | ASR + normalisasi + LLM-DST |
| Alur pemesanan multi-domain | LLM yang dipandu skema dengan model Pydantic per domain |
| Peka terhadap kepatuhan | Primer berbasis aturan, fallback LLM dengan aliran konfirmasi |

## Kirim

Simpan sebagai `outputs/skill-dst-designer.md`:

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## Latihan

1. **Mudah.** Buat pelacak negara bagian berbasis aturan di `code/main.py` untuk 3 slot (masakan, area, harga). Uji 10 dialog buatan tangan. Ukur JGA.
2. **Sedang.** Dataset yang sama dengan Instruktur + Pydantic + LLM kecil. Bandingkan JGA. Periksa belokan yang paling sulit.
3. **Sulit.** Menerapkan keduanya dan merutekan: primer berbasis aturan, penggantian LLM ketika berbasis aturan mengeluarkan <2 slot dengan yakin. Ukur gabungan JGA dan biaya inference per giliran.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| DST | Pelacakan status dialog | Pertahankan dikte nilai slot di seluruh putaran dialog. |
| celah | Unit niat pengguna | Parameter bernama kebutuhan backend (masakan, tanggal). |
| Domain | Area tugas | Restoran, hotel, taksi — set slot. |
| JGA | Akurasi Sasaran Bersama | Bagian belokan yang setiap slotnya benar. Semua atau tidak sama sekali. |
| MultiWOZ | Tolok ukur | Dataset WOZ multi-domain; evaluasi DST standar. |
| DST bebas ontologi | Tidak ada skema | Hasilkan nama dan nilai slot secara langsung, tanpa daftar tetap. |
| Koreksi | "Sebenarnya..." | Putaran itu akan menimpa slot yang telah diisi sebelumnya. |

## Bacaan Lanjutan- [Budzianowski dkk. (2018). MultiWOZ — Wizard-of-Oz Multi-Domain Berskala Besar](https://arxiv.org/abs/1810.00278) — tolok ukur kanonik.
- [Feng dkk. (2023). Menuju Pelacakan Status Dialog (LDST) yang digerakkan oleh LLM](https://arxiv.org/abs/2310.14970) — Penyetelan instruksi LLaMA + LoRA untuk DST.
- [Heck dkk. (2020). TripPy — Strategi Tiga Salinan untuk Pelacakan Status Dialog Neural Independen Nilai](https://arxiv.org/abs/2005.02877) — pekerja keras DST berbasis salinan.
- [Raja, Flanigan (2024). Dialog Berorientasi Tugas End-to-End Tanpa Pengawasan dengan LLM](https://arxiv.org/abs/2404.10753) — TOD tanpa pengawasan berbasis EM.
- [Papan peringkat MultiWOZ](https://github.com/budzianowski/multiwoz) — hasil DST kanonik.
