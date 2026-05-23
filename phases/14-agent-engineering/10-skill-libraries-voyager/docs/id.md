# Perpustakaan Keterampilan dan Pembelajaran Seumur Hidup (Voyager)

> Voyager (Wang et al., TMLR 2024) memperlakukan code yang dapat dieksekusi sebagai keterampilan. Keterampilan diberi nama, dapat diambil, disusun, dan disempurnakan berdasarkan umpan balik lingkungan. Ini adalah arsitektur referensi untuk skill, skillkit, dan pola perpustakaan skill Claude Agent SDK 2026.

**Type:** Build
**Language:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Blok Letta)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Sebutkan tiga komponen Voyager — kurikulum otomatis, pustaka keterampilan, prompt berulang — dan peran masing-masing komponen.
- Jelaskan mengapa Voyager membuat code ruang tindakan, bukan prompt primitif.
- Menerapkan perpustakaan keterampilan stdlib dengan registrasi, pengambilan, komposisi, dan penyempurnaan yang didorong oleh kegagalan.
- Memetakan pola Voyager ke keterampilan SDK Agen Claude 2026 dan ekosistem kit keterampilan.

## Masalah

Agen yang membangun kembali setiap kemampuan dari awal di setiap sesi melakukan tiga kesalahan:

1. **Token limbah.** Setiap tugas memunculkan kembali alasan yang sama.
2. **Kehilangan kemajuan.** Koreksi yang dipelajari di sesi A tidak ditransfer ke sesi B.
3. **Gagal dalam komposisi jangka panjang.** Tugas kompleks memerlukan hierarki kemampuan; prompt sekali pakai tidak dapat mengungkapkannya.

Jawaban Voyager: perlakukan setiap kemampuan yang dapat digunakan kembali sebagai potongan code bernama yang disimpan di perpustakaan, dapat diambil berdasarkan kesamaan, dapat disusun dengan keterampilan lain, dan disempurnakan dengan umpan balik eksekusi.

## Konsep

### Tiga komponen

Voyager (arXiv:2305.16291) menyusun agen di sekitar:

1. **Kurikulum otomatis.** Pengusul yang didorong oleh rasa ingin tahu memilih tugas berikutnya berdasarkan keahlian dan kondisi lingkungan agen saat ini. Eksplorasi bersifat bottom-up.
2. **Perpustakaan keterampilan.** Setiap keterampilan adalah code yang dapat dieksekusi. Keterampilan baru ditambahkan ketika tugas berhasil. Keterampilan diambil berdasarkan kesamaan kueri-ke-deskripsi.
3. **Mekanisme permintaan berulang.** Jika gagal, agen menerima kesalahan eksekusi, input lingkungan, dan output verifikasi mandiri, lalu menyempurnakan keterampilan.

Evaluasi Minecraft (Wang dkk., 2024): item unik 3,3x lebih banyak, perkakas batu 8,5x lebih cepat, perkakas besi 6,4x lebih cepat, penjelajahan peta 2,3x lebih lama dibandingkan garis dasar. Angka-angkanya khusus untuk Minecraft, tetapi polanya dapat berubah.

### Ruang tindakan = code

Kebanyakan agen mengeluarkan prompt primitif. Voyager mengeluarkan fungsi JavaScript. Sebuah keterampilan adalah:

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

Terdiri dari sub-keterampilan. Disimpan dengan kunci pada deskripsi dan embedding. Diambil sebagai program, bukan prompt.

Ini adalah keterampilan Claude Agent SDK 2026: potongan code yang diberi nama dan dapat diambil serta instruksi yang dimuat agen sesuai permintaan.

### Pengambilan keterampilan

Tugas baru "membuat beliung berlian". Agen:

1. Sematkan deskripsi tugas.
2. Menanyakan perpustakaan keterampilan untuk k keterampilan serupa teratas.
3. Mengambil `craftIronPickaxe`, `mineDiamond`, `placeCraftingTable` dll.
4. Menyusun keterampilan baru dari primitif yang diambil + logika baru.

Ini adalah pola yang diterapkan sumber daya MCP (Fase 13) dan keterampilan Agen SDK: pengambilan melalui permukaan pengetahuan/code, yang tercakup dalam tugas saat ini.

### Penyempurnaan berulang

Putaran umpan balik Voyager:

1. Agen menulis keterampilan.
2. Keterampilan berlari melawan lingkungan.
3. Salah satu dari tiga sinyal kembali: `success`, `error` (dengan pelacakan tumpukan), `self-verification failure`.
4. Agen menulis ulang skill menggunakan sinyal sebagai konteks.
5. Ulangi hingga sukses atau putaran maksimal.Ini adalah Penyempurnaan Mandiri (Lesson 05) yang diterapkan pada pembuatan code dengan verifikasi berbasis lingkungan. CRITIC (Lesson 05) adalah pola yang sama dengan alat eksternal sebagai verifikator.

### Kurikulum dan eksplorasi

Modul kurikulum Voyager mengusulkan tugas seperti "membangun tempat berlindung di dekat danau" berdasarkan apa yang dimiliki agen dan apa yang belum dilakukan. Pengusul menggunakan kondisi lingkungan + inventaris keterampilan untuk memilih tugas yang berada tepat di atas kemampuan saat ini — sweet spot eksplorasi.

Bagi agen produksi, hal ini berarti operator "apa yang hilang": mengingat perpustakaan keterampilan dan domain saat ini, keterampilan apa yang belum kita kuasai? Tim biasanya menerapkan ini secara manual sebagai tinjauan kurikulum.

### Dimana letak kesalahan pola ini

- **Skill Library Rot.** Skill yang sama ditambahkan 10 kali dengan deskripsi yang sedikit berbeda. Tambahkan deduplikasi saat menulis; pengambilan hanya mengembalikan satu.
- **Penyimpangan keterampilan komposisi.** Keterampilan orang tua bergantung pada anak yang disempurnakan. Keterampilan versi; orang tua yang di-embed ke v1 tidak secara ajaib mengambil v3.
- **Kualitas pengambilan.** Pengambilan vector atas deskripsi keterampilan menurun seiring bertambahnya jumlah perpustakaan. Lengkapi dengan filter tag dan batasan keras ("hanya keterampilan dengan `category=tooling`").

## Build

`code/main.py` mengimplementasikan perpustakaan keterampilan stdlib:

- `Skill` — nama, deskripsi, code (sebagai string), versi, tag, dependensi.
- `SkillLibrary` — mendaftar, mencari (token tumpang tindih), menulis (semacam deps topologi), dan menyaring (versi bertambah saat pembaruan).
- Agen bernaskah yang mendaftarkan tiga keterampilan primitif, menyusun keterampilan keempat, mengalami kegagalan, dan menyempurnakan.

Jalankan:

```
python3 code/main.py
```

Jejak menunjukkan penulisan perpustakaan, pengambilan, komposisi, eksekusi yang gagal, dan penyempurnaan v2 — loop Voyager ujung ke ujung.

## Pakai

- **Keterampilan Claude Agent SDK** (Antropik) — referensi tahun 2026: setiap keterampilan memiliki deskripsi, code, dan instruksi; dimuat sesuai permintaan selama sesi agen.
- **skillkit** (npm: skillkit) — manajemen keterampilan lintas agen untuk 32+ agen pengkodean AI.
- **Perpustakaan keterampilan khusus** — khusus domain (keterampilan SQL untuk agen data, keterampilan Terraform untuk agen infra). Pola Voyager diperkecil.
- **OpenAI Agents SDK `tools`** — di kelas bawah; setiap alat adalah keterampilan ringan.

## Kirim

`outputs/skill-skill-library.md` menghasilkan perpustakaan keterampilan berbentuk Voyager dengan registrasi, pengambilan, pembuatan versi, dan penyempurnaan yang terhubung untuk setiap target runtime.

## Latihan

1. Tambahkan detektor siklus ketergantungan ke `compose()`. Apa jadinya jika skill A bergantung pada B yang bergantung pada A? Kesalahan vs peringatan?
2. Menerapkan embedding versi per keterampilan. Ketika keterampilan orang tua menyusun anak `crafting@1`, penyempurnaan ke `crafting@2` tidak boleh memutakhirkan induk secara diam-diam.
3. Ganti pengambilan token yang tumpang tindih dengan embedding pengubah kalimat (atau impl stdlib BM25). Ukur pengambilan@5 di perpustakaan mainan dengan 50 keterampilan.
4. Tambahkan agen "kurikulum": dengan mempertimbangkan perpustakaan saat ini dan deskripsi domain, usulkan 5 keterampilan yang hilang. Sebut saja mingguan.
5. Baca dokumen keterampilan Claude Agent SDK Anthropic. Port perpustakaan mainan ke skema keterampilan SDK. Apa saja perubahan terkait kemampuan untuk ditemukan?

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Keterampilan | "Kemampuan yang dapat digunakan kembali" | Dinamakan potongan code + deskripsi, dapat diambil berdasarkan kesamaan |
| Perpustakaan Keterampilan | "Ingatan agen tentang caranya" | Penyimpanan keterampilan yang gigih, dapat dicari dan disusun |
| Kurikulum | "Pengusul tugas" | Generator tujuan dari bawah ke atas didorong oleh kesenjangan kemampuan saat ini |
| Komposisi | "Keterampilan DAG" | Keterampilan memanggil keterampilan; diurutkan secara topologi pada eksekusi |
| Penyempurnaan berulang | "Lingkaran koreksi diri" | Umpan balik Env + kesalahan + verifikasi mandiri dilipat kembali ke versi berikutnya |
| Ruang tindakan sebagai code | "Tindakan terprogram" | Memancarkan fungsi, bukan prompt primitif, untuk perilaku yang diperpanjang sementara |
| Dedup saat menulis | "Keterampilan runtuh" ​​| Deskripsi yang hampir duplikat diciutkan menjadi satu keterampilan kanonik |

## Bacaan Lanjutan

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — makalah perpustakaan keterampilan asli
- [Ikhtisar Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — keterampilan sebagai produksi tahun 2026
- [Antropik, Agen bangunan dengan SDK Agen Claude](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — keterampilan dan subagen dalam praktik
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — loop penyempurnaan di bawah Voyager
