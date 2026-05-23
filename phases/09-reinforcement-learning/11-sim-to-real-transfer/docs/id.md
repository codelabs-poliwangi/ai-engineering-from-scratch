# Transfer Sim-ke-Nyata

> Kebijakan yang dilatih pada simulator yang gagal pada perangkat keras adalah kebijakan yang mengingat simulator tersebut. Pengacakan domain, adaptasi domain, dan identifikasi sistem adalah tiga alat untuk membuat pengontrol terpelajar melewati kesenjangan realitas.

**Type:** Learn
**Language:** Python
**Prerequisites:** Fase 9 · 08 (PPO), Fase 2 · 10 (Bias/Varians)
**Waktu:** ~45 menit

## Masalah

Melatih robot sungguhan itu lambat, berbahaya, dan mahal. Seorang biped membutuhkan jutaan episode training untuk belajar berjalan; makhluk berkaki dua sungguhan yang terjatuh bahkan setelah merusak perangkat keras. Simulasi memberi kamu pengaturan ulang tanpa batas, reproduktifitas deterministik, lingkungan paralel, dan tidak ada kerusakan fisik.

Tapi simulator salah. Bantalan memiliki lebih banyak gesekan dibandingkan model MuJoCo. Kamera memiliki distorsi lensa yang tidak termasuk dalam simulator. Motor memiliki penundaan, reaksi balik, dan saturasi yang dilewati oleh 99% model sim. Angin, debu, dan pencahayaan yang bervariasi menyabot kebijakan yang dilatih tentang rendering yang steril. **Kesenjangan realitas** — perbedaan sistematis antara distribusi sim dan distribusi nyata — adalah masalah utama penerapan RL untuk robotika.

kamu memerlukan kebijakan yang *kuat untuk peralihan distribusi sim-to-real*. Tiga pendekatan historis: mengacak simulator (pengacakan domain), mengadaptasi kebijakan dengan sedikit data nyata (adaptasi/penyempurnaan domain), atau mengidentifikasi parameter sistem nyata dan mencocokkannya (identifikasi sistem). Pada tahun 2026 resep dominan menggabungkan ketiganya dengan simulasi paralel masif (Isaac Sim, Isaac Lab, Mujoco MJX pada GPU).

## Konsep

![Tiga rezim sim-to-real: pengacakan domain, adaptasi, identifikasi sistem](../assets/sim-to-real.svg)

**Pengacakan Domain (DR).** Tobin dkk. 2017, Peng dkk. 2018. Selama training, acak setiap parameter sim yang mungkin berbeda pada robot sebenarnya: massa, koefisien gesekan, penguatan PD motor, kebisingan sensor, posisi kamera, pencahayaan, tekstur, model kontak. Kebijakan ini mempelajari distribusi bersyarat mengenai "sim mana yang ada saat ini" dan menggeneralisasikannya ke seluruh rentang. Jika robot sebenarnya termasuk dalam cakupan training, kebijakan tersebut akan berhasil.

- **Kelebihannya:** tidak diperlukan data nyata. Satu resep, banyak robot.
- **Kelemahan:** Training yang terlalu acak menghasilkan kebijakan yang "universal" namun terlalu hati-hati. Terlalu banyak kebisingan ≈ terlalu banyak regularisasi.

**Identifikasi Sistem (SI).** Sesuaikan parameter simulator dengan data dunia nyata sebelum training. Jika kamu dapat mengukur gesekan sendi lengan pada robot asli, sambungkan ke sim. Kemudian latih kebijakan yang mengharapkan nilai-nilai tersebut. Membutuhkan akses ke sistem nyata namun mengurangi kesenjangan realitas secara langsung.

- **Kelebihannya:** target latihan yang presisi dan tidak menimbulkan kebisingan.
- **Kelemahan:** kesalahan model sisa tidak terlihat dalam kebijakan; efek kecil yang tidak teridentifikasi (misalnya, deadband motor) masih mengganggu penerapan.

**Adaptasi Domain.** Berlatih dalam sim, sempurnakan dengan sedikit data nyata. Dua rasa:

- **Real2Sim2Real:** pelajari simulator sisa `f(s, a, z) - f_sim(s, a)` menggunakan peluncuran nyata, latih sim yang telah diperbaiki. Menutup kesenjangan tanpa banyak data nyata.
- **Adaptasi observasi:** latih kebijakan yang memetakan obs nyata → obs mirip sim melalui ekstraktor feature yang dipelajari (misalnya, GAN piksel-ke-piksel). Pengontrol tetap di sim.**Pembelajaran istimewa / guru-siswa.** Miki dkk. 2022 (SETIAP hewan berkaki empat). Latih *guru* dalam simulasi yang memiliki akses ke informasi istimewa (gesekan kebenaran lapangan, ketinggian medan, penyimpangan IMU). Saring *siswa* yang hanya melihat pengamatan sensor nyata. Siswa belajar menyimpulkan feature-feature istimewa dari sejarah, yang kuat di seluruh parameter fisik.

**Simulasi paralel besar-besaran.** 2024–2026. Isaac Lab, Mujoco MJX, Brax semuanya menjalankan ribuan robot paralel dalam satu GPU. PPO dengan 4.096 humanoid paralel mengumpulkan pengalaman bertahun-tahun dalam hitungan jam. “Kesenjangan realitas” mengecil seiring dengan meluasnya distribusi training; DR menjadi hampir bebas ketika masing-masing dari 4.096 env tersebut memiliki parameter acak yang berbeda.

**Resep dunia nyata tahun 2026 (contoh berjalan kaki berkaki empat):**

1. Sim paralel besar-besaran dengan gravitasi acak domain, gesekan, penguatan motor, muatan.
2. Kebijakan guru dilatih dengan informasi istimewa (peta medan, kebenaran dasar kecepatan tubuh).
3. Kebijakan siswa disaring dari guru hanya dengan menggunakan proprioception (pengkode sendi kaki).
4. Adaptasi observasi opsional melalui autoencoder pada IMU nyata.
5. Terapkan. Zero-shot di 10+ lingkungan. Jika gagal, lakukan beberapa menit penyesuaian di dunia nyata dengan PPO dengan batasan keamanan.

## Build

Code lesson ini adalah demonstrasi kecil pengacakan domain di GridWorld dengan transisi *berisik*. Kami melatih kebijakan yang mengalami probabilitas slip acak dalam "sim" dan mengevaluasi secara "nyata" dengan tingkat slip yang tidak pernah dilihat selama training. Bentuknya dipetakan langsung ke transfer MuJoCo-ke-perangkat keras.

### Langkah 1: sim berparameter

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` adalah parameter yang diekspos oleh simulator. Dalam robotika nyata, hal itu bisa berupa gesekan, massa, penguatan motor — apa pun yang berubah antara simulasi dan nyata.

### Langkah 2: berlatih bersama DR

Di awal setiap episode, contoh `slip ~ Uniform[0.0, 0.4]`. Latih PPO / Q-learning / apa saja. Lakukan ini untuk banyak episode.

### Langkah 3: evaluasi zero-shot pada slip "nyata".

Evaluasi di `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`. Empat yang pertama berada dalam dukungan training; `0.5` dan `0.7` berada di luar. Kebijakan yang dilatih oleh DR harus tetap memberikan dukungan yang optimal di dalam negeri dan menurun dengan baik di luar. Kebijakan yang dilatih dengan slip tetap akan menjadi rapuh di luar slip training-nya.

### Langkah 4: bandingkan dengan training sempit

Latih kebijakan kedua hanya dengan `slip = 0.0`. Evaluasi pada sapuan `slip` yang sama. kamu akan melihat penurunan yang sangat besar segera setelah slip nyata > 0.

## Jebakan- **Terlalu banyak pengacakan.** Berlatihlah di `slip ∈ [0, 0.9]` dan kebijakan kamu sangat menghindari risiko sehingga tidak pernah mencoba jalur optimal. Cocokkan dengan *yang diharapkan* distribusi di dunia nyata, bukan "apa pun bisa terjadi".
- **Pengacakan terlalu sedikit.** Berlatihlah secara tipis-tipis dan kebijakan tidak dapat digeneralisasikan sama sekali. Gunakan kurikulum adaptif (Pengacakan Domain Otomatis) yang memperluas distribusi seiring dengan perbaikan kebijakan.
- **Ruang parameter salah diidentifikasi.** Mengacak hal yang salah (warna kamera saat celah sebenarnya adalah penundaan motor) dan DR tidak membantu. Profil robot asli terlebih dahulu.
- **Kebocoran info istimewa.** Seorang guru yang menggunakan keadaan global untuk bertindak, bukan hanya observasi, dapat menghasilkan siswa yang tidak dapat mengejar ketertinggalan. Pastikan kebijakan guru dapat diwujudkan oleh siswa dengan diberikan riwayat observasi.
- **Kegagalan transfer sim-ke-sim.** Jika kebijakan kamu tidak kuat terhadap varian sim yang lebih sulit, kebijakan tersebut juga tidak akan kuat terhadap dunia nyata. Selalu uji pada varian sim yang tersedia sebelum digunakan.
- **Tidak ada batasan keamanan di dunia nyata.** Kebijakan yang berfungsi dalam simulasi dan "berfungsi secara nyata" tanpa pelindung keamanan tingkat rendah masih dapat merusak perangkat keras. Tambahkan batas kecepatan, batas torsi, batas sambungan pada pengontrol yang tidak dipelajari.

## Pakai

Tumpukan sim-to-real 2026:

| Domain | Tumpukan |
|--------|-------|
| Penggerak berkaki (APA PUN, Bintik, humanoid) | Isaac Lab + DR + guru / siswa istimewa |
| Manipulasi (tangan cekatan, pick-and-place) | Isaac Lab + DR + DR-GAN untuk penglihatan |
| Mengemudi otonom | CARLA / NVIDIA DRIVE Sim + DR + penyempurnaan nyata |
| Balapan drone | RotorS / Flightmare + DR + adaptasi online |
| Manipulasi jari/di tangan | OpenAI Dactyl (DR pada skala yang belum pernah terjadi sebelumnya) |
| Senjata industri | MuJoCo-Warp + SI + penyempurnaan nyata kecil |

Untuk kontrol di semua skala, alur kerjanya konsisten: sesuaikan sim sebaik mungkin, acak apa yang tidak bisa kamu sesuaikan, latih kebijakan besar, saring, terapkan dengan perisai keselamatan.

## Kirim

Simpan sebagai `outputs/skill-sim2real-planner.md`:

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## Latihan

1. **Mudah.** Latih agen Q-learning di GridWorld dengan slip tetap (slip=0,0). Evaluasi pada slip ∈ {0.0, 0.1, 0.3, 0.5}. Pengembalian plot vs slip.
2. **Sedang.** Latih pengambilan sample agen pembelajaran DR Q `slip ~ Uniform[0, 0.3]`. Evaluasi sapuan yang sama. Berapa harga pembelian DR pada slip=0,5 (di luar distribusi)?
3. **Sulit.** Menerapkan kurikulum: mulai dengan slip=0,0, perluas rentang DR setiap kali kebijakan mencapai 90% dari optimal. Ukur total langkah lingkungan untuk mencapai slip=0,3 zero-shot vs. garis dasar DR yang tetap.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Kesenjangan realitas | "Perbedaan nyata" | Pergeseran distribusi antara training dan penerapan fisika/penginderaan. |
| Pengacakan domain (DR) | "Berlatih melintasi sim acak" | Acak parameter sim selama training sehingga kebijakan dapat digeneralisasi. |
| Identifikasi sistem (SI) | "Ukur sim yang nyata dan fit" | Perkirakan parameter fisik nyata; atur sim agar cocok. |
| Adaptasi domain | "Sempurnakan data nyata" | Penyempurnaan kecil di dunia nyata setelah training sim; dapat mengadaptasi obs atau dinamika. |
| Info istimewa | "Kebenaran dasar untuk guru" | Informasi yang hanya dimiliki oleh sim; siswa harus menyimpulkannya dari riwayat obs. |
| Guru/siswa | "Singkirkan hak istimewa -> dapat diamati" | Guru dilatih dengan jalan pintas; siswa belajar meniru tanpa mereka. |
| ADR | "Pengacakan Domain Otomatis" | Kurikulum yang memperluas rentang DR seiring dengan perbaikan kebijakan. |
| Nyata2Sim | "Menutup kesenjangan dengan data nyata" | Learn sisa untuk membuat sim meniru peluncuran sebenarnya. |

## Bacaan Lanjutan

- [Tobin dkk. (2017). Pengacakan Domain untuk Mentransfer Jaringan Syaraf Dalam dari Simulasi ke Dunia Nyata](https://arxiv.org/abs/1703.06907) — makalah DR asli (visi untuk robotika).
- [Peng dkk. (2018). Transfer Kontrol Robot Sim-ke-Nyata dengan Pengacakan Dinamik](https://arxiv.org/abs/1710.06537) — DR untuk dinamika, penggerak berkaki empat.
- [OpenAI dkk. (2019). Memecahkan Kubus Rubik dengan Tangan Robot](https://arxiv.org/abs/1910.07113) — Dactyl, ADR dalam skala besar.
- [Miki dkk. (2022). Mempelajari gerak perseptif yang kuat untuk robot berkaki empat di alam liar](https://www.science.org/doi/10.1126/scirobotics.abk2822) — guru-siswa untuk ANYmal.
- [Makoviychuk dkk. (2021). Isaac Gym: Simulasi Fisika Berbasis GPU Kinerja Tinggi untuk Pembelajaran Robot](https://arxiv.org/abs/2108.10470) — sim paralel besar yang mendorong penerapan pada tahun 2025–2026.
- [Akkaya dkk. (2019). Pengacakan Domain Otomatis](https://arxiv.org/abs/1910.07113) — Metode kurikulum ADR.
- [Sutton & Barto (2018). Bab. 8 — Perencanaan dan Pembelajaran dengan Metode Tabular](http://incompleteideas.net/book/RLbook2020.pdf) — framing Dyna (menggunakan model untuk perencanaan + peluncuran) yang mendasari pipeline sim-to-real modern.
- [Zhao, Queralta & Westerlund (2020). Transfer Sim-to-Real dalam Pembelajaran Penguatan Mendalam untuk Robotika: Survei](https://arxiv.org/abs/2009.13303) — taksonomi metode sim-to-real dengan hasil benchmark.
