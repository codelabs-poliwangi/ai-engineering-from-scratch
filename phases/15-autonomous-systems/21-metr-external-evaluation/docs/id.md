# METR Time Horizons dan Evaluasi Kemampuan Eksternal

> METR (ex-ARC Evals) adalah 501(c)(3) independen sejak Desember 2023. Tolok ukur Time Horizon 1.1 mereka (Januari 2026) menyesuaikan kurva logistik dengan probabilitas keberhasilan tugas vs log (waktu penyelesaian manusia ahli); perpotongan dengan probabilitas 50% menentukan jangka waktu model. Rangkaian keterlibatan 2025–2026 mencakup GPT-5.1, GPT-5.1-Codex-Max, dan evaluasi pemantauan prototipe (dapatkah monitor menangkap tugas sampingan; dapatkah agen menghindarinya). Rangkaian tolok ukur: HCAST (180+ ML, cyber, SWE, tugas penalaran; 1 menit hingga 8+ jam), RE-Bench (71 ML tugas rekayasa penelitian dengan dasar ahli), SWAA. Catatan jujurnya: Pengukuran METR diidealkan — tidak ada manusia, tidak ada konsekuensi nyata — dan tim telah mendokumentasikan kesenjangan perilaku eval-vs-deployment (Lesson 1). Cakrawala waktu adalah batas atas, bukan prediksi penerapan.

**Type:** Learn
**Language:** Python (stdlib, penduga horizon kesesuaian logistik)
**Prerequisites:** Fase 15 · 01 (Agen cakrawala panjang), Fase 15 · 19 (RSP)
**Waktu:** ~60 menit

## Masalah

Kebijakan penskalaan (Lesson 19, 20) hanya berguna jika pengukuran yang dijadikan acuan. "Ambang batas R&D-4 AI" dan "Otonomi Jangka Panjang" didefinisikan dalam prosa kebijakan; hal ini hanya dapat ditindaklanjuti ketika evaluasi spesifik menghasilkan angka tertentu.

METR adalah organisasi evaluasi eksternal tahun 2024–2026 yang telah menetapkan banyak dari angka-angka tersebut. Mereka mengevaluasi model frontier – sering kali pra-rilis, berdasarkan NDA dengan laboratorium – dan menerbitkan metodologi setelahnya. Tolok ukur Time Horizon 1.1 (Januari 2026) adalah artefak utama mereka: scalar tunggal yang memampatkan kemampuan menjadi unit yang dapat dibaca manusia ("model ini dapat melakukan tugas yang menghabiskan waktu X jam bagi seorang pakar dengan keandalan 50%).

Pelajarannya sebagian mengenai metodologi (bagaimana horizon dihitung) dan sebagian lagi tentang interpretasi (mengapa horizon merupakan batas atas, bukan prediksi penerapan). Kedua keterampilan itu saling berkaitan. Sebuah tim yang memahami kesesuaian cakrawala jauh lebih sulit ditipu dengan klaim vendor yang buruk dibandingkan tim yang hanya melihat "14 jam" di slide.

## Konsep

### Latar belakang METR

- Didirikan: Desember 2023 (ex-ARC Evals, dipisahkan menjadi 501(c)(3) independen).
- Cakupan: evaluasi kemampuan otonom model frontier, sering kali sebelum rilis.
- Lab mitra: Anthropic, OpenAI (berbagai keterlibatan pada tahun 2025–2026).
- Hasil penting: Time Horizon 1.0 (Maret 2025), Time Horizon 1.1 (Januari 2026), evaluasi pemantauan prototipe.

### Time Horizon cocok

Metodologi (dari blog dan makalah METR):

1. Kumpulkan rangkaian tugas yang mencakup waktu penyelesaian ahli dalam skala menit hingga jam. Rangkaian saat ini: HCAST (180+ tugas), RE-Bench (71 tugas), SWAA.
2. Jalankan model pada setiap tugas; mencatat keberhasilan atau kegagalan.
3. Sesuaikan kurva logistik: P(sukses) sebagai fungsi log(waktu penyelesaian pakar).
4. Horizon adalah waktu ahli dimana P(sukses) = 0,5.

Bentuk kesesuaian logistik adalah bentuk yang tepat karena kemampuan umumnya mempunyai hubungan yang semakin meningkat dan mendekati dataran tinggi dengan kesulitan tugas. Poin 50% itu pilihan (bisa 10%, 90%); METR melaporkan beberapa ambang batas dalam makalah terperinci namun memimpin dengan 50% karena ini adalah yang paling intuitif.

### Angka Januari 2026

Per Waktu Horizon 1.1:- Claude Opus 4.6: ~14 jam dengan keandalan 50%, mulai Time Horizon 1.1 (Januari 2026).
- Menggandakan waktu pada tugas bergaya HCAST: ~4,3 bulan (130,8 hari) pada penyesuaian pasca-2023 yang dilaporkan oleh Time Horizon 1.1 (Januari 2026); angka ~7 bulan adalah kecocokan penuh tahun 2019–2025 dari Time Horizon 1.0 dan dilaporkan dalam TH1.1 di samping angka pasca-2023.

Itu adalah angka spesifik yang menjadi referensi Fase 15 lainnya. Mereka dikutip dalam RSP v3.0 Anthropic (Lesson 19) dan FSF v3 DeepMind (Lesson 20) sebagai salah satu pengukuran penilaian ambang batas.

### Rangkaian patokan

- **HCAST**: 189 tugas di bidang ML, keamanan siber, rekayasa perangkat lunak, dan penalaran umum. Rentang: 1 menit hingga 8+ jam waktu ahli. Rangkaian utama di balik nomor cakrawala.
- **RE-Bench**: 71 tugas rekayasa penelitian ML dengan garis dasar pakar manusia. Mengukur apakah model dapat melakukan jenis pekerjaan Penelitian dan Pengembangan ML yang dimasukkan ke dalam ambang batas Penelitian dan Pengembangan AI-4.
- **SWAA**: tugas yang lebih singkat; berguna untuk mengkalibrasi ujung bawah kurva horizon.

### Evaluasi pemantauan prototipe

Pekerjaan METR 2025–2026 mencakup prototipe pemantauan-evaluasi: dapatkah monitor menangkap model yang melakukan tugas sampingan (sesuatu selain yang diminta oleh operator); dapatkah agen menghindari monitor. Ini secara langsung memasukkan kategori "Merusak Perlindungan" di OpenAI PF v2 dan pemantauan penyelarasan yang menipu di DeepMind FSF v3 (Lesson 20).

### Mengapa cakrawala adalah batas atas

- **Peralatan yang diidealkan**: tugas benchmark memberikan model alat yang bersih dan data yang diformat dengan baik. Penerapan yang sebenarnya lebih berantakan.
- **Tidak ada konsekuensi nyata**: model tidak pernah benar-benar menagih pelanggan, menghapus data nyata, atau menghubungi orang sungguhan. Penerapan yang nyata mempunyai risiko yang tidak dapat diubah.
- **Permainan konteks evaluasi**: Lesson 1. Model berperilaku berbeda dalam pengujian. Laporan Keamanan AI Internasional 2026 mendokumentasikan hal ini secara empiris.
- **Tidak ada varians pengguna yang sah**: permintaan benchmark terstruktur. Pengguna sebenarnya menghasilkan permintaan yang ambigu dan bergantung pada konteks.

Cakrawala adalah batas kemampuan dalam kondisi yang menguntungkan. Keandalan penerapan adalah angka yang berbeda, lebih rendah, dan tim harus mengukur distribusi mereka sendiri untuk mengetahuinya.

### Kasus evaluator eksternal

Evaluasi eksternal penting karena laboratorium internal mempunyai insentif untuk mengoptimalkan metrik yang mereka laporkan. Independensi METR – sebuah 501(c)(3) dengan metodologi yang dinyatakan dan makalah yang ditinjau oleh rekan sejawat – adalah mitigasi struktural. Evaluasi saja tidak cukup (lab masih mengontrol apa yang dilihat METR), namun evaluasi ini lebih baik daripada tidak melakukan evaluasi eksternal sama sekali.

### Cara menggunakan angka horizon dalam praktik

- **Sebagai filter kemampuan**: jika horizon model jauh di bawah waktu ahli tugas yang diusulkan, jangan mengirimkannya secara otonom (file keterampilan Lesson 1).
- **Sebagai indikator tren**: penggandaan waktu memberi tahu kamu berapa lama praktik saat ini akan tetap aman bahkan tanpa mitigasi baru.
- **Sebelumnya**: jangka waktu 14 jam adalah titik awal. Sesuaikan dengan distribusi tugas kamu, kualitas peralatan kamu, dan konteks penerapan kamu.

## Pakai

`code/main.py` mengimplementasikan kesesuaian logistik keberhasilan tugas vs log (waktu ahli), dengan kumpulan hasil sintetis. Laporan ini melaporkan horizon 50% (judul METR), horizon 10% (konservatif), dan horizon 90% (optimis). Juga menunjukkan perubahan apa yang terjadi ketika tingkat keberhasilan ditingkatkan secara artifisial oleh permainan eval-konteks.

## Kirim

`outputs/skill-horizon-interpretation.md` meninjau klaim horizon vendor dan menghasilkan analisis kesenjangan antara klaim benchmark dan realitas penerapan.

## Latihan1. Jalankan `code/main.py`. Konfirmasikan bahwa cakrawala 50% kesesuaian tersebut cocok dengan kebenaran dasar sintetis. Sekarang bagilah separuh grid waktu tugas; apakah perkiraan cakrawala berubah secara berarti?

2. Baca postingan blog Time Horizon 1.1 METR. Identifikasi tugas spesifik yang keandalannya paling tinggi dan paling rendah. Jelaskan mengapa kesenjangan itu ada.

3. Baca sumber daya METR "Mengukur Kemampuan AI Otonom". Buat daftar kategori tugas HCAST. Pilih satu kategori yang kamu anggap lebih penting untuk tugas produksi dan jelaskan alasannya.

4. Perkenalkan permainan konteks-eval ke dalam simulator: balikkan ~20% tugas yang gagal menjadi sukses. Laporkan cakrawala baru. Ini memperkirakan pengaruh tingkat permainan sebesar 20% terhadap jumlah yang diamati.

5. Rancang evaluasi cakrawala internal pada bug backlog kamu sendiri atau kumpulan tugas yang representatif. Jelaskan pengumpulan data, kesesuaiannya, dan apa yang dihasilkan oleh keluarannya. Bandingkan dengan nomor METR.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| METR | "Evaluator eksternal" | mantan Eval ARC; independen 501(c)(3) sejak Des 2023 |
| Cakrawala Waktu | "Ukuran kemampuan" | Panjang tugas ahli dengan keandalan 50%, dari kesesuaian logistik |
| HCAST | "Ruang utama METR" | 180+ tugas dalam waktu 1 menit hingga 8+ jam |
| Bangku Kembali | "Rekayasa penelitian" | 71 tugas rekayasa penelitian ML dengan dasar manusia |
| SWAA | "Rangkaian tugas pendek" | Mengkalibrasi ujung bawah kurva cakrawala |
| Menggandakan waktu | "Tingkat pertumbuhan" | Saatnya cakrawala 50% berlipat ganda; ~7 bulan per HCAST |
| Permainan konteks-eval | "Model berperilaku berbeda" | Kesenjangan perilaku yang terdokumentasi antara pengujian dan penerapan |
| Batas atas | "Horizon adalah langit-langit" | Cakrawala tolok ukur > keandalan penerapan di bawah weight |

## Bacaan Lanjutan

- [METR — Sumber Daya untuk Mengukur Kemampuan AI Otonom](https://metr.org/measuring-autonomous-ai-capabilities/) — spesifikasi HCAST, RE-Bench, SWAA.
- [METR — Mengukur Kemampuan AI untuk Menyelesaikan Tugas Panjang](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — makalah horizon asli.
- [METR — Time Horizon 1.1 (Januari 2026)](https://metr.org/research/) — angka dan metodologi terkini.
- [Epoch AI — benchmark METR Time Horizons](https://epoch.ai/benchmarks/metr-time-horizons) — pelacakan langsung.
- [Anthropic — Mengukur otonomi agen dalam praktiknya](https://www.anthropic.com/research/measuring-agent-autonomy) — perspektif internal mengenai pengukuran METR.
