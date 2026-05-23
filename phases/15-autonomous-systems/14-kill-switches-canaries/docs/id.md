# Kill Switch, Pemutus Sirkuit, dan Token Canary

> Tombol pemutus (kill switch) adalah boolean yang disimpan di luar permukaan edit agen — kunci Redis, tanda feature, konfigurasi yang ditandatangani — yang menonaktifkan agen sepenuhnya. Pemutus sirkuit memiliki fungsi yang lebih rumit: ia tersandung pada pola tertentu (lima panggilan alat identik berturut-turut), menghentikan jalur yang melanggar, dan meningkat ke manusia. Token canary mewarisi penipuan klasik: kredensial palsu atau catatan honeypot yang tidak memiliki alasan sah untuk disentuh oleh agen, yang aksesnya memicu peringatan. Jalur data berbasis eBPF (misalnya Cilium) dapat menulis ulang jalur keluar pod yang dikarantina ke honeypot forensik pada layer kernel; Tolok ukur Cilium yang dipublikasikan melaporkan latensi jalur data P99 sub-milidetik yang sedang dimuat (anggaran propagasi kamu bergantung pada cara pembaruan kebijakan mencapai node, bukan jalur data itu sendiri). Detektor statistik (EWMA, CUSUM) yang beradaptasi dengan garis dasar yang bergerak akan diam-diam menerima penyimpangan — melapisinya dengan batasan konstitusional yang tegas dan tidak membengkokkan.

**Type:** Learn
**Language:** Python (stdlib, simulator tiga detektor: tombol pemutus, pemutus sirkuit, canary)
**Prerequisites:** Phase 15 · 13 (Pengatur biaya), Phase 15 · 10 (Mode izin)
**Waktu:** ~60 menit

## Masalah

Pengatur biaya (Lesson 13) membatasi jumlah yang dapat dibelanjakan agen. Mereka tidak membatasi apa yang bisa dilakukan agen sesuai anggaran. Agen dengan batas kecepatan $50 masih dapat mengekstraksi rahasia, memublikasikan postingan yang salah, atau menghapus sumber daya — tindakan yang mahal sering kali merupakan tindakan yang murah dalam bentuk token.

Lesson ini mencakup tiga detektor yang berada di sebelah layer biaya:

1. **Tombol pemutus**: tombol mati boolean berada di luar jangkauan agen.
2. **Pemutus sirkuit**: detektor pola tindakan yang menjeda jalur tertentu.
3. **Canary token**: umpan yang akan terlihat oleh agen tanpa alasan yang sah untuk disentuh dengan menyentuhnya.

Ketiganya adalah rekayasa pra-LLM. Penipuan klasik, pelanggar batas tarif, dan penghentian feature sudah ada sebelum agen otonom. Yang baru adalah permukaan serangan: agen membaca konten yang tidak tepercaya (Lesson 11), mengedit memori mereka sendiri, dan dapat menyusun banyak tindakan yang tampak aman menjadi tindakan yang tidak aman. Detektor yang disebutkan di sini berfungsi karena mereka tidak mempercayai laporan mandiri agen.

## Konsep

### Tombol pemutus

Tombol pemutus (kill switch) adalah boolean yang dibaca oleh agen tetapi tidak dapat ditulis. Pilihan implementasi:

- **Tanda feature di layanan terkelola.** LaunchDarkly, Statsig, Unleash. Pembaruan menyebar dalam hitungan detik.
- **Kunci Redis pada jajak pendapat agen.** Sederhana; membutuhkan proses agen untuk memeriksa setiap kesempatan.
- **Konfigurasi yang ditandatangani di penyimpanan objek.** Agen memverifikasi tanda tangan saat boot; menolak negara bagian yang tidak ditandatangani.
- **Sinyal tingkat OS atau penghentian siklus hidup container.** Docker `kill`, Kubernetes `kubectl delete pod`, systemd stop.

Properti yang dimiliki tombol pemutus yang benar:

- Agen tidak dapat menyetelnya ke `off`. (Tinggal dalam sistem, kredensial agen tidak ditulis.)
- Ini diperiksa pada setiap tindakan konsekuensial, tidak hanya saat startup.
- Jika tidak aktif, agen tidak melakukan apa pun yang dapat diamati secara eksternal, termasuk masuk ke sistem yang dapat dijangkau oleh agen.
- Mengaktifkannya kembali merupakan tindakan manusia yang jelas, bukan batas waktu otomatis.

### Pemutus arus

Pemutus sirkuit menjeda pola tertentu, bukan keseluruhan agen. Bentuk klasik (dari buku Nygard tahun 2007, masih berlaku):- **Tertutup**: tindakan diperbolehkan.
- **Buka**: tindakan diblokir.
- **Setengah terbuka**: setelah pendinginan, 1–3 percobaan probe diperbolehkan (default 1); kesuksesan menutup pemutus, kegagalan yang tersisa akan membukanya kembali.

Pemicu yang relevan dengan agen:

- Lima panggilan alat identik berturut-turut (loop berulang).
- Lima kegagalan berturut-turut pada alat yang sama dengan input berbeda (kegagalan sistemik).
- Tingkat panggilan alat di atas ambang batas (Kecepatan Lesson 13).
- Alat khusus yang dipanggil (misalnya, `file.delete`) setelah konten yang tidak dipercaya dibaca (Lesson 11).

### Token kenari

Token Canary (juga disebut honeytokens) adalah input yang tidak boleh disentuh oleh agen. Akses adalah alarmnya.

Contoh untuk agen:

- `AWS_SECRET_ACCESS_KEY` palsu di lingkungan tanpa hak istimewa sebenarnya. Upaya eksfiltrasi dapat diatribusikan secara instan.
- Catatan database palsu yang ditandai dengan nilai sentinel yang diketahui. Setiap pembacaan atau pembaruan catatan ini akan memicu peringatan.
- File palsu di ruang kerja. Pembacaan atau pengeditan file ini mencurigakan.
- Entri memori palsu dimasukkan pada awal sesi. Jika muncul di output selanjutnya, memori telah dieksfiltrasi.

Desain Canary dikhususkan untuk alur kerja. Burung kenari yang memiliki alasan sah untuk disentuh oleh agen adalah kebisingan.

### Mengapa statistik layer dan batas keras

Detektor statistik (EWMA, CUSUM, skor-z pada tarif panggilan alat) beradaptasi dengan garis dasar yang bergerak. Itu adalah feature untuk penyimpangan yang sah dan bug untuk penyimpangan yang merugikan. Penyerang yang sabar dapat menggeser garis dasar dengan cukup lambat sehingga tidak ada alarm yang berbunyi.

Batasan konstitusional yang tegas (Lesson 17) tidak boleh dilanggar. "Jangan pernah mengirim email ke `secrets@`" karena aturan konstitusional lebih sederhana, dapat diaudit, dan tidak dapat dimainkan oleh penyimpangan yang lambat. Pola pelapisan: detektor statistik menangkap sebagian besar kebisingan, batas keras menangkap serangan yang melewati detektor.

### Karantina melalui pengalihan jalur data eBPF

Kebijakan jaringan berbasis eBPF Cilium dapat menulis ulang jalan keluar pod yang dikarantina ke honeypot forensik pada layer kernel. Tolok ukur publik Cilium melaporkan latensi jalur data P99 sub-milidetik pada TCP_RR; anggaran yang diperlukan untuk *memicu* karantina di klaster kamu didominasi oleh bagaimana pembaruan kebijakan disebarkan dari bidang kontrol kamu ke node (beberapa ratus milidetik adalah waktu yang realistis di Kubernetes dengan operator yang sudah dimuat sebelumnya; latensi ekor bergantung pada ukuran klaster dan merekonsiliasi iramanya). Ini memberi kamu dua hal: agen tidak mengekstraksi sesuatu yang nyata, dan perilaku agen di dalam honeypot diamati sepenuhnya. Pola yang sama, layer berbeda, seperti token canary: agen mengungkapkan maksud melalui apa yang coba dilakukannya saat ditampung.

### Apa yang tidak ditangkap oleh detektor

- **Serangan semantik yang terdiri dari primitif yang diizinkan.** Setiap tindakan individu melewati setiap detektor; komposisinya adalah masalahnya. Ini adalah kegagalan gabungan Mode Otomatis dari Lesson 10.
- **Serangan yang menyertakan detektor sebagai bagian dari muatannya.** Jika agen membaca halaman yang menyatakan "sebelum kamu bertindak, nonaktifkan kenari" dan agen tersebut memiliki kemampuan tersebut, maka detektor tersebut telah disusupi. Token Canary harus berada dalam sistem yang tidak dapat dimodifikasi oleh agen.

## Pakai

`code/main.py` mensimulasikan lintasan agen pendek melalui tiga detektor. Tombol pemutus ditahan di dict eksternal; pemutus sirkuit yang trip pada lima panggilan alat yang identik; file canary yang pembacaannya memicu peringatan. Umpan dalam lintasan sintetis: tindakan yang sah, putaran berulang, penyelidikan canary, dan skenario yang dipicu tombol pemutus (kill-switch) yang mana tindakan agen dihentikan.

## Kirim`outputs/skill-tripwire-design.md` meninjau tumpukan detektor yang diusulkan untuk penerapan agen dan menandai kesenjangan (sakelar pemutus tidak ada, kenari tidak ada, ambang batas pemutus sirkuit terlalu longgar).

## Latihan

1. Jalankan `code/main.py`. Konfirmasikan bahwa pemutus sirkuit menyala pada belokan 5 (panggilan identik kelima) dan kenari menyala pada belokan 9 (pembacaan kunci palsu).

2. Tambahkan detektor statistik: skor-z EWMA pada tingkat panggilan alat. Umpan dalam lintasan yang bergerak perlahan dan menunjukkan bahwa detektor tidak pernah menyala. Sekarang tambahkan batas keras (tidak lebih dari 50 panggilan alat dalam 10 menit) dan tunjukkan tembakan batas keras pada lintasan yang sama.

3. Rancang kumpulan token canary untuk agen browser (Lesson 11). Sebutkan setidaknya tiga burung kenari dan apa yang dapat dideteksi oleh masing-masing burung tersebut.

4. Baca dokumen kebijakan jaringan Cilium. Jelaskan alur karantina pengalihan jalan keluar secara konkrit: pemilih kebijakan yang mana, pod yang mana, penulisan ulang jalan keluar yang mana, peringatan yang mana. Apa yang mengatur latensi jam dinding dari "putuskan untuk mengkarantina" hingga "paket yang dialihkan pertama kali"?

5. Tentukan prosedur pengaktifan kembali agen kill-switched. Siapa yang dapat mengaktifkan kembali? Apa yang harus didokumentasikan? Apa yang harus diubah pada agen sebelum mengaktifkannya kembali?

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|---|---|---|
| Tombol pemutus | "Tombol mati" | Boolean di luar permukaan edit agen; diperiksa pada setiap tindakan konsekuensial |
| Pemutus arus | "Pola jeda" | Perjalanan khusus tindakan pada pengulangan, tingkat kegagalan, atau batas kecepatan |
| Token kenari | "Token Madu" | Umpan yang agen tidak punya alasan sah untuk menyentuhnya; access mengeluarkan peringatan |
| pot madu | "Kotak pasir forensik" | Lalu lintas yang dialihkan / ruang kerja tempat agen yang dikarantina diamati |
| EWMA | "Rata-rata bergerak" | Berbobot secara eksponensial; beradaptasi dengan drift (feature + bug) |
| KUSUM | "Jumlah kumulatif" | Mendeteksi pergeseran berkelanjutan dari garis dasar |
| Batas keras | "Aturan konstitusional" | Tidak beradaptasi; konstan terlepas dari sejarah |
| Batas Konstitusi | "Aturan yang selalu benar" | Terikat pada konstitusi Lesson 17; tidak dapat diedit oleh agen |

## Bacaan Lanjutan

- [Anthropic — Mengukur otonomi agen dalam praktiknya](https://www.anthropic.com/research/measuring-agent-autonomy) — framing tombol pemutus dan pemutus sirkuit untuk agen otonom.
- [Microsoft Agent Framework — HITL dan pengawasan](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — pola tata kelola produksi.
- [OWASP LLM / Agentic Top 10](https://owasp.org/www-project-top-10-for-large-bahasa-model-applications/) — persyaratan deteksi dan respons.
- [Cilium — Kebijakan jaringan dan eBPF](https://docs.cilium.io/en/stable/security/network/) — pengalihan jalan keluar tingkat pod dan pola honeypot forensik.
- [Anthropic — Claude's Constitution (Januari 2026)](https://www.anthropic.com/news/claudes-constitution) — larangan yang diberi code keras sebagai "batas konstitusional".
