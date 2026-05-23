# Keamanan — Rahasia, Rotasi Kunci API, Log Audit, Pagar Pembatas

> Hilangkan penyebaran rahasia melalui brankas terpusat (HashiCorp Vault, AWS Secrets Manager, Azure Key Vault). Jangan pernah menyimpan kredensial di file konfigurasi, file env di VCS, spreadsheet. Gunakan peran IAM melalui kunci statis; OIDC untuk CI/CD. Pola AI-gateway adalah solusi tahun 2026: aplikasi → gateway → penyedia model, dengan gateway menarik kredensial dari vault saat runtime. Putar di vault dan semua aplikasi diambil dalam hitungan menit — tidak ada penerapan ulang, tidak ada pesan Slack "siapa yang memiliki kunci baru". Kebijakan rotasi ≤90 hari; pindai dengan TruffleHog / GitGuardian / Gitleaks pada setiap komit. Zero-trust: MFA, SSO, RBAC/ABAC, token berumur pendek, postur perangkat. Scrubbing PII menggunakan pengenalan entitas untuk menutupi PHI/PII sebelum meneruskan; tokenization yang konsisten (pendekatan Mesh) memetakan nilai-nilai sensitif ke placeholder yang stabil sehingga LLM mempertahankan semantik code/hubungan. Jalan keluar jaringan: Layanan LLM dalam daftar putih subnet VPC/VNet khusus hanya `api.openai.com`, `api.anthropic.com` dll; memblokir semua keluar lainnya. Penyebab insiden tahun 2026: Serangan rantai pasokan Vercel melalui kredensial CI/CD yang disusupi menyaring env vars di ribuan penerapan pelanggan.

**Type:** Learn
**Language:** Python (stdlib, mainan PII-scrubber + penulis log audit)
**Prerequisites:** Fase 17 · 19 (AI Gateways), Fase 17 · 13 (Observabilitas)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Hitung empat anti-pola manajemen rahasia (file konfigurasi dalam VCS, env hardcoded, spreadsheet, kunci statis) dan beri nama penggantinya.
- Menjelaskan pola AI-gateway-pulls-from-vault sebagai standar produksi tahun 2026.
- Menerapkan scrubber PII dengan tokenization yang konsisten (nilai yang sama → placeholder yang sama) sehingga semantik tetap bertahan.
- Sebutkan insiden rantai pasokan Vercel tahun 2026 dan apa yang diajarkannya tentang kebersihan kredensial CI/CD.

## Masalah

Seorang pekerja magang melakukan `.env` dengan kunci API. Mereka menghapusnya dengan cepat. Kuncinya sudah ada dalam riwayat git — Pemindaian GitGuardian menangkapnya, proses rotasi kamu adalah "Kendurkan tim, perbarui 40 file konfigurasi, sebarkan ulang semua layanan." 8 jam kemudian, separuh layanan kamu aktif dan separuh lagi menunggu jendela penerapan.

Secara terpisah, prompt pengguna mencakup "SSN saya adalah 123-45-6789." Prompt pergi ke OpenAI. kamu memiliki BAA tetapi kebijakan internal kamu adalah menutupi PII sebelum meneruskan. kamu tidak melakukannya.

Secara terpisah, pod LLM klaster EKS kamu dapat menjangkau host internet mana pun. Seseorang mengeluarkan data melalui pencarian DNS ke domain yang dikendalikan penyerang. Tidak ada yang menghalanginya.

Keamanan untuk layanan LLM harus mengatasi ketiga vector tersebut. Kredensial yang didukung Vault. Penggosokan PII. Pemfilteran jalan keluar jaringan. Log audit.

## Konsep

### Vault terpusat + penarikan peran IAM

**Vault**: HashiCorp Vault, Manajer Rahasia AWS, Azure Key Vault, Manajer Rahasia GCP. Salah satu sumber kebenaran.

**Peran IAM**: aplikasi/gateway mengautentikasi melalui identitas IAM-nya, bukan kunci statis. Vault mengembalikan rahasia seumur hidup token.

**Pola AI-gateway**: gateway menarik `OPENAI_API_KEY` dari vault pada waktu permintaan. Putar di brankas; permintaan berikutnya mendapatkan kunci baru. Tidak ada penempatan ulang.

### Kebijakan rotasi ≤ 90 hari

Semua kunci API, token root vault, kredensial CI/CD. Rotasi otomatis jika memungkinkan. Rotasi manual dicatat dan dilacak.

### Pemindaian rahasia

- **TruffleHog** — regex + entropi saat dilakukan.
- **GitGuardian** — komersial, akurasi tinggi.
- **Gitleaks** — OSS, berjalan di CI.

Jalankan di setiap komit. Blokir PR jika rahasia baru terdeteksi.

### Postur tanpa rasa percaya- MFA diperlukan di semua akun.
- SSO melalui SAML/OIDC.
- RBAC (berbasis peran) atau ABAC (berbasis atribut) untuk akses terperinci.
- Token berumur pendek (jam, bukan hari).
- Postur perangkat — hanya perangkat perusahaan dengan enkripsi disk.

### Penggosokan PII / PHI

Sebelum prompt meninggalkan infra kamu:

1. Pengenalan entitas (spaCy NER, Presidio, komersial).
2. Entitas yang cocok dengan topeng: `"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`.
3. Tokenization yang konsisten (pendekatan Mesh): nilai yang sama dipetakan ke placeholder yang sama sehingga LLM menjaga hubungan.
4. Pemetaan terbalik opsional untuk respons LLM.

Filter regex statis menangkap pola dasar; NER menangkap lebih banyak. Gunakan keduanya.

### Pagar pembatas input + output

Input: memblokir jailbreak yang diketahui, topik terlarang; batas tarif per pengguna.

Output: scrub regex untuk rahasia yang bocor (pola kunci API, pola email dalam konteks penolakan), pengklasifikasi untuk pelanggaran kebijakan.

### Daftar putih keluar jaringan

Layanan LLM di subnet khusus:
- Daftar Putih: `api.openai.com`, `api.anthropic.com`, titik akhir vector DB, titik akhir vault.
- Yang lainnya: jatuhkan.
- DNS melalui penyelesai khusus daftar yang diizinkan (hindari exfil kanalisasi DNS).

### Catatan audit

Log yang tidak dapat diubah dari setiap panggilan LLM dengan:
- Stempel waktu.
- Pengguna / penyewa.
- Hash cepat (bukan prompt mentah untuk privasi).
- Model + versi.
- Token penting.
- Biaya.
- Respon hash.
- Setiap perjalanan pagar pembatas.

Simpan sesuai persyaratan peraturan (SOC 2 1 tahun, HIPAA 6 tahun).

### Insiden Vercel tahun 2026

Serangan rantai pasokan: kredensial CI/CD yang disusupi menyaring env vars di ribuan penerapan pelanggan. Lesson: Kredensial CI/CD setara dengan produk. Simpan di lemari besi. Cakupannya sempit. Putar secara agresif.

### Nomor yang harus kamu ingat

- Kebijakan rotasi: ≤ 90 hari.
- Pindai setiap komit: TruffleHog / GitGuardian / Gitleaks.
- Vercel 2026: Kredibilitas CI/CD dikompromikan → ribuan variabel lingkungan pelanggan bocor.
- Retensi log audit: SOC 2 = 1 tahun, HIPAA = 6 tahun.

## Pakai

`code/main.py` mengimplementasikan scrubber PII mainan dengan tokenization yang konsisten dan log audit khusus tambahan.

## Kirim

Lesson ini menghasilkan `outputs/skill-llm-security-plan.md`. Mengingat cakupan peraturan dan kondisi saat ini, rencanakan migrasi vault, scrubber, jalan keluar, dan log audit.

## Latihan

1. Jalankan `code/main.py`. Kirim dua prompt yang merujuk pada SSN yang sama. Konfirmasikan keduanya mendapatkan placeholder yang sama.
2. Rancang kebijakan keluar jaringan untuk penerapan vLLM-on-EKS yang memanggil OpenAI + Anthropic + Weaviate.
3. kamu menemukan kunci dalam riwayat git (2 tahun). Apa respons yang benar — memutar kunci, menghapus riwayat, atau keduanya? Membenarkan.
4. Log audit kamu bertambah 10 GB/hari. Tingkat retensi desain (panas 30 hari, hangat 12 bulan, dingin 6 tahun).
5. Berdebat apakah tokenization terbalik (mengganti nilai nyata kembali ke respons LLM) sepadan dengan kompleksitasnya dibandingkan menjaga placeholder tetap terlihat.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| Gudang | "toko rahasia" | Layanan manajemen kredensial terpusat |
| Peran IAM | "autentikasi berbasis identitas" | Peran diambil oleh aplikasi; mengembalikan kredit berumur pendek |
| OIDC untuk CI/CD | "token yang dikeluarkan cloud" | Tidak ada kunci statis di CI — identitas melalui OIDC |
| TruffleHog / GitGuardian / Gitleaks | "pemindai rahasia" | Deteksi rahasia waktu komitmen |
| RBAC / ABAC | "kontrol akses" | Berbasis peran vs berbasis atribut |
| Penggosokan PII | "penyembunyian data" | Hapus atau tokenization entitas sensitif |
| Tokenization yang konsisten | "placeholder stabil" | Nilai yang sama → token yang sama setiap kali |
| Pendekatan jala | "Tokenization mesh" | Pola tokenization pelestarian semantik |
| Daftar putih jalan keluar | "daftar keluar yang diizinkan" | Hanya domain yang diizinkan yang dapat dijangkau |
| Catatan audit | "sejarah yang tidak dapat diubah" | Catatan tambahkan saja untuk kepatuhan |

## Bacaan Lanjutan

- [Doppler — Keamanan LLM Tingkat Lanjut](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Kelola kunci API LLM dengan referensi rahasia](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — Praktik Terbaik Pagar Pembatas LLM](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Praktik Terbaik Manajemen Rahasia 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — Deteksi dan anonimisasi PII.
- [Dokumen HashiCorp Vault](https://developer.hashicorp.com/vault/docs)
