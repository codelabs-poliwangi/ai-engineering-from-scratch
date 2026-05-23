# Otentikasi MCP dalam Produksi — DCR, Rotasi JWKS, Token yang Di-embed Audiens pada iii Primitif

> Lesson 16 menampilkan mesin status OAuth 2.1 di memori. Pada tahun 2026, setiap server MCP yang kamu kirim ke organisasi sebenarnya berada di belakang autentikasi produksi: pendaftaran klien dinamis (RFC 7591), penemuan metadata server otorisasi (RFC 8414), rotasi JWKS yang tidak merusak validasi token pukul 3 pagi, dan token yang dipasangi audiens yang menolak penggunaan kembali deputi yang bingung. Lesson ini menghubungkan semua itu melalui iii primitif — `iii.registerTrigger` untuk HTTP dan cron, `iii.registerFunction` untuk logika autentikasi, `state::set/get` untuk kunci cache — sehingga permukaan autentikasi dapat diamati, dapat dimulai ulang, dan dapat diputar ulang seperti setiap weight kerja lainnya di mesin.

**Type:** Build
**Language:** Python (stdlib, iii primitif diejek untuk lingkungan lesson)
**Prerequisites:** Fase 13 · 16 (mesin status OAuth 2.1), Fase 13 · 17 (gateway)
**Waktu:** ~90 menit

## Tujuan Pembelajaran

- Temukan server otorisasi melalui metadata RFC 8414 dan verifikasi kontrak.
- Menerapkan pendaftaran klien dinamis RFC 7591 sehingga klien MCP mendaftar tanpa intervensi admin.
- Cache dan putar kunci JWKS menggunakan pemicu cron sehingga verifikasi tanda tangan bertahan dalam roll-over kunci.
- Sematkan token ke satu sumber daya MCP menggunakan indikator sumber daya RFC 8707 dan tolak penggunaan kembali yang membingungkan.
- Hubungkan setiap titik akhir dan pekerjaan latar belakang sebagai iii primitif — pemicu HTTP, pemicu cron, fungsi bernama, dan pembacaan `state::*` — sehingga satu kali restart akan membangun kembali permukaan autentikasi.
- Membaca matrix kemampuan IdP dan menolak penerapan ketika IdP tidak dapat memenuhi profil autentikasi MCP.

## Masalah

Simulator Lesson 16 menjalankan OAuth 2.1 di memori. Produksi memiliki tiga celah operasional yang tidak dapat dilihat oleh simulator memori saja.

Kesenjangan pertama adalah pendaftaran. Sebuah organisasi nyata menjalankan ratusan server MCP dan ribuan klien MCP. Operator tidak mendaftarkan setiap pengguna Cursor sebagai klien OAuth. Pendaftaran klien dinamis RFC 7591 memungkinkan klien `POST /register` melawan server otorisasi dan menerima `client_id` (dan opsional `client_secret`) saat itu juga. Server menerbitkan `registration_endpoint` dalam metadata RFC 8414; klien menemukannya tanpa konfigurasi out-of-band.

Kesenjangan kedua adalah rotasi kunci. Validasi JWT bergantung pada kunci penandatanganan server otorisasi, yang dipublikasikan sebagai JSON Web Key Set (JWKS). Server otorisasi memutarnya sesuai jadwal (sering kali setiap jam, terkadang lebih cepat dalam respons insiden). Server MCP yang mengambil JWKS sekali saat boot divalidasi dengan baik hingga jendela rotasi — kemudian setiap permintaan gagal hingga dimulai ulang. Produksi menghubungkan JWKS sebagai nilai yang di-cache dengan tugas penyegaran yang menimpa cache sebelum kunci sebelumnya kedaluwarsa, ditambah pengambilan cadangan pada cache yang hilang untuk kasus di mana token ditandatangani oleh kunci yang lebih baru daripada cache yang tiba.

Kesenjangan ketiga adalah pengikatan penonton. Lesson 16 memperkenalkan indikator sumber daya RFC 8707. Dalam produksi, indikator tersebut menjadi pemeriksaan klaim yang sulit pada setiap permintaan. Server MCP membandingkan `token.aud` dengan URL sumber daya kanoniknya sendiri dan menolak ketidakcocokan dengan HTTP 401. Ini adalah satu-satunya pertahanan terhadap server MCP hulu (atau klien jahat yang memegang token yang ditujukan untuk satu server) yang memutar ulang token tersebut terhadap server lain dalam jaringan kepercayaan yang sama.Lesson ini memperlakukan setiap kesenjangan tersebut sebagai sesuatu yang primitif. Dokumen metadata adalah pemicu HTTP yang mengembalikan output suatu fungsi. Rotasi JWKS adalah pemicu cron yang memanggil `auth::rotate-jwks`, yang menulis ke `state::set("auth/jwks/<issuer>", ...)`. Validasi JWT adalah fungsi yang dipanggil orang lain melalui `iii.trigger("auth::validate-jwt", token)`. Server MCP sendiri hanyalah pemicu HTTP lain yang memanggil validasi sebelum pengiriman. Nyalakan ulang mesin: registri pemicu dibangun kembali; negara bertahan; permukaan autentikasi dapat dioperasikan tanpa rekonsiliasi manual.

## Konsep

### RFC 8414 — Metadata Server Otorisasi OAuth

Dokumen di `/.well-known/oauth-authorization-server` menjelaskan semua yang dibutuhkan klien:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

Klien diberi penemuan rantai URL sumber daya MCP: `oauth-protected-resource` dari RFC 9728 (dokumen server sumber daya) memberi nama penerbitnya, lalu `oauth-authorization-server` (RFC ini) memberi nama setiap titik akhir. Klien tidak pernah melakukan hard-code pada URL otorisasi.

Kontrak yang kamu verifikasi sebelum memercayai IdP untuk MCP:

- `code_challenge_methods_supported` termasuk `S256` (PKCE per RFC 7636).
- `grant_types_supported` termasuk `authorization_code` dan menolak `password` dan `implicit`.
- `registration_endpoint` hadir (dukungan RFC 7591).
- `response_types_supported` sama persis dengan `["code"]` untuk OAuth 2.1.

Jika salah satu dari hal tersebut tidak ada, server MCP akan menolak penerapan terhadap IdP ini. Manifes penerapan yang salah, bukan kodenya.

### RFC 9728 (rekap) — Metadata Sumber Daya yang Dilindungi

Lesson 16 membahas RFC 9728. Delta dalam produksi: dokumen ini adalah satu-satunya tempat klien mencari untuk menemukan server otorisasi yang dipercaya oleh server MCP *ini*. Satu server MCP dapat menerima token dari beberapa IdP (satu untuk staf, satu untuk mitra). RFC 9728 menyatakan himpunan itu; RFC 8414 mendokumentasikan apa yang didukung setiap IdP.

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### RFC 7591 — Pendaftaran Klien Dinamis

Tanpa DCR, setiap klien MCP (Cursor, Claude Desktop, agen kustom) memerlukan pertukaran out-of-band dengan admin IdP. Dengan DCR, klien memposting:

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

Server merespons dengan `client_id` dan `registration_access_token` untuk pembaruan selanjutnya:

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` adalah default yang tepat untuk klien MCP yang berjalan di perangkat pengguna. Mereka hanya mendapatkan `client_id` — tidak ada `client_secret` untuk melakukan eksfiltrasi. PKCE memberikan bukti kepemilikan yang dibutuhkan klien publik.

Tiga kendala produksi:

- Titik akhir pendaftaran harus dibatasi tarifnya berdasarkan IP sumber. Tanpa itu, aktor yang bermusuhan akan membuat jutaan registrasi palsu dan menghabiskan namespace `client_id`. iii menjadikan hal ini sepele: pemicu HTTP registrasi memanggil fungsi `auth::rate-limit` sebelum mengirim ke registrar.
- `software_statement` (vouching JWT yang ditandatangani untuk klien) diperlukan oleh beberapa IdP perusahaan. Tiruan lesson melewatkannya; produksi menghubungkan langkah verifikasi yang menolak pendaftaran yang tidak ditandatangani dari apa pun selain URI pengalihan localhost.
- `registration_access_token` harus disimpan sebagai hash, bukan teks biasa. Pencurian token ini berarti penyerang dapat menulis ulang URI pengalihan klien.

### RFC 8707 (rekap) — Indikator Sumber DayaPelajaran 16 menetapkan bentuknya. Aturan produksi: setiap permintaan token menyertakan `resource=<canonical-mcp-url>`, dan server MCP memverifikasi `token.aud` cocok dengan URL sumber dayanya sendiri pada setiap panggilan. Jika server MCP dapat dijangkau di `https://notes.example.com/mcp`, URL kanonisnya adalah `https://notes.example.com` — komponen jalur dikecualikan sehingga satu server menghosting beberapa jalur di bawah satu audiens.

### RFC 7636 (rekap) — PKCE

PKCE bersifat wajib di OAuth 2.1. Alur code otorisasi lesson selalu membawa `code_challenge` dan `code_verifier`. Server menolak permintaan token apa pun tanpa pemverifikasi atau dengan pemverifikasi yang tidak melakukan hash terhadap tantangan yang disimpan.

### Spesifikasi MCP Profil Otentikasi 25-11-2025

Spesifikasi MCP (25-11-2025) menjelaskan secara tepat apa yang harus dilakukan layer otorisasi server MCP:

- Publikasikan `/.well-known/oauth-protected-resource` (RFC 9728).
- Terima token hanya melalui `Authorization: Bearer ...`.
- Validasi `aud`, `iss`, `exp`, dan cakupan yang diperlukan per permintaan.
- Tanggapi dengan `WWW-Authenticate` yang membawa `Bearer error=...` untuk setiap 401 dan 403, termasuk parameter `scope=` dan `resource=` jika berlaku.
- Tolak token yang `aud` tidak cocok dengan sumber daya kanonik.
- Tolak token yang `iss` tidak ada dalam daftar `authorization_servers` metadata sumber daya yang dilindungi.

Draf OAuth 2.1 adalah substratnya; RFC 8414/7591/8707/9728 + RFC 7636 adalah permukaannya; spesifikasi MCP adalah profilnya.

### Matrix kemampuan IdP

Tidak semua IdP mendukung profil MCP lengkap. Matrix di bawah mendokumentasikan pernyataan kemampuan faktual pada spesifikasi 25-11-2025. Ini adalah *gerbang penerapan*, bukan rekomendasi.

| Kategori IdP | Metadata RFC 8414 | RFC 7591 DCR | Sumber daya RFC 8707 | RFC 7636 S256 PKCE | Catatan |
|---|---|---|---|---|---|
| Dihosting sendiri (Keycloak) | ya | ya | ya (sejak 24.x) | ya | Referensi IdP untuk profil MCP dalam lesson ini; mendukung setiap RFC ujung ke ujung. |
| SSO Perusahaan (Microsoft Entra ID) | ya | ya (tingkatan premium) | ya | ya | Ketersediaan DCR berbeda-beda berdasarkan tingkat penyewa; verifikasi di penyewa target sebelum menerapkan. |
| SSO Perusahaan (Okta) | ya | ya (Okta CIC / Auth0) | ya | ya | DCR tersedia di Auth0 (sekarang Okta CIC); organisasi Okta klasik memerlukan prapendaftaran admin. |
| IdP login sosial (umum) | bervariasi | jarang | jarang | ya | Kebanyakan IdP sosial memperlakukan klien sebagai mitra statis; jangan mengandalkan DCR. Gunakan hanya sebagai sumber identitas, lapisi server otorisasi MCP-aware kamu di atasnya. |
| Custom/buatan sendiri | tergantung | tergantung | tergantung | tergantung | Jika kamu mengirimkan sendiri, kirimkan profil lengkap. Melewatkan salah satu dari empat RFC di atas akan merusak kontrak autentikasi MCP. |

Aturan penolakan untuk manifes penerapan: jika IdP yang dipilih tidak mengembalikan `registration_endpoint` dan tidak mencantumkan `S256` di `code_challenge_methods_supported`, server MCP menolak untuk memulai. Tidak ada mode terdegradasi.

### Pola rotasi JWKS dengan iii

Mode kegagalan produksi adalah cache JWKS yang basi. Selesaikan dengan pemicu cron dan cache `state::*`:

```python
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *", "name": "auth::jwks-refresh"},
    "auth::rotate-jwks",
)
```Setiap enam jam, pemicu cron memanggil `auth::rotate-jwks`, yang mengambil `<issuer>/.well-known/jwks.json` dan menulis ke `state::set("auth/jwks/<issuer>", {keys, fetched_at})`. Validator membaca dari `state::get`. Token yang `kid` hilang dari cache memicu panggilan sinkron `auth::rotate-jwks` sebagai pengganti. Ini menangani dua kasus sekaligus: rotasi terjadwal (cron) dan jendela yang tumpang tindih dengan kunci (fall-back sinkron).

Bentuk negara:

```json
{
  "auth/jwks/https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

Dua kunci sekaligus merupakan kondisi stabil. Server otorisasi dirotasi dengan memasukkan kunci berikutnya (`k_2026_04`) sebelum menghentikan kunci sebelumnya (`k_2026_03`), sehingga token yang diterbitkan dengan kunci lama tetap valid hingga habis masa berlakunya. Cache menampung gabungan; validator memilih oleh `kid`.

### iii pengkabelan primitif (bagian dari lesson ini sebenarnya)

Lima primitif menyusun permukaan autentikasi:

```python
# 1. RFC 8414 metadata document
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591 dynamic client registration
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT validation as a callable function (the resource server triggers it)
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. Step-up issuance for incremental scope (SEP-835 from L16)
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. Cron-driven JWKS rotation
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)
```

Server MCP sendiri tidak pernah memanggil validasi secara langsung. Itu:

```python
result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}
```

Tipuan ini adalah taruhan ketiga. Besok kamu menukar validator dengan fanout yang berkonsultasi dengan dua IdP secara paralel, atau kamu menambahkan emitor rentang, atau kamu menyimpan validasi positif dalam cache. Server MCP tidak berubah.

### Panduan wakil yang bingung dengan pengikatan audiens

Server A (`notes.example.com`) dan Server B (`tasks.example.com`) keduanya mendaftar pada server otorisasi yang sama. Server A disusupi. Penyerang mengambil token catatan pengguna dan memutarnya kembali ke Server B.

Validator Server B:

1. Dekode JWT, ambil JWKS melalui `kid`, verifikasi tanda tangan.
2. Periksa `iss` dengan metadata sumber daya yang dilindungi `authorization_servers`. (Lulus — IdP yang sama.)
3. Periksa `aud == "https://tasks.example.com"`. (Gagal — token `aud` adalah `https://notes.example.com`.)
4. Kembalikan 401 dengan `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch"`.

Klaim audiens adalah satu-satunya pertahanan terhadap serangan di layer protokol ini. Melewatkannya demi performa adalah kesalahan produksi yang paling umum; validator harus dijalankan pada setiap permintaan, tidak hanya pada awal sesi.

### Mode kegagalan

- **JWKS basi.** Validator menolak token yang valid setelah rotasi kunci. Cara mengatasinya adalah pola cron+fall-back di atas. Jangan pernah melakukan cache JWKS tanpa pekerjaan penyegaran.
- **Klaim `aud` tidak ada.** Beberapa IdP secara default menghilangkan `aud` kecuali `resource` ada dalam permintaan token. Validator harus menolak token yang `aud` hilang, tidak memperlakukan ketidakhadiran sebagai wildcard.
- **Perlombaan peningkatan cakupan.** Dua alur peningkatan secara bersamaan untuk pengguna yang sama dapat berhasil dan menghasilkan dua token akses dengan cakupan berbeda. Validator harus menggunakan token yang diberikan pada permintaan, bukan mencari "cakupan pengguna saat ini" — yang menciptakan jendela TOCTOU.
- **Pencurian token pendaftaran.** `registration_access_token` yang bocor memungkinkan penyerang menulis ulang URI pengalihan. Hash ini saat istirahat; mengharuskan klien untuk menyajikan teks yang jelas pada setiap pembaruan; berputar karena curiga.
- **`iss` tidak dipasangi pin.** Validator yang menerima `iss` memungkinkan penyerang membuat server otorisasinya sendiri, mendaftarkan klien untuk audiens target, dan menerbitkan token. Daftar `authorization_servers` metadata sumber daya yang dilindungi adalah daftar yang diizinkan; menegakkannya.

## Pakai`code/main.py` menjalankan alur produksi penuh dengan stdlib Python dan registri kecil `iii_mock` yang meniru `iii.registerFunction`, `iii.registerTrigger`, `iii.trigger`, dan `state::set/get`. Aliran:

1. Server otorisasi menerbitkan metadata RFC 8414 di `/.well-known/oauth-authorization-server`.
2. Klien MCP memanggil titik akhir metadata, menemukan titik akhir pendaftaran.
3. Klien MCP memposting ke `/register` (RFC 7591) dan menerima `client_id`.
4. Klien MCP menjalankan aliran code otorisasi yang dilindungi PKCE (RFC 7636) dengan indikator `resource` (RFC 8707).
5. Klien MCP memanggil alat di server MCP dengan `Authorization: Bearer ...`.
6. Server MCP memicu `auth::validate-jwt`, yang membaca JWKS dari `state::get`.
7. Pemicu cron mengaktifkan `auth::rotate-jwks`, menggantikan status JWKS.
8. Panggilan berikutnya memvalidasi terhadap kunci baru tanpa memulai ulang.
9. Upaya wakil yang bingung terhadap sumber daya MCP yang berbeda mendapat 401 dengan ketidakcocokan audiens.

JWT tiruan di sini menggunakan HS256 dengan rahasia bersama (jadi lesson hanya berjalan di stdlib). Produksi menggunakan RS256 atau EdDSA dengan pola JWKS di atas; logika validasinya identik.

## Kirim

Lesson ini menghasilkan `outputs/skill-mcp-auth-iii.md`. Dengan adanya konfigurasi server MCP dan rangkaian kemampuan IdP, keterampilan tersebut memancarkan iii primitif untuk mendaftar, jadwal rotasi JWKS, pemetaan cakupan, dan aturan penolakan untuk diterapkan ketika IdP tidak mendukung profil RFC lengkap.

## Latihan

1. Jalankan `code/main.py`. Telusuri alur 9 langkah. Perhatikan di mana `state::get` mengembalikan data lama segera sebelum `auth::rotate-jwks` menimpanya, dan bagaimana permintaan berikutnya sekarang divalidasi terhadap kunci baru.

2. Tambahkan IdP baru ke daftar `authorization_servers` metadata sumber daya yang dilindungi. Keluarkan token yang ditandatangani oleh IdP baru dan konfirmasikan validator menerimanya. Terbitkan token yang ditandatangani oleh IdP yang tidak terdaftar dan konfirmasikan penolakan validator dengan `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`.

3. Terapkan `auth::rate-limit` sebagai fungsi iii dan panggil fungsi tersebut dari dalam pemicu HTTP registrasi sebelum registrar dijalankan. Gunakan token-bucket per IP sumber yang disimpan di `state::set("auth/ratelimit/<ip>", ...)`.

4. Baca RFC 7591 dan kenali dua bidang yang tidak divalidasi oleh pengendali `/register` lesson. Tambahkan validasi. (Petunjuk: `software_statement` dan `redirect_uris` skema URI.)

5. Baca bagian otorisasi spesifikasi MCP 25-11-2025. Temukan satu persyaratan normatif pada header `WWW-Authenticate` yang saat ini tidak dikeluarkan oleh validator lesson. Tambahkan itu.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| ASM | "Dokumen metadata OAuth" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| DCR | "Pendaftaran klien swalayan" | RFC 7591 `POST /register` aliran |
| JWKS | "Kunci publik untuk validasi JWT" | Kumpulan Kunci Web JSON, diambil dari `jwks_uri`, diindeks oleh `kid` |
| Indikator sumber daya | "Parameter audiens" | Parameter RFC 8707 `resource` embed token ke satu server |
| `aud` klaim | "Penonton" | JWT mengklaim validator membandingkannya dengan URL sumber daya kanonik |
| Deputi yang bingung | "Pemutaran ulang token" | Serangan dimana token yang dikeluarkan untuk Server A disajikan ke Server B |
| `iss` daftar yang diizinkan | "Server otorisasi tepercaya" | Kumpulan yang diberi nama dalam `authorization_servers` | metadata sumber daya yang dilindungi
| Rotasi kunci | "Bergulir JWKS" | Penggantian kunci penandatanganan secara berkala dengan jendela yang tumpang tindih |
| Klien publik | "Klien asli atau browser" | Klien OAuth tanpa `client_secret`; PKCE memberikan kompensasi |
| `WWW-Authenticate` | "Tajuk respons 401/403" | Membawa arahan `Bearer error=...` yang mendorong pemulihan klien |

## Bacaan Lanjutan

- [MCP — Spesifikasi otorisasi (25-11-2025)](https://modelcontextprotocol.io/spesification/draft/basic/authorization) — profil autentikasi MCP yang diterapkan dalam lesson ini
- [RFC 8414 — Metadata Server Otorisasi OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8414) — kontrak penemuan
- [RFC 7591 — Protokol Pendaftaran Klien Dinamis OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc7591) — DCR
- [RFC 7636 — Kunci Bukti untuk Pertukaran Code (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — bukti kepemilikan klien publik
- [RFC 8707 — Indikator Sumber Daya untuk OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — embedding audiens
- [RFC 9728 — Metadata Sumber Daya yang Dilindungi OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc9728) — penemuan server sumber daya
- [Draf OAuth 2.1](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — substrat OAuth terkonsolidasi
