# Keamanan MCP II — OAuth 2.1, Indikator Sumber Daya, Cakupan Tambahan

> Server MCP distance jauh memerlukan otorisasi, bukan hanya otentikasi. Spesifikasi 25-11-2025 selaras dengan OAuth 2.1 + PKCE + indikator sumber daya (RFC 8707) + metadata sumber daya yang dilindungi (RFC 9728). SEP-835 menambahkan persetujuan cakupan tambahan dengan otorisasi bertahap pada 403 WWW-Authenticate. Lesson ini mengimplementasikan alur step-up sebagai mesin status sehingga kamu dapat melihat setiap lompatan.

**Type:** Build
**Language:** Python (stdlib, simulator mesin status OAuth)
**Prerequisites:** Phase 13 · 09 (transportasi), Phase 13 · 15 (keamanan I)
**Waktu:** ~75 menit

## Tujuan Pembelajaran

- Bedakan server sumber daya dari tanggung jawab server otorisasi.
- Jalani alur code otorisasi OAuth 2.1 yang dilindungi PKCE.
- Gunakan `resource` (RFC 8707) dan metadata sumber daya yang dilindungi (RFC 9728) untuk mencegah serangan deputi yang membingungkan.
- Menerapkan otorisasi bertahap: server merespons 403 dengan WWW-Authenticate yang meminta cakupan yang lebih tinggi; klien meminta kembali izin pengguna dan mencoba lagi.

## Masalah

MCP awal (sebelum tahun 2025) mengirimkan server distance jauh dengan kunci API ad-hoc atau bahkan tanpa autentikasi. Spesifikasi 25-11-2025 menutup kesenjangan tersebut dengan profil OAuth 2.1 lengkap.

Tiga kebutuhan dunia nyata:

- **Server distance jauh biasa.** Pengguna memasang server MCP distance jauh yang mengakses Notion / GitHub / Gmail mereka. OAuth 2.1 dengan PKCE adalah bentuk yang tepat.
- **Eskalasi cakupan.** Server catatan yang diberikan `notes:read` nantinya dapat memerlukan `notes:write` untuk tindakan tertentu. Daripada mengulangi seluruh alur, step-up (SEP-835) meminta cakupan tambahan.
- **Deputi pencegahan yang bingung.** Klien memegang token dalam cakupan audiens untuk Server A. Server A berbahaya dan mencoba menyajikan token ke Server B. Indikator sumber daya (RFC 8707) embed token ke audiens yang dituju.

OAuth 2.1 bukanlah hal baru. Yang baru adalah profil MCP: alur spesifik yang diperlukan (code otorisasi + PKCE saja; tidak ada implisit, tidak ada kredensial klien secara default), indikator sumber daya wajib pada setiap permintaan token, dan metadata sumber daya dilindungi dipublikasikan sehingga klien tahu ke mana harus pergi.

## Konsep

### Peran

- **Klien.** Klien MCP (Claude Desktop, Kursor, dll.).
- **Server sumber daya.** Server MCP (catatan, GitHub, Postgres, apa pun).
- **Server otorisasi.** Mengeluarkan token. Mungkin layanan yang sama dengan server sumber daya atau IdP terpisah (Auth0, Keycloak, Cognito).

Dalam profil MCP, server sumber daya dan otorisasi BISA menjadi host yang sama tetapi HARUS dibedakan berdasarkan URL.

### Code otorisasi + PKCE

Aliran:

1. Klien menghasilkan `code_verifier` (acak) dan `code_challenge` (SHA256).
2. Klien mengalihkan pengguna ke `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`.
3. Persetujuan pengguna. Server otorisasi dialihkan ke `redirect_uri?code=...`.
4. Klien POST ke `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`.
5. Server otorisasi memvalidasi hash pemverifikasi terhadap tantangan yang disimpan dan mengeluarkan token akses.
6. Klien menggunakan token: `Authorization: Bearer ...` pada setiap permintaan ke server sumber daya.

PKCE mencegah serangan intersepsi code otorisasi. Indikator sumber daya mencegah token menjadi valid di tempat lain.

### Metadata sumber daya yang dilindungi (RFC 9728)

Server sumber daya menerbitkan dokumen `.well-known/oauth-protected-resource`:

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

Klien menemukan server otorisasi dari server sumber daya. Mengurangi konfigurasi — klien hanya memerlukan URL sumber daya.

### Indikator sumber daya (RFC 8707)Parameter `resource` dalam permintaan token embed audiens yang dituju token. Token yang diterbitkan berisi `aud: "https://notes.example.com"`. Server MCP lain yang menerima token ini memeriksa `aud` dan menolaknya.

### Model cakupan

Cakupan adalah string yang dipisahkan oleh ruang. Konvensi umum MCP:

- `notes:read`, `notes:write`, `notes:delete`
- `admin:*` untuk kemampuan admin (gunakan dengan hemat)
- `profile:read` untuk identitas

Pemilihan cakupan harus memiliki hak istimewa yang paling rendah: mintalah apa yang kamu butuhkan sekarang, majulah ketika kamu membutuhkan lebih banyak.

### Otorisasi bertahap (SEP-835)

Hibah pengguna `notes:read`. Mereka kemudian meminta agen untuk menghapus catatan tersebut. Server merespons:

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

Klien melihat kesalahan tidak mencukupi_scope, meminta pengguna dengan dialog persetujuan untuk cakupan tambahan, melakukan alur OAuth mini untuk cakupan tersebut, mencoba ulang permintaan dengan token baru.

### Validasi audiens token

Setiap permintaan: server memeriksa `token.aud == self.resource_url`. Ketidakcocokan = 401. Ini menghentikan penggunaan kembali token lintas server.

### Token dan rotasi berumur pendek

Token akses HARUS berumur pendek (default 1 jam). Token penyegaran diputar pada setiap penyegaran. Klien menangani penyegaran senyap di latar belakang.

### Tidak ada jalur token

Server pengambilan sample (Fase 13 · 11) TIDAK BOLEH meneruskan token klien ke layanan lain. Permintaan pengambilan sample adalah batasannya.

### Deputi pencegahan yang bingung

Token terikat ke `aud`. Klien terikat ke `client_id`. Setiap permintaan divalidasi terhadap keduanya. Spesifikasi tersebut secara eksplisit melarang pola lama "pass-the-token" yang umum terjadi di ekosistem alat distance jauh sebelum MCP.

### Penemuan ID Klien

Setiap klien MCP menerbitkan metadatanya pada URL tetap. Server otorisasi dapat mengambil dokumen metadata klien untuk menemukan URI pengalihan dan informasi kontak. Ini menghapus registrasi klien manual.

### Gateway dan OAuth

Fase 13 · 17 menunjukkan bagaimana gateway perusahaan menangani OAuth: gateway menyimpan kredensial untuk server upstream, token ke klien dikeluarkan oleh gateway, dan token upstream tidak pernah meninggalkan gateway. Hal ini membalikkan model kepercayaan — pengguna mengautentikasi dengan gateway satu kali; gateway menangani otorisasi server N.

## Pakai

`code/main.py` menyimulasikan alur peningkatan OAuth 2.1 penuh sebagai mesin negara. Ini mengimplementasikan:

- Pemverifikasi code / pembuatan tantangan PKCE.
- Aliran code otorisasi dengan indikator sumber daya.
- Titik akhir metadata sumber daya yang dilindungi.
- Validasi token dengan pemeriksaan audiens.
- Naik ke `insufficient_scope`.

Tidak ada server HTTP dalam lesson ini; mesin negara berjalan di memori sehingga kamu dapat melacak setiap lompatan. Lesson gerbang Fase 13 · 17 menghubungkannya ke transportasi sebenarnya.

## Kirim

Lesson ini menghasilkan `outputs/skill-oauth-scope-planner.md`. Dengan adanya server MCP distance jauh yang dilengkapi alat, keterampilan merancang kumpulan cakupan, embed aturan, dan kebijakan peningkatan.

## Latihan

1. Jalankan `code/main.py`. Telusuri alur peningkatan dua cakupan. Perhatikan lompatan mana yang berulang pada saat step-up.

2. Tambahkan rotasi token penyegaran: setiap penyegaran mengeluarkan token penyegaran baru dan membatalkan yang lama. Simulasikan token penyegaran yang dicuri yang digunakan setelah rotasi dan konfirmasikan gagal.

3. Menerapkan titik akhir metadata sumber daya yang dilindungi sebagai respons HTTP nyata menggunakan stdlib http.server. Cerminkan titik akhir /mcp dari Lesson 09.

4. Rancang hierarki cakupan untuk server GitHub MCP: baca repo, tulis PR, setujui PR, gabungkan PR, admin. Gunakan step-up di antara setiap level.5. Baca RFC 8707 dan RFC 9728. Identifikasi satu bidang di 9728 yang digunakan MCP secara berbeda dari contoh RFC. (Petunjuk: ini menyangkut `scopes_supported`.)

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|------------------------|
| OAuth 2.1 | "OAuth Modern" | RFC terkonsolidasi yang mengamanatkan PKCE dan melarang aliran implisit |
| PKCE | "Bukti kepemilikan" | Pemverifikasi code + tantangan mengalahkan intersepsi code otorisasi |
| Indikator sumber daya | "Pemirsa Token" | RFC 8707 `resource` parameter embed token ke satu server |
| Metadata sumber daya yang dilindungi | "Dokumen penemuan" | RFC 9728 `.well-known/oauth-protected-resource` |
| Otorisasi bertahap | "Persetujuan tambahan" | Alur SEP-835 untuk menambahkan cakupan sesuai permintaan |
| `insufficient_scope` | "403 dengan WWW-Otentikasi" | Sinyal server untuk menyetujui kembali cakupan yang lebih besar |
| Deputi yang bingung | "Penggunaan kembali token di seluruh layanan" | Menyerang ketika pemegang tepercaya meneruskan token secara tidak tepat |
| Token berumur pendek | "Akses token TTL" | Pembawa yang masa berlakunya cepat habis; penyegaran token diperbarui |
| Hierarki ruang lingkup | "Tumpukan hak istimewa paling rendah" | Lingkup kelulusan diatur dengan peningkatan antar level |
| Metadata ID Klien | "Dokumen penemuan klien" | URL tempat klien memublikasikan metadata OAuth miliknya |

## Bacaan Lanjutan

- [MCP — Spesifikasi otorisasi](https://modelcontextprotocol.io/spesification/draft/basic/authorization) — profil OAuth MCP kanonik
- [den.dev — spesifikasi otorisasi MCP November](https://den.dev/blog/mcp-november-authorization-spec/) — panduan perubahan 25-11-2025
- [RFC 8707 — Indikator sumber daya untuk OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — RFC yang embed audiens
- [RFC 9728 — metadata sumber daya yang dilindungi OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc9728) — dokumen penemuan RFC
- [Aembit — MCP OAuth 2.1, PKCE, dan masa depan otorisasi AI](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — panduan alur langkah praktis
