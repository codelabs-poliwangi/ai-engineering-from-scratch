# Capstone 06 — Agen Pemecahan Masalah DevOps untuk Kubernetes

> Agen DevOps AWS menggunakan GA, Resolve AI menerbitkan buku pedoman K8, NeuBird mendemonstrasikan pemantauan semantik, dan Metoro mengikat AI SRE ke SLO per layanan. Bentuk produksi telah diselesaikan: webhook peringatan diaktifkan, agen membaca telemetri, menelusuri grafik objek K8, membuat peringkat hipotesis akar penyebab, dan memposting ringkasan Slack dengan tombol persetujuan. Hanya baca secara default. Setiap remediasi dijaga oleh manusia. Batu penjuru ini adalah agen tersebut, yang dievaluasi pada 20 insiden sintetis dan dibandingkan dengan Agen AWS pada tiga kasus bersama.

**Type:** Batu penjuru
**Language:** Python (agen), TypeScript (integrasi Slack)
**Prerequisites:** Fase 11 (rekayasa LLM), Fase 13 (peralatan dan MCP), Fase 14 (agen), Fase 15 (otonom), Fase 17 (infrastruktur), Fase 18 (keselamatan)
**Fase yang dilakukan:** P11 · P13 · P14 · P15 · P17 · P18
**Waktu:** 30 jam

## Masalah

Narasi SRE 2025-2026 menjadi: "Agen AI melakukan triase terhadap insiden, manusia menyetujui remediasi." Agen AWS DevOps, Resolve AI, NeuBird, Metoro, PagerDuty AIOps semuanya mengirimkan bentuk ini dalam produksi. Agen membaca metrik Prometheus, log Loki, jejak Tempo, metrik kube-state, dan grafik pengetahuan objek K8. Ini menghasilkan hipotesis akar penyebab dengan kutipan telemetri dalam waktu kurang dari lima menit. Itu tidak pernah menjalankan prompt destruktif tanpa persetujuan manusia secara eksplisit melalui Slack.

Sebagian besar kerja kerasnya adalah pelingkupan dan keamanan, bukan penalaran. Agen memerlukan permukaan RBAC yang hanya dapat dibaca secara default, server alat MCP yang diperkuat, dan log audit dari setiap prompt yang dipertimbangkan vs dijalankan. Ia perlu mengetahui kapan ia berada di luar kedalamannya dan meningkat. Dan itu harus dijalankan dengan cukup murah sehingga rangkaian pembunuhan OOM tidak menghasilkan tagihan agen sebesar $5k.

## Konsep

Agen beroperasi pada grafik pengetahuan. Node adalah objek K8 (Pod, Deployment, Layanan, Node, HPA, PVC) ditambah sumber telemetri (seri Prometheus, aliran Loki, jejak Tempo). Edge mengkodekan kepemilikan (Pod -> ReplicaSet -> Deployment), penjadwalan (Pod -> Node), dan observasi (Pod -> seri Prometheus). Grafik ini selalu diperbarui dengan sinkronisasi metrik kube-state dan pengambilan sample ulang pada setiap peringatan.

Saat peringatan menyala, agen melakukan akar penyebab dari objek yang terpengaruh. Ia menelusuri tepian, menarik potongan telemetri yang relevan (15 menit terakhir), dan menyusun hipotesis. Hipotesis ini diberi peringkat berdasarkan bukti: berapa banyak kutipan telemetri yang mendukungnya, seberapa terkini, seberapa spesifik. 3 hipotesis teratas masuk ke Slack dengan visualisasi jalur grafik dan tombol persetujuan untuk tindakan remediasi.

Remediasi bersifat gated. Tindakan default yang diperbolehkan bersifat hanya baca. Tindakan destruktif (menurunkan skala, memutar kembali, menghapus Pod) memerlukan persetujuan Slack; Kait rollback ArgoCD memerlukan token autentikasi yang tidak pernah dimiliki agen. Log audit mencatat setiap prompt yang *dipertimbangkan* oleh agen — tidak hanya dijalankan — sehingga proses peninjauan hampir menemukan kesalahan.

## Arsitektur

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## Tumpukan- Sumber observasi: Prometheus, Loki, Tempo, kube-state-metrics
- Grafik pengetahuan: Neo4j (dikelola) atau kuzu (tertanam) objek K8s + tepi telemetri
- Agen: LangGraph dengan daftar izin per alat, hanya baca secara default
- Transportasi alat: FastMCP melalui StreamableHTTP; server terpisah untuk alat perusak di belakang gerbang persetujuan
- Model: Claude Sonnet 4.7 untuk penalaran akar permasalahan, Gemini 2.5 Flash untuk ringkasan log
- Remediasi: Webhook rollback ArgoCD, eskalasi PagerDuty, kartu persetujuan Slack
- Audit: log terstruktur khusus tambahan (dipertimbangkan, dilaksanakan, disetujui, hasil)
- Penerapan: penerapan K8 dengan peran RBAC yang sempit; ruang nama terpisah

## Build

1. **Penyerapan grafik.** Sinkronkan kube-state-metrics ke Neo4j/kuzu setiap 30 detik. Node: Pod, Deployment, Node, Layanan, PVC, HPA. Tepian: OWNED_BY, SCHEDULED_ON, EXPOSES, MOUNTS, SKALA. Tepi hamparan telemetri: OBSERVED_BY (Sebuah Pod diamati oleh rangkaian Prometheus).

2. **Penerima peringatan.** Titik akhir FastAPI yang menerima webhook PagerDuty atau Alertmanager. Ekstrak objek yang terpengaruh dan pelanggaran SLO.

3. **Permukaan alat hanya-baca.** Bungkus kubectl, kueri Prometheus, Loki logql, Tempo traceql melalui FastMCP. Setiap alat memiliki kata kerja RBAC yang sempit ("dapatkan", "daftar", "jelaskan"). Tidak ada "hapus", "exec", "skala" di server default.

4. **Agen akar penyebab.** LangGraph dengan tiga node: `sample` mengambil potongan telemetri 15 menit terakhir, `walk` mengkueri grafik untuk objek di sekitarnya, `hypothesize` draf memberi peringkat kandidat akar penyebab dengan kutipan telemetri.

5. **Penilaian bukti.** Setiap hipotesis memiliki skor = keterkinian * kekhususan * inverse panjang jalur grafik * jumlah kutipan. Kembali ke-3 teratas.

6. **Slack brief.** Posting lampiran dengan hipotesis, visualisasi jalur grafik (gambar subgraf yang dirender di sisi server), dan tombol persetujuan untuk maksimal satu tindakan remediasi.

7. **Gerbang remediasi.** Alat perusak (menurunkan skala, memutar kembali, menghapus) ada di server MCP kedua di belakang token persetujuan. Agen dapat menghubungi mereka hanya setelah kartu Slack disetujui oleh manusia.

8. **Log audit.** JSONL khusus tambahan: untuk setiap prompt kandidat, catat apakah prompt tersebut dipertimbangkan, apakah dijalankan, siapa yang menyetujuinya. Kirim ke S3 setiap hari.

9. **Rangkaian insiden sintetik.** Buat 20 skenario: OOMKill cascade, DNS flap, HPA thrash, PVC fill, berisik tetangga, sidecar rusak, peluncuran ConfigMap buruk, rotasi sertifikat, image-pull backoff, dll. Beri skor pada agen berdasarkan keakuratan akar penyebab dan waktu pembuatan hipotesis.

## Pakai

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## Kirim

`outputs/skill-devops-agent.md` adalah hasil yang dapat dicapai. Dengan adanya klaster K8 dan sumber peringatan, agen menghasilkan hipotesis akar penyebab yang diberi peringkat dan alur remediasi dengan gerbang Slack.

| Berat | Kriteria | Bagaimana cara mengukurnya |
|:-:|---|---|
| 25 | Akurasi RCA pada rangkaian skenario | ≥80% akar permasalahan yang benar pada 20 insiden sintetis |
| 20 | Keamanan | Penjaga tindakan destruktif tidak pernah diaktifkan tanpa persetujuan Slack di log audit |
| 20 | Waktu-untuk-hipotesis | p50 kurang dari 5 menit dari peringatan ke Slack brief |
| 20 | Penjelasan | Setiap hipotesis memiliki jalur grafik dan kutipan telemetri |
| 15 | Kelengkapan integrasi | PagerDuty, Slack, ArgoCD, Prometheus berfungsi secara menyeluruh |
| **100** | | |

## Latihan

1. Jalankan agen kamu pada tiga insiden yang sama dengan Demo Agen DevOps AWS. Publikasikan secara berdampingan. Laporkan di mana agen menyimpang.2. Tambahkan audit "nyaris celaka" yang menandai prompt apa pun yang *dianggap* oleh agen yang dapat merusak tanpa persetujuan. Ukur tingkat kejadian nyaris celaka selama satu minggu.

3. Tukar model hipotesis dari Claude Sonnet 4.7 ke Llama 3.3 70B yang dihosting sendiri. Ukur akurasi RCA delta dan dolar per insiden.

4. Membangun filter sebab akibat: membedakan lonjakan telemetri yang berkorelasi dari akar permasalahan sebenarnya. Latih pengklasifikasi kecil pada label 20 skenario.

5. Tambahkan rollback dry-run: ArgoCD rollback terhadap staging cluster dengan manifes yang sama. Verifikasi rencana rollback dalam klaster langsung sebelum tombol persetujuan Slack.

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|------------------------|
| Grafik pengetahuan K8 | "Grafik cluster" | Node = objek K8 + rangkaian telemetri; edge = kepemilikan, penjadwalan, observasi |
| Baca-saja-secara-default | "RBAC Cakupan" | Akun layanan agen hanya memiliki kata kerja get/list/deskripsikan; kata kerja destruktif tinggal di server terpisah di belakang persetujuan |
| Catatan audit | "Dianggap vs dieksekusi" | Append-only record dari setiap calon komando, baik yang mencalonkan, siapa yang menyetujui |
| Pemeringkatan hipotesis | "Skor bukti" | Kekinian × kekhususan × inverse panjang jalur grafik × jumlah kutipan |
| Kartu persetujuan kendur | "Gerbang HITL" | Pesan Slack interaktif dengan tombol remediasi; agen tidak dapat melanjutkan sampai manusia mengklik |
| Kutipan telemetri | "Penunjuk bukti" | Kueri Prometheus, pemilih Loki, atau URL jejak Tempo yang mendukung klaim |
| MTTR | "Waktunya untuk resolusi" | Jam dinding dari peringatan kebakaran hingga pemulihan SLO |

## Bacaan Lanjutan

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — referensi kanonikal tahun 2026
- [Menyelesaikan pemecahan masalah AI K8](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — referensi pesaing
- [Pemantauan semantik NeuBird](https://www.neubird.ai) — pendekatan grafik semantik
- [Metoro AI SRE](https://metoro.io) — Pembingkaian produksi pertama SLO
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — sumber status cluster
- [LangGraph](https://langchain-ai.github.io/langgraph/) — orkestrator agen referensi
- [FastMCP](https://github.com/jlowin/fastmcp) — Kerangka server Python MCP
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — target remediasi yang terjaga keamanannya
