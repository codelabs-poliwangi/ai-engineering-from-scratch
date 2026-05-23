# Mengapa Multi-Agen?

> Salah satu agen menabrak tembok. Langkah cerdas bukanlah agen yang lebih besar - melainkan lebih banyak agen.

**Type:** Learn
**Language:** TypeScript
**Prerequisites:** Fase 14 (Rekayasa Agen)
**Waktu:** ~60 menit

## Tujuan Pembelajaran

- Identifikasi batas maksimum agen tunggal (limpahan konteks, keahlian campuran, kemacetan berurutan) dan jelaskan kapan pemisahan menjadi beberapa agen adalah langkah yang tepat
- Bandingkan pola orkestrasi (pipa, fan-out paralel, supervisor, hierarki) dan pilih yang tepat untuk struktur tugas tertentu
- Rancang sistem multi-agen dengan batasan peran yang jelas, status bersama, dan kontrak komunikasi
- Menganalisis trade-off kompleksitas multi-agen (latensi, biaya, kesulitan debugging) versus kesederhanaan agen tunggal

## Masalah

kamu membuat agen tunggal di Fase 14. Berhasil. Itu dapat membaca file, menjalankan prompt, memanggil API, dan memikirkan hasil. Kemudian kamu mengarahkannya ke basis code sebenarnya: 200 file, tiga bahasa, pengujian yang bergantung pada infrastruktur, dan persyaratan untuk meneliti API eksternal sebelum menulis code.

Agen itu tersedak. Bukan karena LLM itu bodoh, tapi karena tugasnya melebihi apa yang bisa ditangani oleh satu loop agen. Jendela konteks terisi dengan konten file. Agen lupa apa yang dibacanya 40 panggilan alat yang lalu. Ia mencoba menjadi peneliti, pembuat code, dan peninjau sekaligus, dan melakukan ketiganya dengan buruk.

Ini adalah batas maksimum agen tunggal. kamu menekannya setiap kali tugas memerlukan:

- **Lebih banyak konteks daripada yang muat dalam satu jendela** - membaca 50 file melampaui 200 ribu token
- **Keahlian yang berbeda pada tahapan yang berbeda** - penelitian memerlukan dorongan yang berbeda dari pembuatan code
- **Pekerjaan yang dapat dilakukan secara paralel** - mengapa membaca tiga file secara berurutan jika kamu dapat membacanya secara bersamaan?

## Konsep

### Batasan Agen Tunggal

Agen tunggal adalah satu loop, satu jendela konteks, satu prompt sistem. Bayangkan:

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

Tiga hal yang rusak:

1. **Saturasi konteks** - hasil alat menumpuk. Pada usia 30, agen telah menggunakan 150 ribu token konten file, output prompt, dan alasan sebelumnya. Detail penting dari tikungan 5 hilang.

2. **Perplexity peran** - system prompt yang menyatakan "kamu adalah peneliti, pembuat code, peninjau, dan penguji" menghasilkan agen yang setengah meneliti, setengah membuat code, dan tidak pernah menyelesaikan peninjauan.

3. **Kemacetan berurutan** - agen membaca file A, lalu file B, lalu file C. Tiga panggilan LLM serial. Tiga eksekusi alat serial. Tidak ada paralelisme.

### Solusi Multi-Agen

Pisahkan pekerjaan. Berikan setiap agen satu pekerjaan, satu jendela konteks, dan satu system prompt yang disesuaikan untuk pekerjaan tersebut:

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

Setiap agen memiliki:
- System prompt yang terfokus ("kamu adalah peninjau code. Satu-satunya tugas kamu adalah menemukan bug.")
- Jendela konteksnya sendiri (tidak tercemar oleh pekerjaan agen lain)
- Kontrak input/output yang jelas (menerima catatan penelitian, code output)

### Sistem Nyata yang Melakukan Ini

**Subagen Code Claude** - saat Code Claude memunculkan subagen dengan `Task`, agen turunan dengan tugas tercakup akan dibuat. Orang tua menjaga konteksnya tetap bersih. Anak tersebut melakukan pekerjaan terfokus dan mengembalikan ringkasan.

**Devin** - menjalankan agen perencana, agen pembuat code, dan agen browser. Perencana membagi pekerjaan menjadi beberapa langkah. Pembuat code menulis code. Browser meneliti dokumentasi. Masing-masing mempunyai konteks tersendiri.**Tim pengkodean multi-agen (SWE-bench)** - sistem berkinerja terbaik di SWE-bench menggunakan peneliti yang membaca basis code, perencana yang merancang perbaikan, dan pembuat code yang mengimplementasikannya. Skor sistem agen tunggal lebih rendah.

**ChatGPT Deep Research** - memunculkan beberapa agen pencarian secara paralel, masing-masing mengeksplorasi sudut pandang yang berbeda, lalu menyatukan hasilnya.

### Spektrum

Multi-agen bukan biner. Ini adalah spektrum:

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**Agen tunggal** - satu putaran, satu prompt. Bagus untuk tugas sederhana.

**Subagen** - orang tua memunculkan anak untuk subtugas terfokus. Orang tua mempertahankan rencananya. Anak-anak melapor kembali. Inilah yang dilakukan Claude Code.

**Pipeline** - agen dijalankan secara berurutan. Output Agen A menjadi input Agen B. Cocok untuk alur kerja bertahap: penelitian -> code -> ulasan -> pengujian.

**Tim** - agen berjalan secara paralel dengan bus pesan bersama. Masing-masing punya peran. Seorang orkestra berkoordinasi. Bagus bila keterampilan yang berbeda dibutuhkan secara bersamaan.

**Swarm** - banyak agen yang identik atau hampir identik dengan status bersama. Tidak ada orkestra tetap. Agen mengambil pekerjaan dari antrian. Cocok untuk tugas paralel dengan throughput tinggi.

### Empat Pola Multi-Agen

#### Pola 1: Pipeline Pipa

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

Setiap agen mengubah data dan meneruskannya. Sederhana untuk dipikirkan. Kegagalan dalam satu phase menghalangi phase lainnya.

#### Pola 2: Fan-out / Fan-in

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

Pisahkan pekerjaan di seluruh agen paralel, lalu gabungkan hasilnya. Cocok untuk tugas yang dipecah menjadi subtugas independen.

#### Pola 3: Pekerja Orkestra

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

Seorang orkestra yang cerdas memutuskan apa yang harus dilakukan, mendelegasikannya kepada pekerja, dan menyatukan hasil. Orkestra itu sendiri adalah agen yang memiliki alat untuk memunculkan pekerja.

#### Pola 4: Kawanan Sejawat

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

Tidak ada orkestra pusat. Agen berkomunikasi secara peer-to-peer. Keputusan muncul dari interaksi. Lebih sulit untuk di-debug, tetapi dapat diperluas ke banyak agen.

### Kapan TIDAK Menggunakan Multi-Agen

Multi-agen menambah kompleksitas. Setiap pesan antar agen merupakan titik kegagalan potensial. Proses debug berubah dari "membaca satu percakapan" menjadi "melacak pesan di lima agen".

**Tetap menjadi agen tunggal ketika:**
- Tugas muat dalam satu jendela konteks (di bawah ~100 ribu token data kerja)
- kamu tidak memerlukan system prompt yang berbeda untuk tahapan yang berbeda
- Eksekusi berurutan cukup cepat
- Tugasnya cukup sederhana sehingga pemisahannya menambah lebih banyak overhead daripada nilai

**Biaya kerumitan:**
- Setiap batasan agen merupakan langkah kompresi yang lossy: konteks lengkap agen A diringkas menjadi pesan untuk agen B
- Logika koordinasi (siapa melakukan apa, kapan, dalam urutan apa) merupakan sumber bugnya sendiri
- Latensi meningkat: N agen berarti minimum N panggilan serial LLM, lebih banyak jika mereka perlu berbicara bolak-balik
- Biaya berlipat ganda: setiap agen membakar token secara independen

Aturan praktisnya: jika suatu tugas memerlukan kurang dari 20 panggilan alat dan dapat memuat 100 ribu token, pertahankan agar tetap menjadi agen tunggal.

## Build

### Langkah 1: Agen Tunggal yang Kelebihan Weight

Inilah agen tunggal yang mencoba melakukan segalanya. Ini memiliki satu sistem prompt besar dan satu jendela konteks yang menampung penelitian, code, dan ulasan:

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Masalah dengan pendekatan ini:
- Jendela konteks berkembang seiring dengan setiap tahapan. Pada langkah review berisi catatan penelitian DAN code DAN alasan sebelumnya.
- System prompt bersifat umum. Itu tidak dapat disetel untuk setiap phase.
- Tidak ada yang berjalan secara paralel.

### Langkah 2: Agen Spesialis

Sekarang bagilah. Setiap agen mendapat satu pekerjaan:

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```Setiap spesialis memiliki prompt yang terfokus. Masing-masing mendapat jendela konteks bersih dengan hanya input yang dibutuhkan.

### Langkah 3: Koordinasikan Melalui Pesan

Hubungkan para spesialis dengan penyampaian pesan eksplisit:

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Setiap agen hanya menerima pesan yang ditujukan kepadanya. Tidak ada polusi konteks. 50 ribu token pembacaan dokumentasi milik peneliti tidak pernah masuk ke dalam konteks pengulas.

### Langkah 4: Bandingkan

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

Versi multi-agen menggunakan lebih banyak token total (tiga agen, tiga panggilan LLM terpisah) namun konteks masing-masing agen tetap bersih. Kualitas setiap phase meningkat karena system prompt dikhususkan.

## Pakai

Lesson ini menghasilkan prompt yang dapat digunakan kembali untuk memutuskan kapan harus beralih ke multi-agen. Lihat `outputs/prompt-multi-agent-decision.md`.

## Latihan

1. Tambahkan spesialis keempat: agen "penguji" yang menerima code dari pembuat code dan meninjau input dari peninjau, lalu menulis pengujian
2. Memodifikasi alur sehingga peninjau dapat mengirimkan umpan balik kembali ke pembuat code untuk putaran revisi (maks 2 putaran)
3. Ubah pipeline sekuensial menjadi fan-out: jalankan peneliti dan agen "penganalisis persyaratan" secara paralel, lalu gabungkan outputnya sebelum diteruskan ke pembuat code

## Istilah Kunci

| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|----------------|----------------------|
| Kawanan | "Sarang agen AI" | Sekumpulan agen sejawat dengan status bersama dan tidak ada pemimpin tetap. Perilaku muncul dari interaksi lokal. |
| Orkestrator | "Agen bos" | Agen yang alatnya mencakup pemijahan dan pengelolaan agen lainnya. Ia merencanakan dan mendelegasikan tetapi mungkin tidak melakukan pekerjaan sebenarnya. |
| Koordinator | "Polisi lalu lintas" | Komponen non-agen (seringkali hanya code, bukan LLM) yang merutekan pesan antar agen berdasarkan aturan. |
| Konsensus | "Agen setuju" | Sebuah protokol di mana banyak agen harus mencapai kesepakatan sebelum melanjutkan. Digunakan ketika output yang bertentangan memerlukan penyelesaian. |
| Perilaku yang muncul | "Para agen menemukan jawabannya sendiri" | Pola tingkat sistem yang muncul dari interaksi agen tetapi tidak diprogram secara eksplisit. Bisa bermanfaat atau merugikan. |
| Fan-out / fan-in | "Pengurangan peta untuk agen" | Membagi tugas di seluruh agen paralel (fan-out), lalu menggabungkan hasilnya (fan-in). |
| Pesan lewat | "Agen berbicara satu sama lain" | Mekanisme komunikasi antar agen: data terstruktur dikirim dari satu agen ke agen lainnya, menggantikan jendela konteks bersama. |

## Bacaan Lanjutan

- [Lanskap Arsitektur Agen AI yang Berkembang](https://arxiv.org/abs/2409.02977) - survei pola multi-agen
- [AutoGen: Mengaktifkan Aplikasi LLM Generasi Berikutnya](https://arxiv.org/abs/2308.08155) - Kerangka kerja percakapan multi-agen Microsoft
- [Dokumentasi subagen Claude Code](https://docs.anthropic.com/en/docs/claude-code) - cara Claude Code mendelegasikan Tugas
- [Dokumentasi CrewAI](https://docs.crewai.com/) - framework multi-agen berbasis peran
