# LangGraph — Mesin Negara untuk Agen

> Loop ReAct yang ditulis dengan tangan adalah `while True`. Perulangan ReAct yang ditulis dalam LangGraph adalah grafik yang dapat kamu periksa, interupsi, percabangan, dan jelajahi waktu. Agennya tidak berubah. Tali pengaman di sekelilingnya punya.

**Type:** Build
**Language:** Python
**Prerequisites:** Fase 11 · 09 (Pemanggilan Fungsi), Fase 11 · 14 (Protokol Konteks Model)
**Waktu:** ~75 menit

## Masalah

kamu mengirimkan agen pemanggil fungsi. Ini berfungsi selama tiga putaran, lalu terjadi kesalahan: model mencoba alat yang menghasilkan 500, pengguna berubah pikiran di tengah tugas, atau agen memutuskan untuk mengembalikan dana pesanan tanpa ada manusia yang menandatanganinya. Loop `while True:` tidak memiliki kait. kamu tidak dapat menjedanya, tidak dapat memundurkannya, dan kamu tidak dapat melanjutkan ke "bagaimana jika model memilih alat yang lain". Saat kamu mengirimkan demo ini, agen menjadi kotak hitam yang berfungsi atau tidak.

Langkah selanjutnya sudah jelas setelah kamu melihatnya. Agen sudah menjadi mesin negara — system prompt ditambah riwayat pesan ditambah panggilan alat yang tertunda ditambah tindakan selanjutnya. Jadikan mesin negara eksplisit: node untuk "model berpikir", "alat berjalan", "manusia menyetujui", dan edge untuk transisi bersyarat di antara node tersebut. Setelah grafiknya eksplisit, harness mendapatkan empat hal secara gratis: checkpointing (menyimpan status di antara langkah-langkah), interupsi (menjeda untuk manusia), streaming (streaming token dan peristiwa perantara), dan perjalanan waktu (mundur ke keadaan sebelumnya dan mencoba cabang yang berbeda).

LangGraph adalah perpustakaan yang mengirimkan abstraksi ini. Ini bukan kerangka agen dalam pengertian LangChain ("ini adalah AgentExecutor, semoga berhasil"). Ini adalah waktu proses grafik dengan status kelas satu, persistensi kelas satu, dan interupsi kelas satu. Lingkaran agen adalah sesuatu yang kamu gambar, bukan sesuatu yang kamu tulis tangan.

## Konsep

![LangGraph StateGraph: node, edge, dan checkpointer](../assets/langgraph-stategraph.svg)

`StateGraph` memiliki tiga hal.

1. **State.** Dict yang diketik (model TypedDict atau Pydantic) yang mengalir melalui grafik. Setiap node menerima status penuh dan mengembalikan pembaruan sebagian, yang digabungkan LangGraph menggunakan *peredam* per bidang — `operator.add` untuk daftar yang harus terakumulasi, ditimpa secara default.
2. **Node.** Fungsi Python `state -> partial_state`. Masing-masing merupakan langkah tersendiri: "panggil model", "jalankan alat", "ringkas".
3. **Tepi.** Transisi antar node. Tepi statis berada di satu tempat. Tepi bersyarat mengambil fungsi router `state -> next_node_name` sehingga grafik dapat bercabang pada output model.

kamu menyusun grafiknya. Kompilasi mengikat topologi, melampirkan checkpointer (opsional tetapi penting untuk produksi), dan mengembalikan runnable. kamu memanggilnya dengan status awal dan `thread_id`. Setiap langkah eksekusi tetap memiliki pos pemeriksaan yang dikunci pada `(thread_id, checkpoint_id)`.

### Empat negara adidaya

**Checkpointing.** Setiap transisi node menulis status baru ke penyimpanan (dalam memori untuk pengujian, Postgres/Redis/SQLite untuk produksi). Lanjutkan dengan memanggil grafik lagi dengan `thread_id` yang sama. Grafik melanjutkan saat jeda.

**Interupsi.** Tandai node dengan `interrupt_before=["human_review"]` dan eksekusi berhenti sebelum node tersebut berjalan. Negara tetap bertahan. API kamu merespons pengguna dengan "menunggu persetujuan". Permintaan berikutnya ke `thread_id` yang sama dengan `Command(resume=...)` akan melanjutkan eksekusi.**Streaming.** `graph.stream(state, mode="updates")` menghasilkan delta status saat terjadi. `mode="messages"` mengalirkan token LLM di dalam node model. `mode="values"` menghasilkan cuplikan lengkap. kamu memilih apa yang akan ditampilkan di UI kamu.

**Perjalanan waktu.** `graph.get_state_history(thread_id)` mengembalikan log pos pemeriksaan lengkap. Berikan `checkpoint_id` sebelumnya ke `graph.invoke` dan kamu bercabang dari titik itu. Cocok untuk proses debug ("bagaimana jika model memilih alat B?") dan untuk pengujian regresi yang memutar ulang jejak produksi.

### Reducer adalah intinya

Setiap bidang negara bagian memiliki peredam. Sebagian besar nilai default baik-baik saja — nilai baru akan menimpa nilai lama. Namun daftar pesan memerlukan `operator.add` agar pesan baru ditambahkan, bukan diganti. Tepi paralel menggabungkan pembaruannya melalui peredam. Jika dua node memperbarui `messages` dan kamu lupa `Annotated[list, add_messages]`, node kedua menang secara diam-diam dan kamu kehilangan separuh giliran. Peredam adalah satu-satunya hal halus di perpustakaan; melakukannya dengan benar dan sisanya menulis.

### Grafik ReAct dalam empat node

Agen ReAct produksi terdiri dari empat node dan dua edge:

1. `agent` — memanggil LLM dengan riwayat pesan saat ini. Mengembalikan pesan asisten (yang mungkin berisi tool_calls).
2. `tools` — menjalankan panggilan_alat apa pun dalam pesan asisten terakhir, menambahkan hasil alat sebagai pesan alat.
3. Tepi bersyarat dari `agent` yang dirutekan ke `tools` jika pesan terakhir memiliki tool_calls, selain itu ke `END`.
4. Tepi statis dari `tools` kembali ke `agent`.

Hanya itu saja. kamu mendapatkan loop ReAct lengkap (Pemikiran → Tindakan → Pengamatan → Pemikiran → …) dengan pos pemeriksaan, interupsi, dan streaming, dalam sekitar 40 baris code.

### StateGraph vs Kirim (fanout)

`Send(node_name, state)` memungkinkan node mengirimkan subgraf paralel. Contoh: agen memutuskan untuk menanyakan tiga retriever sekaligus. Setiap `Send` memunculkan eksekusi paralel dari node target; keluarannya digabungkan melalui peredam keadaan. Beginilah cara LangGraph mengekspresikan pola pekerja orkestra tanpa memasukkan primitif.

### Subgraf

Graf yang dikompilasi dapat menjadi simpul pada graf lain. Grafik luar melihat satu simpul; grafik bagian dalam memiliki statusnya sendiri dan pos pemeriksaannya sendiri. Beginilah cara tim membangun agen supervisor-pekerja: grafik supervisor mengarahkan niat pengguna ke subgraf pekerja per domain.

## Build

### Langkah 1: status dan node

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` adalah peredam yang membuat daftar pesan terakumulasi alih-alih ditimpa. Melupakannya adalah bug LangGraph yang paling umum.

### Langkah 2: jalankan dengan thread

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

Setiap pembaruan adalah dikte `{node_name: state_delta}`. Frontend kamu dapat mengalirkannya ke UI sehingga pengguna melihat "agen sedang berpikir… menelepon search_web… mendapat hasil… menjawab."

### Langkah 3: tambahkan interupsi human-in-the-loop

Tandai sebuah node sehingga eksekusi dijeda sebelum dijalankan.

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

Status, pos pemeriksaan, dan thread semuanya tetap ada di seluruh interupsi. Tidak ada yang ada di memori kecuali selama eksekusi.

### Langkah 4: perjalanan waktu untuk debugging

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

Melewati `None` sebagai input yang diputar ulang dari pos pemeriksaan yang diberikan; meneruskan nilai akan menambahkannya sebagai pembaruan pada status pos pemeriksaan tersebut sebelum melanjutkan. Ini adalah cara kamu mereproduksi proses agen yang buruk tanpa menjalankan kembali seluruh percakapan.

### Langkah 5: tukar checkpointer dengan produksi

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```SQLite, Redis, dan Postgres dikirimkan. `MemorySaver` adalah untuk ujian. Apa pun yang bertahan saat restart menginginkan penyimpanan nyata.

## Keterampilan

> kamu membuat agen sebagai grafik, bukan sebagai `while True` loop.

Sebelum kamu menggunakan LangGraph, lakukan desain 60 detik:

1. **Beri nama nodenya.** Setiap keputusan terpisah atau tindakan yang memiliki efek samping adalah sebuah node. "Agen berpikir", "alat berjalan", "peninjau menyetujui", "aliran respons". Jika kamu tidak dapat mencantumkannya, maka tugas tersebut belum berbentuk agen.
2. **Deklarasikan status.** Minimal TypedDict dengan peredam untuk setiap kolom daftar. Jangan memasukkan semuanya ke dalam `messages`; angkat bidang khusus tugas (penghitung `plan` yang berfungsi, penghitung `budget`, daftar `retrieved_docs`) ke tingkat atas.
3. **Gambar tepinya.** Statis kecuali langkah berikutnya bergantung pada output model. Setiap tepi bersyarat memerlukan fungsi router dengan cabang bernama.
4. **Pilih checkpointer di awal.** `MemorySaver` untuk pengujian, Postgres/Redis/SQLite untuk hal lainnya. Jangan mengirim tanpa checkpointer — tidak ada checkpointer berarti tidak ada resume, tidak ada interupsi, tidak ada perjalanan waktu.
5. **Putuskan interupsi sebelum alat dijalankan, bukan setelahnya.** Persetujuan diteruskan ke node yang memiliki efek samping sehingga kamu dapat membatalkan sebelum terjadi kerusakan; validasi lebih unggul dari model sehingga kamu dapat menolak panggilan buruk dengan harga murah.
6. **Streaming secara default.** `mode="updates"` untuk UI, `mode="messages"` untuk streaming tingkat token di dalam node model, `mode="values"` untuk snapshot penuh selama evaluasi.

Menolak mengirimkan agen LangGraph yang tidak memiliki pos pemeriksaan. Menolak mengirimkan produk yang mengganggu *setelah* efek sampingnya. Menolak mengirimkan bidang `messages` tanpa `add_messages` sebagai peredamnya.

## Latihan

1. **Mudah.** Implementasikan grafik ReAct empat node di atas dengan alat kalkulator dan alat penelusuran web. Verifikasi bahwa `list(app.get_state_history(config))` mengembalikan setidaknya empat pos pemeriksaan untuk percakapan dua putaran.
2. **Medium.** Tambahkan node `planner` yang berjalan sebelum `agent` dan menulis `plan: list[str]` terstruktur ke dalam status. Minta `agent` menandai langkah rencana sebagai selesai. Gagal dalam pengujian jika `plan` hilang di resume pos pemeriksaan (peredam salah).
3. **Sulit.** Buat grafik pengawas yang merutekan antara tiga subgraf (`researcher`, `writer`, `reviewer`) menggunakan `Send`. Setiap subgraf memiliki status dan pos pemeriksaannya sendiri. Tambahkan `interrupt_before=["writer"]` pada grafik luar sehingga manusia dapat menyetujui ringkasan penelitian. Konfirmasikan bahwa perjalanan waktu dari pos pemeriksaan sebelumnya hanya menjalankan kembali cabang bercabang.

## Istilah Kunci| Istilah | Apa kata orang | Apa sebenarnya arti |
|------|-----------------|-----------------------|
| Grafik Negara | "Grafik LangGraph" | Objek pembangun tempat kamu menambahkan node dan tepi sebelum dikompilasi. |
| Peredam | "Bagaimana bidang tersebut menyatu" | Fungsi `(old, new) -> merged` diterapkan ketika sebuah node mengembalikan pembaruan untuk bidang tersebut; defaultnya ditimpa, `add_messages` ditambahkan. |
| Benang | "ID percakapan" | String `thread_id` yang mencakup semua pos pemeriksaan untuk satu sesi. |
| Pos pemeriksaan | "Keadaan dijeda" | Cuplikan status grafik lengkap yang disimpan setelah transisi node, dengan kunci `(thread_id, checkpoint_id)`. |
| Interupsi | "Jeda untuk manusia" | `interrupt_before` / `interrupt_after` menghentikan eksekusi pada batas node; lanjutkan dengan `Command(resume=...)`. |
| Perjalanan waktu | "Fork dari langkah sebelumnya" | `graph.invoke(None, config_with_old_checkpoint_id)` memutar ulang dari pos pemeriksaan itu ke depan. |
| Kirim | "Pengiriman subgraf paralel" | Sebuah konstruktor sebuah node dapat kembali untuk menelurkan N eksekusi paralel dari sebuah node target. |
| Subgraf | "Grafik yang dikompilasi sebagai simpul" | StateGraph yang dikompilasi digunakan sebagai node di grafik lain; mempertahankan ruang lingkup negaranya sendiri. |

## Bacaan Lanjutan

- [Dokumentasi LangGraph](https://langchain-ai.github.io/langgraph/) — referensi kanonik untuk StateGraph, reduksi, checkpointer, dan interupsi.
- [Konsep LangGraph: status, reduksi, checkpointer](https://langchain-ai.github.io/langgraph/concepts/low_level/) — model mental yang digunakan dalam lesson ini, langsung dari sumbernya.
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) — detail tentang penyimpanan Postgres/SQLite/Redis, namespace checkpoint, dan ID thread.
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`, `interrupt_after`, `Command(resume=...)`, dan pola status edit.
- [Yao dkk., "ReAct: Mensinergikan Penalaran dan Tindakan dalam Model Bahasa" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — pola yang diterapkan setiap agen LangGraph; membacanya untuk alasan jejak penalaran.
- [Anthropic — Membangun agen yang efektif (Des 2024)](https://www.anthropic.com/research/building- Effective-agents) — yang bentuk grafiknya (rantai, router, pekerja orkestra, optimizer-evaluator) yang dipilih dan waktunya.
- Fase 11 · 09 (Pemanggilan Fungsi) — primitif panggilan alat yang digunakan kembali oleh setiap node agen LangGraph.
- Fase 11 · 14 (Model Context Protocol) — penemuan alat eksternal yang dihubungkan ke LangGraph `ToolNode` melalui adaptor MCP.
- Fase 11 · 17 (Pengorbanan kerangka agen) — kapan harus memilih LangGraph daripada CrewAI, AutoGen, atau Agno.
