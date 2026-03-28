# TurboQuant-RS

Fast vector compression and semantic memory for AI agents.

TurboQuant compresses high-dimensional embedding vectors to 1-4 bits per dimension, enabling semantic search over thousands of documents with minimal memory. Built for AI coding agents that need to recall prior context across sessions.

## How It Works

<p align="center">
  <img src="docs/turboquant-animation.svg" alt="TurboQuant Algorithm Animation" width="800">
</p>

> Open `docs/algorithm-explainer.html` in a browser for the full interactive version with randomizable demos.

## Architecture

```
turboquant-rs/
  crates/
    turboquant/       Core compression: QJL (1-bit) + PolarQuant (4-bit) + two-stage search
    agent-memory/     Session context ranking + persistent cross-session semantic recall
    memory-mcp/       MCP server: 4 tools for AI agent integration
    codesearch-mcp/   Semantic code search (scaffolded, not yet implemented)
  scripts/
    export_onnx.py    Download & export embedding models to ONNX format
```

## Compression Algorithms

### QJL (Quantized Johnson-Lindenstrauss) - 1-bit

Compresses each vector to **1 bit per dimension** (32x compression vs float32).

1. Generate a random Gaussian projection matrix R (deterministic from seed)
2. Project the vector: `y = R @ x`
3. Keep only the sign bits: `b = sign(y)`

Vectors that are similar in the original space will share most sign bits, so **Hamming distance between bit vectors approximates cosine distance**. Used as a fast pre-filter.

### PolarQuant - 4-bit angles + 8-bit radii

Compresses each vector to **~0.75 bytes per dimension** (4x compression).

1. Apply a random orthogonal rotation (Gram-Schmidt, deterministic from seed)
2. Pair adjacent dimensions into 2D coordinates
3. Convert each pair to polar form: `(x, y) -> (r, theta)`
4. Quantize angles to 4 bits (16 levels) and radii to 8 bits (256 levels)

Reconstructed similarity uses: `sum_i(r_a * r_b * cos(theta_a - theta_b))`, which closely approximates the true dot product. Used for accurate re-ranking.

### Two-Stage Search

```
Query vector
    |
    v
[QJL compress] -> Hamming scan over all vectors -> top-K candidates (fast, approximate)
    |
    v
[PolarQuant re-rank] -> accurate similarity on K candidates -> final top-k results
```

This gives near-exact recall at a fraction of the cost of brute-force search.

## Crates

### `turboquant`

Core compression library with no runtime dependencies beyond ndarray.

```rust
use turboquant::{TurboIndex, SearchResult};

// Create an index (stored on disk via mmap)
let mut index = TurboIndex::create("./my_index", 384, /*qjl_seed=*/42, /*polar_seed=*/99)?;

// Insert vectors (e.g., from an embedding model)
index.insert(1, &embedding_vec)?;
index.insert_batch(&ids, &embedding_matrix)?;

// Search
let results: Vec<SearchResult> = index.search(&query_vec, 10);
for r in &results {
    println!("id={} score={:.3} hamming={}", r.id, r.score, r.distance);
}

// Maintenance
index.delete(1)?;
index.compact()?;  // rebuild without deleted vectors
```

**Storage format:** Memory-mapped files with 16-byte headers. QJL vectors (`*.tqjl`) and PolarQuant vectors (`*.tqpl`) are stored separately for cache-friendly scanning.

### `agent-memory`

Two memory systems designed for AI agent workflows:

**SessionMemory** - in-conversation context management:
- Tracks conversation turns with token counts
- Selects relevant context within a token budget
- Ranks by `relevance_weight * search_score + recency_weight * recency_score`
- Always includes the N most recent turns (configurable)

```rust
use agent_memory::{SessionMemory, MockEmbedder};

let mut session = SessionMemory::new(Arc::new(MockEmbedder::new(384)))?;
session.add_turn("user", "How does the auth middleware work?", 12)?;
session.add_turn("assistant", "The auth middleware validates JWT tokens...", 85)?;

// Select context that fits in 4096 tokens, ranked by relevance to current message
let context = session.select_context("Now fix the token expiry bug", 4096, None)?;
```

**PersistentMemory** - cross-session knowledge base:
- Ingests documents (plans, RCAs, investigation notes) with semantic indexing
- Recalls by natural language query with type filtering
- Tracks file changes via SHA-256 for incremental sync
- Supports glob patterns for bulk ingestion

```rust
use agent_memory::{PersistentMemory, OnnxEmbedder, DocumentType};

let embedder = Arc::new(OnnxEmbedder::load(Path::new("models/minilm"), 384)?);
let mut mem = PersistentMemory::open(Path::new("./memory_store"), embedder)?;

// Ingest
mem.ingest("Auth rewrite: moved to asymmetric JWT...", DocumentType::RCA, Some("docs/rca_auth.md"))?;
mem.ingest_glob("docs/plans/*.md", DocumentType::Plan)?;

// Recall
let results = mem.recall("JWT token validation", 5)?;
let rca_only = mem.recall_typed("auth bug", 5, &[DocumentType::RCA])?;

// Sync after files change on disk
let stats = mem.sync()?;
println!("{} added, {} updated, {} removed", stats.added, stats.updated, stats.removed);
```

**Document types:** `Plan`, `Memory`, `RCA`, `CommitContext`, `ConversationTurn`, `ToolResult`, `Custom(String)`

**Embedder trait:** Pluggable embedding backend. Ships with:
- `MockEmbedder` - deterministic SHA-256-based pseudo-embeddings (for testing)
- `OnnxEmbedder` - production embeddings via ONNX Runtime (supports any HuggingFace model exported to ONNX)

### `memory-mcp`

MCP (Model Context Protocol) server that exposes agent-memory to AI coding agents like Claude Code.

**Tools:**

| Tool | Description |
|------|-------------|
| `memory_ingest` | Ingest text, a file, or files matching a glob pattern into the index |
| `memory_recall` | Semantic search over ingested documents by natural language query |
| `memory_sync` | Re-index changed files, remove deleted files |
| `memory_stats` | Index statistics: document count, token count, type breakdown |

**Model auto-detection:** On startup, the server looks for an ONNX model in:
1. `MEMORY_MODEL_DIR` environment variable
2. `<binary_dir>/../models/minilm/`
3. `~/.cache/agent-memory/models/minilm/`

Falls back to `MockEmbedder` if no model is found (semantic recall will be degraded).

**Storage:** Per-project memory stored at `~/.cache/agent-memory/<project_hash>/`.

### `codesearch-mcp` (scaffolded)

Semantic code search server. Will chunk code files (`.gitignore`-aware), embed with BGE-code-v1, and expose `index_codebase` + `semantic_search` MCP tools. Not yet implemented.

## Quick Start

### 1. Build

```bash
cargo build --release -p memory-mcp
```

### 2. Download the embedding model

```bash
pip install transformers optimum[onnxruntime] torch
python scripts/export_onnx.py --output ~/.cache/agent-memory/models/minilm
```

This downloads `all-MiniLM-L6-v2` (384 dimensions, ~80MB) and exports it to ONNX format.

Use `--model` to export a different model:
```bash
python scripts/export_onnx.py --model sentence-transformers/all-mpnet-base-v2 --output ~/.cache/agent-memory/models/mpnet
```

### 3. Configure Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "/path/to/turboquant-rs/target/release/memory-mcp"
    }
  }
}
```

Or with a custom model path:

```json
{
  "mcpServers": {
    "memory": {
      "command": "/path/to/turboquant-rs/target/release/memory-mcp",
      "env": {
        "MEMORY_MODEL_DIR": "/path/to/your/model/directory"
      }
    }
  }
}
```

### 4. Seed the index

In a Claude Code conversation:
```
Ingest my plans and RCAs:
- memory_ingest(glob: "~/.claude/plans/*.md")
- memory_ingest(glob: "~/workspace/docs/RCA_*.md")
```

### 5. Recall

```
memory_recall(query: "auth middleware JWT token validation")
```

## Compression Efficiency

| Method | Bits/dim | Compression vs f32 | Use case |
|--------|----------|-------------------|----------|
| QJL | 1 | 32x | Fast pre-filtering |
| PolarQuant (4-bit angle, 8-bit radius) | ~6 | ~5x | Accurate re-ranking |
| Two-stage combined | - | - | Best of both: speed + accuracy |

For a 384-dim embedding model (MiniLM):
- Raw: 1,536 bytes per vector
- QJL: 48 bytes per vector
- PolarQuant: 576 bytes per vector
- 10,000 documents: ~6MB total index (vs ~15MB raw)

## Running Tests

```bash
# All tests (52 total)
cargo test --workspace

# With real ONNX model (requires export_onnx.py first)
cargo test -p agent-memory test_onnx_embedder_real_model
```

## License

TBD
