# TurboQuant-RS

Fast vector compression and semantic memory for AI agents.

TurboQuant-RS implements the TurboQuant family of algorithms for compressing high-dimensional embedding vectors to 1-4 bits per dimension, enabling semantic search over thousands of documents with minimal memory. Built for AI coding agents that need to recall prior context across sessions.

## How It Works

<p align="center">
  <img src="docs/turboquant-animation.svg" alt="TurboQuant Algorithm Animation" width="800">
</p>

> Open `docs/algorithm-explainer.html` in a browser for the full interactive version with randomizable demos.

## Architecture

```
turboquant-rs/
  crates/
    turboquant/       Core: QJL (1-bit) + TurboQuant_mse (2-4 bit) + two-stage search
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

Vectors that are similar in the original space will share most sign bits, so **Hamming distance between bit vectors approximates cosine distance**. Used as a fast pre-filter in the first search stage.

> Zandieh, A., Daliri, M., & Han, I. (2024). *QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead.* [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)

### TurboQuant_mse - Optimal Scalar Quantization (2-4 bit)

Compresses each vector to **b bits per dimension** with provably near-optimal MSE distortion (within 2.7x of the information-theoretic lower bound).

1. Normalize the vector and store its L2 norm separately
2. Multiply by a random orthogonal matrix Q (Gram-Schmidt on Gaussian, deterministic from seed)
3. After rotation, each coordinate follows ≈ N(0, 1/d) — quantize independently with a precomputed [Lloyd-Max codebook](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm)
4. Pack b-bit indices into bytes

**Key insight for search:** Since Q is orthogonal, `<x, y> = <Qx, Qy>`. Similarity is computed directly in the rotated domain using codebook lookups — no matrix multiply during search.

| Bits | Centroids | Storage (384-dim) | MSE bound |
|------|-----------|-------------------|-----------|
| 2 | 4 | 100 bytes | ≤ 0.117 |
| 3 | 8 | 148 bytes | ≤ 0.038 |
| 4 | 16 | 196 bytes | ≤ 0.009 |

> Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

### PolarQuant (also included)

Available as an alternative compressor. Pairs adjacent dimensions into 2D polar coordinates, quantizes angles (4-bit) and radii (8-bit). Not used in the default search pipeline but kept for comparison.

> Han, I., Kacham, P., Karbasi, A., Mirrokni, V., & Zandieh, A. (2025). *PolarQuant.* [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)

### Two-Stage Search

```
Query vector
    |
    v
[QJL 1-bit] --> Hamming scan over all vectors --> top-K candidates  (fast, approximate)
    |
    v
[TurboQuant_mse 3-bit] --> codebook similarity on K candidates --> final top-k results
```

QJL eliminates ~98% of candidates with a single XOR + popcount per vector. TurboQuant_mse re-ranks only the survivors with near-optimal accuracy.

## Crates

### `turboquant`

Core compression library with no runtime dependencies beyond ndarray.

```rust
use turboquant::{TurboIndex, SearchResult};

// Create an index (stored on disk via mmap)
// Default: QJL 1-bit pre-filter + TurboQuant_mse 3-bit re-ranking
let mut index = TurboIndex::create("./my_index", 384, /*qjl_seed=*/42, /*tqmse_seed=*/99)?;

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

Use `TurboIndex::create_with_bits()` to choose 2-bit (smallest) or 4-bit (most accurate) re-ranking.

**Storage format:** Memory-mapped files with 16-byte headers. QJL vectors (`*.qjl`) and TurboQuant_mse vectors (`*.tqsq`) are stored separately for cache-friendly scanning.

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

| Method | Bits/dim | Bytes/vec (384-dim) | Compression vs f32 | Use case |
|--------|----------|--------------------|--------------------|----------|
| QJL | 1 | 48 | 32x | Fast pre-filtering |
| TurboQuant_mse 2-bit | 2 | 100 | 15x | Compact re-ranking |
| TurboQuant_mse 3-bit | 3 | 148 | 10x | Default re-ranking |
| TurboQuant_mse 4-bit | 4 | 196 | 8x | High-accuracy re-ranking |
| PolarQuant | ~6 | 288 | 5x | Alternative (legacy) |

For 10,000 documents at 384 dimensions:
- Raw float32: 15.0 MB
- QJL + TurboQuant_mse 3-bit: **1.9 MB** (7.9x total compression)
- QJL + PolarQuant: 3.3 MB

## Running Tests

```bash
# All tests (65 total)
cargo test --workspace

# With real ONNX model (requires export_onnx.py first)
cargo test -p agent-memory test_onnx_embedder_real_model
```

## Citations

This project implements algorithms from the following papers:

```bibtex
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}

@article{zandieh2024qjl,
  title={QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead},
  author={Zandieh, Amir and Daliri, Majid and Han, Insu},
  journal={arXiv preprint arXiv:2406.03482},
  year={2024}
}

@article{han2025polarquant,
  title={PolarQuant},
  author={Han, Insu and Kacham, Praneeth and Karbasi, Amin and Mirrokni, Vahab and Zandieh, Amir},
  journal={arXiv preprint arXiv:2502.02617},
  year={2025}
}
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
