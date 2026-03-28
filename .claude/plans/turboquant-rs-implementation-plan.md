# TurboQuant-RS: Vector Compression Library + Applications

## Context

Google's TurboQuant paper (ICLR 2026) introduces calibration-free vector compression — PolarQuant and QJL — achieving extreme compression with zero accuracy loss. We're building a **Rust workspace** with three crates: a core compression library, a unified agent memory system (session + persistent), and a semantic code search MCP server.

## Workspace Layout

```
~/workspace/hr/turboquant-rs/
├── Cargo.toml                          # Workspace root
├── crates/
│   ├── turboquant/                     # Core compression + search library
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── compression/
│   │       │   ├── mod.rs
│   │       │   ├── qjl.rs             # Random projection + sign-bit (1-bit/dim)
│   │       │   ├── polarquant.rs       # Rotation + polar coords + grid quantize
│   │       │   └── hamming.rs          # SIMD Hamming distance + bulk top-k
│   │       ├── index.rs               # Append/delete/compact bit vector index
│   │       ├── storage.rs             # Memory-mapped binary file management
│   │       └── search.rs             # Two-stage: QJL pre-filter → PolarQuant re-rank
│   │
│   ├── agent-memory/                   # Unified agent memory (session + persistent)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── embedder.rs            # ONNX Runtime: all-MiniLM-L6-v2 (384 dims)
│   │       ├── session.rs             # SessionMemory — within-conversation context ranking
│   │       ├── persistent.rs          # PersistentMemory — cross-session semantic recall
│   │       ├── document.rs            # Document types (Plan, RCA, Memory, CommitCtx, Turn)
│   │       ├── ranker.rs              # Relevance + recency scoring (shared by both)
│   │       └── budget.rs              # Token-budget-aware context selection
│   │
│   └── codesearch-mcp/                 # MCP server for semantic code search
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs                 # MCP stdio server entrypoint
│           ├── mcp_tools.rs           # index_codebase + semantic_search tool defs
│           ├── embedder.rs            # ONNX Runtime: bge-code-v1 (768 dims)
│           ├── chunker.rs             # .gitignore-aware code file chunking
│           └── indexer.rs             # chunk → embed → compress → store + SQLite
│
├── models/                             # ONNX model files (gitignored)
│   ├── bge-code-v1/                   # For codesearch-mcp (768 dims, code-trained)
│   │   ├── model.onnx
│   │   └── tokenizer.json
│   └── minilm/                        # For agent-memory (384 dims, NL-trained)
│       ├── model.onnx
│       └── tokenizer.json
├── scripts/
│   └── export_onnx.py                 # Export both models to ONNX
└── tests/
    └── integration.rs
```

## Embedding Models

| Crate | Model | Dims | Why |
|-------|-------|------|-----|
| `agent-memory` | `all-MiniLM-L6-v2` | 384 | Memory/plans/conversations are natural language. Small = fast per-turn. |
| `codesearch-mcp` | `BAAI/bge-code-v1` | 768 | Code-trained on GitHub + SO. Understands code semantics. |

---

## Crate 1: `turboquant` (Core Library)

Pure compression + search. No embedding model, no I/O opinions. Dimension-agnostic.

### Dependencies
```toml
[dependencies]
ndarray = "0.16"
rand = "0.8"
rand_distr = "0.4"
memmap2 = "0.9"
bytemuck = "1.14"
serde = { version = "1", features = ["derive"] }
bincode = "1.3"
thiserror = "2"
```

### Public API
```rust
// --- QJL Compression (1-bit per dimension) ---
pub struct QjlCompressor { /* projection matrix R ∈ R^(d×d), dim */ }
impl QjlCompressor {
    pub fn new(dim: usize, seed: u64) -> Self;
    pub fn compress(&self, vector: &[f32]) -> BitVector;
    pub fn compress_batch(&self, vectors: &Array2<f32>) -> Vec<BitVector>;
}
pub struct BitVector(Vec<u8>);  // ceil(dim/8) bytes

// --- PolarQuant Compression (4-bit angles + 8-bit radii) ---
pub struct PolarQuantCompressor { /* orthogonal Q ∈ R^(d×d), angle_bits, radius_bits */ }
impl PolarQuantCompressor {
    pub fn new(dim: usize, seed: u64, angle_bits: u8, radius_bits: u8) -> Self;
    pub fn compress(&self, vector: &[f32]) -> PolarVector;
    pub fn decompress(&self, pv: &PolarVector) -> Vec<f32>;
    pub fn similarity(&self, a: &PolarVector, b: &PolarVector) -> f32;
}
pub struct PolarVector { angles: Vec<u8>, radii: Vec<u8> }

// --- Index (append, delete, two-stage search) ---
pub struct TurboIndex { /* QJL + PolarQuant compressors, mmap storage */ }
impl TurboIndex {
    pub fn create(path: &Path, dim: usize, qjl_seed: u64, polar_seed: u64) -> Result<Self>;
    pub fn open(path: &Path) -> Result<Self>;
    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<()>;
    pub fn insert_batch(&mut self, ids: &[u64], vectors: &Array2<f32>) -> Result<()>;
    pub fn delete(&mut self, id: u64) -> Result<()>;
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult>;
    pub fn compact(&mut self) -> Result<()>;
    pub fn len(&self) -> usize;
}
pub struct SearchResult { pub id: u64, pub score: f32, pub distance: u32 }

// --- Hamming primitives ---
pub fn hamming_distance(a: &BitVector, b: &BitVector) -> u32;
pub fn hamming_top_k(query: &BitVector, index: &[BitVector], k: usize) -> Vec<(usize, u32)>;
```

### Compression Math

**QJL** (1-bit/dim):
```
x ∈ R^d (unit normalized) → z = R @ x → b = sign(z) → pack into ceil(d/8) bytes
R ∈ R^(d×d), R[i,j] ~ N(0, 1/d), deterministic from seed
384 dims → 48 bytes (32× compression)
768 dims → 96 bytes (16× compression)
Similarity: cos(x,y) ≈ cos(π × hamming(b_x, b_y) / d)
```

**PolarQuant** (4-bit angles + 8-bit radii):
```
x ∈ R^d → y = Q @ x (orthogonal rotation, QR of Gaussian)
Pair dims: (y[2i], y[2i+1]) → polar: (r_i, θ_i)
θ_q = round(θ_i × 16 / 2π) mod 16   → 4 bits
r_q = round(clamp(r_i/r_max, 0, 1) × 255) → 8 bits
384 dims → 192 pairs → 288 bytes (5.3× compression)
768 dims → 384 pairs → 576 bytes (5.3× compression)
```

**Two-stage search**: QJL Hamming scan → top-50 → PolarQuant re-rank → top-k.

### Implementation Checklist
- [ ] `compression/qjl.rs` — Gaussian matrix gen (deterministic seed), sign-bit quantize, bit pack/unpack
- [ ] `compression/hamming.rs` — u64 popcount distance, bulk top-k with argpartition
- [ ] `compression/polarquant.rs` — QR orthogonal matrix, polar convert, grid quantize, reconstruct, similarity
- [ ] `search.rs` — two-stage pipeline (QJL → PolarQuant)
- [ ] `index.rs` — TurboIndex: insert, insert_batch, delete, compact, len
- [ ] `storage.rs` — memory-mapped binary files (header + packed vectors), deletion bitmap
- [ ] `lib.rs` — public re-exports
- [ ] Unit tests: distance correlation (Spearman ρ > 0.95), round-trip, pack/unpack, edge cases

---

## Crate 2: `agent-memory` (Unified Agent Memory)

Two modes sharing the same embedding + ranking engine.

### Dependencies
```toml
[dependencies]
turboquant = { path = "../turboquant" }
ort = "2"
tokenizers = "0.20"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"
glob = "0.3"
thiserror = "2"
```

### Module: `embedder.rs` (shared)
- Load `all-MiniLM-L6-v2` ONNX model + tokenizer on init
- `embed(&self, text: &str) -> Vec<f32>` — tokenize → infer → mean pool → L2 normalize
- `embed_batch(&self, texts: &[&str]) -> Array2<f32>` — batch inference
- Lazy model loading, cached after first use

### Module: `document.rs` (shared types)
```rust
pub enum DocumentType {
    Plan,           // ~/.claude/plans/*.md, ~/notion-plans/**/*.html
    Memory,         // ~/.claude/projects/*/memory/*.md
    RCA,            // docs/RCA_*.md
    CommitContext,   // git log summaries
    ConversationTurn, // within-session messages
    ToolResult,     // file reads, grep outputs, etc.
    Custom(String),
}

pub struct Document {
    pub id: u64,
    pub doc_type: DocumentType,
    pub source_path: Option<String>,  // file path if from disk
    pub content: String,
    pub content_preview: String,      // first 200 chars
    pub token_count: usize,
    pub timestamp: u64,               // unix millis
    pub metadata: HashMap<String, String>, // flexible k-v (turn_number, role, etc.)
}
```

### Module: `ranker.rs` (shared scoring)
```rust
pub struct RankingConfig {
    pub relevance_weight: f32,       // default 0.6
    pub recency_weight: f32,         // default 0.4
    pub always_keep_recent: usize,   // default 4 (turns for session, 0 for persistent)
}

/// Score = relevance_weight × (1 - hamming/dim) + recency_weight × (1 / (1 + ln(age)))
pub fn rank(query_bits: &BitVector, candidates: &[(u64, &Document)],
            index: &TurboIndex, config: &RankingConfig) -> Vec<RankedResult>;
```

### Module: `budget.rs` (shared selection)
```rust
/// Greedily select top-ranked documents until token_budget is exhausted.
/// Always keeps `always_keep_recent` most recent items regardless of rank.
/// Returns selected documents in original chronological order.
pub fn select_within_budget(ranked: &[RankedResult], token_budget: usize,
                            config: &RankingConfig) -> Vec<&Document>;
```

### Module: `session.rs` — SessionMemory
```rust
/// Ephemeral within-conversation context ranking.
/// Index lives in memory, dies with the session.
pub struct SessionMemory {
    embedder: Arc<Embedder>,
    index: TurboIndex,       // in-memory (tmpdir)
    documents: Vec<Document>,
    turn_counter: usize,
}

impl SessionMemory {
    pub fn new(embedder: Arc<Embedder>) -> Result<Self>;

    /// Add a conversation turn or tool result
    pub fn add_turn(&mut self, role: &str, content: &str, token_count: usize) -> Result<u64>;

    /// Select the most relevant context for the current message within token budget
    pub fn select_context(&self, current_message: &str, token_budget: usize,
                          config: Option<RankingConfig>) -> Result<Vec<&Document>>;

    pub fn turn_count(&self) -> usize;
    pub fn total_tokens(&self) -> usize;
}
```

### Module: `persistent.rs` — PersistentMemory
```rust
/// Cross-session semantic memory. Index lives on disk, persists forever.
/// Location: ~/.cache/agent-memory/<project-hash>/
pub struct PersistentMemory {
    embedder: Arc<Embedder>,
    index: TurboIndex,         // on-disk, memory-mapped
    documents: Vec<Document>,  // loaded from metadata store
    db_path: PathBuf,
}

impl PersistentMemory {
    pub fn open(storage_dir: &Path, embedder: Arc<Embedder>) -> Result<Self>;

    /// Ingest a single document
    pub fn ingest(&mut self, content: &str, doc_type: DocumentType,
                  source_path: Option<&str>) -> Result<u64>;

    /// Ingest all files matching a glob pattern
    pub fn ingest_glob(&mut self, pattern: &str, doc_type: DocumentType) -> Result<usize>;

    /// Incremental re-ingest: only process files that changed since last ingest
    pub fn sync(&mut self) -> Result<SyncStats>;

    /// Recall relevant prior knowledge for a query
    pub fn recall(&self, query: &str, top_k: usize) -> Result<Vec<RecallResult>>;

    /// Recall with document type filter
    pub fn recall_typed(&self, query: &str, top_k: usize,
                        types: &[DocumentType]) -> Result<Vec<RecallResult>>;

    pub fn stats(&self) -> MemoryStats;
}

pub struct RecallResult {
    pub document: Document,
    pub relevance_score: f32,
    pub recency_score: f32,
    pub combined_score: f32,
}

pub struct SyncStats {
    pub added: usize,
    pub updated: usize,
    pub removed: usize,
    pub unchanged: usize,
}

pub struct MemoryStats {
    pub total_documents: usize,
    pub total_tokens: usize,
    pub index_size_bytes: usize,
    pub by_type: HashMap<DocumentType, usize>,
}
```

### Storage (PersistentMemory)
```
~/.cache/agent-memory/<project-hash>/
├── turbo_index/           # TurboIndex files (QJL + PolarQuant binary, managed by turboquant)
├── documents.json         # Serialized document metadata (id, type, path, preview, tokens, timestamp)
└── file_hashes.json       # SHA256 hashes for incremental sync
```

### Implementation Checklist
- [ ] `embedder.rs` — ONNX model load, tokenize, infer, pool, normalize, batch
- [ ] `document.rs` — Document, DocumentType, RecallResult, SyncStats types
- [ ] `ranker.rs` — relevance + recency scoring, combined ranking
- [ ] `budget.rs` — greedy token-budget selection, always-keep-recent logic
- [ ] `session.rs` — SessionMemory: add_turn, select_context (in-memory TurboIndex)
- [ ] `persistent.rs` — PersistentMemory: open, ingest, ingest_glob, sync, recall, recall_typed
- [ ] `lib.rs` — public re-exports, shared Embedder initialization
- [ ] Tests: session ranking correctness, persistent round-trip, sync incremental, budget limits

---

## Crate 3: `codesearch-mcp` (MCP Server Binary)

### Dependencies
```toml
[dependencies]
turboquant = { path = "../turboquant" }
rmcp = { version = "*", features = ["server", "transport-io"] }
ort = "2"
tokenizers = "0.20"
rusqlite = { version = "0.32", features = ["bundled"] }
ignore = "0.4"
sha2 = "0.10"
tokio = { version = "1", features = ["full"] }
serde_json = "1"
thiserror = "2"
```

### Modules

**`embedder.rs`**: Load `bge-code-v1` ONNX (768 dims). Tokenize → ONNX infer → mean pool → L2 normalize. Batch processing. Same structure as agent-memory embedder but different model.

**`chunker.rs`**: `ignore` crate for .gitignore-aware walking. 60-line blocks, 10-line overlap. Context header prepended (`// File: path (lang)\n// Lines N-M`). Default patterns: `*.rs,*.py,*.ts,*.js,*.rb,*.go,*.java,*.md`. Skip binary files.

**`indexer.rs`**: Full + incremental pipeline. SQLite for file/chunk metadata. TurboIndex for compressed vectors. Diff file hashes → re-index only changed files.

**`mcp_tools.rs`**: Two tools:
- `index_codebase(directory, file_patterns?, exclude_patterns?, force_reindex?)`
- `semantic_search(query, directory, top_k?, file_pattern?)`

**`main.rs`**: rmcp stdio server, register tools, run.

### SQLite Schema
```sql
CREATE TABLE files (path TEXT PRIMARY KEY, content_hash TEXT, mtime_ns INTEGER, indexed_at INTEGER);
CREATE TABLE chunks (id INTEGER PRIMARY KEY, file_path TEXT, start_line INTEGER, end_line INTEGER, preview TEXT);
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
```

### Index Location
`~/.cache/codesearch/<sha256(directory)[:16]>/` — `metadata.db` + TurboIndex files.

### Claude Code Integration
```json
{ "mcpServers": { "codesearch": { "command": "~/workspace/hr/turboquant-rs/target/release/codesearch-mcp" } } }
```

### Implementation Checklist
- [ ] `embedder.rs` — ONNX bge-code-v1, tokenize, infer, pool, normalize
- [ ] `chunker.rs` — directory walk (.gitignore aware), line-block chunking
- [ ] `indexer.rs` — full + incremental pipeline, SQLite metadata
- [ ] `mcp_tools.rs` — tool schemas, input validation, handler dispatch
- [ ] `main.rs` — rmcp stdio server setup
- [ ] Integration test: index a test dir → search → verify results

---

## ONNX Model Setup

One-time Python script (`scripts/export_onnx.py`):
```python
# Exports both models to ONNX format
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

for model_name, output_dir in [
    ("sentence-transformers/all-MiniLM-L6-v2", "models/minilm"),
    ("BAAI/bge-code-v1", "models/bge-code-v1"),  # verify exact HF name
]:
    model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
```

Rust binaries check `models/` dir on startup, print setup instructions if missing.

---

## Build Order

### Step 1 — Workspace scaffold
- [ ] Create workspace Cargo.toml + all three crate Cargo.tomls
- [ ] Create directory structure, empty src files with module declarations
- [ ] `cargo check` passes on empty workspace

### Step 2 — `turboquant` core (no external dependencies beyond ndarray/rand)
- [ ] QJL compression + bit packing
- [ ] Hamming distance + bulk top-k
- [ ] PolarQuant compression + reconstruction + similarity
- [ ] Two-stage search pipeline
- [ ] TurboIndex (insert, delete, compact, search)
- [ ] Memory-mapped storage
- [ ] Unit tests

### Step 3 — `agent-memory` (depends on turboquant + ort + tokenizers)
- [ ] Embedder (ONNX MiniLM)
- [ ] Document types + ranking + budget selection
- [ ] SessionMemory (in-memory)
- [ ] PersistentMemory (on-disk, incremental sync)
- [ ] Tests

### Step 4 — `codesearch-mcp` (depends on turboquant + ort + rmcp)
- [ ] Embedder (ONNX bge-code-v1)
- [ ] Chunker + Indexer + SQLite
- [ ] MCP tools + server
- [ ] Integration tests + Claude Code config

### Step 5 — ONNX model export + end-to-end testing
- [ ] export_onnx.py script
- [ ] Self-search test (index this repo)
- [ ] Monolith test (index hackerrank)
- [ ] Memory test (ingest plans/memories, recall)

---

## Verification

1. `cargo test` — all crates
2. Compression quality: 1000 random pairs, Spearman ρ > 0.95 between Hamming and cosine
3. Self-search: index turboquant-rs, "hamming distance" → hamming.rs top-3
4. Monolith: index hackerrank, "question insights worker" → relevant worker files
5. Memory recall: ingest plans + RCAs, query "auth optimization" → role-service plans
6. Session test: 50-turn synthetic conversation, verify relevant turns selected within budget
7. Claude Code: configure MCP server, use semantic_search in real session
