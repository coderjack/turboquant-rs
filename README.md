# TurboQuant-RS

Rust implementation of Google's [TurboQuant](https://arxiv.org/abs/2504.19874) algorithm for compressing high-dimensional embedding vectors. Achieves **6.2x compression** at 384 dimensions with calibration-free, near-optimal distortion guarantees.

```rust
use turboquant::{TurboIndex, SearchResult};

let mut index = TurboIndex::create("./my_index", 384, 42, 99)?;
index.insert(0, &embedding)?;

let results: Vec<SearchResult> = index.search(&query, 10);
```

## How It Works

TurboQuant compresses vectors in three steps:

```
Input vector (384 × f32 = 1,536 bytes)
    │
    ▼
┌───────────────────────────────────────┐
│  Step 1: Random Orthogonal Rotation   │  Spreads energy evenly — each
│  Q × x (deterministic from seed)      │  coordinate → N(0, 1/√d)
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│  Step 2: Lloyd-Max Scalar Quantize    │  MSE-optimal centroids for
│  b bits per dimension (default b=4)   │  the Gaussian distribution
└───────────────────┬───────────────────┘
                    │ residual = rotated − reconstructed
                    ▼
┌───────────────────────────────────────┐
│  Step 3: QJL Residual Correction      │  sign(S × residual)
│  1-bit sign per dimension             │  → unbiased IP estimator
└───────────────────┬───────────────────┘
                    │
                    ▼
Output: 248 bytes (6.2x compression, 5.17 bits/dim)
```

The rotation decorrelates coordinates so each follows a known Gaussian. Lloyd-Max finds the provably optimal quantization centroids for this distribution (within 2.7x of information-theoretic lower bound). QJL corrects the residual, making the inner product estimator **unbiased**.

> **Interactive demo:** Open [`docs/turbo-quant-visualizer.html`](docs/turbo-quant-visualizer.html) in a browser to see the algorithm animate on 3D vectors — step through rotation, quantization, and QJL correction with orbit controls.

## Compression Numbers

Measured on 384-dim random unit vectors (from `cargo test --test compression_claims`):

| Config | Bytes/vec | Bits/dim | Compression | Avg Cosine |
|--------|-----------|----------|-------------|------------|
| 1-bit Lloyd-Max | 104 | 2.17 | 14.8x | 0.630 |
| 2-bit Lloyd-Max | 152 | 3.17 | 10.1x | 0.756 |
| 3-bit Lloyd-Max | 200 | 4.17 | 7.7x | 0.818 |
| **4-bit Lloyd-Max (default)** | **248** | **5.17** | **6.2x** | **0.852** |
| 6-bit Lloyd-Max | 344 | 7.17 | 4.5x | 0.911 |

For 10,000 embeddings at 384 dimensions:
- Raw float32: **15.0 MB**
- TurboQuant 4-bit: **2.4 MB**

## Quick Start

### As a dependency

```toml
[dependencies]
turboquant = { git = "https://github.com/coderjack/turboquant-rs" }
```

### High-level: TurboIndex

Create an index, insert vectors, search:

```rust
use turboquant::{TurboIndex, SearchResult};

// Create a persistent index on disk (default: 4-bit Lloyd-Max)
let mut index = TurboIndex::create(
    "./my_index",
    384,  // embedding dimension
    42,   // rotation seed (any u64, deterministic)
    99,   // QJL seed (any u64, deterministic)
)?;

// Or choose a specific bit width: 1–8
let mut index = TurboIndex::create_with_bits("./my_index", 384, 42, 99, 3)?;

// Insert embeddings with unique IDs
index.insert(0, &embedding_vec)?;
index.insert(1, &another_vec)?;

// Search: returns top-k by approximate inner product
let results: Vec<SearchResult> = index.search(&query_vec, 10);
for r in &results {
    println!("id={} score={:.4}", r.id, r.score);
}

// Maintenance
index.delete(1)?;     // soft-delete
index.compact()?;      // rebuild without deleted vectors

// Re-open from disk in another process
let index = TurboIndex::open("./my_index")?;
```

### Low-level: LloydMaxCompressor

For direct access to compression without the index:

```rust
use turboquant::{LloydMaxCompressor, PreparedQuery};

let compressor = LloydMaxCompressor::new(
    384,  // dimension
    42,   // rotation seed
    99,   // QJL seed
    4,    // bits per dimension
);

// Compress a vector
let compressed = compressor.compress(&vector);
println!("{} bytes", compressed.byte_size()); // 248

// Decompress (Lloyd-Max reconstruction, no QJL)
let reconstructed = compressor.decompress(&compressed);

// Similarity search: prepare query once, score many candidates
let prepared: PreparedQuery = compressor.prepare_query(&query);
let score = compressor.similarity_prepared(&prepared, &compressed);
```

`prepare_query` does the O(d^2) rotation + QJL projection once. Each `similarity_prepared` call is only O(d).

## Examples

### Basic index (no external deps)

```bash
cargo run -p turboquant --example basic_index
```

Creates an index with synthetic vectors, inserts, searches, deletes, compacts, and reopens from disk.

### Semantic search with a local ONNX model

End-to-end semantic search using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384-dim):

```bash
# 1. Export the model to ONNX (one-time)
pip install optimum[exporters]
optimum-cli export onnx \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    models/all-MiniLM-L6-v2

# 2. Run the example
cargo run -p turboquant --example semantic_search -- \
    --model-dir models/all-MiniLM-L6-v2
```

## Architecture

```
turboquant-rs/
  crates/
    turboquant/          Core compression library
      src/
        compression/
          rotation.rs      Step 1: Random orthogonal rotation (Gram-Schmidt)
          lloyd_max.rs     Step 2: Lloyd-Max scalar quantization (default)
          polarquant.rs    Step 2 alt: Polar coordinate quantization
          qjl.rs           Step 3: QJL 1-bit residual correction
        turboquant.rs      Pipeline composition (LloydMaxCompressor + TurboQuantCompressor)
        index.rs           TurboIndex: insert, delete, search, compact
        storage.rs         File-based vector storage
        lib.rs             Public API re-exports
      examples/
        basic_index.rs     Minimal usage example
        semantic_search.rs End-to-end with ONNX embedding model
      tests/
        compression_claims.rs  Property tests + baseline comparisons
        head_to_head.rs        Lloyd-Max vs PolarQuant benchmark
```

### Dependencies

The core library has **zero ML dependencies** — pure math + compression:

- `ndarray` — matrix operations
- `rand`, `rand_chacha`, `rand_distr` — deterministic RNG
- `serde`, `bincode` — serialization
- `thiserror` — error types

ONNX Runtime and tokenizers are dev-dependencies only, used by the `semantic_search` example.

## Running Tests

```bash
# All tests
cargo test -p turboquant

# With verbose output (shows compression tables, recall numbers)
cargo test -p turboquant --test compression_claims -- --nocapture

# Lloyd-Max vs PolarQuant comparison
cargo test -p turboquant --test head_to_head -- --nocapture
```

## References

This project implements algorithms from:

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

Google Research blog post: [TurboQuant: Redefining AI Efficiency with Extreme Compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
