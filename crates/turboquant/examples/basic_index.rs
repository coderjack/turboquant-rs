//! Basic TurboIndex usage — create, insert, search, delete.
//!
//! Run with: cargo run -p turboquant --example basic_index

use turboquant::{LloydMaxCompressor, TurboIndex};

fn main() {
    let dim = 384;
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = dir.path().join("demo_index");

    // -----------------------------------------------------------------------
    // 1. Create an index
    // -----------------------------------------------------------------------
    println!("Creating index at {:?} (dim={dim})", index_path);

    let mut index = TurboIndex::create(
        &index_path,
        dim,
        42,  // rotation seed  — deterministic, pick any u64
        99,  // QJL seed        — deterministic, pick any u64
    )
    .expect("failed to create index");

    // -----------------------------------------------------------------------
    // 2. Insert some vectors
    // -----------------------------------------------------------------------
    // In a real app these would be embeddings from an ONNX model.
    // Here we use simple synthetic vectors for demonstration.
    let documents = [
        "Rust is a systems programming language",
        "Python is popular for machine learning",
        "TurboQuant compresses embeddings efficiently",
        "Vector databases enable semantic search",
        "ONNX Runtime runs models on CPU and GPU",
    ];

    // Fake embeddings: hash each document into a deterministic vector.
    for (i, doc) in documents.iter().enumerate() {
        let embedding = fake_embedding(doc, dim);
        index
            .insert(i as u64, &embedding)
            .expect("failed to insert");
    }
    println!("Inserted {} vectors", index.len());

    // -----------------------------------------------------------------------
    // 3. Search
    // -----------------------------------------------------------------------
    let query = "how to compress neural network embeddings";
    let query_vec = fake_embedding(query, dim);

    let results = index.search(&query_vec, 3);

    println!("\nQuery: \"{query}\"");
    println!("Top-3 results:");
    for (rank, r) in results.iter().enumerate() {
        println!(
            "  #{}: id={}, score={:.4} — \"{}\"",
            rank + 1,
            r.id,
            r.score,
            documents[r.id as usize]
        );
    }

    // -----------------------------------------------------------------------
    // 4. Delete + compact
    // -----------------------------------------------------------------------
    println!("\nDeleting id=1...");
    index.delete(1).expect("failed to delete");
    println!("Live vectors: {}", index.len());

    index.compact().expect("failed to compact");
    println!("Compacted. Live vectors: {}", index.len());

    // -----------------------------------------------------------------------
    // 5. Re-open from disk
    // -----------------------------------------------------------------------
    let reopened = TurboIndex::open(&index_path).expect("failed to reopen");
    println!("\nRe-opened index from disk: {} vectors", reopened.len());

    // -----------------------------------------------------------------------
    // 6. Show compression stats
    // -----------------------------------------------------------------------
    let compressor = LloydMaxCompressor::new(dim, 42, 99, 4);
    let raw_bytes = dim * 4; // f32
    let compressed_bytes = compressor.bytes_per_vector();
    let bits_per_dim = (compressed_bytes as f64 * 8.0) / dim as f64;
    println!(
        "\nCompression: {} bytes → {} bytes ({:.1}x, {:.2} bits/dim)",
        raw_bytes,
        compressed_bytes,
        raw_bytes as f64 / compressed_bytes as f64,
        bits_per_dim,
    );
}

/// Deterministic fake embedding from text (for demo purposes only).
/// In production, use a real embedding model (see semantic_search example).
fn fake_embedding(text: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut v = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut h = DefaultHasher::new();
        text.hash(&mut h);
        i.hash(&mut h);
        let bits = h.finish();
        // Map hash to [-1, 1] range
        v.push(((bits as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32);
    }
    // L2 normalize
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}
