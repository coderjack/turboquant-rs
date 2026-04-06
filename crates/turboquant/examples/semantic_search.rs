//! Semantic search using a local ONNX embedding model + TurboQuant.
//!
//! Embeds a corpus of text snippets with all-MiniLM-L6-v2 (384-dim),
//! builds a TurboQuant index, and runs queries against it.
//!
//! ## Setup
//!
//! 1. Download the model (one-time):
//!    ```bash
//!    pip install optimum[exporters]
//!    optimum-cli export onnx \
//!        --model sentence-transformers/all-MiniLM-L6-v2 \
//!        models/all-MiniLM-L6-v2
//!    ```
//!
//! 2. Run the example:
//!    ```bash
//!    cargo run -p turboquant --example semantic_search -- \
//!        --model-dir models/all-MiniLM-L6-v2
//!    ```
//!
//! The model directory should contain:
//!   - `model.onnx` (or any `.onnx` file)
//!   - `tokenizer.json`

use std::path::PathBuf;
use turboquant::TurboIndex;

// ---------------------------------------------------------------------------
// Sample corpus — replace with your own documents
// ---------------------------------------------------------------------------

const CORPUS: &[&str] = &[
    "Rust is a systems programming language focused on safety, speed, and concurrency.",
    "Python is widely used for machine learning and data science applications.",
    "TurboQuant compresses high-dimensional vectors with near-optimal distortion.",
    "Vector databases enable fast similarity search over embedding collections.",
    "ONNX Runtime provides cross-platform inference for machine learning models.",
    "Transformers revolutionized natural language processing with attention mechanisms.",
    "Embedding models convert text into dense numerical representations.",
    "Cosine similarity measures the angle between two vectors in high-dimensional space.",
    "Quantization reduces memory usage by representing values with fewer bits.",
    "Semantic search finds documents by meaning rather than keyword matching.",
    "The Johnson-Lindenstrauss lemma states random projections preserve distances.",
    "Lloyd-Max quantization finds optimal centroids for Gaussian-distributed data.",
    "Graph neural networks learn representations over structured relational data.",
    "Retrieval-augmented generation combines search with language model outputs.",
    "Approximate nearest neighbor algorithms trade accuracy for search speed.",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = parse_model_dir(&args);

    // -----------------------------------------------------------------------
    // 1. Load the embedding model
    // -----------------------------------------------------------------------
    println!("Loading ONNX model from: {}", model_dir.display());
    let mut embedder = match OnnxEmbedder::load(&model_dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error loading model: {e}");
            eprintln!("\nTo download the model:");
            eprintln!("  pip install optimum[exporters]");
            eprintln!("  optimum-cli export onnx \\");
            eprintln!("      --model sentence-transformers/all-MiniLM-L6-v2 \\");
            eprintln!("      models/all-MiniLM-L6-v2");
            std::process::exit(1);
        }
    };
    println!("Model loaded (dim={})\n", embedder.dim());

    // -----------------------------------------------------------------------
    // 2. Embed the corpus
    // -----------------------------------------------------------------------
    println!("Embedding {} documents...", CORPUS.len());
    let texts: Vec<&str> = CORPUS.to_vec();
    let embeddings = embedder.embed_batch(&texts).expect("embedding failed");
    println!("Done.\n");

    // -----------------------------------------------------------------------
    // 3. Build a TurboQuant index
    // -----------------------------------------------------------------------
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let index_path = dir.path().join("semantic_index");

    let mut index = TurboIndex::create(
        &index_path,
        embedder.dim(),
        42, // rotation seed
        99, // QJL seed
    )
    .expect("failed to create index");

    for (i, emb) in embeddings.iter().enumerate() {
        index.insert(i as u64, emb).expect("failed to insert");
    }

    let compressor = turboquant::LloydMaxCompressor::new(embedder.dim(), 42, 99, 4);
    let raw_bytes = embedder.dim() * 4;
    let comp_bytes = compressor.bytes_per_vector();
    println!(
        "Index built: {} vectors, {} -> {} bytes/vec ({:.1}x compression)\n",
        index.len(),
        raw_bytes,
        comp_bytes,
        raw_bytes as f64 / comp_bytes as f64,
    );

    // -----------------------------------------------------------------------
    // 4. Search
    // -----------------------------------------------------------------------
    let queries = [
        "how to make AI models smaller and faster",
        "programming languages for building reliable software",
        "finding similar documents using neural networks",
    ];

    for query in &queries {
        println!("Query: \"{query}\"");

        let query_vec = embedder.embed(query).expect("query embedding failed");
        let results = index.search(&query_vec, 5);

        println!("  Top-5 results:");
        for (rank, r) in results.iter().enumerate() {
            println!(
                "    #{}: [{:.4}] {}",
                rank + 1,
                r.score,
                CORPUS[r.id as usize],
            );
        }
        println!();
    }
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

fn parse_model_dir(args: &[String]) -> PathBuf {
    for i in 0..args.len() {
        if args[i] == "--model-dir" {
            if let Some(dir) = args.get(i + 1) {
                return PathBuf::from(dir);
            }
        }
    }
    PathBuf::from("models/all-MiniLM-L6-v2")
}

// ---------------------------------------------------------------------------
// Minimal ONNX embedder (self-contained for the example)
//
// Uses ort 2.0.0-rc.12 API:
//   - Session at ort::session::Session
//   - Tensor::from_array((shape, vec)) for tensor creation
//   - inputs! macro returns Vec, not Result
//   - try_extract_tensor() returns (&Shape, &[T])
// ---------------------------------------------------------------------------

use ort::session::Session;
use ort::value::Tensor;

struct OnnxEmbedder {
    session: Session,
    tokenizer: tokenizers::Tokenizer,
    dim: usize,
    has_token_type_ids: bool,
}

impl OnnxEmbedder {
    fn load(model_dir: &std::path::Path) -> Result<Self, String> {
        let model_path = std::fs::read_dir(model_dir)
            .map_err(|e| format!("cannot read model dir: {e}"))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().map_or(false, |ext| ext == "onnx"))
            .ok_or_else(|| format!("no .onnx file found in {}", model_dir.display()))?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(format!(
                "tokenizer.json not found in {}",
                model_dir.display()
            ));
        }

        let session = Session::builder()
            .map_err(|e| format!("ort session builder: {e}"))?
            .commit_from_file(&model_path)
            .map_err(|e| format!("ort load model: {e}"))?;

        let has_token_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");

        // Infer dim from model output shape
        let dim = match session.outputs().first().map(|o| o.dtype()) {
            Some(ort::value::ValueType::Tensor { shape, .. }) => {
                shape.last().copied().unwrap_or(384) as usize
            }
            _ => 384,
        };

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("tokenizer load: {e}"))?;

        Ok(Self {
            session,
            tokenizer,
            dim,
            has_token_type_ids,
        })
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&mut self, text: &str) -> Result<Vec<f32>, String> {
        Ok(self.embed_batch(&[text])?.remove(0))
    }

    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        let strings: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let encodings = self
            .tokenizer
            .encode_batch(strings, true)
            .map_err(|e| format!("tokenize: {e}"))?;

        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Build padded input tensors
        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attn_mask = vec![0i64; batch_size * max_len];
        let mut type_ids = vec![0i64; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            for (j, (&id, &mask)) in
                enc.get_ids().iter().zip(enc.get_attention_mask()).enumerate()
            {
                let idx = i * max_len + j;
                input_ids[idx] = id as i64;
                attn_mask[idx] = mask as i64;
                type_ids[idx] = enc.get_type_ids()[j] as i64;
            }
        }

        // Use (shape, Vec<T>) form to avoid ndarray version mismatch with ort
        let shape = [batch_size as i64, max_len as i64];

        let input_ids_tensor =
            Tensor::from_array((shape, input_ids)).map_err(|e| format!("tensor: {e}"))?;
        let attn_mask_tensor =
            Tensor::from_array((shape, attn_mask)).map_err(|e| format!("tensor: {e}"))?;

        let outputs = if self.has_token_type_ids {
            let type_ids_tensor =
                Tensor::from_array((shape, type_ids)).map_err(|e| format!("tensor: {e}"))?;

            self.session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attn_mask_tensor,
                "token_type_ids" => type_ids_tensor,
            ])
        } else {
            self.session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attn_mask_tensor,
            ])
        }
        .map_err(|e| format!("ort run: {e}"))?;

        // Extract hidden states: (batch, seq_len, dim)
        let (_shape, hidden_slice) = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract: {e}"))?;

        // Mean-pool over non-padding tokens, then L2 normalize
        let mut result = Vec::with_capacity(batch_size);
        for (i, enc) in encodings.iter().enumerate() {
            let mask = enc.get_attention_mask();
            let mut pooled = vec![0.0f32; self.dim];
            let mut count = 0usize;

            for (j, &m) in mask.iter().enumerate() {
                if m > 0 {
                    count += 1;
                    let offset = (i * max_len + j) * self.dim;
                    for d in 0..self.dim {
                        pooled[d] += hidden_slice[offset + d];
                    }
                }
            }

            if count > 0 {
                for v in &mut pooled {
                    *v /= count as f32;
                }
            }

            // L2 normalize
            let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut pooled {
                    *v /= norm;
                }
            }

            result.push(pooled);
        }

        Ok(result)
    }
}
