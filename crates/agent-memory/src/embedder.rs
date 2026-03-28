use std::sync::Mutex;

use ort::session::Session;
use ort::value::Tensor;
use sha2::{Digest, Sha256};

use crate::AgentMemoryError;

/// Trait for embedding text into fixed-dimensional vectors.
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Vec<f32>;
    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>>;
    fn dim(&self) -> usize;
}

/// Mock embedder for testing. Produces deterministic pseudo-embeddings from
/// the SHA-256 hash of the input text. Identical texts produce identical
/// embeddings; different texts produce (pseudo-)random unit vectors.
pub struct MockEmbedder {
    dim: usize,
}

impl MockEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Embedder for MockEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        // Hash the text with SHA-256 to get 32 bytes of seed material.
        let hash = Sha256::digest(text.as_bytes());

        // Use the hash bytes to seed a simple deterministic PRNG (xorshift64)
        // to generate `dim` f32 values, then L2-normalize.
        let mut state: u64 = 0;
        for chunk in hash.chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            state ^= u64::from_le_bytes(buf);
        }
        // Ensure non-zero state
        if state == 0 {
            state = 0xdeadbeef;
        }

        let mut values = Vec::with_capacity(self.dim);
        for _ in 0..self.dim {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to roughly normal-ish f32 in [-1, 1]
            let val = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
            values.push(val);
        }

        // L2-normalize
        let norm: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut values {
                *v /= norm;
            }
        }

        values
    }

    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// ONNX Runtime embedder. Loads a tokenizer and ONNX model from a directory,
/// runs inference, and returns mean-pooled, L2-normalized embeddings.
///
/// Expected directory layout:
/// ```text
/// model_dir/
///   *.onnx          (the exported model)
///   tokenizer.json  (HuggingFace tokenizer)
/// ```
///
/// Model inputs expected: `input_ids`, `attention_mask`, and optionally
/// `token_type_ids` (detected automatically from the model's input metadata).
/// Model output expected: `last_hidden_state` of shape `[batch, seq, dim]`.
pub struct OnnxEmbedder {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    dim: usize,
    has_token_type_ids: bool,
}

impl OnnxEmbedder {
    /// Load a model from `model_dir`. `dim` must match the model's hidden size
    /// (e.g. 384 for MiniLM, 768 for BGE-code-v1).
    pub fn load(model_dir: &std::path::Path, dim: usize) -> Result<Self, AgentMemoryError> {
        let model_path = std::fs::read_dir(model_dir)
            .map_err(AgentMemoryError::Io)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().map_or(false, |ext| ext == "onnx"))
            .ok_or_else(|| {
                AgentMemoryError::Embedder(format!(
                    "no .onnx file found in {}",
                    model_dir.display()
                ))
            })?;

        let tokenizer_path = model_dir.join("tokenizer.json");

        let session = Session::builder()
            .map_err(|e| AgentMemoryError::Embedder(format!("ort builder: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| AgentMemoryError::Embedder(format!("ort load: {e}")))?;

        let has_token_type_ids = session.inputs().iter().any(|i| i.name() == "token_type_ids");

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| AgentMemoryError::Embedder(format!("tokenizer: {e}")))?;

        Ok(Self { session: Mutex::new(session), tokenizer, dim, has_token_type_ids })
    }

    fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, AgentMemoryError> {
        let strings: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let encodings = self
            .tokenizer
            .encode_batch(strings, true)
            .map_err(|e| AgentMemoryError::Embedder(format!("encode: {e}")))?;

        let batch_size = encodings.len();
        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);

        let mut input_ids_data = vec![0i64; batch_size * max_len];
        let mut attn_mask_data = vec![0i64; batch_size * max_len];
        let mut type_ids_data = vec![0i64; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let types = enc.get_type_ids();
            for j in 0..ids.len() {
                let idx = i * max_len + j;
                input_ids_data[idx] = ids[j] as i64;
                attn_mask_data[idx] = mask[j] as i64;
                type_ids_data[idx] = types[j] as i64;
            }
        }

        let shape = vec![batch_size as i64, max_len as i64];

        let input_ids_tensor = Tensor::from_array((shape.clone(), input_ids_data))
            .map_err(|e| AgentMemoryError::Embedder(format!("input_ids tensor: {e}")))?;
        let attn_mask_tensor = Tensor::from_array((shape.clone(), attn_mask_data))
            .map_err(|e| AgentMemoryError::Embedder(format!("attn_mask tensor: {e}")))?;

        let mut session = self.session.lock()
            .map_err(|e| AgentMemoryError::Embedder(format!("session lock: {e}")))?;

        let outputs = if self.has_token_type_ids {
            let type_ids_tensor = Tensor::from_array((shape, type_ids_data))
                .map_err(|e| AgentMemoryError::Embedder(format!("type_ids tensor: {e}")))?;
            let session_inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attn_mask_tensor,
                "token_type_ids" => type_ids_tensor,
            ];
            session.run(session_inputs)
        } else {
            let session_inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attn_mask_tensor,
            ];
            session.run(session_inputs)
        }
        .map_err(|e| AgentMemoryError::Embedder(format!("ort run: {e}")))?;

        // last_hidden_state: [batch, seq, dim] — mean pool over non-padding tokens
        let hidden = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| AgentMemoryError::Embedder(format!("extract tensor: {e}")))?;
        let hidden_slice = hidden.1;

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

impl Embedder for OnnxEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        self.embed_texts(&[text])
            .expect("ONNX embed failed")
            .remove(0)
    }

    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        self.embed_texts(texts).expect("ONNX embed_batch failed")
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_embedder_deterministic() {
        let emb = MockEmbedder::new(64);
        let v1 = emb.embed("hello world");
        let v2 = emb.embed("hello world");
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_mock_embedder_different_texts_differ() {
        let emb = MockEmbedder::new(64);
        let v1 = emb.embed("hello");
        let v2 = emb.embed("world");
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_mock_embedder_unit_vector() {
        let emb = MockEmbedder::new(128);
        let v = emb.embed("test text");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {}", norm);
    }

    #[test]
    fn test_mock_embedder_dim() {
        let emb = MockEmbedder::new(384);
        assert_eq!(emb.dim(), 384);
        assert_eq!(emb.embed("foo").len(), 384);
    }

    #[test]
    fn test_mock_embedder_batch() {
        let emb = MockEmbedder::new(32);
        let batch = emb.embed_batch(&["a", "b", "c"]);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0], emb.embed("a"));
    }

    #[test]
    fn test_onnx_embedder_load_fails_missing_dir() {
        let result = OnnxEmbedder::load(std::path::Path::new("/nonexistent"), 384);
        assert!(result.is_err());
    }

    #[test]
    fn test_onnx_embedder_load_fails_no_onnx_file() {
        let dir = tempfile::tempdir().unwrap();
        let result = OnnxEmbedder::load(dir.path(), 384);
        assert!(result.is_err());
    }

    /// Integration test — only runs when models/minilm/ exists.
    /// Run `python scripts/export_onnx.py` first.
    #[test]
    fn test_onnx_embedder_real_model() {
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("models/minilm");
        let has_onnx = model_dir
            .read_dir()
            .ok()
            .and_then(|mut d| d.find(|e| {
                e.as_ref().ok().map_or(false, |e| {
                    e.path().extension().map_or(false, |ext| ext == "onnx")
                })
            }))
            .is_some();
        if !has_onnx {
            eprintln!("Skipping ONNX test: run `python scripts/export_onnx.py` first");
            return;
        }

        let emb = OnnxEmbedder::load(&model_dir, 384).expect("failed to load model");
        assert_eq!(emb.dim(), 384);

        // Deterministic
        let v1 = emb.embed("hello world");
        let v2 = emb.embed("hello world");
        assert_eq!(v1.len(), 384);
        assert_eq!(v1, v2);

        // Unit vector
        let norm: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm = {}", norm);

        // Semantic similarity: similar texts closer than dissimilar ones
        let cat = emb.embed("The cat sat on the mat");
        let kitten = emb.embed("A kitten rested on the rug");
        let quantum = emb.embed("Quantum mechanics describes wave functions");

        let sim_similar: f32 = cat.iter().zip(&kitten).map(|(a, b)| a * b).sum();
        let sim_different: f32 = cat.iter().zip(&quantum).map(|(a, b)| a * b).sum();
        assert!(
            sim_similar > sim_different,
            "expected cat/kitten ({sim_similar}) > cat/quantum ({sim_different})"
        );

        // Batch
        let batch = emb.embed_batch(&["hello", "world"]);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0], emb.embed("hello"));
    }
}
