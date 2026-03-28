// ONNX Runtime embedder: bge-code-v1 (768 dims)

use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbedderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("embedder error: {0}")]
    Embedder(String),
}

/// ONNX Runtime embedder backed by bge-code-v1 (768-dim).
///
/// Expected directory layout:
/// ```
/// model_dir/
///   *.onnx          (the exported model)
///   tokenizer.json  (HuggingFace tokenizer)
/// ```
///
/// Inputs: `input_ids`, `attention_mask`, optionally `token_type_ids`.
/// Output: `last_hidden_state` → mean-pooled + L2-normalized.
pub struct OnnxEmbedder {
    session: ort::Session,
    tokenizer: tokenizers::Tokenizer,
    dim: usize,
    has_token_type_ids: bool,
}

impl OnnxEmbedder {
    pub const DIM: usize = 768;

    /// Load the bge-code-v1 model from `model_dir`.
    pub fn load(model_dir: &std::path::Path) -> Result<Self, EmbedderError> {
        let model_path = std::fs::read_dir(model_dir)
            .map_err(EmbedderError::Io)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().map_or(false, |ext| ext == "onnx"))
            .ok_or_else(|| {
                EmbedderError::Embedder(format!(
                    "no .onnx file found in {}",
                    model_dir.display()
                ))
            })?;

        let tokenizer_path = model_dir.join("tokenizer.json");

        let session = ort::Session::builder()
            .map_err(|e| EmbedderError::Embedder(format!("ort builder: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| EmbedderError::Embedder(format!("ort load: {e}")))?;

        let has_token_type_ids = session.inputs.iter().any(|i| i.name == "token_type_ids");

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedderError::Embedder(format!("tokenizer: {e}")))?;

        Ok(Self { session, tokenizer, dim: Self::DIM, has_token_type_ids })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        Ok(self.embed_batch(&[text])?.remove(0))
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        let strings: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let encodings = self
            .tokenizer
            .encode_batch(strings, true)
            .map_err(|e| EmbedderError::Embedder(format!("encode: {e}")))?;

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

        let input_ids =
            ndarray::Array2::from_shape_vec((batch_size, max_len), input_ids_data)
                .map_err(|e| EmbedderError::Embedder(e.to_string()))?;
        let attn_mask =
            ndarray::Array2::from_shape_vec((batch_size, max_len), attn_mask_data)
                .map_err(|e| EmbedderError::Embedder(e.to_string()))?;

        let outputs = if self.has_token_type_ids {
            let type_ids =
                ndarray::Array2::from_shape_vec((batch_size, max_len), type_ids_data)
                    .map_err(|e| EmbedderError::Embedder(e.to_string()))?;
            let session_inputs = ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attn_mask,
                "token_type_ids" => type_ids,
            ]
            .map_err(|e| EmbedderError::Embedder(format!("inputs: {e}")))?;
            self.session.run(session_inputs)
        } else {
            let session_inputs = ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attn_mask,
            ]
            .map_err(|e| EmbedderError::Embedder(format!("inputs: {e}")))?;
            self.session.run(session_inputs)
        }
        .map_err(|e| EmbedderError::Embedder(format!("ort run: {e}")))?;

        let hidden = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| EmbedderError::Embedder(format!("extract tensor: {e}")))?;
        let hidden_slice = hidden
            .as_slice()
            .ok_or_else(|| EmbedderError::Embedder("non-contiguous hidden state".into()))?;

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
