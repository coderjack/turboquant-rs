use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

use agent_memory::embedder::{Embedder, MockEmbedder, OnnxEmbedder};
use agent_memory::persistent::PersistentMemory;
use agent_memory::AgentMemoryError;

/// Server state: manages one PersistentMemory instance per project directory.
pub struct ServerState {
    memories: RwLock<HashMap<String, Arc<RwLock<PersistentMemory>>>>,
    embedder: Arc<dyn Embedder>,
}

impl ServerState {
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        Self {
            memories: RwLock::new(HashMap::new()),
            embedder,
        }
    }

    /// Create a ServerState with the default MockEmbedder (dim=384).
    pub fn with_mock_embedder() -> Self {
        Self::new(Arc::new(MockEmbedder::new(384)))
    }

    /// Create a ServerState with auto-detected embedder.
    ///
    /// Looks for an ONNX model in this order:
    /// 1. `MEMORY_MODEL_DIR` env var
    /// 2. `<exe_dir>/../models/minilm/`
    /// 3. `~/.cache/agent-memory/models/minilm/`
    ///
    /// Falls back to MockEmbedder if no model is found.
    pub fn auto() -> Self {
        match Self::try_load_onnx() {
            Ok(embedder) => {
                tracing::info!("Loaded ONNX embedder (dim={})", embedder.dim());
                Self::new(Arc::new(embedder))
            }
            Err(reason) => {
                tracing::warn!("ONNX model not available ({reason}), using MockEmbedder — semantic recall will be degraded");
                Self::with_mock_embedder()
            }
        }
    }

    fn try_load_onnx() -> Result<OnnxEmbedder, String> {
        let candidates: Vec<PathBuf> = [
            // 1. Explicit env var
            std::env::var("MEMORY_MODEL_DIR").ok().map(PathBuf::from),
            // 2. Relative to binary: <exe>/../models/minilm
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("../models/minilm"))),
            // 3. User cache directory
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".cache/agent-memory/models/minilm")),
        ]
        .into_iter()
        .flatten()
        .collect();

        for dir in &candidates {
            let has_onnx = dir
                .read_dir()
                .ok()
                .and_then(|mut d| {
                    d.find(|e| {
                        e.as_ref()
                            .ok()
                            .map_or(false, |e| e.path().extension().is_some_and(|ext| ext == "onnx"))
                    })
                })
                .is_some();

            if has_onnx {
                tracing::info!("Found ONNX model at {}", dir.display());
                return OnnxEmbedder::load(dir, 384)
                    .map_err(|e| format!("failed to load {}: {e}", dir.display()));
            }
        }

        Err(format!(
            "no ONNX model found in any of: {}",
            candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
        ))
    }

    /// Get or create a PersistentMemory for the given project directory.
    /// Memory indexes are stored at `~/.cache/agent-memory/<hash>/`.
    pub async fn get_memory(
        &self,
        project_dir: &str,
    ) -> Result<Arc<RwLock<PersistentMemory>>, AgentMemoryError> {
        // Fast path: check if already cached.
        {
            let memories = self.memories.read().await;
            if let Some(mem) = memories.get(project_dir) {
                return Ok(Arc::clone(mem));
            }
        }

        // Slow path: create and cache.
        let mut memories = self.memories.write().await;
        // Double-check after acquiring write lock.
        if let Some(mem) = memories.get(project_dir) {
            return Ok(Arc::clone(mem));
        }

        let storage_dir = storage_dir_for(project_dir);
        let mem = PersistentMemory::open(&storage_dir, Arc::clone(&self.embedder))?;
        let mem = Arc::new(RwLock::new(mem));
        memories.insert(project_dir.to_string(), Arc::clone(&mem));
        Ok(mem)
    }
}

/// Compute the storage directory for a project: `~/.cache/agent-memory/<sha256[:16]>/`
fn storage_dir_for(project_dir: &str) -> PathBuf {
    let hash = Sha256::digest(project_dir.as_bytes());
    let short_hash = format!("{:x}", hash);
    let short_hash = &short_hash[..16.min(short_hash.len())];

    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("agent-memory")
        .join(short_hash)
}

/// Resolve the project directory: use the provided value or fall back to CWD.
pub fn resolve_project_dir(project_dir: Option<&str>) -> String {
    project_dir
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| ".".to_string())
        })
}
