use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

use agent_memory::embedder::{Embedder, MockEmbedder};
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
