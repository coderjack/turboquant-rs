pub mod budget;
pub mod document;
pub mod embedder;
pub mod persistent;
pub mod ranker;
pub mod session;

#[derive(thiserror::Error, Debug)]
pub enum AgentMemoryError {
    #[error("turboquant error: {0}")]
    Turbo(#[from] turboquant::TurboError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("embedder error: {0}")]
    Embedder(String),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("not found: {0}")]
    NotFound(String),
}

// Re-export core types for convenience.
pub use document::{Document, DocumentType, MemoryStats, RecallResult, SyncStats};
pub use embedder::{Embedder, MockEmbedder, OnnxEmbedder};
pub use persistent::PersistentMemory;
pub use session::SessionMemory;
