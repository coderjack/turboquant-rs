pub mod compression;
pub mod index;
pub mod storage;
pub mod turboquant;

/// Error type for all TurboQuant operations.
#[derive(thiserror::Error, Debug)]
pub enum TurboError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("index not found: {0}")]
    NotFound(u64),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("storage error: {0}")]
    Storage(String),
}

// Re-export core types for convenience.
//
// Primary API (Lloyd-Max based — default):
pub use compression::lloyd_max::{LloydMaxQuantizer, ScalarQuantVector};
pub use compression::qjl::{BitVector, QjlCompressor};
pub use compression::rotation::Rotation;
pub use index::{SearchResult, TurboIndex};
pub use storage::VectorStorage;
pub use turboquant::{LloydMaxCompressor, LloydMaxVector, PreparedQuery};

// PolarQuant variant (kept for benchmarking / backward compatibility):
pub use compression::polarquant::{PolarQuantizer, PolarVector};
pub use turboquant::{TurboQuantCompressor, TurboQuantVector};
