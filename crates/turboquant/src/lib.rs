pub mod compression;
pub mod index;
pub mod search;
pub mod storage;

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
pub use compression::hamming::{hamming_distance, hamming_top_k};
pub use compression::polarquant::{PolarQuantCompressor, PolarVector};
pub use compression::qjl::{BitVector, QjlCompressor};
pub use compression::turboquant_mse::{TqMseCompressor, TqMseVector};
pub use index::TurboIndex;
pub use search::{two_stage_search, SearchResult};
pub use storage::{MmapBitVectors, MmapPolarVectors, MmapTqMseVectors};
