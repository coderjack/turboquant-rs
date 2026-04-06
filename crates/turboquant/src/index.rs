// TurboIndex: append, delete, compact, brute-force search
//
// Stores vectors compressed with the full TurboQuant pipeline
// (rotation + Lloyd-Max scalar quantization + QJL residual).
// Search is brute-force with O(d) per-candidate cost after an
// O(d²) query preparation.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::storage::VectorStorage;
use crate::turboquant::LloydMaxCompressor;
use crate::TurboError;

/// A search result with ID and similarity score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
}

/// Metadata stored alongside the index for re-opening.
#[derive(serde::Serialize, serde::Deserialize)]
struct IndexMeta {
    dim: usize,
    rotation_seed: u64,
    qjl_seed: u64,
    bits: u8,
    ids: Vec<u64>,
    deleted: Vec<u64>,
    next_id: u64,
    version: u8,
}

/// TurboIndex: vector search using the TurboQuant pipeline.
///
/// Default quantizer: Lloyd-Max scalar (Algorithm 1 from the TurboQuant paper)
/// with QJL residual correction for unbiased inner product estimation.
pub struct TurboIndex {
    compressor: LloydMaxCompressor,
    storage: VectorStorage,
    ids: Vec<u64>,
    deleted: HashSet<u64>,
    next_id: u64,
    path: PathBuf,
    dim: usize,
    rotation_seed: u64,
    qjl_seed: u64,
    bits: u8,
}

impl TurboIndex {
    fn storage_path(base: &Path) -> PathBuf {
        base.join("index.tqlm")
    }

    fn meta_path(base: &Path) -> PathBuf {
        base.join("index.meta")
    }

    /// Create a new index at the given directory path.
    ///
    /// Default: 4-bit Lloyd-Max quantization (6.2x compression at 384-dim).
    pub fn create(
        path: &Path,
        dim: usize,
        rotation_seed: u64,
        qjl_seed: u64,
    ) -> Result<Self, TurboError> {
        Self::create_with_bits(path, dim, rotation_seed, qjl_seed, 4)
    }

    /// Create a new index with explicit quantization bit width.
    ///
    /// - `bits`: 1–8. Higher = better quality, lower compression.
    ///   Recommended: 2 (compact), 3 (balanced), 4 (quality).
    pub fn create_with_bits(
        path: &Path,
        dim: usize,
        rotation_seed: u64,
        qjl_seed: u64,
        bits: u8,
    ) -> Result<Self, TurboError> {
        fs::create_dir_all(path)?;

        let compressor = LloydMaxCompressor::new(dim, rotation_seed, qjl_seed, bits);
        let storage =
            VectorStorage::create(&Self::storage_path(path), dim, bits)?;

        let index = Self {
            compressor,
            storage,
            ids: Vec::new(),
            deleted: HashSet::new(),
            next_id: 0,
            path: path.to_path_buf(),
            dim,
            rotation_seed,
            qjl_seed,
            bits,
        };

        index.save_meta()?;
        Ok(index)
    }

    /// Open an existing index from a directory.
    pub fn open(path: &Path) -> Result<Self, TurboError> {
        let meta_bytes = fs::read(Self::meta_path(path))?;
        let meta: IndexMeta = bincode::deserialize(&meta_bytes)
            .map_err(|e| TurboError::Storage(format!("failed to deserialize meta: {e}")))?;

        let compressor = LloydMaxCompressor::new(
            meta.dim,
            meta.rotation_seed,
            meta.qjl_seed,
            meta.bits,
        );
        let storage = VectorStorage::open(&Self::storage_path(path))?;

        Ok(Self {
            compressor,
            storage,
            ids: meta.ids,
            deleted: meta.deleted.into_iter().collect(),
            next_id: meta.next_id,
            path: path.to_path_buf(),
            dim: meta.dim,
            rotation_seed: meta.rotation_seed,
            qjl_seed: meta.qjl_seed,
            bits: meta.bits,
        })
    }

    fn save_meta(&self) -> Result<(), TurboError> {
        let meta = IndexMeta {
            dim: self.dim,
            rotation_seed: self.rotation_seed,
            qjl_seed: self.qjl_seed,
            bits: self.bits,
            ids: self.ids.clone(),
            deleted: self.deleted.iter().copied().collect(),
            next_id: self.next_id,
            version: 2, // v2 = Lloyd-Max format
        };
        let bytes = bincode::serialize(&meta)
            .map_err(|e| TurboError::Storage(format!("failed to serialize meta: {e}")))?;
        fs::write(Self::meta_path(&self.path), bytes)?;
        Ok(())
    }

    /// Insert a single vector with the given ID.
    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), TurboError> {
        if vector.len() != self.dim {
            return Err(TurboError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }

        let lv = self.compressor.compress(vector);
        self.storage.append(&lv)?;
        self.ids.push(id);

        if id >= self.next_id {
            self.next_id = id + 1;
        }

        self.save_meta()?;
        Ok(())
    }

    /// Soft-delete a vector by ID.
    pub fn delete(&mut self, id: u64) -> Result<(), TurboError> {
        if !self.ids.contains(&id) {
            return Err(TurboError::NotFound(id));
        }
        self.deleted.insert(id);
        self.save_meta()?;
        Ok(())
    }

    /// Search for the top-k nearest vectors to the query.
    ///
    /// Uses brute-force scan with the TurboQuant unbiased estimator.
    /// Query preparation (rotation + QJL projection) is O(d²), done once.
    /// Per-candidate similarity is O(d).
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        if query.len() != self.dim || self.ids.is_empty() {
            return vec![];
        }

        let prepared = self.compressor.prepare_query(query);

        let mut results: Vec<SearchResult> = Vec::new();
        for (i, id) in self.ids.iter().enumerate() {
            if self.deleted.contains(id) {
                continue;
            }
            let lv = self.storage.get(i);
            let score = self.compressor.similarity_prepared(&prepared, &lv);
            results.push(SearchResult { id: *id, score });
        }

        results
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Compact: rebuild the index without deleted vectors.
    pub fn compact(&mut self) -> Result<(), TurboError> {
        if self.deleted.is_empty() {
            return Ok(());
        }

        let old_ids = self.ids.clone();
        let mut live_vectors = Vec::new();
        let mut live_ids = Vec::new();

        for (i, id) in old_ids.iter().enumerate() {
            if !self.deleted.contains(id) {
                live_vectors.push(self.storage.get(i));
                live_ids.push(*id);
            }
        }

        let mut new_storage = VectorStorage::create(
            &Self::storage_path(&self.path),
            self.dim,
            self.bits,
        )?;

        for lv in &live_vectors {
            new_storage.append(lv)?;
        }

        self.storage = new_storage;
        self.ids = live_ids;
        self.deleted.clear();
        self.save_meta()?;

        Ok(())
    }

    /// Number of live (non-deleted) vectors.
    pub fn len(&self) -> usize {
        self.ids.len() - self.deleted.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn unit_vec(dim: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[idx] = 1.0;
        v
    }

    #[test]
    fn test_create_insert_search() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("idx");
        let dim = 32;

        let mut index = TurboIndex::create(&path, dim, 42, 99).unwrap();
        index.insert(1, &unit_vec(dim, 0)).unwrap();
        index.insert(2, &unit_vec(dim, 1)).unwrap();
        index.insert(3, &unit_vec(dim, 2)).unwrap();

        assert_eq!(index.len(), 3);

        let results = index.search(&unit_vec(dim, 0), 2);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, 1, "expected id=1 got id={}", results[0].id);
    }

    #[test]
    fn test_delete_and_compact() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("idx");
        let dim = 16;

        let mut index = TurboIndex::create(&path, dim, 10, 20).unwrap();
        index.insert(1, &unit_vec(dim, 0)).unwrap();
        index.insert(2, &unit_vec(dim, 1)).unwrap();
        index.insert(3, &unit_vec(dim, 2)).unwrap();

        assert_eq!(index.len(), 3);

        index.delete(2).unwrap();
        assert_eq!(index.len(), 2);

        let results = index.search(&unit_vec(dim, 1), 3);
        for r in &results {
            assert_ne!(r.id, 2, "deleted vector should not appear in results");
        }

        index.compact().unwrap();
        assert_eq!(index.len(), 2);

        let reopened = TurboIndex::open(&path).unwrap();
        assert_eq!(reopened.len(), 2);
    }

    #[test]
    fn test_open_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("idx");
        let dim = 16;

        {
            let mut index = TurboIndex::create(&path, dim, 1, 2).unwrap();
            index.insert(10, &unit_vec(dim, 0)).unwrap();
            index.insert(20, &unit_vec(dim, 3)).unwrap();
        }

        let index = TurboIndex::open(&path).unwrap();
        assert_eq!(index.len(), 2);

        let results = index.search(&unit_vec(dim, 0), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 10);
    }

    #[test]
    fn test_create_with_bits() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("idx");
        let dim = 32;

        let mut index = TurboIndex::create_with_bits(&path, dim, 42, 99, 2).unwrap();
        index.insert(1, &unit_vec(dim, 0)).unwrap();

        let results = index.search(&unit_vec(dim, 0), 1);
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn test_delete_not_found() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("idx");

        let mut index = TurboIndex::create(&path, 16, 1, 2).unwrap();
        let result = index.delete(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("idx");

        let mut index = TurboIndex::create(&path, 16, 1, 2).unwrap();
        let result = index.insert(1, &[1.0, 2.0]);
        assert!(result.is_err());
    }
}
