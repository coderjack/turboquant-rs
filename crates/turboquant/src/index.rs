// TurboIndex: append, delete, compact, two-stage search

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use ndarray::Array2;

use crate::compression::{
    polarquant::PolarQuantCompressor,
    qjl::QjlCompressor,
};
use crate::search::{two_stage_search, SearchResult};
use crate::storage::{MmapBitVectors, MmapPolarVectors};
use crate::TurboError;

const DEFAULT_PRE_FILTER_K: usize = 50;

/// Metadata stored alongside the index for re-opening.
#[derive(serde::Serialize, serde::Deserialize)]
struct IndexMeta {
    dim: usize,
    qjl_seed: u64,
    polar_seed: u64,
    angle_bits: u8,
    radius_bits: u8,
    ids: Vec<u64>,
    deleted: Vec<u64>,
    next_id: u64,
}

/// TurboIndex: two-stage vector search with QJL + PolarQuant.
pub struct TurboIndex {
    qjl: QjlCompressor,
    polar: PolarQuantCompressor,
    qjl_storage: MmapBitVectors,
    polar_storage: MmapPolarVectors,
    ids: Vec<u64>,
    deleted: HashSet<u64>,
    next_id: u64,
    path: PathBuf,
    dim: usize,
    qjl_seed: u64,
    polar_seed: u64,
}

impl TurboIndex {
    fn qjl_path(base: &Path) -> PathBuf {
        base.join("index.qjl")
    }

    fn polar_path(base: &Path) -> PathBuf {
        base.join("index.polar")
    }

    fn meta_path(base: &Path) -> PathBuf {
        base.join("index.meta")
    }

    /// Create a new index at the given directory path.
    pub fn create(path: &Path, dim: usize, qjl_seed: u64, polar_seed: u64) -> Result<Self, TurboError> {
        fs::create_dir_all(path)?;

        let qjl = QjlCompressor::new(dim, qjl_seed);
        let polar = PolarQuantCompressor::new(dim, polar_seed, 4, 8);

        let qjl_storage = MmapBitVectors::create(&Self::qjl_path(path), dim)?;
        let polar_storage = MmapPolarVectors::create(&Self::polar_path(path), dim, 4)?;

        let index = Self {
            qjl,
            polar,
            qjl_storage,
            polar_storage,
            ids: Vec::new(),
            deleted: HashSet::new(),
            next_id: 0,
            path: path.to_path_buf(),
            dim,
            qjl_seed,
            polar_seed,
        };

        index.save_meta()?;
        Ok(index)
    }

    /// Open an existing index from a directory.
    pub fn open(path: &Path) -> Result<Self, TurboError> {
        let meta_bytes = fs::read(Self::meta_path(path))?;
        let meta: IndexMeta = bincode::deserialize(&meta_bytes)
            .map_err(|e| TurboError::Storage(format!("failed to deserialize meta: {e}")))?;

        let qjl = QjlCompressor::new(meta.dim, meta.qjl_seed);
        let polar = PolarQuantCompressor::new(meta.dim, meta.polar_seed, meta.angle_bits, meta.radius_bits);

        let qjl_storage = MmapBitVectors::open(&Self::qjl_path(path))?;
        let polar_storage = MmapPolarVectors::open(&Self::polar_path(path))?;

        Ok(Self {
            qjl,
            polar,
            qjl_storage,
            polar_storage,
            ids: meta.ids,
            deleted: meta.deleted.into_iter().collect(),
            next_id: meta.next_id,
            path: path.to_path_buf(),
            dim: meta.dim,
            qjl_seed: meta.qjl_seed,
            polar_seed: meta.polar_seed,
        })
    }

    fn save_meta(&self) -> Result<(), TurboError> {
        let meta = IndexMeta {
            dim: self.dim,
            qjl_seed: self.qjl_seed,
            polar_seed: self.polar_seed,
            angle_bits: self.polar.angle_bits(),
            radius_bits: self.polar.radius_bits(),
            ids: self.ids.clone(),
            deleted: self.deleted.iter().copied().collect(),
            next_id: self.next_id,
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

        let bv = self.qjl.compress(vector);
        let pv = self.polar.compress(vector);

        self.qjl_storage.append(&bv)?;
        self.polar_storage.append(&pv)?;
        self.ids.push(id);

        if id >= self.next_id {
            self.next_id = id + 1;
        }

        self.save_meta()?;
        Ok(())
    }

    /// Insert a batch of vectors. Each row of `vectors` is one vector.
    pub fn insert_batch(&mut self, ids: &[u64], vectors: &Array2<f32>) -> Result<(), TurboError> {
        if vectors.ncols() != self.dim {
            return Err(TurboError::DimensionMismatch {
                expected: self.dim,
                got: vectors.ncols(),
            });
        }
        if ids.len() != vectors.nrows() {
            return Err(TurboError::Storage(
                "ids length must match number of vectors".into(),
            ));
        }

        let bvs = self.qjl.compress_batch(vectors);
        for (i, bv) in bvs.into_iter().enumerate() {
            let pv = self.polar.compress(vectors.row(i).as_slice().unwrap());
            self.qjl_storage.append(&bv)?;
            self.polar_storage.append(&pv)?;
            self.ids.push(ids[i]);

            if ids[i] >= self.next_id {
                self.next_id = ids[i] + 1;
            }
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
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        if query.len() != self.dim || self.ids.is_empty() {
            return vec![];
        }

        // Build in-memory index excluding deleted vectors
        let all_qjl = self.qjl_storage.get_all();
        let mut live_qjl = Vec::new();
        let mut live_polar = Vec::new();
        let mut live_ids = Vec::new();

        for (i, id) in self.ids.iter().enumerate() {
            if !self.deleted.contains(id) {
                live_qjl.push(all_qjl[i].clone());
                live_polar.push(self.polar_storage.get(i));
                live_ids.push(*id);
            }
        }

        if live_ids.is_empty() {
            return vec![];
        }

        let query_qjl = self.qjl.compress(query);
        let query_polar = self.polar.compress(query);

        let pre_filter_k = DEFAULT_PRE_FILTER_K.min(live_ids.len());

        two_stage_search(
            &query_qjl,
            &query_polar,
            &live_qjl,
            &live_polar,
            &live_ids,
            top_k,
            pre_filter_k,
            &self.polar,
        )
    }

    /// Compact: rebuild the index without deleted vectors.
    pub fn compact(&mut self) -> Result<(), TurboError> {
        if self.deleted.is_empty() {
            return Ok(());
        }

        let all_qjl = self.qjl_storage.get_all();
        let old_ids = self.ids.clone();

        // Collect live vectors
        let mut live_qjl = Vec::new();
        let mut live_polar = Vec::new();
        let mut live_ids = Vec::new();

        for (i, id) in old_ids.iter().enumerate() {
            if !self.deleted.contains(id) {
                live_qjl.push(all_qjl[i].clone());
                live_polar.push(self.polar_storage.get(i));
                live_ids.push(*id);
            }
        }

        // Recreate storage files
        let mut new_qjl = MmapBitVectors::create(&Self::qjl_path(&self.path), self.dim)?;
        let mut new_polar = MmapPolarVectors::create(
            &Self::polar_path(&self.path),
            self.dim,
            self.polar.angle_bits(),
        )?;

        for bv in &live_qjl {
            new_qjl.append(bv)?;
        }
        for pv in &live_polar {
            new_polar.append(pv)?;
        }

        self.qjl_storage = new_qjl;
        self.polar_storage = new_polar;
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
        // The closest should be id=1 (same vector)
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

        // Search should not return deleted vector
        let results = index.search(&unit_vec(dim, 1), 3);
        for r in &results {
            assert_ne!(r.id, 2, "deleted vector should not appear in results");
        }

        // Compact
        index.compact().unwrap();
        assert_eq!(index.len(), 2);

        // Re-open and verify
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
    fn test_insert_batch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("idx");
        let dim = 16;

        let mut index = TurboIndex::create(&path, dim, 5, 6).unwrap();

        let ids = vec![100, 200, 300];
        let mut data = vec![0.0f32; 3 * dim];
        data[0] = 1.0;           // vec 0: unit along dim 0
        data[dim + 1] = 1.0;     // vec 1: unit along dim 1
        data[2 * dim + 2] = 1.0; // vec 2: unit along dim 2
        let mat = Array2::from_shape_vec((3, dim), data).unwrap();

        index.insert_batch(&ids, &mat).unwrap();
        assert_eq!(index.len(), 3);

        let results = index.search(&unit_vec(dim, 0), 1);
        assert_eq!(results[0].id, 100);
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
        let result = index.insert(1, &[1.0, 2.0]); // wrong dim
        assert!(result.is_err());
    }
}
