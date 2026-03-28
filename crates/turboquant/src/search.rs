// Two-stage search: QJL Hamming pre-filter -> PolarQuant re-rank

use crate::compression::{
    hamming,
    polarquant::{PolarQuantCompressor, PolarVector},
    qjl::BitVector,
};

/// A search result with ID, score, and raw Hamming distance.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
    pub distance: u32,
}

/// Two-stage search: QJL pre-filter -> PolarQuant re-rank.
///
/// 1. Compute Hamming distances from `query_qjl` against all `index_qjl` vectors.
/// 2. Take top `pre_filter_k` candidates.
/// 3. Re-rank those candidates using PolarQuant similarity.
/// 4. Return top `top_k` results sorted by descending score.
pub fn two_stage_search(
    query_qjl: &BitVector,
    query_polar: &PolarVector,
    index_qjl: &[BitVector],
    index_polar: &[PolarVector],
    ids: &[u64],
    top_k: usize,
    pre_filter_k: usize,
    polar_compressor: &PolarQuantCompressor,
) -> Vec<SearchResult> {
    if index_qjl.is_empty() {
        return vec![];
    }

    // Stage 1: QJL Hamming pre-filter
    let effective_pre_k = pre_filter_k.min(index_qjl.len());
    let candidates = hamming::hamming_top_k(query_qjl, index_qjl, effective_pre_k);

    // Stage 2: PolarQuant re-rank
    let mut results: Vec<SearchResult> = candidates
        .iter()
        .map(|&(idx, dist)| {
            let score = polar_compressor.similarity(query_polar, &index_polar[idx]);
            SearchResult {
                id: ids[idx],
                score,
                distance: dist,
            }
        })
        .collect();

    // Sort by score descending
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_k);

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::{polarquant::PolarQuantCompressor, qjl::QjlCompressor};

    #[test]
    fn test_two_stage_search_basic() {
        let dim = 32;
        let qjl = QjlCompressor::new(dim, 42);
        let polar = PolarQuantCompressor::new(dim, 99, 4, 8);

        // Create some vectors
        let mut vectors = Vec::new();
        let mut ids = Vec::new();

        // v0 = [1, 0, 0, ...]
        let mut v0 = vec![0.0f32; dim];
        v0[0] = 1.0;
        vectors.push(v0);
        ids.push(100);

        // v1 close to v0
        let mut v1 = vec![0.0f32; dim];
        let n = (0.95f32 * 0.95 + 0.05 * 0.05).sqrt();
        v1[0] = 0.95 / n;
        v1[1] = 0.05 / n;
        vectors.push(v1);
        ids.push(101);

        // v2 orthogonal to v0
        let mut v2 = vec![0.0f32; dim];
        v2[1] = 1.0;
        vectors.push(v2);
        ids.push(102);

        let index_qjl: Vec<BitVector> = vectors.iter().map(|v| qjl.compress(v)).collect();
        let index_polar: Vec<PolarVector> = vectors.iter().map(|v| polar.compress(v)).collect();

        // Query = v0
        let query_qjl = qjl.compress(&vectors[0]);
        let query_polar = polar.compress(&vectors[0]);

        let results = two_stage_search(
            &query_qjl,
            &query_polar,
            &index_qjl,
            &index_polar,
            &ids,
            2,
            3,
            &polar,
        );

        assert_eq!(results.len(), 2);
        // The top result should be v0 itself (id=100) or v1 (id=101)
        assert!(
            results[0].id == 100 || results[0].id == 101,
            "top result id = {}",
            results[0].id
        );
    }

    #[test]
    fn test_empty_index() {
        let dim = 16;
        let polar = PolarQuantCompressor::new(dim, 1, 4, 8);
        let qjl = crate::compression::qjl::QjlCompressor::new(dim, 1);

        let mut v = vec![0.0f32; dim];
        v[0] = 1.0;

        let results = two_stage_search(
            &qjl.compress(&v),
            &polar.compress(&v),
            &[],
            &[],
            &[],
            5,
            10,
            &polar,
        );
        assert!(results.is_empty());
    }
}
