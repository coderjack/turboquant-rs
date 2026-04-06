// TurboQuant — full pipeline composing all three steps
//
// Default (LloydMaxCompressor — Algorithm 1+2 from the paper):
//   Step 1: Random orthogonal rotation  (rotation.rs)
//   Step 2: Lloyd-Max scalar quantization (lloyd_max.rs)
//   Step 3: QJL 1-bit residual correction (qjl.rs)
//
// Alternative (TurboQuantCompressor — PolarQuant variant):
//   Step 1: Random orthogonal rotation  (rotation.rs)
//   Step 2: PolarQuant quantization     (polarquant.rs)
//   Step 3: QJL 1-bit residual correction (qjl.rs)
//
// Based on: Zandieh, Daliri, Hadian, Mirrokni (2025).
// "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate."
// arXiv:2504.19874.
//
// The rotation decorrelates coordinates so each follows ≈ N(0, 1/√d).
// Lloyd-Max finds the MSE-optimal scalar quantization centroids for this
// distribution. QJL corrects the residual with 1 bit per dimension,
// making the inner product estimator unbiased.

use crate::compression::lloyd_max::{LloydMaxQuantizer, ScalarQuantVector};
use crate::compression::polarquant::{PolarQuantizer, PolarVector};
use crate::compression::qjl::{BitVector, QjlCompressor};
use crate::compression::rotation::Rotation;

/// Compressed vector produced by the full TurboQuant pipeline.
#[derive(Clone, Debug)]
pub struct TurboQuantVector {
    /// PolarQuant compressed data (angles + radii in rotated domain).
    pub polar: PolarVector,
    /// QJL sign bits of the residual (1 bit per dimension).
    pub residual_signs: BitVector,
    /// L2 norm of the residual in the rotated domain.
    pub residual_norm: f32,
    /// Original L2 norm of the input vector.
    pub original_norm: f32,
}

impl TurboQuantVector {
    /// Total serialized byte size.
    pub fn byte_size(&self) -> usize {
        // polar data + residual signs + 4 bytes residual_norm + 4 bytes original_norm
        self.polar.byte_size() + self.residual_signs.byte_len() + 8
    }
}

/// Pre-computed query projections for efficient batch similarity.
///
/// Computing these is O(d²) due to the rotation and QJL projection.
/// Call `prepare_query` once per search query, then `similarity_prepared`
/// for each candidate — the per-candidate cost is only O(d).
pub struct PreparedQuery {
    /// Q × query (rotated into PolarQuant domain).
    pub rotated: Vec<f32>,
    /// S × (Q × query) (projected for QJL correction).
    pub qjl_projected: Vec<f32>,
}

/// TurboQuant compressor — the full three-step pipeline.
pub struct TurboQuantCompressor {
    rotation: Rotation,
    polar: PolarQuantizer,
    qjl: QjlCompressor,
    dim: usize,
}

impl TurboQuantCompressor {
    /// Create a new compressor.
    ///
    /// - `rotation_seed`: deterministic seed for the orthogonal rotation matrix
    /// - `qjl_seed`: deterministic seed for the QJL projection matrix
    /// - `angle_bits`: bits per angle in PolarQuant (typically 4)
    /// - `radius_bits`: bits per radius in PolarQuant (typically 8)
    pub fn new(
        dim: usize,
        rotation_seed: u64,
        qjl_seed: u64,
        angle_bits: u8,
        radius_bits: u8,
    ) -> Self {
        Self {
            rotation: Rotation::new(dim, rotation_seed),
            polar: PolarQuantizer::new(dim, angle_bits, radius_bits),
            qjl: QjlCompressor::new(dim, qjl_seed),
            dim,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn polar(&self) -> &PolarQuantizer {
        &self.polar
    }

    /// Total bytes per compressed vector.
    pub fn bytes_per_vector(&self) -> usize {
        // polar + qjl signs + residual_norm(4) + original_norm(4)
        self.polar.bytes_per_vector() + self.qjl.bytes_per_vector() + 8
    }

    /// Compress a vector using the full TurboQuant pipeline.
    pub fn compress(&self, vector: &[f32]) -> TurboQuantVector {
        assert_eq!(vector.len(), self.dim);

        // Compute original norm and normalize to unit sphere.
        let original_norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        let unit: Vec<f32> = if original_norm > 1e-10 {
            vector.iter().map(|&v| v / original_norm).collect()
        } else {
            vec![0.0; self.dim]
        };

        // Step 1: Rotate
        let rotated = self.rotation.rotate(&unit);

        // Step 2: PolarQuant
        let polar = self.polar.compress(&rotated);
        let reconstructed_rotated = self.polar.decompress(&polar);

        // Step 3: Compute residual in rotated domain, apply QJL
        let residual: Vec<f32> = rotated
            .iter()
            .zip(&reconstructed_rotated)
            .map(|(a, b)| a - b)
            .collect();
        let residual_norm: f32 = residual.iter().map(|v| v * v).sum::<f32>().sqrt();
        let residual_signs = self.qjl.compress(&residual);

        TurboQuantVector {
            polar,
            residual_signs,
            residual_norm,
            original_norm,
        }
    }

    /// Decompress: PolarQuant decompress → inverse rotate → scale by norm.
    ///
    /// Note: QJL residual is not added back (it's 1-bit, lossy).
    /// This reconstructs the PolarQuant approximation only.
    pub fn decompress(&self, tv: &TurboQuantVector) -> Vec<f32> {
        let rotated = self.polar.decompress(&tv.polar);
        let unit = self.rotation.inverse_rotate(&rotated);
        unit.iter().map(|&v| v * tv.original_norm).collect()
    }

    /// Pre-compute query projections for efficient batch search.
    ///
    /// Call once per search query. The returned `PreparedQuery` can be
    /// used with `similarity_prepared` for each candidate at O(d) cost.
    pub fn prepare_query(&self, query: &[f32]) -> PreparedQuery {
        assert_eq!(query.len(), self.dim);
        let rotated = self.rotation.rotate(query);
        let qjl_projected = self.qjl.project(&rotated);
        PreparedQuery {
            rotated,
            qjl_projected,
        }
    }

    /// Compute similarity using pre-computed query projections.
    ///
    /// Unbiased estimator:
    ///   <query, x> ≈ ‖x‖ · (<q_rot, ŷ> + ‖r‖ · √(π/(2d)) · Σ sign_i · (S·q_rot)_i)
    ///
    /// where ŷ is the PolarQuant reconstruction and r is the residual.
    pub fn similarity_prepared(
        &self,
        prepared: &PreparedQuery,
        compressed: &TurboQuantVector,
    ) -> f32 {
        // PolarQuant dot product in rotated domain
        let polar_sim = self.polar.similarity_raw(&prepared.rotated, &compressed.polar);

        // QJL correction for the residual
        let correction = if compressed.residual_norm > 1e-10 {
            let scale = compressed.residual_norm
                * (std::f32::consts::PI / (2.0 * self.dim as f32)).sqrt();

            let mut dot = 0.0f32;
            for i in 0..self.dim {
                let sign =
                    if (compressed.residual_signs.0[i / 8] >> (i % 8)) & 1 == 1 {
                        1.0f32
                    } else {
                        -1.0f32
                    };
                dot += sign * prepared.qjl_projected[i];
            }

            scale * dot
        } else {
            0.0
        };

        // Scale by original norm: <query, x> = ‖x‖ · <query, x_unit>
        compressed.original_norm * (polar_sim + correction)
    }

    /// Convenience: compute similarity without pre-computing projections.
    ///
    /// For single comparisons. Use `prepare_query` + `similarity_prepared`
    /// when comparing against multiple compressed vectors.
    pub fn similarity_raw(&self, query: &[f32], compressed: &TurboQuantVector) -> f32 {
        let prepared = self.prepare_query(query);
        self.similarity_prepared(&prepared, compressed)
    }
}

// ===========================================================================
// LloydMaxCompressor — Algorithm 1 variant (Rotation + Lloyd-Max + QJL)
// ===========================================================================

/// Compressed vector produced by the Lloyd-Max pipeline.
#[derive(Clone, Debug)]
pub struct LloydMaxVector {
    /// Lloyd-Max scalar quantized data in rotated domain.
    pub scalar: ScalarQuantVector,
    /// QJL sign bits of the residual.
    pub residual_signs: BitVector,
    /// L2 norm of the residual in the rotated domain.
    pub residual_norm: f32,
    /// Original L2 norm of the input vector.
    pub original_norm: f32,
}

impl LloydMaxVector {
    pub fn byte_size(&self) -> usize {
        self.scalar.byte_size() + self.residual_signs.byte_len() + 8
    }
}

/// TurboQuant compressor using Lloyd-Max scalar quantization (Algorithm 1).
///
/// Same three-step pipeline as `TurboQuantCompressor`, but replaces PolarQuant
/// with Lloyd-Max optimal scalar quantization in step 2.
pub struct LloydMaxCompressor {
    rotation: Rotation,
    lloyd_max: LloydMaxQuantizer,
    qjl: QjlCompressor,
    dim: usize,
}

impl LloydMaxCompressor {
    pub fn new(dim: usize, rotation_seed: u64, qjl_seed: u64, bits: u8) -> Self {
        Self {
            rotation: Rotation::new(dim, rotation_seed),
            lloyd_max: LloydMaxQuantizer::new(dim, bits),
            qjl: QjlCompressor::new(dim, qjl_seed),
            dim,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn lloyd_max(&self) -> &LloydMaxQuantizer {
        &self.lloyd_max
    }

    /// Total bytes per compressed vector.
    pub fn bytes_per_vector(&self) -> usize {
        self.lloyd_max.bytes_per_vector() + self.qjl.bytes_per_vector() + 8
    }

    /// Compress using Rotation → Lloyd-Max → QJL residual.
    pub fn compress(&self, vector: &[f32]) -> LloydMaxVector {
        assert_eq!(vector.len(), self.dim);

        let original_norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        let unit: Vec<f32> = if original_norm > 1e-10 {
            vector.iter().map(|&v| v / original_norm).collect()
        } else {
            vec![0.0; self.dim]
        };

        let rotated = self.rotation.rotate(&unit);
        let scalar = self.lloyd_max.compress(&rotated);
        let reconstructed_rotated = self.lloyd_max.decompress(&scalar);

        let residual: Vec<f32> = rotated
            .iter()
            .zip(&reconstructed_rotated)
            .map(|(a, b)| a - b)
            .collect();
        let residual_norm: f32 = residual.iter().map(|v| v * v).sum::<f32>().sqrt();
        let residual_signs = self.qjl.compress(&residual);

        LloydMaxVector {
            scalar,
            residual_signs,
            residual_norm,
            original_norm,
        }
    }

    /// Decompress: Lloyd-Max decompress → inverse rotate → scale by norm.
    pub fn decompress(&self, lv: &LloydMaxVector) -> Vec<f32> {
        let rotated = self.lloyd_max.decompress(&lv.scalar);
        let unit = self.rotation.inverse_rotate(&rotated);
        unit.iter().map(|&v| v * lv.original_norm).collect()
    }

    /// Pre-compute query projections (same as TurboQuantCompressor).
    pub fn prepare_query(&self, query: &[f32]) -> PreparedQuery {
        assert_eq!(query.len(), self.dim);
        let rotated = self.rotation.rotate(query);
        let qjl_projected = self.qjl.project(&rotated);
        PreparedQuery {
            rotated,
            qjl_projected,
        }
    }

    /// Similarity using the unbiased estimator.
    pub fn similarity_prepared(
        &self,
        prepared: &PreparedQuery,
        compressed: &LloydMaxVector,
    ) -> f32 {
        let scalar_sim = self
            .lloyd_max
            .similarity_raw(&prepared.rotated, &compressed.scalar);

        let correction = if compressed.residual_norm > 1e-10 {
            let scale = compressed.residual_norm
                * (std::f32::consts::PI / (2.0 * self.dim as f32)).sqrt();

            let mut dot = 0.0f32;
            for i in 0..self.dim {
                let sign =
                    if (compressed.residual_signs.0[i / 8] >> (i % 8)) & 1 == 1 {
                        1.0f32
                    } else {
                        -1.0f32
                    };
                dot += sign * prepared.qjl_projected[i];
            }
            scale * dot
        } else {
            0.0
        };

        compressed.original_norm * (scalar_sim + correction)
    }

    pub fn similarity_raw(&self, query: &[f32], compressed: &LloydMaxVector) -> f32 {
        let prepared = self.prepare_query(query);
        self.similarity_prepared(&prepared, compressed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, Normal};

    fn random_unit_vec(dim: usize, rng: &mut ChaCha8Rng) -> Vec<f32> {
        let normal = Normal::new(0.0, 1.0f64).unwrap();
        let raw: Vec<f32> = (0..dim).map(|_| normal.sample(rng) as f32).collect();
        let norm: f32 = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        raw.iter().map(|v| v / norm).collect()
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let dim = 64;
        let comp = TurboQuantCompressor::new(dim, 42, 99, 4, 8);

        let mut v = vec![0.0f32; dim];
        v[0] = 1.0;

        let tv = comp.compress(&v);
        assert!((tv.original_norm - 1.0).abs() < 1e-6);
        assert!(tv.residual_norm >= 0.0);

        let reconstructed = comp.decompress(&tv);
        assert_eq!(reconstructed.len(), dim);

        let dot: f32 = v.iter().zip(&reconstructed).map(|(a, b)| a * b).sum();
        let norm_r: f32 = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = if norm_r > 0.0 { dot / norm_r } else { 0.0 };
        assert!(cosine > 0.7, "roundtrip cosine = {cosine}, expected > 0.7");
    }

    #[test]
    fn test_similarity_ordering() {
        let dim = 64;
        let comp = TurboQuantCompressor::new(dim, 55, 77, 4, 8);

        let mut v1 = vec![0.0f32; dim];
        v1[0] = 1.0;

        let mut v2 = vec![0.0f32; dim];
        let n = (0.9f32 * 0.9 + 0.1 * 0.1).sqrt();
        v2[0] = 0.9 / n;
        v2[1] = 0.1 / n;

        let mut v3 = vec![0.0f32; dim];
        v3[1] = 1.0;

        let tv2 = comp.compress(&v2);
        let tv3 = comp.compress(&v3);

        let sim12 = comp.similarity_raw(&v1, &tv2);
        let sim13 = comp.similarity_raw(&v1, &tv3);

        assert!(
            sim12 > sim13,
            "similar vectors should score higher: sim12={sim12}, sim13={sim13}"
        );
    }

    #[test]
    fn test_prepared_matches_raw() {
        let dim = 32;
        let comp = TurboQuantCompressor::new(dim, 42, 99, 4, 8);
        let mut rng = ChaCha8Rng::seed_from_u64(1);

        let query = random_unit_vec(dim, &mut rng);
        let target = random_unit_vec(dim, &mut rng);
        let tv = comp.compress(&target);

        let sim_raw = comp.similarity_raw(&query, &tv);
        let prepared = comp.prepare_query(&query);
        let sim_prepared = comp.similarity_prepared(&prepared, &tv);

        assert!(
            (sim_raw - sim_prepared).abs() < 1e-6,
            "raw={sim_raw}, prepared={sim_prepared}"
        );
    }

    #[test]
    fn test_similarity_preserves_ranking() {
        let dim = 384;
        let comp = TurboQuantCompressor::new(dim, 42, 99, 4, 8);
        let mut rng = ChaCha8Rng::seed_from_u64(77);

        let query = random_unit_vec(dim, &mut rng);
        let vecs: Vec<Vec<f32>> = (0..50).map(|_| random_unit_vec(dim, &mut rng)).collect();

        let true_sims: Vec<f32> = vecs
            .iter()
            .map(|v| query.iter().zip(v).map(|(a, b)| a * b).sum())
            .collect();

        let prepared = comp.prepare_query(&query);
        let compressed: Vec<_> = vecs.iter().map(|v| comp.compress(v)).collect();
        let comp_sims: Vec<f32> = compressed
            .iter()
            .map(|tv| comp.similarity_prepared(&prepared, tv))
            .collect();

        // Top-5 by true similarity
        let mut true_ranking: Vec<(usize, f32)> =
            true_sims.iter().copied().enumerate().collect();
        true_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let true_top5: Vec<usize> = true_ranking[..5].iter().map(|(i, _)| *i).collect();

        // Top-5 by compressed similarity
        let mut comp_ranking: Vec<(usize, f32)> =
            comp_sims.iter().copied().enumerate().collect();
        comp_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let comp_top5: Vec<usize> = comp_ranking[..5].iter().map(|(i, _)| *i).collect();

        let overlap = true_top5
            .iter()
            .filter(|i| comp_top5.contains(i))
            .count();

        assert!(
            overlap >= 2,
            "at least 2 of true top-5 should appear in compressed top-5, got {overlap}/5"
        );
    }

    #[test]
    fn test_byte_sizes() {
        // 384-dim: polar(288) + qjl(48) + norms(8) = 344
        let comp = TurboQuantCompressor::new(384, 42, 99, 4, 8);
        assert_eq!(comp.bytes_per_vector(), 344);
    }

    #[test]
    fn test_deterministic() {
        let c1 = TurboQuantCompressor::new(32, 42, 99, 4, 8);
        let c2 = TurboQuantCompressor::new(32, 42, 99, 4, 8);

        let v = vec![1.0f32; 32];
        let tv1 = c1.compress(&v);
        let tv2 = c2.compress(&v);

        assert_eq!(tv1.polar.angles, tv2.polar.angles);
        assert_eq!(tv1.polar.radii, tv2.polar.radii);
        assert_eq!(tv1.residual_signs.0, tv2.residual_signs.0);
        assert_eq!(tv1.residual_norm, tv2.residual_norm);
        assert_eq!(tv1.original_norm, tv2.original_norm);
    }
}
