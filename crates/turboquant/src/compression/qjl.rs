// QJL (Quantized Johnson-Lindenstrauss) — Step 3 of TurboQuant
//
// Applied to the RESIDUAL error left over after PolarQuant.
// Projects via a random Gaussian matrix, then extracts sign bits.
// These 1-bit corrections eliminate bias in the inner product estimator,
// making the combined TurboQuant similarity unbiased.
//
// The `project` method returns the full-precision projection (before sign
// extraction) for use in the correction term during search.

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// Packed sign-bit vector — ceil(dim/8) bytes.
#[derive(Clone, Debug)]
pub struct BitVector(pub Vec<u8>);

impl BitVector {
    /// Number of bytes in the packed representation.
    pub fn byte_len(&self) -> usize {
        self.0.len()
    }
}

/// QJL compressor: random Gaussian projection + sign-bit extraction.
///
/// Projection matrix entries are drawn from N(0, 1/d), preserving
/// inner products in expectation (Johnson-Lindenstrauss property).
pub struct QjlCompressor {
    projection_matrix: Array2<f32>,
    dim: usize,
}

impl QjlCompressor {
    /// Create with deterministic seed. Generates d×d Gaussian random matrix.
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, (1.0 / dim as f64).sqrt()).unwrap();

        let data: Vec<f32> = (0..dim * dim)
            .map(|_| normal.sample(&mut rng) as f32)
            .collect();

        let projection_matrix = Array2::from_shape_vec((dim, dim), data).unwrap();

        Self {
            projection_matrix,
            dim,
        }
    }

    /// Return the dimensionality this compressor was created for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of bytes per compressed vector.
    pub fn bytes_per_vector(&self) -> usize {
        (self.dim + 7) / 8
    }

    /// Full-precision projection: S × x (no sign extraction).
    ///
    /// Used to project the query vector for the QJL correction term
    /// during search: correction = ||r|| · √(π/(2d)) · Σ sign_i · (S·q)_i
    pub fn project(&self, vector: &[f32]) -> Vec<f32> {
        assert_eq!(vector.len(), self.dim, "vector length must match dim");
        let x = Array1::from_vec(vector.to_vec());
        self.projection_matrix.dot(&x).to_vec()
    }

    /// Compress one vector: sign(S × x) → packed bits.
    /// Positive → 1, negative/zero → 0.
    pub fn compress(&self, vector: &[f32]) -> BitVector {
        assert_eq!(vector.len(), self.dim, "vector length must match dim");

        let x = Array1::from_vec(vector.to_vec());
        let z = self.projection_matrix.dot(&x);

        let num_bytes = (self.dim + 7) / 8;
        let mut bits = vec![0u8; num_bytes];

        for (i, &val) in z.iter().enumerate() {
            if val > 0.0 {
                bits[i / 8] |= 1 << (i % 8);
            }
        }

        BitVector(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_seed() {
        let c1 = QjlCompressor::new(64, 42);
        let c2 = QjlCompressor::new(64, 42);
        assert_eq!(c1.projection_matrix, c2.projection_matrix);
    }

    #[test]
    fn test_compress_basic() {
        let dim = 32;
        let comp = QjlCompressor::new(dim, 123);
        let v: Vec<f32> = (0..dim)
            .map(|i| if i == 0 { 1.0 } else { 0.0 })
            .collect();
        let bv = comp.compress(&v);
        assert_eq!(bv.byte_len(), 4); // 32/8 = 4
    }

    #[test]
    fn test_project_preserves_inner_product() {
        // JL property: E[<Sx, Sy>] = <x, y>
        let dim = 128;
        let comp = QjlCompressor::new(dim, 42);

        let mut v1 = vec![0.0f32; dim];
        v1[0] = 1.0;
        let mut v2 = vec![0.0f32; dim];
        v2[0] = 0.8;
        v2[1] = 0.6;

        let true_dot: f32 = v1.iter().zip(&v2).map(|(a, b)| a * b).sum();
        let p1 = comp.project(&v1);
        let p2 = comp.project(&v2);
        let proj_dot: f32 = p1.iter().zip(&p2).map(|(a, b)| a * b).sum();

        // Allow some variance — JL is approximate
        assert!(
            (proj_dot - true_dot).abs() < 0.5,
            "projected dot = {proj_dot}, true dot = {true_dot}"
        );
    }

    #[test]
    fn test_bytes_per_vector() {
        let comp = QjlCompressor::new(384, 42);
        assert_eq!(comp.bytes_per_vector(), 48); // 384/8 = 48
    }
}
