// QJL (Quantized Johnson-Lindenstrauss) compression
// Random projection + sign-bit extraction -> 1-bit per dimension

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// Packed bit vector -- ceil(dim/8) bytes.
#[derive(Clone, Debug)]
pub struct BitVector(pub Vec<u8>);

impl BitVector {
    /// Number of bytes in the packed representation.
    pub fn byte_len(&self) -> usize {
        self.0.len()
    }
}

/// QJL compressor: random Gaussian projection + sign-bit extraction.
pub struct QjlCompressor {
    projection_matrix: Array2<f32>, // R in R^(d x d), R[i,j] ~ N(0, 1/d)
    dim: usize,
}

impl QjlCompressor {
    /// Create with deterministic seed. Generates d x d Gaussian random matrix.
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, (1.0 / dim as f64).sqrt()).unwrap();

        let mut data = Vec::with_capacity(dim * dim);
        for _ in 0..(dim * dim) {
            data.push(normal.sample(&mut rng) as f32);
        }

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

    /// Compress one vector: sign(R @ x) -> packed bits.
    /// Positive -> 1, negative/zero -> 0.
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

    /// Batch compress: each row of `vectors` is one vector.
    pub fn compress_batch(&self, vectors: &Array2<f32>) -> Vec<BitVector> {
        assert_eq!(vectors.ncols(), self.dim, "vector dim must match");

        // Z = vectors @ R^T  (each row of Z is R @ x_i)
        let z = vectors.dot(&self.projection_matrix.t());

        let num_bytes = (self.dim + 7) / 8;
        let n = vectors.nrows();
        let mut result = Vec::with_capacity(n);

        for row_idx in 0..n {
            let mut bits = vec![0u8; num_bytes];
            for (i, &val) in z.row(row_idx).iter().enumerate() {
                if val > 0.0 {
                    bits[i / 8] |= 1 << (i % 8);
                }
            }
            result.push(BitVector(bits));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

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
        let v: Vec<f32> = (0..dim).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let bv = comp.compress(&v);
        assert_eq!(bv.byte_len(), 4); // 32/8 = 4
    }

    #[test]
    fn test_hamming_correlates_with_cosine() {
        use crate::compression::hamming::hamming_distance;

        let dim = 128;
        let comp = QjlCompressor::new(dim, 99);

        // Two identical vectors should have Hamming distance 0
        let v: Vec<f32> = {
            let mut v = vec![0.0f32; dim];
            v[0] = 1.0;
            v
        };
        let bv1 = comp.compress(&v);
        let bv2 = comp.compress(&v);
        assert_eq!(hamming_distance(&bv1, &bv2), 0);

        // Orthogonal vectors should have Hamming distance ~dim/2
        let mut v2 = vec![0.0f32; dim];
        v2[1] = 1.0;
        let bv3 = comp.compress(&v2);
        let dist = hamming_distance(&bv1, &bv3);
        // Expect roughly dim/2 = 64, allow wide margin for randomness
        assert!(dist > 30 && dist < 100, "dist={dist} expected ~64");
    }

    #[test]
    fn test_batch_matches_single() {
        let dim = 64;
        let comp = QjlCompressor::new(dim, 77);

        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f32> = (0..3 * dim).map(|_| normal.sample(&mut rng) as f32).collect();
        let mat = Array2::from_shape_vec((3, dim), data).unwrap();

        let batch = comp.compress_batch(&mat);
        for i in 0..3 {
            let single = comp.compress(mat.row(i).as_slice().unwrap());
            assert_eq!(batch[i].0, single.0);
        }
    }
}
