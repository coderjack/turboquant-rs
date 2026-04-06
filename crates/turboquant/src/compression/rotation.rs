// Random orthogonal rotation — Step 1 of TurboQuant
//
// Generates a uniformly random orthogonal matrix via Modified Gram-Schmidt
// on a Gaussian random matrix. Deterministic given the seed.
//
// The rotation spreads "energy" of any spike in the vector evenly across
// all dimensions, so each coordinate of a unit-sphere vector concentrates
// around N(0, 1/d) in high dimensions.

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// Random orthogonal rotation matrix.
pub struct Rotation {
    matrix: Array2<f32>,
    dim: usize,
}

impl Rotation {
    /// Generate a random orthogonal matrix of size dim×dim.
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0f64).unwrap();

        let data: Vec<f32> = (0..dim * dim)
            .map(|_| normal.sample(&mut rng) as f32)
            .collect();
        let mut q = Array2::from_shape_vec((dim, dim), data).unwrap();

        // Modified Gram-Schmidt orthogonalization (column-wise)
        for i in 0..dim {
            let norm: f32 = q.column(i).dot(&q.column(i)).sqrt();
            if norm > 1e-10 {
                let inv_norm = 1.0 / norm;
                for r in 0..dim {
                    q[[r, i]] *= inv_norm;
                }
            }
            for j in (i + 1)..dim {
                let proj: f32 = q.column(i).dot(&q.column(j));
                for r in 0..dim {
                    q[[r, j]] -= proj * q[[r, i]];
                }
            }
        }

        Self { matrix: q, dim }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Rotate a vector: y = Q * x.
    pub fn rotate(&self, vector: &[f32]) -> Vec<f32> {
        assert_eq!(vector.len(), self.dim);
        let x = Array1::from_vec(vector.to_vec());
        self.matrix.dot(&x).to_vec()
    }

    /// Inverse rotate: x = Q^T * y (Q is orthogonal, so Q^{-1} = Q^T).
    pub fn inverse_rotate(&self, rotated: &[f32]) -> Vec<f32> {
        assert_eq!(rotated.len(), self.dim);
        let y = Array1::from_vec(rotated.to_vec());
        self.matrix.t().dot(&y).to_vec()
    }

    /// Access the raw matrix (for testing).
    pub fn matrix(&self) -> &Array2<f32> {
        &self.matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orthogonality() {
        let dim = 16;
        let rot = Rotation::new(dim, 42);
        let qtq = rot.matrix.t().dot(&rot.matrix);
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[[i, j]] - expected).abs() < 1e-4,
                    "Q^T Q[{i},{j}] = {} (expected {expected})",
                    qtq[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_rotate_inverse_roundtrip() {
        let dim = 32;
        let rot = Rotation::new(dim, 42);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32) / dim as f32).collect();
        let rotated = rot.rotate(&v);
        let recovered = rot.inverse_rotate(&rotated);
        for (a, b) in v.iter().zip(&recovered) {
            assert!((a - b).abs() < 1e-4, "rotate/inverse mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_norm_preservation() {
        let dim = 64;
        let rot = Rotation::new(dim, 99);
        let mut v = vec![0.0f32; dim];
        v[0] = 1.0;
        let rotated = rot.rotate(&v);
        let norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "rotation should preserve norm: {norm}"
        );
    }

    #[test]
    fn test_deterministic() {
        let r1 = Rotation::new(16, 42);
        let r2 = Rotation::new(16, 42);
        assert_eq!(r1.matrix, r2.matrix);
    }
}
