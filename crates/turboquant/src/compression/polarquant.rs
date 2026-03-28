// PolarQuant compression
// Random orthogonal rotation -> polar coordinates -> grid quantization

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// Quantized polar representation.
#[derive(Clone, Debug)]
pub struct PolarVector {
    /// Packed angle nibbles (4-bit each when angle_bits=4, two per byte).
    pub angles: Vec<u8>,
    /// 8-bit quantized radii, one per dimension pair.
    pub radii: Vec<u8>,
    /// Original vector dimensionality.
    pub dim: usize,
    /// Bits per angle quantization level.
    pub angle_bits: u8,
}

impl PolarVector {
    /// Number of dimension pairs = ceil(dim / 2).
    pub fn num_pairs(&self) -> usize {
        (self.dim + 1) / 2
    }

    /// Total serialized byte size (angles + radii).
    pub fn byte_size(&self) -> usize {
        self.angles.len() + self.radii.len()
    }
}

/// PolarQuant compressor: orthogonal rotation -> polar coords -> grid quantize.
pub struct PolarQuantCompressor {
    rotation_matrix: Array2<f32>,
    angle_bits: u8,
    radius_bits: u8,
    dim: usize,
    max_radius: f32,
}

impl PolarQuantCompressor {
    /// Create with deterministic seed. Generates orthogonal matrix via Gram-Schmidt.
    pub fn new(dim: usize, seed: u64, angle_bits: u8, radius_bits: u8) -> Self {
        let rotation_matrix = generate_orthogonal_matrix(dim, seed);
        Self {
            rotation_matrix,
            angle_bits,
            radius_bits,
            dim,
            max_radius: 0.5,
        }
    }

    /// Return the dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn angle_bits(&self) -> u8 {
        self.angle_bits
    }

    pub fn radius_bits(&self) -> u8 {
        self.radius_bits
    }

    /// Bytes needed to store angles for one vector.
    pub fn angle_bytes(&self) -> usize {
        let num_pairs = (self.dim + 1) / 2;
        if self.angle_bits == 4 {
            (num_pairs + 1) / 2 // two nibbles per byte
        } else {
            num_pairs // 8-bit angles, one per byte
        }
    }

    /// Bytes needed to store radii for one vector.
    pub fn radii_bytes(&self) -> usize {
        (self.dim + 1) / 2
    }

    /// Total bytes per compressed vector (angles + radii).
    pub fn bytes_per_vector(&self) -> usize {
        self.angle_bytes() + self.radii_bytes()
    }

    /// Compress: rotate -> pair dims -> polar -> quantize.
    pub fn compress(&self, vector: &[f32]) -> PolarVector {
        assert_eq!(vector.len(), self.dim, "vector length must match dim");

        let x = Array1::from_vec(vector.to_vec());
        let y = self.rotation_matrix.dot(&x);

        let num_pairs = (self.dim + 1) / 2;
        let angle_levels = 1u32 << self.angle_bits;
        let radius_levels = (1u32 << self.radius_bits) - 1;

        let mut raw_angles = Vec::with_capacity(num_pairs);
        let mut radii = Vec::with_capacity(num_pairs);

        for i in 0..num_pairs {
            let y0 = y[2 * i];
            let y1 = if 2 * i + 1 < self.dim { y[2 * i + 1] } else { 0.0 };

            let r = (y0 * y0 + y1 * y1).sqrt();
            let theta = y1.atan2(y0); // in [-pi, pi]

            // Quantize angle: map [-pi, pi] to [0, 2^b)
            let theta_norm = (theta + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
            let theta_q = ((theta_norm * angle_levels as f32).round() as u32) % angle_levels;
            raw_angles.push(theta_q as u8);

            // Quantize radius
            let r_norm = (r / self.max_radius).clamp(0.0, 1.0);
            let r_q = (r_norm * radius_levels as f32).round() as u8;
            radii.push(r_q);
        }

        // Pack angles as nibbles if angle_bits == 4
        let angles = if self.angle_bits == 4 {
            pack_nibbles(&raw_angles)
        } else {
            raw_angles
        };

        PolarVector {
            angles,
            radii,
            dim: self.dim,
            angle_bits: self.angle_bits,
        }
    }

    /// Decompress: reconstruct approximate vector from quantized polar.
    pub fn decompress(&self, pv: &PolarVector) -> Vec<f32> {
        assert_eq!(pv.dim, self.dim);

        let num_pairs = (self.dim + 1) / 2;
        let angle_levels = 1u32 << self.angle_bits;
        let radius_levels = (1u32 << self.radius_bits) - 1;

        let raw_angles = if pv.angle_bits == 4 {
            unpack_nibbles(&pv.angles, num_pairs)
        } else {
            pv.angles.clone()
        };

        let mut y = vec![0.0f32; self.dim];

        for i in 0..num_pairs {
            let theta_q = raw_angles[i] as u32;
            let r_q = pv.radii[i];

            // Dequantize angle
            let theta = (theta_q as f32 / angle_levels as f32) * 2.0 * std::f32::consts::PI
                - std::f32::consts::PI;

            // Dequantize radius
            let r = (r_q as f32 / radius_levels as f32) * self.max_radius;

            y[2 * i] = r * theta.cos();
            if 2 * i + 1 < self.dim {
                y[2 * i + 1] = r * theta.sin();
            }
        }

        // Inverse rotation: x_approx = Q^T @ y
        let y_arr = Array1::from_vec(y);
        let x_approx = self.rotation_matrix.t().dot(&y_arr);
        x_approx.to_vec()
    }

    /// Compute similarity between two PolarVectors (approximate dot product).
    /// Uses: sim = sum_i r_a_i * r_b_i * cos(theta_a_i - theta_b_i)
    pub fn similarity(&self, a: &PolarVector, b: &PolarVector) -> f32 {
        assert_eq!(a.dim, b.dim);
        assert_eq!(a.dim, self.dim);

        let num_pairs = (self.dim + 1) / 2;
        let angle_levels = 1u32 << self.angle_bits;
        let radius_levels = (1u32 << self.radius_bits) - 1;

        let a_angles = if a.angle_bits == 4 {
            unpack_nibbles(&a.angles, num_pairs)
        } else {
            a.angles.clone()
        };
        let b_angles = if b.angle_bits == 4 {
            unpack_nibbles(&b.angles, num_pairs)
        } else {
            b.angles.clone()
        };

        let mut sim = 0.0f32;

        for i in 0..num_pairs {
            let r_a = (a.radii[i] as f32 / radius_levels as f32) * self.max_radius;
            let r_b = (b.radii[i] as f32 / radius_levels as f32) * self.max_radius;

            let theta_a = (a_angles[i] as f32 / angle_levels as f32) * 2.0 * std::f32::consts::PI;
            let theta_b = (b_angles[i] as f32 / angle_levels as f32) * 2.0 * std::f32::consts::PI;

            sim += r_a * r_b * (theta_a - theta_b).cos();
        }

        sim
    }
}

/// Generate an orthogonal matrix via Gram-Schmidt on random Gaussian columns.
fn generate_orthogonal_matrix(dim: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0f64).unwrap();

    // Generate random Gaussian matrix
    let data: Vec<f32> = (0..dim * dim)
        .map(|_| normal.sample(&mut rng) as f32)
        .collect();
    let mut q = Array2::from_shape_vec((dim, dim), data).unwrap();

    // Modified Gram-Schmidt orthogonalization (column-wise)
    for i in 0..dim {
        // Normalize column i
        let norm: f32 = q.column(i).dot(&q.column(i)).sqrt();
        if norm > 1e-10 {
            let inv_norm = 1.0 / norm;
            for r in 0..dim {
                q[[r, i]] *= inv_norm;
            }
        }

        // Subtract projection of column i from all subsequent columns
        for j in (i + 1)..dim {
            let proj: f32 = q.column(i).dot(&q.column(j));
            for r in 0..dim {
                q[[r, j]] -= proj * q[[r, i]];
            }
        }
    }

    q
}

/// Pack an array of nibbles (values 0..15) into bytes, two per byte.
fn pack_nibbles(values: &[u8]) -> Vec<u8> {
    let num_bytes = (values.len() + 1) / 2;
    let mut packed = vec![0u8; num_bytes];
    for (i, &v) in values.iter().enumerate() {
        if i % 2 == 0 {
            packed[i / 2] |= v & 0x0F;
        } else {
            packed[i / 2] |= (v & 0x0F) << 4;
        }
    }
    packed
}

/// Unpack nibbles from packed bytes.
fn unpack_nibbles(packed: &[u8], count: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let byte = packed[i / 2];
        let nibble = if i % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
        values.push(nibble);
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orthogonal_matrix() {
        let dim = 16;
        let q = generate_orthogonal_matrix(dim, 42);

        // Q^T @ Q should be close to identity
        let qtq = q.t().dot(&q);
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
    fn test_pack_unpack_nibbles() {
        let values = vec![3, 12, 7, 15, 0];
        let packed = pack_nibbles(&values);
        let unpacked = unpack_nibbles(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let dim = 32;
        let comp = PolarQuantCompressor::new(dim, 42, 4, 8);

        // Create a unit vector
        let mut v = vec![0.0f32; dim];
        v[0] = 1.0;

        let pv = comp.compress(&v);
        let reconstructed = comp.decompress(&pv);

        // Compute cosine similarity between original and reconstructed
        let dot: f32 = v.iter().zip(reconstructed.iter()).map(|(a, b)| a * b).sum();
        let norm_r: f32 = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Should be reasonably close for a unit vector
        let cosine = if norm_r > 0.0 { dot / norm_r } else { 0.0 };
        assert!(
            cosine > 0.5,
            "roundtrip cosine similarity = {cosine}, expected > 0.5"
        );
    }

    #[test]
    fn test_similarity_self() {
        let dim = 64;
        let comp = PolarQuantCompressor::new(dim, 99, 4, 8);

        let mut v = vec![0.0f32; dim];
        v[0] = 1.0;

        let pv = comp.compress(&v);
        let self_sim = comp.similarity(&pv, &pv);

        // Self-similarity should be positive
        assert!(self_sim > 0.0, "self similarity = {self_sim}");
    }

    #[test]
    fn test_similarity_ordering() {
        let dim = 64;
        let comp = PolarQuantCompressor::new(dim, 55, 4, 8);

        // v1 = [1, 0, 0, ...]
        let mut v1 = vec![0.0f32; dim];
        v1[0] = 1.0;

        // v2 is close to v1
        let mut v2 = vec![0.0f32; dim];
        let norm = (0.9f32 * 0.9 + 0.1 * 0.1).sqrt();
        v2[0] = 0.9 / norm;
        v2[1] = 0.1 / norm;

        // v3 is orthogonal
        let mut v3 = vec![0.0f32; dim];
        v3[1] = 1.0;

        let pv1 = comp.compress(&v1);
        let pv2 = comp.compress(&v2);
        let pv3 = comp.compress(&v3);

        let sim12 = comp.similarity(&pv1, &pv2);
        let sim13 = comp.similarity(&pv1, &pv3);

        assert!(
            sim12 > sim13,
            "similar vectors should have higher similarity: sim12={sim12}, sim13={sim13}"
        );
    }

    #[test]
    fn test_deterministic() {
        let dim = 16;
        let c1 = PolarQuantCompressor::new(dim, 42, 4, 8);
        let c2 = PolarQuantCompressor::new(dim, 42, 4, 8);
        assert_eq!(c1.rotation_matrix, c2.rotation_matrix);
    }
}
