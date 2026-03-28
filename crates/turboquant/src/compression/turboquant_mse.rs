// TurboQuant_mse: random orthogonal rotation -> optimal scalar quantization
//
// Based on: Zandieh, Daliri, Hadian, Mirrokni (2025).
// "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate."
// arXiv:2504.19874.
//
// After rotation by a uniformly random orthogonal matrix, each coordinate of a
// unit-sphere vector follows a distribution that concentrates around N(0, 1/d)
// in high dimensions. We quantize each coordinate independently using a
// precomputed Lloyd-Max codebook for this distribution.

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// Compressed vector: packed b-bit scalar quantizer indices + original norm.
#[derive(Clone, Debug)]
pub struct TqMseVector {
    /// Packed b-bit indices, ceil(dim * bits / 8) bytes.
    pub data: Vec<u8>,
    /// Original L2 norm (TurboQuant assumes unit vectors; we store norm to
    /// handle arbitrary vectors by normalizing, quantizing, then scaling back).
    pub norm: f32,
    /// Dimensionality.
    pub dim: usize,
    /// Bits per coordinate.
    pub bits: u8,
}

impl TqMseVector {
    /// Total serialized byte size (4 bytes norm + packed indices).
    pub fn byte_size(&self) -> usize {
        4 + self.data.len()
    }

    /// Number of bytes for the packed index data (without norm).
    pub fn index_bytes(dim: usize, bits: u8) -> usize {
        (dim * bits as usize + 7) / 8
    }
}

/// Lloyd-Max optimal scalar quantizer codebook for N(0, 1).
/// The actual distribution after rotation is N(0, 1/d), so we scale by 1/sqrt(d).
///
/// These are the well-known optimal centroids for the standard normal distribution.
/// Reference: Max (1960), "Quantizing for minimum distortion".
struct Codebook {
    /// Centroids sorted ascending, length = 2^bits.
    centroids: &'static [f32],
    /// Decision boundaries (midpoints), length = 2^bits - 1.
    /// boundaries[i] = (centroids[i] + centroids[i+1]) / 2
    boundaries: &'static [f32],
}

// Lloyd-Max quantizer centroids for N(0,1). Symmetric around 0.
// b=2 (4 levels):
static CODEBOOK_2_CENTROIDS: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];
static CODEBOOK_2_BOUNDARIES: [f32; 3] = [-0.9816, 0.0, 0.9816];

// b=3 (8 levels):
static CODEBOOK_3_CENTROIDS: [f32; 8] = [
    -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
];
static CODEBOOK_3_BOUNDARIES: [f32; 7] = [
    -1.7480, -1.0500, -0.5006, 0.0, 0.5006, 1.0500, 1.7480,
];

// b=4 (16 levels):
static CODEBOOK_4_CENTROIDS: [f32; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284,
    0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6180, 2.0690, 2.7326,
];
static CODEBOOK_4_BOUNDARIES: [f32; 15] = [
    -2.4008, -1.8435, -1.4371, -1.0993, -0.7996, -0.5224, -0.2582, 0.0, 0.2582, 0.5224, 0.7996,
    1.0993, 1.4371, 1.8435, 2.4008,
];

fn get_codebook(bits: u8) -> Codebook {
    match bits {
        2 => Codebook {
            centroids: &CODEBOOK_2_CENTROIDS,
            boundaries: &CODEBOOK_2_BOUNDARIES,
        },
        3 => Codebook {
            centroids: &CODEBOOK_3_CENTROIDS,
            boundaries: &CODEBOOK_3_BOUNDARIES,
        },
        4 => Codebook {
            centroids: &CODEBOOK_4_CENTROIDS,
            boundaries: &CODEBOOK_4_BOUNDARIES,
        },
        _ => panic!("TqMse supports bits=2,3,4; got {bits}"),
    }
}

/// TurboQuant_mse compressor.
///
/// Compress: normalize → rotate → quantize each coordinate via Lloyd-Max codebook.
/// Decompress: look up centroids → inverse rotate → scale by norm.
/// Similarity: dot product in rotated domain (no matrix multiply needed).
pub struct TqMseCompressor {
    rotation_matrix: Array2<f32>,
    /// Scaled codebook centroids: standard centroids * (1/sqrt(d)).
    scaled_centroids: Vec<f32>,
    /// Scaled decision boundaries: standard boundaries * (1/sqrt(d)).
    scaled_boundaries: Vec<f32>,
    dim: usize,
    bits: u8,
}

impl TqMseCompressor {
    /// Create with deterministic seed. `bits` must be 2, 3, or 4.
    pub fn new(dim: usize, seed: u64, bits: u8) -> Self {
        assert!(
            bits >= 2 && bits <= 4,
            "TqMse supports bits=2,3,4; got {bits}"
        );

        let rotation_matrix = generate_orthogonal_matrix(dim, seed);
        let codebook = get_codebook(bits);
        let scale = 1.0 / (dim as f32).sqrt();

        let scaled_centroids: Vec<f32> = codebook.centroids.iter().map(|&c| c * scale).collect();
        let scaled_boundaries: Vec<f32> =
            codebook.boundaries.iter().map(|&b| b * scale).collect();

        Self {
            rotation_matrix,
            scaled_centroids,
            scaled_boundaries,
            dim,
            bits,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Bytes per compressed vector (4 bytes norm + packed indices).
    pub fn bytes_per_vector(&self) -> usize {
        4 + TqMseVector::index_bytes(self.dim, self.bits)
    }

    /// Quantize a single coordinate value to its codebook index.
    #[inline]
    fn quantize_scalar(&self, val: f32) -> u8 {
        // Binary search through boundaries.
        // For 2^b levels this is at most b comparisons.
        let mut idx = 0u8;
        for &boundary in &self.scaled_boundaries {
            if val >= boundary {
                idx += 1;
            } else {
                break;
            }
        }
        idx
    }

    /// Compress a vector.
    pub fn compress(&self, vector: &[f32]) -> TqMseVector {
        assert_eq!(vector.len(), self.dim, "vector length must match dim");

        // Compute norm and normalize.
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        let x = if norm > 1e-10 {
            let inv = 1.0 / norm;
            Array1::from_vec(vector.iter().map(|&v| v * inv).collect())
        } else {
            Array1::zeros(self.dim)
        };

        // Rotate.
        let y = self.rotation_matrix.dot(&x);

        // Quantize each coordinate and pack bits.
        let num_bytes = TqMseVector::index_bytes(self.dim, self.bits);
        let mut data = vec![0u8; num_bytes];
        let mut bit_offset = 0usize;

        for j in 0..self.dim {
            let idx = self.quantize_scalar(y[j]);
            pack_bits(&mut data, bit_offset, idx, self.bits);
            bit_offset += self.bits as usize;
        }

        TqMseVector {
            data,
            norm,
            dim: self.dim,
            bits: self.bits,
        }
    }

    /// Decompress: look up centroids -> inverse rotate -> scale by norm.
    pub fn decompress(&self, tv: &TqMseVector) -> Vec<f32> {
        assert_eq!(tv.dim, self.dim);

        let mut y = Array1::zeros(self.dim);
        let mut bit_offset = 0usize;

        for j in 0..self.dim {
            let idx = unpack_bits(&tv.data, bit_offset, tv.bits) as usize;
            y[j] = self.scaled_centroids[idx];
            bit_offset += tv.bits as usize;
        }

        // Inverse rotate: x = Q^T @ y
        let x = self.rotation_matrix.t().dot(&y);

        // Scale by original norm.
        x.iter().map(|&v| v * tv.norm).collect()
    }

    /// Compute approximate dot product between two compressed vectors.
    ///
    /// Key insight: since Q is orthogonal, <x, y> = <Qx, Qy>.
    /// We compute the dot product directly in the rotated domain using
    /// codebook centroids — no matrix multiply needed.
    pub fn similarity(&self, a: &TqMseVector, b: &TqMseVector) -> f32 {
        assert_eq!(a.dim, self.dim);
        assert_eq!(b.dim, self.dim);

        let mut dot = 0.0f32;
        let mut bit_offset_a = 0usize;
        let mut bit_offset_b = 0usize;

        for _ in 0..self.dim {
            let idx_a = unpack_bits(&a.data, bit_offset_a, a.bits) as usize;
            let idx_b = unpack_bits(&b.data, bit_offset_b, b.bits) as usize;
            dot += self.scaled_centroids[idx_a] * self.scaled_centroids[idx_b];
            bit_offset_a += a.bits as usize;
            bit_offset_b += b.bits as usize;
        }

        // Scale by both norms.
        dot * a.norm * b.norm
    }

    /// Compute approximate dot product between a compressed vector and a raw
    /// query vector. Avoids compressing the query (preserves full precision).
    pub fn similarity_raw(&self, query: &[f32], compressed: &TqMseVector) -> f32 {
        assert_eq!(query.len(), self.dim);
        assert_eq!(compressed.dim, self.dim);

        // Rotate the query into the same domain.
        let q = Array1::from_vec(query.to_vec());
        let qr = self.rotation_matrix.dot(&q);

        let mut dot = 0.0f32;
        let mut bit_offset = 0usize;

        for j in 0..self.dim {
            let idx = unpack_bits(&compressed.data, bit_offset, compressed.bits) as usize;
            dot += qr[j] * self.scaled_centroids[idx];
            bit_offset += compressed.bits as usize;
        }

        dot * compressed.norm
    }
}

// ---------------------------------------------------------------------------
// Bit packing utilities
// ---------------------------------------------------------------------------

/// Pack a `bits`-wide value at the given bit offset into a byte array.
#[inline]
fn pack_bits(data: &mut [u8], bit_offset: usize, value: u8, bits: u8) {
    let mask = (1u8 << bits) - 1;
    let val = value & mask;

    let byte_idx = bit_offset / 8;
    let bit_idx = bit_offset % 8;

    // Value may span two bytes.
    data[byte_idx] |= val << bit_idx;
    let remaining = 8 - bit_idx;
    if (bits as usize) > remaining {
        data[byte_idx + 1] |= val >> remaining;
    }
}

/// Unpack a `bits`-wide value from the given bit offset.
#[inline]
fn unpack_bits(data: &[u8], bit_offset: usize, bits: u8) -> u8 {
    let mask = (1u16 << bits) - 1;
    let byte_idx = bit_offset / 8;
    let bit_idx = bit_offset % 8;

    // Read up to 2 bytes and shift.
    let lo = data[byte_idx] as u16;
    let hi = if byte_idx + 1 < data.len() {
        data[byte_idx + 1] as u16
    } else {
        0
    };
    let combined = lo | (hi << 8);
    ((combined >> bit_idx) & mask) as u8
}

// ---------------------------------------------------------------------------
// Random orthogonal matrix generation (same as in polarquant.rs)
// ---------------------------------------------------------------------------

/// Generate a uniformly random orthogonal matrix via Modified Gram-Schmidt
/// on a Gaussian random matrix. Deterministic given the seed.
fn generate_orthogonal_matrix(dim: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0f64).unwrap();

    let data: Vec<f32> = (0..dim * dim)
        .map(|_| normal.sample(&mut rng) as f32)
        .collect();
    let mut q = Array2::from_shape_vec((dim, dim), data).unwrap();

    // Modified Gram-Schmidt (column-wise).
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

    q
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_bits_2() {
        let mut data = vec![0u8; 4];
        pack_bits(&mut data, 0, 3, 2); // 11
        pack_bits(&mut data, 2, 1, 2); // 01
        pack_bits(&mut data, 4, 0, 2); // 00
        pack_bits(&mut data, 6, 2, 2); // 10

        assert_eq!(unpack_bits(&data, 0, 2), 3);
        assert_eq!(unpack_bits(&data, 2, 2), 1);
        assert_eq!(unpack_bits(&data, 4, 2), 0);
        assert_eq!(unpack_bits(&data, 6, 2), 2);
    }

    #[test]
    fn test_pack_unpack_bits_3() {
        let mut data = vec![0u8; 4];
        // Values: 5 (101), 7 (111), 2 (010)
        pack_bits(&mut data, 0, 5, 3);
        pack_bits(&mut data, 3, 7, 3);
        pack_bits(&mut data, 6, 2, 3);

        assert_eq!(unpack_bits(&data, 0, 3), 5);
        assert_eq!(unpack_bits(&data, 3, 3), 7);
        assert_eq!(unpack_bits(&data, 6, 3), 2);
    }

    #[test]
    fn test_pack_unpack_bits_4() {
        let mut data = vec![0u8; 2];
        pack_bits(&mut data, 0, 12, 4);
        pack_bits(&mut data, 4, 5, 4);
        pack_bits(&mut data, 8, 15, 4);

        assert_eq!(unpack_bits(&data, 0, 4), 12);
        assert_eq!(unpack_bits(&data, 4, 4), 5);
        assert_eq!(unpack_bits(&data, 8, 4), 15);
    }

    #[test]
    fn test_quantize_scalar_symmetry() {
        let comp = TqMseCompressor::new(384, 42, 3);
        // Zero should map to a central index.
        let idx = comp.quantize_scalar(0.0);
        // For 8 centroids symmetric around 0, boundary at 0 means idx=4
        assert_eq!(idx, 4, "zero should map to centroid index 4 (first positive)");
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let dim = 64;
        let comp = TqMseCompressor::new(dim, 42, 3);

        let mut v = vec![0.0f32; dim];
        v[0] = 1.0;

        let tv = comp.compress(&v);
        assert_eq!(tv.dim, dim);
        assert_eq!(tv.bits, 3);
        assert!((tv.norm - 1.0).abs() < 1e-6);

        let reconstructed = comp.decompress(&tv);
        assert_eq!(reconstructed.len(), dim);

        // Cosine similarity should be high.
        let dot: f32 = v.iter().zip(&reconstructed).map(|(a, b)| a * b).sum();
        let norm_r: f32 = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = if norm_r > 0.0 { dot / norm_r } else { 0.0 };
        assert!(
            cosine > 0.8,
            "roundtrip cosine = {cosine}, expected > 0.8"
        );
    }

    #[test]
    fn test_compress_decompress_4bit() {
        let dim = 64;
        let comp = TqMseCompressor::new(dim, 42, 4);

        let mut v = vec![0.0f32; dim];
        v[0] = 0.8;
        v[1] = 0.6;

        let tv = comp.compress(&v);
        let reconstructed = comp.decompress(&tv);

        let dot: f32 = v.iter().zip(&reconstructed).map(|(a, b)| a * b).sum();
        let norm_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_r: f32 = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = dot / (norm_v * norm_r);
        assert!(
            cosine > 0.9,
            "4-bit roundtrip cosine = {cosine}, expected > 0.9"
        );
    }

    #[test]
    fn test_similarity_self() {
        let dim = 64;
        let comp = TqMseCompressor::new(dim, 99, 3);

        let mut v = vec![0.0f32; dim];
        v[0] = 1.0;

        let tv = comp.compress(&v);
        let self_sim = comp.similarity(&tv, &tv);
        assert!(self_sim > 0.0, "self similarity = {self_sim}");
    }

    #[test]
    fn test_similarity_ordering() {
        let dim = 64;
        let comp = TqMseCompressor::new(dim, 55, 3);

        let mut v1 = vec![0.0f32; dim];
        v1[0] = 1.0;

        let mut v2 = vec![0.0f32; dim];
        let n = (0.9f32 * 0.9 + 0.1 * 0.1).sqrt();
        v2[0] = 0.9 / n;
        v2[1] = 0.1 / n;

        let mut v3 = vec![0.0f32; dim];
        v3[1] = 1.0;

        let tv1 = comp.compress(&v1);
        let tv2 = comp.compress(&v2);
        let tv3 = comp.compress(&v3);

        let sim12 = comp.similarity(&tv1, &tv2);
        let sim13 = comp.similarity(&tv1, &tv3);

        assert!(
            sim12 > sim13,
            "similar vectors should have higher similarity: sim12={sim12}, sim13={sim13}"
        );
    }

    #[test]
    fn test_similarity_raw_matches() {
        let dim = 32;
        let comp = TqMseCompressor::new(dim, 42, 3);

        let mut v1 = vec![0.0f32; dim];
        v1[0] = 1.0;
        let mut v2 = vec![0.0f32; dim];
        v2[0] = 0.8;
        v2[1] = 0.6;

        let tv1 = comp.compress(&v1);
        let tv2 = comp.compress(&v2);

        let sim_compressed = comp.similarity(&tv1, &tv2);
        let sim_raw = comp.similarity_raw(&v1, &tv2);

        // similarity_raw uses full-precision query, so it should be close but
        // not identical to compressed-compressed similarity.
        assert!(
            (sim_compressed - sim_raw).abs() < 0.15,
            "compressed={sim_compressed}, raw={sim_raw}"
        );
    }

    #[test]
    fn test_deterministic() {
        let dim = 32;
        let c1 = TqMseCompressor::new(dim, 42, 3);
        let c2 = TqMseCompressor::new(dim, 42, 3);
        assert_eq!(c1.rotation_matrix, c2.rotation_matrix);
        assert_eq!(c1.scaled_centroids, c2.scaled_centroids);
    }

    #[test]
    fn test_byte_sizes() {
        // 384-dim, 3-bit: 384 * 3 / 8 = 144 bytes + 4 norm = 148
        assert_eq!(TqMseVector::index_bytes(384, 3), 144);
        let comp = TqMseCompressor::new(384, 42, 3);
        assert_eq!(comp.bytes_per_vector(), 148);

        // 384-dim, 2-bit: 384 * 2 / 8 = 96 bytes + 4 norm = 100
        assert_eq!(TqMseVector::index_bytes(384, 2), 96);

        // 384-dim, 4-bit: 384 * 4 / 8 = 192 bytes + 4 norm = 196
        assert_eq!(TqMseVector::index_bytes(384, 4), 192);
    }

    #[test]
    fn test_mse_distortion() {
        // Verify MSE distortion at 3-bit is reasonable for random unit vectors.
        let dim = 384;
        let comp = TqMseCompressor::new(dim, 42, 3);

        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let normal = Normal::new(0.0, 1.0f64).unwrap();

        let mut total_mse = 0.0f64;
        let n_trials = 100;

        for _ in 0..n_trials {
            let raw: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng) as f32).collect();
            let norm: f32 = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
            let unit: Vec<f32> = raw.iter().map(|v| v / norm).collect();

            let tv = comp.compress(&unit);
            let recon = comp.decompress(&tv);

            let mse: f32 = unit
                .iter()
                .zip(&recon)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            total_mse += mse as f64;
        }

        let avg_mse = total_mse / n_trials as f64;
        // Theoretical bound for b=3: sqrt(3*pi)/2 * 4^{-3} ≈ 0.038
        // In practice should be lower.
        assert!(
            avg_mse < 0.05,
            "average MSE = {avg_mse}, expected < 0.05 for 3-bit"
        );
    }
}
