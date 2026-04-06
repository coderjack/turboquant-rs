// Lloyd-Max scalar quantizer — Algorithm 1 from TurboQuant paper
//
// After rotation, each coordinate follows ≈ N(0, 1/√d). Lloyd-Max finds
// the optimal scalar quantization centroids for this distribution, minimizing
// reconstruction MSE. The centroids only depend on (dim, bits) and are
// computed once at construction time.
//
// References:
//   Lloyd (1982), "Least squares quantization in PCM"
//   TurboQuant paper, Algorithm 1 (TurboQuant_MSE)

/// Packed scalar quantization indices — one index per dimension.
#[derive(Clone, Debug)]
pub struct ScalarQuantVector {
    /// Packed indices. For bits ≤ 4, multiple indices per byte.
    pub data: Vec<u8>,
    /// Original vector dimensionality.
    pub dim: usize,
    /// Bits per index.
    pub bits: u8,
}

impl ScalarQuantVector {
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }
}

/// Lloyd-Max scalar quantizer for Gaussian-distributed coordinates.
///
/// After random rotation of a unit vector in d dimensions, each coordinate
/// follows approximately N(0, σ) where σ = 1/√d. This quantizer finds
/// the MSE-optimal centroids for that distribution.
pub struct LloydMaxQuantizer {
    /// Sorted centroids (2^bits values).
    centroids: Vec<f32>,
    /// Partition boundaries (2^bits - 1 midpoints, plus -∞ and +∞ implicit).
    boundaries: Vec<f32>,
    bits: u8,
    dim: usize,
    #[allow(dead_code)]
    sigma: f32,
}

impl LloydMaxQuantizer {
    /// Create a quantizer. Computes optimal centroids via Lloyd-Max iteration.
    ///
    /// - `dim`: vector dimension (determines σ = 1/√d)
    /// - `bits`: quantization bits per dimension (1–8)
    pub fn new(dim: usize, bits: u8) -> Self {
        assert!(bits >= 1 && bits <= 8, "bits must be 1–8");

        let sigma = 1.0 / (dim as f32).sqrt();
        let centroids = compute_lloyd_max_centroids(sigma, bits);
        let boundaries = compute_boundaries(&centroids);

        Self {
            centroids,
            boundaries,
            bits,
            dim,
            sigma,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Bytes needed per compressed vector.
    pub fn bytes_per_vector(&self) -> usize {
        let total_bits = self.dim * self.bits as usize;
        (total_bits + 7) / 8
    }

    /// Compress a rotated vector: find nearest centroid per dimension, pack indices.
    pub fn compress(&self, rotated: &[f32]) -> ScalarQuantVector {
        assert_eq!(rotated.len(), self.dim);

        let n_levels = self.centroids.len();
        let mut indices = Vec::with_capacity(self.dim);

        for &val in rotated {
            // Binary search on boundaries for the right partition
            let mut idx = 0usize;
            for (b, &boundary) in self.boundaries.iter().enumerate() {
                if val > boundary {
                    idx = b + 1;
                } else {
                    break;
                }
            }
            indices.push(idx.min(n_levels - 1) as u8);
        }

        let data = pack_indices(&indices, self.bits);
        ScalarQuantVector {
            data,
            dim: self.dim,
            bits: self.bits,
        }
    }

    /// Decompress: look up centroids for each index.
    pub fn decompress(&self, sqv: &ScalarQuantVector) -> Vec<f32> {
        assert_eq!(sqv.dim, self.dim);
        let indices = unpack_indices(&sqv.data, self.dim, sqv.bits);
        indices
            .iter()
            .map(|&idx| self.centroids[idx as usize])
            .collect()
    }

    /// Dot product between a raw rotated query and a compressed vector.
    pub fn similarity_raw(&self, rotated_query: &[f32], compressed: &ScalarQuantVector) -> f32 {
        let reconstructed = self.decompress(compressed);
        rotated_query
            .iter()
            .zip(&reconstructed)
            .map(|(a, b)| a * b)
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Lloyd-Max centroid computation
// ---------------------------------------------------------------------------

/// Gaussian PDF: (1 / (σ√(2π))) * exp(-x²/(2σ²))
fn gaussian_pdf(x: f32, sigma: f32) -> f32 {
    let coeff = 1.0 / (sigma * (2.0 * std::f32::consts::PI).sqrt());
    let exponent = -(x * x) / (2.0 * sigma * sigma);
    coeff * exponent.exp()
}

/// Compute E[x | a ≤ x ≤ b] for Gaussian using numerical integration (trapezoid rule).
///
/// Returns (numerator, denominator) where:
///   numerator = ∫_a^b x · pdf(x) dx
///   denominator = ∫_a^b pdf(x) dx
fn gaussian_conditional_mean(a: f32, b: f32, sigma: f32, n_points: usize) -> (f64, f64) {
    let h = (b - a) as f64 / n_points as f64;
    let mut num = 0.0f64;
    let mut den = 0.0f64;

    for i in 0..=n_points {
        let x = a as f64 + i as f64 * h;
        let p = gaussian_pdf(x as f32, sigma) as f64;
        let weight = if i == 0 || i == n_points { 0.5 } else { 1.0 };
        num += weight * x * p;
        den += weight * p;
    }

    (num * h, den * h)
}

/// Run Lloyd-Max iteration to find optimal centroids for N(0, σ).
fn compute_lloyd_max_centroids(sigma: f32, bits: u8) -> Vec<f32> {
    let n_levels = 1usize << bits;
    let range = 5.0 * sigma; // cover ±5σ (>99.99997%)

    // Initialize: uniform spacing
    let mut centroids: Vec<f32> = (0..n_levels)
        .map(|i| {
            -range + (2.0 * range) * (i as f32 + 0.5) / n_levels as f32
        })
        .collect();

    let n_integration_points = 200;
    let max_iter = 100;
    let tol = 1e-8f32;

    for _iter in 0..max_iter {
        // Compute partition boundaries (midpoints)
        let boundaries = compute_boundaries(&centroids);

        // Update each centroid to the conditional mean of its partition
        let mut new_centroids = vec![0.0f32; n_levels];
        let mut max_change = 0.0f32;

        for i in 0..n_levels {
            let lo = if i == 0 { -range } else { boundaries[i - 1] };
            let hi = if i == n_levels - 1 {
                range
            } else {
                boundaries[i]
            };

            let (num, den) =
                gaussian_conditional_mean(lo, hi, sigma, n_integration_points);

            new_centroids[i] = if den.abs() > 1e-15 {
                (num / den) as f32
            } else {
                centroids[i] // keep old if partition has negligible mass
            };

            max_change = max_change.max((new_centroids[i] - centroids[i]).abs());
        }

        centroids = new_centroids;

        if max_change < tol {
            break;
        }
    }

    // Ensure sorted
    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap());
    centroids
}

/// Compute partition boundaries as midpoints between adjacent centroids.
fn compute_boundaries(centroids: &[f32]) -> Vec<f32> {
    centroids
        .windows(2)
        .map(|w| 0.5 * (w[0] + w[1]))
        .collect()
}

// ---------------------------------------------------------------------------
// Bit packing utilities
// ---------------------------------------------------------------------------

/// Bit mask for `bits` bits, handling bits=8 without overflow.
fn bit_mask(bits: u8) -> u8 {
    if bits >= 8 { 0xFF } else { (1u8 << bits) - 1 }
}

/// Pack indices (0..2^bits-1) into bytes.
fn pack_indices(indices: &[u8], bits: u8) -> Vec<u8> {
    // Special case: 8-bit = 1 byte per index, no packing needed
    if bits == 8 {
        return indices.to_vec();
    }

    let total_bits = indices.len() * bits as usize;
    let num_bytes = (total_bits + 7) / 8;
    let mut packed = vec![0u8; num_bytes];
    let mask = bit_mask(bits);

    let mut bit_offset = 0usize;
    for &idx in indices {
        let byte_pos = bit_offset / 8;
        let bit_pos = bit_offset % 8;

        packed[byte_pos] |= (idx & mask) << bit_pos;
        if bit_pos + bits as usize > 8 {
            let overflow = bit_pos + bits as usize - 8;
            packed[byte_pos + 1] |= idx >> (bits as usize - overflow);
        }

        bit_offset += bits as usize;
    }

    packed
}

/// Unpack indices from packed bytes.
fn unpack_indices(packed: &[u8], count: usize, bits: u8) -> Vec<u8> {
    // Special case: 8-bit = 1 byte per index
    if bits == 8 {
        return packed[..count].to_vec();
    }

    let mask = bit_mask(bits);
    let mut indices = Vec::with_capacity(count);

    let mut bit_offset = 0usize;
    for _ in 0..count {
        let byte_pos = bit_offset / 8;
        let bit_pos = bit_offset % 8;

        let mut val = packed[byte_pos] >> bit_pos;
        if bit_pos + bits as usize > 8 && byte_pos + 1 < packed.len() {
            val |= packed[byte_pos + 1] << (8 - bit_pos);
        }
        indices.push(val & mask);

        bit_offset += bits as usize;
    }

    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        for bits in 1..=8u8 {
            let n = 32;
            let max_val = bit_mask(bits);
            let indices: Vec<u8> = (0..n).map(|i| (i as u8) & max_val).collect();
            let packed = pack_indices(&indices, bits);
            let unpacked = unpack_indices(&packed, n, bits);
            assert_eq!(indices, unpacked, "roundtrip failed for bits={bits}");
        }
    }

    #[test]
    fn test_centroids_symmetry() {
        // For a symmetric distribution (Gaussian), centroids should be
        // approximately symmetric around 0
        let centroids = compute_lloyd_max_centroids(0.05, 2); // dim=400, σ≈0.05
        assert_eq!(centroids.len(), 4);
        // Check approximate symmetry
        assert!(
            (centroids[0] + centroids[3]).abs() < 0.001,
            "c[0]={} c[3]={} should be symmetric",
            centroids[0],
            centroids[3]
        );
        assert!(
            (centroids[1] + centroids[2]).abs() < 0.001,
            "c[1]={} c[2]={} should be symmetric",
            centroids[1],
            centroids[2]
        );
    }

    #[test]
    fn test_centroids_sorted() {
        for bits in 1..=4u8 {
            let centroids = compute_lloyd_max_centroids(0.051, bits);
            for w in centroids.windows(2) {
                assert!(w[0] < w[1], "centroids not sorted at bits={bits}");
            }
        }
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let dim = 64;
        let lm = LloydMaxQuantizer::new(dim, 4);
        let sigma = 1.0 / (dim as f32).sqrt();

        // Simulate a rotated vector
        let rotated: Vec<f32> = (0..dim)
            .map(|i| ((i as f32) * 0.7).sin() * sigma)
            .collect();

        let compressed = lm.compress(&rotated);
        let reconstructed = lm.decompress(&compressed);

        assert_eq!(reconstructed.len(), dim);

        // MSE should be small for 4-bit quantization
        let mse: f32 = rotated
            .iter()
            .zip(&reconstructed)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / dim as f32;

        assert!(
            mse < sigma * sigma * 0.1,
            "MSE {mse} too high for 4-bit Lloyd-Max"
        );
    }

    #[test]
    fn test_bytes_per_vector() {
        assert_eq!(LloydMaxQuantizer::new(384, 1).bytes_per_vector(), 48);
        assert_eq!(LloydMaxQuantizer::new(384, 2).bytes_per_vector(), 96);
        assert_eq!(LloydMaxQuantizer::new(384, 3).bytes_per_vector(), 144);
        assert_eq!(LloydMaxQuantizer::new(384, 4).bytes_per_vector(), 192);
        assert_eq!(LloydMaxQuantizer::new(384, 8).bytes_per_vector(), 384);
    }

    #[test]
    fn test_similarity_raw_matches_decompress() {
        let dim = 32;
        let lm = LloydMaxQuantizer::new(dim, 3);
        let sigma = 1.0 / (dim as f32).sqrt();

        let rotated: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.5).sin() * sigma).collect();
        let query: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.3).cos() * sigma).collect();

        let compressed = lm.compress(&rotated);
        let sim_raw = lm.similarity_raw(&query, &compressed);

        let decompressed = lm.decompress(&compressed);
        let sim_manual: f32 = query.iter().zip(&decompressed).map(|(a, b)| a * b).sum();

        assert!(
            (sim_raw - sim_manual).abs() < 1e-6,
            "similarity_raw={sim_raw} vs manual={sim_manual}"
        );
    }
}
