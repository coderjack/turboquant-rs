// PolarQuant — Step 2 of TurboQuant
//
// Converts pairs of rotated Cartesian coordinates to polar form (radius, angle),
// then quantizes each independently. This eliminates the per-block overhead that
// traditional quantizers carry.
//
// Input: a vector in the ROTATED domain (after Step 1 rotation).
// Output: packed quantized angles + radii.

/// Quantized polar representation of a rotated vector.
#[derive(Clone, Debug)]
pub struct PolarVector {
    /// Packed angle data (nibbles if angle_bits=4, bytes otherwise).
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

/// PolarQuant quantizer — operates on already-rotated vectors.
///
/// No rotation matrix is stored here; the caller (TurboQuantCompressor)
/// handles rotation as a separate step.
pub struct PolarQuantizer {
    angle_bits: u8,
    radius_bits: u8,
    dim: usize,
    max_radius: f32,
}

impl PolarQuantizer {
    /// Create a quantizer for the given dimension.
    ///
    /// `max_radius` is set automatically to cover ~4σ of the Rayleigh
    /// distribution that pair-radii follow after rotation of a unit vector.
    pub fn new(dim: usize, angle_bits: u8, radius_bits: u8) -> Self {
        // After rotation of a unit vector, each coordinate ≈ N(0, 1/d).
        // Pair radius follows Rayleigh(σ) with σ = 1/sqrt(d).
        // 4σ covers >99.99% of the distribution.
        let max_radius = 4.0 / (dim as f32).sqrt();
        Self {
            angle_bits,
            radius_bits,
            dim,
            max_radius,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn angle_bits(&self) -> u8 {
        self.angle_bits
    }
    pub fn radius_bits(&self) -> u8 {
        self.radius_bits
    }
    pub fn max_radius(&self) -> f32 {
        self.max_radius
    }

    /// Bytes needed for the angle data of one vector.
    pub fn angle_bytes(&self) -> usize {
        let num_pairs = (self.dim + 1) / 2;
        if self.angle_bits == 4 {
            (num_pairs + 1) / 2 // two nibbles per byte
        } else {
            num_pairs // one byte per angle
        }
    }

    /// Bytes needed for the radii data of one vector.
    pub fn radii_bytes(&self) -> usize {
        (self.dim + 1) / 2
    }

    /// Total bytes per compressed vector (angles + radii).
    pub fn bytes_per_vector(&self) -> usize {
        self.angle_bytes() + self.radii_bytes()
    }

    /// Compress a rotated vector into polar coordinates.
    pub fn compress(&self, rotated: &[f32]) -> PolarVector {
        assert_eq!(rotated.len(), self.dim);

        let num_pairs = (self.dim + 1) / 2;
        let angle_levels = 1u32 << self.angle_bits;
        let radius_levels = (1u32 << self.radius_bits) - 1;

        let mut raw_angles = Vec::with_capacity(num_pairs);
        let mut radii = Vec::with_capacity(num_pairs);

        for i in 0..num_pairs {
            let y0 = rotated[2 * i];
            let y1 = if 2 * i + 1 < self.dim {
                rotated[2 * i + 1]
            } else {
                0.0
            };

            let r = (y0 * y0 + y1 * y1).sqrt();
            let theta = y1.atan2(y0); // in [-π, π]

            // Quantize angle: map [-π, π] → [0, 2^b)
            let theta_norm =
                (theta + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
            let theta_q =
                ((theta_norm * angle_levels as f32).round() as u32) % angle_levels;
            raw_angles.push(theta_q as u8);

            // Quantize radius: map [0, max_radius] → [0, 2^b - 1]
            let r_norm = (r / self.max_radius).clamp(0.0, 1.0);
            let r_q = (r_norm * radius_levels as f32).round() as u8;
            radii.push(r_q);
        }

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

    /// Decompress to rotated-domain Cartesian coordinates.
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
            let theta = (raw_angles[i] as f32 / angle_levels as f32)
                * 2.0
                * std::f32::consts::PI
                - std::f32::consts::PI;
            let r = (pv.radii[i] as f32 / radius_levels as f32) * self.max_radius;

            y[2 * i] = r * theta.cos();
            if 2 * i + 1 < self.dim {
                y[2 * i + 1] = r * theta.sin();
            }
        }

        y
    }

    /// Dot product between a raw rotated query and a compressed vector.
    ///
    /// Decompresses the polar vector to rotated-domain Cartesian
    /// and computes the dot product. O(d).
    pub fn similarity_raw(&self, rotated_query: &[f32], compressed: &PolarVector) -> f32 {
        let y_hat = self.decompress(compressed);
        rotated_query
            .iter()
            .zip(&y_hat)
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Dot product between two compressed vectors using the polar identity:
    /// sim = Σ r_a · r_b · cos(θ_a − θ_b)
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
            let theta_a = (a_angles[i] as f32 / angle_levels as f32)
                * 2.0
                * std::f32::consts::PI;
            let theta_b = (b_angles[i] as f32 / angle_levels as f32)
                * 2.0
                * std::f32::consts::PI;
            sim += r_a * r_b * (theta_a - theta_b).cos();
        }

        sim
    }
}

// ---------------------------------------------------------------------------
// Nibble packing utilities
// ---------------------------------------------------------------------------

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
    fn test_pack_unpack_nibbles() {
        let values = vec![3, 12, 7, 15, 0];
        let packed = pack_nibbles(&values);
        let unpacked = unpack_nibbles(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let dim = 32;
        let pq = PolarQuantizer::new(dim, 4, 8);

        // Simulate a rotated unit vector (small values ≈ N(0, 1/d))
        let scale = 1.0 / (dim as f32).sqrt();
        let rotated: Vec<f32> = (0..dim)
            .map(|i| ((i as f32 * 0.7).sin()) * scale)
            .collect();

        let pv = pq.compress(&rotated);
        let reconstructed = pq.decompress(&pv);

        // Compute MSE
        let mse: f32 = rotated
            .iter()
            .zip(&reconstructed)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / dim as f32;

        assert!(
            mse < 1e-3,
            "roundtrip MSE = {mse}, expected < 1e-3 for well-scaled input"
        );
    }

    #[test]
    fn test_similarity_self() {
        let dim = 64;
        let pq = PolarQuantizer::new(dim, 4, 8);

        let scale = 1.0 / (dim as f32).sqrt();
        let rotated: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.3).cos() * scale).collect();

        let pv = pq.compress(&rotated);
        let self_sim = pq.similarity(&pv, &pv);
        assert!(self_sim > 0.0, "self similarity = {self_sim}");
    }

    #[test]
    fn test_similarity_raw_matches_decompress() {
        let dim = 32;
        let pq = PolarQuantizer::new(dim, 4, 8);

        let scale = 1.0 / (dim as f32).sqrt();
        let rotated: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.5).sin() * scale).collect();
        let query: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.3).cos() * scale).collect();

        let pv = pq.compress(&rotated);
        let sim_raw = pq.similarity_raw(&query, &pv);

        // Compare with manual decompress + dot
        let decompressed = pq.decompress(&pv);
        let sim_manual: f32 = query.iter().zip(&decompressed).map(|(a, b)| a * b).sum();

        assert!(
            (sim_raw - sim_manual).abs() < 1e-6,
            "similarity_raw={sim_raw} vs manual={sim_manual}"
        );
    }

    #[test]
    fn test_byte_sizes() {
        // 384-dim, 4-bit angles, 8-bit radii
        let pq = PolarQuantizer::new(384, 4, 8);
        // 192 pairs: angle_bytes = 96, radii_bytes = 192 → 288
        assert_eq!(pq.angle_bytes(), 96);
        assert_eq!(pq.radii_bytes(), 192);
        assert_eq!(pq.bytes_per_vector(), 288);
    }
}
