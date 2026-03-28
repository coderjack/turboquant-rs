// SIMD-accelerated Hamming distance + bulk top-k search

use super::qjl::BitVector;

/// Hamming distance between two BitVectors using u64 popcount.
pub fn hamming_distance(a: &BitVector, b: &BitVector) -> u32 {
    assert_eq!(a.0.len(), b.0.len(), "BitVector lengths must match");

    let a_bytes = &a.0;
    let b_bytes = &b.0;
    let len = a_bytes.len();

    let mut dist: u32 = 0;

    // Process 8 bytes at a time using u64 XOR + popcount
    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let offset = i * 8;
        let a_u64 = u64::from_le_bytes(a_bytes[offset..offset + 8].try_into().unwrap());
        let b_u64 = u64::from_le_bytes(b_bytes[offset..offset + 8].try_into().unwrap());
        dist += (a_u64 ^ b_u64).count_ones();
    }

    // Handle remaining bytes
    let tail_offset = chunks * 8;
    for i in 0..remainder {
        dist += (a_bytes[tail_offset + i] ^ b_bytes[tail_offset + i]).count_ones();
    }

    dist
}

/// Find top-k closest vectors by minimum Hamming distance.
/// Returns (index, distance) pairs sorted by distance ascending.
pub fn hamming_top_k(query: &BitVector, index: &[BitVector], k: usize) -> Vec<(usize, u32)> {
    if index.is_empty() || k == 0 {
        return vec![];
    }

    let mut distances: Vec<(usize, u32)> = index
        .iter()
        .enumerate()
        .map(|(i, bv)| (i, hamming_distance(query, bv)))
        .collect();

    let k = k.min(distances.len());

    // Partial sort: place the k smallest at the front
    distances.select_nth_unstable_by_key(k - 1, |&(_, d)| d);
    distances.truncate(k);
    distances.sort_by_key(|&(_, d)| d);

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical() {
        let a = BitVector(vec![0b10110011, 0b11001010]);
        let b = BitVector(vec![0b10110011, 0b11001010]);
        assert_eq!(hamming_distance(&a, &b), 0);
    }

    #[test]
    fn test_known_pattern() {
        // Differ in exactly 3 bits
        let a = BitVector(vec![0b00000000]);
        let b = BitVector(vec![0b00000111]);
        assert_eq!(hamming_distance(&a, &b), 3);
    }

    #[test]
    fn test_all_differ() {
        let a = BitVector(vec![0x00]);
        let b = BitVector(vec![0xFF]);
        assert_eq!(hamming_distance(&a, &b), 8);
    }

    #[test]
    fn test_multi_byte_u64_path() {
        // 9 bytes: exercises both the u64 path (first 8) and remainder (1)
        let a = BitVector(vec![0u8; 9]);
        let b = BitVector(vec![0xFF; 9]);
        assert_eq!(hamming_distance(&a, &b), 72); // 9 * 8
    }

    #[test]
    fn test_top_k() {
        let query = BitVector(vec![0b00000000]);
        let index = vec![
            BitVector(vec![0b11111111]), // dist 8
            BitVector(vec![0b00000001]), // dist 1
            BitVector(vec![0b00000011]), // dist 2
            BitVector(vec![0b01111111]), // dist 7
        ];
        let results = hamming_top_k(&query, &index, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (1, 1));
        assert_eq!(results[1], (2, 2));
    }

    #[test]
    fn test_top_k_larger_than_index() {
        let query = BitVector(vec![0x00]);
        let index = vec![BitVector(vec![0x01]), BitVector(vec![0x03])];
        let results = hamming_top_k(&query, &index, 10);
        assert_eq!(results.len(), 2);
    }
}
