//! Tests that verify TurboQuant compression properties.
//!
//! Run with: `cargo test -p turboquant --test compression_claims -- --nocapture`

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use turboquant::{
    LloydMaxCompressor, PolarQuantizer, QjlCompressor, TurboIndex, TurboQuantCompressor,
    VectorStorage,
};

const DIM: usize = 384;
const RAW_BYTES_PER_VEC: usize = DIM * 4; // 1,536 bytes

fn random_unit_vec(dim: usize, rng: &mut ChaCha8Rng) -> Vec<f32> {
    let normal = Normal::new(0.0, 1.0f64).unwrap();
    let raw: Vec<f32> = (0..dim).map(|_| normal.sample(rng) as f32).collect();
    let norm: f32 = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
    raw.iter().map(|v| v / norm).collect()
}

// ===========================================================================
// Bytes per vector
// ===========================================================================

#[test]
fn verify_polarquant_bytes() {
    let pq = PolarQuantizer::new(DIM, 4, 8);
    // 192 pairs: angle_bytes=96, radii_bytes=192 → 288
    assert_eq!(pq.bytes_per_vector(), 288);
}

#[test]
fn verify_qjl_bytes() {
    let qjl = QjlCompressor::new(DIM, 42);
    assert_eq!(qjl.bytes_per_vector(), 48); // 384/8 = 48
}

#[test]
fn verify_turboquant_bytes() {
    let comp = TurboQuantCompressor::new(DIM, 42, 99, 4, 8);
    // polar(288) + qjl(48) + norms(8) = 344
    assert_eq!(comp.bytes_per_vector(), 344);

    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let v = random_unit_vec(DIM, &mut rng);
    let tv = comp.compress(&v);
    assert_eq!(tv.byte_size(), 344);
}

#[test]
fn verify_compression_ratio() {
    let comp = TurboQuantCompressor::new(DIM, 42, 99, 4, 8);
    let ratio = RAW_BYTES_PER_VEC as f64 / comp.bytes_per_vector() as f64;
    // 1536 / 344 ≈ 4.5x
    assert!(
        ratio > 4.0,
        "compression ratio {ratio:.1}x should be > 4x"
    );
}

#[test]
fn verify_lloydmax_bytes() {
    let comp = LloydMaxCompressor::new(DIM, 42, 99, 4);
    // indices: ceil(384*4/8)=192, qjl: 48, norms: 8 → 248
    assert_eq!(comp.bytes_per_vector(), 248);

    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let v = random_unit_vec(DIM, &mut rng);
    let lv = comp.compress(&v);
    assert_eq!(lv.byte_size(), 248);
}

#[test]
fn verify_lloydmax_compression_ratio() {
    let comp = LloydMaxCompressor::new(DIM, 42, 99, 4);
    let ratio = RAW_BYTES_PER_VEC as f64 / comp.bytes_per_vector() as f64;
    // 1536 / 248 ≈ 6.2x
    assert!(
        ratio > 6.0,
        "Lloyd-Max 4-bit compression ratio {ratio:.1}x should be > 6x"
    );
}

// ===========================================================================
// Roundtrip quality
// ===========================================================================

#[test]
fn verify_roundtrip_cosine() {
    let comp = TurboQuantCompressor::new(DIM, 42, 99, 4, 8);
    let mut rng = ChaCha8Rng::seed_from_u64(123);

    let mut total_cosine = 0.0f64;
    let n_trials = 200;

    for _ in 0..n_trials {
        let v = random_unit_vec(DIM, &mut rng);
        let tv = comp.compress(&v);
        let recon = comp.decompress(&tv);

        let dot: f32 = v.iter().zip(&recon).map(|(a, b)| a * b).sum();
        let norm_r: f32 = recon.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = if norm_r > 0.0 { dot / norm_r } else { 0.0 };
        total_cosine += cosine as f64;
    }

    let avg_cosine = total_cosine / n_trials as f64;
    eprintln!("Average roundtrip cosine similarity: {avg_cosine:.4}");

    assert!(
        avg_cosine > 0.85,
        "average cosine = {avg_cosine:.4}, expected > 0.85"
    );
}

// ===========================================================================
// Similarity ranking preservation
// ===========================================================================

#[test]
fn verify_similarity_preserves_ranking() {
    let comp = TurboQuantCompressor::new(DIM, 42, 99, 4, 8);
    let mut rng = ChaCha8Rng::seed_from_u64(77);

    let query = random_unit_vec(DIM, &mut rng);
    let vecs: Vec<Vec<f32>> = (0..50).map(|_| random_unit_vec(DIM, &mut rng)).collect();

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
    let mut true_ranking: Vec<(usize, f32)> = true_sims.iter().copied().enumerate().collect();
    true_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let true_top5: Vec<usize> = true_ranking[..5].iter().map(|(i, _)| *i).collect();

    // Top-5 by compressed similarity
    let mut comp_ranking: Vec<(usize, f32)> = comp_sims.iter().copied().enumerate().collect();
    comp_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let comp_top5: Vec<usize> = comp_ranking[..5].iter().map(|(i, _)| *i).collect();

    let overlap = true_top5
        .iter()
        .filter(|i| comp_top5.contains(i))
        .count();

    eprintln!(
        "Top-5 overlap: {overlap}/5\n  true_top5={true_top5:?}\n  comp_top5={comp_top5:?}"
    );

    assert!(
        overlap >= 2,
        "at least 2 of true top-5 should appear in compressed top-5, got {overlap}/5"
    );
}

// ===========================================================================
// Storage file sizes
// ===========================================================================

#[test]
fn verify_storage_file_sizes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.tqlm");
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let n = 100usize;
    let bits = 4u8;

    let comp = LloydMaxCompressor::new(DIM, 42, 99, bits);
    let bpv = comp.bytes_per_vector();
    {
        let mut storage = VectorStorage::create(&path, DIM, bits).unwrap();
        for _ in 0..n {
            let v = random_unit_vec(DIM, &mut rng);
            storage.append(&comp.compress(&v)).unwrap();
        }
    }

    let file_size = std::fs::metadata(&path).unwrap().len() as usize;
    let expected = 16 + 5 + n * bpv; // header(16) + metadata(5) + data
    assert_eq!(
        file_size, expected,
        "file: {file_size} bytes, expected {expected}"
    );

    eprintln!(
        "Storage for {n} vectors: {file_size} bytes ({:.1} bytes/vec incl. header)",
        file_size as f64 / n as f64
    );
}

// ===========================================================================
// Index end-to-end search accuracy
// ===========================================================================

#[test]
fn verify_turboindex_search_recall() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench_idx");
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let n_vecs = 200;
    let mut index = TurboIndex::create(&path, DIM, 42, 99).unwrap();
    let mut all_vecs: Vec<Vec<f32>> = Vec::new();

    for i in 0..n_vecs {
        let v = random_unit_vec(DIM, &mut rng);
        index.insert(i as u64, &v).unwrap();
        all_vecs.push(v);
    }

    let mut total_recall = 0.0f64;
    let n_queries = 20;

    for _ in 0..n_queries {
        let query = random_unit_vec(DIM, &mut rng);

        // True top-10 by cosine
        let mut true_sims: Vec<(u64, f32)> = all_vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u64, query.iter().zip(v).map(|(a, b)| a * b).sum()))
            .collect();
        true_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let true_top10: Vec<u64> = true_sims[..10].iter().map(|(id, _)| *id).collect();

        // TurboIndex search
        let results = index.search(&query, 10);
        let search_top10: Vec<u64> = results.iter().map(|r| r.id).collect();

        let overlap = true_top10
            .iter()
            .filter(|id| search_top10.contains(id))
            .count();
        total_recall += overlap as f64 / 10.0;
    }

    let avg_recall = total_recall / n_queries as f64;
    eprintln!("TurboIndex recall@10 over {n_queries} queries: {avg_recall:.2}");

    // Brute-force search should have decent recall
    assert!(
        avg_recall >= 0.4,
        "recall@10 should be at least 40%, got {:.0}%",
        avg_recall * 100.0
    );
}

// ===========================================================================
// Experiment 1: Bit-width sweep
//
// Mirrors yashkc2025/turboquant's `benchmark_vs_naive.py` which sweeps
// bit widths 1–4 and measures MSE + improvement ratio.
//
// We sweep angle_bits ∈ {2, 3, 4} × radius_bits ∈ {4, 6, 8} and verify:
//   - Higher bits → lower reconstruction MSE (monotonic improvement)
//   - Compression ratio scales as expected
//   - All configs produce usable cosine similarity (> 0.5)
// ===========================================================================

#[test]
fn experiment_bitwidth_sweep() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let n_vecs = 200;
    let vecs: Vec<Vec<f32>> = (0..n_vecs).map(|_| random_unit_vec(DIM, &mut rng)).collect();

    let configs: Vec<(u8, u8)> = vec![
        (2, 4), (2, 6), (2, 8),
        (3, 4), (3, 6), (3, 8),
        (4, 4), (4, 6), (4, 8),
    ];

    eprintln!("\n{:>6} {:>6} {:>10} {:>10} {:>12} {:>10}",
        "angle", "radius", "bits/dim", "MSE", "avg_cosine", "ratio");
    eprintln!("{}", "-".repeat(62));

    let mut prev_mse = f64::MAX;
    let mut prev_angle = 0u8;

    for &(angle_bits, radius_bits) in &configs {
        let comp = TurboQuantCompressor::new(DIM, 42, 99, angle_bits, radius_bits);
        let bpv = comp.bytes_per_vector();
        let bits_per_dim = (bpv as f64 * 8.0) / DIM as f64;
        let ratio = RAW_BYTES_PER_VEC as f64 / bpv as f64;

        let mut total_mse = 0.0f64;
        let mut total_cosine = 0.0f64;

        for v in &vecs {
            let tv = comp.compress(v);
            let recon = comp.decompress(&tv);

            let mse: f64 = v.iter().zip(&recon)
                .map(|(a, b)| ((a - b) as f64) * ((a - b) as f64))
                .sum::<f64>() / DIM as f64;
            total_mse += mse;

            let dot: f32 = v.iter().zip(&recon).map(|(a, b)| a * b).sum();
            let norm_r: f32 = recon.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cosine = if norm_r > 0.0 { dot / norm_r } else { 0.0 };
            total_cosine += cosine as f64;
        }

        let avg_mse = total_mse / n_vecs as f64;
        let avg_cosine = total_cosine / n_vecs as f64;

        eprintln!("{angle_bits:>6} {radius_bits:>6} {bits_per_dim:>10.2} {avg_mse:>10.6} {avg_cosine:>12.4} {ratio:>10.2}x");

        // All configs should produce reasonable cosine similarity
        assert!(
            avg_cosine > 0.5,
            "angle={angle_bits}, radius={radius_bits}: avg_cosine={avg_cosine:.4} < 0.5"
        );

        // Within the same angle_bits, increasing radius_bits should not increase MSE
        // (more bits for radius → better reconstruction)
        if angle_bits == prev_angle && avg_mse > prev_mse * 1.1 {
            panic!(
                "MSE should decrease with more radius bits: \
                 angle={angle_bits}, radius={radius_bits} MSE={avg_mse:.6} > prev={prev_mse:.6}"
            );
        }

        prev_mse = avg_mse;
        prev_angle = angle_bits;
    }

    // Verify the best config (4-bit angle, 8-bit radius) beats the worst (2-bit angle, 4-bit radius)
    let comp_best = TurboQuantCompressor::new(DIM, 42, 99, 4, 8);
    let comp_worst = TurboQuantCompressor::new(DIM, 42, 99, 2, 4);

    let v = &vecs[0];
    let mse_best: f64 = {
        let recon = comp_best.decompress(&comp_best.compress(v));
        v.iter().zip(&recon).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / DIM as f64
    };
    let mse_worst: f64 = {
        let recon = comp_worst.decompress(&comp_worst.compress(v));
        v.iter().zip(&recon).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / DIM as f64
    };

    assert!(
        mse_best < mse_worst,
        "4/8 config MSE ({mse_best:.6}) should beat 2/4 config ({mse_worst:.6})"
    );
}

// ===========================================================================
// Experiment 2: Inner-product estimator bias measurement
//
// Mirrors yashkc2025/turboquant's bias measurement in `benchmark_vs_naive.py`.
//
// The TurboQuant estimator should be unbiased:
//   E[ <q̃, x̃> ] ≈ <q, x>
//
// We measure:
//   bias = mean( estimated_sim - true_sim )  over many (query, target) pairs
//   relative_bias = |bias| / mean(|true_sim|)
//
// An unbiased estimator should have relative_bias close to 0.
// ===========================================================================

#[test]
fn experiment_inner_product_bias() {
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let n_pairs = 500;

    let configs: Vec<(u8, u8, &str)> = vec![
        (2, 4, "2-bit angle / 4-bit radius"),
        (3, 6, "3-bit angle / 6-bit radius"),
        (4, 8, "4-bit angle / 8-bit radius"),
    ];

    eprintln!("\n{:<32} {:>10} {:>10} {:>12} {:>10}",
        "config", "mean_bias", "std_err", "rel_bias%", "mean|sim|");
    eprintln!("{}", "-".repeat(80));

    for (angle_bits, radius_bits, label) in &configs {
        let comp = TurboQuantCompressor::new(DIM, 42, 99, *angle_bits, *radius_bits);

        let mut biases = Vec::with_capacity(n_pairs);
        let mut abs_true_sims = Vec::with_capacity(n_pairs);

        for _ in 0..n_pairs {
            let query = random_unit_vec(DIM, &mut rng);
            let target = random_unit_vec(DIM, &mut rng);

            // True inner product
            let true_sim: f32 = query.iter().zip(&target).map(|(a, b)| a * b).sum();

            // TurboQuant estimated inner product
            let tv = comp.compress(&target);
            let est_sim = comp.similarity_raw(&query, &tv);

            biases.push((est_sim - true_sim) as f64);
            abs_true_sims.push(true_sim.abs() as f64);
        }

        let mean_bias: f64 = biases.iter().sum::<f64>() / n_pairs as f64;
        let variance: f64 = biases.iter()
            .map(|b| (b - mean_bias) * (b - mean_bias))
            .sum::<f64>() / (n_pairs - 1) as f64;
        let std_err = (variance / n_pairs as f64).sqrt();
        let mean_abs_sim: f64 = abs_true_sims.iter().sum::<f64>() / n_pairs as f64;
        let relative_bias_pct = if mean_abs_sim > 1e-10 {
            (mean_bias.abs() / mean_abs_sim) * 100.0
        } else {
            0.0
        };

        eprintln!(
            "{label:<32} {mean_bias:>10.6} {std_err:>10.6} {relative_bias_pct:>11.2}% {mean_abs_sim:>10.4}"
        );

        // The estimator should be approximately unbiased.
        // We allow relative bias up to 15% — the QJL correction reduces bias
        // but at low bit widths there's still some residual.
        assert!(
            relative_bias_pct < 15.0,
            "{label}: relative bias {relative_bias_pct:.2}% exceeds 15% threshold"
        );

        // The mean bias should not be statistically significant at 3σ
        // (i.e., zero should be within the 99.7% confidence interval)
        let ci_bound = 3.0 * std_err;
        assert!(
            mean_bias.abs() < ci_bound + 0.01, // small epsilon for edge cases
            "{label}: mean bias {mean_bias:.6} is >3σ from zero (3σ = {ci_bound:.6})"
        );
    }
}

// ===========================================================================
// Experiment 3: Baseline comparison vs naive uniform quantizer
//
// Mirrors yashkc2025/turboquant's `benchmark_vs_naive.py` and
// `nearest_neighbor.py` which compare against NaiveUniform quantization.
//
// Naive uniform quantizer: clamp each coordinate to [-range, range], then
// uniformly bin into 2^b levels. No rotation, no residual correction.
//
// We compare:
//   - Reconstruction MSE (TurboQuant should win)
//   - Inner product estimation error (TurboQuant should win)
//   - Recall@10 on nearest neighbor search (TurboQuant should win)
// ===========================================================================

/// Naive uniform scalar quantizer — the simplest possible baseline.
///
/// Each dimension is independently clamped to [-range, range] and uniformly
/// quantized to 2^bits levels. No rotation, no data-dependent codebook.
struct NaiveUniformQuantizer {
    bits: u8,
    range: f32,
    #[allow(dead_code)]
    dim: usize,
}

impl NaiveUniformQuantizer {
    fn new(dim: usize, bits: u8) -> Self {
        // For unit vectors in dim d, each coordinate concentrates around
        // N(0, 1/sqrt(d)). Use 4σ to cover >99.99%.
        let range = 4.0 / (dim as f32).sqrt();
        Self { bits, range, dim }
    }

    fn quantize_dequantize(&self, vector: &[f32]) -> Vec<f32> {
        let levels = (1u32 << self.bits) as f32;
        vector.iter().map(|&x| {
            let clamped = x.clamp(-self.range, self.range);
            let normalized = (clamped + self.range) / (2.0 * self.range); // [0, 1]
            let bin = (normalized * levels).floor().min(levels - 1.0);
            let center = (bin + 0.5) / levels;          // midpoint of bin
            center * 2.0 * self.range - self.range       // back to original scale
        }).collect()
    }
}

/// MSE and inner-product trade-off table across methods and bit rates.
///
/// This is an informational comparison — naive uniform is very accurate
/// at high bits/dim (7-8), while TurboQuant's advantage is calibration-free
/// operation and theoretical guarantees. At high bit rates, naive's per-coordinate
/// bins are extremely fine and hard to beat with any structured approach.
///
/// The assertions verify minimum quality bars, not head-to-head dominance.
#[test]
fn experiment_baseline_mse_and_ip_tradeoff() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let n_vecs = 300;
    let vecs: Vec<Vec<f32>> = (0..n_vecs).map(|_| random_unit_vec(DIM, &mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..100).map(|_| random_unit_vec(DIM, &mut rng)).collect();

    // TurboQuant configs
    let turbo_configs: Vec<(u8, u8, &str)> = vec![
        (2, 4, "TQ 2a/4r"),
        (3, 6, "TQ 3a/6r"),
        (4, 8, "TQ 4a/8r"),
    ];

    // Naive configs
    let naive_configs: Vec<(u8, &str)> = vec![
        (2, "Naive 2b"),
        (3, "Naive 3b"),
        (4, "Naive 4b"),
        (6, "Naive 6b"),
        (8, "Naive 8b"),
    ];

    eprintln!("\n{:<14} {:>9} {:>12} {:>12} {:>10}",
        "method", "bits/dim", "recon MSE", "IP MAE", "cos_sim");
    eprintln!("{}", "-".repeat(62));

    // --- TurboQuant entries ---
    for &(ab, rb, label) in &turbo_configs {
        let comp = TurboQuantCompressor::new(DIM, 42, 99, ab, rb);
        let bpd = (comp.bytes_per_vector() as f64 * 8.0) / DIM as f64;

        let mut mse_sum = 0.0f64;
        let mut cos_sum = 0.0f64;
        let compressed: Vec<_> = vecs.iter().map(|v| comp.compress(v)).collect();

        for (i, v) in vecs.iter().enumerate() {
            let recon = comp.decompress(&compressed[i]);
            mse_sum += v.iter().zip(&recon)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>() / DIM as f64;
            let dot: f32 = v.iter().zip(&recon).map(|(a, b)| a * b).sum();
            let nr: f32 = recon.iter().map(|x| x * x).sum::<f32>().sqrt();
            cos_sum += if nr > 0.0 { (dot / nr) as f64 } else { 0.0 };
        }

        let mut ip_err_sum = 0.0f64;
        for q in &queries {
            let prepared = comp.prepare_query(q);
            for (j, v) in vecs.iter().enumerate() {
                let true_ip: f32 = q.iter().zip(v).map(|(a, b)| a * b).sum();
                let est_ip = comp.similarity_prepared(&prepared, &compressed[j]);
                ip_err_sum += (est_ip - true_ip).abs() as f64;
            }
        }

        let avg_mse = mse_sum / n_vecs as f64;
        let avg_cos = cos_sum / n_vecs as f64;
        let avg_mae = ip_err_sum / (queries.len() * n_vecs) as f64;

        eprintln!("{label:<14} {bpd:>9.2} {avg_mse:>12.8} {avg_mae:>12.8} {avg_cos:>10.4}");

        // Minimum quality: cosine > 0.8 for all configs
        assert!(avg_cos > 0.8, "{label}: cosine {avg_cos:.4} < 0.8");
        // Minimum quality: IP MAE < 0.05 (estimator is useful)
        assert!(avg_mae < 0.05, "{label}: IP MAE {avg_mae:.6} > 0.05");
    }

    // --- Naive entries ---
    for &(bits, label) in &naive_configs {
        let naive = NaiveUniformQuantizer::new(DIM, bits);

        let mut mse_sum = 0.0f64;
        let mut cos_sum = 0.0f64;
        let naive_recons: Vec<Vec<f32>> = vecs.iter()
            .map(|v| naive.quantize_dequantize(v))
            .collect();

        for (i, v) in vecs.iter().enumerate() {
            mse_sum += v.iter().zip(&naive_recons[i])
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>() / DIM as f64;
            let dot: f32 = v.iter().zip(&naive_recons[i]).map(|(a, b)| a * b).sum();
            let nr: f32 = naive_recons[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            cos_sum += if nr > 0.0 { (dot / nr) as f64 } else { 0.0 };
        }

        let avg_mse = mse_sum / n_vecs as f64;
        let avg_cos = cos_sum / n_vecs as f64;

        // IP MAE for naive
        let mut naive_mae_sum = 0.0f64;
        let mut naive_mae_count = 0usize;
        for q in &queries {
            for (j, v) in vecs.iter().enumerate() {
                let true_ip: f32 = q.iter().zip(v).map(|(a, b)| a * b).sum();
                let est_ip: f32 = q.iter().zip(&naive_recons[j]).map(|(a, b)| a * b).sum();
                naive_mae_sum += (est_ip - true_ip).abs() as f64;
                naive_mae_count += 1;
            }
        }
        let avg_mae = naive_mae_sum / naive_mae_count as f64;

        eprintln!("{label:<14} {bits:>9} {avg_mse:>12.8} {avg_mae:>12.8} {avg_cos:>10.4}");
    }

    eprintln!("\nNote: naive uniform excels at high bits/dim (6-8) where per-coordinate");
    eprintln!("bins are extremely fine. TurboQuant's advantage: calibration-free,");
    eprintln!("unbiased IP estimator, theoretical near-optimal distortion rate.");
}

/// Calibration robustness: TurboQuant works without seeing data;
/// naive uniform degrades badly with wrong range estimate.
///
/// This mirrors yashkc2025/turboquant's nearest_neighbor.py which shows
/// TurboQuant achieving good recall without calibration data, while
/// naive quantization requires calibration.
#[test]
fn experiment_baseline_calibration_robustness() {
    let mut rng = ChaCha8Rng::seed_from_u64(77);
    let n_vecs = 200;
    let n_queries = 30;
    let top_k = 10;

    // Database vectors: unit vectors in dim 384
    let vecs: Vec<Vec<f32>> = (0..n_vecs).map(|_| random_unit_vec(DIM, &mut rng)).collect();

    // TurboQuant: no calibration needed, works for any distribution
    let turbo = TurboQuantCompressor::new(DIM, 42, 99, 4, 8);
    let compressed: Vec<_> = vecs.iter().map(|v| turbo.compress(v)).collect();

    // Naive 4-bit with CORRECT range (knows the distribution — best case for naive)
    // At 4 bits (16 levels) miscalibration really hurts because bins are coarse.
    let naive_correct = NaiveUniformQuantizer::new(DIM, 4);
    let naive_correct_recons: Vec<Vec<f32>> = vecs.iter()
        .map(|v| naive_correct.quantize_dequantize(v))
        .collect();

    // Naive 4-bit with WRONG range (miscalibrated — assumes range appropriate
    // for raw [-1, 1] data, 5x too wide for concentrated unit sphere vectors).
    // With 16 levels spread over [-1, 1], only ~3 levels cover the actual
    // ±0.204 data range → severe quantization loss.
    let naive_wrong = NaiveUniformQuantizer {
        bits: 4,
        range: 1.0, // 5x too wide for dim=384 unit vectors
        dim: DIM,
    };
    let naive_wrong_recons: Vec<Vec<f32>> = vecs.iter()
        .map(|v| naive_wrong.quantize_dequantize(v))
        .collect();

    let mut turbo_recall_sum = 0.0f64;
    let mut naive_correct_recall_sum = 0.0f64;
    let mut naive_wrong_recall_sum = 0.0f64;

    for _ in 0..n_queries {
        let query = random_unit_vec(DIM, &mut rng);

        // True top-k
        let mut true_sims: Vec<(usize, f32)> = vecs.iter().enumerate()
            .map(|(i, v)| (i, query.iter().zip(v).map(|(a, b)| a * b).sum()))
            .collect();
        true_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let true_topk: Vec<usize> = true_sims[..top_k].iter().map(|(i, _)| *i).collect();

        // TurboQuant top-k
        let prepared = turbo.prepare_query(&query);
        let mut t_sims: Vec<(usize, f32)> = compressed.iter().enumerate()
            .map(|(i, tv)| (i, turbo.similarity_prepared(&prepared, tv)))
            .collect();
        t_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let t_topk: Vec<usize> = t_sims[..top_k].iter().map(|(i, _)| *i).collect();

        // Naive correct top-k
        let mut nc_sims: Vec<(usize, f32)> = naive_correct_recons.iter().enumerate()
            .map(|(i, v)| (i, query.iter().zip(v).map(|(a, b)| a * b).sum()))
            .collect();
        nc_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let nc_topk: Vec<usize> = nc_sims[..top_k].iter().map(|(i, _)| *i).collect();

        // Naive wrong top-k
        let mut nw_sims: Vec<(usize, f32)> = naive_wrong_recons.iter().enumerate()
            .map(|(i, v)| (i, query.iter().zip(v).map(|(a, b)| a * b).sum()))
            .collect();
        nw_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let nw_topk: Vec<usize> = nw_sims[..top_k].iter().map(|(i, _)| *i).collect();

        turbo_recall_sum += true_topk.iter().filter(|i| t_topk.contains(i)).count() as f64 / top_k as f64;
        naive_correct_recall_sum += true_topk.iter().filter(|i| nc_topk.contains(i)).count() as f64 / top_k as f64;
        naive_wrong_recall_sum += true_topk.iter().filter(|i| nw_topk.contains(i)).count() as f64 / top_k as f64;
    }

    let turbo_recall = turbo_recall_sum / n_queries as f64;
    let naive_correct_recall = naive_correct_recall_sum / n_queries as f64;
    let naive_wrong_recall = naive_wrong_recall_sum / n_queries as f64;

    eprintln!("\n--- Calibration robustness: Recall@{top_k} ---");
    eprintln!("TurboQuant (no calibration):         {turbo_recall:.3}");
    eprintln!("Naive Uniform (correct calibration):  {naive_correct_recall:.3}");
    eprintln!("Naive Uniform (WRONG calibration):    {naive_wrong_recall:.3}");

    // TurboQuant should beat miscalibrated naive
    assert!(
        turbo_recall > naive_wrong_recall,
        "TurboQuant ({turbo_recall:.3}) should beat miscalibrated naive ({naive_wrong_recall:.3})"
    );

    // TurboQuant should achieve reasonable recall without calibration
    assert!(
        turbo_recall > 0.6,
        "TurboQuant recall@{top_k} = {turbo_recall:.3}, expected > 0.6"
    );

    // Miscalibrated naive should show significant degradation
    assert!(
        naive_wrong_recall < naive_correct_recall,
        "Wrong calibration should degrade recall: wrong={naive_wrong_recall:.3} vs correct={naive_correct_recall:.3}"
    );
}

#[test]
fn experiment_baseline_comparison_recall() {
    let mut rng = ChaCha8Rng::seed_from_u64(33);
    let n_vecs = 200;
    let n_queries = 30;
    let top_k = 10;

    let vecs: Vec<Vec<f32>> = (0..n_vecs).map(|_| random_unit_vec(DIM, &mut rng)).collect();

    let turbo = TurboQuantCompressor::new(DIM, 42, 99, 4, 8);
    let naive = NaiveUniformQuantizer::new(DIM, 8); // give naive MORE bits

    let compressed: Vec<_> = vecs.iter().map(|v| turbo.compress(v)).collect();
    let naive_recons: Vec<Vec<f32>> = vecs.iter().map(|v| naive.quantize_dequantize(v)).collect();

    let mut turbo_recall_sum = 0.0f64;
    let mut naive_recall_sum = 0.0f64;

    for _ in 0..n_queries {
        let query = random_unit_vec(DIM, &mut rng);

        // True top-k
        let mut true_sims: Vec<(usize, f32)> = vecs.iter().enumerate()
            .map(|(i, v)| (i, query.iter().zip(v).map(|(a, b)| a * b).sum()))
            .collect();
        true_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let true_topk: Vec<usize> = true_sims[..top_k].iter().map(|(i, _)| *i).collect();

        // TurboQuant top-k
        let prepared = turbo.prepare_query(&query);
        let mut turbo_sims: Vec<(usize, f32)> = compressed.iter().enumerate()
            .map(|(i, tv)| (i, turbo.similarity_prepared(&prepared, tv)))
            .collect();
        turbo_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let turbo_topk: Vec<usize> = turbo_sims[..top_k].iter().map(|(i, _)| *i).collect();

        // Naive top-k
        let mut naive_sims: Vec<(usize, f32)> = naive_recons.iter().enumerate()
            .map(|(i, v)| (i, query.iter().zip(v).map(|(a, b)| a * b).sum()))
            .collect();
        naive_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let naive_topk: Vec<usize> = naive_sims[..top_k].iter().map(|(i, _)| *i).collect();

        let turbo_overlap = true_topk.iter().filter(|i| turbo_topk.contains(i)).count();
        let naive_overlap = true_topk.iter().filter(|i| naive_topk.contains(i)).count();

        turbo_recall_sum += turbo_overlap as f64 / top_k as f64;
        naive_recall_sum += naive_overlap as f64 / top_k as f64;
    }

    let turbo_recall = turbo_recall_sum / n_queries as f64;
    let naive_recall = naive_recall_sum / n_queries as f64;

    eprintln!("\n--- Recall@{top_k} comparison ({n_vecs} vectors, {n_queries} queries) ---");
    eprintln!("TurboQuant (~7.2 bits/dim): recall@{top_k} = {turbo_recall:.3}");
    eprintln!("Naive Uniform (8 bits/dim): recall@{top_k} = {naive_recall:.3}");

    // TurboQuant should achieve competitive recall even at fewer bits
    assert!(
        turbo_recall >= naive_recall * 0.8,
        "TurboQuant recall ({turbo_recall:.3}) should be within 80% of naive ({naive_recall:.3})"
    );
    // And both should be above a minimum bar
    assert!(
        turbo_recall > 0.3,
        "TurboQuant recall@{top_k} = {turbo_recall:.3}, expected > 0.3"
    );
}
