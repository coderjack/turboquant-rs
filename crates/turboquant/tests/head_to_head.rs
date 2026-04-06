//! Head-to-head benchmark: PolarQuant vs Lloyd-Max at matched bit rates.
//!
//! Both use the same rotation (step 1) and QJL residual correction (step 3).
//! The only difference is step 2: PolarQuant vs Lloyd-Max scalar quantization.
//!
//! Run with: cargo test -p turboquant --test head_to_head -- --nocapture

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use turboquant::{LloydMaxCompressor, TurboQuantCompressor};

const DIM: usize = 384;

fn random_unit_vec(dim: usize, rng: &mut ChaCha8Rng) -> Vec<f32> {
    let normal = Normal::new(0.0, 1.0f64).unwrap();
    let raw: Vec<f32> = (0..dim).map(|_| normal.sample(rng) as f32).collect();
    let norm: f32 = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
    raw.iter().map(|v| v / norm).collect()
}

// ===========================================================================
// Matched bit-rate configurations
//
// PolarQuant: each pair of dims uses (angle_bits + radius_bits) bits,
//   plus 1 bit/dim for QJL, plus 8 bytes (norms) amortized.
//   Effective bits/dim = (angle_bits + radius_bits) / 2 + 1 + 8*8/dim
//
// Lloyd-Max: each dim uses `bits` bits,
//   plus 1 bit/dim for QJL, plus 8 bytes (norms) amortized.
//   Effective bits/dim = bits + 1 + 8*8/dim
//
// We compare at the TOTAL bytes/vec level (what actually matters for storage).
// ===========================================================================

struct PQConfig {
    angle_bits: u8,
    radius_bits: u8,
    label: &'static str,
}

struct LMConfig {
    bits: u8,
    label: &'static str,
}

fn pq_configs() -> Vec<PQConfig> {
    vec![
        PQConfig { angle_bits: 2, radius_bits: 2, label: "PQ 2a/2r" },
        PQConfig { angle_bits: 2, radius_bits: 4, label: "PQ 2a/4r" },
        PQConfig { angle_bits: 3, radius_bits: 4, label: "PQ 3a/4r" },
        PQConfig { angle_bits: 3, radius_bits: 6, label: "PQ 3a/6r" },
        PQConfig { angle_bits: 4, radius_bits: 6, label: "PQ 4a/6r" },
        PQConfig { angle_bits: 4, radius_bits: 8, label: "PQ 4a/8r" },
    ]
}

fn lm_configs() -> Vec<LMConfig> {
    vec![
        LMConfig { bits: 1, label: "LM 1-bit" },
        LMConfig { bits: 2, label: "LM 2-bit" },
        LMConfig { bits: 3, label: "LM 3-bit" },
        LMConfig { bits: 4, label: "LM 4-bit" },
        LMConfig { bits: 6, label: "LM 6-bit" },
        LMConfig { bits: 8, label: "LM 8-bit" },
    ]
}

// ===========================================================================
// Test 1: Reconstruction MSE + cosine at all configs
// ===========================================================================

#[test]
fn head_to_head_reconstruction_quality() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let n_vecs = 300;
    let vecs: Vec<Vec<f32>> = (0..n_vecs).map(|_| random_unit_vec(DIM, &mut rng)).collect();

    eprintln!("\n============================================================");
    eprintln!("  RECONSTRUCTION QUALITY: PolarQuant vs Lloyd-Max ({DIM}-dim)");
    eprintln!("============================================================\n");
    eprintln!(
        "{:<12} {:>10} {:>10} {:>12} {:>10}",
        "method", "bytes/vec", "bits/dim", "recon MSE", "avg cos"
    );
    eprintln!("{}", "-".repeat(58));

    // --- PolarQuant configs ---
    for cfg in pq_configs() {
        let comp = TurboQuantCompressor::new(DIM, 42, 99, cfg.angle_bits, cfg.radius_bits);
        let bpv = comp.bytes_per_vector();
        let bpd = (bpv as f64 * 8.0) / DIM as f64;

        let (mse, cos) = measure_reconstruction(&comp, &vecs);
        eprintln!(
            "{:<12} {:>10} {:>10.2} {:>12.8} {:>10.4}",
            cfg.label, bpv, bpd, mse, cos
        );
    }

    eprintln!("{}", "-".repeat(58));

    // --- Lloyd-Max configs ---
    for cfg in lm_configs() {
        let comp = LloydMaxCompressor::new(DIM, 42, 99, cfg.bits);
        let bpv = comp.bytes_per_vector();
        let bpd = (bpv as f64 * 8.0) / DIM as f64;

        let (mse, cos) = measure_reconstruction_lm(&comp, &vecs);
        eprintln!(
            "{:<12} {:>10} {:>10.2} {:>12.8} {:>10.4}",
            cfg.label, bpv, bpd, mse, cos
        );
    }

    eprintln!();
}

fn measure_reconstruction(
    comp: &TurboQuantCompressor,
    vecs: &[Vec<f32>],
) -> (f64, f64) {
    let mut mse_sum = 0.0f64;
    let mut cos_sum = 0.0f64;
    for v in vecs {
        let tv = comp.compress(v);
        let recon = comp.decompress(&tv);
        mse_sum += per_dim_mse(v, &recon);
        cos_sum += cosine_sim(v, &recon);
    }
    (mse_sum / vecs.len() as f64, cos_sum / vecs.len() as f64)
}

fn measure_reconstruction_lm(
    comp: &LloydMaxCompressor,
    vecs: &[Vec<f32>],
) -> (f64, f64) {
    let mut mse_sum = 0.0f64;
    let mut cos_sum = 0.0f64;
    for v in vecs {
        let lv = comp.compress(v);
        let recon = comp.decompress(&lv);
        mse_sum += per_dim_mse(v, &recon);
        cos_sum += cosine_sim(v, &recon);
    }
    (mse_sum / vecs.len() as f64, cos_sum / vecs.len() as f64)
}

fn per_dim_mse(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| ((x - y) as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if nb > 0.0 { (dot / nb) as f64 } else { 0.0 }
}

// ===========================================================================
// Test 2: Inner product estimation accuracy (with QJL correction)
// ===========================================================================

#[test]
fn head_to_head_inner_product_accuracy() {
    let mut rng = ChaCha8Rng::seed_from_u64(55);
    let n_pairs = 400;

    let queries: Vec<Vec<f32>> = (0..n_pairs).map(|_| random_unit_vec(DIM, &mut rng)).collect();
    let targets: Vec<Vec<f32>> = (0..n_pairs).map(|_| random_unit_vec(DIM, &mut rng)).collect();

    let true_ips: Vec<f32> = queries
        .iter()
        .zip(&targets)
        .map(|(q, t)| q.iter().zip(t).map(|(a, b)| a * b).sum())
        .collect();

    eprintln!("\n============================================================");
    eprintln!("  INNER PRODUCT ACCURACY: PolarQuant vs Lloyd-Max ({DIM}-dim)");
    eprintln!("============================================================\n");
    eprintln!(
        "{:<12} {:>10} {:>10} {:>12} {:>10}",
        "method", "bytes/vec", "bits/dim", "IP MAE", "IP bias"
    );
    eprintln!("{}", "-".repeat(58));

    // PolarQuant
    for cfg in pq_configs() {
        let comp = TurboQuantCompressor::new(DIM, 42, 99, cfg.angle_bits, cfg.radius_bits);
        let bpv = comp.bytes_per_vector();
        let bpd = (bpv as f64 * 8.0) / DIM as f64;

        let mut errors = Vec::with_capacity(n_pairs);
        for (i, (q, t)) in queries.iter().zip(&targets).enumerate() {
            let tv = comp.compress(t);
            let est = comp.similarity_raw(q, &tv);
            errors.push((est - true_ips[i]) as f64);
        }

        let mae: f64 = errors.iter().map(|e| e.abs()).sum::<f64>() / n_pairs as f64;
        let bias: f64 = errors.iter().sum::<f64>() / n_pairs as f64;
        eprintln!(
            "{:<12} {:>10} {:>10.2} {:>12.6} {:>+10.6}",
            cfg.label, bpv, bpd, mae, bias
        );
    }

    eprintln!("{}", "-".repeat(58));

    // Lloyd-Max
    for cfg in lm_configs() {
        let comp = LloydMaxCompressor::new(DIM, 42, 99, cfg.bits);
        let bpv = comp.bytes_per_vector();
        let bpd = (bpv as f64 * 8.0) / DIM as f64;

        let mut errors = Vec::with_capacity(n_pairs);
        for (i, (q, t)) in queries.iter().zip(&targets).enumerate() {
            let lv = comp.compress(t);
            let est = comp.similarity_raw(q, &lv);
            errors.push((est - true_ips[i]) as f64);
        }

        let mae: f64 = errors.iter().map(|e| e.abs()).sum::<f64>() / n_pairs as f64;
        let bias: f64 = errors.iter().sum::<f64>() / n_pairs as f64;
        eprintln!(
            "{:<12} {:>10} {:>10.2} {:>12.6} {:>+10.6}",
            cfg.label, bpv, bpd, mae, bias
        );
    }

    eprintln!();
}

// ===========================================================================
// Test 3: Recall@10 search accuracy
// ===========================================================================

#[test]
fn head_to_head_recall() {
    let mut rng = ChaCha8Rng::seed_from_u64(33);
    let n_vecs = 200;
    let n_queries = 30;
    let top_k = 10;

    let vecs: Vec<Vec<f32>> = (0..n_vecs).map(|_| random_unit_vec(DIM, &mut rng)).collect();
    let queries: Vec<Vec<f32>> =
        (0..n_queries).map(|_| random_unit_vec(DIM, &mut rng)).collect();

    // True top-k for each query
    let true_topks: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| {
            let mut sims: Vec<(usize, f32)> = vecs
                .iter()
                .enumerate()
                .map(|(i, v)| (i, q.iter().zip(v).map(|(a, b)| a * b).sum()))
                .collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            sims[..top_k].iter().map(|(i, _)| *i).collect()
        })
        .collect();

    eprintln!("\n============================================================");
    eprintln!("  RECALL@{top_k}: PolarQuant vs Lloyd-Max ({DIM}-dim, {n_vecs} vecs)");
    eprintln!("============================================================\n");
    eprintln!(
        "{:<12} {:>10} {:>10} {:>12}",
        "method", "bytes/vec", "bits/dim", "recall@10"
    );
    eprintln!("{}", "-".repeat(46));

    // Subset of configs to keep test fast
    let pq_test = vec![
        PQConfig { angle_bits: 2, radius_bits: 4, label: "PQ 2a/4r" },
        PQConfig { angle_bits: 3, radius_bits: 6, label: "PQ 3a/6r" },
        PQConfig { angle_bits: 4, radius_bits: 8, label: "PQ 4a/8r" },
    ];

    let lm_test = vec![
        LMConfig { bits: 2, label: "LM 2-bit" },
        LMConfig { bits: 3, label: "LM 3-bit" },
        LMConfig { bits: 4, label: "LM 4-bit" },
    ];

    for cfg in &pq_test {
        let comp = TurboQuantCompressor::new(DIM, 42, 99, cfg.angle_bits, cfg.radius_bits);
        let bpv = comp.bytes_per_vector();
        let bpd = (bpv as f64 * 8.0) / DIM as f64;

        let compressed: Vec<_> = vecs.iter().map(|v| comp.compress(v)).collect();
        let mut recall_sum = 0.0f64;

        for (qi, q) in queries.iter().enumerate() {
            let prepared = comp.prepare_query(q);
            let mut sims: Vec<(usize, f32)> = compressed
                .iter()
                .enumerate()
                .map(|(i, tv)| (i, comp.similarity_prepared(&prepared, tv)))
                .collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let est_topk: Vec<usize> = sims[..top_k].iter().map(|(i, _)| *i).collect();
            let overlap = true_topks[qi]
                .iter()
                .filter(|i| est_topk.contains(i))
                .count();
            recall_sum += overlap as f64 / top_k as f64;
        }

        let recall = recall_sum / n_queries as f64;
        eprintln!(
            "{:<12} {:>10} {:>10.2} {:>12.3}",
            cfg.label, bpv, bpd, recall
        );
    }

    eprintln!("{}", "-".repeat(46));

    for cfg in &lm_test {
        let comp = LloydMaxCompressor::new(DIM, 42, 99, cfg.bits);
        let bpv = comp.bytes_per_vector();
        let bpd = (bpv as f64 * 8.0) / DIM as f64;

        let compressed: Vec<_> = vecs.iter().map(|v| comp.compress(v)).collect();
        let mut recall_sum = 0.0f64;

        for (qi, q) in queries.iter().enumerate() {
            let prepared = comp.prepare_query(q);
            let mut sims: Vec<(usize, f32)> = compressed
                .iter()
                .enumerate()
                .map(|(i, lv)| (i, comp.similarity_prepared(&prepared, lv)))
                .collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let est_topk: Vec<usize> = sims[..top_k].iter().map(|(i, _)| *i).collect();
            let overlap = true_topks[qi]
                .iter()
                .filter(|i| est_topk.contains(i))
                .count();
            recall_sum += overlap as f64 / top_k as f64;
        }

        let recall = recall_sum / n_queries as f64;
        eprintln!(
            "{:<12} {:>10} {:>10.2} {:>12.3}",
            cfg.label, bpv, bpd, recall
        );
    }

    eprintln!();
}

// ===========================================================================
// Test 4: Compression ratio summary
// ===========================================================================

#[test]
fn head_to_head_compression_summary() {
    let raw_bytes = DIM * 4;

    eprintln!("\n============================================================");
    eprintln!("  COMPRESSION SUMMARY: all configs ({DIM}-dim, f32 = {raw_bytes} bytes)");
    eprintln!("============================================================\n");
    eprintln!(
        "{:<12} {:>7} {:>10} {:>10} {:>12}",
        "method", "q-bits", "bytes/vec", "bits/dim", "ratio"
    );
    eprintln!("{}", "-".repeat(55));

    for cfg in pq_configs() {
        let comp = TurboQuantCompressor::new(DIM, 42, 99, cfg.angle_bits, cfg.radius_bits);
        let bpv = comp.bytes_per_vector();
        let bpd = (bpv as f64 * 8.0) / DIM as f64;
        let ratio = raw_bytes as f64 / bpv as f64;
        let q_bits = format!("{}a+{}r", cfg.angle_bits, cfg.radius_bits);
        eprintln!(
            "{:<12} {:>7} {:>10} {:>10.2} {:>11.1}x",
            cfg.label, q_bits, bpv, bpd, ratio
        );
    }

    eprintln!("{}", "-".repeat(55));

    for cfg in lm_configs() {
        let comp = LloydMaxCompressor::new(DIM, 42, 99, cfg.bits);
        let bpv = comp.bytes_per_vector();
        let bpd = (bpv as f64 * 8.0) / DIM as f64;
        let ratio = raw_bytes as f64 / bpv as f64;
        eprintln!(
            "{:<12} {:>7} {:>10} {:>10.2} {:>11.1}x",
            cfg.label, cfg.bits, bpv, bpd, ratio
        );
    }

    eprintln!();
}
