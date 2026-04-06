//! Using each TurboQuant component independently.
//!
//! Shows how to use Rotation, LloydMaxQuantizer, and QjlCompressor as
//! standalone building blocks — useful for KV cache compression, custom
//! pipelines, or understanding what each step does.
//!
//! Run with: cargo run -p turboquant --example components

use turboquant::{LloydMaxCompressor, LloydMaxQuantizer, QjlCompressor, Rotation};

fn main() {
    let dim = 128; // small dim so output is readable
    let vector = random_unit_vec(dim, 42);

    println!("TurboQuant component-by-component demo (dim={dim})\n");
    println!(
        "Input: unit vector, first 8 values: {:?}",
        &vector[..8]
            .iter()
            .map(|v| format!("{v:+.4}"))
            .collect::<Vec<_>>()
    );
    println!(
        "Input norm: {:.6}",
        vector.iter().map(|v| v * v).sum::<f32>().sqrt()
    );

    // ===================================================================
    // Step 1: Random Orthogonal Rotation
    //
    // Spreads energy evenly across all dimensions. After rotation, each
    // coordinate of a unit vector follows ≈ N(0, 1/√d).
    //
    // In KV cache: apply once when a new KV pair enters the cache.
    // The rotation is deterministic from the seed — same seed = same
    // matrix across all layers / heads.
    // ===================================================================
    println!("\n--- Step 1: Rotation ---");

    let rotation = Rotation::new(dim, /*seed=*/ 42);
    let rotated = rotation.rotate(&vector);

    println!(
        "Rotated: first 8 values: {:?}",
        &rotated[..8]
            .iter()
            .map(|v| format!("{v:+.4}"))
            .collect::<Vec<_>>()
    );

    let rot_norm: f32 = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("Rotated norm: {rot_norm:.6} (should ≈ 1.0, rotation preserves norm)");

    let sigma = 1.0 / (dim as f32).sqrt();
    let mean: f32 = rotated.iter().sum::<f32>() / dim as f32;
    let variance: f32 =
        rotated.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
    println!(
        "Expected σ = 1/√{dim} = {sigma:.4}, measured std = {:.4}",
        variance.sqrt()
    );

    // Inverse rotation recovers the original
    let recovered = rotation.inverse_rotate(&rotated);
    let recovery_error: f32 = vector
        .iter()
        .zip(&recovered)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    println!("Inverse rotation error: {recovery_error:.2e} (should ≈ 0)");

    // ===================================================================
    // Step 2: Lloyd-Max Scalar Quantization
    //
    // Quantizes each rotated coordinate independently using MSE-optimal
    // centroids for the Gaussian distribution. No calibration data needed.
    //
    // In KV cache: this is the main compression step. Each key/value
    // dimension gets mapped to one of 2^b centroids. At 3-bit, that's
    // 8 centroids per dimension — enough for quality-neutral attention
    // (per the TurboQuant paper).
    // ===================================================================
    println!("\n--- Step 2: Lloyd-Max Quantization ---");

    for bits in [2, 3, 4] {
        let lm = LloydMaxQuantizer::new(dim, bits);

        println!("\n  {bits}-bit Lloyd-Max ({} centroids):", 1 << bits);
        println!(
            "  Centroids: {:?}",
            lm.centroids()
                .iter()
                .map(|c| format!("{c:+.5}"))
                .collect::<Vec<_>>()
        );

        let compressed = lm.compress(&rotated);
        let reconstructed = lm.decompress(&compressed);

        let mse: f32 = rotated
            .iter()
            .zip(&reconstructed)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / dim as f32;

        let cosine = dot(&rotated, &reconstructed)
            / (norm(&rotated) * norm(&reconstructed));

        println!(
            "  Compressed: {} bytes (raw: {} bytes, {:.1}x compression)",
            compressed.byte_size(),
            dim * 4,
            (dim * 4) as f64 / compressed.byte_size() as f64
        );
        println!("  Recon MSE: {mse:.8}");
        println!("  Cosine similarity: {cosine:.4}");
    }

    // ===================================================================
    // Step 3: QJL Residual Correction
    //
    // After Lloyd-Max quantization, there's a small residual error.
    // QJL projects this residual through a random Gaussian matrix and
    // keeps only the sign bits (1 bit per dimension). This makes the
    // inner product estimator unbiased.
    //
    // In KV cache: store the sign bits alongside the quantized values.
    // During attention, the QJL correction term is added to the
    // approximate dot product to debias it.
    // ===================================================================
    println!("\n--- Step 3: QJL Residual Correction ---");

    let bits = 3;
    let lm = LloydMaxQuantizer::new(dim, bits);
    let compressed = lm.compress(&rotated);
    let reconstructed = lm.decompress(&compressed);

    // Compute residual
    let residual: Vec<f32> = rotated
        .iter()
        .zip(&reconstructed)
        .map(|(a, b)| a - b)
        .collect();
    let residual_norm = norm(&residual);
    println!("Residual norm: {residual_norm:.6}");

    let qjl = QjlCompressor::new(dim, /*seed=*/ 99);
    let sign_bits = qjl.compress(&residual);
    println!(
        "QJL sign bits: {} bytes (1 bit per dimension)",
        sign_bits.byte_len()
    );

    // Show how the correction works during search
    let query = random_unit_vec(dim, 77);
    let rotated_query = rotation.rotate(&query);

    let true_ip: f32 = dot(&vector, &query);

    // Without QJL: just use Lloyd-Max reconstruction
    let approx_ip_no_qjl = dot(&rotated_query, &reconstructed);

    // With QJL: add the correction term
    let qjl_projection = qjl.project(&rotated_query);
    let correction = {
        let scale =
            residual_norm * (std::f32::consts::PI / (2.0 * dim as f32)).sqrt();
        let mut correction_dot = 0.0f32;
        for i in 0..dim {
            let sign = if (sign_bits.0[i / 8] >> (i % 8)) & 1 == 1 {
                1.0f32
            } else {
                -1.0f32
            };
            correction_dot += sign * qjl_projection[i];
        }
        scale * correction_dot
    };
    let approx_ip_with_qjl = approx_ip_no_qjl + correction;

    println!("\nInner product estimation (single pair):");
    println!("  True <query, vector>:     {true_ip:+.6}");
    println!("  Lloyd-Max only:           {approx_ip_no_qjl:+.6} (error: {:.6})", (approx_ip_no_qjl - true_ip).abs());
    println!("  Lloyd-Max + QJL:          {approx_ip_with_qjl:+.6} (error: {:.6})", (approx_ip_with_qjl - true_ip).abs());
    println!("  (QJL may increase error on a single pair — it's unbiased in expectation)");

    // Show bias averaging out over many pairs
    let n_trials = 500;
    let mut bias_no_qjl = 0.0f64;
    let mut bias_with_qjl = 0.0f64;
    let mut mae_no_qjl = 0.0f64;
    let mut mae_with_qjl = 0.0f64;

    let comp_3bit = LloydMaxCompressor::new(dim, 42, 99, 3);
    for trial in 0..n_trials {
        let q = random_unit_vec(dim, 1000 + trial);
        let t = random_unit_vec(dim, 2000 + trial);
        let true_dot: f32 = dot(&q, &t);

        // Without QJL: rotate, Lloyd-Max, decompress, dot
        let rq = rotation.rotate(&q);
        let rt = rotation.rotate(&t);
        let ct = lm.compress(&rt);
        let dt = lm.decompress(&ct);
        let est_no_qjl: f32 = dot(&rq, &dt);

        // With QJL: full pipeline
        let lv = comp_3bit.compress(&t);
        let est_with_qjl = comp_3bit.similarity_raw(&q, &lv);

        bias_no_qjl += (est_no_qjl - true_dot) as f64;
        bias_with_qjl += (est_with_qjl - true_dot) as f64;
        mae_no_qjl += (est_no_qjl - true_dot).abs() as f64;
        mae_with_qjl += (est_with_qjl - true_dot).abs() as f64;
    }

    println!("\n  Over {n_trials} random pairs:");
    println!(
        "  Lloyd-Max only:  bias={:+.6}, MAE={:.6}",
        bias_no_qjl / n_trials as f64,
        mae_no_qjl / n_trials as f64
    );
    println!(
        "  Lloyd-Max + QJL: bias={:+.6}, MAE={:.6}",
        bias_with_qjl / n_trials as f64,
        mae_with_qjl / n_trials as f64
    );
    println!("  (Both should have near-zero bias; QJL reduces MAE on average)");

    // ===================================================================
    // Full pipeline via LloydMaxCompressor (all 3 steps in one call)
    // ===================================================================
    println!("\n--- Full pipeline (LloydMaxCompressor) ---");

    let comp = LloydMaxCompressor::new(dim, 42, 99, 3);
    let full_compressed = comp.compress(&vector);

    println!(
        "Total: {} bytes (Lloyd-Max: {}, QJL signs: {}, norms: 8)",
        full_compressed.byte_size(),
        compressed.byte_size(),
        sign_bits.byte_len()
    );
    println!(
        "Compression: {} → {} bytes ({:.1}x)",
        dim * 4,
        full_compressed.byte_size(),
        (dim * 4) as f64 / full_compressed.byte_size() as f64,
    );

    let full_sim = comp.similarity_raw(&query, &full_compressed);
    println!("\nFull pipeline similarity: {full_sim:+.6}");
    println!("True inner product:      {true_ip:+.6}");
    println!("Error:                   {:.6}", (full_sim - true_ip).abs());

    // ===================================================================
    // KV Cache use case summary
    // ===================================================================
    println!("\n--- KV Cache Application ---");
    println!("For a transformer with head_dim={dim}:");
    println!("  Raw KV pair:     {} bytes per head", dim * 4 * 2); // key + value
    println!(
        "  3-bit TurboQuant: {} bytes per head ({:.1}x savings)",
        full_compressed.byte_size() * 2,
        (dim * 4 * 2) as f64 / (full_compressed.byte_size() * 2) as f64,
    );
    let seq_len = 4096;
    let n_heads = 32;
    println!(
        "  At seq_len={seq_len}, {n_heads} heads:");
    println!(
        "    Raw:        {:.1} MB",
        (seq_len * n_heads * dim * 4 * 2) as f64 / 1e6
    );
    println!(
        "    Compressed: {:.1} MB",
        (seq_len * n_heads * full_compressed.byte_size() * 2) as f64 / 1e6
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn random_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut v = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut h = DefaultHasher::new();
        seed.hash(&mut h);
        i.hash(&mut h);
        let bits = h.finish();
        v.push(((bits as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32);
    }
    let n = norm(&v);
    if n > 0.0 {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
