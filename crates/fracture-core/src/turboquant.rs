use crate::Result;
use std::collections::HashMap;
use std::f64::consts::PI;

/// TurboQuant configuration, passed through from CLI to cache manager.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    /// Key quantization bits for normal layers (2 or 4, default: 4).
    pub key_bits: u8,
    /// Value quantization bits for normal layers (2 or 4, default: 2).
    pub value_bits: u8,
    /// Bit width for protected (first/last N) layers (default: 8).
    pub protected_bits: u8,
    /// Number of first/last layers to protect with higher bits (default: 0).
    pub protected_layers: usize,
    /// Recent tokens kept in FP16 per sequence (default: 0).
    pub residual_tokens: usize,
    /// Base seed for rotation matrix generation (default: 42).
    pub seed: u64,
}

impl Default for TurboQuantConfig {
    fn default() -> Self {
        Self {
            key_bits: 4,
            value_bits: 2,
            protected_bits: 8,
            protected_layers: 0,
            residual_tokens: 0,
            seed: 42,
        }
    }
}

impl TurboQuantConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        const VALID_BITS: &[u8] = &[2, 4, 8];
        if !VALID_BITS.contains(&self.key_bits) {
            return Err(crate::FractureError::ModelConfig(format!(
                "TurboQuant key_bits must be 2, 4, or 8, got {}",
                self.key_bits
            )));
        }
        if !VALID_BITS.contains(&self.value_bits) {
            return Err(crate::FractureError::ModelConfig(format!(
                "TurboQuant value_bits must be 2, 4, or 8, got {}",
                self.value_bits
            )));
        }
        if !VALID_BITS.contains(&self.protected_bits) {
            return Err(crate::FractureError::ModelConfig(format!(
                "TurboQuant protected_bits must be 2, 4, or 8, got {}",
                self.protected_bits
            )));
        }
        Ok(())
    }

    /// Returns the effective key bit-width for a given layer.
    pub fn key_bits_for_layer(&self, layer: usize, num_layers: usize) -> u8 {
        if self.is_protected(layer, num_layers) {
            self.protected_bits
        } else {
            self.key_bits
        }
    }

    /// Returns the effective value bit-width for a given layer.
    pub fn value_bits_for_layer(&self, layer: usize, num_layers: usize) -> u8 {
        if self.is_protected(layer, num_layers) {
            self.protected_bits
        } else {
            self.value_bits
        }
    }

    /// Returns true if a layer is protected (first/last N layers).
    pub fn is_protected(&self, layer: usize, num_layers: usize) -> bool {
        self.protected_layers > 0
            && (layer < self.protected_layers || layer >= num_layers.saturating_sub(self.protected_layers))
    }

    /// Returns the set of distinct bit-widths in use.
    pub fn distinct_bit_widths(&self) -> Vec<u8> {
        let mut bits = vec![self.key_bits, self.value_bits];
        if self.protected_layers > 0 {
            bits.push(self.protected_bits);
        }
        bits.sort_unstable();
        bits.dedup();
        bits
    }

    /// Computes the packed byte dimension for a given head_dim and bit-width.
    ///
    /// For `bits=4, head_dim=128`: each head packs into 64 bytes.
    /// For `bits=2, head_dim=128`: each head packs into 32 bytes.
    pub fn packed_dim_per_head(head_dim: usize, bits: u8) -> usize {
        (head_dim * bits as usize).div_ceil(8)
    }

    /// Computes bytes per quantized block for one layer (K+V, one layer).
    pub fn bytes_per_block_layer(
        num_kv_heads: usize,
        head_dim: usize,
        key_bits: u8,
        value_bits: u8,
    ) -> usize {
        let block_size = 16;
        let k_packed_bytes = block_size * num_kv_heads * Self::packed_dim_per_head(head_dim, key_bits);
        let v_packed_bytes = block_size * num_kv_heads * Self::packed_dim_per_head(head_dim, value_bits);
        let k_norm_bytes = block_size * num_kv_heads * 2; // FP16
        let v_norm_bytes = block_size * num_kv_heads * 2;
        k_packed_bytes + v_packed_bytes + k_norm_bytes + v_norm_bytes
    }

    /// Computes total bytes per physical block across all layers, accounting
    /// for per-layer bit widths (protected vs normal).
    pub fn bytes_per_block_total(
        &self,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> usize {
        (0..num_layers)
            .map(|layer| {
                let kb = self.key_bits_for_layer(layer, num_layers);
                let vb = self.value_bits_for_layer(layer, num_layers);
                Self::bytes_per_block_layer(num_kv_heads, head_dim, kb, vb)
            })
            .sum()
    }

    /// Computes how many blocks fit in a given memory budget.
    pub fn compute_num_blocks(
        &self,
        gpu_available: usize,
        scratch_reserve: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> usize {
        let cache_budget = gpu_available.saturating_sub(scratch_reserve);
        let bytes_per_block = self.bytes_per_block_total(num_layers, num_kv_heads, head_dim);
        if bytes_per_block == 0 {
            return 0;
        }
        cache_budget / bytes_per_block
    }
}

// ── Lloyd-Max Codebook ──────────────────────────────────────────────

/// Precomputed Lloyd-Max codebook: centroids and decision boundaries for
/// quantizing N(0, 1/d) distributed coordinates.
#[derive(Debug, Clone)]
pub struct LloydMaxCodebook {
    pub centroids: Vec<f32>,
    pub boundaries: Vec<f32>,
    pub bits: u8,
    pub dim: usize,
}

/// Gaussian PDF for N(0, sigma^2).
fn gaussian_pdf(x: f64, sigma2: f64) -> f64 {
    (1.0 / (2.0 * PI * sigma2).sqrt()) * (-x * x / (2.0 * sigma2)).exp()
}

/// Numerical integration of `f` over `[a, b]` using composite Simpson's rule.
fn integrate_simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let n = if n % 2 == 1 { n + 1 } else { n };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    sum * h / 3.0
}

impl LloydMaxCodebook {
    /// Solve the Lloyd-Max optimal quantizer for N(0, 1/d).
    ///
    /// Uses the Gaussian approximation which is accurate for d >= 64.
    pub fn compute(dim: usize, bits: u8) -> Self {
        let n_levels = 1usize << bits;
        let sigma2 = 1.0 / dim as f64;
        let sigma = sigma2.sqrt();

        let lo = -3.5 * sigma;
        let hi = 3.5 * sigma;

        // Initialize centroids uniformly in [lo, hi]
        let mut centroids: Vec<f64> = (0..n_levels)
            .map(|i| lo + (hi - lo) * (i as f64 + 0.5) / n_levels as f64)
            .collect();

        let max_iter = 200;
        let tol = 1e-10;
        let n_quad = 500;

        for _ in 0..max_iter {
            // Step 1: boundaries = midpoints between adjacent centroids
            let boundaries: Vec<f64> = (0..n_levels - 1)
                .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
                .collect();

            // Step 2: update centroids as conditional expectations
            let edges_lo = lo * 3.0;
            let edges_hi = hi * 3.0;
            let mut new_centroids = Vec::with_capacity(n_levels);
            let mut max_shift = 0.0f64;

            for i in 0..n_levels {
                let a = if i == 0 { edges_lo } else { boundaries[i - 1] };
                let b = if i == n_levels - 1 { edges_hi } else { boundaries[i] };

                let numerator = integrate_simpson(|x| x * gaussian_pdf(x, sigma2), a, b, n_quad);
                let denominator = integrate_simpson(|x| gaussian_pdf(x, sigma2), a, b, n_quad);

                let new_c = if denominator.abs() > 1e-15 {
                    numerator / denominator
                } else {
                    centroids[i]
                };

                max_shift = max_shift.max((new_c - centroids[i]).abs());
                new_centroids.push(new_c);
            }

            centroids = new_centroids;

            if max_shift < tol {
                break;
            }
        }

        // Final boundaries
        let boundaries: Vec<f64> = (0..n_levels - 1)
            .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
            .collect();

        Self {
            centroids: centroids.iter().map(|&c| c as f32).collect(),
            boundaries: boundaries.iter().map(|&b| b as f32).collect(),
            bits,
            dim,
        }
    }

    /// Number of quantization levels (2^bits).
    pub fn n_levels(&self) -> usize {
        1 << self.bits
    }
}

// ── Precomputed Codebooks for d=128 ──────────────────────────────────

/// Precomputed Lloyd-Max centroids for head_dim=128 (Llama family).
/// These are optimal reconstruction levels for N(0, 1/128).
pub fn precomputed_codebook(dim: usize, bits: u8) -> Option<LloydMaxCodebook> {
    if dim != 128 {
        return None;
    }

    // Compute lazily rather than hardcode — the computation takes < 1ms
    // and avoids maintaining large const arrays that could drift from the
    // solver output. We verify correctness via unit tests.
    match bits {
        2 | 4 | 8 => Some(LloydMaxCodebook::compute(dim, bits)),
        _ => None,
    }
}

/// Get or compute the codebook for a given (dim, bits) pair.
/// Returns precomputed tables for d=128, computes on-the-fly for other dims.
pub fn get_codebook(dim: usize, bits: u8) -> LloydMaxCodebook {
    precomputed_codebook(dim, bits).unwrap_or_else(|| {
        tracing::warn!(
            "no precomputed TurboQuant codebook for head_dim={dim}, bits={bits}; \
             computing at startup"
        );
        LloydMaxCodebook::compute(dim, bits)
    })
}

/// Compute all codebook tables needed for a given config.
pub fn compute_codebook_tables(
    config: &TurboQuantConfig,
    head_dim: usize,
) -> HashMap<u8, LloydMaxCodebook> {
    config
        .distinct_bit_widths()
        .into_iter()
        .map(|bits| (bits, get_codebook(head_dim, bits)))
        .collect()
}

// ── Rotation Matrix Generation ──────────────────────────────────────

/// Generates a deterministic random orthogonal matrix (Haar-distributed)
/// via QR decomposition of a seeded Gaussian matrix.
///
/// Returns the matrix as a flat row-major `[d, d]` array of f32.
pub fn generate_rotation_matrix(d: usize, seed: u64) -> Vec<f32> {
    // Xoshiro256** PRNG seeded deterministically
    let mut rng = Xoshiro256::new(seed);

    // Generate d×d Gaussian matrix using Box-Muller transform
    let n = d * d;
    let mut g = Vec::with_capacity(n);
    let pairs_needed = n.div_ceil(2);
    for _ in 0..pairs_needed {
        let (z0, z1) = box_muller(&mut rng);
        g.push(z0);
        if g.len() < n {
            g.push(z1);
        }
    }

    // QR decomposition via modified Gram-Schmidt
    let q = qr_orthogonalize(&g, d);

    // Fix sign ambiguity: ensure diagonal elements are positive
    // (equivalent to Q * sign(diag(R)) from full QR)
    let mut result = q;
    for col in 0..d {
        let diag_val = result[col * d + col];
        if diag_val < 0.0 {
            for row in 0..d {
                result[row * d + col] = -result[row * d + col];
            }
        }
    }

    result
}

/// Compute K and V rotation matrix seeds for a given layer.
pub fn rotation_seeds(base_seed: u64, layer: usize) -> (u64, u64) {
    let k_seed = base_seed + layer as u64 * 1000;
    let v_seed = base_seed + layer as u64 * 1000 + 500;
    (k_seed, v_seed)
}

// ── Xoshiro256** PRNG (deterministic, no external deps) ──────────────

struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    fn new(seed: u64) -> Self {
        // SplitMix64 to initialize state from a single seed
        let mut sm = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            sm = sm.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            *slot = z;
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Returns a f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Box-Muller transform: two uniform samples → two Gaussian samples.
fn box_muller(rng: &mut Xoshiro256) -> (f32, f32) {
    let u1 = rng.next_f64().max(1e-15); // avoid log(0)
    let u2 = rng.next_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * PI * u2;
    ((r * theta.cos()) as f32, (r * theta.sin()) as f32)
}

/// Modified Gram-Schmidt QR orthogonalization.
/// Input: column-major `g[row * d + col]` but stored row-major, d×d.
/// Output: orthogonal matrix Q as flat row-major `[d, d]`.
fn qr_orthogonalize(g: &[f32], d: usize) -> Vec<f32> {
    // Work in column-major for Gram-Schmidt, then output row-major.
    // Extract columns from row-major input.
    let mut cols: Vec<Vec<f64>> = (0..d)
        .map(|col| (0..d).map(|row| g[row * d + col] as f64).collect())
        .collect();

    for i in 0..d {
        // Normalize column i
        let norm: f64 = cols[i].iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for val in cols[i].iter_mut().take(d) {
                *val /= norm;
            }
        }

        // Subtract projection from remaining columns
        for j in (i + 1)..d {
            let dot: f64 = (0..d).map(|row| cols[i][row] * cols[j][row]).sum();
            let (col_i, col_j) = if i < j {
                let (left, right) = cols.split_at_mut(j);
                (&left[i], &mut right[0])
            } else {
                unreachable!()
            };
            for (cj, ci) in col_j.iter_mut().zip(col_i.iter()) {
                *cj -= dot * ci;
            }
        }
    }

    // Convert back to row-major
    let mut result = vec![0.0f32; d * d];
    for row in 0..d {
        for col in 0..d {
            result[row * d + col] = cols[col][row] as f32;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let cfg = TurboQuantConfig::default();
        assert_eq!(cfg.key_bits, 4);
        assert_eq!(cfg.value_bits, 2);
        assert_eq!(cfg.protected_bits, 8);
        assert_eq!(cfg.protected_layers, 0);
        assert_eq!(cfg.residual_tokens, 0);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn test_config_validate() {
        let mut cfg = TurboQuantConfig::default();
        assert!(cfg.validate().is_ok());

        cfg.key_bits = 3;
        assert!(cfg.validate().is_err());

        cfg.key_bits = 4;
        cfg.value_bits = 5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_layer_bits() {
        let cfg = TurboQuantConfig {
            key_bits: 4,
            value_bits: 2,
            protected_bits: 8,
            protected_layers: 2,
            ..Default::default()
        };

        let num_layers = 32;
        // First 2 layers protected
        assert_eq!(cfg.key_bits_for_layer(0, num_layers), 8);
        assert_eq!(cfg.key_bits_for_layer(1, num_layers), 8);
        // Middle layers normal
        assert_eq!(cfg.key_bits_for_layer(2, num_layers), 4);
        assert_eq!(cfg.key_bits_for_layer(15, num_layers), 4);
        assert_eq!(cfg.key_bits_for_layer(29, num_layers), 4);
        // Last 2 layers protected
        assert_eq!(cfg.key_bits_for_layer(30, num_layers), 8);
        assert_eq!(cfg.key_bits_for_layer(31, num_layers), 8);

        // Values
        assert_eq!(cfg.value_bits_for_layer(0, num_layers), 8);
        assert_eq!(cfg.value_bits_for_layer(15, num_layers), 2);
        assert_eq!(cfg.value_bits_for_layer(31, num_layers), 8);
    }

    #[test]
    fn test_config_no_protection() {
        let cfg = TurboQuantConfig::default();
        for layer in 0..32 {
            assert_eq!(cfg.key_bits_for_layer(layer, 32), 4);
            assert_eq!(cfg.value_bits_for_layer(layer, 32), 2);
        }
    }

    #[test]
    fn test_packed_dim_per_head() {
        assert_eq!(TurboQuantConfig::packed_dim_per_head(128, 4), 64);
        assert_eq!(TurboQuantConfig::packed_dim_per_head(128, 2), 32);
        assert_eq!(TurboQuantConfig::packed_dim_per_head(128, 8), 128);
        // Non-power-of-2 head_dim
        assert_eq!(TurboQuantConfig::packed_dim_per_head(96, 4), 48);
        assert_eq!(TurboQuantConfig::packed_dim_per_head(96, 2), 24);
    }

    #[test]
    fn test_bytes_per_block_layer() {
        // K4/V2, 8 heads, d=128
        let bytes = TurboQuantConfig::bytes_per_block_layer(8, 128, 4, 2);
        // K: 16 * 8 * (64 + 2) = 8448 packed + norms, but let's compute:
        // k_packed = 16 * 8 * 64 = 8192
        // v_packed = 16 * 8 * 32 = 4096
        // k_norms = 16 * 8 * 2 = 256
        // v_norms = 16 * 8 * 2 = 256
        // Total = 8192 + 4096 + 256 + 256 = 12800
        assert_eq!(bytes, 12800);
    }

    #[test]
    fn test_bytes_per_block_total_uniform() {
        let cfg = TurboQuantConfig::default(); // K4/V2, no protection
        let total = cfg.bytes_per_block_total(32, 8, 128);
        assert_eq!(total, 12800 * 32);
    }

    #[test]
    fn test_bytes_per_block_total_with_protection() {
        let cfg = TurboQuantConfig {
            key_bits: 4,
            value_bits: 2,
            protected_bits: 8,
            protected_layers: 4,
            ..Default::default()
        };
        let total = cfg.bytes_per_block_total(32, 8, 128);

        // 8 protected layers (first 4 + last 4): K8/V8
        let protected = TurboQuantConfig::bytes_per_block_layer(8, 128, 8, 8);
        // 24 normal layers: K4/V2
        let normal = TurboQuantConfig::bytes_per_block_layer(8, 128, 4, 2);
        assert_eq!(total, 8 * protected + 24 * normal);
    }

    #[test]
    fn test_distinct_bit_widths() {
        let cfg = TurboQuantConfig::default();
        let bits = cfg.distinct_bit_widths();
        assert_eq!(bits, vec![2, 4]); // K4/V2, no protection

        let cfg = TurboQuantConfig {
            protected_layers: 2,
            ..Default::default()
        };
        let bits = cfg.distinct_bit_widths();
        assert_eq!(bits, vec![2, 4, 8]);
    }

    // ── Lloyd-Max tests ──────────────────────────────────────────────

    #[test]
    fn test_lloyd_max_2bit_symmetry() {
        let cb = LloydMaxCodebook::compute(128, 2);
        assert_eq!(cb.centroids.len(), 4);
        assert_eq!(cb.boundaries.len(), 3);

        // Codebook should be symmetric around 0
        for i in 0..2 {
            let pos = cb.centroids[3 - i];
            let neg = cb.centroids[i];
            assert!(
                (pos + neg).abs() < 1e-6,
                "centroids should be symmetric: {} vs {}",
                neg,
                pos
            );
        }

        // Middle boundary should be ~0
        assert!(
            cb.boundaries[1].abs() < 1e-6,
            "middle boundary should be ~0, got {}",
            cb.boundaries[1]
        );
    }

    #[test]
    fn test_lloyd_max_4bit_properties() {
        let cb = LloydMaxCodebook::compute(128, 4);
        assert_eq!(cb.centroids.len(), 16);
        assert_eq!(cb.boundaries.len(), 15);
        assert_eq!(cb.n_levels(), 16);

        // Centroids should be sorted
        for i in 1..cb.centroids.len() {
            assert!(
                cb.centroids[i] > cb.centroids[i - 1],
                "centroids should be sorted: [{}]={} <= [{}]={}",
                i - 1,
                cb.centroids[i - 1],
                i,
                cb.centroids[i]
            );
        }

        // Boundaries should be between adjacent centroids
        for i in 0..cb.boundaries.len() {
            assert!(
                cb.boundaries[i] > cb.centroids[i] && cb.boundaries[i] < cb.centroids[i + 1],
                "boundary[{}]={} should be between centroids {} and {}",
                i,
                cb.boundaries[i],
                cb.centroids[i],
                cb.centroids[i + 1]
            );
        }
    }

    #[test]
    fn test_lloyd_max_8bit_count() {
        let cb = LloydMaxCodebook::compute(128, 8);
        assert_eq!(cb.centroids.len(), 256);
        assert_eq!(cb.boundaries.len(), 255);
    }

    #[test]
    fn test_lloyd_max_centroid_scale() {
        // Centroids should scale with 1/sqrt(d)
        let cb_128 = LloydMaxCodebook::compute(128, 4);
        let cb_64 = LloydMaxCodebook::compute(64, 4);

        let max_128 = cb_128.centroids.last().unwrap();
        let max_64 = cb_64.centroids.last().unwrap();

        // max_64 / max_128 should be approximately sqrt(128/64) = sqrt(2) ≈ 1.414
        let ratio = max_64 / max_128;
        assert!(
            (ratio - std::f32::consts::SQRT_2).abs() < 0.1,
            "centroid scale ratio should be ~sqrt(2), got {}",
            ratio
        );
    }

    #[test]
    fn test_lloyd_max_deterministic() {
        let cb1 = LloydMaxCodebook::compute(128, 4);
        let cb2 = LloydMaxCodebook::compute(128, 4);
        assert_eq!(cb1.centroids, cb2.centroids);
        assert_eq!(cb1.boundaries, cb2.boundaries);
    }

    // ── Rotation matrix tests ────────────────────────────────────────

    #[test]
    fn test_rotation_matrix_orthogonality() {
        let d = 16; // small for test speed
        let q = generate_rotation_matrix(d, 42);

        // Q^T @ Q should be identity
        for i in 0..d {
            for j in 0..d {
                let dot: f32 = (0..d).map(|k| q[k * d + i] * q[k * d + j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "Q^T @ Q [{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_rotation_matrix_deterministic() {
        let q1 = generate_rotation_matrix(32, 42);
        let q2 = generate_rotation_matrix(32, 42);
        assert_eq!(q1, q2);
    }

    #[test]
    fn test_rotation_matrix_different_seeds() {
        let q1 = generate_rotation_matrix(16, 42);
        let q2 = generate_rotation_matrix(16, 43);
        // Different seeds should produce different matrices
        assert_ne!(q1, q2);
    }

    #[test]
    fn test_rotation_matrix_128d() {
        let d = 128;
        let q = generate_rotation_matrix(d, 42);
        assert_eq!(q.len(), d * d);

        // Spot-check orthogonality on a few row pairs
        for (i, j) in [(0, 1), (0, 63), (63, 127)] {
            let dot: f32 = (0..d).map(|k| q[i * d + k] * q[j * d + k]).sum();
            assert!(
                dot.abs() < 1e-3,
                "rows {i} and {j} should be orthogonal, dot = {dot}"
            );
        }

        // Check a diagonal element (row self-dot = 1)
        let self_dot: f32 = (0..d).map(|k| q[0 * d + k] * q[0 * d + k]).sum();
        assert!(
            (self_dot - 1.0).abs() < 1e-3,
            "row 0 self-dot should be 1.0, got {self_dot}"
        );
    }

    #[test]
    fn test_rotation_seeds() {
        let (k, v) = rotation_seeds(42, 0);
        assert_eq!(k, 42);
        assert_eq!(v, 542);

        let (k, v) = rotation_seeds(42, 5);
        assert_eq!(k, 5042);
        assert_eq!(v, 5542);

        // K and V seeds should always differ
        for layer in 0..32 {
            let (k, v) = rotation_seeds(42, layer);
            assert_ne!(k, v);
        }
    }

    #[test]
    fn test_codebook_tables() {
        let cfg = TurboQuantConfig::default();
        let tables = compute_codebook_tables(&cfg, 128);
        assert!(tables.contains_key(&4));
        assert!(tables.contains_key(&2));
        assert!(!tables.contains_key(&8)); // no protection

        let cfg = TurboQuantConfig {
            protected_layers: 2,
            ..Default::default()
        };
        let tables = compute_codebook_tables(&cfg, 128);
        assert!(tables.contains_key(&8));
    }

    // ── Integration test: compress round-trip in pure Rust ───────────

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let d = 128;
        let cb = LloydMaxCodebook::compute(d, 4);
        let pi = generate_rotation_matrix(d, 42);

        // Create a test vector
        let mut rng = Xoshiro256::new(123);
        let x: Vec<f32> = (0..d).map(|_| {
            let (z, _) = box_muller(&mut rng);
            z
        }).collect();

        // Normalize
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let x_norm: Vec<f32> = x.iter().map(|v| v / (norm + 1e-8)).collect();

        // Rotate: y = Pi @ x_norm
        let mut y = vec![0.0f32; d];
        for i in 0..d {
            y[i] = (0..d).map(|j| pi[i * d + j] * x_norm[j]).sum();
        }

        // Quantize: find nearest centroid per coordinate
        let indices: Vec<usize> = y.iter().map(|&yi| {
            cb.centroids.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (yi - *a).abs().partial_cmp(&(yi - *b).abs()).unwrap()
                })
                .unwrap().0
        }).collect();

        // Dequantize: lookup centroids
        let y_hat: Vec<f32> = indices.iter().map(|&i| cb.centroids[i]).collect();

        // Unrotate: x_hat = Pi^T @ y_hat
        let mut x_hat = vec![0.0f32; d];
        for i in 0..d {
            x_hat[i] = (0..d).map(|j| pi[j * d + i] * y_hat[j]).sum();
        }

        // Rescale
        let x_recon: Vec<f32> = x_hat.iter().map(|v| v * norm).collect();

        // Compute MSE
        let mse: f32 = x.iter().zip(x_recon.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>() / d as f32;

        // Cosine similarity
        let dot: f32 = x.iter().zip(x_recon.iter()).map(|(a, b)| a * b).sum();
        let norm_orig: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_recon: f32 = x_recon.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cosine_sim = dot / (norm_orig * norm_recon + 1e-8);

        // 4-bit should give high fidelity
        assert!(
            cosine_sim > 0.99,
            "4-bit round-trip cosine similarity should be > 0.99, got {cosine_sim}"
        );
        assert!(
            mse < norm * norm * 0.05,
            "4-bit MSE should be small relative to vector magnitude, got {mse}"
        );
    }
}
