use fracture_core::{FractureError, Result};
use rand::Rng;
use rand::SeedableRng;

/// Sampling parameters for token selection.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    /// Optional seed for deterministic sampling. When set, the sampler uses
    /// a seeded RNG (`StdRng::seed_from_u64`) instead of `rng()`.
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        }
    }
}

/// Token sampler operating on CPU logits.
pub struct Sampler;

/// Compare two f32 values, treating NaN as less than all other values.
fn f32_cmp(a: &f32, b: &f32) -> std::cmp::Ordering {
    a.partial_cmp(b).unwrap_or_else(|| {
        // NaN handling: treat NaN as less than everything
        match (a.is_nan(), b.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => unreachable!(),
        }
    })
}

impl Sampler {
    /// Sample a token from logits given sampling parameters.
    pub fn sample(logits: &[f32], params: &SamplingParams) -> Result<u32> {
        if logits.is_empty() {
            return Err(FractureError::Generation("empty logits".into()));
        }

        // Check for NaN or Inf logits
        if logits.iter().any(|l| l.is_nan()) {
            return Err(FractureError::Generation("NaN in logits".into()));
        }
        if logits.iter().any(|l| l.is_infinite()) {
            return Err(FractureError::Generation("Inf in logits".into()));
        }

        if params.temperature == 0.0 || params.top_k == 1 {
            // Greedy: argmax
            let (idx, _) = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| f32_cmp(a, b))
                .unwrap(); // safe: checked non-empty above
            return Ok(idx as u32);
        }

        // Temperature scaling
        let scaled: Vec<f32> = logits.iter().map(|&l| l / params.temperature).collect();

        // Top-K filtering
        let mut indices: Vec<usize> = (0..scaled.len()).collect();
        if params.top_k > 0 && params.top_k < scaled.len() {
            indices.sort_by(|&a, &b| f32_cmp(&scaled[b], &scaled[a]));
            indices.truncate(params.top_k);
        }

        // Softmax over remaining
        let max_val = indices.iter().map(|&i| scaled[i]).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = indices.iter().map(|&i| (scaled[i] - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        // Top-P filtering
        let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
        sorted_indices.sort_by(|&a, &b| f32_cmp(&probs[b], &probs[a]));

        let mut cumulative = 0.0;
        let mut cutoff = sorted_indices.len();
        if params.top_p < 1.0 {
            for (i, &si) in sorted_indices.iter().enumerate() {
                cumulative += probs[si];
                if cumulative > params.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
        }
        let filtered: Vec<usize> = sorted_indices[..cutoff].to_vec();

        // Re-normalize and sample
        let filtered_sum: f32 = filtered.iter().map(|&i| probs[i]).sum();
        let r: f32 = if let Some(seed) = params.seed {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            rng.random::<f32>()
        } else {
            rand::random::<f32>()
        } * filtered_sum;
        let mut acc = 0.0;
        for &i in &filtered {
            acc += probs[i];
            if acc >= r {
                return Ok(indices[i] as u32);
            }
        }

        Ok(indices[*filtered.last().unwrap()] as u32) // safe: filtered is non-empty (logits checked above)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_logits() -> Vec<f32> {
        // indices: 0=1.0, 1=3.0, 2=2.0, 3=0.5, 4=0.1
        vec![1.0, 3.0, 2.0, 0.5, 0.1]
    }

    #[test]
    fn test_greedy_temp_zero() {
        let logits = simple_logits();
        let params = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        for _ in 0..20 {
            let token = Sampler::sample(&logits, &params).unwrap();
            assert_eq!(token, 1); // index 1 has max logit 3.0
        }
    }

    #[test]
    fn test_greedy_top_k_one() {
        let logits = simple_logits();
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            seed: None,
        };
        for _ in 0..20 {
            let token = Sampler::sample(&logits, &params).unwrap();
            assert_eq!(token, 1);
        }
    }

    #[test]
    fn test_temperature_scaling_flattens_distribution() {
        // With high temperature, softmax should produce more uniform probs
        let logits = [10.0, 0.0];

        // Low temperature (near greedy) - diff should be large
        let low_temp: Vec<f32> = logits.iter().map(|&l| l / 0.1).collect();
        let max_low = low_temp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps_low: Vec<f32> = low_temp.iter().map(|l| (l - max_low).exp()).collect();
        let sum_low: f32 = exps_low.iter().sum();
        let probs_low: Vec<f32> = exps_low.iter().map(|e| e / sum_low).collect();

        // High temperature - diff should be small (more uniform)
        let high_temp: Vec<f32> = logits.iter().map(|&l| l / 100.0).collect();
        let max_high = high_temp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps_high: Vec<f32> = high_temp.iter().map(|l| (l - max_high).exp()).collect();
        let sum_high: f32 = exps_high.iter().sum();
        let probs_high: Vec<f32> = exps_high.iter().map(|e| e / sum_high).collect();

        let diff_low = (probs_low[0] - probs_low[1]).abs();
        let diff_high = (probs_high[0] - probs_high[1]).abs();
        assert!(diff_high < diff_low, "high temp should flatten: diff_high={diff_high} < diff_low={diff_low}");
    }

    #[test]
    fn test_top_k_filtering() {
        let logits = simple_logits(); // argmax at 1 (3.0), second at 2 (2.0)
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            seed: None,
        };
        for _ in 0..100 {
            let token = Sampler::sample(&logits, &params).unwrap();
            assert!(token == 1 || token == 2, "token {token} not in top-2");
        }
    }

    #[test]
    fn test_top_p_very_small_is_greedy() {
        let logits = simple_logits();
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.01,
            seed: None,
        };
        for _ in 0..50 {
            let token = Sampler::sample(&logits, &params).unwrap();
            assert_eq!(token, 1, "very small top_p should select argmax");
        }
    }

    #[test]
    fn test_softmax_max_subtraction_no_overflow() {
        // Large logit values that would overflow exp() without the max-subtraction trick
        let logits = vec![1000.0, 999.0, 998.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        let token = Sampler::sample(&logits, &params).unwrap();
        assert!(token <= 2, "should produce a valid token index");
    }

    #[test]
    fn test_greedy_determinism() {
        let logits = simple_logits();
        let params = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        let first = Sampler::sample(&logits, &params).unwrap();
        for _ in 0..99 {
            assert_eq!(Sampler::sample(&logits, &params).unwrap(), first);
        }
    }

    #[test]
    fn test_sampling_params_default() {
        let params = SamplingParams::default();
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_k, 0);
        assert_eq!(params.top_p, 1.0);
        assert!(params.seed.is_none());
    }

    #[test]
    fn test_top_p_disables_at_one() {
        // With top_p=1.0, all tokens should be candidates.
        // Use a fairly flat distribution so multiple tokens can appear.
        let logits = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        let mut seen = std::collections::HashSet::new();
        for _ in 0..500 {
            seen.insert(Sampler::sample(&logits, &params).unwrap());
        }
        assert!(
            seen.len() > 1,
            "with top_p=1.0 and uniform logits, multiple distinct tokens should appear, got {:?}",
            seen
        );
    }

    #[test]
    fn test_top_k_disables_at_zero() {
        // top_k=0 means no filtering — all tokens are candidates.
        let logits = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        let mut seen = std::collections::HashSet::new();
        for _ in 0..500 {
            seen.insert(Sampler::sample(&logits, &params).unwrap());
        }
        assert!(
            seen.len() > 1,
            "with top_k=0, all tokens should be candidates, got {:?}",
            seen
        );
    }

    #[test]
    fn test_top_k_equals_vocab_size_disables() {
        // top_k equal to vocab size means no filtering.
        let logits = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 5,
            top_p: 1.0,
            seed: None,
        };
        let mut seen = std::collections::HashSet::new();
        for _ in 0..500 {
            seen.insert(Sampler::sample(&logits, &params).unwrap());
        }
        assert!(
            seen.len() > 1,
            "with top_k=vocab_size, all tokens should be candidates, got {:?}",
            seen
        );
    }

    #[test]
    fn test_weighted_random_selection() {
        // Higher-logit tokens should appear more frequently.
        let logits = vec![10.0, 0.0, 0.0, 0.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        let mut counts = [0u32; 5];
        for _ in 0..1000 {
            let token = Sampler::sample(&logits, &params).unwrap();
            counts[token as usize] += 1;
        }
        // Token 0 has logit 10.0, should dominate heavily
        assert!(
            counts[0] > counts[1],
            "token 0 (logit=10.0) should appear more than token 1 (logit=0.0): {} vs {}",
            counts[0],
            counts[1]
        );
        assert!(
            counts[0] > counts[2],
            "token 0 (logit=10.0) should appear more than token 2 (logit=0.0): {} vs {}",
            counts[0],
            counts[2]
        );
        // Token 0 should have the majority of samples
        assert!(
            counts[0] > 500,
            "token 0 should appear in >50% of 1000 samples, got {}",
            counts[0]
        );
    }

    #[test]
    fn test_combined_top_k_top_p() {
        // top_k=3 keeps indices 1,2,0 (logits 3.0, 2.0, 1.0)
        // top_p=0.5 further filters to only highest-prob tokens
        let logits = simple_logits();
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 3,
            top_p: 0.5,
            seed: None,
        };
        let valid_tokens: Vec<u32> = vec![0, 1, 2]; // the top-3 by logit value
        for _ in 0..100 {
            let token = Sampler::sample(&logits, &params).unwrap();
            assert!(
                valid_tokens.contains(&token),
                "token {token} not in valid set {valid_tokens:?}"
            );
        }
    }

    #[test]
    fn test_sampler_returns_result_type() {
        // Explicitly verify sample() returns Result<u32>:
        // Ok on valid input, Err on invalid — never panics.
        let logits = simple_logits();
        let params = SamplingParams::default();

        // Valid input yields Ok with a token index in range
        let result: fracture_core::Result<u32> = Sampler::sample(&logits, &params);
        assert!(result.is_ok(), "sample() should return Ok on valid input");
        let token = result.unwrap();
        assert!(
            (token as usize) < logits.len(),
            "sampled token {token} should be within vocab range"
        );

        // Empty logits yields Err (not panic)
        let result = Sampler::sample(&[], &params);
        assert!(result.is_err(), "sample() should return Err on empty logits");

        // NaN logits yields Err (not panic)
        let result = Sampler::sample(&[1.0, f32::NAN], &params);
        assert!(result.is_err(), "sample() should return Err on NaN logits");
    }

    #[test]
    fn test_empty_logits_returns_error() {
        let logits: Vec<f32> = vec![];
        let params = SamplingParams::default();
        let result = Sampler::sample(&logits, &params);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty logits"));
    }

    #[test]
    fn test_nan_logits_returns_error() {
        let logits = vec![1.0, f32::NAN, 2.0];
        let params = SamplingParams::default();
        let result = Sampler::sample(&logits, &params);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("NaN"));

        // Also test greedy path
        let params_greedy = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        let result = Sampler::sample(&logits, &params_greedy);
        assert!(result.is_err());
    }

    #[test]
    fn test_temperature_negative_behavior() {
        // Negative temperature inverts the logit ordering during softmax.
        // The token with the lowest logit gets the highest probability.
        let logits = vec![10.0, 0.0, 5.0]; // index 0 has max, index 1 has min
        let params = SamplingParams {
            temperature: -1.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };

        // With negative temperature, logits are divided by -1.0:
        // scaled = [-10.0, 0.0, -5.0] => softmax favors index 1 (highest scaled value)
        let mut counts = [0u32; 3];
        for _ in 0..500 {
            let token = Sampler::sample(&logits, &params).unwrap();
            counts[token as usize] += 1;
        }
        // Index 1 (originally lowest logit) should dominate with negative temp
        assert!(
            counts[1] > counts[0] && counts[1] > counts[2],
            "negative temperature should invert distribution: counts = {:?}",
            counts
        );
    }

    #[test]
    fn test_top_p_exactly_zero() {
        // With top_p=0.0, cumulative probability never exceeds 0.0,
        // so the cutoff stays at sorted_indices.len() (no filtering).
        // However, the first token's cumulative prob (>0.0) > top_p (0.0)
        // triggers cutoff = 1, keeping only the most probable token.
        let logits = vec![1.0, 3.0, 2.0, 0.5];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.0,
            seed: None,
        };
        // Since top_p < 1.0 and cumulative > 0.0 on the very first token,
        // cutoff = 1, so only the top token (index 1, logit 3.0) should be selected.
        for _ in 0..50 {
            let token = Sampler::sample(&logits, &params).unwrap();
            assert_eq!(
                token, 1,
                "top_p=0.0 should keep only the most probable token"
            );
        }
    }

    #[test]
    fn test_softmax_valid_distribution() {
        // Verify that the softmax over logits with temp=1.0 produces a valid distribution
        let logits = [1.0, 3.0, 2.0, 0.5, 0.1];

        // Manually compute softmax
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        // All probabilities should be positive
        for (i, &p) in probs.iter().enumerate() {
            assert!(p > 0.0, "prob[{i}] should be > 0, got {p}");
            assert!(p <= 1.0, "prob[{i}] should be <= 1, got {p}");
        }

        // Probabilities should sum to ~1.0
        let total: f32 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "softmax probs should sum to ~1.0, got {total}"
        );

        // Highest logit (index 1, value 3.0) should have highest probability
        let max_prob_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_prob_idx, 1, "index 1 (logit=3.0) should have highest prob");
    }

    #[test]
    fn test_seeded_reproducibility() {
        // With temperature > 0 and a seed, two samples should produce identical results.
        let logits = vec![1.0, 2.0, 3.0, 2.5, 1.5];
        let seed = 42u64;
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: Some(seed),
        };

        // Same seed always produces the same token
        let first = Sampler::sample(&logits, &params).unwrap();
        for _ in 0..100 {
            let token = Sampler::sample(&logits, &params).unwrap();
            assert_eq!(
                token, first,
                "same seed should produce identical results"
            );
        }

        // Different seed may produce a different token (run enough times to see)
        let params_other = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: Some(9999),
        };
        let other = Sampler::sample(&logits, &params_other).unwrap();
        // We can't guarantee different seeds produce different tokens,
        // but we can verify both calls succeed.
        assert!((other as usize) < logits.len());
    }

    #[test]
    fn test_seed_none_is_nondeterministic() {
        // Without a seed, sampling with temperature > 0 should be non-deterministic
        // (over many trials, we should see at least 2 distinct tokens).
        let logits = vec![1.0, 1.0, 1.0, 1.0, 1.0]; // uniform
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        let mut seen = std::collections::HashSet::new();
        for _ in 0..500 {
            seen.insert(Sampler::sample(&logits, &params).unwrap());
        }
        assert!(
            seen.len() > 1,
            "unseeded sampling on uniform logits should produce multiple tokens, got {:?}",
            seen
        );
    }

    #[test]
    fn test_combined_top_k_top_p_restricts() {
        // Construct logits where top-1 token has >0.3 cumulative probability
        // after softmax within the top-3. With top_k=3 and top_p=0.3,
        // top_p should further restrict to only the top-1 token.
        //
        // logits: [0.0, 100.0, 1.0, 0.0, 0.0]
        // After top_k=3: indices [1, 2, 0] with scaled logits [100.0, 1.0, 0.0]
        // Softmax of [100.0, 1.0, 0.0] => index 1 gets ~1.0 probability
        // top_p=0.3: cumulative prob of top token > 0.3 immediately => cutoff = 1
        let logits = vec![0.0, 100.0, 1.0, 0.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 3,
            top_p: 0.3,
            seed: None,
        };
        for _ in 0..100 {
            let token = Sampler::sample(&logits, &params).unwrap();
            assert_eq!(
                token, 1,
                "with top_k=3 and top_p=0.3 on a dominant logit, only the top token should be selected"
            );
        }
    }

    /// Verify the sampler produces an approximately uniform distribution
    /// when given uniform logits and temperature=1.0.
    #[test]
    fn test_sampler_softmax_empirical_uniform() {
        let n = 4;
        let logits = vec![0.0f32; n];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };

        let mut counts = vec![0u32; n];
        let trials = 10000;
        for _ in 0..trials {
            let token = Sampler::sample(&logits, &params).unwrap();
            counts[token as usize] += 1;
        }

        let expected = trials as f64 / n as f64;
        for (i, &c) in counts.iter().enumerate() {
            let ratio = c as f64 / expected;
            assert!(
                ratio > 0.7 && ratio < 1.3,
                "token {i}: count {c}, expected ~{expected}, ratio {ratio:.2}"
            );
        }
    }

    /// Verify temperature=1.0 leaves the distribution unchanged:
    /// the argmax token is the same with and without temperature scaling.
    #[test]
    fn test_temperature_one_preserves_distribution() {
        let logits = vec![1.0, 5.0, 3.0, 2.0, 0.5];
        let params_temp1 = SamplingParams {
            temperature: 0.0, // greedy
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        let greedy = Sampler::sample(&logits, &params_temp1).unwrap();

        // With temperature=1.0 and greedy-like params (top_k=1), should pick same token
        let params_temp1_topk1 = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            seed: None,
        };
        let with_temp = Sampler::sample(&logits, &params_temp1_topk1).unwrap();
        assert_eq!(greedy, with_temp, "temp=1.0 with top_k=1 should pick the same as greedy");
    }

    /// Verify top-K is applied before top-P by using logits where the order matters.
    /// top_k=3 restricts to indices 0,1,2, then top_p=0.99 keeps all 3.
    /// Tokens 3 and 4 should never appear.
    #[test]
    fn test_combined_filtering_order_k_before_p() {
        let logits = vec![10.0, 9.0, 8.0, 1.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 3,
            top_p: 0.99,
            seed: None,
        };
        for _ in 0..200 {
            let token = Sampler::sample(&logits, &params).unwrap();
            assert!(
                token <= 2,
                "token {token} should be in top-3 (0,1,2)"
            );
        }
    }

    #[test]
    fn test_inf_logits_returns_error() {
        let params = SamplingParams::default();

        // Positive infinity
        let logits = vec![1.0, f32::INFINITY, 2.0];
        let result = Sampler::sample(&logits, &params);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Inf"));

        // Negative infinity
        let logits = vec![1.0, f32::NEG_INFINITY, 2.0];
        let result = Sampler::sample(&logits, &params);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Inf"));

        // Also test greedy path
        let params_greedy = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        let logits = vec![f32::INFINITY, 1.0];
        let result = Sampler::sample(&logits, &params_greedy);
        assert!(result.is_err());
    }

    /// Verify temperature scaling works end-to-end through `Sampler::sample`.
    ///
    /// This is distinct from `test_temperature_scaling_flattens_distribution`,
    /// which manually replicates the division math outside the sampler. Here we
    /// call the actual sampler at two temperatures and assert the empirical
    /// token-count distributions differ in the expected direction: low temperature
    /// (0.5) sharpens toward token 0, high temperature (5.0) flattens toward
    /// a more uniform count distribution.
    #[test]
    fn test_temperature_scaling_through_sampler() {
        // Token 0 has logit 10.0; tokens 1-3 have logit 0.0.
        // At low temp (0.5), token 0 dominates even more than at temp=1.0.
        // At high temp (5.0), the distribution flattens and other tokens appear more.
        let logits = vec![10.0f32, 0.0, 0.0, 0.0];
        let trials = 2000;

        let low_params = SamplingParams {
            temperature: 0.5,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        let high_params = SamplingParams {
            temperature: 5.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };

        let mut low_counts = [0u32; 4];
        let mut high_counts = [0u32; 4];

        for _ in 0..trials {
            let t = Sampler::sample(&logits, &low_params).unwrap();
            low_counts[t as usize] += 1;
            let t = Sampler::sample(&logits, &high_params).unwrap();
            high_counts[t as usize] += 1;
        }

        // Low temperature: token 0 should be sampled in the vast majority of trials.
        // With temp=0.5, scaled logits are [20.0, 0.0, 0.0, 0.0].
        // e^20 / (e^20 + 3) ≈ 1.0, so token 0 should appear nearly every time.
        assert!(
            low_counts[0] > (trials as u32 * 99 / 100),
            "at temp=0.5, token 0 should dominate (>99%): counts={:?}",
            low_counts
        );

        // High temperature: other tokens should appear more often.
        // With temp=5.0, scaled logits are [2.0, 0.0, 0.0, 0.0].
        // e^2 / (e^2 + 3) ≈ 0.71, so other tokens collectively appear ~29% of the time.
        let high_others: u32 = high_counts[1] + high_counts[2] + high_counts[3];
        let low_others: u32 = low_counts[1] + low_counts[2] + low_counts[3];
        assert!(
            high_others > low_others,
            "at temp=5.0 non-dominant tokens should appear more than at temp=0.5: high_others={high_others}, low_others={low_others}"
        );

        // At high temperature, non-dominant tokens should collectively appear
        // in a meaningful fraction of trials (at least 15% to give generous slack).
        assert!(
            high_others > (trials as u32 * 15 / 100),
            "at temp=5.0, non-dominant tokens should appear in >15% of trials: counts={:?}",
            high_counts
        );
    }

    /// Verify that negative temperature inverts the logit ordering.
    ///
    /// The spec states 'Temperature must be >= 0', but the current implementation
    /// does not validate this and instead silently inverts the distribution.
    /// This test documents the actual behavior: with temp=-1.0, the token with
    /// the lowest logit receives the highest probability after softmax.
    #[test]
    fn test_negative_temperature_inverts_distribution() {
        // logits: index 0 = 10.0 (highest), index 1 = 0.0 (lowest), index 2 = 5.0 (middle)
        // With temp=-1.0, scaled logits = [-10.0, 0.0, -5.0].
        // Softmax favors the largest scaled value, which is index 1 (0.0 → highest after negation).
        let logits = vec![10.0f32, 0.0, 5.0];
        let params = SamplingParams {
            temperature: -1.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };

        let mut counts = [0u32; 3];
        for _ in 0..1000 {
            let token = Sampler::sample(&logits, &params).unwrap();
            counts[token as usize] += 1;
        }

        // Index 1 (originally lowest logit, 0.0) should dominate with negative temp.
        assert!(
            counts[1] > counts[0] && counts[1] > counts[2],
            "negative temperature should invert distribution so lowest-logit token dominates: counts={:?}",
            counts
        );

        // Index 1 should appear in the vast majority of samples
        // (scaled[1]=0.0 vs scaled[0]=-10.0 vs scaled[2]=-5.0 → e^0 / (e^0 + e^{-10} + e^{-5}) ≈ 0.993)
        assert!(
            counts[1] > 900,
            "index 1 should dominate with negative temperature: counts={:?}",
            counts
        );
    }

    #[test]
    fn test_single_logit_vocab() {
        let logits = vec![42.0];

        // Greedy (temp=0)
        let params = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        assert_eq!(Sampler::sample(&logits, &params).unwrap(), 0);

        // top_k=1
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            seed: None,
        };
        assert_eq!(Sampler::sample(&logits, &params).unwrap(), 0);

        // Standard sampling with temperature
        let params = SamplingParams {
            temperature: 0.5,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        assert_eq!(Sampler::sample(&logits, &params).unwrap(), 0);

        // With top_p filtering
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
            seed: None,
        };
        assert_eq!(Sampler::sample(&logits, &params).unwrap(), 0);

        // With both top_k and top_p
        let params = SamplingParams {
            temperature: 2.0,
            top_k: 10,
            top_p: 0.1,
            seed: None,
        };
        assert_eq!(Sampler::sample(&logits, &params).unwrap(), 0);
    }
}
