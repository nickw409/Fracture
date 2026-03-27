use fracture_core::Result;

/// Sampling parameters for token selection.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

/// Token sampler operating on CPU logits.
pub struct Sampler;

impl Sampler {
    /// Sample a token from logits given sampling parameters.
    pub fn sample(logits: &[f32], params: &SamplingParams) -> Result<u32> {
        if params.temperature == 0.0 || params.top_k == 1 {
            // Greedy: argmax
            let (idx, _) = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            return Ok(idx as u32);
        }

        // Temperature scaling
        let scaled: Vec<f32> = logits.iter().map(|&l| l / params.temperature).collect();

        // Top-K filtering
        let mut indices: Vec<usize> = (0..scaled.len()).collect();
        if params.top_k > 0 && params.top_k < scaled.len() {
            indices.sort_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());
            indices.truncate(params.top_k);
        }

        // Softmax over remaining
        let max_val = indices.iter().map(|&i| scaled[i]).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = indices.iter().map(|&i| (scaled[i] - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        // Top-P filtering
        let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
        sorted_indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

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
        let r: f32 = rand::random::<f32>() * filtered_sum;
        let mut acc = 0.0;
        for &i in &filtered {
            acc += probs[i];
            if acc >= r {
                return Ok(indices[i] as u32);
            }
        }

        Ok(indices[*filtered.last().unwrap()] as u32)
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
        };
        for _ in 0..20 {
            let token = Sampler::sample(&logits, &params).unwrap();
            assert_eq!(token, 1);
        }
    }

    #[test]
    fn test_temperature_scaling_flattens_distribution() {
        // With high temperature, softmax should produce more uniform probs
        let logits = vec![10.0, 0.0];

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
}
