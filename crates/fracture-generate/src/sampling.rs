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
