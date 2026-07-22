use rayon::prelude::*;

use super::{
    request::HbmRequest,
    heatmap::Heatmap,
    index::RoutingIndex,
    channel::HbmChannel,
};

/// Multilayer scratchpad reinforcement memory.
/// Tracks per-layer routing history, failures, and adaptive bias.
#[derive(Debug, Clone)]
pub struct Scratchpad {
    pub layers: usize,
    pub history: Vec<Vec<Option<usize>>>, // [layer][recent exit]
    pub failures: Vec<u32>,               // per-layer failure counters
}

impl Scratchpad {
    pub fn new(layers: usize) -> Self {
        Self {
            layers,
            history: vec![vec![None; 8]; layers], // last 8 exits per layer
            failures: vec![0; layers],
        }
    }

    /// Record a successful exit for reinforcement.
    pub fn record_success(&mut self, layer: usize, exit_id: usize) {
        if let Some(layer_hist) = self.history.get_mut(layer) {
            layer_hist.rotate_right(1);
            layer_hist[0] = Some(exit_id);
        }
    }

    /// Record a failure (circulation without exit).
    pub fn record_failure(&mut self, layer: usize) {
        if let Some(f) = self.failures.get_mut(layer) {
            *f += 1;
        }
    }

    /// Parallel multilayer bias computation.
    /// FIXED: compute biases in parallel, apply AFTER.
    pub fn apply_bias_parallel(
        &self,
        req: &mut HbmRequest,
        heatmap: &Heatmap,
        channels: &[HbmChannel],
    ) {
        // Compute all biases in parallel
        let biases: Vec<f32> = (0..self.layers)
            .into_par_iter()
            .map(|layer| {
                // Failure bias
                let fail_bias = self.failures[layer] as f32 * 0.05;

                // Recent exit bias
                let recent_bias = self.history[layer][0].map(|exit_id| {
                    let channel = channels.iter().find(|c| c.id == exit_id);
                    if let Some(ch) = channel {
                        let idx_score = RoutingIndex::score_channel(req, ch, self.layers);
                        -0.1 + (idx_score * 0.01)
                    } else {
                        -0.1
                    }
                }).unwrap_or(0.0);

                // Heatmap bias
                let heat_bias = if let Some(layer_vec) = heatmap.layers.get(layer) {
                    if !layer_vec.is_empty() {
                        let avg_heat = layer_vec.iter().copied().sum::<f32>() 
                            / layer_vec.len() as f32;
                        avg_heat * 0.10
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                fail_bias + recent_bias + heat_bias
            })
            .collect();

        // Apply biases sequentially (safe)
        for layer in 0..self.layers {
            req.update_layer_bias(layer, req.layer_bias[layer] + biases[layer]);
        }
    }
}
