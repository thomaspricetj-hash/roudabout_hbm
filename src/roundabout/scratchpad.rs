use rayon::prelude::*;

use super::{
    request::HbmRequest,
    heatmap::Heatmap,
    index::RoutingIndex,
    channel::HbmChannel,
    grid::CrossConnectGrid,
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

    /// MAX‑tier parallel multilayer bias computation (heat + grid + index + failures)
    pub fn apply_bias_parallel(
        &self,
        req: &mut HbmRequest,
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
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
                        let idx_score = RoutingIndex::score_channel_parallel_with_grid(
                            req,
                            ch,
                            heatmap,
                            ccg,
                            self.layers,
                        );
                        -0.1 + (idx_score * 0.01)
                    } else {
                        -0.1
                    }
                }).unwrap_or(0.0);

                // Heatmap bias (per-layer)
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

                // Grid bias (cluster + zone + door + geom)
                let grid_bias =
                    0.35 * ccg.cluster_bias[layer][req.channel_id] +
                    0.25 * ccg.zone_bias[layer][req.channel_id] +
                    0.20 * ccg.door_bias[layer][req.channel_id] +
                    0.20 * ccg.geom_bias[layer][req.channel_id];

                // Rotating door bias
                let door_rot = ccg.door_rotation[layer][req.channel_id] as f32 * 0.01;

                fail_bias + recent_bias + heat_bias + door_rot - grid_bias
            })
            .collect();

        // Apply biases sequentially (safe)
        for layer in 0..self.layers {
            req.update_layer_bias(layer, req.layer_bias[layer] + biases[layer]);
        }
    }
}

