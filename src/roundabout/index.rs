use rayon::prelude::*;

use super::{
    channel::HbmChannel,
    request::HbmRequest,
    heatmap::Heatmap,
};

#[derive(Debug, Clone)]
pub struct RoutingIndex;

impl RoutingIndex {
    /// Original sequential scoring (kept for compatibility)
    pub fn score_channel(
        request: &HbmRequest,
        channel: &HbmChannel,
        layer_count: usize,
    ) -> f32 {
        let mut score = 0.0;

        for layer in 0..layer_count {
            let base = match layer {
                0 => channel.metrics.load,
                1 => channel.metrics.refresh_pressure,
                2 => channel.metrics.jitter_cycles,
                3 => 1.0 - channel.metrics.stability_score,
                _ => 0.0,
            };

            let bias = request.layer_bias.get(layer).copied().unwrap_or(0.0);
            score += base + bias;
        }

        score
    }

    /// MAX‑tier parallel multilayer scoring
    pub fn score_channel_parallel(
        request: &HbmRequest,
        channel: &HbmChannel,
        heatmap: &Heatmap,
        layer_count: usize,
    ) -> f32 {
        // Compute per-layer contributions in parallel
        let layer_sum: f32 = (0..layer_count)
            .into_par_iter()
            .map(|layer| {
                // Base metric for this layer
                let base = match layer {
                    0 => channel.metrics.load,
                    1 => channel.metrics.refresh_pressure,
                    2 => channel.metrics.jitter_cycles,
                    3 => 1.0 - channel.metrics.stability_score,
                    _ => 0.0,
                };

                // Request bias
                let req_bias = request.layer_bias.get(layer).copied().unwrap_or(0.0);

                // Heatmap bias
                let heat_bias = heatmap.layers.get(layer).map(|layer_vec| {
                    if !layer_vec.is_empty() {
                        let avg_heat = layer_vec.iter().copied().sum::<f32>()
                            / layer_vec.len() as f32;
                        avg_heat * 0.10
                    } else {
                        0.0
                    }
                }).unwrap_or(0.0);

                // Combine contributions
                base + req_bias + heat_bias
            })
            .sum();

        layer_sum
    }

    /// MAX‑tier: full composite index score including channel metrics
    pub fn composite_index_score(
        request: &HbmRequest,
        channel: &HbmChannel,
        heatmap: &Heatmap,
        layer_count: usize,
    ) -> f32 {
        let layer_score = Self::score_channel_parallel(request, channel, heatmap, layer_count);

        // Channel-level contributions
        let channel_score =
            (channel.metrics.load * 0.20)
            + (channel.metrics.refresh_pressure * 0.30)
            + (channel.metrics.ecc_activity * 0.25)
            + (channel.metrics.jitter_cycles * 0.10)
            + ((1.0 - channel.metrics.stability_score) * 0.20);

        layer_score + channel_score
    }
}
