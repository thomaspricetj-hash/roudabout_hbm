use rayon::prelude::*;

use super::{
    channel::HbmChannel,
    request::HbmRequest,
    heatmap::Heatmap,
    grid::CrossConnectGrid,
};

#[derive(Debug, Clone)]
pub struct RoutingIndex;

impl RoutingIndex {
    /// Legacy sequential scoring (kept for compatibility)
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

    /// Legacy MAX‑tier parallel multilayer scoring (heatmap only)
    /// Used by existing call sites that don't know about CrossConnectGrid yet.
    pub fn score_channel_parallel(
        request: &HbmRequest,
        channel: &HbmChannel,
        heatmap: &Heatmap,
        layer_count: usize,
    ) -> f32 {
        (0..layer_count)
            .into_par_iter()
            .map(|layer| {
                let base = match layer {
                    0 => channel.metrics.load,
                    1 => channel.metrics.refresh_pressure,
                    2 => channel.metrics.jitter_cycles,
                    3 => 1.0 - channel.metrics.stability_score,
                    _ => 0.0,
                };

                let req_bias = request.layer_bias.get(layer).copied().unwrap_or(0.0);

                let heat_bias = heatmap.layers.get(layer).map(|layer_vec| {
                    if !layer_vec.is_empty() {
                        let avg_heat = layer_vec.iter().copied().sum::<f32>()
                            / layer_vec.len() as f32;
                        avg_heat * 0.10
                    } else {
                        0.0
                    }
                }).unwrap_or(0.0);

                base + req_bias + heat_bias
            })
            .sum()
    }

    /// MAX‑tier multilayer parallel scoring (Heatmap + CrossConnectGrid)
    /// New path for grid‑aware routing.
    pub fn score_channel_parallel_with_grid(
        request: &HbmRequest,
        channel: &HbmChannel,
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        layer_count: usize,
    ) -> f32 {
        (0..layer_count)
            .into_par_iter()
            .map(|layer| {
                let base = match layer {
                    0 => channel.metrics.load,
                    1 => channel.metrics.refresh_pressure,
                    2 => channel.metrics.jitter_cycles,
                    3 => 1.0 - channel.metrics.stability_score,
                    _ => 0.0,
                };

                let req_bias = request.layer_bias.get(layer).copied().unwrap_or(0.0);

                let heat = heatmap.layers[layer][channel.id];

                let grid_bias =
                    0.35 * ccg.cluster_bias[layer][channel.id] +
                    0.25 * ccg.zone_bias[layer][channel.id] +
                    0.20 * ccg.door_bias[layer][channel.id] +
                    0.20 * ccg.geom_bias[layer][channel.id];

                let scratch = heatmap.scratch[layer][channel.id];

                let door_rot = ccg.door_rotation[layer][channel.id] as f32 * 0.01;

                base + req_bias + heat + scratch + door_rot - grid_bias
            })
            .sum()
    }

    /// MAX‑tier composite index score (metrics + heatmap + grid)
    pub fn composite_index_score(
        request: &HbmRequest,
        channel: &HbmChannel,
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        layer_count: usize,
    ) -> f32 {
        let layer_score =
            Self::score_channel_parallel_with_grid(request, channel, heatmap, ccg, layer_count);

        let channel_score =
            (channel.metrics.load * 0.20)
            + (channel.metrics.refresh_pressure * 0.30)
            + (channel.metrics.ecc_activity * 0.25)
            + (channel.metrics.jitter_cycles * 0.10)
            + ((1.0 - channel.metrics.stability_score) * 0.20);

        let fused_heat = heatmap.fused_heat(channel.id);
        let fused_grid = ccg.fused_bias(channel.id);

        layer_score + channel_score + fused_heat - fused_grid
    }
}


