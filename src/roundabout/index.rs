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

    pub fn locality_score(channel: &HbmChannel, heatmap: &Heatmap) -> f32 {
        let mut score = 0.0;

        score += heatmap.row_conflict[channel.id] * 0.40;
        score += heatmap.bank_busy[channel.id] * 0.35;
        score += heatmap.channel_sat[channel.id] * 0.25;

        score -= heatmap.refresh_heat[channel.id] * 0.30;
        score -= heatmap.ecc_heat[channel.id] * 0.25;

        score
    }

    pub fn geometry_score(channel: &HbmChannel, ccg: &CrossConnectGrid) -> f32 {
        let id = channel.id;

        let mut score = 0.0;

        score += ccg.geom_bias[0][id] * 0.30;
        score += ccg.geom_bias[1][id] * 0.25;
        score += ccg.geom_bias[2][id] * 0.20;
        score += ccg.geom_bias[3][id] * 0.15;

        score
    }

    pub fn reliability_score(channel: &HbmChannel) -> f32 {
        (channel.metrics.stability_score * 0.50)
            - (channel.metrics.ecc_activity * 0.30)
            - (channel.metrics.refresh_pressure * 0.20)
    }

    /// NEW: grouped‑pair index contribution
    pub fn pair_index_score(channel: &HbmChannel) -> f32 {
        channel.pair_score_component()
    }

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

        let locality = Self::locality_score(channel, heatmap);
        let geometry = Self::geometry_score(channel, ccg);
        let reliability = Self::reliability_score(channel);
        let pair_score = Self::pair_index_score(channel);

        layer_score
            + channel_score
            + fused_heat
            - fused_grid
            + locality
            + geometry
            + reliability
            + pair_score
    }
}


