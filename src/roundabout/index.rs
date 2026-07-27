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

                let heat = heatmap
                    .layers
                    .get(layer)
                    .and_then(|layer_vec| layer_vec.get(channel.id))
                    .copied()
                    .unwrap_or(0.0);

                let grid_bias =
                    0.35 * ccg
                        .cluster_bias
                        .get(layer)
                        .and_then(|v| v.get(channel.id))
                        .copied()
                        .unwrap_or(0.0)
                    + 0.25 * ccg
                        .zone_bias
                        .get(layer)
                        .and_then(|v| v.get(channel.id))
                        .copied()
                        .unwrap_or(0.0)
                    + 0.20 * ccg
                        .door_bias
                        .get(layer)
                        .and_then(|v| v.get(channel.id))
                        .copied()
                        .unwrap_or(0.0)
                    + 0.20 * ccg
                        .geom_bias
                        .get(layer)
                        .and_then(|v| v.get(channel.id))
                        .copied()
                        .unwrap_or(0.0);

                let scratch = heatmap
                    .scratch
                    .get(layer)
                    .and_then(|v| v.get(channel.id))
                    .copied()
                    .unwrap_or(0.0);

                let door_rot = ccg
                    .door_rotation
                    .get(layer)
                    .and_then(|v| v.get(channel.id))
                    .copied()
                    .unwrap_or(0) as f32
                    * 0.01;

                let bitdrop_layer =
                    request.locality_score * 0.02 +
                    request.refresh_pressure * 0.02 +
                    request.ecc_pressure * 0.02 +
                    request.tunnel_preference * 0.03 +
                    request.tunnel_heat * 0.02 +
                    request.tunnel_score * 0.03 +
                    (1.0 - request.stability_factor) * 0.02 +
                    (request.adaptive_weight - 1.0) * 0.02 +
                    channel.locality_score * 0.02 +
                    channel.heat_affinity * 0.02 +
                    (1.0 - channel.reliability_score) * 0.03 +
                    channel.tunnel_bias * 0.03 +
                    (1.0 - channel.tunnel_reliability) * 0.03;

                base + req_bias + heat + scratch + door_rot + bitdrop_layer - grid_bias
            })
            .sum()
    }

    pub fn locality_score(channel: &HbmChannel, heatmap: &Heatmap) -> f32 {
        let mut score = 0.0;

        score += heatmap.row_conflict.get(channel.id).copied().unwrap_or(0.0) * 0.40;
        score += heatmap.bank_busy.get(channel.id).copied().unwrap_or(0.0) * 0.35;
        score += heatmap.channel_sat.get(channel.id).copied().unwrap_or(0.0) * 0.25;

        score -= heatmap.refresh_heat.get(channel.id).copied().unwrap_or(0.0) * 0.30;
        score -= heatmap.ecc_heat.get(channel.id).copied().unwrap_or(0.0) * 0.25;

        score
    }

    pub fn geometry_score(channel: &HbmChannel, ccg: &CrossConnectGrid) -> f32 {
        let id = channel.id;
        let mut score = 0.0;

        let weights = [0.30, 0.25, 0.20, 0.15];

        for (layer, weight) in weights.iter().enumerate() {
            if let Some(row) = ccg.geom_bias.get(layer).and_then(|v| v.get(id)) {
                score += *row * *weight;
            }
        }

        score
    }

    pub fn reliability_score(channel: &HbmChannel) -> f32 {
        (channel.metrics.stability_score * 0.50)
            - (channel.metrics.ecc_activity * 0.30)
            - (channel.metrics.refresh_pressure * 0.20)
    }

    pub fn pair_index_score(channel: &HbmChannel) -> f32 {
        channel.pair_score_component()
    }

    pub fn bitdrop_request_index(request: &HbmRequest) -> f32 {
        let mut score = 0.0;

        score += request.locality_score * 0.08;
        score += request.refresh_pressure * 0.06;
        score += request.ecc_pressure * 0.06;

        score += request.tunnel_preference * 0.08;
        score += request.tunnel_heat * 0.05;
        score += request.tunnel_score * 0.07;

        score += (request.adaptive_weight - 1.0) * 0.05;
        score += (1.0 - request.stability_factor) * 0.05;

        if request.is_tunnel_escalated {
            score += 0.06;
        }

        score
    }

    pub fn bitdrop_channel_index(channel: &HbmChannel) -> f32 {
        let mut score = 0.0;

        score += channel.heat_affinity * 0.06;
        score += channel.locality_score * 0.07;

        score += (1.0 - channel.reliability_score) * 0.08;

        if channel.is_tunnel {
            score += channel.tunnel_bias * 0.08;
            score += (1.0 - channel.tunnel_reliability) * 0.07;
        }

        if channel.group_size > 1 {
            score += (channel.group_size as f32 - 1.0) * 0.02;
        }

        score
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

        let bitdrop_req = Self::bitdrop_request_index(request);
        let bitdrop_ch = Self::bitdrop_channel_index(channel);

        layer_score
            + channel_score
            + fused_heat
            - fused_grid
            + locality
            + geometry
            + reliability
            + pair_score
            + bitdrop_req
            + bitdrop_ch
    }
}

