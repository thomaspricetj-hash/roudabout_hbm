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
    // -------------------------------------------------------------------------
    // Tesla valve directional scoring helper
    // -------------------------------------------------------------------------

    fn valve_component(req: &HbmRequest, ch: &HbmChannel) -> f32 {
        let forward = req.valve_forward * -0.10 + ch.valve_forward * -0.10;
        let reverse = req.valve_reverse * 0.12 + ch.valve_reverse * 0.12;
        let oscillation = req.valve_oscillation * 0.15 + ch.valve_oscillation * 0.15;
        forward + reverse + oscillation
    }

    // -------------------------------------------------------------------------
    // BASIC SCORING
    // -------------------------------------------------------------------------

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

        // Tesla valve directional influence
        score += Self::valve_component(request, channel);

        score
    }

    // -------------------------------------------------------------------------
    // PARALLEL SCORING
    // -------------------------------------------------------------------------

    pub fn score_channel_parallel(
        request: &HbmRequest,
        channel: &HbmChannel,
        heatmap: &Heatmap,
        layer_count: usize,
    ) -> f32 {
        let base_score: f32 = (0..layer_count)
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
            .sum();

        base_score + Self::valve_component(request, channel)
    }

    // -------------------------------------------------------------------------
    // PARALLEL + GRID + BITDROP + TESLA VALVE
    // -------------------------------------------------------------------------

    pub fn score_channel_parallel_with_grid(
        request: &HbmRequest,
        channel: &HbmChannel,
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        layer_count: usize,
    ) -> f32 {
        let base_score: f32 = (0..layer_count)
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
                    0.35 * ccg.cluster_bias[layer][channel.id] +
                    0.25 * ccg.zone_bias[layer][channel.id] +
                    0.20 * ccg.door_bias[layer][channel.id] +
                    0.20 * ccg.geom_bias[layer][channel.id];

                let scratch = heatmap
                    .scratch
                    .get(layer)
                    .and_then(|v| v.get(channel.id))
                    .copied()
                    .unwrap_or(0.0);

                let door_rot =
                    ccg.door_rotation[layer][channel.id] as f32 * 0.01;

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
            .sum();

        base_score + Self::valve_component(request, channel)
    }

    // -------------------------------------------------------------------------
    // LOCALITY SCORE (Tesla valve integrated)
    // -------------------------------------------------------------------------

    pub fn locality_score(channel: &HbmChannel, heatmap: &Heatmap) -> f32 {
        let mut score = 0.0;

        score += heatmap.row_conflict[channel.id] * 0.40;
        score += heatmap.bank_busy[channel.id] * 0.35;
        score += heatmap.channel_sat[channel.id] * 0.25;

        score -= heatmap.refresh_heat[channel.id] * 0.30;
        score -= heatmap.ecc_heat[channel.id] * 0.25;

        // Tesla valve directional influence
        score += channel.valve_forward * -0.05;
        score += channel.valve_reverse * 0.06;
        score += channel.valve_oscillation * 0.08;

        score
    }

    // -------------------------------------------------------------------------
    // GEOMETRY SCORE (Tesla valve integrated)
    // -------------------------------------------------------------------------

    pub fn geometry_score(channel: &HbmChannel, ccg: &CrossConnectGrid) -> f32 {
        let id = channel.id;
        let mut score = 0.0;

        let weights = [0.30, 0.25, 0.20, 0.15];

        for (layer, weight) in weights.iter().enumerate() {
            if let Some(row) = ccg.geom_bias.get(layer).and_then(|v| v.get(id)) {
                score += *row * *weight;
            }
        }

        // Tesla valve directional influence
        score += channel.valve_forward * -0.04;
        score += channel.valve_reverse * 0.05;
        score += channel.valve_oscillation * 0.06;

        score
    }

    // -------------------------------------------------------------------------
    // RELIABILITY SCORE (Tesla valve integrated)
    // -------------------------------------------------------------------------

    pub fn reliability_score(channel: &HbmChannel) -> f32 {
        let mut score =
            (channel.metrics.stability_score * 0.50)
            - (channel.metrics.ecc_activity * 0.30)
            - (channel.metrics.refresh_pressure * 0.20);

        // Tesla valve directional influence
        score += channel.valve_forward * 0.05;
        score -= channel.valve_reverse * 0.08;
        score -= channel.valve_oscillation * 0.10;

        score
    }

    // -------------------------------------------------------------------------
    // BITDROP INDEX (Tesla valve integrated)
    // -------------------------------------------------------------------------

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

        // Tesla valve directional influence
        score += request.valve_forward * -0.06;
        score += request.valve_reverse * 0.08;
        score += request.valve_oscillation * 0.10;

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

        // Tesla valve directional influence
        score += channel.valve_forward * -0.05;
        score += channel.valve_reverse * 0.07;
        score += channel.valve_oscillation * 0.09;

        score
    }

    // -------------------------------------------------------------------------
    // COMPOSITE INDEX SCORE (Tesla valve integrated)
    // -------------------------------------------------------------------------

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
        let pair_score = channel.pair_score_component();

        let bitdrop_req = Self::bitdrop_request_index(request);
        let bitdrop_ch = Self::bitdrop_channel_index(channel);

        let valve_component = Self::valve_component(request, channel);

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
            + valve_component
    }

    // -------------------------------------------------------------------------
    // DF‑HBM DELTA INDEX SCORE (Tesla valve integrated)
    // -------------------------------------------------------------------------

    pub fn dfhbm_index_score(
        request: &HbmRequest,
        channel: &HbmChannel,
        ccg: &CrossConnectGrid,
        heatmap: &Heatmap,
    ) -> f32 {
        let df_req = request.dfhbm_score() * 0.40;

        let df_ch =
            channel.metrics.delta_load * 0.25 +
            channel.metrics.delta_stability * 0.25 -
            channel.metrics.delta_row_conflict * 0.20 -
            channel.metrics.delta_bank_busy * 0.20 -
            channel.metrics.delta_channel_sat * 0.15 +
            channel.metrics.delta_refresh_pressure * 0.15 +
            channel.metrics.delta_ecc_activity * 0.15;

        let df_geom = ccg.fused_delta_bias(channel.id) * 0.35;

        let df_heat =
            heatmap.row_conflict[channel.id] * 0.10 +
            heatmap.bank_busy[channel.id] * 0.10 +
            heatmap.channel_sat[channel.id] * 0.10;

        let valve_component = Self::valve_component(request, channel);

        df_req + df_ch + df_geom + df_heat + valve_component
    }
}



