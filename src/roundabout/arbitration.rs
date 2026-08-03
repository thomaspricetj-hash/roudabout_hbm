use rayon::prelude::*;

use super::{
    request::{HbmRequest, RequestPriority},
    channel::HbmChannel,
    heatmap::Heatmap,
    index::RoutingIndex,
    grid::CrossConnectGrid,
    controller::{DeltaBuffer, DeltaStore, EffectiveView},
};

#[derive(Debug, Default)]
pub struct ArbitrationEngine;

impl ArbitrationEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Original priority weight (kept intact)
    pub fn priority_weight(req: &HbmRequest) -> f32 {
        match req.priority {
            RequestPriority::High => 0.5,
            RequestPriority::Standard => 1.0,
            RequestPriority::Low => 1.5,
        }
    }

    /// HBM‑aware priority escalation
    pub fn hbm_priority_escalation(req: &HbmRequest) -> f32 {
        let mut esc = 0.0;

        if req.locality_score > 0.5 {
            esc -= 0.10;
        }

        if req.refresh_pressure > 0.5 {
            esc -= 0.05;
        }
        if req.ecc_pressure > 0.5 {
            esc -= 0.05;
        }

        if req.is_tunnel_escalated {
            esc -= 0.08;
        }

        esc += (req.adaptive_weight - 1.0) * 0.03;
        esc += (1.0 - req.stability_factor) * 0.02;

        esc
    }

    // -------------------------------------------------------------------------
    // NEW: Tesla‑style one‑way valve logic (Directional Flow Valve)
    // -------------------------------------------------------------------------

    fn tesla_valve_score(payload: &[u8]) -> f32 {
        if payload.is_empty() {
            return 0.0;
        }

        let mut forward_flow = 0.0;
        let mut reverse_flow = 0.0;
        let mut oscillation = 0.0;

        let mut last_bit = payload[0] & 1;

        for byte in payload {
            for i in 0..8 {
                let bit = (byte >> i) & 1;

                if bit == last_bit {
                    forward_flow += 0.015;
                } else {
                    reverse_flow += 0.030;
                }

                if bit != last_bit {
                    oscillation += 0.020;
                }

                last_bit = bit;
            }
        }

        forward_flow - reverse_flow - oscillation
    }

    // -------------------------------------------------------------------------
    // MAX‑tier parallel arbitration upgrades
    // -------------------------------------------------------------------------

    pub fn arbitration_score_parallel(
        req: &HbmRequest,
        channel: &HbmChannel,
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        layer_count: usize,
    ) -> f32 {
        let priority = Self::priority_weight(req) + Self::hbm_priority_escalation(req);

        let index_score = RoutingIndex::score_channel_parallel_with_grid(
            req,
            channel,
            heatmap,
            ccg,
            layer_count,
        );

        let metrics_score =
            (channel.metrics.load * 0.20)
            + (channel.metrics.refresh_pressure * 0.30)
            + (channel.metrics.ecc_activity * 0.25)
            + (channel.metrics.jitter_cycles * 0.10)
            + ((1.0 - channel.metrics.stability_score) * 0.20);

        let bank_busy_score = channel.bank_busy_score_parallel();

        let heat_affinity = heatmap.layers
            .par_iter()
            .map(|layer| layer.get(channel.id).copied().unwrap_or(0.0))
            .sum::<f32>();

        let locality =
            heatmap.row_conflict[channel.id] * 0.40 +
            heatmap.bank_busy[channel.id] * 0.35 +
            heatmap.channel_sat[channel.id] * 0.25;

        let penalties =
            -heatmap.refresh_heat[channel.id] * 0.30 -
            heatmap.ecc_heat[channel.id] * 0.25;

        let grid_bias = ccg.fused_bias(channel.id);

        let pair_component = channel.pair_score_component();

        let channel_bitdrop_component =
            channel.heat_affinity * 0.05 +
            (1.0 - channel.reliability_score) * 0.10 +
            channel.locality_score * 0.06 +
            channel.tunnel_bias * 0.08 +
            (1.0 - channel.tunnel_reliability) * 0.07;

        let request_bitdrop_component =
            req.locality_score * 0.06 +
            req.refresh_pressure * 0.05 +
            req.ecc_pressure * 0.05 +
            req.tunnel_preference * 0.08 +
            req.tunnel_heat * 0.04 +
            req.tunnel_score * 0.06 +
            (1.0 - req.stability_factor) * 0.05;

        priority
            + index_score
            + metrics_score
            + bank_busy_score
            + heat_affinity
            + locality
            + penalties
            + pair_component
            + channel_bitdrop_component
            + request_bitdrop_component
            - grid_bias
    }

    /// Parallel arbitration across all channels (BitDrop‑aware + Tesla Valve)
    pub fn choose_best_channel_parallel_with_payload(
        &self,
        req: &HbmRequest,
        channels: &[HbmChannel],
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        layer_count: usize,
        raw_payload: &[u8],
        profile_hint: Option<&str>,
    ) -> Option<usize> {
        let valve_score = Self::tesla_valve_score(raw_payload);

        channels
            .par_iter()
            .filter_map(|ch| {
                let mut ch_clone = ch.clone();
                ch_clone.update_bitdrop_biases(raw_payload, profile_hint);

                let other_load = ch_clone.pair_id.and_then(|pid| {
                    channels
                        .iter()
                        .find(|c| c.pair_id == Some(pid) && c.id != ch_clone.id)
                        .map(|c| c.metrics.load)
                });

                if !ch_clone.can_accept_with_pair(req.bank_id, other_load) {
                    return None;
                }

                let mut score = Self::arbitration_score_parallel(
                    req,
                    &ch_clone,
                    heatmap,
                    ccg,
                    layer_count,
                );

                score += valve_score;

                if req.is_tunnel_escalated {
                    if ch_clone.is_tunnel {
                        score -= 0.10;
                    } else {
                        score += 0.05;
                    }
                }

                if ch_clone.group_size > 1 {
                    score -= (ch_clone.group_size as f32 - 1.0) * 0.02;
                }

                Some((ch_clone.id, score))
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    /// Original parallel arbitration (Tesla Valve added)
    pub fn choose_best_channel_parallel(
        &self,
        req: &HbmRequest,
        channels: &[HbmChannel],
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        layer_count: usize,
    ) -> Option<usize> {
        let valve_score = 0.0;

        channels
            .par_iter()
            .filter_map(|ch| {
                let other_load = ch.pair_id.and_then(|pid| {
                    channels
                        .iter()
                        .find(|c| c.pair_id == Some(pid) && c.id != ch.id)
                        .map(|c| c.metrics.load)
                });

                if !ch.can_accept_with_pair(req.bank_id, other_load) {
                    return None;
                }

                let mut score = Self::arbitration_score_parallel(
                    req,
                    ch,
                    heatmap,
                    ccg,
                    layer_count,
                );

                score += valve_score;

                if req.is_tunnel_escalated {
                    if ch.is_tunnel {
                        score -= 0.10;
                    } else {
                        score += 0.05;
                    }
                }

                if ch.group_size > 1 {
                    score -= (ch.group_size as f32 - 1.0) * 0.02;
                }

                Some((ch.id, score))
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    // -------------------------------------------------------------------------
    // DF‑HBM arbitration (Tesla Valve added)
    // -------------------------------------------------------------------------

    pub fn choose_best_channel_dfhbm(
        &self,
        req: &HbmRequest,
        channels: &[HbmChannel],
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        layer_count: usize,
    ) -> Option<usize> {
        let valve_score = 0.0;

        channels
            .par_iter()
            .filter_map(|ch| {
                let other_load = ch.pair_id.and_then(|pid| {
                    channels
                        .iter()
                        .find(|c| c.pair_id == Some(pid) && c.id != ch.id)
                        .map(|c| c.metrics.load)
                });

                if !ch.can_accept_with_pair(req.bank_id, other_load) {
                    return None;
                }

                let base_score = Self::arbitration_score_parallel(
                    req,
                    ch,
                    heatmap,
                    ccg,
                    layer_count,
                );

                let df_req = req.dfhbm_score() * 0.35;

                let df_ch =
                    ch.metrics.delta_load * 0.20 +
                    ch.metrics.delta_stability * 0.20 -
                    ch.metrics.delta_row_conflict * 0.15 -
                    ch.metrics.delta_bank_busy * 0.15 -
                    ch.metrics.delta_channel_sat * 0.10 +
                    ch.metrics.delta_refresh_pressure * 0.10 +
                    ch.metrics.delta_ecc_activity * 0.10;

                let score = base_score + df_req + df_ch + valve_score;

                Some((ch.id, score))
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    // -------------------------------------------------------------------------
    // DAX‑aware arbitration (views, rollback, deltas)
    // -------------------------------------------------------------------------

    pub fn choose_best_channel_with_dax_view(
        &self,
        req: &HbmRequest,
        channels: &[HbmChannel],
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        layer_count: usize,
        store: &DeltaStore,
        view: &EffectiveView,
    ) -> Option<usize> {
        channels
            .par_iter()
            .filter_map(|ch| {
                let mut ch_clone = ch.clone();
                ch_clone.apply_effective_view(view, store);

                let other_load = ch_clone.pair_id.and_then(|pid| {
                    channels
                        .iter()
                        .find(|c| c.pair_id == Some(pid) && c.id != ch_clone.id)
                        .map(|c| c.metrics.load)
                });

                if !ch_clone.can_accept_with_pair(req.bank_id, other_load) {
                    return None;
                }

                let mut score = Self::arbitration_score_parallel(
                    req,
                    &ch_clone,
                    heatmap,
                    ccg,
                    layer_count,
                );

                if req.is_tunnel_escalated {
                    if ch_clone.is_tunnel {
                        score -= 0.10;
                    } else {
                        score += 0.05;
                    }
                }

                if ch_clone.group_size > 1 {
                    score -= (ch_clone.group_size as f32 - 1.0) * 0.02;
                }

                Some((ch_clone.id, score))
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    pub fn choose_best_channel_with_rollback(
        &self,
        req: &HbmRequest,
        channels: &[HbmChannel],
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        layer_count: usize,
        store: &DeltaStore,
        master_id: usize,
        target_seq: u64,
    ) -> Option<usize> {
        channels
            .par_iter()
            .filter_map(|ch| {
                let mut ch_clone = ch.clone();
                ch_clone.rollback_to(master_id, store, target_seq);

                let other_load = ch_clone.pair_id.and_then(|pid| {
                    channels
                        .iter()
                        .find(|c| c.pair_id == Some(pid) && c.id != ch_clone.id)
                        .map(|c| c.metrics.load)
                });

                if !ch_clone.can_accept_with_pair(req.bank_id, other_load) {
                    return None;
                }

                let mut score = Self::arbitration_score_parallel(
                    req,
                    &ch_clone,
                    heatmap,
                    ccg,
                    layer_count,
                );

                if req.is_tunnel_escalated {
                    if ch_clone.is_tunnel {
                        score -= 0.10;
                    } else {
                        score += 0.05;
                    }
                }

                if ch_clone.group_size > 1 {
                    score -= (ch_clone.group_size as f32 - 1.0) * 0.02;
                }

                Some((ch_clone.id, score))
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    pub fn dax_delta_influence(
        &self,
        _req: &HbmRequest,
        _delta: &DeltaBuffer,
        _channel: &HbmChannel,
    ) -> f32 {
        // Hook for future fine‑grained delta‑based arbitration shaping
        0.0
    }
}


