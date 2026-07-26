use rayon::prelude::*;

use super::{
    request::{HbmRequest, RequestPriority},
    channel::HbmChannel,
    heatmap::Heatmap,
    index::RoutingIndex,
    grid::CrossConnectGrid,
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

    /// NEW: HBM‑aware priority escalation
    pub fn hbm_priority_escalation(req: &HbmRequest) -> f32 {
        let mut esc = 0.0;

        // row / bank locality escalation
        if req.locality_score > 0.5 {
            esc -= 0.10;
        }

        // refresh / ECC pressure escalation
        if req.refresh_pressure > 0.5 {
            esc -= 0.05;
        }
        if req.ecc_pressure > 0.5 {
            esc -= 0.05;
        }

        esc
    }

    // -------------------------------------------------------------------------
    // MAX‑tier parallel arbitration upgrades
    // -------------------------------------------------------------------------

    /// Parallel multilayer arbitration score for a single channel
    pub fn arbitration_score_parallel(
        req: &HbmRequest,
        channel: &HbmChannel,
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        layer_count: usize,
    ) -> f32 {
        // Priority weight
        let priority = Self::priority_weight(req) + Self::hbm_priority_escalation(req);

        // Parallel multilayer index scoring (heat + grid + metrics)
        let index_score = RoutingIndex::score_channel_parallel_with_grid(
            req,
            channel,
            heatmap,
            ccg,
            layer_count,
        );

        // Channel‑metric contribution (legacy + multilayer)
        let metrics_score =
            (channel.metrics.load * 0.20)
            + (channel.metrics.refresh_pressure * 0.30)
            + (channel.metrics.ecc_activity * 0.25)
            + (channel.metrics.jitter_cycles * 0.10)
            + ((1.0 - channel.metrics.stability_score) * 0.20);

        // Bank busy contribution (parallel)
        let bank_busy_score = channel.bank_busy_score_parallel();

        // Heat affinity contribution (parallel)
        let heat_affinity = heatmap.layers
            .par_iter()
            .map(|layer| layer.get(channel.id).copied().unwrap_or(0.0))
            .sum::<f32>();

        // NEW: HBM locality contribution
        let locality =
            heatmap.row_conflict[channel.id] * 0.40 +
            heatmap.bank_busy[channel.id] * 0.35 +
            heatmap.channel_sat[channel.id] * 0.25;

        // NEW: refresh/ECC penalties
        let penalties =
            -heatmap.refresh_heat[channel.id] * 0.30 -
            -heatmap.ecc_heat[channel.id] * 0.25;

        // Grid fused bias (cluster + zone + door + geom + locality)
        let grid_bias = ccg.fused_bias(channel.id);

        // NEW: grouped‑pair contribution
        let pair_component = channel.pair_score_component();

        // Composite arbitration score
        priority
            + index_score
            + metrics_score
            + bank_busy_score
            + heat_affinity
            + locality
            + penalties
            + pair_component
            - grid_bias
    }

    /// Parallel arbitration across all channels
    pub fn choose_best_channel_parallel(
        &self,
        req: &HbmRequest,
        channels: &[HbmChannel],
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        layer_count: usize,
    ) -> Option<usize> {
        channels
            .par_iter()
            .filter_map(|ch| {
                // find paired channel load if this channel is in a pair
                let other_load = ch.pair_id.and_then(|pid| {
                    channels
                        .iter()
                        .find(|c| c.pair_id == Some(pid) && c.id != ch.id)
                        .map(|c| c.metrics.load)
                });

                // pair‑aware acceptance
                if !ch.can_accept_with_pair(req.bank_id, other_load) {
                    return None;
                }

                let score = Self::arbitration_score_parallel(
                    req,
                    ch,
                    heatmap,
                    ccg,
                    layer_count,
                );

                Some((ch.id, score))
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }
}
