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
        let priority = Self::priority_weight(req);

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

        // Grid fused bias (cluster + zone + door + geom)
        let grid_bias = ccg.fused_bias(channel.id);

        // Composite arbitration score
        priority + index_score + metrics_score + bank_busy_score + heat_affinity - grid_bias
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
                if !ch.can_accept(req.bank_id) {
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

