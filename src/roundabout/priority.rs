use rayon::prelude::*;

use super::{
    request::{HbmRequest, RequestPriority},
    channel::HbmChannel,
    metrics::ChannelMetrics,
    heatmap::Heatmap,
    index::RoutingIndex,
    grid::CrossConnectGrid,
};

/// PriorityEngine computes multilayer priority weights for HBM requests.
/// MAX‑tier version: parallel multilayer scoring, heatmap bias, grid bias, index bias.
#[derive(Debug, Default)]
pub struct PriorityEngine;

impl PriorityEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Base priority weight (lower = higher priority)
    pub fn base_priority_weight(priority: RequestPriority) -> f32 {
        match priority {
            RequestPriority::High => 0.25,
            RequestPriority::Standard => 1.0,
            RequestPriority::Low => 1.75,
        }
    }

    /// Multilayer priority bias based on request state
    pub fn multilayer_request_bias(req: &HbmRequest, layer: usize) -> f32 {
        let heat = req.layer_heat.get(layer).copied().unwrap_or(0.0);
        let score = req.layer_scores.get(layer).copied().unwrap_or(0.0);
        let bias = req.layer_bias.get(layer).copied().unwrap_or(0.0);

        (heat * 0.15) + (score * 0.10) + bias
    }

    /// Channel‑aware priority bias (HBM‑specific)
    pub fn channel_bias(metrics: &ChannelMetrics) -> f32 {
        let load = metrics.load;
        let refresh = metrics.refresh_pressure;
        let ecc = metrics.ecc_activity;
        let jitter = metrics.jitter_cycles;

        (load * 0.20) + (refresh * 0.30) + (ecc * 0.25) + (jitter * 0.10)
    }

    /// Bank‑busy priority bias
    pub fn bank_bias(channel: &HbmChannel, bank_id: usize) -> f32 {
        if let Some(bank) = channel.banks.iter().find(|b| b.bank_id == bank_id) {
            if bank.busy {
                return 1.0;
            }
        }
        0.0
    }

    /// MAX‑tier parallel multilayer priority scoring (heat + grid + index + metrics)
    pub fn composite_priority_parallel(
        req: &HbmRequest,
        channel: &HbmChannel,
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        layer_count: usize,
    ) -> f32 {
        let base = Self::base_priority_weight(req.priority);

        // Compute multilayer bias in parallel
        let layer_bias_sum: f32 = (0..layer_count)
            .into_par_iter()
            .map(|layer| {
                // Request bias
                let req_bias = Self::multilayer_request_bias(req, layer);

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

                // Grid bias (cluster + zone + door + geom)
                let grid_bias =
                    0.35 * ccg.cluster_bias[layer][channel.id] +
                    0.25 * ccg.zone_bias[layer][channel.id] +
                    0.20 * ccg.door_bias[layer][channel.id] +
                    0.20 * ccg.geom_bias[layer][channel.id];

                // Index bias (multilayer fused index scoring)
                let idx_bias = RoutingIndex::score_channel_parallel_with_grid(
                    req,
                    channel,
                    heatmap,
                    ccg,
                    layer_count,
                ) * 0.02;

                req_bias + heat_bias - grid_bias + idx_bias
            })
            .sum();

        // Channel‑level bias
        let channel_bias = Self::channel_bias(&channel.metrics);

        // Bank‑busy bias
        let bank_bias = Self::bank_bias(channel, req.bank_id);

        // Composite score
        let mut score = base + layer_bias_sum + channel_bias + bank_bias;

        // Reinforcement learning: stable requests get lower priority weight
        score *= req.stability_factor;

        score
    }

    /// Original composite priority (kept for compatibility)
    pub fn composite_priority(
        req: &HbmRequest,
        channel: &HbmChannel,
        layer_count: usize,
    ) -> f32 {
        let mut score = Self::base_priority_weight(req.priority);

        for layer in 0..layer_count {
            score += Self::multilayer_request_bias(req, layer);
        }

        score += Self::channel_bias(&channel.metrics);
        score += Self::bank_bias(channel, req.bank_id);

        score *= req.stability_factor;

        score
    }

    /// Priority escalation logic
    pub fn escalate(req: &mut HbmRequest) {
        req.priority = match req.priority {
            RequestPriority::High => RequestPriority::High,
            RequestPriority::Standard => RequestPriority::High,
            RequestPriority::Low => RequestPriority::Standard,
        };

        req.escalations += 1;
        req.adaptive_weight *= 1.15;
        req.stability_factor *= 0.95;
    }
}

