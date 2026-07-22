use rayon::prelude::*;
use std::time::Instant;

use super::{
    heatmap::Heatmap,
    index::RoutingIndex,
    request::HbmRequest,
    channel::HbmChannel,
};

#[derive(Debug, Clone)]
pub struct ChannelMetrics {
    pub load: f32,
    pub row_availability: f32,
    pub refresh_pressure: f32,
    pub ecc_activity: f32,

    pub jitter_cycles: f32,
    pub error_rate: f32,
    pub throughput_gbps: f32,
    pub stability_score: f32,
    pub last_refresh: Instant,
}

impl ChannelMetrics {
    pub fn new() -> Self {
        Self {
            load: 0.0,
            row_availability: 1.0,
            refresh_pressure: 0.0,
            ecc_activity: 0.0,
            jitter_cycles: 0.0,
            error_rate: 0.0,
            throughput_gbps: 0.0,
            stability_score: 1.0,
            last_refresh: Instant::now(),
        }
    }

    /// MAX‑tier: parallel multilayer metric scoring
    pub fn multilayer_score_parallel(
        &self,
        req: &HbmRequest,
        heatmap: &Heatmap,
        channel: &HbmChannel,
        layer_count: usize,
    ) -> f32 {
        // Compute per-layer contributions in parallel
        let layer_sum: f32 = (0..layer_count)
            .into_par_iter()
            .map(|layer| {
                // Heatmap contribution
                let heat_bias = heatmap.layers.get(layer).map(|layer_vec| {
                    if !layer_vec.is_empty() {
                        let avg_heat = layer_vec.iter().copied().sum::<f32>()
                            / layer_vec.len() as f32;
                        avg_heat * 0.10
                    } else {
                        0.0
                    }
                }).unwrap_or(0.0);

                // Index scoring contribution
                let idx_score = RoutingIndex::score_channel(req, channel, layer_count) * 0.02;

                // Request layer bias contribution
                let req_bias = req.layer_bias.get(layer).copied().unwrap_or(0.0);

                heat_bias + idx_score + req_bias
            })
            .sum();

        // Core channel metrics contribution
        let core = (self.load * 0.20)
            + (self.refresh_pressure * 0.30)
            + (self.ecc_activity * 0.25)
            + (self.jitter_cycles * 0.10)
            + (self.error_rate * 0.15)
            + ((1.0 - self.stability_score) * 0.20);

        core + layer_sum
    }

    /// MAX‑tier: adaptive stability update
    pub fn update_stability(&mut self, success: bool) {
        if success {
            self.stability_score = (self.stability_score + 0.03).min(2.0);
        } else {
            self.stability_score = (self.stability_score - 0.03).max(0.1);
        }
    }

    /// MAX‑tier: refresh pressure update
    pub fn update_refresh(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refresh).as_secs_f32();

        // More time since refresh → higher pressure
        self.refresh_pressure = (elapsed * 0.05).min(1.0);
    }

    /// MAX‑tier: ECC activity update
    pub fn update_ecc(&mut self, ecc_events: u32) {
        self.ecc_activity = (ecc_events as f32 * 0.02).min(1.0);
    }

    /// MAX‑tier: jitter update
    pub fn update_jitter(&mut self, jitter: f32) {
        self.jitter_cycles = jitter.clamp(0.0, 1.0);
    }

    /// MAX‑tier: throughput update
    pub fn update_throughput(&mut self, gbps: f32) {
        self.throughput_gbps = gbps.max(0.0);
    }
}
