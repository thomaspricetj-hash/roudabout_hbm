use rayon::prelude::*;
use std::time::Instant;

use super::{
    heatmap::Heatmap,
    request::HbmRequest,
    channel::HbmChannel,
    grid::CrossConnectGrid,
};


#[derive(Debug, Clone)]
pub struct ChannelMetrics {
    // Core metrics (legacy)
    pub load: f32,
    pub row_availability: f32,
    pub refresh_pressure: f32,
    pub ecc_activity: f32,
    pub jitter_cycles: f32,
    pub error_rate: f32,
    pub throughput_gbps: f32,
    pub stability_score: f32,
    pub last_refresh: Instant,

    // NEW: multilayer metric arrays
    pub layer_load: Vec<f32>,
    pub layer_refresh: Vec<f32>,
    pub layer_jitter: Vec<f32>,
    pub layer_stability: Vec<f32>,

    // NEW: multilayer scratchpad
    pub scratch: Vec<f32>,
}

impl ChannelMetrics {
    pub fn new(layer_count: usize) -> Self {
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

            layer_load: vec![0.0; layer_count],
            layer_refresh: vec![0.0; layer_count],
            layer_jitter: vec![0.0; layer_count],
            layer_stability: vec![1.0; layer_count],

            scratch: vec![0.0; layer_count],
        }
    }

    /// MAX‑tier: multilayer parallel metric scoring (Heatmap + Grid + Metrics)
    pub fn multilayer_score_parallel(
        &self,
        req: &HbmRequest,
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        channel: &HbmChannel,
        layer_count: usize,
    ) -> f32 {
        (0..layer_count)
            .into_par_iter()
            .map(|layer| {
                // Base metric per layer
                let base =
                    (self.layer_load[layer] * 0.20) +
                    (self.layer_refresh[layer] * 0.30) +
                    (self.layer_jitter[layer] * 0.10) +
                    ((1.0 - self.layer_stability[layer]) * 0.20);

                // Request bias
                let req_bias = req.layer_bias.get(layer).copied().unwrap_or(0.0);

                // Heatmap contribution
                let heat = heatmap.layers[layer][channel.id];

                // Grid contribution
                let grid_bias =
                    0.35 * ccg.cluster_bias[layer][channel.id] +
                    0.25 * ccg.zone_bias[layer][channel.id] +
                    0.20 * ccg.door_bias[layer][channel.id] +
                    0.20 * ccg.geom_bias[layer][channel.id];

                // Scratchpad contribution
                let scratch = self.scratch[layer];

                // Rotating door bias
                let door_rot = ccg.door_rotation[layer][channel.id] as f32 * 0.01;

                base + req_bias + heat + scratch + door_rot - grid_bias
            })
            .sum()
    }

    /// MAX‑tier: adaptive stability update (multilayer)
    pub fn update_stability(&mut self, success: bool) {
        let delta = if success { 0.03 } else { -0.03 };

        self.stability_score = (self.stability_score + delta).clamp(0.1, 2.0);

        self.layer_stability
            .iter_mut()
            .for_each(|s| *s = (*s + delta).clamp(0.1, 2.0));
    }

    /// MAX‑tier: refresh pressure update (multilayer)
    pub fn update_refresh(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refresh).as_secs_f32();

        let pressure = (elapsed * 0.05).min(1.0);

        self.refresh_pressure = pressure;

        self.layer_refresh
            .iter_mut()
            .for_each(|r| *r = pressure);
    }

    /// MAX‑tier: ECC activity update (multilayer)
    pub fn update_ecc(&mut self, ecc_events: u32) {
        let ecc = (ecc_events as f32 * 0.02).min(1.0);

        self.ecc_activity = ecc;

        self.layer_load
            .iter_mut()
            .for_each(|l| *l += ecc * 0.01);
    }

    /// MAX‑tier: jitter update (multilayer)
    pub fn update_jitter(&mut self, jitter: f32) {
        let j = jitter.clamp(0.0, 1.0);

        self.jitter_cycles = j;

        self.layer_jitter
            .iter_mut()
            .for_each(|v| *v = j);
    }

    /// MAX‑tier: throughput update (multilayer)
    pub fn update_throughput(&mut self, gbps: f32) {
        let t = gbps.max(0.0);

        self.throughput_gbps = t;

        self.layer_load
            .iter_mut()
            .for_each(|l| *l += t * 0.01);
    }
}
