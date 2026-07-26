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

    // NEW: tunneling metrics
    pub tunnel_latency_ms: f32,
    pub tunnel_jitter_ms: f32,
    pub tunnel_loss_rate: f32,
    pub tunnel_stability_score: f32,
    pub tunnel_congestion_level: f32,
    pub tunnel_bias: f32,

    // NEW: HBM-specific counters
    pub row_conflicts: u32,
    pub bank_busy_events: u32,
    pub channel_saturation_events: u32,
    pub refresh_events: u32,
    pub ecc_events: u32,

    // NEW: locality reliability
    pub locality_score: f32,
    pub geometry_score: f32,

    // NEW: grouped‑pair metrics
    pub pair_successes: u32,
    pub pair_failures: u32,
    pub pair_imbalance: f32,
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

            tunnel_latency_ms: 0.0,
            tunnel_jitter_ms: 0.0,
            tunnel_loss_rate: 0.0,
            tunnel_stability_score: 1.0,
            tunnel_congestion_level: 0.0,
            tunnel_bias: 0.0,

            row_conflicts: 0,
            bank_busy_events: 0,
            channel_saturation_events: 0,
            refresh_events: 0,
            ecc_events: 0,

            locality_score: 0.0,
            geometry_score: 0.0,

            pair_successes: 0,
            pair_failures: 0,
            pair_imbalance: 0.0,
        }
    }

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
                let base =
                    (self.layer_load[layer] * 0.20) +
                    (self.layer_refresh[layer] * 0.30) +
                    (self.layer_jitter[layer] * 0.10) +
                    ((1.0 - self.layer_stability[layer]) * 0.20);

                let req_bias = req.layer_bias.get(layer).copied().unwrap_or(0.0);

                let heat = heatmap.layers[layer][channel.id];

                let grid_bias =
                    0.35 * ccg.cluster_bias[layer][channel.id] +
                    0.25 * ccg.zone_bias[layer][channel.id] +
                    0.20 * ccg.door_bias[layer][channel.id] +
                    0.20 * ccg.geom_bias[layer][channel.id];

                let scratch = self.scratch[layer];

                let door_rot = ccg.door_rotation[layer][channel.id] as f32 * 0.01;

                let tunnel_score = if channel.is_tunnel {
                    let mut score = 0.0;

                    score += (self.tunnel_latency_ms / 100.0) * 0.10;
                    score += (self.tunnel_jitter_ms / 100.0) * 0.10;
                    score += self.tunnel_congestion_level * 0.15;
                    score += (1.0 - self.tunnel_stability_score) * 0.20;
                    score -= (1.0 - self.tunnel_loss_rate) * 0.10;
                    score += self.tunnel_bias * 0.10;

                    score
                } else {
                    0.0
                };

                let locality =
                    heatmap.row_conflict[channel.id] * 0.40 +
                    heatmap.bank_busy[channel.id] * 0.35 +
                    heatmap.channel_sat[channel.id] * 0.25;

                let penalties =
                    -heatmap.refresh_heat[channel.id] * 0.30 -
                    heatmap.ecc_heat[channel.id] * 0.25;

                base
                    + req_bias
                    + heat
                    + scratch
                    + door_rot
                    - grid_bias
                    + tunnel_score
                    + locality
                    + penalties
            })
            .sum()
    }

    pub fn update_stability(&mut self, success: bool) {
        let delta = if success { 0.03 } else { -0.03 };

        self.stability_score = (self.stability_score + delta).clamp(0.1, 2.0);

        self.layer_stability
            .iter_mut()
            .for_each(|s| *s = (*s + delta).clamp(0.1, 2.0));

        self.tunnel_stability_score = (self.tunnel_stability_score + delta).clamp(0.1, 2.0);

        let success_factor = if success { 0.05 } else { -0.05 };
        self.locality_score = (self.locality_score + success_factor).clamp(-1.0, 1.0);
    }

    pub fn update_refresh(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refresh).as_secs_f32();

        let pressure = (elapsed * 0.05).min(1.0);

        self.refresh_pressure = pressure;

        self.layer_refresh
            .iter_mut()
            .for_each(|r| *r = pressure);

        self.refresh_events = self.refresh_events.saturating_add(1);
    }

    pub fn update_ecc(&mut self, ecc_events: u32) {
        let ecc = (ecc_events as f32 * 0.02).min(1.0);

        self.ecc_activity = ecc;

        self.layer_load
            .iter_mut()
            .for_each(|l| *l += ecc * 0.01);

        self.ecc_events = self.ecc_events.saturating_add(ecc_events);
    }

    pub fn update_jitter(&mut self, jitter: f32) {
        let j = jitter.clamp(0.0, 1.0);

        self.jitter_cycles = j;

        self.layer_jitter
            .iter_mut()
            .for_each(|v| *v = j);

        self.tunnel_jitter_ms = j * 100.0;
    }

    pub fn update_throughput(&mut self, gbps: f32) {
        let t = gbps.max(0.0);

        self.throughput_gbps = t;

        self.layer_load
            .iter_mut()
            .for_each(|l| *l += t * 0.01);

        self.tunnel_congestion_level = (t * 0.02).min(1.0);

        self.channel_saturation_events = self.channel_saturation_events.saturating_add(1);
    }

    pub fn update_row_conflict(&mut self, conflicts: u32) {
        self.row_conflicts = self.row_conflicts.saturating_add(conflicts);
        self.row_availability = (self.row_availability - conflicts as f32 * 0.01).max(0.0);
    }

    pub fn update_bank_busy(&mut self, busy_events: u32) {
        self.bank_busy_events = self.bank_busy_events.saturating_add(busy_events);
        self.load += busy_events as f32 * 0.01;
    }

    pub fn update_geometry_reliability(&mut self, ccg: &CrossConnectGrid, channel_id: usize) {
        let fused = ccg.fused_bias(channel_id);
        self.geometry_score = fused.clamp(-2.0, 2.0);
    }

    /// NEW: grouped‑pair metrics update
    pub fn update_pair_metrics(&mut self, success: bool, load_delta: f32) {
        if success {
            self.pair_successes = self.pair_successes.saturating_add(1);
        } else {
            self.pair_failures = self.pair_failures.saturating_add(1);
        }

        self.pair_imbalance =
            (self.pair_imbalance + load_delta * 0.05).clamp(-1.0, 1.0);
    }
}
