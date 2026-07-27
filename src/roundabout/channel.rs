use rayon::prelude::*;

use super::{
    metrics::ChannelMetrics,
    heatmap::Heatmap,
    grid::CrossConnectGrid,
};

// BitDrop‑V2 integration
use bitdrop_v2::{
    compress_with_profile,
    estimate_entropy,
    looks_like_text_or_structured,
    looks_like_u32_counter,
    gpu_available,
};

#[derive(Debug, Clone)]
pub struct BankState {
    pub bank_id: usize,
    pub busy: bool,
    pub open_row: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct HbmChannel {
    pub id: usize,
    pub banks: Vec<BankState>,
    pub metrics: ChannelMetrics,
    pub max_load: f32,

    pub heat_affinity: f32,
    pub reliability_score: f32,
    pub locality_score: f32,

    // Tunneling fields
    pub is_tunnel: bool,
    pub tunnel_bias: f32,
    pub tunnel_reliability: f32,

    // Grouped‑pair / group fields
    pub pair_id: Option<usize>,
    pub is_pair_primary: bool,
    pub pair_affinity_score: f32,
    pub pair_load_bias: f32,

    // group size (2 = pair, 3 = triplet, 4 = quad)
    pub group_size: usize,

    // BitDrop‑aware channel hints
    pub payload_entropy_bias: f32,
    pub payload_size_bias: f32,
    pub payload_structure_bias: f32,
    pub payload_numeric_bias: f32,

    // NEW: structure‑aware channel metrics
    pub structured_lane_bias: f32,
    pub structured_tunnel_bias: f32,
    pub structured_geom_bias: f32,
    pub structured_load_bias: f32,
    pub structured_stability_bias: f32,
}

impl HbmChannel {
    pub fn new(id: usize, bank_count: usize, max_load: f32, layer_count: usize) -> Self {
        let banks = (0..bank_count)
            .map(|b| BankState {
                bank_id: b,
                busy: false,
                open_row: None,
            })
            .collect();

        Self {
            id,
            banks,
            metrics: ChannelMetrics::new(layer_count),
            max_load,
            heat_affinity: 0.0,
            reliability_score: 1.0,
            locality_score: 0.0,

            is_tunnel: false,
            tunnel_bias: 0.0,
            tunnel_reliability: 1.0,

            pair_id: None,
            is_pair_primary: false,
            pair_affinity_score: 0.0,
            pair_load_bias: 0.0,

            group_size: 1,

            payload_entropy_bias: 0.0,
            payload_size_bias: 0.0,
            payload_structure_bias: 0.0,
            payload_numeric_bias: 0.0,

            // NEW structure‑aware fields
            structured_lane_bias: 0.0,
            structured_tunnel_bias: 0.0,
            structured_geom_bias: 0.0,
            structured_load_bias: 0.0,
            structured_stability_bias: 0.0,
        }
    }

    // -------------------------------------------------------------------------
    // NEW: Structure‑aware scoring
    // -------------------------------------------------------------------------

    pub fn update_structure_biases(&mut self, req: &super::request::HbmRequest) {
        let is_structured = req.payload_is_structured;
        let is_numeric = req.payload_is_numeric_counter;
        let size = req.payload_compressed_size as f32;

        // Lane bias: structured payloads prefer stable, low‑heat channels
        self.structured_lane_bias =
            if is_structured {
                (self.metrics.stability_score * 0.10) - (self.metrics.load * 0.06)
            } else {
                0.0
            };

        // Numeric payloads prefer tunnels + low conflict
        self.structured_tunnel_bias =
            if is_numeric && self.is_tunnel {
                (self.metrics.stability_score * 0.12)
                    - (self.metrics.tunnel_congestion_level * 0.08)
            } else {
                0.0
            };

        // Geometry bias: structured payloads benefit from strong geom channels
        self.structured_geom_bias =
            if is_structured {
                self.metrics.geometry_score * 0.08
            } else {
                0.0
            };

        // Load bias: small structured payloads prefer lightly loaded channels
        let usage = self.metrics.load / self.max_load;
        self.structured_load_bias =
            if size < 256.0 {
                (1.0 - usage) * 0.06
            } else {
                0.0
            };

        // Stability bias: structured payloads prefer stable channels
        self.structured_stability_bias =
            if is_structured {
                self.metrics.stability_score * 0.10
            } else {
                0.0
            };
    }

    // -------------------------------------------------------------------------
    // BitDrop‑aware payload scoring (per‑channel)
    // -------------------------------------------------------------------------

    pub fn update_bitdrop_biases(&mut self, raw_payload: &[u8], profile_hint: Option<&str>) {
        let entropy = estimate_entropy(raw_payload);
        let is_structured = looks_like_text_or_structured(raw_payload);
        let is_numeric = looks_like_u32_counter(raw_payload);

        let profile = if let Some(p) = profile_hint {
            p
        } else if is_numeric {
            "numbin"
        } else if is_structured {
            "pymid"
        } else if gpu_available() {
            "adaptive"
        } else {
            "fast"
        };

        let compressed = compress_with_profile(raw_payload, profile);
        let size = compressed.len() as f32;

        self.payload_size_bias = (1_000_000.0 / size.max(64.0)).min(10.0);
        self.payload_entropy_bias = (8.0 - entropy).clamp(-4.0, 4.0);
        self.payload_structure_bias = if is_structured { 1.5 } else { 0.0 };
        self.payload_numeric_bias = if is_numeric { 2.0 } else { 0.0 };
    }

    // -------------------------------------------------------------------------
    // TUNNEL METHODS
    // -------------------------------------------------------------------------

    pub fn attach_tunnel(
        &mut self,
        latency_ms: f32,
        jitter_ms: f32,
        loss_rate: f32,
        stability: f32,
        congestion: f32,
    ) {
        self.is_tunnel = true;

        self.metrics.tunnel_latency_ms = latency_ms;
        self.metrics.tunnel_jitter_ms = jitter_ms;
        self.metrics.tunnel_loss_rate = loss_rate;
        self.metrics.tunnel_stability_score = stability;
        self.metrics.tunnel_congestion_level = congestion;

        self.tunnel_reliability = stability;

        self.tunnel_bias =
            (1.0 - loss_rate) * 0.20
            - (congestion * 0.10)
            + (stability * 0.20)
            - (latency_ms / 100.0) * 0.05
            - (jitter_ms / 100.0) * 0.05;
    }

    pub fn update_tunnel_metrics(
        &mut self,
        latency_ms: f32,
        jitter_ms: f32,
        loss_rate: f32,
        stability: f32,
        congestion: f32,
    ) {
        if !self.is_tunnel { return; }

        self.metrics.tunnel_latency_ms = latency_ms;
        self.metrics.tunnel_jitter_ms = jitter_ms;
        self.metrics.tunnel_loss_rate = loss_rate;
        self.metrics.tunnel_stability_score = stability;
        self.metrics.tunnel_congestion_level = congestion;

        self.tunnel_reliability = stability;

        self.tunnel_bias =
            (1.0 - loss_rate) * 0.20
            - (congestion * 0.10)
            + (stability * 0.20)
            - (latency_ms / 100.0) * 0.05
            - (jitter_ms / 100.0) * 0.05;
    }

    pub fn reinforce_tunnel(&mut self) {
        if !self.is_tunnel { return; }

        self.tunnel_reliability = (self.tunnel_reliability + 0.03).clamp(0.1, 2.0);
        self.metrics.tunnel_stability_score =
            (self.metrics.tunnel_stability_score + 0.03).clamp(0.1, 2.0);

        self.tunnel_bias += 0.02;
    }

    pub fn cool_tunnel(&mut self) {
        if !self.is_tunnel { return; }

        self.tunnel_reliability = (self.tunnel_reliability - 0.03).clamp(0.1, 2.0);
        self.metrics.tunnel_stability_score =
            (self.metrics.tunnel_stability_score - 0.03).clamp(0.1, 2.0);

        self.tunnel_bias -= 0.02;
    }

    // -------------------------------------------------------------------------
    // GROUPED PAIRS / TRIPLETS / QUADS
    // -------------------------------------------------------------------------

    pub fn attach_pair(&mut self, pair_id: usize, is_primary: bool) {
        self.pair_id = Some(pair_id);
        self.is_pair_primary = is_primary;
        self.pair_affinity_score = 0.0;
        self.pair_load_bias = if is_primary { -0.05 } else { 0.0 };
        self.group_size = 2;
    }

    pub fn attach_group(&mut self, group_id: usize, group_size: usize, is_primary: bool) {
        self.pair_id = Some(group_id);
        self.is_pair_primary = is_primary;
        self.pair_affinity_score = 0.0;
        self.pair_load_bias = if is_primary { -0.05 } else { 0.0 };
        self.group_size = group_size;
    }

    pub fn clear_group(&mut self) {
        self.pair_id = None;
        self.is_pair_primary = false;
        self.pair_affinity_score = 0.0;
        self.pair_load_bias = 0.0;
        self.group_size = 1;
    }

    pub fn update_pair_affinity(&mut self, other_load: f32) {
        let load_delta = self.metrics.load - other_load;

        self.pair_affinity_score =
            (-load_delta * 0.20) + (self.metrics.stability_score * 0.15);

        self.pair_load_bias =
            (self.pair_load_bias + (-load_delta * 0.05)).clamp(-0.20, 0.20);
    }

    pub fn maybe_switch_primary(&mut self, other_load: f32) {
        let load_delta = self.metrics.load - other_load;

        if !self.is_pair_primary && load_delta < -0.30 {
            self.is_pair_primary = true;
            self.pair_load_bias = -0.05;
        }

        if self.is_pair_primary && load_delta > 0.30 {
            self.is_pair_primary = false;
            self.pair_load_bias = 0.0;
        }
    }

    pub fn tunnel_pair_component(&self) -> f32 {
        if !self.is_tunnel || self.pair_id.is_none() {
            return 0.0;
        }

        let base =
            (1.0 - self.metrics.tunnel_loss_rate) * 0.10
            + (self.metrics.tunnel_stability_score * 0.10)
            - (self.metrics.tunnel_congestion_level * 0.05);

        let primary_bonus = if self.is_pair_primary { 0.03 } else { 0.0 };

        base + primary_bonus
    }

    pub fn pair_score_component(&self) -> f32 {
        if self.pair_id.is_none() {
            return 0.0;
        }

        let primary_bias = if self.is_pair_primary { -0.05 } else { 0.0 };

        self.pair_affinity_score
            + self.pair_load_bias
            + primary_bias
            + self.tunnel_pair_component()
    }

    // -------------------------------------------------------------------------
    // ACCEPTANCE LOGIC (structure‑aware)
    // -------------------------------------------------------------------------

    pub fn can_accept(&self, bank_id: usize) -> bool {
        if self.metrics.load >= self.max_load { return false; }

        if let Some(bank) = self.banks.iter().find(|b| b.bank_id == bank_id) {
            if bank.busy { return false; }
        } else {
            return false;
        }

        if self.is_tunnel {
            if self.metrics.tunnel_congestion_level >= 0.95 { return false; }
            if self.metrics.tunnel_loss_rate >= 0.10 { return false; }
        }

        // NEW: structure‑aware acceptance
        if self.structured_load_bias < -0.10 {
            return false;
        }

        true
    }

    pub fn can_accept_with_pair(&self, bank_id: usize, other_load: Option<f32>) -> bool {
        if !self.can_accept(bank_id) { return false; }

        if let Some(other) = other_load {
            let load_delta = self.metrics.load - other;
            if load_delta > 0.25 { return false; }
        }

        true
    }

    // -------------------------------------------------------------------------
    // PARALLEL SCORING (structure‑aware)
    // -------------------------------------------------------------------------

    pub fn bank_busy_score_parallel(&self) -> f32 {
        self.banks.par_iter().map(|b| if b.busy { 1.0 } else { 0.0 }).sum()
    }

    pub fn open_row_affinity_parallel(&self, target_row: u64) -> f32 {
        self.banks.par_iter().map(|b| {
            if let Some(open) = b.open_row {
                if open == target_row { 0.0 } else { 0.5 }
            } else {
                0.25
            }
        }).sum()
    }

    pub fn update_reliability_parallel(&mut self) {
        let ecc = self.metrics.ecc_activity;
        let jitter = self.metrics.jitter_cycles;
        let err = self.metrics.error_rate;

        let drop = [ecc, jitter, err].par_iter().map(|v| v * 0.10).sum::<f32>();

        self.reliability_score = (self.reliability_score - drop).clamp(0.1, 1.0);
    }

    pub fn update_heat_affinity_parallel(&mut self, heat_layers: &[Vec<f32>]) {
        let affinity = heat_layers.par_iter()
            .map(|layer| layer.get(self.id).copied().unwrap_or(0.0))
            .sum::<f32>();

        self.heat_affinity = affinity;
    }

    pub fn composite_channel_score_parallel(
        &self,
        target_row: u64,
        heat_layers: &[Vec<f32>],
    ) -> f32 {
        let bank_busy = self.bank_busy_score_parallel();
        let row_affinity = self.open_row_affinity_parallel(target_row);

        let heat_affinity = heat_layers.par_iter()
            .map(|layer| layer.get(self.id).copied().unwrap_or(0.0))
            .sum::<f32>();

        let metrics_score =
            (self.metrics.load * 0.20)
            + (self.metrics.refresh_pressure * 0.30)
            + (self.metrics.ecc_activity * 0.25)
            + (self.metrics.jitter_cycles * 0.10)
            + ((1.0 - self.metrics.stability_score) * 0.20);

        let tunnel_score = if self.is_tunnel {
            let mut score = 0.0;

            score += (self.metrics.tunnel_latency_ms / 100.0) * 0.10;
            score += (self.metrics.tunnel_jitter_ms / 100.0) * 0.10;
            score += self.metrics.tunnel_congestion_level * 0.15;
            score += (1.0 - self.metrics.tunnel_stability_score) * 0.20;
            score -= (1.0 - self.metrics.tunnel_loss_rate) * 0.10;

            score += self.tunnel_bias * 0.10;
            score += (1.0 - self.tunnel_reliability) * 0.15;

            score
        } else {
            0.0
        };

        bank_busy
            + row_affinity
            + heat_affinity
            + metrics_score
            + tunnel_score
            + self.pair_score_component()
            + self.payload_size_bias * 0.05
            + self.payload_entropy_bias * 0.03
            + self.payload_structure_bias * 0.02
            + self.payload_numeric_bias * 0.02
            + self.structured_lane_bias * 0.08
            + self.structured_tunnel_bias * 0.09
            + self.structured_geom_bias * 0.07
            + self.structured_load_bias * 0.06
            + self.structured_stability_bias * 0.08
    }

    pub fn composite_channel_score_parallel_with_grid(
        &self,
        target_row: u64,
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        _layer_count: usize,
    ) -> f32 {
        let bank_busy = self.bank_busy_score_parallel();
        let row_affinity = self.open_row_affinity_parallel(target_row);

        let heat_affinity = heatmap.layers.par_iter()
            .map(|layer| layer.get(self.id).copied().unwrap_or(0.0))
            .sum::<f32>();

        let metrics_score =
            (self.metrics.load * 0.20)
            + (self.metrics.refresh_pressure * 0.30)
            + (self.metrics.ecc_activity * 0.25)
            + (self.metrics.jitter_cycles * 0.10)
            + ((1.0 - self.metrics.stability_score) * 0.20);

        let grid_bias = ccg.fused_bias(self.id);

        let tunnel_score = if self.is_tunnel {
            let mut score = 0.0;

            score += (self.metrics.tunnel_latency_ms / 100.0) * 0.10;
            score += (self.metrics.tunnel_jitter_ms / 100.0) * 0.10;
            score += self.metrics.tunnel_congestion_level * 0.15;
            score += (1.0 - self.metrics.tunnel_stability_score) * 0.20;
            score -= (1.0 - self.metrics.tunnel_loss_rate) * 0.10;

            score += self.tunnel_bias * 0.10;
            score += (1.0 - self.tunnel_reliability) * 0.15;

            score
        } else {
            0.0
        };

        bank_busy
            + row_affinity
            + heat_affinity
            + metrics_score
            + tunnel_score
            + self.pair_score_component()
            - grid_bias
            + self.payload_size_bias * 0.05
            + self.payload_entropy_bias * 0.03
            + self.payload_structure_bias * 0.02
            + self.payload_numeric_bias * 0.02
            + self.structured_lane_bias * 0.08
            + self.structured_tunnel_bias * 0.09
            + self.structured_geom_bias * 0.07
            + self.structured_load_bias * 0.06
            + self.structured_stability_bias * 0.08
    }
}
