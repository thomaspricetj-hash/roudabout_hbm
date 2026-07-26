use rayon::prelude::*;

use super::{
    metrics::ChannelMetrics,
    heatmap::Heatmap,
    grid::CrossConnectGrid,
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
        }
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

    /// Pair imbalance correction: adjust affinity/load bias based on peer load.
    pub fn update_pair_affinity(&mut self, other_load: f32) {
        let load_delta = self.metrics.load - other_load;

        self.pair_affinity_score =
            (-load_delta * 0.20) + (self.metrics.stability_score * 0.15);

        self.pair_load_bias =
            (self.pair_load_bias + (-load_delta * 0.05)).clamp(-0.20, 0.20);
    }

    /// Dynamic pair switching: flip primary/secondary when imbalance is too high.
    pub fn maybe_switch_primary(&mut self, other_load: f32) {
        let load_delta = self.metrics.load - other_load;

        // If secondary is much lighter, let it become primary.
        if !self.is_pair_primary && load_delta < -0.30 {
            self.is_pair_primary = true;
            self.pair_load_bias = -0.05;
        }

        // If primary is much heavier, demote it.
        if self.is_pair_primary && load_delta > 0.30 {
            self.is_pair_primary = false;
            self.pair_load_bias = 0.0;
        }
    }

    /// Pair‑aware tunnel routing: extra bias when group contains tunnels.
    pub fn tunnel_pair_component(&self) -> f32 {
        if !self.is_tunnel || self.pair_id.is_none() {
            return 0.0;
        }

        // Favor stable, low‑loss tunnels inside a group.
        let base =
            (1.0 - self.metrics.tunnel_loss_rate) * 0.10
            + (self.metrics.tunnel_stability_score * 0.10)
            - (self.metrics.tunnel_congestion_level * 0.05);

        // Primary tunnel in a group gets a small extra bias.
        let primary_bonus = if self.is_pair_primary { 0.03 } else { 0.0 };

        base + primary_bonus
    }

    pub fn pair_score_component(&self) -> f32 {
        if self.pair_id.is_none() {
            return 0.0;
        }

        let primary_bias = if self.is_pair_primary { -0.05 } else { 0.0 };

        self.pair_affinity_score + self.pair_load_bias + primary_bias + self.tunnel_pair_component()
    }

    // -------------------------------------------------------------------------
    // ACCEPTANCE LOGIC
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
    // PARALLEL SCORING
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

        bank_busy + row_affinity + heat_affinity + metrics_score + tunnel_score
            + self.pair_score_component()
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
    }
}

