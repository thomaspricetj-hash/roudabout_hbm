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

    // NEW: grid/topology‑aware locality score
    pub locality_score: f32,

    // NEW: tunneling fields
    pub is_tunnel: bool,
    pub tunnel_bias: f32,
    pub tunnel_reliability: f32,
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

            // tunneling defaults
            is_tunnel: false,
            tunnel_bias: 0.0,
            tunnel_reliability: 1.0,
        }
    }

    /// Attach tunneling characteristics to this channel (HBM tunnel path).
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

        // Initial tunnel bias: favor stable, low‑loss, low‑congestion tunnels
        self.tunnel_bias =
            (1.0 - loss_rate) * 0.20
            - (congestion * 0.10)
            + (stability * 0.20)
            - (latency_ms / 100.0) * 0.05
            - (jitter_ms / 100.0) * 0.05;
    }

    /// Update tunnel metrics over time (e.g., from telemetry).
    pub fn update_tunnel_metrics(
        &mut self,
        latency_ms: f32,
        jitter_ms: f32,
        loss_rate: f32,
        stability: f32,
        congestion: f32,
    ) {
        if !self.is_tunnel {
            return;
        }

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

    /// Reinforce tunnel reliability after successful routing.
    pub fn reinforce_tunnel(&mut self) {
        if !self.is_tunnel {
            return;
        }

        self.tunnel_reliability = (self.tunnel_reliability + 0.03).clamp(0.1, 2.0);
        self.metrics.tunnel_stability_score =
            (self.metrics.tunnel_stability_score + 0.03).clamp(0.1, 2.0);

        self.tunnel_bias += 0.02;
    }

    /// Cool tunnel bias after failure or congestion.
    pub fn cool_tunnel(&mut self) {
        if !self.is_tunnel {
            return;
        }

        self.tunnel_reliability = (self.tunnel_reliability - 0.03).clamp(0.1, 2.0);
        self.metrics.tunnel_stability_score =
            (self.metrics.tunnel_stability_score - 0.03).clamp(0.1, 2.0);

        self.tunnel_bias -= 0.02;
    }

    /// Original acceptance logic + tunnel‑aware constraints.
    pub fn can_accept(&self, bank_id: usize) -> bool {
        if self.metrics.load >= self.max_load {
            return false;
        }

        if let Some(bank) = self.banks.iter().find(|b| b.bank_id == bank_id) {
            if bank.busy {
                return false;
            }
        } else {
            return false;
        }

        // Tunnel‑specific acceptance rules
        if self.is_tunnel {
            if self.metrics.tunnel_congestion_level >= 0.95 {
                return false;
            }
            if self.metrics.tunnel_loss_rate >= 0.10 {
                return false;
            }
        }

        true
    }

    // -------------------------------------------------------------------------
    // MAX‑tier parallel upgrades
    // -------------------------------------------------------------------------

    /// Parallel bank busy scoring (used by arbitration + index)
    pub fn bank_busy_score_parallel(&self) -> f32 {
        self.banks
            .par_iter()
            .map(|b| if b.busy { 1.0 } else { 0.0 })
            .sum::<f32>()
    }

    /// Parallel open‑row affinity scoring
    pub fn open_row_affinity_parallel(&self, target_row: u64) -> f32 {
        self.banks
            .par_iter()
            .map(|b| {
                if let Some(open) = b.open_row {
                    if open == target_row {
                        0.0 // perfect row hit → no penalty
                    } else {
                        0.5 // row miss penalty
                    }
                } else {
                    0.25 // closed row penalty
                }
            })
            .sum::<f32>()
    }

    /// Parallel reliability update based on ECC + jitter + error rate
    pub fn update_reliability_parallel(&mut self) {
        let ecc = self.metrics.ecc_activity;
        let jitter = self.metrics.jitter_cycles;
        let err = self.metrics.error_rate;

        let drop = [ecc, jitter, err]
            .par_iter()
            .map(|v| v * 0.10)
            .sum::<f32>();

        self.reliability_score = (self.reliability_score - drop).clamp(0.1, 1.0);
    }

    /// Parallel heat affinity update based on heatmap layer values
    pub fn update_heat_affinity_parallel(&mut self, heat_layers: &[Vec<f32>]) {
        let affinity = heat_layers
            .par_iter()
            .map(|layer| layer.get(self.id).copied().unwrap_or(0.0))
            .sum::<f32>();

        self.heat_affinity = affinity;
    }

    /// Composite parallel channel score (used by controller + arbitration)
    pub fn composite_channel_score_parallel(
        &self,
        target_row: u64,
        heat_layers: &[Vec<f32>],
    ) -> f32 {
        let bank_busy = self.bank_busy_score_parallel();
        let row_affinity = self.open_row_affinity_parallel(target_row);

        let heat_affinity = heat_layers
            .par_iter()
            .map(|layer| layer.get(self.id).copied().unwrap_or(0.0))
            .sum::<f32>();

        let metrics_score =
            (self.metrics.load * 0.20)
            + (self.metrics.refresh_pressure * 0.30)
            + (self.metrics.ecc_activity * 0.25)
            + (self.metrics.jitter_cycles * 0.10)
            + ((1.0 - self.metrics.stability_score) * 0.20);

        // NEW: tunneling contribution
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
    }

    /// NEW: composite parallel channel score with Heatmap + CrossConnectGrid
    pub fn composite_channel_score_parallel_with_grid(
        &self,
        target_row: u64,
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        _layer_count: usize,
    ) -> f32 {
        let bank_busy = self.bank_busy_score_parallel();
        let row_affinity = self.open_row_affinity_parallel(target_row);

        // use full multilayer heatmap
        let heat_affinity = heatmap.layers
            .par_iter()
            .map(|layer| layer.get(self.id).copied().unwrap_or(0.0))
            .sum::<f32>();

        // metrics score as before
        let metrics_score =
            (self.metrics.load * 0.20)
            + (self.metrics.refresh_pressure * 0.30)
            + (self.metrics.ecc_activity * 0.25)
            + (self.metrics.jitter_cycles * 0.10)
            + ((1.0 - self.metrics.stability_score) * 0.20);

        // grid fused bias
        let grid_bias = ccg.fused_bias(self.id);

        // tunneling contribution
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
            - grid_bias
    }
}

