use rayon::prelude::*;
use super::metrics::ChannelMetrics;

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
}

impl HbmChannel {
    pub fn new(id: usize, bank_count: usize, max_load: f32) -> Self {
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
            metrics: ChannelMetrics::new(),
            max_load,
            heat_affinity: 0.0,
            reliability_score: 1.0,
        }
    }

    /// Original acceptance logic (kept intact)
    pub fn can_accept(&self, bank_id: usize) -> bool {
        if self.metrics.load >= self.max_load {
            return false;
        }
        if let Some(bank) = self.banks.iter().find(|b| b.bank_id == bank_id) {
            !bank.busy
        } else {
            false
        }
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

        // Compute reliability drop in parallel
        let drop = [ecc, jitter, err]
            .par_iter()
            .map(|v| v * 0.10)
            .sum::<f32>();

        self.reliability_score = (self.reliability_score - drop).clamp(0.1, 1.0);
    }

    /// Parallel heat affinity update based on heatmap layer values
    pub fn update_heat_affinity_parallel(&mut self, heat_layers: &[Vec<f32>]) {
        // Average heat across all layers for this channel
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

        bank_busy + row_affinity + heat_affinity + metrics_score
    }
}
