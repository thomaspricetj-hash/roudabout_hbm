use rayon::prelude::*;

use super::{
    request::HbmRequest,
    heatmap::Heatmap,
    index::RoutingIndex,
    channel::HbmChannel,
    grid::CrossConnectGrid,
};

/// Multilayer scratchpad reinforcement memory.
/// Tracks per-layer routing history, failures, and adaptive bias.
#[derive(Debug, Clone)]
pub struct Scratchpad {
    pub layers: usize,
    pub history: Vec<Vec<Option<usize>>>, // [layer][recent exit]
    pub failures: Vec<u32>,               // per-layer failure counters

    // NEW: HBM locality + event memory
    pub last_row: Vec<Option<u32>>,       // last row accessed per layer
    pub last_bank: Vec<Option<u32>>,      // last bank accessed per layer
    pub last_channel: Vec<Option<usize>>, // last channel per layer

    pub refresh_events: Vec<u32>,         // refresh storm counters per layer
    pub ecc_events: Vec<u32>,             // ECC correction counters per layer

    pub success_reinforce: Vec<f32>,      // success reinforcement per layer
    pub failure_penalty: Vec<f32>,        // failure penalty per layer

    // NEW: BitDrop temporal geometry memory
    pub entropy_memory: Vec<f32>,         // entropy-driven temporal bias
    pub size_memory: Vec<f32>,            // compressed-size temporal bias
    pub structure_memory: Vec<f32>,       // structure-driven temporal bias
    pub numeric_memory: Vec<f32>,         // numeric-counter locality memory
    pub tunnel_memory: Vec<f32>,          // tunnel physics temporal bias

    pub adaptive_memory: Vec<f32>,        // adaptive-weight temporal bias
    pub stability_memory: Vec<f32>,       // stability-factor temporal penalty
}

impl Scratchpad {
    pub fn new(layers: usize) -> Self {
        Self {
            layers,
            history: vec![vec![None; 8]; layers],
            failures: vec![0; layers],

            last_row: vec![None; layers],
            last_bank: vec![None; layers],
            last_channel: vec![None; layers],

            refresh_events: vec![0; layers],
            ecc_events: vec![0; layers],

            success_reinforce: vec![0.0; layers],
            failure_penalty: vec![0.0; layers],

            // NEW: BitDrop temporal geometry memory
            entropy_memory: vec![0.0; layers],
            size_memory: vec![0.0; layers],
            structure_memory: vec![0.0; layers],
            numeric_memory: vec![0.0; layers],
            tunnel_memory: vec![0.0; layers],

            adaptive_memory: vec![0.0; layers],
            stability_memory: vec![0.0; layers],
        }
    }

    /// Record a successful exit for reinforcement.
    pub fn record_success(&mut self, layer: usize, exit_id: usize) {
        if let Some(layer_hist) = self.history.get_mut(layer) {
            layer_hist.rotate_right(1);
            layer_hist[0] = Some(exit_id);
        }
        if let Some(sr) = self.success_reinforce.get_mut(layer) {
            *sr += 0.05;
        }
        if let Some(fp) = self.failure_penalty.get_mut(layer) {
            *fp *= 0.90;
        }

        // NEW: BitDrop temporal reinforcement
        self.entropy_memory[layer] *= 0.95;
        self.size_memory[layer] *= 0.95;
        self.structure_memory[layer] *= 0.95;
        self.numeric_memory[layer] *= 0.95;
        self.tunnel_memory[layer] *= 0.95;

        self.adaptive_memory[layer] *= 0.97;
        self.stability_memory[layer] *= 0.97;
    }

    /// Record a failure (circulation without exit).
    pub fn record_failure(&mut self, layer: usize) {
        if let Some(f) = self.failures.get_mut(layer) {
            *f += 1;
        }
        if let Some(fp) = self.failure_penalty.get_mut(layer) {
            *fp += 0.05;
        }

        // NEW: BitDrop temporal penalty
        self.entropy_memory[layer] += 0.02;
        self.size_memory[layer] += 0.02;
        self.structure_memory[layer] += 0.02;
        self.numeric_memory[layer] += 0.02;
        self.tunnel_memory[layer] += 0.02;

        self.adaptive_memory[layer] += 0.03;
        self.stability_memory[layer] += 0.03;
    }

    /// NEW: record locality info for HBM access
    pub fn record_locality(&mut self, layer: usize, row: u32, bank: u32, channel_id: usize) {
        self.last_row[layer] = Some(row);
        self.last_bank[layer] = Some(bank);
        self.last_channel[layer] = Some(channel_id);

        // NEW: BitDrop locality temporal reinforcement
        self.numeric_memory[layer] += 0.03;
    }

    /// NEW: record refresh storm
    pub fn record_refresh_event(&mut self, layer: usize) {
        self.refresh_events[layer] += 1;

        // NEW: BitDrop stability penalty
        self.stability_memory[layer] += 0.02;
    }

    /// NEW: record ECC correction
    pub fn record_ecc_event(&mut self, layer: usize) {
        self.ecc_events[layer] += 1;

        // NEW: BitDrop stability penalty
        self.stability_memory[layer] += 0.03;
    }

    /// NEW: record BitDrop payload geometry
    pub fn record_bitdrop(
        &mut self,
        layer: usize,
        entropy: f32,
        size: f32,
        structure: f32,
        numeric: f32,
        tunnel: f32,
        adaptive: f32,
        stability: f32,
    ) {
        self.entropy_memory[layer] += entropy * 0.05;
        self.size_memory[layer] += size * 0.05;
        self.structure_memory[layer] += structure * 0.05;
        self.numeric_memory[layer] += numeric * 0.05;
        self.tunnel_memory[layer] += tunnel * 0.05;

        self.adaptive_memory[layer] += adaptive * 0.04;
        self.stability_memory[layer] += (1.0 - stability) * 0.04;
    }

    /// MAX‑tier parallel multilayer bias computation (HBM + BitDrop)
    pub fn apply_bias_parallel(
        &self,
        req: &mut HbmRequest,
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        channels: &[HbmChannel],
    ) {
        let biases: Vec<f32> = (0..self.layers)
            .into_par_iter()
            .map(|layer| {
                // Failure bias
                let fail_bias =
                    self.failures[layer] as f32 * 0.05 + self.failure_penalty[layer];

                // Success reinforcement
                let success_bias = self.success_reinforce[layer] * 0.10;

                // Recent exit bias
                let recent_bias = self.history[layer][0].map(|exit_id| {
                    let channel = channels.iter().find(|c| c.id == exit_id);
                    if let Some(ch) = channel {
                        let idx_score = RoutingIndex::score_channel_parallel_with_grid(
                            req,
                            ch,
                            heatmap,
                            ccg,
                            self.layers,
                        );
                        -0.1 + (idx_score * 0.01)
                    } else {
                        -0.1
                    }
                }).unwrap_or(0.0);

                // Heatmap bias
                let heat_bias = heatmap.layers.get(layer).map(|layer_vec| {
                    if !layer_vec.is_empty() {
                        let avg_heat =
                            layer_vec.iter().copied().sum::<f32>() / layer_vec.len() as f32;
                        avg_heat * 0.10
                    } else {
                        0.0
                    }
                }).unwrap_or(0.0);

                // Grid bias
                let grid_bias =
                    0.35 * ccg.cluster_bias[layer][req.channel_id] +
                    0.25 * ccg.zone_bias[layer][req.channel_id] +
                    0.20 * ccg.door_bias[layer][req.channel_id] +
                    0.20 * ccg.geom_bias[layer][req.channel_id];

                // Door rotation bias
                let door_rot = ccg.door_rotation[layer][req.channel_id] as f32 * 0.01;

                // Locality bias
                let locality_bias = {
                    let mut lb = 0.0;
                    if let Some(last_ch) = self.last_channel[layer] {
                        if last_ch == req.channel_id {
                            lb += 0.15;
                        }
                    }
                    if let Some(last_row) = self.last_row[layer] {
                        if last_row == req.row {
                            lb += 0.20;
                        }
                    }
                    if let Some(last_bank) = self.last_bank[layer] {
                        if last_bank == req.bank {
                            lb += 0.20;
                        }
                    }
                    lb
                };

                // Refresh/ECC penalties
                let refresh_penalty = self.refresh_events[layer] as f32 * 0.03;
                let ecc_penalty = self.ecc_events[layer] as f32 * 0.04;

                // NEW: BitDrop temporal geometry bias
                let bitdrop_bias =
                    self.entropy_memory[layer] * 0.06 +
                    self.size_memory[layer] * 0.06 +
                    self.structure_memory[layer] * 0.05 +
                    self.numeric_memory[layer] * 0.05 +
                    self.tunnel_memory[layer] * 0.05 +
                    self.adaptive_memory[layer] * 0.04 +
                    self.stability_memory[layer] * 0.04;

                fail_bias
                    + success_bias
                    + recent_bias
                    + heat_bias
                    + door_rot
                    + locality_bias
                    + bitdrop_bias
                    - grid_bias
                    - refresh_penalty
                    - ecc_penalty
            })
            .collect();

        for layer in 0..self.layers {
            req.update_layer_bias(layer, req.layer_bias[layer] + biases[layer]);
        }
    }
}

