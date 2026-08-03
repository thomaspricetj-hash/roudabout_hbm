use rayon::prelude::*;

use super::{
    request::HbmRequest,
    heatmap::Heatmap,
    index::RoutingIndex,
    channel::HbmChannel,
    grid::CrossConnectGrid,
    controller::{DeltaBuffer, DeltaStore, EffectiveView},
};

#[derive(Debug, Clone)]
pub struct Scratchpad {
    pub layers: usize,
    pub history: Vec<Vec<Option<usize>>>,
    pub failures: Vec<u32>,

    pub last_row: Vec<Option<u32>>,
    pub last_bank: Vec<Option<u32>>,
    pub last_channel: Vec<Option<usize>>,

    pub refresh_events: Vec<u32>,
    pub ecc_events: Vec<u32>,

    pub success_reinforce: Vec<f32>,
    pub failure_penalty: Vec<f32>,

    pub entropy_memory: Vec<f32>,
    pub size_memory: Vec<f32>,
    pub structure_memory: Vec<f32>,
    pub numeric_memory: Vec<f32>,
    pub tunnel_memory: Vec<f32>,

    pub adaptive_memory: Vec<f32>,
    pub stability_memory: Vec<f32>,

    // Tesla valve directional memory
    pub valve_forward_memory: Vec<f32>,
    pub valve_reverse_memory: Vec<f32>,
    pub valve_oscillation_memory: Vec<f32>,
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

            entropy_memory: vec![0.0; layers],
            size_memory: vec![0.0; layers],
            structure_memory: vec![0.0; layers],
            numeric_memory: vec![0.0; layers],
            tunnel_memory: vec![0.0; layers],

            adaptive_memory: vec![0.0; layers],
            stability_memory: vec![0.0; layers],

            valve_forward_memory: vec![0.0; layers],
            valve_reverse_memory: vec![0.0; layers],
            valve_oscillation_memory: vec![0.0; layers],
        }
    }

    // =========================================================================
    // DAX INTEGRATION — FULL MAX LOGIC
    // =========================================================================

    /// Reset scratchpad state before applying deltas or views
    pub fn clear(&mut self) {
        for layer in 0..self.layers {
            self.history[layer].fill(None);
            self.failures[layer] = 0;

            self.last_row[layer] = None;
            self.last_bank[layer] = None;
            self.last_channel[layer] = None;

            self.refresh_events[layer] = 0;
            self.ecc_events[layer] = 0;

            self.success_reinforce[layer] = 0.0;
            self.failure_penalty[layer] = 0.0;

            self.entropy_memory[layer] = 0.0;
            self.size_memory[layer] = 0.0;
            self.structure_memory[layer] = 0.0;
            self.numeric_memory[layer] = 0.0;
            self.tunnel_memory[layer] = 0.0;

            self.adaptive_memory[layer] = 0.0;
            self.stability_memory[layer] = 0.0;

            self.valve_forward_memory[layer] = 0.0;
            self.valve_reverse_memory[layer] = 0.0;
            self.valve_oscillation_memory[layer] = 0.0;
        }
    }

    /// Apply a single delta to scratchpad state
    pub fn apply_delta(&mut self, delta: &DeltaBuffer) {
        let layer = delta.layer.min(self.layers - 1);

        // Update locality
        self.last_row[layer] = Some(delta.row as u32);
        self.last_bank[layer] = Some(delta.bank as u32);

        // Update history (channel exit)
        self.history[layer].rotate_right(1);
        self.history[layer][0] = Some(delta.row % 64); // synthetic exit mapping

        // Update memories based on delta payload
        let payload = &delta.payload;

        if !payload.is_empty() {
            let entropy = (payload[0] as f32) / 255.0;
            let structure = (payload[1] as f32) / 255.0;
            let valve = (payload[2] as f32) / 255.0;

            self.entropy_memory[layer] += entropy * 0.05;
            self.structure_memory[layer] += structure * 0.05;
            self.valve_forward_memory[layer] += valve * 0.04;
        }

        // Temporal memory from sequence
        self.stability_memory[layer] += (delta.seq as f32 * 0.0001).min(0.05);
    }

    /// Apply an effective view (all deltas) to scratchpad
    pub fn apply_effective_view(&mut self, view: &EffectiveView, store: &DeltaStore) {
        self.clear();
        for delta_id in &view.delta_ids {
            if let Some(delta) = store.deltas.iter().find(|d| d.id == *delta_id) {
                self.apply_delta(delta);
            }
        }
    }

    /// Rollback scratchpad to a target sequence
    pub fn rollback_to(&mut self, master_id: usize, store: &DeltaStore, target_seq: u64) {
        self.clear();
        let deltas = store.deltas_for_master(master_id);
        for d in deltas {
            if d.seq <= target_seq {
                self.apply_delta(d);
            }
        }
    }

    /// Switch scratchpad to a specific DAX view
    pub fn switch_view(&mut self, view_idx: usize, store: &DeltaStore) {
        if let Some(view) = store.get_view(view_idx) {
            self.apply_effective_view(view, store);
        }
    }

    // =========================================================================
    // SUCCESS / FAILURE reinforcement (Tesla valve integrated)
    // =========================================================================

    pub fn record_success(&mut self, layer: usize, exit_id: usize) {
        if let Some(hist) = self.history.get_mut(layer) {
            hist.rotate_right(1);
            hist[0] = Some(exit_id);
        }
        if let Some(sr) = self.success_reinforce.get_mut(layer) {
            *sr += 0.05;
        }
        if let Some(fp) = self.failure_penalty.get_mut(layer) {
            *fp *= 0.90;
        }

        if let Some(e) = self.entropy_memory.get_mut(layer) { *e *= 0.95; }
        if let Some(s) = self.size_memory.get_mut(layer) { *s *= 0.95; }
        if let Some(st) = self.structure_memory.get_mut(layer) { *st *= 0.95; }
        if let Some(n) = self.numeric_memory.get_mut(layer) { *n *= 0.95; }
        if let Some(t) = self.tunnel_memory.get_mut(layer) { *t *= 0.95; }

        if let Some(a) = self.adaptive_memory.get_mut(layer) { *a *= 0.97; }
        if let Some(stab) = self.stability_memory.get_mut(layer) { *stab *= 0.97; }

        if let Some(vf) = self.valve_forward_memory.get_mut(layer) {
            *vf += 0.04;
        }
        if let Some(vr) = self.valve_reverse_memory.get_mut(layer) {
            *vr *= 0.90;
        }
        if let Some(vo) = self.valve_oscillation_memory.get_mut(layer) {
            *vo *= 0.90;
        }
    }

    pub fn record_failure(&mut self, layer: usize) {
        if let Some(f) = self.failures.get_mut(layer) { *f += 1; }
        if let Some(fp) = self.failure_penalty.get_mut(layer) { *fp += 0.05; }

        if let Some(e) = self.entropy_memory.get_mut(layer) { *e += 0.02; }
        if let Some(s) = self.size_memory.get_mut(layer) { *s += 0.02; }
        if let Some(st) = self.structure_memory.get_mut(layer) { *st += 0.02; }
        if let Some(n) = self.numeric_memory.get_mut(layer) { *n += 0.02; }
        if let Some(t) = self.tunnel_memory.get_mut(layer) { *t += 0.02; }

        if let Some(a) = self.adaptive_memory.get_mut(layer) { *a += 0.03; }
        if let Some(stab) = self.stability_memory.get_mut(layer) { *stab += 0.03; }

        if let Some(vr) = self.valve_reverse_memory.get_mut(layer) {
            *vr += 0.04;
        }
        if let Some(vo) = self.valve_oscillation_memory.get_mut(layer) {
            *vo += 0.05;
        }
    }

    // =========================================================================
    // Locality + refresh + ECC
    // =========================================================================

    pub fn record_locality(&mut self, layer: usize, row: u32, bank: u32, channel_id: usize) {
        self.last_row[layer] = Some(row);
        self.last_bank[layer] = Some(bank);
        self.last_channel[layer] = Some(channel_id);

        if let Some(n) = self.numeric_memory.get_mut(layer) { *n += 0.03; }
    }

    pub fn record_refresh_event(&mut self, layer: usize) {
        self.refresh_events[layer] += 1;
        self.stability_memory[layer] += 0.02;
    }

    pub fn record_ecc_event(&mut self, layer: usize) {
        self.ecc_events[layer] += 1;
        self.stability_memory[layer] += 0.03;
    }

    // =========================================================================
    // BitDrop + Tesla valve bitflow memory
    // =========================================================================

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
        valve_forward: f32,
        valve_reverse: f32,
        valve_oscillation: f32,
    ) {
        self.entropy_memory[layer] += entropy * 0.05;
        self.size_memory[layer] += size * 0.05;
        self.structure_memory[layer] += structure * 0.05;
        self.numeric_memory[layer] += numeric * 0.05;
        self.tunnel_memory[layer] += tunnel * 0.05;

        self.adaptive_memory[layer] += adaptive * 0.04;
        self.stability_memory[layer] += (1.0 - stability) * 0.04;

        self.valve_forward_memory[layer] += valve_forward * 0.05;
        self.valve_reverse_memory[layer] += valve_reverse * 0.06;
        self.valve_oscillation_memory[layer] += valve_oscillation * 0.07;
    }

    // =========================================================================
    // APPLY BIAS (Tesla valve integrated)
    // =========================================================================

    pub fn apply_bias_parallel(
        &self,
        req: &mut HbmRequest,
        heatmap: &Heatmap,
        ccg: &CrossConnectGrid,
        channels: &[HbmChannel],
    ) {
        let max_layers = self.layers.min(req.layer_bias.len());

        let biases: Vec<f32> = (0..max_layers)
            .into_par_iter()
            .map(|layer| {
                let fail_bias =
                    self.failures[layer] as f32 * 0.05 +
                    self.failure_penalty[layer];

                let success_bias =
                    self.success_reinforce[layer] * 0.10;

                let recent_bias = self.history[layer][0]
                    .and_then(|exit_id| {
                        channels.iter().find(|c| c.id == exit_id).map(|ch| {
                            let idx = RoutingIndex::score_channel_parallel_with_grid(
                                req, ch, heatmap, ccg, max_layers);
                            -0.1 + idx * 0.01
                        })
                    })
                    .unwrap_or(0.0);

                let heat_bias = heatmap.layers[layer]
                    .iter()
                    .copied()
                    .sum::<f32>()
                    / heatmap.layers[layer].len().max(1) as f32
                    * 0.10;

                let cid = req.channel_id.min(channels.len().saturating_sub(1));

                let grid_bias =
                    0.35 * ccg.cluster_bias[layer][cid] +
                    0.25 * ccg.zone_bias[layer][cid] +
                    0.20 * ccg.door_bias[layer][cid] +
                    0.20 * ccg.geom_bias[layer][cid];

                let door_rot =
                    ccg.door_rotation[layer][cid] as f32 * 0.01;

                let locality_bias = {
                    let mut lb = 0.0;
                    if self.last_channel[layer] == Some(req.channel_id) { lb += 0.15; }
                    if self.last_row[layer] == Some(req.row) { lb += 0.20; }
                    if self.last_bank[layer] == Some(req.bank) { lb += 0.20; }
                    lb
                };

                let refresh_penalty = self.refresh_events[layer] as f32 * 0.03;
                let ecc_penalty = self.ecc_events[layer] as f32 * 0.04;

                let bitdrop_bias =
                    self.entropy_memory[layer] * 0.06 +
                    self.size_memory[layer] * 0.06 +
                    self.structure_memory[layer] * 0.05 +
                    self.numeric_memory[layer] * 0.05 +
                    self.tunnel_memory[layer] * 0.05 +
                    self.adaptive_memory[layer] * 0.04 +
                    self.stability_memory[layer] * 0.04;

                let valve_bias =
                    self.valve_forward_memory[layer] * -0.08 +
                    self.valve_reverse_memory[layer] * 0.10 +
                    self.valve_oscillation_memory[layer] * 0.12;

                fail_bias
                    + success_bias
                    + recent_bias
                    + heat_bias
                    + door_rot
                    + locality_bias
                    + bitdrop_bias
                    + valve_bias
                    - grid_bias
                    - refresh_penalty
                    - ecc_penalty
            })
            .collect();

        for layer in 0..max_layers {
            let current = req.layer_bias[layer];
            req.update_layer_bias(layer, current + biases[layer]);
        }
    }
}



