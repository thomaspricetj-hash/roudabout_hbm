use rayon::prelude::*;
use super::grid::CrossConnectGrid;
use super::controller::{DeltaBuffer, DeltaStore, EffectiveView};

#[derive(Debug, Clone)]
pub struct Heatmap {
    pub layers: Vec<Vec<f32>>,        // [layer][channel]
    pub decay: f32,

    pub scratch: Vec<Vec<f32>>,       // [layer][channel]
    pub door_rotation: Vec<Vec<usize>>,
    pub layer_weights: Vec<f32>,

    // HBM-specific multilayer metrics
    pub row_conflict: Vec<f32>,
    pub bank_busy: Vec<f32>,
    pub channel_sat: Vec<f32>,
    pub refresh_heat: Vec<f32>,
    pub ecc_heat: Vec<f32>,

    // BitDrop-aware multilayer metrics
    pub bitdrop_payload_heat: Vec<f32>,
    pub bitdrop_tunnel_heat: Vec<f32>,
    pub bitdrop_locality_heat: Vec<f32>,

    // Tesla valve directional heat fields
    pub valve_forward_heat: Vec<f32>,
    pub valve_reverse_heat: Vec<f32>,
    pub valve_oscillation_heat: Vec<f32>,
}

impl Heatmap {
    pub fn new(layer_count: usize, channel_count: usize, decay: f32) -> Self {
        let zero_layer = || vec![0.0; channel_count];
        let zero_usize_layer = || vec![0; channel_count];

        Self {
            layers: (0..layer_count).map(|_| zero_layer()).collect(),
            scratch: (0..layer_count).map(|_| zero_layer()).collect(),
            door_rotation: (0..layer_count).map(|_| zero_usize_layer()).collect(),
            layer_weights: vec![1.0; layer_count],
            decay,

            row_conflict: vec![0.0; channel_count],
            bank_busy: vec![0.0; channel_count],
            channel_sat: vec![0.0; channel_count],
            refresh_heat: vec![0.0; channel_count],
            ecc_heat: vec![0.0; channel_count],

            bitdrop_payload_heat: vec![0.0; channel_count],
            bitdrop_tunnel_heat: vec![0.0; channel_count],
            bitdrop_locality_heat: vec![0.0; channel_count],

            valve_forward_heat: vec![0.0; channel_count],
            valve_reverse_heat: vec![0.0; channel_count],
            valve_oscillation_heat: vec![0.0; channel_count],
        }
    }

    // =========================================================================
    // DAX INTEGRATION — FULL MAX LOGIC
    // =========================================================================

    pub fn clear(&mut self) {
        for layer_vec in &mut self.layers {
            layer_vec.fill(0.0);
        }
        for layer_vec in &mut self.scratch {
            layer_vec.fill(0.0);
        }
        for rot in &mut self.door_rotation {
            rot.fill(0);
        }

        self.row_conflict.fill(0.0);
        self.bank_busy.fill(0.0);
        self.channel_sat.fill(0.0);
        self.refresh_heat.fill(0.0);
        self.ecc_heat.fill(0.0);

        self.bitdrop_payload_heat.fill(0.0);
        self.bitdrop_tunnel_heat.fill(0.0);
        self.bitdrop_locality_heat.fill(0.0);

        self.valve_forward_heat.fill(0.0);
        self.valve_reverse_heat.fill(0.0);
        self.valve_oscillation_heat.fill(0.0);
    }

    pub fn apply_delta(&mut self, delta: &DeltaBuffer) {
        let layer_idx = delta.layer.min(self.layers.len().saturating_sub(1));
        let channel_count = self.layers.get(0).map(|v| v.len()).unwrap_or(0);
        if channel_count == 0 {
            return;
        }
        let ch = (delta.row as usize) % channel_count;

        if let Some(layer_vec) = self.layers.get_mut(layer_idx) {
            if let Some(v) = layer_vec.get_mut(ch) {
                *v += 0.08;
            }
        }

        self.row_conflict[ch] += 0.03;
        self.bank_busy[ch] += 0.03;
        self.channel_sat[ch] += 0.02;

        let payload = &delta.payload;
        if payload.len() >= 3 {
            let entropy = (payload[0] as f32) / 255.0;
            let tunnel = (payload[1] as f32) / 255.0;
            let locality = (payload[2] as f32) / 255.0;

            self.bitdrop_payload_heat[ch] += entropy * 0.06;
            self.bitdrop_tunnel_heat[ch] += tunnel * 0.06;
            self.bitdrop_locality_heat[ch] += locality * 0.05;

            self.valve_forward_heat[ch] += (1.0 - entropy) * 0.04;
            self.valve_reverse_heat[ch] += tunnel * 0.05;
            self.valve_oscillation_heat[ch] += locality * 0.06;
        }

        let seq_heat = (delta.seq as f32 * 0.0001).min(0.05);
        self.refresh_heat[ch] += seq_heat * 0.5;
        self.ecc_heat[ch] += seq_heat * 0.3;
    }

    pub fn apply_effective_view(&mut self, view: &EffectiveView, store: &DeltaStore) {
        self.clear();
        for delta_id in &view.delta_ids {
            if let Some(delta) = store.deltas.iter().find(|d| d.id == *delta_id) {
                self.apply_delta(delta);
            }
        }
    }

    pub fn rollback_to(&mut self, master_id: usize, store: &DeltaStore, target_seq: u64) {
        self.clear();
        let deltas = store.deltas_for_master(master_id);
        for d in deltas {
            if d.seq <= target_seq {
                self.apply_delta(d);
            }
        }
    }

    pub fn switch_view(&mut self, view_idx: usize, store: &DeltaStore) {
        if let Some(view) = store.get_view(view_idx) {
            self.apply_effective_view(view, store);
        }
    }

    // -------------------------------------------------------------------------
    // NEW: Tesla valve heat injection
    // -------------------------------------------------------------------------

    pub fn add_valve_forward(&mut self, channel: usize, amount: f32) {
        if let Some(v) = self.valve_forward_heat.get_mut(channel) {
            *v += amount;
        }
    }

    pub fn add_valve_reverse(&mut self, channel: usize, amount: f32) {
        if let Some(v) = self.valve_reverse_heat.get_mut(channel) {
            *v += amount;
        }
    }

    pub fn add_valve_oscillation(&mut self, channel: usize, amount: f32) {
        if let Some(v) = self.valve_oscillation_heat.get_mut(channel) {
            *v += amount;
        }
    }

    // -------------------------------------------------------------------------
    // Parallel multilayer decay (Tesla valve included)
    // -------------------------------------------------------------------------

    pub fn decay_step(&mut self) {
        self.layers.par_iter_mut().for_each(|layer_vec| {
            layer_vec.iter_mut().for_each(|v| *v *= self.decay);
        });

        self.row_conflict.par_iter_mut().for_each(|v| *v *= self.decay);
        self.bank_busy.par_iter_mut().for_each(|v| *v *= self.decay);
        self.channel_sat.par_iter_mut().for_each(|v| *v *= self.decay);
        self.refresh_heat.par_iter_mut().for_each(|v| *v *= self.decay);
        self.ecc_heat.par_iter_mut().for_each(|v| *v *= self.decay);

        self.bitdrop_payload_heat.par_iter_mut().for_each(|v| *v *= self.decay);
        self.bitdrop_tunnel_heat.par_iter_mut().for_each(|v| *v *= self.decay);
        self.bitdrop_locality_heat.par_iter_mut().for_each(|v| *v *= self.decay);

        self.valve_forward_heat.par_iter_mut().for_each(|v| *v *= self.decay);
        self.valve_reverse_heat.par_iter_mut().for_each(|v| *v *= self.decay);
        self.valve_oscillation_heat.par_iter_mut().for_each(|v| *v *= self.decay);
    }

    // -------------------------------------------------------------------------
    // Parallel heat injection
    // -------------------------------------------------------------------------

    pub fn add_heat_parallel(&mut self, layer: usize, channel: usize, amount: f32) {
        if let Some(layer_vec) = self.layers.get_mut(layer) {
            if let Some(v) = layer_vec.get_mut(channel) {
                *v += amount;
            }
        }
    }

    pub fn add_row_conflict(&mut self, channel: usize, amount: f32) {
        if let Some(v) = self.row_conflict.get_mut(channel) {
            *v += amount;
        }
    }

    pub fn add_bank_busy(&mut self, channel: usize, amount: f32) {
        if let Some(v) = self.bank_busy.get_mut(channel) {
            *v += amount;
        }
    }

    pub fn add_channel_sat(&mut self, channel: usize, amount: f32) {
        if let Some(v) = self.channel_sat.get_mut(channel) {
            *v += amount;
        }
    }

    pub fn add_refresh_heat(&mut self, channel: usize, amount: f32) {
        if let Some(v) = self.refresh_heat.get_mut(channel) {
            *v += amount;
        }
    }

    pub fn add_ecc_heat(&mut self, channel: usize, amount: f32) {
        if let Some(v) = self.ecc_heat.get_mut(channel) {
            *v += amount;
        }
    }

    pub fn add_bitdrop_payload_heat(&mut self, channel: usize, amount: f32) {
        if let Some(v) = self.bitdrop_payload_heat.get_mut(channel) {
            *v += amount;
        }
    }

    pub fn add_bitdrop_tunnel_heat(&mut self, channel: usize, amount: f32) {
        if let Some(v) = self.bitdrop_tunnel_heat.get_mut(channel) {
            *v += amount;
        }
    }

    pub fn add_bitdrop_locality_heat(&mut self, channel: usize, amount: f32) {
        if let Some(v) = self.bitdrop_locality_heat.get_mut(channel) {
            *v += amount;
        }
    }

    // -------------------------------------------------------------------------
    // Parallel normalization (Tesla valve included)
    // -------------------------------------------------------------------------

    pub fn normalize(&mut self) {
        self.layers.par_iter_mut().for_each(|layer_vec| {
            let max_val = layer_vec.iter().copied().fold(0.0_f32, |acc, x| acc.max(x));
            if max_val > 0.0 {
                let inv = 1.0 / max_val;
                layer_vec.iter_mut().for_each(|v| *v *= inv);
            }
        });

        let normalize_vec = |vec: &mut Vec<f32>| {
            let max_val = vec.iter().copied().fold(0.0_f32, |acc, x| acc.max(x));
            if max_val > 0.0 {
                let inv = 1.0 / max_val;
                vec.iter_mut().for_each(|v| *v *= inv);
            }
        };

        normalize_vec(&mut self.row_conflict);
        normalize_vec(&mut self.bank_busy);
        normalize_vec(&mut self.channel_sat);
        normalize_vec(&mut self.refresh_heat);
        normalize_vec(&mut self.ecc_heat);

        normalize_vec(&mut self.bitdrop_payload_heat);
        normalize_vec(&mut self.bitdrop_tunnel_heat);
        normalize_vec(&mut self.bitdrop_locality_heat);

        normalize_vec(&mut self.valve_forward_heat);
        normalize_vec(&mut self.valve_reverse_heat);
        normalize_vec(&mut self.valve_oscillation_heat);
    }

    // -------------------------------------------------------------------------
    // Reinforcement / cooling
    // -------------------------------------------------------------------------

    pub fn reinforce_parallel(&mut self, layer: usize, channel: usize) {
        if let Some(layer_vec) = self.layers.get_mut(layer) {
            if let Some(v) = layer_vec.get_mut(channel) {
                *v += 0.05;
            }
        }
    }

    pub fn cool_parallel(&mut self, layer: usize, channel: usize) {
        if let Some(layer_vec) = self.layers.get_mut(layer) {
            if let Some(v) = layer_vec.get_mut(channel) {
                *v *= 0.90;
            }
        }
    }

    pub fn cool_refresh(&mut self, channel: usize) {
        if let Some(v) = self.refresh_heat.get_mut(channel) {
            *v *= 0.85;
        }
    }

    pub fn cool_ecc(&mut self, channel: usize) {
        if let Some(v) = self.ecc_heat.get_mut(channel) {
            *v *= 0.80;
        }
    }

    pub fn inject_layer_heat(&mut self, layer: usize, amount: f32) {
        if let Some(layer_vec) = self.layers.get_mut(layer) {
            layer_vec.par_iter_mut().for_each(|v| *v += amount);
        }
    }

    pub fn cool_layer(&mut self, layer: usize, factor: f32) {
        if let Some(layer_vec) = self.layers.get_mut(layer) {
            layer_vec.par_iter_mut().for_each(|v| *v *= factor);
        }
    }

    pub fn rotate_doors(&mut self, layer: usize) {
        if let Some(rot) = self.door_rotation.get_mut(layer) {
            if !rot.is_empty() {
                rot.rotate_left(1);
            }
        }
    }

    pub fn cache_scratch(&mut self, layer: usize, channel: usize, value: f32) {
        if let Some(layer_vec) = self.scratch.get_mut(layer) {
            if let Some(v) = layer_vec.get_mut(channel) {
                *v = value;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Fused heat scoring (Tesla valve included)
    // -------------------------------------------------------------------------

    pub fn fused_heat(&self, channel: usize) -> f32 {
        let mut acc = 0.0;

        for (layer, w) in self.layer_weights.iter().enumerate() {
            acc += w * self.layers[layer][channel];
        }

        acc += self.row_conflict[channel] * 0.40;
        acc += self.bank_busy[channel] * 0.35;
        acc += self.channel_sat[channel] * 0.25;
        acc += self.refresh_heat[channel] * 0.20;
        acc += self.ecc_heat[channel] * 0.15;

        acc += self.bitdrop_payload_heat[channel] * 0.25;
        acc += self.bitdrop_tunnel_heat[channel] * 0.20;
        acc += self.bitdrop_locality_heat[channel] * 0.20;

        acc += self.valve_forward_heat[channel] * -0.20;
        acc += self.valve_reverse_heat[channel] * 0.25;
        acc += self.valve_oscillation_heat[channel] * 0.30;

        acc
    }

    pub fn fused_heat_with_grid(
        &self,
        channel: usize,
        ccg: &CrossConnectGrid
    ) -> f32 {
        let heat = self.fused_heat(channel);
        let bias = ccg.fused_bias(channel);
        heat - bias
    }
}
