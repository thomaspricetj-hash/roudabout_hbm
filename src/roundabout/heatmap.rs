use rayon::prelude::*;
use super::grid::CrossConnectGrid;

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

    // NEW: Tesla valve directional heat fields
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

            // Tesla valve directional heat fields
            valve_forward_heat: vec![0.0; channel_count],
            valve_reverse_heat: vec![0.0; channel_count],
            valve_oscillation_heat: vec![0.0; channel_count],
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

        // Tesla valve directional decay
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

        // Tesla valve directional normalization
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

        // Tesla valve directional heat
        acc += self.valve_forward_heat[channel] * -0.20;   // forward flow reduces heat
        acc += self.valve_reverse_heat[channel] * 0.25;    // reverse flow increases heat
        acc += self.valve_oscillation_heat[channel] * 0.30; // oscillation increases heat sharply

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


