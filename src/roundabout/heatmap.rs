use rayon::prelude::*;
use super::grid::CrossConnectGrid;

#[derive(Debug, Clone)]
pub struct Heatmap {
    pub layers: Vec<Vec<f32>>,        // [layer][channel]
    pub decay: f32,

    // multilayer scratchpad cache
    pub scratch: Vec<Vec<f32>>,       // [layer][channel]

    // rotating doors per layer
    pub door_rotation: Vec<Vec<usize>>,

    // per-layer weights for fused scoring
    pub layer_weights: Vec<f32>,

    // NEW: HBM-specific multilayer metrics
    pub row_conflict: Vec<f32>,       // row conflict heat
    pub bank_busy: Vec<f32>,          // bank busy heat
    pub channel_sat: Vec<f32>,        // channel saturation heat
    pub refresh_heat: Vec<f32>,       // refresh cycle heat
    pub ecc_heat: Vec<f32>,           // ECC correction heat
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

            // NEW: HBM multilayer heat sources
            row_conflict: vec![0.0; channel_count],
            bank_busy: vec![0.0; channel_count],
            channel_sat: vec![0.0; channel_count],
            refresh_heat: vec![0.0; channel_count],
            ecc_heat: vec![0.0; channel_count],
        }
    }

    /// Parallel multilayer decay
    pub fn decay_step(&mut self) {
        self.layers.par_iter_mut().for_each(|layer_vec| {
            layer_vec.iter_mut().for_each(|v| *v *= self.decay);
        });

        // NEW: decay HBM-specific layers
        self.row_conflict.par_iter_mut().for_each(|v| *v *= self.decay);
        self.bank_busy.par_iter_mut().for_each(|v| *v *= self.decay);
        self.channel_sat.par_iter_mut().for_each(|v| *v *= self.decay);
        self.refresh_heat.par_iter_mut().for_each(|v| *v *= self.decay);
        self.ecc_heat.par_iter_mut().for_each(|v| *v *= self.decay);
    }

    /// Parallel heat injection across layers
    pub fn add_heat_parallel(&mut self, layer: usize, channel: usize, amount: f32) {
        if let Some(layer_vec) = self.layers.get_mut(layer) {
            if let Some(v) = layer_vec.get_mut(channel) {
                *v += amount;
            }
        }
    }

    /// NEW: HBM row conflict heat injection
    pub fn add_row_conflict(&mut self, channel: usize, amount: f32) {
        if let Some(v) = self.row_conflict.get_mut(channel) {
            *v += amount;
        }
    }

    /// NEW: HBM bank busy heat injection
    pub fn add_bank_busy(&mut self, channel: usize, amount: f32) {
        if let Some(v) = self.bank_busy.get_mut(channel) {
            *v += amount;
        }
    }

    /// NEW: HBM channel saturation heat injection
    pub fn add_channel_sat(&mut self, channel: usize, amount: f32) {
        if let Some(v) = self.channel_sat.get_mut(channel) {
            *v += amount;
        }
    }

    /// NEW: HBM refresh cycle heat injection
    pub fn add_refresh_heat(&mut self, channel: usize, amount: f32) {
        if let Some(v) = self.refresh_heat.get_mut(channel) {
            *v += amount;
        }
    }

    /// NEW: HBM ECC correction heat injection
    pub fn add_ecc_heat(&mut self, channel: usize, amount: f32) {
        if let Some(v) = self.ecc_heat.get_mut(channel) {
            *v += amount;
        }
    }

    /// Parallel normalization: keeps heatmap stable under heavy load
    pub fn normalize(&mut self) {
        self.layers.par_iter_mut().for_each(|layer_vec| {
            let max_val = layer_vec.iter().copied().fold(0.0_f32, |acc, x| acc.max(x));
            if max_val > 0.0 {
                let inv = 1.0 / max_val;
                layer_vec.iter_mut().for_each(|v| *v *= inv);
            }
        });

        // NEW: normalize HBM-specific layers
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
    }

    /// Parallel reinforcement: boost channels that recently succeeded
    pub fn reinforce_parallel(&mut self, layer: usize, channel: usize) {
        if let Some(layer_vec) = self.layers.get_mut(layer) {
            if let Some(v) = layer_vec.get_mut(channel) {
                *v += 0.05;
            }
        }
    }

    /// Parallel cooling: reduce heat on channels that failed
    pub fn cool_parallel(&mut self, layer: usize, channel: usize) {
        if let Some(layer_vec) = self.layers.get_mut(layer) {
            if let Some(v) = layer_vec.get_mut(channel) {
                *v *= 0.90;
            }
        }
    }

    /// NEW: refresh-aware cooling
    pub fn cool_refresh(&mut self, channel: usize) {
        if let Some(v) = self.refresh_heat.get_mut(channel) {
            *v *= 0.85;
        }
    }

    /// NEW: ECC-aware cooling
    pub fn cool_ecc(&mut self, channel: usize) {
        if let Some(v) = self.ecc_heat.get_mut(channel) {
            *v *= 0.80;
        }
    }

    /// Parallel full-layer heat injection (used for global events)
    pub fn inject_layer_heat(&mut self, layer: usize, amount: f32) {
        if let Some(layer_vec) = self.layers.get_mut(layer) {
            layer_vec.par_iter_mut().for_each(|v| *v += amount);
        }
    }

    /// Parallel full-layer cooling (used for refresh events)
    pub fn cool_layer(&mut self, layer: usize, factor: f32) {
        if let Some(layer_vec) = self.layers.get_mut(layer) {
            layer_vec.par_iter_mut().for_each(|v| *v *= factor);
        }
    }

    /// rotate doors for a given layer
    pub fn rotate_doors(&mut self, layer: usize) {
        if let Some(rot) = self.door_rotation.get_mut(layer) {
            if !rot.is_empty() {
                rot.rotate_left(1);
            }
        }
    }

    /// cache scratchpad values for fused scoring
    pub fn cache_scratch(&mut self, layer: usize, channel: usize, value: f32) {
        if let Some(layer_vec) = self.scratch.get_mut(layer) {
            if let Some(v) = layer_vec.get_mut(channel) {
                *v = value;
            }
        }
    }

    /// fused multilayer heat score (HBM-aware)
    pub fn fused_heat(&self, channel: usize) -> f32 {
        let mut acc = 0.0;

        // multilayer base heat
        for (layer, w) in self.layer_weights.iter().enumerate() {
            acc += w * self.layers[layer][channel];
        }

        // NEW: HBM locality heat sources
        acc += self.row_conflict[channel] * 0.40;
        acc += self.bank_busy[channel] * 0.35;
        acc += self.channel_sat[channel] * 0.25;
        acc += self.refresh_heat[channel] * 0.20;
        acc += self.ecc_heat[channel] * 0.15;

        acc
    }

    /// fused multilayer heat + grid bias (HBM geometry)
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

