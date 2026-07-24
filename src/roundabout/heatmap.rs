use rayon::prelude::*;
use super::grid::CrossConnectGrid;

#[derive(Debug, Clone)]
pub struct Heatmap {
    pub layers: Vec<Vec<f32>>,        // [layer][channel]
    pub decay: f32,

    // NEW: multilayer scratchpad cache
    pub scratch: Vec<Vec<f32>>,       // [layer][channel]

    // NEW: rotating doors per layer
    pub door_rotation: Vec<Vec<usize>>,

    // NEW: per-layer weights for fused scoring
    pub layer_weights: Vec<f32>,
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
        }
    }

    /// Parallel multilayer decay
    pub fn decay_step(&mut self) {
        self.layers.par_iter_mut().for_each(|layer_vec| {
            layer_vec.iter_mut().for_each(|v| *v *= self.decay);
        });
    }

    /// Parallel heat injection across layers
    pub fn add_heat_parallel(&mut self, layer: usize, channel: usize, amount: f32) {
        if let Some(layer_vec) = self.layers.get_mut(layer) {
            if let Some(v) = layer_vec.get_mut(channel) {
                *v += amount;
            }
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

    // NEW: rotate doors for a given layer
    pub fn rotate_doors(&mut self, layer: usize) {
        if let Some(rot) = self.door_rotation.get_mut(layer) {
            if !rot.is_empty() {
                rot.rotate_left(1);
            }
        }
    }

    // NEW: cache scratchpad values for fused scoring
    pub fn cache_scratch(&mut self, layer: usize, channel: usize, value: f32) {
        if let Some(layer_vec) = self.scratch.get_mut(layer) {
            if let Some(v) = layer_vec.get_mut(channel) {
                *v = value;
            }
        }
    }

    // NEW: fused multilayer heat score
    pub fn fused_heat(&self, channel: usize) -> f32 {
        let mut acc = 0.0;
        for (layer, w) in self.layer_weights.iter().enumerate() {
            acc += w * self.layers[layer][channel];
        }
        acc
    }

    // NEW: fused multilayer heat + grid bias
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
