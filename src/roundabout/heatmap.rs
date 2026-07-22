use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Heatmap {
    pub layers: Vec<Vec<f32>>, // [layer][channel]
    pub decay: f32,
}

impl Heatmap {
    pub fn new(layer_count: usize, channel_count: usize, decay: f32) -> Self {
        Self {
            layers: vec![vec![0.0; channel_count]; layer_count],
            decay,
        }
    }

    /// Parallel multilayer decay
    pub fn decay_step(&mut self) {
        self.layers.par_iter_mut().for_each(|layer_vec| {
            layer_vec.iter_mut().for_each(|v| {
                *v *= self.decay;
            });
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
            // FIX: explicit max computation, no type inference issue
            let max_val = layer_vec
                .iter()
                .copied()
                .fold(0.0_f32, |acc, x| acc.max(x));

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
            layer_vec.par_iter_mut().for_each(|v| {
                *v += amount;
            });
        }
    }

    /// Parallel full-layer cooling (used for refresh events)
    pub fn cool_layer(&mut self, layer: usize, factor: f32) {
        if let Some(layer_vec) = self.layers.get_mut(layer) {
            layer_vec.par_iter_mut().for_each(|v| {
                *v *= factor;
            });
        }
    }
}
