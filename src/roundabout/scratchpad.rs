use rayon::prelude::*;

use super::{
    request::HbmRequest,
    heatmap::Heatmap,
    index::RoutingIndex,
    channel::HbmChannel,
    grid::CrossConnectGrid,
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
        }
    }

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
    }

    pub fn record_locality(&mut self, layer: usize, row: u32, bank: u32, channel_id: usize) {
        if let Some(r) = self.last_row.get_mut(layer) { *r = Some(row); }
        if let Some(b) = self.last_bank.get_mut(layer) { *b = Some(bank); }
        if let Some(c) = self.last_channel.get_mut(layer) { *c = Some(channel_id); }

        if let Some(n) = self.numeric_memory.get_mut(layer) { *n += 0.03; }
    }

    pub fn record_refresh_event(&mut self, layer: usize) {
        if let Some(r) = self.refresh_events.get_mut(layer) { *r += 1; }
        if let Some(stab) = self.stability_memory.get_mut(layer) { *stab += 0.02; }
    }

    pub fn record_ecc_event(&mut self, layer: usize) {
        if let Some(e) = self.ecc_events.get_mut(layer) { *e += 1; }
        if let Some(stab) = self.stability_memory.get_mut(layer) { *stab += 0.03; }
    }

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
        if let Some(e) = self.entropy_memory.get_mut(layer) { *e += entropy * 0.05; }
        if let Some(s) = self.size_memory.get_mut(layer) { *s += size * 0.05; }
        if let Some(st) = self.structure_memory.get_mut(layer) { *st += structure * 0.05; }
        if let Some(n) = self.numeric_memory.get_mut(layer) { *n += numeric * 0.05; }
        if let Some(t) = self.tunnel_memory.get_mut(layer) { *t += tunnel * 0.05; }

        if let Some(a) = self.adaptive_memory.get_mut(layer) { *a += adaptive * 0.04; }
        if let Some(stab) = self.stability_memory.get_mut(layer) { *stab += (1.0 - stability) * 0.04; }
    }

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
                    self.failures.get(layer).copied().unwrap_or(0) as f32 * 0.05
                    + self.failure_penalty.get(layer).copied().unwrap_or(0.0);

                let success_bias =
                    self.success_reinforce.get(layer).copied().unwrap_or(0.0) * 0.10;

                let recent_bias = self.history.get(layer)
                    .and_then(|h| h.get(0))
                    .and_then(|exit| exit.map(|exit_id| {
                        channels.iter().find(|c| c.id == exit_id)
                            .map(|ch| {
                                let idx = RoutingIndex::score_channel_parallel_with_grid(
                                    req, ch, heatmap, ccg, max_layers);
                                -0.1 + idx * 0.01
                            })
                            .unwrap_or(-0.1)
                    }))
                    .unwrap_or(0.0);

                let heat_bias = heatmap.layers.get(layer)
                    .map(|lv| {
                        if lv.is_empty() { 0.0 }
                        else {
                            let avg = lv.iter().copied().sum::<f32>() / lv.len() as f32;
                            avg * 0.10
                        }
                    })
                    .unwrap_or(0.0);

                let cid = req.channel_id.min(channels.len().saturating_sub(1));

                let grid_bias =
                    0.35 * ccg.cluster_bias.get(layer).and_then(|v| v.get(cid)).copied().unwrap_or(0.0)
                    + 0.25 * ccg.zone_bias.get(layer).and_then(|v| v.get(cid)).copied().unwrap_or(0.0)
                    + 0.20 * ccg.door_bias.get(layer).and_then(|v| v.get(cid)).copied().unwrap_or(0.0)
                    + 0.20 * ccg.geom_bias.get(layer).and_then(|v| v.get(cid)).copied().unwrap_or(0.0);

                let door_rot =
                    ccg.door_rotation.get(layer).and_then(|v| v.get(cid)).copied().unwrap_or(0) as f32
                    * 0.01;

                let locality_bias = {
                    let mut lb = 0.0;
                    if let Some(ch) = self.last_channel.get(layer).and_then(|c| *c) {
                        if ch == req.channel_id { lb += 0.15; }
                    }
                    if let Some(r) = self.last_row.get(layer).and_then(|r| *r) {
                        if r == req.row { lb += 0.20; }
                    }
                    if let Some(b) = self.last_bank.get(layer).and_then(|b| *b) {
                        if b == req.bank { lb += 0.20; }
                    }
                    lb
                };

                let refresh_penalty =
                    self.refresh_events.get(layer).copied().unwrap_or(0) as f32 * 0.03;
                let ecc_penalty =
                    self.ecc_events.get(layer).copied().unwrap_or(0) as f32 * 0.04;

                let bitdrop_bias =
                    self.entropy_memory.get(layer).copied().unwrap_or(0.0) * 0.06 +
                    self.size_memory.get(layer).copied().unwrap_or(0.0) * 0.06 +
                    self.structure_memory.get(layer).copied().unwrap_or(0.0) * 0.05 +
                    self.numeric_memory.get(layer).copied().unwrap_or(0.0) * 0.05 +
                    self.tunnel_memory.get(layer).copied().unwrap_or(0.0) * 0.05 +
                    self.adaptive_memory.get(layer).copied().unwrap_or(0.0) * 0.04 +
                    self.stability_memory.get(layer).copied().unwrap_or(0.0) * 0.04;

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

        for layer in 0..max_layers {
            let current = req.layer_bias.get(layer).copied().unwrap_or(0.0);
            req.update_layer_bias(layer, current + biases[layer]);
        }
    }
}
