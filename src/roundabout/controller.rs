use rayon::prelude::*;

use super::{
    arbitration::ArbitrationEngine,
    channel::HbmChannel,
    heatmap::Heatmap,
    index::RoutingIndex,
    request::HbmRequest,
    grid::CrossConnectGrid,
    scratchpad::Scratchpad,
};

#[derive(Debug)]
pub struct HbmRoundaboutController {
    pub channels: Vec<HbmChannel>,
    pub heatmap: Heatmap,
    pub layers: usize,
    pub arb: ArbitrationEngine,
    pub ccg: CrossConnectGrid,
    pub scratchpad: Scratchpad,
}

impl HbmRoundaboutController {
    pub fn new(channels: Vec<HbmChannel>, layers: usize, decay: f32) -> Self {
        let channel_count = channels.len();

        Self {
            ccg: CrossConnectGrid::new(layers, channel_count),
            channels,
            heatmap: Heatmap::new(layers, channel_count, decay),
            layers,
            arb: ArbitrationEngine::new(),
            scratchpad: Scratchpad::new(layers),
        }
    }

    /// MAX‑tier parallel routing + multilayer Cross‑Connect Grid + tunneling + scratchpad
    pub fn route_request(&mut self, mut req: HbmRequest) -> Option<usize> {
        // touch attempt / circulation tracking
        req.touch_attempt();

        // multilayer heatmap decay
        self.heatmap.decay_step();

        // rotate doors for both heatmap and grid
        for layer in 0..self.layers {
            self.heatmap.rotate_doors(layer);
            self.ccg.rotate_doors(layer);
        }

        // apply scratchpad‑driven multilayer bias (heat + grid + index + failures)
        self.scratchpad.apply_bias_parallel(
            &mut req,
            &self.heatmap,
            &self.ccg,
            &self.channels,
        );

        // parallel arbitration across all channels (heatmap + grid + metrics)
        let best = self.arb.choose_best_channel_parallel(
            &req,
            &self.channels,
            &self.heatmap,
            &self.ccg,
            self.layers,
        );

        if let Some(ch_id) = best {
            req.update_last_exit(Some(ch_id));

            // compute fused heat + grid score for caching
            let fused_heat = self.heatmap.fused_heat(ch_id);
            let fused_heat_grid = self.heatmap.fused_heat_with_grid(ch_id, &self.ccg);

            // reinforce heatmap + grid + scratchpad across all layers for this successful exit
            for layer in 0..self.layers {
                // heatmap reinforcement + scratch cache
                self.heatmap.reinforce_parallel(layer, ch_id);
                self.heatmap.cache_scratch(layer, ch_id, fused_heat_grid);

                // grid reinforcement + scratch cache
                let cluster = self.ccg.cluster_bias[layer][ch_id];
                let zone = self.ccg.zone_bias[layer][ch_id];
                let door = self.ccg.door_bias[layer][ch_id];
                let geom = self.ccg.geom_bias[layer][ch_id];

                self.ccg.reinforce(layer, ch_id);
                self.ccg.cache_scratch(layer, ch_id, cluster, zone, door, geom);

                // scratchpad success history
                self.scratchpad.record_success(layer, ch_id);

                // tunnel reinforcement
                if self.channels[ch_id].is_tunnel {
                    self.channels[ch_id].reinforce_tunnel();
                }
            }

            // update request route score with composite index
            let idx_score = RoutingIndex::composite_index_score(
                &req,
                &self.channels[ch_id],
                &self.heatmap,
                &self.ccg,
                self.layers,
            );
            req.update_route_score(idx_score);
            req.update_heat_signature(fused_heat);

            Some(ch_id)
        } else {
            // no exit → circulation
            req.circulations += 1;

            // cool heatmap + grid + scratchpad across all layers for the current channel_id
            for layer in 0..self.layers {
                self.heatmap.cool_parallel(layer, req.channel_id);
                self.ccg.cool(layer, req.channel_id);
                self.scratchpad.record_failure(layer);

                if let Some(ch) = self.channels.get_mut(req.channel_id) {
                    if ch.is_tunnel {
                        ch.cool_tunnel();
                    }
                }
            }

            None
        }
    }
}

