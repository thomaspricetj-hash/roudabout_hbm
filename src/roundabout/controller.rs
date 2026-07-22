use rayon::prelude::*;

use super::{
    arbitration::ArbitrationEngine,
    channel::HbmChannel,
    heatmap::Heatmap,
    index::RoutingIndex,
    request::HbmRequest,
};

#[derive(Debug)]
pub struct HbmRoundaboutController {
    pub channels: Vec<HbmChannel>,
    pub heatmap: Heatmap,
    pub layers: usize,
    pub arb: ArbitrationEngine,
}

impl HbmRoundaboutController {
    pub fn new(channels: Vec<HbmChannel>, layers: usize, decay: f32) -> Self {
        let channel_count = channels.len();

        Self {
            channels,
            heatmap: Heatmap::new(layers, channel_count, decay),
            layers,
            arb: ArbitrationEngine::new(),
        }
    }

    /// MAX‑tier parallel routing
    pub fn route_request(&mut self, mut req: HbmRequest) -> Option<usize> {
        // Parallel multilayer heatmap decay
        self.heatmap.decay_step();

        // Compute scores for all channels in parallel
        let results: Vec<(usize, f32)> = self
            .channels
            .par_iter()
            .filter_map(|ch| {
                if !ch.can_accept(req.bank_id) {
                    return None;
                }

                // Parallel multilayer index scoring
                let score = RoutingIndex::score_channel_parallel(
                    &req,
                    ch,
                    &self.heatmap,
                    self.layers,
                );

                Some((ch.id, score))
            })
            .collect();

        // Find best channel (lowest score)
        let best = results
            .into_iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((ch_id, _)) = best {
            req.update_last_exit(Some(ch_id));

            // Reinforce heatmap for successful exit
            self.heatmap.reinforce_parallel(0, ch_id);

            Some(ch_id)
        } else {
            // No exit → circulation
            req.circulations += 1;

            // Cool heatmap for failure
            for layer in 0..self.layers {
                self.heatmap.cool_parallel(layer, req.channel_id);
            }

            None
        }
    }
}
