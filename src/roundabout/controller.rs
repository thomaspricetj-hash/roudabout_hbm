use rayon::prelude::*;

use super::{
    arbitration::ArbitrationEngine,
    channel::HbmChannel,
    heatmap::Heatmap,
    index::RoutingIndex,
    request::HbmRequest,
    grid::CrossConnectGrid,
};

#[derive(Debug)]
pub struct HbmRoundaboutController {
    pub channels: Vec<HbmChannel>,
    pub heatmap: Heatmap,
    pub layers: usize,
    pub arb: ArbitrationEngine,
    pub ccg: CrossConnectGrid,
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
        }
    }

    /// MAX‑tier parallel routing + multilayer Cross‑Connect Grid + tunneling
    pub fn route_request(&mut self, mut req: HbmRequest) -> Option<usize> {
        // multilayer heatmap decay
        self.heatmap.decay_step();

        // compute scores for all channels in parallel
        let results: Vec<(usize, f32)> = self
            .channels
            .par_iter()
            .filter_map(|ch| {
                if !ch.can_accept(req.bank_id) {
                    return None;
                }

                // base multilayer index score
                let base_score = RoutingIndex::score_channel_parallel(
                    &req,
                    ch,
                    &self.heatmap,
                    self.layers,
                );

                // fused multilayer grid bias
                let fused_bias = self.ccg.fused_bias(ch.id);

                // NEW: tunnel scoring
                let tunnel_score = if ch.is_tunnel {
                    let tm = &ch.metrics;


                    let mut score = 0.0;

                    score += (tm.tunnel_latency_ms / 100.0) * 0.10;
                    score += (tm.tunnel_jitter_ms / 100.0) * 0.10;
                    score += tm.tunnel_congestion_level * 0.15;
                    score += (1.0 - tm.tunnel_stability_score) * 0.20;
                    score -= (1.0 - tm.tunnel_loss_rate) * 0.10;

                    // tunnel bias + topology bias blending
                    score += ch.tunnel_bias * 0.10;
                    score += (1.0 - ch.tunnel_reliability) * 0.15;
                    score += fused_bias * 0.10;

                    Some(score)
                } else {
                    Some(0.0)
                }?;

                let final_score = base_score - fused_bias + tunnel_score;

                Some((ch.id, final_score))
            })
            .collect();

        // best channel (lowest score)
        let best = results
            .into_iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((ch_id, _)) = best {
            req.update_last_exit(Some(ch_id));

            // reinforce heatmap + grid across all layers for this successful exit
            for layer in 0..self.layers {
                self.heatmap.reinforce_parallel(layer, ch_id);
                self.ccg.reinforce(layer, ch_id);

                // NEW: tunnel reinforcement
                if self.channels[ch_id].is_tunnel {
                    self.channels[ch_id].reinforce_tunnel();
                }
            }

            Some(ch_id)
        } else {
            // no exit → circulation
            req.circulations += 1;

            // cool heatmap + grid across all layers for failed channel
            for layer in 0..self.layers {
                self.heatmap.cool_parallel(layer, req.channel_id);
                self.ccg.cool(layer, req.channel_id);

                // NEW: tunnel cooling
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


