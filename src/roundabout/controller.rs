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
struct CascadeFiberResult {
    pub ch_id: Option<usize>,
    pub route_score: f32,
    pub fused_heat: f32,
    pub fused_heat_grid: f32,
    pub locality_score: f32,
    pub tunnel_score: f32,
    pub geom_score: f32,
}

#[derive(Debug)]
pub struct HbmRoundaboutController {
    pub channels: Vec<HbmChannel>,
    pub heatmap: Heatmap,
    pub layers: usize,
    pub arb: ArbitrationEngine,
    pub ccg: CrossConnectGrid,
    pub scratchpad: Scratchpad,
}

fn compute_dynamic_fiber_count(
    heatmap: &Heatmap,
    scratchpad: &Scratchpad,
    channels: &[HbmChannel],
    req: &HbmRequest,
    layers: usize,
) -> usize {
    // Heat volatility
    let mut heat_volatility = 0.0;
    for layer in 0..layers {
        if let Some(layer_vec) = heatmap.layers.get(layer) {
            if layer_vec.len() > 1 {
                let avg = layer_vec.iter().copied().sum::<f32>() / layer_vec.len() as f32;
                let var = layer_vec
                    .iter()
                    .map(|v| (v - avg).abs())
                    .sum::<f32>()
                    / layer_vec.len() as f32;
                heat_volatility += var;
            }
        }
    }

    // Tunnel pressure
    let mut tunnel_pressure = 0.0;
    for ch in channels {
        if ch.is_tunnel {
            tunnel_pressure += 0.1;
        }
    }

    // Bank conflict probability
    let mut bank_conflict = 0.0;
    for layer in 0..layers {
        if let Some(last_bank) = scratchpad.last_bank[layer] {
            if last_bank == req.bank {
                bank_conflict += 0.2;
            }
        }
        bank_conflict += scratchpad.refresh_events[layer] as f32 * 0.05;
        bank_conflict += scratchpad.ecc_events[layer] as f32 * 0.07;
    }

    let score = heat_volatility * 0.4 + tunnel_pressure * 0.3 + bank_conflict * 0.3;

    let fibers = 3 + (score * 10.0) as usize;
    fibers.clamp(3, 12)
}

// ---------- RULES OF THE ROAD HELPERS (local, no extra structs) ----------

fn no_u_turn_penalty(scratchpad: &Scratchpad, ch: usize, window: usize) -> f32 {
    for layer in 0..scratchpad.layers {
        for i in 0..window {
            if let Some(prev) = scratchpad.history[layer][i] {
                if prev == ch {
                    return -0.10;
                }
            }
        }
    }
    0.0
}

fn stability_yield_bias(scratchpad: &Scratchpad, ch: usize, bias: f32) -> f32 {
    let mut b = 0.0;
    for layer in 0..scratchpad.layers {
        if let Some(last_exit) = scratchpad.last_channel[layer] {
            if last_exit == ch {
                b += bias;
            }
        }
    }
    b
}

fn emergency_preempt_bonus(req: &HbmRequest) -> f32 {
    match req.priority {
        super::request::RequestPriority::High => 0.20,
        _ => 0.0,
    }
}

fn construction_zone_penalty(heatmap: &Heatmap, ch: usize) -> f32 {
    let mut penalty = 0.0;
    for layer in 0..heatmap.layers.len() {
        let heat = heatmap.layers[layer][ch];
        if heat > 0.85 {
            penalty -= 0.25;
        }
    }
    penalty
}

fn lane_discipline_penalty(_req: &HbmRequest, _channel: &HbmChannel) -> f32 {
    // Placeholder for future lane_type integration:
    // if req.lane_type != channel.lane_type { -0.10 } else { 0.0 }
    0.0
}

fn speed_limit_penalty(channel: &HbmChannel) -> f32 {
    let usage = channel.metrics.load / channel.max_load;
    if usage > 0.90 {
        -0.15
    } else {
        0.0
    }
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

    pub fn route_request(&mut self, mut req: HbmRequest) -> Option<usize> {
        req.touch_attempt();
        self.heatmap.decay_step();

        for layer in 0..self.layers {
            self.heatmap.rotate_doors(layer);
            self.ccg.rotate_doors(layer);
        }

        // BitDrop‑V2: update channel‑side payload biases for this request
        for ch in &mut self.channels {
            ch.update_bitdrop_biases(&req.payload, Some(&req.payload_profile));
        }

        self.scratchpad
            .apply_bias_parallel(&mut req, &self.heatmap, &self.ccg, &self.channels);

        let fiber_count = compute_dynamic_fiber_count(
            &self.heatmap,
            &self.scratchpad,
            &self.channels,
            &req,
            self.layers,
        );

        let heatmap = &self.heatmap;
        let ccg = &self.ccg;
        let scratchpad = &self.scratchpad;
        let channels = &self.channels;
        let layers = self.layers;
        let arb = &self.arb;

        let fiber_results: Vec<CascadeFiberResult> = (0..fiber_count)
            .into_par_iter()
            .map(|fiber_id| {
                let mut fiber_req = req.clone();
                let jitter = (fiber_id as f32) * 0.01;
                for layer in 0..layers {
                    fiber_req.update_layer_bias(layer, fiber_req.layer_bias[layer] + jitter);
                }

                let ch_id = arb.choose_best_channel_parallel(
                    &fiber_req,
                    channels,
                    heatmap,
                    ccg,
                    layers,
                );

                if let Some(ch) = ch_id {
                    let fused_heat = heatmap.fused_heat(ch);
                    let fused_heat_grid = heatmap.fused_heat_with_grid(ch, ccg);

                    let mut route_score = RoutingIndex::composite_index_score(
                        &fiber_req,
                        &channels[ch],
                        heatmap,
                        ccg,
                        layers,
                    );

                    // Locality fusion
                    let mut locality_score = 0.0;
                    for layer in 0..scratchpad.layers {
                        if let Some(last_ch) = scratchpad.last_channel[layer] {
                            if last_ch == fiber_req.channel_id {
                                locality_score += 0.05;
                            }
                        }
                        if let Some(last_row) = scratchpad.last_row[layer] {
                            if last_row == fiber_req.row {
                                locality_score += 0.07;
                            }
                        }
                        if let Some(last_bank) = scratchpad.last_bank[layer] {
                            if last_bank == fiber_req.bank {
                                locality_score += 0.07;
                            }
                        }
                    }

                    // Geometry fusion
                    let mut geom_score = 0.0;
                    for layer in 0..layers {
                        let cluster = ccg.cluster_bias[layer][ch];
                        let zone = ccg.zone_bias[layer][ch];
                        let door = ccg.door_bias[layer][ch];
                        let geom = ccg.geom_bias[layer][ch];

                        geom_score += 0.25 * cluster
                            + 0.25 * zone
                            + 0.25 * door
                            + 0.25 * geom;
                    }
                    geom_score /= layers as f32;

                    // Tunnel physics
                    let tunnel_score = if channels[ch].is_tunnel { 0.20 } else { 0.0 };

                    // Temporal heatmap forecasting
                    let mut forecast_heat = 0.0;
                    for layer in 0..layers {
                        if let Some(layer_vec) = heatmap.layers.get(layer) {
                            if !layer_vec.is_empty() {
                                let avg_heat =
                                    layer_vec.iter().copied().sum::<f32>() / layer_vec.len() as f32;
                                forecast_heat += avg_heat;
                            }
                        }
                    }
                    if layers > 0 {
                        forecast_heat /= layers as f32;
                    }

                    // Predictive arbitration
                    let mut predictive_bonus = 0.0;
                    for layer in 0..scratchpad.layers {
                        if let Some(last_exit) = scratchpad.history[layer][0] {
                            if last_exit == ch {
                                predictive_bonus += 0.05;
                            }
                        }
                        predictive_bonus -= scratchpad.failures[layer] as f32 * 0.005;
                    }

                    // Row‑hammer avoidance
                    let mut rowhammer_penalty = 0.0;
                    for layer in 0..scratchpad.layers {
                        if let Some(last_row) = scratchpad.last_row[layer] {
                            if last_row == fiber_req.row {
                                rowhammer_penalty += 0.05;
                            }
                        }
                        rowhammer_penalty += scratchpad.refresh_events[layer] as f32 * 0.01;
                        rowhammer_penalty += scratchpad.ecc_events[layer] as f32 * 0.015;
                    }

                    // Temporal tunnel forecasting
                    let mut tunnel_forecast = 0.0;
                    if channels[ch].is_tunnel {
                        tunnel_forecast = fused_heat * 0.6 + fused_heat_grid * 0.4;
                    }

                    // Bank‑conflict predictor
                    let mut bank_conflict_score = 0.0;
                    for layer in 0..scratchpad.layers {
                        if let Some(last_bank) = scratchpad.last_bank[layer] {
                            if last_bank == fiber_req.bank {
                                bank_conflict_score += 0.12;
                            }
                        }
                        bank_conflict_score += scratchpad.refresh_events[layer] as f32 * 0.03;
                        bank_conflict_score += scratchpad.ecc_events[layer] as f32 * 0.04;
                    }

                    // Thermal‑geometry coupling
                    let mut thermal_geom = 0.0;
                    for layer in 0..layers {
                        let heat = heatmap.layers[layer][ch];
                        let cluster = ccg.cluster_bias[layer][ch];
                        let zone = ccg.zone_bias[layer][ch];
                        let door = ccg.door_bias[layer][ch];
                        let geom = ccg.geom_bias[layer][ch];

                        thermal_geom +=
                            heat * (0.25 * cluster + 0.25 * zone + 0.25 * door + 0.25 * geom);
                    }
                    thermal_geom /= layers as f32;

                    // Existing scoring fusion
                    route_score += locality_score;
                    route_score += tunnel_score;
                    route_score += geom_score * 0.10;
                    route_score += forecast_heat * 0.02;
                    route_score += predictive_bonus;
                    route_score += tunnel_forecast * 0.15;
                    route_score += thermal_geom * 0.05;
                    route_score -= rowhammer_penalty;
                    route_score -= bank_conflict_score;

                    // ---------- RULES OF THE ROAD APPLIED HERE ----------

                    route_score += no_u_turn_penalty(scratchpad, ch, 3);
                    route_score += stability_yield_bias(scratchpad, ch, 0.05);
                    route_score += emergency_preempt_bonus(&fiber_req);
                    route_score += construction_zone_penalty(heatmap, ch);
                    route_score += lane_discipline_penalty(&fiber_req, &channels[ch]);
                    route_score += speed_limit_penalty(&channels[ch]);

                    // grouped‑pair routing contribution
                    route_score += channels[ch].pair_score_component();

                    // BitDrop payload geometry contribution (channel‑side biases)
                    route_score += channels[ch].payload_size_bias * 0.05;
                    route_score += channels[ch].payload_entropy_bias * 0.03;
                    route_score += channels[ch].payload_structure_bias * 0.02;
                    route_score += channels[ch].payload_numeric_bias * 0.02;

                    // ----------------------------------------------------

                    CascadeFiberResult {
                        ch_id: Some(ch),
                        route_score,
                        fused_heat,
                        fused_heat_grid,
                        locality_score,
                        tunnel_score,
                        geom_score,
                    }
                } else {
                    CascadeFiberResult {
                        ch_id: None,
                        route_score: f32::MIN,
                        fused_heat: 0.0,
                        fused_heat_grid: 0.0,
                        locality_score: 0.0,
                        tunnel_score: 0.0,
                        geom_score: 0.0,
                    }
                }
            })
            .collect();

        let best_fiber = fiber_results
            .iter()
            .filter(|f| f.ch_id.is_some())
            .max_by(|a, b| {
                a.route_score
                    .partial_cmp(&b.route_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some(fiber) = best_fiber {
            let ch_id = fiber.ch_id.unwrap();

            req.update_last_exit(Some(ch_id));
            req.update_route_score(fiber.route_score);
            req.update_heat_signature(fiber.fused_heat);

            let valid_count = fiber_results.iter().filter(|f| f.ch_id.is_some()).count() as f32;

            let avg_fused_heat = fiber_results
                .iter()
                .filter(|f| f.ch_id.is_some())
                .map(|f| f.fused_heat)
                .sum::<f32>()
                / valid_count;

            let avg_fused_heat_grid = fiber_results
                .iter()
                .filter(|f| f.ch_id.is_some())
                .map(|f| f.fused_heat_grid)
                .sum::<f32>()
                / valid_count;

            let _avg_locality = fiber_results
                .iter()
                .filter(|f| f.ch_id.is_some())
                .map(|f| f.locality_score)
                .sum::<f32>()
                / valid_count;

            let _avg_tunnel = fiber_results
                .iter()
                .filter(|f| f.ch_id.is_some())
                .map(|f| f.tunnel_score)
                .sum::<f32>()
                / valid_count;

            let _avg_geom = fiber_results
                .iter()
                .filter(|f| f.ch_id.is_some())
                .map(|f| f.geom_score)
                .sum::<f32>()
                / valid_count;

            for layer in 0..self.layers {
                self.heatmap.reinforce_parallel(layer, ch_id);
                self.heatmap.cache_scratch(layer, ch_id, avg_fused_heat_grid);

                let cluster = self.ccg.cluster_bias[layer][ch_id];
                let zone = self.ccg.zone_bias[layer][ch_id];
                let door = self.ccg.door_bias[layer][ch_id];
                let geom = self.ccg.geom_bias[layer][ch_id];

                self.ccg.reinforce(layer, ch_id);
                self.ccg
                    .cache_scratch(layer, ch_id, cluster, zone, door, geom);

                self.scratchpad.record_success(layer, ch_id);

                if self.channels[ch_id].is_tunnel {
                    self.channels[ch_id].reinforce_tunnel();
                }
            }

            Some(ch_id)
        } else {
            req.circulations += 1;

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

