use rayon::prelude::*;

use super::{
    arbitration::ArbitrationEngine,
    channel::HbmChannel,
    grid::CrossConnectGrid,
    heatmap::Heatmap,
    index::RoutingIndex,
    request::HbmRequest,
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

// ---------- structure‑aware helpers ----------

fn payload_structure_lane_bias(req: &HbmRequest, ch: &HbmChannel) -> f32 {
    let mut bias = 0.0;

    if req.payload_is_structured {
        let heat = ch.heat_affinity;
        let stability = ch.metrics.stability_score;
        bias += (stability * 0.08) - (heat * 0.05);
    }

    if req.payload_is_numeric_counter {
        if ch.is_tunnel {
            bias += 0.10;
        }
        if ch.metrics.row_conflicts < 2 {
            bias += 0.06;
        }
    }

    let size = req.payload_compressed_size as f32;
    if size < 256.0 {
        let usage = ch.metrics.load / ch.max_load;
        bias += (1.0 - usage) * 0.05;
    }

    bias
}

fn payload_structure_tunnel_bias(req: &HbmRequest, ch: &HbmChannel) -> f32 {
    if !ch.is_tunnel {
        return 0.0;
    }

    let mut bias = 0.0;

    if req.payload_is_structured {
        let geom = ch.metrics.geometry_score;
        let heat = ch.heat_affinity;
        bias += geom * 0.07;
        bias -= heat * 0.04;
    }

    if req.payload_is_numeric_counter {
        let stability = ch.metrics.stability_score;
        bias += stability * 0.09;
    }

    bias
}

fn payload_structure_locality_bias(req: &HbmRequest, scratchpad: &Scratchpad) -> f32 {
    let mut bias = 0.0;

    for layer in 0..scratchpad.layers {
        if let Some(last_row) = scratchpad.last_row[layer] {
            if last_row == req.row {
                bias += 0.03;
            }
        }
        if let Some(last_bank) = scratchpad.last_bank[layer] {
            if last_bank == req.bank {
                bias += 0.03;
            }
        }
    }

    if req.payload_is_structured {
        bias *= 1.3;
    }

    bias
}

fn compute_dynamic_fiber_count(
    heatmap: &Heatmap,
    scratchpad: &Scratchpad,
    channels: &[HbmChannel],
    req: &HbmRequest,
    layers: usize,
) -> usize {
    let mut heat_volatility = 0.0;
    let max_layers = layers.min(heatmap.layers.len());
    for layer in 0..max_layers {
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

    let mut tunnel_pressure = 0.0;
    for ch in channels {
        if ch.is_tunnel {
            tunnel_pressure += 0.1;
        }
    }

    let mut bank_conflict = 0.0;
    let max_sp_layers = layers.min(scratchpad.layers);
    for layer in 0..max_sp_layers {
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

// ---------- rules of the road helpers ----------

fn no_u_turn_penalty(scratchpad: &Scratchpad, ch: usize, window: usize) -> f32 {
    let max_layers = scratchpad.layers;
    for layer in 0..max_layers {
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
    let max_layers = scratchpad.layers;
    for layer in 0..max_layers {
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
        if let Some(layer_vec) = heatmap.layers.get(layer) {
            if let Some(&heat) = layer_vec.get(ch) {
                if heat > 0.85 {
                    penalty -= 0.25;
                }
            }
        }
    }
    penalty
}

fn lane_discipline_penalty(_req: &HbmRequest, _channel: &HbmChannel) -> f32 {
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

// ---------- cascade context ----------

struct CascadeContext<'a> {
    pub heatmap: &'a Heatmap,
    pub ccg: &'a CrossConnectGrid,
    pub scratchpad: &'a Scratchpad,
    pub channels: &'a [HbmChannel],
    pub layers: usize,
}

// ---------- compression shape (cube / pyramid) ----------

#[derive(Copy, Clone, Debug)]
enum CompressionShape {
    Cube,
    Pyramid,
}

fn choose_compression_shape(
    req: &HbmRequest,
    ch: &HbmChannel,
    heatmap: &Heatmap,
    layers: usize,
) -> CompressionShape {
    let mut avg_heat = 0.0;
    let mut count = 0.0;
    let max_layers = layers.min(heatmap.layers.len());
    for layer in 0..max_layers {
        if let Some(layer_vec) = heatmap.layers.get(layer) {
            for h in layer_vec {
                avg_heat += *h;
                count += 1.0;
            }
        }
    }
    if count > 0.0 {
        avg_heat /= count;
    }

    if req.payload_is_structured || req.payload_is_numeric_counter {
        CompressionShape::Pyramid
    } else if avg_heat > 0.85 {
        CompressionShape::Pyramid
    } else {
        CompressionShape::Cube
    }
}

// ---------- micro‑structors (vmax: 15 structors) ----------

struct LocalityStructor;
struct GeomStructor;
struct TunnelStructor;
struct ThermalStructor;
struct BankStructor;

struct RoadRulesStructor<'a> {
    scratchpad: &'a Scratchpad,
    heatmap: &'a Heatmap,
}

struct BitDropStructor; // BitDrop‑V5

struct PredictiveStructor<'a> {
    scratchpad: &'a Scratchpad,
}

struct AdaptiveWeightsStructor<'a> {
    scratchpad: &'a Scratchpad,
    heatmap: &'a Heatmap,
}

struct TunnelGeomStructor;
struct HistoryStructor;
struct TemporalStructor;
struct CollapseStructor;
struct EntropyStructor;
struct LoadStructor;

struct MicroScores {
    pub locality: f32,
    pub geom: f32,
    pub tunnel: f32,
    pub thermal: f32,
    pub bank: f32,
    pub road: f32,
    pub bitdrop: f32,
    pub predictive: f32,
    pub tunnel_geom: f32,
    pub history: f32,
    pub temporal: f32,
    pub collapse: f32,
    pub entropy: f32,
    pub load: f32,
}

struct MicroWeights {
    pub w_locality: f32,
    pub w_geom: f32,
    pub w_tunnel: f32,
    pub w_thermal: f32,
    pub w_bank: f32,
    pub w_road: f32,
    pub w_bitdrop: f32,
    pub w_predictive: f32,
    pub w_tunnel_geom: f32,
    pub w_history: f32,
    pub w_temporal: f32,
    pub w_collapse: f32,
    pub w_entropy: f32,
    pub w_load: f32,
    pub w_base: f32,
}

impl MicroScores {
    pub fn fused(&self, w: &MicroWeights) -> f32 {
        self.locality * w.w_locality
            + self.geom * w.w_geom
            + self.tunnel * w.w_tunnel
            + self.thermal * w.w_thermal
            - self.bank * w.w_bank
            + self.road * w.w_road
            + self.bitdrop * w.w_bitdrop
            + self.predictive * w.w_predictive
            + self.tunnel_geom * w.w_tunnel_geom
            + self.history * w.w_history
            + self.temporal * w.w_temporal
            + self.collapse * w.w_collapse
            + self.entropy * w.w_entropy
            + self.load * w.w_load
    }
}

// ---------- self‑evolving temporal‑adaptive weights (vmax) ----------

impl AdaptiveWeightsStructor<'_> {
    pub fn weights(&self, ctx: &CascadeContext, req: &HbmRequest) -> MicroWeights {
        let mut w = MicroWeights {
            w_locality: 1.0,
            w_geom: 0.14,
            w_tunnel: 1.0,
            w_thermal: 0.08,
            w_bank: 1.0,
            w_road: 1.0,
            w_bitdrop: 1.0,
            w_predictive: 1.0,
            w_tunnel_geom: 0.30,
            w_history: 0.70,
            w_temporal: 0.50,
            w_collapse: 0.60,
            w_entropy: 0.50,
            w_load: 0.50,
            w_base: 1.0,
        };

        let mut failure_sum = 0.0;
        let mut refresh_sum = 0.0;
        let mut ecc_sum = 0.0;
        let max_sp_layers = self.scratchpad.layers;
        for layer in 0..max_sp_layers {
            failure_sum += self.scratchpad.failures[layer] as f32;
            refresh_sum += self.scratchpad.refresh_events[layer] as f32;
            ecc_sum += self.scratchpad.ecc_events[layer] as f32;
        }

        let mut avg_heat = 0.0;
        let mut heat_count = 0.0;
        let max_layers = ctx.layers.min(self.heatmap.layers.len());
        for layer in 0..max_layers {
            if let Some(layer_vec) = self.heatmap.layers.get(layer) {
                for h in layer_vec {
                    avg_heat += *h;
                    heat_count += 1.0;
                }
            }
        }
        if heat_count > 0.0 {
            avg_heat /= heat_count;
        }

        let failure_factor = (failure_sum * 0.01).min(2.0);
        let conflict_factor = ((refresh_sum + ecc_sum) * 0.01).min(2.0);
        let heat_factor = (avg_heat * 0.7).min(2.0);
        let temporal_factor = (req.circulations as f32 * 0.12).min(2.5);

        // self‑evolving: lean harder on what historically fixes problems
        w.w_bank *= 1.0 + conflict_factor * 0.7;
        w.w_road *= 1.0 + failure_factor * 0.6;
        w.w_predictive *= 1.0 + failure_factor * 0.5;
        w.w_locality *= 1.0 + heat_factor * 0.4;
        w.w_history *= 1.0 + failure_factor * 0.5;
        w.w_temporal *= 1.0 + temporal_factor * 0.6;
        w.w_entropy *= 1.0 + heat_factor * 0.4;
        w.w_load *= 1.0 + conflict_factor * 0.4;

        if req.payload_is_structured {
            w.w_geom *= 1.6;
            w.w_bitdrop *= 1.5;
            w.w_tunnel_geom *= 1.4;
            w.w_collapse *= 1.4;
            w.w_entropy *= 1.3;
        }

        if req.payload_is_numeric_counter {
            w.w_tunnel *= 1.6;
            w.w_predictive *= 1.5;
            w.w_tunnel_geom *= 1.5;
            w.w_temporal *= 1.3;
        }

        if avg_heat > 0.85 {
            w.w_base *= 0.75;
            w.w_bank *= 1.3;
            w.w_thermal *= 1.3;
            w.w_entropy *= 1.2;
        }

        w
    }
}

// ---------- micro‑structor implementations ----------

impl LocalityStructor {
    pub fn score(ctx: &CascadeContext, req: &HbmRequest, _ch: usize) -> f32 {
        let mut s = 0.0;
        let max_sp_layers = ctx.scratchpad.layers;
        for layer in 0..max_sp_layers {
            if let Some(last_ch) = ctx.scratchpad.last_channel[layer] {
                if last_ch == req.channel_id {
                    s += 0.05;
                }
            }
            if let Some(last_row) = ctx.scratchpad.last_row[layer] {
                if last_row == req.row {
                    s += 0.07;
                }
            }
            if let Some(last_bank) = ctx.scratchpad.last_bank[layer] {
                if last_bank == req.bank {
                    s += 0.07;
                }
            }
        }
        s
    }
}

impl GeomStructor {
    pub fn score(ctx: &CascadeContext, ch: usize) -> f32 {
        let mut geom_score = 0.0;
        let max_layers = ctx.layers.min(ctx.ccg.cluster_bias.len());
        for layer in 0..max_layers {
            let cluster = ctx
                .ccg
                .cluster_bias
                .get(layer)
                .and_then(|v| v.get(ch))
                .copied()
                .unwrap_or(0.0);
            let zone = ctx
                .ccg
                .zone_bias
                .get(layer)
                .and_then(|v| v.get(ch))
                .copied()
                .unwrap_or(0.0);
            let door = ctx
                .ccg
                .door_bias
                .get(layer)
                .and_then(|v| v.get(ch))
                .copied()
                .unwrap_or(0.0);
            let geom = ctx
                .ccg
                .geom_bias
                .get(layer)
                .and_then(|v| v.get(ch))
                .copied()
                .unwrap_or(0.0);

            geom_score += 0.25 * cluster
                + 0.25 * zone
                + 0.25 * door
                + 0.25 * geom;
        }
        geom_score / max_layers.max(1) as f32
    }
}

impl TunnelStructor {
    pub fn score(ctx: &CascadeContext, ch: usize) -> f32 {
        if ctx.channels[ch].is_tunnel {
            0.20
        } else {
            0.0
        }
    }
}

impl ThermalStructor {
    pub fn score(ctx: &CascadeContext, ch: usize) -> f32 {
        let mut thermal_geom = 0.0;
        let max_layers = ctx.layers.min(ctx.heatmap.layers.len()).min(ctx.ccg.cluster_bias.len());
        for layer in 0..max_layers {
            let heat = ctx
                .heatmap
                .layers
                .get(layer)
                .and_then(|v| v.get(ch))
                .copied()
                .unwrap_or(0.0);
            let cluster = ctx
                .ccg
                .cluster_bias
                .get(layer)
                .and_then(|v| v.get(ch))
                .copied()
                .unwrap_or(0.0);
            let zone = ctx
                .ccg
                .zone_bias
                .get(layer)
                .and_then(|v| v.get(ch))
                .copied()
                .unwrap_or(0.0);
            let door = ctx
                .ccg
                .door_bias
                .get(layer)
                .and_then(|v| v.get(ch))
                .copied()
                .unwrap_or(0.0);
            let geom = ctx
                .ccg
                .geom_bias
                .get(layer)
                .and_then(|v| v.get(ch))
                .copied()
                .unwrap_or(0.0);

            thermal_geom +=
                heat * (0.25 * cluster + 0.25 * zone + 0.25 * door + 0.25 * geom);
        }
        thermal_geom / max_layers.max(1) as f32
    }
}

impl BankStructor {
    pub fn score(ctx: &CascadeContext, req: &HbmRequest) -> f32 {
        let mut bank_conflict = 0.0;
        let max_sp_layers = ctx.scratchpad.layers;
        for layer in 0..max_sp_layers {
            if let Some(last_bank) = ctx.scratchpad.last_bank[layer] {
                if last_bank == req.bank {
                    bank_conflict += 0.12;
                }
            }
            bank_conflict += ctx.scratchpad.refresh_events[layer] as f32 * 0.03;
            bank_conflict += ctx.scratchpad.ecc_events[layer] as f32 * 0.04;
        }
        bank_conflict
    }
}

impl<'a> RoadRulesStructor<'a> {
    pub fn score(&self, req: &HbmRequest, ch: usize, channel: &HbmChannel) -> f32 {
        let mut s = 0.0;
        s += no_u_turn_penalty(self.scratchpad, ch, 3);
        s += stability_yield_bias(self.scratchpad, ch, 0.05);
        s += emergency_preempt_bonus(req);
        s += construction_zone_penalty(self.heatmap, ch);
        s += lane_discipline_penalty(req, channel);
        s += speed_limit_penalty(channel);
        s
    }
}

// BitDrop‑V5: tuned for vmax cascade collapse patterns, now cube/pyramid‑aware
impl BitDropStructor {
    pub fn score(
        req: &HbmRequest,
        ch: &HbmChannel,
        shape: CompressionShape,
    ) -> f32 {
        let mut s = 0.0;

        let shape_factor = match shape {
            CompressionShape::Cube => 1.0,
            CompressionShape::Pyramid => 1.15,
        };

        s += ch.pair_score_component() * 1.2 * shape_factor;
        s += ch.payload_size_bias * 0.07 * shape_factor;
        s += ch.payload_entropy_bias * 0.05 * shape_factor;
        s += ch.payload_structure_bias * 0.04 * shape_factor;
        s += ch.payload_numeric_bias * 0.04 * shape_factor;
        s += payload_structure_lane_bias(req, ch) * 1.0 * shape_factor;
        s += payload_structure_tunnel_bias(req, ch) * 1.1 * shape_factor;

        s
    }
}

impl<'a> PredictiveStructor<'a> {
    pub fn score(&self, ch: usize) -> f32 {
        let mut predictive_bonus = 0.0;
        let max_sp_layers = self.scratchpad.layers;
        for layer in 0..max_sp_layers {
            if let Some(last_exit) = self.scratchpad.history[layer][0] {
                if last_exit == ch {
                    predictive_bonus += 0.05;
                }
            }
            predictive_bonus -= self.scratchpad.failures[layer] as f32 * 0.005;
        }
        predictive_bonus
    }
}

impl TunnelGeomStructor {
    pub fn score(ctx: &CascadeContext, ch: usize) -> f32 {
        if !ctx.channels[ch].is_tunnel {
            return 0.0;
        }

        let mut s = 0.0;
        let max_layers = ctx.layers.min(ctx.heatmap.layers.len()).min(ctx.ccg.cluster_bias.len());
        for layer in 0..max_layers {
            let heat = ctx
                .heatmap
                .layers
                .get(layer)
                .and_then(|v| v.get(ch))
                .copied()
                .unwrap_or(0.0);
            let cluster = ctx
                .ccg
                .cluster_bias
                .get(layer)
                .and_then(|v| v.get(ch))
                .copied()
                .unwrap_or(0.0);
            let zone = ctx
                .ccg
                .zone_bias
                .get(layer)
                .and_then(|v| v.get(ch))
                .copied()
                .unwrap_or(0.0);
            let door = ctx
                .ccg
                .door_bias
                .get(layer)
                .and_then(|v| v.get(ch))
                .copied()
                .unwrap_or(0.0);
            let geom = ctx
                .ccg
                .geom_bias
                .get(layer)
                .and_then(|v| v.get(ch))
                .copied()
                .unwrap_or(0.0);

            let geom_mix = 0.25 * cluster + 0.25 * zone + 0.25 * door + 0.25 * geom;
            s += (1.0 - heat) * geom_mix;
        }

        s / max_layers.max(1) as f32
    }
}

impl HistoryStructor {
    pub fn score(ctx: &CascadeContext, ch: usize) -> f32 {
        let mut s = 0.0;
        let max_sp_layers = ctx.scratchpad.layers;
        for layer in 0..max_sp_layers {
            if let Some(last_exit) = ctx.scratchpad.last_channel[layer] {
                if last_exit == ch {
                    s += 0.04;
                }
            }
        }
        s
    }
}

impl TemporalStructor {
    pub fn score(_ctx: &CascadeContext, req: &HbmRequest) -> f32 {
        let attempts = req.circulations as f32;
        (attempts * 0.05).min(0.6)
    }
}

impl CollapseStructor {
    pub fn score(_ctx: &CascadeContext, _req: &HbmRequest, ch: &HbmChannel) -> f32 {
        let entropy = ch.payload_entropy_bias;
        let size = ch.payload_size_bias;
        (1.0 - entropy).max(0.0) * 0.06 + (1.0 - size).max(0.0) * 0.06
    }
}

impl EntropyStructor {
    pub fn score(_ctx: &CascadeContext, _req: &HbmRequest, ch: &HbmChannel) -> f32 {
        let entropy = ch.payload_entropy_bias;
        (1.0 - entropy).max(0.0) * 0.08
    }
}

impl LoadStructor {
    pub fn score(_ctx: &CascadeContext, _req: &HbmRequest, ch: &HbmChannel) -> f32 {
        let usage = ch.metrics.load / ch.max_load;
        (1.0 - usage).max(0.0) * 0.08
    }
}

// ---------- fiber evaluation (vmax) ----------

fn evaluate_fiber(
    ctx: &CascadeContext,
    road_rules: &RoadRulesStructor,
    predictive: &PredictiveStructor,
    adaptive: &AdaptiveWeightsStructor,
    req: &HbmRequest,
    ch: usize,
) -> CascadeFiberResult {
    let fused_heat = ctx.heatmap.fused_heat(ch);
    let fused_heat_grid = ctx.heatmap.fused_heat_with_grid(ch, ctx.ccg);

    let base = RoutingIndex::composite_index_score(
        req,
        &ctx.channels[ch],
        ctx.heatmap,
        ctx.ccg,
        ctx.layers,
    );

    let locality = LocalityStructor::score(ctx, req, ch);
    let geom = GeomStructor::score(ctx, ch);
    let tunnel = TunnelStructor::score(ctx, ch);
    let thermal = ThermalStructor::score(ctx, ch);
    let bank = BankStructor::score(ctx, req);
    let road = road_rules.score(req, ch, &ctx.channels[ch]);

    let shape = choose_compression_shape(req, &ctx.channels[ch], ctx.heatmap, ctx.layers);
    let bitdrop = BitDropStructor::score(req, &ctx.channels[ch], shape);

    let predictive_score = predictive.score(ch);
    let tunnel_geom = TunnelGeomStructor::score(ctx, ch);
    let history = HistoryStructor::score(ctx, ch);
    let temporal = TemporalStructor::score(ctx, req);
    let collapse = CollapseStructor::score(ctx, req, &ctx.channels[ch]);
    let entropy = EntropyStructor::score(ctx, req, &ctx.channels[ch]);
    let load = LoadStructor::score(ctx, req, &ctx.channels[ch]);

    let micro = MicroScores {
        locality,
        geom,
        tunnel,
        thermal,
        bank,
        road,
        bitdrop,
        predictive: predictive_score,
        tunnel_geom,
        history,
        temporal,
        collapse,
        entropy,
        load,
    };

    let weights = adaptive.weights(ctx, req);

    let route_score = base * weights.w_base + micro.fused(&weights);

    CascadeFiberResult {
        ch_id: Some(ch),
        route_score,
        fused_heat,
        fused_heat_grid,
        locality_score: locality,
        tunnel_score: tunnel,
        geom_score: geom,
    }
}

// ---------- impl HbmRoundaboutController (vmax) ----------

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

        let max_layers = self
            .layers
            .min(self.heatmap.layers.len())
            .min(self.ccg.cluster_bias.len())
            .min(self.scratchpad.layers);

        for layer in 0..max_layers {
            self.heatmap.rotate_doors(layer);
            self.ccg.rotate_doors(layer);
        }

        for ch in &mut self.channels {
            ch.update_bitdrop_biases(&req.payload, Some(&req.payload_profile));
        }

        let locality_bias = payload_structure_locality_bias(&req, &self.scratchpad);
        req.update_locality_score(req.locality_score + locality_bias);

        self.scratchpad
            .apply_bias_parallel(&mut req, &self.heatmap, &self.ccg, &self.channels);

        let fiber_count = compute_dynamic_fiber_count(
            &self.heatmap,
            &self.scratchpad,
            &self.channels,
            &req,
            max_layers,
        );

        let ctx = CascadeContext {
            heatmap: &self.heatmap,
            ccg: &self.ccg,
            scratchpad: &self.scratchpad,
            channels: &self.channels,
            layers: max_layers,
        };

        let road_rules = RoadRulesStructor {
            scratchpad: &self.scratchpad,
            heatmap: &self.heatmap,
        };

        let predictive = PredictiveStructor {
            scratchpad: &self.scratchpad,
        };

        let adaptive = AdaptiveWeightsStructor {
            scratchpad: &self.scratchpad,
            heatmap: &self.heatmap,
        };

        let arb = &self.arb;

        let fiber_results: Vec<CascadeFiberResult> = (0..fiber_count)
            .into_par_iter()
            .map(|fiber_id| {
                let mut fiber_req = req.clone();
                let jitter = (fiber_id as f32) * 0.01;
                for layer in 0..max_layers {
                    let current = fiber_req.layer_bias.get(layer).copied().unwrap_or(0.0);
                    fiber_req.update_layer_bias(layer, current + jitter);
                }

                let ch_id = arb.choose_best_channel_parallel(
                    &fiber_req,
                    ctx.channels,
                    ctx.heatmap,
                    ctx.ccg,
                    ctx.layers,
                );

                if let Some(ch) = ch_id {
                    evaluate_fiber(&ctx, &road_rules, &predictive, &adaptive, &fiber_req, ch)
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

            let _avg_fused_heat = fiber_results
                .iter()
                .filter(|f| f.ch_id.is_some())
                .map(|f| f.fused_heat)
                .sum::<f32>()
                / valid_count.max(1.0);

            let avg_fused_heat_grid = fiber_results
                .iter()
                .filter(|f| f.ch_id.is_some())
                .map(|f| f.fused_heat_grid)
                .sum::<f32>()
                / valid_count.max(1.0);

            let _avg_locality = fiber_results
                .iter()
                .filter(|f| f.ch_id.is_some())
                .map(|f| f.locality_score)
                .sum::<f32>()
                / valid_count.max(1.0);

            let _avg_tunnel = fiber_results
                .iter()
                .filter(|f| f.ch_id.is_some())
                .map(|f| f.tunnel_score)
                .sum::<f32>()
                / valid_count.max(1.0);

            let _avg_geom = fiber_results
                .iter()
                .filter(|f| f.ch_id.is_some())
                .map(|f| f.geom_score)
                .sum::<f32>()
                / valid_count.max(1.0);

            for layer in 0..max_layers {
                self.heatmap.reinforce_parallel(layer, ch_id);
                self.heatmap.cache_scratch(layer, ch_id, avg_fused_heat_grid);

                let cluster = self
                    .ccg
                    .cluster_bias
                    .get(layer)
                    .and_then(|v| v.get(ch_id))
                    .copied()
                    .unwrap_or(0.0);
                let zone = self
                    .ccg
                    .zone_bias
                    .get(layer)
                    .and_then(|v| v.get(ch_id))
                    .copied()
                    .unwrap_or(0.0);
                let door = self
                    .ccg
                    .door_bias
                    .get(layer)
                    .and_then(|v| v.get(ch_id))
                    .copied()
                    .unwrap_or(0.0);
                let geom = self
                    .ccg
                    .geom_bias
                    .get(layer)
                    .and_then(|v| v.get(ch_id))
                    .copied()
                    .unwrap_or(0.0);

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

            for layer in 0..max_layers {
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
