use rayon::prelude::*;
use std::time::Instant;

// BitDrop‑V2 integration
use bitdrop_v2::{
    compress_with_profile,
    estimate_entropy,
    looks_like_text_or_structured,
    looks_like_u32_counter,
    gpu_available,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPriority {
    High,
    Standard,
    Low,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
    Load,
    Store,
    Prefetch,
}

#[derive(Debug, Clone)]
pub struct HbmRequest {
    pub id: u64,
    pub priority: RequestPriority,
    pub kind: RequestKind,

    pub channel_id: usize,
    pub bank_id: usize,
    pub row_addr: u64,

    // NEW: explicit HBM locality fields
    pub row: u32,
    pub bank: u32,

    pub created_at: Instant,
    pub last_attempt: Instant,
    pub circulations: u32,

    pub last_exit_channel: Option<usize>,
    pub heat_signature: f32,
    pub route_score: f32,
    pub escalations: u32,

    pub layer_scores: Vec<f32>,
    pub layer_heat: Vec<f32>,
    pub layer_bias: Vec<f32>,
    pub layer_exit_history: Vec<Option<usize>>,

    pub adaptive_weight: f32,
    pub stability_factor: f32,

    // NEW: HBM‑aware request state
    pub locality_score: f32,
    pub refresh_pressure: f32,
    pub ecc_pressure: f32,

    pub is_tunnel_escalated: bool,
    pub tunnel_preference: f32,
    pub tunnel_history: Vec<Option<usize>>,
    pub tunnel_heat: f32,
    pub tunnel_score: f32,

    // BitDrop‑V2 integration
    pub payload: Vec<u8>,
    pub payload_profile: String,
    pub payload_entropy: f32,
    pub payload_is_structured: bool,
    pub payload_is_numeric_counter: bool,
    pub payload_compressed_size: usize,
}

impl HbmRequest {
    pub fn new(
        id: u64,
        channel_id: usize,
        bank_id: usize,
        row_addr: u64,
        priority: RequestPriority,
        kind: RequestKind,
        layers: usize,
        raw_payload: Vec<u8>,
        profile: &str,
    ) -> Self {
        let now = Instant::now();

        // BitDrop profiling
        let entropy = estimate_entropy(&raw_payload);
        let is_structured = looks_like_text_or_structured(&raw_payload);
        let is_numeric = looks_like_u32_counter(&raw_payload);

        // Choose effective profile
        let effective_profile = if profile.is_empty() {
            if is_numeric {
                "numbin"
            } else if is_structured {
                "pymid"
            } else if gpu_available() {
                "adaptive"
            } else {
                "fast"
            }
        } else {
            profile
        };

        let compressed = compress_with_profile(&raw_payload, effective_profile);
        let compressed_size = compressed.len();

        Self {
            id,
            priority,
            kind,
            channel_id,
            bank_id,
            row_addr,
            row: (row_addr & 0xFFFF_FFFF) as u32,
            bank: bank_id as u32,
            created_at: now,
            last_attempt: now,
            circulations: 0,
            last_exit_channel: None,
            heat_signature: 0.0,
            route_score: 0.0,
            escalations: 0,

            layer_scores: vec![0.0; layers],
            layer_heat: vec![0.0; layers],
            layer_bias: vec![0.0; layers],
            layer_exit_history: vec![None; layers],

            adaptive_weight: 1.0,
            stability_factor: 1.0,

            locality_score: 0.0,
            refresh_pressure: 0.0,
            ecc_pressure: 0.0,

            is_tunnel_escalated: false,
            tunnel_preference: 0.0,
            tunnel_history: vec![None; layers],
            tunnel_heat: 0.0,
            tunnel_score: 0.0,

            payload: compressed,
            payload_profile: effective_profile.to_string(),
            payload_entropy: entropy,
            payload_is_structured: is_structured,
            payload_is_numeric_counter: is_numeric,
            payload_compressed_size: compressed_size,
        }
    }

    pub fn escalate(&mut self) {
        self.priority = match self.priority {
            RequestPriority::High => RequestPriority::High,
            RequestPriority::Standard => RequestPriority::High,
            RequestPriority::Low => RequestPriority::Standard,
        };

        self.escalations += 1;
        self.adaptive_weight *= 1.1;

        self.is_tunnel_escalated = true;
        self.tunnel_preference += 0.05;
    }

    pub fn reinforce_tunnel(&mut self, success: bool) {
        let delta = if success { 0.04 } else { -0.04 };

        self.tunnel_preference = (self.tunnel_preference + delta).clamp(-1.0, 2.0);
        self.stability_factor = (self.stability_factor + delta).clamp(0.1, 2.0);
        self.tunnel_heat = (self.tunnel_heat + delta).clamp(0.0, 5.0);
    }

    pub fn update_tunnel_score(&mut self, score: f32) {
        self.tunnel_score = score;
    }

    pub fn update_tunnel_exit(&mut self, layer: usize, exit_id: Option<usize>) {
        if layer < self.tunnel_history.len() {
            self.tunnel_history[layer] = exit_id;
        }
    }

    pub fn payload_size_bias(&self) -> f32 {
        let s = self.payload_compressed_size as f32;
        (1_000_000.0 / (s.max(64.0))).min(10.0)
    }

    pub fn payload_entropy_bias(&self) -> f32 {
        (8.0 - self.payload_entropy).clamp(-4.0, 4.0)
    }

    pub fn payload_structure_bias(&self) -> f32 {
        if self.payload_is_structured {
            1.5
        } else {
            0.0
        }
    }

    pub fn payload_numeric_bias(&self) -> f32 {
        if self.payload_is_numeric_counter {
            2.0
        } else {
            0.0
        }
    }

    pub fn update_route_score(&mut self, score: f32) {
        let size_bias = self.payload_size_bias();
        let entropy_bias = self.payload_entropy_bias();
        let struct_bias = self.payload_structure_bias();
        let numeric_bias = self.payload_numeric_bias();

        self.route_score = score
            + size_bias * 0.05
            + entropy_bias * 0.03
            + struct_bias * 0.02
            + numeric_bias * 0.02;
    }

    pub fn update_last_exit(&mut self, channel: Option<usize>) {
        self.last_exit_channel = channel;
    }

    pub fn update_heat_signature(&mut self, heat: f32) {
        self.heat_signature = heat;
    }

    pub fn update_layer_score(&mut self, layer: usize, score: f32) {
        if layer < self.layer_scores.len() {
            self.layer_scores[layer] = score;
        }
    }

    pub fn update_layer_heat(&mut self, layer: usize, heat: f32) {
        if layer < self.layer_heat.len() {
            self.layer_heat[layer] = heat;
        }
    }

    pub fn update_layer_bias(&mut self, layer: usize, bias: f32) {
        if layer < self.layer_bias.len() {
            self.layer_bias[layer] = bias;
        }
    }

    pub fn update_layer_exit(&mut self, layer: usize, channel: Option<usize>) {
        if layer < self.layer_exit_history.len() {
            self.layer_exit_history[layer] = channel;
        }
    }

    pub fn reinforce_parallel(&mut self, success: bool) {
        let adjustments: Vec<f32> = (0..self.layer_scores.len())
            .into_par_iter()
            .map(|layer| {
                let base = if success { 0.05 } else { -0.05 };
                let heat = self.layer_heat[layer] * 0.02;
                let score = self.layer_scores[layer] * 0.01;
                base + heat + score
            })
            .collect();

        for adj in adjustments {
            self.stability_factor = (self.stability_factor + adj).clamp(0.1, 2.0);
        }
    }

    pub fn reinforce(&mut self, success: bool) {
        if success {
            self.stability_factor = (self.stability_factor + 0.05).min(2.0);
        } else {
            self.stability_factor = (self.stability_factor - 0.05).max(0.1);
        }
    }

    // NEW: update locality score from heatmap/grid
    pub fn update_locality_score(&mut self, score: f32) {
        self.locality_score = score.clamp(-1.0, 1.0);
    }

    // NEW: update refresh/ecc pressures (for arbitration priority escalation)
    pub fn update_refresh_pressure(&mut self, pressure: f32) {
        self.refresh_pressure = pressure.clamp(0.0, 1.0);
    }

    pub fn update_ecc_pressure(&mut self, pressure: f32) {
        self.ecc_pressure = pressure.clamp(0.0, 1.0);
    }

    pub fn touch_attempt(&mut self) {
        self.last_attempt = Instant::now();
        self.circulations = self.circulations.saturating_add(1);
    }
}
