use rayon::prelude::*;
use std::time::Instant;

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

    // Core timing
    pub created_at: Instant,
    pub last_attempt: Instant,
    pub circulations: u32,

    // Multilayer routing state
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
    ) -> Self {
        let now = Instant::now();
        Self {
            id,
            priority,
            kind,
            channel_id,
            bank_id,
            row_addr,
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
    }

    pub fn update_route_score(&mut self, score: f32) {
        self.route_score = score;
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

    /// Parallel multilayer reinforcement update.
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

        // FIX: silence unused variable warning, keep logic intact
        for (_layer, adj) in adjustments.into_iter().enumerate() {
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
}

