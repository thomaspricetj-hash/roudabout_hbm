use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct CrossConnectGrid {
    // [layer][channel]
    pub cluster_bias: Vec<Vec<f32>>,
    pub zone_bias: Vec<Vec<f32>>,
    pub door_bias: Vec<Vec<f32>>,
    pub geom_bias: Vec<Vec<f32>>,

    pub door_rotation: Vec<Vec<usize>>,
    pub index_weights: Vec<f32>,

    pub scratch_cluster: Vec<Vec<f32>>,
    pub scratch_zone: Vec<Vec<f32>>,
    pub scratch_door: Vec<Vec<f32>>,
    pub scratch_geom: Vec<Vec<f32>>,

    pub row_locality: Vec<Vec<f32>>,
    pub bank_locality: Vec<Vec<f32>>,
    pub channel_locality: Vec<Vec<f32>>,
    pub die_locality: Vec<Vec<f32>>,
    pub stack_locality: Vec<Vec<f32>>,

    pub refresh_penalty: Vec<Vec<f32>>,
    pub ecc_penalty: Vec<Vec<f32>>,

    pub bitdrop_entropy_geom: Vec<Vec<f32>>,
    pub bitdrop_size_geom: Vec<Vec<f32>>,
    pub bitdrop_structure_geom: Vec<Vec<f32>>,
    pub bitdrop_numeric_geom: Vec<Vec<f32>>,
    pub bitdrop_tunnel_geom: Vec<Vec<f32>>,

    // ---------- NEW: Tesla valve directional geometry ----------
    pub valve_forward_geom: Vec<Vec<f32>>,
    pub valve_reverse_geom: Vec<Vec<f32>>,
    pub valve_oscillation_geom: Vec<Vec<f32>>,

    // ---------- NEW: Option‑C DF‑HBM delta‑geometry ----------
    pub delta_cluster: Vec<Vec<f32>>,
    pub delta_zone: Vec<Vec<f32>>,
    pub delta_door: Vec<Vec<f32>>,
    pub delta_geom: Vec<Vec<f32>>,

    pub delta_row_locality: Vec<Vec<f32>>,
    pub delta_bank_locality: Vec<Vec<f32>>,
    pub delta_channel_locality: Vec<Vec<f32>>,
    pub delta_die_locality: Vec<Vec<f32>>,
    pub delta_stack_locality: Vec<Vec<f32>>,

    pub delta_refresh_penalty: Vec<Vec<f32>>,
    pub delta_ecc_penalty: Vec<Vec<f32>>,

    pub delta_bitdrop_entropy: Vec<Vec<f32>>,
    pub delta_bitdrop_size: Vec<Vec<f32>>,
    pub delta_bitdrop_structure: Vec<Vec<f32>>,
    pub delta_bitdrop_numeric: Vec<Vec<f32>>,
    pub delta_bitdrop_tunnel: Vec<Vec<f32>>,

    // ---------- NEW: Tesla valve delta geometry ----------
    pub delta_valve_forward_geom: Vec<Vec<f32>>,
    pub delta_valve_reverse_geom: Vec<Vec<f32>>,
    pub delta_valve_oscillation_geom: Vec<Vec<f32>>,
}

impl CrossConnectGrid {
    pub fn new(layers: usize, count: usize) -> Self {
        let zero_layer = || vec![0.0; count];
        let zero_usize_layer = || vec![0; count];

        Self {
            cluster_bias: (0..layers).map(|_| zero_layer()).collect(),
            zone_bias: (0..layers).map(|_| zero_layer()).collect(),
            door_bias: (0..layers).map(|_| zero_layer()).collect(),
            geom_bias: (0..layers).map(|_| zero_layer()).collect(),

            door_rotation: (0..layers).map(|_| zero_usize_layer()).collect(),
            index_weights: vec![1.0; layers],

            scratch_cluster: (0..layers).map(|_| zero_layer()).collect(),
            scratch_zone: (0..layers).map(|_| zero_layer()).collect(),
            scratch_door: (0..layers).map(|_| zero_layer()).collect(),
            scratch_geom: (0..layers).map(|_| zero_layer()).collect(),

            row_locality: (0..layers).map(|_| zero_layer()).collect(),
            bank_locality: (0..layers).map(|_| zero_layer()).collect(),
            channel_locality: (0..layers).map(|_| zero_layer()).collect(),
            die_locality: (0..layers).map(|_| zero_layer()).collect(),
            stack_locality: (0..layers).map(|_| zero_layer()).collect(),

            refresh_penalty: (0..layers).map(|_| zero_layer()).collect(),
            ecc_penalty: (0..layers).map(|_| zero_layer()).collect(),

            bitdrop_entropy_geom: (0..layers).map(|_| zero_layer()).collect(),
            bitdrop_size_geom: (0..layers).map(|_| zero_layer()).collect(),
            bitdrop_structure_geom: (0..layers).map(|_| zero_layer()).collect(),
            bitdrop_numeric_geom: (0..layers).map(|_| zero_layer()).collect(),
            bitdrop_tunnel_geom: (0..layers).map(|_| zero_layer()).collect(),

            // Tesla valve directional geometry
            valve_forward_geom: (0..layers).map(|_| zero_layer()).collect(),
            valve_reverse_geom: (0..layers).map(|_| zero_layer()).collect(),
            valve_oscillation_geom: (0..layers).map(|_| zero_layer()).collect(),

            delta_cluster: (0..layers).map(|_| zero_layer()).collect(),
            delta_zone: (0..layers).map(|_| zero_layer()).collect(),
            delta_door: (0..layers).map(|_| zero_layer()).collect(),
            delta_geom: (0..layers).map(|_| zero_layer()).collect(),

            delta_row_locality: (0..layers).map(|_| zero_layer()).collect(),
            delta_bank_locality: (0..layers).map(|_| zero_layer()).collect(),
            delta_channel_locality: (0..layers).map(|_| zero_layer()).collect(),
            delta_die_locality: (0..layers).map(|_| zero_layer()).collect(),
            delta_stack_locality: (0..layers).map(|_| zero_layer()).collect(),

            delta_refresh_penalty: (0..layers).map(|_| zero_layer()).collect(),
            delta_ecc_penalty: (0..layers).map(|_| zero_layer()).collect(),

            delta_bitdrop_entropy: (0..layers).map(|_| zero_layer()).collect(),
            delta_bitdrop_size: (0..layers).map(|_| zero_layer()).collect(),
            delta_bitdrop_structure: (0..layers).map(|_| zero_layer()).collect(),
            delta_bitdrop_numeric: (0..layers).map(|_| zero_layer()).collect(),
            delta_bitdrop_tunnel: (0..layers).map(|_| zero_layer()).collect(),

            delta_valve_forward_geom: (0..layers).map(|_| zero_layer()).collect(),
            delta_valve_reverse_geom: (0..layers).map(|_| zero_layer()).collect(),
            delta_valve_oscillation_geom: (0..layers).map(|_| zero_layer()).collect(),
        }
    }

    // -------------------------------------------------------------------------
    // Tesla valve reinforcement
    // -------------------------------------------------------------------------

    pub fn reinforce_valve(&mut self, layer: usize, id: usize, forward: f32, reverse: f32, oscillation: f32) {
        self.valve_forward_geom[layer][id] += forward * 0.04;
        self.valve_reverse_geom[layer][id] += reverse * 0.05;
        self.valve_oscillation_geom[layer][id] += oscillation * 0.06;
    }

    pub fn cool_valve(&mut self, layer: usize, id: usize) {
        self.valve_forward_geom[layer][id] *= 0.95;
        self.valve_reverse_geom[layer][id] *= 0.92;
        self.valve_oscillation_geom[layer][id] *= 0.90;
    }

    // -------------------------------------------------------------------------
    // Reinforce (HBM + BitDrop + Tesla valve)
    // -------------------------------------------------------------------------

    pub fn reinforce(&mut self, layer: usize, id: usize) {
        self.cluster_bias[layer][id] += 0.01;
        self.zone_bias[layer][id] += 0.01;
        self.door_bias[layer][id] += 0.02;
        self.geom_bias[layer][id] += 0.01;

        self.row_locality[layer][id] += 0.03;
        self.bank_locality[layer][id] += 0.03;
        self.channel_locality[layer][id] += 0.02;
        self.die_locality[layer][id] += 0.01;
        self.stack_locality[layer][id] += 0.01;

        self.bitdrop_entropy_geom[layer][id] += 0.02;
        self.bitdrop_size_geom[layer][id] += 0.02;
        self.bitdrop_structure_geom[layer][id] += 0.02;
        self.bitdrop_numeric_geom[layer][id] += 0.02;
        self.bitdrop_tunnel_geom[layer][id] += 0.02;

        // Tesla valve reinforcement
        self.valve_forward_geom[layer][id] += 0.03;
        self.valve_reverse_geom[layer][id] *= 0.95;
        self.valve_oscillation_geom[layer][id] *= 0.95;
    }

    // -------------------------------------------------------------------------
    // Cooling (HBM + BitDrop + Tesla valve)
    // -------------------------------------------------------------------------

    pub fn cool(&mut self, layer: usize, id: usize) {
        self.cluster_bias[layer][id] *= 0.98;
        self.zone_bias[layer][id] *= 0.98;
        self.door_bias[layer][id] *= 0.97;
        self.geom_bias[layer][id] *= 0.98;

        self.row_locality[layer][id] *= 0.95;
        self.bank_locality[layer][id] *= 0.95;
        self.channel_locality[layer][id] *= 0.95;
        self.die_locality[layer][id] *= 0.97;
        self.stack_locality[layer][id] *= 0.97;

        self.refresh_penalty[layer][id] *= 0.90;
        self.ecc_penalty[layer][id] *= 0.90;

        self.bitdrop_entropy_geom[layer][id] *= 0.94;
        self.bitdrop_size_geom[layer][id] *= 0.94;
        self.bitdrop_structure_geom[layer][id] *= 0.94;
        self.bitdrop_numeric_geom[layer][id] *= 0.94;
        self.bitdrop_tunnel_geom[layer][id] *= 0.94;

        // Tesla valve cooling
        self.valve_forward_geom[layer][id] *= 0.95;
        self.valve_reverse_geom[layer][id] *= 0.92;
        self.valve_oscillation_geom[layer][id] *= 0.90;
    }

    // -------------------------------------------------------------------------
    // Fused multilayer bias (HBM + BitDrop + Tesla valve)
    // -------------------------------------------------------------------------

    pub fn fused_bias(&self, id: usize) -> f32 {
        let mut acc = 0.0;

        for (layer, w) in self.index_weights.iter().enumerate() {
            let base =
                0.35 * self.cluster_bias[layer][id] +
                0.25 * self.zone_bias[layer][id] +
                0.20 * self.door_bias[layer][id] +
                0.20 * self.geom_bias[layer][id];

            let locality =
                0.40 * self.row_locality[layer][id] +
                0.35 * self.bank_locality[layer][id] +
                0.25 * self.channel_locality[layer][id] +
                0.20 * self.die_locality[layer][id] +
                0.15 * self.stack_locality[layer][id];

            let penalties =
                -0.30 * self.refresh_penalty[layer][id] +
                -0.25 * self.ecc_penalty[layer][id];

            let bitdrop =
                0.25 * self.bitdrop_entropy_geom[layer][id] +
                0.25 * self.bitdrop_size_geom[layer][id] +
                0.20 * self.bitdrop_structure_geom[layer][id] +
                0.15 * self.bitdrop_numeric_geom[layer][id] +
                0.15 * self.bitdrop_tunnel_geom[layer][id];

            let valve =
                -0.20 * self.valve_forward_geom[layer][id] +
                 0.25 * self.valve_reverse_geom[layer][id] +
                 0.30 * self.valve_oscillation_geom[layer][id];

            acc += w * (base + locality + penalties + bitdrop + valve);
        }

        acc
    }

    // -------------------------------------------------------------------------
    // Delta update (HBM + BitDrop + Tesla valve)
    // -------------------------------------------------------------------------

    pub fn update_deltas_from_prev(&mut self, prev: &CrossConnectGrid) {
        for layer in 0..self.cluster_bias.len() {
            for id in 0..self.cluster_bias[layer].len() {
                self.delta_cluster[layer][id] =
                    self.cluster_bias[layer][id] - prev.cluster_bias[layer][id];
                self.delta_zone[layer][id] =
                    self.zone_bias[layer][id] - prev.zone_bias[layer][id];
                self.delta_door[layer][id] =
                    self.door_bias[layer][id] - prev.door_bias[layer][id];
                self.delta_geom[layer][id] =
                    self.geom_bias[layer][id] - prev.geom_bias[layer][id];

                self.delta_row_locality[layer][id] =
                    self.row_locality[layer][id] - prev.row_locality[layer][id];
                self.delta_bank_locality[layer][id] =
                    self.bank_locality[layer][id] - prev.bank_locality[layer][id];
                self.delta_channel_locality[layer][id] =
                    self.channel_locality[layer][id] - prev.channel_locality[layer][id];
                self.delta_die_locality[layer][id] =
                    self.die_locality[layer][id] - prev.die_locality[layer][id];
                self.delta_stack_locality[layer][id] =
                    self.stack_locality[layer][id] - prev.stack_locality[layer][id];

                self.delta_refresh_penalty[layer][id] =
                    self.refresh_penalty[layer][id] - prev.refresh_penalty[layer][id];
                self.delta_ecc_penalty[layer][id] =
                    self.ecc_penalty[layer][id] - prev.ecc_penalty[layer][id];

                self.delta_bitdrop_entropy[layer][id] =
                    self.bitdrop_entropy_geom[layer][id] - prev.bitdrop_entropy_geom[layer][id];
                self.delta_bitdrop_size[layer][id] =
                    self.bitdrop_size_geom[layer][id] - prev.bitdrop_size_geom[layer][id];
                self.delta_bitdrop_structure[layer][id] =
                    self.bitdrop_structure_geom[layer][id] - prev.bitdrop_structure_geom[layer][id];
                self.delta_bitdrop_numeric[layer][id] =
                    self.bitdrop_numeric_geom[layer][id] - prev.bitdrop_numeric_geom[layer][id];
                self.delta_bitdrop_tunnel[layer][id] =
                    self.bitdrop_tunnel_geom[layer][id] - prev.bitdrop_tunnel_geom[layer][id];

                // Tesla valve delta geometry
                self.delta_valve_forward_geom[layer][id] =
                    self.valve_forward_geom[layer][id] - prev.valve_forward_geom[layer][id];
                self.delta_valve_reverse_geom[layer][id] =
                    self.valve_reverse_geom[layer][id] - prev.valve_reverse_geom[layer][id];
                self.delta_valve_oscillation_geom[layer][id] =
                    self.valve_oscillation_geom[layer][id] - prev.valve_oscillation_geom[layer][id];
            }
        }
    }

    // -------------------------------------------------------------------------
    // Fused delta bias (HBM + BitDrop + Tesla valve)
    // -------------------------------------------------------------------------

    pub fn fused_delta_bias(&self, id: usize) -> f32 {
        let mut acc = 0.0;

        for (layer, w) in self.index_weights.iter().enumerate() {
            let delta_base =
                0.30 * self.delta_cluster[layer][id] +
                0.25 * self.delta_zone[layer][id] +
                0.20 * self.delta_door[layer][id] +
                0.20 * self.delta_geom[layer][id];

            let delta_locality =
                0.35 * self.delta_row_locality[layer][id] +
                0.30 * self.delta_bank_locality[layer][id] +
                0.25 * self.delta_channel_locality[layer][id] +
                0.20 * self.delta_die_locality[layer][id] +
                0.15 * self.delta_stack_locality[layer][id];

            let delta_penalties =
                -0.25 * self.delta_refresh_penalty[layer][id] +
                -0.20 * self.delta_ecc_penalty[layer][id];

            let delta_bitdrop =
                0.25 * self.delta_bitdrop_entropy[layer][id] +
                0.25 * self.delta_bitdrop_size[layer][id] +
                0.20 * self.delta_bitdrop_structure[layer][id] +
                0.15 * self.delta_bitdrop_numeric[layer][id] +
                0.15 * self.delta_bitdrop_tunnel[layer][id];

            let delta_valve =
                -0.18 * self.delta_valve_forward_geom[layer][id] +
                 0.22 * self.delta_valve_reverse_geom[layer][id] +
                 0.28 * self.delta_valve_oscillation_geom[layer][id];

            acc += w * (delta_base + delta_locality + delta_penalties + delta_bitdrop + delta_valve);
        }

        acc
    }
}

