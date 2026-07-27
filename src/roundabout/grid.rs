use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct CrossConnectGrid {
    // [layer][channel]
    pub cluster_bias: Vec<Vec<f32>>,   // HBM stack-level bias
    pub zone_bias: Vec<Vec<f32>>,      // die-level bias
    pub door_bias: Vec<Vec<f32>>,      // bank-level bias
    pub geom_bias: Vec<Vec<f32>>,      // row/channel locality bias

    // rotating doors per layer [layer][channel]
    pub door_rotation: Vec<Vec<usize>>,

    // multilayer index weights [layer]
    pub index_weights: Vec<f32>,

    // multilayer scratchpad cache [layer][channel]
    pub scratch_cluster: Vec<Vec<f32>>,
    pub scratch_zone: Vec<Vec<f32>>,
    pub scratch_door: Vec<Vec<f32>>,
    pub scratch_geom: Vec<Vec<f32>>,

    // NEW: HBM geometry layers
    pub row_locality: Vec<Vec<f32>>,       // row distance / conflict locality
    pub bank_locality: Vec<Vec<f32>>,      // bank distance / busy locality
    pub channel_locality: Vec<Vec<f32>>,   // channel distance / saturation locality
    pub die_locality: Vec<Vec<f32>>,       // die distance (HBM stack)
    pub stack_locality: Vec<Vec<f32>>,     // stack-level locality

    // NEW: refresh/ECC geometry penalties
    pub refresh_penalty: Vec<Vec<f32>>,
    pub ecc_penalty: Vec<Vec<f32>>,

    // NEW: BitDrop geometry layers
    pub bitdrop_entropy_geom: Vec<Vec<f32>>,     // entropy → geometry coupling
    pub bitdrop_size_geom: Vec<Vec<f32>>,        // compressed size → zone/cluster
    pub bitdrop_structure_geom: Vec<Vec<f32>>,   // structured payload → door/geom
    pub bitdrop_numeric_geom: Vec<Vec<f32>>,     // numeric-counter → locality
    pub bitdrop_tunnel_geom: Vec<Vec<f32>>,      // tunnel physics → geometry
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

            // NEW: HBM geometry layers
            row_locality: (0..layers).map(|_| zero_layer()).collect(),
            bank_locality: (0..layers).map(|_| zero_layer()).collect(),
            channel_locality: (0..layers).map(|_| zero_layer()).collect(),
            die_locality: (0..layers).map(|_| zero_layer()).collect(),
            stack_locality: (0..layers).map(|_| zero_layer()).collect(),

            // NEW: refresh/ECC penalties
            refresh_penalty: (0..layers).map(|_| zero_layer()).collect(),
            ecc_penalty: (0..layers).map(|_| zero_layer()).collect(),

            // NEW: BitDrop geometry layers
            bitdrop_entropy_geom: (0..layers).map(|_| zero_layer()).collect(),
            bitdrop_size_geom: (0..layers).map(|_| zero_layer()).collect(),
            bitdrop_structure_geom: (0..layers).map(|_| zero_layer()).collect(),
            bitdrop_numeric_geom: (0..layers).map(|_| zero_layer()).collect(),
            bitdrop_tunnel_geom: (0..layers).map(|_| zero_layer()).collect(),
        }
    }

    /// Reinforce a specific layer + id (successful channel selection).
    /// Tuned for HBM locality + BitDrop geometry.
    pub fn reinforce(&mut self, layer: usize, id: usize) {
        // HBM locality
        self.cluster_bias[layer][id] += 0.01;
        self.zone_bias[layer][id] += 0.01;
        self.door_bias[layer][id] += 0.02;
        self.geom_bias[layer][id] += 0.01;

        self.row_locality[layer][id] += 0.03;
        self.bank_locality[layer][id] += 0.03;
        self.channel_locality[layer][id] += 0.02;
        self.die_locality[layer][id] += 0.01;
        self.stack_locality[layer][id] += 0.01;

        // BitDrop geometry reinforcement
        self.bitdrop_entropy_geom[layer][id] += 0.02;
        self.bitdrop_size_geom[layer][id] += 0.02;
        self.bitdrop_structure_geom[layer][id] += 0.02;
        self.bitdrop_numeric_geom[layer][id] += 0.02;
        self.bitdrop_tunnel_geom[layer][id] += 0.02;
    }

    /// Cool a specific layer + id (failed/avoided channel).
    /// Tuned for HBM locality + BitDrop geometry.
    pub fn cool(&mut self, layer: usize, id: usize) {
        // HBM locality cooling
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

        // BitDrop geometry cooling
        self.bitdrop_entropy_geom[layer][id] *= 0.94;
        self.bitdrop_size_geom[layer][id] *= 0.94;
        self.bitdrop_structure_geom[layer][id] *= 0.94;
        self.bitdrop_numeric_geom[layer][id] *= 0.94;
        self.bitdrop_tunnel_geom[layer][id] *= 0.94;
    }

    /// Rotate doors for a given layer (BitDrop-aware)
    pub fn rotate_doors(&mut self, layer: usize) {
        let rot = &mut self.door_rotation[layer];
        if !rot.is_empty() {
            rot.rotate_left(1);
        }
    }

    /// Cache multilayer scratchpad values for a given id.
    pub fn cache_scratch(
        &mut self,
        layer: usize,
        id: usize,
        cluster: f32,
        zone: f32,
        door: f32,
        geom: f32,
    ) {
        self.scratch_cluster[layer][id] = cluster;
        self.scratch_zone[layer][id] = zone;
        self.scratch_door[layer][id] = door;
        self.scratch_geom[layer][id] = geom;
    }

    /// NEW: cache HBM geometry locality
    pub fn cache_locality(
        &mut self,
        layer: usize,
        id: usize,
        row: f32,
        bank: f32,
        channel: f32,
        die: f32,
        stack: f32,
    ) {
        self.row_locality[layer][id] = row;
        self.bank_locality[layer][id] = bank;
        self.channel_locality[layer][id] = channel;
        self.die_locality[layer][id] = die;
        self.stack_locality[layer][id] = stack;
    }

    /// NEW: cache refresh/ECC penalties
    pub fn cache_penalties(
        &mut self,
        layer: usize,
        id: usize,
        refresh: f32,
        ecc: f32,
    ) {
        self.refresh_penalty[layer][id] = refresh;
        self.ecc_penalty[layer][id] = ecc;
    }

    /// NEW: cache BitDrop geometry contributions
    pub fn cache_bitdrop_geom(
        &mut self,
        layer: usize,
        id: usize,
        entropy: f32,
        size: f32,
        structure: f32,
        numeric: f32,
        tunnel: f32,
    ) {
        self.bitdrop_entropy_geom[layer][id] = entropy;
        self.bitdrop_size_geom[layer][id] = size;
        self.bitdrop_structure_geom[layer][id] = structure;
        self.bitdrop_numeric_geom[layer][id] = numeric;
        self.bitdrop_tunnel_geom[layer][id] = tunnel;
    }

    /// Fused multilayer bias for an id (HBM + BitDrop geometry)
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

            acc += w * (base + locality + penalties + bitdrop);
        }

        acc
    }
}

