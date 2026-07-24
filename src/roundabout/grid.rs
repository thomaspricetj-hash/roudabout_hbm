#[derive(Debug, Clone)]
pub struct CrossConnectGrid {
    // [layer][channel]
    pub cluster_bias: Vec<Vec<f32>>,
    pub zone_bias: Vec<Vec<f32>>,
    pub door_bias: Vec<Vec<f32>>,
    pub geom_bias: Vec<Vec<f32>>,

    // rotating doors per layer
    pub door_rotation: Vec<Vec<usize>>,

    // multilayer index weights [layer]
    pub index_weights: Vec<f32>,

    // multilayer scratchpad cache [layer][channel]
    pub scratch_cluster: Vec<Vec<f32>>,
    pub scratch_zone: Vec<Vec<f32>>,
    pub scratch_door: Vec<Vec<f32>>,
    pub scratch_geom: Vec<Vec<f32>>,
}

impl CrossConnectGrid {
    pub fn new(layers: usize, channel_count: usize) -> Self {
        let zero_layer = || vec![0.0; channel_count];
        let zero_usize_layer = || vec![0; channel_count];

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
        }
    }

    /// Reinforce a specific layer + channel
    pub fn reinforce(&mut self, layer: usize, ch_id: usize) {
        self.cluster_bias[layer][ch_id] += 0.01;
        self.zone_bias[layer][ch_id] += 0.01;
        self.door_bias[layer][ch_id] += 0.02;
        self.geom_bias[layer][ch_id] += 0.01;
    }

    /// Cool a specific layer + channel
    pub fn cool(&mut self, layer: usize, ch_id: usize) {
        self.cluster_bias[layer][ch_id] *= 0.98;
        self.zone_bias[layer][ch_id] *= 0.98;
        self.door_bias[layer][ch_id] *= 0.97;
        self.geom_bias[layer][ch_id] *= 0.98;
    }

    /// Rotate doors for a given layer (simple round‑robin)
    pub fn rotate_doors(&mut self, layer: usize) {
        let rot = &mut self.door_rotation[layer];
        if !rot.is_empty() {
            rot.rotate_left(1);
        }
    }

    /// Cache multilayer scratchpad values
    pub fn cache_scratch(
        &mut self,
        layer: usize,
        ch_id: usize,
        cluster: f32,
        zone: f32,
        door: f32,
        geom: f32,
    ) {
        self.scratch_cluster[layer][ch_id] = cluster;
        self.scratch_zone[layer][ch_id] = zone;
        self.scratch_door[layer][ch_id] = door;
        self.scratch_geom[layer][ch_id] = geom;
    }

    /// Fused multilayer bias for a channel
    pub fn fused_bias(&self, ch_id: usize) -> f32 {
        let mut acc = 0.0;
        for (layer, w) in self.index_weights.iter().enumerate() {
            acc += w * (
                0.35 * self.cluster_bias[layer][ch_id] +
                0.25 * self.zone_bias[layer][ch_id] +
                0.20 * self.door_bias[layer][ch_id] +
                0.20 * self.geom_bias[layer][ch_id]
            );
        }
        acc
    }
}
