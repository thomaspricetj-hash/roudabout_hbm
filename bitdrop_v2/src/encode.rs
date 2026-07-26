use std::collections::HashMap;

use crate::blocks::{
    flatten_from_cubes,
    inverse_orientation,
    lift_to_cubes,
    rotate_cube_data,
    Cube,
    CubeId,
};
use crate::cluster::cluster_cubes;
use crate::collapse::collapse_cluster;
use crate::container::{pack_cubes_and_log, unpack_cubes_and_log};
use crate::metrics::choose_best_orientation;
use crate::transform::{apply_shift, inverse_shift, Transform, TransformLog};

pub struct BitDrop3DEngine {
    pub block_shape: (usize, usize, usize),
    pub max_layers: u16,
}

impl BitDrop3DEngine {
    pub fn new(block_shape: (usize, usize, usize), max_layers: u16) -> Self {
        Self { block_shape, max_layers }
    }

    pub fn encode(&self, payload: &[u8]) -> Vec<u8> {
        // DEBUG: first 16 bytes entering encode
        println!(
            "RUST_ENCODE_FIRST16: {:?}",
            &payload[..16.min(payload.len())]
        );

        let (mut cubes, original_len) = lift_to_cubes(payload, self.block_shape);

        let mut log = TransformLog::new();
        let mut next_id: u32 = cubes.len() as u32;
        let mut current_layer: u16 = 0;

        // Orientation selection + rotation
        for cube in cubes.iter_mut() {
            let best_ori = choose_best_orientation(cube);
            if best_ori.0 != 0 {
                let rotated = rotate_cube_data(cube.shape, &cube.data, best_ori);
                cube.data = rotated;

                log.push(Transform::Rotate {
                    cube_id: cube.id,
                    layer: cube.layer,
                    orientation: best_ori,
                });
            }
        }

        // Layered collapse
        while current_layer < self.max_layers {
            let clusters = cluster_cubes(&cubes);
            let mut next_cubes = Vec::new();

            for cluster in clusters {
                let collapsed =
                    collapse_cluster(cluster, current_layer + 1, &mut next_id, &mut log);
                next_cubes.extend(collapsed);
            }

            if next_cubes.len() == cubes.len() {
                break;
            }

            cubes = next_cubes;
            current_layer += 1;
        }

        let out = pack_cubes_and_log(&cubes, original_len, &log);

        // DEBUG: first 16 bytes of encoded blob
        println!(
            "RUST_ENCODE_OUTPUT_FIRST16: {:?}",
            &out[..16.min(out.len())]
        );

        out
    }

    pub fn decode(&self, blob: &[u8]) -> Vec<u8> {
        // DEBUG: first 16 bytes entering decode
        println!(
            "RUST_DECODE_INPUT_FIRST16: {:?}",
            &blob[..16.min(blob.len())]
        );

        let (final_cubes, original_len, log) = unpack_cubes_and_log(blob);

        let mut cubes: HashMap<CubeId, Cube> = HashMap::with_capacity(final_cubes.len());
        for c in final_cubes {
            cubes.insert(c.id, c);
        }

        // Apply inverse transforms in reverse order
        for t in log.transforms.iter().rev() {
            match t {
                Transform::DropLayer { .. } => {
                    // no-op for now; layer is informational in this engine
                }

                Transform::Merge {
                    new_cube_id,
                    members,
                    offsets,
                    original_positions,
                    original_shapes,
                    original_layers,
                    ..
                } => {
                    let merged = match cubes.remove(new_cube_id) {
                        Some(c) => c,
                        None => {
                            eprintln!(
                                "WARN: merged cube {:?} missing during inverse merge",
                                new_cube_id
                            );
                            continue;
                        }
                    };

                    let data = merged.data;

                    for idx in 0..members.len() {
                        let start = offsets[idx] as usize;
                        let end = if idx + 1 < offsets.len() {
                            offsets[idx + 1] as usize
                        } else {
                            data.len()
                        };

                        if start > end || end > data.len() {
                            eprintln!(
                                "WARN: invalid slice range in inverse merge: start={}, end={}, len={}",
                                start,
                                end,
                                data.len()
                            );
                            continue;
                        }

                        let slice = data[start..end].to_vec();

                        let pos = original_positions[idx];
                        let shape = original_shapes[idx];
                        let layer = original_layers[idx];
                        let cube_id = members[idx];

                        let cube = Cube::new(cube_id, layer, pos, shape, slice);
                        cubes.insert(cube_id, cube);
                    }
                }

                Transform::Shift { cube_id, dx, dy, dz, .. } => {
                    if let Some(c) = cubes.get_mut(cube_id) {
                        let (ix, iy, iz) = inverse_shift(*dx, *dy, *dz);
                        c.pos = apply_shift(c.pos, ix, iy, iz);
                    } else {
                        eprintln!("WARN: cube {:?} missing during inverse shift", cube_id);
                    }
                }

                Transform::Rotate { cube_id, orientation, .. } => {
                    if let Some(c) = cubes.get_mut(cube_id) {
                        let inv = inverse_orientation(*orientation);
                        let rotated = rotate_cube_data(c.shape, &c.data, inv);
                        c.data = rotated;
                    } else {
                        eprintln!("WARN: cube {:?} missing during inverse rotate", cube_id);
                    }
                }
            }
        }

        let all_cubes: Vec<Cube> = cubes.into_values().collect();
        let out = flatten_from_cubes(&all_cubes, original_len);

        // DEBUG: first 16 bytes of decoded output
        println!(
            "RUST_DECODE_OUTPUT_FIRST16: {:?}",
            &out[..16.min(out.len())]
        );

        out
    }
}
