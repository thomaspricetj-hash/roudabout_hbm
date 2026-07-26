// container.rs

use crate::blocks::{Cube, CubeId, CubePos, CubeData, Orientation};
use crate::transform::{PatternTag, Transform, TransformLog};
use std::io::{Cursor, Read};

const MAGIC: &[u8; 8] = b"BDROP3D\0";
const VERSION: u32 = 3;

// ------------------------------------------------------------
// Micro‑helpers
// ------------------------------------------------------------
#[inline] fn write_u16(w: &mut Vec<u8>, v: u16) { w.extend_from_slice(&v.to_le_bytes()); }
#[inline] fn write_u32(w: &mut Vec<u8>, v: u32) { w.extend_from_slice(&v.to_le_bytes()); }
#[inline] fn write_u64(w: &mut Vec<u8>, v: u64) { w.extend_from_slice(&v.to_le_bytes()); }
#[inline] fn write_i8(w: &mut Vec<u8>, v: i8) { w.push(v as u8); }
#[inline] fn write_u8(w: &mut Vec<u8>, v: u8) { w.push(v); }

#[inline] fn read_u16(r: &mut Cursor<&[u8]>) -> u16 { let mut b=[0;2]; r.read_exact(&mut b).unwrap(); u16::from_le_bytes(b) }
#[inline] fn read_u32(r: &mut Cursor<&[u8]>) -> u32 { let mut b=[0;4]; r.read_exact(&mut b).unwrap(); u32::from_le_bytes(b) }
#[inline] fn read_u64(r: &mut Cursor<&[u8]>) -> u64 { let mut b=[0;8]; r.read_exact(&mut b).unwrap(); u64::from_le_bytes(b) }
#[inline] fn read_i8(r: &mut Cursor<&[u8]>) -> i8 { let mut b=[0;1]; r.read_exact(&mut b).unwrap(); b[0] as i8 }
#[inline] fn read_u8(r: &mut Cursor<&[u8]>) -> u8 { let mut b=[0;1]; r.read_exact(&mut b).unwrap(); b[0] }

/// ============================================================
/// PACK
/// ============================================================
pub fn pack_cubes_and_log(
    cubes: &[Cube],
    original_len: usize,
    log: &TransformLog,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        64 + cubes.len() * 64 + log.transforms.len() * 64 + original_len
    );

    // Header
    out.extend_from_slice(MAGIC);
    write_u32(&mut out, VERSION);

    // Group cubes by layer
    let mut layer_map: std::collections::BTreeMap<u16, Vec<&Cube>> =
        std::collections::BTreeMap::new();
    for c in cubes {
        layer_map.entry(c.layer).or_default().push(c);
    }

    write_u32(&mut out, layer_map.len() as u32);
    write_u64(&mut out, original_len as u64);

    // Pattern buffer (engine may fill later)
    let pattern_entries: Vec<(u32, Vec<u8>)> = Vec::new();

    // Raw data section
    let mut data_section = Vec::with_capacity(original_len);

    // ------------------------------------------------------------
    // Write cube metadata + raw/ref data
    // ------------------------------------------------------------
    for (layer, list) in &layer_map {
        write_u16(&mut out, *layer);
        write_u32(&mut out, list.len() as u32);

        for cube in list.iter().copied() {
            let CubeId(id) = cube.id;
            write_u32(&mut out, id);
            write_u32(&mut out, cube.pos.x as u32);
            write_u32(&mut out, cube.pos.y as u32);
            write_u32(&mut out, cube.pos.z as u32);
            write_u32(&mut out, cube.shape.0 as u32);
            write_u32(&mut out, cube.shape.1 as u32);
            write_u32(&mut out, cube.shape.2 as u32);

            write_u8(&mut out, cube.quant_bits);
            write_u8(&mut out, cube.quant_vmin);
            write_u8(&mut out, cube.quant_vmax);

            match &cube.data {
                CubeData::Raw(bytes) => {
                    write_u8(&mut out, 0); // raw
                    let offset = data_section.len() as u64;
                    let len = bytes.len() as u32;
                    write_u64(&mut out, offset);
                    write_u32(&mut out, len);
                    data_section.extend_from_slice(bytes);
                }
                CubeData::Ref(ref_id) => {
                    write_u8(&mut out, 1); // reference
                    write_u32(&mut out, *ref_id);
                }
            }
        }
    }

    // ------------------------------------------------------------
    // Write transform log
    // ------------------------------------------------------------
    write_u32(&mut out, log.transforms.len() as u32);

    for t in &log.transforms {
        match t {
            Transform::Rotate { cube_id, layer, orientation } => {
                write_u8(&mut out, 1);
                write_u32(&mut out, cube_id.0);
                write_u16(&mut out, *layer);
                write_u8(&mut out, orientation.0);
            }

            Transform::Shift { cube_id, layer, dx, dy, dz } => {
                write_u8(&mut out, 2);
                write_u32(&mut out, cube_id.0);
                write_u16(&mut out, *layer);
                write_i8(&mut out, *dx);
                write_i8(&mut out, *dy);
                write_i8(&mut out, *dz);
            }

            Transform::Merge {
                new_cube_id,
                layer_from,
                layer_to,
                members,
                offsets,
                original_positions,
                original_shapes,
                original_layers,
            } => {
                write_u8(&mut out, 3);
                write_u32(&mut out, new_cube_id.0);
                write_u16(&mut out, *layer_from);
                write_u16(&mut out, *layer_to);

                write_u32(&mut out, members.len() as u32);
                for m in members {
                    write_u32(&mut out, m.0);
                }
                for off in offsets {
                    write_u32(&mut out, *off);
                }
                for pos in original_positions {
                    write_u32(&mut out, pos.x as u32);
                    write_u32(&mut out, pos.y as u32);
                    write_u32(&mut out, pos.z as u32);
                }
                for (sx, sy, sz) in original_shapes {
                    write_u32(&mut out, *sx as u32);
                    write_u32(&mut out, *sy as u32);
                    write_u32(&mut out, *sz as u32);
                }
                for lay in original_layers {
                    write_u16(&mut out, *lay);
                }
            }

            Transform::DropLayer { cube_id, from_layer, to_layer } => {
                write_u8(&mut out, 4);
                write_u32(&mut out, cube_id.0);
                write_u16(&mut out, *from_layer);
                write_u16(&mut out, *to_layer);
            }

            Transform::PatternTag { cube_id, layer, tags } => {
                write_u8(&mut out, 5);
                write_u32(&mut out, cube_id.0);
                write_u16(&mut out, *layer);
                write_u32(&mut out, tags.len() as u32);
                for pt in tags {
                    write_u32(&mut out, pt.pattern);
                    write_u32(&mut out, pt.tag);
                }
            }

            Transform::PatternRef { cube_id, ref_id, layer } => {
                write_u8(&mut out, 6);
                write_u32(&mut out, cube_id.0);
                write_u32(&mut out, *ref_id);
                write_u16(&mut out, *layer);
            }
        }
    }

    // ------------------------------------------------------------
    // Pattern buffer section
    // ------------------------------------------------------------
    write_u32(&mut out, pattern_entries.len() as u32);
    for (ref_id, bytes) in pattern_entries {
        write_u32(&mut out, ref_id);
        write_u32(&mut out, bytes.len() as u32);
        out.extend_from_slice(&bytes);
    }

    out.extend_from_slice(&data_section);
    out
}

/// ============================================================
/// UNPACK
/// ============================================================
pub fn unpack_cubes_and_log(blob: &[u8]) -> (Vec<Cube>, usize, TransformLog) {
    let mut cur = Cursor::new(blob);

    let mut magic = [0u8; 8];
    cur.read_exact(&mut magic).unwrap();
    if &magic != MAGIC {
        panic!("Invalid BitDrop3D magic");
    }

    let _version = read_u32(&mut cur);
    let layer_count = read_u32(&mut cur) as usize;
    let original_len = read_u64(&mut cur) as usize;

    #[derive(Clone)]
    struct CubeMeta {
        id: u32,
        layer: u16,
        pos: CubePos,
        shape: (usize, usize, usize),
        quant_bits: u8,
        quant_vmin: u8,
        quant_vmax: u8,
        is_ref: bool,
        ref_id: u32,
        offset: u64,
        len: u32,
    }

    let mut metas = Vec::with_capacity(layer_count * 8);

    // ------------------------------------------------------------
    // Read cube metadata
    // ------------------------------------------------------------
    for _ in 0..layer_count {
        let layer = read_u16(&mut cur);
        let count = read_u32(&mut cur) as usize;

        for _ in 0..count {
            let id = read_u32(&mut cur);
            let x = read_u32(&mut cur) as i32;
            let y = read_u32(&mut cur) as i32;
            let z = read_u32(&mut cur) as i32;
            let sx = read_u32(&mut cur) as usize;
            let sy = read_u32(&mut cur) as usize;
            let sz = read_u32(&mut cur) as usize;

            let quant_bits = read_u8(&mut cur);
            let quant_vmin = read_u8(&mut cur);
            let quant_vmax = read_u8(&mut cur);

            let flag = read_u8(&mut cur);

            let (is_ref, ref_id, offset, len) = if flag == 1 {
                let ref_id = read_u32(&mut cur);
                (true, ref_id, 0, 0)
            } else {
                let offset = read_u64(&mut cur);
                let len = read_u32(&mut cur);
                (false, 0, offset, len)
            };

            metas.push(CubeMeta {
                id,
                layer,
                pos: CubePos { x, y, z },
                shape: (sx, sy, sz),
                quant_bits,
                quant_vmin,
                quant_vmax,
                is_ref,
                ref_id,
                offset,
                len,
            });
        }
    }

    // ------------------------------------------------------------
    // Read transform log
    // ------------------------------------------------------------
    let log_len = read_u32(&mut cur) as usize;
    let mut log = TransformLog::new();
    log.transforms.reserve(log_len);

    for _ in 0..log_len {
        let tag = read_u8(&mut cur);

        match tag {
            1 => {
                let id = read_u32(&mut cur);
                let layer = read_u16(&mut cur);
                let ori = read_u8(&mut cur);
                log.push(Transform::Rotate {
                    cube_id: CubeId(id),
                    layer,
                    orientation: Orientation(ori),
                });
            }

            2 => {
                let id = read_u32(&mut cur);
                let layer = read_u16(&mut cur);
                let dx = read_i8(&mut cur);
                let dy = read_i8(&mut cur);
                let dz = read_i8(&mut cur);
                log.push(Transform::Shift {
                    cube_id: CubeId(id),
                    layer,
                    dx,
                    dy,
                    dz,
                });
            }

            3 => {
                let id = read_u32(&mut cur);
                let layer_from = read_u16(&mut cur);
                let layer_to = read_u16(&mut cur);

                let count = read_u32(&mut cur) as usize;

                let mut members = Vec::with_capacity(count);
                for _ in 0..count {
                    members.push(CubeId(read_u32(&mut cur)));
                }

                let mut offsets = Vec::with_capacity(count);
                for _ in 0..count {
                    offsets.push(read_u32(&mut cur));
                }

                let mut original_positions = Vec::with_capacity(count);
                for _ in 0..count {
                    let x = read_u32(&mut cur) as i32;
                    let y = read_u32(&mut cur) as i32;
                    let z = read_u32(&mut cur) as i32;
                    original_positions.push(CubePos { x, y, z });
                }

                let mut original_shapes = Vec::with_capacity(count);
                for _ in 0..count {
                    let sx = read_u32(&mut cur) as usize;
                    let sy = read_u32(&mut cur) as usize;
                    let sz = read_u32(&mut cur) as usize;
                    original_shapes.push((sx, sy, sz));
                }

                let mut original_layers = Vec::with_capacity(count);
                for _ in 0..count {
                    original_layers.push(read_u16(&mut cur));
                }

                log.push(Transform::Merge {
                    new_cube_id: CubeId(id),
                    layer_from,
                    layer_to,
                    members,
                    offsets,
                    original_positions,
                    original_shapes,
                    original_layers,
                });
            }

            4 => {
                let id = read_u32(&mut cur);
                let from_layer = read_u16(&mut cur);
                let to_layer = read_u16(&mut cur);
                log.push(Transform::DropLayer {
                    cube_id: CubeId(id),
                    from_layer,
                    to_layer,
                });
            }

            5 => {
                let id = read_u32(&mut cur);
                let layer = read_u16(&mut cur);
                let count = read_u32(&mut cur) as usize;
                let mut tags = Vec::with_capacity(count);
                for _ in 0..count {
                    let pattern = read_u32(&mut cur);
                    let tag_val = read_u32(&mut cur);
                    tags.push(PatternTag { pattern, tag: tag_val });
                }
                log.push(Transform::PatternTag {
                    cube_id: CubeId(id),
                    layer,
                    tags,
                });
            }

            6 => {
                let id = read_u32(&mut cur);
                let ref_id = read_u32(&mut cur);
                let layer = read_u16(&mut cur);
                log.push(Transform::PatternRef {
                    cube_id: CubeId(id),
                    ref_id,
                    layer,
                });
            }

            _ => panic!("Unknown transform tag {}", tag),
        }
    }

    // ------------------------------------------------------------
    // Pattern buffer section
    // ------------------------------------------------------------
    let pattern_count = read_u32(&mut cur) as usize;
    let mut pattern_buffer: Vec<Vec<u8>> = Vec::with_capacity(pattern_count);

    for _ in 0..pattern_count {
        let _ref_id = read_u32(&mut cur);
        let len = read_u32(&mut cur) as usize;
        let mut buf = vec![0u8; len];
        cur.read_exact(&mut buf).unwrap();
        pattern_buffer.push(buf);
    }

    // ------------------------------------------------------------
    // Raw data section
    // ------------------------------------------------------------
    let data_start = cur.position() as usize;
    let data_slice = &blob[data_start..];

    let mut cubes = Vec::with_capacity(metas.len());

    for meta in metas {
        let data = if meta.is_ref {
            pattern_buffer[meta.ref_id as usize].clone()
        } else {
            let start = meta.offset as usize;
            let end = start + meta.len as usize;
            data_slice[start..end].to_vec()
        };

        cubes.push(Cube {
            id: CubeId(meta.id),
            layer: meta.layer,
            pos: meta.pos,
            shape: meta.shape,
            data: CubeData::Raw(data),
            quant_bits: meta.quant_bits,
            quant_vmin: meta.quant_vmin,
            quant_vmax: meta.quant_vmax,
        });
    }

    (cubes, original_len, log)
}











