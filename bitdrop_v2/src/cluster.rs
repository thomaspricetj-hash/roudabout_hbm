use crate::blocks::{Cube, CubePos};
use crate::metrics::cube_signature;
use std::collections::HashMap;

/// Base adjacency radius (Manhattan distance).
const BASE_ADJ_RADIUS: i32 = 4;

/// Hard caps for cluster sizes by tier.
const SMALL_MAX_CLUSTER: usize = 256;
const MEDIUM_MAX_CLUSTER: usize = 512;
const LARGE_MAX_CLUSTER: usize = 1024;

/// ----------------------------------------------------------
/// Tiny helpers over CubeData
/// ----------------------------------------------------------
#[inline]
fn cube_bytes(c: &Cube) -> &[u8] {
    c.bytes()
}

#[inline]
fn cube_len(c: &Cube) -> usize {
    c.bytes().len()
}

/// Fast entropy estimate for a cube.
#[inline]
fn quick_entropy(c: &Cube) -> f32 {
    let bytes = cube_bytes(c);
    let mut hist = [0u16; 256];
    for &b in bytes {
        hist[b as usize] += 1;
    }
    let len = bytes.len() as f32;
    if len == 0.0 {
        return 0.0;
    }

    let mut e = 0.0;
    for &h in &hist {
        if h > 0 {
            let p = h as f32 / len;
            e -= p * p.log2();
        }
    }
    e
}

/// Rich signature: base signature + entropy bucket + skew bucket.
#[inline]
fn rich_signature(c: &Cube) -> u64 {
    let base = cube_signature(c);

    let e = quick_entropy(c);
    let e_bucket = ((e * 4.0).min(15.0).max(0.0) as u64) & 0xF;

    let bytes = cube_bytes(c);
    let (mut minb, mut maxb) = (255u8, 0u8);
    for &b in bytes {
        if b < minb {
            minb = b;
        }
        if b > maxb {
            maxb = b;
        }
    }
    let skew = (maxb as i32 - minb as i32).clamp(0, 255) as u64;

    base ^ (e_bucket << 48) ^ (skew << 32)
}

/// Size tier based on cube count and average cube size.
#[inline]
fn size_tier(cubes: &[Cube]) -> u8 {
    if cubes.is_empty() {
        return 0;
    }

    let count = cubes.len();
    let avg_size = cubes.iter().map(cube_len).sum::<usize>() / count;

    if count < 512 && avg_size < 16 * 1024 {
        0
    } else if count < 4096 && avg_size < 64 * 1024 {
        1
    } else {
        2
    }
}

/// Adaptive adjacency radius based on size and entropy.
#[inline]
fn adaptive_adj_radius(cubes: &[&Cube]) -> i32 {
    if cubes.is_empty() {
        return BASE_ADJ_RADIUS;
    }

    let avg_size = cubes.iter().map(|c| cube_len(c)).sum::<usize>() / cubes.len();
    let avg_entropy: f32 =
        cubes.iter().map(|c| quick_entropy(c)).sum::<f32>() / cubes.len() as f32;

    let mut r = BASE_ADJ_RADIUS;

    if avg_size > 16 * 1024 {
        r += 2;
    }
    if avg_size > 64 * 1024 {
        r += 2;
    }

    if avg_entropy < 3.0 {
        r += 1;
    }
    if avg_entropy < 2.0 {
        r += 1;
    }

    r
}

/// Manhattan adjacency check.
#[inline]
fn is_adjacent(a: CubePos, b: CubePos, radius: i32) -> bool {
    let dx = (a.x - b.x).abs();
    let dy = (a.y - b.y).abs();
    let dz = (a.z - b.z).abs();
    dx + dy + dz <= radius
}

/// Minimum position in a cluster for deterministic ordering.
#[inline]
fn min_pos(cluster: &[Cube]) -> CubePos {
    cluster
        .iter()
        .map(|c| c.pos)
        .min_by_key(|p| (p.x, p.y, p.z))
        .unwrap()
}

/// Adaptive clustering with rich signatures and adjacency.
pub fn cluster_cubes(cubes: &[Cube]) -> Vec<Vec<Cube>> {
    if cubes.is_empty() {
        return Vec::new();
    }

    let tier = size_tier(cubes);

    let max_cluster = match tier {
        0 => SMALL_MAX_CLUSTER,
        1 => MEDIUM_MAX_CLUSTER,
        _ => LARGE_MAX_CLUSTER,
    };

    let mut buckets: HashMap<u64, Vec<&Cube>> =
        HashMap::with_capacity(cubes.len().max(8));

    for cube in cubes {
        let sig = rich_signature(cube);
        buckets.entry(sig).or_default().push(cube);
    }

    let mut sig_keys: Vec<u64> = buckets.keys().copied().collect();
    sig_keys.sort_unstable();

    let mut final_clusters: Vec<Vec<Cube>> = Vec::new();

    for sig in sig_keys {
        let group = buckets.remove(&sig).unwrap();

        if group.len() == 1 {
            final_clusters.push(vec![group[0].clone()]);
            continue;
        }

        let adj_radius = adaptive_adj_radius(&group);

        let mut visited = vec![false; group.len()];
        let mut local_clusters: Vec<Vec<&Cube>> = Vec::new();

        for i in 0..group.len() {
            if visited[i] {
                continue;
            }

            let mut stack = vec![i];
            visited[i] = true;
            let mut cluster_refs: Vec<&Cube> = Vec::new();

            let ref_entropy = quick_entropy(group[i]);
            let ref_size = cube_len(group[i]);

            while let Some(idx) = stack.pop() {
                let c = group[idx];
                cluster_refs.push(c);

                if cluster_refs.len() >= max_cluster {
                    continue;
                }

                for j in 0..group.len() {
                    if visited[j] {
                        continue;
                    }

                    let c2 = group[j];

                    if !is_adjacent(c.pos, c2.pos, adj_radius) {
                        continue;
                    }

                    let e2 = quick_entropy(c2);
                    if (e2 - ref_entropy).abs() > 1.5 {
                        continue;
                    }

                    let s2 = cube_len(c2);
                    if (s2 as i64 - ref_size as i64).abs() > (ref_size as i64 / 2) {
                        continue;
                    }

                    visited[j] = true;
                    stack.push(j);
                }
            }

            local_clusters.push(cluster_refs);
        }

        for lc in local_clusters {
            let mut cluster = Vec::with_capacity(lc.len());
            for c in lc {
                cluster.push(c.clone());
            }
            final_clusters.push(cluster);
        }
    }

    final_clusters.sort_by(|a, b| {
        let sa = rich_signature(&a[0]);
        let sb = rich_signature(&b[0]);
        sa.cmp(&sb).then_with(|| {
            let pa = min_pos(a);
            let pb = min_pos(b);
            (pa.x, pa.y, pa.z).cmp(&(pb.x, pb.y, pb.z))
        })
    });

    final_clusters
}




