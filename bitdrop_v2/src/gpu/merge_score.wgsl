@group(0) @binding(0)
var<storage, read> cube_data: array<u32>;

@group(0) @binding(1)
var<storage, read_write> out_scores: array<i32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;

    // Placeholder scoring logic:
    out_scores[idx] = -i32(idx);
}
