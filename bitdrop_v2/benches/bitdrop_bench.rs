use bitdrop_v2::{compress, decompress, gpu_available, init_gpu_backend};
use std::time::Instant;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn bench_case(name: &str, data: &[u8]) {
    println!("\n=== {name} ===");

    let start = Instant::now();
    let out = compress(data);
    let t_comp = start.elapsed();

    let start = Instant::now();
    let back = decompress(&out);
    let t_decomp = start.elapsed();

    let ok = back == data;

    println!("input:   {} bytes", data.len());
    println!("output:  {} bytes", out.len());
    println!("ratio:   {:.3}", out.len() as f64 / data.len() as f64);
    println!("correct: {}", ok);
    println!("compress: {:?}", t_comp);
    println!("decomp:   {:?}", t_decomp);
}

fn random_bytes(n: usize) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(12345);
    (0..n).map(|_| rng.gen::<u8>()).collect()
}

fn repetitive_bytes(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 4) as u8).collect()
}

fn structured_payload_40mb() -> Vec<u8> {
    let target = 40 * 1024 * 1024;
    let mut v = Vec::with_capacity(target);
    let mut i: u32 = 0;

    while v.len() < target {
        v.extend_from_slice(&i.to_le_bytes());
        i = i.wrapping_add(1);
    }

    v
}

fn main() {
    println!("BitDrop v2 Benchmark");
    println!("GPU available: {}", gpu_available());

    // Warmup GPU if present
    init_gpu_backend();

    // Test cases
    bench_case("small text", b"hello world, this is a test payload");

    bench_case("random 64 KB", &random_bytes(64 * 1024));

    bench_case("random 40 MB", &random_bytes(40 * 1024 * 1024));

    bench_case("repetitive 40 MB", &repetitive_bytes(40 * 1024 * 1024));

    bench_case("structured 40 MB", &structured_payload_40mb());
}
