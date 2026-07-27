// src/gpu/wgpu_backend.rs

use crate::blocks::Cube;
use wgpu::util::DeviceExt;

/// ============================================================
/// WGPU backend — GPU-accelerated merge scoring (chunked, safe)
/// ============================================================

pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_layout: wgpu::BindGroupLayout,
}

impl WgpuBackend {
    /// Micro-helper: create WGPU backend if a suitable adapter/device exists.
    pub async fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("bitdrop-wgpu-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("merge-score-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("merge_score.wgsl").into()),
        });

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("merge-score-bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("merge-score-pipeline-layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("merge-score-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        Some(Self {
            device,
            queue,
            pipeline,
            bind_layout,
        })
    }

    /// Micro-helper: flatten cube bytes into a single buffer.
    #[inline]
    fn flatten_cubes(cluster: &[Cube]) -> Vec<u8> {
        let total: usize = cluster.iter().map(|c| c.bytes().len()).sum();
        let mut flat = Vec::with_capacity(total);
        for c in cluster {
            flat.extend_from_slice(c.bytes());
        }
        flat
    }

    /// Micro-helper: safe padded copy to 4-byte alignment.
    #[inline]
    fn pad_to_4(mut data: Vec<u8>) -> Vec<u8> {
        while data.len() % 4 != 0 {
            data.push(0);
        }
        data
    }

    /// Micro-helper: compute pair count from cube count.
    #[inline]
    fn pair_count_for(n: u32) -> u32 {
        (n * (n - 1)) / 2
    }

    /// Compute best merge pair on GPU, returning (i, j, score), with chunking.
    pub fn best_merge_pair(&self, cluster: &[Cube]) -> Option<(usize, usize, i64)> {
        if cluster.len() < 2 {
            return None;
        }

        const GPU_MAX_BIND: usize = 128 * 1024 * 1024;

        let mut best_global_score: i64 = i64::MAX;
        let mut best_global_pair: Option<(usize, usize)> = None;

        let mut start_idx = 0;

        while start_idx < cluster.len() {
            let mut end_idx = start_idx;
            let mut acc_bytes: usize = 0;

            // Build a chunk whose total bytes do not exceed GPU_MAX_BIND
            while end_idx < cluster.len() {
                let cube_bytes = cluster[end_idx].bytes().len();
                if acc_bytes + cube_bytes > GPU_MAX_BIND {
                    break;
                }
                acc_bytes += cube_bytes;
                end_idx += 1;
            }

            if end_idx - start_idx < 2 {
                start_idx = end_idx;
                continue;
            }

            let sub = &cluster[start_idx..end_idx];

            let flat = Self::flatten_cubes(sub);
            let padded = Self::pad_to_4(flat);

            let cube_count = sub.len() as u32;
            let pair_count = Self::pair_count_for(cube_count);

            if pair_count == 0 {
                start_idx = end_idx;
                continue;
            }

            let input_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cube-data-chunk"),
                contents: &padded,
                usage: wgpu::BufferUsages::STORAGE,
            });

            let output_size_bytes =
                (pair_count as usize * std::mem::size_of::<i32>()) as u64;

            let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("merge-scores-chunk"),
                size: output_size_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("merge-score-bind-group-chunk"),
                layout: &self.bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("merge-score-encoder-chunk"),
                    });

            const MAX_WG: u32 = 65535;
            let total = pair_count;
            let groups_x = MAX_WG;
            let groups_y = (total + MAX_WG - 1) / MAX_WG;

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("merge-score-pass-chunk"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            }

            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging-buffer-chunk"),
                size: output_size_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, staging.size());
            self.queue.submit(Some(encoder.finish()));

            let buffer_slice = staging.slice(..);
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();

            buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
                let _ = tx.send(v);
            });

            let _ = futures_lite::future::block_on(rx.receive());

            let data = buffer_slice.get_mapped_range();
            let scores: &[i32] = bytemuck::cast_slice(&data);

            let mut best_chunk_score: i32 = i32::MAX;
            let mut best_chunk_pair = (0usize, 1usize);

            let mut idx = 0;
            for i in 0..sub.len() {
                for j in (i + 1)..sub.len() {
                    let s = scores[idx];
                    if s < best_chunk_score {
                        best_chunk_score = s;
                        best_chunk_pair = (i, j);
                    }
                    idx += 1;
                }
            }

            drop(data);
            staging.unmap();

            let global_i = best_chunk_pair.0 + start_idx;
            let global_j = best_chunk_pair.1 + start_idx;
            let global_score = best_chunk_score as i64;

            if global_score < best_global_score {
                best_global_score = global_score;
                best_global_pair = Some((global_i, global_j));
            }

            start_idx = end_idx;
        }

        best_global_pair.map(|(i, j)| (i, j, best_global_score))
    }
}




