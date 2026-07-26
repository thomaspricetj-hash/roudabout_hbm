use rand::Rng;
use crate::roundabout::{
    channel::HbmChannel,
    controller::HbmRoundaboutController,
    request::{HbmRequest, RequestKind, RequestPriority},
};

fn attach_pairs(channels: &mut [HbmChannel]) {
    for (pair_id, pair) in channels.chunks_mut(2).enumerate() {
        if pair.len() == 2 {
            pair[0].attach_group(pair_id, 2, true);  // primary
            pair[1].attach_group(pair_id, 2, false); // secondary
        }
    }
}

fn attach_triplets(channels: &mut [HbmChannel]) {
    for (group_id, group) in channels.chunks_mut(3).enumerate() {
        if group.len() == 3 {
            group[0].attach_group(group_id, 3, true);
            group[1].attach_group(group_id, 3, false);
            group[2].attach_group(group_id, 3, false);
        }
    }
}

fn attach_quads(channels: &mut [HbmChannel]) {
    for (group_id, group) in channels.chunks_mut(4).enumerate() {
        if group.len() == 4 {
            group[0].attach_group(group_id, 4, true);
            group[1].attach_group(group_id, 4, false);
            group[2].attach_group(group_id, 4, false);
            group[3].attach_group(group_id, 4, false);
        }
    }
}

/// Pair/group imbalance correction + dynamic primary switching.
/// Call this periodically (e.g., every N requests).
fn rebalance_groups(channels: &mut [HbmChannel]) {
    let len = channels.len();

    // Pairs
    for pair in channels.chunks_mut(2) {
        if pair.len() == 2 {
            let load_a = pair[0].metrics.load;
            let load_b = pair[1].metrics.load;

            pair[0].update_pair_affinity(load_b);
            pair[1].update_pair_affinity(load_a);

            pair[0].maybe_switch_primary(load_b);
            pair[1].maybe_switch_primary(load_a);
        }
    }

    // Triplets
    for group in channels.chunks_mut(3) {
        if group.len() == 3 {
            let loads = [
                group[0].metrics.load,
                group[1].metrics.load,
                group[2].metrics.load,
            ];
            let avg = (loads[0] + loads[1] + loads[2]) / 3.0;

            for ch in group.iter_mut() {
                ch.update_pair_affinity(avg);
                ch.maybe_switch_primary(avg);
            }
        }
    }

    // Quads
    for group in channels.chunks_mut(4) {
        if group.len() == 4 {
            let loads = [
                group[0].metrics.load,
                group[1].metrics.load,
                group[2].metrics.load,
                group[3].metrics.load,
            ];
            let avg = (loads[0] + loads[1] + loads[2] + loads[3]) / 4.0;

            for ch in group.iter_mut() {
                ch.update_pair_affinity(avg);
                ch.maybe_switch_primary(avg);
            }
        }
    }

    let _ = len;
}

pub fn run_stress_simulation() {
    let layer_count = 4;
    let mut rng = rand::thread_rng();

    // Build channels (some normal, some tunnels)
    let mut channels = vec![
        HbmChannel::new(0, 16, 0.85, layer_count),
        HbmChannel::new(1, 16, 0.85, layer_count),
        HbmChannel::new(2, 16, 0.85, layer_count),
        HbmChannel::new(3, 16, 0.85, layer_count),
    ];

    attach_pairs(&mut channels);

    // Attach tunnel characteristics
    channels[2].attach_tunnel(1.5, 0.3, 0.01, 1.2, 0.25);
    channels[3].attach_tunnel(3.0, 0.8, 0.03, 0.9, 0.6);

    let mut ctrl = HbmRoundaboutController::new(channels, layer_count, 0.85);

    let total_requests = 10_000;
    let mut exited = 0;
    let mut max_circulations = 0u32;
    let mut emergency_exits = 0;
    let mut standard_exits = 0;
    let mut low_exits = 0;

    for id in 0..total_requests {
        if id % 256 == 0 {
            rebalance_groups(&mut ctrl.channels);
        }

        let priority = match rng.gen_range(0..100) {
            0..=5 => RequestPriority::High,
            6..=70 => RequestPriority::Standard,
            _ => RequestPriority::Low,
        };

        let kind = match rng.gen_range(0..3) {
            0 => RequestKind::Load,
            1 => RequestKind::Store,
            _ => RequestKind::Prefetch,
        };

        let start_channel = rng.gen_range(0..4);
        let bank_id = rng.gen_range(0..16);
        let row_addr: u64 = rng.gen_range(0..0xFFFF);

        // ============================================================
        // FIXED: HbmRequest::new now requires raw_payload + profile
        // ============================================================
        let mut req = HbmRequest::new(
            id as u64,
            start_channel,
            bank_id,
            row_addr,
            priority,
            kind,
            layer_count,
            vec![],     // <-- FIX: raw_payload
            "sim",      // <-- FIX: profile
        );

        let mut local_circulations = 0u32;
        for _ in 0..16 {
            if let Some(ch) = ctrl.route_request(req.clone()) {
                exited += 1;
                max_circulations = max_circulations.max(local_circulations);

                match priority {
                    RequestPriority::High => emergency_exits += 1,
                    RequestPriority::Standard => standard_exits += 1,
                    RequestPriority::Low => low_exits += 1,
                }

                break;
            } else {
                req.circulations += 1;
                local_circulations += 1;
            }
        }
    }

    println!("=== HBM Roundabout Stress Simulation ===");
    println!("Total requests:        {}", total_requests);
    println!("Total exited:          {}", exited);
    println!("Max circulations seen: {}", max_circulations);
    println!("Emergency exits (High):   {}", emergency_exits);
    println!("Standard exits:           {}", standard_exits);
    println!("Low-priority exits:       {}", low_exits);
    println!(
        "Exit rate: {:.2}%",
        (exited as f32 / total_requests as f32) * 100.0
    );
}
