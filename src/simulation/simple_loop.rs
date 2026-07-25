use rand::Rng;
use crate::roundabout::{
    channel::HbmChannel,
    controller::HbmRoundaboutController,
    request::{HbmRequest, RequestKind, RequestPriority},
};

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

    // Attach tunnel characteristics to a subset to showcase tunnel routing
    channels[2].attach_tunnel(
        1.5,   // latency_ms
        0.3,   // jitter_ms
        0.01,  // loss_rate
        1.2,   // stability
        0.25,  // congestion
    );
    channels[3].attach_tunnel(
        3.0,
        0.8,
        0.03,
        0.9,
        0.6,
    );

    let mut ctrl = HbmRoundaboutController::new(channels, layer_count, 0.85);

    let total_requests = 10_000;
    let mut exited = 0;
    let mut max_circulations = 0u32;
    let mut emergency_exits = 0;
    let mut standard_exits = 0;
    let mut low_exits = 0;

    for id in 0..total_requests {
        // Randomize request properties
        let priority = match rng.gen_range(0..100) {
            0..=5 => RequestPriority::High,      // small fraction: emergencies
            6..=70 => RequestPriority::Standard, // majority: normal traffic
            _ => RequestPriority::Low,          // background traffic
        };

        let kind = match rng.gen_range(0..3) {
            0 => RequestKind::Load,
            1 => RequestKind::Store,
            _ => RequestKind::Prefetch,
        };

        let start_channel = rng.gen_range(0..4);
        let bank_id = rng.gen_range(0..16);
        let row_addr: u64 = rng.gen_range(0..0xFFFF);

        let mut req = HbmRequest::new(
            id as u64,
            start_channel,
            bank_id,
            row_addr,
            priority,
            kind,
            layer_count,
        );

        // Let the roundabout try to route this request
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
