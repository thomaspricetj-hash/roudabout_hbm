use crate::roundabout::{
    channel::HbmChannel,
    controller::HbmRoundaboutController,
    request::{HbmRequest, RequestKind, RequestPriority},
};

pub fn run_simulation() {
    // Build channels (unchanged logic)
    let channels = vec![
        HbmChannel::new(0, 16, 0.9),
        HbmChannel::new(1, 16, 0.9),
        HbmChannel::new(2, 16, 0.9),
        HbmChannel::new(3, 16, 0.9),
    ];

    // MAX‑tier controller (parallel routing)
    let mut ctrl = HbmRoundaboutController::new(channels, 4, 0.85);

    // Request (no logic removed)
    let req = HbmRequest::new(
        1,
        0,
        3,
        0x1234,
        RequestPriority::Standard,
        RequestKind::Load,
        4,
    );

    // Simulation loop
    for _ in 0..10 {
        // Parallel routing inside controller
        if let Some(ch) = ctrl.route_request(req.clone()) {
            println!("Request {} exited via channel {}", req.id, ch);
            break;
        } else {
            println!("Request {} circulating (count: {})", req.id, req.circulations);
        }
    }
}
