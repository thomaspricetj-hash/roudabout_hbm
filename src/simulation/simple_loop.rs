use crate::roundabout::{
    channel::HbmChannel,
    controller::HbmRoundaboutController,
    request::{HbmRequest, RequestKind, RequestPriority},
};

pub fn run_simulation() {
    // Multilayer engine uses 4 layers (same as controller + request)
    let layer_count = 4;

    // Build channels (unchanged logic, just fixed constructor)
    let channels = vec![
        HbmChannel::new(0, 16, 0.9, layer_count),
        HbmChannel::new(1, 16, 0.9, layer_count),
        HbmChannel::new(2, 16, 0.9, layer_count),
        HbmChannel::new(3, 16, 0.9, layer_count),
    ];

    // MAX‑tier controller (parallel routing)
    let mut ctrl = HbmRoundaboutController::new(channels, layer_count, 0.85);

    // Request (updated to match upgraded HbmRequest fields)
    let mut req = HbmRequest::new(
        1,                      // id
        0,                      // starting channel
        3,                      // bank
        0x1234,                 // row address
        RequestPriority::Standard,
        RequestKind::Load,
        layer_count,
    );

    // Simulation loop
    for _ in 0..10 {
        if let Some(ch) = ctrl.route_request(req.clone()) {
            println!("Request {} exited via channel {}", req.id, ch);
            break;
        } else {
            req.circulations += 1;
            println!("Request {} circulating (count: {})", req.id, req.circulations);
        }
    }
}
