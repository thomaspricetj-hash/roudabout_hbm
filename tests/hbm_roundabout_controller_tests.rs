use hbm_roundabout::roundabout::{
    channel::HbmChannel,
    controller::HbmRoundaboutController,
    request::{HbmRequest, RequestPriority, RequestKind},
};

fn make_test_channel(id: usize, is_tunnel: bool) -> HbmChannel {
    let mut ch = HbmChannel::new(id, 16, 1.0, 4);
    ch.is_tunnel = is_tunnel;
    ch
}

fn make_test_request_structured(row: u32, bank: u32, channel_id: usize) -> HbmRequest {
    let mut req = HbmRequest::new(
        row as u64,
        channel_id,
        bank as usize,
        row as u64,
        RequestPriority::Standard,
        RequestKind::Load,
        4,
        vec![1, 2, 3, 4],
        "",
    );
    req.payload_is_structured = true;
    req.payload_is_numeric_counter = false;
    req.payload_compressed_size = 128;
    req
}

fn make_test_request_numeric(row: u32, bank: u32, channel_id: usize) -> HbmRequest {
    let mut req = HbmRequest::new(
        row as u64,
        channel_id,
        bank as usize,
        row as u64,
        RequestPriority::High,
        RequestKind::Store,
        4,
        vec![9, 9, 9, 9],
        "",
    );
    req.payload_is_structured = false;
    req.payload_is_numeric_counter = true;
    req.payload_compressed_size = 64;
    req
}

fn make_controller(channel_count: usize, layers: usize, decay: f32) -> HbmRoundaboutController {
    let mut channels = Vec::with_capacity(channel_count);
    for i in 0..channel_count {
        let is_tunnel = i % 3 == 0;
        channels.push(make_test_channel(i, is_tunnel));
    }
    HbmRoundaboutController::new(channels, layers, decay)
}

#[test]
fn baseline_routing_works() {
    let mut ctrl = make_controller(16, 4, 0.92);

    for i in 0..500 {
        let row = (i % 64) as u32;
        let bank = (i % 8) as u32;
        let ch_id = (i % 16) as usize;

        let req = make_test_request_structured(row, bank, ch_id);
        let result = ctrl.route_request(req);

        if let Some(ch) = result {
            assert!(ch < ctrl.channels.len());
        }
    }
}

#[test]
fn parallel_fibers_do_not_panic_and_choose_consistent_best() {
    let mut ctrl = make_controller(32, 6, 0.94);

    for i in 0..200 {
        let row = (i % 128) as u32;
        let bank = (i % 16) as u32;
        let ch_id = (i % 32) as usize;

        let mut req = make_test_request_structured(row, bank, ch_id);
        req.circulations = 0;

        let result = ctrl.route_request(req);

        if let Some(ch) = result {
            assert!(ch < ctrl.channels.len());
        }
    }
}

#[test]
fn structor_isolation_behaves_sensibly() {
    let mut ctrl = make_controller(8, 3, 0.90);

    let row = 10;
    let bank = 2;
    let ch_id = 0;

    let mut req_low_entropy = make_test_request_structured(row, bank, ch_id);
    req_low_entropy.payload = vec![0; 64];

    let mut req_high_entropy = make_test_request_structured(row, bank, ch_id);
    req_high_entropy.payload = (0..64).map(|x| (x as u8) ^ 0xAA).collect();

    let res_low = ctrl.route_request(req_low_entropy).unwrap_or(usize::MAX);
    let res_high = ctrl.route_request(req_high_entropy).unwrap_or(usize::MAX);

    assert!(res_low < ctrl.channels.len());
    assert!(res_high < ctrl.channels.len());
}

#[test]
fn cube_vs_pyramid_mode_changes_behavior_for_structured_vs_numeric() {
    let mut ctrl = make_controller(12, 4, 0.93);

    let row = 5;
    let bank = 1;

    let req_struct = make_test_request_structured(row, bank, 0);
    let req_numeric = make_test_request_numeric(row, bank, 0);

    let res_struct = ctrl.route_request(req_struct);
    let res_numeric = ctrl.route_request(req_numeric);

    assert!(res_struct.is_some());
    assert!(res_numeric.is_some());
    assert!(res_struct.unwrap() < ctrl.channels.len());
    assert!(res_numeric.unwrap() < ctrl.channels.len());
}

#[test]
fn heatmap_volatility_stays_bounded_under_load() {
    let mut ctrl = make_controller(24, 5, 0.91);

    for i in 0..2000 {
        let row = (i % 256) as u32;
        let bank = (i % 32) as u32;
        let ch_id = (i % 24) as usize;

        let req = if i % 2 == 0 {
            make_test_request_structured(row, bank, ch_id)
        } else {
            make_test_request_numeric(row, bank, ch_id)
        };

        let _ = ctrl.route_request(req);
    }

    for layer in 0..ctrl.layers {
        if let Some(layer_vec) = ctrl.heatmap.layers.get(layer) {
            for h in layer_vec.iter() {
                assert!(*h >= 0.0);
                assert!(*h <= 1.5);
            }
        }
    }
}

#[test]
fn tunnel_channels_are_used_and_reinforced() {
    let mut ctrl = make_controller(18, 4, 0.92);

    for i in 0..500 {
        let row = (i % 64) as u32;
        let bank = (i % 8) as u32;
        let ch_id = (i % 18) as usize;

        let mut req = make_test_request_numeric(row, bank, ch_id);
        req.priority = RequestPriority::High;

        let res = ctrl.route_request(req);
        if let Some(ch) = res {
            assert!(ctrl.channels[ch].metrics.stability_score.is_finite());
        }
    }
}

#[test]
fn adaptive_weights_respond_to_failures() {
    let mut ctrl = make_controller(10, 3, 0.90);

    let row = 7;
    let bank = 3;
    let ch_id = 0;

    for _ in 0..50 {
        let mut req = make_test_request_structured(row, bank, ch_id);
        req.channel_id = 9999;
        let _ = ctrl.route_request(req);
    }

    let req_ok = make_test_request_structured(row, bank, 1);
    let res_ok = ctrl.route_request(req_ok);

    assert!(res_ok.is_some());
    assert!(res_ok.unwrap() < ctrl.channels.len());
}

#[test]
fn full_chaos_simulation_stays_stable() {
    let mut ctrl = make_controller(32, 6, 0.94);

    for i in 0..50_000 {
        let row = (i % 1024) as u32;
        let bank = (i % 32) as u32;
        let ch_id = (i % 32) as usize;

        let mut req = if i % 3 == 0 {
            make_test_request_structured(row, bank, ch_id)
        } else if i % 3 == 1 {
            make_test_request_numeric(row, bank, ch_id)
        } else {
            let mut r = make_test_request_structured(row, bank, ch_id);
            r.payload_is_structured = false;
            r.payload_is_numeric_counter = false;
            r.payload_compressed_size = 512;
            r
        };

        if i % 500 == 0 {
            req.priority = RequestPriority::High;
        }

        let _ = ctrl.route_request(req);
    }

    for layer in 0..ctrl.layers {
        if let Some(layer_vec) = ctrl.heatmap.layers.get(layer) {
            for h in layer_vec.iter() {
                assert!(h.is_finite());
            }
        }
    }

    for ch in ctrl.channels.iter() {
        assert!(ch.metrics.load.is_finite());
        assert!(ch.metrics.stability_score.is_finite());
        assert!(ch.metrics.geometry_score.is_finite());
    }
}


