// Roundabout / BitDrop‑V2 structure‑aware module root

pub mod request;      // structured payloads (frames/blocks/flags/profile)
pub mod channel;      // structure‑aware channel behavior + BitDrop biases
pub mod metrics;      // structure‑aware channel metrics + multilayer stats
pub mod controller;   // structure‑aware routing + fiber cascade
pub mod arbitration;  // structure‑aware arbitration + BitDrop coupling
pub mod priority;     // priority enums / helpers
pub mod heatmap;      // structure‑aware thermal model + HBM + BitDrop heat
pub mod scratchpad;   // structure‑aware temporal memory + locality/events
pub mod index;        // structure‑aware composite routing index
pub mod grid;         // structure‑aware geometry (CrossConnectGrid)
