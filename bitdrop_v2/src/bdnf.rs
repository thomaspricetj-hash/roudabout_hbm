use serde_json::Value;
use byteorder::{BigEndian, WriteBytesExt};

/// Encode raw JSON bytes into BDNF.
/// Safe, deterministic, zero‑panic version.
pub fn encode_bdnf_json(json_bytes: &[u8]) -> Vec<u8> {
    match serde_json::from_slice::<Value>(json_bytes) {
        Ok(v) => encode_bdnf_value(&v),
        Err(_) => Vec::new(), // fail‑safe: return empty BDNF
    }
}

/// Encode a pre‑parsed JSON value into BDNF.
pub fn encode_bdnf_struct(v: &Value) -> Vec<u8> {
    encode_bdnf_value(v)
}

/// Core encoder entry point.
fn encode_bdnf_value(v: &Value) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    encode_value(v, &mut out);
    out
}

/// Encode a single JSON value into BDNF.
fn encode_value(v: &Value, out: &mut Vec<u8>) {
    match v {
        // -----------------------------
        // STRING
        // -----------------------------
        Value::String(s) => {
            out.push(0x01);
            let bs = s.as_bytes();
            let len = bs.len().min(u32::MAX as usize) as u32;
            out.write_u32::<BigEndian>(len).unwrap();
            out.extend_from_slice(&bs[..len as usize]);
        }

        // -----------------------------
        // NUMBER
        // -----------------------------
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push(0x03);
                out.write_u32::<BigEndian>(8).unwrap();
                out.write_i64::<BigEndian>(i).unwrap();
            } else if let Some(f) = n.as_f64() {
                out.push(0x04);
                out.write_u32::<BigEndian>(8).unwrap();
                out.write_f64::<BigEndian>(f).unwrap();
            } else {
                // unreachable for valid JSON, but safe fallback
                out.push(0x06);
                out.write_u32::<BigEndian>(0).unwrap();
            }
        }

        // -----------------------------
        // BOOL
        // -----------------------------
        Value::Bool(b) => {
            out.push(0x05);
            out.write_u32::<BigEndian>(1).unwrap();
            out.push(if *b { 1 } else { 0 });
        }

        // -----------------------------
        // NULL
        // -----------------------------
        Value::Null => {
            out.push(0x06);
            out.write_u32::<BigEndian>(0).unwrap();
        }

        // -----------------------------
        // ARRAY
        // -----------------------------
        Value::Array(arr) => {
            out.push(0x07);

            // Reserve space for length
            let len_pos = out.len();
            out.write_u32::<BigEndian>(0).unwrap();

            let start = out.len();
            for item in arr {
                encode_value(item, out);
            }

            // Patch length
            let inner_len = (out.len() - start).min(u32::MAX as usize) as u32;
            out[len_pos..len_pos + 4].copy_from_slice(&inner_len.to_be_bytes());
        }

        // -----------------------------
        // OBJECT
        // -----------------------------
        Value::Object(map) => {
            out.push(0x08);

            // Reserve space for length
            let len_pos = out.len();
            out.write_u32::<BigEndian>(0).unwrap();

            let start = out.len();

            // Deterministic ordering
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();

            for k in keys {
                encode_value(&Value::String(k.clone()), out);
                encode_value(&map[k], out);
            }

            // Patch length
            let inner_len = (out.len() - start).min(u32::MAX as usize) as u32;
            out[len_pos..len_pos + 4].copy_from_slice(&inner_len.to_be_bytes());
        }
    }
}
