// ============================================================================
// PyMid v3/v4 — Text/Structured Dictionary Codec (HCB + segmented)
// ============================================================================

pub mod tokenizer;
pub mod dictionary;
pub mod model;
pub mod encode;
pub mod decode;
pub mod structbin;   // ★ NEW: required for structured → binary → BitDrop pipeline

// Single‑frame API (backward‑compatible)
pub use encode::pymid_encode;
pub use decode::pymid_decode;

use std::io::{Cursor, Read};
use crate::pymid::model::{read_header, FLAG_TINY, FLAG_DICT_CDB};

const TAG_DICT_REF: u8 = 0x00;
const TAG_LITERAL:  u8 = 0x01;
const TAG_RAW:      u8 = 0xFF;

const DICT_MODE_RAW: u8 = 0x00;
const DICT_MODE_CDB: u8 = 0x01;

// ============================================================================
// Tiny‑text codec (raw or RLE)
// ============================================================================

const TINY_MAX_LEN: usize = 128;

fn tiny_compress(payload: &[u8]) -> Vec<u8> {
    if payload.is_empty() {
        return vec![0];
    }

    let mut rle = Vec::with_capacity(1 + 2 * payload.len());
    rle.push(1);

    let mut i = 0;
    while i < payload.len() {
        let b = payload[i];
        let mut len = 1u8;
        i += 1;
        while i < payload.len() && payload[i] == b && len < u8::MAX {
            len += 1;
            i += 1;
        }
        rle.push(b);
        rle.push(len);
    }

    if rle.len() >= 1 + payload.len() {
        let mut out = Vec::with_capacity(1 + payload.len());
        out.push(0);
        out.extend_from_slice(payload);
        out
    } else {
        rle
    }
}

fn tiny_decompress(data: &[u8]) -> Option<Vec<u8>> {
    if data.is_empty() {
        return Some(Vec::new());
    }

    let mode = data[0];
    match mode {
        0 => Some(data[1..].to_vec()),
        1 => {
            let mut out = Vec::with_capacity(data.len());
            let mut i = 1;
            while i + 1 <= data.len() {
                if i + 1 >= data.len() {
                    return None;
                }
                let b = data[i];
                let len = data[i + 1];
                i += 2;
                for _ in 0..len {
                    out.push(b);
                }
            }
            Some(out)
        }
        _ => None,
    }
}

// ============================================================================
// Segmented Encode (legacy compatibility)
// ============================================================================

pub fn pymid_encode_segmented(input: &[u8], _max_block: usize) -> Vec<u8> {
    pymid_encode(input)
}

// ============================================================================
// Helpers
// ============================================================================

fn read_u16(cur: &mut Cursor<&[u8]>) -> Option<u16> {
    let mut buf = [0u8; 2];
    cur.read_exact(&mut buf).ok()?;
    Some(u16::from_le_bytes(buf))
}

fn read_tag(cur: &mut Cursor<&[u8]>) -> Option<u8> {
    let mut tag = [0u8; 1];
    cur.read_exact(&mut tag).ok()?;
    Some(tag[0])
}

// ============================================================================
// Segmented Decode (multi‑frame support)
// ============================================================================

pub fn pymid_decode_segmented(input: &[u8]) -> Option<Vec<u8>> {
    if input.is_empty() {
        return Some(Vec::new());
    }

    let mut out = Vec::new();
    let mut cursor: usize = 0;

    while cursor < input.len() {
        let slice = &input[cursor..];

        let (meta, header_len) = read_header(slice)?;
        let original_len = meta.original_len as usize;
        let dict_size = meta.dict_size as usize;

        let decoded = pymid_decode(slice)?;
        if decoded.len() != original_len {
            return None;
        }
        out.extend_from_slice(&decoded);

        let mut cur = Cursor::new(&slice[header_len..]);
        let mut consumed: usize = 0;

        if dict_size == 0 {
            if meta.flags & FLAG_TINY != 0 {
                consumed += slice.len() - header_len;
            } else {
                let _tag = read_tag(&mut cur)?;
                consumed += 1 + original_len;
            }
        } else {
            let mut dict: Vec<Vec<u8>> = Vec::with_capacity(dict_size);

            let start_pos = cur.position();
            let mode = read_tag(&mut cur);
            cur.set_position(start_pos);

            if meta.flags & FLAG_DICT_CDB != 0 {
                if let Some(m) = mode {
                    if m == DICT_MODE_CDB {
                        let _ = read_tag(&mut cur)?;
                        consumed += 1;
                    }
                }

                let mut prev: Vec<u8> = Vec::new();
                for _ in 0..dict_size {
                    let prefix_len = read_u16(&mut cur)? as usize;
                    let suffix_len = read_u16(&mut cur)? as usize;
                    consumed += 2 + 2 + suffix_len;

                    let mut suffix = vec![0u8; suffix_len];
                    cur.read_exact(&mut suffix).ok()?;

                    let mut token = Vec::with_capacity(prefix_len + suffix_len);
                    token.extend_from_slice(&prev[..prefix_len]);
                    token.extend_from_slice(&suffix);

                    prev = token.clone();
                    dict.push(token);
                }
            } else {
                if let Some(m) = mode {
                    if m == DICT_MODE_RAW {
                        let _ = read_tag(&mut cur)?;
                        consumed += 1;
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }

                for _ in 0..dict_size {
                    let len = read_u16(&mut cur)? as usize;
                    consumed += 2 + len;
                    if len == 0 || len > 65535 {
                        return None;
                    }
                    let mut buf = vec![0u8; len];
                    cur.read_exact(&mut buf).ok()?;
                    dict.push(buf);
                }
            }

            let mut produced = 0usize;
            while produced < original_len {
                let tag = read_tag(&mut cur)?;
                consumed += 1;

                match tag {
                    TAG_DICT_REF => {
                        let idx = read_u16(&mut cur)? as usize;
                        consumed += 2;
                        if idx >= dict.len() {
                            return None;
                        }
                        produced += dict[idx].len();
                    }
                    TAG_LITERAL => {
                        let len = read_u16(&mut cur)? as usize;
                        consumed += 2 + len;
                        if len == 0 || len > 65535 {
                            return None;
                        }
                        let mut buf = vec![0u8; len];
                        cur.read_exact(&mut buf).ok()?;
                        produced += len;
                    }
                    TAG_RAW => return None,
                    _ => return None,
                }
            }
        }

        let frame_len = header_len + consumed;
        cursor += frame_len;
    }

    Some(out)
}














