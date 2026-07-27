use std::io::{Cursor, Read};
use crate::pymid::model::{
    read_header,
    FLAG_TINY, FLAG_DICTIONARY, FLAG_DICT_CDB, FLAG_STRUCTURED,
};

use crate::pymid::structbin::decode_structured_binary;
use crate::bitdrop::binary::BitDropBinaryEngine;

const TAG_DICT_REF: u8 = 0x00;
const TAG_LITERAL:  u8 = 0x01;
const TAG_RAW:      u8 = 0xFF;

const DICT_MODE_RAW: u8 = 0x00;
const DICT_MODE_CDB: u8 = 0x01;

const MAX_ORIGINAL_LEN: usize = 1 << 30; // 1 GiB safety
const MAX_DICT_SIZE: usize = 1 << 16;    // 65536 entries

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

fn read_raw_dict(cur: &mut Cursor<&[u8]>, dict: &mut Vec<Vec<u8>>, dict_size: usize) -> Option<()> {
    for _ in 0..dict_size {
        let len = read_u16(cur)? as usize;
        if len == 0 || len > 65535 {
            return None;
        }
        let mut buf = vec![0u8; len];
        cur.read_exact(&mut buf).ok()?;
        dict.push(buf);
    }
    Some(())
}

fn read_cdb_dict(cur: &mut Cursor<&[u8]>, dict: &mut Vec<Vec<u8>>, dict_size: usize) -> Option<()> {
    let mut prev: Vec<u8> = Vec::new();

    for _ in 0..dict_size {
        let prefix_len = read_u16(cur)? as usize;
        let suffix_len = read_u16(cur)? as usize;

        if prefix_len > prev.len() || suffix_len > 65535 {
            return None;
        }

        let mut suffix = vec![0u8; suffix_len];
        cur.read_exact(&mut suffix).ok()?;

        let mut token = Vec::with_capacity(prefix_len + suffix_len);
        token.extend_from_slice(&prev[..prefix_len]);
        token.extend_from_slice(&suffix);

        prev = token.clone();
        dict.push(token);
    }
    Some(())
}

// Tiny‑text decompressor (matches classic tiny_compress)
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

// Predictive tiny‑text decoder (mode = 2)
fn tiny_decompress_predictive(data: &[u8], dict_size: usize, original_len: usize) -> Option<Vec<u8>> {
    let mut cur = 0usize;

    // Dictionary
    let mut dict: Vec<Vec<u8>> = Vec::with_capacity(dict_size);
    for _ in 0..dict_size {
        if cur >= data.len() {
            return None;
        }
        let prefix_len = data[cur] as usize;
        cur += 1;

        if cur + prefix_len > data.len() {
            return None;
        }
        let _prefix = &data[cur..cur + prefix_len];
        cur += prefix_len;

        if cur + 2 > data.len() {
            return None;
        }
        let full_len = u16::from_le_bytes([data[cur], data[cur + 1]]) as usize;
        cur += 2;

        if cur + full_len > data.len() {
            return None;
        }
        let token = data[cur..cur + full_len].to_vec();
        cur += full_len;

        dict.push(token);
    }

    // Stream
    let mut out = Vec::with_capacity(original_len);
    while cur < data.len() && out.len() < original_len {
        let tag = data[cur];
        cur += 1;

        match tag {
            0 => {
                if cur >= data.len() {
                    return None;
                }
                let idx = data[cur] as usize;
                cur += 1;
                if idx >= dict.len() {
                    return None;
                }
                let tok = &dict[idx];
                if out.len() + tok.len() > original_len {
                    return None;
                }
                out.extend_from_slice(tok);
            }
            1 => {
                if cur >= data.len() {
                    return None;
                }
                let len = data[cur] as usize;
                cur += 1;
                if cur + len > data.len() {
                    return None;
                }
                if out.len() + len > original_len {
                    return None;
                }
                out.extend_from_slice(&data[cur..cur + len]);
                cur += len;
            }
            _ => return None,
        }
    }

    if out.len() != original_len {
        return None;
    }

    Some(out)
}

pub fn pymid_decode(input: &[u8]) -> Option<Vec<u8>> {
    let (meta, header_len) = read_header(input)?;
    let original_len = meta.original_len as usize;
    let dict_size = meta.dict_size as usize;

    if original_len == 0 {
        return Some(Vec::new());
    }
    if original_len > MAX_ORIGINAL_LEN {
        return None;
    }
    if dict_size > MAX_DICT_SIZE {
        return None;
    }
    if header_len >= input.len() {
        return None;
    }

    let body = &input[header_len..];

    // ------------------------------------------------------------------------
    // TINY PREDICTIVE MODE (FLAG_TINY + dict_size > 0, mode = 2)
    // ------------------------------------------------------------------------
    if (meta.flags & FLAG_TINY) != 0 && dict_size > 0 {
        if body.is_empty() {
            return None;
        }
        let mode = body[0];
        if mode == 2 {
            return tiny_decompress_predictive(&body[1..], dict_size, original_len);
        }
    }

    let mut cur = Cursor::new(body);

    // ------------------------------------------------------------------------
    // RAW / TINY / STRUCTURED-BINARY MODE (dict_size == 0)
    // ------------------------------------------------------------------------
    if dict_size == 0 {
        let remaining = input.len() - header_len;

        // Tiny‑text
        if meta.flags & FLAG_TINY != 0 {
            if remaining == 0 {
                return None;
            }
            let mut buf = vec![0u8; remaining];
            cur.read_exact(&mut buf).ok()?;
            let out = tiny_decompress(&buf)?;
            if out.len() != original_len {
                return None;
            }
            return Some(out);
        }

        // Structured-binary
        if (meta.flags & FLAG_STRUCTURED) != 0 {
            let tag = read_tag(&mut cur)?;
            if tag != TAG_RAW {
                return None;
            }

            let mut compressed = Vec::with_capacity(remaining.saturating_sub(1));
            cur.read_to_end(&mut compressed).ok()?;

            // FIXED: remove invalid `.ok()`
            let binary = BitDropBinaryEngine::decompress(&compressed)?;

            let out = decode_structured_binary(&binary)?;
            return Some(out);
        }

        // Legacy raw
        if remaining < original_len + 1 {
            return None;
        }

        let tag = read_tag(&mut cur)?;
        if tag != TAG_RAW {
            return None;
        }

        let mut out = vec![0u8; original_len];
        cur.read_exact(&mut out).ok()?;
        return Some(out);
    }

    // ------------------------------------------------------------------------
    // DICTIONARY MODE
    // ------------------------------------------------------------------------
    let mut dict: Vec<Vec<u8>> = Vec::with_capacity(dict_size);

    if meta.flags & FLAG_DICTIONARY != 0 {
        let start_pos = cur.position();
        let mode = read_tag(&mut cur);
        cur.set_position(start_pos);

        if meta.flags & FLAG_DICT_CDB != 0 {
            if let Some(m) = mode {
                if m == DICT_MODE_CDB {
                    let _ = read_tag(&mut cur)?;
                    read_cdb_dict(&mut cur, &mut dict, dict_size)?;
                } else {
                    read_cdb_dict(&mut cur, &mut dict, dict_size)?;
                }
            } else {
                return None;
            }
        } else {
            if let Some(m) = mode {
                if m == DICT_MODE_RAW {
                    let _ = read_tag(&mut cur)?;
                    read_raw_dict(&mut cur, &mut dict, dict_size)?;
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }
    } else {
        return None;
    }

    // ------------------------------------------------------------------------
    // STREAM DECODE
    // ------------------------------------------------------------------------
    let mut out = Vec::with_capacity(original_len);

    while out.len() < original_len {
        let tag = match read_tag(&mut cur) {
            Some(t) => t,
            None => break,
        };

        match tag {
            TAG_DICT_REF => {
                let idx = read_u16(&mut cur)? as usize;
                if idx >= dict.len() {
                    return None;
                }
                let token = &dict[idx];
                if out.len() + token.len() > original_len {
                    return None;
                }
                out.extend_from_slice(token);
            }

            TAG_LITERAL => {
                let len = read_u16(&mut cur)? as usize;
                if len == 0 || len > 65535 {
                    return None;
                }
                if out.len() + len > original_len {
                    return None;
                }
                let mut buf = vec![0u8; len];
                cur.read_exact(&mut buf).ok()?;
                out.extend_from_slice(&buf);
            }

            TAG_RAW => return None,

            _ => return None,
        }
    }

    if out.len() != original_len {
        return None;
    }

    Some(out)
}





