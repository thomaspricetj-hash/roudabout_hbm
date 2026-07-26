// ============================================================================
// PyMid v4 — Structured → Binary → BitDrop Compression
// Structured-binary codec with JSON dictionary + mode tagging + auto chunking
// ============================================================================

use std::collections::HashMap;

use crate::pymid::tokenizer::tokenize;
use crate::bitdrop::binary::BitDropBinaryEngine;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn is_jsonish(input: &[u8]) -> bool {
    input
        .iter()
        .any(|&b| matches!(b, b'{' | b'}' | b'[' | b']' | b':' | b'"'))
}

#[inline]
fn is_csvish(input: &[u8]) -> bool {
    input.iter().any(|&b| matches!(b, b',' | b';' | b'|'))
}

#[inline]
fn is_newline(b: u8) -> bool {
    matches!(b, b'\n' | b'\r')
}

// Mode tags
const MODE_JSON: u8 = 0;
const MODE_CSV: u8 = 1;
const MODE_LOG: u8 = 2;

// Token tags (JSON mode)
const TAG_PUNCT: u8 = 0;
const TAG_WORD: u8 = 1;
const TAG_NUMBER: u8 = 2;
const TAG_WHITESPACE: u8 = 3;
const TAG_DICT_REF: u8 = 4;

// Chunking
const MIN_CHUNK: usize = 32 * 1024;
const MAX_CHUNK: usize = 256 * 1024;

fn choose_chunk_size(len: usize) -> usize {
    if len <= MIN_CHUNK {
        len
    } else {
        let target = len / 16; // aim for ~16 chunks
        target.clamp(MIN_CHUNK, MAX_CHUNK)
    }
}

// ============================================================================
// PUBLIC API
// ============================================================================

pub fn encode_structured(input: &[u8]) -> Vec<u8> {
    let chunk_size = choose_chunk_size(input.len());

    // Single-chunk fast path (no chunking overhead)
    if chunk_size == input.len() {
        if is_jsonish(input) {
            encode_jsonish_binary(input)
        } else if is_csvish(input) {
            encode_csvish_binary(input)
        } else {
            encode_log_binary(input)
        }
    } else {
        encode_structured_chunked(input, chunk_size)
    }
}

pub fn compress_binary(input: &[u8]) -> Vec<u8> {
    BitDropBinaryEngine::compress(input)
}

pub fn decode_structured_binary(binary: &[u8]) -> Option<Vec<u8>> {
    if binary.is_empty() {
        return Some(Vec::new());
    }

    // Chunked format starts with 0xFF marker
    if binary[0] == 0xFF {
        return decode_structured_chunked(&binary[1..]);
    }

    // Legacy single-chunk format
    match binary[0] {
        MODE_JSON => decode_json_binary(&binary[1..]),
        MODE_CSV => decode_csv_binary(&binary[1..]),
        MODE_LOG => decode_log_binary(&binary[1..]),
        _ => None,
    }
}

// ============================================================================
// CHUNKED STRUCTURED ENCODING
// ============================================================================
//
// Format:
// [0xFF]
//   repeated:
//     [u8 mode]
//     [u32 chunk_len]
//     [chunk_payload...]
//
// Each chunk_payload is the same as the legacy single-chunk *body* for that
// mode (i.e., without the leading MODE_* byte).
//

fn encode_structured_chunked(input: &[u8], chunk_size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() / 2);
    out.push(0xFF); // chunked marker

    let mut offset = 0;
    while offset < input.len() {
        let end = (offset + chunk_size).min(input.len());
        let chunk = &input[offset..end];

        if is_jsonish(chunk) {
            let payload = encode_jsonish_binary(chunk);
            // payload = [MODE_JSON][body...]
            let body = &payload[1..];
            out.push(MODE_JSON);
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
            out.extend_from_slice(body);
        } else if is_csvish(chunk) {
            let payload = encode_csvish_binary(chunk);
            let body = &payload[1..];
            out.push(MODE_CSV);
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
            out.extend_from_slice(body);
        } else {
            let payload = encode_log_binary(chunk);
            let body = &payload[1..];
            out.push(MODE_LOG);
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
            out.extend_from_slice(body);
        }

        offset = end;
    }

    out
}

fn decode_structured_chunked(binary: &[u8]) -> Option<Vec<u8>> {
    let mut i = 0;
    let mut out = Vec::new();

    while i < binary.len() {
        if i + 1 + 4 > binary.len() {
            return None;
        }
        let mode = binary[i];
        i += 1;

        let len = u32::from_le_bytes([binary[i], binary[i + 1], binary[i + 2], binary[i + 3]]) as usize;
        i += 4;

        if i + len > binary.len() {
            return None;
        }

        let chunk_payload = &binary[i..i + len];
        i += len;

        let decoded = match mode {
            MODE_JSON => decode_json_binary(chunk_payload),
            MODE_CSV => decode_csv_binary(chunk_payload),
            MODE_LOG => decode_log_binary(chunk_payload),
            _ => None,
        }?;

        if !out.is_empty() {
            out.push(b'\n');
        }
        out.extend_from_slice(&decoded);
    }

    Some(out)
}

// ============================================================================
// JSON-ish → Binary (with dictionary)
// ============================================================================

fn encode_jsonish_binary(input: &[u8]) -> Vec<u8> {
    let tokens = tokenize(input);

    // Build frequency map
    let mut freq: HashMap<Vec<u8>, usize> = HashMap::new();
    for t in &tokens {
        if t.len() >= 3 {
            *freq.entry(t.clone()).or_insert(0) += 1;
        }
    }

    // Build dictionary: tokens with freq >= 4
    let mut dict: Vec<Vec<u8>> = freq
        .into_iter()
        .filter_map(|(tok, c)| if c >= 4 { Some(tok) } else { None })
        .collect();

    // Stable order for determinism
    dict.sort();

    let mut dict_map: HashMap<&[u8], u16> = HashMap::new();
    for (i, t) in dict.iter().enumerate() {
        if i >= u16::MAX as usize {
            break;
        }
        dict_map.insert(t.as_slice(), i as u16);
    }

    // Output: [MODE_JSON][dict_len][dict entries][stream...]
    let mut out = Vec::with_capacity(input.len() / 2);
    out.push(MODE_JSON);

    let dict_len = dict_map.len() as u16;
    out.extend_from_slice(&dict_len.to_le_bytes());

    // Dict entries: [u16 len][bytes...]
    for t in &dict {
        let len = t.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(t);
    }

    // Stream
    for t in tokens {
        if t.len() == 1 {
            let b = t[0];

            // structural punctuation
            if matches!(b, b'{' | b'}' | b'[' | b']' | b':' | b',' | b'"') {
                out.push(TAG_PUNCT);
                out.push(1);
                out.push(b);
                continue;
            }

            // whitespace
            if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
                out.push(TAG_WHITESPACE);
                out.push(1);
                out.push(b);
                continue;
            }
        }

        // dictionary reference?
        if let Some(&idx) = dict_map.get(t.as_slice()) {
            out.push(TAG_DICT_REF);
            out.extend_from_slice(&idx.to_le_bytes());
            continue;
        }

        // number?
        let is_num = t.iter().all(|c| {
            (*c as char).is_ascii_digit() || matches!(*c, b'.' | b'-' | b'+' | b'e' | b'E')
        });

        if is_num {
            out.push(TAG_NUMBER);
        } else {
            out.push(TAG_WORD);
        }

        let len = t.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&t);
    }

    out
}

// ============================================================================
// CSV-ish → Binary
// ============================================================================

fn encode_csvish_binary(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() / 2);
    out.push(MODE_CSV);

    let mut row_start = 0;
    for (i, &b) in input.iter().enumerate() {
        if is_newline(b) {
            if i > row_start {
                encode_csv_row(&input[row_start..i], &mut out);
            }
            row_start = i + 1;
        }
    }

    if row_start < input.len() {
        encode_csv_row(&input[row_start..], &mut out);
    }

    out
}

fn encode_csv_row(row: &[u8], out: &mut Vec<u8>) {
    let mut cols = row.split(|&b| matches!(b, b',' | b';' | b'|'));

    let mut col_count = 0u16;
    let count_pos = out.len();
    out.extend_from_slice(&0u16.to_le_bytes()); // placeholder

    while let Some(col) = cols.next() {
        col_count += 1;
        let len = col.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(col);
    }

    out[count_pos..count_pos + 2].copy_from_slice(&col_count.to_le_bytes());
}

// ============================================================================
// Logs → Binary
// ============================================================================

fn encode_log_binary(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() / 2);
    out.push(MODE_LOG);

    let mut line_start = 0;
    for (i, &b) in input.iter().enumerate() {
        if is_newline(b) {
            if i > line_start {
                let len = (i - line_start) as u16;
                out.extend_from_slice(&len.to_le_bytes());
                out.extend_from_slice(&input[line_start..i]);
            }
            line_start = i + 1;
        }
    }

    if line_start < input.len() {
        let len = (input.len() - line_start) as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&input[line_start..]);
    }

    out
}

// ============================================================================
// DECODERS
// ============================================================================

fn decode_json_binary(binary: &[u8]) -> Option<Vec<u8>> {
    let mut i = 0;

    if i + 2 > binary.len() {
        return None;
    }
    let dict_len = u16::from_le_bytes([binary[i], binary[i + 1]]) as usize;
    i += 2;

    let mut dict: Vec<Vec<u8>> = Vec::with_capacity(dict_len);
    for _ in 0..dict_len {
        if i + 2 > binary.len() {
            return None;
        }
        let len = u16::from_le_bytes([binary[i], binary[i + 1]]) as usize;
        i += 2;
        if i + len > binary.len() {
            return None;
        }
        dict.push(binary[i..i + len].to_vec());
        i += len;
    }

    let mut out = Vec::new();

    while i < binary.len() {
        let tag = binary[i];
        i += 1;

        match tag {
            TAG_PUNCT | TAG_WHITESPACE => {
                if i + 1 > binary.len() {
                    return None;
                }
                let len = binary[i] as usize;
                i += 1;
                if i + len > binary.len() {
                    return None;
                }
                out.extend_from_slice(&binary[i..i + len]);
                i += len;
            }

            TAG_WORD | TAG_NUMBER => {
                if i + 2 > binary.len() {
                    return None;
                }
                let len = u16::from_le_bytes([binary[i], binary[i + 1]]) as usize;
                i += 2;
                if i + len > binary.len() {
                    return None;
                }
                out.extend_from_slice(&binary[i..i + len]);
                i += len;
            }

            TAG_DICT_REF => {
                if i + 2 > binary.len() {
                    return None;
                }
                let idx = u16::from_le_bytes([binary[i], binary[i + 1]]) as usize;
                i += 2;
                if idx >= dict.len() {
                    return None;
                }
                out.extend_from_slice(&dict[idx]);
            }

            _ => return None,
        }
    }

    Some(out)
}

fn decode_csv_binary(binary: &[u8]) -> Option<Vec<u8>> {
    let mut i = 0;
    let mut out = Vec::new();

    while i + 2 <= binary.len() {
        let col_count = u16::from_le_bytes([binary[i], binary[i + 1]]) as usize;
        i += 2;

        if col_count == 0 {
            continue;
        }

        for col_idx in 0..col_count {
            if i + 2 > binary.len() {
                return None;
            }
            let len = u16::from_le_bytes([binary[i], binary[i + 1]]) as usize;
            i += 2;
            if i + len > binary.len() {
                return None;
            }
            out.extend_from_slice(&binary[i..i + len]);
            i += len;

            if col_idx + 1 < col_count {
                out.push(b',');
            }
        }

        out.push(b'\n');
    }

    Some(out)
}

fn decode_log_binary(binary: &[u8]) -> Option<Vec<u8>> {
    let mut i = 0;
    let mut out = Vec::new();
    let mut first = true;

    while i + 2 <= binary.len() {
        let len = u16::from_le_bytes([binary[i], binary[i + 1]]) as usize;
        i += 2;
        if i + len > binary.len() {
            return None;
        }
        if !first {
            out.push(b'\n');
        }
        out.extend_from_slice(&binary[i..i + len]);
        i += len;
        first = false;
    }

    Some(out)
}



