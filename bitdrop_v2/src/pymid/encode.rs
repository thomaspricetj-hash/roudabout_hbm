// ============================================================================
// PyMid v4 — Simple Encoder (tokenize → dict → encode)
// ============================================================================

use std::collections::HashMap;

use crate::pymid::dictionary::build_dictionary;
use crate::pymid::model::{
    write_header, PyMidMeta,
    FLAG_DICTIONARY, FLAG_STRUCTURED, FLAG_UTF8, FLAG_DICT_CDB, FLAG_TINY,
    FLAG_FIRST_TAG_RAW, FLAG_FIRST_TAG_DICT, FLAG_FIRST_TAG_LITERAL,
};
use crate::pymid::tokenizer::tokenize;

// NEW: structured → binary → BitDrop
use crate::pymid::structbin::{encode_structured, compress_binary};

const TAG_DICT_REF: u8 = 0x00;
const TAG_LITERAL:  u8 = 0x01;
const TAG_RAW:      u8 = 0xFF;

const MAX_DICT: usize = 4096;

// ============================================================================
// Helpers
// ============================================================================

#[inline]
fn write_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn detect_utf8(input: &[u8]) -> bool {
    std::str::from_utf8(input).is_ok()
}

#[inline]
fn detect_structured(input: &[u8]) -> bool {
    if input.is_empty() {
        return false;
    }

    let mut jsonish = 0usize;
    let mut csvish = 0usize;
    let mut newlines = 0usize;

    for &b in input {
        match b {
            b'{' | b'}' | b'[' | b']' | b':' | b'"' => jsonish += 1,
            b',' | b';' | b'|' => csvish += 1,
            b'\n' | b'\r' => newlines += 1,
            _ => {}
        }
    }

    jsonish > 0 || csvish > 0 || newlines > 0
}

// ============================================================================
// Predictive Tiny Codec (Mode = 2)
// ============================================================================

fn encode_tiny_predictive(input: &[u8], mut flags: u16) -> Vec<u8> {
    flags |= FLAG_TINY;

    let tokens = tokenize(input);

    // Build tiny dictionary
    let mut map: HashMap<&[u8], usize> = HashMap::new();
    let mut dict: Vec<&[u8]> = Vec::new();

    for t in &tokens {
        if !map.contains_key(t.as_slice()) {
            map.insert(t.as_slice(), dict.len());
            dict.push(t.as_slice());
        }
    }

    let meta = PyMidMeta::new(input.len(), dict.len(), flags);

    let mut out = Vec::with_capacity(64 + input.len());
    write_header(&mut out, &meta);

    // Mode = 2 (predictive tiny)
    out.push(2);

    // Dictionary entries
    for &tok in &dict {
        let prefix_len = tok.len().min(4) as u8;
        out.push(prefix_len);
        out.extend_from_slice(&tok[..prefix_len as usize]);

        let full_len = tok.len() as u16;
        out.extend_from_slice(&full_len.to_le_bytes());

        out.extend_from_slice(tok);
    }

    // Stream
    for t in &tokens {
        if let Some(&idx) = map.get(t.as_slice()) {
            out.push(0);
            out.push(idx as u8);
        } else {
            out.push(1);
            out.push(t.len() as u8);
            out.extend_from_slice(t);
        }
    }

    out
}

// ============================================================================
// Tiny‑text codec (raw or RLE)
// ============================================================================

const TINY_MAX_LEN: usize = 256;

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

fn encode_tiny(input: &[u8], mut flags: u16) -> Vec<u8> {
    flags |= FLAG_TINY;
    flags |= FLAG_FIRST_TAG_RAW;

    // Ultra-small inputs: classic tiny is faster
    if input.len() <= 64 {
        let meta = PyMidMeta::new(input.len(), 0, flags);
        let mut out = Vec::with_capacity(32 + input.len());
        write_header(&mut out, &meta);

        let tiny = tiny_compress(input);
        out.extend_from_slice(&tiny);
        return out;
    }

    // Try predictive tiny mode
    let pred = encode_tiny_predictive(input, flags);
    if pred.len() < input.len() + 8 {
        return pred;
    }

    // Fallback to classic tiny
    let meta = PyMidMeta::new(input.len(), 0, flags);
    let mut out = Vec::with_capacity(32 + input.len());
    write_header(&mut out, &meta);

    let tiny = tiny_compress(input);
    out.extend_from_slice(&tiny);

    out
}

// ============================================================================
// Estimate encoded size
// ============================================================================

fn estimate_encoded_size(tokens: &[Vec<u8>], dict: &[Vec<u8>], original_len: usize) -> usize {
    if dict.is_empty() {
        return 6 + original_len;
    }

    let mut size = 6;

    let mut prev: &[u8] = &[];
    for t in dict {
        let mut prefix_len = 0usize;
        let max_prefix = prev.len().min(t.len());
        while prefix_len < max_prefix && prev[prefix_len] == t[prefix_len] {
            prefix_len += 1;
        }
        let suffix_len = t.len() - prefix_len;
        size += 2 + 2 + suffix_len;
        prev = t;
    }

    let mut dict_map: HashMap<&[u8], u16> = HashMap::new();
    for (i, t) in dict.iter().enumerate() {
        dict_map.insert(t.as_slice(), i as u16);
    }

    for t in tokens {
        if dict_map.contains_key(t.as_slice()) {
            size += 1 + 2;
        } else {
            size += 1 + 2 + t.len();
        }
    }

    size
}

// ============================================================================
// PyMid encoder (no grouping, no clustering)
// ============================================================================

pub fn pymid_encode(input: &[u8]) -> Vec<u8> {
    let original_len = input.len();
    let is_utf8 = detect_utf8(input);
    let is_structured = detect_structured(input);

    // ---------------------------------------------------------
    // STRUCTURED → BINARY → BITDROP → PYMID
    // ---------------------------------------------------------
    if is_structured && original_len > 4096 {
        let binary = encode_structured(input);
        let compressed = compress_binary(&binary);

        // Structured frame: FLAG_STRUCTURED + RAW tag + BitDrop payload
        let mut flags: u16 = FLAG_STRUCTURED | FLAG_FIRST_TAG_RAW;
        if is_utf8 {
            flags |= FLAG_UTF8;
        }

        let meta = PyMidMeta::new(original_len, 0, flags);
        let mut out = Vec::with_capacity(32 + 1 + compressed.len());
        write_header(&mut out, &meta);

        out.push(TAG_RAW);
        out.extend_from_slice(&compressed);

        return out;
    }

    // ---------------------------------------------------------
    // Tiny fast path (now predictive)
    // ---------------------------------------------------------
    if original_len <= TINY_MAX_LEN {
        let mut flags = 0;
        if is_utf8 { flags |= FLAG_UTF8; }
        if is_structured { flags |= FLAG_STRUCTURED; }
        return encode_tiny(input, flags);
    }

    // ---------------------------------------------------------
    // Tokenize (no grouping)
    // ---------------------------------------------------------
    let tokens = tokenize(input);

    // ---------------------------------------------------------
    // Build dictionary
    // ---------------------------------------------------------
    let dict = build_dictionary(&tokens, MAX_DICT);

    // ---------------------------------------------------------
    // Flags
    // ---------------------------------------------------------
    let mut flags: u16 = 0;
    if is_utf8 { flags |= FLAG_UTF8; }
    if is_structured { flags |= FLAG_STRUCTURED; }

    // ---------------------------------------------------------
    // Cost/benefit
    // ---------------------------------------------------------
    let dict_encoded_size = estimate_encoded_size(&tokens, &dict, original_len);
    let raw_encoded_size = 6 + original_len;

    let use_dict =
        !dict.is_empty() &&
        dict.len() >= 4 &&
        dict_encoded_size < raw_encoded_size;

    if !use_dict {
        return encode_tiny(input, flags);
    }

    // ---------------------------------------------------------
    // Dictionary mode
    // ---------------------------------------------------------
    flags |= FLAG_DICTIONARY | FLAG_DICT_CDB;

    if let Some(first) = tokens.first() {
        if dict.contains(first) {
            flags |= FLAG_FIRST_TAG_DICT;
        } else {
            flags |= FLAG_FIRST_TAG_LITERAL;
        }
    }

    let meta = PyMidMeta::new(original_len, dict.len(), flags);

    let mut out = Vec::with_capacity(dict_encoded_size);
    write_header(&mut out, &meta);

    // Write dictionary (CDB)
    let mut prev: Vec<u8> = Vec::new();
    for t in &dict {
        let mut prefix_len = 0usize;
        let max_prefix = prev.len().min(t.len());
        while prefix_len < max_prefix && prev[prefix_len] == t[prefix_len] {
            prefix_len += 1;
        }

        let suffix = &t[prefix_len..];
        write_u16(&mut out, prefix_len as u16);
        write_u16(&mut out, suffix.len() as u16);
        out.extend_from_slice(suffix);

        prev.clear();
        prev.extend_from_slice(t);
    }

    // Map
    let mut dict_map: HashMap<&[u8], u16> = HashMap::new();
    for (i, t) in dict.iter().enumerate() {
        dict_map.insert(t.as_slice(), i as u16);
    }

    // Encode stream
    for t in tokens {
        if let Some(&idx) = dict_map.get(t.as_slice()) {
            out.push(TAG_DICT_REF);
            write_u16(&mut out, idx);
        } else {
            out.push(TAG_LITERAL);
            write_u16(&mut out, t.len() as u16);
            out.extend_from_slice(&t);
        }
    }

    out
}








