// ============================================================================
// PyMid v3 — Metadata + Pro‑grade Compressed Header Block (HCB)
// Backward‑compatible with PM2\0, forward‑compatible with PH
// ============================================================================

use std::io::{Cursor, Read};

// ---------------------------------------------------------------------------
// CONSTANTS
// ---------------------------------------------------------------------------

pub const PYMID_MAGIC_OLD: &[u8; 4] = b"PM2\0";
pub const PYMID_MAGIC_HCB: &[u8; 2] = b"PH";

pub const PYMID_VERSION: u16 = 3;

pub const FLAG_UTF8:        u16 = 0x0001;
pub const FLAG_STRUCTURED:  u16 = 0x0002;
pub const FLAG_DICTIONARY:  u16 = 0x0004;
pub const FLAG_DICT_CDB:    u16 = 0x0008;
pub const FLAG_TINY:        u16 = 0x0010;

pub const STRUCT_JSON:      u16 = 0x0100;
pub const STRUCT_CSV:       u16 = 0x0200;
pub const STRUCT_LOG:       u16 = 0x0400;
pub const STRUCT_CONF:      u16 = 0x0800;

pub const FLAG_FIRST_TAG_MASK:    u16 = 0x3000;
pub const FLAG_FIRST_TAG_RAW:     u16 = 0x1000;
pub const FLAG_FIRST_TAG_DICT:    u16 = 0x2000;
pub const FLAG_FIRST_TAG_LITERAL: u16 = 0x3000;

// ---------------------------------------------------------------------------
// METADATA STRUCT
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PyMidMeta {
    pub version: u16,
    pub flags: u16,
    pub original_len: u32,
    pub dict_size: u16,
}

impl PyMidMeta {
    pub fn new(original_len: usize, dict_size: usize, flags: u16) -> Self {
        Self {
            version: PYMID_VERSION,
            flags,
            original_len: original_len as u32,
            dict_size: dict_size as u16,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: varint + zigzag + bit‑packing
// ---------------------------------------------------------------------------

fn zigzag_encode(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn zigzag_decode(v: u64) -> i64 {
    ((v >> 1) as i64) ^ (-((v & 1) as i64))
}

fn write_varint(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push(((v as u8) & 0x7F) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn read_varint(input: &[u8], pos: &mut usize) -> Option<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;

    while *pos < input.len() {
        let b = input[*pos];
        *pos += 1;

        result |= ((b & 0x7F) as u64) << shift;
        if (b & 0x80) == 0 {
            return Some(result);
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
    None
}

// Pack flags into a single byte when possible
fn pack_flags(flags: u16) -> (u8, u16) {
    let base_bits =
        (if flags & FLAG_UTF8 != 0 { 1 } else { 0 }) |
        (if flags & FLAG_STRUCTURED != 0 { 1 << 1 } else { 0 }) |
        (if flags & FLAG_DICTIONARY != 0 { 1 << 2 } else { 0 }) |
        (if flags & FLAG_DICT_CDB != 0 { 1 << 3 } else { 0 }) |
        (if flags & FLAG_TINY != 0 { 1 << 4 } else { 0 });

    let first_tag_bits = match flags & FLAG_FIRST_TAG_MASK {
        FLAG_FIRST_TAG_RAW      => 0b01 << 5,
        FLAG_FIRST_TAG_DICT     => 0b10 << 5,
        FLAG_FIRST_TAG_LITERAL  => 0b11 << 5,
        _                       => 0,
    };

    let packed = (base_bits | first_tag_bits) as u8;
    (packed, flags & !(
        FLAG_UTF8 |
        FLAG_STRUCTURED |
        FLAG_DICTIONARY |
        FLAG_DICT_CDB |
        FLAG_TINY |
        FLAG_FIRST_TAG_MASK
    ))
}

fn unpack_flags(packed: u8, extra: u16) -> u16 {
    let mut flags = extra;

    if packed & 0x01 != 0 { flags |= FLAG_UTF8; }
    if packed & 0x02 != 0 { flags |= FLAG_STRUCTURED; }
    if packed & 0x04 != 0 { flags |= FLAG_DICTIONARY; }
    if packed & 0x08 != 0 { flags |= FLAG_DICT_CDB; }
    if packed & 0x10 != 0 { flags |= FLAG_TINY; }

    match (packed >> 5) & 0x03 {
        0b01 => flags |= FLAG_FIRST_TAG_RAW,
        0b10 => flags |= FLAG_FIRST_TAG_DICT,
        0b11 => flags |= FLAG_FIRST_TAG_LITERAL,
        _    => {}
    }

    flags
}

// ---------------------------------------------------------------------------
// Minimal RLE for header bytes
// ---------------------------------------------------------------------------
//
// Format:
//   mode 0: raw packed header
//   mode 1: RLE over packed header bytes
// ---------------------------------------------------------------------------

fn rle_compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 2 * data.len());
    out.push(1); // mode = 1

    let mut i = 0;
    while i < data.len() {
        let v = data[i];
        let mut len = 1u8;
        i += 1;
        while i < data.len() && data[i] == v && len < u8::MAX {
            len += 1;
            i += 1;
        }
        out.push(v);
        out.push(len);
    }

    if out.len() >= 1 + data.len() {
        let mut raw = Vec::with_capacity(1 + data.len());
        raw.push(0); // mode = 0
        raw.extend_from_slice(data);
        raw
    } else {
        out
    }
}

fn rle_decompress(data: &[u8]) -> Option<Vec<u8>> {
    if data.is_empty() {
        return None;
    }

    let mode = data[0];
    match mode {
        0 => {
            Some(data[1..].to_vec())
        }
        1 => {
            let mut out = Vec::with_capacity(16);
            let mut i = 1;
            while i + 1 <= data.len() {
                if i + 1 >= data.len() {
                    return None;
                }
                let v = data[i];
                let len = data[i + 1];
                i += 2;
                for _ in 0..len {
                    out.push(v);
                }
            }
            Some(out)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Cube header codec — varint + bitpack + RLE
// ---------------------------------------------------------------------------
//
// Logical header fields:
//   version:      u16
//   flags:        u16
//   original_len: u32
//   dict_size:    u16
//
// Packed format (before RLE):
//   [packed_flags (u8)]
//   [extra_flags (varint, zigzag over i16)]
//   [version (varint)]
//   [original_len (varint)]
//   [dict_size (varint)]
// ---------------------------------------------------------------------------

fn cube_compress_header(raw: &[u8]) -> Vec<u8> {
    if raw.len() != 10 {
        let mut out = Vec::with_capacity(1 + raw.len());
        out.push(0);
        out.extend_from_slice(raw);
        return out;
    }

    let version      = u16::from_le_bytes([raw[0], raw[1]]);
    let flags        = u16::from_le_bytes([raw[2], raw[3]]);
    let original_len = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    let dict_size    = u16::from_le_bytes([raw[8], raw[9]]);

    let (packed_flags, extra_flags) = pack_flags(flags);

    let mut packed = Vec::with_capacity(16);
    packed.push(packed_flags);

    let extra_zigzag = zigzag_encode(extra_flags as i16 as i64);
    write_varint(extra_zigzag, &mut packed);

    write_varint(version as u64, &mut packed);
    write_varint(original_len as u64, &mut packed);
    write_varint(dict_size as u64, &mut packed);

    rle_compress(&packed)
}

fn cube_decompress_header(compressed: &[u8]) -> Option<Vec<u8>> {
    let packed = rle_decompress(compressed)?;
    if packed.is_empty() {
        return None;
    }

    let mut pos = 0;

    let packed_flags = packed[pos];
    pos += 1;

    let extra_zigzag = read_varint(&packed, &mut pos)?;
    let extra_flags = zigzag_decode(extra_zigzag) as i16 as u16;

    let version = read_varint(&packed, &mut pos)? as u16;
    let original_len = read_varint(&packed, &mut pos)? as u32;
    let dict_size = read_varint(&packed, &mut pos)? as u16;

    let flags = unpack_flags(packed_flags, extra_flags);

    let mut raw = Vec::with_capacity(10);
    raw.extend_from_slice(&version.to_le_bytes());
    raw.extend_from_slice(&flags.to_le_bytes());
    raw.extend_from_slice(&original_len.to_le_bytes());
    raw.extend_from_slice(&dict_size.to_le_bytes());

    Some(raw)
}

// ---------------------------------------------------------------------------
// WRITE HEADER (HCB or legacy PM2\0)
// ---------------------------------------------------------------------------

pub fn write_header(buf: &mut Vec<u8>, meta: &PyMidMeta) {
    let mut raw = Vec::with_capacity(10);
    raw.extend_from_slice(&meta.version.to_le_bytes());
    raw.extend_from_slice(&meta.flags.to_le_bytes());
    raw.extend_from_slice(&meta.original_len.to_le_bytes());
    raw.extend_from_slice(&meta.dict_size.to_le_bytes());

    let compressed = cube_compress_header(&raw);

    if compressed.len() < 255 {
        buf.extend_from_slice(PYMID_MAGIC_HCB);
        buf.push(compressed.len() as u8);
        buf.extend_from_slice(&compressed);
        return;
    }

    buf.extend_from_slice(PYMID_MAGIC_OLD);
    buf.extend_from_slice(&meta.version.to_le_bytes());
    buf.extend_from_slice(&meta.flags.to_le_bytes());
    buf.extend_from_slice(&meta.original_len.to_le_bytes());
    buf.extend_from_slice(&meta.dict_size.to_le_bytes());
}

// ---------------------------------------------------------------------------
// READ HEADER (supports PH and PM2\0)
// ---------------------------------------------------------------------------

pub fn read_header(input: &[u8]) -> Option<(PyMidMeta, usize)> {
    if input.len() < 3 {
        return None;
    }

    if &input[0..2] == PYMID_MAGIC_HCB {
        let header_len = input[2] as usize;
        let start = 3;
        let end = start + header_len;

        if end > input.len() {
            return None;
        }

        let compressed = &input[start..end];
        let raw = cube_decompress_header(compressed)?;

        if raw.len() != 10 {
            return None;
        }

        let version      = u16::from_le_bytes([raw[0], raw[1]]);
        let flags        = u16::from_le_bytes([raw[2], raw[3]]);
        let original_len = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
        let dict_size    = u16::from_le_bytes([raw[8], raw[9]]);

        return Some((
            PyMidMeta { version, flags, original_len, dict_size },
            end,
        ));
    }

    if input.len() < 14 {
        return None;
    }

    let mut cur = Cursor::new(input);

    let mut magic = [0u8; 4];
    cur.read_exact(&mut magic).ok()?;
    if &magic != PYMID_MAGIC_OLD {
        return None;
    }

    let mut vbuf = [0u8; 2];
    cur.read_exact(&mut vbuf).ok()?;
    let version = u16::from_le_bytes(vbuf);

    let mut fbuf = [0u8; 2];
    cur.read_exact(&mut fbuf).ok()?;
    let flags = u16::from_le_bytes(fbuf);

    let mut obuf = [0u8; 4];
    cur.read_exact(&mut obuf).ok()?;
    let original_len = u32::from_le_bytes(obuf);

    let mut dbuf = [0u8; 2];
    cur.read_exact(&mut dbuf).ok()?;
    let dict_size = u16::from_le_bytes(dbuf);

    Some((
        PyMidMeta { version, flags, original_len, dict_size },
        14,
    ))
}

// ---------------------------------------------------------------------------
// FLAG HELPERS
// ---------------------------------------------------------------------------

impl PyMidMeta {
    pub fn is_utf8(&self) -> bool { self.flags & FLAG_UTF8 != 0 }
    pub fn is_structured(&self) -> bool { self.flags & FLAG_STRUCTURED != 0 }
    pub fn uses_dictionary(&self) -> bool { self.flags & FLAG_DICTIONARY != 0 }
    pub fn uses_cdb(&self) -> bool { self.flags & FLAG_DICT_CDB != 0 }
    pub fn is_tiny(&self) -> bool { self.flags & FLAG_TINY != 0 }

    pub fn first_tag(&self) -> Option<u8> {
        match self.flags & FLAG_FIRST_TAG_MASK {
            FLAG_FIRST_TAG_RAW      => Some(0xFF),
            FLAG_FIRST_TAG_DICT     => Some(0x00),
            FLAG_FIRST_TAG_LITERAL  => Some(0x01),
            _ => None,
        }
    }
}





