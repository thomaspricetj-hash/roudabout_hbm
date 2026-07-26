// ============================================================================
// PyMid v4 — Structure‑Aware Tokenizer (NO chunking, NO grouping)
// ============================================================================

use std::str as _;

// ---------------------------------------------------------------------------
// Character classifiers
// ---------------------------------------------------------------------------

#[inline]
fn is_word_char(b: u8) -> bool {
    (b as char).is_alphanumeric() || b == b'_' || b == b'$'
}

#[inline]
fn is_number_char(b: u8) -> bool {
    (b as char).is_ascii_digit() || matches!(b, b'.' | b'-' | b'+' | b'e' | b'E')
}

#[inline]
fn is_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n')
}

#[inline]
fn is_json_punct(b: u8) -> bool {
    matches!(b, b'{' | b'}' | b'[' | b']' | b':' | b',' | b'"')
}

#[inline]
fn is_csv_sep(b: u8) -> bool {
    matches!(b, b',' | b';' | b'|')
}

#[inline]
fn is_utf8_start(b: u8) -> bool {
    b < 0x80 || (b & 0xC0) == 0xC0
}

#[inline]
fn utf8_char_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if (b & 0xE0) == 0xC0 {
        2
    } else if (b & 0xF0) == 0xE0 {
        3
    } else if (b & 0xF8) == 0xF0 {
        4
    } else {
        1
    }
}

// ---------------------------------------------------------------------------
// Multi‑char punctuation sequences
// ---------------------------------------------------------------------------

const MULTI_PUNCT: &[&[u8]] = &[
    b"::", b"==", b"->", b"=>", b">=", b"<=", b"&&", b"||",
];

// ---------------------------------------------------------------------------
// Timestamp detection (YYYY-MM-DD, HH:MM:SS)
// ---------------------------------------------------------------------------

fn is_timestamp_start(input: &[u8], i: usize) -> Option<usize> {
    let rem = &input[i..];

    if rem.len() >= 10
        && rem[4] == b'-'
        && rem[7] == b'-'
        && rem[..4].iter().all(|c| c.is_ascii_digit())
        && rem[5..7].iter().all(|c| c.is_ascii_digit())
        && rem[8..10].iter().all(|c| c.is_ascii_digit())
    {
        return Some(10);
    }

    if rem.len() >= 8
        && rem[2] == b':'
        && rem[5] == b':'
        && rem[..2].iter().all(|c| c.is_ascii_digit())
        && rem[3..5].iter().all(|c| c.is_ascii_digit())
        && rem[6..8].iter().all(|c| c.is_ascii_digit())
    {
        return Some(8);
    }

    None
}

// ---------------------------------------------------------------------------
// Config key=value detection
// ---------------------------------------------------------------------------

fn scan_key_value(input: &[u8], i: usize) -> Option<(usize, usize)> {
    let mut pos = i;

    while pos < input.len() && is_word_char(input[pos]) {
        pos += 1;
    }

    if pos >= input.len() || input[pos] != b'=' {
        return None;
    }

    let eq_pos = pos;
    pos += 1;

    while pos < input.len() && !is_whitespace(input[pos]) {
        pos += 1;
    }

    Some((eq_pos - i, pos - eq_pos))
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

pub fn tokenize(input: &[u8]) -> Vec<Vec<u8>> {
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < input.len() {
        let b = input[i];

        if is_whitespace(b) {
            let start = i;
            i += 1;
            while i < input.len() && is_whitespace(input[i]) {
                i += 1;
            }
            tokens.push(input[start..i].to_vec());
            continue;
        }

        let mut matched = false;
        for seq in MULTI_PUNCT {
            if i + seq.len() <= input.len() && &input[i..i + seq.len()] == *seq {
                tokens.push(seq.to_vec());
                i += seq.len();
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }

        if is_json_punct(b) {
            tokens.push(vec![b]);
            i += 1;
            continue;
        }

        if is_csv_sep(b) {
            tokens.push(vec![b]);
            i += 1;
            continue;
        }

        if let Some(len) = is_timestamp_start(input, i) {
            tokens.push(input[i..i + len].to_vec());
            i += len;
            continue;
        }

        if let Some((klen, vlen)) = scan_key_value(input, i) {
            let total = klen + 1 + vlen;
            tokens.push(input[i..i + total].to_vec());
            i += total;
            continue;
        }

        if is_word_char(b) {
            let start = i;
            i += 1;
            while i < input.len() && is_word_char(input[i]) {
                i += 1;
            }
            tokens.push(input[start..i].to_vec());
            continue;
        }

        if is_number_char(b) {
            let start = i;
            i += 1;
            while i < input.len() && is_number_char(input[i]) {
                i += 1;
            }
            tokens.push(input[start..i].to_vec());
            continue;
        }

        if is_utf8_start(b) {
            let len = utf8_char_len(b);
            if i + len <= input.len() {
                tokens.push(input[i..i + len].to_vec());
                i += len;
                continue;
            }
        }

        tokens.push(vec![b]);
        i += 1;
    }

    tokens
}
