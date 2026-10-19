// Demo input for scripts/scan.py. Not part of the corpus and not used in
// training -- this file exists so `./run_model.sh` can demonstrate the scanner
// on something the model has definitely never seen.
//
// It deliberately mixes genuinely dangerous functions with careful ones, so the
// output shows both the ranking and its limits. Do not read the scanner's
// verdicts here as ground truth: at a measured precision of 0.078 it is a
// triage aid, not an oracle.

use std::ptr;
use std::slice;

/// Raw pointers, no length validation, unsafe block.
pub unsafe fn copy_region(dst: *mut u8, src: *const u8, len: usize) {
    unsafe {
        ptr::copy_nonoverlapping(src, dst, len);
    }
}

/// Builds a slice from caller-supplied parts with no checks at all.
pub unsafe fn as_slice<'a>(data: *const u8, len: usize) -> &'a [u8] {
    unsafe { slice::from_raw_parts(data, len) }
}

/// Unchecked indexing plus a cast chain: panics on a short buffer.
pub fn parse_length(buf: &[u8], offset: usize) -> usize {
    let hi = buf[offset] as usize;
    let lo = buf[offset + 1] as usize;
    (hi << 8) | lo
}

/// Panics on malformed input instead of returning an error.
pub fn decode_name(raw: &[u8]) -> String {
    let text = std::str::from_utf8(raw).unwrap();
    let first = text.split(',').next().expect("no comma present");
    first.to_string()
}

/// Careful: bounds-checked, checked arithmetic, no panics.
pub fn checked_total(values: &[u32]) -> Option<u32> {
    let mut total: u32 = 0;
    for value in values {
        total = total.checked_add(*value)?;
    }
    Some(total)
}

/// Safe accessor returning an Option rather than indexing directly.
pub fn byte_at(buf: &[u8], index: usize) -> Option<u8> {
    if index < buf.len() {
        Some(buf[index])
    } else {
        None
    }
}

/// Transmute between types: a classic unsoundness source.
pub fn reinterpret(value: u64) -> f64 {
    unsafe { std::mem::transmute::<u64, f64>(value) }
}
