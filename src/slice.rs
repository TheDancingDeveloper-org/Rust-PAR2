//! Per-slice checksums for in-flight verification.
//!
//! The IFSC packet stores an MD5 and a CRC32 for every input slice, computed
//! over the slice zero-padded to the full slice size. [`SliceHasher`] produces
//! the same pair from bytes in memory, so a caller can check a slice before
//! feeding it to a [`Repair`](crate::repair::Repair) builder: a corrupt slice
//! that is fed poisons every recovered slice and cannot be taken back out.

use md5::{Digest, Md5};

use crate::types::{Par2File, SliceChecksum};

/// Incremental MD5 + CRC32 over one input slice, zero-padded to `slice_size`.
///
/// ```
/// use rust_par2::slice::SliceHasher;
///
/// let mut h = SliceHasher::new(8);
/// h.update(b"abc");
/// h.update(b"de");
/// let sum = h.finish(); // checksums of b"abcde\0\0\0"
/// assert_eq!(sum, SliceHasher::checksum(b"abcde", 8));
/// ```
pub struct SliceHasher {
    md5: Md5,
    crc: crc32fast::Hasher,
    slice_size: u64,
    len: u64,
}

impl SliceHasher {
    /// Start hashing a slice of a set whose slice size is `slice_size`.
    pub fn new(slice_size: u64) -> Self {
        Self {
            md5: Md5::new(),
            crc: crc32fast::Hasher::new(),
            slice_size,
            len: 0,
        }
    }

    /// Feed the next bytes of the slice.
    ///
    /// Bytes beyond `slice_size` are hashed too, so the result will not match
    /// any IFSC entry; callers should not pass more than one slice.
    pub fn update(&mut self, data: &[u8]) {
        self.md5.update(data);
        self.crc.update(data);
        self.len += data.len() as u64;
    }

    /// Bytes fed so far, before padding.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether no bytes have been fed yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Zero-pad to `slice_size` and return the checksum pair.
    pub fn finish(mut self) -> SliceChecksum {
        const ZEROS: [u8; 4096] = [0; 4096];
        let mut pad = self.slice_size.saturating_sub(self.len);
        while pad > 0 {
            let n = pad.min(ZEROS.len() as u64) as usize;
            self.md5.update(&ZEROS[..n]);
            self.crc.update(&ZEROS[..n]);
            pad -= n as u64;
        }
        SliceChecksum {
            md5: self.md5.finalize().into(),
            crc32: self.crc.finalize(),
        }
    }

    /// Checksum a whole slice in one call.
    pub fn checksum(data: &[u8], slice_size: u64) -> SliceChecksum {
        let mut h = Self::new(slice_size);
        h.update(data);
        h.finish()
    }
}

impl Par2File {
    /// Whether `data` matches the IFSC checksum of this file's slice
    /// `local_index` (0-based within the file).
    ///
    /// `data` is the slice as it appears in the file: `slice_size` bytes, or
    /// fewer for the file's last slice. Returns `false` if the index is out of
    /// range or the length is wrong for that slice.
    pub fn slice_matches(&self, local_index: usize, data: &[u8], slice_size: u64) -> bool {
        let Some(expected) = self.slices.get(local_index) else {
            return false;
        };
        let offset = local_index as u64 * slice_size;
        let expected_len = self.size.saturating_sub(offset).min(slice_size);
        if data.len() as u64 != expected_len {
            return false;
        }
        SliceHasher::checksum(data, slice_size) == *expected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_matches_explicit_zeroes() {
        let data = vec![0xABu8; 10_000];
        let mut padded = data.clone();
        padded.resize(16_384, 0);
        let a = SliceHasher::checksum(&data, 16_384);
        let b = SliceHasher::checksum(&padded, 16_384);
        assert_eq!(a, b);
        assert_eq!(a.crc32, crc32fast::hash(&padded));
        let md5: [u8; 16] = Md5::digest(&padded).into();
        assert_eq!(a.md5, md5);
    }

    #[test]
    fn incremental_matches_one_shot() {
        let data: Vec<u8> = (0..5000u32).map(|i| (i * 7) as u8).collect();
        let mut h = SliceHasher::new(8192);
        for chunk in data.chunks(333) {
            h.update(chunk);
        }
        assert_eq!(h.len(), 5000);
        assert_eq!(h.finish(), SliceHasher::checksum(&data, 8192));
    }
}
