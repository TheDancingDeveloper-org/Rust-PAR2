//! Recovery block reading from PAR2 volume files.
//!
//! [`RecoveryReader`] streams RecoverySlice packets out of any [`Read`] source
//! (a file, a network body, `&[u8]`), one volume at a time, keeping at most a
//! budgeted number of rows and preferring the lowest exponents. Packets it will
//! not keep are skipped without being buffered, so a volume much larger than
//! the budget streams past in constant memory.
//!
//! [`load_recovery_blocks`] is the directory-scanning convenience wrapper.
//!
//! ```
//! use rust_par2::recovery::{RecoveryBudget, RecoveryReader};
//!
//! # fn demo(set_id: [u8; 16], slice_size: u64, volumes: Vec<Vec<u8>>, missing: usize) {
//! let mut reader = RecoveryReader::new(set_id, slice_size)
//!     .with_budget(RecoveryBudget::rows(missing));
//! for volume in &volumes {
//!     let scan = reader.read_volume(&volume[..]);
//!     if !scan.read_to_end() {
//!         // This volume may carry more rows than we saw: don't conclude the
//!         // set is unrepairable from it.
//!     }
//! }
//! let rows = reader.into_blocks(); // feed to `Repair::new_owning`
//! # let _ = rows; }
//! ```

use std::collections::BTreeSet;
use std::fmt;
use std::io::{self, Read};
use std::path::Path;

use md5::{Digest, Md5};
use tracing::{debug, info, warn};

use crate::packets::{HEADER_SIZE, MAGIC};

/// RecoverySlice packet type identifier.
const TYPE_RECOVERY: &[u8; 16] = b"PAR 2.0\x00RecvSlic";

/// Default cap on bytes read from one volume by a [`RecoveryReader`].
pub const DEFAULT_MAX_VOLUME_BYTES: u64 = 512 * 1024 * 1024;

/// A recovery block extracted from a RecoverySlice packet.
#[derive(Debug, Clone)]
pub struct RecoveryBlock {
    /// The exponent (recovery block index) used in the Vandermonde matrix.
    pub exponent: u32,
    /// The recovery data. Length = slice_size.
    pub data: Vec<u8>,
}

/// How many recovery rows a [`RecoveryReader`] may hold at once.
///
/// Both limits apply when both are set; the tighter one wins. Rows over the
/// budget are dropped highest exponent first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RecoveryBudget {
    /// Keep at most this many rows.
    pub max_rows: Option<usize>,
    /// Keep at most this many bytes of row data (`rows × slice_size`).
    pub max_bytes: Option<u64>,
}

impl RecoveryBudget {
    /// No limit: keep every row seen.
    pub const UNLIMITED: Self = Self {
        max_rows: None,
        max_bytes: None,
    };

    /// Keep at most `n` rows.
    pub fn rows(n: usize) -> Self {
        Self {
            max_rows: Some(n),
            max_bytes: None,
        }
    }

    /// Keep at most `bytes` of row data.
    pub fn bytes(bytes: u64) -> Self {
        Self {
            max_rows: None,
            max_bytes: Some(bytes),
        }
    }

    /// Row limit for a set with this slice size.
    pub fn row_limit(&self, slice_size: u64) -> usize {
        let by_bytes = match self.max_bytes {
            Some(b) if slice_size > 0 => usize::try_from(b / slice_size).unwrap_or(usize::MAX),
            _ => usize::MAX,
        };
        self.max_rows.unwrap_or(usize::MAX).min(by_bytes)
    }
}

/// How reading one volume ended.
#[derive(Debug)]
pub enum VolumeEnd {
    /// Read to the end of the stream: the volume carries no more rows than
    /// were seen.
    Complete,
    /// The stream ended inside a packet. Rows after the cut are unknown.
    Truncated,
    /// Stopped at the per-volume byte cap with data left unread.
    CapReached,
    /// A read failed. Rows after the failure are unknown.
    Io(io::Error),
}

impl fmt::Display for VolumeEnd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Complete => write!(f, "read to end"),
            Self::Truncated => write!(f, "truncated inside a packet"),
            Self::CapReached => write!(f, "stopped at the per-volume byte cap"),
            Self::Io(e) => write!(f, "read error: {e}"),
        }
    }
}

/// What [`RecoveryReader::read_volume`] found in one volume.
#[derive(Debug)]
pub struct VolumeScan {
    /// Intact RecoverySlice packets for this set found in the volume,
    /// duplicates of exponents already held included. Packets skipped for
    /// budget are counted without their hash being checked.
    pub seen: usize,
    /// Rows from this volume that were added to the reader. A later volume can
    /// evict some of them again.
    pub kept: usize,
    /// RecoverySlice packets for this set dropped as damaged: a body too
    /// short for the slice size, or a packet hash mismatch.
    pub corrupt: usize,
    /// Bytes consumed from the stream.
    pub bytes_read: u64,
    /// Whether the volume was read to its end.
    pub end: VolumeEnd,
}

impl VolumeScan {
    /// Whether the whole volume was read. When `false`, the volume may carry
    /// rows that were never seen, so a shortfall must not be taken to mean
    /// the set is unrepairable.
    pub fn read_to_end(&self) -> bool {
        matches!(self.end, VolumeEnd::Complete)
    }
}

/// Streaming RecoverySlice reader for one recovery set, accumulating rows
/// across volumes under a [`RecoveryBudget`].
///
/// Rows are held in the order their packets were read. When the budget is
/// full, a new row is kept only if its exponent is lower than the highest one
/// held, which is then dropped. Peak memory is one slice over the budget.
#[derive(Debug, Clone)]
pub struct RecoveryReader {
    set_id: [u8; 16],
    slice_size: u64,
    limit: usize,
    max_volume_bytes: u64,
    blocks: Vec<RecoveryBlock>,
    exponents: BTreeSet<u32>,
    seen: usize,
}

impl RecoveryReader {
    /// Reader for recovery set `set_id` with slices of `slice_size` bytes, an
    /// unlimited budget and the [`DEFAULT_MAX_VOLUME_BYTES`] cap.
    pub fn new(set_id: [u8; 16], slice_size: u64) -> Self {
        Self {
            set_id,
            slice_size,
            limit: usize::MAX,
            max_volume_bytes: DEFAULT_MAX_VOLUME_BYTES,
            blocks: Vec::new(),
            exponents: BTreeSet::new(),
            seen: 0,
        }
    }

    /// Set the budget (builder form of [`set_budget`](Self::set_budget)).
    pub fn with_budget(mut self, budget: RecoveryBudget) -> Self {
        self.set_budget(budget);
        self
    }

    /// Cap the bytes read from one volume. A volume longer than this reports
    /// [`VolumeEnd::CapReached`].
    pub fn with_max_volume_bytes(mut self, max: u64) -> Self {
        self.max_volume_bytes = max;
        self
    }

    /// Change the budget. Shrinking it drops the highest-exponent rows now.
    pub fn set_budget(&mut self, budget: RecoveryBudget) {
        self.limit = budget.row_limit(self.slice_size);
        while self.blocks.len() > self.limit {
            self.evict_highest();
        }
    }

    /// Current row limit.
    pub fn row_limit(&self) -> usize {
        self.limit
    }

    /// Intact RecoverySlice packets for this set seen across all volumes.
    pub fn seen(&self) -> usize {
        self.seen
    }

    /// Rows held now.
    pub fn kept(&self) -> usize {
        self.blocks.len()
    }

    /// Rows held, in packet order.
    pub fn blocks(&self) -> &[RecoveryBlock] {
        &self.blocks
    }

    /// Take the held rows, in packet order.
    pub fn into_blocks(self) -> Vec<RecoveryBlock> {
        self.blocks
    }

    /// Read RecoverySlice packets from one volume.
    ///
    /// Packets for other sets, other packet types and rows outside the budget
    /// are read past without being buffered. Never fails: how the read ended
    /// is in [`VolumeScan::end`], and rows found before a failure are kept.
    pub fn read_volume<R: Read>(&mut self, reader: R) -> VolumeScan {
        let mut src = Source::new(reader, self.max_volume_bytes);
        let mut scan = VolumeScan {
            seen: 0,
            kept: 0,
            corrupt: 0,
            bytes_read: 0,
            end: VolumeEnd::Complete,
        };
        scan.end = match self.scan(&mut src, &mut scan) {
            Ok(()) => VolumeEnd::Complete,
            Err(Stop::Eof) => VolumeEnd::Truncated,
            Err(Stop::Cap) => VolumeEnd::CapReached,
            Err(Stop::Io(e)) => VolumeEnd::Io(e),
        };
        scan.bytes_read = src.pos;
        self.seen += scan.seen;
        scan
    }

    fn scan<R: Read>(&mut self, src: &mut Source<R>, scan: &mut VolumeScan) -> Result<(), Stop> {
        let mut hdr = [0u8; HEADER_SIZE];
        let mut have = 0;
        loop {
            let got = src.read_full(&mut hdr[have..])?;
            have += got;
            if have < HEADER_SIZE {
                // End of stream between packets. A partial packet header is
                // a cut; anything else is trailing garbage.
                return if have > 0 && starts_like_magic(&hdr[..have]) {
                    Err(Stop::Eof)
                } else {
                    Ok(())
                };
            }

            if &hdr[0..8] != MAGIC {
                // Resync: keep the window from the next possible magic start.
                let next = find_magic_start(&hdr[1..]).map_or(HEADER_SIZE, |i| i + 1);
                hdr.copy_within(next.., 0);
                have = HEADER_SIZE - next;
                continue;
            }
            let packet_len = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
            if packet_len < HEADER_SIZE as u64 || packet_len % 4 != 0 {
                let next = find_magic_start(&hdr[1..]).map_or(HEADER_SIZE, |i| i + 1);
                hdr.copy_within(next.., 0);
                have = HEADER_SIZE - next;
                continue;
            }
            have = 0;
            let body_len = packet_len - HEADER_SIZE as u64;

            if hdr[32..48] != self.set_id || &hdr[48..64] != TYPE_RECOVERY {
                src.skip(body_len)?;
                continue;
            }
            if body_len < 4 + self.slice_size {
                scan.corrupt += 1;
                src.skip(body_len)?;
                continue;
            }

            let mut exp_buf = [0u8; 4];
            src.read_exact(&mut exp_buf)?;
            let exponent = u32::from_le_bytes(exp_buf);
            let rest = body_len - 4;

            if !self.wants(exponent) {
                scan.seen += 1;
                src.skip(rest)?;
                continue;
            }

            let mut hasher = Md5::new();
            hasher.update(&hdr[32..64]);
            hasher.update(exp_buf);
            let mut data = vec![0u8; self.slice_size as usize];
            src.read_exact(&mut data)?;
            hasher.update(&data);
            src.skip_hashing(rest - self.slice_size, &mut hasher)?;
            let digest: [u8; 16] = hasher.finalize().into();
            if digest[..] != hdr[16..32] {
                scan.corrupt += 1;
                continue;
            }

            scan.seen += 1;
            scan.kept += 1;
            if self.blocks.len() >= self.limit {
                self.evict_highest();
            }
            self.exponents.insert(exponent);
            self.blocks.push(RecoveryBlock { exponent, data });
        }
    }

    /// Whether a new row with this exponent would be kept.
    fn wants(&self, exponent: u32) -> bool {
        if self.limit == 0 || self.exponents.contains(&exponent) {
            return false;
        }
        self.blocks.len() < self.limit || self.exponents.last().is_some_and(|&max| exponent < max)
    }

    fn evict_highest(&mut self) {
        let Some(max) = self.exponents.pop_last() else {
            return;
        };
        if let Some(i) = self.blocks.iter().position(|b| b.exponent == max) {
            self.blocks.remove(i);
        }
    }
}

/// Why a volume scan stopped before a clean end.
enum Stop {
    Eof,
    Cap,
    Io(io::Error),
}

/// A reader that counts bytes and refuses to go past a cap.
struct Source<R> {
    inner: R,
    pos: u64,
    cap: u64,
}

impl<R: Read> Source<R> {
    fn new(inner: R, cap: u64) -> Self {
        Self { inner, pos: 0, cap }
    }

    /// Read until `buf` is full or the stream ends; returns bytes read.
    fn read_full(&mut self, buf: &mut [u8]) -> Result<usize, Stop> {
        let mut n = 0;
        while n < buf.len() {
            let room = self.cap - self.pos;
            if room == 0 {
                return if self.at_eof()? {
                    Ok(n)
                } else {
                    Err(Stop::Cap)
                };
            }
            let want = (buf.len() - n).min(usize::try_from(room).unwrap_or(usize::MAX));
            match self.inner.read(&mut buf[n..n + want]) {
                Ok(0) => break,
                Ok(k) => {
                    n += k;
                    self.pos += k as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(Stop::Io(e)),
            }
        }
        Ok(n)
    }

    /// At the cap: probe whether the stream has anything left.
    fn at_eof(&mut self) -> Result<bool, Stop> {
        let mut probe = [0u8; 1];
        loop {
            match self.inner.read(&mut probe) {
                Ok(0) => return Ok(true),
                Ok(_) => return Ok(false),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(Stop::Io(e)),
            }
        }
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), Stop> {
        if self.read_full(buf)? < buf.len() {
            return Err(Stop::Eof);
        }
        Ok(())
    }

    fn skip(&mut self, n: u64) -> Result<(), Stop> {
        self.skip_with(n, |_| {})
    }

    fn skip_hashing(&mut self, n: u64, hasher: &mut Md5) -> Result<(), Stop> {
        self.skip_with(n, |chunk| hasher.update(chunk))
    }

    fn skip_with(&mut self, mut n: u64, mut f: impl FnMut(&[u8])) -> Result<(), Stop> {
        let mut buf = [0u8; 8192];
        while n > 0 {
            let want = n.min(buf.len() as u64) as usize;
            self.read_exact(&mut buf[..want])?;
            f(&buf[..want]);
            n -= want as u64;
        }
        Ok(())
    }
}

/// Offset of the first position in `buf` where the packet magic could start
/// (a full match, or a prefix of it running to the end of `buf`).
fn find_magic_start(buf: &[u8]) -> Option<usize> {
    (0..buf.len()).find(|&i| starts_like_magic(&buf[i..]))
}

fn starts_like_magic(buf: &[u8]) -> bool {
    let n = buf.len().min(MAGIC.len());
    buf[..n] == MAGIC[..n]
}

/// Load all recovery blocks from PAR2 files in a directory.
///
/// Scans all `.par2` files (index and volumes) for RecoverySlice packets
/// matching the given recovery set ID. Returns blocks sorted by exponent.
pub fn load_recovery_blocks(dir: &Path, set_id: &[u8; 16], slice_size: u64) -> Vec<RecoveryBlock> {
    // Find all .par2 files in the directory
    let mut par2_files: Vec<_> = match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("par2"))
            })
            .collect(),
        Err(e) => {
            warn!(error = %e, "Failed to read directory for recovery files");
            return Vec::new();
        }
    };
    par2_files.sort();

    let mut reader = RecoveryReader::new(*set_id, slice_size).with_max_volume_bytes(u64::MAX);
    for par2_path in &par2_files {
        let file = match std::fs::File::open(par2_path) {
            Ok(f) => f,
            Err(e) => {
                debug!(file = %par2_path.display(), error = %e, "Skipping unreadable file");
                continue;
            }
        };
        let scan = reader.read_volume(io::BufReader::with_capacity(64 * 1024, file));
        debug!(
            file = %par2_path.display(),
            count = scan.kept,
            corrupt = scan.corrupt,
            end = %scan.end,
            "Loaded recovery blocks"
        );
    }

    let mut blocks = reader.into_blocks();
    blocks.sort_by_key(|b| b.exponent);

    info!(total_blocks = blocks.len(), "Recovery blocks loaded");

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    const SET: [u8; 16] = [7u8; 16];

    fn packet(set_id: &[u8; 16], ptype: &[u8; 16], body: &[u8]) -> Vec<u8> {
        let mut hashed = Vec::new();
        hashed.extend_from_slice(set_id);
        hashed.extend_from_slice(ptype);
        hashed.extend_from_slice(body);
        let md5: [u8; 16] = Md5::digest(&hashed).into();
        let mut p = Vec::new();
        p.extend_from_slice(MAGIC);
        p.extend_from_slice(&((HEADER_SIZE + body.len()) as u64).to_le_bytes());
        p.extend_from_slice(&md5);
        p.extend_from_slice(&hashed);
        p
    }

    fn recv(exponent: u32, slice_size: usize) -> Vec<u8> {
        let mut body = exponent.to_le_bytes().to_vec();
        body.extend((0..slice_size).map(|i| (i as u32 ^ exponent) as u8));
        packet(&SET, TYPE_RECOVERY, &body)
    }

    fn volume(exponents: &[u32], slice_size: usize) -> Vec<u8> {
        exponents
            .iter()
            .flat_map(|&e| recv(e, slice_size))
            .collect()
    }

    fn exps(r: &RecoveryReader) -> Vec<u32> {
        r.blocks().iter().map(|b| b.exponent).collect()
    }

    #[test]
    fn test_load_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let set_id = [0u8; 16];
        let blocks = load_recovery_blocks(dir.path(), &set_id, 768000);
        assert!(blocks.is_empty());
    }

    #[test]
    fn packet_order_and_data() {
        let vol = volume(&[5, 2, 9, 0], 16);
        let mut r = RecoveryReader::new(SET, 16);
        let scan = r.read_volume(&vol[..]);
        assert!(scan.read_to_end());
        assert_eq!((scan.seen, scan.kept, scan.corrupt), (4, 4, 0));
        assert_eq!(scan.bytes_read, vol.len() as u64);
        assert_eq!(exps(&r), [5, 2, 9, 0]);
        assert_eq!(r.blocks()[1].data, recv(2, 16)[HEADER_SIZE + 4..]);
    }

    #[test]
    fn budget_keeps_lowest_exponents_in_packet_order() {
        let vol = volume(&[5, 2, 9, 0, 7, 1], 16);
        let mut r = RecoveryReader::new(SET, 16).with_budget(RecoveryBudget::rows(3));
        let scan = r.read_volume(&vol[..]);
        assert!(scan.read_to_end());
        assert_eq!(scan.seen, 6);
        assert_eq!(r.seen(), 6);
        assert_eq!(r.kept(), 3);
        assert_eq!(exps(&r), [2, 0, 1]);
    }

    #[test]
    fn byte_budget_and_shrinking() {
        let vol = volume(&[3, 1, 2, 0], 16);
        let mut r = RecoveryReader::new(SET, 16).with_budget(RecoveryBudget::bytes(16 * 3 + 5));
        r.read_volume(&vol[..]);
        assert_eq!(exps(&r), [1, 2, 0]);
        r.set_budget(RecoveryBudget::rows(1));
        assert_eq!(exps(&r), [0]);
    }

    #[test]
    fn zero_budget_counts_without_keeping() {
        let vol = volume(&[1, 2, 3], 16);
        let mut r = RecoveryReader::new(SET, 16).with_budget(RecoveryBudget::rows(0));
        let scan = r.read_volume(&vol[..]);
        assert!(scan.read_to_end());
        assert_eq!((scan.seen, scan.kept), (3, 0));
    }

    #[test]
    fn accumulates_across_volumes_and_skips_duplicates() {
        let mut r = RecoveryReader::new(SET, 16).with_budget(RecoveryBudget::rows(3));
        r.read_volume(&volume(&[4, 5, 6], 16)[..]);
        let scan = r.read_volume(&volume(&[5, 1], 16)[..]);
        assert_eq!((scan.seen, scan.kept), (2, 1));
        assert_eq!(exps(&r), [4, 5, 1]);
        assert_eq!(r.seen(), 5);
    }

    #[test]
    fn truncated_volume_is_not_complete() {
        let vol = volume(&[0, 1, 2], 16);
        for cut in [vol.len() - 1, vol.len() - 10, 2 * vol.len() / 3 + 3] {
            let mut r = RecoveryReader::new(SET, 16);
            let scan = r.read_volume(&vol[..cut]);
            assert!(!scan.read_to_end(), "cut at {cut}");
            assert!(matches!(scan.end, VolumeEnd::Truncated));
            assert_eq!(exps(&r), [0, 1]);
        }
        // Cut inside a skipped (over-budget) packet is still a truncation.
        let mut r = RecoveryReader::new(SET, 16).with_budget(RecoveryBudget::rows(1));
        let scan = r.read_volume(&vol[..vol.len() - 3]);
        assert!(matches!(scan.end, VolumeEnd::Truncated));
        // Cut inside the next packet's header.
        let mut r = RecoveryReader::new(SET, 16);
        let one = recv(0, 16);
        let scan = r.read_volume(&vol[..one.len() + 5]);
        assert!(matches!(scan.end, VolumeEnd::Truncated));
    }

    #[test]
    fn cap_reached_is_not_complete() {
        let vol = volume(&[0, 1, 2], 16);
        let one = recv(0, 16).len() as u64;
        let mut r = RecoveryReader::new(SET, 16).with_max_volume_bytes(one * 2);
        let scan = r.read_volume(&vol[..]);
        assert!(matches!(scan.end, VolumeEnd::CapReached));
        assert_eq!(scan.bytes_read, one * 2);
        assert_eq!(exps(&r), [0, 1]);

        // A volume exactly at the cap is complete.
        let mut r = RecoveryReader::new(SET, 16).with_max_volume_bytes(vol.len() as u64);
        assert!(r.read_volume(&vol[..]).read_to_end());
    }

    #[test]
    fn io_error_is_reported() {
        struct Failing<'a>(&'a [u8]);
        impl Read for Failing<'_> {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                if self.0.is_empty() {
                    return Err(io::Error::other("boom"));
                }
                let n = self.0.read(buf)?;
                Ok(n)
            }
        }
        let vol = volume(&[0, 1], 16);
        let mut r = RecoveryReader::new(SET, 16);
        let scan = r.read_volume(Failing(&vol[..vol.len() - 4]));
        assert!(matches!(scan.end, VolumeEnd::Io(_)));
        assert_eq!(exps(&r), [0]);
    }

    #[test]
    fn skips_other_sets_types_garbage_and_corrupt() {
        let mut vol = b"junk!".to_vec();
        vol.extend(packet(&[9u8; 16], TYPE_RECOVERY, &[0u8; 20]));
        vol.extend(packet(&SET, b"PAR 2.0\x00Creator\x00", b"test"));
        vol.extend(recv(3, 16));
        let mut bad = recv(4, 16);
        let last = bad.len() - 1;
        bad[last] ^= 1;
        vol.extend(bad);
        vol.extend(packet(&SET, TYPE_RECOVERY, &[0u8; 8])); // too short
        vol.extend(recv(1, 16));
        vol.extend(b"tail");

        let mut r = RecoveryReader::new(SET, 16);
        let scan = r.read_volume(&vol[..]);
        assert!(scan.read_to_end(), "{}", scan.end);
        assert_eq!((scan.seen, scan.kept, scan.corrupt), (2, 2, 2));
        assert_eq!(exps(&r), [3, 1]);
    }

    #[test]
    fn empty_input_is_complete() {
        let mut r = RecoveryReader::new(SET, 16);
        let scan = r.read_volume(&[][..]);
        assert!(scan.read_to_end());
        assert_eq!(scan.seen, 0);
    }
}
