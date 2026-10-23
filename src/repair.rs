//! PAR2 repair engine.
//!
//! Repairs damaged or missing files using Reed-Solomon recovery data.
//!
//! Algorithm (D×D reduced approach — D = damaged blocks, N = total blocks):
//! 1. Verify to identify damaged/missing blocks
//! 2. Load recovery blocks from volume files
//! 3. Build a D×D Vandermonde submatrix (only damaged columns × recovery rows)
//! 4. Invert the D×D matrix — O(D³) instead of O(N²D) for the full N×N
//! 5. Compute adjusted recovery: subtract intact-block contributions from recovery data
//! 6. Apply the D×D inverse to produce repaired blocks
//! 7. Write repaired blocks back to files
//!
//! Steps 3–6 are the [`Repair`] builder, which works on slices in memory and
//! does no I/O. The file-based functions ([`repair`], [`repair_from_verify`],
//! [`repair_from_verify_no_reverify`]) read intact slices from disk, feed them
//! to a [`Repair`], and write the solved slices back.
//!
//! # In-memory repair
//!
//! ```
//! use rust_par2::recovery::RecoveryBlock;
//! use rust_par2::repair::Repair;
//! # fn demo(recovery: Vec<RecoveryBlock>, present: Vec<(usize, Vec<u8>)>) -> Result<(), rust_par2::RepairError> {
//! // Global slice indices run across files in the Main packet's file order;
//! // `BlockMap` maps them to (file, offset, length).
//! let (slice_size, total_slices) = (768_000, 1_000);
//!
//! let mut r = Repair::new_owning(recovery, slice_size, total_slices)?;
//! for (index, bytes) in &present {
//!     // Check each slice against its IFSC checksum first: see `slice::SliceHasher`.
//!     r.feed(*index, bytes)?;
//! }
//! let missing = [17, 18, 404];
//! let rebuilt = r.solve(&missing)?; // index -> slice_size bytes (zero-padded)
//! # let _ = rebuilt; Ok(()) }
//! ```
//!
//! # Memory model
//!
//! A [`Repair`] holds one accumulator per recovery row: peak memory is
//! `rows × slice_size`, plus `D × slice_size` for the solved output, plus
//! whatever the caller buffers before feeding. Feeding allocates nothing for
//! [`Repair::feed`]; [`Repair::feed_batch`] allocates one small coefficient
//! vector per row per call. To bound memory, load or keep only as many recovery
//! rows as there are missing slices ([`Repair::truncate_rows`] drops the rest at
//! any time — each row is an independent equation).

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use rayon::prelude::*;
use tracing::{debug, info};

use crate::gf;
use crate::gf_simd;
use crate::matrix::{GfMatrix, par2_input_constants};
use crate::recovery::{RecoveryBlock, load_recovery_blocks};
use crate::types::{Id16, Par2FileSet, VerifyResult};
use crate::verify;

/// Result of a repair operation.
#[derive(Debug)]
pub struct RepairResult {
    /// Whether the repair succeeded (all files now intact).
    pub success: bool,
    /// Number of blocks repaired.
    pub blocks_repaired: u32,
    /// Number of files repaired.
    pub files_repaired: usize,
    /// Descriptive message.
    pub message: String,
}

impl fmt::Display for RepairResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.success {
            write!(
                f,
                "Repair complete: {} blocks repaired across {} files",
                self.blocks_repaired, self.files_repaired
            )
        } else {
            write!(f, "Repair failed: {}", self.message)
        }
    }
}

/// Errors that can occur during repair.
#[derive(Debug, thiserror::Error)]
pub enum RepairError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Insufficient recovery data: need {needed} blocks, have {available}")]
    InsufficientRecovery { needed: u32, available: u32 },
    #[error("Decode matrix is singular — cannot repair with these recovery blocks")]
    SingularMatrix,
    #[error("No damage detected — nothing to repair")]
    NoDamage,
    #[error("Verification after repair failed: {0}")]
    VerifyFailed(String),
    #[error("Invalid slice size {0}: must be a non-zero multiple of 4")]
    InvalidSliceSize(usize),
    #[error("Recovery block {exponent} is {len} bytes, expected the slice size {slice_size}")]
    InvalidRecoveryBlock {
        exponent: u32,
        len: usize,
        slice_size: usize,
    },
    #[error("Slice index {index} out of range: the set has {total} input slices")]
    SliceOutOfRange { index: usize, total: usize },
    #[error("Slice {index} is {len} bytes, longer than the slice size {slice_size}")]
    SliceTooLong {
        index: usize,
        len: usize,
        slice_size: usize,
    },
    #[error("Slice {index} was already fed (feeding twice cancels it out)")]
    DuplicateSlice { index: usize },
    #[error("{unaccounted} input slices were neither fed nor listed as missing")]
    IncompleteFeed { unaccounted: usize },
}

// ---------------------------------------------------------------------------
// In-memory builder
// ---------------------------------------------------------------------------

/// Incremental, I/O-free PAR2 repair over slices held in memory.
///
/// Load recovery rows, [`feed`](Self::feed) every present input slice once,
/// then [`solve`](Self::solve) for the missing ones. Each feed XOR-accumulates
/// `c_i^exp_e × D_i` into every recovery row `e`, so the slices can arrive in
/// any order and need not be kept after they are fed.
///
/// A fed slice cannot be removed. Feeding a corrupt slice makes every solved
/// slice wrong, so verify slices first (see
/// [`SliceHasher`](crate::slice::SliceHasher) and
/// [`Par2File::slice_matches`](crate::Par2File::slice_matches)).
///
/// Slice indices are global: they run across the set's files in the Main
/// packet's file order. [`BlockMap`] converts between global indices and file
/// positions.
#[derive(Debug, Clone)]
pub struct Repair {
    slice_size: usize,
    /// PAR2 input constant for every global slice index.
    constants: Vec<u16>,
    /// Recovery exponent of each row, ascending.
    exponents: Vec<u32>,
    /// Accumulators, one per row: `R_e ⊕ Σ_fed c_i^exp_e × D_i`.
    rows: Vec<Vec<u8>>,
    fed: Vec<bool>,
    fed_count: usize,
}

impl Repair {
    /// Build from borrowed recovery blocks (their data is copied).
    pub fn new(
        recovery: &[RecoveryBlock],
        slice_size: usize,
        total_input_slices: usize,
    ) -> Result<Self, RepairError> {
        Self::new_owning(recovery.to_vec(), slice_size, total_input_slices)
    }

    /// Build from owned recovery blocks, reusing their buffers as the row
    /// accumulators.
    ///
    /// Rows are sorted by exponent and duplicate exponents dropped. Every block
    /// must be exactly `slice_size` bytes.
    pub fn new_owning(
        mut recovery: Vec<RecoveryBlock>,
        slice_size: usize,
        total_input_slices: usize,
    ) -> Result<Self, RepairError> {
        if slice_size == 0 || slice_size % 4 != 0 {
            return Err(RepairError::InvalidSliceSize(slice_size));
        }
        if let Some(bad) = recovery.iter().find(|b| b.data.len() != slice_size) {
            return Err(RepairError::InvalidRecoveryBlock {
                exponent: bad.exponent,
                len: bad.data.len(),
                slice_size,
            });
        }
        recovery.sort_by_key(|b| b.exponent);
        recovery.dedup_by_key(|b| b.exponent);

        let (exponents, rows) = recovery.into_iter().map(|b| (b.exponent, b.data)).unzip();
        Ok(Self {
            slice_size,
            constants: par2_input_constants(total_input_slices),
            exponents,
            rows,
            fed: vec![false; total_input_slices],
            fed_count: 0,
        })
    }

    /// Number of recovery rows loaded: the most missing slices `solve` can
    /// rebuild.
    pub fn rows(&self) -> usize {
        self.rows.len()
    }

    /// Recovery exponents of the loaded rows, ascending.
    pub fn exponents(&self) -> &[u32] {
        &self.exponents
    }

    /// Slice size in bytes.
    pub fn slice_size(&self) -> usize {
        self.slice_size
    }

    /// Total number of input slices in the set.
    pub fn total_slices(&self) -> usize {
        self.fed.len()
    }

    /// Number of slices fed so far.
    pub fn fed_count(&self) -> usize {
        self.fed_count
    }

    /// Whether slice `index` has been fed.
    pub fn is_fed(&self, index: usize) -> bool {
        self.fed.get(index).copied().unwrap_or(false)
    }

    /// Keep only the first `n` rows (lowest exponents), freeing the rest.
    ///
    /// Safe at any point: each row is an independent equation. `solve` only
    /// uses as many rows as there are missing slices, so once that number is
    /// known the others are dead weight.
    pub fn truncate_rows(&mut self, n: usize) {
        self.rows.truncate(n);
        self.exponents.truncate(n);
    }

    /// Feed one present input slice.
    ///
    /// `data` is the slice as stored: `slice_size` bytes, or fewer for the last
    /// slice of a file, which is treated as zero-padded.
    pub fn feed(&mut self, index: usize, data: &[u8]) -> Result<(), RepairError> {
        self.check_feed(index, data)?;
        let c = self.constants[index];
        self.rows
            .par_iter_mut()
            .zip(self.exponents.par_iter())
            .for_each(|(row, &exp)| mul_add_prefix(row, data, gf::pow(c, exp)));
        self.fed[index] = true;
        self.fed_count += 1;
        Ok(())
    }

    /// Feed several present slices at once.
    ///
    /// Each row is updated against the whole batch while it is hot in cache,
    /// using the pair-batched SIMD path, so batches of a few dozen slices are
    /// much faster than the same slices fed one by one. The batch is validated
    /// before anything is accumulated: on error, nothing from it was fed.
    pub fn feed_batch(&mut self, slices: &[(usize, &[u8])]) -> Result<(), RepairError> {
        for (k, &(index, data)) in slices.iter().enumerate() {
            if let Err(e) = self.check_feed(index, data) {
                for &(earlier, _) in &slices[..k] {
                    self.fed[earlier] = false;
                }
                return Err(e);
            }
            // Marking as we go also rejects duplicates within the batch.
            self.fed[index] = true;
        }

        let slice_size = self.slice_size;
        let (full, short): (Vec<_>, Vec<_>) = slices
            .iter()
            .copied()
            .partition(|(_, data)| data.len() == slice_size);
        let full_data: Vec<&[u8]> = full.iter().map(|&(_, d)| d).collect();

        let constants = &self.constants;
        self.rows
            .par_iter_mut()
            .zip(self.exponents.par_iter())
            .for_each(|(row, &exp)| {
                let coeffs: Vec<u16> = full
                    .iter()
                    .map(|&(i, _)| gf::pow(constants[i], exp))
                    .collect();
                gf_simd::mul_add_multi(row, &full_data, &coeffs);
                for &(i, data) in &short {
                    mul_add_prefix(row, data, gf::pow(constants[i], exp));
                }
            });

        self.fed_count += slices.len();
        Ok(())
    }

    /// Rebuild the `missing` slices.
    ///
    /// Every slice must be either fed or listed in `missing`. Returns each
    /// missing index mapped to its full `slice_size` bytes; the last slice of a
    /// file comes back zero-padded, and the caller writes only its real length
    /// (see [`BlockMap::global_to_file`]).
    ///
    /// Uses the first `missing.len()` rows. Errors with
    /// [`RepairError::InsufficientRecovery`] if there are fewer rows than
    /// missing slices, and [`RepairError::SingularMatrix`] if the chosen rows
    /// are linearly dependent.
    pub fn solve(&self, missing: &[usize]) -> Result<HashMap<usize, Vec<u8>>, RepairError> {
        let mut missing = missing.to_vec();
        missing.sort_unstable();
        missing.dedup();
        let outputs = self.reconstruct(&missing)?;
        Ok(missing.into_iter().zip(outputs).collect())
    }

    fn check_feed(&self, index: usize, data: &[u8]) -> Result<(), RepairError> {
        let total = self.fed.len();
        if index >= total {
            return Err(RepairError::SliceOutOfRange { index, total });
        }
        if data.len() > self.slice_size {
            return Err(RepairError::SliceTooLong {
                index,
                len: data.len(),
                slice_size: self.slice_size,
            });
        }
        if self.fed[index] {
            return Err(RepairError::DuplicateSlice { index });
        }
        Ok(())
    }

    /// Solve for sorted, deduplicated `missing`; outputs are in the same order.
    fn reconstruct(&self, missing: &[usize]) -> Result<Vec<Vec<u8>>, RepairError> {
        let total = self.fed.len();
        for &index in missing {
            if index >= total {
                return Err(RepairError::SliceOutOfRange { index, total });
            }
            if self.fed[index] {
                return Err(RepairError::DuplicateSlice { index });
            }
        }
        let unaccounted = total - self.fed_count - missing.len();
        if unaccounted != 0 {
            return Err(RepairError::IncompleteFeed { unaccounted });
        }

        let d = missing.len();
        if d == 0 {
            return Ok(Vec::new());
        }
        if d > self.rows.len() {
            return Err(RepairError::InsufficientRecovery {
                needed: d as u32,
                available: self.rows.len() as u32,
            });
        }

        // Rearranging the encoding equation R_e = Σ_i c_i^exp_e × D_i for the
        // missing slices only:
        //   R_e ⊕ Σ_{fed i} c_i^exp_e × D_i = Σ_{missing j} c_j^exp_e × D_j
        // The left side is what the rows hold. Invert V[e][j] = c_j^exp_e.
        let mut vandermonde = GfMatrix::zeros(d, d);
        for (e, &exp) in self.exponents[..d].iter().enumerate() {
            for (j, &idx) in missing.iter().enumerate() {
                vandermonde.set(e, j, gf::pow(self.constants[idx], exp));
            }
        }
        let inverse = vandermonde.invert().ok_or(RepairError::SingularMatrix)?;
        debug!(d, "D×D decode matrix inverted");

        // repaired[j] = Σ_e V⁻¹[j][e] × row[e]
        let row_refs: Vec<&[u8]> = self.rows[..d].iter().map(Vec::as_slice).collect();
        let mut outputs: Vec<Vec<u8>> = (0..d).map(|_| vec![0u8; self.slice_size]).collect();
        outputs.par_iter_mut().enumerate().for_each(|(j, dst)| {
            let coeffs: Vec<u16> = (0..d).map(|e| inverse.get(j, e)).collect();
            gf_simd::mul_add_multi(dst, &row_refs, &coeffs);
        });
        Ok(outputs)
    }
}

/// `dst ^= c × src` over `src.len()` bytes, `src` treated as zero-padded to
/// `dst.len()`. Padding contributes nothing, so only the prefix is touched; an
/// odd trailing byte is the low half of a GF(2^16) word whose high byte is 0.
fn mul_add_prefix(dst: &mut [u8], src: &[u8], c: u16) {
    let even = src.len() & !1;
    gf_simd::mul_add_buffer(&mut dst[..even], &src[..even], c);
    if even < src.len() {
        let p = gf::mul(c, src[even] as u16);
        dst[even] ^= p as u8;
        dst[even + 1] ^= (p >> 8) as u8;
    }
}

// ---------------------------------------------------------------------------
// File-based repair
// ---------------------------------------------------------------------------

/// Repair damaged/missing files in a PAR2 set.
///
/// This is a blocking operation. For async contexts, wrap in `spawn_blocking`.
///
/// Runs verification internally to identify damage. If you already have a
/// [`VerifyResult`] from a prior [`verify()`](crate::verify) call, use
/// [`repair_from_verify`] instead to skip the redundant verification pass.
pub fn repair(file_set: &Par2FileSet, dir: &Path) -> Result<RepairResult, RepairError> {
    let verify_result = verify::verify(file_set, dir);
    repair_from_verify_inner(file_set, dir, &verify_result, true)
}

/// Repair using a pre-computed [`VerifyResult`].
///
/// This skips the initial verification pass, saving significant time when the
/// caller has already called [`verify()`](crate::verify). The `verify_result`
/// must have been computed against the same `file_set` and `dir`.
///
/// Uses a reduced D×D matrix approach where D = number of damaged blocks.
/// The full N×N decode matrix is never constructed — only a small Vandermonde
/// submatrix covering damaged positions is built and inverted. This reduces
/// matrix inversion from O(N²D) to O(D³), which is the difference between
/// minutes and milliseconds for typical repair scenarios.
///
/// After repair, a full MD5 verification pass confirms all files are correct.
/// To skip re-verification, use [`repair_from_verify_no_reverify`].
///
/// This is a blocking operation. For async contexts, wrap in `spawn_blocking`.
pub fn repair_from_verify(
    file_set: &Par2FileSet,
    dir: &Path,
    verify_result: &VerifyResult,
) -> Result<RepairResult, RepairError> {
    repair_from_verify_inner(file_set, dir, verify_result, true)
}

/// Like [`repair_from_verify`], but skips re-verification after repair.
///
/// The Reed-Solomon math is deterministic — if the matrix inverted and I/O
/// succeeded, the output is correct. Skipping re-verify saves a full read
/// of all files (~6s per 5GB).
pub fn repair_from_verify_no_reverify(
    file_set: &Par2FileSet,
    dir: &Path,
    verify_result: &VerifyResult,
) -> Result<RepairResult, RepairError> {
    repair_from_verify_inner(file_set, dir, verify_result, false)
}

fn repair_from_verify_inner(
    file_set: &Par2FileSet,
    dir: &Path,
    verify_result: &VerifyResult,
    re_verify: bool,
) -> Result<RepairResult, RepairError> {
    if verify_result.all_correct() {
        return Err(RepairError::NoDamage);
    }

    let blocks_needed = verify_result.blocks_needed();
    info!(
        blocks_needed,
        damaged = verify_result.damaged.len(),
        missing = verify_result.missing.len(),
        "Repair: damage detected"
    );

    // Step 1: Load recovery blocks from all volume files
    let recovery_blocks = load_recovery_blocks(dir, &file_set.recovery_set_id, file_set.slice_size);

    // Step 2: Map files to global block indices
    let block_map = build_block_map(file_set);
    let total_input_blocks = block_map.total_blocks as usize;
    let slice_size = file_set.slice_size as usize;

    let mut fold = Repair::new_owning(recovery_blocks, slice_size, total_input_blocks)?;
    if (fold.rows() as u32) < blocks_needed {
        return Err(RepairError::InsufficientRecovery {
            needed: blocks_needed,
            available: fold.rows() as u32,
        });
    }

    let damaged_indices = find_damaged_block_indices(verify_result, &block_map);
    info!(
        damaged_block_count = damaged_indices.len(),
        total_input_blocks, "Mapped damaged blocks to global indices"
    );

    // Exactly D recovery rows are needed (one per damaged block).
    fold.truncate_rows(damaged_indices.len());

    // Step 3: Fold intact blocks into the recovery rows.
    //
    // Cache-tiled: the D row buffers (D × slice_size) typically exceed L3.
    // Iterating source-major thrashes cache on every source. Instead we read
    // batches of B source blocks and feed_batch processes each row against the
    // whole batch, keeping the row hot in L1/L2 while applying B sources via
    // mul_add_multi's pair-batched AVX2 path.
    let damaged_set: HashSet<usize> = damaged_indices.iter().copied().collect();
    let intact_indices: Vec<usize> = (0..total_input_blocks)
        .filter(|i| !damaged_set.contains(i))
        .collect();

    // Batch size: chosen so B × slice_size fits comfortably in L3 cache (~16-32MB).
    // With 768KB slices, 24 blocks = ~18MB. Leaves room for output buffer in L1/L2.
    const BATCH_SIZE: usize = 24;

    // I/O is sequential so OS readahead keeps the pipeline full.
    let mut file_handles: HashMap<String, std::fs::File> = HashMap::new();

    for batch in intact_indices.chunks(BATCH_SIZE) {
        let batch_data: Vec<Vec<u8>> = batch
            .iter()
            .map(|&idx| read_source_block(dir, &block_map, idx, slice_size, &mut file_handles))
            .collect::<std::io::Result<Vec<_>>>()?;
        let pairs: Vec<(usize, &[u8])> = batch
            .iter()
            .copied()
            .zip(batch_data.iter().map(Vec::as_slice))
            .collect();
        fold.feed_batch(&pairs)?;
    }

    info!("Intact-block contributions subtracted from recovery data");

    // Steps 4–5: invert the D×D matrix and apply it.
    let mut repaired = fold.solve(&damaged_indices)?;
    drop(fold);

    info!("Repaired blocks reconstructed via D×D inverse");

    // Step 6: Write repaired blocks back to files
    let mut files_touched = HashSet::new();

    for global_idx in &damaged_indices {
        let data = repaired
            .remove(global_idx)
            .expect("solve returns every missing index");
        let loc = block_map
            .global_to_file(*global_idx)
            .expect("damaged index comes from the block map");
        let filename = &loc.file.filename;

        let file_path = dir.join(filename);
        debug!(
            filename,
            global_block = global_idx,
            offset = loc.offset,
            len = loc.len,
            "Writing repaired block"
        );

        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&file_path)?;

        // Ensure file is the right size (for missing files)
        if f.metadata()?.len() < loc.file.file_size {
            f.set_len(loc.file.file_size)?;
        }

        f.seek(SeekFrom::Start(loc.offset))?;
        f.write_all(&data[..loc.len])?;
        files_touched.insert(filename.clone());
    }

    let blocks_repaired = damaged_indices.len() as u32;

    // Step 7: Optional re-verify
    if re_verify {
        let verification = verify::verify(file_set, dir);
        if verification.all_correct() {
            info!(
                blocks = blocks_repaired,
                files = files_touched.len(),
                "Repair successful — all files verified"
            );
            Ok(RepairResult {
                success: true,
                blocks_repaired,
                files_repaired: files_touched.len(),
                message: "All files repaired and verified".to_string(),
            })
        } else {
            Err(RepairError::VerifyFailed(format!("{verification}")))
        }
    } else {
        info!(
            blocks = blocks_repaired,
            files = files_touched.len(),
            "Repair complete (re-verify skipped)"
        );
        Ok(RepairResult {
            success: true,
            blocks_repaired,
            files_repaired: files_touched.len(),
            message: "All files repaired (re-verify skipped)".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Block mapping
// ---------------------------------------------------------------------------

/// Maps between global slice indices and per-file slice positions.
///
/// Global indices run across the set's files in the Main packet's file order
/// (falling back to File ID order if the Main packet listed none); this is the
/// order the PAR2 input constants are assigned in.
#[derive(Debug, Clone)]
pub struct BlockMap {
    /// Files in global order.
    pub files: Vec<BlockFile>,
    /// Total number of input slices across all files.
    pub total_blocks: u32,
    /// Slice size in bytes.
    pub slice_size: u64,
}

/// One file's place in a [`BlockMap`].
#[derive(Debug, Clone)]
pub struct BlockFile {
    /// File ID from the FileDesc packet.
    pub file_id: Id16,
    /// Filename from the FileDesc packet.
    pub filename: String,
    /// File size in bytes.
    pub file_size: u64,
    /// Number of slices in this file.
    pub block_count: u32,
    /// First global slice index for this file.
    pub start_block: u32,
}

/// Where a global slice lives on disk. Returned by [`BlockMap::global_to_file`].
#[derive(Debug, Clone, Copy)]
pub struct SliceLocation<'a> {
    /// The file holding the slice.
    pub file: &'a BlockFile,
    /// Slice index within the file (0-based).
    pub local_index: u32,
    /// Byte offset of the slice within the file.
    pub offset: u64,
    /// Bytes of real data: `slice_size`, or less for the file's last slice.
    pub len: usize,
}

/// Build the global slice map for a file set.
pub fn build_block_map(file_set: &Par2FileSet) -> BlockMap {
    let slice_size = file_set.slice_size;
    let mut files = Vec::new();
    let mut block_offset = 0u32;

    let ordered_files: Vec<_> = if file_set.file_order.is_empty() {
        let mut fallback: Vec<_> = file_set.files.values().collect();
        fallback.sort_by_key(|f| f.file_id);
        fallback
    } else {
        file_set
            .file_order
            .iter()
            .filter_map(|id| file_set.files.get(id))
            .collect()
    };

    for f in ordered_files {
        let block_count = if slice_size == 0 {
            0
        } else {
            f.size.div_ceil(slice_size) as u32
        };
        files.push(BlockFile {
            file_id: f.file_id,
            filename: f.filename.clone(),
            file_size: f.size,
            block_count,
            start_block: block_offset,
        });
        block_offset += block_count;
    }

    BlockMap {
        files,
        total_blocks: block_offset,
        slice_size,
    }
}

impl BlockMap {
    /// Build the global slice map for a file set. Same as [`build_block_map`].
    pub fn new(file_set: &Par2FileSet) -> Self {
        build_block_map(file_set)
    }

    /// Locate a global slice index: its file, byte offset and real length.
    /// `None` if the index is out of range.
    pub fn global_to_file(&self, global_idx: usize) -> Option<SliceLocation<'_>> {
        let global = u32::try_from(global_idx).ok()?;
        // Files are in start_block order; empty files end where they start and
        // are skipped.
        let pos = self
            .files
            .partition_point(|f| f.start_block + f.block_count <= global);
        let file = self.files.get(pos)?;
        if global < file.start_block {
            return None;
        }
        let local_index = global - file.start_block;
        let offset = local_index as u64 * self.slice_size;
        let len = (file.file_size - offset).min(self.slice_size) as usize;
        Some(SliceLocation {
            file,
            local_index,
            offset,
            len,
        })
    }

    /// Global index of slice `local_index` of the file named `filename`.
    pub fn global_index(&self, filename: &str, local_index: u32) -> Option<usize> {
        let f = self.file(filename)?;
        (local_index < f.block_count).then(|| (f.start_block + local_index) as usize)
    }

    /// The entry for the file named `filename`.
    pub fn file(&self, filename: &str) -> Option<&BlockFile> {
        self.files.iter().find(|f| f.filename == filename)
    }
}

fn find_damaged_block_indices(verify_result: &VerifyResult, block_map: &BlockMap) -> Vec<usize> {
    let mut indices = Vec::new();

    for damaged in &verify_result.damaged {
        if let Some(bf) = block_map.file(&damaged.filename) {
            if damaged.damaged_block_indices.is_empty() {
                // No per-block info — assume all blocks damaged
                for i in 0..bf.block_count {
                    indices.push((bf.start_block + i) as usize);
                }
            } else {
                // Use precise per-block damage info
                for &local_idx in &damaged.damaged_block_indices {
                    indices.push((bf.start_block + local_idx) as usize);
                }
            }
        }
    }

    for missing in &verify_result.missing {
        if let Some(bf) = block_map.file(&missing.filename) {
            for i in 0..bf.block_count {
                indices.push((bf.start_block + i) as usize);
            }
        }
    }

    indices.sort();
    indices.dedup();
    indices
}

/// Read a single source block from disk, reusing file handles.
fn read_source_block(
    dir: &Path,
    block_map: &BlockMap,
    global_idx: usize,
    slice_size: usize,
    file_handles: &mut HashMap<String, std::fs::File>,
) -> std::io::Result<Vec<u8>> {
    let loc = block_map
        .global_to_file(global_idx)
        .expect("intact index comes from the block map");

    let handle = match file_handles.entry(loc.file.filename.clone()) {
        Entry::Occupied(e) => e.into_mut(),
        Entry::Vacant(e) => {
            let path = dir.join(&loc.file.filename);
            e.insert(std::fs::File::open(&path)?)
        }
    };
    handle.seek(SeekFrom::Start(loc.offset))?;

    let mut buf = vec![0u8; slice_size]; // zero-initialized for last-block padding
    let mut total = 0;
    while total < slice_size {
        match handle.read(&mut buf[total..]) {
            Ok(0) => break,
            Ok(n) => total += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crc32fast::hash as crc32_hash;
    use md5::{Digest, Md5};

    use crate::packets::{HEADER_SIZE, MAGIC};

    const TYPE_MAIN: &[u8; 16] = b"PAR 2.0\x00Main\x00\x00\x00\x00";
    const TYPE_FILE_DESC: &[u8; 16] = b"PAR 2.0\x00FileDesc";
    const TYPE_IFSC: &[u8; 16] = b"PAR 2.0\x00IFSC\x00\x00\x00\x00";
    const TYPE_RECOVERY: &[u8; 16] = b"PAR 2.0\x00RecvSlic";

    fn build_packet(set_id: [u8; 16], packet_type: &[u8; 16], body: &[u8]) -> Vec<u8> {
        let mut data = Vec::with_capacity(32 + body.len());
        data.extend_from_slice(&set_id);
        data.extend_from_slice(packet_type);
        data.extend_from_slice(body);

        let mut packet = Vec::with_capacity(32 + data.len());
        packet.extend_from_slice(MAGIC);
        packet.extend_from_slice(&((HEADER_SIZE + body.len()) as u64).to_le_bytes());
        packet.extend_from_slice(&Md5::digest(&data));
        packet.extend_from_slice(&data);
        packet
    }

    fn file_md5(data: &[u8]) -> [u8; 16] {
        Md5::digest(data).into()
    }

    fn hash_16k(data: &[u8]) -> [u8; 16] {
        Md5::digest(&data[..data.len().min(16_384)]).into()
    }

    fn padded_block(data: &[u8], block_index: usize, slice_size: usize) -> Vec<u8> {
        let start = block_index * slice_size;
        let end = (start + slice_size).min(data.len());
        let mut block = vec![0u8; slice_size];
        if start < data.len() {
            block[..end - start].copy_from_slice(&data[start..end]);
        }
        block
    }

    fn write_par2_fixture(
        dir: &Path,
        slice_size: usize,
        files: &[(&str, [u8; 16], Vec<u8>)],
        main_order: &[[u8; 16]],
        recovery_exponents: &[u32],
    ) -> PathBuf {
        let set_id = [0x5Au8; 16];
        let mut index_bytes = Vec::new();

        let mut main_body = Vec::new();
        main_body.extend_from_slice(&(slice_size as u64).to_le_bytes());
        main_body.extend_from_slice(&(main_order.len() as u32).to_le_bytes());
        for file_id in main_order {
            main_body.extend_from_slice(file_id);
        }
        index_bytes.extend_from_slice(&build_packet(set_id, TYPE_MAIN, &main_body));

        for (filename, file_id, data) in files {
            std::fs::write(dir.join(filename), data).unwrap();

            let mut desc_body = Vec::new();
            desc_body.extend_from_slice(file_id);
            desc_body.extend_from_slice(&file_md5(data));
            desc_body.extend_from_slice(&hash_16k(data));
            desc_body.extend_from_slice(&(data.len() as u64).to_le_bytes());
            desc_body.extend_from_slice(filename.as_bytes());
            desc_body.push(0);
            while desc_body.len() % 4 != 0 {
                desc_body.push(0);
            }
            index_bytes.extend_from_slice(&build_packet(set_id, TYPE_FILE_DESC, &desc_body));

            let block_count = data.len().div_ceil(slice_size);
            let mut ifsc_body = Vec::new();
            ifsc_body.extend_from_slice(file_id);
            for block_idx in 0..block_count {
                let block = padded_block(data, block_idx, slice_size);
                ifsc_body.extend_from_slice(&file_md5(&block));
                ifsc_body.extend_from_slice(&crc32_hash(&block).to_le_bytes());
            }
            index_bytes.extend_from_slice(&build_packet(set_id, TYPE_IFSC, &ifsc_body));
        }

        std::fs::write(dir.join("fixture.par2"), &index_bytes).unwrap();

        let ordered_files: Vec<_> = main_order
            .iter()
            .map(|wanted| {
                files
                    .iter()
                    .find(|(_, file_id, _)| file_id == wanted)
                    .unwrap()
            })
            .collect();
        let input_blocks: Vec<Vec<u8>> = ordered_files
            .iter()
            .flat_map(|(_, _, data)| {
                let count = data.len().div_ceil(slice_size);
                (0..count).map(move |block_idx| padded_block(data, block_idx, slice_size))
            })
            .collect();

        let input_count = input_blocks.len();
        let enc = GfMatrix::par2_encoding_matrix(input_count, recovery_exponents);
        let srcs: Vec<&[u8]> = input_blocks.iter().map(Vec::as_slice).collect();
        let mut vol_bytes = index_bytes.clone();

        for (row_idx, &exp) in recovery_exponents.iter().enumerate() {
            let coeffs: Vec<u16> = (0..input_count)
                .map(|col| enc.get(input_count + row_idx, col))
                .collect();
            let mut recovery = vec![0u8; slice_size];
            gf_simd::mul_add_multi(&mut recovery, &srcs, &coeffs);

            let mut body = Vec::with_capacity(4 + slice_size);
            body.extend_from_slice(&exp.to_le_bytes());
            body.extend_from_slice(&recovery);
            vol_bytes.extend_from_slice(&build_packet(set_id, TYPE_RECOVERY, &body));
        }

        std::fs::write(dir.join("fixture.vol00+2.par2"), &vol_bytes).unwrap();
        dir.join("fixture.par2")
    }

    /// Test that the basic RS encode→decode round-trip works with the D×D approach.
    /// 2 data blocks, 2 recovery blocks, lose both data blocks, recover.
    #[test]
    fn test_rs_roundtrip_simple() {
        // 2 input "blocks" of 4 bytes each (2 u16 values)
        let input0: Vec<u8> = vec![0x01, 0x00, 0x02, 0x00]; // [1, 2] as u16 LE
        let input1: Vec<u8> = vec![0x03, 0x00, 0x04, 0x00]; // [3, 4] as u16 LE

        let input_count = 2;
        let recovery_exponents = vec![0u32, 1u32];

        // Build encoding matrix to compute recovery blocks
        let enc = GfMatrix::par2_encoding_matrix(input_count, &recovery_exponents);

        // Compute recovery blocks
        let slice_size = 4;
        let u16_per_slice = slice_size / 2;
        let inputs = [&input0, &input1];

        let mut recovery0 = vec![0u8; slice_size];
        let mut recovery1 = vec![0u8; slice_size];

        for pos in 0..u16_per_slice {
            let off = pos * 2;
            let mut r0: u16 = 0;
            let mut r1: u16 = 0;
            for (i, inp) in inputs.iter().enumerate() {
                let val = u16::from_le_bytes([inp[off], inp[off + 1]]);
                r0 = gf::add(r0, gf::mul(enc.get(2, i), val));
                r1 = gf::add(r1, gf::mul(enc.get(3, i), val));
            }
            recovery0[off] = r0 as u8;
            recovery0[off + 1] = (r0 >> 8) as u8;
            recovery1[off] = r1 as u8;
            recovery1[off + 1] = (r1 >> 8) as u8;
        }

        // Now "lose" both input blocks. Use D×D approach to recover.
        // D = 2 (both blocks damaged), N = 2.
        // Since all blocks are damaged, there are no intact blocks to subtract.
        // adjusted[e] = recovery[e] (no intact contributions to remove).

        let constants = par2_input_constants(input_count);
        let damaged_indices = [0usize, 1usize];

        // Build D×D Vandermonde: V[e][j] = c_{damaged_j}^{exp_e}
        let num_damaged = damaged_indices.len();
        let mut vandermonde = GfMatrix::zeros(num_damaged, num_damaged);
        for (e, &exp) in recovery_exponents.iter().enumerate() {
            for (j, &dmg_idx) in damaged_indices.iter().enumerate() {
                vandermonde.set(e, j, gf::pow(constants[dmg_idx], exp));
            }
        }

        let inv = vandermonde.invert().expect("Should be invertible");

        // Apply inverse: repaired[j] = Σ_e inv[j][e] * adjusted[e]
        let adjusted = [&recovery0[..], &recovery1[..]];
        let mut result0 = vec![0u8; slice_size];
        let mut result1 = vec![0u8; slice_size];

        for pos in 0..u16_per_slice {
            let off = pos * 2;
            let mut out0: u16 = 0;
            let mut out1: u16 = 0;
            for (e, adj) in adjusted.iter().enumerate() {
                let val = u16::from_le_bytes([adj[off], adj[off + 1]]);
                out0 = gf::add(out0, gf::mul(inv.get(0, e), val));
                out1 = gf::add(out1, gf::mul(inv.get(1, e), val));
            }
            result0[off] = out0 as u8;
            result0[off + 1] = (out0 >> 8) as u8;
            result1[off] = out1 as u8;
            result1[off + 1] = (out1 >> 8) as u8;
        }

        assert_eq!(result0, input0, "Recovered block 0 should match original");
        assert_eq!(result1, input1, "Recovered block 1 should match original");
    }

    /// Test D×D approach with partial damage (some blocks intact).
    /// 4 data blocks, lose 2, recover using 2 recovery blocks.
    #[test]
    fn test_rs_roundtrip_partial_damage() {
        let slice_size = 4;
        let input_count = 4;
        let recovery_exponents = vec![0u32, 1u32];

        let inputs: Vec<Vec<u8>> = vec![
            vec![0x01, 0x00, 0x02, 0x00],
            vec![0x03, 0x00, 0x04, 0x00],
            vec![0x05, 0x00, 0x06, 0x00],
            vec![0x07, 0x00, 0x08, 0x00],
        ];

        let enc = GfMatrix::par2_encoding_matrix(input_count, &recovery_exponents);

        // Compute recovery blocks
        let mut recovery = vec![vec![0u8; slice_size]; 2];
        for pos in 0..(slice_size / 2) {
            let off = pos * 2;
            for (e, rec) in recovery.iter_mut().enumerate() {
                let mut val: u16 = 0;
                for (i, inp) in inputs.iter().enumerate() {
                    let d = u16::from_le_bytes([inp[off], inp[off + 1]]);
                    val = gf::add(val, gf::mul(enc.get(input_count + e, i), d));
                }
                rec[off] = val as u8;
                rec[off + 1] = (val >> 8) as u8;
            }
        }

        // "Lose" blocks 1 and 3. Blocks 0 and 2 are intact.
        let damaged_indices = [1usize, 3usize];
        let intact_indices: Vec<usize> = (0..input_count)
            .filter(|i| !damaged_indices.contains(i))
            .collect();
        let num_damaged = damaged_indices.len();

        let constants = par2_input_constants(input_count);

        // Build D×D Vandermonde
        let mut vandermonde = GfMatrix::zeros(num_damaged, num_damaged);
        for (e, &exp) in recovery_exponents.iter().enumerate() {
            for (j, &dmg_idx) in damaged_indices.iter().enumerate() {
                vandermonde.set(e, j, gf::pow(constants[dmg_idx], exp));
            }
        }
        let inv = vandermonde.invert().expect("Should be invertible");

        // Compute adjusted recovery: subtract intact contributions
        let mut adjusted = recovery.clone();
        for &intact_idx in &intact_indices {
            let c_i = constants[intact_idx];
            for (e, adj) in adjusted.iter_mut().enumerate() {
                let coeff = gf::pow(c_i, recovery_exponents[e]);
                gf_simd::mul_add_buffer(adj, &inputs[intact_idx], coeff);
            }
        }

        // Apply inverse
        let adj_refs: Vec<&[u8]> = adjusted.iter().map(|v| v.as_slice()).collect();
        let mut outputs: Vec<Vec<u8>> = (0..num_damaged).map(|_| vec![0u8; slice_size]).collect();

        for (j, dst) in outputs.iter_mut().enumerate() {
            let coeffs: Vec<u16> = (0..num_damaged).map(|e| inv.get(j, e)).collect();
            gf_simd::mul_add_multi(dst, &adj_refs, &coeffs);
        }

        // Verify recovery
        assert_eq!(outputs[0], inputs[1], "Recovered block 1 should match");
        assert_eq!(outputs[1], inputs[3], "Recovered block 3 should match");
    }

    #[test]
    fn test_repair_cross_file_damage_respects_main_packet_file_order() {
        let dir = tempfile::tempdir().unwrap();
        let slice_size = 4096usize;
        let file_a_id = [0x20; 16];
        let file_b_id = [0x10; 16];
        let file_a = vec![0x41; slice_size * 2];
        let file_b = vec![0x42; slice_size * 2];

        let par2_path = write_par2_fixture(
            dir.path(),
            slice_size,
            &[
                ("file_a.bin", file_a_id, file_a.clone()),
                ("file_b.bin", file_b_id, file_b.clone()),
            ],
            &[file_a_id, file_b_id],
            &[0, 1],
        );

        let file_set = crate::parse(&par2_path).unwrap();
        assert_eq!(file_set.file_order, vec![file_a_id, file_b_id]);

        {
            let file_a_path = dir.path().join("file_a.bin");
            let file_b_path = dir.path().join("file_b.bin");
            let mut a = std::fs::OpenOptions::new()
                .write(true)
                .open(&file_a_path)
                .unwrap();
            let mut b = std::fs::OpenOptions::new()
                .write(true)
                .open(&file_b_path)
                .unwrap();
            a.seek(SeekFrom::Start(slice_size as u64)).unwrap();
            b.seek(SeekFrom::Start(slice_size as u64)).unwrap();
            a.write_all(&vec![0xDE; slice_size]).unwrap();
            b.write_all(&vec![0xAD; slice_size]).unwrap();
        }

        let pre = verify::verify(&file_set, dir.path());
        assert_eq!(pre.damaged.len(), 2);
        assert_eq!(pre.blocks_needed(), 2);
        assert!(pre.repair_possible);

        let repaired = repair(&file_set, dir.path()).unwrap();
        assert!(repaired.success);

        assert_eq!(
            std::fs::read(dir.path().join("file_a.bin")).unwrap(),
            file_a
        );
        assert_eq!(
            std::fs::read(dir.path().join("file_b.bin")).unwrap(),
            file_b
        );
    }
}
