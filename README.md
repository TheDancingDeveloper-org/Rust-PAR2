# rust-par2

[![Crates.io](https://img.shields.io/crates/v/rust-par2.svg)](https://crates.io/crates/rust-par2)
[![Docs.rs](https://docs.rs/rust-par2/badge.svg)](https://docs.rs/rust-par2)
[![CI](https://github.com/AusAgentSmith-org/Rust-PAR2/actions/workflows/ci.yml/badge.svg)](https://github.com/AusAgentSmith-org/Rust-PAR2/actions/workflows/ci.yml)

Pure Rust implementation of PAR2 (Parity Archive 2.0) verification and repair.

Parses PAR2 parity files, verifies data file integrity, and repairs damaged or
missing files using Reed-Solomon error correction -- all without any C
dependencies. Competitive with par2cmdline-turbo on repair performance, and
faster on verification.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rust-par2 = "0.1"
```

## Features

- **Full PAR2 verify and repair** -- detects damaged/missing files and
  reconstructs them from parity data
- **Pure Rust** -- no C/C++ dependencies, no FFI, no unsafe outside of SIMD
  intrinsics
- **SIMD-accelerated** -- AVX2 and SSSE3 Galois field multiplication with
  automatic runtime detection and scalar fallback
- **Parallel** -- rayon-based multithreading for verification (per-file) and
  repair (per-block)
- **Streaming I/O** -- double-buffered reader thread overlaps disk reads with
  GF compute during repair
- **Low memory** -- streams source blocks instead of loading entire files into
  memory; repair allocates only O(damaged_blocks * slice_size)
- **Structured logging** -- uses `tracing` for optional debug/trace output with
  zero overhead in release builds

## Quick start

```rust
use std::path::Path;

let par2_path = Path::new("/downloads/movie/movie.par2");
let dir = Path::new("/downloads/movie");

// Parse the PAR2 index file
let file_set = rust_par2::parse(par2_path).unwrap();

// Verify all files
let result = rust_par2::verify(&file_set, dir);

if result.all_correct() {
    println!("All files intact");
} else if result.repair_possible {
    // Repair damaged/missing files
    let repair = rust_par2::repair(&file_set, dir).unwrap();
    println!("{repair}");
}
```

### Skipping redundant verification

If you already have a `VerifyResult` (e.g., from displaying status to the user),
pass it directly to `repair_from_verify` to avoid re-verifying:

```rust
# use std::path::Path;
# let par2_path = Path::new("/downloads/movie/movie.par2");
# let dir = Path::new("/downloads/movie");
# let file_set = rust_par2::parse(par2_path).unwrap();
let result = rust_par2::verify(&file_set, dir);

if !result.all_correct() && result.repair_possible {
    // Skips the internal verify pass -- saves significant time on large files
    let repair = rust_par2::repair_from_verify(&file_set, dir, &result).unwrap();
    println!("{repair}");
}
```

### In-memory repair (no file I/O)

`Repair` rebuilds missing slices from slices you already hold in memory, such
as article bodies from a Usenet download, plus recovery blocks. It reads and
writes no files. Feed each present slice once, in any order, then solve:

```rust
use rust_par2::recovery::RecoveryBlock;
use rust_par2::{BlockMap, Repair};
# fn demo(file_set: &rust_par2::Par2FileSet, recovery: Vec<RecoveryBlock>,
#         present: Vec<(usize, Vec<u8>)>, mut missing: Vec<usize>) -> Result<(), rust_par2::RepairError> {
// Global slice indices run across files in the Main packet's file order.
let map = BlockMap::new(file_set);
let slice_size = file_set.slice_size as usize;

let mut repair = Repair::new_owning(recovery, slice_size, map.total_blocks as usize)?;

for (index, bytes) in &present {
    // Verify before feeding: a corrupt slice cannot be taken back out.
    let loc = map.global_to_file(*index).unwrap();
    let file = &file_set.files[&loc.file.file_id];
    if file.slice_matches(loc.local_index as usize, bytes, file_set.slice_size) {
        repair.feed(*index, bytes)?; // the last slice of a file may be short
    } else {
        missing.push(*index);
    }
}
repair.truncate_rows(missing.len()); // optional: free rows solve won't use

// Every slice must be either fed or listed as missing.
for (index, slice) in repair.solve(&missing)? {
    let loc = map.global_to_file(index).unwrap();
    // `slice` is zero-padded to slice_size; write only loc.len bytes at loc.offset.
    let _ = (&loc.file.filename, loc.offset, &slice[..loc.len]);
}
# Ok(()) }
```

`solve` fails with `RepairError::InsufficientRecovery` when more slices are
missing than recovery rows are loaded. Peak memory is `rows × slice_size` for
the accumulators plus `missing × slice_size` for the output, plus whatever you
buffer before feeding. `feed_batch` takes several slices at once and uses the
same cache-tiled SIMD path as the file-based repair, which is built on `Repair`.
`SliceHasher` computes the IFSC MD5 + CRC32 pair incrementally if a slice
arrives in pieces.

## Performance

Benchmarked against [par2cmdline-turbo](https://github.com/animetosho/par2cmdline-turbo)
on 1 GB random data with 3% damage, 8% redundancy, 768 KB block size
(16-thread AMD/Intel system):

| Operation         | rust-par2 | par2cmdline-turbo | Ratio            |
|-------------------|-----------|-------------------|------------------|
| Verify (intact)   | 1.4s      | 1.3s              | ~1.0x            |
| Verify (damaged)  | 2.8s      | 4.1s              | **1.5x faster**  |
| Repair            | 3.4s      | 3.7s              | **1.1x faster**  |

Repair time uses `repair_from_verify` (typical workflow where verify has already
been called). The standalone `repair()` function includes an internal verify pass
and takes ~6.5s.

## Design

### PAR2 format

PAR2 files are a container of typed packets. This crate parses five packet types:

| Packet          | Purpose                                          |
|-----------------|--------------------------------------------------|
| Main            | Slice size, file count, recovery block count     |
| FileDesc        | Per-file: filename, size, MD5, 16 KiB hash       |
| IFSC            | Per-block checksums (MD5 + CRC32) for each file  |
| RecoverySlice   | Reed-Solomon recovery data + exponent            |
| Creator         | Software identification string                   |

Packets can appear in any order and across multiple `.par2` files (an index file
plus `.volNNN+NNN.par2` volume files). All multi-byte fields are little-endian.
Packet bodies are verified with MD5 checksums.

### Galois field arithmetic

PAR2 mandates GF(2^16) with irreducible polynomial
`x^16 + x^12 + x^3 + x + 1` (0x1100B) and generator element 2.

Scalar operations use log/antilog tables (two 65536-entry u16 arrays,
initialized once via `OnceLock`) giving O(1) multiply, divide, and inverse.

For bulk operations (the repair inner loop), the crate uses the **PSHUFB
nibble-decomposition technique**:

1. Decompose each 16-bit GF element into four 4-bit nibbles
2. For each nibble position, precompute a 16-entry lookup table:
   `table[n] = constant * (n << shift)` in GF(2^16)
3. Use VPSHUFB/PSHUFB as a parallel 16-way table lookup
4. XOR the four partial products to get the final result

This gives 8 VPSHUFB lookups + 7 XORs per 32 bytes (16 GF elements) with AVX2,
or the same at 16 bytes with SSSE3. Runtime feature detection selects the best
available path, with a pure scalar fallback for non-x86 platforms.

The dispatch hierarchy:

| Path     | Width   | Throughput (single core)  |
|----------|---------|--------------------------|
| AVX2     | 256-bit | ~15 GB/s                 |
| SSSE3    | 128-bit | ~8 GB/s                  |
| Scalar   | 16-bit  | ~0.3 GB/s                |

### Reed-Solomon decoding

Repair works by constructing and inverting a Vandermonde matrix over GF(2^16):

1. **Encoding matrix construction**: Rows 0..N are the identity (data blocks
   pass through). Rows N..N+K encode recovery blocks using the PAR2 input
   constants `c[i] = 2^n` (where n skips multiples of 3, 5, 17, 257).

2. **Submatrix selection**: Select rows corresponding to available data: identity
   rows for intact blocks, recovery rows for damaged positions.

3. **Gaussian elimination**: Invert the submatrix over GF(2^16). If the matrix
   is singular (linearly dependent recovery blocks), repair fails.

4. **Block reconstruction**: Multiply the inverse matrix by the available data
   (intact blocks + recovery data) using SIMD multiply-accumulate.

### Repair architecture

The repair pipeline uses a source-major streaming design:

```
Reader thread                    Compute threads (rayon)
    |                                    |
    |  read block[0] ──sync_channel──>   |  dst[0..D] ^= coeff * block[0]
    |  read block[1] ──sync_channel──>   |  dst[0..D] ^= coeff * block[1]
    |  read block[2] ──sync_channel──>   |  dst[0..D] ^= coeff * block[2]
    |  ...                               |  ...
    v                                    v
```

- A dedicated reader thread reads source blocks sequentially, reusing file
  handles (one open per file, not per block)
- A `sync_channel(2)` provides double-buffering: the reader can be two blocks
  ahead of compute
- For each source block, rayon applies the GF multiply-accumulate to all damaged
  output buffers in parallel
- The source block (~768 KB) stays hot in shared L3 cache across rayon threads
- Each output buffer (~768 KB) fits in L2 cache per thread

This design reduces memory usage from O(total_blocks * slice_size) to
O(damaged_blocks * slice_size + 2 * slice_size), and overlaps disk I/O with
computation.

### Verification

Verification is parallelized per-file using rayon. For each file:

1. Compare file size against expected
2. Compute full-file MD5 using double-buffered reads (2 MB buffers)
3. If MD5 mismatches, identify specific damaged blocks via per-slice CRC32 + MD5

The per-slice identification allows repair to target only damaged blocks rather
than treating the entire file as lost.

## Scope and limitations

- **Verify and repair only** -- PAR2 file creation is not supported. Use
  par2cmdline or par2cmdline-turbo to create `.par2` files.
- **x86_64 optimized** -- SIMD acceleration requires AVX2 or SSSE3 (available
  on essentially all x86_64 CPUs from 2013+). Other architectures fall back to
  scalar arithmetic.
- **Blocking I/O** -- all operations are synchronous. For async contexts, wrap
  calls in `spawn_blocking`.
- **Single recovery set** -- the crate assumes one recovery set per directory,
  which is the standard PAR2 usage pattern.

## API reference

### Core functions

| Function             | Description                                              |
|----------------------|----------------------------------------------------------|
| `parse(path)`        | Parse a PAR2 index file, returns `Par2FileSet`           |
| `parse_par2_reader(reader, size)` | Parse from any `Read + Seek` source       |
| `verify(file_set, dir)` | Verify files on disk, returns `VerifyResult`         |
| `repair(file_set, dir)` | Verify + repair in one call                          |
| `repair_from_verify(file_set, dir, verify_result)` | Repair with pre-computed verify |
| `compute_hash_16k(path)` | MD5 of first 16 KiB (for file identification)      |
| `Repair::new_owning(recovery, slice_size, total)` | In-memory repair builder: `feed`, `feed_batch`, `solve` |
| `BlockMap::new(file_set)` | Map global slice indices to (file, offset, length) |
| `SliceHasher::checksum(data, slice_size)` | IFSC MD5 + CRC32 of a slice (zero-padded) |

### Key types

| Type             | Description                                                 |
|------------------|-------------------------------------------------------------|
| `Par2FileSet`    | Parsed PAR2 metadata: files, slice size, recovery set ID    |
| `Par2File`       | Per-file metadata: hash, size, filename, per-slice checksums|
| `VerifyResult`   | Verification outcome: intact, damaged, and missing files    |
| `RepairResult`   | Repair outcome: success flag, blocks/files repaired         |
| `DamagedFile`    | File with damage: includes specific damaged block indices   |
| `MissingFile`    | File that is entirely absent                                |

### Error types

| Error                  | Cause                                               |
|------------------------|-----------------------------------------------------|
| `ParseError`           | Invalid/corrupt PAR2 file                           |
| `RepairError::Io`      | Filesystem error during repair                      |
| `RepairError::InsufficientRecovery` | Not enough recovery blocks for repair  |
| `RepairError::SingularMatrix` | Recovery blocks are linearly dependent         |
| `RepairError::NoDamage` | Nothing to repair (all files intact)               |
| `RepairError::VerifyFailed` | Post-repair verification failed                |
| `RepairError::SliceOutOfRange` / `SliceTooLong` / `DuplicateSlice` / `IncompleteFeed` | Invalid input to the `Repair` builder |
| `RepairError::InvalidSliceSize` / `InvalidRecoveryBlock` | Bad `Repair` construction arguments |

## Dependencies

| Crate       | Purpose                          |
|-------------|----------------------------------|
| `md-5`      | MD5 hashing (ASM-accelerated)    |
| `crc32fast` | CRC32 checksums                  |
| `rayon`     | Data parallelism                 |
| `thiserror` | Error type derivation            |
| `tracing`   | Structured logging (zero-cost)   |

No system dependencies. No build scripts. No proc macros beyond `thiserror`.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)