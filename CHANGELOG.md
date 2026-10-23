# Changelog

## 0.2.0

### Added

- `RecoveryReader` (`recovery` module): streams RecoverySlice packets from any
  `Read` source (file, network body, `&[u8]`) for one recovery set. Rows are
  accumulated across volumes under a row or byte budget (`RecoveryBudget`) that
  keeps the lowest exponents; packets outside the budget stream past without
  being buffered, and reads per volume are capped (512 MiB by default). Each
  volume reports seen/kept/corrupt counts and how it ended (`Complete`,
  `Truncated`, `CapReached`, `Io`), so callers never treat a set as
  unrepairable from a volume they failed to finish. `load_recovery_blocks` is
  now a thin wrapper over the reader.
- Slice-level in-memory repair: the `Repair` builder loads recovery rows, takes
  present slices with `feed`/`feed_batch` (short last slices are zero-padded
  without copying), and `solve`s for the missing ones with no file I/O.
  `truncate_rows` drops rows the solve will not use. Errors cover insufficient
  recovery, singular matrices, and out-of-range, too-long, duplicate or
  unaccounted slices.
- New `slice` module: `SliceHasher` (IFSC MD5 + CRC32, zero-padded) and
  `Par2File::slice_matches` for checking a slice before feeding it.
- `BlockMap`, `BlockFile`, `SliceLocation` and `build_block_map` are now public.

### Changed

- **Behaviour change:** RecoverySlice packets whose packet MD5 does not match
  are now dropped (and counted as corrupt) instead of being used for repair.
  A volume with damaged recovery packets yields fewer rows than before.
- `BlockMap::global_to_file` returns an `Option`.
- The file-based repair functions now read intact slices and feed them to
  `Repair`, so file and in-memory repair share one code path.
