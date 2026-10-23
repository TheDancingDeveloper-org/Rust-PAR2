//! In-memory slice repair (`Repair` builder) tests.
//!
//! The `multi` fixture was made by par2cmdline 1.0.0:
//! `par2 c -s4096 -c8 -n1 multi.par2 alpha.bin beta.bin gamma.bin`
//! — 12 input slices of 4096 bytes (alpha and beta end in short slices) and 8
//! recovery slices.

use std::collections::HashMap;
use std::path::Path;

use rust_par2::matrix::GfMatrix;
use rust_par2::recovery::{RecoveryBlock, load_recovery_blocks};
use rust_par2::{BlockMap, Par2FileSet, Repair, RepairError, SliceHasher};

fn multi_dir() -> &'static Path {
    Path::new("tests/fixtures/multi")
}

/// Small deterministic PRNG (splitmix64) so failures reproduce.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// `k` distinct values from `0..n`, sorted.
    fn choose(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut all: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + self.below(n - i);
            all.swap(i, j);
        }
        let mut out = all[..k].to_vec();
        out.sort_unstable();
        out
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }
}

struct Fixture {
    file_set: Par2FileSet,
    map: BlockMap,
    recovery: Vec<RecoveryBlock>,
    /// Every input slice as stored in its file (last slice of a file is short).
    slices: Vec<Vec<u8>>,
}

fn load_multi() -> Fixture {
    let dir = multi_dir();
    let file_set = rust_par2::parse(&dir.join("multi.par2")).unwrap();
    let map = BlockMap::new(&file_set);
    let recovery = load_recovery_blocks(dir, &file_set.recovery_set_id, file_set.slice_size);
    let slices = (0..map.total_blocks as usize)
        .map(|g| {
            let loc = map.global_to_file(g).unwrap();
            let data = std::fs::read(dir.join(&loc.file.filename)).unwrap();
            data[loc.offset as usize..loc.offset as usize + loc.len].to_vec()
        })
        .collect();
    Fixture {
        file_set,
        map,
        recovery,
        slices,
    }
}

fn padded(data: &[u8], slice_size: usize) -> Vec<u8> {
    let mut v = data.to_vec();
    v.resize(slice_size, 0);
    v
}

/// Feed every slice not in `missing`, alternating single feeds and batches.
fn feed_present(r: &mut Repair, slices: &[Vec<u8>], missing: &[usize], rng: &mut Rng) {
    let present: Vec<usize> = (0..slices.len()).filter(|i| !missing.contains(i)).collect();
    let mut rest = present.as_slice();
    while !rest.is_empty() {
        let take = 1 + rng.below(rest.len().min(5));
        let (now, later) = rest.split_at(take);
        if take == 1 {
            r.feed(now[0], &slices[now[0]]).unwrap();
        } else {
            let batch: Vec<(usize, &[u8])> =
                now.iter().map(|&i| (i, slices[i].as_slice())).collect();
            r.feed_batch(&batch).unwrap();
        }
        rest = later;
    }
}

#[test]
fn fixture_layout() {
    let fx = load_multi();
    assert_eq!(fx.file_set.slice_size, 4096);
    assert_eq!(fx.map.total_blocks, 12);
    assert_eq!(fx.recovery.len(), 8);
    let short: Vec<usize> = (0..12)
        .filter(|&g| fx.map.global_to_file(g).unwrap().len < 4096)
        .collect();
    assert!(!short.is_empty(), "fixture should have short last slices");
    assert!(fx.map.global_to_file(12).is_none());
}

#[test]
fn slices_match_ifsc_checksums() {
    let fx = load_multi();
    for (g, data) in fx.slices.iter().enumerate() {
        let loc = fx.map.global_to_file(g).unwrap();
        let file = &fx.file_set.files[&loc.file.file_id];
        assert!(file.slice_matches(loc.local_index as usize, data, 4096));
        assert_eq!(
            SliceHasher::checksum(data, 4096),
            file.slices[loc.local_index as usize]
        );
        let mut bad = data.clone();
        bad[0] ^= 1;
        assert!(!file.slice_matches(loc.local_index as usize, &bad, 4096));
    }
}

/// Every damage count from 1 to the recovery count, several sets each, solved
/// from par2cmdline's recovery data.
#[test]
fn par2cmdline_fixture_round_trip() {
    let fx = load_multi();
    let n = fx.slices.len();
    let mut rng = Rng(153);
    for d in 1..=fx.recovery.len() {
        for _ in 0..6 {
            let missing = rng.choose(n, d);
            let mut r = Repair::new(&fx.recovery, 4096, n).unwrap();
            assert_eq!(r.rows(), 8);
            feed_present(&mut r, &fx.slices, &missing, &mut rng);
            let out = r.solve(&missing).unwrap();
            assert_eq!(out.len(), d);
            for &g in &missing {
                let loc = fx.map.global_to_file(g).unwrap();
                assert_eq!(
                    out[&g],
                    padded(&fx.slices[g], 4096),
                    "slice {g} with missing {missing:?}"
                );
                let file = &fx.file_set.files[&loc.file.file_id];
                assert!(file.slice_matches(loc.local_index as usize, &out[&g][..loc.len], 4096));
            }
        }
    }
}

#[test]
fn par2cmdline_fixture_too_many_missing() {
    let fx = load_multi();
    let n = fx.slices.len();
    let missing: Vec<usize> = (0..9).collect();
    let mut r = Repair::new(&fx.recovery, 4096, n).unwrap();
    feed_present(&mut r, &fx.slices, &missing, &mut Rng(1));
    match r.solve(&missing) {
        Err(RepairError::InsufficientRecovery {
            needed: 9,
            available: 8,
        }) => {}
        other => panic!("expected InsufficientRecovery, got {other:?}"),
    }
}

#[test]
fn truncated_rows_still_solve_and_bound_errors() {
    let fx = load_multi();
    let n = fx.slices.len();
    let missing = vec![2, 5, 8];
    let mut r = Repair::new_owning(fx.recovery.clone(), 4096, n).unwrap();
    r.truncate_rows(3);
    assert_eq!(r.rows(), 3);
    feed_present(&mut r, &fx.slices, &missing, &mut Rng(2));
    let out = r.solve(&missing).unwrap();
    for g in missing {
        assert_eq!(out[&g], padded(&fx.slices[g], 4096));
    }

    let mut r = Repair::new(&fx.recovery, 4096, n).unwrap();
    r.truncate_rows(2);
    let missing = vec![0, 1, 2];
    feed_present(&mut r, &fx.slices, &missing, &mut Rng(3));
    assert!(matches!(
        r.solve(&missing),
        Err(RepairError::InsufficientRecovery {
            needed: 3,
            available: 2
        })
    ));
}

/// Damage files on disk, repair them with the file-based path, and check the
/// builder gives the same bytes for the same damage.
#[test]
fn file_repair_and_builder_agree() {
    let fx = load_multi();
    let tmp = tempfile::tempdir().unwrap();
    for e in std::fs::read_dir(multi_dir()).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
    }

    // Corrupt one slice in alpha and the short last slice of beta, and delete
    // gamma entirely.
    let damage = |name: &str, local: u32| {
        let g = fx
            .map
            .global_to_file(fx.map.global_index(name, local).unwrap())
            .unwrap();
        let path = tmp.path().join(name);
        let mut data = std::fs::read(&path).unwrap();
        data[g.offset as usize] ^= 0xFF;
        std::fs::write(&path, data).unwrap();
    };
    damage("alpha.bin", 2);
    let beta_last = fx.map.file("beta.bin").unwrap().block_count - 1;
    damage("beta.bin", beta_last);
    std::fs::remove_file(tmp.path().join("gamma.bin")).unwrap();

    let gamma = fx.map.file("gamma.bin").unwrap();
    let mut missing = vec![
        fx.map.global_index("alpha.bin", 2).unwrap(),
        fx.map.global_index("beta.bin", beta_last).unwrap(),
    ];
    missing.extend((0..gamma.block_count).map(|i| (gamma.start_block + i) as usize));

    let pre = rust_par2::verify(&fx.file_set, tmp.path());
    assert_eq!(pre.blocks_needed() as usize, missing.len());
    let result = rust_par2::repair_from_verify(&fx.file_set, tmp.path(), &pre).unwrap();
    assert!(result.success);
    assert_eq!(result.blocks_repaired as usize, missing.len());

    let mut r = Repair::new(&fx.recovery, 4096, fx.slices.len()).unwrap();
    feed_present(&mut r, &fx.slices, &missing, &mut Rng(4));
    let out = r.solve(&missing).unwrap();

    let mut on_disk: HashMap<String, Vec<u8>> = HashMap::new();
    for &g in &missing {
        let loc = fx.map.global_to_file(g).unwrap();
        let file = on_disk
            .entry(loc.file.filename.clone())
            .or_insert_with(|| std::fs::read(tmp.path().join(&loc.file.filename)).unwrap());
        let start = loc.offset as usize;
        assert_eq!(&file[start..start + loc.len], &out[&g][..loc.len]);
        assert!(out[&g][loc.len..].iter().all(|&b| b == 0));
    }
    for name in ["alpha.bin", "beta.bin", "gamma.bin"] {
        assert_eq!(
            std::fs::read(tmp.path().join(name)).unwrap(),
            std::fs::read(multi_dir().join(name)).unwrap()
        );
    }
}

/// Same equivalence on the original single-file `damaged` fixture.
#[test]
fn file_repair_and_builder_agree_on_damaged_fixture() {
    let src = Path::new("tests/fixtures/damaged");
    let tmp = tempfile::tempdir().unwrap();
    for e in std::fs::read_dir(src).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
    }
    let file_set = rust_par2::parse(&tmp.path().join("testdata.bin.par2")).unwrap();
    let map = BlockMap::new(&file_set);
    let slice_size = file_set.slice_size as usize;
    let pre = rust_par2::verify(&file_set, tmp.path());
    let damaged = &pre.damaged[0];
    let missing: Vec<usize> = damaged
        .damaged_block_indices
        .iter()
        .map(|&l| map.global_index(&damaged.filename, l).unwrap())
        .collect();

    let before = std::fs::read(tmp.path().join("testdata.bin")).unwrap();
    let recovery = load_recovery_blocks(tmp.path(), &file_set.recovery_set_id, file_set.slice_size);
    let mut r = Repair::new_owning(recovery, slice_size, map.total_blocks as usize).unwrap();
    for g in 0..map.total_blocks as usize {
        if !missing.contains(&g) {
            let loc = map.global_to_file(g).unwrap();
            r.feed(
                g,
                &before[loc.offset as usize..loc.offset as usize + loc.len],
            )
            .unwrap();
        }
    }
    let out = r.solve(&missing).unwrap();

    rust_par2::repair_from_verify(&file_set, tmp.path(), &pre).unwrap();
    let after = std::fs::read(tmp.path().join("testdata.bin")).unwrap();
    for g in missing {
        let loc = map.global_to_file(g).unwrap();
        let start = loc.offset as usize;
        assert_eq!(&after[start..start + loc.len], &out[&g][..loc.len]);
    }
}

/// Encode random file layouts with random recovery exponents, damage up to the
/// recovery count, and check the builder reproduces the original bytes.
#[test]
fn property_random_layouts_and_damage() {
    let mut rng = Rng(0x5EED);
    for case in 0..60 {
        let slice_size = 4 * (1 + rng.below(40)); // 4..=160, exercises SIMD tails
        let file_count = 1 + rng.below(4);
        let mut slices: Vec<Vec<u8>> = Vec::new();
        for _ in 0..file_count {
            let len = 1 + rng.below(slice_size * 6);
            let data = rng.bytes(len);
            slices.extend(data.chunks(slice_size).map(<[u8]>::to_vec));
        }
        let n = slices.len();
        let k = 1 + rng.below(8);
        let mut exponents: Vec<u32> = rng.choose(40, k).into_iter().map(|e| e as u32).collect();
        exponents.sort_unstable();

        let enc = GfMatrix::par2_encoding_matrix(n, &exponents);
        let padded_slices: Vec<Vec<u8>> = slices.iter().map(|s| padded(s, slice_size)).collect();
        let srcs: Vec<&[u8]> = padded_slices.iter().map(Vec::as_slice).collect();
        let recovery: Vec<RecoveryBlock> = exponents
            .iter()
            .enumerate()
            .map(|(row, &exponent)| {
                let coeffs: Vec<u16> = (0..n).map(|col| enc.get(n + row, col)).collect();
                let mut data = vec![0u8; slice_size];
                rust_par2::gf_simd::mul_add_multi(&mut data, &srcs, &coeffs);
                RecoveryBlock { exponent, data }
            })
            .collect();

        let d = 1 + rng.below(k.min(n));
        let missing = rng.choose(n, d);
        let mut r = Repair::new_owning(recovery, slice_size, n).unwrap();
        feed_present(&mut r, &slices, &missing, &mut rng);
        let out = r.solve(&missing).unwrap_or_else(|e| {
            panic!("case {case}: n={n} k={k} exps={exponents:?} missing={missing:?}: {e}")
        });
        for &g in &missing {
            assert_eq!(out[&g], padded_slices[g], "case {case} slice {g}");
        }
    }
}

/// A corrupt slice fed once poisons the result and cannot be taken back:
/// re-feeding the good bytes is refused. Callers must verify before feeding.
#[test]
fn corrupt_feed_cannot_be_undone() {
    let fx = load_multi();
    let n = fx.slices.len();
    let missing = vec![3];
    let mut r = Repair::new(&fx.recovery, 4096, n).unwrap();

    let mut corrupt = fx.slices[0].clone();
    corrupt[100] ^= 0x55;
    r.feed(0, &corrupt).unwrap();
    assert!(matches!(
        r.feed(0, &fx.slices[0]),
        Err(RepairError::DuplicateSlice { index: 0 })
    ));
    for g in 1..n {
        if g != 3 {
            r.feed(g, &fx.slices[g]).unwrap();
        }
    }
    let out = r.solve(&missing).unwrap();
    assert_ne!(out[&3], padded(&fx.slices[3], 4096));

    // The slice hasher would have caught it.
    let loc = fx.map.global_to_file(0).unwrap();
    let file = &fx.file_set.files[&loc.file.file_id];
    assert!(!file.slice_matches(0, &corrupt, 4096));
}

#[test]
fn input_validation() {
    let fx = load_multi();
    let n = fx.slices.len();
    let mut r = Repair::new(&fx.recovery, 4096, n).unwrap();

    assert!(matches!(
        r.feed(n, &fx.slices[0]),
        Err(RepairError::SliceOutOfRange { index, total }) if index == n && total == n
    ));
    assert!(matches!(
        r.feed(0, &[0u8; 4097]),
        Err(RepairError::SliceTooLong { len: 4097, .. })
    ));

    // A bad batch feeds nothing, including the valid entries before the error.
    let bad_batch: Vec<(usize, &[u8])> =
        vec![(0, &fx.slices[0]), (1, &fx.slices[1]), (1, &fx.slices[1])];
    assert!(matches!(
        r.feed_batch(&bad_batch),
        Err(RepairError::DuplicateSlice { index: 1 })
    ));
    assert_eq!(r.fed_count(), 0);
    assert!(!r.is_fed(0));

    // Slices neither fed nor missing.
    r.feed(0, &fx.slices[0]).unwrap();
    assert!(matches!(
        r.solve(&[1]),
        Err(RepairError::IncompleteFeed { unaccounted }) if unaccounted == n - 2
    ));
    // A fed slice cannot also be missing.
    assert!(matches!(
        r.solve(&[0]),
        Err(RepairError::DuplicateSlice { index: 0 })
    ));

    assert!(matches!(
        Repair::new(&fx.recovery, 4098, n),
        Err(RepairError::InvalidSliceSize(4098))
    ));
    assert!(matches!(
        Repair::new(&fx.recovery, 8192, n),
        Err(RepairError::InvalidRecoveryBlock { len: 4096, .. })
    ));
}

#[test]
fn duplicate_recovery_exponents_are_dropped() {
    let fx = load_multi();
    let mut doubled = fx.recovery.clone();
    doubled.extend(fx.recovery.iter().cloned());
    let r = Repair::new_owning(doubled, 4096, fx.slices.len()).unwrap();
    assert_eq!(r.rows(), 8);
    assert!(r.exponents().windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn nothing_missing_solves_to_empty() {
    let fx = load_multi();
    let mut r = Repair::new(&fx.recovery, 4096, fx.slices.len()).unwrap();
    feed_present(&mut r, &fx.slices, &[], &mut Rng(5));
    assert!(r.solve(&[]).unwrap().is_empty());
}
