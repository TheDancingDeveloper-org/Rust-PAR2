//! Streaming recovery reader (`RecoveryReader`) tests against the par2cmdline
//! `multi` fixture (12 input slices of 4096 bytes, 8 recovery slices in
//! `multi.vol0+8.par2`).

use std::path::Path;

use rust_par2::recovery::{RecoveryBudget, RecoveryReader, VolumeEnd, load_recovery_blocks};
use rust_par2::{BlockMap, Repair};

const SLICE: usize = 4096;

fn dir() -> &'static Path {
    Path::new("tests/fixtures/multi")
}

fn set() -> rust_par2::Par2FileSet {
    rust_par2::parse(&dir().join("multi.par2")).unwrap()
}

fn volume() -> Vec<u8> {
    std::fs::read(dir().join("multi.vol0+8.par2")).unwrap()
}

/// Split a volume into whole packets by their length fields.
fn packets(vol: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + 64 <= vol.len() {
        let len = u64::from_le_bytes(vol[pos + 8..pos + 16].try_into().unwrap()) as usize;
        out.push(&vol[pos..pos + len]);
        pos += len;
    }
    assert_eq!(pos, vol.len());
    out
}

fn exponent_of(packet: &[u8]) -> Option<u32> {
    (&packet[48..64] == b"PAR 2.0\x00RecvSlic")
        .then(|| u32::from_le_bytes(packet[64..68].try_into().unwrap()))
}

fn input_slices(fs: &rust_par2::Par2FileSet) -> Vec<Vec<u8>> {
    let map = BlockMap::new(fs);
    (0..map.total_blocks as usize)
        .map(|g| {
            let loc = map.global_to_file(g).unwrap();
            let data = std::fs::read(dir().join(&loc.file.filename)).unwrap();
            data[loc.offset as usize..loc.offset as usize + loc.len].to_vec()
        })
        .collect()
}

fn padded(data: &[u8]) -> Vec<u8> {
    let mut v = data.to_vec();
    v.resize(SLICE, 0);
    v
}

/// Solve `missing` from `rows` and check against the real slices.
fn assert_repairs(rows: Vec<rust_par2::recovery::RecoveryBlock>, missing: &[usize]) {
    let fs = set();
    let slices = input_slices(&fs);
    let mut r = Repair::new_owning(rows, SLICE, slices.len()).unwrap();
    for (i, s) in slices.iter().enumerate() {
        if !missing.contains(&i) {
            r.feed(i, s).unwrap();
        }
    }
    let solved = r.solve(missing).unwrap();
    for &m in missing {
        assert_eq!(solved[&m], padded(&slices[m]), "slice {m}");
    }
}

#[test]
fn reader_matches_directory_loader() {
    let fs = set();
    let from_dir = load_recovery_blocks(dir(), &fs.recovery_set_id, fs.slice_size);

    let mut reader = RecoveryReader::new(fs.recovery_set_id, fs.slice_size);
    let index = std::fs::read(dir().join("multi.par2")).unwrap();
    assert!(reader.read_volume(&index[..]).read_to_end());
    let scan = reader.read_volume(std::fs::File::open(dir().join("multi.vol0+8.par2")).unwrap());
    assert!(scan.read_to_end());
    assert_eq!((scan.seen, scan.kept, scan.corrupt), (8, 8, 0));
    assert_eq!(scan.bytes_read, volume().len() as u64);

    let mut streamed = reader.into_blocks();
    streamed.sort_by_key(|b| b.exponent);
    assert_eq!(streamed.len(), from_dir.len());
    for (a, b) in streamed.iter().zip(&from_dir) {
        assert_eq!(a.exponent, b.exponent);
        assert_eq!(a.data, b.data);
    }
}

/// Recovery packets reordered so exponents arrive out of order; a budget of
/// three keeps the three lowest, in packet order, and they repair three
/// missing slices.
#[test]
fn out_of_order_exponents_under_budget() {
    let fs = set();
    let vol = volume();
    let mut pkts = packets(&vol);
    pkts.sort_by_key(|p| exponent_of(p).map(|e| (e * 5 + 3) % 8));
    let order: Vec<u32> = pkts.iter().filter_map(|p| exponent_of(p)).collect();
    assert_eq!(order, [1, 6, 3, 0, 5, 2, 7, 4]);
    let shuffled: Vec<u8> = pkts.concat();

    let mut reader =
        RecoveryReader::new(fs.recovery_set_id, fs.slice_size).with_budget(RecoveryBudget::rows(3));
    let scan = reader.read_volume(&shuffled[..]);
    assert!(scan.read_to_end());
    assert_eq!(scan.seen, 8);
    assert_eq!(reader.kept(), 3);
    let kept: Vec<u32> = reader.blocks().iter().map(|b| b.exponent).collect();
    assert_eq!(kept, [1, 0, 2]);

    assert_repairs(reader.into_blocks(), &[0, 5, 11]);
}

#[test]
fn byte_budget_bounds_rows() {
    let fs = set();
    let mut reader = RecoveryReader::new(fs.recovery_set_id, fs.slice_size)
        .with_budget(RecoveryBudget::bytes(2 * SLICE as u64 + 100));
    assert!(reader.read_volume(&volume()[..]).read_to_end());
    let kept: Vec<u32> = reader.blocks().iter().map(|b| b.exponent).collect();
    assert_eq!(kept, [0, 1]);
}

/// A volume cut short reports `Truncated`, not a clean end, and still yields
/// the rows before the cut.
#[test]
fn truncated_volume() {
    let fs = set();
    let vol = volume();
    let pkts = packets(&vol);
    let recv_ends: Vec<usize> = pkts
        .iter()
        .scan(0, |end, p| {
            *end += p.len();
            Some((*end, exponent_of(p).is_some()))
        })
        .filter_map(|(end, is_recv)| is_recv.then_some(end))
        .collect();
    // Cut half way through the fourth recovery packet.
    let cut = recv_ends[3] - SLICE / 2;

    let mut reader = RecoveryReader::new(fs.recovery_set_id, fs.slice_size);
    let scan = reader.read_volume(&vol[..cut]);
    assert!(!scan.read_to_end());
    assert!(matches!(scan.end, VolumeEnd::Truncated), "{}", scan.end);
    assert_eq!(scan.seen, 3);
    assert_eq!(scan.bytes_read, cut as u64);

    assert_repairs(reader.into_blocks(), &[2, 3, 9]);
}

#[test]
fn per_volume_cap() {
    let fs = set();
    let vol = volume();
    let mut reader =
        RecoveryReader::new(fs.recovery_set_id, fs.slice_size).with_max_volume_bytes(10_000);
    let scan = reader.read_volume(&vol[..]);
    assert!(matches!(scan.end, VolumeEnd::CapReached), "{}", scan.end);
    assert_eq!(scan.bytes_read, 10_000);
    assert!(scan.seen < 8);
}
