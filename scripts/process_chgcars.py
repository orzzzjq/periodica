#!/usr/bin/env python
"""Topological descriptors of Materials Project charge densities.

For each material the script reads the float32 density grid
<data-dir>/<mpid>.npy (shape = VASP FFT grid, index [i, j, k] at fractional
coordinates (i/N1, j/N2, k/N3)) and its lattice from <data-dir>/manifest.jsonl,
builds the periodic cubical (grid) quotient complex and its periodic merge
tree, and writes

    <out-dir>/barcodes/<mpid>.npz   Periodica barcode file (core.write_barcodes;
                                    load with Periodica.load_barcodes)
    <out-dir>/curves/<mpid>.npz     the d + 1 filtration curves: the exact step
                                    curves (times_k / values_k) and their uniform
                                    samples 'vectors' ((d + 1) x resolution,
                                    float32) on the grid 'x'

By default the SUPERLEVEL sets of the density are analyzed (the filtration
value of a grid point is -rho, so charge maxima at the atoms are born first);
--sublevel switches to sublevel sets of rho. Filtration values, barcode births
/ deaths and the curve x-axis are all in the same (possibly negated) e/A^3.

The curve sampling interval defaults to the material's own barcode range
(the plot_curves x-range); pass --interval LO HI (in filtration units) to
sample every material on the same grid so that the vectors are comparable.

Examples
    scripts/process_chgcars.py mp-1
    scripts/process_chgcars.py mp-1 mp-10 --out-dir /tmp/topo
    scripts/process_chgcars.py --all --workers 8 --skip-existing
"""
import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from periodica.core import Periodica, barcode_range, curves_to_vectors  # noqa: E402

DEFAULT_DATA_DIR = Path('~/data/mpchg/chg_f32').expanduser()
DEFAULT_OUT_DIR = Path('~/data/mpchg/periodica').expanduser()
CURVE_FILE_FORMAT = 'periodica-curves-1'


def load_manifest(path):
    """mpid -> manifest record (lattice, shape, task, ...)."""
    records = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records[rec['mpid']] = rec
    return records


def lattice_matrix(rec):
    """Periodica lattice U (vectors as COLUMNS) from the pymatgen Structure
    dict, whose lattice matrix has the vectors as ROWS."""
    return np.array(rec['structure']['lattice']['matrix'], dtype=float).T


def composition(rec):
    counts = {}
    for site in rec['structure']['sites']:
        for sp in site['species']:
            counts[sp['element']] = counts.get(sp['element'], 0) + sp.get('occu', 1)
    return counts


def write_curves(file, d, curves, interval, resolution, meta=None):
    """Compressed .npz with the exact step curves and their uniform samples."""
    x = np.linspace(interval[0], interval[1], resolution)
    vectors = curves_to_vectors(curves, interval, resolution)
    arrays = {
        'format': np.array(CURVE_FILE_FORMAT),
        'd': np.array(d, dtype=np.int8),
        'interval': np.array(interval, dtype=np.float64),
        'x': x,
        'vectors': vectors,
        'meta': np.array(json.dumps({} if meta is None else dict(meta))),
    }
    for k, (times, values) in enumerate(curves):
        arrays[f'times_{k}'] = np.asarray(times, dtype=np.float64)
        arrays[f'values_{k}'] = np.asarray(values, dtype=np.float64)
    np.savez_compressed(file, **arrays)


def process(mpid, rec, data_dir, out_dir, sublevel, interval, resolution):
    """Full pipeline for one material; returns a small summary dict."""
    t0 = time.time()
    grid_file = Path(data_dir) / f'{mpid}.npy'
    values = np.load(grid_file)
    U = lattice_matrix(rec)
    if tuple(values.shape) != tuple(rec['shape']):
        raise ValueError(f'{mpid}: grid shape {values.shape} != manifest shape {rec["shape"]}')

    p = Periodica()
    p.periodic_grid(U, values, sublevel=sublevel, build_arcs=False)
    p.merge_tree()
    bcodes = p.barcodes(drop_zero_length=True)
    curves = p.curves()
    iv = tuple(interval) if interval is not None else tuple(barcode_range(bcodes))

    meta = {
        'mpid': mpid,
        'task': rec.get('task'),
        'source': str(grid_file),
        'shape': list(values.shape),
        'lattice_columns': U.tolist(),
        'volume': rec['structure']['lattice'].get('volume'),
        'composition': composition(rec),
        'units': 'e/A^3',
        'filtration': 'sublevel' if sublevel else 'superlevel',
        'filtration_sign': 1 if sublevel else -1,  # filtration value = sign * rho
        'rho_min': float(values.min()),
        'rho_max': float(values.max()),
    }
    bc_file = Path(out_dir) / 'barcodes' / f'{mpid}.npz'
    cv_file = Path(out_dir) / 'curves' / f'{mpid}.npz'
    p.save_barcodes(bc_file, meta=meta)
    write_curves(cv_file, p.d, curves, iv, resolution,
                 meta={**meta, 'resolution': resolution,
                       'interval_source': 'argument' if interval is not None else 'barcode_range'})
    return {
        'mpid': mpid,
        'shape': list(values.shape),
        'bars': [len(bars) for bars in bcodes],
        'interval': iv,
        'seconds': time.time() - t0,
    }


def _worker(args):
    mpid, rec, data_dir, out_dir, sublevel, interval, resolution = args
    try:
        return process(mpid, rec, data_dir, out_dir, sublevel, interval, resolution)
    except Exception:
        return {'mpid': mpid, 'error': traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('materials', nargs='*', help='mpids (mp-1) or paths to <mpid>.npy files')
    ap.add_argument('--all', action='store_true', help='process every material in the manifest')
    ap.add_argument('--limit', type=int, default=None, help='stop after this many materials')
    ap.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument('--sublevel', action='store_true', help='sublevel sets of rho (default: superlevel)')
    ap.add_argument('--interval', type=float, nargs=2, metavar=('LO', 'HI'), default=None,
                    help='curve sampling interval in filtration units (default: per-material barcode range)')
    ap.add_argument('--resolution', type=int, default=100, help='samples per curve (default 100)')
    ap.add_argument('--workers', type=int, default=1, help='parallel processes (each needs a few GB)')
    ap.add_argument('--skip-existing', action='store_true', help='skip materials whose two outputs exist')
    args = ap.parse_args()

    manifest = load_manifest(args.data_dir / 'manifest.jsonl')
    if args.all:
        mpids = list(manifest)
    else:
        mpids = [Path(m).stem if m.endswith('.npy') else m for m in args.materials]
    if not mpids:
        ap.error('give mpids / .npy paths or --all')
    missing = [m for m in mpids if m not in manifest]
    if missing:
        ap.error(f'not in manifest: {missing[:10]}{" ..." if len(missing) > 10 else ""}')
    if args.skip_existing:
        mpids = [m for m in mpids
                 if not ((args.out_dir / 'barcodes' / f'{m}.npz').exists()
                         and (args.out_dir / 'curves' / f'{m}.npz').exists())]
    if args.limit is not None:
        mpids = mpids[:args.limit]
    (args.out_dir / 'barcodes').mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'curves').mkdir(parents=True, exist_ok=True)
    sublevel = args.sublevel
    interval = tuple(args.interval) if args.interval is not None else None

    jobs = [(m, manifest[m], args.data_dir, args.out_dir, sublevel, interval, args.resolution) for m in mpids]
    print(f'{len(jobs)} materials -> {args.out_dir}  ({"sublevel" if sublevel else "superlevel"}, '
          f'resolution {args.resolution}, workers {args.workers})', flush=True)
    t0 = time.time()
    n_ok = n_err = 0
    failures = open(args.out_dir / 'failures.jsonl', 'a')

    def report(i, r):
        nonlocal n_ok, n_err
        if 'error' in r:
            n_err += 1
            failures.write(json.dumps({'mpid': r['mpid'], 'error': r['error']}) + '\n')
            failures.flush()
            print(f'[{i}/{len(jobs)}] {r["mpid"]} FAILED: {r["error"].strip().splitlines()[-1]}', flush=True)
        else:
            n_ok += 1
            print(f'[{i}/{len(jobs)}] {r["mpid"]} shape {r["shape"]} bars {r["bars"]} '
                  f'interval ({r["interval"][0]:.4g}, {r["interval"][1]:.4g}) {r["seconds"]:.1f}s', flush=True)

    if args.workers > 1:
        with Pool(args.workers, maxtasksperchild=50) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, jobs), 1):
                report(i, r)
    else:
        for i, job in enumerate(jobs, 1):
            report(i, _worker(job))
    failures.close()
    print(f'done: {n_ok} ok, {n_err} failed, {time.time() - t0:.0f}s', flush=True)
    return 1 if n_err else 0


if __name__ == '__main__':
    sys.exit(main())
