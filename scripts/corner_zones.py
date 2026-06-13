"""
Corner zone detection module.
Builds a Car-Pos-Norm → zone label map from training LD files.
Zones: 0=straight, 1=eingang, 2=mitte, 3=apex
"""

import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from ldparser import ldData

N_GRID = 3000
SMOOTH_W = 80
APEX_PCTILE = 85
MIN_GAP_FRAC = 0.04
APPROACH_FRAC = 0.05
EXIT_FRAC = 0.025

ZONE_STRAIGHT = 0
ZONE_EINGANG = 1
ZONE_MITTE = 2
ZONE_APEX = 3
ZONE_NAMES = {0: 'straight', 1: 'eingang', 2: 'mitte', 3: 'apex'}


def _load_ld_first_lap(ld_path):
    """Load pos_norm, coord_x, coord_y from first clean lap of an LD file."""
    ld = ldData.fromfile(str(ld_path))
    available = list(ld)
    required = ['Car Pos Norm', 'Car Coord X', 'Car Coord Y']
    if not all(c in available for c in required):
        return None, None, None
    pos = np.array(ld['Car Pos Norm'].data, dtype=float)
    x   = np.array(ld['Car Coord X'].data,  dtype=float)
    y   = np.array(ld['Car Coord Y'].data,  dtype=float)
    # Truncate to first complete lap (before first pos_norm wrap)
    wraps = np.where(np.diff(pos) < -0.5)[0]
    if len(wraps) > 0:
        pos = pos[:wraps[0] + 1]
        x   = x  [:wraps[0] + 1]
        y   = y  [:wraps[0] + 1]
    return pos, x, y


def _rolling_mean(arr, w):
    return np.convolve(arr, np.ones(w) / w, mode='same')


def _compute_curvature(x, y):
    xs = _rolling_mean(x, SMOOTH_W)
    ys = _rolling_mean(y, SMOOTH_W)
    dx,  dy  = np.gradient(xs), np.gradient(ys)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    num   = np.abs(dx * ddy - dy * ddx)
    denom = np.maximum((dx**2 + dy**2) ** 1.5, 1e-10)
    return num / denom


def _build_average_line(ld_files):
    """Interpolate all drivers' first laps onto a common pos_norm grid and average."""
    pos_grid = np.linspace(0.02, 0.98, N_GRID)
    all_x, all_y = [], []
    for path in ld_files:
        pos, x, y = _load_ld_first_lap(path)
        if pos is None or len(pos) < 100:
            continue
        idx = np.argsort(pos)
        all_x.append(np.interp(pos_grid, pos[idx], x[idx]))
        all_y.append(np.interp(pos_grid, pos[idx], y[idx]))
    if not all_x:
        raise ValueError("No valid LD files found to build corner zone map.")
    return pos_grid, np.mean(all_x, axis=0), np.mean(all_y, axis=0)


def _find_apex_positions(curv, pos_grid):
    """Return list of pos_norm values at corner apexes."""
    threshold = np.percentile(curv, APEX_PCTILE)
    min_gap   = MIN_GAP_FRAC

    apexes = []
    above = curv > threshold
    in_peak, peak_start = False, 0

    for i in range(len(curv)):
        if above[i] and not in_peak:
            in_peak, peak_start = True, i
        elif not above[i] and in_peak:
            local_max = peak_start + int(np.argmax(curv[peak_start:i]))
            pos_val   = float(pos_grid[local_max])
            if not apexes or (pos_val - apexes[-1]) >= min_gap:
                apexes.append(pos_val)
            in_peak = False

    return apexes


def build_corner_zone_map(ld_files):
    """
    Build corner zone boundaries from a list of LD file paths.

    Returns a list of corner dicts:
    [
      {
        'corner_id':     0,
        'apex_pos':      0.123,
        'eingang_start': 0.073,   # start of entry approach
        'eingang_end':   0.098,   # midpoint of approach
        'mitte_end':     0.123,   # = apex_pos
        'exit_end':      0.148    # apex + exit zone
      }, ...
    ]
    """
    print(f"Building corner zone map from {len(ld_files)} LD files...")
    pos_grid, avg_x, avg_y = _build_average_line(ld_files)
    curv = _compute_curvature(avg_x, avg_y)
    apex_positions = _find_apex_positions(curv, pos_grid)
    print(f"  Found {len(apex_positions)} apexes")

    corners = []
    for corner_id, apex_pos in enumerate(apex_positions):
        half_approach = APPROACH_FRAC / 2.0
        corners.append({
            'corner_id':     corner_id,
            'apex_pos':      apex_pos,
            'eingang_start': apex_pos - APPROACH_FRAC,
            'eingang_end':   apex_pos - half_approach,
            'mitte_end':     apex_pos,
            'exit_end':      apex_pos + EXIT_FRAC,
        })
    return corners


def label_samples(pos_norm_array, corners):
    """
    Label each telemetry sample with a zone and corner_id.

    Args:
        pos_norm_array: numpy array of Car Pos Norm values (0.0-1.0, may wrap)
        corners: list of corner dicts from build_corner_zone_map / load_corner_zones

    Returns:
        zones:      int array  (0=straight, 1=eingang, 2=mitte, 3=apex)
        corner_ids: int array  (-1 for straight)
    """
    n = len(pos_norm_array)
    zones      = np.zeros(n, dtype=int)
    corner_ids = np.full(n, -1, dtype=int)

    for corner in corners:
        cid = corner['corner_id']
        es  = corner['eingang_start']
        ee  = corner['eingang_end']
        me  = corner['mitte_end']
        xe  = corner['exit_end']

        # Try direct and both wrap directions (handles corners near 0 or 1)
        for offset in [0.0, 1.0, -1.0]:
            p = pos_norm_array + offset
            eingang_m = (p >= es) & (p < ee)
            mitte_m   = (p >= ee) & (p < me)
            apex_m    = (p >= me) & (p <= xe)
            zones[eingang_m]      = ZONE_EINGANG
            zones[mitte_m]        = ZONE_MITTE
            zones[apex_m]         = ZONE_APEX
            corner_ids[eingang_m] = cid
            corner_ids[mitte_m]   = cid
            corner_ids[apex_m]    = cid

    return zones, corner_ids


def save_corner_zones(corners, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(corners, f, indent=2)
    print(f"Saved {len(corners)} corner zones -> {save_path}")


def load_corner_zones(load_path):
    with open(load_path) as f:
        return json.load(f)


if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))
    from utils import get_raw_data_path, get_features_path

    ld_files = sorted(get_raw_data_path().glob('*.ld'))
    if not ld_files:
        print("No .ld files in raw_data/ - trying training_data/")
        from utils import get_training_data_path
        ld_files = sorted(get_training_data_path().glob('*.ld'))

    corners = build_corner_zone_map(ld_files)
    save_corner_zones(corners, get_features_path() / 'corner_zones.json')

    print("\nCorner zone boundaries (pos_norm):")
    for c in corners:
        print(f"  Corner {c['corner_id']:2d}: eingang [{c['eingang_start']:.3f}-{c['eingang_end']:.3f}]"
              f"  mitte [{c['eingang_end']:.3f}-{c['mitte_end']:.3f}]"
              f"  apex [{c['mitte_end']:.3f}-{c['exit_end']:.3f}]")
