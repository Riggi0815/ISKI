"""
Corner zone detection module.
Builds a Car-Pos-Norm → zone label map from training LD files.
Zones: 0=straight, 1=eingang, 2=mitte, 3=apex
"""

import sys
import json
from pathlib import Path
import numpy as np

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


def _first_lap_from_driver(driver_df):
    """Extract (pos_norm, x, y) for the first clean lap of a driver.

    LD path:  pos_norm + coord_x/coord_y  — direct, wrap-detected.
    HTF path: gps_lat/gps_long            — first lap via n_lap_number,
              pos_norm derived from cumulative GPS distance.
    """
    # --- LD path ---
    if {'pos_norm', 'coord_x', 'coord_y'}.issubset(driver_df.columns):
        pos = driver_df['pos_norm'].values.astype(float)
        x   = driver_df['coord_x'].values.astype(float)
        y   = driver_df['coord_y'].values.astype(float)
        wraps = np.where(np.diff(pos) < -0.5)[0]
        if len(wraps) > 0:
            pos, x, y = pos[:wraps[0]+1], x[:wraps[0]+1], y[:wraps[0]+1]
        return pos, x, y

    # --- HTF path: GPS coordinates ---
    if {'gps_lat', 'gps_long'}.issubset(driver_df.columns):
        if 'n_lap_number' in driver_df.columns:
            first_lap_num = driver_df['n_lap_number'].min()
            lap_df = driver_df[driver_df['n_lap_number'] == first_lap_num]
        else:
            lap_df = driver_df
        x = lap_df['gps_long'].values.astype(float)
        y = lap_df['gps_lat'].values.astype(float)
        if len(x) < 100:
            return None, None, None
        # Cumulative distance along GPS path → normalized 0-1 pos_norm
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        cum_dist = np.cumsum(np.sqrt(dx**2 + dy**2))
        total = cum_dist[-1]
        if total < 1e-6:
            return None, None, None
        return cum_dist / total, x, y

    return None, None, None


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


def _build_average_line(telemetry_df):
    """Interpolate all drivers' first laps onto a common pos_norm grid and average."""
    pos_grid = np.linspace(0.02, 0.98, N_GRID)
    all_x, all_y = [], []
    for driver_id, driver_df in telemetry_df.groupby('driver_id'):
        pos, x, y = _first_lap_from_driver(driver_df)
        if pos is None or len(pos) < 100:
            continue
        idx = np.argsort(pos)
        all_x.append(np.interp(pos_grid, pos[idx], x[idx]))
        all_y.append(np.interp(pos_grid, pos[idx], y[idx]))
    if not all_x:
        raise ValueError("No coordinate data found in telemetry (need LD or GPS columns).")
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


def build_corner_zone_map(telemetry_df):
    """
    Build corner zone boundaries from a parsed telemetry DataFrame.
    DataFrame must have columns: driver_id, pos_norm, coord_x, coord_y.

    Returns a list of corner dicts:
    [
      {
        'corner_id':     0,
        'apex_pos':      0.123,
        'eingang_start': 0.073,
        'eingang_end':   0.098,
        'mitte_end':     0.123,
        'exit_end':      0.148
      }, ...
    ]
    """
    n_drivers = telemetry_df['driver_id'].nunique()
    print(f"Building corner zone map from {n_drivers} drivers...")
    pos_grid, avg_x, avg_y = _build_average_line(telemetry_df)
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
    import pandas as pd
    sys.path.append(str(Path(__file__).parent))
    from utils import get_processed_data_path, get_features_path

    processed = get_processed_data_path()

    # Load best available telemetry. Accepts LD (pos_norm/coord_x/coord_y) or
    # HTF (gps_lat/gps_long). Priority: combined > ld-only > htf-only.
    LD_COLS  = {'pos_norm', 'coord_x', 'coord_y'}
    GPS_COLS = {'gps_lat', 'gps_long'}
    telemetry_df = None
    for candidate in ['telemetry_combined', 'telemetry_ld', 'telemetry_all']:
        pkl = processed / f'{candidate}.pkl'
        if not pkl.exists():
            continue
        df = pd.read_pickle(pkl)
        if LD_COLS.issubset(df.columns) or GPS_COLS.issubset(df.columns):
            src = 'LD coords' if LD_COLS.issubset(df.columns) else 'GPS coords'
            print(f"Loading {pkl.name}... ({src})")
            telemetry_df = df
            break
        else:
            print(f"  Skipping {pkl.name} (no usable coordinate columns)")

    if telemetry_df is None:
        print("No telemetry with coordinate data found.")
        print("Need either LD (pos_norm/coord_x/coord_y) or HTF (gps_lat/gps_long).")
        sys.exit(1)

    corners = build_corner_zone_map(telemetry_df)
    save_corner_zones(corners, get_features_path() / 'corner_zones.json')

    print("\nCorner zone boundaries (pos_norm):")
    for c in corners:
        print(f"  Corner {c['corner_id']:2d}: eingang [{c['eingang_start']:.3f}-{c['eingang_end']:.3f}]"
              f"  mitte [{c['eingang_end']:.3f}-{c['mitte_end']:.3f}]"
              f"  apex [{c['mitte_end']:.3f}-{c['exit_end']:.3f}]")
