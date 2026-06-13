import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent))
from ldparser import ldData

RAW_DATA = Path(__file__).parent.parent / "raw_data"
OUTPUT   = Path(__file__).parent.parent / "results" / "track_map_corners.png"

N_GRID       = 3000   # resolution of average racing line
SMOOTH_W     = 80     # smoothing window for curvature on average line
APEX_PCTILE  = 85     # curvature percentile threshold for apex detection
MIN_GAP_FRAC = 0.04   # min gap between apexes as fraction of track length
APPROACH_FRAC = 0.05  # approach zone before apex (fraction of track length)
EXIT_FRAC     = 0.025 # exit zone after apex


def load_driver_data(path: Path):
    ld  = ldData.fromfile(str(path))
    x   = np.array(ld["Car Coord X"].data, dtype=float)
    y   = np.array(ld["Car Coord Y"].data, dtype=float)
    pos = np.array(ld["Car Pos Norm"].data, dtype=float)
    return x, y, pos


def rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(arr, np.ones(w) / w, mode="same")


def compute_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xs = rolling_mean(x, SMOOTH_W)
    ys = rolling_mean(y, SMOOTH_W)
    dx, dy   = np.gradient(xs), np.gradient(ys)
    ddx, ddy = np.gradient(dx),  np.gradient(dy)
    num   = np.abs(dx * ddy - dy * ddx)
    denom = np.maximum((dx**2 + dy**2) ** 1.5, 1e-10)
    return num / denom


def arc_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return np.concatenate([[0.0], np.cumsum(ds)])


def build_average_line(ld_files):
    """Interpolate all drivers onto a common Car Pos Norm grid and average."""
    pos_grid = np.linspace(0.02, 0.98, N_GRID)
    all_x, all_y = [], []
    for path in ld_files:
        x, y, pos = load_driver_data(path)
        idx = np.argsort(pos)
        all_x.append(np.interp(pos_grid, pos[idx], x[idx]))
        all_y.append(np.interp(pos_grid, pos[idx], y[idx]))
    return np.mean(all_x, axis=0), np.mean(all_y, axis=0)


def find_apexes(curv: np.ndarray, arc_s: np.ndarray) -> np.ndarray:
    """Find local curvature maxima above threshold with minimum spacing."""
    threshold = np.percentile(curv, APEX_PCTILE)
    total_len = arc_s[-1]
    min_gap   = MIN_GAP_FRAC * total_len

    apexes = []
    above  = curv > threshold
    in_peak, peak_start = False, 0

    for i in range(len(curv)):
        if above[i] and not in_peak:
            in_peak, peak_start = True, i
        elif not above[i] and in_peak:
            local_max = peak_start + int(np.argmax(curv[peak_start:i]))
            if not apexes or (arc_s[local_max] - arc_s[apexes[-1]]) >= min_gap:
                apexes.append(local_max)
            in_peak = False

    return np.array(apexes, dtype=int)


def build_color_values(apex_idxs: np.ndarray, arc_s: np.ndarray) -> np.ndarray:
    """
    For each point on the average line assign a value 0..1:
      0    = straight (no coloring)
      0..1 = approach ramp  (entry → apex)
      1..0 = exit ramp      (apex → just after)
    """
    total_len   = arc_s[-1]
    approach_d  = APPROACH_FRAC * total_len
    exit_d      = EXIT_FRAC     * total_len
    color_val   = np.zeros(N_GRID)

    for apex_idx in apex_idxs:
        s_apex = arc_s[apex_idx]
        for i in range(N_GRID):
            dist = arc_s[i] - s_apex          # negative = before apex
            if -approach_d <= dist <= 0:       # approach zone
                t = (dist + approach_d) / approach_d   # 0 at entry, 1 at apex
                color_val[i] = max(color_val[i], t)
            elif 0 < dist <= exit_d:           # exit zone
                t = 1.0 - dist / exit_d        # 1 at apex, 0 at exit
                color_val[i] = max(color_val[i], t)

    return color_val


def main():
    ld_files = sorted(RAW_DATA.glob("*.ld"))
    if not ld_files:
        print(f"Keine .ld Dateien in {RAW_DATA}")
        return

    print("Berechne Durchschnittslinie…")
    avg_x, avg_y = build_average_line(ld_files)

    arc_s = arc_length(avg_x, avg_y)
    print(f"  Streckenlänge: {arc_s[-1]:.1f} Koordinaten-Einheiten")

    curv      = compute_curvature(avg_x, avg_y)
    apex_idxs = find_apexes(curv, arc_s)
    print(f"  {len(apex_idxs)} Apexe gefunden")

    color_val = build_color_values(apex_idxs, arc_s)

    # --- LineCollection ---
    corner_cmap = LinearSegmentedColormap.from_list(
        "corner", ["#ffee00", "#ff8800", "#cc0000"], N=256
    )
    pts      = np.array([avg_x, avg_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    vals     = color_val[:-1]
    is_corner = vals > 0.02

    # --- figure ---
    fig, ax = plt.subplots(figsize=(8, 14))
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")

    # all drivers as faint background
    for path in ld_files:
        x, y, _ = load_driver_data(path)
        ax.plot(x, y, color="#282828", linewidth=0.5, alpha=0.5, zorder=1)

    # straights
    lc_s = LineCollection(
        segments[~is_corner], colors="#505050", linewidths=1.0, alpha=0.8, zorder=2
    )
    ax.add_collection(lc_s)

    # corner zones with gradient
    lc_c = LineCollection(
        segments[is_corner],
        colors=corner_cmap(vals[is_corner]),
        linewidths=2.5,
        alpha=0.95,
        zorder=3,
    )
    ax.add_collection(lc_c)

    # mark apexes
    ax.scatter(
        avg_x[apex_idxs], avg_y[apex_idxs],
        color="white", s=18, zorder=5, linewidths=0.5,
        edgecolors="#cc0000", label="Apex"
    )

    ax.set_xlim(avg_x.min() - 60, avg_x.max() + 60)
    ax.set_ylim(avg_y.min() - 60, avg_y.max() + 60)
    ax.set_title("Streckenplot – Kurvenabschnitte (Ø Rennspur)", fontsize=13, pad=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    sm = plt.cm.ScalarMappable(cmap=corner_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Kurvenabschnitt", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(labelcolor="white")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Eingang", "Mitte", "Apex"], fontsize=8)
    plt.setp(cbar.ax.get_yticklabels(), color="white")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Gespeichert: {OUTPUT}")


if __name__ == "__main__":
    main()
