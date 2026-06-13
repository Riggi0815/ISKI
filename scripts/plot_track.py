import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from ldparser import ldData

RAW_DATA = Path(__file__).parent.parent / "raw_data"
OUTPUT = Path(__file__).parent.parent / "results" / "track_map.png"


def extract_driver_name(path: Path) -> str:
    parts = path.stem.split("_&_")
    return parts[2] if len(parts) >= 3 else path.stem


def load_coords(path: Path):
    ld = ldData.fromfile(str(path))
    x = np.array(ld["Car Coord X"].data, dtype=float)
    y = np.array(ld["Car Coord Y"].data, dtype=float)
    return x, y


def main():
    ld_files = sorted(RAW_DATA.glob("*.ld"))
    if not ld_files:
        print(f"Keine .ld Dateien gefunden in {RAW_DATA}")
        return

    colors = cm.tab10(np.linspace(0, 1, len(ld_files)))

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    for path, color in zip(ld_files, colors):
        driver = extract_driver_name(path)
        try:
            x, y = load_coords(path)
            ax.plot(x, y, linewidth=0.8, alpha=0.85, color=color, label=driver)
        except Exception as e:
            print(f"Fehler bei {path.name}: {e}")

    ax.set_title("Streckenplot – Car Coord X/Y", fontsize=14, pad=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(
        loc="best",
        fontsize=8,
        framealpha=0.4,
        facecolor="#333333",
        labelcolor="white",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Gespeichert: {OUTPUT}")


if __name__ == "__main__":
    main()
