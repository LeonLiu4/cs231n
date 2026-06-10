#!/usr/bin/env python3
"""Collect the reduced K-sweep results and emit the Results tables + plots.

Fetches per-run ``eval_results.json`` from the Modal volume (or reads them
locally), aggregates Chamfer / F-score for the four embedding variants across
view counts, regenerates ``chamfer_plot.png`` / ``fscore_plot.png`` in
``figures/report/`` from REAL numbers, and prints LaTeX-ready table rows.

Usage:
    python3 scripts/collect_sweep_results.py --from-modal
    python3 scripts/collect_sweep_results.py            # local runs/ dir only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VOLUME = "pose-aware-data-v2"
VIEW_COUNTS = [1, 2, 4, 8]

# (display label, run-name prefix on the volume)
VARIANTS: list[tuple[str, str]] = [
    ("2D Positional", "rt_2d_positional"),
    ("View-ID", "rt_view_id"),
    ("Camera Pose", "rt_camera_pose"),
    ("Geometry-Aware", "rt_geometry_aware"),
]

SERIES_COLORS = {
    "2D Positional": "#2f6fed",
    "View-ID": "#8a5cf6",
    "Camera Pose": "#e0892f",
    "Geometry-Aware": "#e2553d",
}

OUT_FIG = ROOT / "figures" / "report"
OUT_DATA = ROOT / "runs" / "sweep"


def _run_name(prefix: str, k: int) -> str:
    return f"{prefix}_sweep_k{k}"


def fetch_from_modal(dest_root: Path) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    for _, prefix in VARIANTS:
        for k in VIEW_COUNTS:
            name = _run_name(prefix, k)
            local = dest_root / f"{name}.json"
            remote = f"runs/baselines/{name}/eval_results.json"
            subprocess.run(
                ["modal", "volume", "get", "--force", VOLUME, remote, str(local)],
                check=False,
                capture_output=True,
            )


def load_results(dest_root: Path) -> dict:
    """Return {label: {K: {"test_seen": {...}, "test_generalization": {...}}}}."""
    table: dict = {label: {} for label, _ in VARIANTS}
    for label, prefix in VARIANTS:
        for k in VIEW_COUNTS:
            path = dest_root / f"{_run_name(prefix, k)}.json"
            if not path.is_file():
                print(f"[missing] {path.name}")
                continue
            data = json.loads(path.read_text())
            table[label][k] = data.get("splits", {})
    return table


def _cell(table: dict, label: str, k: int, split: str, metric: str):
    splits = table.get(label, {}).get(k)
    if not splits or split not in splits:
        return None
    return splits[split].get(metric)


def latex_tables(table: dict, split: str = "test_seen") -> str:
    lines: list[str] = []
    for metric, scale, caption in [
        ("chamfer", 1e3, "Chamfer Distance ($\\times 10^3$, lower is better)"),
        ("f_score", 1.0, "F-score @ $\\tau=0.01$ (higher is better)"),
    ]:
        lines.append(f"% --- {caption} on {split} ---")
        for label, _ in VARIANTS:
            cells = []
            for k in VIEW_COUNTS:
                v = _cell(table, label, k, split, metric)
                cells.append("--" if v is None else f"{v * scale:.2f}")
            lines.append(f"{label:<14} & " + " & ".join(cells) + r" \\")
        lines.append("")
    return "\n".join(lines)


def _plot(table: dict, metric: str, scale: float, ylabel: str, title: str,
          fname: str, split: str = "test_seen") -> None:
    x = list(range(len(VIEW_COUNTS)))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, _ in VARIANTS:
        ys, xs = [], []
        for i, k in enumerate(VIEW_COUNTS):
            v = _cell(table, label, k, split, metric)
            if v is not None:
                xs.append(i)
                ys.append(v * scale)
        if not ys:
            continue
        emph = label == "Geometry-Aware"
        ax.plot(xs, ys, color=SERIES_COLORS[label], marker="o",
                lw=2.4 if emph else 1.7, markersize=6 if emph else 5,
                label=label, zorder=5 if emph else 3)
    ax.set_xticks(x)
    ax.set_xticklabels(VIEW_COUNTS)
    ax.set_xlabel("Number of input views (K)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color="#e7ebf1", lw=0.9)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG / fname}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-modal", action="store_true")
    parser.add_argument("--split", default="test_seen")
    args = parser.parse_args()

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    if args.from_modal:
        fetch_from_modal(OUT_DATA)

    table = load_results(OUT_DATA)
    (OUT_DATA / "sweep_results.json").write_text(json.dumps(table, indent=2))

    _plot(table, "chamfer", 1e3,
          r"Chamfer distance ($\times 10^3$, lower better)",
          "Chamfer Distance vs. Number of Views", "chamfer_plot.png", args.split)
    _plot(table, "f_score", 1.0,
          r"F-score @ $\tau=0.01$ (higher better)",
          "F-score vs. Number of Views", "fscore_plot.png", args.split)

    print("\n" + "=" * 60 + "\nLaTeX table rows (test_seen):\n" + "=" * 60)
    print(latex_tables(table, args.split))


if __name__ == "__main__":
    main()
