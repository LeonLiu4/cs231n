#!/usr/bin/env python3
"""Generate report figures for the LaTeX writeup into ``figures/report/``.

Most figures use real assets (renders, meshes, GT point clouds, dataset
counts). Five figures depend on trained-model outputs that do not exist yet
(baseline_comparison, chamfer_plot, fscore_plot, qualitative_results,
failure_cases); those are emitted as clearly labeled placeholders with the
correct layout/axes so the report compiles and can be swapped out later.

Usage:
    python3 scripts/make_report_figures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image

from src.data.camera import NESTED_VIEW_SCHEDULES, NUM_AZIMUTH, pose_for_bin
from src.data.render import normalize_mesh, render_rgb_view

OUT = ROOT / "figures" / "report"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Shared style
# ----------------------------------------------------------------------------
ACCENT = "#2f6fed"
ACCENT_DARK = "#1f3f85"
OURS = "#e2553d"
GT = "#3a7d44"
INK = "#1b2330"
MUTED = "#6b788c"
PANEL_EDGE = "#cdd5e0"
BG = "#ffffff"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "font.size": 12,
        "axes.edgecolor": "#9aa6b6",
        "axes.linewidth": 1.0,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
    }
)

CATEGORIES = [
    ("03001627", "chair"),
    ("04379243", "table"),
    ("04256520", "sofa"),
    ("02958343", "car"),
    ("02691156", "airplane"),
    ("03636649", "lamp"),
    ("02933112", "cabinet"),
]


# ----------------------------------------------------------------------------
# Asset helpers
# ----------------------------------------------------------------------------
PREFERRED_CHAIR = "03001627_8117c55b8bbdbbc54c5c5c89015f1980"
AIRPLANE_DIR_NAME = "02691156_a36d00e2f7414043f2b0736dd4d8afe0"


def _first_chair_object() -> Path:
    objs = ROOT / "data/master_shapenet/objects"
    pref = objs / PREFERRED_CHAIR
    if pref.is_dir():
        return pref
    chairs = sorted(d for d in objs.iterdir() if d.is_dir() and d.name.startswith("03001627_"))
    return chairs[0]


def _chair_mesh_path(obj_dir: Path) -> Path:
    syn, mid = obj_dir.name.split("_", 1)
    return ROOT / "data/ShapeNetCore.v2" / syn / mid / "models/model_normalized.obj"


def _load_points(obj_dir: Path, n: int = 4000) -> np.ndarray:
    pts = np.load(obj_dir / "gt_points.npy")
    if pts.ndim == 3:
        pts = pts.reshape(-1, pts.shape[-1])
    pts = pts[:, :3]
    if len(pts) > n:
        idx = np.linspace(0, len(pts) - 1, n, dtype=int)
        pts = pts[idx]
    return pts


def _view_image(obj_dir: Path, prefer: int = 21) -> Image.Image:
    p = obj_dir / f"view_{prefer:02d}.png"
    if not p.exists():
        cands = sorted(obj_dir.glob("view_*.png"))
        p = cands[len(cands) // 2] if cands else None
    return Image.open(p).convert("RGB")


def _load_mesh(path: Path, upright: bool = True):
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if upright:
        # ShapeNet meshes are y-up but the CPU renderer is z-up; rotate +90 deg
        # about x so objects stand upright in renders.
        mesh = mesh.copy()
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0]))
    return normalize_mesh(mesh)


def _render_mesh_pil(mesh, image_size: int = 320, bin_az: int = 2, bin_el: int = 1) -> Image.Image:
    rgb = render_rgb_view(mesh, pose_for_bin(bin_az, bin_el), image_size=image_size)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8"))


def _scatter3d(ax, pts, color=None, s=2.0):
    c = pts[:, 2] if color is None else color
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=c, cmap="viridis", s=s, linewidths=0)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=42)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis.line.set_color((1, 1, 1, 0))


def _placeholder(ax, text="Pending model outputs"):
    ax.add_patch(
        FancyBboxPatch(
            (0.04, 0.04),
            0.92,
            0.92,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            transform=ax.transAxes,
            facecolor="#f4f6fa",
            edgecolor=PANEL_EDGE,
            linestyle=(0, (5, 4)),
            linewidth=1.6,
        )
    )
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=12,
        style="italic",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _save(fig, name: str):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------------------------
# 1. task.png
# ----------------------------------------------------------------------------
def fig_task(hero_dir):
    img = _view_image(hero_dir)
    pts = _load_points(hero_dir)
    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.55, 1.0], wspace=0.05)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img)
    ax0.set_title("Input 2D views", fontsize=14)
    ax0.axis("off")

    axm = fig.add_subplot(gs[1])
    axm.axis("off")
    axm.set_xlim(0, 1)
    axm.set_ylim(0, 1)
    axm.add_patch(
        FancyBboxPatch(
            (0.18, 0.4),
            0.64,
            0.2,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=ACCENT,
            edgecolor="none",
        )
    )
    axm.text(0.5, 0.5, "Reconstruction\nmodel", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    axm.add_patch(FancyArrowPatch((0.0, 0.5), (0.17, 0.5), arrowstyle="-|>", mutation_scale=22, color=INK, lw=2))
    axm.add_patch(FancyArrowPatch((0.83, 0.5), (1.0, 0.5), arrowstyle="-|>", mutation_scale=22, color=INK, lw=2))

    ax1 = fig.add_subplot(gs[2], projection="3d")
    _scatter3d(ax1, pts)
    ax1.set_title("Predicted 3D point cloud", fontsize=14)

    fig.suptitle("2D-to-3D Reconstruction Task", fontsize=16, fontweight="bold", y=1.02)
    _save(fig, "task.png")


# ----------------------------------------------------------------------------
# 2. patch_tokens.png
# ----------------------------------------------------------------------------
def fig_patch_tokens(hero_dir):
    img = _view_image(hero_dir).resize((224, 224))
    fig = plt.figure(figsize=(13, 3.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.25, 1.0], wspace=0.28)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img)
    ax0.set_title("224 x 224 image", fontsize=12)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(img)
    n = 14
    for i in range(n + 1):
        ax1.axhline(i * 224 / n, color="white", lw=0.8, alpha=0.8)
        ax1.axvline(i * 224 / n, color="white", lw=0.8, alpha=0.8)
    ax1.set_title("Split into patches", fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[2])
    ax2.axis("off")
    ax2.set_title("Sequence of tokens", fontsize=12)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 3)
    for i in range(8):
        x = 0.3 + i * 1.15
        ax2.add_patch(FancyBboxPatch((x, 1.2), 0.95, 0.95, boxstyle="round,pad=0.02,rounding_size=0.08",
                                     facecolor="#dbe4fb", edgecolor=ACCENT, lw=1.4))
        ax2.text(x + 0.47, 1.67, f"t{i+1}", ha="center", va="center", fontsize=9, color=ACCENT_DARK)
    ax2.text(9.5, 1.67, "...", fontsize=16, va="center")

    ax3 = fig.add_subplot(gs[3])
    ax3.axis("off")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 3)
    ax3.set_title("+ embedding", fontsize=12)
    labels = ["2D position", "view ID", "camera pose", "geometry / ray"]
    colors = [ACCENT, "#8a5cf6", "#e0892f", OURS]
    for i, (lab, col) in enumerate(zip(labels, colors)):
        y = 2.5 - i * 0.62
        ax3.add_patch(FancyBboxPatch((0.05, y - 0.2), 0.9, 0.4, boxstyle="round,pad=0.02,rounding_size=0.08",
                                     facecolor=col, edgecolor="none", alpha=0.9))
        ax3.text(0.5, y, lab, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")

    fig.suptitle("Image Patches to Visual Tokens with Embeddings", fontsize=15, fontweight="bold", y=1.04)
    _save(fig, "patch_tokens.png")


# ----------------------------------------------------------------------------
# 3. representation_comparison.png
# ----------------------------------------------------------------------------
def fig_representation(chair_dir):
    mesh = _load_mesh(_chair_mesh_path(chair_dir))
    pose = pose_for_bin(2, 1)
    rgb = render_rgb_view(mesh, pose, image_size=420)

    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 3, wspace=0.08)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(np.clip(rgb, 0, 1))
    ax0.set_title("Mesh", fontsize=14)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1], projection="3d")
    try:
        vox = mesh.voxelized(pitch=0.06)
        matrix = np.asarray(vox.matrix)
        # downsample dense grids for speed
        if matrix.size > 60000:
            f = int(np.ceil((matrix.size / 60000) ** (1 / 3)))
            matrix = matrix[::f, ::f, ::f]
        ax1.voxels(matrix.transpose(0, 2, 1), facecolors="#9bb7f0", edgecolor="#5d82d6", linewidth=0.2)
    except Exception as exc:  # pragma: no cover
        ax1.text2D(0.5, 0.5, f"voxel failed\n{exc}", ha="center", transform=ax1.transAxes)
    ax1.set_title("Voxel grid", fontsize=14)
    ax1.set_box_aspect((1, 1, 1))
    ax1.view_init(elev=18, azim=42)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
    ax1.grid(False)
    for axis in (ax1.xaxis, ax1.yaxis, ax1.zaxis):
        axis.set_pane_color((1, 1, 1, 0))

    ax2 = fig.add_subplot(gs[2], projection="3d")
    _scatter3d(ax2, _load_points(chair_dir), s=2.2)
    ax2.set_title("Point cloud (2048 x 3, ours)", fontsize=14)

    fig.suptitle("3D Output Representations", fontsize=16, fontweight="bold", y=1.02)
    _save(fig, "representation_comparison.png")


# ----------------------------------------------------------------------------
# 4. category_distribution.png
# ----------------------------------------------------------------------------
def fig_category_distribution():
    names = [c[1] for c in CATEGORIES] + ["broad"]
    counts = [200] * 7 + [600]
    colors = ["#4c78d6"] * 7 + [OURS]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, padding=3, fontsize=11, color=INK)
    ax.set_ylabel("Objects")
    ax.set_title("Dataset Composition by Category")
    ax.set_ylim(0, 720)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.02)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4c78d6"),
        plt.Rectangle((0, 0), 1, 1, color=OURS),
    ]
    ax.legend(handles, ["7 main categories (200 each)", "broad generalization group"], frameon=False, loc="upper left")
    _save(fig, "category_distribution.png")


# ----------------------------------------------------------------------------
# 5. dataset_split.png
# ----------------------------------------------------------------------------
def fig_dataset_split():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 1620)
    ax.set_ylim(-0.5, 1.8)

    # main row (1400 -> 980/210/210)
    segs_main = [("train", 980, ACCENT), ("val", 210, "#8a5cf6"), ("test_seen", 210, "#e0892f")]
    x = 0
    for lab, w, col in segs_main:
        ax.barh(1.2, w, left=x, height=0.55, color=col, edgecolor="white")
        ax.text(x + w / 2, 1.2, f"{lab}\n{w}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        x += w
    ax.text(-15, 1.2, "Main\n1,400", ha="right", va="center", fontsize=11, color=INK, fontweight="bold")

    # broad row (600 -> generalization)
    ax.barh(0.4, 600, left=0, height=0.55, color=GT, edgecolor="white")
    ax.text(300, 0.4, "test_generalization\n600", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.text(-15, 0.4, "Broad\n600", ha="right", va="center", fontsize=11, color=INK, fontweight="bold")

    ax.set_title("Object-Level Train / Val / Test Split", loc="left")
    ax.axis("off")
    _save(fig, "dataset_split.png")


# ----------------------------------------------------------------------------
# 6. mesh_normalization.png
# ----------------------------------------------------------------------------
def fig_mesh_normalization(chair_dir):
    import trimesh

    base = _load_mesh(_chair_mesh_path(chair_dir))
    pose = pose_for_bin(2, 1)

    raw = base.copy()
    raw.apply_scale(0.6)
    raw.apply_translation(np.array([0.35, 0.15, 0.0]))

    centered = raw.copy()
    centered.apply_translation(-centered.vertices.mean(axis=0))

    normalized = normalize_mesh(centered)

    panels = [("Raw mesh", raw), ("Centered", centered), ("Scaled to unit sphere", normalized)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (title, m) in zip(axes, panels):
        rgb = render_rgb_view(m, pose, image_size=360)
        ax.imshow(np.clip(rgb, 0, 1))
        ax.set_title(title, fontsize=13)
        ax.axhline(180, color=ACCENT, lw=0.6, alpha=0.4)
        ax.axvline(180, color=ACCENT, lw=0.6, alpha=0.4)
        ax.set_xticks([]); ax.set_yticks([])
    axes[2].add_patch(Circle((180, 180), 150, fill=False, color=OURS, lw=1.8, linestyle=(0, (5, 3))))
    fig.suptitle("Mesh Normalization (center + scale + align)", fontsize=16, fontweight="bold", y=1.02)
    _save(fig, "mesh_normalization.png")


# ----------------------------------------------------------------------------
# 7. camera_layout.png
# ----------------------------------------------------------------------------
def fig_camera_layout():
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    R = 1.0
    ax.add_patch(Circle((0, 0), R, fill=False, color=PANEL_EDGE, lw=1.5, linestyle=(0, (4, 4))))
    # object
    ax.scatter([0], [0], s=420, marker="*", color=OURS, zorder=5)
    ax.text(0, -0.16, "object", ha="center", va="top", fontsize=11, color=INK)
    for b in range(NUM_AZIMUTH):
        th = 2 * np.pi * b / NUM_AZIMUTH
        cx, cy = R * np.cos(th), R * np.sin(th)
        ax.scatter([cx], [cy], s=130, marker="^", color=ACCENT, edgecolor="white", zorder=4,
                   transform=ax.transData)
        ax.add_patch(FancyArrowPatch((cx * 0.92, cy * 0.92), (cx * 0.18, cy * 0.18),
                                     arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1.0, alpha=0.7))
    ax.text(0, R + 0.13, "16 azimuth bins x 4 elevations", ha="center", fontsize=11, color=MUTED)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Camera Layout Around Object", fontsize=15)
    _save(fig, "camera_layout.png")


# ----------------------------------------------------------------------------
# 8. nested_views.png
# ----------------------------------------------------------------------------
def fig_nested_views():
    ks = [1, 2, 4, 8, 16]
    fig, axes = plt.subplots(1, len(ks), figsize=(14, 3.2))
    for ax, k in zip(axes, ks):
        active = {a for a, _ in NESTED_VIEW_SCHEDULES[k]}
        ax.add_patch(Circle((0, 0), 1, fill=False, color=PANEL_EDGE, lw=1.2, linestyle=(0, (4, 4))))
        ax.scatter([0], [0], s=120, marker="*", color=OURS, zorder=5)
        for b in range(NUM_AZIMUTH):
            th = 2 * np.pi * b / NUM_AZIMUTH
            cx, cy = np.cos(th), np.sin(th)
            on = b in active
            ax.scatter([cx], [cy], s=90 if on else 36, marker="^" if on else "o",
                       color=ACCENT if on else "#d4dbe6",
                       edgecolor="white" if on else "none", zorder=4 if on else 2)
        ax.set_title(f"K = {k}", fontsize=13)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("Nested Evaluation View Subsets (smaller K ⊂ larger K)", fontsize=15, fontweight="bold", y=1.05)
    _save(fig, "nested_views.png")


# ----------------------------------------------------------------------------
# 9. embedding_variants.png
# ----------------------------------------------------------------------------
def fig_embedding_variants():
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.4, 2.5), 2.2, 1.0, boxstyle="round,pad=0.05,rounding_size=0.12",
                                facecolor="#dbe4fb", edgecolor=ACCENT, lw=1.6))
    ax.text(1.5, 3.0, "Patch / view\ntoken $x_{i,p}$", ha="center", va="center", fontsize=11, color=ACCENT_DARK)

    variants = [
        ("2D Positional", "ViT patch position", ACCENT),
        ("View-ID", "learned per-view index", "#8a5cf6"),
        ("Camera Pose", "azimuth, elev, eye", "#e0892f"),
        ("Geometry-Aware", "ray / extrinsic geometry", OURS),
    ]
    for i, (title, sub, col) in enumerate(variants):
        y = 5.0 - i * 1.25
        ax.add_patch(FancyArrowPatch((2.65, 3.0), (5.0, y + 0.0), arrowstyle="-|>",
                                     mutation_scale=14, color=MUTED, lw=1.3))
        ax.add_patch(FancyBboxPatch((5.1, y - 0.45), 6.4, 0.9, boxstyle="round,pad=0.04,rounding_size=0.1",
                                    facecolor=col, edgecolor="none", alpha=0.92))
        ax.text(5.45, y, f"+  {title}", ha="left", va="center", color="white", fontsize=12, fontweight="bold")
        ax.text(11.3, y, sub, ha="right", va="center", color="white", fontsize=9.5, style="italic")

    ax.text(6.0, 5.85, "Embedding injected before fusion", ha="center", fontsize=12, color=INK, fontweight="bold")
    ax.set_title("Embedding Variants Compared", fontsize=15, loc="left")
    _save(fig, "embedding_variants.png")


# ----------------------------------------------------------------------------
# 10. metric_intuition.png
# ----------------------------------------------------------------------------
def fig_embedding_comparison():
    """Side-by-side contrast of what each embedding variant injects."""
    variants = [
        ("2D Positional", ACCENT, "+ (row, col)", "2D grid location only"),
        ("View-ID", "#8a5cf6", "+ view #k", "abstract view index"),
        ("Camera Pose", "#e0892f", "+ (az, el, dist)", "global pose per view"),
        ("Geometry-Aware", OURS, "+ ray(o, d)", "per-patch ray geometry"),
    ]

    fig = plt.figure(figsize=(15, 6.8))
    gs = fig.add_gridspec(
        2, 4, height_ratios=[1.0, 0.62], hspace=0.18, wspace=0.12,
        left=0.06, right=0.98, top=0.86, bottom=0.06,
    )

    def _grid(ax, color, hi=(2, 1)):
        for i in range(4):
            ax.plot([0, 3], [i, i], color="#b9c2d0", lw=0.9, zorder=1)
            ax.plot([i, i], [0, 3], color="#b9c2d0", lw=0.9, zorder=1)
        ax.add_patch(plt.Rectangle((hi[0], hi[1]), 1, 1, facecolor=color, alpha=0.85, zorder=2))

    for col, (name, color, tag, sub) in enumerate(variants):
        ax = fig.add_subplot(gs[0, col])
        ax.set_xlim(-0.4, 5.4)
        ax.set_ylim(-1.5, 3.8)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch((-0.3, 3.05), 5.6, 0.7, boxstyle="round,pad=0.02,rounding_size=0.1",
                           facecolor=color, edgecolor="none", clip_on=False)
        )
        ax.text(2.5, 3.4, name, ha="center", va="center", color="white", fontsize=12.5, fontweight="bold")

        if col == 0:  # 2D positional: just the grid + coordinate
            _grid(ax, color)
            ax.annotate("(r, c)", xy=(2.5, 1.5), xytext=(4.1, 2.3), fontsize=11, color=color,
                        ha="center", arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))
        elif col == 1:  # view-id: grid + badge
            _grid(ax, color)
            ax.add_patch(FancyBboxPatch((3.6, 0.3), 1.5, 0.7, boxstyle="round,pad=0.02,rounding_size=0.1",
                                        facecolor=color, edgecolor="none"))
            ax.text(4.35, 0.65, "view k", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        elif col == 2:  # camera pose: object + camera + az/el arrows
            ax.scatter([1.4], [1.3], s=260, marker="*", color=INK, zorder=3)
            cam = (4.2, 2.6)
            ax.add_patch(FancyBboxPatch((cam[0] - 0.45, cam[1] - 0.3), 0.9, 0.6,
                                        boxstyle="round,pad=0.02,rounding_size=0.08",
                                        facecolor=color, edgecolor="none"))
            ax.add_patch(FancyArrowPatch(cam, (1.7, 1.45), arrowstyle="-|>", mutation_scale=14,
                                         color=MUTED, lw=1.5))
            ax.annotate("", xy=(2.6, 1.3), xytext=(1.4, 1.3),
                        arrowprops=dict(arrowstyle="-", color=color, lw=1.2, linestyle=(0, (3, 2))))
            ax.text(2.3, 0.8, "azimuth /\nelevation", ha="center", va="center", fontsize=9, color=color)
        else:  # geometry-aware: camera origin + ray through patch to object
            _grid(ax, color, hi=(2, 1))
            origin = (4.7, 3.0)
            patch_c = (2.5, 1.5)
            ax.scatter([origin[0]], [origin[1]], s=70, color=color, zorder=4)
            ax.text(origin[0], origin[1] + 0.35, "origin o", ha="center", fontsize=9, color=color)
            ax.add_patch(FancyArrowPatch(origin, (patch_c[0] - 0.9, patch_c[1] - 0.9),
                                         arrowstyle="-|>", mutation_scale=14, color=color, lw=1.8))
            ax.text(3.7, 0.2, "ray d", ha="center", fontsize=9, color=color)

        ax.text(2.5, -1.15, sub, ha="center", va="center", fontsize=10, color=INK, style="italic")

    # comparison table
    rows = [
        ("Granularity", ["per-patch", "per-view", "per-view", "per-patch"]),
        ("Camera info", ["none", "implicit", "explicit", "explicit"]),
        ("3D geometry", ["none", "none", "partial", "explicit (rays)"]),
    ]
    axt = fig.add_subplot(gs[1, :])
    axt.set_xlim(0, 10)
    axt.set_ylim(0, len(rows) + 0.4)
    axt.axis("off")
    col_x = [3.0, 4.75, 6.5, 8.4]
    colors = [v[1] for v in variants]
    for cx, c in zip(col_x, colors):
        axt.add_patch(plt.Rectangle((cx - 0.82, len(rows) - 0.05), 1.64, 0.42, color=c, clip_on=False))
    for cx, (name, _, _, _) in zip(col_x, variants):
        axt.text(cx, len(rows) + 0.16, name.split()[0], ha="center", va="center", color="white",
                 fontsize=9.5, fontweight="bold")
    for r, (label, vals) in enumerate(rows):
        y = len(rows) - 1 - r + 0.35
        if r % 2 == 0:
            axt.add_patch(plt.Rectangle((0.2, y - 0.22), 9.6, 0.5, color="#f2f5fa", zorder=0))
        axt.text(0.4, y, label, ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")
        for cx, val in zip(col_x, vals):
            strong = val in ("explicit", "explicit (rays)", "per-patch")
            axt.text(cx, y, val, ha="center", va="center", fontsize=10,
                     color=OURS if val == "explicit (rays)" else INK,
                     fontweight="bold" if strong else "normal")

    fig.suptitle("Embedding Variants: What Information Each Injects", fontsize=16, fontweight="bold", y=0.97)
    fig.text(0.52, 0.905, "increasing 3D / camera awareness  →", ha="center", fontsize=11, color=MUTED)
    _save(fig, "embedding_comparison.png")


def fig_metric_intuition():
    rng = np.random.default_rng(7)
    gt = rng.normal(size=(14, 2)) * np.array([1.0, 0.7])
    pred = gt + rng.normal(scale=0.22, size=gt.shape)
    pred = np.vstack([pred, rng.normal(size=(3, 2)) * np.array([1.0, 0.7])])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.scatter(gt[:, 0], gt[:, 1], s=70, color=GT, label="ground truth", zorder=3)
    ax.scatter(pred[:, 0], pred[:, 1], s=70, color=OURS, marker="^", label="prediction", zorder=3)
    for p in pred:
        d = np.linalg.norm(gt - p, axis=1)
        q = gt[d.argmin()]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=MUTED, lw=0.9, zorder=1)
    ax.set_title("Chamfer: nearest-neighbour distances")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.scatter(gt[:, 0], gt[:, 1], s=70, color=GT, zorder=3)
    ax.scatter(pred[:, 0], pred[:, 1], s=70, color=OURS, marker="^", zorder=3)
    tau = 0.35
    for q in gt[:5]:
        ax.add_patch(Circle((q[0], q[1]), tau, fill=False, color=GT, lw=1.0, alpha=0.6, linestyle=(0, (3, 2))))
    ax.set_title("F-score: matches within threshold τ")
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Reconstruction Metrics", fontsize=16, fontweight="bold", y=1.02)
    _save(fig, "metric_intuition.png")


# ----------------------------------------------------------------------------
# 11. baseline_comparison.png  (partial: real input + GT, model cols pending)
# ----------------------------------------------------------------------------
def fig_baseline_comparison(hero_dir):
    img = _view_image(hero_dir)
    pts = _load_points(hero_dir)
    fig = plt.figure(figsize=(14, 4.0))
    gs = fig.add_gridspec(1, 4, wspace=0.12)

    ax0 = fig.add_subplot(gs[0]); ax0.imshow(img); ax0.axis("off")
    ax0.set_title("Input view", fontsize=13)

    ax1 = fig.add_subplot(gs[1]); _placeholder(ax1, "2D Positional\n(pending)"); ax1.set_title("2D Positional", fontsize=13)
    ax2 = fig.add_subplot(gs[2]); _placeholder(ax2, "Geometry-Aware\n(pending)"); ax2.set_title("Geometry-Aware (ours)", fontsize=13)

    ax3 = fig.add_subplot(gs[3], projection="3d"); _scatter3d(ax3, pts); ax3.set_title("Ground truth", fontsize=13)

    fig.suptitle("Baseline vs. Geometry-Aware (qualitative)", fontsize=15, fontweight="bold", y=1.03)
    _save(fig, "baseline_comparison.png")


# ----------------------------------------------------------------------------
# 12 / 13. chamfer_plot.png, fscore_plot.png  (axis templates, no fake data)
# ----------------------------------------------------------------------------
def _results_plot(name, ylabel, title, better):
    ks = [1, 2, 4, 8, 16]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    series = ["2D Positional", "View-ID", "Camera Pose", "Geometry-Aware"]
    colors = [ACCENT, "#8a5cf6", "#e0892f", OURS]
    for s, c in zip(series, colors):
        ax.plot([], [], color=c, marker="o", label=s)
    ax.set_xticks(ks)
    ax.set_xlabel("Number of input views (K)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.text(0.5, 0.5, f"Pending results\n({better})", transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color=MUTED, style="italic")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, name)


def fig_chamfer_plot():
    _results_plot("chamfer_plot.png", "Chamfer distance", "Chamfer vs. Number of Views", "lower is better")


def fig_fscore_plot():
    _results_plot("fscore_plot.png", "F-score @ 1% bbox", "F-score vs. Number of Views", "higher is better")


# ----------------------------------------------------------------------------
# 14. qualitative_results.png  (real input + GT, model cols pending)
# ----------------------------------------------------------------------------
def fig_qualitative_results(rows):
    """rows: list of (label, input_PIL_image, gt_points)."""
    n = len(rows)
    fig = plt.figure(figsize=(13, 3.3 * n))
    gs = fig.add_gridspec(n, 4, wspace=0.1, hspace=0.16)
    cols = ["Input", "Baseline", "Ours", "Ground truth"]
    for r, (label, img, pts) in enumerate(rows):
        a0 = fig.add_subplot(gs[r, 0]); a0.imshow(img)
        a0.set_xticks([]); a0.set_yticks([])
        a0.set_ylabel(label, fontsize=12, fontweight="bold")
        a1 = fig.add_subplot(gs[r, 1]); _placeholder(a1, "pending")
        a2 = fig.add_subplot(gs[r, 2]); _placeholder(a2, "pending")
        a3 = fig.add_subplot(gs[r, 3], projection="3d"); _scatter3d(a3, pts)
        if r == 0:
            a0.set_title(cols[0], fontsize=13)
            a1.set_title(cols[1], fontsize=13)
            a2.set_title(cols[2], fontsize=13)
            a3.set_title(cols[3], fontsize=13)
    fig.suptitle("Qualitative Reconstructions", fontsize=16, fontweight="bold", y=1.0)
    _save(fig, "qualitative_results.png")


# ----------------------------------------------------------------------------
# 15. failure_cases.png  (real renders + expected failure annotations)
# ----------------------------------------------------------------------------
def fig_failure_cases(chair_img, airplane_img):
    cases = [
        (chair_img, "Thin structures (chair legs)"),
        (airplane_img, "Distorted wings / tail"),
        (chair_img, "Hidden / occluded parts"),
        (airplane_img, "Ambiguous single view"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (img, label) in zip(axes, cases):
        ax.imshow(img)
        ax.set_title(label, fontsize=11)
        ax.axis("off")
        ax.text(0.5, -0.08, "prediction pending", transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=MUTED, style="italic")
    fig.suptitle("Anticipated Failure Cases", fontsize=15, fontweight="bold", y=1.04)
    _save(fig, "failure_cases.png")


def main():
    chair = _first_chair_object()
    airplane_dir = ROOT / "data/modal_preview/objects" / AIRPLANE_DIR_NAME

    # Clean upright chair render reused across several figures.
    chair_mesh = _load_mesh(_chair_mesh_path(chair), upright=True)
    chair_img = _render_mesh_pil(chair_mesh, image_size=360)
    airplane_img = _view_image(airplane_dir)

    fig_task(airplane_dir)
    fig_patch_tokens(airplane_dir)
    fig_representation(chair)
    fig_category_distribution()
    fig_dataset_split()
    fig_mesh_normalization(chair)
    fig_camera_layout()
    fig_nested_views()
    fig_embedding_variants()
    fig_metric_intuition()
    fig_baseline_comparison(airplane_dir)
    fig_chamfer_plot()
    fig_fscore_plot()
    fig_qualitative_results(
        [
            ("airplane", airplane_img, _load_points(airplane_dir)),
            ("chair", chair_img, _load_points(chair)),
        ]
    )
    fig_failure_cases(chair_img, airplane_img)
    print("\nAll report figures written to", OUT)


if __name__ == "__main__":
    main()
