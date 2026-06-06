"""Figures and HTML gallery for the master / processed ShapeNet dataset."""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.data.camera import SUPPORTED_VIEW_COUNTS
from src.data.master_splits import MAIN_CATEGORIES

CATEGORY_COLORS = {
    "03001627": "#059669",
    "04379243": "#7c3aed",
    "04256520": "#be185d",
    "02958343": "#dc2626",
    "02691156": "#1d4ed8",
    "03642806": "#d97706",
    "02933112": "#0891b2",
}


def _list_rendered_samples(objects_dir: Path) -> list[str]:
    if not objects_dir.is_dir():
        return []
    ids = []
    for d in sorted(objects_dir.iterdir()):
        if d.is_dir() and (d / "gt_points.npy").is_file():
            ids.append(d.name)
    return ids


def _entry_for_sample(manifest: dict | None, sample_id: str) -> dict | None:
    if manifest is None:
        return None
    for e in manifest.get("entries", []):
        if e.get("sample_id") == sample_id:
            return e
    return None


def _category_name(entry: dict | None, sample_id: str) -> str:
    if entry:
        return entry.get("category_name") or entry.get("synset_id", sample_id)
    if "_" in sample_id:
        return sample_id.split("_", 1)[0]
    return sample_id


class DatasetVisualizer:
    def __init__(
        self,
        dataset_dir: Path,
        output_dir: Path,
        dpi: int = 150,
    ) -> None:
        self.dataset_dir = dataset_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.objects_dir = self.dataset_dir / "objects"
        self.dpi = dpi
        self.output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = self.dataset_dir / "master_manifest.json"
        self.manifest = None
        if manifest_path.is_file():
            with manifest_path.open() as f:
                self.manifest = json.load(f)

    def available_samples(self) -> list[str]:
        return _list_rendered_samples(self.objects_dir)

    def run(
        self,
        sample_ids: list[str] | None = None,
        max_samples: int | None = None,
        make_gallery: bool = True,
    ) -> list[Path]:
        samples = sample_ids or self.available_samples()
        if not samples:
            raise FileNotFoundError(
                f"No rendered objects under {self.objects_dir}. "
                "Run: python scripts/build_master_dataset.py --allow-partial"
            )
        if max_samples is not None:
            samples = samples[:max_samples]

        saved: list[Path] = []
        saved.append(self.plot_overview(samples))
        saved.append(self.plot_pointclouds_available(samples))
        saved.append(self.plot_pointclouds_by_main_category())

        for sid in samples:
            p = self.plot_sample_panel(sid)
            if p:
                saved.append(p)
            p = self.plot_all_renders_grid(sid)
            if p:
                saved.append(p)
            p = self.plot_input_view_strips(sid)
            if p:
                saved.append(p)

        if make_gallery:
            gallery = self.write_html_gallery(samples)
            saved.append(gallery)

        return [p for p in saved if p is not None]

    def plot_overview(self, samples: list[str]) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        split_counts: dict[str, int] = {}
        splits_dir = self.dataset_dir / "splits"
        if splits_dir.is_dir():
            for p in sorted(splits_dir.glob("*.json")):
                with p.open() as f:
                    data = json.load(f)
                split_counts[data.get("split", p.stem)] = len(data.get("samples", []))

        if split_counts:
            axes[0].bar(split_counts.keys(), split_counts.values(), color="#3b82f6")
            axes[0].set_ylabel("objects in manifest")
            axes[0].set_title("Split sizes (manifest)")
            axes[0].tick_params(axis="x", rotation=20)
        else:
            axes[0].text(0.5, 0.5, "No split manifests", ha="center", va="center")
            axes[0].set_axis_off()

        rendered = len(self.available_samples())
        axes[1].bar(
            ["rendered on disk", "this run"],
            [rendered, len(samples)],
            color=["#10b981", "#6366f1"],
        )
        axes[1].set_title("Objects with GT + views")
        for i, v in enumerate([rendered, len(samples)]):
            axes[1].text(i, v + 0.3, str(v), ha="center", fontsize=11)

        title = f"Dataset: {self.dataset_dir.name}"
        if self.manifest:
            title += f"  |  manifest total: {self.manifest.get('total_objects', '?')}"
        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        out = self.output_dir / "00_overview.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_pointclouds_available(self, samples: list[str]) -> Path:
        n = len(samples)
        cols = min(4, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows), subplot_kw={"projection": "3d"})
        axes_flat = np.atleast_1d(axes).flatten()

        for ax, sid in zip(axes_flat, samples):
            pts = np.load(self.objects_dir / sid / "gt_points.npy")
            entry = _entry_for_sample(self.manifest, sid)
            name = _category_name(entry, sid)
            syn = entry.get("synset_id", "") if entry else ""
            color = CATEGORY_COLORS.get(syn, "#2563eb")
            self._scatter_pc(ax, pts, color)
            split = entry.get("split", "?") if entry else "?"
            ax.set_title(f"{name}\n{sid}\n({split})", fontsize=8)
            ax.view_init(elev=22, azim=-58)
            ax.set_axis_off()

        for ax in axes_flat[len(samples) :]:
            ax.set_axis_off()

        fig.suptitle("Ground-truth point clouds (2048 pts)", fontsize=12)
        plt.tight_layout()
        out = self.output_dir / "01_pointclouds_all_samples.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_pointclouds_by_main_category(self) -> Path:
        """One sample per main category when available."""
        by_synset: dict[str, str] = {}
        for sid in self.available_samples():
            entry = _entry_for_sample(self.manifest, sid)
            syn = entry["synset_id"] if entry else sid.split("_", 1)[0]
            if syn not in by_synset:
                by_synset[syn] = sid

        fig, axes = plt.subplots(2, 4, figsize=(14, 7), subplot_kw={"projection": "3d"})
        name_map = {c["synset_id"]: c["name"] for c in MAIN_CATEGORIES}

        for ax, cat in zip(axes.flatten(), MAIN_CATEGORIES):
            sid = by_synset.get(cat["synset_id"])
            if sid is None:
                ax.set_title(f"{cat['name']}\n(not downloaded)", fontsize=9)
                ax.set_axis_off()
                continue
            pts = np.load(self.objects_dir / sid / "gt_points.npy")
            self._scatter_pc(ax, pts, CATEGORY_COLORS.get(cat["synset_id"], "#2563eb"))
            ax.set_title(cat["name"], fontsize=10)
            ax.view_init(elev=22, azim=-58)
            ax.set_axis_off()

        axes.flatten()[-1].set_axis_off()
        fig.suptitle("Point clouds by main category", fontsize=12)
        plt.tight_layout()
        out = self.output_dir / "02_pointclouds_by_category.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_sample_panel(self, sample_id: str) -> Path | None:
        sample_dir = self.objects_dir / sample_id
        gt_path = sample_dir / "gt_points.npy"
        if not gt_path.is_file():
            return None

        entry = _entry_for_sample(self.manifest, sample_id)
        name = _category_name(entry, sample_id)
        color = CATEGORY_COLORS.get(entry["synset_id"] if entry else "", "#2563eb")

        fig = plt.figure(figsize=(14, 5))
        ax_pc = fig.add_subplot(121, projection="3d")
        pts = np.load(gt_path)
        self._scatter_pc(ax_pc, pts, color)
        ax_pc.set_title(f"GT point cloud — {name}", fontsize=11)
        ax_pc.view_init(elev=25, azim=-55)
        ax_pc.set_axis_off()

        ax_img = fig.add_subplot(122)
        strip = self._render_strip(sample_dir, num_views=8)
        if strip is None:
            ax_img.set_title("No renders")
            ax_img.axis("off")
        else:
            ax_img.imshow(strip)
            ax_img.set_title("8 evenly spaced RGB views", fontsize=11)
            ax_img.axis("off")

        split = entry.get("split", "") if entry else ""
        fig.suptitle(f"{sample_id}  {split}", fontsize=12, y=1.02)
        plt.tight_layout()
        out = self.output_dir / f"sample_{sample_id}_panel.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_all_renders_grid(self, sample_id: str) -> Path | None:
        sample_dir = self.objects_dir / sample_id
        views = sorted(sample_dir.glob("view_*.png"))
        views = [v for v in views if "_pose" not in v.name]
        if not views:
            return None

        n = len(views)
        cols = 8 if n >= 16 else 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
        axes_flat = np.atleast_1d(axes).flatten()

        for ax, vf in zip(axes_flat, views):
            ax.imshow(Image.open(vf).convert("RGB"))
            ax.set_title(vf.stem.replace("view_", ""), fontsize=7)
            ax.axis("off")
        for ax in axes_flat[len(views) :]:
            ax.axis("off")

        fig.suptitle(f"All renders ({n} views) — {sample_id}", fontsize=12)
        plt.tight_layout()
        out = self.output_dir / f"sample_{sample_id}_renders_grid.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_input_view_strips(self, sample_id: str) -> Path | None:
        sample_dir = self.objects_dir / sample_id
        if not (sample_dir / "view_00.png").is_file():
            return None

        view_counts = list(SUPPORTED_VIEW_COUNTS)
        fig, axes = plt.subplots(len(view_counts), 1, figsize=(12, 2.3 * len(view_counts)))

        for ax, k in zip(axes, view_counts):
            strip = self._render_strip(sample_dir, num_views=k)
            if strip is None:
                ax.axis("off")
                continue
            ax.imshow(strip)
            ax.set_title(f"{k} input view{'s' if k > 1 else ''}", fontsize=10, loc="left")
            ax.axis("off")

        fig.suptitle(f"Input-view settings — {sample_id}", fontsize=12)
        plt.tight_layout()
        out = self.output_dir / f"sample_{sample_id}_input_views.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def write_html_gallery(self, samples: list[str]) -> Path:
        gallery_dir = self.output_dir / "gallery"
        gallery_dir.mkdir(exist_ok=True)

        cards = []
        for sid in samples:
            entry = _entry_for_sample(self.manifest, sid)
            name = _category_name(entry, sid)
            split = entry.get("split", "") if entry else ""
            panel = f"../sample_{sid}_panel.png"
            grid = f"../sample_{sid}_renders_grid.png"
            cards.append(
                f"""
                <section class="card">
                  <h2>{name} <span class="muted">{sid}</span></h2>
                  <p class="meta">split: {split or "n/a"}</p>
                  <img src="{panel}" alt="panel {sid}" />
                  <img src="{grid}" alt="renders {sid}" class="grid-img" />
                </section>
                """
            )

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>ShapeNet dataset gallery</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; }}
    h1 {{ font-size: 1.4rem; }}
    .card {{ background: #1e293b; border-radius: 12px; padding: 16px; margin-bottom: 24px; }}
    .card img {{ max-width: 100%; border-radius: 8px; margin-top: 8px; }}
    .grid-img {{ margin-top: 12px; }}
    .muted {{ color: #94a3b8; font-weight: normal; font-size: 0.9rem; }}
    .meta {{ color: #94a3b8; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <h1>Dataset gallery — {self.dataset_dir.name}</h1>
  <p>{len(samples)} object(s) visualized. Figures also saved as PNG next to this folder.</p>
  {"".join(cards)}
</body>
</html>
"""
        index = gallery_dir / "index.html"
        index.write_text(html)
        return index

    def _render_strip(self, sample_dir: Path, num_views: int) -> np.ndarray | None:
        from src.data.camera import eval_view_indices

        try:
            indices = eval_view_indices(num_views)
        except ValueError:
            view_files = sorted(
                p for p in sample_dir.glob("view_*.png") if "_pose" not in p.name
            )
            if not view_files:
                return None
            n = min(num_views, len(view_files))
            indices = np.linspace(0, len(view_files) - 1, n, dtype=int)
            imgs = [
                np.asarray(Image.open(view_files[i]).convert("RGB")) for i in indices
            ]
            return np.concatenate(imgs, axis=1)

        imgs = []
        for vi in indices:
            path = sample_dir / f"view_{vi:02d}.png"
            if not path.is_file():
                return None
            imgs.append(np.asarray(Image.open(path).convert("RGB")))
        return np.concatenate(imgs, axis=1) if imgs else None

    @staticmethod
    def _scatter_pc(ax, points: np.ndarray, color: str) -> None:
        if len(points) > 1500:
            idx = np.linspace(0, len(points) - 1, 1500, dtype=int)
            points = points[idx]
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=color, s=1.0, alpha=0.7, linewidths=0,
        )
        mins, maxs = points.min(0), points.max(0)
        c = (mins + maxs) / 2
        r = max((maxs - mins).max() / 2, 1e-3) * 1.08
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)

    def open_gallery(self) -> None:
        index = self.output_dir / "gallery" / "index.html"
        if index.is_file():
            webbrowser.open(index.as_uri())
