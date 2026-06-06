"""Mesh loading, normalization, and surface point sampling."""

from __future__ import annotations

import numpy as np
import trimesh


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center at origin and scale vertices to fit inside the unit sphere."""
    mesh = mesh.copy()
    vertices = mesh.vertices.astype(np.float64)
    centroid = vertices.mean(axis=0)
    vertices -= centroid
    radii = np.linalg.norm(vertices, axis=1)
    max_radius = float(radii.max())
    if max_radius > 0:
        vertices /= max_radius
    mesh.vertices = vertices.astype(np.float32)
    return mesh


def sample_surface_points(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """Sample points uniformly over mesh surface area."""
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points.astype(np.float32)


def load_mesh(mesh_path: str) -> trimesh.Trimesh:
    """Load an OBJ mesh as a single Trimesh."""
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(f"No mesh geometry in {mesh_path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh from {mesh_path}")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Empty mesh: {mesh_path}")
    return loaded


def mesh_to_pointcloud(mesh_path: str, num_points: int = 2048) -> np.ndarray:
    """Load mesh, normalize, sample surface points. Returns (N, 3) float32."""
    mesh = load_mesh(mesh_path)
    mesh = normalize_mesh(mesh)
    return sample_surface_points(mesh, num_points)
