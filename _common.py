"""Shared utilities. Loads the active axes/models configs at import time.

Configs are looked up via env vars (with sensible defaults so the repo runs
out-of-the-box):
  PHASE_H_AXES        — path to alternate axes config (default: configs/axes.py)
  PHASE_H_MODELS      — path to alternate models config (default: configs/models.py)
  PHASE_H_DATA_ROOT   — root dir holding per-model subdirectories (default: ./data)
                        Each model's subdirectory is `<root>/<MODELS[tag]['path']>`.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent


def _load_config(env_var: str, default_relative: str):
    path = Path(os.environ.get(env_var, REPO_ROOT / default_relative))
    spec = importlib.util.spec_from_file_location(env_var.lower(), path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load config at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_axes_cfg = _load_config("PHASE_H_AXES", "configs/axes.py")
_models_cfg = _load_config("PHASE_H_MODELS", "configs/models.py")

AXIS_ORDER: list[str] = _axes_cfg.AXIS_ORDER
ANCHOR_AXES: dict[str, tuple[list[str], list[str]]] = _axes_cfg.ANCHOR_AXES
MODELS: dict[str, dict] = _models_cfg.MODELS

DATA_ROOT = Path(os.environ.get("PHASE_H_DATA_ROOT", REPO_ROOT / "data")).resolve()


def model_dir(tag: str) -> Path:
    """Return the absolute directory for model `tag` (using DATA_ROOT)."""
    if tag not in MODELS:
        raise KeyError(f"Unknown model tag {tag!r}; configured: {list(MODELS)}")
    return DATA_ROOT / MODELS[tag]["path"]


def _extract_tensor(obj) -> torch.Tensor:
    """Some role-vector .pt files store {'vector': tensor, ...}; return the tensor."""
    if isinstance(obj, dict):
        obj = obj.get("vector", obj.get("mean", next(iter(obj.values()))))
    return obj


def load_role_vectors(vec_dir: Path) -> dict[str, np.ndarray]:
    """Load every .pt file in vec_dir as {role_name: float32 ndarray}."""
    vectors = {}
    for f in sorted(vec_dir.glob("*.pt")):
        v = _extract_tensor(torch.load(f, map_location="cpu", weights_only=False))
        vectors[f.stem] = v.float().squeeze().numpy()
    if not vectors:
        raise FileNotFoundError(f"No .pt role vectors in {vec_dir}")
    return vectors


def load_role_matrix(model_path: Path, kind: str = "") -> tuple[np.ndarray, list[str]]:
    """Stack all role vectors from {model_path}/vectors{kind}/ into an (N, d) matrix.
    Returns (matrix, sorted role names)."""
    vectors = load_role_vectors(model_path / f"vectors{kind}")
    roles = sorted(vectors.keys())
    return np.stack([vectors[r] for r in roles]), roles


def load_default(model_path: Path) -> np.ndarray:
    """Load the default-Assistant activation vector for a model."""
    v = _extract_tensor(torch.load(model_path / "default.pt", map_location="cpu", weights_only=False))
    return v.float().squeeze().numpy()


def load_contrast(model_path: Path, axis_name: str) -> np.ndarray:
    """Load a contrast vector (e.g. 'v_assistant') from <model_path>/contrasts/<axis_name>.pt."""
    v = _extract_tensor(torch.load(model_path / "contrasts" / f"{axis_name}.pt", map_location="cpu", weights_only=False))
    return v.float().squeeze().numpy()


def z_score_default_on_axis(
    X: np.ndarray, default: np.ndarray, pos_idxs: Iterable[int], neg_idxs: Iterable[int]
) -> float:
    """Build axis v = mean(X[pos]) − mean(X[neg]); project default; z-score against role distribution.

    Returns NaN if the role-projection std is degenerate (< 1e-12).
    """
    pos_idxs = np.asarray(list(pos_idxs))
    neg_idxs = np.asarray(list(neg_idxs))
    v = X[pos_idxs].mean(axis=0) - X[neg_idxs].mean(axis=0)
    role_proj = X @ v
    sd = role_proj.std()
    if sd < 1e-12:
        return float("nan")
    return float((default @ v - role_proj.mean()) / sd)


def procrustes_orthogonal(X: np.ndarray, Y: np.ndarray) -> dict:
    """argmin_R ||X R - Y||_F  s.t.  R^T R = I.   X, Y: (N, K).

    Retained for the deferred procrustes.py module — not used by core scripts.
    """
    M = X.T @ Y
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    Xrot = X @ R
    resid = float(np.linalg.norm(Xrot - Y))
    rel_resid = resid / (float(np.linalg.norm(Y)) + 1e-12)
    cos_per_row = (Y * Xrot).sum(axis=1) / (
        np.linalg.norm(Y, axis=1) * np.linalg.norm(Xrot, axis=1) + 1e-12
    )
    cos_per_row = np.clip(cos_per_row, -1.0, 1.0)
    return {
        "R": R,
        "Xrot": Xrot,
        "resid_frob": resid,
        "rel_resid": rel_resid,
        "mean_row_rot_deg": float(np.degrees(np.arccos(cos_per_row)).mean()),
        "R_minus_I_frob": float(np.linalg.norm(R - np.eye(R.shape[0]))),
    }


def to_native(obj):
    """Recursively convert numpy/torch scalars and arrays to native Python for json.dump."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def polar_radar_setup(ax, axes_labels: list[str], r_min: float, r_max: float) -> list[float]:
    """Configure a matplotlib polar Axes for a radar plot. Returns angles per spoke."""
    K = len(axes_labels)
    angles = list(np.linspace(0, 2 * np.pi, K, endpoint=False))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlim(r_min, r_max)
    ax.set_rorigin(r_min - 0.05)
    tick_vals = sorted(set(
        [round(v, 1) for v in np.arange(np.ceil(r_min), r_max + 0.01, 1.0)] + [0.0]
    ))
    ax.set_rgrids(
        tick_vals,
        labels=[f"{t:+.0f}σ" if t != 0 else "0" for t in tick_vals],
        angle=22.5, fontsize=9, color="#444",
    )
    theta_full = np.linspace(0, 2 * np.pi, 200)
    ax.plot(theta_full, np.zeros_like(theta_full), color="#666", lw=1.0, ls="--", zorder=1.5)
    ax.set_xticks(angles)
    ax.set_xticklabels(axes_labels, fontsize=12)
    return angles
