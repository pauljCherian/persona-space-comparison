#!/usr/bin/env python3
"""Build contrast vectors for one or more models, then project all roles + z-score.

For each model directory:
  - Build v_assistant = default − mean(roles)
  - Build each anchor axis v_X = mean(pos roles) − mean(neg roles)
  - Project all roles onto each axis (raw scalars)
  - Z-score per axis within model

Outputs (per model_dir):
  contrasts/<axis>.pt           — float32 1-D tensor per axis
  projections/raw.pt            — {projections: (N, K), axes: [...]}
  projections/zscore.pt         — same shape, within-model z-scored
  projections/role_index.json   — {roles: [...], axes: [...]}

Usage:
  python compute_axes.py --model_dir phi-3.5-mini
  python compute_axes.py --all                       # uses configs/models.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch

from _common import (
    ANCHOR_AXES, AXIS_ORDER, DATA_ROOT, MODELS,
    _extract_tensor, load_default, model_dir,
)


def load_role_vectors_torch(vec_dir: Path) -> dict[str, torch.Tensor]:
    """Load .pt role vectors as 1-D float32 torch tensors keyed by filename stem."""
    return {
        f.stem: _extract_tensor(torch.load(f, map_location="cpu", weights_only=False)).float().squeeze()
        for f in sorted(vec_dir.glob("*.pt"))
    }


def build_contrasts(vectors: dict[str, torch.Tensor], default: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return {axis_name: 1-D contrast tensor}. v_assistant uses full role distribution."""
    out: dict[str, torch.Tensor] = {}
    role_keys = [k for k in vectors if k != "default"]
    if "v_assistant" in AXIS_ORDER:
        mean_all = torch.stack([vectors[k] for k in role_keys]).mean(dim=0)
        out["v_assistant"] = default - mean_all

    for axis, (pos, neg) in ANCHOR_AXES.items():
        pos_present = [vectors[a] for a in pos if a in vectors]
        neg_present = [vectors[a] for a in neg if a in vectors]
        if not pos_present or not neg_present:
            print(f"  WARNING: {axis} has empty pole — skipping")
            continue
        out[axis] = torch.stack(pos_present).mean(dim=0) - torch.stack(neg_present).mean(dim=0)
    return out


def project_and_zscore(
    role_vectors: dict[str, torch.Tensor],
    contrasts: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    role_names = sorted(role_vectors.keys())
    role_stack = torch.stack([role_vectors[r] for r in role_names])
    axes = [a for a in AXIS_ORDER if a in contrasts]
    contrast_stack = torch.stack([contrasts[a] for a in axes])

    raw = role_stack @ contrast_stack.T
    mean = raw.mean(dim=0, keepdim=True)
    std = raw.std(dim=0, keepdim=True).clamp(min=1e-8)
    z = (raw - mean) / std
    return raw, z, role_names, axes


def process_one(mdir: Path) -> None:
    if not mdir.exists():
        sys.exit(f"FATAL: {mdir} does not exist (run pipeline.sh first)")
    vec_dir = mdir / "vectors"
    if not vec_dir.exists():
        sys.exit(f"FATAL: {vec_dir} does not exist")

    print(f"\n=== {mdir.name} ===")
    vectors = load_role_vectors_torch(vec_dir)
    default = torch.from_numpy(load_default(mdir)).float()
    print(f"  loaded {len(vectors)} role vectors; default shape {tuple(default.shape)}")

    contrasts = build_contrasts(vectors, default)
    out_contrasts = mdir / "contrasts"
    out_contrasts.mkdir(parents=True, exist_ok=True)
    for axis, v in contrasts.items():
        torch.save(v, out_contrasts / f"{axis}.pt")
        print(f"  {axis:>15}: norm={v.norm().item():.3f}  → {out_contrasts}/{axis}.pt")

    raw, z, roles, axes = project_and_zscore(vectors, contrasts)
    out_proj = mdir / "projections"
    out_proj.mkdir(parents=True, exist_ok=True)
    torch.save({"projections": raw, "axes": axes}, out_proj / "raw.pt")
    torch.save({"projections": z, "axes": axes}, out_proj / "zscore.pt")
    (out_proj / "role_index.json").write_text(
        json.dumps({"roles": roles, "axes": axes}, indent=2)
    )
    print(f"  projections: shape {tuple(raw.shape)} → {out_proj}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model_dir", help="path under DATA_ROOT, or absolute path")
    g.add_argument("--all", action="store_true", help="process every model in configs/models.py")
    g.add_argument("--tag", help="model tag from configs/models.py")
    args = ap.parse_args()

    if args.all:
        for tag in MODELS:
            process_one(model_dir(tag))
    elif args.tag:
        process_one(model_dir(args.tag))
    else:
        p = Path(args.model_dir)
        process_one(p if p.is_absolute() else DATA_ROOT / p)


if __name__ == "__main__":
    main()
