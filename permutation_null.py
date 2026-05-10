#!/usr/bin/env python3
"""Permutation null: does each anchor axis carry concept-specific signal beyond
random 8+8 partitions, after removing PC1 of the role-only covariance?

Math note: residualizing against a=unit(default-mean(roles)) makes z-scores
identically zero (a·v_perp=0 by construction). We use PC1 of the role matrix
instead — ~0.92 correlated with the Assistant Axis (Lu 2026) but distinct from
it, so the centroid-to-default signal survives residualization.

For each axis: v_perp = unit(v − (v·pc1)·pc1); z-score default's projection onto
v_perp against role projections. Same random 8+8 indices applied to all models
per iter (paired). p (two-sided) = mean(|null| ≥ |real|).
"""
import argparse, json
from itertools import combinations
from pathlib import Path
import numpy as np

from _common import ANCHOR_AXES, MODELS, load_contrast, load_default, load_role_matrix, model_dir, to_native


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.full_like(v, np.nan)


def _z(X, default, v):
    """Default's z-score on direction v, against role projections."""
    role_proj = X @ v
    sd = role_proj.std()
    return float("nan") if sd < 1e-12 else float((default @ v - role_proj.mean()) / sd)


def _two_sided_p(observed, null_arr):
    null_arr = null_arr[~np.isnan(null_arr)]
    if null_arr.size == 0 or np.isnan(observed):
        return float("nan")
    return float((np.abs(null_arr) >= abs(observed)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iter", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/radar/permutation_null.json")
    args = ap.parse_args()

    # Load each model: X (276, d), default, and PC1 of role-only matrix.
    print("Loading models...")
    data = {}
    for tag in MODELS:
        X, roles = load_role_matrix(model_dir(tag))
        default = load_default(model_dir(tag))
        non_default = np.array([r != "default" for r in roles])
        X_roles = X[non_default]
        _, _, Vt = np.linalg.svd(X_roles - X_roles.mean(axis=0), full_matrices=False)
        pc1 = Vt[0]
        if pc1 @ (default - X_roles.mean(axis=0)) < 0:   # sign-align with default direction
            pc1 = -pc1
        a_lu = _unit(default - X_roles.mean(axis=0))
        data[tag] = {"X": X, "roles": roles, "default": default, "pc1": pc1}
        print(f"  {tag:>6}: X={X.shape}, cos(pc1, lu_axis)={pc1 @ a_lu:.3f}")

    tags = list(data)
    roles = data[tags[0]]["roles"]
    for t in tags[1:]:
        if data[t]["roles"] != roles:
            raise SystemExit(f"Role lists differ between {t!r} and {tags[0]!r}")

    all_anchors = {r for pos, neg in ANCHOR_AXES.values() for r in pos + neg}
    pool = np.array([i for i, r in enumerate(roles) if r not in all_anchors and r != "default"])
    print(f"  Neutral pool: {len(pool)} roles (275 char − {len(all_anchors)} anchors)\n")

    def resid_z(v, info):
        v_perp = _unit(v - (v @ info["pc1"]) * info["pc1"])
        return _z(info["X"], info["default"], v_perp)

    # Real residualized z-scores.
    real = {}
    print("Real residualized z (post-PC1):")
    for axis in ANCHOR_AXES:
        per_tag = {tag: resid_z(load_contrast(model_dir(tag), axis), info) for tag, info in data.items()}
        real[axis] = {"per_model": per_tag,
                      "gaps": {f"{a}-{b}": per_tag[a] - per_tag[b] for a, b in combinations(tags, 2)}}
        print(f"  {axis:>15}: " + "  ".join(f"{tag}={per_tag[tag]:+.3f}" for tag in tags))

    # Null distribution.
    print(f"\nSampling {args.n_iter} null partitions from {len(pool)}-role pool...")
    rng = np.random.default_rng(args.seed)
    null_z = {tag: np.empty(args.n_iter) for tag in tags}
    null_gaps = {f"{a}-{b}": np.empty(args.n_iter) for a, b in combinations(tags, 2)}
    for it in range(args.n_iter):
        idx = rng.choice(pool, size=16, replace=False)
        pos_idx, neg_idx = idx[:8], idx[8:]
        zs = {tag: resid_z(info["X"][pos_idx].mean(0) - info["X"][neg_idx].mean(0), info)
              for tag, info in data.items()}
        for tag in tags:
            null_z[tag][it] = zs[tag]
        for a, b in combinations(tags, 2):
            null_gaps[f"{a}-{b}"][it] = zs[a] - zs[b]

    # Summarize.
    summary = {
        "meta": {"n_iter": args.n_iter, "seed": args.seed, "anchors_per_pole": 8,
                 "pool_size": int(len(pool)), "models": tags, "axes": list(ANCHOR_AXES),
                 "residualization": "PC1 of role-only matrix (sign-aligned with default)"},
        "null_per_model": {tag: {"mean": float(null_z[tag].mean()), "std": float(null_z[tag].std()),
                                 "p2.5": float(np.percentile(null_z[tag], 2.5)),
                                 "p97.5": float(np.percentile(null_z[tag], 97.5))} for tag in tags},
        "axes": {},
    }
    print("\nPer-cell concept-specific significance:")
    for axis in ANCHOR_AXES:
        cell = {tag: {"real_z_resid": real[axis]["per_model"][tag],
                      "p_value": _two_sided_p(real[axis]["per_model"][tag], null_z[tag])} for tag in tags}
        gap = {f"{a}-{b}": {"real_gap": real[axis]["gaps"][f"{a}-{b}"],
                            "p_value": _two_sided_p(real[axis]["gaps"][f"{a}-{b}"], null_gaps[f"{a}-{b}"])}
               for a, b in combinations(tags, 2)}
        summary["axes"][axis] = {"per_model": cell, "gaps": gap}
        for tag in tags:
            r, p = cell[tag]["real_z_resid"], cell[tag]["p_value"]
            print(f"  {axis:>15} {tag:>6}: z_resid={r:+.3f}  p={p:.4f}  {'SIG' if p<0.05 else 'n.s.'}")

    print("\nPair-gap concept-specific significance:")
    for axis in ANCHOR_AXES:
        for pair, info in summary["axes"][axis]["gaps"].items():
            print(f"  {axis:>15} {pair:>10}: gap_resid={info['real_gap']:+.3f}  p={info['p_value']:.4f}  {'SIG' if info['p_value']<0.05 else 'n.s.'}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_native(summary), indent=2))
    np.savez(out_path.with_suffix(".npz"),
             **{f"null_z__{tag}": null_z[tag] for tag in tags},
             **{f"null_gap__{p}": null_gaps[p] for p in null_gaps})
    print(f"\nSaved: {out_path}\nSaved: {out_path.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
