#!/usr/bin/env python3
"""Paired anchor bootstrap on the default-Assistant z-score per (model × axis).
Renders the radar plot (with optional CI bars).

For each axis, resample its 8+8 anchors with replacement; the SAME resampled
indices are used for ALL models so per-axis variability is shared and the
between-model gap CI is conservative (paired bootstrap).

Outputs (under <out_dir>):
  default_z_scores.npz          — point estimates per (model, axis)
  default_bootstrap_ci.json     — per-(model, axis) CIs + per-pair gap CIs
  default_bootstrap_arrays.npz  — raw bootstrap arrays
  three_model_radar.png         — radar with point-estimate polygons
  three_model_radar_zoom.png    — same, tight zoom
  (with --ci, also produces)
  three_model_radar_with_ci.png        — radar with bootstrap CI bars
  three_model_radar_with_ci_zoom.png   — same, tight zoom

Usage:
  python bootstrap_radar.py --n_boot 2000
  python bootstrap_radar.py --n_boot 200 --no-bootstrap     # just the plain radar
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import (
    ANCHOR_AXES, AXIS_ORDER, MODELS,
    load_contrast, load_default, load_role_matrix, model_dir, polar_radar_setup, to_native,
    z_score_default_on_axis,
)


def load_all_models() -> dict[str, dict]:
    data = {}
    need_v_assistant = "v_assistant" in AXIS_ORDER
    for tag, info in MODELS.items():
        mdir = model_dir(tag)
        X, roles = load_role_matrix(mdir)
        default = load_default(mdir)
        v_asst = None
        if need_v_assistant:
            v_asst_path = mdir / "contrasts" / "v_assistant.pt"
            if not v_asst_path.exists():
                raise SystemExit(f"FATAL: {v_asst_path} not found — run `compute_axes.py --all` before bootstrap_radar.py")
            v_asst = load_contrast(mdir, "v_assistant")
        data[tag] = {
            "X": X, "roles": roles, "default": default,
            "v_assistant": v_asst,
            "role_to_idx": {r: i for i, r in enumerate(roles)},
            "color": info["color"], "display": info["display"],
        }
        print(f"  {tag:>6}: X={X.shape}, default={default.shape}")
    tags = list(data)
    for t in tags[1:]:
        if data[t]["roles"] != data[tags[0]]["roles"]:
            raise SystemExit(f"Role list for {t!r} differs from {tags[0]!r} — paired bootstrap requires identical role lists")
    return data


def point_estimates(data: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Default-on-axis z-score per model per axis (no resampling)."""
    out: dict[str, dict[str, float]] = {tag: {} for tag in data}
    for axis in AXIS_ORDER:
        if axis == "v_assistant":
            for tag, info in data.items():
                v = info["v_assistant"]  # loaded from contrasts/v_assistant.pt; built with default excluded from role-mean (compute_axes.py:44-48)
                role_proj = info["X"] @ v
                sd = role_proj.std()
                out[tag][axis] = float("nan") if sd < 1e-12 else float((info["default"] @ v - role_proj.mean()) / sd)
        elif axis in ANCHOR_AXES:
            pos, neg = ANCHOR_AXES[axis]
            for tag, info in data.items():
                pos_idx = [info["role_to_idx"][r] for r in pos]
                neg_idx = [info["role_to_idx"][r] for r in neg]
                out[tag][axis] = z_score_default_on_axis(info["X"], info["default"], pos_idx, neg_idx)
    return out


def paired_bootstrap(data: dict[str, dict], n_boot: int, seed: int) -> tuple[dict, dict]:
    """For each anchor axis, resample (pos, neg) anchor indices with replacement;
    compute per-model default z-score and per-pair gap. Returns (boot, gaps)."""
    rng = np.random.default_rng(seed)
    tags = list(data)
    boot: dict[str, dict[str, list[float]]] = {ax: {tag: [] for tag in tags} for ax in ANCHOR_AXES}
    gaps: dict[str, dict[str, list[float]]] = {
        ax: {f"{a}-{b}": [] for a, b in combinations(tags, 2)} for ax in ANCHOR_AXES
    }
    for _ in range(n_boot):
        for axis, (pos, neg) in ANCHOR_AXES.items():
            pos_resample = rng.integers(0, len(pos), size=len(pos))
            neg_resample = rng.integers(0, len(neg), size=len(neg))
            zs: dict[str, float] = {}
            for tag, info in data.items():
                pos_idx = [info["role_to_idx"][pos[i]] for i in pos_resample]
                neg_idx = [info["role_to_idx"][neg[i]] for i in neg_resample]
                zs[tag] = z_score_default_on_axis(info["X"], info["default"], pos_idx, neg_idx)
                boot[axis][tag].append(zs[tag])
            for a, b in combinations(tags, 2):
                gaps[axis][f"{a}-{b}"].append(zs[a] - zs[b])
    return boot, gaps


def _ci_summary(arr: list[float]) -> dict:
    a = np.asarray(arr, float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {k: float("nan") for k in ("mean", "median", "std", "lo", "hi", "frac_neg", "frac_pos")}
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "lo": float(np.percentile(a, 2.5)),
        "hi": float(np.percentile(a, 97.5)),
        "frac_neg": float((a < 0).mean()),
        "frac_pos": float((a > 0).mean()),
    }


def summarize(point: dict, boot: dict, gaps: dict, tags: list[str]) -> dict:
    summary: dict = {"axes": {}, "gaps": {}}
    for ax in ANCHOR_AXES:
        summary["axes"][ax] = {}
        for tag in tags:
            c = _ci_summary(boot[ax][tag])
            sign_stable = c["frac_neg"] > 0.95 or c["frac_pos"] > 0.95
            summary["axes"][ax][tag] = {
                "point": point[tag][ax],
                "boot_mean": c["mean"], "boot_median": c["median"], "boot_std": c["std"],
                "ci95_lo": c["lo"], "ci95_hi": c["hi"],
                "frac_negative": c["frac_neg"], "frac_positive": c["frac_pos"],
                "sign_stable_95pct": bool(sign_stable),
            }
        summary["gaps"][ax] = {}
        for a, b in combinations(tags, 2):
            pair = f"{a}-{b}"
            point_gap = point[a][ax] - point[b][ax]
            c = _ci_summary(gaps[ax][pair])
            excludes = (c["lo"] > 0) or (c["hi"] < 0) if not np.isnan(c["lo"]) else False
            summary["gaps"][ax][pair] = {
                "point": point_gap,
                "boot_mean": c["mean"], "boot_median": c["median"], "boot_std": c["std"],
                "ci95_lo": c["lo"], "ci95_hi": c["hi"],
                "ci95_excludes_zero": bool(excludes),
            }
    return summary


def render_radar(
    point: dict,
    summary: dict | None,
    tags: list[str],
    displays: dict[str, str],
    colors: dict[str, str],
    out_path: Path,
    title: str,
    r_min: float | None = None,
    r_max: float | None = None,
) -> None:
    """Render the radar. If `summary` is provided, overlay anchor-bootstrap CI bars."""
    z_arr = np.array([[point[tag][ax] for ax in AXIS_ORDER] for tag in tags])
    lo, hi = z_arr.min(), z_arr.max()
    if summary is not None:
        for ax_name, per_model in summary["axes"].items():
            for tag in tags:
                lo = min(lo, per_model[tag]["ci95_lo"])
                hi = max(hi, per_model[tag]["ci95_hi"])
    if r_min is None:
        r_min = float(np.floor(lo * 2) / 2 - 0.5)
    if r_max is None:
        r_max = float(np.ceil(hi * 2) / 2 + 0.5)

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    label_axes = [a.replace("v_", "") for a in AXIS_ORDER]
    angles = polar_radar_setup(ax, label_axes, r_min, r_max)

    if summary is not None:
        offsets = np.linspace(-2.0, 2.0, len(tags))
        for tag, off in zip(tags, offsets):
            offset = np.radians(off)
            cap_w = np.radians(1.0)
            for i, ax_name in enumerate(AXIS_ORDER):
                if ax_name not in summary["axes"]:
                    continue
                lo_b = summary["axes"][ax_name][tag]["ci95_lo"]
                hi_b = summary["axes"][ax_name][tag]["ci95_hi"]
                a = angles[i] + offset
                ax.plot([a, a], [lo_b, hi_b], color=colors[tag], lw=2.5, alpha=0.55,
                        solid_capstyle="butt", zorder=3)
                ax.plot([a - cap_w, a + cap_w], [lo_b, lo_b], color=colors[tag], lw=1.2, alpha=0.7, zorder=3)
                ax.plot([a - cap_w, a + cap_w], [hi_b, hi_b], color=colors[tag], lw=1.2, alpha=0.7, zorder=3)

    angles_closed = angles + [angles[0]]
    for tag in tags:
        vals = [point[tag][a] for a in AXIS_ORDER]
        vals_closed = vals + [vals[0]]
        ax.plot(angles_closed, vals_closed, "-", lw=2.0, color=colors[tag],
                label=displays[tag], zorder=4)
        ax.fill(angles_closed, vals_closed, color=colors[tag], alpha=0.08, zorder=2)
        for a, v in zip(angles, vals):
            ax.plot(a, v, "o", color=colors[tag], ms=5, zorder=5)

    ax.set_title(title, pad=24, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.10), fontsize=10, frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="out/radar")
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="Skip the bootstrap; produce only the plain radar from point estimates.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    data = load_all_models()
    tags = list(data)
    displays = {tag: data[tag]["display"] for tag in tags}
    colors = {tag: data[tag]["color"] for tag in tags}

    print("\nPoint estimates:")
    point = point_estimates(data)
    label_w = max(len(a) for a in AXIS_ORDER)
    for ax in AXIS_ORDER:
        line = f"  {ax:>{label_w}}: " + "  ".join(f"{tag}={point[tag][ax]:+.3f}" for tag in tags)
        print(line)

    np.savez(out_dir / "default_z_scores.npz",
             axes=np.array(AXIS_ORDER),
             **{tag: np.array([point[tag][ax] for ax in AXIS_ORDER]) for tag in tags})
    print(f"  saved: {out_dir / 'default_z_scores.npz'}")

    if args.no_bootstrap:
        print("\nRendering plain radar (no bootstrap)...")
        render_radar(point, None, tags, displays, colors,
                     out_dir / "three_model_radar.png",
                     "Default Assistant persona profile — point estimates")
        render_radar(point, None, tags, displays, colors,
                     out_dir / "three_model_radar_zoom.png",
                     "Default Assistant persona profile — point estimates",
                     r_min=-1.4, r_max=2.1)
        return

    print(f"\nPaired anchor bootstrap (B={args.n_boot}, seed={args.seed})...")
    boot, gaps = paired_bootstrap(data, args.n_boot, args.seed)
    summary = summarize(point, boot, gaps, tags)
    summary["meta"] = {
        "n_boot": args.n_boot, "seed": args.seed,
        "models": tags, "axes": list(ANCHOR_AXES),
        "pairs": [f"{a}-{b}" for a, b in combinations(tags, 2)],
    }
    (out_dir / "default_bootstrap_ci.json").write_text(json.dumps(to_native(summary), indent=2))
    print(f"  saved: {out_dir / 'default_bootstrap_ci.json'}")

    np.savez(
        out_dir / "default_bootstrap_arrays.npz",
        **{f"{ax}__{tag}": np.array(boot[ax][tag]) for ax in ANCHOR_AXES for tag in tags},
        **{f"{ax}__GAP_{a}-{b}": np.array(gaps[ax][f"{a}-{b}"])
           for ax in ANCHOR_AXES for a, b in combinations(tags, 2)},
    )
    print(f"  saved: {out_dir / 'default_bootstrap_arrays.npz'}")

    print("\nPer-model sign-stability:")
    for ax in ANCHOR_AXES:
        for tag in tags:
            s = summary["axes"][ax][tag]
            stable = "stable" if s["sign_stable_95pct"] else "noise"
            print(f"  {ax:>15} {tag:>6}: point={s['point']:+.3f}  "
                  f"CI=[{s['ci95_lo']:+.3f}, {s['ci95_hi']:+.3f}]  ({stable})")

    print("\nPair-gap robustness:")
    for ax in ANCHOR_AXES:
        for pair, info in summary["gaps"][ax].items():
            tag = "EXCLUDES 0" if info["ci95_excludes_zero"] else "spans 0"
            print(f"  {ax:>15} {pair:>10}: gap={info['point']:+.3f}  "
                  f"CI=[{info['ci95_lo']:+.3f}, {info['ci95_hi']:+.3f}]  ({tag})")

    print("\nRendering radars...")
    render_radar(point, None, tags, displays, colors,
                 out_dir / "three_model_radar.png",
                 "Default Assistant persona profile (z-score within each model)")
    render_radar(point, None, tags, displays, colors,
                 out_dir / "three_model_radar_zoom.png",
                 "Default Assistant persona profile — zoomed",
                 r_min=-1.4, r_max=2.1)
    render_radar(point, summary, tags, displays, colors,
                 out_dir / "three_model_radar_with_ci.png",
                 f"Default Assistant — anchor-bootstrap 95% CIs (B={args.n_boot})")
    render_radar(point, summary, tags, displays, colors,
                 out_dir / "three_model_radar_with_ci_zoom.png",
                 f"Default Assistant — anchor-bootstrap 95% CIs (B={args.n_boot})",
                 r_min=-1.4, r_max=2.1)


if __name__ == "__main__":
    main()
