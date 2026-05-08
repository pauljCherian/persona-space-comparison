#!/usr/bin/env python3
"""Pre-flight check. Validates env, configs, and data root before running pipeline/analysis.

Exit 0 → ready; exit 1 → fix the reported issues first.
"""
import os
import sys
from pathlib import Path

from _common import ANCHOR_AXES, AXIS_ORDER, DATA_ROOT, MODELS, REPO_ROOT


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    print("=== persona-space-comparison pre-flight ===\n")
    fails = 0

    print("Configs:")
    fails += not check("AXIS_ORDER non-empty", len(AXIS_ORDER) > 0, f"{len(AXIS_ORDER)} axes")
    fails += not check("ANCHOR_AXES non-empty", len(ANCHOR_AXES) > 0, f"{len(ANCHOR_AXES)} anchor axes")
    fails += not check("'v_assistant' in AXIS_ORDER", "v_assistant" in AXIS_ORDER)
    for axis in ANCHOR_AXES:
        fails += not check(f"  {axis} in AXIS_ORDER", axis in AXIS_ORDER)
        pos, neg = ANCHOR_AXES[axis]
        fails += not check(f"  {axis} pos+neg roles non-empty",
                          len(pos) > 0 and len(neg) > 0,
                          f"{len(pos)} pos, {len(neg)} neg")
    all_anchor_roles = [r for pos, neg in ANCHOR_AXES.values() for r in pos + neg]
    fails += not check("anchor roles unique across axes",
                      len(all_anchor_roles) == len(set(all_anchor_roles)),
                      f"{len(all_anchor_roles)} total, {len(set(all_anchor_roles))} unique")

    print("\nData root:")
    fails += not check(f"PHASE_H_DATA_ROOT exists ({DATA_ROOT})", DATA_ROOT.exists(),
                      "" if DATA_ROOT.exists() else "set PHASE_H_DATA_ROOT or `mkdir -p data/`")

    print("\nModels:")
    fails += not check("MODELS non-empty", len(MODELS) > 0, f"{len(MODELS)} models")
    for tag, info in MODELS.items():
        for key in ("path", "layer", "hidden_dim", "color", "display"):
            fails += not check(f"  {tag}.{key} present", key in info)
        if "path" in info:
            mdir = DATA_ROOT / info["path"]
            present = mdir.exists()
            fails += not check(f"  {tag} model_dir exists",
                              present,
                              str(mdir) if present else f"missing {mdir} — run pipeline.sh first")

    print("\nVendored library:")
    try:
        import assistant_axis  # noqa: F401
        check("assistant_axis importable", True)
    except ImportError as e:
        check("assistant_axis importable", False, str(e))
        fails += 1

    print("\nGPU (only required for pipeline.sh, not analysis):")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        check(f"CUDA available", cuda_ok,
              f"{torch.cuda.device_count()} device(s)" if cuda_ok else "OK to skip if running analysis only")
    except Exception as e:
        check("torch importable", False, str(e))
        fails += 1

    print()
    if fails:
        print(f"FAIL: {fails} issue(s). Fix before running pipeline / analysis.")
        return 1
    print("PASS — ready to run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
