# persona-space-comparison

Minimal pipeline for comparing language models on a small panel of contrast-vector axes derived from the [assistant-axis](https://github.com/safety-research/assistant-axis) methodology (Lu et al., 2026), with anchor-bootstrap CIs on the headline radar.

**What this does** — for each model:
1. Run the assistant-axis pipeline (no judge — see *Skipping the judge* below).
2. Build five contrast vectors per model: `v_assistant` (Lu canonical) plus four anchor-pair contrasts (`v_benevolence`, `v_authority`, `v_humor`, `v_critic`).
3. Project all 275 roles onto the contrasts; z-score within model; render a paired-anchor-bootstrap radar with 95% CIs.

That's the whole repo. ~1,020 lines of code. Two researcher knobs (axes panel + model set) live in `configs/`.

## Install

### Option A — share parent project's venv (fastest for development on this machine)

The parent project at `/jumbo/lisp/fl1/assistant-axis-abliteration/` already has the assistant-axis library editable-installed in its venv. Use it directly:

```bash
git clone https://github.com/pjcherian7/persona-space-comparison.git
cd persona-space-comparison
export PYTHON=/jumbo/lisp/fl1/assistant-axis-abliteration/.venv/bin/python
```

That's it — no fresh install needed. Run scripts via `$PYTHON script.py`.

### Option B — fresh standalone install (for distribution / clean repro)

```bash
git clone https://github.com/pjcherian7/persona-space-comparison.git
cd persona-space-comparison
python -m venv .venv && source .venv/bin/activate
pip install -e git+https://github.com/safety-research/assistant-axis.git#egg=assistant-axis
pip install -r requirements.txt
```

The `pipeline.sh` script auto-discovers the assistant-axis library location, so either path works.

## Configuration: the two researcher knobs

Both live in `configs/` as plain Python files. Edit in place to fork; or copy + override via env var.

### `configs/axes.py` — the contrast-axis panel
```python
AXIS_ORDER = ["v_assistant", "v_benevolence", "v_authority", "v_humor", "v_critic"]

ANCHOR_AXES: dict[str, tuple[list[str], list[str]]] = {
    "v_benevolence": (
        ["counselor", "parent", ...],   # positive pole
        ["criminal", "saboteur", ...],  # negative pole
    ),
    ...
}
```

`v_assistant` is special (computed at runtime as default − mean(roles)) — it appears in `AXIS_ORDER` but not `ANCHOR_AXES`.

To run with a different panel:
```bash
cp configs/axes.py configs/no_critic.py    # then edit configs/no_critic.py
PHASE_H_AXES=configs/no_critic.py python compute_axes.py --all
```

### `configs/models.py` — the model set
```python
MODELS: dict[str, dict] = {
    "phi": {
        "path": "phi-3.5-mini",   # subdir under PHASE_H_DATA_ROOT
        "layer": 16,              # extraction layer (Lu canonical: N/2)
        "hidden_dim": 3072,
        "color": "#1f77b4",
        "display": "Phi-3.5-mini",
    },
    ...
}
```

To run with a different model set:
```bash
cp configs/models.py configs/abliterated_set.py    # edit
PHASE_H_MODELS=configs/abliterated_set.py python bootstrap_radar.py
```

### Where data lives — `PHASE_H_DATA_ROOT`

The repo ships **code only**. Per-model artifacts are read from / written to `$PHASE_H_DATA_ROOT/<MODELS[tag]['path']>/`.

```bash
export PHASE_H_DATA_ROOT=/path/to/your/data            # default: ./data
```

Layout per model:
```
$PHASE_H_DATA_ROOT/phi-3.5-mini/
  responses/        # produced by pipeline.sh step 1 (vLLM gen)
  activations/      # produced by step 2 (per-role activation dicts)
  vectors/          # produced by step 3 (unfiltered means; one .pt per role)
  axis.pt           # produced by step 4 (Lu canonical assistant axis)
  default.pt        # produced by step 5 (mean default activation)
  contrasts/        # produced by compute_axes.py (5 contrast vector .pts)
  projections/      # produced by compute_axes.py (raw + zscore + role_index)
```

For development on this machine, use the parent project's pre-computed activations to skip the ~16 GPU-hour generation step:
```bash
export PHASE_H_DATA_ROOT=/jumbo/lisp/fl1/assistant-axis-abliteration/results
```

## Usage — three steps

### 1. Pre-flight
Validates configs, env vars, model directories, vendored library, GPU availability:
```bash
$PYTHON check_ready.py
```
Exit 0 = ready, exit 1 = fix the listed issues first.

### 2. Pipeline (one model at a time)
Runs vLLM generation + activation extraction + unfiltered vectors + axis + default. Args: HuggingFace model ID, output dir (under `$PHASE_H_DATA_ROOT`), extraction layer, hidden dim.

```bash
./pipeline.sh microsoft/Phi-3.5-mini-instruct $PHASE_H_DATA_ROOT/phi-3.5-mini 16 3072
./pipeline.sh meta-llama/Llama-3.2-3B-Instruct $PHASE_H_DATA_ROOT/llama-3.2-3b 14 3072
./pipeline.sh Qwen/Qwen2.5-3B-Instruct $PHASE_H_DATA_ROOT/qwen2.5-3b 18 2048
```

Cost per model: ~12–16 GPU-hours, ~50 GB output (mostly responses + activations). Steps are resume-safe (skips role files that already exist).

### 3. Analysis

```bash
# Build contrast vectors and project roles for every model in configs/models.py:
$PYTHON compute_axes.py --all

# (alternatively, single model:)
$PYTHON compute_axes.py --tag phi
$PYTHON compute_axes.py --model_dir phi-3.5-mini

# Paired-anchor-bootstrap on the default-Assistant z-score; render the radar:
$PYTHON bootstrap_radar.py --n_boot 2000
```

### Outputs (under `out/radar/`)

- `three_model_radar.png` / `_zoom.png` — point-estimate radar (no CI overlay).
- `three_model_radar_with_ci.png` / `_zoom.png` — radar with anchor-bootstrap 95% CI bars per spoke.
- `default_z_scores.npz` — point estimates per (model, axis).
- `default_bootstrap_ci.json` — per-(model, axis) CI summaries + per-pair gap CI summaries (with `ci95_excludes_zero` flags).
- `default_bootstrap_arrays.npz` — raw bootstrap distributions for diagnostics.

Skip the bootstrap and just plot point estimates:
```bash
$PYTHON bootstrap_radar.py --no-bootstrap
```

## Skipping the judge

The Lu et al. pipeline includes an LLM-judge step that scores each rollout for role-fidelity (0–3) and filters to score=3 before averaging. **We skip this step.** Justification:

1. **Empirical** — for the three small instruct models in `configs/models.py` (Phi-3.5-mini, Llama-3.2-3B, Qwen-2.5-3B), the filtered and unfiltered role-vector matrices are byte-identical (filter rate effectively 100%). Verified in the parent project.
2. **Cost** — judging takes ~24 GPU-hours and ~$70/model in OpenAI API spend.
3. **Robustness** — Lu et al. App. B.3 reports base ↔ instruct role-vector cosine > 0.99, so the persona structure is robust to data-pruning choices.

If you swap in a model where filter rate genuinely matters (a base model, a noisy fine-tune, an unreliable role-player), restore the judge step from upstream `assistant-axis/pipeline/3_judge.py` between steps 2 and 4 of `pipeline.sh`. The judge's outputs are byte-compatible with our unfiltered step 3 — just substitute.

## Deferred analyses

This repo does the **headline only** — pipeline → contrasts → projections → radar with CIs. Reviewer-defensibility analyses are deliberately deferred. They include:

| Deferred module | What it adds |
|---|---|
| **Pre-registered tests A–I** | independence, PC1 alignment, cross-model Spearman with bootstrap CI + permutation null, anchor sanity, validation pairs, null-role purity, magnitude comparison |
| **Anchor jackknife (J) + per-role bootstrap (K)** | "did you cherry-pick anchors?" + "which specific roles drive cross-model differences?" |
| **Persona-space sanity gate** | PCA variance, PC1=Assistant, default-extreme, semantic clustering — exit-1 if extraction is broken |
| **Lu-style cross-model PCA correlation** | the methodological link to Lu et al.'s reported PC1 r > 0.92 |
| **Procrustes alignment (3 variants)** | "do the persona spaces align as wholes, not just on the named axes?" |
| **Diagnostic plots** | per-axis scatter, independence heatmap, alignment plots |

Each deferred module's source code lives in the parent project at `/jumbo/lisp/fl1/assistant-axis-abliteration/scripts/` and is documented in detail in a project memory note (priority order, source paths, reviewer-attack each closes). They port cleanly because this repo's `_common.py` preserves the same API as the parent.

## Troubleshooting

**`ImportError: No module named assistant_axis`** — the vendored library isn't installed. See *Install* above.

**`FileNotFoundError: No .pt role vectors in .../vectors`** — pipeline.sh hasn't run yet for that model, or `PHASE_H_DATA_ROOT` is wrong. Run `check_ready.py` to see which model directories exist.

**`Role list for 'X' differs from 'Y' — paired bootstrap requires identical role lists`** — different models have different role files (probably different filter rates or one was processed with a different upstream-library version). The paired bootstrap requires identical role sets across models. Either re-run the pipeline for the offending model or drop it from `configs/models.py`.

**Bootstrap CI bars look weird / inverted on the radar** — matplotlib polar plots fold negative `r` to the opposite spoke. The radar code uses `set_rorigin` to a value below the data minimum to render negatives correctly. If you customize the plot, preserve that.

**Layer mismatch between `pipeline.sh` and `configs/models.py`** — the layer is declared twice: once as a positional CLI arg to `pipeline.sh` and once in `configs/models.py`. Mismatch will compute the wrong contrast (analysis reads from the layer that pipeline.sh extracted to). Keep them in sync; consider a wrapper script that pulls layer from the config.

## Pushing to GitHub

```bash
# Option A: gh CLI
gh repo create pjcherian7/persona-space-comparison --public --source=. --remote=origin --push \
  --description "Minimal Phase H reproducibility: assistant-axis pipeline + contrast-vector projection"

# Option B: web UI + git remote
# 1. Create empty repo at https://github.com/new (no README/LICENSE/.gitignore — already have those)
# 2. Then locally:
git remote add origin git@github.com:pjcherian7/persona-space-comparison.git
git push -u origin main
```

## License

MIT (set at first push).

## Citation

If using this code, cite Lu et al.'s original Assistant Axis paper:
```
@article{lu2026assistantaxis,
  title={The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models},
  author={Lu, Christina and Gallagher, ... and Lindsey, Jack},
  journal={arXiv preprint arXiv:2601.10387},
  year={2026}
}
```
