# persona-space-comparison

Minimal pipeline for comparing language models on a small panel of contrast-vector axes derived from the [assistant-axis](https://github.com/safety-research/assistant-axis) methodology (Lu et al., 2026), with anchor-bootstrap CIs on the headline radar.

**Three operations:**
1. Run the assistant-axis pipeline on a model (no judge — see `DESIGN.md`).
2. Build contrast vectors per model (default + 4 anchor-defined axes).
3. Project all roles onto the contrasts, z-score within model, render a radar with anchor-bootstrap CIs.

That's the whole repo. ~940 lines of code.

## Install

```bash
git clone https://github.com/pjcherian7/persona-space-comparison.git
cd persona-space-comparison
python -m venv .venv && source .venv/bin/activate
pip install -e git+https://github.com/safety-research/assistant-axis.git#egg=assistant-axis
pip install -r requirements.txt
```

## The two researcher knobs

Both live in `configs/` and can be edited in place or swapped via env vars:

- `configs/axes.py` — `AXIS_ORDER` and `ANCHOR_AXES` (which roles define each contrast).
- `configs/models.py` — `MODELS` dict (paths, layer, hidden_dim, color, display name).

To run with an alternate axis panel:
```bash
cp configs/axes.py configs/no_critic.py   # edit
PHASE_H_AXES=configs/no_critic.py python compute_axes.py --all
```

## Where data lives

The repo is code-only. Per-model artifacts live under `$PHASE_H_DATA_ROOT/<model_path>/`. Default `$PHASE_H_DATA_ROOT` is `./data/`.

```
$PHASE_H_DATA_ROOT/
  phi-3.5-mini/
    responses/        # produced by pipeline.sh step 1 (vLLM)
    activations/      # produced by step 2
    vectors/          # produced by step 3 (unfiltered means)
    axis.pt           # produced by step 4
    default.pt        # produced by step 5
    contrasts/        # produced by compute_axes.py
    projections/      # produced by compute_axes.py
  llama-3.2-3b/
    ...
  qwen2.5-3b/
    ...
```

## Run

### Pre-flight
```bash
python check_ready.py
```

### Pipeline (one model at a time, ~16 GPU-hours per 3B model)
```bash
./pipeline.sh microsoft/Phi-3.5-mini-instruct $PHASE_H_DATA_ROOT/phi-3.5-mini 16 3072
./pipeline.sh meta-llama/Llama-3.2-3B-Instruct $PHASE_H_DATA_ROOT/llama-3.2-3b 14 3072
./pipeline.sh Qwen/Qwen2.5-3B-Instruct $PHASE_H_DATA_ROOT/qwen2.5-3b 18 2048
```

### Analysis
```bash
python compute_axes.py --all                    # all models in configs/models.py
python bootstrap_radar.py --n_boot 2000          # paired anchor bootstrap + radar PNGs
```

### Outputs (under `out/radar/`)
- `three_model_radar.png` / `_zoom.png` — point-estimate radar.
- `three_model_radar_with_ci.png` / `_zoom.png` — radar with anchor-bootstrap 95% CIs.
- `default_bootstrap_ci.json` — per-model and per-pair-gap CI summaries.
- `default_z_scores.npz` — point estimates.
- `default_bootstrap_arrays.npz` — raw bootstrap distributions.

## Skipping the judge

The Lu et al. pipeline judges each rollout for role-fidelity (score 0–3) and filters to score=3 before averaging. We **skip this** because: (a) for the small instruct models in our `configs/models.py` (Phi-3.5-mini, Llama-3.2-3B, Qwen-2.5-3B), the empirical filter rate is effectively 100% — we verified that filtered and unfiltered role-vector matrices are byte-identical; (b) the cost is ~24 GPU-hours and ~$70/model in OpenAI API spend; (c) Lu et al. App. B.3 reports base ↔ instruct role vectors agree at cos > 0.99, so the underlying structure is robust to data-pruning choices. If you swap in a model where the filter rate genuinely matters (e.g., a base model or noisy fine-tune), restore the judge step from upstream `assistant-axis/pipeline/3_judge.py`.

## Deferred analyses

This repo does the *headline* — pipeline → contrasts → projections → radar with CIs. Reviewer-defensibility analyses are deliberately deferred and tracked in a separate memory note. They include:

- Pre-registered tests A–I (cross-model Spearman, anchor sanity, validation pairs, null purity, magnitude comparison).
- Anchor-jackknife (Test J) and per-role hierarchical bootstrap (Test K).
- Three Procrustes alignment variants (5-D contrast, K-anchor 3072-D, full-cloud 3072-D).
- Lu-style cross-model PC role-loading correlation.
- Persona-space sanity gate (PCA variance, PC1=Assistant, default-extreme, semantic clustering).
- Diagnostic plots (cross-model scatter, independence heatmap, alignment diagnostics).

Source code for each lives in the parent project (the same `_common.py` API is preserved, so they drop in cleanly).

## License

MIT (or whatever — set at first push).
