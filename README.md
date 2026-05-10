# persona-space-comparison

Minimal pipeline for comparing language models on a panel of contrast-vector axes derived from the [assistant-axis](https://github.com/safety-research/assistant-axis) methodology (Lu et al., 2026), with anchor-bootstrap CIs + permutation null on the headline radar.

**What this does** — for each model:
1. Run the assistant-axis pipeline (no judge — see *Skipping the judge* below).
2. Build seven contrast vectors per model: `v_assistant` (Lu canonical) plus six anchor-pair contrasts (`v_benevolence`, `v_authority`, `v_humor`, `v_critic`, `v_mystical`, `v_edgy`).
3. Project all 275 roles onto the contrasts; z-score within model; render a paired-anchor-bootstrap radar with 95% CIs.
4. (Optional) Run a PC1-residualized permutation null: are our hand-curated anchors doing concept-specific work beyond random 8+8 partitions?

Two researcher knobs (axes panel + model set) live in `configs/`.

## Install

```bash
git clone https://github.com/pauljCherian/persona-space-comparison.git
cd persona-space-comparison
python -m venv .venv && source .venv/bin/activate
pip install -e git+https://github.com/safety-research/assistant-axis.git#egg=assistant-axis
pip install -r requirements.txt
source .envrc   # sets $PYTHON + $PHASE_H_DATA_ROOT defaults (or use direnv)
```

`pipeline.sh` auto-discovers the assistant-axis library location from `assistant_axis.__file__`.

The pre-computed artifacts (responses, activations, vectors, radar PNGs) for Phi-3.5-mini, Llama-3.2-3B, and Qwen-2.5-3B are published at [`pandaman007/persona-space-comparison`](https://huggingface.co/datasets/pandaman007/persona-space-comparison) if you want to skip the ~16 GPU-hour generation step per model. To use them, download to `$PHASE_H_DATA_ROOT` and run `compute_axes.py` + `bootstrap_radar.py` directly.

## Configuration: the two researcher knobs

Both live in `configs/` as plain Python files. Edit in place to fork; or copy + override via env var.

### `configs/axes.py` — the contrast-axis panel
```python
AXIS_ORDER = ["v_assistant", "v_benevolence", "v_authority", "v_humor", "v_critic", "v_mystical", "v_edgy"]

ANCHOR_AXES: dict[str, tuple[list[str], list[str]]] = {
    "v_benevolence": (
        ["counselor", "parent", ...],   # positive pole
        ["criminal", "saboteur", ...],  # negative pole
    ),
    ...
}
```

`v_assistant` is special (computed at runtime as default − mean(roles)) — it appears in `AXIS_ORDER` but not `ANCHOR_AXES`. Rationale for each anchor list in `ANCHORS.md`.

To run with a different panel:
```bash
cp configs/axes.py configs/no_critic.py    # then edit configs/no_critic.py
PHASE_H_AXES=configs/no_critic.py python compute_axes.py --all
```

### `configs/models.py` — the model set
```python
MODELS: dict[str, dict] = {
    "phi": {
        "model_id":   "microsoft/Phi-3.5-mini-instruct",
        "path":       "phi-3.5-mini",   # subdir under PHASE_H_DATA_ROOT
        "layer":      16,               # extraction layer (Lu canonical: N/2)
        "hidden_dim": 3072,
        "color":      "#1f77b4",
        "display":    "Phi-3.5-mini",
    },
    ...
}
```

Override via `PHASE_H_MODELS=configs/alt_models.py` the same way.

### Where data lives — `PHASE_H_DATA_ROOT`

The repo ships **code only**. Per-model artifacts are read from / written to `$PHASE_H_DATA_ROOT/<MODELS[tag]['path']>/`. Default: `./data`.

Layout per model:
```
$PHASE_H_DATA_ROOT/phi-3.5-mini/
  responses/        # produced by pipeline.sh step 1 (vLLM gen)
  activations/      # produced by step 2 (per-role activation dicts)
  vectors/          # produced by step 3 (unfiltered means; one .pt per role)
  axis.pt           # produced by step 4 (Lu canonical assistant axis)
  default.pt        # produced by step 5 (mean default activation)
  contrasts/        # produced by compute_axes.py (7 contrast vector .pts)
  projections/      # produced by compute_axes.py (raw + zscore + role_index)
```

## Usage

### 1. Pre-flight
```bash
$PYTHON check_ready.py
```
Validates configs, env vars, model directories, vendored library, anchor existence, GPU availability.

### 2. Pipeline (one model at a time)
```bash
./pipeline.sh phi
./pipeline.sh llama
./pipeline.sh qwen
```
Cost per model: ~12–16 GPU-hours, ~2–3 GB output. Resume-safe (skips role files that already exist).

### 3. Analysis
```bash
$PYTHON compute_axes.py --all                       # build contrasts + projections
$PYTHON bootstrap_radar.py --n_boot 2000            # paired anchor bootstrap + radar PNGs
$PYTHON permutation_null.py --n_iter 2000           # PC1-residualized permutation null
```

### Outputs (under `out/radar/`)

- `three_model_radar.png` / `_zoom.png` — point-estimate radar.
- `three_model_radar_with_ci.png` / `_zoom.png` — radar with anchor-bootstrap 95% CI bars.
- `default_z_scores.npz` — point estimates per (model, axis).
- `default_bootstrap_ci.json` — per-(model, axis) CI summaries + per-pair gap CI summaries.
- `default_bootstrap_arrays.npz` — raw bootstrap distributions.
- `permutation_null.json` — per-axis null distribution stats, real residualized z-scores, p-values.
- `permutation_null.npz` — raw null arrays (per-model z and per-pair gap).

Skip the bootstrap and just plot point estimates:
```bash
$PYTHON bootstrap_radar.py --no-bootstrap
```

## Skipping the judge

The Lu et al. pipeline includes an LLM-judge step that scores each rollout for role-fidelity (0–3) and filters to score=3 before averaging. **We skip this step.** Justification:

1. **Empirical** — for the three small instruct models in `configs/models.py` (Phi-3.5-mini, Llama-3.2-3B, Qwen-2.5-3B), filtered and unfiltered role-vector matrices are byte-identical (filter rate effectively 100%).
2. **Cost** — judging takes ~24 GPU-hours and ~$70/model in OpenAI API spend.
3. **Robustness** — Lu et al. App. B.3 reports base ↔ instruct role-vector cosine > 0.99, so the persona structure is robust to data-pruning choices.

If you swap in a model where filter rate genuinely matters (a base model, a noisy fine-tune, an unreliable role-player), restore the judge step from upstream `assistant-axis/pipeline/3_judge.py` between steps 2 and 4 of `pipeline.sh`. The judge's outputs are byte-compatible with our unfiltered step 3 — just substitute.

## Troubleshooting

**`ImportError: No module named assistant_axis`** — the vendored library isn't installed. See *Install* above.

**`FileNotFoundError: No .pt role vectors in .../vectors`** — pipeline.sh hasn't run yet for that model, or `PHASE_H_DATA_ROOT` is wrong. Run `check_ready.py` to see which model directories exist.

**`Role list for 'X' differs from 'Y'`** — different models have different role files (probably different filter rates or one was processed with a different upstream-library version). The paired bootstrap requires identical role sets. Either re-run the pipeline for the offending model or drop it from `configs/models.py`.

**Bootstrap CI bars look inverted on the radar** — matplotlib polar plots fold negative `r` to the opposite spoke. The radar code uses `set_rorigin` to a value below the data minimum to render negatives correctly. Preserve that if customizing.

## License

MIT.

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
