# Project: persona-space-comparison

## One-sentence summary

Cross-model comparison pipeline built on the [assistant-axis](https://github.com/safety-research/assistant-axis) methodology (Lu et al., 2026): for each model, runs the generation + activation extraction pipeline (no judge), builds seven contrast vectors (canonical Lu assistant axis + six anchor-pair contrasts), projects all 275 roles, and produces a paired-anchor-bootstrap radar with 95% CIs plus a PC1-residualized permutation null per axis.

## Repository layout

```
persona-space-comparison/
├── CLAUDE.md                    ← this file
├── README.md                    user-facing install + usage
├── DESIGN.md                    methodology, locked decisions
├── ANCHORS.md                   per-axis anchor rationale
├── requirements.txt
├── .envrc                       direnv config — sets $PYTHON + $PHASE_H_DATA_ROOT
├── .gitignore
├── pipeline.sh                  vLLM gen + activations + vectors + axis + default
├── check_ready.py               pre-flight checks
├── _common.py                   config loader, data loaders, math helpers
├── compute_axes.py              build contrasts + project all roles + z-score
├── bootstrap_radar.py           paired anchor bootstrap + radar PNGs
├── permutation_null.py          PC1-residualized null vs random 8+8 partitions
└── configs/
    ├── axes.py                  AXIS_ORDER + ANCHOR_AXES  (researcher knob 1)
    └── models.py                MODELS dict               (researcher knob 2)
```

## Researcher knobs

Two parameters live in dedicated, swappable Python config modules:

1. **`configs/axes.py`** — `AXIS_ORDER` + `ANCHOR_AXES`. Edit in place to fork, or copy + override:
   ```bash
   cp configs/axes.py configs/alt.py    # then edit
   PHASE_H_AXES=configs/alt.py python compute_axes.py --all
   ```

2. **`configs/models.py`** — `MODELS: dict[str, dict]` with keys `model_id`, `path`, `layer`, `hidden_dim`, `color`, `display`. Single source of truth — `pipeline.sh <tag>` reads everything from here.

Third knob: `PHASE_H_DATA_ROOT` (default `./data/`). Each model's artifacts live at `$PHASE_H_DATA_ROOT/$MODELS[tag]['path']/`.

`_common.py` loads both configs at import time via `importlib`, honoring env vars.

## Data flow

```
pipeline.sh <tag>          → data/<model>/{responses,activations,vectors}/ + axis.pt + default.pt
compute_axes.py --all      → data/<model>/{contrasts,projections}/
bootstrap_radar.py         → out/radar/three_model_radar*.png + default_bootstrap_*.{json,npz}
permutation_null.py        → out/radar/permutation_null.{json,npz}
```

The HF dataset [`pandaman007/persona-space-comparison`](https://huggingface.co/datasets/pandaman007/persona-space-comparison) mirrors all artifacts for the three models in the default config (Phi-3.5-mini, Llama-3.2-3B, Qwen-2.5-3B), so downstream researchers can skip the ~16 GPU-hour generation step.

## Methodology highlights (see DESIGN.md and ANCHORS.md for full detail)

- **No judge step.** Empirically the filter rate is ~100% for our small instruct models, so unfiltered ≡ filtered. Saves ~$70/model in OpenAI spend. Restore upstream `assistant-axis/pipeline/3_judge.py` between steps 2 and 4 of `pipeline.sh` if running a model where filter rate matters.
- **Layer = N/2** per Lu canonical (16 for Phi-3.5-mini, 14 for Llama-3.2-3B, 18 for Qwen-2.5-3B).
- **Within-model z-scoring** of role projections — makes magnitudes commensurable across models with different hidden dimensions.
- **Paired anchor bootstrap** (B=2000 default): same resampled anchor indices applied to all models per iteration. Conservative on per-cell CIs; powerful on cross-model gap CIs (shared anchor noise cancels).
- **PC1-residualized permutation null**: tests whether each hand-curated axis carries concept-specific signal beyond what random 8+8 partitions of the 179 non-anchor roles would give, after removing the role-matrix PC1 direction. Critical detail: residualization is against PC1 of the role-only covariance matrix (~0.92 correlated with Lu's Assistant Axis but distinct from it); residualizing against the Assistant Axis directly is mathematically degenerate.

## How to collaborate with the user (Paul Cherian)

- Reads every line by hand. Concise replies, no bloat.
- Pushes back on over-engineering. Prefers deletion over preservation when in doubt.
- Sharp clarifying questions over confident wrong answers.
- Honest "I don't know yet, let me check" over guesses.
- When committed to a scope ("core only"), respect it.
- Dartmouth student / researcher (GitHub `pauljCherian`, HF `pandaman007`, email `pjcherian7@gmail.com`).

## What this project is NOT

- **NOT** abliteration (Arditi et al.).
- **NOT** persona-vector steering (Anthropic-style).
- **NOT** a benchmark suite.

Just one experiment: cross-model contrast-vector comparison via the Lu et al. methodology, plus an anchor-bootstrap + permutation null defense.
