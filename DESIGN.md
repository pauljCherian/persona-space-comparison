# Phase H Design

## Goal

Compare a small panel of language models on their *default Assistant persona* in a model-independent coordinate system, defined by anchor-based contrast vectors over the Lu et al. (2026) 275-role activation set.

## Methodology

### 1. Pipeline (per model)

For each model:
1. Generate 1,200 rollouts per role × 276 roles using vLLM. (`assistant-axis/pipeline/1_generate.py`)
2. Extract residual-stream activations at layer L = N/2 (Lu canonical). (`assistant-axis/pipeline/2_activations.py`)
3. **Skip the judge step.** See `README.md → "Skipping the judge"`.
4. Compute per-role mean activation across all rollouts → role vector ∈ ℝᵈ.
5. Compute Lu's canonical assistant axis: `axis = mean(default activations) − mean(role activations)`. (`assistant-axis/pipeline/5_axis.py`)
6. Compute the model's default activation vector (mean over default rollouts).

### 2. Contrast vectors (per model)

Five axes. v_assistant is canonical (default vs. role mean). The other four are anchor-pair contrasts:

```
v_assistant   = default − mean(all_roles)
v_benevolence = mean(helpers)   − mean(harmers)
v_authority   = mean(experts)   − mean(novices)
v_humor       = mean(playful)   − mean(serious)
v_critic      = mean(oppositional) − mean(affirmative)
```

Anchor lists are locked in `configs/axes.py` (8 positive + 8 negative per anchor axis = 64 anchors total, no overlap).

### 3. Per-role projections

For each model: project all 275 role vectors onto each contrast axis to get raw scalars (shape 275 × 5). Z-score within model per axis (mean 0, std 1) so projection magnitudes are commensurable across models.

### 4. Default-Assistant z-score on each axis

The "default" Assistant activation is also projected onto each axis and z-scored against the model's role-distribution mean and std. This yields one scalar per (model × axis) that the radar plot displays.

### 5. Anchor bootstrap (paired)

For each anchor axis, resample its 8+8 anchor *positions* with replacement. The same resampled indices are applied to all models so per-axis variability is shared. For each bootstrap iteration:
- For each model: rebuild axis from resampled anchors, project default, z-score → bootstrap default-z.
- Compute per-pair gap (model A's default-z) − (model B's default-z) for every pair.

After B iterations:
- Per-(model, axis) 95% CI on default-z.
- Per-pair-axis 95% CI on the gap.

### 6. Radar plot

Polar plot with one spoke per axis, one polygon per model. Optionally overlays per-spoke CI bars. The zero ring (z = 0 = role-mean) is shown as a dashed reference.

## Layer choice

Lu et al. use layer N/2 (the middle residual stream layer) as canonical:
- Phi-3.5-mini-instruct: 32 layers → L = 16.
- Llama-3.2-3B-Instruct: 28 layers → L = 14.
- Qwen2.5-3B-Instruct: 36 layers → L = 18.

These are configured per-model in `configs/models.py`.

## Why no judge

Three reasons:
1. **Empirical:** for the three target models, filtered (judge ≥ "somewhat role-playing") and unfiltered role-vector matrices are byte-identical. Filter rate ≈ 100%.
2. **Cost:** the judge step is ~24 GPU-hours and ~$70/model in OpenAI API spend.
3. **Robustness:** Lu et al. (App. B.3) report base ↔ instruct role-vector cosine > 0.99, suggesting the underlying persona structure is robust to data-pruning choices.

If you swap in a model where filter rate genuinely differs (e.g., a base model or a noisy fine-tune), restore upstream `assistant-axis/pipeline/3_judge.py` between steps 2 and 4.

## What this repo does NOT include

Reviewer-defensibility analyses are deliberately deferred. See `README.md → "Deferred analyses"` for the list and provenance.

## Locked decisions

- 4 anchor axes (v_humanness and v_collective tested and dropped in the parent project — see parent's PHASE_H_DESIGN.md history if interested).
- Layer = N/2 per Lu canonical (no per-model layer sweep here).
- Z-score within-model normalization (so cross-model magnitudes are commensurable).
- Paired anchor bootstrap with B = 2000 default.
- "PASS" sign-stable threshold = 95% of bootstrap samples on same side of zero.
- Pair-gap robustness threshold = 95% CI excludes zero.

To change any of these, edit `configs/` (for axes/models) or the relevant CLI flag (for B, seed).
