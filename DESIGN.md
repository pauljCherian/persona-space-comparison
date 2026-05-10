# Design — persona-space-comparison

## Research question

For a given panel of small language models, do their default Assistant personas occupy comparable positions on a small set of interpretable axes (assistant-likeness, benevolence, authority, humor, criticality)? And how robust are those positions to anchor choice?

This is a *cross-model* operationalization of Lu et al. (2026)'s persona-space methodology. Lu et al. measured an "Assistant Axis" within each model and reported that PC1 of role-vector space correlates >0.92 across large models (Gemma-2-27B, Qwen-3-32B, Llama-3.3-70B). We extend with anchor-pair contrast axes for *named* concepts (benevolence, authority, humor, criticality), use small instruct models, and add an anchor-bootstrap CI on the headline radar.

## Pre-registered methodology

### 1. Pipeline (per model)

For each model in `configs/models.py`:

1. Generate 1,200 rollouts per role × 276 roles using vLLM.
   - 5 system-prompt variants × 240 extraction questions × 1 sample per (prompt, question) per role.
   - Roles + system prompts + extraction questions come from the upstream `assistant-axis/data/roles/`. 275 character-archetype roles + 1 default "neutral Assistant" role.
2. Extract residual-stream activation at layer L = N/2 (Lu canonical) for each rollout.
   - Activation is the post-MLP residual stream at the response-token positions, averaged over response tokens. Token-position spans are computed by `assistant_axis.internals.SpanMapper`.
3. **Skip judging.** See justification below.
4. Compute per-role mean activation across all 1,200 rollouts → role vector ∈ ℝᵈ.
5. Compute Lu's canonical assistant axis: `axis = mean(default activations) − mean(role activations)` (using `assistant-axis/pipeline/5_axis.py`).
6. Compute the model's default activation vector (mean of default rollouts).

### 2. Contrast vectors (per model)

Five axes, all computed in the model's native ℝᵈ:

```
v_assistant   = default                       − mean(all_roles)
v_benevolence = mean(8 helper roles)          − mean(8 harmer roles)
v_authority   = mean(8 expert roles)          − mean(8 novice roles)
v_humor       = mean(8 playful roles)         − mean(8 serious roles)
v_critic      = mean(8 oppositional roles)    − mean(8 affirmative roles)
```

`v_assistant` is computed from the entire role distribution (canonical Lu definition). The four anchor-pair contrasts are computed from 16 hand-curated roles per axis (8 positive pole, 8 negative pole). All anchor lists locked in `configs/axes.py`. The anchor-pair contrast formulation is the standard one from the linear-representation-hypothesis literature (Marks & Tegmark 2023; Park et al. ICML 2024) and from Anthropic's persona-vectors work (Chen et al. 2025).

64 unique anchors total, no overlap between axes (verified by `check_ready.py`).

### 3. Per-role projections

Project all 276 role vectors (275 character roles plus the default) onto each contrast axis:

```
raw[i, k] = role_vector_i · contrast_k                  for i in 1..276, k in 1..5
```

Z-score within model per axis (subtract column mean, divide by column std):

```
zscore[i, k] = (raw[i, k] − mean_i(raw[·, k])) / std_i(raw[·, k])
```

This makes magnitudes commensurable across models (each model's role distribution is its own reference frame). Outputs go to `<model_dir>/projections/{raw.pt, zscore.pt, role_index.json}`.

Note that the default vector is included in the role distribution used for per-axis mean/std. Because `default · v_assistant` is by construction extreme, this slightly inflates the std and shifts the mean upward, mildly reducing default's apparent z-score magnitude. We retain this for byte-equality with parent project's reference values; if you want strict 275-only normalization, filter `"default"` out in `compute_axes.py:project_and_zscore` and `bootstrap_radar.py:load_all_models`.

### 4. Default-Assistant z-score per axis

The default Assistant activation is also projected onto each axis and z-scored against the model's role-projection distribution:

```
default_z[k] = (default · v_k − mean_i(role_projection[i, k])) / std_i(role_projection[i, k])
```

This yields one scalar per (model × axis) — the radar-plot value.

### 5. Paired anchor bootstrap

For each anchor axis (skipping `v_assistant` which has no anchors), resample its 8 positive and 8 negative anchor *positions* with replacement. The same resampled indices are applied to *all* models in the analysis, so per-axis anchor variability is shared and the between-model gap CI is conservative (paired-bootstrap structure).

For each bootstrap iteration b ∈ 1..B:
- For each axis × model: rebuild `v_axis_b` from resampled anchors, project default, z-score against the model's (resampled-axis) role distribution → `default_z_b`.
- For each model pair (a, b): compute `gap_b = default_z_b[a] − default_z_b[b]`.

After B iterations:
- Per-(model, axis) 95% CI: `[percentile(default_z_b, 2.5), percentile(default_z_b, 97.5)]`.
- Per-(pair, axis) 95% CI on the gap.

**Default B = 2000** (sufficient for stable percentile estimates; smaller B for quick smoke tests).

**Sign-stable verdict:** a (model, axis) cell is "sign-stable at 95%" iff > 95% of bootstrap samples lie on the same side of zero.

**Pair-gap-robust verdict:** a (pair, axis) cell is robust iff the 95% CI excludes zero.

### 6. Radar plot

Polar plot, one spoke per axis, one polygon per model. Optional CI bars per spoke (offset slightly between models so they don't overlap). Zero ring shown as dashed reference (because `default_z = 0` corresponds to "default is at the role-mean").

## Layer choice

Lu et al. use layer N/2 (the middle residual stream layer) as canonical. Verified across their three models: Gemma-2-27B (22/46), Qwen-3-32B (32/64), Llama-3.3-70B (40/80). For our small models:

| Model | Total layers | L = N/2 |
|---|---|---|
| Phi-3.5-mini-instruct | 32 | **16** |
| Llama-3.2-3B-Instruct | 28 | **14** |
| Qwen2.5-3B-Instruct | 36 | **18** |

These are configured per-model in `configs/models.py`. To override, edit the file.

Lu et al. App. B.4 reports that nearby layers (L ± 2) yield essentially identical persona structure. We do **not** sweep layers in this minimal repo — the parent project includes a layer-sweep script (`scripts/36_phase_h_layer_sweep.sh`) which was deleted as ancillary; if needed for robustness analysis, it can be re-added.

## Why no judge

The standard Lu et al. pipeline includes an LLM-judge filter (step 3) that scores each rollout for role-fidelity on a 0–3 scale and keeps only score=3 samples before averaging. We omit this step. Three reasons:

1. **Empirical** — for all three models in `configs/models.py`, the parent project verified that `vectors/` (judge-filtered) and `vectors_unfiltered/` (no judge) directories contain byte-identical role vectors. Filter rate is effectively 100% for these small instruct models on the role/question set: every rollout passes the judge.

2. **Cost** — judging is ~24 GPU-hours of API time and ~$70/model in OpenAI API spend. This is the most expensive step in the entire pipeline.

3. **Robustness** — Lu et al. App. B.3 shows base ↔ instruct role-vector cosine > 0.99 on the same architecture, suggesting the underlying persona structure is robust to data-pruning choices like judge filtering.

If you swap in a model where filter rate genuinely differs (e.g., a base model, an unreliable role-player, or a noisy fine-tune), restore upstream `assistant-axis/pipeline/3_judge.py` between steps 2 and 4 of `pipeline.sh`. The judge's outputs are byte-compatible with our unfiltered step 3.

## Anchor selection rationale

The four anchor-pair axes (`v_benevolence`, `v_authority`, `v_humor`, `v_critic`) were finalized in the parent project after iteration. Two earlier candidates were tested and dropped:

- `v_humanness` — non-human pole (vampire, undead, etc.) spanned three incompatible sub-clusters (liminal-undead, fantasy-mythic, transformative); no coherent semantic meaning. Dropped.
- `v_collective` — anchors (virus, zeitgeist, vampire, etc.) cross-loaded heavily on other axes; bio-network vs cosmic confound. Dropped.

The four remaining axes were selected for: (a) interpretability — each pole names a coherent concept; (b) anchor independence — pairwise cosines between contrast vectors below 0.5 in pilot runs; (c) coverage of orthogonal dimensions. The anchor lists themselves are 8+8 per axis chosen to span the conceptual range without semantic redundancy.

The pre-registration discipline matters: the anchor lists are locked in `configs/axes.py` *before* projecting. To explore alternative panels, copy to a new config file and load via `PHASE_H_AXES=...` — don't edit the locked file post-hoc.

## What this repo does NOT include

Reviewer-defensibility analyses are deliberately deferred. See `README.md → "Deferred analyses"` and the project memory note for the catalogue + porting priority. In particular:

- No within-model independence test (Test A: cosines between contrasts).
- No PC1-alignment test (Test B: |cos(v_axis, PC1)|).
- No cross-model Spearman + bootstrap on held-out roles (Test C).
- No anchor sanity (Test D) or held-out validation pairs (Test G) or null-role purity (Test H).
- No anchor jackknife / per-role bootstrap (Tests J + K).
- No persona-space sanity gate.
- No Procrustes alignment (5-D, K-anchor, full-cloud).
- No Lu-style cross-model PCA correlation.
- No diagnostic plots (scatter, heatmap, alignment).

The headline radar without these is a pretty picture; with them, it's a defensible result. Add them when you write up.

## Locked decisions (do not change without pre-registration update)

- 4 anchor axes + 1 canonical = 5 total. (v_humanness and v_collective were dropped during iteration; do not re-add without re-running the full panel.)
- L = N/2 per Lu canonical (no per-model layer sweep in this repo).
- 16 anchors per axis (8 positive + 8 negative). 64 unique anchors total.
- Z-score normalization within-model (so cross-model magnitudes are commensurable).
- Paired anchor bootstrap with B = 2000 default.
- "Sign-stable" threshold: 95% of bootstrap samples on same side of zero.
- "Pair-gap robust" threshold: 95% CI excludes zero.

To explore alternatives, copy the relevant config and override via env var. Don't edit locked decisions in place — keep the audit trail.

## References

- **Lu, Gallagher, Michala, Fish, Lindsey** (2026). *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models.* arXiv:2601.10387.
- **Marks & Tegmark** (2023). *The Geometry of Truth.* arXiv:2310.06824.
- **Park, Choe, Veitch** (2024). *The Linear Representation Hypothesis.* ICML 2024. arXiv:2311.03658.
- **Chen, Arditi, Sleight, Evans, Lindsey** (2025). *Persona Vectors: Monitoring and Controlling Character Traits.* arXiv:2507.21509.
- Upstream library: [github.com/safety-research/assistant-axis](https://github.com/safety-research/assistant-axis).
- Pre-computed vectors (parent project, partial): [huggingface.co/datasets/pandaman007/assistant-axis-abliteration-vectors](https://huggingface.co/datasets/pandaman007/assistant-axis-abliteration-vectors).
