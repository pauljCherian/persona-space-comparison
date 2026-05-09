# Project: persona-space-comparison

## Two-sentence summary

Minimal pipeline for comparing language models on a small panel of contrast-vector axes derived from the [assistant-axis](https://github.com/safety-research/assistant-axis) methodology (Lu et al., 2026). For each model, run the upstream pipeline (no judge), build five contrast axes (canonical Lu assistant axis + four anchor-pair contrasts), project all 275 roles onto them with within-model z-scoring, and produce a paired-anchor-bootstrap radar with 95% CIs.

## Why this repo exists

This repo was extracted from a larger research project at `/jumbo/lisp/fl1/assistant-axis-abliteration/` (~9,640 LOC across Phases A–H of an exploratory program on persona-space geometry, abliteration, persona-vector steering, and cross-model comparison). The parent project produced the methodology and verified empirical findings. This repo distills *only the cross-model comparison pipeline* (parent's "Phase H") into the smallest reading surface that runs end-to-end — about 1,020 LOC with two clean researcher knobs.

The decision to keep this minimal was deliberate: the user (Paul Cherian) is reading every line by hand. A "full reproducibility" version with all reviewer-defensibility analyses would be ~2,260 LOC; the missing 1,240 LOC are catalogued as deferred modules (see "Deferred analyses" below) and can be ported back in priority order when the project moves toward writeup.

## Background: the methodology

### Lu et al. 2026 — "The Assistant Axis"

[arXiv:2601.10387](https://arxiv.org/abs/2601.10387). The paper:
1. Defines 275 character-archetype roles (pirate, analyst, demon, etc.), each with 5 system prompts × 240 questions = 1,200 rollouts per role.
2. For each rollout, extracts the post-MLP residual stream activation at the middle layer (L = N/2), averaged across response tokens.
3. Filters rollouts using an LLM judge (we **skip this** — see DESIGN.md).
4. Averages surviving rollouts per role into one role vector in ℝᵈ.
5. Defines the **Assistant Axis** as `axis = mean(default Assistant activation) − mean(role vectors)`.
6. Reports cross-model PC1 correlation > 0.92 across Gemma-2-27B, Qwen-3-32B, Llama-3.3-70B.

### What this repo adds on top

A panel of **anchor-pair contrast axes** beyond the canonical assistant axis:

| Axis | Positive pole anchors | Negative pole anchors |
|---|---|---|
| `v_assistant` | (canonical) default | mean(all roles) |
| `v_benevolence` | counselor, parent, guardian, pacifist, peacekeeper, altruist, healer, angel | criminal, saboteur, narcissist, zealot, hoarder, smuggler, demon, predator |
| `v_authority` | judge, scientist, ambassador, polymath, virtuoso, sage, leviathan, ancient | amateur, dilettante, student, infant, refugee, prey, prisoner, orphan |
| `v_humor` | comedian, jester, fool, absurdist, bohemian, surfer, improviser, bard | philosopher, mathematician, ascetic, scholar, hermit, traditionalist, conservator, statistician |
| `v_critic` | contrarian, devils_advocate, skeptic, cynic, perfectionist, evaluator, auditor, examiner | synthesizer, optimist, idealist, evangelist, romantic, advocate, facilitator, instructor |

Each anchor axis = `mean(positive-pole role vectors) − mean(negative-pole role vectors)`. Rationale and pre-registration in `DESIGN.md`.

## Repository layout

```
persona-space-comparison/
├── CLAUDE.md                   ← this file
├── README.md                   user-facing install + usage
├── DESIGN.md                   methodology, locked decisions
├── requirements.txt            python deps
├── .gitignore
├── pipeline.sh                 vLLM gen + activations + unfiltered vectors + axis + default
├── check_ready.py              pre-flight: configs, data root, models, library, GPU
├── _common.py                  config loader (env-var-aware), data loaders, math helpers
├── compute_axes.py             build contrasts + project all roles + z-score per model
├── bootstrap_radar.py          paired anchor bootstrap + radar PNGs (with optional CI bars)
└── configs/
    ├── axes.py                 ANCHOR_AXES + AXIS_ORDER  (researcher knob 1)
    └── models.py               MODELS dict               (researcher knob 2)
```

**Total: 1,020 LOC across 11 files.**

## Researcher knobs

The two parameters a researcher will tune most often live in dedicated, swappable Python config modules — never hardcoded inside analysis scripts:

1. **Contrast-axis definitions** — `configs/axes.py`. Defines `AXIS_ORDER` and `ANCHOR_AXES`. Edit in place to fork the panel; or copy + override:
   ```bash
   cp configs/axes.py configs/no_critic.py    # then edit
   PHASE_H_AXES=configs/no_critic.py python compute_axes.py --all
   ```
2. **Model set + per-model metadata** — `configs/models.py`. Defines `MODELS: dict[str, dict]` with keys `model_id` (HuggingFace ID for vLLM in `pipeline.sh`), `path` (subdir under `$PHASE_H_DATA_ROOT`), `layer`, `hidden_dim`, `color`, `display`. Single source of truth — `pipeline.sh <tag>` reads everything from here, so layer/hidden_dim can never drift between pipeline and analysis. Override the same way:
   ```bash
   cp configs/models.py configs/abliterated_set.py    # edit
   PHASE_H_MODELS=configs/abliterated_set.py python bootstrap_radar.py
   ```

`_common.py` loads both configs at import via `importlib`, honoring the env vars. Every downstream script just does `from _common import ANCHOR_AXES, MODELS, ...`.

The third knob is data location: `PHASE_H_DATA_ROOT` (default `./data/`). Each model lives at `$PHASE_H_DATA_ROOT/$MODELS[tag]['path']/`.

## Where data lives

The repo is **code-only**. Per-model artifacts (responses, activations, vectors, axes, defaults, contrasts, projections) are produced by `pipeline.sh` and `compute_axes.py` into `$PHASE_H_DATA_ROOT/<model_path>/`.

For development on this machine, the parent project's pre-computed activations are reused via the included `.envrc`:
```bash
source .envrc
# or with direnv installed: cd in, vars auto-load
```
which sets `PYTHON` to the parent's venv and `PHASE_H_DATA_ROOT` to the parent's `results/`. This avoids re-running the ~16 GPU-hour generation for each model. Existing per-model directories there contain `vectors_unfiltered/`, `default.pt`, etc.

The canonical artifacts are also published to HuggingFace at [`pandaman007/assistant-axis-abliteration-vectors`](https://huggingface.co/datasets/pandaman007/assistant-axis-abliteration-vectors) — not all keys map cleanly to this repo's conventions, but the vectors and defaults are there.

## Vendored library setup

The repo expects `assistant_axis` (Lu et al.'s Python library) to be importable. Two options:

**Development on lisplab** — the parent project's venv at `/jumbo/lisp/fl1/assistant-axis-abliteration/.venv/` already has it installed editably. Just `source .envrc` (or use direnv) and the included default `PYTHON` points there.

**Clean standalone setup** (e.g., after cloning fresh elsewhere):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e git+https://github.com/safety-research/assistant-axis.git#egg=assistant-axis
pip install -r requirements.txt
```

`pipeline.sh` auto-discovers the library location from `assistant_axis.__file__`, so either install method works.

## Verification status

As of initial commit `c8fb0fd` (built 2026-05-08), verified end-to-end against parent project's pre-computed activations:

1. **Parse + import:** all 6 Python files parse cleanly and import.
2. **`check_ready.py`:** 28 pre-flight checks pass.
3. **`compute_axes.py --tag phi`:** produces contrast vectors with norms `1.835 / 3.389 / 1.847 / 7.563 / 2.728` (v_assistant / v_benevolence / v_authority / v_humor / v_critic). These match the parent project byte-identically.
4. **`bootstrap_radar.py --n_boot 200`:** produces 4 radar PNGs + bootstrap CI JSON + raw bootstrap NPZ. Point estimates match the parent radar (Phi v_critic = −0.758, Llama v_humor = −0.828, etc.).
5. **Config swap:** `PHASE_H_AXES=configs/_test_3_axis.py` correctly drops `v_critic` from the output radar.

The pipeline half (steps 1–2: vLLM gen + activation extraction) was NOT re-run from scratch (would cost ~16 GPU-hours per model). The user may want to verify this end-to-end on one model before fully trusting the repo.

## Why no judge

The Lu et al. pipeline includes step 3 (LLM-judge filters rollouts to score=3 only). We skip it because:

1. **Empirical:** for the three models in `configs/models.py` (Phi-3.5-mini, Llama-3.2-3B, Qwen-2.5-3B), the parent project verified that filtered and unfiltered role-vector matrices are byte-identical for every role. Filter rate ≈ 100%.
2. **Cost:** ~24 GPU-hours and ~$70/model in OpenAI API spend.
3. **Robustness:** Lu et al. App. B.3 reports base ↔ instruct role-vector cosine > 0.99 — the persona structure is robust to data-pruning choices.

If you swap in a model where filter rate genuinely matters (a base model, or a noisy fine-tune), restore upstream `assistant-axis/pipeline/3_judge.py` between steps 2 and 4 of `pipeline.sh`. The judge step's outputs are byte-compatible with our unfiltered step 3 — just substitute.

## Deferred analyses

This repo ships only the headline pipeline. Reviewer-defensibility analyses are deliberately deferred. Each closes a specific reviewer attack on the radar:

| Module | LOC | Source in parent | Closes |
|---|---|---|---|
| `tests.py` (A–I) | ~330 | `scripts/34_compare_axes.py` | "Are axes independent? Do they transfer cross-model? Are they confounds (vocab, topic) instead of the named concept?" Test C is the *quantitative* cross-model claim. |
| `anchor_robustness.py` (J + K) | ~150 | `scripts/37_anchor_robustness.py` | "Did you cherry-pick anchors?" "Which roles drive cross-model differences?" |
| `sanity.py` | ~120 | `scripts/38_persona_space_sanity.py` | "Is the persona space meaningful at all before you start projecting?" Should be a hard gate before contrast-axis work. |
| `pca_cross_model.py` | ~140 | `scripts/44_pca_cross_model.py` | "Lu reports PC1 correlation > 0.92 across models — what do you get?" The methodological link to the published paper. |
| `procrustes.py` (3 modes) | ~250 | `scripts/{39,41,43}_*.py` | "Do the persona spaces align as wholes, not just on your hand-picked axes?" |
| `plots.py` (multi-mode) | ~250 | `scripts/{35,40,42}_*.py` | "Show me the data, not just summary stats." |

**Priority order for porting** (in the project memory at `~/.claude/projects/-jumbo-lisp-fl1-assistant-axis-abliteration/memory/phase_h_clean_deferred_modules.md`): tests → anchor_robustness → sanity → pca_cross_model → procrustes → plots. Each ports cleanly because `_common.py` preserves the same API as the parent.

## How to collaborate with the user (Paul)

Paul wants to read every line by hand. Concise replies, no bloat. He will push back on any over-engineering. He prefers:
- Asking sharp clarifying questions early over making assumptions late.
- Concrete file:line references over abstract description.
- Honest "I don't know yet, let me check" over confident wrong answers.
- Deletion over preservation when in doubt.

When he asks "is X needed", actually check empirically (he caught the no-judge insight by exactly that sort of question).

When he commits to a scope ("core only, full later"), respect it. Memory notes carry forward intent across sessions.

He is a Dartmouth student/researcher (login `f006vv2`, GitHub `pjcherian7`, email `pjcherian7@gmail.com`). The git config in this repo is set to that identity.

## What's next

Likely user actions, in order of probability:
1. Read every line of code in this repo. Probably ask clarifying questions.
2. Push to GitHub at `pjcherian7/persona-space-comparison` (instructions in README).
3. Re-run `bootstrap_radar.py --n_boot 2000` (full bootstrap) to regenerate the canonical radar from the cleaned code. Compare to parent's radar.
4. Optionally re-run `pipeline.sh` end-to-end on one model to verify generation+extraction. Costs ~16 GPU-hours; mostly redundant.
5. When ready for writeup: port the deferred modules in the priority order above.

## What this project is NOT (in case it's confused with the parent)

- **NOT abliteration.** Parent project's Phase E covered abliteration (Arditi et al.) on Llama-3.1-8B; this repo does not.
- **NOT persona-vector steering.** Parent's Phase F covered Anthropic-style persona vectors; not here.
- **NOT a LizaT (dangerous medical) experiment.** Parent's Phase G; not here.
- **NOT a benchmark suite.** Just one experiment: contrast-vector cross-model comparison via the Lu et al. methodology.

The parent project at `/jumbo/lisp/fl1/assistant-axis-abliteration/` has all of those.
