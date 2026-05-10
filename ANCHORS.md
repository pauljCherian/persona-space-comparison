# Defense of Contrast Vectors

Rationale for the positive/negative archetypes that define each contrast axis. All anchor names verified to exist in the 275-role library and to not collide across axes.

## Existing 4 axes (kept)

- `v_benevolence`, `v_authority`, `v_humor`, `v_critic` — locked from parent project. 64 anchors.

## Candidates considered

- **Warmth ↔ Sterility** — drop. Warmth pole (`counselor, parent, healer, altruist, romantic, advocate, idealist`) is already `v_benevolence`. Would be a relabel.
- **Moralistic ↔ Irreverent** — drop. Irreverent pole (`jester, fool, comedian, surfer, bohemian, dilettante`) is already `v_humor`.
- **Sycophantic ↔ Contentious** — drop. Contentious pole (`contrarian, devils_advocate, skeptic, cynic, judge, auditor, evaluator, examiner`) *is* `v_critic`. Bootstrap CIs on `v_critic` already speak to the sycophancy concern.
- **Hedging ↔ Authoritative** — drop. No clean archetype mapping in the 275. Better measured behaviorally (count "perhaps", "it depends", etc. on neutral prompts).
- **Witty ↔ Dry** — drop. Wit pole = `v_humor` positive pole. Dry pole has no coherent cluster (`scholar` is rigorous, not dry). Better measured by lexical-diversity / register classifier.
- **Sanitized ↔ Edgy** — add as `v_edgy`. Best user-discourse match outside the existing 4 (Grok-vs-ChatGPT axis). Curated 8+8 has zero collision with existing anchors.
- **Mystical ↔ Grounded** — add as `v_mystical`. Lu et al. specifically identify mystical/theatrical as a failure direction when steering far from default. Lowest cross-axis overlap. Methodologically the strongest add.

## New anchor lists (8+8 each, all unique vs existing 64)

- `v_mystical` positive: `eldritch, void, oracle, wraith, shaman, mystic, witch, genie`
- `v_mystical` negative: `realist, pragmatist, technologist, engineer, mechanic, programmer, debugger, accountant`
- `v_edgy` positive: `anarchist, rebel, destroyer, provocateur, daredevil, maverick, trickster, rogue`
- `v_edgy` negative: `archivist, validator, screener, moderator, librarian, secretary, coordinator, supervisor`

Final panel: 6 anchor-pair axes + `v_assistant`, 96 unique anchors total.

## User-discourse evidence

- Sycophancy: ~58% of LLM responses sycophantic ([UNU C3](https://c3.unu.edu/blog/how-sycophancy-shapes-the-reliability-of-large-language-models)); covered by `v_critic`.
- Preachy/moralistic: ChatGPT 5.2 "Karen persona" complaints ([Vertu](https://vertu.com/lifestyle/why-is-chatgpt-5-2-so-argumentative-the-rise-of-the-karen-ai-persona/)); overlaps `v_humor` / `v_critic`.
- Warmth: Claude described as "personable, empathetic" vs GPT-4 "formal" ([Tom's Guide](https://www.tomsguide.com/ai/i-tested-chatgpt-vs-gemini-vs-claude-for-emotional-intelligence-heres-the-winner)); overlaps `v_benevolence`.
- Edgy: Grok positioned vs ChatGPT — "fewer constraints, edgy, sarcastic" ([Glory Webs](https://www.glorywebs.com/blog/grok-vs-chatgpt)); mapped by `v_edgy`.
