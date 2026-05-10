"""Contrast-axis definitions — the primary researcher knob.

To run a different axis panel:
  cp configs/axes.py configs/my_axes.py  # edit
  PHASE_H_AXES=configs/my_axes.py python compute_axes.py ...

`v_assistant` is the canonical (default vs role-mean) axis from Lu et al. — it
has no anchor list because it's defined by the default activation, not by
anchor groups. It is included in AXIS_ORDER but not in ANCHOR_AXES.

Each anchor axis: v_X = mean(positive-pole roles) − mean(negative-pole roles).
"""

AXIS_ORDER = ["v_assistant", "v_benevolence", "v_authority", "v_humor", "v_critic", "v_mystical", "v_edgy"]

ANCHOR_AXES: dict[str, tuple[list[str], list[str]]] = {
    "v_benevolence": (
        ["counselor", "parent", "guardian", "pacifist", "peacekeeper", "altruist", "healer", "angel"],
        ["criminal", "saboteur", "narcissist", "zealot", "hoarder", "smuggler", "demon", "predator"],
    ),
    "v_authority": (
        ["judge", "scientist", "ambassador", "polymath", "virtuoso", "sage", "leviathan", "ancient"],
        ["amateur", "dilettante", "student", "infant", "refugee", "prey", "prisoner", "orphan"],
    ),
    "v_humor": (
        ["comedian", "jester", "fool", "absurdist", "bohemian", "surfer", "improviser", "bard"],
        ["philosopher", "mathematician", "ascetic", "scholar", "hermit", "traditionalist", "conservator", "statistician"],
    ),
    "v_critic": (
        ["contrarian", "devils_advocate", "skeptic", "cynic", "perfectionist", "evaluator", "auditor", "examiner"],
        ["synthesizer", "optimist", "idealist", "evangelist", "romantic", "advocate", "facilitator", "instructor"],
    ),
    "v_mystical": (
        ["eldritch", "void", "oracle", "wraith", "shaman", "mystic", "witch", "genie"],
        ["realist", "pragmatist", "technologist", "engineer", "mechanic", "programmer", "debugger", "accountant"],
    ),
    "v_edgy": (
        ["anarchist", "rebel", "destroyer", "provocateur", "daredevil", "maverick", "trickster", "rogue"],
        ["archivist", "validator", "screener", "moderator", "librarian", "secretary", "coordinator", "supervisor"],
    ),
}
