#!/bin/bash
# Run the assistant-axis pipeline (no judge) for a single model.
# Usage:
#   ./pipeline.sh <hf_model_id> <output_dir> <layer> <hidden_dim>
# Example:
#   ./pipeline.sh microsoft/Phi-3.5-mini-instruct ./data/phi-3.5-mini 16 3072
#
# Steps:
#   1. vLLM generation: 276 roles × 1200 rollouts → responses/<role>.jsonl
#   2. Activation extraction at layer L → activations/<role>.pt
#   3. (skipped — judge not needed; see DESIGN.md)
#   4. Unfiltered role vectors (mean over all rollouts) → vectors/<role>.pt
#   5. Lu-style assistant axis (mean(default) − mean(roles)) → axis.pt
#   6. Default activation vector → default.pt
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <hf_model_id> <output_dir> <layer> <hidden_dim>" >&2
    exit 2
fi

MODEL_ID="$1"
OUTPUT="$(realpath "$2")"
LAYER="$3"
HIDDEN_DIM="$4"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

# Locate the assistant-axis library (must be pip-installed editable, or available on path)
PIPELINE_DIR="$($PYTHON -c 'import os, assistant_axis; print(os.path.dirname(os.path.dirname(assistant_axis.__file__)) + "/pipeline")')"
ROLES_DIR="$($PYTHON -c 'import os, assistant_axis; print(os.path.dirname(os.path.dirname(assistant_axis.__file__)) + "/data/roles/instructions")')"
QUESTIONS_FILE="$($PYTHON -c 'import os, assistant_axis; print(os.path.dirname(os.path.dirname(assistant_axis.__file__)) + "/data/extraction_questions.jsonl")')"

mkdir -p "$OUTPUT"

echo "=== Step 1/5: Generate responses (vLLM) ==="
$PYTHON "$PIPELINE_DIR/1_generate.py" \
    --model "$MODEL_ID" \
    --roles_dir "$ROLES_DIR" \
    --questions_file "$QUESTIONS_FILE" \
    --output_dir "$OUTPUT/responses" \
    --question_count 240 \
    --max_tokens 512

echo "=== Step 2/5: Extract activations at layer $LAYER ==="
$PYTHON "$PIPELINE_DIR/2_activations.py" \
    --model "$MODEL_ID" \
    --responses_dir "$OUTPUT/responses" \
    --output_dir "$OUTPUT/activations" \
    --layers "$LAYER" \
    --batch_size 32

echo "=== Step 3/5: Compute unfiltered role vectors (no judge) ==="
$PYTHON - "$OUTPUT/activations" "$OUTPUT/vectors" <<'PYEOF'
import sys
import torch
from pathlib import Path

act_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

n = 0
for af in sorted(act_dir.glob("*.pt")):
    role = af.stem
    activations = torch.load(af, map_location="cpu", weights_only=False)
    stacked = torch.stack([v.squeeze(0) if v.ndim > 1 else v for v in activations.values()])
    mean_vec = stacked.mean(dim=0)
    torch.save({"vector": mean_vec, "type": "mean", "role": role}, out_dir / f"{role}.pt")
    n += 1
print(f"Wrote {n} unfiltered role vectors → {out_dir}")
PYEOF

echo "=== Step 4/5: Compute Lu-style assistant axis ==="
$PYTHON "$PIPELINE_DIR/5_axis.py" \
    --vectors_dir "$OUTPUT/vectors" \
    --output "$OUTPUT/axis.pt"

echo "=== Step 5/5: Save default activation vector ==="
$PYTHON - "$OUTPUT/activations/default.pt" "$OUTPUT/default.pt" <<'PYEOF'
import sys
import torch
from pathlib import Path

act_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
activations = torch.load(act_path, map_location="cpu", weights_only=False)
default_vec = torch.stack([v.squeeze(0) if v.ndim > 1 else v for v in activations.values()]).mean(dim=0)
torch.save(default_vec, out_path)
print(f"Wrote default vector {tuple(default_vec.shape)} → {out_path}")
PYEOF

echo ""
echo "=== Done. Outputs in $OUTPUT ==="
echo "  responses/    role response JSONL files"
echo "  activations/  per-role activation dicts"
echo "  vectors/      per-role unfiltered mean vectors"
echo "  axis.pt       Lu-style assistant axis"
echo "  default.pt    default activation vector"
