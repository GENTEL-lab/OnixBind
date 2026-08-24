#!/usr/bin/env bash
# Run the bundled example, or a directory of AF3 JSON records.
#
#   bash predict.sh                       # the bundled example
#   bash predict.sh /path/to/af3_inputs   # your own records
set -euo pipefail

: "${CUTLASS_PATH:?set CUTLASS_PATH to a cutlass checkout, see docs/installation.md}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-fast_layernorm}"
export USE_DEEPSPEED_EVO_ATTENTION="${USE_DEEPSPEED_EVO_ATTENTION:-true}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/onixbind/torch_extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="${1:-$HERE/examples/5S8I_A.json}"
OUTPUT="${2:-$HERE/../output}"

python "$HERE/run_onixbind.py" "$INPUT" \
  --out_dir "$OUTPUT" \
  --weights "$HERE/weights" \
  --skip_completed
  # --save_features
