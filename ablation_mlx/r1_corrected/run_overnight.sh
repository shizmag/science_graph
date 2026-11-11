#!/bin/bash
set -Eeuo pipefail

ROOT="/Volumes/happy-disk/codex-worktrees/r1-mlx-structured-context"
PYTHON="/Volumes/happy-disk/dev/uv/venvs/r1-mlx-structured-context/bin/python"
MODEL="/Volumes/happy-disk/models/llm/OCC-RAG-1.7B"
DATABASE="/Users/vladimirkasterin/python/graph/ablation_portability/snapshots/candidates/external_db/v1/database.db"
SOURCE="$ROOT/revision_round1/experiments/r1_mlx_corrected"
RUNTIME="$ROOT/revision_round1/experiments/r1_mlx_corrected_runtime_2026-11-12_full_v2"
MANIFEST="dffd5b400b86035eb270961496399f90495b122891da9e2e0eb37a9fcc7b4607"
LOG_DIR="$RUNTIME/logs"
VALIDATION_MARKER="$RUNTIME/.live-validation-passed"
SKIP_LIVE_VALIDATION="${R1_SKIP_LIVE_VALIDATION:-0}"

cd "$ROOT"
test -x "$PYTHON"; test -d "$MODEL"; test -f "$DATABASE"; test -d "$SOURCE"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/overnight-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

run() { "$PYTHON" -m ablation_mlx.r1_corrected.cli --model "$MODEL" --database "$DATABASE" --artifact-root "$SOURCE" --runtime-artifact-root "$RUNTIME" "$@"; }

echo "R1 MLX overnight run; log=$LOG"
caffeinate -dimsu -w $$ & CAFFEINATE_PID=$!
trap 'kill "$CAFFEINATE_PID" 2>/dev/null || true' EXIT
run preflight
run status --manifest-hash "$MANIFEST"
run prepare-contexts --manifest-hash "$MANIFEST"
test "$(find "$RUNTIME/prepared_inputs" -type f -name '*.json' | wc -l | tr -d ' ')" = 600
run validate --manifest-hash "$MANIFEST"

if [[ "$SKIP_LIVE_VALIDATION" = 1 ]]; then
  test -f "$VALIDATION_MARKER" || { echo "refusing skip: $VALIDATION_MARKER is absent" >&2; exit 2; }
else
  # This is the only bounded real-model gate.  A failure exits because of
  # pipefail/set -e; it must never fall through to the full run.
  run generate-prepared --manifest-hash "$MANIFEST" --max-new-slots 2
  run validate --manifest-hash "$MANIFEST"
  # Validation must report predictive telemetry for both newly accepted rows.
  grep -q '"predictive_telemetry_valid_records": [1-9]' "$LOG"
  touch "$VALIDATION_MARKER"
fi

echo "Starting/resuming remaining sequential generation"
run generate-prepared --manifest-hash "$MANIFEST"
run validate --manifest-hash "$MANIFEST"
run status --manifest-hash "$MANIFEST"
echo "Final status complete; log=$LOG"
