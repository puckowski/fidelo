#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

STARTING_PROMPT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --starting-prompt)
      if [ $# -lt 2 ]; then
        echo "Missing value for --starting-prompt" >&2
        exit 1
      fi
      STARTING_PROMPT="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./scripts/run_tokenizer_reconstruction_opencode_loop.sh [--starting-prompt "extra guidance"]

Options:
  --starting-prompt  Extra initial guidance appended to each opencode iteration prompt.
  -h, --help         Show this help text.

Most other settings are configured through environment variables, such as MODEL, ITERATIONS,
TRAIN_EPOCHS, TRAIN_LR, TEST_OUTPUT_SECONDS, EXTRA_TRAIN_ARGS, and EXTRA_TEST_ARGS.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

MODEL="${MODEL:-ollama/qwen3.6:latest}"
PYTHON_CMD="${PYTHON_CMD:-/c/Users/fooba/anaconda3/python.exe}"
ITERATIONS="${ITERATIONS:-50}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-2}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-1e-5}"
VAL_RATIO="${VAL_RATIO:-0.02}"
BASE_TOKENIZER_DIR="${BASE_TOKENIZER_DIR:-latent_audio_tokenizer_out}"
OUTPUT_ROOT="${OUTPUT_ROOT:-tokenizer_reconstruction_loop_runs}"
TEST_OUTPUT_SECONDS="${TEST_OUTPUT_SECONDS:-3.33}"
INPUT_AUDIO="${INPUT_AUDIO:-}"
ALLOW_CPU="${ALLOW_CPU:-0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_TEST_ARGS="${EXTRA_TEST_ARGS:-}"
TRAIN_FAILURE_STREAK=0
COMPILE_FAILURE_STREAK=0
STASH_HELPER="$REPO_ROOT/scripts/stash_repo_changes.py"

if ! command -v opencode >/dev/null 2>&1; then
  echo "Missing required command: opencode" >&2
  exit 1
fi

if [ ! -x "$PYTHON_CMD" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_CMD="$(command -v python)"
  else
    echo "Could not find a usable Python interpreter. Set PYTHON_CMD first." >&2
    exit 1
  fi
fi

if [ ! -d "$BASE_TOKENIZER_DIR" ]; then
  echo "Base tokenizer directory not found: $BASE_TOKENIZER_DIR" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$REPO_ROOT/$OUTPUT_ROOT/run_$RUN_ID"
mkdir -p "$RUN_DIR"

REPORT_FILE="$RUN_DIR/EXPERIMENT_LOG.md"
BASELINE_DIR="$RUN_DIR/baseline"
mkdir -p "$BASELINE_DIR"

relative_path() {
  local abs="$1"
  abs="${abs#$REPO_ROOT/}"
  printf '%s' "$abs"
}

extract_metric() {
  local metric_name="$1"
  local log_path="$2"
  awk -F': ' -v name="$metric_name" '$1 == name {print $2; exit}' "$log_path"
}

metric_is_better() {
  local old_mae="$1"
  local old_mse="$2"
  local new_mae="$3"
  local new_mse="$4"
  "$PYTHON_CMD" - "$old_mae" "$old_mse" "$new_mae" "$new_mse" <<'PY'
import sys
old_mae, old_mse, new_mae, new_mse = map(float, sys.argv[1:5])
eps = 1e-9
is_better = (new_mae < old_mae - eps) or (abs(new_mae - old_mae) <= eps and new_mse < old_mse - eps)
print("yes" if is_better else "no")
PY
}

append_report_block() {
  local iteration_label="$1"
  local model_dir="$2"
  local train_log="$3"
  local recon_log="$4"
  local mae="$5"
  local mse="$6"
  local improved="$7"
  local notes="$8"
  cat >> "$REPORT_FILE" <<EOF

### $iteration_label

- Tokenizer directory: $(relative_path "$model_dir")
- Train log: $(relative_path "$train_log")
- Reconstruction log: $(relative_path "$recon_log")
- MAE: $mae
- MSE: $mse
- Improved best: $improved
- Loop note: $notes
EOF
}

run_reconstruction() {
  local tokenizer_dir="$1"
  local output_dir="$2"
  local log_path="$3"
  mkdir -p "$output_dir"
  local cmd=(
    "$PYTHON_CMD" "./test_latent_tokenizer_reconstruction.py"
    --tokenizer-dir "$tokenizer_dir"
    --output-dir "$output_dir"
    --output-seconds "$TEST_OUTPUT_SECONDS"
  )
  if [ -n "$INPUT_AUDIO" ]; then
    cmd+=(--input-audio "$INPUT_AUDIO")
  fi
  if [ "$ALLOW_CPU" = "1" ]; then
    cmd+=(--allow-cpu)
  fi
  if [ -n "$EXTRA_TEST_ARGS" ]; then
    # shellcheck disable=SC2206
    local extra_test_parts=( $EXTRA_TEST_ARGS )
    cmd+=("${extra_test_parts[@]}")
  fi
  "${cmd[@]}" | tee "$log_path"
}

cat > "$REPORT_FILE" <<EOF
# Tokenizer Reconstruction Improvement Loop

- Started: $(date -Iseconds)
- Model harness: opencode
- Local model: $MODEL
- Python: $PYTHON_CMD
- Base tokenizer dir: $(relative_path "$BASE_TOKENIZER_DIR")
- Iterations requested: $ITERATIONS
- Train epochs per iteration: $TRAIN_EPOCHS
- Reconstruction output seconds: $TEST_OUTPUT_SECONDS
- Starting prompt guidance: ${STARTING_PROMPT:-<none>}

## Baseline
EOF

BASELINE_RECON_LOG="$BASELINE_DIR/reconstruction.log"
run_reconstruction "$BASE_TOKENIZER_DIR" "$BASELINE_DIR/reconstruction_files" "$BASELINE_RECON_LOG"

BEST_MAE="$(extract_metric "MAE" "$BASELINE_RECON_LOG")"
BEST_MSE="$(extract_metric "MSE" "$BASELINE_RECON_LOG")"
BEST_TOKENIZER_DIR="$REPO_ROOT/$BASE_TOKENIZER_DIR"

append_report_block "Baseline" "$BEST_TOKENIZER_DIR" "$BASELINE_RECON_LOG" "$BASELINE_RECON_LOG" "$BEST_MAE" "$BEST_MSE" "yes" "Initial checkpoint before loop edits"

stash_repo_changes_after_failures() {
  local iteration_label="$1"
  local reason_message="$2"
  "$PYTHON_CMD" "$STASH_HELPER" \
    --repo-root "$REPO_ROOT" \
    --exclude-prefix "$OUTPUT_ROOT" \
    --exclude-prefix "scripts/stash_repo_changes.py" \
    --exclude-prefix "scripts/run_tokenizer_reconstruction_opencode_loop.sh" \
    --message "$reason_message"
}

compile_changed_python_files() {
  local log_path="$1"
  local -a tracked_python_files=()
  local -a untracked_python_files=()
  local -a python_files=()
  local file_path

  mapfile -t tracked_python_files < <(git diff --name-only --diff-filter=ACM -- '*.py' || true)
  mapfile -t untracked_python_files < <(git ls-files --others --exclude-standard -- '*.py' || true)

  for file_path in "${tracked_python_files[@]}" "${untracked_python_files[@]}"; do
    if [ -z "$file_path" ]; then
      continue
    fi
    if [ ! -f "$file_path" ]; then
      continue
    fi
    python_files+=("$file_path")
  done

  if [ ${#python_files[@]} -eq 0 ]; then
    echo "No changed Python files to compile." | tee "$log_path"
    return 0
  fi

  printf 'Compiling changed Python files:\n' | tee "$log_path"
  printf ' - %s\n' "${python_files[@]}" | tee -a "$log_path"
  "$PYTHON_CMD" -m py_compile "${python_files[@]}" 2>&1 | tee -a "$log_path"
}

for ((iteration = 1; iteration <= ITERATIONS; iteration++)); do
  ITER_LABEL="$(printf '%03d' "$iteration")"
  ITER_DIR="$RUN_DIR/iteration_$ITER_LABEL"
  mkdir -p "$ITER_DIR"

  PROMPT_FILE="$ITER_DIR/opencode_prompt.txt"
  STATUS_FILE="$ITER_DIR/git_status_after_agent.txt"
  DIFF_FILE="$ITER_DIR/git_diff_after_agent.patch"
  COMPILE_LOG="$ITER_DIR/compile.log"
  TRAIN_LOG="$ITER_DIR/train.log"
  RECON_LOG="$ITER_DIR/reconstruction.log"
  ITER_TOKENIZER_DIR="$ITER_DIR/tokenizer_out"
  RECON_OUT_DIR="$ITER_DIR/reconstruction_files"

  cat > "$PROMPT_FILE" <<EOF
You are working in the music4 repository.

Goal:
Improve tokenizer reconstruction quality measured by test_latent_tokenizer_reconstruction.py.

Current best metrics:
- MAE: $BEST_MAE
- MSE: $BEST_MSE

Constraints:
- Only use local code and local inference. No cloud APIs.
- Prefer small, targeted edits.
- Focus on files directly relevant to tokenizer quality, especially:
  - train_latent_audio_tokenizer.py
  - latent_audio_token_pipeline.py
  - test_latent_tokenizer_reconstruction.py
- Keep CLI compatibility unless there is a strong reason to change it.
- Do not run long training loops yourself; the surrounding harness will train and evaluate.

Task:
1. Make one focused code improvement aimed at better reconstruction quality.
2. Append a markdown section to $(relative_path "$REPORT_FILE") with heading "Agent Iteration $ITER_LABEL".
3. In that section, write:
   - what code you changed
   - why you think it can improve reconstruction
   - what risk or tradeoff it introduces
4. Stop after the code edit and markdown update.

Additional initial guidance:
${STARTING_PROMPT:-<none>}

The harness will fine-tune the tokenizer next and then append measured MAE/MSE.
EOF

  echo "[$(date -Iseconds)] Agent iteration $ITER_LABEL" | tee -a "$TRAIN_LOG"
  if ! opencode run -m "$MODEL" "$(cat "$PROMPT_FILE")" | tee "$ITER_DIR/opencode.log"; then
    append_report_block "Iteration $ITER_LABEL" "$BEST_TOKENIZER_DIR" "$ITER_DIR/opencode.log" "$ITER_DIR/opencode.log" "$BEST_MAE" "$BEST_MSE" "no" "opencode failed before training"
    continue
  fi

  git status --short > "$STATUS_FILE" || true
  git diff --binary > "$DIFF_FILE" || true

  if ! compile_changed_python_files "$COMPILE_LOG"; then
    COMPILE_FAILURE_STREAK=$((COMPILE_FAILURE_STREAK + 1))
    NOTE="python compile check failed"
    if [ "$COMPILE_FAILURE_STREAK" -ge 2 ]; then
      STASH_MESSAGE="Auto stash after two consecutive Python compile failures at iteration $ITER_LABEL"
      if stash_repo_changes_after_failures "$ITER_LABEL" "$STASH_MESSAGE" | tee -a "$COMPILE_LOG"; then
        NOTE="python compile check failed; stashed code changes after two consecutive compile failures"
      else
        NOTE="python compile check failed; attempted stash after two consecutive compile failures but stash command failed"
      fi
      COMPILE_FAILURE_STREAK=0
    fi
    append_report_block "Iteration $ITER_LABEL" "$BEST_TOKENIZER_DIR" "$COMPILE_LOG" "$COMPILE_LOG" "$BEST_MAE" "$BEST_MSE" "no" "$NOTE"
    continue
  fi

  COMPILE_FAILURE_STREAK=0

  TRAIN_CMD=(
    "$PYTHON_CMD" "./train_latent_audio_tokenizer.py"
    --finetune-from "$BEST_TOKENIZER_DIR"
    --out-dir "$ITER_TOKENIZER_DIR"
    --epochs "$TRAIN_EPOCHS"
    --batch-size "$TRAIN_BATCH_SIZE"
    --grad-accum-steps "$TRAIN_GRAD_ACCUM"
    --lr "$TRAIN_LR"
    --weight-decay "$TRAIN_WEIGHT_DECAY"
    --val-ratio "$VAL_RATIO"
  )

  if [ "$ALLOW_CPU" = "1" ]; then
    TRAIN_CMD+=(--allow-cpu)
  fi
  if [ -n "$EXTRA_TRAIN_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_TRAIN_PARTS=( $EXTRA_TRAIN_ARGS )
    TRAIN_CMD+=("${EXTRA_TRAIN_PARTS[@]}")
  fi

  if ! "${TRAIN_CMD[@]}" | tee "$TRAIN_LOG"; then
    TRAIN_FAILURE_STREAK=$((TRAIN_FAILURE_STREAK + 1))
    NOTE="training failed"
    if [ "$TRAIN_FAILURE_STREAK" -ge 2 ]; then
      STASH_MESSAGE="Auto stash after two consecutive tokenizer training failures at iteration $ITER_LABEL"
      if stash_repo_changes_after_failures "$ITER_LABEL" "$STASH_MESSAGE" | tee -a "$TRAIN_LOG"; then
        NOTE="training failed; stashed code changes after two consecutive failures"
      else
        NOTE="training failed; attempted stash after two consecutive failures but stash command failed"
      fi
      TRAIN_FAILURE_STREAK=0
    fi
    append_report_block "Iteration $ITER_LABEL" "$BEST_TOKENIZER_DIR" "$TRAIN_LOG" "$TRAIN_LOG" "$BEST_MAE" "$BEST_MSE" "no" "$NOTE"
    continue
  fi

  TRAIN_FAILURE_STREAK=0

  if ! run_reconstruction "$ITER_TOKENIZER_DIR" "$RECON_OUT_DIR" "$RECON_LOG"; then
    append_report_block "Iteration $ITER_LABEL" "$ITER_TOKENIZER_DIR" "$TRAIN_LOG" "$RECON_LOG" "$BEST_MAE" "$BEST_MSE" "no" "reconstruction evaluation failed"
    continue
  fi

  ITER_MAE="$(extract_metric "MAE" "$RECON_LOG")"
  ITER_MSE="$(extract_metric "MSE" "$RECON_LOG")"
  IMPROVED="$(metric_is_better "$BEST_MAE" "$BEST_MSE" "$ITER_MAE" "$ITER_MSE")"

  if [ "$IMPROVED" = "yes" ]; then
    BEST_MAE="$ITER_MAE"
    BEST_MSE="$ITER_MSE"
    BEST_TOKENIZER_DIR="$ITER_TOKENIZER_DIR"
    NOTE="new best tokenizer"
  else
    NOTE="kept previous best tokenizer"
  fi

  append_report_block "Iteration $ITER_LABEL" "$ITER_TOKENIZER_DIR" "$TRAIN_LOG" "$RECON_LOG" "$ITER_MAE" "$ITER_MSE" "$IMPROVED" "$NOTE"
done

cat >> "$REPORT_FILE" <<EOF

## Final Best

- Best tokenizer dir: $(relative_path "$BEST_TOKENIZER_DIR")
- Best MAE: $BEST_MAE
- Best MSE: $BEST_MSE
- Finished: $(date -Iseconds)
EOF

echo "Loop finished. Report written to $(relative_path "$REPORT_FILE")"