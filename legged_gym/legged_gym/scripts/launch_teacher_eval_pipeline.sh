#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRIPT_DIR="$REPO_ROOT/legged_gym/legged_gym/scripts"
LOG_ROOT="$REPO_ROOT/legged_gym/logs/aaai_ablation"
RESULT_ROOT="$REPO_ROOT/legged_gym/results/aaai_ablation"
MANIFEST="$SCRIPT_DIR/manifests/hollow_stairs_v1.json"
CONDA_SETUP="/home/ps/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-txc_go2parkour}"
PIPELINE_LOG="$LOG_ROOT/teacher_eval_pipeline.log"

TEACHER_RUNS=(
  "AAAI-T0-EP-S1"
  "AAAI-T1-EP3-S1"
  "AAAI-T2-FULL-S1"
)
TEACHER_PROFILES=(
  "ep"
  "ep_plus_three"
  "current_full"
)

mkdir -p "$LOG_ROOT" "$RESULT_ROOT"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

wait_for_sessions() {
  local sessions=("$@")
  while true; do
    local active=0
    for session in "${sessions[@]}"; do
      if tmux has-session -t "$session" 2>/dev/null; then
        active=$((active + 1))
      fi
    done
    if [[ "$active" -eq 0 ]]; then
      return
    fi
    echo "$(timestamp) waiting for $active session(s): ${sessions[*]}"
    sleep 60
  done
}

start_eval_job() {
  local session="$1"
  local gpu="$2"
  local run="$3"
  local checkpoint="$4"
  local profile="$5"
  local suite="$6"
  local output_dir="$7"
  mkdir -p "$output_dir"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "bash -lc 'source $CONDA_SETUP && conda activate $CONDA_ENV && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH} && cd $SCRIPT_DIR && PYTHONUNBUFFERED=1 python run_eval_suite.py --exptid $run --checkpoint $checkpoint --reward_profile $profile --suite $suite --manifest $MANIFEST --episodes 200 --device cuda:$gpu --proj_name aaai_ablation --output_dir $output_dir > $output_dir/eval.log 2>&1'"
}

echo "$(timestamp) waiting for teacher training"
wait_for_sessions "${TEACHER_RUNS[@]}"
for run in "${TEACHER_RUNS[@]}"; do
  checkpoint="$LOG_ROOT/$run/model_5000.pt"
  if [[ ! -s "$checkpoint" ]]; then
    echo "Missing teacher checkpoint: $checkpoint" >&2
    exit 1
  fi
done
echo "$(timestamp) all teacher checkpoints are ready"

source "$CONDA_SETUP"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$SCRIPT_DIR"
python evaluate_ablation.py \
  --task go2 \
  --exptid AAAI-T2-FULL-S1 \
  --proj_name aaai_ablation \
  --checkpoint 5000 \
  --reward_profile current_full \
  --eval_condition regular_d1 \
  --episodes 32 \
  --output_dir "$RESULT_ROOT/eval_smoke" \
  --device cuda:3 \
  --headless \
  --no_wandb

echo "$(timestamp) starting experiment-3 quick evaluation"
quick_sessions=()
for index in 0 1 2; do
  session="AAAI-EVAL-QUICK-T$index"
  quick_sessions+=("$session")
  start_eval_job \
    "$session" \
    "$index" \
    "${TEACHER_RUNS[$index]}" \
    5000 \
    "${TEACHER_PROFILES[$index]}" \
    quick \
    "$RESULT_ROOT/experiment3_quick/${TEACHER_RUNS[$index]}"
done
wait_for_sessions "${quick_sessions[@]}"
for index in 0 1 2; do
  summaries="$RESULT_ROOT/experiment3_quick/${TEACHER_RUNS[$index]}"
  if [[ "$(find "$summaries" -name '*.summary.json' | wc -l)" -ne 3 ]]; then
    echo "Quick evaluation failed for ${TEACHER_RUNS[$index]}" >&2
    exit 1
  fi
done
python summarize_ablation.py \
  "$RESULT_ROOT/experiment3_quick" \
  --output "$RESULT_ROOT/experiment3_quick/summary.json" \
  --compare "AAAI-T0-EP-S1:AAAI-T1-EP3-S1" \
  --compare "AAAI-T1-EP3-S1:AAAI-T2-FULL-S1"
echo "$(timestamp) experiment-3 quick report is ready"

echo "$(timestamp) starting full experiment-3 evaluation"
teacher_full_sessions=()
for index in 0 1 2; do
  session="AAAI-EVAL-FULL-T$index"
  teacher_full_sessions+=("$session")
  start_eval_job \
    "$session" \
    "$index" \
    "${TEACHER_RUNS[$index]}" \
    5000 \
    "${TEACHER_PROFILES[$index]}" \
    full \
    "$RESULT_ROOT/experiment3_full/${TEACHER_RUNS[$index]}"
done
wait_for_sessions "${teacher_full_sessions[@]}"
python summarize_ablation.py \
  "$RESULT_ROOT/experiment3_full" \
  --output "$RESULT_ROOT/experiment3_full/summary.json" \
  --compare "AAAI-T0-EP-S1:AAAI-T1-EP3-S1" \
  --compare "AAAI-T1-EP3-S1:AAAI-T2-FULL-S1"

echo "$(timestamp) teacher evaluations and reports are complete"
