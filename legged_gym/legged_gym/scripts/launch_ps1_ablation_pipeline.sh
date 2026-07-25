#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRIPT_DIR="$REPO_ROOT/legged_gym/legged_gym/scripts"
LOG_ROOT="$REPO_ROOT/legged_gym/logs/aaai_ablation"
RESULT_ROOT="$REPO_ROOT/legged_gym/results/aaai_ablation"
MANIFEST="$SCRIPT_DIR/manifests/hollow_stairs_v1.json"
CONDA_SETUP="/home/ps/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="txc_go2parkour"
PIPELINE_LOG="$LOG_ROOT/pipeline.log"

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
STUDENT_RUNS=(
  "AAAI-A-CONCAT-LSTM-S1"
  "AAAI-B-CROSSATTN-LSTM-S1"
  "AAAI-C-CROSSATTN-SRU-S1"
)
STUDENT_VARIANTS=(
  "concat_lstm"
  "crossattn_lstm"
  "crossattn_sru"
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
  local variant="${8:-}"
  local camera_args=""
  if [[ -n "$variant" ]]; then
    camera_args="--policy_variant $variant --use_camera --delay"
  fi
  mkdir -p "$output_dir"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "bash -lc 'source $CONDA_SETUP && conda activate $CONDA_ENV && cd $SCRIPT_DIR && PYTHONUNBUFFERED=1 python run_eval_suite.py --exptid $run --checkpoint $checkpoint --reward_profile $profile --suite $suite --manifest $MANIFEST --episodes 200 --device cuda:$gpu --proj_name aaai_ablation --output_dir $output_dir $camera_args > $output_dir/eval.log 2>&1'"
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

# A small post-training evaluation validates the terminal bookkeeping before
# committing all three GPUs to the paper table.
source "$CONDA_SETUP"
conda activate "$CONDA_ENV"
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

echo "$(timestamp) smoke-testing student variants"
for index in 0 1 2; do
  smoke_run="SMOKE-${STUDENT_RUNS[$index]}"
  mkdir -p "$LOG_ROOT/$smoke_run"
  python train.py \
    --task go2 \
    --exptid "$smoke_run" \
    --proj_name aaai_ablation \
    --device cuda:3 \
    --num_envs 16 \
    --rows 10 \
    --cols 4 \
    --seed 1 \
    --max_iterations 2 \
    --save_interval 1 \
    --reward_profile current_full \
    --policy_variant "${STUDENT_VARIANTS[$index]}" \
    --resume \
    --resumeid AAAI-T2-FULL-S1 \
    --checkpoint 5000 \
    --use_camera \
    --delay \
    --no_wandb \
    > "$LOG_ROOT/$smoke_run/run.log" 2>&1
  test -s "$LOG_ROOT/$smoke_run/model_2.pt"
done

echo "$(timestamp) starting A/B/C student training"
for index in 0 1 2; do
  run="${STUDENT_RUNS[$index]}"
  variant="${STUDENT_VARIANTS[$index]}"
  gpu="$index"
  if [[ -e "$LOG_ROOT/$run/model_1.pt" ]]; then
    echo "Refusing to overwrite existing student run: $run" >&2
    exit 1
  fi
  mkdir -p "$LOG_ROOT/$run"
  tmux kill-session -t "$run" 2>/dev/null || true
  tmux new-session -d -s "$run" \
    "bash -lc 'source $CONDA_SETUP && conda activate $CONDA_ENV && cd $SCRIPT_DIR && PYTHONUNBUFFERED=1 python train.py --task go2 --exptid $run --proj_name aaai_ablation --device cuda:$gpu --seed 1 --max_iterations 6000 --save_interval 500 --reward_profile current_full --policy_variant $variant --resume --resumeid AAAI-T2-FULL-S1 --checkpoint 5000 --use_camera --delay --no_wandb > $LOG_ROOT/$run/run.log 2>&1'"
done

monitor_session="AAAI-STUDENT-STOP-MONITOR"
tmux kill-session -t "$monitor_session" 2>/dev/null || true
tmux new-session -d -s "$monitor_session" \
  "bash -lc 'source $CONDA_SETUP && conda activate $CONDA_ENV && cd $SCRIPT_DIR && PYTHONUNBUFFERED=1 python monitor_student_ablation.py $LOG_ROOT/${STUDENT_RUNS[0]} $LOG_ROOT/${STUDENT_RUNS[1]} $LOG_ROOT/${STUDENT_RUNS[2]} > $LOG_ROOT/student_stop_monitor.log 2>&1'"

wait_for_sessions "${STUDENT_RUNS[@]}" "$monitor_session"
decision="$LOG_ROOT/${STUDENT_RUNS[0]}/common_stop_decision.json"
if [[ ! -s "$decision" ]]; then
  echo "Student common-stop decision was not produced" >&2
  exit 1
fi
student_checkpoint="$(
  python -c "import json; print(json.load(open('$decision'))['common_stop_iteration'])"
)"
for run in "${STUDENT_RUNS[@]}"; do
  if [[ ! -s "$LOG_ROOT/$run/model_${student_checkpoint}.pt" ]]; then
    echo "Missing common student checkpoint for $run: $student_checkpoint" >&2
    exit 1
  fi
done
echo "$(timestamp) students stopped at common iteration $student_checkpoint"

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

echo "$(timestamp) starting full experiment-1 evaluation"
student_eval_sessions=()
for index in 0 1 2; do
  session="AAAI-EVAL-STUDENT-$index"
  student_eval_sessions+=("$session")
  start_eval_job \
    "$session" \
    "$index" \
    "${STUDENT_RUNS[$index]}" \
    "$student_checkpoint" \
    current_full \
    full \
    "$RESULT_ROOT/experiment1_full/${STUDENT_RUNS[$index]}" \
    "${STUDENT_VARIANTS[$index]}"
done
wait_for_sessions "${student_eval_sessions[@]}"
python summarize_ablation.py \
  "$RESULT_ROOT/experiment1_full" \
  --output "$RESULT_ROOT/experiment1_full/summary.json" \
  --compare "AAAI-A-CONCAT-LSTM-S1:AAAI-B-CROSSATTN-LSTM-S1" \
  --compare "AAAI-B-CROSSATTN-LSTM-S1:AAAI-C-CROSSATTN-SRU-S1"

echo "$(timestamp) all requested experiments and reports are complete"
