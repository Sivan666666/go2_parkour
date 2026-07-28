#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRIPT_DIR="$REPO_ROOT/legged_gym/legged_gym/scripts"
MANIFEST="$SCRIPT_DIR/manifests/table_i_strict_teacher_v1.json"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/legged_gym/results/aaai_ablation/table_i_teacher325_students_ckpt2000_3000_v1}"
PROJECT_NAME="${PROJECT_NAME:-aaai_teacher32500}"
DEVICE="${DEVICE:-cuda:0}"
EPISODES="${EPISODES:-200}"

runs=(
  "A|AAAI325-A-CONCAT-LSTM-E176-S1|concat_lstm"
  "B|AAAI325-B-CROSSATTN-LSTM-E176-S1|crossattn_lstm"
  "C|AAAI325-C-CROSSATTN-SRU-E176-S1|crossattn_sru"
)
checkpoints=(2000 3000)

mkdir -p "$OUTPUT_ROOT"
cp "$MANIFEST" "$OUTPUT_ROOT/manifest.json"

for checkpoint in "${checkpoints[@]}"; do
  for spec in "${runs[@]}"; do
    IFS="|" read -r method run_name variant <<<"$spec"
    output_dir="$OUTPUT_ROOT/${method}_ckpt${checkpoint}"
    echo "[$(date --iso-8601=seconds)] evaluating $method checkpoint $checkpoint"
    python "$SCRIPT_DIR/run_eval_suite.py" \
      --exptid "$run_name" \
      --checkpoint "$checkpoint" \
      --reward_profile current_full \
      --domain_rand_profile teacher32500 \
      --policy_variant "$variant" \
      --suite table_i \
      --manifest "$MANIFEST" \
      --episodes "$EPISODES" \
      --device "$DEVICE" \
      --proj_name "$PROJECT_NAME" \
      --output_dir "$output_dir" \
      --use_camera \
      --delay
  done
done

echo "[$(date --iso-8601=seconds)] all Table I evaluations completed"
