#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 RUN_NAME POLICY_VARIANT" >&2
  exit 2
fi

RUN_NAME="$1"
POLICY_VARIANT="$2"
case "$POLICY_VARIANT" in
  concat_lstm|crossattn_lstm|crossattn_sru) ;;
  *)
    echo "Unknown policy variant: $POLICY_VARIANT" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRIPT_DIR="$REPO_ROOT/legged_gym/legged_gym/scripts"
PROJECT_NAME="${PROJECT_NAME:-aaai_teacher32500}"
TEACHER_RUN="${TEACHER_RUN:-Tea-ps2-mlp_armature_53_noise_reward_dof}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-32500}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
PHYSICAL_GPU="${PHYSICAL_GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
CONDA_SETUP="${CONDA_SETUP:-/home/ps/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-txc_go2parkour}"

LOG_ROOT="$REPO_ROOT/legged_gym/logs/$PROJECT_NAME"
RUN_DIR="$LOG_ROOT/$RUN_NAME"
TEACHER_PATH="$LOG_ROOT/$TEACHER_RUN/model_${TEACHER_CHECKPOINT}.pt"

if [[ ! -s "$TEACHER_PATH" ]]; then
  echo "Missing Teacher checkpoint: $TEACHER_PATH" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing run: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
source "$CONDA_SETUP"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export PYTHONPATH="$REPO_ROOT/legged_gym:$REPO_ROOT/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"

cd "$SCRIPT_DIR"
exec python train.py \
  --task go2 \
  --exptid "$RUN_NAME" \
  --proj_name "$PROJECT_NAME" \
  --device "$DEVICE" \
  --seed 1 \
  --max_iterations "$MAX_ITERATIONS" \
  --save_interval "$SAVE_INTERVAL" \
  --reward_profile current_full \
  --domain_rand_profile teacher32500 \
  --policy_variant "$POLICY_VARIANT" \
  --resume \
  --resumeid "$TEACHER_RUN" \
  --checkpoint "$TEACHER_CHECKPOINT" \
  --use_camera \
  --delay \
  --no_wandb
