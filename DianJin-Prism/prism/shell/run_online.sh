#!/bin/bash
set -euo pipefail
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH}
SEEDS=${SEEDS:-data/train_prompts.jsonl}
OUT=${OUT:-outputs/prism_online}
GPUS=${GPUS:-0}
PROCESSES=${PROCESSES:-1}
GRAD_ACCUM=${GRAD_ACCUM:-1}
EPOCHS=${EPOCHS:-1}
LR=${LR:-3e-5}
NUM_STEPS=${NUM_STEPS:-50}
REGEN_NUM_STEPS=${REGEN_NUM_STEPS:-50}
WIDTH=${WIDTH:-1024}
HEIGHT=${HEIGHT:-1024}
GEN_MAX_NEW_TOKENS=${GEN_MAX_NEW_TOKENS:-8192}
AUDIT_MAX_NEW_TOKENS=${AUDIT_MAX_NEW_TOKENS:-4096}
MAX_IMAGES=${MAX_IMAGES:-8}
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-2}
KEEP_STEP_CKPTS=${KEEP_STEP_CKPTS:-2}
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES="$GPUS" python -m accelerate.commands.launch \
  --num_processes "$PROCESSES" --config_file prism/configs/accelerate/zero2.yaml \
  prism/scripts/run_online.py --accelerate \
  --model_path "$MODEL_PATH" --seeds "$SEEDS" --out "$OUT" \
  --lora_rank 64 --lora_alpha 128 --grad_accum "$GRAD_ACCUM" \
  --epochs "$EPOCHS" --lr "$LR" --num_steps "$NUM_STEPS" \
  --regen_num_steps "$REGEN_NUM_STEPS" --width "$WIDTH" --height "$HEIGHT" \
  --gen_max_new_tokens "$GEN_MAX_NEW_TOKENS" \
  --audit_max_new_tokens "$AUDIT_MAX_NEW_TOKENS" --max_images "$MAX_IMAGES" \
  --save_every_steps "$SAVE_EVERY_STEPS" --keep_step_ckpts "$KEEP_STEP_CKPTS"
