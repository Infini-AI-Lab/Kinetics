#!/bin/bash
MODEL=Qwen/Qwen3-0.6B
OUTPUT_DIR=Qwen3-0.6B
MODEL_CLASS=qwen3
MAX_LEN=5120
GEN_LEN=4096
SEED=123
PORT=$((29500 + SEED))
TRIAL=1
TASK="gsm8k"
OMP_NUM_THREADS=64 torchrun  --master-port=$PORT --nproc_per_node=2 eval_math_dist.py \
    --model "$MODEL" \
    --model_class "$MODEL_CLASS" \
    --max_len "$MAX_LEN" \
    --gen_len "$GEN_LEN" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"\
    --task "$TASK"\
    --trial "$TRIAL"