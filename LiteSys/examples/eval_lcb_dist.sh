#!/bin/bash
MODEL=Qwen/Qwen3-32B
OUTPUT_DIR=Qwen3-32B
MODEL_CLASS=qwen3
MAX_LEN=32768
GEN_LEN=30500
SEED=123
PORT=$((29500 + SEED))
TRIAL=32
OMP_NUM_THREADS=64 torchrun  --master-port=$PORT --nproc_per_node=8 eval_lcb_dist.py \
    --model "$MODEL" \
    --model_class "$MODEL_CLASS" \
    --max_len "$MAX_LEN" \
    --gen_len "$GEN_LEN" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR" \
    --trial "$TRIAL"