#!/bin/bash
MODEL=Qwen/Qwen3-8B
OUTPUT_DIR=Qwen3-8B
MODEL_CLASS=qwen3
NPROC=8
SEED=123
MAX_LEN=32768
GEN_LEN=30500
TRIAL=32
python eval_lcb.py \
    --model "$MODEL" \
    --model_class "$MODEL_CLASS" \
    --nproc "$NPROC" \
    --max_len "$MAX_LEN" \
    --gen_len "$GEN_LEN" \
    --output_dir "$OUTPUT_DIR" \
    --trial "$TRIAL" \
    --seed "$SEED"