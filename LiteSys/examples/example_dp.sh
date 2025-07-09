#!/bin/bash
MODEL=Qwen/Qwen3-0.6B
OUTPUT_DIR=Qwen3-0.6B
MODEL_CLASS=qwen3
NPROC=1
SEED=123
MAX_LEN=5120
GEN_LEN=4096
TASK="gsm8k"
TRIAL=1
python eval_math.py \
    --model "$MODEL" \
    --model_class "$MODEL_CLASS" \
    --nproc "$NPROC" \
    --max_len "$MAX_LEN" \
    --gen_len "$GEN_LEN" \
    --output_dir "$OUTPUT_DIR" \
    --task "$TASK"\
    --trial "$TRIAL"