#!/bin/bash
MODEL=Qwen/Qwen3-1.7B
OUTPUT_DIR=Qwen3-1.7B
BATCH_SIZE=11
MODEL_CLASS=qwen3
NPROC=1
MAX_LEN=32768
GEN_LEN=30768
TASK="aime24"
python ref_hf.py \
    --model "$MODEL" \
    --model_class "$MODEL_CLASS" \
    --nproc "$NPROC" \
    --batch_size "$BATCH_SIZE" \
    --max_len "$MAX_LEN" \
    --gen_len "$GEN_LEN" \
    --output_dir "$OUTPUT_DIR" \
    --task "$TASK"