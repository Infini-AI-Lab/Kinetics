# LiteSys

**An Inference/Evaluation Framework for Algorithm Developers**


## Introduction

If you:

- ✅ Want an inference framework that's easy to extend with new models and algorithms
- 🐢 Find Hugging Face Transformers too slow for long-generation evaluation
- 🎯 Feel painful to align performance and accuracy on benchmarks like AIME, AMC, or LiveCodeBench

<div align="center">
<h1><img src="assets/litesys.png" height="40px" align="top"/> LiteSys is a great choice.
</h1>

**LiteSys** is a lightweight, flexible serving system designed specifically for **academic research on large language models (LLMs)** (and implementation for [Kinetics: Rethinking Test-Time Scaling Laws](https://arxiv.org/abs/2506.05333)).

The goal of LiteSys is to offer a minimal yet efficient alternative to existing solutions:

- ⚡ **Faster than Hugging Face Transformers** for inference workloads, especially in batched decoding.
- 🧩 **Easier to modify than vLLM or SGLang**, making it ideal for prototyping custom model architectures, attention mechanisms, and scheduling policies.

LiteSys emphasizes **modularity**, **hackability**, and **ease of debugging**, without sacrificing key features like continuous batching, KV cache management, and model-parallel support.

Whether you're working on novel LLM execution strategies, experimenting with sparse attention, or evaluating training artifacts—LiteSys gives you full control with minimal overhead.

## Installation
```bash
pip install -r requirements.txt
```

For dryrun

**single GPU**
```bash
cd examples
bash example_dp.sh # Eval Qwen3-0.6B on GSM8K, 4K generation length. 
```
ETA: 25mins (L40 48GB), Pass@1: 74.4

**Multi GPU**
```bash
cd examples
bash example_tp.sh # Eval Qwen3-0.6B on GSM8K, 4K generation length. 
```
ETA: 16mins (2xL40 48GB), Pass@1: 74.3

## Reference Evaluation Results
**AIME24**

| Model                         | LiteSys⚡| Qwen3 Reported |
|:------------------------------|:-----------------------:|:----------------------------:|
| Qwen3-0.6B |          11.5           |            10.7             |
| Qwen3-1.7B   |        45.9           |            48.3             |
| Qwen3-4B  |          71.7           |             73.8             |
| Qwen3-8B  |          75.2           |             76             |
| Qwen3-14B  |         79.6           |             79.3             |
| Qwen3-32B |          82.1           |             81.4             |

**AIME25**

| Model                         | LiteSys⚡| Qwen3 Reported |
|:------------------------------|:-----------------------:|:----------------------------:|
| Qwen3-0.6B |          16.5           |            15.1             |
| Qwen3-1.7B   |        38.6           |            36.8             |
| Qwen3-4B  |          65.5           |             65.6             |
| Qwen3-8B  |          68.8           |             67.3            |
| Qwen3-14B  |         73.6           |             70.4             |
| Qwen3-32B |          72.9           |             72.9             |


**LiveCodeBench v5 (167 examples)**

| Model                         | LiteSys⚡| Qwen3 Reported |
|:------------------------------|:-----------------------:|:----------------------------:|
| Qwen3-1.7B   |        32.0           |            33.2             |
| Qwen3-4B  |          52.7          |              54.2            |
| Qwen3-8B  |          56.6           |             57.5            |
| Qwen3-14B  |         61.7           |             63.5             |
| Qwen3-32B |          63.8           |             65.7             |



## Examples

We provide evaluations for AIME24, AIME25, MATH500, GSM8K and LiveCodeBench v5.

#### Data Parallel
```bash
#!/bin/bash
MODEL=Qwen/Qwen3-8B
OUTPUT_DIR=Qwen3-8B
MODEL_CLASS=qwen3
NPROC=8
SEED=123
MAX_LEN=32768
GEN_LEN=30768
TASK="aime25"
TRIAL=4
python eval_math.py \
    --model "$MODEL" \
    --model_class "$MODEL_CLASS" \
    --nproc "$NPROC" \
    --max_len "$MAX_LEN" \
    --gen_len "$GEN_LEN" \
    --output_dir "$OUTPUT_DIR" \
    --task "$TASK"\
    --trial "$TRIAL"
```

#### Tensor Parallel
```bash
#!/bin/bash
MODEL=Qwen/Qwen3-32B
OUTPUT_DIR=Qwen3-32B
MODEL_CLASS=qwen3
MAX_LEN=32768
GEN_LEN=30768
NPROC=8
SEED=123
PORT=$((29500 + SEED))
TRIAL=4
TASK="aime25"
OMP_NUM_THREADS=64 torchrun  --master-port=$PORT --nproc_per_node=$NPROC eval_math_dist.py \
    --model "$MODEL" \
    --model_class "$MODEL_CLASS" \
    --max_len "$MAX_LEN" \
    --gen_len "$GEN_LEN" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"\
    --task "$TASK"\
    --trial "$TRIAL"
```
**Explanations**:

`--model`: Model name in Huggingface or local directory.

`--model_class`: Model architecture. `qwen2`, `qwen3` or `qwen3moe`. 

`--nproc`: Numbers of available GPUs. 

`--max_len`: Maximum lengths including prompts. 

`--gen_len`: Maximum generation lengths (excluding prompts). 

`--output_dir`: The directory where the evaluation results, including generation tokens and grades, are saved. 

`--task`: The evaluation task: `aime24`, `aime25`, `math500` or `gsm8k`. 

`--trial`: Number of reasoning trials. Finally, we will report `pass@1` and `pass@trial`. All generation results will be saved for later processing. 

For LiveCodeBench, we provide an additional python script for grading. 
To generate the responses
```bash
#!/bin/bash
MODEL=Qwen/Qwen3-8B
OUTPUT_DIR=Qwen3-8B
MODEL_CLASS=qwen3
NPROC=8
SEED=123
MAX_LEN=32768
GEN_LEN=30500
TRIAL=4
python eval_lcb.py \
    --model "$MODEL" \
    --model_class "$MODEL_CLASS" \
    --nproc "$NPROC" \
    --max_len "$MAX_LEN" \
    --gen_len "$GEN_LEN" \
    --output_dir "$OUTPUT_DIR" \
    --trial "$TRIAL" \
    --seed "$SEED"
```
This command will only generate responses without grading in a jsonl file under `--output_dir`.

To grade the responses
```bash
python grade_lcb.py --output_dir Qwen3-32B/ --output_file Qwen3-32B/lcb_results.jsonl
```

The above command will collect all jsonl files under `--output_dir` and grade together (e.g., we can generate the responses multiple times).

## Components

### Model Executor

The **Model Executor** defines how LLM layers or modules are applied to inputs—essentially controlling the model's forward workflow. Model implementations are located in `litesys/models/`.

Currently supported models:

- `Qwen2.5`
- `Qwen3`
- `Qwen3-MoE`  

These correspond to model classes:

- `"qwen2"`
- `"qwen3"`
- `"qwen3moe"`

#### To add a new model:

1. **Define layer parameters** in  
   `litesys/models/<new_model>_layer.py`.

2. **Implement execution logic** in  
   `litesys/models/<new_model>.py`.

3. **(Optional)** Add tensor-parallelism support in  
   `litesys/models/<new_model>_dist.py`.  
   > Recommended when model weights use more than 40% of GPU VRAM.

4. **Register the model** in  
   `litesys/models/auto_model.py`.

---

### KV Manager

The **KV Manager** handles both:

- **Prefill stage**: single-token, single-request input.
- **Decoding stage**: efficient batched generation.

Key features:

- Optimized for **Grouped Query Attention (GQA)**:
  - KV caches are loaded once per group.
- Exposes **raw attention logits** (not fused), enabling rapid experimentation with sparse attention ideas like:
  - Native Sparse Attention  
  - Block Sparse Attention  
  - Quest  
  - StreamingLLM  
  - Mixed-head sparse attention variants

This modular design supports flexible and efficient attention strategies. This is why we do not used paged attention/flash attention.

---

### Scheduler

The **Scheduler** supports **continuous batching**, meaning:

- As soon as a request finishes, a new one is fetched to fill its slot.
- Maximizes hardware utilization during long-generation workloads.

#### Repetition Detection

To avoid degenerate repetition loops (common in model evaluation), the scheduler supports early stopping:

| Parameter              | Description                                                     |
|------------------------|-----------------------------------------------------------------|
| `repeat_check`         | Enable or disable repetition detection                          |
| `repeat_check_window`  | How many recent tokens to scan for repeated patterns (e.g. 1024)|
| `repeat_block_size`    | Length of repeated chunks to look for (e.g. 64 tokens)          |

> A request is terminated early if the last `repeat_check_window` tokens contain a number of exactly repeated `repeat_block_size`-sized blocks (e.g., 16).

This is especially useful for evaluating newly trained or unstable models.

---

## Citation

If you use **LiteSys** in your research, please consider citing:

```bibtex
@misc{sadhukhan2025kineticsrethinkingtesttimescaling,
      title={Kinetics: Rethinking Test-Time Scaling Laws}, 
      author={Ranajoy Sadhukhan and Zhuoming Chen and Haizhong Zheng and Yang Zhou and Emma Strubell and Beidi Chen},
      year={2025},
      eprint={2506.05333},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.05333}, 
}
```
