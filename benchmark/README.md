## Benchmarking
This repository provides a simple yet effective setup to benchmark the throughput of **block-topk attention** and **dense attention** for different model sizes. The benchmarking process leverages [Flashinfer](https://github.com/Dao-AILab/flashinfer) attention kernels and utilizes `torch.compile` for efficient decoding runs.

> **Note:** `gen_len = 2N` corresponds to a context length of `N`.

## Dense Attention
To benchmark throughput of dense attention with page size 16, run the following command:
```
python3 dense.py --model Qwen/Qwen3-8B --gen_len 32768 --batch_size 256 --page_size 16 --world_size 8
```

## Block-topk Attention
To benchmark throughput of block-topk attention with page size 16 and 64 topk pages (KV cache budget = 1024), run the following command:
```
python3 blocktopk.py --model Qwen/Qwen3-8B --gen_len 32768 --batch_size 256 --page_size 16 --topk_page 64 --world_size 8
```
