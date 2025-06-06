## Benchmarking
We provide a simplistic set-up to benchmark the throughput of block-topk attention for different model sizes. We use Flashinfer kernels for attention and use torch.compile to compile the decoding runs.

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

## TODO
Currently our benchmark does not show end-to-end throughput improvements. Stay tuned for the upcoming updates.

- [ ] **SGLang Integration**

    - We are adding support for block top-k attention into SGLang to demonstrate end-to-end throughput improvements with sparse attention.

 

