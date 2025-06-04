import argparse
import time
import torch
import torch.nn as nn
from transformers import AutoConfig
from dense_decoder import DecoderLayer, init_weights  # 将 DecoderLayer 拆出去更好维护

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gen_len", type=int, default=32768)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--page_size", type=int, default=16)
    parser.add_argument("--repeat_time", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--world_size", type=int, default=8)
    return parser.parse_args()

def build_decoder(config, args):
    prefix_len = (args.gen_len) // 2
    max_len = 32768
    dtype = getattr(torch, args.dtype)

    decoder = DecoderLayer(
        config=config,
        world_size=args.world_size,
        batch_size=args.batch_size,
        prefix_len=prefix_len,
        max_len=max_len,
        page_size=args.page_size,
        device=args.device,
        dtype=dtype
    )
    decoder.apply(init_weights)
    decoder.eval()
    decoder.to(args.device, dtype=dtype)
    decoder.forward = torch.compile(decoder.forward, mode="max-autotune", fullgraph=True)
    decoder.prepare_wrapper()
    return decoder, prefix_len

def run_benchmark(decoder, hidden_states, repeat_time):
    for _ in range(200):  # warmup
        _ = decoder(hidden_states.clone())

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat_time):
        _ = decoder(hidden_states.clone())
    torch.cuda.synchronize()
    end = time.time()

    throughput = (repeat_time * hidden_states.size(0)) / (end - start)
    return throughput

def main():
    args = parse_args()
    config = AutoConfig.from_pretrained(args.model)
    dtype = getattr(torch, args.dtype)
    hidden_size = config.hidden_size

    hidden_states = torch.randn(
        args.batch_size, 1, hidden_size,
        device=args.device, dtype=dtype
    )

    decoder, prefix_len = build_decoder(config, args)
    throughput = run_benchmark(decoder, hidden_states, args.repeat_time)

    print("\n=== Benchmark Result ===")
    print(f"Model:           {args.model.split('/')[-1]}")
    print(f"Prefix length:   {prefix_len}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Page size:       {args.page_size}")
    print(f"Repeat time:     {args.repeat_time}")
    print(f"Throughput:      {throughput/(args.world_size * config.num_hidden_layers):.2f} tokens/sec")

if __name__ == "__main__":
    main()
