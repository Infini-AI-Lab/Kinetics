import argparse
import time
import torch
import torch.nn as nn
from transformers import AutoConfig
from blocktopk_decoder import BlockTopkDecoderLayer, init_weights


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gen_len", type=int, default=32768)
    parser.add_argument("--page_size", type=int, default=16)
    parser.add_argument("--topk_page", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--repeat_time", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--world_size", type=int, default=8)
    return parser.parse_args()

def build_decoder_layer(config, args):
    prefix_len = args.gen_len // 2
    max_len = 32768
    dtype = getattr(torch, args.dtype)

    decoder_layer = BlockTopkDecoderLayer(
        config=config,
        world_size=args.world_size,
        batch_size=args.batch_size,
        prefix_len=prefix_len,
        max_len=max_len,
        page_size=args.page_size,
        topk_page=args.topk_page,
        device=args.device,
        dtype=dtype,
    )
    decoder_layer.apply(init_weights)
    decoder_layer.eval()
    decoder_layer.to(args.device, dtype=dtype)
    decoder_layer.forward = torch.compile(
        decoder_layer.forward, mode="max-autotune", fullgraph=True
    )
    decoder_layer.prepare_wrapper(prefix_len)
    return decoder_layer

def run_benchmark(decoder_layer, hidden_states, repeat_time):
    # Warmup
    for _ in range(200):
        _ = decoder_layer(hidden_states.clone())
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(repeat_time):
        _ = decoder_layer(hidden_states.clone())
    torch.cuda.synchronize()
    end_time = time.time()

    elapsed = end_time - start_time
    throughput = (repeat_time * hidden_states.size(0)) / elapsed
    return throughput

def main():
    args = get_args()
    config = AutoConfig.from_pretrained(args.model)
    hidden_size = config.hidden_size
    hidden_states = torch.randn(
        args.batch_size, 1, hidden_size, 
        device=args.device, 
        dtype=getattr(torch, args.dtype)
    )

    decoder_layer = build_decoder_layer(config, args)
    throughput = run_benchmark(decoder_layer, hidden_states, args.repeat_time)

    print(f"\n=== Benchmark Summary ===")
    print(f"Model: {args.model.split('/')[-1]}")
    print(f"Prefix length: {args.gen_len // 2}")
    print(f"Page size: {args.page_size}")
    print(f"Topk pages: {args.topk_page}")
    print(f"Batch size: {args.batch_size}")
    print(f"Throughput: {throughput:.2f} tokens/sec")

if __name__ == "__main__":
    main()
