import sys
sys.path.append("..")
from litesys.helper import generate_code_requests
from litesys.engine.dist_engine import LLM
from datasets import load_dataset
import os
import argparse
import torch
import jsonlines
from datetime import datetime
from typing import Any

import random
import numpy as np
import torch.distributed as dist


def prepare_prompt(line: dict[str, Any]) -> str:
    query = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    query += f"Question: {line['question_content']}\n\n"
    
    if starter_code := line.get("starter_code", None):
        query += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        query += f"```python\n{starter_code}\n```\n\n"
    else:
        query += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        query += "```python\n# YOUR CODE HERE\n```\n\n"
    return query

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import glob

def check_seed_file_exists(output_dir, seed):

    pattern = os.path.join(output_dir, f"*seed{seed}_raw.jsonl")
    matching_files = glob.glob(pattern)
    return len(matching_files) > 0

def eval(args):
    set_seed(args.seed)
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    data = "livecodebench/code_generation_lite"
    if not args.local_data:
        dataset = load_dataset(data, version_tag=args.version, split="test")
    else:
        dataset = load_dataset(path=f"{args.data}/code_generation_lite.py",data_dir=args.data, name="v5", split="test", trust_remote_code=True)
        
    requests = generate_code_requests(dataset=dataset, formatter=prepare_prompt, trial=args.trial, rank=0, world_size=1)

    
    llm = LLM(model_name=args.model, model_class_str=args.model_class, dtype=torch.bfloat16, device=f"cuda:{rank}",
            max_seq_len=args.max_len, top_p=0.95, temperature=0.6, top_k=20,
            attn_topk=args.attn_topk, attn_nsample=args.attn_nsample, attn_local=args.attn_local, attn_block_topk=args.attn_block_topk,
            attn_block_size=args.attn_block_size, attn_stride=args.attn_stride,
            attn_random_select=args.attn_random_select, attn_random_nsample=args.attn_random_nsample,
            quest_block_size=args.quest_block_size, quest_topk_blocks=args.quest_topk_blocks)
    
    all_processed_request = []
    processed_request = llm.offline_exec(requests, args.gen_len)
    all_processed_request.extend(processed_request)
    results = []
    
    for data in all_processed_request: 
            data.pop("private_test_cases")
            data.pop("inputs")
            data.pop("outputs")
            results.append(data)
            
    return results
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-0.6B", help='Model name')
    parser.add_argument('--model_class', type=str, default="qwen3", help='Model Class')
    parser.add_argument('--version', type=str, default="v5", help='LCB version')
    parser.add_argument('--task', type=str, default="livecodebench", help='Task identifier')
    parser.add_argument('--local_data', type=bool, default=False, help='Whether to use local data')
    parser.add_argument('--data', type=str, default=None, help='Evaluation Data Path')
    parser.add_argument('--nproc', type=int, default=8, help='Number of processes to launch')
    parser.add_argument('--trial', type=int, default=4, help='Number of trials')
    parser.add_argument('--max_len', type=int, default=32768, help='Max Length')
    parser.add_argument('--gen_len', type=int, default=30768, help='Max Generation Length')
    parser.add_argument('--output_dir', type=str, default="results", help='output dir')
    parser.add_argument('--attn_topk', type=int, default=None, help='Attention Top-k')
    parser.add_argument('--attn_nsample', type=int, default=None, help='Attention N-sample')
    parser.add_argument('--attn_local', type=int, default=None, help='Attention Local')
    parser.add_argument('--attn_block_topk', type=int, default=None, help='Attention Block Top-k')
    parser.add_argument('--attn_block_size', type=int, default=None, help='Attention Block Size')
    parser.add_argument('--attn_stride', type=int, default=None, help='Attention Stride')
    parser.add_argument('--attn_random_select', type=int, default=None, help='Attention Random Select')
    parser.add_argument('--attn_random_nsample', type=int, default=None, help='Attention Random N-sample')
    parser.add_argument('--quest_block_size', type=int, default=16, help='Quest Block Size')
    parser.add_argument('--quest_topk_blocks', type=int, default=None, help='Quest Top-k Blocks')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    args = parser.parse_args()
    output_dir = args.output_dir
    if check_seed_file_exists(output_dir, args.seed):
        return
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank)
    else:
        raise RuntimeError("Distributed environment not initialized. Please use torchrun to launch the script.")
    
    

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        raw_output_file = os.path.join(output_dir, f"{args.task}_{current_time}_seed{args.seed}_raw.jsonl")
    dist.barrier()
    result_list = eval(args)
    dist.barrier()
    if rank == 0:
        with jsonlines.open(raw_output_file, "w") as f:
            f.write_all(result_list)
        print("Results saved to", raw_output_file)
        

if __name__ == "__main__":
    main()