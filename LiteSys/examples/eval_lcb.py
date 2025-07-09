import sys
sys.path.append("..")
from litesys.helper import  generate_code_requests
from litesys.engine.engine import LLM
from datasets import load_dataset
import os
import argparse
import torch
import torch.multiprocessing as mp
import jsonlines
from datetime import datetime
from typing import Any

import random
import numpy as np



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

def worker(rank, world_size, args, shared_list):
    
    torch.cuda.set_device(rank)
    set_seed(args.seed + rank)
    data = "livecodebench/code_generation_lite"
    if not args.local_data:
        dataset = load_dataset(data, version_tag=args.version, split="test")
    else:
        dataset = load_dataset(path=f"{args.data}/code_generation_lite.py",data_dir=args.data, name="v5", split="test", trust_remote_code=True)
    
    requests = generate_code_requests(dataset=dataset, formatter=prepare_prompt, trial=args.trial, rank=rank, world_size=world_size)

    
    llm = LLM(model_name=args.model, model_class_str=args.model_class, dtype=torch.bfloat16, device=f"cuda:{rank}",
            max_seq_len=args.max_len, top_p=0.95, temperature=0.6, top_k=20,
            attn_topk=args.attn_topk, attn_nsample=args.attn_nsample, attn_local=args.attn_local, attn_block_topk=args.attn_block_topk,
            attn_block_size=args.attn_block_size, attn_stride=args.attn_stride,
            attn_random_select=args.attn_random_select, attn_random_nsample=args.attn_random_nsample,
            quest_block_size=args.quest_block_size, quest_topk_blocks=args.quest_topk_blocks)
    
    all_processed_request = []
    processed_request = llm.offline_exec(requests, args.gen_len)
    all_processed_request.extend(processed_request)
    
    for data in all_processed_request: 
            data.pop("private_test_cases")
            data.pop("inputs")
            data.pop("outputs")
            shared_list.append(data)
        
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-0.6B", help='Model name')
    parser.add_argument('--model_class', type=str, default="qwen3", help='Model Class')
    parser.add_argument('--version', type=str, default="v5", help='LCB version')
    parser.add_argument('--task', type=str, default="livecodebench", help='Task identifier')
    parser.add_argument('--nproc', type=int, default=8, help='Number of processes to launch')
    parser.add_argument('--trial', type=int, default=4, help='Number of trials')
    parser.add_argument('--local_data', type=bool, default=False, help='Whether to use local data')
    parser.add_argument('--data', type=str, default=None, help='Evaluation Data Path')
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
    
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    raw_output_file = os.path.join(output_dir, f"{args.task}_{current_time}_seed{args.seed}_raw.jsonl")
    if check_seed_file_exists(output_dir, args.seed):
        return
    # Use mp.Manager to create a shared list for process-safe communication
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    shared_list = manager.list()

    # Launch multiple processes using mp.spawn; each process handles a subset of the dataset
    world_size = args.nproc
    mp.spawn(worker, args=(world_size, args, shared_list), nprocs=world_size, join=True)

    with jsonlines.open(raw_output_file, "w") as f:
            f.write_all(shared_list)
    # Gather results and compute global accuracy (optional)
    print("Results saved to", raw_output_file)
    
    

if __name__ == "__main__":
    main()