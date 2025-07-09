
import sys
sys.path.append("..")
from litesys.helper import generate_requests, generate_code_requests
from litesys.engine.engine import LLM
from datasets import load_dataset, load_from_disk
import os
import argparse
import torch
import json
import torch.multiprocessing as mp
import jsonlines
from lighteval.tasks.extended.lcb.codegen_metrics import codegen_metrics, extract_code
from datetime import datetime
from typing import Any
import random
import numpy as np
import glob
import re


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

def list_seed_files(output_dir):
    pattern = os.path.join(output_dir, "*.jsonl")
    matching_files = sorted(glob.glob(pattern))

    if not matching_files:
        print("No seed files found.")
    return matching_files

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))
    return data

def load_data(args):
    data = "livecodebench/code_generation_lite"
    if not args.local_data:
        dataset = load_dataset(data, version_tag="v5", split="test")
    else:
        dataset = load_dataset(path=f"{args.data}/code_generation_lite.py",data_dir=args.data, name="v5", split="test", trust_remote_code=True)
    requests = generate_code_requests(dataset=dataset, formatter=prepare_prompt, trial=1, rank=0, world_size=1)
    evaluation_samples = {}
    for data in requests:
        evaluation_samples[data["query"]] = {
            "inputs": data["inputs"],
            "outputs": data["outputs"],
            "fn_name": data["fn_name"],
        }
    return evaluation_samples         

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="checkpoints/Qwen3-32B/", help='output dir')
    parser.add_argument('--output_file', type=str, default="/checkpoints/Qwen3-32B/lcb_results.jsonl", help='output file')
    parser.add_argument('--local_data', type=bool, default=False, help='Whether to use local data')
    parser.add_argument('--data', type=str, default=None, help='Evaluation Data Path')
    args = parser.parse_args()
    evaluation_samples = load_data(args)
    result_files = list_seed_files(args.output_dir)
    
    if len(result_files)>64:
        result_files = result_files[:64]

    shared_list = []
    for file in result_files:
        print(f"Processing file: {file}")
        if not os.path.exists(file):
            print(f"File {file} does not exist, skipping.")
            continue
        
        # Read the JSONL file
        data = read_jsonl(file)
        shared_list += data

    trial = len(shared_list)/167
    print("trials:", trial)

    # Gather results and compute global accuracy (optional)
    
    unique_result: dict[str, list] = {}
    
    for item in shared_list:
            if item['query'] not in unique_result:
                unique_result[item['query']] = [item]
            else:
                unique_result[item['query']].append(item)
    
    results = []
    print("questions:", len(unique_result.keys()))
    all_code_snippets = []
    all_evaluation_sample = []
    for query in unique_result.keys():
        data = unique_result[query]
        generated_code_snippets = [[extract_code(g["output_text"]) for g in unique_result[query]]]
         
        evaluation_sample = evaluation_samples[query]
        evaluation_sample = [{"input_output": json.dumps(evaluation_sample)}]
        all_code_snippets.extend(generated_code_snippets)
        all_evaluation_sample.extend(evaluation_sample)
    
    
    metrics, _ = codegen_metrics(
                        all_evaluation_sample,
                        all_code_snippets,
                        k_list=[1, trial],
                        num_process_evaluate=64,
                        timeout=20
                )
    
    results.append(metrics)
    # Save the aggregated results to a single JSONLines file
    with jsonlines.open(args.output_file, "w") as f:
        f.write_all(results)

    print("Results saved to", args.output_file)
    

if __name__ == "__main__":
    main()
