import os
import sys
import math
from typing import List, Tuple, Dict, Any, Optional, Callable
from functools import partial, reduce
import json
import numpy as np
from scipy.special import comb
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import *

def compute_tradeoff(log_file, tokenizer, **kwargs):
    with open(log_file, 'r') as f:
        # read metadata dict from first line
        metadata = json.loads(f.readline())
        # read rest of the lines as dicts
        data = [json.loads(line) for line in f]
        
    # cehck if attn_local is there, or return None
    is_sparse = metadata.get("attn_local", None) is not None
    is_block_sparse = is_sparse and metadata.get("attn_block_topk", None) is not None
    
    # step 1: aggregate responses for the same query
    df = pd.DataFrame(data)
    
    # only take the first item from the predictions list and choices list
    df["prediction"] = df["prediction"].apply(lambda x: x[0])
    df["choices"] = df["choices"].apply(lambda x: x[0])
    
    # Use partial to pass tokenizer and template to map
    template_func = partial(apply_chat_template, tokenizer=tokenizer, template=MATH_QUERY_TEMPLATE)

    # Apply with multiprocessing
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(template_func, num_proc=8, desc="Applying chat template")

    # Optionally convert to pandas for groupby
    df_tokenized = tokenized_dataset.to_pandas()
    
    # Group by 'query' and 'choices' and aggregate 'token_ids' and 'score' into lists
    grouped_df = (
        df_tokenized
        .groupby(["query", "choices", "prefix_length"])   
        .agg({"token_length": list, "score": list})
        .reset_index()
    )
    
    result_cols = ["trial", "coverage", "cost"]
    if is_sparse:
        result_cols.append("sparsity")
        
    result_df = []
    
    if "32b" not in log_file:
        num_trials = [1, 2, 4, 8, 16, 32]
    else:
        num_trials = [1, 2, 4, 8]
    for index, row in tqdm(grouped_df.iterrows(), total=len(grouped_df), desc="Processing rows"):
        context_length = row["prefix_length"]
        sequence_lengths = row["token_length"]
        generation_lengths = [sl - context_length for sl in sequence_lengths]
        
        scores = row["score"]   
        if any(score > 1 for score in scores):
            scores = [score / 100 for score in scores]
        cov = coverage(scores, num_trials)
        
        if not is_sparse:
            cost_fn = dense_cost
        elif is_block_sparse:
            cost_fn = block_sparse_cost
            kwargs["block_size"] = metadata["attn_block_size"]
            kwargs["block_topk"] = metadata["attn_block_topk"]
            kwargs["local"] = metadata["attn_local"]
            budget = kwargs["local"] + kwargs["block_topk"] * kwargs["block_size"]
        else:
            cost_fn = zero_overhead_sparse_cost
            attn_topk = metadata.get("attn_topk", 0)
            attn_topk = 0 if attn_topk is None else attn_topk
            kwargs["budget"] = metadata["attn_local"] + attn_topk
            budget = kwargs["budget"]
        
        for ntrial, cov_trial in zip(num_trials, cov):
            compute_cost, memory_cost = expected_cost(cost_fn, generation_lengths, ntrial, context_length=context_length, **kwargs)
            
            res_dict = {
                "query_id": row["question_id"] if "question_id" in row else index,
                "generation_length": generation_lengths,
                "trial": ntrial,
                "coverage": cov_trial,
                "compute_cost": compute_cost,
                "memory_cost": memory_cost,
                "total_cost": compute_cost + memory_cost
            }
            
            if "choices" in row:
                res_dict["choices"] = row["choices"]
            if "query" in row:
                res_dict["query"] = row["query"]
            if "difficulty" in row:
                res_dict["difficulty"] = row["difficulty"]
            
            if is_sparse:
                res_dict["budget"] = budget
                
            result_df.append(res_dict)
    result_df = pd.DataFrame(result_df)
    
    return result_df

if __name__ == "__main__":
    E_flops = 562.5
    task = "aime24"
    
    model_sizes = {
        "0.6B": 0.752,
        "1.7B": 2.03,
        "4B": 4.02,
        "8B": 8.19,
        "14B": 14.77,
        "32B": 32.76
    }
    
    task = sys.argv[1]
    sparse_arg = sys.argv[2]
    res_dir = f"{task}/{sparse_arg}"
    
    for model in ["Qwen3-32B", "Qwen3-14B", "Qwen3-8B", "Qwen3-4B", "Qwen3-1.7B", "Qwen3-0.6B"]:
        config = AutoConfig.from_pretrained(f"Qwen/{model}")
        tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{model}")
        
        nparams = model_sizes[model.split("-")[1]] * 1e9
        
        num_attn_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim * config.num_hidden_layers
        
        log_files = os.listdir(res_dir)
        model_name = model.lower().replace(".", "-")
        
        for log_file in log_files:
            if log_file.endswith(".jsonl") and f"_{sparse_arg}" in log_file and model_name in log_file:
                result_df = compute_tradeoff(f"{res_dir}/{log_file}", tokenizer, nparams=nparams, num_attn_heads=num_attn_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, E_flops=E_flops)
                result_df.to_csv(f"{res_dir}/{log_file.split('/')[-1].split('.')[0]}_ntrial_tradeoff.csv", index=False)