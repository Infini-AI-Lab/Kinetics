import os
import math
from typing import List, Tuple, Dict, Any, Optional, Callable
from functools import partial, reduce
import json
import numpy as np
from scipy.special import comb
import pandas as pd

MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}{Response}
""".strip()

# Define the mapping function
def apply_chat_template(example, tokenizer, template):
    chat = [
        {"role": "user", "content": template.format(Question=example["query"], Response=example["prediction"])}
    ]
    example["token_length"] = len(tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True
    ))
    prefix = [
        {"role": "user", "content": template.format(Question=example["query"], Response="")}
    ]
    example["prefix_length"] = len(tokenizer.apply_chat_template(
        prefix, tokenize=True, add_generation_prompt=True
    ))
    return example

def coverage(scores, nsamples: List[int]) -> List[float]:
    num_trials = len(scores)
    ncorrect = int(sum(scores))
    coverage_scores = []
    for nsample in nsamples:
        cov = 1 - comb(num_trials - ncorrect, nsample) / comb(num_trials, nsample)
        coverage_scores.append(cov)
    return coverage_scores

#########################
# Cost Model Functions #
#########################

def dense_cost(nparams, num_attn_heads, num_kv_heads, head_dim, E_flops, context_length, generation_length, num_trials=1):
    compute_cost = (
        2 * nparams * generation_length + \
        2 * (2 * context_length + generation_length) * generation_length * num_attn_heads * head_dim
    ) * num_trials * 1e-12
    
    memory_cost = (
        # 2 * nparams * generation_length * num_trials + \
        2 * (2 * context_length + generation_length * num_trials) * generation_length * num_kv_heads * head_dim
    ) * E_flops * 1e-12
    
    return compute_cost, memory_cost

def zero_overhead_sparse_cost(nparams, num_attn_heads, num_kv_heads, head_dim, E_flops, context_length, budget, generation_length, num_trials=1):
    kv_scaling_factor =  budget * generation_length + 0.5 * max(0, budget - context_length)**2 
    
    compute_cost = (
        2 * nparams * generation_length + \
        2 * 2 * kv_scaling_factor * num_attn_heads * head_dim
    ) * num_trials * 1e-12
    
    # Not considering any kind of prefix sharing in sparse setting: although it is possible for large context length
    memory_cost = (
        # 2 * nparams * generation_length + \
        2 * 2 * kv_scaling_factor * num_kv_heads * head_dim
    ) * num_trials * E_flops * 1e-12
    
    return compute_cost, memory_cost
    
def block_sparse_cost(nparams, num_attn_heads, num_kv_heads, head_dim, E_flops, context_length, local, block_size, block_topk, generation_length, num_trials=1):
    budget = local + block_size * block_topk
    
    kv_scaling_factor =  budget * generation_length - 0.5 * max(0, budget - context_length)**2 
    
    search_compute_cost = (2 * context_length + generation_length) // block_size * generation_length * num_attn_heads * head_dim
    
    compute_cost = (
        2 * nparams * generation_length + \
        2 * 2 * kv_scaling_factor * num_attn_heads * head_dim + \
        search_compute_cost
    ) * num_trials * 1e-12
    
    search_memory_cost = (2 * context_length + generation_length * num_trials) // block_size * generation_length * num_kv_heads * head_dim
    
    memory_cost = (
        # 2 * nparams * generation_length + \
        2 * 2 * kv_scaling_factor * num_kv_heads * head_dim 
    ) * num_trials + search_memory_cost
    
    memory_cost *= E_flops * 1e-12
    
    return compute_cost, memory_cost
        
def expected_cost(cost_fn: callable, generation_lengths, num_trials=1, **kwargs):
    compute_costs, memory_costs = zip(*[cost_fn(generation_length=x, num_trials=num_trials, **kwargs) for x in generation_lengths])
    return np.mean(compute_costs), np.mean(memory_costs)