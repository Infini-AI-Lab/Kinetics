import torch
import math
from einops import rearrange
from functools import partial

def topk_sparsity(logits_triton, offset, num_query_heads, num_key_value_heads, topk, local, sink=4):
    batch_size = logits_triton.shape[0]
    group_size = num_query_heads // num_key_value_heads
    n_full = sink + topk + local
    for batch_id in range(batch_size):
        seq_len = offset[batch_id]
        if seq_len > n_full:
            logits = logits_triton[batch_id, :, sink:seq_len - local]   # dynamic part
            mean_logits = rearrange(logits, '(h r) s -> h r s', h=num_key_value_heads).mean(dim=1, keepdim=True)    # average over the query group (h 1 s)
            topk_ids = torch.topk(mean_logits, topk, dim=-1).indices # topk indices based on logits avged over query group (h 1 k)
            mask = torch.zeros_like(mean_logits, dtype=torch.bool)  # (h 1 s)
            mask.scatter_(-1, topk_ids, True)    # mask for topk (h 1 s)
            mask = mask.expand(-1, group_size, -1).reshape(num_query_heads, -1)   # expand to all query heads in a query group
            logits_triton[batch_id, :, sink:seq_len - local] = torch.where(mask, logits, float('-inf'))   # apply mask to dynamic part

    return logits_triton


def blockwise_topk_sparsity(logits_triton, offset, num_query_heads, num_key_value_heads, topk_block, local, sink=4, block_size=16):
    batch_size = logits_triton.shape[0]
    group_size = num_query_heads // num_key_value_heads
    n_full = sink + local + topk_block * block_size 
    n_full = (n_full + block_size - 1) // block_size * block_size
    
    for batch_id in range(batch_size):
        seq_len = offset[batch_id]
        if seq_len > n_full:
            _n_dynamic = (seq_len - local - sink) // block_size * block_size
            _local = seq_len - _n_dynamic - sink
            logits = logits_triton[batch_id, :, sink:seq_len - _local]   # dynamic part
            mean_logits = rearrange(logits, '(h r) s -> h r s', h=num_key_value_heads).mean(dim=1, keepdim=True)    # average over the query group (h 1 s)
            mean_logits = rearrange(mean_logits, 'h 1 (n b) -> h 1 n b', b = block_size).mean(dim=-1)   # average over blocks (h 1 n)
            block_topk_ids = torch.topk(mean_logits, topk_block, dim=-1).indices # topk indices based on logits avged over query group (h 1 k)
            mask = torch.zeros_like(mean_logits, dtype=torch.bool)  # (h 1 n)
            mask.scatter_(-1, block_topk_ids, True)    # mask for topk (h 1 n)
            mask = mask.unsqueeze(-1).expand(-1, -1, -1, block_size)  # expand to blocks (h 1 s)
            mask = mask.reshape(*mask.shape[:-2], -1)
            mask = mask.expand(-1, group_size, -1)   # expand to all query heads in a query group
            mask = mask.reshape(num_query_heads, -1)
            logits_triton[batch_id, :, sink:seq_len - _local] = torch.where(mask, logits, float('-inf'))   # apply mask to dynamic part

    return logits_triton

def strided_blockwise_topk_sparsity(logits_triton, offset, num_query_heads, num_key_value_heads, topk_block, local, sink=4, block_size=16, stride=8):
    batch_size = logits_triton.shape[0]
    group_size = num_query_heads // num_key_value_heads
    kv_budget = topk_block * block_size  # total number of allowed unique KV tokens

    for batch_id in range(batch_size):
        seq_len = offset[batch_id]
        if seq_len <= sink + local:
            continue

        dynamic_start = sink
        dynamic_end = seq_len - local
        dynamic_len = dynamic_end - dynamic_start

        if dynamic_len < block_size:
            continue  # too short for block sparsity

        # 1. Extract dynamic logits
        logits = logits_triton[batch_id, :, dynamic_start:dynamic_end]  # (H, S_dyn)
        
        # 2. Reshape to group KV heads and average over query group
        logits_grouped = rearrange(logits, '(h r) s -> h r s', h=num_key_value_heads)
        mean_logits = logits_grouped.mean(dim=1, keepdim=True)  # (H_kv, 1, S_dyn)

        # 3. Create overlapping strided blocks
        num_blocks = (dynamic_len - block_size) // stride + 1
        block_indices = torch.arange(num_blocks, device=logits.device) * stride  # (n_blocks,)
        block_ranges = block_indices[:, None] + torch.arange(block_size, device=logits.device)  # (n_blocks, block_size)
        block_ranges = block_ranges.clamp(0, dynamic_len - 1)  # Ensure valid indices
        
        # 4. Average logits over each block for each head
        gathered = mean_logits[:, :, block_ranges]  # (H_kv, 1, n_blocks, block_size)
        block_means = gathered.mean(dim=-1)  # (H_kv, 1, n_blocks)

        # 5. Select more blocks (oversample) and deduplicate later
        k_blocks = min(topk_block * 2, num_blocks)
        top_blocks = torch.topk(block_means, k_blocks, dim=-1).indices  # (H_kv, 1, k_blocks)

        # 6. Get union of token indices from selected blocks
        selected_token_mask = torch.zeros((num_key_value_heads, 1, dynamic_len), dtype=torch.bool, device=logits.device)
        for h in range(num_key_value_heads):
            block_ids = top_blocks[h, 0]  # (k_blocks,)
            tokens = block_ranges[block_ids].flatten()
            tokens = tokens.unique()
            if tokens.numel() > kv_budget:
                tokens = tokens[:kv_budget]  # trim to KV budget
            selected_token_mask[h, 0, tokens] = True

        # 7. Expand to query heads
        selected_token_mask = selected_token_mask.expand(-1, group_size, -1).reshape(num_query_heads, -1)

        # 8. Apply mask
        logits_triton[batch_id, :, dynamic_start:dynamic_end] = torch.where(
            selected_token_mask, logits, float('-inf')
        )

    return logits_triton


def local_sparsity(logits_triton, offset, local, sink=4):
    batch_size = logits_triton.shape[0]
    for batch_id in range(batch_size):
            seq_len = offset[batch_id]
            if seq_len > sink + local:
                logits_triton[batch_id, :, sink:seq_len - local].fill_(float('-inf'))   # mask dynamic part

    return logits_triton

def oracle_sample_sparsity(logits_triton, offset, num_query_heads, num_key_value_heads, nsample, local, sink=4):
    batch_size = logits_triton.shape[0]
    group_size = num_query_heads // num_key_value_heads
    n_full = sink + local + nsample
    budget = nsample + sink + local
    for batch_id in range(batch_size):
        seq_len = offset[batch_id]
        if seq_len > n_full:
            sample_ids = torch.multinomial(logits_triton[batch_id, :, :seq_len], budget)
            logits_triton[batch_id, :, :seq_len].zero_()
            logits_triton[batch_id, :, :seq_len].scatter_add_(-1, 
                                                             sample_ids, 
                                                             torch.ones_like(sample_ids, dtype=logits_triton.dtype))
            logits_triton[batch_id] = logits_triton[batch_id] / budget
            
    return logits_triton

def fair_oracle_sample_sparsity(logits_triton, offset, num_query_heads, num_key_value_heads, nsample, local, sink=4, oversample_factor=1):
    batch_size = logits_triton.shape[0]
    group_size = num_query_heads // num_key_value_heads
    budget = nsample + sink + local
    for batch_id in range(batch_size):
        seq_len = offset[batch_id]
        if seq_len > budget:
            logits = logits_triton[batch_id, :, :seq_len]

            # Oversample and ensure we get `nsample` unique items
            sampled_unique = None
            while True:
                oversample_n = oversample_factor * nsample
                sampled = torch.multinomial(logits, oversample_n, replacement=True)
                sampled_unique = torch.unique(sampled, dim=-1)
                if sampled_unique.shape[-1] >= nsample:
                    sampled_unique = sampled_unique[..., :nsample]
                    break
                oversample_factor *= 2  # increase oversampling if not enough unique

            # Zero out all logits and set sampled positions
            logits.zero_()
            logits.scatter_add_(-1, sampled_unique, torch.ones_like(sampled_unique, dtype=logits.dtype))

            # Normalize (optional; comment out if not needed)
            logits_triton[batch_id, :, :seq_len] = logits / budget

    return logits_triton

        
def random_sample_with_static_sparsity(logits_triton, offset, num_query_heads, num_key_value_heads, nsample, local, sink=4):
    batch_size = logits_triton.shape[0]
    group_size = num_query_heads // num_key_value_heads
    n_full = nsample + sink + local
    for batch_id in range(batch_size):
        seq_len = offset[batch_id]
        if seq_len > n_full:
            logits = logits_triton[batch_id, :, sink:seq_len - local]   # dynamic part
            random_sample_ids = torch.randint(0, seq_len - sink - local, (num_key_value_heads, nsample), dtype=torch.long, device=logits.device)
            logits = logits + math.log(seq_len - sink - local) - math.log(nsample)  # importance sampling
            mask = torch.zeros(num_key_value_heads, seq_len - sink - local, device=logits.device, dtype=torch.bool)
            mask.scatter_(-1, random_sample_ids, True)    # mask for topk (h 1 s)
            mask = mask.unsqueeze(1).expand(-1, group_size, -1).reshape(num_query_heads, -1)   # expand to all query heads in a query group
            logits_triton[batch_id, :, sink:seq_len - local] = torch.where(mask, logits, float('-inf'))   # apply mask to dynamic part

    return logits_triton

def random_select_sparsity(logits_triton, offset, num_query_heads, num_key_value_heads, nsample, local, sink=4):
    batch_size = logits_triton.shape[0]
    group_size = num_query_heads // num_key_value_heads
    n_full = nsample + sink + local
    for batch_id in range(batch_size):
        seq_len = offset[batch_id]
        if seq_len > n_full:
            logits = logits_triton[batch_id, :, sink:seq_len - local]   # dynamic part
            random_select_ids = torch.randint(0, seq_len - sink - local, (num_key_value_heads, nsample), dtype=torch.long, device=logits.device)
            mask = torch.zeros(num_key_value_heads, seq_len - sink - local, device=logits.device, dtype=torch.bool)
            mask.scatter_(-1, random_select_ids, True)    # mask for topk (h 1 s)
            mask = mask.unsqueeze(1).expand(-1, group_size, -1).reshape(num_query_heads, -1)   # expand to all query heads in a query group
            logits_triton[batch_id, :, sink:seq_len - local] = torch.where(mask, logits, float('-inf'))   # apply mask to dynamic part

    return logits_triton


def preprocess_sparse_args(**kwargs):
    
    attn_topk = kwargs.get("attn_topk", None)
    attn_local = kwargs.get("attn_local", None)
    attn_block_topk = kwargs.get("attn_block_topk", None)
    attn_random_select = kwargs.get("attn_random_select", None)
    attn_random_nsample = kwargs.get("attn_random_nsample", None)
    use_nonlocal_sparsity = sum([attn_topk is not None, attn_block_topk is not None, attn_random_select is not None, attn_random_nsample is not None])
    assert use_nonlocal_sparsity <= 1, "Only one of the sparsity parameters can be provided"
    if use_nonlocal_sparsity:
        assert attn_local is not None, "attn_local must be provided if use_nonlocal_sparsity is True"
    kwargs["use_nonlocal_sparsity"] = use_nonlocal_sparsity
    kwargs["use_sparsity"] = use_nonlocal_sparsity + (attn_local is not None)
    kwargs["attn_layer_skip"] = kwargs.get("attn_layer_skip", [0])
    
    return kwargs
        
    

def attn_logits_processor(logits_triton, offset, num_query_heads, num_key_value_heads, layer_idx, **kwargs):
    
    use_sparse_attn = kwargs["use_sparsity"]
    attn_layer_skip = kwargs["attn_layer_skip"]
    if (not use_sparse_attn) or (layer_idx in attn_layer_skip):
        return  logits_triton
    
    attn_topk = kwargs["attn_topk"]
    attn_local = kwargs.get("attn_local", None)
    attn_block_topk = kwargs.get("attn_block_topk", None)
    attn_block_size = kwargs.get("attn_block_size", 16)
    attn_stride = kwargs.get("attn_stride", None)
    attn_random_select = kwargs.get("attn_random_select", None)
    attn_random_nsample = kwargs.get("attn_random_nsample", None)
    use_nonlocal_sparsity = kwargs["use_nonlocal_sparsity"]

    if use_nonlocal_sparsity:
        if attn_topk is not None:
            logits_triton = topk_sparsity(logits_triton, offset, num_query_heads, num_key_value_heads, attn_topk, attn_local)
        elif attn_block_topk is not None:
            if attn_stride is not None:
                logits_triton = strided_blockwise_topk_sparsity(logits_triton, offset, num_query_heads, num_key_value_heads, attn_block_topk, attn_local, block_size=attn_block_size, stride=attn_stride)
            else:
                logits_triton = blockwise_topk_sparsity(logits_triton, offset, num_query_heads, num_key_value_heads, attn_block_topk, attn_local, block_size=attn_block_size)
        elif attn_random_select is not None:
            logits_triton = random_select_sparsity(logits_triton, offset, num_query_heads, num_key_value_heads, attn_random_select, attn_local)
        elif attn_random_nsample is not None:
            logits_triton = random_sample_with_static_sparsity(logits_triton, offset, num_query_heads, num_key_value_heads, attn_random_nsample, attn_local)
                    
    elif attn_local is not None:
        logits_triton = local_sparsity(logits_triton, offset, attn_local)
    
    return logits_triton
    
    