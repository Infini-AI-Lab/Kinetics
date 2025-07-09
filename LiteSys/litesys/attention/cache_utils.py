import triton
import triton.language as tl

@triton.jit
def copy_to_cache_kernel(
    key_cache_ptr, key_ptr, batch_id_ptr, offset_ptr,
    num_heads, head_dim,
    stride_kc_b, stride_kc_h, stride_kc_s, stride_kc_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    BLOCK_HEAD_DIM: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_dim = tl.arange(0, BLOCK_HEAD_DIM)

    batch_id = tl.load(batch_id_ptr + batch_idx)
    offset = tl.load(offset_ptr + batch_idx)

    key_offset = (
        batch_idx * stride_k_b +
        head_idx * stride_k_h +
        0 * stride_k_s + 
        offs_dim * stride_k_d
    )

    cache_offset = (
        batch_id * stride_kc_b +
        head_idx * stride_kc_h +
        offset * stride_kc_s +
        offs_dim * stride_kc_d
    )

    mask = offs_dim < head_dim

    key_vals = tl.load(key_ptr + key_offset, mask=mask, other=0)
    tl.store(key_cache_ptr + cache_offset, key_vals, mask=mask)


def copy_to_cache(key_cache, key, batch_id, offset):
    batch_size, num_heads, _, head_dim = key.shape

    grid = (batch_size, num_heads)

    copy_to_cache_kernel[grid](
        key_cache, key, batch_id, offset,
        num_heads, head_dim,
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        BLOCK_HEAD_DIM=triton.next_power_of_2(head_dim)
    )

@triton.jit
def attention_score_kernel(
    key_cache_ptr, query_ptr, batch_idx_ptr, offset_ptr, output_ptr,
    num_key_value_heads: tl.constexpr,
    num_query_heads: tl.constexpr,
    max_seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    batch_id = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    seq_block = tl.program_id(2)

    seq_offsets = seq_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    offset = tl.load(offset_ptr + batch_id)
    if (seq_block * BLOCK_SEQ >= offset): return
    valid_mask = seq_offsets < offset
    batch_idx = tl.load(batch_idx_ptr + batch_id)

    key_offset = batch_idx * num_key_value_heads * max_seq_len * head_dim \
                 + kv_head_id * max_seq_len * head_dim
    key_ptrs = key_cache_ptr + key_offset + \
               (seq_offsets[:, None] * head_dim + tl.arange(0, head_dim)[None, :])

    key = tl.load(key_ptrs, mask=valid_mask[:, None], other=0.0)

    group_size = num_query_heads // num_key_value_heads
    for qh_in_group in range(group_size):
        query_head_id = kv_head_id * group_size + qh_in_group
        query_offset = batch_id * num_query_heads * head_dim + query_head_id * head_dim
        query = tl.load(query_ptr + query_offset + tl.arange(0, head_dim))

        attn_score = tl.sum(key * query[None, :], axis=1)
        attn_score *= (1.0 / (head_dim ** 0.5))
        attn_score = tl.where(valid_mask, attn_score, float('-inf'))

        output_offset = batch_id * num_query_heads * max_seq_len \
                        + query_head_id * max_seq_len
        tl.store(output_ptr + output_offset + seq_offsets,
                 attn_score,
                 mask=seq_offsets < max_seq_len)


@triton.jit
def softmax_kernel_inplace(
    ptr, offset_ptr,
    num_heads: tl.constexpr, max_seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Load offset
    seq_len = tl.load(offset_ptr + batch_id)

    seq_offsets = tl.arange(0, BLOCK_SIZE)
    mask = seq_offsets < seq_len

    # Pointers
    offset = batch_id * num_heads * max_seq_len + head_id * max_seq_len
    ptrs = ptr + offset + seq_offsets

    # Load inputs
    logits = tl.load(ptrs, mask=mask, other=float('-inf'))

    # Compute max for numerical stability
    logits_max = tl.max(logits, axis=0)
    logits = logits - logits_max

    # Compute exponentials and sum
    exp_logits = tl.where(mask, tl.exp(logits), 0.0)
    exp_sum = tl.sum(exp_logits, axis=0)

    # Compute softmax in-place
    softmax = exp_logits / exp_sum

    # Store results in-place
    tl.store(ptrs, softmax, mask=mask)



@triton.jit
def attention_value_stage1_kernel(
    score_ptr, value_ptr, buffer_ptr, batch_idx_ptr, offset_ptr,
    num_query_heads: tl.constexpr,
    num_key_value_heads: tl.constexpr,
    max_seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    NUM_BLOCK_SEQ: tl.constexpr,
    BLOCK_SEQ: tl.constexpr
):
    
    batch_id = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    seq_block = tl.program_id(2)

    seq_len = tl.load(offset_ptr + batch_id)
    if (seq_block * BLOCK_SEQ >= seq_len): return
    
    seq_offsets = seq_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    mask = seq_offsets  < seq_len
    dim_offsets = tl.arange(0, head_dim)
    batch_idx = tl.load(batch_idx_ptr + batch_id)
    blk_value_ptr = value_ptr + batch_idx * num_key_value_heads * max_seq_len * head_dim + kv_head_id * max_seq_len * head_dim + \
        seq_block * BLOCK_SEQ * head_dim + tl.arange(0, BLOCK_SEQ)[:,None] * head_dim + dim_offsets
    blk_values = tl.load(blk_value_ptr, mask=mask[:,None], other=0.0)

    group_size = num_query_heads // num_key_value_heads
    for qh_in_group in range(group_size):
        query_head_id = kv_head_id * group_size + qh_in_group
        blk_score_ptr = score_ptr + batch_id * num_query_heads * max_seq_len + query_head_id * max_seq_len + seq_offsets
        blk_scores = tl.load(blk_score_ptr, mask=mask, other=0.0)
        acc = tl.sum(blk_scores[:, None] * blk_values, axis=0)
        output_offset = batch_id * num_query_heads * NUM_BLOCK_SEQ * head_dim + \
            query_head_id * NUM_BLOCK_SEQ * head_dim + seq_block * head_dim
        
        tl.store(buffer_ptr + output_offset + dim_offsets, acc)
    
    

def qk(query, key, batch_idx, offset,
        logits_buffer, BLOCK_SEQ):
    
    batch_size, num_query_heads, _, head_dim = query.shape
    _, num_key_value_heads, max_seq_len, _ = key.shape
    NUM_BLOCK_SEQ = triton.cdiv(max_seq_len, BLOCK_SEQ)

    
    grid = (batch_size, num_key_value_heads, NUM_BLOCK_SEQ)
    
    attention_score_kernel[grid](
        key, query, batch_idx, offset, logits_buffer,
        num_key_value_heads,
        num_query_heads,
        max_seq_len,
        head_dim,
        BLOCK_SEQ=BLOCK_SEQ
    )


def sv(value, batch_idx, offset, 
    logits_buffer, output_buffer, BLOCK_SEQ):
    
    batch_size = len(batch_idx)
    num_query_heads = output_buffer.shape[1]
    _, num_key_value_heads, max_seq_len, head_dim = value.shape
    NUM_BLOCK_SEQ = triton.cdiv(max_seq_len, BLOCK_SEQ)

    grid = (batch_size, num_key_value_heads, NUM_BLOCK_SEQ)
    attention_value_stage1_kernel[grid](
        logits_buffer, value, output_buffer, batch_idx, offset,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        NUM_BLOCK_SEQ=NUM_BLOCK_SEQ,
        BLOCK_SEQ=BLOCK_SEQ
    )
    output = output_buffer[:batch_size].sum(dim=2).to(value.dtype).contiguous()
    return output



def softmax(
    logits_buffer, batch_idx, offset
):
    batch_size = len(batch_idx)
    num_query_heads = logits_buffer.shape[1]
    max_seq_len = logits_buffer.shape[2]
    softmax_kernel_inplace[(batch_size, num_query_heads)](
        logits_buffer, offset,
        num_heads=num_query_heads, max_seq_len=max_seq_len,
        BLOCK_SIZE=triton.next_power_of_2(max_seq_len)
    )

    


