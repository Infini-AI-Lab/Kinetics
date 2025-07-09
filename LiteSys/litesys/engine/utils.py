import triton
import triton.language as tl
import torch

@triton.jit
def copy_new_tokens_kernel(
    slot_tokens_ptr,  # [max_batch_size, max_seq_len]
    new_tokens_ptr,   # [batch_size]
    offset_ptr,       # [batch_size]
    batch_idx_ptr,    # [batch_size]
    max_seq_len: tl.constexpr,
    batch_size: tl.constexpr,
):
    pid = tl.program_id(0)  # 每个 program 处理一个 new_token

    if pid >= batch_size:
        return

    # Load indices
    offset = tl.load(offset_ptr + pid)
    batch_idx = tl.load(batch_idx_ptr + pid)
    new_token = tl.load(new_tokens_ptr + pid)

    # Compute flat index into slot_tokens
    out_ptr = slot_tokens_ptr + batch_idx * max_seq_len + offset

    # Write
    tl.store(out_ptr, new_token)


def copy_new_tokens(slot_tokens, new_tokens, offset, batch_idx):
    # slot_tokens: [max_batch_size, max_seq_len]
    # new_tokens, offset, batch_idx: [batch_size]
    assert new_tokens.shape == offset.shape == batch_idx.shape
    batch_size = new_tokens.shape[0]
    max_batch_size, max_seq_len = slot_tokens.shape

    grid = (batch_size,)  # 每个 thread 处理一个 token

    copy_new_tokens_kernel[grid](
        slot_tokens, new_tokens, offset, batch_idx,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
    )


def recent_subseq_repeat_check_torch(tokens: torch.Tensor, subseq_len: int = 10, 
                               recent_window: int = 100, repeat_thresh: int = 4):
    """
    Check if the most recent subsequence (length subseq_len) has appeared 
    at least repeat_thresh times within the recent_window tokens.
    
    Args:
        tokens (torch.Tensor): Tensor of shape [seq_len], token ids.
        subseq_len (int): Length of subsequence to check.
        recent_window (int): The number of recent tokens to consider.
        repeat_thresh (int): Number of times subsequence should appear to return True.
        
    Returns:
        bool: True if subsequence appears at least repeat_thresh times, else False.
    """
    seq_len = tokens.size(0)
    if seq_len < subseq_len:
        return False

    # Define the start index for the recent window
    start_idx = max(0, seq_len - recent_window)

    # Extract recent window
    recent_tokens = tokens[start_idx:]

    # Ensure there are enough tokens to form subsequences
    if recent_tokens.size(0) < subseq_len:
        return False

    # Extract the most recent subsequence
    recent_subseq = recent_tokens[-subseq_len:]

    # Generate sliding windows within recent_tokens
    windows = recent_tokens.unfold(0, subseq_len, 1)

    # Compare windows with recent subsequence
    matches = (windows == recent_subseq).all(dim=-1)

    # Count occurrences
    count = matches.sum().item()

    # Check if count reaches threshold
    return count >= repeat_thresh



# Triton Kernel: Check subsequence matches
@triton.jit
def subseq_match_kernel(tokens_ptr, recent_ptr, result_ptr, window_size, subseq_len, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = idx < (window_size - subseq_len + 1)

    match = tl.full((BLOCK_SIZE,), True, dtype=tl.int1)

    for offset in range(subseq_len):
        tok_seq = tl.load(tokens_ptr + idx + offset, mask=mask, other=-1)
        tok_recent = tl.load(recent_ptr + offset)
        match = match & (tok_seq == tok_recent)

    tl.store(result_ptr + idx, match.to(tl.int32), mask=mask)


# High-level PyTorch wrapper function
def recent_subseq_repeat_check(tokens: torch.Tensor, subseq_len: int = 10, 
                               recent_window: int = 100, repeat_thresh: int = 4) -> bool:
    """
    Checks if the most recent subsequence has occurred at least 'repeat_thresh'
    times within the recent_window tokens using a Triton kernel.

    Args:
        tokens (torch.Tensor): 1D tensor of tokens (device='cuda', dtype=torch.int32).
        subseq_len (int): Length of the subsequence to check.
        recent_window (int): Number of recent tokens to consider.
        repeat_thresh (int): Threshold for the number of repetitions.

    Returns:
        bool: True if recent subsequence occurs at least repeat_thresh times, else False.
    """
    seq_len = tokens.size(0)
    if seq_len < subseq_len:
        return False

    # Determine the recent window start index
    window_start_idx = max(0, seq_len - recent_window - subseq_len)
    window_tokens = tokens[window_start_idx : seq_len - subseq_len].contiguous()
    window_size = window_tokens.size(0)

    if window_size < subseq_len:
        return False

    recent_subseq = tokens[-subseq_len:].contiguous()

    # Allocate results tensor
    result = torch.zeros(window_size - subseq_len + 1, dtype=torch.int32, device=tokens.device)

    # Kernel launch configuration
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(window_size - subseq_len + 1, BLOCK_SIZE),)

    # Launch Triton kernel
    subseq_match_kernel[grid](
        tokens_ptr=window_tokens,
        recent_ptr=recent_subseq,
        result_ptr=result,
        window_size=window_size,
        subseq_len=subseq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Count matches
    match_count = result.sum().item()

    return match_count >= repeat_thresh

