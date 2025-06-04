import torch
import flashinfer

def layer_norm(
    hidden_states: torch.Tensor,
    layernorm_variance_epsilon: float,
    layernorm_weight: torch.Tensor,
):  
    b, s, h = hidden_states.shape
    
    hidden_states = hidden_states.reshape(b * s, h)
    hidden_states = flashinfer.rmsnorm(hidden_states, layernorm_weight, layernorm_variance_epsilon)
    hidden_states = hidden_states.reshape(b, s, h)
    return hidden_states

def qk_norm(
    hidden_states: torch.Tensor,
    layernorm_variance_epsilon: float,
    layernorm_weight: torch.Tensor,
):  
    b, s, h, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b * s * h, d)
    hidden_states = flashinfer.rmsnorm(hidden_states, layernorm_weight, layernorm_variance_epsilon)
    hidden_states = hidden_states.reshape(b, s, h, d)
    return hidden_states

def fused_layer_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    layernorm_variance_epsilon: float,
    layernorm_weight: torch.Tensor,
):  
    b, s, h = hidden_states.shape
    
    hidden_states = hidden_states.reshape(b * s, h)
    residual = residual.reshape(b * s, h)
    flashinfer.norm.fused_add_rmsnorm(hidden_states, residual, layernorm_weight, layernorm_variance_epsilon)
    hidden_states = hidden_states.reshape(b, s, h)
    residual = residual.reshape(b, s, h)
    
    return hidden_states, residual

def layer_norm_gemma(
    hidden_states: torch.Tensor,
    layernorm_variance_epsilon: float,
    layernorm_weight: torch.Tensor,
):  
    b, s, h = hidden_states.shape
    
    hidden_states = hidden_states.reshape(b * s, h)
    hidden_states = flashinfer.gemma_rmsnorm(hidden_states, layernorm_weight, layernorm_variance_epsilon)
    hidden_states = hidden_states.reshape(b, s, h)
    return hidden_states

def capture_graph(
    llm, decoding_seqlen :int =1, mempool=None, n_warmups :int=3
):
    device = llm.device
    bsz = llm.batch_size
    static_input_ids = torch.full((bsz, decoding_seqlen), 0, dtype=torch.long, device=device)
    static_position_ids = torch.full((bsz, decoding_seqlen), 0, dtype=torch.long, device=device)
    static_storage_ids = torch.arange(decoding_seqlen, dtype=torch.long, device=device)
    static_attn_mask = torch.full((decoding_seqlen, llm.max_length), 1, dtype=torch.bool, device=device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_logits = llm.inference(
                    input_ids=static_input_ids, 
                    position_ids=static_position_ids, 
                    attention_mask=static_attn_mask,
                    storage_ids=static_storage_ids, 
                    )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_logits = llm.inference(
                input_ids=static_input_ids,  
                position_ids=static_position_ids, 
                attention_mask=static_attn_mask,
                storage_ids=static_storage_ids,
                )
    def run(input_ids, storage_ids, position_ids, attention_mask):
        static_input_ids.copy_(input_ids)
        static_storage_ids.copy_(storage_ids)
        static_position_ids.copy_(position_ids)
        static_attn_mask.copy_(attention_mask)
        graph.replay()
        return static_logits.clone()
    
    return run


def topk_along_last_dim(input_tensor, k):
    
    topk_values, topk_indices = torch.topk(input_tensor, k=k, dim=-1)
    output = torch.zeros_like(input_tensor)
    output.scatter_(-1, topk_indices, topk_values)
    
    return output

def topk_along_last_dim_abs(input_tensor, k):
    abs_tensor = torch.abs(input_tensor)
    topk_values, topk_indices = torch.topk(abs_tensor, k=k, dim=-1)

    original_topk_values = torch.gather(input_tensor, dim=-1, index=topk_indices)

    output = torch.zeros_like(input_tensor)
    output.scatter_(-1, topk_indices, original_topk_values)

    return output

def topk_along_last_dim_abs_inplace_(input_tensor: torch.Tensor, k: int):
    
    abs_tensor = input_tensor.abs()
    
    _, indices_to_zero = torch.topk(abs_tensor, input_tensor.size(-1) - k, dim=-1, largest=False)
    
    input_tensor.scatter_(-1, indices_to_zero, 0)

    return input_tensor