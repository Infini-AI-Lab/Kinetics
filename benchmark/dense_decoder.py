import torch
import torch.nn as nn
import torch.nn.functional as F
import flashinfer
from einops import rearrange
import time
import math
import torch._dynamo
import torch._inductor.config
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True
torch._functorch.config.enable_autograd_cache = True
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = False

from model_utils import layer_norm, qk_norm
from transformers.activations import ACT2FN
from transformers import AutoConfig
import sys
torch.library.define(
    "mylib::update_kv",
    "(Tensor k, Tensor v, Tensor kv_append_indptr, Tensor(a!) kv_cache, Tensor kv_page_indices, Tensor kv_page_indptr, Tensor cachelen) -> ()",
)

@torch.library.impl("mylib::update_kv", "cuda")
def update_kv(
            k,
            v,
            kv_append_indptr,
            kv_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_page_last_len,
        ):
    flashinfer.append_paged_kv_cache(
            k,
            v,
            kv_append_indptr,
            kv_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_page_last_len,
        )
    
@torch.library.register_fake("mylib::update_kv")
def update_kv_abstract(
            k,
            v,
            kv_append_indptr,
            kv_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_page_last_len,
        ):
    return None

# also wrap qk_norm and layer_norm using torch.ops
torch.library.define(
    "mylib::qk_norm",
    "(Tensor q, float variance_epsilon, Tensor weight) -> Tensor",
)

@torch.library.impl("mylib::qk_norm", "cuda")
def qk_norm_impl(q, variance_epsilon, weight):
    return qk_norm(q, variance_epsilon, weight)

@torch.library.register_fake("mylib::qk_norm")
def qk_norm_abstract(q, variance_epsilon, weight):
    return torch.empty_like(q)

torch.library.define(
    "mylib::layer_norm",
    "(Tensor input, float variance_epsilon, Tensor weight) -> Tensor",
)

@torch.library.impl("mylib::layer_norm", "cuda")
def layer_norm_impl(input, variance_epsilon, weight):
    return layer_norm(input, variance_epsilon, weight)

@torch.library.register_fake("mylib::layer_norm")
def layer_norm_abstract(input, weight, variance_epsilon):
    return torch.empty_like(input)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

class DecoderLayer(nn.Module):
    def __init__(self, config, world_size,
                 batch_size, prefix_len, max_len, page_size, 
                 device, dtype):
        super().__init__()
        self.num_heads = config.num_attention_heads // world_size
        self.num_kv_heads = config.num_key_value_heads // world_size
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.dim = config.hidden_size
        self.intermediate_size = config.intermediate_size // world_size
        
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.eos_tokens = config.eos_token_id if (isinstance(config.eos_token_id, list)) else [config.eos_token_id]
        self.activation = ACT2FN[config.hidden_act]
        self.ffn_frac = getattr(config, "ffn_frac", 1.0)
        
        self.qnorm_weight = nn.Parameter(torch.ones(self.head_dim))
        self.qnorm_variance_epsilon = config.rms_norm_eps
        self.knorm_weight = nn.Parameter(torch.ones(self.head_dim))
        self.knorm_variance_epsilon = config.rms_norm_eps
        self.input_layernorm_weight = nn.Parameter(torch.ones(self.dim))
        self.input_layernorm_variance_epsilon = config.rms_norm_eps
        self.post_attention_layernorm_weight = nn.Parameter(torch.ones(self.dim))
        self.post_attention_layernorm_variance_epsilon = config.rms_norm_eps
        
        total_head_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.wqkv = nn.Linear(self.dim, total_head_dim, bias=True)
        self.wo = nn.Linear(self.head_dim * self.num_heads, self.dim, bias=False)
        
        self.up_proj = nn.Linear(self.dim, self.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(self.dim, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.dim, bias=False)
        
        self.batch_size = batch_size
        self.prefix_len = prefix_len
        self.page_size = page_size
        self.device = device
        self.dtype = dtype
        
        # setup flashinfer buffers and KV Cache
        max_num_pages = max_len // page_size
        num_pages_per_req = prefix_len // page_size
        paged_kv_indices = torch.arange(num_pages_per_req + 1, dtype=torch.int32, device=device) * batch_size + torch.arange(batch_size, dtype=torch.int32, device=device)[:, None]
        self.paged_kv_indices = paged_kv_indices.view(-1)
        self.paged_kv_indptr = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(torch.full((batch_size,), num_pages_per_req + 1, dtype=torch.int32, device=device), dim=0)
        ]).to(torch.int32)
        
        self.last_kv_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.qkv_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, "NHD",
            use_tensor_cores=True,
            use_cuda_graph=True,
            paged_kv_indices_buffer=self.paged_kv_indices,
            paged_kv_indptr_buffer=self.paged_kv_indptr,
            paged_kv_last_page_len_buffer=self.last_kv_lens
        )
        self.kv_cache = torch.randn(
            max_num_pages * batch_size, 2, page_size, 
            self.num_kv_heads, self.head_dim, dtype=dtype, device=device
        )
        
        # register decode_wrapper run functions
        torch.library.define(
            "mylib::decode",
            "(Tensor q, Tensor kv_cache) -> Tensor",
        )        
        
        @torch.library.impl("mylib::decode", "cuda")
        def decode_impl(q, kv_cache):
            return self.decode_wrapper.run(q, kv_cache)
        
        @torch.library.register_fake("mylib::decode")
        def decode_abstract(q, kv_cache):
            return torch.empty_like(q)
        
        self.decode_run = torch.ops.mylib.decode
        
    def prepare_wrapper(self):
        self.decode_wrapper.plan(
            self.paged_kv_indptr,
            self.paged_kv_indices,
            self.last_kv_lens,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            pos_encoding_mode="NONE",
            data_type="float16"
        )
        
    @torch.inference_mode()
    def forward(self, 
        hidden_states,
    ):
        residual = hidden_states
        
        hidden_states = torch.ops.mylib.layer_norm(hidden_states, self.input_layernorm_variance_epsilon, self.input_layernorm_weight)
        
        bsz, q_len, _ = hidden_states.size()
        q_dim, kv_dim = self.head_dim * self.num_heads, self.head_dim * self.num_kv_heads
        split_sizes = (q_dim, kv_dim, kv_dim)
        proj = self.wqkv(hidden_states).contiguous()  
        query_states, key_states, value_states = proj.split(split_sizes, dim=-1)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).contiguous()
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).contiguous()
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).squeeze(dim=1).contiguous()
        
        query_states = torch.ops.mylib.qk_norm(query_states, self.qnorm_variance_epsilon, self.qnorm_weight).squeeze(dim=1)
        key_states = torch.ops.mylib.qk_norm(key_states, self.knorm_variance_epsilon, self.knorm_weight).squeeze(dim=1)
        
        torch.ops.mylib.update_kv(
            key_states, value_states, self.qkv_indptr,
            self.kv_cache,
            self.paged_kv_indices, self.paged_kv_indptr, self.last_kv_lens
        )
        
        o = self.decode_run(
            query_states, self.kv_cache
        ) 
        o = o.reshape(bsz, q_len, self.head_dim * self.num_heads)
        hidden_states = self.wo(o)
        
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = torch.ops.mylib.layer_norm(hidden_states, self.post_attention_layernorm_variance_epsilon, self.post_attention_layernorm_weight)
            
        up = self.up_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        gate = self.activation(gate)
        hidden_states = gate * up
        hidden_states = self.down_proj(hidden_states)
        
        hidden_states = residual + hidden_states
        return hidden_states
        
