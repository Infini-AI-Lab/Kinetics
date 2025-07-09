from __future__ import annotations
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer
class QwenLayer:
    def __init__(self, layer_idx, device = "cpu") -> None:
        
        self.wq :torch.Tensor = None
        self.wk :torch.Tensor = None
        self.wv :torch.Tensor = None
        self.wo :torch.Tensor = None

        self.bq :torch.Tensor = None
        self.bk :torch.Tensor = None
        self.bv :torch.Tensor = None
        
        self.gate_proj :torch.Tensor = None 
        self.up_proj :torch.Tensor = None
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.layer_idx = layer_idx
        self.device = device
    def init_parameters(self, hf_layer: Qwen2DecoderLayer):

        self.wq :torch.Tensor= hf_layer.self_attn.q_proj.weight.detach()
        self.wk :torch.Tensor= hf_layer.self_attn.k_proj.weight.detach()
        self.wv :torch.Tensor= hf_layer.self_attn.v_proj.weight.detach()
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()

        self.bq :torch.Tensor= hf_layer.self_attn.q_proj.bias.detach()
        self.bk :torch.Tensor= hf_layer.self_attn.k_proj.bias.detach()
        self.bv :torch.Tensor= hf_layer.self_attn.v_proj.bias.detach()
        
        
        
        self.gate_proj = hf_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_layer.mlp.up_proj.weight.detach()
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

    
    def to(self, device:str = 'cuda:0', non_blocking = True):

        self.device = device
        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=non_blocking)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=non_blocking)
        self.wq = self.wq.to(device, non_blocking=non_blocking)
        self.wk = self.wk.to(device, non_blocking=non_blocking)
        self.wv = self.wv.to(device, non_blocking=non_blocking)
        self.bq = self.bq.to(device, non_blocking=non_blocking)
        self.bk = self.bk.to(device, non_blocking=non_blocking)
        self.bv = self.bv.to(device, non_blocking=non_blocking)
        self.wo = self.wo.to(device, non_blocking=non_blocking)
        self.gate_proj = self.gate_proj.to(device, non_blocking=non_blocking)
        self.up_proj = self.up_proj.to(device, non_blocking=non_blocking)
        self.down_proj =  self.down_proj.to(device, non_blocking=non_blocking)

    def copy(self, layer: QwenLayer):

        self.wq.copy_(layer.wq, non_blocking=True)
        self.wk.copy_(layer.wk, non_blocking=True)
        self.wv.copy_(layer.wv, non_blocking=True)
        
        self.bq.copy_(layer.bq, non_blocking=True)
        self.bk.copy_(layer.bk, non_blocking=True)
        self.bv.copy_(layer.bv, non_blocking=True)
        
        self.wo.copy_(layer.wo, non_blocking=True)
        self.gate_proj.copy_(layer.gate_proj, non_blocking=True)
        self.up_proj.copy_(layer.up_proj, non_blocking=True)
        self.down_proj.copy_(layer.down_proj, non_blocking=True)
        
        self.input_layernorm_weight.copy_(layer.input_layernorm_weight, non_blocking=True)
        self.post_attention_layernorm_weight.copy_(layer.post_attention_layernorm_weight, non_blocking=True)
        self.input_layernorm_variance_epsilon= layer.input_layernorm_variance_epsilon
        self.post_attention_layernorm_variance_epsilon = layer.post_attention_layernorm_variance_epsilon
        self.layer_idx = layer.layer_idx
        
    def alloc_space(self, layer: QwenLayer, device):

        self.device = device
        self.wq = torch.zeros_like(layer.wq).to(device)
        self.wk = torch.zeros_like(layer.wk).to(device)
        self.wv = torch.zeros_like(layer.wv).to(device)
        self.bq = torch.zeros_like(layer.bq).to(device)
        self.bk = torch.zeros_like(layer.bk).to(device)
        self.bv = torch.zeros_like(layer.bv).to(device)
        
        
        self.wo = torch.zeros_like(layer.wo).to(device)


        self.gate_proj = torch.zeros_like(layer.gate_proj).to(device)
        self.up_proj = torch.zeros_like(layer.up_proj).to(device)
        self.down_proj = torch.zeros_like(layer.down_proj).to(device)
        self.input_layernorm_weight = torch.zeros_like(layer.input_layernorm_weight).to(device)
        self.post_attention_layernorm_weight = torch.zeros_like(layer.post_attention_layernorm_weight).to(device)

class Qwen3Layer:
    def __init__(self, layer_idx, device = "cpu") -> None:
        
        self.wq :torch.Tensor = None
        self.wk :torch.Tensor = None
        self.wv :torch.Tensor = None
        self.wo :torch.Tensor = None
        
        self.gate_proj :torch.Tensor = None 
        self.up_proj :torch.Tensor = None
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0
        self.qnorm_weight :torch.Tensor = None
        self.qnorm_variance_epsilon :float = 0.0
        self.knorm_weight :torch.Tensor = None
        self.knorm_variance_epsilon :float = 0.0
        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.layer_idx = layer_idx
        self.device = device
    def init_parameters(self, hf_layer: Qwen3DecoderLayer):

        self.wq :torch.Tensor= hf_layer.self_attn.q_proj.weight.detach()
        self.wk :torch.Tensor= hf_layer.self_attn.k_proj.weight.detach()
        self.wv :torch.Tensor= hf_layer.self_attn.v_proj.weight.detach()
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()

        self.qnorm_weight :torch.Tensor= hf_layer.self_attn.q_norm.weight.detach()
        self.knorm_weight :torch.Tensor= hf_layer.self_attn.k_norm.weight.detach()
        self.qnorm_variance_epsilon =  hf_layer.self_attn.q_norm.variance_epsilon
        self.knorm_variance_epsilon =  hf_layer.self_attn.k_norm.variance_epsilon
        
        self.gate_proj = hf_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_layer.mlp.up_proj.weight.detach()
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

    
    def to(self, device:str = 'cuda:0', non_blocking = True):

        self.device = device
        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=non_blocking)
        self.qnorm_weight = self.qnorm_weight.to(device, non_blocking=non_blocking)
        self.knorm_weight = self.knorm_weight.to(device, non_blocking=non_blocking)
        
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=non_blocking)
        self.wq = self.wq.to(device, non_blocking=non_blocking)
        self.wk = self.wk.to(device, non_blocking=non_blocking)
        self.wv = self.wv.to(device, non_blocking=non_blocking)
        self.wo = self.wo.to(device, non_blocking=non_blocking)
        self.gate_proj = self.gate_proj.to(device, non_blocking=non_blocking)
        self.up_proj = self.up_proj.to(device, non_blocking=non_blocking)
        self.down_proj =  self.down_proj.to(device, non_blocking=non_blocking)

    def copy(self, layer: Qwen3Layer):

        self.wq.copy_(layer.wq, non_blocking=True)
        self.wk.copy_(layer.wk, non_blocking=True)
        self.wv.copy_(layer.wv, non_blocking=True)
    
        self.wo.copy_(layer.wo, non_blocking=True)
        self.gate_proj.copy_(layer.gate_proj, non_blocking=True)
        self.up_proj.copy_(layer.up_proj, non_blocking=True)
        self.down_proj.copy_(layer.down_proj, non_blocking=True)
        
        self.input_layernorm_weight.copy_(layer.input_layernorm_weight, non_blocking=True)
        self.qnorm_weight.copy_(layer.qnorm_weight, non_blocking=True)
        self.knorm_weight.copy_(layer.knorm_weight, non_blocking=True)
        self.post_attention_layernorm_weight.copy_(layer.post_attention_layernorm_weight, non_blocking=True)
        self.input_layernorm_variance_epsilon= layer.input_layernorm_variance_epsilon
        self.post_attention_layernorm_variance_epsilon = layer.post_attention_layernorm_variance_epsilon
        self.knorm_variance_epsilon = layer.knorm_variance_epsilon
        self.qnorm_variance_epsilon = layer.qnorm_variance_epsilon
        self.layer_idx = layer.layer_idx
        
    def alloc_space(self, layer: Qwen3Layer, device):

        self.device = device
        self.wq = torch.zeros_like(layer.wq).to(device)
        self.wk = torch.zeros_like(layer.wk).to(device)
        self.wv = torch.zeros_like(layer.wv).to(device)    
        self.wo = torch.zeros_like(layer.wo).to(device)


        self.gate_proj = torch.zeros_like(layer.gate_proj).to(device)
        self.up_proj = torch.zeros_like(layer.up_proj).to(device)
        self.down_proj = torch.zeros_like(layer.down_proj).to(device)
        self.input_layernorm_weight = torch.zeros_like(layer.input_layernorm_weight).to(device)
        self.post_attention_layernorm_weight = torch.zeros_like(layer.post_attention_layernorm_weight).to(device)
        self.qnorm_weight = torch.zeros_like(layer.qnorm_weight).to(device)
        self.knorm_weight = torch.zeros_like(layer.knorm_weight).to(device)
        
from transformers import AutoConfig
import torch.distributed as dist

class QwenTPLayer:
    def __init__(self, layer_idx, config: AutoConfig, device = "cpu") -> None:
        
        self.wq :torch.Tensor = None
        self.wk :torch.Tensor = None
        self.wv :torch.Tensor = None
        self.wo :torch.Tensor = None

        
        self.bq :torch.Tensor = None
        self.bk :torch.Tensor = None
        self.bv :torch.Tensor = None
        
        self.gate_proj :torch.Tensor = None 
        self.up_proj :torch.Tensor = None
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.cos_cache :torch.Tensor = None
        self.sin_cache :torch.Tensor = None

        self.layer_idx = layer_idx
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.world_size

        self.intermediate_size = config.intermediate_size
        self.mlp_slice = self.intermediate_size // self.world_size
    
    def init_parameters(self, hf_layer: Qwen2DecoderLayer):

        self.wq :torch.Tensor= hf_layer.self_attn.q_proj.weight.detach()
        self.wq :torch.Tensor= self.wq.split((self.num_heads * self.head_dim) // self.world_size, dim=0)[self.rank]
        if hf_layer.self_attn.q_proj.bias is not None:
            self.bq :torch.Tensor= hf_layer.self_attn.q_proj.bias.detach()
            self.bq :torch.Tensor= self.bq.split((self.num_heads * self.head_dim) // self.world_size, dim=0)[self.rank]
        
        self.wk :torch.Tensor= hf_layer.self_attn.k_proj.weight.detach()
        self.wk :torch.Tensor= self.wk.split(self.key_value_slicing, dim=0)[self.rank]
        if hf_layer.self_attn.k_proj.bias is not None:
            self.bk :torch.Tensor= hf_layer.self_attn.k_proj.bias.detach()
            self.bk :torch.Tensor= self.bk.split(self.key_value_slicing, dim=0)[self.rank]
            
        self.wv :torch.Tensor= hf_layer.self_attn.v_proj.weight.detach()
        self.wv :torch.Tensor= self.wv.split(self.key_value_slicing, dim=0)[self.rank]
        if hf_layer.self_attn.v_proj.bias is not None:
            self.bv :torch.Tensor= hf_layer.self_attn.v_proj.bias.detach()
            self.bv :torch.Tensor= self.bv.split(self.key_value_slicing, dim=0)[self.rank]
        
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()
        self.wo :torch.Tensor=self.wo.split(self.hidden_size // self.world_size, dim=1)[self.rank]

        

        self.gate_proj :torch.Tensor= hf_layer.mlp.gate_proj.weight.detach()
        self.gate_proj :torch.Tensor = self.gate_proj.split(self.mlp_slice, dim=0)[self.rank]

        self.up_proj :torch.Tensor= hf_layer.mlp.up_proj.weight.detach()
        self.up_proj :torch.Tensor= self.up_proj.split(self.mlp_slice, dim=0)[self.rank]

        self.down_proj :torch.Tensor= hf_layer.mlp.down_proj.weight.detach()
        self.down_proj :torch.Tensor= self.down_proj.split(self.mlp_slice, dim=1)[self.rank]
        

        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon
    
    def to(self, device:str = 'cuda:0'):

        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wq = self.wq.to(device, non_blocking=True)
        self.wk = self.wk.to(device, non_blocking=True)
        self.wv = self.wv.to(device, non_blocking=True)
        
        if self.bq is not None:
            self.bq = self.bq.to(device, non_blocking=True)
        if self.bk is not None:
            self.bk = self.bk.to(device, non_blocking=True)
        if self.bv is not None:
            self.bv = self.bv.to(device, non_blocking=True)

        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_proj = self.gate_proj.to(device, non_blocking=True)
        self.up_proj = self.up_proj.to(device, non_blocking=True)
        self.down_proj =  self.down_proj.to(device, non_blocking=True)
        

class Qwen3TPLayer:
    def __init__(self, layer_idx, config: AutoConfig, device = "cpu") -> None:
        
        self.wq :torch.Tensor = None
        self.wk :torch.Tensor = None
        self.wv :torch.Tensor = None
        self.wo :torch.Tensor = None
        
        self.gate_proj :torch.Tensor = None 
        self.up_proj :torch.Tensor = None
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0
        self.qnorm_weight :torch.Tensor = None
        self.qnorm_variance_epsilon :float = 0.0
        self.knorm_weight :torch.Tensor = None
        self.knorm_variance_epsilon :float = 0.0
        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.layer_idx = layer_idx
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.world_size

        self.intermediate_size = config.intermediate_size
        self.mlp_slice = self.intermediate_size // self.world_size
        
    def init_parameters(self, hf_layer: Qwen3DecoderLayer):

        self.wq :torch.Tensor= hf_layer.self_attn.q_proj.weight.detach()
        self.wq :torch.Tensor= self.wq.split((self.num_heads * self.head_dim) // self.world_size, dim=0)[self.rank]
        
        self.wk :torch.Tensor= hf_layer.self_attn.k_proj.weight.detach()
        self.wk :torch.Tensor= self.wk.split(self.key_value_slicing, dim=0)[self.rank]
        
        self.wv :torch.Tensor= hf_layer.self_attn.v_proj.weight.detach()
        self.wv :torch.Tensor= self.wv.split(self.key_value_slicing, dim=0)[self.rank]
        
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()
        self.wo :torch.Tensor= self.wo.split((self.num_heads * self.head_dim) // self.world_size, dim=1)[self.rank]
        

        self.qnorm_weight :torch.Tensor= hf_layer.self_attn.q_norm.weight.detach()
        self.knorm_weight :torch.Tensor= hf_layer.self_attn.k_norm.weight.detach()
        self.qnorm_variance_epsilon =  hf_layer.self_attn.q_norm.variance_epsilon
        self.knorm_variance_epsilon =  hf_layer.self_attn.k_norm.variance_epsilon
        
        self.gate_proj = hf_layer.mlp.gate_proj.weight.detach()
        self.gate_proj :torch.Tensor = self.gate_proj.split(self.mlp_slice, dim=0)[self.rank]

        self.up_proj = hf_layer.mlp.up_proj.weight.detach()
        self.up_proj :torch.Tensor= self.up_proj.split(self.mlp_slice, dim=0)[self.rank]

        self.down_proj = hf_layer.mlp.down_proj.weight.detach()
        self.down_proj :torch.Tensor= self.down_proj.split(self.mlp_slice, dim=1)[self.rank]

        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon


    def to(self, device:str = 'cuda:0'):

        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.qnorm_weight = self.qnorm_weight.to(device, non_blocking=True)
        self.knorm_weight = self.knorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)

        self.wq = self.wq.to(device, non_blocking=True)
        self.wk = self.wk.to(device, non_blocking=True)
        self.wv = self.wv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_proj = self.gate_proj.to(device, non_blocking=True)
        self.up_proj = self.up_proj.to(device, non_blocking=True)
        self.down_proj =  self.down_proj.to(device, non_blocking=True)


class Qwen3MoETPLayer:
    def __init__(self, layer_idx, config: AutoConfig, device = "cpu") -> None:
        
        self.wq :torch.Tensor = None
        self.wk :torch.Tensor = None
        self.wv :torch.Tensor = None
        self.wo :torch.Tensor = None
        
        self.gate :torch.Tensor = None 
        self.experts_gates :torch.Tensor = None 
        self.experts_up :torch.Tensor = None
        self.experts_down :torch.Tensor = None
        
        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0
        self.qnorm_weight :torch.Tensor = None
        self.qnorm_variance_epsilon :float = 0.0
        self.knorm_weight :torch.Tensor = None
        self.knorm_variance_epsilon :float = 0.0
        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.layer_idx = layer_idx
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.world_size
        self.num_experts = config.num_experts // self.world_size

        
        
    def init_parameters(self, hf_layer: Qwen3MoeDecoderLayer):

        self.wq :torch.Tensor= hf_layer.self_attn.q_proj.weight.detach()
        self.wq :torch.Tensor= self.wq.split((self.num_heads * self.head_dim) // self.world_size, dim=0)[self.rank]
        
        self.wk :torch.Tensor= hf_layer.self_attn.k_proj.weight.detach()
        self.wk :torch.Tensor= self.wk.split(self.key_value_slicing, dim=0)[self.rank]
        
        self.wv :torch.Tensor= hf_layer.self_attn.v_proj.weight.detach()
        self.wv :torch.Tensor= self.wv.split(self.key_value_slicing, dim=0)[self.rank]
        
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()
        self.wo :torch.Tensor= self.wo.split((self.num_heads * self.head_dim) // self.world_size, dim=1)[self.rank]
        

        self.qnorm_weight :torch.Tensor= hf_layer.self_attn.q_norm.weight.detach()
        self.knorm_weight :torch.Tensor= hf_layer.self_attn.k_norm.weight.detach()
        self.qnorm_variance_epsilon =  hf_layer.self_attn.q_norm.variance_epsilon
        self.knorm_variance_epsilon =  hf_layer.self_attn.k_norm.variance_epsilon
        
        self.gate = hf_layer.mlp.gate.weight.detach()
        
        self.experts_gates = torch.cat([hf_layer.mlp.experts[i].gate_proj.weight.detach() for i in range(self.rank * self.num_experts, (self.rank + 1) * self.num_experts)], dim=0)
        self.experts_up = torch.cat([hf_layer.mlp.experts[i].up_proj.weight.detach() for i in range(self.rank * self.num_experts, (self.rank + 1) * self.num_experts)], dim=0)
        self.experts_down = torch.cat([hf_layer.mlp.experts[i].down_proj.weight.detach() for i in range(self.rank * self.num_experts, (self.rank + 1) * self.num_experts)], dim=1)
        self.experts_down.transpose_(0, 1)
        self.experts_down = self.experts_down.reshape(self.num_experts, -1, self.hidden_size)
        
       
        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon


    def to(self, device:str = 'cuda:0'):

        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.qnorm_weight = self.qnorm_weight.to(device, non_blocking=True)
        self.knorm_weight = self.knorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)

        self.wq = self.wq.to(device, non_blocking=True)
        self.wk = self.wk.to(device, non_blocking=True)
        self.wv = self.wv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate = self.gate.to(device, non_blocking=True)
        self.experts_up = self.experts_up.to(device, non_blocking=True)
        self.experts_down = self.experts_down.to(device, non_blocking=True)
        self.experts_gates = self.experts_gates.to(device, non_blocking=True)