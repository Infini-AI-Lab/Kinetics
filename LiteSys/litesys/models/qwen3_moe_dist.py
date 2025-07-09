from time import sleep
from transformers import Qwen3MoeForCausalLM, Qwen3MoeConfig
import torch
import torch.nn.functional as F
import gc
import flashinfer
from .qwen_layer import Qwen3MoETPLayer
from .base import LLMBase
from .model_utils import apply_rotary_pos_emb, layer_norm, qk_norm
from ..attention.batch_cache import BatchKVManager
from transformers.activations import ACT2FN
import torch.distributed as dist

class Qwen3MoETP(LLMBase):
    def __init__(self, 
        model_name: str,
        max_length :int = 32768, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16) -> None:
        
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.config = Qwen3MoeConfig.from_pretrained(model_name)
        
        self.model_name = model_name
        self.max_length = max_length
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads // self.world_size
        self.head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        self.num_key_value_heads = self.config.num_key_value_heads // self.world_size
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.eos_tokens = self.config.eos_token_id if (isinstance(self.config.eos_token_id, list)) else [self.config.eos_token_id]
        self.activation = ACT2FN[self.config.hidden_act]
        self.ffn_frac = getattr(self.config, "ffn_frac", 1.0)
        self.num_parameters = 0
        self.topk = self.config.num_experts_per_tok
        self.num_experts = self.config.num_experts
        self.num_local_experts = self.config.num_experts // self.world_size
    def alloc(self):
        hf_model = Qwen3MoeForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.num_parameters = sum(p.numel() for p in hf_model.parameters()) 
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        if self.config.tie_word_embeddings:
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = hf_model.lm_head.weight.detach().to(self.device)

        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        
        self.inv_freq = hf_model.model.rotary_emb.inv_freq.detach().to(self.device)
        self.attention_scaling = hf_model.model.rotary_emb.attention_scaling
        position_ids = torch.arange(0, self.max_length).unsqueeze(0).to(self.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[0]
        self.sin_cache = emb.sin()[0]
        self.cos_cache = self.cos_cache * self.attention_scaling
        self.sin_cache = self.sin_cache * self.attention_scaling
        self.cos_cache = self.cos_cache.to(self.dtype)
        self.sin_cache = self.sin_cache.to(self.dtype)
        
        self.layers :list[Qwen3MoETPLayer] = []
        
        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = Qwen3MoETPLayer(idx, config=self.config)
            layer.init_parameters(hf_layer=hf_layer)
            layer.to(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()
            
        self.num_layers = len(self.layers)
        
    @torch.inference_mode()
    def layer_compute(self, 
            buffer: Qwen3MoETPLayer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor,
            batch_idx: torch.LongTensor,
            kv_cache: BatchKVManager
            ):

        residual = hidden_states
        
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        
        bsz, q_len, _ = hidden_states.size()
        query_states = F.linear(hidden_states, buffer.wq)
        key_states = F.linear(hidden_states, buffer.wk)
        value_states = F.linear(hidden_states, buffer.wv)
        
        query_states = qk_norm(query_states.view(bsz, q_len, self.num_heads, self.head_dim), buffer.qnorm_variance_epsilon, buffer.qnorm_weight).transpose(1, 2)
        key_states = qk_norm(key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim), buffer.knorm_variance_epsilon, buffer.knorm_weight).transpose(1, 2)
        
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, self.cos_cache, self.sin_cache, position_ids)
        
        hidden_states = kv_cache.compute_attention(
            query_states, key_states, value_states, layer_idx, batch_idx
        )
        
        
        hidden_states = F.linear(hidden_states, buffer.wo)
        dist.all_reduce(hidden_states, dist.ReduceOp.SUM)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)

        bsz, q_len, d = hidden_states.size()
        hidden_states = hidden_states.view(-1, d)
        router_logits = F.linear(hidden_states, buffer.gate)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk_weights, selected_experts = torch.topk(routing_weights, self.topk, dim=-1)
        normalized_topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        full_routing_weights = torch.zeros_like(routing_weights)
        full_routing_weights.scatter_(1, selected_experts, normalized_topk_weights)
        routing_weights = full_routing_weights.to(hidden_states.dtype)
        routing_weights = routing_weights[:,self.rank * self.num_local_experts:(self.rank + 1) * self.num_local_experts]
        up = F.linear(hidden_states, buffer.experts_up)
        gate = F.linear(hidden_states, buffer.experts_gates)
        gate = self.activation(gate)
        hidden_states = gate * up
        hidden_states = hidden_states.reshape(bsz * q_len, self.num_local_experts, -1).transpose(0, 1)
        hidden_states = torch.matmul(hidden_states, buffer.experts_down).transpose(0, 1)
        hidden_states = (hidden_states * routing_weights.unsqueeze(-1)).sum(dim=1)
        hidden_states = hidden_states.reshape(bsz, q_len, d)
        dist.all_reduce(hidden_states, dist.ReduceOp.SUM)
        
        hidden_states = residual + hidden_states
        return hidden_states


    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            batch_idx: torch.LongTensor,
            kv_cache: BatchKVManager):
        
        self.validate_input(input_ids, position_ids, batch_idx)
        
        hidden_states = F.embedding(input_ids, self.embed_tokens)  
        
        for idx in range(self.num_layers):
                hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids, batch_idx, kv_cache=kv_cache)
    
        b, s, h = hidden_states.shape

        hidden_states = hidden_states.reshape(b * s, h)
        hidden_states = flashinfer.rmsnorm(hidden_states, self.norm_weight, self.norm_variance_epsilon)
        hidden_states = hidden_states.reshape(b, s, h)
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits

    def validate_input(self, 
        input_ids: torch.LongTensor, 
        position_ids: torch.LongTensor,
        batch_idx:  torch.LongTensor):
        
        bsz, seq_len = input_ids.shape
        assert (bsz == 1) or (seq_len) == 1, f"Operation Not Supported with input shape {input_ids.shape}"
        assert input_ids.shape == position_ids.shape, f"input ids {input_ids.shape} is expected to have the same shape as position ids {position_ids.shape}"
        assert bsz == batch_idx.shape[0], f"input ids {input_ids.shape[0]} is expected to have the same batch size as batch idx {batch_idx.shape}"
        
        
        
            
        