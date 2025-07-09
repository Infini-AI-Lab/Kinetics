import torch
from ..models.auto_model import AutoModelLM
from ..attention.batch_cache import BatchKVManager
from transformers import AutoConfig, AutoTokenizer
from .utils import copy_new_tokens, recent_subseq_repeat_check
import copy
from ..logging_config import setup_logger  
from tqdm import tqdm
import time
import math
import torch.distributed as dist
logger = setup_logger()



class LLM:
    def __init__(self, 
    model_name: str, 
    dtype: torch.dtype, 
    device: str, 
    max_seq_len: int,
    top_p: float,
    temperature: float,
    top_k: int,
    eos: list[int] = [151643, 151645],
    model_class_str: str = "qwen2",
    repeat_check: bool = True,
    repeat_check_window: int = 1024,
    repeat_block_size: int = 64,
    **sparsity_kwargs):
    
        self.model_name = model_name
        self.dtype = dtype
        self.device = device
        self.max_seq_len = max_seq_len
        
        self.top_p = top_p
        self.temperature = temperature
        self.top_k = top_k
        
        self.repeat_check = repeat_check
        self.repeat_block_size = repeat_block_size
        self.repeat_check_window = repeat_check_window
        self.repeat_thr = max((self.repeat_check_window // self.repeat_block_size) - 1, 2)
        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.world_size = dist.get_world_size()
        model_class = AutoModelLM.from_pretrained(model_class_str, tp=True)
        
        self.model_executor = model_class(self.model_name, self.max_seq_len, self.device, self.dtype)
        self.model_executor.alloc()
        
        head_dim = getattr(self.model_config, "head_dim", self.model_config.hidden_size // self.model_config.num_attention_heads)
        kv_size_for_single_request = head_dim * self.model_config.num_key_value_heads * self.model_config.num_hidden_layers * self.max_seq_len * 4
        model_size = getattr(self.model_executor, "num_parameters") * 2.5
        total_memory = torch.cuda.get_device_properties(device).total_memory * self.world_size
        self.max_batch_size = math.floor((total_memory * 0.9 - model_size) / kv_size_for_single_request)
        if dist.get_rank() == 0:
                logger.info(f"  Model Size    {model_size / (1024 * 1024 * 1024)} GB")
                logger.info(f"  KV Size    {kv_size_for_single_request / (1024 * 1024 * 1024)} GB")
                logger.info(f"  Available GPU VRAM Size    {0.9 * total_memory / (1024 * 1024 * 1024)} GB")
                logger.info(f"  Batch Size Auto Set    {self.max_batch_size}")
                
        
        self.kv_cache = BatchKVManager(self.model_config, self.max_batch_size, self.max_seq_len, self.device, self.dtype, **sparsity_kwargs)
        
        self.slots_occupy_status = [False for _ in range(self.max_batch_size)]
        self.slots_offsets = torch.zeros((self.max_batch_size,), dtype=torch.long, device=self.device)
        self.slots_prompt_len = torch.zeros((self.max_batch_size,), dtype=torch.long, device=self.device)
        self.slots_tokens = torch.zeros((self.max_batch_size, self.max_seq_len), dtype=torch.long, device=self.device)
        
        self.eos_tensor = eos
        self.eos_tensor = torch.tensor(self.eos_tensor, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if dist.get_rank() == 0:
            logger.info("LLM Initialized:")
            logger.info(f"  Model Name      : {self.model_name}")
            logger.info(f"  Model Class     : {model_class}")
            logger.info(f"  Device          : {self.device}")
            logger.info(f"  Dtype           : {self.dtype}")
            logger.info(f"  Max Seq Length  : {self.max_seq_len}")
            logger.info(f"  Max Batch Size  : {self.max_batch_size}")
            logger.info(f"  Top-p           : {self.top_p}")
            logger.info(f"  Temperature     : {self.temperature}")
        
    def clear_request(self, batch_idx: int):
        
        self.slots_occupy_status[batch_idx] = False
        self.slots_offsets[batch_idx] = 0
        self.slots_prompt_len[batch_idx] = 0
        self.slots_tokens[batch_idx].zero_()
        self.kv_cache.clear_cache(batch_idx)
       
    def prefill(self, batch_idx: int, input_ids: torch.LongTensor):
        
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
        
        self.slots_tokens[batch_idx][:seq_len].copy_(input_ids[0])
        self.slots_offsets[batch_idx] = seq_len
        self.slots_prompt_len[batch_idx] = seq_len
        self.slots_occupy_status[batch_idx] = True
        
        batch_idx = torch.tensor([batch_idx], device=self.device)
        input_ids = input_ids.to(self.device)
        
        logits = self.model_executor.inference(input_ids, position_ids, batch_idx, self.kv_cache)[:,-1:,:]
        next_tokens = self.sample_next_tokens(logits)
        
        self.slots_tokens[batch_idx.item()][seq_len] = next_tokens.item()
        
        
    def decode(self, batch_idx: torch.LongTensor):
        
        offset = self.slots_offsets[batch_idx].unsqueeze(1)
        
        input_ids = self.slots_tokens[batch_idx].gather(dim=-1, index=offset)
        
        logits = self.model_executor.inference(input_ids, offset, batch_idx, self.kv_cache)
        next_tokens = self.sample_next_tokens(logits)    
        dist.broadcast(next_tokens, src=0)
        self.slots_offsets[batch_idx] += 1
        copy_new_tokens(self.slots_tokens, next_tokens.squeeze(1), self.slots_offsets[batch_idx], batch_idx)
        is_eos = (next_tokens == self.eos_tensor.view(1, -1)).any(dim=1)
        
        return is_eos
        
    def sample_next_tokens(self, logits: torch.Tensor):
        """
        logits: Tensor of shape [batch_size, 1, vocab_size]
        returns: Tensor of shape [batch_size, 1]
        """
        
        if self.temperature < 0.02:
            return logits.argmax(dim=-1)
        

        logits = logits.squeeze(1)

        # Top-k filtering
        if self.top_k > 0:
            topk_values, topk_indices = torch.topk(logits, self.top_k, dim=-1)
            topk_logits = torch.full_like(logits, float('-inf'))
            topk_logits.scatter_(dim=-1, index=topk_indices, src=topk_values)
            logits = topk_logits
        
        probs = torch.softmax(logits / self.temperature, dim=-1)  # [batch_size, vocab_size]

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)  # [batch_size, vocab_size]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > self.top_p

        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = 0

    
        sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)

        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        sampled_indices = torch.multinomial(sorted_probs, num_samples=1)  # [batch_size, 1]

        next_tokens = sorted_indices.gather(dim=1, index=sampled_indices)  # [batch_size, 1]

        return next_tokens


    def offline_exec(self, requests: list, max_new_tokens: int): 
        
        processed_requests = []
        request_idx = 0
        batch_size = self.max_batch_size
        total_requests = len(requests)
        if dist.get_rank() == 0:
            logger.critical("Offline JOB Started:")
            logger.critical(f"  Total Requests      : {total_requests}")
            logger.critical(f"  Max New Tokens      : {max_new_tokens}")
        
        gen_token_counts = torch.zeros((batch_size,), dtype=torch.long, device=self.device)

        # Track metadata for each slot
        slot_to_request_meta = [None for _ in range(batch_size)]
        total_generate_tokens = 0
        torch.cuda.synchronize()
        t1 = time.time()
        with tqdm(total=total_requests, desc="Processing Requests", ncols=100) as pbar:
            # Initialize the batch
            for i in range(min(batch_size, total_requests)):
                req = requests[request_idx]
                self.preprocess_request(req)
                input_ids = torch.tensor(req["input_ids"]).unsqueeze(0)
                self.prefill(i, input_ids)
                gen_token_counts[i] = 1
                slot_to_request_meta[i] = req
                request_idx += 1

            while any(self.slots_occupy_status):
                active_slots = [i for i, status in enumerate(self.slots_occupy_status) if status]
                active_batch_idx = torch.tensor(active_slots, device=self.device)

                is_eos = self.decode(active_batch_idx)

                for i, b in enumerate(active_slots):
                    gen_token_counts[b] += 1
                    
                    seq_len = self.slots_offsets[b].item()
                    prompt_len = self.slots_prompt_len[b].item()
                        
                    is_repeat = False if not (self.repeat_check and seq_len > self.repeat_check_window) else recent_subseq_repeat_check(self.slots_tokens[b, prompt_len:seq_len], 
                        subseq_len=self.repeat_block_size, recent_window=self.repeat_check_window, repeat_thresh=self.repeat_thr)

                    track_back = self.repeat_block_size if is_repeat else 0
                    
                    if is_eos[i] or gen_token_counts[b] >= max_new_tokens or is_repeat:
                        
                        tokens = self.slots_tokens[b, prompt_len:seq_len - track_back].tolist()
                        
                        # Prepare output dict with metadata
                        total_generate_tokens += seq_len - prompt_len
                        result = copy.deepcopy(slot_to_request_meta[b])
                        result["output_tokens"] = tokens
                        self.postprocess_request(result)
                        processed_requests.append(result)

                        # Clear and reuse
                        self.clear_request(b)
                        slot_to_request_meta[b] = None
                        pbar.update(1)
                        if request_idx < total_requests:
                            req = requests[request_idx]
                            self.preprocess_request(req)
                            input_ids = torch.tensor(req["input_ids"]).unsqueeze(0)
                            self.prefill(b, input_ids)
                            gen_token_counts[b] = 1
                            slot_to_request_meta[b] = req
                            request_idx += 1

        torch.cuda.synchronize()
        t2 = time.time()
        
        logger.info("Total Generated Tokens {:.2f} | Throughput {:.2f} TPS".format(total_generate_tokens, total_generate_tokens/(t2-t1)))
        
        return processed_requests

    
    def preprocess_request(self, req):
        if "input_ids" in req:
            return
        
        prompt = self.tokenizer.apply_chat_template(
                req["conversations"],
                tokenize=False,
                add_generation_prompt=True
            )

        tokens = self.tokenizer.encode(prompt)
        
        req["input_ids"] = tokens
    
    def postprocess_request(self, req):
        req["output_text"] = self.tokenizer.decode(req["output_tokens"])

                    
                
                
                
                