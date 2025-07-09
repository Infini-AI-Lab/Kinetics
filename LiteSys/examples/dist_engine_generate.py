import sys
sys.path.append("..")
from transformers import AutoTokenizer
import torch
from litesys.attention.batch_cache import AutoConfig
from litesys.engine.dist_engine import LLM
import json
import torch.distributed as dist
model_name = "Qwen/Qwen3-30B-A3B"
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model_name=model_name, max_seq_len=5120, max_batch_size=5,
device=device, dtype=torch.bfloat16, top_p=0.95, temperature=0.6, topk=20, model_class_str="qwen3moe")

requests = []
texts = [
r"Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
r"If the original price of a shirt is $25, a discount of 20% is applied. How much will you pay for the shirt after the discount?",
r"Tell me about Reinforcement Learning in 200 words."
]
for text in texts:
    requests.append(
        {
            "conversations": [{"role": "user", "content": text}]
        }
    )

processed_request = llm.offline_exec(requests, max_new_tokens=4096)

if local_rank == 0:
    with open("output.jsonl", "w", encoding="utf-8") as f:
        for item in processed_request:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

dist.destroy_process_group()