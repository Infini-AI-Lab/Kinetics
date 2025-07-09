import sys
sys.path.append("..")
from litesys.helper import generate_all_requests
from litesys.engine.dist_engine import LLM
from datasets import load_dataset
import os
import argparse
import torch
import jsonlines
from datasets import load_dataset
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from datetime import datetime
import torch.distributed as dist
MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

import random
import numpy as np

TASKS = [
    "amc23",
    "math500",
    "aime24",
    "aime25",
    "gsm8k"
]

DATAMAP = {
    "amc23": "math-ai/amc23",
    "aime24": "HuggingFaceH4/aime_2024",
    "aime25": "math-ai/aime25",
    "gsm8k": "InfiniAILab/gsm8k",
    "math500": "HuggingFaceH4/MATH-500"
}


SPLITMAP = {
    "amc23": "test",
    "aime24": "train",
    "aime25": "test",
    "gsm8k": "test",
    "math500": "test"
}

COLUMNMAP = {
    "amc23": "question",
    "aime24": "problem",
    "aime25": "problem",
    "gsm8k": "question",
    "math500": "problem"
}

ANSWERMAP = {
    "amc23": "answer",
    "aime24": "answer",
    "aime25": "answer",
    "gsm8k": "answer",
    "math500": "solution"
}

JUDGEMAP = {
    "amc23": "expr",
    "aime24": "expr",
    "aime25": "expr",
    "gsm8k": "expr",
    "math500": "latex"
}
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval(args, trial, task):
    
    set_seed(args.seed)
    result_list = []    
    data = DATAMAP[task]
    split = SPLITMAP[task]
    column = COLUMNMAP[task]
    
    answer_column = ANSWERMAP[task]
    judge = JUDGEMAP[task]
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    llm = LLM(model_name=args.model, model_class_str=args.model_class, dtype=torch.bfloat16, device=f"cuda:{rank}",
        max_seq_len=args.max_len, top_p=0.95, temperature=0.6, top_k=20,
        attn_topk=args.attn_topk, attn_nsample=args.attn_nsample, attn_local=args.attn_local, attn_block_topk=args.attn_block_topk,
        attn_block_size=args.attn_block_size, attn_stride=args.attn_stride,
        attn_random_select=args.attn_random_select, attn_random_nsample=args.attn_random_nsample,
        quest_block_size=args.quest_block_size, quest_topk_blocks=args.quest_topk_blocks
    )
    dataset = load_dataset(data, split=split)
    all_processed_request = []
    requests = generate_all_requests(dataset=dataset, field_name=column, data_format=MATH_QUERY_TEMPLATE, trial=trial, rank=rank, world_size=world_size)
    processed_request = llm.offline_exec(requests, args.gen_len)
    all_processed_request.extend(processed_request)
    
    if judge == "expr":
        gold_metric = multilingual_extractive_match_metric(
            language=Language.ENGLISH,
            fallback_mode="first_match",
            precision=5,
            gold_extraction_target=(ExprExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
            aggregation_function=max,
        )
    
    else:
        gold_metric = multilingual_extractive_match_metric(
            language=Language.ENGLISH,
            fallback_mode="first_match",
            precision=5,
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
            aggregation_function=max,
        )

    for data in all_processed_request:
        golds = [data[answer_column]]
        target = Doc(query=data[column],choices=golds, gold_index=0)
        predictions = data["output_text"]
        try:
            result = gold_metric.compute(golds=golds,predictions=[predictions],formatted_doc=target)
        except:
            result = {"extractive_match": 0.0}
            
        result_list.append(
                    {   
                        "score": result["extractive_match"] * 100,
                        "prediction": [predictions],
                        "choices": golds,
                        "query": data[column]
                    }
                )
    
    return result_list   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-0.6B", help='Model name')
    parser.add_argument('--model_class', type=str, default="qwen3", help='Model Class')
    parser.add_argument('--task', type=str, default="amc23", help='Task')
    parser.add_argument('--max_len', type=int, default=32768, help='Max Length')
    parser.add_argument('--gen_len', type=int, default=30768, help='Max Generation Length')
    parser.add_argument('--trial', type=int, default=4, help='Number of trials')
    parser.add_argument('--output_dir', type=str, default="results", help='output dir')
    parser.add_argument('--attn_topk', type=int, default=None, help='Attention Top-k')
    parser.add_argument('--attn_nsample', type=int, default=None, help='Attention N-sample')
    parser.add_argument('--attn_local', type=int, default=None, help='Attention Local')
    parser.add_argument('--attn_block_topk', type=int, default=None, help='Attention Block Top-k')
    parser.add_argument('--attn_block_size', type=int, default=16, help='Attention Block Size')
    parser.add_argument('--attn_stride', type=int, default=None, help='Attention Stride')
    parser.add_argument('--attn_random_select', type=int, default=None, help='Attention Random Select')
    parser.add_argument('--attn_random_nsample', type=int, default=None, help='Attention Random N-sample')
    parser.add_argument('--quest_block_size', type=int, default=16, help='Quest Block Size')
    parser.add_argument('--quest_topk_blocks', type=int, default=None, help='Quest Top-k Blocks')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    args = parser.parse_args()

    # Initialize distributed environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank)
    else:
        raise RuntimeError("Distributed environment not initialized. Please use torchrun to launch the script.")

    output_dir = args.output_dir
    rank = dist.get_rank()
    
    output_dir = args.output_dir
    
    run_name = f"{args.output_dir}"
    if args.attn_local is not None:
        run_name += f"_l{args.attn_local}"
    if args.attn_topk is not None:
        run_name += f"_topk{args.attn_topk}"
    if args.attn_block_topk is not None:
        run_name += f"_blktopk{args.attn_block_topk}"
        run_name += f"_blk{args.attn_block_size}"
    if args.attn_stride is not None:
        run_name += f"_stride{args.attn_stride}"
    
    output_dir = run_name
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    
    task = args.task
    trial = args.trial
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_dir, f"{task}_{current_time}_{args.seed}.jsonl")

    dist.barrier()
    result_list = eval(args, trial, task)
    dist.barrier()
    if rank == 0:
            # Gather results and compute global accuracy (optional)
            total_accuracy = 0.0
            count = 0
            results = []
            unique_result = {}
            for item in result_list:
                    results.append(item)
                    total_accuracy += item['score']
                    count += 1
                    
                    if item['query'] not in unique_result:
                        unique_result[item['query']] = item["score"]
                    else:
                        unique_result[item['query']] = max(item["score"], unique_result[item['query']])

            global_summary = {
                'task': task,
                f'pass@1': total_accuracy / count if count > 0 else 0,
                'total_example': count,
                f"pass@{trial}": sum(unique_result.values()) / len(unique_result)
            }
            
            results.insert(0, global_summary)
            print(global_summary)
            # Save the aggregated results to a single JSONLines file
            with jsonlines.open(output_file, "w") as f:
                f.write_all(results)

            print("Results saved to", output_file)

if __name__ == "__main__":
    main()
