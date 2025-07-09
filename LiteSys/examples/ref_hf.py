import sys
sys.path.append("..")
from litesys.helper import generate_requests
from transformers import Qwen3ForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import os
import argparse
import torch
import wandb
import torch.multiprocessing as mp
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
from tqdm import tqdm

MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

import random
import numpy as np

TASKS = [
    "amc23",
    "aime24",
    "aime25",
    "gsm8k",
    "math500"
]

TRIALS = {
    "amc23": 8,
    "aime24": 8,
    "aime25": 8,
    "gsm8k": 1,
    "math500": 2
}

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

def worker(rank, world_size, args, shared_list, trial, task):
    
    torch.cuda.set_device(rank)
    
    data = DATAMAP[task]
    split = SPLITMAP[task]
    column = COLUMNMAP[task]
    
    answer_column = ANSWERMAP[task]
    judge = JUDGEMAP[task]
    
    llm = Qwen3ForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, _attn_implementation="flash_attention_2").to(rank)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    llm.eval()
    dataset = load_dataset(data, split=split)
    
    all_processed_request = []
    
    
    requests = generate_requests(dataset=dataset, field_name=column, data_format=MATH_QUERY_TEMPLATE, trial=trial, rank=rank, world_size=world_size)
    batch_size = args.batch_size
    
    for i in tqdm(range(0, len(requests), batch_size), desc=f"Process {rank}", position=rank):
        
        end = min(i + batch_size, len(requests))
        batch = requests[i:end]

        prompts = [
            tokenizer.apply_chat_template(
                req["conversations"],
                tokenize=False,
                add_generation_prompt=True
            )
            for req in batch
        ]

        inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                truncation=False
            ).to(rank)
    
        
        with torch.inference_mode():
            output = llm.generate(**inputs,
                                temperature=0.6,
                                top_p = 0.95, 
                                top_k = 20,
                                max_new_tokens=args.gen_len)
    
        predictions = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for i in range(end - i):
                req = batch[i]
                req["output_text"] = predictions[i]
                all_processed_request.append(req)
            
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
            
        shared_list.append(
                    {   
                        "score": result["extractive_match"] * 100,
                        "prediction": [predictions],
                        "choices": golds,
                        "query": data[column]
                    }
                )
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Model name')
    parser.add_argument('--model_class', type=str, default="qwen3", help='Model Class')
    parser.add_argument('--task', type=str, default="amc23", help='Task')
    parser.add_argument('--nproc', type=int, default=8, help='Number of processes to launch')
    parser.add_argument('--batch_size', type=int, default=4, help='Max Batch Size on GPUs')
    parser.add_argument('--max_len', type=int, default=32768, help='Max Length')
    parser.add_argument('--gen_len', type=int, default=30768, help='Max Generation Length')
    parser.add_argument('--output_dir', type=str, default="results", help='output dir')
    args = parser.parse_args()

    output_dir = args.output_dir
    
    run_name = f"{args.task}_{args.output_dir}"
    run = wandb.init(
        project="evaluations",
        name=run_name,
        config=vars(args)
    )
    os.makedirs(output_dir, exist_ok=True)
    
    task = args.task
    assert task in TASKS, f"Task {task} not in {TASKS}"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_dir, f"{task}_{current_time}.jsonl")
    trial = TRIALS[task] 
    # Use mp.Manager to create a shared list for process-safe communication
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    shared_list = manager.list()

    # Launch multiple processes using mp.spawn; each process handles a subset of the dataset
    world_size = args.nproc
    mp.spawn(worker, args=(world_size, args, shared_list, trial, task), nprocs=world_size, join=True)

    # Gather results and compute global accuracy (optional)
    total_accuracy = 0.0
    count = 0
    results = []
    unique_result = {}
    
    # Create a wandb Table for individual predictions
    columns = ["query", "prediction", "gold_answer", "score"]
    prediction_table = wandb.Table(columns=columns)

    for item in shared_list:
            results.append(item)
            total_accuracy += item['score']
            count += 1
            
            if item['query'] not in unique_result:
                unique_result[item['query']] = item["score"]
            else:
                unique_result[item['query']] = max(item["score"], unique_result[item['query']])

            prediction_table.add_data(item['query'], item['prediction'][0], item['choices'][0], item['score'])
    
    global_summary = {
        'task': task,
        f'pass@{trial}': total_accuracy / count if count > 0 else 0,
        'total_example': count,
        f"cov@{trial}": sum(unique_result.values()) / len(unique_result)
    }
    run.log(
        { 
        f"{task}/pass@{trial}{task}": total_accuracy / count if count > 0 else 0,
        f"{task}/cov@{trial}": sum(unique_result.values()) / len(unique_result)
        }
    )
    results.insert(0, global_summary)
    print(global_summary)
    # Save the aggregated results to a single JSONLines file
    with jsonlines.open(output_file, "w") as f:
        f.write_all(results)

    print("Results saved to", output_file)

    # Log summary and result table to wandb
    wandb.log(global_summary)
    wandb.log({"predictions": prediction_table})
    wandb.finish()

if __name__ == "__main__":
    main()