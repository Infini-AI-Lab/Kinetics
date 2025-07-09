from typing import Callable, Any
import datasets
from datasets import Dataset,concatenate_datasets
import json
from lighteval.tasks.extended.lcb.codegen_metrics import translate_private_test_cases


def generate_requests(dataset: Dataset, field_name: str, data_format: str, trial: int = 1, rank: int = 0, world_size: int = 1):
    requests = []

    # Step 1: Expand dataset trial times
    if trial > 1:
        dataset = Dataset.from_dict(dataset.to_dict().copy())  # ensure copy
        datasets = [dataset] * trial
        dataset = concatenate_datasets(datasets)
    
    total = len(dataset)
    
    # Step 2: Partition across ranks
    per_proc = total // world_size
    remainder = total % world_size
    start = rank * per_proc + min(rank, remainder)
    end = start + per_proc + (1 if rank < remainder else 0)
    subset = dataset.select(list(range(start, end)))

    # Step 3: Format requests
    for data in subset:
        conversations = [
            {"role": "user", "content": data_format.format(Question=data[field_name])}
        ]
        data["conversations"] = conversations
        requests.append(data)

    return requests

def generate_code_requests(dataset: Dataset, formatter: Callable[[dict[str, Any]], str], trial: int = 1, rank: int = 0, world_size: int = 1):

    requests = []

    # Step 1: Expand dataset trial times
    if trial > 1:
        dataset = Dataset.from_dict(dataset.to_dict().copy())  # ensure copy
        datasets = [dataset] * trial
        dataset = concatenate_datasets(datasets)
    
    total = len(dataset)
    
    # Step 2: Partition across ranks
    per_proc = total // world_size
    remainder = total % world_size
    start = rank * per_proc + min(rank, remainder)
    end = start + per_proc + (1 if rank < remainder else 0)
    
    subset = dataset.select(list(range(start, end)))

    # Step 3: Format requests
    for data in subset:
        query = formatter(data)
        public_test_cases = json.loads(data["public_test_cases"])
        private_test_cases = translate_private_test_cases(data["private_test_cases"])
        conversations = [
            {"role": "user", "content": query},
        ]
        data["conversations"] = conversations
        data["query"] = query
        data["inputs"] = [test["input"] for test in public_test_cases + private_test_cases]
        data["outputs"] = [test["output"] for test in public_test_cases + private_test_cases]
        data["fn_name"] = json.loads(data["metadata"]).get("func_name", None)
        requests.append(data)

    return requests
        
def generate_all_requests(dataset: Dataset, field_name: str, data_format: str, trial: int = 1, rank: int = 0, world_size: int = 1):
    requests = []

    # Step 1: Expand dataset trial times
    if trial > 1:
        dataset = Dataset.from_dict(dataset.to_dict().copy())  # ensure copy
        datasets = [dataset] * trial
        dataset = concatenate_datasets(datasets)
    
    # Step 3: Format requests
    for data in dataset:
        conversations = [
            {"role": "user", "content": data_format.format(Question=data[field_name])}
        ]
        data["conversations"] = conversations
        requests.append(data)

    return requests

def generate_all_code_requests(dataset: Dataset, formatter: Callable[[dict[str, Any]], str], trial: int = 1, rank: int = 0, world_size: int = 1):
    requests = []

    # Step 1: Expand dataset trial times
    if trial > 1:
        dataset = Dataset.from_dict(dataset.to_dict().copy())  # ensure copy
        datasets = [dataset] * trial
        dataset = concatenate_datasets(datasets)
    
    total = len(dataset)

    for data in dataset:
        query = formatter(data)
        public_test_cases = json.loads(data["public_test_cases"])
        private_test_cases = translate_private_test_cases(data["private_test_cases"])
        conversations = [
            {"role": "user", "content": query},
        ]
        data["conversations"] = conversations
        data["query"] = query
        data["inputs"] = [test["input"] for test in public_test_cases + private_test_cases]
        data["outputs"] = [test["output"] for test in public_test_cases + private_test_cases]
        data["fn_name"] = json.loads(data["metadata"]).get("func_name", None)
        requests.append(data)

    return requests