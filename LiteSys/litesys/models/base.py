from abc import ABC, abstractmethod
import torch
from ..attention.batch_cache import BatchKVManager

class LLMBase(ABC):
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def alloc(self, **kwargs):
        
        raise NotImplementedError("Subclasses must implement the `alloc` method.")

    
    @abstractmethod
    def inference(self, 
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        batch_idx: torch.LongTensor,
        kv_cache: BatchKVManager):
        
        raise NotImplementedError("Subclasses must implement the `inference` method.")
    
    
    