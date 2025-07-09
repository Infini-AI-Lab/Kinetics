from .qwen import Qwen
from .qwen_dist import QwenTP
from .qwen3 import Qwen3
from .qwen3_dist import Qwen3TP
from .qwen3_moe_dist import Qwen3MoETP
from .base import LLMBase

class AutoModelLM(LLMBase):
    _MODEL_MAPPING = {
        "qwen2": Qwen,
        "qwen3": Qwen3
    }
    
    _TP_MODEL_MAPPING = {
        "qwen2": QwenTP,
        "qwen3": Qwen3TP,
        "qwen3moe": Qwen3MoETP,
    }

    
    @classmethod
    def from_pretrained(cls, model_class, tp=False):
        if not tp:
            if model_class not in cls._MODEL_MAPPING:
                    raise ValueError(f"Model type '{model_class}' is not supported. "
                                    f"Supported types: {list(cls._MODEL_MAPPING.keys())}")
            else:
                model_class = cls._MODEL_MAPPING[model_class]
        else:
            if model_class not in cls._TP_MODEL_MAPPING:
                    raise ValueError(f"Model type '{model_class}' is not supported. "
                                    f"Supported types: {list(cls._TP_MODEL_MAPPING.keys())}")
            else:
                model_class = cls._TP_MODEL_MAPPING[model_class]
                
        return model_class