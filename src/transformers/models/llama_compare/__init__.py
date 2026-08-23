# Llama COMPARE variant: identical to the patched llama variant but also
# captures tensors via .copy_() to pre-allocated buffers for bitwise
# comparison against ClickHouse output. Used for transport correctness
# testing under HF TP. Mirrors transformers/models/qwen3_compare.

from .configuration_llama import LlamaConfig
from .modeling_llama import (
    CompareLlamaAttention,
    CompareLlamaDecoderLayer,
    CompareLlamaForCausalLM,
    CompareLlamaModel,
    LlamaPreTrainedModel,
)

__all__ = [
    "CompareLlamaAttention",
    "CompareLlamaDecoderLayer",
    "CompareLlamaForCausalLM",
    "CompareLlamaModel",
    "LlamaConfig",
    "LlamaPreTrainedModel",
]
