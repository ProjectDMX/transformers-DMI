# DMI hooked variant of LlamaForCausalLM. Mirrors the gpt2_p / qwen3_p
# layout: canonical model lives in the sibling `llama/` directory; this
# subpackage subclasses those classes and inserts HookPoint instances at
# the observation sites (Q/K/V/Z, residual stream, MLP I/O, etc.).

from .configuration_llama import LlamaConfig
from .modeling_llama import (
    HookedLlamaAttention,
    HookedLlamaDecoderLayer,
    HookedLlamaForCausalLM,
    HookedLlamaModel,
)

__all__ = [
    "HookedLlamaAttention",
    "HookedLlamaDecoderLayer",
    "HookedLlamaForCausalLM",
    "HookedLlamaModel",
    "LlamaConfig",
]
