# Re-export of the canonical LlamaConfig so users can import it through
# the hooked subpackage as `from transformers.models.llama_p import LlamaConfig`.

from ..llama.configuration_llama import LlamaConfig

__all__ = ["LlamaConfig"]
